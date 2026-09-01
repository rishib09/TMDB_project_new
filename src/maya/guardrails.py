"""Deterministic security guardrails for Maya (issue #8).

Pure, zero-LLM, zero-network: the LangGraph entry node (#5) calls these
before routing. Design notes:
- Injection filtering is pattern-based with three verdicts — BLOCKED for
  attack patterns, SUSPICIOUS for proceed-but-flag (false positives must
  never brick the chat; #6 logs flagged queries for tuning).
- Off-topic deflection reuses the router's existing OUT_OF_SCOPE intent
  (#3) — no second LLM call, and persona text lives only here, never in
  the router prompt (see #10 / ADR 0005 role separation).
"""

import re
from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, Field

from src.domain.memory import ConversationState


class GuardrailVerdict(str, Enum):
    """Pipeline action for a guardrail check."""

    CLEAN = "clean"            # proceed normally
    SUSPICIOUS = "suspicious"  # proceed, but flag for eval logging (#6)
    BLOCKED = "blocked"        # deflect with a guardrail response


class GuardrailResult(BaseModel):
    """Outcome of a guardrail inspection."""

    verdict: GuardrailVerdict
    sanitized_query: str
    matched_patterns: list[str] = Field(default_factory=list)
    reason: str = ""


class InjectionFilter:
    """Detects and sanitizes prompt-injection / jailbreak attempts."""

    #: (pattern_id, regex) — BLOCKED on any match, first hit wins.
    ATTACK_PATTERNS: ClassVar[list[tuple[str, re.Pattern[str]]]] = [
        ("instruction_override", re.compile(
            r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+"
            r"(instructions|prompts|rules|directions)", re.IGNORECASE)),
        ("system_prompt_extraction", re.compile(
            r"(reveal|show|repeat|print|output|expose)\s+(me\s+)?(your\s+|the\s+)?"
            r"(system\s+prompt|instructions|rules)", re.IGNORECASE)),
        ("role_hijack", re.compile(
            r"\b(you\s+are\s+now|from\s+now\s+on\s+you\s+are|act\s+as\s+if\s+you\s+are|"
            r"pretend\s+(you\s+are|to\s+be))\s", re.IGNORECASE)),
        ("jailbreak_persona", re.compile(
            r"\b(DAN|do\s+anything\s+now|developer\s+mode)\b", re.IGNORECASE)),
        ("cwa_ground_smuggling", re.compile(
            r"</?\s*(retrieved_movies|movie_record|system|instructions)\s*>", re.IGNORECASE)),
        ("api_invocation_injection", re.compile(
            r"(call|invoke|execute|run)\s+(the\s+)?(function|api|tool|sql|command)",
            re.IGNORECASE)),
    ]

    #: Stripped from the query when verdict is SUSPICIOUS (proceed-but-clean).
    _SMUGGLED_MARKUP = re.compile(
        r"```.*?```|</?\s*\w+\s*/?>", re.DOTALL
    )

    def inspect(self, query: str) -> GuardrailResult:
        """Returns the pipeline verdict for a raw user query."""
        matched = [
            pattern_id
            for pattern_id, pattern in self.ATTACK_PATTERNS
            if pattern.search(query)
        ]
        if matched:
            return GuardrailResult(
                verdict=GuardrailVerdict.BLOCKED,
                sanitized_query="",
                matched_patterns=matched,
                reason="Prompt injection pattern detected",
            )

        markup = self._SMUGGLED_MARKUP.findall(query)
        sanitized = self._SMUGGLED_MARKUP.sub(" ", query)
        sanitized = re.sub(r"\s{2,}", " ", sanitized).strip()
        if markup:
            return GuardrailResult(
                verdict=GuardrailVerdict.SUSPICIOUS,
                sanitized_query=sanitized,
                matched_patterns=["markup_stripped"],
                reason="Stripped smuggled markup; query allowed through",
            )

        return GuardrailResult(verdict=GuardrailVerdict.CLEAN, sanitized_query=query.strip())


class OffTopicPivot:
    """Film-literate deflections for OUT_OF_SCOPE queries."""

    PIVOT_TEMPLATES: ClassVar[list[str]] = [
        "That's outside my reel — I'm strictly a film curator, and my projector only "
        "runs on movies from 1970 to 2026. But speaking of {topic_adj} stories, I could "
        "definitely find you a film with that energy. Want me to try?",
        "Hmm, that's a channel I don't broadcast on — movies are my whole universe. "
        "Tell me a mood or a genre though, and I'll pull something {topic_adj} from the "
        "archives.",
        "I'd be useless there and magnificent here: ask me for anything from heist "
        "thrillers to {topic_adj} romances and I've got you covered.",
    ]

    #: Rejected-domain keywords mapped to a natural film-genre adjective, so the
    #: pivot feels like a bridge rather than a canned refusal.
    _TOPIC_BRIDGES: ClassVar[dict[str, str]] = {
        "weather": "storm-swept survival",
        "stock": "high-stakes financial",
        "sport": "underdog sports",
        "cook": "foodie",
        "medicine": "medical drama",
        "space": "space-travel",
        "code": "hacker-thriller",
    }

    def pivot_response(self, query: str, seed: int = 0) -> str:
        """Deterministic deflection for an off-topic query."""
        query_lower = query.lower()
        topic_adj = next(
            (bridge for keyword, bridge in self._TOPIC_BRIDGES.items() if keyword in query_lower),
            "edge-of-your-seat",
        )
        template = self.PIVOT_TEMPLATES[seed % len(self.PIVOT_TEMPLATES)]
        return template.format(topic_adj=topic_adj)


class SessionTokenLimiter:
    """Throttles sessions that exceed the hard token cap (issue #8: 15,000)."""

    SESSION_CAP: ClassVar[int] = 15_000
    #: Below the cap but close — allow the turn, flag for wrap-up messaging.
    THROTTLE_RATIO: ClassVar[float] = 0.85

    def check(self, state: ConversationState) -> GuardrailResult:
        """Verdict for allowing another turn on this session."""
        used = state.session_tokens
        if used >= self.SESSION_CAP:
            return GuardrailResult(
                verdict=GuardrailVerdict.BLOCKED,
                sanitized_query="",
                reason=(
                    f"Session token budget exhausted ({used}/{self.SESSION_CAP}). "
                    "Please start a new session."
                ),
            )
        if used >= self.SESSION_CAP * self.THROTTLE_RATIO:
            return GuardrailResult(
                verdict=GuardrailVerdict.SUSPICIOUS,
                sanitized_query="",
                matched_patterns=["session_near_cap"],
                reason=(
                    f"Session nearing token cap ({used}/{self.SESSION_CAP}) — "
                    "wrap up gracefully."
                ),
            )
        return GuardrailResult(verdict=GuardrailVerdict.CLEAN, sanitized_query="")
