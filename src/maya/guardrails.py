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

import logging
import re
from datetime import date
from enum import Enum
from typing import ClassVar, Protocol

from pydantic import BaseModel, Field

from src.domain.memory import ConversationState

logger = logging.getLogger(__name__)


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
            r"(reveal|show|repeat|print|output|expose|leak)\s+(me\s+)?(your\s+|the\s+)?"
            r"(system\s+)?(prompt|prompts|instructions|rules)", re.IGNORECASE)),
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
    """Throttles sessions that exceed the hard token cap (issue #8: 15,000).

    The #5 orchestrator wires ``record()`` after every synthesis call;
    ``check()`` gates each turn in the graph's guard node.
    """

    SESSION_CAP: ClassVar[int] = 15_000
    #: Below the cap but close — allow the turn, flag for wrap-up messaging.
    THROTTLE_RATIO: ClassVar[float] = 0.85

    def __init__(self) -> None:
        self._used_tokens = 0

    #: Promised #8 interface: record(model, prompt, completion) -> BudgetStatus.
    #: BudgetStatus aliases GuardrailVerdict (CLEAN / SUSPICIOUS / BLOCKED map
    #: to under-cap / near-cap / over-cap) — one enum, no duplicate taxonomy.
    BudgetStatus = GuardrailVerdict

    def record(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> "SessionTokenLimiter.BudgetStatus":
        """Accumulates one LLM call's usage and returns the resulting budget state."""
        del model  # per-model accounting is #6 territory; the cap is per-session
        self._used_tokens += prompt_tokens + completion_tokens
        return self.check_used(self._used_tokens).verdict

    def check(self, state: ConversationState) -> GuardrailResult:
        """Verdict for allowing another turn on this session."""
        return self.check_used(state.session_tokens)

    def check_current(self) -> GuardrailResult:
        """Verdict for the limiter's own accumulated usage (graph guard node)."""
        return self.check_used(self._used_tokens)

    def check_used(self, used_tokens: int) -> GuardrailResult:
        """Verdict for a raw token count (shared by check() and record())."""
        used = used_tokens
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


# --- weekly API expenditure ceiling (#8, completion of the deferred half) ---


class BudgetSink(Protocol):
    """Duck-typed storage port — keeps this module import-free of storage."""

    def record_budget_entry(
        self, date_str: str, cost_usd: float, tokens_used: int, model_name: str
    ) -> None: ...

    def weekly_spend_usd(self) -> float: ...


#: Blended $/1M-token estimates (input+output mixed), matched by substring
#: against the model id. Estimates for overspend PROTECTION, not billing —
#: the point is an order-of-magnitude ceiling, not accounting precision.
#: Unknown models fall back to the priciest rate (fail-closed on cost).
MODEL_PRICES_PER_MTOK: ClassVar[list[tuple[str, float]]] = [
    ("llama-3.2-3b", 0.06),
    ("llama-3.3-70b", 0.20),
    ("flash-lite", 0.12),
]
DEFAULT_PRICE_PER_MTOK: ClassVar[float] = 1.00


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Blended-cost estimate for one LLM call (pure, deterministic)."""
    tokens = max(0, int(prompt_tokens)) + max(0, int(completion_tokens))
    model_lower = (model or "").lower()
    price = next(
        (p for key, p in MODEL_PRICES_PER_MTOK if key in model_lower),
        DEFAULT_PRICE_PER_MTOK,
    )
    return tokens / 1_000_000 * price


class WeeklyBudgetTracker:
    """Persistent weekly API-spend ceiling (issue #8: $10.00/week).

    Complements the per-session token limiter: that one bounds a single
    conversation, this bounds aggregate spend across all sessions and days.
    Storage errors fail OPEN (logged, treated as no data) — a database
    hiccup must never brick the chat; the ceiling is protection, not a wall.
    """

    WEEKLY_CAP_USD: ClassVar[float] = 10.00
    #: Below the cap but close — allow turns, surface a visible warning.
    WARN_RATIO: ClassVar[float] = 0.80

    def __init__(self, sink: BudgetSink) -> None:
        self._sink = sink

    def record(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> GuardrailVerdict:
        """Logs one call's estimated cost and returns the resulting weekly verdict."""
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        try:
            self._sink.record_budget_entry(
                date.today().isoformat(), cost, prompt_tokens + completion_tokens, model
            )
            spend = self._sink.weekly_spend_usd()
        except Exception:  # noqa: BLE001 — fail-open, see class docstring
            logger.warning("weekly budget sink failed; ceiling unenforced this turn", exc_info=True)
            return GuardrailVerdict.CLEAN
        return self.verdict_for(spend)

    def current_verdict(self) -> GuardrailVerdict:
        """Verdict for gating the NEXT turn (graph guard node) — fail-open."""
        try:
            spend = self.weekly_spend()
        except Exception:  # noqa: BLE001 — sink read failure, see class docstring
            logger.warning("weekly budget read failed; ceiling unenforced", exc_info=True)
            return GuardrailVerdict.CLEAN
        return self.verdict_for(spend)

    def weekly_spend(self) -> float:
        """Current ISO-week spend in USD (raises if the sink fails)."""
        return self._sink.weekly_spend_usd()

    def verdict_for(self, spend_usd: float) -> GuardrailVerdict:
        """Pure threshold logic (CLEAN / SUSPICIOUS warn / BLOCKED stop)."""
        if spend_usd >= self.WEEKLY_CAP_USD:
            return GuardrailVerdict.BLOCKED
        if spend_usd >= self.WEEKLY_CAP_USD * self.WARN_RATIO:
            return GuardrailVerdict.SUSPICIOUS
        return GuardrailVerdict.CLEAN
