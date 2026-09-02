"""Deterministic Maya Router: single-pass intent classification, coreference
query reformulation, and heuristic fallback via OpenRouter structured output.

Design notes:
- The ONLY public seam is :meth:`MayaRouter.route` (spec: public seam testing).
- No persona text here — Maya's voice lives in the synthesis prompt (Issue #10).
- No custom JSON parsing: ``ChatOpenAI.with_structured_output`` validates the
  LLM response directly into the existing ``QueryRoutingDecision`` schema.
- Not wired into LangGraph here; ``route()`` becomes a node callable in the
  Issue #5 StateGraph.
"""

import os
import re
from collections.abc import Collection

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.domain.config import ExperimentConfig
from src.domain.memory import ConversationState
from src.domain.routing import (
    IntentType,
    MetadataFilterCriteria,
    QueryRoutingDecision,
)
from src.maya.probing import extract_probe_answers, strip_markup

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: Intents whose handling requires the retrieval pipeline.
RETRIEVAL_INTENTS = {
    IntentType.SEMANTIC_SEARCH,
    IntentType.ATTRIBUTE_FILTER,
    IntentType.SUPERLATIVE_RANKING,
    IntentType.NEGATION_EXCLUSION,
}

#: Intents allowed to carry structured filters/superlative criteria.
FILTER_INTENTS = {
    IntentType.ATTRIBUTE_FILTER,
    IntentType.NEGATION_EXCLUSION,
}

#: First year covered by the bundled dataset (see ADR 0001).
DATASET_START_YEAR = 1970

#: Textual pre-1970 references (years or decades) — backs the genre guard's
#: temporal exemption (#26): "1950s horror movies" must stay OUT_OF_SCOPE.
_PRE_1970_TEXT_RE = re.compile(r"\b(18\d\d|19[0-6]\d|18\d0s|19[0-6]0s)\b")

#: Max conversation turns (2 messages each) replayed to the router for
#: coreference resolution. Keeps prompt size bounded regardless of history.
MAX_HISTORY_TURNS = 2

ROUTER_SYSTEM_PROMPT = """You are the deterministic intent router for Maya, \
a film curator specialized in US theatrical releases from 1970 to 2026.

You are a classifier. You never search for movies, never recommend movies, \
and never mention specific movie titles in your reasoning. Your only job is \
to classify the utterance and extract structured fields.

Classify the user's utterance into exactly one intent:
- GREETING: salutations or small talk with no film request.
- CAPABILITIES: questions about what Maya can do.
- SEMANTIC_SEARCH: plot, theme, or mood-based movie discovery needing retrieval.
- ATTRIBUTE_FILTER: concrete metadata filters (year, genre, director, cast).
- SUPERLATIVE_RANKING: extremes ("highest", "lowest", "longest", "top N") by a metric.
- NEGATION_EXCLUSION: requests excluding genres or actors ("no horror", "without Tom Cruise").
- OUT_OF_SCOPE: non-film topics, or films released before 1970.

Temporal rule: the dataset covers ONLY 1970-2026. Queries about any year or \
decade before 1970 (e.g. "1950s", "1939", "1968") must be classified \
OUT_OF_SCOPE. Never extract filters or superlatives for pre-1970 years.

Superlative metric mapping:
- "highest-grossing", "biggest box office", "most profitable" -> REVENUE
- "most expensive", "biggest budget" -> BUDGET
- "best", "top rated", "highest rated" -> RATING
- "most popular" -> POPULARITY
- "longest", "shortest" -> RUNTIME
- "most voted" -> VOTE_COUNT

Rules:
1. Resolve pronouns and ellipses in follow-ups using the conversation context \
into `standalone_query` — a self-contained search string. For non-search intents, \
echo the cleaned query.
2. For SUPERLATIVE_RANKING fill `superlative` (metric, direction, year, genre, limit). \
For ATTRIBUTE_FILTER and NEGATION_EXCLUSION fill `filters`. Never fill `filters` \
for other intents.
3. Place newly stated exclusions in `filters.excluded_genres` / `filters.excluded_actors`. \
Never also set `cast_member` to an excluded actor.
4. Set `requires_rag` to false for GREETING, CAPABILITIES, and OUT_OF_SCOPE. \
(Inconsistent values are corrected automatically; focus on intent accuracy.)
5. Set `confidence` to your routing confidence between 0.0 and 1.0.
6. ALWAYS fill `mood` and `audience` when the utterance expresses them, \
regardless of intent: `mood` = emotional flavor in the user's own words \
(e.g. "edge of the seat", "feel-good", "funny", "scary", "mind-bending"); \
`audience` = who is watching (e.g. "kids", "family", "date night", "adults"). \
Leave them as empty strings otherwise. Never invent a mood or audience \
that was not expressed.
7. When the user names a PERSON without stating a role (e.g. "movies of \
Clint Eastwood"), fill `filters.person`. Use `filters.director` or \
`filters.cast_member` ONLY when the role is explicit ("directed by", \
"starring"). The system resolves the role against its own database.
8. Respond with JSON matching the schema only — no prose.

Intent boundary examples (follow these closely):
- "best movies from the 1950s" -> OUT_OF_SCOPE (pre-1970)
- "1980s horror movies directed by John Carpenter starring Kurt Russell" -> \
ATTRIBUTE_FILTER (concrete year+genre+director+cast constraints)
- "highest-grossing film of 1970" -> SUPERLATIVE_RANKING (any "highest/lowest/" \
"longest/shortest/top" superlative wording, even with a year)
- "action movies without Tom Cruise" -> NEGATION_EXCLUSION (the word "without" \
or "no" plus a genre means exclusion, not plain filtering)
- "movies about space exploration and lunar colonies" -> SEMANTIC_SEARCH \
(plot/theme concepts, no concrete filters)
- "who directed it?" (after discussing Inception) -> SEMANTIC_SEARCH with \
standalone_query="Who directed Inception?"
"""


class MayaRouter:
    """Routes one user utterance into a validated :class:`QueryRoutingDecision`.

    Single-pass OpenRouter call at temperature 0.0 with a Pydantic-validated
    structured output. Falls back to a keyword heuristic when the API is
    unreachable or returns low-confidence output, so callers never see an
    exception.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        api_key: str | None = None,
        confidence_threshold: float = 0.5,
        genre_vocabulary: Collection[str] = (),
    ) -> None:
        """Builds the bound structured-output chain once.

        Args:
            config: Active experiment config (router model, temperature).
            api_key: OpenRouter key; defaults to ``OPENROUTER_API_KEY`` env var.
            confidence_threshold: Decisions below this confidence trigger the
                heuristic fallback.
            genre_vocabulary: Genre names present in the dataset (#26 genre
                guard). Fed from ``MovieDatabase.distinct_genres()`` — the
                data is the vocabulary. Empty set simply disables the guard.
        """
        self.config = config
        self.confidence_threshold = confidence_threshold
        self.genre_vocabulary = frozenset(
            g.lower().strip() for g in genre_vocabulary if g.strip()
        )
        self._llm = ChatOpenAI(
            model=config.router_model,
            temperature=config.temperature,
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            max_tokens=1024,  # prevents truncated JSON on long structured outputs
        )
        # Bound once at construction; tests stub this attribute directly.
        self._chain = self._llm.with_structured_output(QueryRoutingDecision)

    def route(
        self,
        query: str,
        state: ConversationState,
        feedback: str | None = None,
    ) -> QueryRoutingDecision:
        """Single public seam: classify one utterance against session state.

        Pipeline: build prompt -> structured LLM call -> merge persistent
        session exclusions into the decision. On API failure or low
        confidence, returns a heuristic fallback decision (flagged via
        ``is_fallback``) instead of raising. The orchestrator (#5) uses that
        flag to drive its bounded re-route cycle; ``feedback`` carries the
        corrective context injected into the prompt on re-route attempts.
        """
        messages = self._build_messages(query, state, feedback)
        try:
            decision = self._chain.invoke(messages)
        except Exception as exc:  # noqa: BLE001 — any API/schema failure degrades gracefully
            fallback = self._heuristic_fallback(
                query, state, reason=f"router API error: {exc}"
            )
            return self._apply_session_exclusions(fallback, state)

        decision = self._normalize_decision(decision, query)
        if decision.confidence < self.confidence_threshold:
            fallback = self._heuristic_fallback(
                query, state, reason=f"low router confidence: {decision.confidence:.2f}"
            )
            return self._apply_session_exclusions(fallback, state)
        return self._apply_session_exclusions(decision, state)

    # --- private helpers (the only custom logic) -----------------------------

    def _normalize_decision(
        self, decision: QueryRoutingDecision, query: str = ""
    ) -> QueryRoutingDecision:
        """Deterministic post-processing: the model proposes, code disposes.

        Enforces the invariants the LLM cannot be trusted to respect (measured
        on live runs — see issues #13 and #26):
        1. ``requires_rag`` is derived from intent, never taken from the model.
        2. Any pre-1970 reference forces OUT_OF_SCOPE and clears criteria.
        3. Spurious filters on non-filter intents are stripped.
        4. ``cast_member`` duplicating an excluded actor is dropped.
        5. ``mood``/``audience`` are whitespace-stripped, with the
           deterministic vocabulary as fallback when the model missed them.
        6. #26-B/C: OUT_OF_SCOPE is FORBIDDEN when the query carries in-scope
           vocabulary — a genre word or a mood/audience answer is a film
           request by definition ("suggest me horror movies" misrouted at
           confidence 1.00 in the walkthrough). The pre-1970 rule above wins:
           "1950s horror" stays out of scope.
        """
        if self._references_pre_1970(decision):
            return decision.model_copy(
                update={
                    "intent": IntentType.OUT_OF_SCOPE,
                    "requires_rag": False,
                    "filters": None,
                    "superlative": None,
                    "is_superlative": False,
                }
            )

        if decision.intent is IntentType.OUT_OF_SCOPE and self._has_in_scope_vocabulary(
            query
        ) and not _PRE_1970_TEXT_RE.search(query):
            decision = decision.model_copy(
                update={
                    "intent": IntentType.SEMANTIC_SEARCH,
                    "reasoning": (
                        (decision.reasoning or "") + " [genre-guard: in-scope "
                        "vocabulary present, OUT_OF_SCOPE overridden (#26)]"
                    ).strip(),
                }
            )

        # Mood/audience fallback (#26-D): the vocab is deterministic where the
        # 3B model is variance-prone; the LLM's own extraction always wins.
        mood = (decision.mood or "").strip()
        audience = (decision.audience or "").strip()
        if not mood or not audience:
            vocab = extract_probe_answers(query)
            mood = mood or vocab.preferred_mood
            audience = audience or vocab.audience
        # Markup sanitizer: these fields reach retrieval queries and
        # deterministic responses downstream (#26-E notice).
        mood = strip_markup(mood).strip()
        audience = strip_markup(audience).strip()

        filters = decision.filters
        if decision.intent not in FILTER_INTENTS:
            filters = None
        elif filters is not None:
            updates = {}
            if filters.cast_member in filters.excluded_actors:
                updates["cast_member"] = None
            if filters.person:
                updates["person"] = filters.person.strip() or None
            if updates:
                filters = filters.model_copy(update=updates)

        return decision.model_copy(
            update={
                "requires_rag": decision.intent in RETRIEVAL_INTENTS,
                "filters": filters,
                "mood": mood,
                "audience": audience,
            }
        )

    def _has_in_scope_vocabulary(self, query: str) -> bool:
        """True when the query contains a genre word or mood/audience vocab (#26)."""
        if not query:
            return False
        lowered = re.sub(r"\bsci fi\b", "sci-fi", query.lower())
        if any(
            re.search(rf"\b{re.escape(genre)}\b", lowered)
            for genre in self.genre_vocabulary
        ):
            return True
        return bool(extract_probe_answers(lowered).answered_axes())

    @staticmethod
    def _references_pre_1970(decision: QueryRoutingDecision) -> bool:
        """True if any structured criteria references a year before the dataset."""
        year_fields = ("exact_year", "year_min", "year_max", "year")
        for criteria in (decision.filters, decision.superlative):
            if criteria is None:
                continue
            if any(
                (year := getattr(criteria, field, None)) is not None
                and year < DATASET_START_YEAR
                for field in year_fields
            ):
                return True
        return False

    def _build_messages(
        self, query: str, state: ConversationState, feedback: str | None = None
    ) -> list[BaseMessage]:
        """System prompt + re-route feedback + entity context + history + query."""
        messages: list[BaseMessage] = [SystemMessage(content=ROUTER_SYSTEM_PROMPT)]
        if feedback:
            messages.append(SystemMessage(content=feedback))

        context_lines: list[str] = []
        if state.focused_entity is not None:
            entity = state.focused_entity
            context_lines.append(
                f"Currently discussed movie: {entity.title} ({entity.release_year}),"
                f" directed by {entity.director}."
            )
        if state.focused_person:
            context_lines.append(f"Currently discussed person: {state.focused_person}.")
        if context_lines:
            messages.append(SystemMessage(content="\n".join(context_lines)))

        # Replay only the recent turns needed for coreference resolution.
        recent = state.messages[-MAX_HISTORY_TURNS * 2 :]
        for chat_message in recent:
            if chat_message.role == "user":
                messages.append(HumanMessage(content=chat_message.content))
            else:
                messages.append(AIMessage(content=chat_message.content))

        messages.append(HumanMessage(content=query))
        return messages

    def _heuristic_fallback(
        self, query: str, state: ConversationState, reason: str
    ) -> QueryRoutingDecision:
        """Keyword-regex safety net used when the API fails or is unsure.

        Detects greetings, capability questions, and pre-1970 temporal
        references; everything else degrades to SEMANTIC_SEARCH with the raw
        query. Confidence is set low so downstream components know the
        decision is degraded.
        """
        lowered = query.lower().strip()

        if re.search(r"\b(hi|hello|hey|yo|good (morning|afternoon|evening))\b", lowered):
            intent = IntentType.GREETING
        elif re.search(r"(what can you do|who are you|how do you work|help me)", lowered):
            intent = IntentType.CAPABILITIES
        elif re.search(r"\b(18\d\d|19[0-6]\d)\b", lowered) or re.search(
            r"\b(18\d0s|19[0-6]0s)\b", lowered
        ):
            intent = IntentType.OUT_OF_SCOPE
        else:
            intent = IntentType.SEMANTIC_SEARCH

        return QueryRoutingDecision(
            intent=intent,
            confidence=0.1,
            standalone_query=query,
            requires_rag=intent in RETRIEVAL_INTENTS,
            reasoning=f"Heuristic fallback: {reason}",
            is_fallback=True,
        )

    def _apply_session_exclusions(
        self, decision: QueryRoutingDecision, state: ConversationState
    ) -> QueryRoutingDecision:
        """Merges persistent session exclusions into the decision's filters.

        Memory is the single source of truth for exclusions — this guarantees
        user-stated negative preferences survive even when the LLM omits them.
        """
        preferences = state.session_preferences
        if not (preferences.excluded_genres or preferences.excluded_actors):
            return decision
        if not decision.requires_rag:
            return decision

        filters = decision.filters or MetadataFilterCriteria()
        # model_copy preserves fields the reconstruction used to silently
        # drop (person, genre_match — #26 latent filter-eating bug).
        merged = filters.model_copy(
            update={
                "excluded_genres": list(
                    dict.fromkeys(filters.excluded_genres + preferences.excluded_genres)
                ),
                "excluded_actors": list(
                    dict.fromkeys(filters.excluded_actors + preferences.excluded_actors)
                ),
            }
        )
        return decision.model_copy(update={"filters": merged})
