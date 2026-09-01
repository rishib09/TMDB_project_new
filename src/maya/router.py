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

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.domain.config import ExperimentConfig
from src.domain.memory import ConversationState
from src.domain.routing import (
    IntentType,
    MetadataFilterCriteria,
    QueryRoutingDecision,
)

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
6. Respond with JSON matching the schema only — no prose.

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
    ) -> None:
        """Builds the bound structured-output chain once.

        Args:
            config: Active experiment config (router model, temperature).
            api_key: OpenRouter key; defaults to ``OPENROUTER_API_KEY`` env var.
            confidence_threshold: Decisions below this confidence trigger the
                heuristic fallback.
        """
        self.config = config
        self.confidence_threshold = confidence_threshold
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

        decision = self._normalize_decision(decision)
        if decision.confidence < self.confidence_threshold:
            fallback = self._heuristic_fallback(
                query, state, reason=f"low router confidence: {decision.confidence:.2f}"
            )
            return self._apply_session_exclusions(fallback, state)
        return self._apply_session_exclusions(decision, state)

    # --- private helpers (the only custom logic) -----------------------------

    def _normalize_decision(self, decision: QueryRoutingDecision) -> QueryRoutingDecision:
        """Deterministic post-processing: the model proposes, code disposes.

        Enforces the invariants the LLM cannot be trusted to respect (measured
        on live runs — see issue #13):
        1. ``requires_rag`` is derived from intent, never taken from the model.
        2. Any pre-1970 reference forces OUT_OF_SCOPE and clears criteria.
        3. Spurious filters on non-filter intents are stripped.
        4. ``cast_member`` duplicating an excluded actor is dropped.
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

        requires_rag = decision.intent in RETRIEVAL_INTENTS

        filters = decision.filters
        if decision.intent not in FILTER_INTENTS:
            filters = None
        elif filters is not None and filters.cast_member in filters.excluded_actors:
            filters = filters.model_copy(update={"cast_member": None})

        return decision.model_copy(
            update={"requires_rag": requires_rag, "filters": filters}
        )

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
        merged = MetadataFilterCriteria(
            exact_year=filters.exact_year,
            year_min=filters.year_min,
            year_max=filters.year_max,
            genres=filters.genres,
            director=filters.director,
            cast_member=filters.cast_member,
            excluded_genres=list(
                dict.fromkeys(filters.excluded_genres + preferences.excluded_genres)
            ),
            excluded_actors=list(
                dict.fromkeys(filters.excluded_actors + preferences.excluded_actors)
            ),
        )
        return decision.model_copy(update={"filters": merged})
