"""LangGraph execution state — the framework-shaped view over pure domain types.

ADR 0006: ``src/domain/`` stays pure Pydantic (zero framework imports). This
module is where LangGraph execution semantics (``Annotated[..., reducer]``)
live, next to the ``StateGraph`` that compiles it (see ``orchestrator.py``).

Note (deviation from the #5 scope comment, recorded deliberately): the pure
merge functions ``merge_unique_ids`` / ``merge_preferences`` stay in
``src/domain/memory.py`` — they are framework-free domain logic that
``ConversationState.add_turn`` also uses. Moving them here would either
duplicate logic or create a domain → graph dependency arrow, both ADR-0006
violations. Only the framework-shaped state schema relocates.
"""

import operator
from collections.abc import Sequence
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.domain.memory import (
    FocusedMovieEntity,
    UserSessionPreferences,
    merge_preferences,
    merge_unique_ids,
)
from src.domain.movie import MovieRecord
from src.domain.routing import QueryRoutingDecision
from src.maya.guardrails import GuardrailResult


class SynthesisUsage(BaseModel):
    """Token usage of one synthesis LLM call, for budget accounting (#8)."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class MayaGraphState(BaseModel):
    """LangGraph 5-Layer Agent State with functional reducers.

    Nodes return simple dict updates (e.g. ``{'messages': [AIMessage(...)],
    'shown_movie_ids': [27205]}``), and LangGraph reducers handle
    deduplication, merging, and accumulation automatically.
    """

    # 1. Message History (Sliding window managed via LangGraph add_messages)
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(default_factory=list)

    # 2. Entity Focus Layer
    focused_entity: FocusedMovieEntity | None = None
    focused_person: str | None = None

    # 3. Seen Recommendations Tracker (Unique Reducer)
    shown_movie_ids: Annotated[list[int], merge_unique_ids] = Field(default_factory=list)

    # 4. Persistent User Preferences & Exclusions (Merge Reducer)
    session_preferences: Annotated[UserSessionPreferences, merge_preferences] = Field(
        default_factory=UserSessionPreferences
    )

    # 5. Session Metrics & Summary
    rolling_summary: str = ""
    session_tokens: Annotated[int, operator.add] = 0

    # Transient per-turn pipeline artifacts
    current_query: str = ""  # sanitized by guard_input; consumed by route/retrieve/synthesize
    guardrail_result: GuardrailResult | None = None
    routing_decision: QueryRoutingDecision | None = None
    #: Bounded re-route cycle (#12): routing attempts so far this turn.
    route_attempts: int = 0
    #: Guided narrowing (#22): probe turns used (persists via UI round-trip,
    #: so no add-reducer — the session passes the running total each turn).
    probe_count: int = 0
    #: Funnel mode (#23): a probe/confirm was just shown; the next message
    #: belongs to the funnel, not the router. Persists via UI round-trip.
    funnel_active: bool = False
    #: Transient (#23): this turn fell through the funnel — OUT_OF_SCOPE
    #: pivots are suppressed for exactly this turn (it may be an answer to
    #: our own question, not an off-topic query).
    from_funnel: bool = False
    retrieved_movies: list[MovieRecord] = Field(default_factory=list)
    synthesis_usage: SynthesisUsage | None = None
    final_response: str = ""
