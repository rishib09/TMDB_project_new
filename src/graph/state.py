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
    merge_shown_ids,
)
from src.domain.movie import MovieRecord
from src.domain.routing import QueryRoutingDecision
from src.domain.usage import LLMUsage
from src.maya.guardrails import GuardrailResult

#: #123: one usage taxonomy for every stack call — synthesis usage IS an
#: LLMUsage. Alias kept so existing imports (agent.py, tests) hold.
SynthesisUsage = LLMUsage


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

    # 3. Seen Recommendations Tracker (reset-aware unique reducer, #80)
    shown_movie_ids: Annotated[list[int], merge_shown_ids] = Field(default_factory=list)
    #: #106/C14: titles parallel the ids — the Understand state block speaks
    #: in titles. Same reducer + reset wrapper as the ids (runtime works for
    #: any hashable; reset is id-free).
    shown_movie_titles: Annotated[list[str], merge_shown_ids] = Field(default_factory=list)

    # 4. Persistent User Preferences & Exclusions (Merge Reducer)
    session_preferences: Annotated[UserSessionPreferences, merge_preferences] = Field(
        default_factory=UserSessionPreferences
    )

    # 5. Session Metrics & Summary
    rolling_summary: str = ""
    session_tokens: Annotated[int, operator.add] = 0
    #: #39: per-session estimated spend — the value the session gate blocks on.
    #: Same currency as the weekly tracker (blended estimate_cost).
    session_cost_usd: Annotated[float, operator.add] = 0.0

    # Transient per-turn pipeline artifacts
    current_query: str = ""  # sanitized by guard_input; consumed by route/retrieve/synthesize
    guardrail_result: GuardrailResult | None = None
    routing_decision: QueryRoutingDecision | None = None
    #: Bounded re-route cycle (#12): routing attempts so far this turn.
    #: Inert on the v2 stack (no re-route cycle) — kept for state shape
    #: stability; begin_turn still zeroes it each turn.
    route_attempts: int = 0
    #: Ask budget (#22): ask/probe turns used. Persists in the thread via
    #: the checkpointer since #93/D16 (the UI round-trip is gone); the v2
    #: disposer's probe-budget invariant consumes it.
    probe_count: int = 0
    #: #26-A: which deterministic stage produced this turn's response
    #: ("ask" | "retrieve" | "") — the UI turn row stays complete on turns
    #: where the route node answered by itself.
    turn_stage: str = ""
    retrieved_movies: list[MovieRecord] = Field(default_factory=list)
    synthesis_usage: SynthesisUsage | None = None
    final_response: str = ""
    #: #93 (Q3 O1): the FINAL filters the retrieve node passed the engine —
    #: None = engine never ran this turn, {} = ran with zero filters (the
    #: engine-invoked signal for the conversation-mode path adapter).
    filters_applied: dict | None = None
    #: #153 (v2): the carry-over transparency notice already fired this
    #: session. Persists in the thread via the checkpointer like probe_count/
    #: funnel_active — deliberately NOT reset by begin_turn (the notice is
    #: once per session, not once per turn).
    carryover_notice_shown: bool = False
