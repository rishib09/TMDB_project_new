"""Maya orchestrator (issue #5): compiled LangGraph StateGraph workflow.

v2 stack only (#156 hard split): the Understand router owns the turn, the
v1 funnel trio is gone. Topology::

    START → guard_input ──blocked──→ refusal ────────────────→ END
               │clean
               ▼
              route ──ask────────────────→ trim ───────────→ END
               │     └─OUT_OF_SCOPE──────→ pivot ───────────→ END
               │     └─no retrieval───────→ synthesize ─────→ END
               ▼
            retrieve ──────────────────────→ synthesize ─────→ END
                 (all synthesize paths) ──→ carryover_notice → END

Every component is already built and tested (#3/#4/#8 guardrails+engine,
#106 v2 router); this module only wires them as LangGraph nodes with
conditional edges — no custom dispatch, no custom state management.

The graph compiles against ``MayaGraphState`` (Pydantic schema, see
``state.py``), so nodes receive the model instance and return partial dict
updates that LangGraph applies through the Annotated reducers.
"""

import re
from typing import Literal

from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.domain.config import ExperimentConfig
from src.domain.memory import (
    FocusedMovieEntity,
    PreferencesUpdate,
    ShownIdsUpdate,
    UserSessionPreferences,
)
from src.domain.moods import (
    expand_query_text,
    merge_profile_floors,
    mood_boost_spec,
    resolve_mood_profile,
)
from src.domain.routing import IntentType, MetadataFilterCriteria, QueryRoutingDecision
from src.domain.usage import LLMUsage
from src.graph.state import MayaGraphState
from src.maya.agent import MayaSynthesizer
from src.maya.guardrails import (
    GuardrailResult,
    GuardrailVerdict,
    InjectionFilter,
    OffTopicPivot,
    SessionCostLimiter,
    WeeklyBudgetTracker,
    estimate_cost,
)
from src.maya.probing import is_fresh_start
from src.maya.v2 import (
    MayaV2Router,
    dispose,
    project_understanding,
    turn_decision,
)
from src.maya.v2.notices import (
    build_filter_carryover_notice as v2_carryover_notice,
    injected_genres,
)
from src.observability.tracer import DualModeObservabilityManager
from src.retrieval.hybrid_engine import HybridRetrievalEngine


def fold_preference_years(
    decision: QueryRoutingDecision, prefs: UserSessionPreferences
) -> QueryRoutingDecision:
    """Fill decision.filters year gaps from session preferences (#78, #116).

    Preference years (era snapshots persisted via merge_preferences) fill a
    decision that carries NO year constraint at all; a decision with any
    year stands completely. Per-field composition was tried and rejected:
    one utterance can be read twice (the model's ``year_min`` AND the era
    extractor's ``year_max`` from the same phrase — the "older movies"
    case), and mixing the two readings over-constrains retrieval against
    the model's intent. The prefs years are merge_preferences output
    (#56-F3), so the fill can never create an impossible range.
    Returns the original object when nothing applies (identity, not a copy).
    """
    f = decision.filters
    if f is not None and (
        f.exact_year is not None or f.year_min is not None or f.year_max is not None
    ):
        return decision  # decision-sourced years stand; no per-field mixing
    if (
        prefs.exact_year is None
        and prefs.year_min is None
        and prefs.year_max is None
    ):
        return decision
    # decision carries no years: the prefs years fill wholesale
    filters = (f or MetadataFilterCriteria()).model_copy(
        update={
            "exact_year": prefs.exact_year,
            "year_min": prefs.year_min,
            "year_max": prefs.year_max,
        }
    )
    return decision.model_copy(update={"filters": filters})


def build_maya_graph(
    config: ExperimentConfig,
    router: "MayaV2Router",
    engine: HybridRetrievalEngine,
    synthesizer: MayaSynthesizer,
    tracer: DualModeObservabilityManager,
    limiter: SessionCostLimiter | None = None,
    budget_tracker: WeeklyBudgetTracker | None = None,
    checkpointer: InMemorySaver | None = None,
) -> CompiledStateGraph:
    """Compiles the Maya workflow with injected components (DI-friendly).

    All heavy collaborators (router, engine, synthesizer) are constructed by
    the caller — the graph itself owns only sequencing and conditional edges,
    which is what makes the unit tests mock-free at the component level.
    """
    limiter = limiter or SessionCostLimiter()
    injection_filter = InjectionFilter()
    pivot = OffTopicPivot()
    router_v2 = router

    def _meter_llm(usage: LLMUsage | None, node: str) -> float:
        """Record one LLM call everywhere it matters (#123).

        Session limiter (the gate), weekly ledger (the ceiling), and one
        ``cost`` trace row per call — then returns the estimated cost so the
        calling node writes ``session_cost_usd`` (and tokens to
        ``session_tokens``, feeding the #93 metrics).

        Unusable usage — ``None`` (stubbed clients, failed calls) or
        non-numeric fields (review P3-2: test doubles that slip the
        isinstance guard) — never silently disappears: one explicit
        ``unmetered`` cost row records the spend-invisibility and the call
        costs nothing (AGENTS.md: fail-open must be explicit and recorded).
        """
        if usage is not None:
            try:
                prompt_tokens = int(usage.prompt_tokens)
                completion_tokens = int(usage.completion_tokens)
            except (TypeError, ValueError, AttributeError):
                usage = None  # non-numeric usage → marker path, not a crash
        if usage is None:
            tracer.record_local(
                "cost",
                {
                    "node": node,
                    "model": "unknown",
                    "tokens": 0,
                    "cost_usd": 0.0,
                    "unmetered": "no_usage_metadata",
                },
            )
            return 0.0
        budget_status = limiter.record(
            usage.model, prompt_tokens, completion_tokens
        )
        weekly_status = None
        if budget_tracker is not None:
            weekly_status = budget_tracker.record(
                usage.model, prompt_tokens, completion_tokens
            )
        cost = estimate_cost(usage.model, prompt_tokens, completion_tokens)
        tracer.record_local(
            "cost",
            {
                "node": node,
                "model": usage.model,
                "tokens": prompt_tokens + completion_tokens,
                "cost_usd": cost,
                "budget": budget_status.value,
                "weekly_budget": weekly_status.value if weekly_status else "off",
            },
        )
        return cost

    def guard_input_node(state: MayaGraphState) -> dict:
        """Injection filter + session budget gate (issue #8, zero LLM)."""
        query = state.messages[-1].text if state.messages else ""
        injection = injection_filter.inspect(query)
        if injection.verdict is GuardrailVerdict.BLOCKED:
            tracer.record_local(
                "guard_input", {"verdict": "blocked", "patterns": injection.matched_patterns}
            )
            return {
                "guardrail_result": injection,
                "final_response": _refusal_text(injection.reason),
            }
        budget = limiter.check_current()
        if budget.verdict is GuardrailVerdict.BLOCKED:
            tracer.record_local("guard_input", {"verdict": "budget_blocked"})
            return {"guardrail_result": budget, "final_response": _refusal_text(budget.reason)}
        # Weekly $ ceiling (#8): aggregate spend across sessions and days.
        if budget_tracker is not None:
            weekly = budget_tracker.current_verdict()
            if weekly is GuardrailVerdict.BLOCKED:
                tracer.record_local("guard_input", {"verdict": "weekly_budget_blocked"})
                weekly_result = GuardrailResult(
                    verdict=GuardrailVerdict.BLOCKED,
                    sanitized_query="",
                    reason=(
                        f"Weekly budget exhausted (${budget_tracker.WEEKLY_CAP_USD:.2f})"
                    ),
                )
                return {
                    "guardrail_result": weekly_result,
                    "final_response": _refusal_text(weekly_result.reason),
                }
        tracer.record_local("guard_input", {"verdict": injection.verdict.value})
        # SUSPICIOUS verdict (stripped markup): proceed with the sanitized query.
        sanitized = injection.sanitized_query or query
        # #26-E: "something completely different" wipes accumulated preferences
        # at the ONE choke point every turn passes — routing proceeds on a
        # clean slate.
        if is_fresh_start(sanitized):
            tracer.record_local("guard_input", {"fresh_start": True})
            return {
                "guardrail_result": injection,
                "current_query": sanitized,
                "session_preferences": UserSessionPreferences(reset_requested=True),
                # #80: the fresh slate covers shown ids too — the union
                # reducer alone could never clear them, so "something
                # completely different" left every earlier title excluded
                # for the rest of the thread.
                "shown_movie_ids": ShownIdsUpdate(reset=True),
                "shown_movie_titles": ShownIdsUpdate(reset=True),
            }
        return {"guardrail_result": injection, "current_query": sanitized}

    def retrieve_node(state: MayaGraphState) -> dict:
        """Hybrid retrieval (#4): SQL path or RRF fusion, per routing decision.

        Probed mood/audience (#22) enrich the semantic query text — the
        embedding model handles natural-language flavor natively — while
        genres/directors stay as the router's deterministic SQL filters.
        """
        decision = state.routing_decision
        prefs = state.session_preferences
        # #137: MoodProfile translation — the mood's concrete retrieval
        # meaning (phrasebook, floors, boosts) is curated data applied by
        # code, in the ONE seam both routing stacks share. Unmapped moods
        # fail open to flavor-only behavior, recorded. Mapped moods must
        # NOT re-append ``(mood: …)`` — that was the measured BM25 failure.
        # The flag-off A/B arm is on the record too: the Trace must
        # distinguish a profile-off run from a no-mood run (telemetry rule).
        session_mood = prefs.preferred_mood or decision.mood
        if not config.mood_profiles_enabled:
            profile = None
            if session_mood:
                tracer.record_local(
                    "retrieve", {"mood_profile": "disabled", "mood": session_mood}
                )
        else:
            profile = resolve_mood_profile(session_mood)
            if not profile and session_mood:
                tracer.record_local("retrieve", {"mood_profile": "unmapped", "mood": session_mood})
        flavor = ", ".join(
            filter(
                None,
                (
                    (
                        f"mood: {prefs.preferred_mood}"
                        if prefs.preferred_mood and profile is None
                        else ""
                    ),
                    f"audience: {prefs.audience}" if prefs.audience else "",
                ),
            )
        )
        query = decision.standalone_query
        if profile:
            query = expand_query_text(query, profile)
        if flavor:
            query = f"{query} ({flavor})"
        # #78/#116: preference years (era snapshots) reach retrieval when the
        # decision carries none — decision years win per-field (see helper).
        # On the record when the fold changed the decision OR resolved pref
        # years differently (a dropped/overridden pref bound is a state
        # transition the trace must show).
        prefs_years = (prefs.exact_year, prefs.year_min, prefs.year_max)
        folded = fold_preference_years(decision, prefs)
        final_f = folded.filters
        effective_years = (
            (final_f.exact_year, final_f.year_min, final_f.year_max)
            if final_f
            else (None, None, None)
        )
        if folded is not decision or (
            any(y is not None for y in prefs_years) and effective_years != prefs_years
        ):
            tracer.record_local("retrieve", {"prefs_years_applied": {
                "exact_year": effective_years[0],
                "year_min": effective_years[1],
                "year_max": effective_years[2],
            }})
        decision = folded
        # Confirmed funnel genres (#25) drive deterministic genre filters.
        if prefs.preferred_genres and not (
            decision.filters and decision.filters.genres
        ):
            from src.domain.routing import MetadataFilterCriteria

            filters = decision.filters or MetadataFilterCriteria()
            # #56-F2: candidate-list genres are a union ("any of these"), only
            # explicitly conjoined genres keep intersection semantics.
            decision = decision.model_copy(update={"filters": filters.model_copy(update={
                "genres": prefs.preferred_genres,
                "genre_match": (
                    "any"
                    if prefs.genres_from_candidates or len(prefs.preferred_genres) == 1
                    else "all"
                ),
            })})
        # #137: profile floors tighten AFTER the genre merge so neither can
        # lose the other's update (both are additive model_copy chains).
        if profile:
            decision = decision.model_copy(
                update={"filters": merge_profile_floors(decision.filters, profile)}
            )
        boost = (
            mood_boost_spec(
                profile,
                scale=config.mood_boost_scale,
                normalizer=config.mood_boost_normalizer,
                runtime_weight=config.mood_boost_runtime_weight,
                term_cap=config.mood_boost_term_cap,
                popularity_log_divisor=config.mood_boost_popularity_log_divisor,
                revenue_log_divisor=config.mood_boost_revenue_log_divisor,
            )
            if profile
            else None
        )
        results = engine.retrieve(
            query=query,
            routing=decision,
            top_k=config.retrieval_top_k,
            shown_ids=list(state.shown_movie_ids),  # #88: store-level exclusion
            boost=boost,
        )
        if profile:
            tracer.record_local("retrieve", {
                "mood_profile": profile.id,
                "phrases_applied": len(profile.query_phrases),
                "floors": profile.floors.model_dump(exclude_none=True),
                # post-merge floors the engine actually enforced (not just declared)
                "floors_enforced": (
                    {"vote_count_min": decision.filters.vote_count_min}
                    if decision.filters and decision.filters.vote_count_min is not None
                    else {}
                ),
                "boost_scale": config.mood_boost_scale,
                # WHICH calibration ran — an A/B comparison of boost weights
                # is meaningless without it (telemetry rule, ADR 0004).
                "calibration": {
                    "normalizer": config.mood_boost_normalizer,
                    "runtime_weight": config.mood_boost_runtime_weight,
                    "term_cap": config.mood_boost_term_cap,
                    "popularity_log_divisor": config.mood_boost_popularity_log_divisor,
                    "revenue_log_divisor": config.mood_boost_revenue_log_divisor,
                },
            })
        # Intersection too narrow? (#25) retry ANY-match, relaxation on record.
        if not results and decision.filters and decision.filters.genre_match == "all":
            relaxed = decision.model_copy(update={"filters": decision.filters.model_copy(
                update={"genre_match": "any"}
            )})
            results = engine.retrieve(
                query=query,
                routing=relaxed,
                top_k=config.retrieval_top_k,
                shown_ids=list(state.shown_movie_ids),  # #88
                boost=boost,
            )
            decision = relaxed  # #93 Q13: filters_applied must record the FINAL
                                # post-relaxation routing — the engine ran "any"
            tracer.record_local("retrieve", {"genre_match_relaxed": True})
        movies = [r.movie for r in results]
        dense_failure = getattr(engine, "last_dense_failure", None)
        if dense_failure:  # #65: BM25-only fallback is explicit in the Trace
            tracer.record_local("retrieve", {"dense_failed": True, "error": dense_failure})
        # #93 (Q3 O1): the FINAL effective filters — post genre-merge, post
        # relaxation retry — so the conversation-mode scorer grades what the
        # engine actually saw. None = engine never ran; {} = ran, zero filters.
        filters_applied = decision.filters.model_dump() if decision.filters else {}
        tracer.record_local(
            "retrieve",
            {"count": len(movies), "ids": [m.id for m in movies],
             "filters_applied": filters_applied,
             "where_applied": getattr(engine, "last_where_applied", None),  # #88 (D5)
             "excluded_shown": len(getattr(engine, "last_excluded_ids", []) or [])},
        )
        # D16: entity focus lives in the thread now — the session-side
        # hand-copy that used to set it never reached the graph, so the
        # router's focused-entity context was silently dead in the UI path.
        focus = FocusedMovieEntity(
            id=movies[0].id, title=movies[0].title,
            release_year=movies[0].release_year, director=movies[0].director,
        ) if movies else None
        if config.routing_stack == "v1" and decision.filters and (
            decision.filters.runtime_max is not None
            or decision.filters.rating_min is not None
        ):
            # #106 review: v1's SQL cannot enforce the C6 predicates — the
            # emission is recorded, never silent (telemetry rule). The strip
            # itself lives at promotion; v1 behavior is unchanged.
            tracer.record_local("retrieve", {"unenforceable_v1_filters": {
                "runtime_max": decision.filters.runtime_max,
                "rating_min": decision.filters.rating_min,
            }})
        return {
            "retrieved_movies": movies,
            "shown_movie_ids": [m.id for m in movies],
            "shown_movie_titles": [m.title for m in movies],  # C14 state block
            "filters_applied": filters_applied,
            "focused_entity": focus,
        }

    def synthesize_node(state: MayaGraphState) -> dict:
        """CWA-grounded synthesis; usage recorded into the session budget (#8).

        Zero-retrieval RAG turns (#21) never reach the LLM: an empty
        ``<retrieved_movies>`` block is the highest hallucination-risk input
        (the model fills the void with popular titles and invented facts),
        so the deterministic branch answers with a grounded refinement
        question instead — model proposes, code disposes taken to its end.
        """
        decision = state.routing_decision
        query = decision.standalone_query or state.current_query
        movies = state.retrieved_movies
        if decision.requires_rag and not movies:
            response_text = _empty_retrieval_text(query)
            tracer.record_local(
                "synthesize",
                {"movies": 0, "retrieval_empty": True, "path": "deterministic"},
            )
            return {
                "final_response": response_text,
                "messages": [AIMessage(content=response_text)],
                "rolling_summary": _update_summary(state, decision),
            }
        history: list = list(state.messages)[:-1]
        response_text, usage = synthesizer.synthesize(query, decision, movies, history)
        # CWA verification + enforcement (#26-G): on no-retrieval turns a
        # flagged response is DISCARDED for the deterministic steer; on
        # retrieval turns foreign-title leaks are still reported (trace) —
        # the user keeps grounded titles, the violation is on record.
        # Runs on EMPTY context too (#21): with no allowed set, any bolded
        # title mention is a violation. The verifier is a MayaSynthesizer
        # method; fakes without it are clean.
        violations = []
        cwa_check = getattr(synthesizer, "cwa_violations", None)
        if cwa_check:
            violations = [v.mentioned_title for v in cwa_check(response_text, movies)]
        # #26-G: CWA ENFORCEMENT, not just detection. On a no-retrieval turn
        # any title mention is a hallucination (nothing is grounded); the
        # flagged response is discarded and replaced by the deterministic
        # steer. Detection without enforcement shipped hallucinated cards
        # four turns in a row in the walkthrough.
        if not movies and violations and not decision.requires_rag:
            tracer.record_local(
                "synthesize",
                {"cwa_gate": True, "discarded_titles": violations[:5]},
            )
            response_text = _no_retrieval_steer(state.current_query)
            violations = []  # the shipped response is title-free
        usage = _metered_usage_of(usage)  # review P3-2: doubles → marker path
        tokens_used = usage.prompt_tokens + usage.completion_tokens if usage else 0
        cost_usd = _meter_llm(usage, "synthesize")
        tracer.record_local(
            "synthesize",
            {"movies": len(movies), "tokens": tokens_used,
             "budget": limiter.check_current().verdict.value,
             "weekly_budget": (
                 budget_tracker.current_verdict().value
                 if budget_tracker is not None else "off"
             ),
             "cwa_violations": violations},
        )
        return {
            "final_response": response_text,
            "synthesis_usage": usage,
            "messages": [AIMessage(content=response_text)],
            "session_tokens": tokens_used,
            "session_cost_usd": cost_usd,
            "rolling_summary": _update_summary(state, decision),
        }

    def carryover_notice_node(state: MayaGraphState) -> dict:
        """#153 (v2): announce remembered filters that silently joined the SQL.

        The retrieve node injects the session's preferred genres when this
        turn declared none (#25). That injection is a code disposal, so the
        explanation is deterministic code too — never a model call (ADR 0005):
        it appends the transparency line for exactly the genres that joined
        (review 2026-10-04: never over-claim filters that did not run), whose
        escape-hatch question is the visitor's way out. Fires at most once per
        session (thread-persistent ``carryover_notice_shown``), only on turns
        that actually retrieved, and only when a preference genre really
        joined.
        """
        injected = injected_genres(
            state.filters_applied,
            state.routing_decision.filters if state.routing_decision else None,
        )
        if (
            not injected
            or state.carryover_notice_shown
            or not state.retrieved_movies
        ):
            return {}
        notice = v2_carryover_notice(injected)
        if not notice:
            return {}
        noticed = state.final_response + notice
        last = state.messages[-1] if state.messages else None
        # Same-id replacement: add_messages swaps the assistant message in
        # place, so the transcript carries the notice without a duplicate.
        # No usable id → the UI still reads final_response (the transcript
        # keeps the un-noticed text) — a fail-open the trace distinguishes
        # via transcript_replaced (telemetry rule: fail-open is recorded).
        replaced = isinstance(last, AIMessage) and bool(last.id)
        tracer.record_local(
            "carryover_notice",
            {"injected_genres": injected, "fired": True,
             "transcript_replaced": replaced},
        )
        update: dict = {"final_response": noticed, "carryover_notice_shown": True}
        if replaced:
            update["messages"] = [last.model_copy(update={"content": noticed})]
        return update

    def refusal_node(state: MayaGraphState) -> dict:
        """Deterministic refusal — guardrail text already in final_response."""
        tracer.record_local("refusal", {})
        return {"messages": [AIMessage(content=state.final_response)]}

    def pivot_node(state: MayaGraphState) -> dict:
        """Deterministic off-topic deflection (#8 OffTopicPivot, zero LLM)."""
        response_text = pivot.pivot_response(state.current_query)
        tracer.record_local("pivot", {})
        return {"final_response": response_text, "messages": [AIMessage(content=response_text)]}

    def route_after_guard(state: MayaGraphState) -> Literal["refusal", "route"]:
        guardrail = state.guardrail_result
        if guardrail and guardrail.verdict is GuardrailVerdict.BLOCKED:
            return "refusal"
        return "route"

    # --- #106: the v2 stack — funnel collapsed into the route node ----------

    def route_node_v2(state: MayaGraphState) -> dict:
        """One Understand call -> disposer -> projection; the ask is INLINE.

        The v2 route node replaces v1's route/probe/funnel trio: the model
        authors the clarifying question (C9), the disposer enforces every
        invariant (C7/C2/C12), and the Turn Decision on Narrowing Axes
        (C8) routes the turn. The preferences snapshot rides verbatim
        (PreferencesUpdate replace) so removals cannot be resurrected by
        the union reducer.
        """
        probe_count = state.probe_count
        u, notes, usage = router_v2.understand(
            state.current_query,
            state.session_preferences,
            shown_titles=state.shown_movie_titles,
            last_assistant=last_assistant_text(state),
            probe_count=probe_count,
        )
        usage = _metered_usage_of(usage)  # review P3-2: doubles → marker path
        understand_cost = _meter_llm(usage, "route_v2")
        for note in notes:  # telemetry rule: every invariant on the record
            tracer.record_local("route_v2", {"note": note})
        disposition = dispose(u, state.session_preferences, config)
        for note in disposition.notes:
            tracer.record_local("route_v2", {"note": note})
        decision = project_understanding(disposition.understanding, disposition.preferences)
        td = turn_decision(
            disposition.understanding, disposition.preferences, config,
            probe_count=probe_count,
        )
        # #113: record the passage UNCONDITIONALLY — guard notes only exist
        # on violations, so a clean ask/retrieve left no route_v2 trace and
        # the Feedback Window inferred "refusal" from the empty channel.
        tracer.record_local("route_v2", {"turn_decision": td.decision})
        update = {
            "routing_decision": decision,
            "session_preferences": PreferencesUpdate(
                prefs=disposition.preferences, replace=True
            ),
            "session_tokens": (
                usage.prompt_tokens + usage.completion_tokens if usage else 0
            ),
            "session_cost_usd": understand_cost,
        }
        if td.decision == "ask":  # C9: the model-authored question IS the reply
            question = disposition.understanding.clarifying_question or ""
            return {
                **update,
                "final_response": question,
                "messages": [AIMessage(content=question)],
                "turn_stage": "ask",
                "probe_count": probe_count + 1,
            }
        return update

    def route_after_router_v2(state: MayaGraphState) -> Literal["retrieve", "synthesize", "pivot", "trim"]:
        """C8 ladder outcomes -> graph targets. ``ask`` already answered
        itself in route_node_v2 (turn_stage), so it trims and ends — the
        funnel's probe/funnel nodes never run on the v2 stack."""
        if state.turn_stage == "ask":
            return "trim"
        decision = state.routing_decision
        if decision.intent is IntentType.OUT_OF_SCOPE:
            return "pivot"
        if not decision.requires_rag:
            return "synthesize"
        return "retrieve"

    def trim_node(state: MayaGraphState) -> dict:
        """#93/D16: the window rides Experiment Config (ADR 0004)."""
        return trim_message_window(state, config.message_window)

    graph = StateGraph(MayaGraphState)
    graph.add_node("begin_turn", begin_turn_node)
    graph.add_node("trim", trim_node)
    graph.add_node("guard_input", guard_input_node)
    graph.add_node("route", route_node_v2)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    # #153: the transparency node — deterministic, never a model call (ADR 0005).
    graph.add_node("carryover_notice", carryover_notice_node)
    graph.add_node("refusal", refusal_node)
    graph.add_node("pivot", pivot_node)

    graph.add_edge(START, "begin_turn")
    graph.add_edge("begin_turn", "guard_input")
    graph.add_conditional_edges("guard_input", route_after_guard)
    graph.add_conditional_edges("route", route_after_router_v2)
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "carryover_notice")
    graph.add_edge("carryover_notice", "trim")
    graph.add_edge("trim", END)
    graph.add_edge("refusal", "trim")
    graph.add_edge("pivot", "trim")

    return graph.compile(checkpointer=checkpointer)


# --- helpers (pure, module-level for testability) ---

def last_assistant_text(state) -> str | None:
    """#106/C14: Maya's last reply for the Understand payload (not a window)."""
    for message in reversed(state.messages):
        if isinstance(message, AIMessage) and message.text:
            return message.text
    return None


def _refusal_text(reason: str) -> str:
    return (
        "I can't help with that request. "
        + (f"({reason})" if reason else "")
        + "\nI'm Maya, a film curator — ask me about movies from 1970 to 2026!"
    )


_EMPTY_QUERY_ECHO_CAP = 120  #: Echo cap for the zero-retrieval response —
                             #: a hostile query must not balloon the reply.

#: #93/D16: the conversation message window kept inside the thread. The
#: tunable lives on ExperimentConfig.message_window; this is the pure helper
#: the trim node calls (module-level for testability).


def trim_message_window(state: MayaGraphState, window: int) -> dict:
    """Keeps the last ``window`` messages inside the thread.

    ``add_messages`` resolves ``RemoveMessage`` by id, so the window is
    maintained by the same reducer that appends — no separate list juggling.
    (Named trim_message_window, not trim_messages: langchain_core's
    ``messages.trim_messages`` has different semantics — token budget.)
    """
    overflow = state.messages[: max(len(state.messages) - window, 0)]
    return {"messages": [RemoveMessage(id=m.id) for m in overflow if m.id]}


def begin_turn_node(state: MayaGraphState) -> dict:
    """#93/D16: resets per-turn scratch before any pipeline node runs.

    What must NOT be reset lives outside this list: probe_count (the v2 ask
    budget carries ACROSS turns by design), plus every reducer-backed field
    (messages, shown_movie_ids, session_preferences, session_tokens).
    """
    return {
        "current_query": "",
        "guardrail_result": None,
        "routing_decision": None,
        "route_attempts": 0,
        "turn_stage": "",
        "retrieved_movies": [],
        "synthesis_usage": None,
        "final_response": "",
        "filters_applied": None,
    }


_SMUGGLED_MARKUP_RE = re.compile(r"</?\s*\w+\s*/?>|```.*?```", re.DOTALL)


def _metered_usage_of(raw: object) -> "LLMUsage | None":
    """Normalizes a caller-reported usage to a safely meterable LLMUsage.

    Review P3-2: doubles like ``Mock(spec=LLMUsage)`` pass the isinstance
    guard but explode on ``int()`` — one choke point decides usability for
    metering AND token accounting. Unusable → ``None`` (the ``_meter_llm``
    marker path), never a crash.
    """
    if isinstance(raw, LLMUsage):
        try:
            return LLMUsage(
                model=str(raw.model),
                prompt_tokens=int(raw.prompt_tokens),
                completion_tokens=int(raw.completion_tokens),
            )
        except (TypeError, ValueError, AttributeError):
            return None
    return None


def _empty_retrieval_text(query: str) -> str:
    """Deterministic zero-retrieval response (#21): grounded, probing, CWA-safe.

    No LLM call happens on this path, so no title can be hallucinated. The
    echoed query is markup-stripped and length-capped to stay inject-safe;
    the follow-up question is the first rung of the #22 narrowing funnel.
    """
    echo = _SMUGGLED_MARKUP_RE.sub(" ", query)
    echo = re.sub(r"\s{2,}", " ", echo).strip()
    if len(echo) > _EMPTY_QUERY_ECHO_CAP:
        echo = echo[:_EMPTY_QUERY_ECHO_CAP].rstrip() + "…"
    echoed = f' for "{echo}"' if echo else ""
    return (
        f"I searched the archive but couldn't find any movies matching that"
        f"{echoed}.\n\n"
        "Help me narrow it down: which decade are you in the mood for, and do "
        "you lean animation or live-action? You can also loosen a filter — "
        "for example, I know films up to PG-13, and telling me a genre or mood "
        "works wonders."
    )


def _no_retrieval_steer(query: str) -> str:
    """Deterministic conversational steer (#26-G): replaces a no-retrieval
    synthesis whose response the CWA verifier flagged for title mentions.

    The LLM response is DISCARDED, never shown — the user still gets a warm,
    grounded, inject-safe reply that steers toward a film request. Same echo
    sanitization contract as the #21 zero-retrieval response.
    """
    echo = _SMUGGLED_MARKUP_RE.sub(" ", query)
    echo = re.sub(r"\s{2,}", " ", echo).strip()
    if len(echo) > _EMPTY_QUERY_ECHO_CAP:
        echo = echo[:_EMPTY_QUERY_ECHO_CAP].rstrip() + "…"
    return (
        "Let's talk movies! Tell me what you're in the mood for — a genre, "
        "a mood, an era, a director — and I'll pull films from the shelf. "
        "For example: \"edge-of-your-seat sci-fi from the 2010s\"."
        + (f' (I heard: "{echo}")' if echo else "")
    )


def _update_summary(state: MayaGraphState, decision) -> str:
    """One-line rolling summary of the latest turn (kept deliberately cheap)."""
    prev = state.rolling_summary or ""
    turn = f"{decision.intent.value}: {decision.standalone_query}"
    return f"{prev} | {turn}".strip(" |")[-500:]
