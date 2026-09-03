"""Maya orchestrator (issue #5): compiled LangGraph StateGraph workflow.

Topology::

    START → guard_input ──blocked──→ refusal ────────────────→ END
               │clean
               ▼
              route ──OUT_OF_SCOPE────────→ pivot ──────────→ END
               │     └─no retrieval───────→ synthesize ─────→ END
               ▼
            retrieve ──────────────────────→ synthesize ─────→ END

Every component is already built and tested (#3 router, #4 hybrid engine,
#8 guardrails); this module only wires them as LangGraph nodes with
conditional edges — no custom dispatch, no custom state management.

The graph compiles against ``MayaGraphState`` (Pydantic schema, see
``state.py``), so nodes receive the model instance and return partial dict
updates that LangGraph applies through the Annotated reducers.
"""

import re
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.domain.config import ExperimentConfig
from src.domain.memory import ConversationState, UserSessionPreferences, merge_preferences
from src.domain.routing import IntentType, QueryRoutingDecision
from src.graph.state import MayaGraphState
from src.maya.agent import MayaSynthesizer
from src.maya.guardrails import (
    GuardrailResult,
    GuardrailVerdict,
    InjectionFilter,
    OffTopicPivot,
    SessionTokenLimiter,
    WeeklyBudgetTracker,
)
from src.maya.probing import (
    build_filter_carryover_notice,
    build_funnel_query,
    build_probe_response,
    extract_probe_answers,
    handle_probe_answer,
    is_fresh_start,
    match_genre_pick,
    next_funnel_step,
    should_probe,
)
from src.maya.router import MayaRouter
from src.observability.tracer import DualModeObservabilityManager
from src.retrieval.hybrid_engine import HybridRetrievalEngine


def build_maya_graph(
    config: ExperimentConfig,
    router: MayaRouter,
    engine: HybridRetrievalEngine,
    synthesizer: MayaSynthesizer,
    tracer: DualModeObservabilityManager,
    limiter: SessionTokenLimiter | None = None,
    budget_tracker: WeeklyBudgetTracker | None = None,
) -> CompiledStateGraph:
    """Compiles the Maya workflow with injected components (DI-friendly).

    All heavy collaborators (router, engine, synthesizer) are constructed by
    the caller — the graph itself owns only sequencing and conditional edges,
    which is what makes the unit tests mock-free at the component level.
    """
    limiter = limiter or SessionTokenLimiter()
    injection_filter = InjectionFilter()
    pivot = OffTopicPivot()

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
        # at the ONE choke point every turn passes — covering funnel turns and
        # post-retrieval turns alike. route_after_guard sees funnel_active=False
        # (LangGraph applies updates before conditional edges) so the funnel
        # cannot re-own the turn; routing proceeds on a clean slate.
        if is_fresh_start(sanitized):
            tracer.record_local("guard_input", {"fresh_start": True})
            return {
                "guardrail_result": injection,
                "current_query": sanitized,
                "session_preferences": UserSessionPreferences(reset_requested=True),
                "funnel_active": False,
                "offered_genre_options": [],
            }
        return {"guardrail_result": injection, "current_query": sanitized}

    def route_node(state: MayaGraphState) -> dict:
        """Structured routing via MayaRouter (#3); re-entry = iteration N+1.

        On re-route attempts (bounded cycle from #12), the corrective feedback
        is built from the previous failed decision and injected into the
        router prompt — the measured fix for low-confidence failures.
        """
        attempts = state.route_attempts + 1
        feedback = None
        if state.routing_decision is not None:
            prev = state.routing_decision
            feedback = (
                f"Your previous routing attempt returned intent={prev.intent.value} "
                f"with confidence {prev.confidence:.2f}, which fell below the "
                "confidence threshold and was rejected. Re-read the user query "
                "carefully (check for pre-1970 references, superlatives, filter "
                "criteria, and greetings) and return a better-reasoned decision."
            )
        decision = router.route(
            state.current_query,
            _to_conversation_state(state),
            feedback=feedback,
        )
        # Guided narrowing (#22/#24): mood/audience extracted by the router
        # itself (open vocabulary), with the deterministic vocab as fallback.
        signals = UserSessionPreferences(
            preferred_mood=(decision.mood or "").strip(),
            audience=(decision.audience or "").strip(),
        )
        if not signals.preferred_mood or not signals.audience:
            vocab = extract_probe_answers(state.current_query)
            signals = UserSessionPreferences(
                preferred_mood=signals.preferred_mood or vocab.preferred_mood,
                audience=signals.audience or vocab.audience,
            )
        tracer.record_local(
            "route",
            {
                "attempt": attempts,
                "intent": decision.intent.value,
                "confidence": decision.confidence,
                "requires_rag": decision.requires_rag,
                "is_fallback": decision.is_fallback,
                "probe_answers": signals.answered_axes(),
            },
        )
        return {
            "routing_decision": decision,
            "route_attempts": attempts,
            "session_preferences": signals,
        }

    def probe_node(state: MayaGraphState) -> dict:
        """Guided narrowing (#22): one deterministic question, zero LLM cost.

        The answer extraction rides on the user's NEXT message: probe
        answers are vocabulary-matched from it in the funnel node (#23),
        so no extra LLM call is needed to understand the reply.
        """
        response_text = build_probe_response(state.session_preferences, state.current_query)
        tracer.record_local(
            "probe",
            {"probe_count": state.probe_count + 1, "query": state.current_query},
        )
        return {
            "final_response": response_text,
            "messages": [AIMessage(content=response_text)],
            "probe_count": state.probe_count + 1,  # session-persisted running total
            "funnel_active": True,  # next message belongs to the funnel (#23)
            "turn_stage": "probe",  # #26-A: UI row stays complete without the router
            "rolling_summary": _update_summary(state, state.routing_decision),
        }

    def funnel_node(state: MayaGraphState) -> dict:
        """Owns the reply to a probe/confirm (#23) — router only on fallthrough.

        Stage machine (#25): pending genre picks are matched deterministically
        against the offered candidates; confirmations retrieve immediately;
        otherwise the router acts as a PURE EXTRACTOR (#24 — its intent
        classification is ignored, so misrouting cannot derail the funnel)
        with the deterministic vocab as fallback. Anything the funnel can't
        own falls through to normal routing — OUT_OF_SCOPE pivots suppressed
        for exactly this turn.
        """
        query = state.current_query
        prefs = state.session_preferences
        outcome = None  # #25 latent crash: options pending + reply not a pick

        # 1. Genre pick pending? (#25) deterministic multiple-choice matching.
        if state.offered_genre_options:
            picks = match_genre_pick(query, state.offered_genre_options)
            if picks is not None:
                outcome = next_funnel_step(merge_preferences(prefs, UserSessionPreferences(
                    preferred_genres=picks, genre_confirmation_done=True,
                )), state.probe_count, query)
                tracer.record_local("probe", {"stage": "genre_pick", "picked": picks})

        # 2. Explicit confirmation → retrieve now; otherwise extract + progress.
        if outcome is None:
            outcome = handle_probe_answer(
                query, prefs, state.probe_count,
                prefs_update=_extract_signals(state, router),
            )

        if outcome.action == "retrieve":
            merged = outcome.prefs_update or prefs
            synthetic = QueryRoutingDecision(
                intent=IntentType.SEMANTIC_SEARCH,
                confidence=1.0,
                standalone_query=build_funnel_query(merged),
                requires_rag=True,
                reasoning="funnel confirmed retrieval (#23)",
            )
            tracer.record_local(
                "probe",
                {"stage": "retrieve", "axes": merged.answered_axes()},
            )
            # from_funnel rides along so synthesize_node can append the #26-E
            # carry-over announcement to this first post-funnel recommendation.
            return {
                "funnel_active": False,
                "routing_decision": synthetic,
                "session_preferences": merged,
                "offered_genre_options": [],
                "from_funnel": True,
                "turn_stage": "retrieve",
            }
        if outcome.action == "fallthrough":
            tracer.record_local("probe", {"stage": "fallthrough"})
            # NOTE: funnel stays ACTIVE — a user who ignores one probe may
            # still say "go ahead" next turn (#23 walkthrough defect); the
            # funnel is a cheap pre-router filter, so lingering is harmless.
            return {"from_funnel": True, "turn_stage": "fallthrough",
                    "offered_genre_options": outcome.offered_genre_options}
        # probe | confirm | confirm_genres → deterministic response, end turn
        tracer.record_local(
            "probe",
            {
                "stage": outcome.action,
                "probe_count": state.probe_count + (1 if outcome.action == "probe" else 0),
            },
        )
        return {
            "final_response": outcome.response,
            "messages": [AIMessage(content=outcome.response)],
            "probe_count": state.probe_count + (1 if outcome.action == "probe" else 0),
            "session_preferences": outcome.prefs_update,
            "funnel_active": True,
            "offered_genre_options": outcome.offered_genre_options,
            "turn_stage": outcome.action,  # #26-A
        }

    def retrieve_node(state: MayaGraphState) -> dict:
        """Hybrid retrieval (#4): SQL path or RRF fusion, per routing decision.

        Probed mood/audience (#22) enrich the semantic query text — the
        embedding model handles natural-language flavor natively — while
        genres/directors stay as the router's deterministic SQL filters.
        """
        decision = state.routing_decision
        prefs = state.session_preferences
        flavor = ", ".join(
            filter(
                None,
                (
                    f"mood: {prefs.preferred_mood}" if prefs.preferred_mood else "",
                    f"audience: {prefs.audience}" if prefs.audience else "",
                ),
            )
        )
        query = decision.standalone_query
        if flavor:
            query = f"{query} ({flavor})"
        # Confirmed funnel genres (#25) drive deterministic genre filters.
        if prefs.preferred_genres and not (
            decision.filters and decision.filters.genres
        ):
            from src.domain.routing import MetadataFilterCriteria

            filters = decision.filters or MetadataFilterCriteria()
            decision = decision.model_copy(update={"filters": filters.model_copy(update={
                "genres": prefs.preferred_genres,
                "genre_match": "all" if len(prefs.preferred_genres) > 1 else "any",
            })})
        results = engine.retrieve(
            query=query,
            routing=decision,
            top_k=config.retrieval_top_k,
        )
        # Intersection too narrow? (#25) retry ANY-match, relaxation on record.
        if not results and decision.filters and decision.filters.genre_match == "all":
            relaxed = decision.model_copy(update={"filters": decision.filters.model_copy(
                update={"genre_match": "any"}
            )})
            results = engine.retrieve(
                query=query,
                routing=relaxed,
                top_k=config.retrieval_top_k,
            )
            tracer.record_local("retrieve", {"genre_match_relaxed": True})
        movies = [r.movie for r in results]
        tracer.record_local("retrieve", {"count": len(movies), "ids": [m.id for m in movies]})
        return {"retrieved_movies": movies, "shown_movie_ids": [m.id for m in movies]}

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
        tokens_used = usage.prompt_tokens + usage.completion_tokens
        budget_status = limiter.record(usage.model, usage.prompt_tokens, usage.completion_tokens)
        # Weekly $ accounting (#8): one row per LLM call, cost-estimated.
        weekly_status = None
        if budget_tracker is not None:
            weekly_status = budget_tracker.record(
                usage.model, usage.prompt_tokens, usage.completion_tokens
            )
        tracer.record_local(
            "synthesize",
            {"movies": len(movies), "tokens": tokens_used, "budget": budget_status.value,
             "weekly_budget": weekly_status.value if weekly_status else "off",
             "cwa_violations": violations},
        )
        # #26-E: the first recommendation after funnel narrowing announces the
        # filters that REMAIN ACTIVE, with the deterministic escape hatch.
        if state.from_funnel and movies:
            notice = build_filter_carryover_notice(state.session_preferences)
            if notice:
                response_text += notice
                tracer.record_local("synthesize", {"carryover_notice": True})
        return {
            "final_response": response_text,
            "synthesis_usage": usage,
            "messages": [AIMessage(content=response_text)],
            "session_tokens": tokens_used,
            "rolling_summary": _update_summary(state, decision),
        }

    def refusal_node(state: MayaGraphState) -> dict:
        """Deterministic refusal — guardrail text already in final_response."""
        tracer.record_local("refusal", {})
        return {"messages": [AIMessage(content=state.final_response)]}

    def pivot_node(state: MayaGraphState) -> dict:
        """Deterministic off-topic deflection (#8 OffTopicPivot, zero LLM)."""
        response_text = pivot.pivot_response(state.current_query)
        tracer.record_local("pivot", {})
        return {"final_response": response_text, "messages": [AIMessage(content=response_text)]}

    def route_after_guard(state: MayaGraphState) -> Literal["refusal", "funnel", "route"]:
        guardrail = state.guardrail_result
        if guardrail and guardrail.verdict is GuardrailVerdict.BLOCKED:
            return "refusal"
        if state.funnel_active:
            return "funnel"
        return "route"

    def route_after_funnel(state: MayaGraphState) -> Literal["route", "retrieve", "__end__"]:
        """#23/#25: probe & confirm responses END the turn — never re-route.

        Walkthrough-defect fix: without the END branch the deterministic
        probe/confirm/genre-confirmation response fell through to ``route``,
        letting a second routing pass overwrite it (the 'edge of the seat'
        GREETING overwrite). Fallthrough still routes on.
        """
        if state.routing_decision is not None:  # funnel confirmed retrieval
            return "retrieve"
        if state.final_response and state.funnel_active:  # probe/confirm ready
            return END
        return "route"  # fallthrough — normal routing takes over

    def route_after_router(state: MayaGraphState) -> Literal["route", "retrieve", "synthesize", "pivot", "probe"]:
        """Bounded re-route cycle (#12) + guided narrowing gate (#22).

        The re-route trigger is deterministic — ``is_fallback`` is set by
        the router's own code (confidence < threshold or API error), never
        by the model (ADR 0005). Probing is a code policy, not a prompt
        suggestion: broad filterless requests ask one narrowing question
        (bounded by MAX_PROBE_TURNS) instead of guessing a movie dump.
        """
        decision = state.routing_decision
        if decision.is_fallback and state.route_attempts < config.route_max_attempts:
            return "route"
        if decision.intent is IntentType.OUT_OF_SCOPE:
            # #23: a message that just fell through the funnel may be an
            # answer to our own question — converse, never pivot.
            if state.from_funnel:
                return "synthesize"
            return "pivot"
        if not decision.requires_rag:
            return "synthesize"
        if should_probe(decision, state.session_preferences, state.probe_count):
            return "probe"
        return "retrieve"

    graph = StateGraph(MayaGraphState)
    graph.add_node("guard_input", guard_input_node)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("refusal", refusal_node)
    graph.add_node("pivot", pivot_node)
    graph.add_node("probe", probe_node)
    graph.add_node("funnel", funnel_node)

    graph.add_edge(START, "guard_input")
    graph.add_conditional_edges("guard_input", route_after_guard)
    graph.add_conditional_edges("funnel", route_after_funnel)
    graph.add_conditional_edges("route", route_after_router)
    # The route→route cycle is implicit: route_after_router may return "route".
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)
    graph.add_edge("refusal", END)
    graph.add_edge("pivot", END)
    graph.add_edge("probe", END)

    return graph.compile()


# --- helpers (pure, module-level for testability) ---

def _refusal_text(reason: str) -> str:
    return (
        "I can't help with that request. "
        + (f"({reason})" if reason else "")
        + "\nI'm Maya, a film curator — ask me about movies from 1970 to 2026!"
    )


#: Echo cap for the zero-retrieval response — a hostile query must not be
#: able to balloon the deterministic reply.
_EMPTY_QUERY_ECHO_CAP = 120
_SMUGGLED_MARKUP_RE = re.compile(r"</?\s*\w+\s*/?>|```.*?```", re.DOTALL)


def _extract_signals(state: "MayaGraphState", router) -> "UserSessionPreferences | None":
    """Router-as-extractor (#24): intent IGNORED, only mood/audience consumed.

    The funnel decides actions deterministically; the LLM only reads meaning.
    Router failure → None (handle_probe_answer falls back to the vocab).
    """
    try:
        decision = router.route(state.current_query, _to_conversation_state(state))
    except Exception:  # noqa: BLE001 — extraction must never break the funnel
        return None
    if not (decision.mood or decision.audience):
        return None
    return UserSessionPreferences(
        preferred_mood=decision.mood.strip(),
        audience=decision.audience.strip(),
    )


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


def _to_conversation_state(state: MayaGraphState) -> ConversationState:
    """Projection of the graph state onto the router's ConversationState input."""
    conversation = ConversationState(
        session_tokens=state.session_tokens,
        focused_entity=state.focused_entity,
        focused_person=state.focused_person,
    )
    conversation.session_preferences = state.session_preferences or conversation.session_preferences
    return conversation


def _update_summary(state: MayaGraphState, decision) -> str:
    """One-line rolling summary of the latest turn (kept deliberately cheap)."""
    prev = state.rolling_summary or ""
    turn = f"{decision.intent.value}: {decision.standalone_query}"
    return f"{prev} | {turn}".strip(" |")[-500:]
