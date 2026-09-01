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

from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.domain.config import ExperimentConfig
from src.domain.memory import ConversationState
from src.domain.routing import IntentType
from src.graph.state import MayaGraphState
from src.maya.agent import MayaSynthesizer
from src.maya.guardrails import (
    GuardrailVerdict,
    InjectionFilter,
    OffTopicPivot,
    SessionTokenLimiter,
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
        tracer.record_local("guard_input", {"verdict": injection.verdict.value})
        # SUSPICIOUS verdict (stripped markup): proceed with the sanitized query.
        sanitized = injection.sanitized_query or query
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
        tracer.record_local(
            "route",
            {
                "attempt": attempts,
                "intent": decision.intent.value,
                "confidence": decision.confidence,
                "requires_rag": decision.requires_rag,
                "is_fallback": decision.is_fallback,
            },
        )
        return {"routing_decision": decision, "route_attempts": attempts}

    def retrieve_node(state: MayaGraphState) -> dict:
        """Hybrid retrieval (#4): SQL path or RRF fusion, per routing decision."""
        decision = state.routing_decision
        results = engine.retrieve(
            query=decision.standalone_query,
            routing=decision,
            top_k=config.retrieval_top_k,
        )
        movies = [r.movie for r in results]
        tracer.record_local("retrieve", {"count": len(movies), "ids": [m.id for m in movies]})
        return {"retrieved_movies": movies, "shown_movie_ids": [m.id for m in movies]}

    def synthesize_node(state: MayaGraphState) -> dict:
        """CWA-grounded synthesis; usage recorded into the session budget (#8)."""
        decision = state.routing_decision
        query = decision.standalone_query or state.current_query
        movies = state.retrieved_movies
        history: list = list(state.messages)[:-1]
        response_text, usage = synthesizer.synthesize(query, decision, movies, history)
        # CWA verification: detect (never silently fix) foreign-title leaks.
        # The verifier is a MayaSynthesizer method; fakes without it are clean.
        violations = []
        cwa_check = getattr(synthesizer, "cwa_violations", None)
        if cwa_check and movies:
            violations = [v.mentioned_title for v in cwa_check(response_text, movies)]
        tokens_used = usage.prompt_tokens + usage.completion_tokens
        budget_status = limiter.record(usage.model, usage.prompt_tokens, usage.completion_tokens)
        tracer.record_local(
            "synthesize",
            {"movies": len(movies), "tokens": tokens_used, "budget": budget_status.value,
             "cwa_violations": violations},
        )
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

    def route_after_guard(state: MayaGraphState) -> Literal["refusal", "route"]:
        guardrail = state.guardrail_result
        if guardrail and guardrail.verdict is GuardrailVerdict.BLOCKED:
            return "refusal"
        return "route"

    def route_after_router(state: MayaGraphState) -> Literal["route", "retrieve", "synthesize", "pivot"]:
        """Bounded re-route cycle (#12): retry while the router signals fallback.

        The trigger is deterministic — ``is_fallback`` is set by the router's
        own code (confidence < threshold or API error), never by the model
        (ADR 0005). Bounded by ``config.route_max_attempts``; when attempts
        are exhausted the (degraded but safe) heuristic decision proceeds.
        """
        decision = state.routing_decision
        if decision.is_fallback and state.route_attempts < config.route_max_attempts:
            return "route"
        if decision.intent is IntentType.OUT_OF_SCOPE:
            return "pivot"
        if not decision.requires_rag:
            return "synthesize"
        return "retrieve"

    graph = StateGraph(MayaGraphState)
    graph.add_node("guard_input", guard_input_node)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("refusal", refusal_node)
    graph.add_node("pivot", pivot_node)

    graph.add_edge(START, "guard_input")
    graph.add_conditional_edges("guard_input", route_after_guard)
    graph.add_conditional_edges("route", route_after_router)
    # The route→route cycle is implicit: route_after_router may return "route".
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)
    graph.add_edge("refusal", END)
    graph.add_edge("pivot", END)

    return graph.compile()


# --- helpers (pure, module-level for testability) ---

def _refusal_text(reason: str) -> str:
    return (
        "I can't help with that request. "
        + (f"({reason})" if reason else "")
        + "\nI'm Maya, a film curator — ask me about movies from 1970 to 2026!"
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
