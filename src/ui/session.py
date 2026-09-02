"""Streamlit session state for Maya (issue #7): one graph, live knobs.

The session owns the compiled LangGraph, the conversational memory, the
tracer, and the ExperimentConfig. Knob changes rebuild the graph once
(config is an ADR-0004 experiment surface — the sidebar edits it live).
Pure logic lives here so it is testable without a Streamlit runtime.
"""

from datetime import UTC, datetime

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.domain.config import ExperimentConfig, PresetType
from src.domain.memory import ConversationState
from src.feedback.langfuse_score import push_feedback_score
from src.feedback.store import FeedbackStore
from src.graph.orchestrator import build_maya_graph
from src.indexing.vector_store import MovieVectorStore
from src.maya.agent import MayaSynthesizer
from src.maya.guardrails import SessionTokenLimiter, WeeklyBudgetTracker
from src.maya.router import MayaRouter
from src.observability.tracer import DualModeObservabilityManager
from src.retrieval.hybrid_engine import HybridRetrievalEngine
from src.storage.database import MovieDatabase

ADMIN_COMMAND = "/admin"


def _to_lc_messages(history) -> list[BaseMessage]:
    """Projects domain ChatMessage history onto LangChain message types."""
    out: list[BaseMessage] = []
    for msg in history:
        if msg.role == "user":
            out.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            out.append(AIMessage(content=msg.content))
    return out


def slice_new_traces(ring_before: int, traces: list[dict]) -> list[dict]:
    """Pure helper (issue #18): only the traces produced during this turn.

    The tracer ring accumulates across turns; counting route traces from the
    whole ring made the transparency chip show stale cross-turn counts.
    """
    return traces[ring_before:]


class MayaSession:
    """Per-browser-session bundle: graph, memory, tracer, config."""

    def __init__(self) -> None:
        self.config = ExperimentConfig()
        self.conversation = ConversationState()
        self.tracer = DualModeObservabilityManager(session_id=f"ui-{datetime.now(UTC):%H%M%S}")
        self.limiter = SessionTokenLimiter()
        self.db = MovieDatabase("data/tmdb_movies.db")  # shared: engine + budget sink
        self.budget_tracker = WeeklyBudgetTracker(self.db)  # weekly $ ceiling (#8)
        self.view = "Chat"  # sidebar navigation: Chat | Evals | Traces
        self.feedback_log: dict[int, int] = {}  # assistant-turn index → ±1 (thumbs)
        self.feedback_store = FeedbackStore()  # SQLite persistence (#9)
        self.rag_version = "v1_1_enriched"  # matches _build_graph engine wiring
        self.admin_mode = False
        self.config_version = 0  # bumped on preset apply → knob widgets remount
        self.turn_log: list[dict] = []  # one row per turn for badges/trace
        self.last_movies = []  # MovieRecords from the most recent retrieval
        self._graph_sig = ""
        self.graph = self._build_graph()

    # --- graph lifecycle ---

    def _build_graph(self):
        engine = HybridRetrievalEngine(
            db=self.db,
            vector_store=MovieVectorStore("data/chroma_db"),
            rag_version=self.rag_version,
            hybrid_alpha=self.config.hybrid_alpha,
            reranker_enabled=self.config.reranker_enabled,
            reranker_model=self.config.reranker_model,
        )
        return build_maya_graph(
            self.config,
            MayaRouter(self.config),
            engine,
            MayaSynthesizer(self.config),
            self.tracer,
            limiter=self.limiter,
            budget_tracker=self.budget_tracker,
        )

    def _graph_signature(self) -> str:
        """Engine + routing knobs that require a graph rebuild when changed."""
        return "|".join(
            str(v) for v in (
                self.config.router_model, self.config.synthesis_model,
                self.config.temperature, self.config.hybrid_alpha,
                self.config.reranker_enabled, self.config.reranker_model,
                self.config.retrieval_top_k, self.config.route_max_attempts,
                self.config.reasoning_effort,
            )
        )

    def ensure_graph(self):
        """Rebuilds the graph only when a rebuild-relevant knob changed."""
        sig = self._graph_signature()
        if sig != self._graph_sig:
            self.graph = self._build_graph()
            self._graph_sig = sig
        return self.graph

    def replace_config(self, new_config: ExperimentConfig) -> None:
        self.config = new_config
        self.ensure_graph()

    def apply_preset(self, preset: PresetType) -> None:
        self.config.apply_preset(preset)
        self.config_version += 1  # remount knob widgets with the preset values
        self.ensure_graph()

    # --- conversation turn ---

    def turn(self, query: str) -> None:
        """One full Maya turn: guard → route → retrieve → synthesize."""
        graph = self.ensure_graph()
        ring_before = len(self.tracer.traces())
        trace_id = self.tracer.new_turn_trace()  # per-turn id for feedback (#9)
        history = _to_lc_messages(self.conversation.messages)
        out = graph.invoke(
            {
                "messages": [*history, HumanMessage(content=query)],
                "session_preferences": self.conversation.session_preferences,
                "session_tokens": self.conversation.session_tokens,
                "probe_count": self.conversation.probe_count,
                "funnel_active": self.conversation.funnel_active,
            },
            # cloud tracing was silently inactive in the UI before #9 — wired
            # every turn now so the trace id and the run actually correlate
            config={"callbacks": self.tracer.callbacks()},
        )
        decision = out["routing_decision"]
        movies = out.get("retrieved_movies", [])
        response = out["final_response"]
        self.last_movies = movies
        tokens = out.get("session_tokens", 0) - self.conversation.session_tokens
        # Guided narrowing (#22): probe answers extracted this turn persist
        # in session state; probe turns carry no movies and no synthesis cost.
        self.conversation.session_preferences = out.get(
            "session_preferences", self.conversation.session_preferences
        )
        self.conversation.probe_count = out.get("probe_count", self.conversation.probe_count)
        self.conversation.funnel_active = out.get(
            "funnel_active", self.conversation.funnel_active
        )
        self.conversation.add_turn(
            query, response, movies, decision, tokens_used=max(tokens, 0)
        )
        route_traces = [
            t
            for t in slice_new_traces(ring_before, self.tracer.traces())
            if t["node"] == "route"
        ]
        self.turn_log.append({
            "timestamp": datetime.now(UTC).isoformat(),
            "trace_id": trace_id,
            "rag_version": self.rag_version,
            "query": query,
            "intent": decision.intent.value,
            "confidence": decision.confidence,
            "path": self._path_taken(route_traces),
            "attempts": sum(1 for t in route_traces),
            "n_movies": len(movies),
            "tokens": max(tokens, 0),
            "response": response,
            "probe": len(self.tracer.traces()) > 0 and any(
                t["node"] == "probe"
                for t in slice_new_traces(ring_before, self.tracer.traces())
            ),
        })

    @staticmethod
    def _path_taken(route_traces: list[dict]) -> str:
        if not route_traces:
            return "refusal"
        return "reroute" if len(route_traces) > 1 else "single-route"

    def record_feedback(self, turn_index: int, value: int) -> bool:
        """Persists a thumb rating and pushes it to Langfuse (#9).

        UPSERT semantics: changing the thumb on the same turn updates the
        row and re-pushes the score (deterministic score_id) — never
        duplicates. Returns True when the cloud push succeeded.
        """
        if value not in (1, -1):
            raise ValueError(f"rating must be +1 or -1, got {value!r}")
        if not (0 <= turn_index < len(self.turn_log)):
            raise IndexError(f"turn_index {turn_index} out of range")
        row = self.turn_log[turn_index]
        self.feedback_store.record(
            row["trace_id"], value, row["rag_version"], intent=row["intent"]
        )
        self.feedback_log[turn_index] = value
        return push_feedback_score(row["trace_id"], value)

    @staticmethod
    def is_admin_command(query: str) -> bool:
        return query.strip().lower() == ADMIN_COMMAND


def get_session() -> MayaSession:
    """Streamlit session_state singleton."""
    if "maya_session" not in st.session_state:
        st.session_state.maya_session = MayaSession()
    return st.session_state.maya_session
