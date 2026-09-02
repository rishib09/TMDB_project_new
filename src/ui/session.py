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
from src.maya.probing import preference_chips
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
            # #26-B: the dataset's own genres are the genre-guard vocabulary.
            MayaRouter(self.config, genre_vocabulary=self.db.distinct_genres()),
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
        """One full Maya turn: guard → route/funnel → retrieve → synthesize.

        The turn_log row is built ATOMICALLY by ``_build_turn_row`` from the
        graph's output alone (#26-A) — chip metadata, response, movie count
        and token count can never come from different turns, even on funnel
        turns where the router never ran (``routing_decision`` is None).
        """
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
                "offered_genre_options": self.conversation.offered_genre_options,
            },
            # cloud tracing was silently inactive in the UI before #9 — wired
            # every turn now so the trace id and the run actually correlate
            config={
                "callbacks": self.tracer.callbacks(),
                "metadata": self.tracer.metadata(),  # v4 session grouping
            },
        )
        row = self._build_turn_row(
            out,
            query=query,
            trace_id=trace_id,
            rag_version=self.rag_version,
            new_traces=slice_new_traces(ring_before, self.tracer.traces()),
            prev_tokens=self.conversation.session_tokens,
        )
        movies = out.get("retrieved_movies", [])
        self.last_movies = movies
        # Guided narrowing (#22): probe answers extracted this turn persist
        # in session state; probe turns carry no movies and no synthesis cost.
        # #26-E: a fresh-start turn clears preferences via the reducer.
        self.conversation.session_preferences = out.get(
            "session_preferences", self.conversation.session_preferences
        )
        self.conversation.probe_count = out.get("probe_count", self.conversation.probe_count)
        self.conversation.funnel_active = out.get(
            "funnel_active", self.conversation.funnel_active
        )
        self.conversation.offered_genre_options = out.get(
            "offered_genre_options", []
        )
        self.conversation.add_turn(
            query, row["response"], movies, out.get("routing_decision"),
            tokens_used=row["tokens"],
        )
        self.turn_log.append(row)

    @staticmethod
    def _build_turn_row(
        out: dict,
        *,
        query: str,
        trace_id: str,
        rag_version: str,
        new_traces: list[dict],
        prev_tokens: int,
    ) -> dict:
        """Pure, atomic turn_log row (#26-A) — unit-testable without a graph.

        Funnel-owned turns (probe/confirm/genre-confirm) end without a
        routing decision: the deterministic funnel stage becomes the intent
        label and the path reads "funnel". Previously ``decision.intent``
        crashed here mid-turn, leaving a stale row that made the UI render
        one turn's chip against another turn's response.
        """
        decision = out.get("routing_decision")
        stage = out.get("turn_stage", "")
        response = out["final_response"]
        tokens = max(out.get("session_tokens", 0) - prev_tokens, 0)
        route_traces = [t for t in new_traces if t["node"] == "route"]
        if decision is None:
            intent = f"FUNNEL_{(stage or 'probe').upper()}"
            confidence = 1.0  # deterministic — no model involved
            path = "funnel"
        else:
            intent = decision.intent.value
            confidence = decision.confidence
            path = "funnel" if stage == "retrieve" else MayaSession._path_taken(route_traces)
        prefs = out.get("session_preferences")
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "trace_id": trace_id,
            "rag_version": rag_version,
            "query": query,
            "intent": intent,
            "confidence": confidence,
            "path": path,
            "stage": stage,
            "attempts": len(route_traces),
            "n_movies": len(out.get("retrieved_movies", [])),
            "tokens": tokens,
            "response": response,
            "probe": any(t["node"] == "probe" for t in new_traces),
            "narrowing": preference_chips(prefs) if prefs else [],
            "filters": MayaSession._filter_chips(decision),
        }

    @staticmethod
    def _filter_chips(decision) -> list[str]:
        """Active SQL filters for this turn's metadata line (#26-F)."""
        if decision is None or decision.filters is None:
            return []
        f = decision.filters
        chips: list[str] = []
        if f.genres:
            mode = f" ({f.genre_match})" if len(f.genres) > 1 else ""
            chips.append("genres: " + ", ".join(f.genres) + mode)
        if f.exact_year:
            chips.append(str(f.exact_year))
        elif f.year_min or f.year_max:
            chips.append(f"{f.year_min or '…'}–{f.year_max or '…'}")
        if f.director:
            chips.append(f"dir. {f.director}")
        if f.cast_member:
            chips.append(f"cast {f.cast_member}")
        if f.person:
            chips.append(f"person: {f.person}")
        chips.extend(f"no {g}" for g in f.excluded_genres)
        chips.extend(f"no {a}" for a in f.excluded_actors)
        return chips

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
