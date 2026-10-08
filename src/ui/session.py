"""Streamlit session state for Maya (issue #7): one graph, live knobs.

The session owns the compiled LangGraph, the conversational memory, the
tracer, and the ExperimentConfig. Knob changes rebuild the graph once
(config is an ADR-0004 experiment surface — the sidebar edits it live).
Pure logic lives here so it is testable without a Streamlit runtime.
"""

import logging
import os
import sqlite3
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.domain.config import ExperimentConfig, PresetType
from src.domain.memory import ConversationState, UserSessionPreferences
from src.domain.routing import IntentType, QueryRoutingDecision
from src.domain.usage import LLMUsage
from src.feedback.inbox import (
    REPORTS_PER_SESSION,
    WINDOW_TURNS,
    ReportResult,
    format_rating_comment,
    format_report_comment,
    post_inbox_comment,
    validate_report,
)
from src.feedback.langfuse_score import push_feedback_score, push_report_comment
from src.feedback.store import FeedbackStore
from src.graph.orchestrator import build_maya_graph
from src.indexing.embeddings import collection_name, lab_collection_names, provider_from_profile
from src.indexing.vector_store import MovieVectorStore
from src.maya.agent import MayaSynthesizer
from src.maya.guardrails import (
    GuardrailResult,
    GuardrailVerdict,
    SessionCostLimiter,
    WeeklyBudgetTracker,
)
from src.maya.probing import preference_chips
from src.maya.router import MayaRouter
from src.observability.tracer import DualModeObservabilityManager
from src.retrieval.hybrid_engine import HybridRetrievalEngine
from src.storage.database import MovieDatabase

logger = logging.getLogger(__name__)

# #119: the checkpointer msgpack-serializes these domain types; register them
# explicitly so the deprecation warning is gone and a future strict
# langgraph-checkpoint cannot degrade session state to plain dicts. Pass the
# TYPES (the lib normalizes to exact (module, qualname) keys — dotted strings
# would silently never match). langgraph's own SAFE_MSGPACK_TYPES stay allowed
# regardless of this list. LLMUsage joins the original five after #123.
# Review: exported as a tuple — the serde registration AND every coverage test
# derive from this one list, so a newly checkpointed type is a single edit.
_CHECKPOINT_ALLOWLIST: tuple[type, ...] = (
    GuardrailVerdict,
    GuardrailResult,
    IntentType,
    QueryRoutingDecision,
    UserSessionPreferences,
    LLMUsage,
)
_CHECKPOINT_SERDE = JsonPlusSerializer(allowed_msgpack_modules=list(_CHECKPOINT_ALLOWLIST))


# --- shared read-only resources (issue #17) ---------------------------------
# One instance per process, shared by every browser session and every graph
# rebuild. Safe to share: MovieDatabase opens a fresh SQLite connection per
# operation, MovieVectorStore wraps a thread-safe ChromaDB PersistentClient,
# and the embedding provider is a stateless API client. Per-session state
# (memory, tracer, limiter, config) stays on MayaSession.


DATA_REPO = "rishib09/maya-data"


def catalog_movie_count(db_path: Path) -> int:
    """Rows in ``movies``, or 0 when the file is missing or not a catalog."""
    if not db_path.is_file() or db_path.stat().st_size == 0:
        return 0
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT COUNT(*) FROM movies").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return 0
    return int(row[0]) if row else 0


def resolve_runtime_data_dir(
    local: Path = Path("data"),
    fallback: Path | None = None,
) -> Path:
    """Where the catalog and the vector index live for this process.

    A populated local ``data/`` wins, so a laptop that already has the catalog
    does not download. ``MAYA_DATA_DIR`` overrides both. Otherwise the files
    go under the system temp directory, outside the app folder: writing the
    dataset into the app folder makes Streamlit restart mid-download and then
    search the empty catalog ``MovieDatabase`` creates.
    """
    configured = os.getenv("MAYA_DATA_DIR")
    if configured:
        return Path(configured)
    if catalog_movie_count(local / "tmdb_movies.db") > 0:
        return local
    return fallback or (Path(tempfile.gettempdir()) / "maya-data")


def _chroma_client(persist_dir: Path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def runtime_files_missing(data_dir: Path = Path("data")) -> bool:
    """True when the catalog or any Lab collection is absent or empty."""
    if catalog_movie_count(data_dir / "tmdb_movies.db") <= 0:
        return True
    sqlite_path = data_dir / "chroma_db" / "chroma.sqlite3"
    if not sqlite_path.is_file():
        return True
    client = _chroma_client(data_dir / "chroma_db")
    counts = {col.name: col.count() for col in client.list_collections()}
    return any(counts.get(name, 0) <= 0 for name in lab_collection_names())


def ensure_runtime_files(
    data_dir: Path | None = None,
    repo_id: str | None = None,
    download=None,
) -> None:
    """Download the catalog and the six Lab collections when either is missing.

    Calls ``huggingface_hub.snapshot_download`` unless ``download`` is passed.
    No-op when the catalog has movies and every Lab collection is already on disk.
    ``repo_id`` defaults to ``MAYA_DATA_REPO`` or ``rishib09/maya-data``.
    With ``data_dir`` omitted, a populated local catalog stays put and an empty
    one is fetched outside the app folder.
    """
    data_dir = data_dir or resolve_runtime_data_dir()
    if not runtime_files_missing(data_dir):
        return
    repo_id = repo_id or os.getenv("MAYA_DATA_REPO", DATA_REPO)
    logger.info("catalog missing, downloading dataset %s into %s", repo_id, data_dir)
    if download is None:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(data_dir))
        if runtime_files_missing(data_dir):
            raise RuntimeError(
                f"The movie catalog at {data_dir} is still empty after downloading {repo_id}."
            )
        return
    download(repo_id, data_dir)


@st.cache_resource(show_spinner=False)
def shared_database(db_path: str | None = None) -> MovieDatabase:
    """Process-wide SQLite handle (also the budget sink)."""
    path = db_path or str(resolve_runtime_data_dir() / "tmdb_movies.db")
    logger.info("building shared MovieDatabase path=%s", path)
    return MovieDatabase(path)


@st.cache_resource(show_spinner=False)
def shared_vector_store(persist_dir: str | None = None) -> MovieVectorStore:
    """Process-wide ChromaDB client + embedder caches."""
    path = persist_dir or str(resolve_runtime_data_dir() / "chroma_db")
    logger.info("building shared MovieVectorStore path=%s", path)
    return MovieVectorStore(path)


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
        # #156: the env overrides died with the other fork — this checkout
        # runs one stack; nothing here is flippable from the environment.
        self.conversation = ConversationState()
        self.tracer = DualModeObservabilityManager(session_id=f"ui-{datetime.now(UTC):%H%M%S}")
        self.limiter = SessionCostLimiter()
        self.db = shared_database()  # process-wide shared handle (#17): engine + budget sink
        self.budget_tracker = WeeklyBudgetTracker(self.db)  # weekly $ ceiling (#8)
        self.view = "Chat"  # sidebar navigation: Chat | Evals | Traces
        self.feedback_log: dict[int, int] = {}  # assistant-turn index → ±1 (thumbs)
        self.feedback_store = FeedbackStore()  # SQLite persistence (#9)
        self.report_count = 0  # /feedback Reports posted this tab (#76 cap)
        # Feedback Receipt (#76 amendment): reported reply's trace id → inbox
        # comment URL, or None when only Langfuse holds the Report.
        self.report_receipts: dict[str, str | None] = {}
        # #11/#30: the dense path derives from config — ONE knob pair names
        # the collection AND the query-embedding provider, so user queries are
        # always embedded by the collection's own model. Default = ADR 0008
        # verdict (gemini-embedding-2 · full: 100% golden hit@5 / MRR 0.964).
        # Fail-closed: without OPENROUTER_API_KEY the app refuses to start
        # rather than silently degrading to a weaker collection.
        # (#96-F2: the stale hard-coded rag_version/search_provider that used
        # to sit here were dead — both names were overwritten two lines below,
        # and the stale assignment built an orphaned cached provider.)
        self.rag_version = collection_name(self.config.column_preset, self.config.embedding_profile)
        self.search_provider = provider_from_profile(self.config.embedding_profile)
        self.config_version = 0  # bumped on preset apply → knob widgets remount
        self.turn_log: list[dict] = []  # one row per turn for badges/trace
        self.last_movies = []  # MovieRecords from the most recent retrieval
        # #93/D16: the graph's memory — one saver + one thread per browser
        # session; every turn sends only the new message (see turn()).
        # #119: allowlisted serde so checkpointed domain types stay typed.
        self._saver = InMemorySaver(serde=_CHECKPOINT_SERDE)
        self._thread_id = uuid.uuid4().hex
        self._graph_sig = ""
        self.graph = self._build_graph()

    # --- graph lifecycle ---

    def _build_graph(self):
        # #30: re-derive the dense pair on every rebuild — a Lab combo change
        # must swap collection and query-embedder together, never one alone.
        self.rag_version = collection_name(self.config.column_preset, self.config.embedding_profile)
        self.search_provider = provider_from_profile(self.config.embedding_profile)
        engine = HybridRetrievalEngine(
            db=self.db,
            vector_store=shared_vector_store(),
            rag_version=self.rag_version,
            hybrid_alpha=self.config.hybrid_alpha,
            reranker_enabled=self.config.reranker_enabled,
            reranker_model=self.config.reranker_model,
            search_provider=self.search_provider,
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
            checkpointer=self._saver,
        )

    def _graph_signature(self) -> str:
        """Engine + routing knobs that require a graph rebuild when changed."""
        return "|".join(
            str(v)
            for v in (
                self.config.router_model,
                self.config.synthesis_model,
                self.config.temperature,
                self.config.hybrid_alpha,
                self.config.reranker_enabled,
                self.config.reranker_model,
                self.config.retrieval_top_k,
                self.config.route_max_attempts,
                self.config.reasoning_effort,
                # #30: combo change swaps collection + query provider
                self.config.embedding_profile,
                self.config.column_preset,
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

    def turn(self, query: str, *, recalled: bool = False) -> None:
        """One full Maya turn: guard → route/funnel → retrieve → synthesize.

        D16 (#93): the graph owns conversation state — the compiled
        checkpointer carries every persistent field across turns on
        ``self._thread_id``, so this method sends ONLY the new message. The
        hand-copied field round-trip (and its bug class — the session
        forgetting shown ids and funnel flags, #80) is gone by construction.
        ``ConversationState`` below is a read model for the transcript and
        turn rows.

        The turn_log row is built ATOMICALLY by ``_build_turn_row`` from the
        graph's output alone (#26-A) — chip metadata, response, movie count
        and token count can never come from different turns, even on funnel
        turns where the router never ran (``routing_decision`` is None).
        """
        graph = self.ensure_graph()
        ring_before = len(self.tracer.traces())
        trace_id = self.tracer.new_turn_trace()  # per-turn id for feedback (#9)
        if recalled:
            # #48 telemetry: recall usage must be observable (ADR 0009) — the
            # evidence base for ever revisiting a latency cache.
            self.tracer.record_local("recall", {"query": query})
        out = graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            # cloud tracing was silently inactive in the UI before #9 — wired
            # every turn now so the trace id and the run actually correlate
            config={
                "callbacks": self.tracer.callbacks(),
                "metadata": self.tracer.metadata(),  # v4 session grouping
                "configurable": {"thread_id": self._thread_id},
            },
        )
        row = self._build_turn_row(
            out,
            query=query,
            trace_id=trace_id,
            rag_version=self.rag_version,
            new_traces=slice_new_traces(ring_before, self.tracer.traces()),
            prev_tokens=self.conversation.session_tokens,
            prev_cost=self.conversation.session_cost_usd,
        )
        row["recalled"] = recalled  # #48: replayed input, evaluated fresh
        movies = list(out.get("retrieved_movies") or [])
        row["movies"] = movies  # #147: Retrieval reads this turn, including after a rerun
        self.last_movies = movies
        # Guided narrowing (#22): probe answers extracted this turn persist
        # in session state; probe turns carry no movies and no synthesis cost.
        # #26-E: a fresh-start turn clears preferences via the reducer.
        self.conversation.session_preferences = out.get(
            "session_preferences", self.conversation.session_preferences
        )
        self.conversation.probe_count = out.get("probe_count", self.conversation.probe_count)
        self.conversation.funnel_active = out.get("funnel_active", self.conversation.funnel_active)
        self.conversation.offered_genre_options = out.get("offered_genre_options", [])
        self.conversation.add_turn(
            query,
            row["response"],
            movies,
            out.get("routing_decision"),
            tokens_used=row["tokens"],
            cost_usd=row["cost_usd"],
            turn_ref=len(self.turn_log),  # #26-K: identity join, stamped pre-append
            window=self.config.message_window,  # #93: read model trims with the knob
        )
        self.turn_log.append(row)

    @staticmethod
    def turn_ref_for(session_or_messages) -> int | None:
        """Latest assistant message's turn_log index (#26-K), None if absent.

        Accepts a MayaSession or any object exposing ``conversation.messages``
        (kept forgiving for tests).
        """
        conversation = getattr(session_or_messages, "conversation", session_or_messages)
        for message in reversed(conversation.messages):
            if message.role == "assistant":
                return message.turn_ref
        return None

    @staticmethod
    def _build_turn_row(
        out: dict,
        *,
        query: str,
        trace_id: str,
        rag_version: str,
        new_traces: list[dict],
        prev_tokens: int,
        prev_cost: float = 0.0,
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
        cost_usd = max(out.get("session_cost_usd", 0.0) - prev_cost, 0.0)
        route_traces = [t for t in new_traces if t["node"] == "route"]
        node_names = {t["node"] for t in new_traces}
        if decision is None:
            intent = f"FUNNEL_{(stage or 'probe').upper()}"
            confidence = 1.0  # deterministic — no model involved
            path = "funnel"
        else:
            intent = decision.intent.value
            confidence = decision.confidence
            if "refusal" in node_names:
                path = "refusal"  # guard-diverted after a projection (#113)
            elif "pivot" in node_names:
                path = "pivot"  # deterministic off-topic deflection (#8)
            elif stage == "retrieve":
                path = "funnel"  # v1 funnel-owned retrieval
            elif route_traces:
                path = MayaSession._path_taken(route_traces)
            else:
                path = "retrieve"
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
            "cost_usd": cost_usd,
            "response": response,
            "probe": any(t["node"] == "probe" for t in new_traces),
            "narrowing": preference_chips(prefs) if prefs else [],
            "filters": MayaSession._effective_filter_chips(
                out.get("filters_applied"), decision
            ),
        }

    @staticmethod
    def _effective_filter_chips(applied, decision) -> list[str]:
        """What the engine ACTUALLY saw this turn (#153, #93 Q3 O1).

        Prefers the post-injection ``filters_applied`` snapshot so remembered
        preferences code folded into the SQL are visible in the chip — the
        "Current filter" line never disagrees with what ran. Falls back to
        the routing decision when the engine never ran this turn.
        """
        if applied is None:
            return MayaSession._filter_chips(decision)
        return MayaSession._filter_chips_from_criteria(applied)

    @staticmethod
    def _filter_chips(decision) -> list[str]:
        """Declared SQL filters for this turn's metadata line (#26-F)."""
        if decision is None or decision.filters is None:
            return []
        return MayaSession._filter_chips_from_criteria(decision.filters.model_dump())

    @staticmethod
    def _filter_chips_from_criteria(data: dict | None) -> list[str]:
        """Chip rendering over a flat MetadataFilterCriteria dict (#153)."""
        if not data:
            return []
        chips: list[str] = []
        if data.get("genres"):
            mode = (
                f" ({data.get('genre_match')})" if len(data["genres"]) > 1 else ""
            )
            chips.append("genres: " + ", ".join(data["genres"]) + mode)
        if data.get("exact_year"):
            chips.append(str(data["exact_year"]))
        elif data.get("year_min") or data.get("year_max"):
            chips.append(f"{data.get('year_min') or '…'}–{data.get('year_max') or '…'}")
        if data.get("director"):
            chips.append(f"dir. {data['director']}")
        if data.get("cast_member"):
            chips.append(f"cast {data['cast_member']}")
        if data.get("person"):
            chips.append(f"person: {data['person']}")
        chips.extend(f"no {g}" for g in (data.get("excluded_genres") or []))
        chips.extend(f"no {a}" for a in (data.get("excluded_actors") or []))
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

        ``turn_index`` is the UI's assistant-message counter (#26-K). It is
        resolved through the message's ``turn_ref`` stamp — the message
        window slides while turn_log grows, so the index alone points at the
        wrong row once trimming begins (feedback was mis-attributed
        accordingly). Falls back to the raw index only for legacy rows.
        """
        if value not in (1, -1):
            raise ValueError(f"rating must be +1 or -1, got {value!r}")
        if not (0 <= turn_index < len(self.turn_log)):
            raise IndexError(f"turn_index {turn_index} out of range")
        row = self._turn_row_for_ui_index(turn_index) or self.turn_log[turn_index]
        self.feedback_store.record(row["trace_id"], value, row["rag_version"], intent=row["intent"])
        self.feedback_log[turn_index] = value
        post_inbox_comment(format_rating_comment(value, row))  # #76: durable copy
        return push_feedback_score(row["trace_id"], value)

    def record_report(self, text: str) -> ReportResult:
        """Posts a ``/feedback`` Report with its Feedback Window (#76).

        REJECTED when the text fails a guard, the per-tab cap is reached, or
        no turn exists to report on — nothing is logged in those cases
        (verdict D9, D11). RECORDED once the Report is durable somewhere:
        the GitHub inbox, or Langfuse when the inbox is unreachable (D12).
        UNDELIVERED when neither backend accepted it; the attempt does not
        count against the cap so the visitor can retry.
        """
        cleaned = validate_report(text)
        if cleaned is None or not self.turn_log:
            return ReportResult.REJECTED
        if self.report_count >= REPORTS_PER_SESSION:
            return ReportResult.REJECTED
        window = self.turn_log[-WINDOW_TURNS:]
        trace_id = window[-1]["trace_id"]
        url = post_inbox_comment(format_report_comment(cleaned, window))
        if url is None and not push_report_comment(trace_id, cleaned):
            logger.warning("Report undelivered: GitHub inbox and Langfuse both unavailable")
            return ReportResult.UNDELIVERED
        self.report_count += 1
        self.report_receipts[trace_id] = url  # Feedback Receipt for the reported reply
        return ReportResult.RECORDED

    def _turn_row_for_ui_index(self, ui_index: int) -> dict | None:
        """Resolves the UI's assistant-message counter to its turn row (#26-K).

        The Nth assistant message carries ``turn_ref`` — the turn_log index
        it was produced under. Returns None when no stamped message exists
        (pre-#26-K sessions) so callers can fall back to index arithmetic.
        """
        assistants = [m for m in self.conversation.messages if m.role == "assistant"]
        if 0 <= ui_index < len(assistants):
            ref = assistants[ui_index].turn_ref
            if ref is not None and 0 <= (ref := ref) < len(self.turn_log):
                return self.turn_log[ref]
        return None


def get_session() -> MayaSession:
    """Streamlit session_state singleton."""
    if "maya_session" not in st.session_state:
        st.session_state.maya_session = MayaSession()
    return st.session_state.maya_session
