"""Issue #9 feedback loop: SQLite persistence + Langfuse score sync.

Three tiers per project convention: mock/unit (offline, :memory:, fake
client), adversarial (validation, failure isolation), live (real DB +
real Langfuse, skipif keys absent).
"""


import pytest

from src.feedback.langfuse_score import FEEDBACK_SCORE_NAME, push_feedback_score
from src.feedback.store import FeedbackStore

# =====================================================================
# Mock / unit tier
# =====================================================================


@pytest.fixture
def store() -> FeedbackStore:
    return FeedbackStore(":memory:")


def test_record_and_read_back(store):
    store.record("trace-1", 1, "v1_1_enriched", intent="SEMANTIC_SEARCH")
    row = store.get_by_trace_id("trace-1")
    assert row["rating"] == 1
    assert row["rag_version"] == "v1_1_enriched"
    assert row["intent"] == "SEMANTIC_SEARCH"


def test_upsert_updates_in_place(store):
    """Re-rating the same trace updates; never duplicates (UPSERT contract)."""
    store.record("trace-1", -1, "v1_1_enriched")
    store.record("trace-1", 1, "v1_1_enriched")
    assert store.count() == 1
    assert store.get_by_trace_id("trace-1")["rating"] == 1


def test_stats_by_version_grouping(store):
    store.record("t1", 1, "v1_0_baseline")
    store.record("t2", -1, "v1_0_baseline")
    store.record("t3", 1, "v1_1_enriched")
    stats = {s["rag_version"]: s for s in store.stats_by_version()}
    assert stats["v1_0_baseline"]["n"] == 2
    assert stats["v1_0_baseline"]["avg_rating"] == pytest.approx(0.0)
    assert stats["v1_0_baseline"]["thumbs_up"] == 1
    assert stats["v1_0_baseline"]["thumbs_down"] == 1
    assert stats["v1_1_enriched"]["avg_rating"] == 1.0


def test_stats_empty_table(store):
    assert store.stats_by_version() == []


def test_push_score_without_keys_is_noop(monkeypatch):
    """Offline-safe: no keys → False, no exception (adversarial contract too)."""
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert push_feedback_score("trace-1", 1) is False


def test_push_score_calls_client_with_numeric_type(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    calls = {}

    class FakeClient:
        def create_score(self, **kwargs):
            calls.update(kwargs)

    import langfuse

    monkeypatch.setattr(langfuse, "get_client", lambda: FakeClient())
    assert push_feedback_score("trace-1", -1) is True
    assert calls["trace_id"] == "trace-1"
    assert calls["value"] == -1.0
    assert calls["data_type"] == "NUMERIC"
    assert calls["name"] == FEEDBACK_SCORE_NAME
    # deterministic score_id → re-rating upserts cloud-side too
    assert calls["score_id"] == f"trace-1:{FEEDBACK_SCORE_NAME}"


def test_turn_trace_ids_are_unique_per_turn(monkeypatch):
    """new_turn_trace mints a fresh id each turn, local uuid4 without keys."""
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    from src.observability.tracer import DualModeObservabilityManager

    tracer = DualModeObservabilityManager(session_id="t")
    ids = {tracer.new_turn_trace() for _ in range(3)}
    assert len(ids) == 3
    assert tracer.current_trace_id in ids


def test_turn_log_carries_trace_id_and_rag_version(monkeypatch):
    """session.turn records the exact trace_id used for feedback (#9).

    Offline fakes for router/engine/synthesizer — the real MayaRouter would
    call OpenRouter from a unit test.
    """

    from src.domain.config import ExperimentConfig
    from src.domain.memory import ConversationState
    from src.domain.movie import MovieRecord
    from src.domain.routing import IntentType, QueryRoutingDecision
    from src.graph.orchestrator import build_maya_graph
    from src.graph.state import SynthesisUsage
    from src.maya.guardrails import SessionTokenLimiter
    from src.observability.tracer import DualModeObservabilityManager
    from src.retrieval.hybrid_engine import RetrievalResult
    from src.ui.session import MayaSession

    class FakeRouter:
        def route(self, query, state, feedback=None):
            return QueryRoutingDecision(
                intent=IntentType.SEMANTIC_SEARCH, confidence=0.9,
                standalone_query=query, requires_rag=True,
            )

    class FakeEngine:
        def retrieve(self, query, routing, top_k=8, candidate_pool=50):
            m = MovieRecord(id=1, title="X", release_year=2020, genres=["Drama"])
            return [RetrievalResult(movie=m, score=1.0, source="sql")]

    class FakeSynth:
        def synthesize(self, query, decision, movies, history):
            return "Answer.", SynthesisUsage(model="fake", prompt_tokens=1, completion_tokens=1)

    session = MayaSession.__new__(MayaSession)  # skip heavy __init__
    session.config = ExperimentConfig()
    session.conversation = ConversationState()
    session.tracer = DualModeObservabilityManager(session_id="t")
    session.limiter = SessionTokenLimiter()
    session.feedback_store = FeedbackStore(":memory:")
    session.feedback_log = {}
    session.rag_version = "v1_1_enriched"
    session.turn_log = []
    # skip ensure_graph()'s rebuild: sig matches → prebuilt graph below is used
    session._graph_sig = session._graph_signature()
    session.graph = build_maya_graph(
        session.config,
        FakeRouter(),
        FakeEngine(),
        FakeSynth(),
        session.tracer,
        limiter=session.limiter,
    )
    session.turn("best movie ever")
    assert session.turn_log[0]["trace_id"] == session.tracer.current_trace_id
    assert session.turn_log[0]["rag_version"] == "v1_1_enriched"
