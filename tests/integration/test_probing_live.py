"""Live tests for guided narrowing (issue #22): real router + real engine.

Runs the real pipeline (OpenRouter router, real hybrid engine); synthesis
LLM is never reached on probe turns, so these stay fast and cheap.
"""

import pytest
from langchain_core.messages import HumanMessage

from src.domain.config import ExperimentConfig
from src.domain.memory import UserSessionPreferences
from src.graph.orchestrator import build_maya_graph
from src.indexing.vector_store import MovieVectorStore
from src.maya.agent import MayaSynthesizer
from src.maya.probing import MAX_PROBE_TURNS
from src.maya.router import MayaRouter
from src.observability.tracer import DualModeObservabilityManager
from src.retrieval.hybrid_engine import HybridRetrievalEngine
from src.storage.database import MovieDatabase

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def graph():
    config = ExperimentConfig()
    db = MovieDatabase("data/tmdb_movies.db")
    store = MovieVectorStore("data/chroma_db")
    engine = HybridRetrievalEngine(db=db, vector_store=store, rag_version="v1_1_enriched")
    return build_maya_graph(
        config, MayaRouter(config), engine, MayaSynthesizer(config),
        DualModeObservabilityManager(session_id="probe-live"),
    )


def test_broad_query_probes_not_dumps(graph):
    """Router-routed broad search ('sci-fi movies') → probe, NOT a dump.

    Note: ultra-vague asks ('suggest me something') classify as GREETING by
    the real router — those steer conversationally via the #10 ethos. The
    deterministic funnel covers broad queries that DO route to retrieval.
    """
    out = graph.invoke({
        "messages": [HumanMessage(content="sci-fi movies")],
        "session_preferences": UserSessionPreferences(),
        "probe_count": 0,
    })
    assert out["probe_count"] == 1
    assert out.get("retrieved_movies", []) == []  # no premature dump
    assert "synthesis_usage" not in out or out["synthesis_usage"] is None  # zero LLM cost
    lowered = out["final_response"].lower()
    assert any(w in lowered for w in ("mood", "watching", "feel")), out["final_response"][:200]


def test_two_probe_turns_then_recommendation_honors_answers(graph):
    """Full funnel: broad → probe → answer → probe → answer → grounded recs."""
    prefs = UserSessionPreferences()
    probe_count = 0
    # turn 1 probes; turn 2's conversational answer still gets extracted;
    # turn 3 routes to retrieval — prefs stop the funnel, recommendations flow
    answers = ["sci-fi movies", "something funny for the kids", "funny movies for kids"]
    final = None
    for i in range(3):
        out = graph.invoke({
            "messages": [HumanMessage(content=answers[i])],
            "session_preferences": prefs,
            "probe_count": probe_count,
        })
        prefs = out["session_preferences"]
        probe_count = out["probe_count"]
        if out.get("retrieved_movies"):
            final = out
            break
    # after "funny for the kids" (conversational turn), extraction still ran:
    assert prefs.preferred_mood == "funny"
    assert prefs.audience == "kids"
    # the funnel terminates: recommendation turn within MAX_PROBE_TURNS probes
    assert probe_count <= MAX_PROBE_TURNS
    assert final is not None, "never reached a recommendation turn"
    assert final["retrieved_movies"], "recommendation turn must retrieve movies"
