"""Live repro of issue #23: the exact walkthrough conversation.

Real router + real engine + real synthesis. Asserts the funnel owns probe
answers, follow-ups never pivot, and confirmed retrieval grounds posters.
"""

import pytest
from langchain_core.messages import HumanMessage

from src.domain.config import ExperimentConfig
from src.domain.memory import UserSessionPreferences
from src.graph.orchestrator import build_maya_graph
from src.indexing.vector_store import MovieVectorStore
from src.maya.agent import MayaSynthesizer
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
        DualModeObservabilityManager(session_id="issue23-live"),
    )


def test_exact_walkthrough_conversation(graph):
    state = {
        "session_preferences": UserSessionPreferences(),
        "probe_count": 0,
        "funnel_active": False,
    }

    # Turn 1: broad scifi search → probe (mood)
    out = graph.invoke({"messages": [HumanMessage(content="show me scifi movies")], **state})
    state["probe_count"] = out["probe_count"]
    state["funnel_active"] = out["funnel_active"]
    assert out["probe_count"] == 1
    assert out["funnel_active"] is True
    assert "mood" in out["final_response"].lower()

    # Turn 2: the ANSWER "edge of the seat" → funnel probes audience (never GREETING)
    out = graph.invoke({"messages": [HumanMessage(content="edge of the seat")], **state})
    state["probe_count"] = out["probe_count"]
    state["funnel_active"] = out["funnel_active"]
    state["session_preferences"] = out["session_preferences"]
    assert state["session_preferences"].preferred_mood == "edge-of-your-seat"
    assert out["final_response"] != "" and "Inception" not in out["final_response"]

    # Turn 3: topical follow-up → conversational, NEVER the OUT_OF_SCOPE pivot
    out = graph.invoke({"messages": [HumanMessage(content="more into the theoretical physics")], **state})
    state["funnel_active"] = out["funnel_active"]
    assert "outside my reel" not in out["final_response"]

    # Turn 4: confirmation → grounded retrieval WITH movies (posters possible)
    out = graph.invoke({"messages": [HumanMessage(content="go ahead show me")], **state})
    assert out["funnel_active"] is False
    assert out["retrieved_movies"], "confirmed retrieval must return movies"
    # synthesis grounded: every title from the retrieved records
    from src.maya.agent import MayaSynthesizer as _M

    synth = _M(ExperimentConfig())
    assert synth.cwa_violations(out["final_response"], out["retrieved_movies"]) == []
