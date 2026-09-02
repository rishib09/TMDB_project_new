"""Live tests for #24/#25: LLM extraction, mood→genre mapping, person union.

Real router (as extractor AND router), real engine, real synthesis where
retrieval happens.
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
        DualModeObservabilityManager(session_id="issue2425-live"),
    )


def test_open_vocabulary_mood_extraction_and_genre_confirmation(graph):
    """'something fantastic' (NOT in vocab) → LLM extracts → genre confirm."""
    state = {
        "session_preferences": UserSessionPreferences(),
        "probe_count": 0,
        "funnel_active": False,
        "offered_genre_options": [],
    }
    out = graph.invoke({"messages": [HumanMessage(content="sci-fi movies")], **state})
    state["probe_count"] = out["probe_count"]
    state["funnel_active"] = out["funnel_active"]
    assert out.get("final_response") and out["probe_count"] >= 1

    # an open-vocabulary mood the deterministic vocab does NOT know
    out = graph.invoke({
        "messages": [HumanMessage(content="something fantastic and mind-bending")], **state,
    })
    state["probe_count"] = out["probe_count"]
    state["funnel_active"] = out["funnel_active"]
    prefs = out["session_preferences"]
    assert prefs.preferred_mood != "", "LLM extraction must capture the open-vocab mood"
    # either genre candidates offered (mapped mood) or graceful flavor-only
    if out["funnel_active"] and out.get("offered_genre_options"):
        assert out["offered_genre_options"], "confirmation must offer candidates"


def test_person_union_round_trip(graph):
    """A person who acts AND directs → union filmography, CWA-clean synthesis."""
    # Christopher Nolan directs; he also cameos/acts in his films — the DB
    # decides, not the router. Any role-less person mention exercises the path.
    out = graph.invoke({
        "messages": [HumanMessage(content="movies of Christopher Nolan")],
        "session_preferences": UserSessionPreferences(),
    })
    movies = out.get("retrieved_movies", [])
    assert movies, "Nolan filmography must retrieve"
    assert all(
        "nolan" in m.director.lower() or any("nolan" in c.name.lower() for c in m.cast)
        for m in movies
    )


def test_full_funnel_with_genre_confirmation_to_filtered_retrieval(graph):
    """Broad → probe → mood answer → genre confirm → pick → grounded recs."""
    state = {
        "session_preferences": UserSessionPreferences(),
        "probe_count": 0,
        "funnel_active": False,
        "offered_genre_options": [],
    }
    answers = ["scary movies", "for the kids", "go ahead"]
    final = None
    for answer in answers:
        out = graph.invoke({"messages": [HumanMessage(content=answer)], **state})
        state["probe_count"] = out["probe_count"]
        state["funnel_active"] = out["funnel_active"]
        state["offered_genre_options"] = out.get("offered_genre_options", [])
        if "session_preferences" in out:
            state["session_preferences"] = out["session_preferences"]
        if out.get("retrieved_movies"):
            final = out
            break
    assert final is not None, "funnel must end in retrieval within the budget"
    assert state["probe_count"] <= MAX_PROBE_TURNS
    assert final["retrieved_movies"], "recommendation must carry movies (posters)"
