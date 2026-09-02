"""Live tests for #26: the walkthrough replayed against the real pipeline.

Pins the acceptance criteria: the #26 walkthrough's incoherent turns are
impossible, genre requests never pivot, the funnel's first recommendation
announces carried filters, and the escape hatch wipes the slate.
"""

import pytest
from langchain_core.messages import HumanMessage

from src.domain.config import ExperimentConfig
from src.domain.memory import UserSessionPreferences
from src.domain.routing import IntentType
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
    engine = HybridRetrievalEngine(
        db=db, vector_store=MovieVectorStore("data/chroma_db"),
        rag_version="v1_1_enriched",
    )
    return build_maya_graph(
        config,
        # the session wires the DB's own genres — mirror that here (#26-B)
        MayaRouter(config, genre_vocabulary=db.distinct_genres()),
        engine, MayaSynthesizer(config),
        DualModeObservabilityManager(session_id="issue26-live"),
    )


def _base(prefs=None, funnel_active=False, offered=None):
    return {
        "session_preferences": prefs or UserSessionPreferences(),
        "funnel_active": funnel_active,
        "offered_genre_options": offered or [],
    }


def test_suggest_me_horror_movies_never_pivots(graph):
    """Walkthrough turn 1: genre request misrouted OUT_OF_SCOPE at 1.00."""
    out = graph.invoke({
        "messages": [HumanMessage(content="suggest me horror movies")], **_base(),
    })
    decision = out["routing_decision"]
    assert decision.intent is not IntentType.OUT_OF_SCOPE
    # coherent next step either way: enough extracted signal → retrieve now,
    # otherwise the funnel arms with a probe — never a pivot, never silence
    assert out["funnel_active"] or out.get("retrieved_movies")


def test_scary_movies_for_kids_refinement_never_pivots(graph):
    """Walkthrough turn 3: post-funnel refinement pivoted — mood vocab guards."""
    out = graph.invoke({
        "messages": [HumanMessage(content="scary movies for kids")], **_base(),
    })
    assert out["routing_decision"].intent is not IntentType.OUT_OF_SCOPE
    assert out["retrieved_movies"], "mood+audience flavor should retrieve"


def test_full_funnel_walkthrough_with_genre_confirmation_and_notice(graph):
    """Funnel-era turns replayed live: narrowing → genre confirm → retrieval
    WITH the #26-E carry-over announcement.

    The funnel is seeded armed (mood=scary) — arming variance is covered by
    the other tests. Confirmation vocabulary forces retrieval at ANY funnel
    stage, so the walkthrough cannot dead-end on LLM extraction variance.
    """
    state = _base(
        prefs=UserSessionPreferences(preferred_mood="scary"), funnel_active=True,
    )
    out = None
    for query in ["for the kids", "both of them", "go ahead", "just show"]:
        out = graph.invoke({"messages": [HumanMessage(content=query)], **state})
        state["session_preferences"] = out["session_preferences"]
        state["funnel_active"] = out["funnel_active"]
        state["offered_genre_options"] = out.get("offered_genre_options", [])
        assert out["final_response"], f"every turn must answer: {query}"
        if out.get("retrieved_movies"):
            break
    assert out.get("retrieved_movies"), "walkthrough must end in grounded retrieval"
    assert "Still filtering by" in out["final_response"]
    assert "mood:" in out["final_response"]


def test_something_completely_different_resets_everything(graph):
    state = _base(prefs=UserSessionPreferences(
        preferred_mood="scary", audience="kids", preferred_genres=["Horror"],
    ), funnel_active=True)
    out = graph.invoke({
        "messages": [HumanMessage(content="actually, something completely different")],
        **state,
    })
    assert out["session_preferences"] == UserSessionPreferences()
    assert out["funnel_active"] is False


def test_person_union_still_retrieves_after_coherence_changes(graph):
    out = graph.invoke({
        "messages": [HumanMessage(content="movies of Christopher Nolan")], **_base(),
    })
    movies = out.get("retrieved_movies", [])
    assert movies
    assert all(
        "nolan" in m.director.lower() or any("nolan" in c.name.lower() for c in m.cast)
        for m in movies
    )
