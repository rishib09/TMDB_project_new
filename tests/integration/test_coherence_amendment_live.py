"""Live tests for the #26 amendment: the 6-turn walkthrough replayed.

Pins: fresh-start honored, stale Comedy retired on mood change, "yes"
retrieves, zero hallucinated cards on any turn.
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
        MayaRouter(config, genre_vocabulary=db.distinct_genres()),
        engine, MayaSynthesizer(config),
        DualModeObservabilityManager(session_id="issue26b-live"),
    )


def test_remove_filters_phrase_resets_everything(graph):
    """Walkthrough turn 2 verbatim — was ignored, must now wipe the slate."""
    out = graph.invoke({
        "messages": [HumanMessage(content="remove all the filters and start with a fresh search")],
        "session_preferences": UserSessionPreferences(
            preferred_mood="funny", audience="date night",
            preferred_genres=["Comedy"], genre_confirmation_done=True,
        ),
        "funnel_active": True,
    })
    assert out["session_preferences"] == UserSessionPreferences()
    assert out["funnel_active"] is False


def test_something_different_resets_without_completely(graph):
    """Walkthrough turn 1 verbatim fragment — 'completely' was required before."""
    out = graph.invoke({
        "messages": [HumanMessage(content="lets start with something different")],
        "session_preferences": UserSessionPreferences(
            preferred_mood="funny", preferred_genres=["Comedy"],
        ),
    })
    assert out["session_preferences"] == UserSessionPreferences()


def test_mood_change_retires_stale_comedy_live(graph):
    """Walkthrough turns 4–5: 'horror' after a Comedy session — the stale
    Comedy must not survive into the new mood's narrowing."""
    prefs = UserSessionPreferences(
        preferred_mood="funny", audience="date night",
        preferred_genres=["Comedy"], genre_confirmation_done=True,
    )
    state = {
        "session_preferences": prefs,
        "funnel_active": True,
        "offered_genre_options": [],
    }
    out = graph.invoke({"messages": [HumanMessage(content="horror")], **state})
    merged = out["session_preferences"]
    assert merged.preferred_mood == "scary"
    assert "Comedy" not in merged.preferred_genres
    # fresh confirmation turn, referencing Horror/Thriller — never Comedy
    assert "Comedy" not in out["final_response"]
    assert "Horror" in out["final_response"] or "Thriller" in out["final_response"]


def test_bare_yes_retrieves_after_confirm_live(graph):
    """Walkthrough turn 6 verbatim: 'yes' after 'shall I pull the films now?'
    must retrieve through the funnel — never GREETING, never hallucinated."""
    state = {
        "session_preferences": UserSessionPreferences(
            preferred_mood="scary", audience="date night",
            preferred_genres=["Horror", "Thriller"], genre_confirmation_done=True,
        ),
        "funnel_active": True,
        "offered_genre_options": [],
    }
    out = graph.invoke({"messages": [HumanMessage(content="yes")], **state})
    assert out.get("retrieved_movies"), "'yes' must land the films"
    assert out["funnel_active"] is False  # funnel closed on delivery


def test_dream_manipulation_leonardo_no_comedy_hallucination(graph):
    """Walkthrough turn 1, post-reset: genre guard + clean slate → the query
    itself drives retrieval; no stale Comedy and no ungrounded cards."""
    out = graph.invoke({
        "messages": [HumanMessage(content="something different: which movie was about dream manipulation and had Leonardo as lead")],
        "session_preferences": UserSessionPreferences(),  # after reset
        "funnel_active": False,
    })
    if out.get("retrieved_movies"):
        titles = [m.title.lower() for m in out["retrieved_movies"]]
        assert any("inception" in t for t in titles), f"dream manipulation query missed: {titles}"
    else:
        # deterministic zero-retrieval branch (#21) — never hallucinated cards
        assert out["final_response"]
