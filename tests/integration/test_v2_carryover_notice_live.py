"""Live test for #153: the carry-over notice fires on a real v2 turn.

The seam under test is deterministic (code injects remembered genres, code
announces them); the only model-discretion risk is the router mirroring the
stored genre into this turn's own filters, which would suppress the
injection. A director-focused query makes that mirroring vanishingly
unlikely — the offline suite pins the deterministic behavior exhaustively
(test_v2_carryover_notice.py).

The stored genre is Action so the filtered pool is non-empty (Tenet,
Odyssey and Dunkirk carry it). A Horror seed was tried first and hit the
OTHER deterministic gate, correctly: Nolan ∧ ≥2017 ∧ Horror matches zero
movies, and a turn that shows nothing must not nag — the reply stayed the
zero-retrieval refinement question.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.domain.config import ExperimentConfig
from src.domain.memory import UserSessionPreferences
from src.graph.orchestrator import build_maya_graph
from src.indexing.vector_store import MovieVectorStore
from src.maya.agent import MayaSynthesizer
from src.maya.v2 import MayaV2Router
from src.observability.tracer import DualModeObservabilityManager
from src.retrieval.hybrid_engine import HybridRetrievalEngine
from src.storage.database import MovieDatabase

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def graph():
    config = ExperimentConfig(routing_stack="v2")
    db = MovieDatabase("data/tmdb_movies.db")
    engine = HybridRetrievalEngine(
        db=db, vector_store=MovieVectorStore("data/chroma_db"),
        rag_version="v1_1_enriched",
    )
    return build_maya_graph(
        config,
        MayaV2Router(config),
        engine, MayaSynthesizer(config),
        DualModeObservabilityManager(session_id="issue153-live"),
        checkpointer=InMemorySaver(),
    )


def test_notice_announces_the_injected_session_genre(graph):
    """Stored Horror + a director-only turn → Horror joins the SQL silently →
    the reply must carry the transparency line with the escape hatch."""
    out = graph.invoke(
        {
            "messages": [HumanMessage(content="latest Christopher Nolan movies")],
            "session_preferences": UserSessionPreferences(preferred_genres=["Action"]),
        },
        {"configurable": {"thread_id": "issue153-live"}},
    )
    applied = out.get("filters_applied") or {}
    assert "Action" in applied.get("genres", []), (
        f"expected the session genre injected into the SQL, got {applied}"
    )
    assert "Still filtering by genres: Action" in out["final_response"]
    assert "completely different?" in out["final_response"]  # escape hatch
    # one message, notice inside — no duplicate bubble
    noticed = [
        m for m in out["messages"]
        if isinstance(m, AIMessage) and "Still filtering by" in m.content
    ]
    assert len(noticed) == 1
    assert noticed[-1].content == out["final_response"]