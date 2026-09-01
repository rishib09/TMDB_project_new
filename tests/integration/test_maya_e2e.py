"""Live end-to-end tests: full Maya graph on real services (issue #5).

Real OpenRouter (router + synthesis), real ChromaDB + SQLite. OPT-IN:
    npx @dotenvx/dotenvx run -- pytest -m live -v

Each test drives one conversation turn through the compiled LangGraph and
asserts on observable outcomes (final response, routing decision, node
path, CWA violations) — never on exact LLM wording.
"""

import os

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

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set — skipping live end-to-end tests",
    ),
]


class EngineSpy:
    """Wraps the real engine, recording retrieve() calls for assertions."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return self.inner.retrieve(**kwargs)


@pytest.fixture(scope="module")
def config():
    return ExperimentConfig()


@pytest.fixture(scope="module")
def graph_bundle(config):
    db = MovieDatabase("data/tmdb_movies.db")
    store = MovieVectorStore("data/chroma_db")
    engine = HybridRetrievalEngine(db=db, vector_store=store, rag_version="v1_1_enriched")
    spy = EngineSpy(engine)
    tracer = DualModeObservabilityManager(session_id="e2e-test")
    return build_maya_graph(
        config,
        MayaRouter(config),
        spy,
        MayaSynthesizer(config),
        tracer,
    ), spy, tracer


def _turn(graph_bundle, query, **extra):
    graph, spy, tracer = graph_bundle
    spy.calls.clear()
    out = graph.invoke(
        {"messages": [HumanMessage(content=query)], **extra},
    )
    return out, spy, tracer


def test_greeting_turn_skips_retrieval(graph_bundle):
    out, spy, _ = _turn(graph_bundle, "Hello! Who are you?")
    assert spy.calls == [], "greeting must not trigger retrieval"
    assert out["routing_decision"].intent == IntentType.GREETING
    assert out["final_response"]  # conversational reply produced


def test_out_of_scope_pivots_without_llm_cost(graph_bundle):
    out, spy, _ = _turn(graph_bundle, "What is the weather in Paris tomorrow?")
    assert spy.calls == []
    assert "film" in out["final_response"].lower()


def test_superlative_uses_sql_path(graph_bundle):
    out, spy, _ = _turn(
        graph_bundle, "What is the highest-grossing movie of all time?"
    )
    assert out["routing_decision"].intent == IntentType.SUPERLATIVE_RANKING
    assert out["retrieved_movies"], "SQL path must return movies"
    assert out["final_response"]
    assert "can't help" not in out["final_response"]


def test_semantic_query_returns_grounded_response(graph_bundle):
    out, spy, tracer = _turn(
        graph_bundle, "recommend a mind-bending dream heist movie"
    )
    assert out["retrieved_movies"], "semantic path must return movies"
    assert out["synthesis_usage"].completion_tokens > 0
    # CWA: every title the LLM bolded must come from the retrieved context
    assert out.get("final_response")
    violations = [
        t["payload"].get("cwa_violations")
        for t in tracer.traces()
        if t["node"] == "synthesize"
    ]
    assert violations and violations[-1] == [], f"CWA violated: {violations}"


def test_exclusion_preference_respected(graph_bundle):
    out, _, _ = _turn(
        graph_bundle,
        "recommend me a scary movie but nothing with too much gore",
        session_preferences=UserSessionPreferences(excluded_genres=["Horror"]),
    )
    # structured exclusions must reach the engine and hold in results
    for movie in out["retrieved_movies"]:
        assert "Horror" not in movie.genres


def test_coreference_followup_uses_standalone_query(graph_bundle):
    # Turn 1 establishes focus; turn 2 corefers to it
    out1, _, _ = _turn(graph_bundle, "Recommend a movie like Inception")
    assert out1["retrieved_movies"]
    out2, spy, _ = _turn(graph_bundle, "what else did that director make?")
    assert spy.calls, "follow-up should retrieve"
    # router must have resolved the coreference into a standalone query
    assert out2["routing_decision"].standalone_query.lower() != "what else did that director make?"


def test_local_traces_captured_for_every_turn(graph_bundle):
    out, _, tracer = _turn(graph_bundle, "a heist comedy from the 2000s")
    nodes = [t["node"] for t in tracer.traces()]
    assert nodes, "every turn must leave local traces"
    assert nodes[-1] in ("synthesize", "pivot", "refusal")
