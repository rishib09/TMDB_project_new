"""Live integration tests for the hybrid retrieval engine (issue #4).

Runs against the REAL SQLite database, REAL ChromaDB collections and the
REAL FlashRank reranker. Parametrized over whichever tier collections are
complete on disk — v1_2 joins automatically once built.
"""

import time

import pytest

from src.domain.routing import (
    IntentType,
    MetadataFilterCriteria,
    QueryRoutingDecision,
    SuperlativeCriteria,
    SuperlativeMetric,
)
from src.indexing.vector_store import MovieVectorStore
from src.retrieval.hybrid_engine import HybridRetrievalEngine
from src.storage.database import MovieDatabase


def _available_versions():
    """TIER_PROFILES ∩ collections complete on disk (count == 9119, metadata matches)."""
    store = MovieVectorStore("data/chroma_db")
    available = []
    for version_name, profile in MovieVectorStore.TIER_PROFILES.items():
        try:
            col = store.client.get_collection(version_name)
            meta = col.metadata or {}
            if (
                col.count() == 9119
                and meta.get("embedding_model") == profile["embedding_model"]
                and meta.get("tier") == profile["tier"]
            ):
                available.append(version_name)
        except Exception:
            continue
    return available


AVAILABLE_VERSIONS = _available_versions()


@pytest.fixture(scope="module", params=AVAILABLE_VERSIONS)
def engine(request):
    db = MovieDatabase("data/tmdb_movies.db")
    store = MovieVectorStore("data/chroma_db")
    return HybridRetrievalEngine(
        db=db, vector_store=store, rag_version=request.param,
        hybrid_alpha=0.5, reranker_enabled=False,
    )


@pytest.fixture(scope="module")
def db():
    return MovieDatabase("data/tmdb_movies.db")


def make_routing(**kwargs) -> QueryRoutingDecision:
    defaults = dict(
        intent=IntentType.SEMANTIC_SEARCH,
        confidence=0.9,
        standalone_query="query",
        requires_rag=True,
    )
    defaults.update(kwargs)
    return QueryRoutingDecision(**defaults)


# --- deterministic SQL path -----------------------------------------------------


@pytest.mark.integration
def test_superlative_matches_ground_truth(engine, db):
    """Highest-grossing 1970 must equal the raw SQL answer — exactly."""
    routing = make_routing(
        intent=IntentType.SUPERLATIVE_RANKING,
        superlative=SuperlativeCriteria(metric=SuperlativeMetric.REVENUE, year=1970),
    )
    results = engine.retrieve("highest-grossing film of 1970", routing, top_k=5)

    assert len(results) == 5
    assert all(r.source == "sql" for r in results)

    ground_truth = db.query_superlative(metric="REVENUE", year=1970, limit=5)
    assert [r.movie.id for r in results] == [m.id for m in ground_truth]


@pytest.mark.integration
def test_exact_attribute_filter_sql(engine):
    routing = make_routing(
        intent=IntentType.ATTRIBUTE_FILTER,
        filters=MetadataFilterCriteria(exact_year=1999, director="Matrix", cast_member="Keanu Reeves"),
    )
    # Director "Matrix" matches nobody — use realistic values instead:
    routing = make_routing(
        intent=IntentType.ATTRIBUTE_FILTER,
        filters=MetadataFilterCriteria(exact_year=1999, cast_member="Keanu Reeves"),
    )
    results = engine.retrieve("Keanu Reeves in 1999", routing, top_k=5)

    assert len(results) > 0
    for r in results:
        assert r.movie.release_year == 1999
        assert any(c.name == "Keanu Reeves" for c in r.movie.cast)


# --- hybrid fusion path ----------------------------------------------------------


@pytest.mark.integration
def test_semantic_plot_query_finds_known_movie(engine):
    routing = make_routing(standalone_query="mind-bending dream heist in the subconscious")
    results = engine.retrieve(
        "mind-bending dream heist in the subconscious", routing, top_k=5
    )

    assert len(results) > 0
    titles = [r.movie.title for r in results]
    assert any("Inception" in t for t in titles), f"Inception missing from {titles}"


@pytest.mark.integration
def test_rrf_merges_both_sources(engine):
    routing = make_routing(standalone_query="space exploration survival")
    results = engine.retrieve("space exploration survival", routing, top_k=10)

    dual = [r for r in results if r.dense_rank is not None and r.sparse_rank is not None]
    assert dual, "expected at least one movie retrieved by BOTH dense and BM25"


@pytest.mark.integration
def test_results_have_poster_paths(engine):
    routing = make_routing(standalone_query="heartwarming family adventure")
    results = engine.retrieve("heartwarming family adventure", routing, top_k=8)
    assert len(results) > 0
    for r in results:
        assert r.movie.poster_path or r.movie.poster_url


@pytest.mark.integration
def test_session_exclusion_enforced_live(engine):
    routing = make_routing(
        intent=IntentType.NEGATION_EXCLUSION,
        filters=MetadataFilterCriteria(excluded_actors=["Tom Cruise"]),
    )
    results = engine.retrieve("action blockbusters", routing, top_k=8)
    for r in results:
        assert "Tom Cruise" not in {c.name for c in r.movie.cast}


@pytest.mark.integration
def test_year_range_filter_live(engine):
    routing = make_routing(
        intent=IntentType.ATTRIBUTE_FILTER,
        filters=MetadataFilterCriteria(year_min=2020),
    )
    results = engine.retrieve("recent superhero blockbusters", routing, top_k=8)
    assert len(results) > 0
    for r in results:
        assert r.movie.release_year >= 2020


# --- reranker mechanism (quality A/B belongs to #6) ------------------------------


@pytest.mark.integration
def test_reranker_runs_and_returns_reranked_page(engine):
    """Reranking is a measured knob, not a default: TinyBERT live-measured WORSE
    than pure RRF on the golden query (2026-08-31). This test pins the MECHANISM."""
    engine.reranker_enabled = True
    routing = make_routing(standalone_query="cyberpunk detective mystery")
    results = engine.retrieve("cyberpunk detective mystery", routing, top_k=5)
    engine.reranker_enabled = False

    assert len(results) == 5
    assert all(r.source == "reranked" for r in results)


# --- latency ---------------------------------------------------------------------


@pytest.mark.integration
def test_hybrid_retrieval_latency(engine):
    """Full pipeline (dense + BM25 + RRF) under 2s on CPU, reranker off."""
    routing = make_routing(standalone_query="cyberpunk detective mystery")
    engine.retrieve("cyberpunk detective mystery", routing)  # warmup

    start = time.perf_counter()
    results = engine.retrieve("cyberpunk detective mystery", routing, top_k=8)
    elapsed = time.perf_counter() - start

    assert len(results) > 0
    assert elapsed < 2.0, f"retrieval took {elapsed:.2f}s"
