"""Unit tests for the hybrid retrieval engine (issue #4).

RRF fusion math, filter matching, and SQL path selection — all with stubs,
no models, no database. Live behavior is covered by integration tests.
"""

import pytest

from src.domain.movie import CastMember, MovieRecord
from src.domain.routing import (
    IntentType,
    MetadataFilterCriteria,
    QueryRoutingDecision,
    SuperlativeCriteria,
    SuperlativeMetric,
)
from src.indexing.vector_store import SearchResult
from src.retrieval.hybrid_engine import HybridRetrievalEngine, RetrievalResult


def make_movie(id: int, title: str, **kwargs) -> MovieRecord:
    kwargs.setdefault("release_year", 2000)
    return MovieRecord(id=id, title=title, **kwargs)


def make_dense(id: int, title: str, rank: int) -> SearchResult:
    return SearchResult(
        id=id,
        score=0.9,
        movie=make_movie(id, title),
        document_text=f"{title} overview text",
    )


def make_sparse(id: int, title: str) -> MovieRecord:
    return make_movie(id, title)


def make_result(id: int, title: str, dense_rank=None, sparse_rank=None) -> RetrievalResult:
    return RetrievalResult(
        movie=make_movie(id, title),
        score=0.0,
        source="rrf",
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
    )


def make_routing(**kwargs) -> QueryRoutingDecision:
    defaults = dict(
        intent=IntentType.SEMANTIC_SEARCH,
        confidence=0.9,
        standalone_query="some query",
        requires_rag=True,
    )
    defaults.update(kwargs)
    return QueryRoutingDecision(**defaults)


@pytest.fixture
def engine():
    from unittest.mock import MagicMock

    return HybridRetrievalEngine(
        db=MagicMock(),
        vector_store=MagicMock(),
        reranker_enabled=False,
    )


# --- RRF fusion math ------------------------------------------------------------


class TestRRFFusion:
    def test_both_lists_score_sum(self, engine):
        """A movie on rank 1 of BOTH lists scores w/(k+1) + (1-w)/(k+1)."""
        dense = [make_dense(1, "Both", rank=0)]
        sparse = [make_sparse(1, "Both")]

        fused = engine._rrf_fuse(dense, sparse)

        expected = 0.5 / 61 + 0.5 / 61  # w/(k + rank) with rank=1, k=60
        assert len(fused) == 1
        assert fused[0].score == pytest.approx(expected, abs=1e-6)
        assert fused[0].dense_rank == 1
        assert fused[0].sparse_rank == 1

    def test_dual_source_beats_single_source(self, engine):
        """Agreement across retrievers is the whole point of RRF."""
        dense = [make_dense(1, "Agrees", rank=0), make_dense(2, "DenseOnly", rank=1)]
        sparse = [make_sparse(1, "Agrees"), make_sparse(3, "SparseOnly")]

        fused = engine._rrf_fuse(dense, sparse)

        assert fused[0].movie.id == 1
        assert fused[0].score > fused[1].score
        assert fused[0].dense_rank == 1 and fused[0].sparse_rank == 1
        assert fused[1].sparse_rank is None
        assert fused[2].dense_rank is None

    def test_alpha_one_is_pure_dense_order(self, engine):
        engine.hybrid_alpha = 1.0
        dense = [make_dense(1, "D1", rank=0), make_dense(2, "D2", rank=1)]
        sparse = [make_sparse(2, "S1"), make_sparse(3, "S2")]

        fused = engine._rrf_fuse(dense, sparse)

        assert [r.movie.id for r in fused[:2]] == [1, 2]

    def test_alpha_zero_is_pure_sparse_order(self, engine):
        engine.hybrid_alpha = 0.0
        dense = [make_dense(1, "D1", rank=0)]
        sparse = [make_sparse(2, "S1"), make_sparse(3, "S2")]

        fused = engine._rrf_fuse(dense, sparse)

        assert [r.movie.id for r in fused[:2]] == [2, 3]

    def test_rrf_k_constant(self, engine):
        assert engine.RRF_K == 60  # standard, per issue #4

    def test_document_text_preserved_from_dense(self, engine):
        dense = [make_dense(1, "T", rank=0)]
        fused = engine._rrf_fuse(dense, [])
        assert fused[0].document_text == "T overview text"


# --- SQL path selection -----------------------------------------------------------


class TestSQLPathSelection:
    def test_superlative_goes_to_sql(self, engine):
        routing = make_routing(
            intent=IntentType.SUPERLATIVE_RANKING,
            superlative=SuperlativeCriteria(metric=SuperlativeMetric.REVENUE),
        )
        assert engine._use_sql_path(routing) is True

    def test_exact_attribute_filter_goes_to_sql(self, engine):
        for exact in [
            MetadataFilterCriteria(exact_year=1999),
            MetadataFilterCriteria(director="Nolan"),
            MetadataFilterCriteria(cast_member="Tom Hanks"),
        ]:
            routing = make_routing(intent=IntentType.ATTRIBUTE_FILTER, filters=exact)
            assert engine._use_sql_path(routing) is True

    def test_fuzzy_filters_go_hybrid(self, engine):
        routing = make_routing(
            intent=IntentType.ATTRIBUTE_FILTER,
            filters=MetadataFilterCriteria(genres=["Horror"], year_min=1980),
        )
        assert engine._use_sql_path(routing) is False

    def test_semantic_search_goes_hybrid(self, engine):
        assert engine._use_sql_path(make_routing()) is False


# --- filter matching ----------------------------------------------------------------


class TestMatchesFilters:
    def test_year_bounds(self):
        movie = make_movie(1, "T", release_year=1985)
        assert HybridRetrievalEngine.matches_filters(
            movie, MetadataFilterCriteria(year_min=1980, year_max=1989)
        )
        assert not HybridRetrievalEngine.matches_filters(
            movie, MetadataFilterCriteria(year_min=1990)
        )
        assert HybridRetrievalEngine.matches_filters(
            movie, MetadataFilterCriteria(exact_year=1985)
        )

    def test_genres_any_of(self):
        movie = make_movie(1, "T", genres=["Horror", "Mystery"])
        assert HybridRetrievalEngine.matches_filters(movie, MetadataFilterCriteria(genres=["Horror"]))
        assert not HybridRetrievalEngine.matches_filters(movie, MetadataFilterCriteria(genres=["Romance"]))

    def test_director_and_cast(self):
        movie = make_movie(
            1, "T", director="John Carpenter",
            cast=[CastMember(name="Kurt Russell", character="Snake")],
        )
        assert HybridRetrievalEngine.matches_filters(movie, MetadataFilterCriteria(director="carpenter"))
        assert HybridRetrievalEngine.matches_filters(movie, MetadataFilterCriteria(cast_member="kurt russell"))
        assert not HybridRetrievalEngine.matches_filters(movie, MetadataFilterCriteria(cast_member="Tom Cruise"))

    def test_none_filters_match_everything(self):
        assert HybridRetrievalEngine.matches_filters(make_movie(1, "T"), None)


# --- exclusions ---------------------------------------------------------------------


class TestExclusions:
    def test_excluded_genre_dropped(self, engine):
        routing = make_routing(
            intent=IntentType.NEGATION_EXCLUSION,
            filters=MetadataFilterCriteria(excluded_genres=["Horror"]),
        )
        movies = [
            make_movie(1, "Fun Comedy", genres=["Comedy"]),
            make_movie(2, "Scary", genres=["Horror"]),
        ]
        allowed = [m.id for m in movies if engine._is_allowed(m, routing.filters)]
        assert allowed == [1]

    def test_excluded_actor_dropped(self, engine):
        engine2 = engine
        routing = make_routing(
            intent=IntentType.NEGATION_EXCLUSION,
            filters=MetadataFilterCriteria(excluded_actors=["Tom Cruise"]),
        )
        movie = make_movie(
            1, "Action", cast=[CastMember(name="Tom Cruise", character="Pilot")]
        )
        assert engine2._is_allowed(movie, routing.filters) is False


# --- retrieve() orchestration (stubbed stores) ----------------------------------------


class TestRetrieveOrchestration:
    def test_requires_rag_false_returns_empty(self, engine):
        routing = make_routing(requires_rag=False)
        assert engine.retrieve("hi", routing) == []

    def test_sql_path_skips_vector_store(self, engine):
        routing = make_routing(
            intent=IntentType.SUPERLATIVE_RANKING,
            superlative=SuperlativeCriteria(metric=SuperlativeMetric.REVENUE, year=1970),
        )
        engine.db.query_superlative.return_value = [make_sparse(1, "Airport")]
        results = engine.retrieve("highest-grossing 1970", routing, top_k=5)

        engine.vector_store.search.assert_not_called()
        assert results[0].source == "sql"
        assert results[0].movie.poster_url  # poster paths present (acceptance criteria)

    def test_dense_failure_degrades_to_bm25(self, engine):
        engine.vector_store.search.side_effect = RuntimeError("chroma down")
        engine.db.search_bm25.return_value = [make_sparse(7, "Fallback")]
        results = engine.retrieve("space movies", make_routing(), top_k=5)

        assert len(results) == 1
        assert results[0].movie.id == 7
        assert results[0].source == "rrf"
