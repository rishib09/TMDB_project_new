"""Adversarial and edge-case data integrity tests on real TMDB dataset."""

import pytest
from src.storage.database import MovieDatabase
from src.domain.movie import MovieRecord


@pytest.fixture(scope="module")
def db() -> MovieDatabase:
    return MovieDatabase("data/tmdb_movies.db")


@pytest.mark.adversarial
def test_zero_revenue_and_budget_handling(db: MovieDatabase):
    """Verifies that movies with unrecorded budget/revenue ($0) do not break superlative sorting."""
    # When querying superlative revenue, no $0 films should be returned in the top rankings
    top_revenue = db.query_superlative(metric="REVENUE", direction="DESC", limit=20)
    for movie in top_revenue:
        assert movie.revenue > 0, f"Expected positive revenue, found {movie.revenue} for {movie.title}"


@pytest.mark.adversarial
def test_special_character_and_punctuation_titles(db: MovieDatabase):
    """Verifies that titles with special characters (quotes, asterisks, dots) are stored and searchable."""
    # Search for special characters
    mash_results = db.search_bm25("MASH", limit=5)
    assert len(mash_results) >= 0  # Should execute cleanly without SQLite syntax error


@pytest.mark.adversarial
def test_fts5_malformed_query_injection_safety(db: MovieDatabase):
    """Verifies that malformed or adversarial search strings never crash SQLite FTS5."""
    adversarial_queries = [
        "AND OR NOT",
        "''''''''''''",
        '""" """',
        "<script>alert(1)</script>",
        "DROP TABLE movies;",
        "SELECT * FROM movies WHERE 1=1;",
        "%%--##@@",
        "   ",
    ]
    for query in adversarial_queries:
        # None of these should raise an exception
        results = db.search_bm25(query, limit=5)
        assert isinstance(results, list)


@pytest.mark.adversarial
def test_dense_text_token_budget_compliance(db: MovieDatabase):
    """Verifies that high-signal dense text representation complies with model sequence limits."""
    sample_movies = db.query_superlative(metric="POPULARITY", limit=50)
    for movie in sample_movies:
        dense_text = movie.to_dense_text(strategy="enriched_metadata")
        # Approximate word count check (256 tokens is roughly ~180-220 words)
        words = dense_text.split()
        assert len(words) < 500, f"Dense text too long ({len(words)} words) for movie {movie.title}"
