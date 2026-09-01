"""Live integration tests running directly against the real TMDB SQLite database."""

import pytest

from src.domain.movie import MovieRecord
from src.storage.database import MovieDatabase


@pytest.fixture(scope="module")
def db() -> MovieDatabase:
    """Fixture providing access to the real ingested SQLite database."""
    return MovieDatabase("data/tmdb_movies.db")


@pytest.mark.integration
def test_real_dataset_volume_and_year_span(db: MovieDatabase):
    """Verifies that thousands of movies across 1970-2026 are present."""
    count = db.count()
    assert count >= 2000, f"Expected at least 2000 movies in dataset, found {count}"

    # Verify distribution across early, middle, and modern eras
    movies_1970 = db.query_superlative(metric="POPULARITY", year=1970, limit=10)
    movies_1999 = db.query_superlative(metric="POPULARITY", year=1999, limit=10)
    movies_2023 = db.query_superlative(metric="POPULARITY", year=2023, limit=10)

    assert len(movies_1970) > 0
    assert len(movies_1999) > 0
    assert len(movies_2023) > 0


@pytest.mark.integration
def test_real_movie_record_hydration_and_computed_properties(db: MovieDatabase):
    """Verifies that real movies hydrate with valid typed fields and computed poster URLs."""
    # Find Inception or Avatar
    all_nolan = db.search_bm25("Inception Christopher Nolan", limit=5)
    assert len(all_nolan) > 0

    movie = all_nolan[0]
    assert isinstance(movie, MovieRecord)
    assert movie.title == "Inception"
    assert movie.release_year == 2010
    assert movie.director == "Christopher Nolan"
    assert movie.poster_url.startswith("https://image.tmdb.org/t/p/w500/")
    assert len(movie.cast) > 0
    assert len(movie.genres) > 0


@pytest.mark.integration
def test_real_superlative_revenue_and_rating(db: MovieDatabase):
    """Verifies deterministic SQL superlative queries against real historical box office data."""
    # Top grossing film of 2009 should be Avatar
    top_2009 = db.query_superlative(metric="REVENUE", direction="DESC", year=2009, limit=1)
    assert len(top_2009) == 1
    assert "Avatar" in top_2009[0].title
    assert top_2009[0].revenue > 2_000_000_000

    # Top grossing film of 1997 should be Titanic
    top_1997 = db.query_superlative(metric="REVENUE", direction="DESC", year=1997, limit=1)
    assert len(top_1997) == 1
    assert "Titanic" in top_1997[0].title

    # Longest runtime in 1970-2026 should be valid (> 180 min)
    longest = db.query_superlative(metric="RUNTIME", direction="DESC", limit=3)
    assert len(longest) == 3
    assert longest[0].runtime >= 180


@pytest.mark.integration
def test_real_fts5_bm25_search_precision(db: MovieDatabase):
    """Verifies full-text lexical search against real cast, plot, and director data."""
    # Searching for "Godfather Mafia Corleone"
    results = db.search_bm25("Godfather Mafia Corleone", limit=5)
    assert len(results) > 0
    titles = [m.title for m in results]
    assert any("Godfather" in t for t in titles)

    # Searching for "Tarantino Pulp"
    pulp_results = db.search_bm25("Tarantino Pulp", limit=5)
    assert len(pulp_results) > 0
    assert any("Pulp Fiction" in m.title for m in pulp_results)
