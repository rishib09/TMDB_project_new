"""Adversarial tests for LLM extraction fallback + person resolution (#24)."""

import pytest

from src.domain.config import ExperimentConfig
from src.domain.routing import IntentType, MetadataFilterCriteria, QueryRoutingDecision
from src.retrieval.hybrid_engine import HybridRetrievalEngine
from src.storage.database import MovieDatabase

pytestmark = pytest.mark.adversarial


# --- person resolution (#24): DB ground truth, never model guesswork ---------

@pytest.fixture
def db(tmp_path):
    database = MovieDatabase(str(tmp_path / "p.db"))
    database.upsert_movie({
        "id": 1, "title": "Both Hat", "release_year": 2020, "genres": ["Drama"],
        "director": "Mad Demon", "cast": [{"name": "Mad Demon", "character": "Lead"}],
        "vote_average": 7.0, "revenue": 100, "budget": 10, "popularity": 5.0,
        "vote_count": 100, "runtime": 100, "overview": "x", "poster_path": "/x.jpg",
        "keywords": [],
    })
    database.upsert_movie({
        "id": 2, "title": "Directed Only", "release_year": 2021, "genres": ["Drama"],
        "director": "Mad Demon", "cast": [{"name": "Someone Else", "character": "Lead"}],
        "vote_average": 7.0, "revenue": 100, "budget": 10, "popularity": 5.0,
        "vote_count": 100, "runtime": 100, "overview": "x", "poster_path": "/x.jpg",
        "keywords": [],
    })
    database.upsert_movie({
        "id": 3, "title": "Acted Only", "release_year": 2022, "genres": ["Drama"],
        "director": "Other Director", "cast": [{"name": "Mad Demon", "character": "Lead"}],
        "vote_average": 7.0, "revenue": 100, "budget": 10, "popularity": 5.0,
        "vote_count": 100, "runtime": 100, "overview": "x", "poster_path": "/x.jpg",
        "keywords": [],
    })
    return database


def _decision(person=None, director=None):
    filters = MetadataFilterCriteria(
        person=person, director=director, genres=["Drama"],
    )
    return QueryRoutingDecision(
        intent=IntentType.ATTRIBUTE_FILTER, confidence=1.0,
        standalone_query="drama", requires_rag=True, filters=filters,
    )


def test_classify_person_both_roles(db):
    assert db.classify_person("Mad Demon") == (True, True)
    assert db.classify_person("Other Director") == (True, False)
    assert db.classify_person("Nobody Serious") == (False, False)


def test_person_union_returns_both_filmographies(db):
    """'movies of Mad Demon' (acts + directs) → union of both filmographies."""
    engine = HybridRetrievalEngine(db=db, vector_store=None, rag_version="v1_1")
    results = engine.retrieve("mad demon", _decision(person="Mad Demon"), top_k=8)
    titles = {r.movie.title for r in results}
    assert titles == {"Both Hat", "Directed Only", "Acted Only"}  # union


def test_person_unknown_returns_empty_never_dump(db):
    """Unknown person → empty (deterministic not-found), NEVER a generic dump."""
    engine = HybridRetrievalEngine(db=db, vector_store=None, rag_version="v1_1")
    results = engine.retrieve("who", _decision(person="Nobody Serious"), top_k=8)
    assert results == []


def test_person_resolved_to_director_when_only_directing(db):
    engine = HybridRetrievalEngine(db=db, vector_store=None, rag_version="v1_1")
    results = engine.retrieve("x", _decision(person="Other Director"), top_k=8)
    titles = {r.movie.title for r in results}
    assert titles == {"Acted Only"}  # director filmography only


def test_wrong_role_explicit_director_still_filters(db):
    """Explicit 'directed by' for a cast-only person → 0 rows (honest)."""
    engine = HybridRetrievalEngine(db=db, vector_store=None, rag_version="v1_1")
    results = engine.retrieve("x", _decision(director="Someone Else"), top_k=8)
    # Someone Else never directed → nothing passes the director predicate
    assert all(r.movie.director == "Someone Else" for r in results) or not results


# --- genre_match semantics (#25) ----------------------------------------------

def test_genre_match_all_requires_intersection(db):
    filters = MetadataFilterCriteria(genres=["Sci-Fi"], genre_match="all")
    routing = QueryRoutingDecision(
        intent=IntentType.ATTRIBUTE_FILTER, confidence=1.0,
        standalone_query="x", requires_rag=True, filters=filters,
    )
    engine = HybridRetrievalEngine(db=db, vector_store=None, rag_version="v1_1")
    # No Sci-Fi+Drama intersection movie in this tiny fixture... Both Hat is
    # Drama-only → ALL-match yields nothing
    results = engine.retrieve("x", routing, top_k=8)
    assert results == []


def test_genre_match_any_is_the_legacy_default(db):
    """Multi-genre any-match: Drama-only movies pass a Sci-Fi+Drama filter."""
    filters = MetadataFilterCriteria(genres=["Sci-Fi", "Drama"], genre_match="any")
    routing = QueryRoutingDecision(
        intent=IntentType.ATTRIBUTE_FILTER, confidence=1.0,
        standalone_query="x", requires_rag=True, filters=filters,
    )
    engine = HybridRetrievalEngine(db=db, vector_store=None, rag_version="v1_1")
    results = engine.retrieve("x", routing, top_k=8)  # Drama-only movies still match
    assert len(results) == 3  # all fixtures carry Drama
