"""Adversarial tests for Mood Profiles (issue #137).

Red on pre-profile code — the whole point. Root class of #34: a mood word
entered retrieval only as BM25 keyword material, so title collisions
(*Epic* (2013)) ranked above true epics and 0.0-vote junk (*Dance 7*,
*Holy Joe*) survived into the top 5.

These tests pin the NEW contract:
- epic floors (vote_count_min) exclude junk deterministically
- phrasebook expansion dilutes the single mood token in the sparse leg
- hidden-gem boosts invert polarity (low-popularity canon beats blockbusters)
- unknown moods resolve to None and change nothing (fail-open, on record)
"""

import pytest

from src.domain.moods import (
    MOOD_PROFILES,
    MoodProfile,
    boost_spec_of,
    expand_query_text,
    merge_profile_floors,
    resolve_mood_profile,
)
from src.domain.movie import MovieRecord
from src.domain.routing import IntentType, MetadataFilterCriteria, QueryRoutingDecision
from src.indexing.vector_store import SearchResult
from src.retrieval.hybrid_engine import HybridRetrievalEngine
from src.storage.database import MovieDatabase


@pytest.fixture(scope="module")
def db():
    return MovieDatabase("data/tmdb_movies.db")


def make_movie(**kwargs) -> MovieRecord:
    defaults = dict(
        id=1,
        title="Some Movie",
        release_year=2000,
        runtime=120,
        vote_average=7.0,
        vote_count=10_000,
        popularity=50.0,
        revenue=100_000_000,
        genres=["Action"],
    )
    defaults.update(kwargs)
    return MovieRecord(**defaults)


def make_routing(**kwargs) -> QueryRoutingDecision:
    defaults = dict(
        intent=IntentType.SEMANTIC_SEARCH,
        confidence=0.9,
        standalone_query="epic movie for solo viewing",
        requires_rag=True,
        mood="epic",
    )
    defaults.update(kwargs)
    return QueryRoutingDecision(**defaults)


# --- registry contract ---------------------------------------------------------


@pytest.mark.adversarial
def test_every_profile_mood_is_canonical():
    """A profile may only exist for a mood the vocabularies can emit."""
    from src.maya.probing import _MOOD_VOCAB

    canonical = set(_MOOD_VOCAB.values())
    for profile in MOOD_PROFILES.values():
        assert profile.mood in canonical, f"{profile.mood}@{profile.version} is off-vocab"


@pytest.mark.adversarial
def test_unknown_mood_resolves_none():
    """Unmapped mood -> None; the caller must fail open with no crash."""
    assert resolve_mood_profile("epic") is not None
    assert resolve_mood_profile("wheelchair-chase-thriller") is None


# --- floors: the junk killer -----------------------------------------------------


@pytest.mark.adversarial
def test_epic_floors_exclude_zero_vote_junk(db):
    """*Dance 7* (1975, 0.0 rating, tiny vote_count) must not survive epic floors.

    The old pipeline had no floor at all — junk reached the top 5 live.
    """
    engine = HybridRetrievalEngine(db=db, vector_store=None, reranker_enabled=False)
    profile = resolve_mood_profile("epic")
    assert profile is not None and profile.floors.min_vote_count is not None

    filters = merge_profile_floors(None, profile)
    assert filters.vote_count_min == profile.floors.min_vote_count

    routing = make_routing(filters=filters)
    results = engine.retrieve("epic movie for solo viewing", routing)
    assert results, "floors must not empty the candidate set"
    assert all(r.movie.vote_count >= filters.vote_count_min for r in results)


@pytest.mark.adversarial
def test_matches_filters_enforces_vote_count_min():
    """The static post-filter honors the additive field on every path."""
    filters = MetadataFilterCriteria(vote_count_min=3000)
    junk = make_movie(id=2, title="Dance 7", vote_count=12, vote_average=0.0)
    real = make_movie(id=3, title="Gladiator", vote_count=18_000)
    assert HybridRetrievalEngine.matches_filters(junk, filters) is False
    assert HybridRetrievalEngine.matches_filters(real, filters) is True
    # None floor / None filters stay permissive (additive, #82 C6 pattern)
    assert HybridRetrievalEngine.matches_filters(junk, MetadataFilterCriteria()) is True
    assert HybridRetrievalEngine.matches_filters(junk, None) is True


# --- phrasebook: the title-collision killer ---------------------------------------


@pytest.mark.adversarial
def test_expansion_dilutes_single_mood_token():
    """The shaped query must carry more than the one stemmed token 'epic'.

    Old code sent 'epic movie for solo viewing' to BM25 — the *Epic* (2013)
    title match dominated. Expansion adds deterministic connotation anchors.
    """
    profile = resolve_mood_profile("epic")
    expanded = expand_query_text("epic movie for solo viewing", profile)
    assert expanded != "epic movie for solo viewing"
    assert "sweeping" in expanded.lower()
    # determinism: same inputs, same output — no LLM in the loop
    assert expanded == expand_query_text("epic movie for solo viewing", profile)


# --- boosts: polarity inversion proof (hidden gem) --------------------------------


def _search(movie: MovieRecord, dense_rank: int) -> SearchResult:
    return SearchResult(id=movie.id, score=0.0, movie=movie, document_text=movie.overview)


@pytest.mark.adversarial
def test_hidden_gem_boost_inverts_popularity(db):
    """At identical RRF contribution, hidden-gem ranks the quiet canon film
    ABOVE the blockbuster — proof boosts can reverse default polarity."""
    engine = HybridRetrievalEngine(db=db, vector_store=None, reranker_enabled=False)
    gem = make_movie(id=10, title="Quiet Gem", popularity=2.0, vote_average=8.1, vote_count=4_100)
    bomb = make_movie(id=11, title="Loud Blockbuster", popularity=300.0, vote_average=6.4, vote_count=9_000)
    dense = [_search(bomb, 1), _search(gem, 2)]
    sparse = [gem, bomb]

    profile = resolve_mood_profile("hidden-gem")
    assert profile is not None
    fused = engine._rrf_fuse(dense, sparse, boost=boost_spec_of(profile))
    order = [r.movie.id for r in fused]
    assert order[0] == gem.id, f"hidden-gem boost must favor {gem.title}, got {order}"


@pytest.mark.adversarial
def test_boost_none_preserves_stock_rrf(db):
    """boost=None must be byte-identical to today's fusion (additive seam)."""
    engine = HybridRetrievalEngine(db=db, vector_store=None, reranker_enabled=False)
    a = make_movie(id=20, title="A", popularity=1.0)
    b = make_movie(id=21, title="B", popularity=900.0)
    dense = [_search(a, 1), _search(b, 2)]
    sparse = [b, a]
    plain = engine._rrf_fuse(dense, sparse)
    assert [r.movie.id for r in plain] == [a.id, b.id]
    assert all(r.source == "rrf" for r in plain)


# --- MoodProfile schema integrity ---------------------------------------------------


@pytest.mark.adversarial
def test_hidden_gem_floors_do_not_excluse_its_own_targets():
    """An inverted profile must not carry floors that kill its constituency."""
    profile: MoodProfile | None = resolve_mood_profile("hidden-gem")
    assert profile is not None
    assert profile.floors.min_vote_count is None or profile.floors.min_vote_count < 5_000
