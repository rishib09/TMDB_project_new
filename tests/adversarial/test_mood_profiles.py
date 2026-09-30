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
from langchain_core.messages import HumanMessage

from src.domain.config import ExperimentConfig
from src.domain.memory import UserSessionPreferences
from src.domain.moods import (
    MOOD_PROFILES,
    MoodProfile,
    expand_query_text,
    merge_profile_floors,
    mood_boost_spec,
    resolve_mood_profile,
)
from src.domain.movie import MovieRecord
from src.domain.routing import IntentType, MetadataFilterCriteria, QueryRoutingDecision
from src.graph.orchestrator import build_maya_graph
from src.graph.state import SynthesisUsage
from src.indexing.vector_store import SearchResult
from src.observability.tracer import DualModeObservabilityManager
from src.retrieval.hybrid_engine import HybridRetrievalEngine, RetrievalResult
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
    fused = engine._rrf_fuse(dense, sparse, boost=mood_boost_spec(profile))
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
def test_hidden_gem_floors_do_not_exclude_its_own_targets():
    """An inverted profile must not carry floors that kill its constituency."""
    profile: MoodProfile | None = resolve_mood_profile("hidden-gem")
    assert profile is not None
    assert profile.floors.min_vote_count is None or profile.floors.min_vote_count < 5_000


# --- review rectifications (#137) -----------------------------------------------------


class _CaptureEngine:
    """Records every retrieve call (query + boost) for orchestrator assertions."""

    def __init__(self, movies=None, empty_first_n: int = 0):
        self.movies = movies or []
        self.empty_first_n = empty_first_n
        self.calls: list[dict] = []

    def retrieve(self, query, routing, top_k=8, candidate_pool=50, shown_ids=None, boost=None):
        self.calls.append({"query": query, "routing": routing, "boost": boost})
        if len(self.calls) <= self.empty_first_n:
            return []
        return [
            RetrievalResult(movie=m, score=1.0, source="sql") for m in self.movies
        ]


class _ScriptedRouter:
    def __init__(self, decision: QueryRoutingDecision):
        self.decision = decision

    def route(self, query, state, feedback=None):
        return self.decision


class _FakeSynth:
    def synthesize(self, query, decision, movies, history):
        return "ok", SynthesisUsage(model="fake", prompt_tokens=1, completion_tokens=1)


@pytest.mark.adversarial
def test_mapped_mood_is_not_reappended_as_text_flavor():
    """Profile-mapped moods must not still leak as '(mood: epic)' text flavor.

    The measured failure mode (#137) was mood-as-BM25-token; phrasebook replaces
    that path — re-appending the mood tag undoes the fix for mapped moods.
    """
    engine = _CaptureEngine(movies=[make_movie(id=98, title="Gladiator", vote_count=18_000)])
    decision = make_routing(
        standalone_query="show me an epic movie for solo viewing",
        mood="epic",
    )
    graph = build_maya_graph(
        ExperimentConfig(mood_profiles_enabled=True),
        _ScriptedRouter(decision),
        engine,
        _FakeSynth(),
        DualModeObservabilityManager(session_id="adv-137-flavor"),
    )
    graph.invoke({
        "messages": [HumanMessage(content=decision.standalone_query)],
        "session_preferences": UserSessionPreferences(
            preferred_mood="epic", audience="solo",
        ),
    })
    assert engine.calls, "retrieve must run"
    query = engine.calls[0]["query"]
    assert "(mood: epic" not in query
    assert "sweeping" in query.lower()  # phrasebook still applied
    assert "audience: solo" in query  # audience flavor is unrelated to MoodProfile


@pytest.mark.adversarial
def test_unmapped_mood_keeps_text_flavor_fail_open():
    """Unknown moods still fail open to pre-#137 flavor-only behavior."""
    engine = _CaptureEngine(movies=[make_movie(id=99, title="Whatever")])
    decision = make_routing(
        standalone_query="something whimsical for solo viewing",
        mood="whimsical",
        audience="solo",
    )
    graph = build_maya_graph(
        ExperimentConfig(mood_profiles_enabled=True),
        _ScriptedRouter(decision),
        engine,
        _FakeSynth(),
        DualModeObservabilityManager(session_id="adv-137-unmapped"),
    )
    graph.invoke({
        "messages": [HumanMessage(content=decision.standalone_query)],
        "session_preferences": UserSessionPreferences(
            preferred_mood="whimsical", audience="solo",
        ),
    })
    assert engine.calls, "mood+audience must retrieve (not probe)"
    assert "mood: whimsical" in engine.calls[0]["query"]
    assert "audience: solo" in engine.calls[0]["query"]


@pytest.mark.adversarial
def test_genre_match_retry_keeps_boost():
    """#25 genre_match relaxation must re-pass the MoodProfile boost."""
    engine = _CaptureEngine(
        movies=[make_movie(id=7, title="Dune", genres=["Science Fiction", "Adventure"])],
        empty_first_n=1,
    )
    decision = make_routing(
        standalone_query="epic sci-fi for solo viewing",
        mood="epic",
        audience="solo",
        filters=MetadataFilterCriteria(
            genres=["War", "History", "Fantasy", "Adventure"],
            genre_match="all",
        ),
    )
    graph = build_maya_graph(
        ExperimentConfig(mood_profiles_enabled=True),
        _ScriptedRouter(decision),
        engine,
        _FakeSynth(),
        DualModeObservabilityManager(session_id="adv-137-genre-retry"),
    )
    graph.invoke({
        "messages": [HumanMessage(content=decision.standalone_query)],
        "session_preferences": UserSessionPreferences(
            preferred_mood="epic", audience="solo",
        ),
    })
    assert len(engine.calls) == 2, "intersection miss must retry ANY-match"
    assert engine.calls[0]["boost"] is not None
    assert engine.calls[1]["boost"] is not None
    assert engine.calls[1]["boost"].profile_id == "epic@1"
    assert engine.calls[1]["routing"].filters.genre_match == "any"


@pytest.mark.adversarial
def test_hard_floors_do_not_fail_open(db):
    """An impossible vote_count_min must stay empty — no silent floor drop."""
    engine = HybridRetrievalEngine(db=db, vector_store=None, reranker_enabled=False)
    profile = resolve_mood_profile("epic")
    assert profile is not None
    boost = mood_boost_spec(profile)
    routing = make_routing(filters=MetadataFilterCriteria(vote_count_min=10_000_000))
    results = engine.retrieve("epic movie for solo viewing", routing, boost=boost)
    assert results == []
    assert engine.last_profile_applied == {"profile_id": "epic@1"}
    assert "floors_relaxed" not in (engine.last_profile_applied or {})


# --- review round: the Trace must tell the whole #137 story -------------------------


def _retrieve_payloads(tracer) -> list[dict]:
    return [t["payload"] for t in tracer.traces() if t["node"] == "retrieve"]


def _run_v1_turn(config: ExperimentConfig, mood: str) -> tuple[object, DualModeObservabilityManager]:
    """One v1 retrieve turn with the given config; returns (graph, tracer)."""
    engine = _CaptureEngine(movies=[make_movie(id=98, title="Gladiator", vote_count=18_000)])
    decision = make_routing(
        standalone_query="epic movies for solo viewing",
        mood=mood,
        audience="solo",
    )
    tracer = DualModeObservabilityManager(session_id=f"adv-137-trace-{id(config)}")
    graph = build_maya_graph(config, _ScriptedRouter(decision), engine, _FakeSynth(), tracer)
    graph.invoke({
        "messages": [HumanMessage(content=decision.standalone_query)],
        "session_preferences": UserSessionPreferences(
            preferred_mood=mood, audience="solo",
        ),
    })
    return tracer


@pytest.mark.adversarial
def test_trace_records_profile_calibration_and_enforced_floors():
    """Telemetry rule: the applied payload must show WHICH calibration ran —
    an A/B comparison of boost weights is meaningless without it — and the
    floors the engine actually enforced (post-merge), not just the declared
    ones."""
    tracer = _run_v1_turn(ExperimentConfig(mood_profiles_enabled=True), "epic")
    payload = next(
        p for p in _retrieve_payloads(tracer) if p.get("mood_profile") == "epic@1"
    )
    assert payload["floors"]["min_vote_count"] == 3000  # declared
    assert payload["floors_enforced"] == {"vote_count_min": 3000}  # post-merge
    assert payload["boost_scale"] == 1.0
    assert payload["calibration"] == {
        "normalizer": 0.01,
        "runtime_weight": 0.5,
        "term_cap": 2.0,
        "popularity_log_divisor": 10.0,
        "revenue_log_divisor": 25.0,
    }


@pytest.mark.adversarial
def test_flag_off_mood_run_is_recorded_in_trace():
    """The mood_profiles_enabled=False A/B arm must be on the record — the
    Trace cannot otherwise distinguish a profile-off run from a no-mood run."""
    tracer = _run_v1_turn(ExperimentConfig(mood_profiles_enabled=False), "epic")
    payload = next(p for p in _retrieve_payloads(tracer) if "mood_profile" in p)
    assert payload["mood_profile"] == "disabled"
    assert payload["mood"] == "epic"
