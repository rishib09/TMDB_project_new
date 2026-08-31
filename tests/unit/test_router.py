"""Offline mock tests for the MayaRouter (Issue #3).

Stubs the bound structured-output chain with recorded/anticipated
``QueryRoutingDecision`` fixtures. No network, runs in < 2s. Live fixtures
get recorded from the ``live`` tier runs and replayed here.
"""

from unittest.mock import MagicMock

import pytest

from src.domain.memory import ConversationState, UserSessionPreferences
from src.domain.routing import IntentType, MetadataFilterCriteria, QueryRoutingDecision
from src.maya.router import MAX_HISTORY_TURNS, ROUTER_SYSTEM_PROMPT, MayaRouter


@pytest.fixture
def config():
    from src.domain.config import ExperimentConfig

    return ExperimentConfig()


@pytest.fixture
def state():
    return ConversationState()


def stub_chain(monkeypatch, router: MayaRouter, decisions: list[QueryRoutingDecision]):
    """Replaces the bound LLM chain with a canned-decision recorder.

    Returns the list of message lists the router submitted per call.
    """
    captured: list[list] = []

    def fake_invoke(messages):
        captured.append(messages)
        return decisions.pop(0)

    chain = MagicMock()
    chain.invoke.side_effect = fake_invoke
    monkeypatch.setattr(router, "_chain", chain)
    return captured


# --- Routing passthrough -----------------------------------------------------

@pytest.mark.unit
def test_route_returns_llm_decision(config, state, monkeypatch):
    router = MayaRouter(config, api_key="test-key")
    expected = QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH,
        confidence=0.95,
        standalone_query="movies about space exploration and lunar colonies",
        requires_rag=True,
    )
    captured = stub_chain(monkeypatch, router, [expected])

    decision = router.route("movies about space exploration", state)

    # Normalization returns a corrected copy (requires_rag derived from intent).
    assert decision == expected
    assert len(captured) == 1
    # Prompt structure: system prompt first, user query last.
    assert ROUTER_SYSTEM_PROMPT in captured[0][0].content
    assert captured[0][-1].content == "movies about space exploration"


@pytest.mark.unit
def test_route_prompt_includes_trimmed_history(config, state, monkeypatch):
    for i in range(10):
        state.add_turn(f"question {i}", f"answer {i}", retrieved_movies=[])
    router = MayaRouter(config, api_key="test-key")
    expected = QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH,
        confidence=0.9,
        standalone_query="q",
        requires_rag=True,
    )
    captured = stub_chain(monkeypatch, router, [expected])

    router.route("follow up question", state)

    # 1 taxonomy system msg + 2*MAX_HISTORY_TURNS history + 1 user query.
    assert len(captured[0]) == 1 + MAX_HISTORY_TURNS * 2 + 1


@pytest.mark.unit
def test_route_prompt_includes_focused_entity(config, state, monkeypatch):
    from src.domain.movie import MovieRecord

    movie = MovieRecord(
        id=27205,
        title="Inception",
        release_date="2010-07-16",
        release_year=2010,
        overview="Dream heist",
        director="Christopher Nolan",
    )
    state.add_turn("show me Inception", "here it is", retrieved_movies=[movie])
    router = MayaRouter(config, api_key="test-key")
    expected = QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH,
        confidence=0.9,
        standalone_query="Inception plot",
        requires_rag=True,
    )
    captured = stub_chain(monkeypatch, router, [expected])

    router.route("who directed it?", state)

    # Context message(s) sit between the taxonomy prompt and the history.
    context_content = captured[0][1].content
    assert "Inception" in context_content
    assert "Christopher Nolan" in context_content


# --- Session exclusion merging ------------------------------------------------

@pytest.mark.unit
def test_session_exclusions_merged_into_filters(config, state, monkeypatch):
    state.session_preferences = UserSessionPreferences(excluded_genres=["Horror"])
    router = MayaRouter(config, api_key="test-key")
    decision = QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH,
        confidence=0.9,
        standalone_query="space movies",
        requires_rag=True,
        filters=MetadataFilterCriteria(),
    )
    stub_chain(monkeypatch, router, [decision])

    result = router.route("space movies", state)

    assert result.filters is not None
    assert result.filters.excluded_genres == ["Horror"]


@pytest.mark.unit
def test_exclusions_merged_when_llm_omits_filters_entirely(config, state, monkeypatch):
    state.session_preferences = UserSessionPreferences(excluded_actors=["Tom Cruise"])
    router = MayaRouter(config, api_key="test-key")
    decision = QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH,
        confidence=0.9,
        standalone_query="action movies",
        requires_rag=True,
        filters=None,
    )
    stub_chain(monkeypatch, router, [decision])

    result = router.route("action movies", state)

    assert result.filters is not None
    assert result.filters.excluded_actors == ["Tom Cruise"]


@pytest.mark.unit
def test_exclusions_not_duplicated(config, state, monkeypatch):
    state.session_preferences = UserSessionPreferences(excluded_genres=["Horror"])
    router = MayaRouter(config, api_key="test-key")
    decision = QueryRoutingDecision(
        intent=IntentType.NEGATION_EXCLUSION,
        confidence=0.9,
        standalone_query="thrillers without horror",
        requires_rag=True,
        filters=MetadataFilterCriteria(excluded_genres=["Horror", "Romance"]),
    )
    stub_chain(monkeypatch, router, [decision])

    result = router.route("thrillers without horror", state)

    assert result.filters is not None
    assert result.filters.excluded_genres == ["Horror", "Romance"]


@pytest.mark.unit
def test_exclusions_skipped_for_non_retrieval_intents(config, state, monkeypatch):
    state.session_preferences = UserSessionPreferences(excluded_genres=["Horror"])
    router = MayaRouter(config, api_key="test-key")
    decision = QueryRoutingDecision(
        intent=IntentType.GREETING,
        confidence=0.99,
        standalone_query="hi",
        requires_rag=False,
    )
    stub_chain(monkeypatch, router, [decision])

    result = router.route("hello there", state)

    assert result.filters is None


# --- Heuristic fallback --------------------------------------------------------

@pytest.mark.unit
def test_api_failure_triggers_fallback(config, state, monkeypatch):
    router = MayaRouter(config, api_key="test-key")
    chain = MagicMock()
    chain.invoke.side_effect = ConnectionError("openrouter unreachable")
    monkeypatch.setattr(router, "_chain", chain)

    decision = router.route("best sci-fi of the 1980s", state)

    assert decision.intent == IntentType.SEMANTIC_SEARCH
    assert decision.confidence == 0.1
    assert "Heuristic fallback" in decision.reasoning
    assert decision.standalone_query == "best sci-fi of the 1980s"


@pytest.mark.unit
def test_low_confidence_triggers_fallback(config, state, monkeypatch):
    router = MayaRouter(config, api_key="test-key", confidence_threshold=0.5)
    low_confidence = QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH,
        confidence=0.3,
        standalone_query="x",
        requires_rag=True,
    )
    stub_chain(monkeypatch, router, [low_confidence])

    decision = router.route("hello", state)

    assert decision.intent == IntentType.GREETING
    assert "low router confidence" in decision.reasoning


@pytest.mark.unit
@pytest.mark.parametrize(
    "query,expected_intent",
    [
        ("hi there", IntentType.GREETING),
        ("hey, good morning", IntentType.GREETING),
        ("what can you do?", IntentType.CAPABILITIES),
        ("best movies from the 1950s", IntentType.OUT_OF_SCOPE),
        ("great films of 1939", IntentType.OUT_OF_SCOPE),
        ("movies about space exploration", IntentType.SEMANTIC_SEARCH),
    ],
)
def test_heuristic_keyword_detection(config, state, query, expected_intent):
    router = MayaRouter(config, api_key="test-key")

    decision = router._heuristic_fallback(query, state, reason="test")

    assert decision.intent == expected_intent
