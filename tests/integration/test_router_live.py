"""Live OpenRouter integration tests for the MayaRouter (Issue #3).

These hit the real OpenRouter API and are OPT-IN:
    pytest -m live -v          (requires OPENROUTER_API_KEY in env)

Recorded decisions from these runs become the replay fixtures for
tests/unit/test_router.py. Assertions check intent correctness only —
standalone-query wording varies between runs.
"""

import os

import pytest

from src.domain.config import ExperimentConfig
from src.domain.memory import ConversationState, UserSessionPreferences
from src.domain.movie import MovieRecord
from src.domain.routing import IntentType
from src.maya.router import MayaRouter

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set — skipping live OpenRouter tests",
    ),
]


@pytest.fixture(scope="module")
def router() -> MayaRouter:
    return MayaRouter(ExperimentConfig())


@pytest.fixture(scope="module")
def state() -> ConversationState:
    return ConversationState()


@pytest.fixture(scope="module")
def focused_state() -> ConversationState:
    """Session state with Inception as the focused entity for follow-ups."""
    state = ConversationState()
    inception = MovieRecord(
        id=27205,
        title="Inception",
        release_year=2010,
        overview="A thief enters dreams to steal secrets.",
        director="Christopher Nolan",
    )
    state.add_turn("tell me about Inception", "Inception (2010), directed by Christopher Nolan.", [inception])
    return state


# --- One live case per intent family ---------------------------------------------

@pytest.mark.live
def test_live_greeting(router, state):
    decision = router.route("hey Maya! what's up?", state)
    assert decision.intent == IntentType.GREETING
    assert decision.requires_rag is False


@pytest.mark.live
def test_live_capabilities(router, state):
    decision = router.route("what can you help me with?", state)
    assert decision.intent == IntentType.CAPABILITIES
    assert decision.requires_rag is False


@pytest.mark.live
def test_live_semantic_search(router, state):
    decision = router.route(
        "movies about space exploration and lunar colonies", state
    )
    assert decision.intent == IntentType.SEMANTIC_SEARCH
    assert decision.requires_rag is True
    assert len(decision.standalone_query) > 0


@pytest.mark.live
def test_live_attribute_filter(router, state):
    decision = router.route(
        "1980s horror movies directed by John Carpenter starring Kurt Russell", state
    )
    assert decision.intent == IntentType.ATTRIBUTE_FILTER
    assert decision.filters is not None
    assert decision.filters.year_min == 1980 or decision.filters.exact_year is None


@pytest.mark.live
def test_live_superlative(router, state):
    decision = router.route("What was the highest-grossing film of 1970?", state)
    assert decision.intent == IntentType.SUPERLATIVE_RANKING
    assert decision.superlative is not None
    assert decision.superlative.metric.value == "REVENUE"
    assert decision.superlative.year == 1970


@pytest.mark.live
def test_live_negation_exclusion(router, state):
    decision = router.route("action movies without Tom Cruise", state)
    assert decision.intent == IntentType.NEGATION_EXCLUSION
    assert decision.filters is not None
    assert "Tom Cruise" in decision.filters.excluded_actors


@pytest.mark.live
def test_live_out_of_scope_pre_1970(router, state):
    decision = router.route("what are the best movies from the 1950s?", state)
    assert decision.intent == IntentType.OUT_OF_SCOPE
    assert decision.requires_rag is False


@pytest.mark.live
def test_live_out_of_scope_non_film(router, state):
    decision = router.route("who won the 1998 FIFA World Cup final?", state)
    assert decision.intent == IntentType.OUT_OF_SCOPE


@pytest.mark.live
def test_live_coreference_reformulation(router, focused_state):
    decision = router.route("who directed it?", focused_state)
    assert decision.standalone_query.lower().count("nolan") > 0 or "direct" in decision.standalone_query.lower()
    assert "it" != decision.standalone_query.lower().strip()


@pytest.mark.live
def test_live_persistent_exclusions_survive_new_query(router):
    state = ConversationState()
    state.session_preferences = UserSessionPreferences(excluded_genres=["Horror"])

    decision = router.route("recommend a mystery thriller", state)

    assert decision.filters is not None
    assert "Horror" in decision.filters.excluded_genres
