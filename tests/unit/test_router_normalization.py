"""Tests for MayaRouter deterministic decision normalization (post-live-run fixes).

Each rule maps to a failure observed in the live run of 2026-08-31 (see issue #13):
the LLM cannot be trusted for requires_rag, temporal boundaries, or filter hygiene.
"""

import pytest

from src.domain.routing import (
    IntentType,
    MetadataFilterCriteria,
    QueryRoutingDecision,
    SuperlativeCriteria,
    SuperlativeMetric,
)
from src.maya.router import MayaRouter


@pytest.fixture
def router():
    from src.domain.config import ExperimentConfig

    return MayaRouter(ExperimentConfig(), api_key="test-key")


# --- Rule 1: requires_rag derived from intent, LLM value ignored -----------------


@pytest.mark.unit
def test_requires_rag_derived_from_intent_not_llm(router):
    """Live failure #1/#6: model set requires_rag=false on a clear search query."""
    decision = QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH,
        confidence=0.8,
        standalone_query="mystery thriller",
        requires_rag=False,  # LLM's wrong claim — must be corrected
    )

    normalized = router._normalize_decision(decision)

    assert normalized.requires_rag is True


@pytest.mark.unit
def test_requires_rag_false_for_non_retrieval_intents(router):
    for intent in (IntentType.GREETING, IntentType.CAPABILITIES, IntentType.OUT_OF_SCOPE):
        decision = QueryRoutingDecision(
            intent=intent,
            confidence=0.9,
            standalone_query="x",
            requires_rag=True,  # LLM's wrong claim — must be corrected
        )
        assert router._normalize_decision(decision).requires_rag is False


# --- Rule 2: pre-1970 references force OUT_OF_SCOPE -------------------------------


@pytest.mark.unit
def test_pre_1970_filters_forced_out_of_scope(router):
    """Live failure #5: model extracted 1950s filters instead of refusing."""
    decision = QueryRoutingDecision(
        intent=IntentType.ATTRIBUTE_FILTER,
        confidence=0.8,
        standalone_query="best movies 1950s",
        requires_rag=True,
        filters=MetadataFilterCriteria(year_min=1950, year_max=1959),
    )

    normalized = router._normalize_decision(decision)

    assert normalized.intent == IntentType.OUT_OF_SCOPE
    assert normalized.requires_rag is False
    assert normalized.filters is None
    assert normalized.superlative is None


@pytest.mark.unit
def test_pre_1970_superlative_forced_out_of_scope(router):
    decision = QueryRoutingDecision(
        intent=IntentType.SUPERLATIVE_RANKING,
        confidence=0.9,
        standalone_query="highest grossing film of 1965",
        requires_rag=True,
        superlative=SuperlativeCriteria(metric=SuperlativeMetric.REVENUE, year=1965),
    )

    normalized = router._normalize_decision(decision)

    assert normalized.intent == IntentType.OUT_OF_SCOPE
    assert normalized.superlative is None


@pytest.mark.unit
def test_in_scope_years_untouched(router):
    decision = QueryRoutingDecision(
        intent=IntentType.SUPERLATIVE_RANKING,
        confidence=0.9,
        standalone_query="highest grossing film of 1970",
        requires_rag=True,
        superlative=SuperlativeCriteria(metric=SuperlativeMetric.REVENUE, year=1970),
    )

    normalized = router._normalize_decision(decision)

    assert normalized.intent == IntentType.SUPERLATIVE_RANKING
    assert normalized.superlative.year == 1970


# --- Rule 3: spurious filters stripped on non-filter intents ----------------------


@pytest.mark.unit
def test_spurious_filters_stripped_from_semantic_search(router):
    """Live failure #1: model filled genres on a plot-based search query."""
    decision = QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH,
        confidence=0.8,
        standalone_query="movies about space exploration",
        requires_rag=True,
        filters=MetadataFilterCriteria(genres=["Science Fiction", "Adventure"]),
    )

    normalized = router._normalize_decision(decision)

    assert normalized.filters is None


@pytest.mark.unit
def test_superlative_filters_stripped(router):
    """Live failure #3: model filled filters alongside superlative criteria."""
    decision = QueryRoutingDecision(
        intent=IntentType.SUPERLATIVE_RANKING,
        confidence=1.0,
        standalone_query="highest-grossing film 1970",
        requires_rag=True,
        filters=MetadataFilterCriteria(exact_year=1970),
        superlative=SuperlativeCriteria(metric=SuperlativeMetric.REVENUE, year=1970),
    )

    normalized = router._normalize_decision(decision)

    assert normalized.filters is None
    assert normalized.superlative is not None


@pytest.mark.unit
def test_filters_kept_for_filter_intents(router):
    decision = QueryRoutingDecision(
        intent=IntentType.ATTRIBUTE_FILTER,
        confidence=0.9,
        standalone_query="1980s John Carpenter horror",
        requires_rag=True,
        filters=MetadataFilterCriteria(year_min=1980, director="John Carpenter"),
    )

    normalized = router._normalize_decision(decision)

    assert normalized.filters is not None
    assert normalized.filters.director == "John Carpenter"


# --- Rule 4: cast_member duplicating an excluded actor is dropped -----------------


@pytest.mark.unit
def test_cast_member_duplicating_excluded_actor_dropped(router):
    """Live failure #4: model set cast_member="Tom Cruise" AND excluded_actors=["Tom Cruise"]."""
    decision = QueryRoutingDecision(
        intent=IntentType.NEGATION_EXCLUSION,
        confidence=0.9,
        standalone_query="action movies without Tom Cruise",
        requires_rag=True,
        filters=MetadataFilterCriteria(
            cast_member="Tom Cruise", excluded_actors=["Tom Cruise"]
        ),
    )

    normalized = router._normalize_decision(decision)

    assert normalized.filters.cast_member is None
    assert normalized.filters.excluded_actors == ["Tom Cruise"]


@pytest.mark.unit
def test_legitimate_cast_member_kept(router):
    decision = QueryRoutingDecision(
        intent=IntentType.ATTRIBUTE_FILTER,
        confidence=0.9,
        standalone_query="Kurt Russell in 1980s horror",
        requires_rag=True,
        filters=MetadataFilterCriteria(cast_member="Kurt Russell"),
    )

    normalized = router._normalize_decision(decision)

    assert normalized.filters.cast_member == "Kurt Russell"


# --- Integration with route(): normalization happens before exclusions merge ------


@pytest.mark.unit
def test_route_normalizes_then_merges_exclusions(router, monkeypatch):
    from src.domain.memory import ConversationState, UserSessionPreferences

    state = ConversationState()
    state.session_preferences = UserSessionPreferences(excluded_genres=["Horror"])

    decision = QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH,
        confidence=0.8,
        standalone_query="recommend a mystery thriller",
        requires_rag=False,  # live failure #6 — must not block exclusion merge
    )

    from unittest.mock import MagicMock

    chain = MagicMock()
    chain.invoke.return_value = decision
    monkeypatch.setattr(router, "_chain", chain)

    result = router.route("recommend a mystery thriller", state)

    assert result.requires_rag is True  # normalized from intent
    assert result.filters is not None
    assert result.filters.excluded_genres == ["Horror"]  # merge no longer skipped
