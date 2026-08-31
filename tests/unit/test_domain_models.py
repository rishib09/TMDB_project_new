"""Unit tests for pure domain models and multi-turn conversation memory."""

import pytest
from src.domain.movie import MovieRecord, CastMember
from src.domain.routing import (
    IntentType,
    QueryRoutingDecision,
    MetadataFilterCriteria,
    SuperlativeCriteria,
    SuperlativeMetric,
)
from src.domain.memory import ConversationState
from src.domain.config import ExperimentConfig, PresetType


@pytest.fixture
def sample_movie() -> MovieRecord:
    return MovieRecord(
        id=27205,
        title="Inception",
        release_year=2010,
        director="Christopher Nolan",
        runtime=148,
        vote_average=8.37,
        revenue=839030630,
        poster_path="/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg",
        genres=["Action", "Science Fiction"],
        cast=[
            CastMember(name="Leonardo DiCaprio", character="Dom Cobb", order=0),
            CastMember(name="Joseph Gordon-Levitt", character="Arthur", order=1),
        ],
        keywords=["dream", "subconscious", "heist"]
    )


@pytest.mark.unit
def test_movie_record_computed_urls(sample_movie: MovieRecord):
    """Verifies that TMDB poster URLs are properly constructed."""
    assert sample_movie.poster_url == "https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg"
    assert sample_movie.cast[0].profile_url is None  # No profile path provided


@pytest.mark.unit
def test_movie_record_dense_text_strategies(sample_movie: MovieRecord):
    """Verifies baseline vs enriched text serialization strategies."""
    baseline = sample_movie.to_dense_text(strategy="baseline")
    assert "Inception (2010)" in baseline

    enriched = sample_movie.to_dense_text(strategy="enriched_metadata")
    assert "Director: Christopher Nolan" in enriched
    assert "Leonardo DiCaprio" in enriched
    assert "Genres: Action, Science Fiction" in enriched


@pytest.mark.unit
def test_query_routing_decision_validation():
    """Verifies Pydantic schema validation for QueryRoutingDecision."""
    decision = QueryRoutingDecision(
        intent=IntentType.SUPERLATIVE_RANKING,
        confidence=0.98,
        standalone_query="highest grossing 1970 movies",
        requires_rag=True,
        is_superlative=True,
        superlative=SuperlativeCriteria(
            metric=SuperlativeMetric.REVENUE,
            direction="DESC",
            year=1970,
            limit=5
        ),
        reasoning="User explicitly asked for top revenue films from 1970."
    )
    assert decision.intent == IntentType.SUPERLATIVE_RANKING
    assert decision.superlative.metric == SuperlativeMetric.REVENUE
    assert decision.superlative.year == 1970


@pytest.mark.unit
def test_conversation_state_multi_turn_transitions(sample_movie: MovieRecord):
    """Verifies 5-layer conversation state updates across turns."""
    state = ConversationState()
    assert len(state.messages) == 0
    assert state.focused_entity is None

    # Turn 1: User asks for sci-fi heist movie
    turn1_decision = QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH,
        confidence=0.95,
        standalone_query="sci-fi dream heist movie",
        requires_rag=True
    )
    state.add_turn(
        user_query="Recommend a great sci-fi dream heist movie",
        assistant_response="I highly recommend Inception (2010) directed by Christopher Nolan.",
        retrieved_movies=[sample_movie],
        decision=turn1_decision,
        tokens_used=450
    )

    assert len(state.messages) == 2
    assert state.focused_entity.id == 27205
    assert state.focused_entity.title == "Inception"
    assert 27205 in state.shown_movie_ids
    assert state.session_tokens == 450

    # Turn 2: User sets persistent exclusion ("No horror movies")
    turn2_decision = QueryRoutingDecision(
        intent=IntentType.NEGATION_EXCLUSION,
        confidence=0.99,
        standalone_query="no horror movies",
        requires_rag=False,
        filters=MetadataFilterCriteria(excluded_genres=["Horror"])
    )
    state.add_turn(
        user_query="Please don't recommend any horror films",
        assistant_response="Understood! I will exclude all horror movies from future searches.",
        retrieved_movies=[],
        decision=turn2_decision,
        tokens_used=200
    )

    assert "Horror" in state.session_preferences.excluded_genres
    assert state.session_tokens == 650


@pytest.mark.unit
def test_experiment_config_preset_switching():
    """Verifies that preset switching properly reconfigures all dynamic control plane knobs."""
    config = ExperimentConfig()
    assert config.hybrid_alpha == 0.5
    assert config.reranker_enabled is True

    # Switch to Fast Budget
    config.apply_preset(PresetType.FAST_BUDGET)
    assert config.router_model == "meta-llama/llama-3.2-3b-instruct"
    assert config.synthesis_model == "google/gemini-2.0-flash-lite"
    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.reranker_enabled is False
    assert config.retrieval_top_k == 3

    # Switch to Production Hybrid
    config.apply_preset(PresetType.PRODUCTION_HYBRID)
    assert config.synthesis_model == "meta-llama/llama-3.3-70b-instruct"
    assert config.embedding_model == "BAAI/bge-small-en-v1.5"
    assert config.hybrid_alpha == 0.5
    assert config.reranker_enabled is True
    assert config.retrieval_top_k == 5
