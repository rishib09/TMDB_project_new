"""Unit tests for pure domain models, LangGraph reducers, and dynamic token budgets."""

import pytest
from src.domain.movie import MovieRecord, CastMember
from src.domain.routing import (
    IntentType,
    QueryRoutingDecision,
    MetadataFilterCriteria,
    SuperlativeCriteria,
    SuperlativeMetric,
)
from src.domain.memory import (
    ConversationState,
    UserSessionPreferences,
    merge_preferences,
    merge_unique_ids,
)
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
        vote_count=35000,
        budget=160000000,
        revenue=839030630,
        poster_path="/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg",
        tagline="Your mind is the scene of the crime.",
        overview="Cobb, a skilled thief who steals corporate secrets through dream-sharing technology, is given the inverse task of planting an idea into the mind of a C.E.O.",
        genres=["Action", "Science Fiction", "Adventure"],
        cast=[
            CastMember(name="Leonardo DiCaprio", character="Dom Cobb", order=0),
            CastMember(name="Joseph Gordon-Levitt", character="Arthur", order=1),
            CastMember(name="Ken Watanabe", character="Saito", order=2),
            CastMember(name="Tom Hardy", character="Eames", order=3),
            CastMember(name="Cillian Murphy", character="Robert Fischer", order=4),
        ],
        keywords=["dream", "subconscious", "heist", "mind", "espionage", "architect", "totem"]
    )


@pytest.mark.unit
def test_movie_record_computed_urls(sample_movie: MovieRecord):
    """Verifies that TMDB poster URLs are properly constructed."""
    assert sample_movie.poster_url == "https://image.tmdb.org/t/p/w500/xlaY2zyzMfkhk0HSC5VUwzoZPU1.jpg"
    assert sample_movie.cast[0].profile_url is None  # No profile path provided


@pytest.mark.unit
def test_movie_record_dense_text_tier_packing(sample_movie: MovieRecord):
    """Verifies dynamic tier-packing across 256, 512, and 1024 token budgets."""
    # 256 tokens: Core + top 4 cast + top 6 keywords + trimmed synopsis
    text_256 = sample_movie.to_dense_text(token_budget=256)
    assert "Title: Inception (2010)" in text_256
    assert "Director: Christopher Nolan" in text_256
    assert "Leonardo DiCaprio, Joseph Gordon-Levitt" in text_256
    assert "Themes: dream, subconscious" in text_256
    assert "Synopsis:" in text_256

    # 512 tokens: Includes character roles ("as Dom Cobb"), runtime, and ratings
    text_512 = sample_movie.to_dense_text(token_budget=512)
    assert "Leonardo DiCaprio as Dom Cobb" in text_512
    assert "Runtime: 148 mins" in text_512
    assert "Rating: 8.4/10" in text_512

    # 1024 tokens: Includes financials (budget and box office)
    text_1024 = sample_movie.to_dense_text(token_budget=1024)
    assert "Financials: Budget $160,000,000 | Box Office $839,030,630" in text_1024


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
def test_langgraph_functional_reducers():
    """Verifies LangGraph functional reducers for unique IDs and session preferences."""
    # Test merge_unique_ids
    merged_ids = merge_unique_ids([101, 102], [102, 103, 104])
    assert merged_ids == [101, 102, 103, 104]

    # Test merge_preferences
    pref1 = UserSessionPreferences(excluded_genres=["Horror"], excluded_actors=["Tom Cruise"])
    pref2 = UserSessionPreferences(excluded_genres=["Comedy"], excluded_actors=["Tom Cruise", "Brad Pitt"])
    merged_pref = merge_preferences(pref1, pref2)
    assert merged_pref.excluded_genres == ["Horror", "Comedy"]
    assert merged_pref.excluded_actors == ["Tom Cruise", "Brad Pitt"]


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
def test_experiment_config_presets_and_budget_clamping():
    """Verifies that preset switching and token budget clamping work properly."""
    config = ExperimentConfig()
    assert config.hybrid_alpha == 0.5
    assert config.reranker_enabled is True

    # Switch to Fast Budget (MiniLM with 256 tokens)
    config.apply_preset(PresetType.FAST_BUDGET)
    assert config.router_model == "meta-llama/llama-3.2-3b-instruct"
    assert config.synthesis_model == "google/gemini-2.0-flash-lite"
    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.token_budget == 256

    # Test auto-clamping on MiniLM (cannot exceed 256 tokens)
    config_invalid = ExperimentConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        token_budget=1024
    )
    assert config_invalid.token_budget == 256  # Auto-clamped to model ceiling
