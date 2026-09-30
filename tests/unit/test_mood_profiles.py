"""Unit tests for Mood Profiles (issue #137): pure domain logic per ADR 0006."""

import pytest
from pydantic import ValidationError

from src.domain.moods import (
    MOOD_BOOST_CALIBRATION,
    MOOD_PROFILES,
    MoodBoostSpec,
    MoodFloorCriteria,
    MoodProfile,
    expand_query_text,
    merge_profile_floors,
    mood_boost_spec,
    resolve_mood_profile,
)
from src.domain.routing import MetadataFilterCriteria

# --- registry integrity ---------------------------------------------------------


@pytest.mark.unit
def test_registry_contains_the_two_proof_profiles():
    assert set(MOOD_PROFILES) == {"epic", "hidden-gem"}
    assert MOOD_PROFILES["epic"].version == 1
    assert MOOD_PROFILES["hidden-gem"].version == 1


@pytest.mark.unit
def test_profile_id_is_stable():
    assert MOOD_PROFILES["epic"].id == "epic@1"


@pytest.mark.unit
def test_profile_rejects_empty_mood():
    with pytest.raises(ValidationError):
        MoodProfile(mood="   ")


# --- resolution -------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_is_casefold_tolerant():
    assert resolve_mood_profile("EPIC") is MOOD_PROFILES["epic"]
    assert resolve_mood_profile("  Hidden-Gem  ") is MOOD_PROFILES["hidden-gem"]


@pytest.mark.unit
def test_resolve_unknown_and_empty_mood_none():
    assert resolve_mood_profile("whimsical") is None
    assert resolve_mood_profile("") is None


@pytest.mark.unit
def test_resolve_custom_registry_isolation():
    """The registry argument keeps the function pure — callers can pass a
    harness variant without mutating the global registry."""
    custom = {"epic": MOOD_PROFILES["epic"].model_copy(update={"version": 2})}
    assert resolve_mood_profile("epic", registry=custom).version == 2
    assert MOOD_PROFILES["epic"].version == 1


# --- query expansion ---------------------------------------------------------------


@pytest.mark.unit
def test_expansion_is_deterministic_and_appends():
    q = "show me an epic movie for solo viewing"
    once = expand_query_text(q, MOOD_PROFILES["epic"])
    assert once.startswith(q)
    assert once == expand_query_text(q, MOOD_PROFILES["epic"])
    for phrase in MOOD_PROFILES["epic"].query_phrases:
        assert phrase in once


@pytest.mark.unit
def test_expansion_without_phrases_is_identity():
    bare = MoodProfile(mood="epic")
    assert expand_query_text("q", bare) == "q"


# --- floor merging -------------------------------------------------------------------


@pytest.mark.unit
def test_merge_into_none_creates_floor_filter():
    filters = merge_profile_floors(None, MOOD_PROFILES["epic"])
    assert filters.vote_count_min == 3000
    assert filters.rating_min is None


@pytest.mark.unit
def test_merge_never_loosens_explicit_user_floor():
    """Tightening only: the user's floor always wins over the profile's."""
    user = MetadataFilterCriteria(vote_count_min=9000)
    merged = merge_profile_floors(user, MOOD_PROFILES["epic"])  # profile says 3000
    assert merged.vote_count_min == 9000


@pytest.mark.unit
def test_merge_tightens_when_profile_stricter():
    user = MetadataFilterCriteria(vote_count_min=100)
    merged = merge_profile_floors(user, MOOD_PROFILES["hidden-gem"])  # profile says 1500
    assert merged.vote_count_min == 1500


@pytest.mark.unit
def test_merge_without_floors_returns_equivalent():
    bare = MoodProfile(mood="epic")  # default empty floors
    filters = MetadataFilterCriteria(year_min=1990)
    merged = merge_profile_floors(filters, bare)
    assert merged == filters


@pytest.mark.unit
def test_floor_criteria_reject_out_of_range():
    with pytest.raises(ValidationError):
        MoodFloorCriteria(min_vote_average=11.0)
    with pytest.raises(ValidationError):
        MoodFloorCriteria(min_vote_count=-1)


# --- boost spec ------------------------------------------------------------------------


@pytest.mark.unit
def test_boost_spec_is_typed_with_profile_id():
    spec = mood_boost_spec(MOOD_PROFILES["epic"])
    assert isinstance(spec, MoodBoostSpec)
    assert spec.profile_id == "epic@1"
    assert spec.genre_boosts["War"] == pytest.approx(0.7)
    assert spec.runtime_boost_min == 120
    assert spec.scale == 1.0
    assert spec.normalizer == pytest.approx(0.01)


@pytest.mark.unit
def test_hidden_gem_spec_carries_negative_weights():
    spec = mood_boost_spec(MOOD_PROFILES["hidden-gem"])
    assert spec.popularity_boost < 0
    assert spec.revenue_boost < 0


@pytest.mark.unit
def test_boost_spec_accepts_experiment_config_calibration():
    spec = mood_boost_spec(
        MOOD_PROFILES["epic"],
        scale=0.5,
        normalizer=0.02,
        runtime_weight=0.75,
        term_cap=1.5,
        popularity_log_divisor=8.0,
        revenue_log_divisor=20.0,
    )
    assert spec.scale == pytest.approx(0.5)
    assert spec.normalizer == pytest.approx(0.02)
    assert spec.runtime_weight == pytest.approx(0.75)
    assert spec.term_cap == pytest.approx(1.5)
    assert spec.popularity_log_divisor == pytest.approx(8.0)
    assert spec.revenue_log_divisor == pytest.approx(20.0)


@pytest.mark.unit
def test_boost_calibration_defaults_are_single_sourced():
    """Drift guard (ADR 0004 single source): the calibration literals live in
    MOOD_BOOST_CALIBRATION only — ExperimentConfig field defaults and
    MoodBoostSpec field defaults must DERIVE from it, never restate it."""
    from src.domain.config import ExperimentConfig

    cal = MOOD_BOOST_CALIBRATION
    config_defaults = {
        "normalizer": ExperimentConfig.model_fields["mood_boost_normalizer"].default,
        "runtime_weight": ExperimentConfig.model_fields["mood_boost_runtime_weight"].default,
        "term_cap": ExperimentConfig.model_fields["mood_boost_term_cap"].default,
        "popularity_log_divisor": ExperimentConfig.model_fields["mood_boost_popularity_log_divisor"].default,
        "revenue_log_divisor": ExperimentConfig.model_fields["mood_boost_revenue_log_divisor"].default,
    }
    assert config_defaults == cal
    spec = MoodBoostSpec(profile_id="drift-check")
    assert {
        "normalizer": spec.normalizer,
        "runtime_weight": spec.runtime_weight,
        "term_cap": spec.term_cap,
        "popularity_log_divisor": spec.popularity_log_divisor,
        "revenue_log_divisor": spec.revenue_log_divisor,
    } == cal
