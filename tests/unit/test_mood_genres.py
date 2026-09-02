"""Unit tests for LLM-assisted extraction + mood→genre mapping (#24/#25)."""

import pytest

from src.domain.memory import UserSessionPreferences, merge_preferences
from src.domain.routing import MetadataFilterCriteria
from src.maya.probing import (
    MOOD_GENRE_MAP,
    build_genre_confirm_response,
    funnel_axes,
    match_genre_pick,
    next_funnel_step,
)

pytestmark = pytest.mark.unit


# --- mood → genre mapping data (#25) ---------------------------------------

def test_mood_genre_map_covers_all_vocab_moods():
    """Every vocab-produced mood with an obvious genre mapping is mapped."""
    from src.maya.probing import _MOOD_VOCAB

    for mood_value in set(_MOOD_VOCAB.values()):
        assert mood_value in MOOD_GENRE_MAP, f"{mood_value} unmapped"


def test_mood_genre_map_values_are_real_genres():
    """Candidates must look like TMDB genre names (capitalized, no dups)."""
    for mood, genres in MOOD_GENRE_MAP.items():
        assert genres == list(dict.fromkeys(genres)), mood
        for g in genres:
            assert g[0].isupper() and "  " not in g, (mood, g)


# --- confirm_genres stage (#25) ---------------------------------------------

def test_multi_candidate_mood_triggers_genre_confirmation():
    prefs = UserSessionPreferences(preferred_mood="edge-of-your-seat")
    outcome = next_funnel_step(prefs, 0)
    assert outcome.action == "confirm_genres"
    assert outcome.offered_genre_options == ["Thriller", "Sci-Fi", "Horror", "Drama"]
    assert "Which of those" in outcome.response


def test_single_candidate_mood_auto_accepts_without_a_turn():
    """'funny' IS comedy — asking would waste a turn."""
    outcome = next_funnel_step(UserSessionPreferences(preferred_mood="funny"), 0)
    assert outcome.action != "confirm_genres"
    assert "Comedy" in outcome.prefs_update.preferred_genres
    assert outcome.prefs_update.genre_confirmation_done is True


def test_narrowing_within_explicit_genre_framing():
    """Explicit genre (sci-fi) → candidates exclude it, framing narrows within."""
    prefs = UserSessionPreferences(preferred_mood="edge-of-your-seat", preferred_genres=["Sci-Fi"])
    outcome = next_funnel_step(prefs, 0)
    assert outcome.action == "confirm_genres"
    assert "Sci-Fi" not in outcome.offered_genre_options
    assert "Within" in outcome.response and "Sci-Fi" in outcome.response


def test_unmapped_mood_skips_confirmation_gracefully():
    outcome = next_funnel_step(UserSessionPreferences(preferred_mood="whimsical"), 0)
    assert outcome.action != "confirm_genres"  # flavor-only, loop never dead-ends


def test_confirmation_never_re_asked_once_done():
    prefs = UserSessionPreferences(
        preferred_mood="edge-of-your-seat", preferred_genres=["Thriller"],
        genre_confirmation_done=True,
    )
    outcome = next_funnel_step(prefs, 0)
    assert outcome.action != "confirm_genres"


# --- genre pick matching (#25) ----------------------------------------------

OPTIONS = ["Thriller", "Sci-Fi", "Horror", "Drama"]


def test_pick_matching_multi_and_all():
    assert sorted(match_genre_pick("sci-fi and thriller", OPTIONS)) == ["Sci-Fi", "Thriller"]
    assert sorted(match_genre_pick("all of them", OPTIONS)) == sorted(OPTIONS)
    assert match_genre_pick("horror", OPTIONS) == ["Horror"]


def test_pick_matching_negation_and_near_miss():
    assert match_genre_pick("no horror, maybe drama", OPTIONS) == ["Drama"]
    assert match_genre_pick("what's the plot of Arrival", OPTIONS) is None
    # spaced "sci fi" normalizes to the candidate name
    assert match_genre_pick("sci fi please", OPTIONS) == ["Sci-Fi"]


def test_genre_picks_continue_funnel_via_next_step():
    merged = merge_preferences(
        UserSessionPreferences(preferred_mood="edge-of-your-seat"),
        UserSessionPreferences(preferred_genres=["Sci-Fi", "Thriller"],
                               genre_confirmation_done=True),
    )
    outcome = next_funnel_step(merged, 1)
    assert outcome.action == "probe"  # audience next
    assert "Sci-Fi" in outcome.prefs_update.preferred_genres


# --- funnel axes: mood + its genre are ONE signal (#25) ---------------------

def test_mood_mapped_genre_does_not_double_count():
    """'something funny' = mood signal; auto Comedy must not jump to confirm."""
    merged = merge_preferences(
        UserSessionPreferences(),
        UserSessionPreferences(preferred_mood="funny", genre_confirmation_done=True),
    )
    # merged has Comedy in genres? No — auto-accept adds it:
    merged_with_genre = merge_preferences(merged, UserSessionPreferences(
        preferred_genres=["Comedy"], genre_confirmation_done=True,
    ))
    assert funnel_axes(merged_with_genre) == ["mood"]  # one signal, one axis


def test_explicit_pick_counts_as_axis_when_mood_absent():
    axes = funnel_axes(UserSessionPreferences(
        preferred_genres=["Sci-Fi"], genre_confirmation_done=True,
    ))
    assert "genres" in axes


# --- #24 schemas -------------------------------------------------------------

def test_metadata_filters_default_match_any_and_person_none():
    f = MetadataFilterCriteria()
    assert f.genre_match == "any"
    assert f.person is None


def test_merge_reopens_genre_confirmation_on_mood_change():
    current = UserSessionPreferences(preferred_mood="funny", genre_confirmation_done=True)
    incoming = UserSessionPreferences(preferred_mood="scary")
    merged = merge_preferences(current, incoming)
    assert merged.preferred_mood == "scary"
    assert merged.genre_confirmation_done is False  # new mood → ask again


def test_build_genre_confirm_response_lists_candidates():
    text = build_genre_confirm_response(
        UserSessionPreferences(preferred_mood="scary"), ["Horror", "Thriller"]
    )
    assert "scary" in text and "Horror" in text and "Thriller" in text
    assert "all of them" in text  # escape hatch offered
