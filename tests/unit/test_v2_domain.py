"""Unit tests for the v2 domain: vocabularies, response models, disposer.

Contract of record: #82 (C1–C14). These tests run offline — no LLM, no
graph, no storage except the dataset pin in the vocabulary test.
"""

import pytest
from pydantic import ValidationError

from src.domain.config import ExperimentConfig
from src.domain.memory import UserSessionPreferences
from src.domain.routing import IntentType, MetadataFilterCriteria
from src.maya.v2 import (
    GENRES,
    PreferenceDelta,
    Understanding,
    deterministic_ask,
    dispose,
    enforce_probe_budget,
    enforce_question,
    known_axes,
    turn_decision,
)

CFG = ExperimentConfig()


def _u(**overrides) -> Understanding:
    """A minimal valid SEMANTIC_SEARCH understanding; overrides applied."""
    defaults = dict(intent=IntentType.SEMANTIC_SEARCH, standalone_query="some query")
    defaults.update(overrides)
    return Understanding(**defaults)


# --- vocabularies: the closed sets -------------------------------------------


def test_genre_vocabulary_matches_the_dataset():
    """C5: the frozen tuple must equal distinct_genres() — the two cannot
    drift. This is the pin that keeps the static set honest. Skipped (not
    failed) on machines without the ingested dataset, so 'offline green'
    stays honest."""
    from pathlib import Path

    from src.storage.database import MovieDatabase

    db_path = Path("data/tmdb_movies.db")
    if not db_path.exists():
        pytest.skip("dataset not ingested on this machine (data/tmdb_movies.db)")

    assert tuple(sorted(GENRES)) == tuple(sorted(MovieDatabase().distinct_genres()))


def test_mood_and_audience_sets_match_contract():
    from src.maya.v2 import AUDIENCES, MOOD_HINTS

    assert len(MOOD_HINTS) == 8
    assert set(AUDIENCES) == {"solo", "date night", "family", "kids", "adults"}


# --- models: schema discipline is the gate -----------------------------------


def test_mood_outside_closed_set_rejected():
    with pytest.raises(ValidationError):
        PreferenceDelta(set_mood="whimsical")  # type: ignore[arg-value]


def test_genre_outside_closed_set_rejected():
    with pytest.raises(ValidationError):
        PreferenceDelta(add_genres=["Space Western"])  # type: ignore[list-item]


def test_era_accepts_labels_only_not_raw_years():
    assert _u(era="recent").era == "recent"
    with pytest.raises(ValidationError):
        _u(era="2015")  # type: ignore[arg-value]


def test_confidence_is_telemetry_bounded():
    assert _u(confidence=0.0).confidence == 0.0
    with pytest.raises(ValidationError):
        _u(confidence=1.5)


def test_missing_slots_limited_to_narrowing_axes():
    with pytest.raises(ValidationError):
        _u(missing_slots=["director"])  # type: ignore[list-item]


# --- known_axes ---------------------------------------------------------------


def test_known_axes_reads_each_preference_slot():
    empty = UserSessionPreferences()
    assert known_axes(empty) == []
    assert known_axes(UserSessionPreferences(preferred_mood="scary")) == ["mood"]
    assert known_axes(UserSessionPreferences(audience="kids")) == ["audience"]
    assert known_axes(UserSessionPreferences(preferred_genres=["Horror"])) == ["genres"]
    assert known_axes(UserSessionPreferences(year_min=2015)) == ["era"]


# --- dispose: C7 era/decade/years ---------------------------------------------


def test_era_recent_maps_through_config_knob():
    out = dispose(_u(era="recent"), UserSessionPreferences(), CFG)
    assert out.preferences.year_min == CFG.era_recent_year_min
    assert "era 'recent'" in " ".join(out.notes)


def test_era_old_maps_through_config_knob():
    out = dispose(_u(era="old"), UserSessionPreferences(), CFG)
    assert out.preferences.year_max == CFG.era_old_year_max


def test_decade_becomes_ten_year_range():
    out = dispose(_u(decade=1980), UserSessionPreferences(), CFG)
    assert (out.preferences.year_min, out.preferences.year_max) == (1980, 1989)


def test_explicit_years_win_over_era_label():
    u = _u(era="old", filters=MetadataFilterCriteria(year_min=2015))
    out = dispose(u, UserSessionPreferences(), CFG)
    assert out.preferences.year_min == 2015
    assert out.preferences.year_max is None  # no era interference


def test_pre1970_decade_forces_out_of_scope_and_cleans_memory():
    out = dispose(_u(decade=1950, filters=MetadataFilterCriteria(genres=["Horror"])),
                  UserSessionPreferences(preferred_mood="scary"), CFG)
    assert out.understanding.intent is IntentType.OUT_OF_SCOPE
    assert out.understanding.filters is None
    assert out.preferences.preferred_mood == "scary"  # this turn poisons nothing


def test_pre1970_explicit_year_forces_out_of_scope():
    out = dispose(_u(filters=MetadataFilterCriteria(exact_year=1939)),
                  UserSessionPreferences(), CFG)
    assert out.understanding.intent is IntentType.OUT_OF_SCOPE


# --- dispose: C2 delta through the reducer -------------------------------------


def test_delta_adds_mood_and_genres():
    u = _u(preference_delta=PreferenceDelta(set_mood="funny", add_genres=["Comedy"]))
    out = dispose(u, UserSessionPreferences(), CFG)
    assert out.preferences.preferred_mood == "funny"
    assert "Comedy" in out.preferences.preferred_genres


def test_remove_genres_retires_only_named():
    prefs = UserSessionPreferences(preferred_genres=["Comedy", "Drama"])
    u = _u(preference_delta=PreferenceDelta(remove_genres=["Comedy"]))
    out = dispose(u, prefs, CFG)
    assert out.preferences.preferred_genres == ["Drama"]


def test_clear_mood_wipes_after_merge():
    prefs = UserSessionPreferences(preferred_mood="scary", preferred_genres=["Horror"])
    u = _u(preference_delta=PreferenceDelta(clear_mood=True))
    out = dispose(u, prefs, CFG)
    assert out.preferences.preferred_mood == ""
    assert out.preferences.preferred_genres == ["Horror"]  # genres untouched


def test_revoke_exclusions_case_insensitive():
    prefs = UserSessionPreferences(excluded_genres=["Horror"], excluded_actors=["Tom Cruise"])
    u = _u(preference_delta=PreferenceDelta(revoke_exclusions=["horror", "tom cruise"]))
    out = dispose(u, prefs, CFG)
    assert out.preferences.excluded_genres == []
    assert out.preferences.excluded_actors == []


def test_reset_context_wipes_the_slate():
    prefs = UserSessionPreferences(preferred_mood="scary", year_min=2015)
    out = dispose(_u(reset_context=True), prefs, CFG)
    assert out.preferences.preferred_mood == ""
    assert out.preferences.year_min is None


def test_donts_are_text_carried_not_filtered():
    u = _u(preference_delta=PreferenceDelta(add_donts=["nothing too sad"]))
    out = dispose(u, UserSessionPreferences(), CFG)
    assert "nothing too sad" in out.preferences.noted_donts


# --- turn_decision: C8 order ----------------------------------------------------


def test_converse_on_greeting_and_capabilities():
    for intent in (IntentType.GREETING, IntentType.CAPABILITIES):
        assert turn_decision(_u(intent=intent), UserSessionPreferences(), CFG).decision == "converse"


def test_pivot_on_out_of_scope():
    u = _u(intent=IntentType.OUT_OF_SCOPE)
    assert turn_decision(u, UserSessionPreferences(), CFG).decision == "pivot"


def test_explicit_filters_retrieve_even_with_zero_axes():
    u = _u(filters=MetadataFilterCriteria(director="Nolan"))
    out = turn_decision(u, UserSessionPreferences(), CFG)
    assert out.decision == "retrieve" and out.why == "explicit filters present"


def test_ready_with_one_axis_retrieves():
    prefs = UserSessionPreferences(preferred_mood="funny")
    u = _u(ready_to_retrieve=True)
    assert turn_decision(u, prefs, CFG).decision == "retrieve"


def test_ready_without_any_axis_still_asks():
    u = _u(ready_to_retrieve=True)
    assert turn_decision(u, UserSessionPreferences(), CFG).decision == "ask"


def test_axes_at_knob_retrieve_without_ready():
    prefs = UserSessionPreferences(preferred_mood="funny", audience="solo")
    out = turn_decision(_u(), prefs, CFG)
    assert out.decision == "retrieve"
    assert "axes" in out.why


def test_below_axes_asks_with_missing_slots():
    out = turn_decision(_u(missing_slots=["mood"]), UserSessionPreferences(), CFG)
    assert out.decision == "ask"
    assert "mood" in out.why


# --- C9 / C12 ---------------------------------------------------------------------


def test_enforce_question_passes_a_good_question_through():
    u = _u(clarifying_question="Something funny for family night?")
    out, notes = enforce_question(u)
    assert out.clarifying_question == u.clarifying_question
    assert notes == []


def test_enforce_question_template_on_overlength():
    u = _u(clarifying_question="x" * 201)
    out, notes = enforce_question(u)
    assert out.clarifying_question == "Tell me a mood or a genre and I'll find something good."
    assert notes and "length check" in notes[0]


def test_enforce_probe_budget_forces_ready_at_cap():
    u = _u()  # not ready
    out, notes = enforce_probe_budget(u, probe_count=2)
    assert out.ready_to_retrieve is True
    assert notes and "exhausted" in notes[0]


def test_enforce_probe_budget_noop_below_cap():
    u = _u()
    out, notes = enforce_probe_budget(u, probe_count=1)
    assert out.ready_to_retrieve is False
    assert notes == []


def test_deterministic_ask_shape():
    u = deterministic_ask("weird input")
    assert u.intent is IntentType.SEMANTIC_SEARCH
    assert u.ready_to_retrieve is False
    assert u.clarifying_question  # the template ask
    assert u.confidence == 0.0
    assert u.preference_delta == PreferenceDelta()


def test_schema_retry_budget_is_a_named_datum():
    """C12: 'one retry with the validation error' must live at one address —
    the #106 understand() wrapper imports it, never invents a literal."""
    from src.maya.v2 import disposer

    assert disposer.MAX_SCHEMA_ATTEMPTS == 2


def test_default_missing_slots_is_a_named_datum():
    from src.maya.v2 import disposer

    assert disposer.DEFAULT_MISSING_SLOTS == ["mood", "genres"]
    u = deterministic_ask("q")
    assert u.missing_slots == disposer.DEFAULT_MISSING_SLOTS
