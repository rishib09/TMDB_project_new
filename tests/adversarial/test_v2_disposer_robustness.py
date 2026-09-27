"""Adversarial tests for the v2 disposer (#105).

The disposer is the trust boundary of the v2 contract (ADR 0005): the
model proposes, this code disposes. Each test attacks one way a hostile
or sloppy model response could poison memory, leak outside the dataset,
or stall the conversation.
"""

import pytest
from pydantic import ValidationError

from src.domain.config import ExperimentConfig
from src.domain.memory import UserSessionPreferences
from src.domain.routing import IntentType, MetadataFilterCriteria
from src.maya.v2 import (
    PreferenceDelta,
    Understanding,
    deterministic_ask,
    dispose,
    enforce_probe_budget,
    turn_decision,
)

CFG = ExperimentConfig()


def _u(**overrides) -> Understanding:
    defaults = dict(intent=IntentType.SEMANTIC_SEARCH, standalone_query="q")
    defaults.update(overrides)
    return Understanding(**defaults)


class TestPre1970Boundary:
    """C7: the dataset boundary is enforced in code, never trusted to the model."""

    def test_decade_straddling_boundary_is_in_scope(self):
        # 1970s starts AT 1970 — in scope; only strictly-before is out.
        out = dispose(_u(decade=1970), UserSessionPreferences(), CFG)
        assert out.understanding.intent is not IntentType.OUT_OF_SCOPE

    def test_year_max_below_boundary_forces_out_of_scope(self):
        out = dispose(_u(filters=MetadataFilterCriteria(year_max=1969)),
                      UserSessionPreferences(), CFG)
        assert out.understanding.intent is IntentType.OUT_OF_SCOPE

    def test_out_of_scope_turn_carries_no_filters_into_memory(self):
        """A pre-1970 request with a delta must not merge anything."""
        u = _u(
            decade=1939,
            preference_delta=PreferenceDelta(add_genres=["Horror"]),
        )
        out = dispose(u, UserSessionPreferences(), CFG)
        assert out.preferences.preferred_genres == []

    def test_boundary_ignores_later_user_statement(self):
        """pre-1970 pivot then a fresh in-scope turn: memory still clean."""
        poisoned = dispose(_u(decade=1950), UserSessionPreferences(), CFG)
        out = dispose(_u(era="recent"), poisoned.preferences, CFG)
        assert out.preferences.year_min == CFG.era_recent_year_min


class TestMemoryIntegrity:
    """C2: the delta can never corrupt or escalate beyond what code allows."""

    def test_delta_cannot_override_reset(self):
        u = _u(
            reset_context=True,
            preference_delta=PreferenceDelta(set_mood="scary", add_genres=["Horror"]),
        )
        out = dispose(u, UserSessionPreferences(preferred_mood="funny"), CFG)
        # reset_requested beats the merge in the reducer — the delta is lost.
        assert out.preferences.preferred_mood == ""

    def test_year_conflict_newer_range_retires_old_exact_year(self):
        prefs = UserSessionPreferences(exact_year=1994)
        u = _u(era="recent")  # incoming range via era mapping
        out = dispose(u, prefs, CFG)
        assert out.preferences.exact_year is None
        assert out.preferences.year_min == CFG.era_recent_year_min

    def test_no_delta_turn_preserves_everything(self):
        prefs = UserSessionPreferences(
            preferred_mood="scary", audience="kids",
            preferred_genres=["Horror"], year_min=2015,
        )
        out = dispose(_u(), prefs, CFG)
        assert out.preferences == prefs

    def test_model_out_of_scope_writes_no_memory(self):
        """The out-of-scope invariant is ONE rule regardless of who ruled
        the turn out: model-emitted OUT_OF_SCOPE ('tell me a joke, and no
        horror') leaves memory exactly as it found it. Red while only the
        code-forced pre-1970 path short-circuited."""
        prefs = UserSessionPreferences(preferred_mood="funny")
        u = _u(
            intent=IntentType.OUT_OF_SCOPE,
            preference_delta=PreferenceDelta(add_excluded_genres=["Horror"]),
        )
        out = dispose(u, prefs, CFG)
        assert out.preferences == prefs
        assert out.understanding.intent is IntentType.OUT_OF_SCOPE

    def test_era_supersedes_decade_with_a_note(self):
        """Era label and decade both set, no explicit years: era wins through
        the knob AND the disposition note says why (telemetry rule)."""
        u = _u(era="recent", decade=1980)
        out = dispose(u, UserSessionPreferences(), CFG)
        assert out.preferences.year_min == CFG.era_recent_year_min
        assert out.preferences.year_max is None
        assert any("supersedes" in n for n in out.notes)


class TestDecisionLadder:
    """C8: the ask/retrieve ladder must never dead-end or over-ask."""

    def test_probe_cap_retrieves_even_with_zero_axes(self):
        """C8: 'ask, at most MAX_PROBE_TURNS per session, then retrieve with
        what exists.' At the cap the ladder retrieves REGARDLESS of axes.
        Red on the pre-fix bug: the ready-branch required >= 1 axis, so a
        zero-axes visitor asked forever."""
        u, _ = enforce_probe_budget(_u(), probe_count=2)
        assert u.ready_to_retrieve is True
        out = turn_decision(u, UserSessionPreferences(), CFG, probe_count=2)
        assert out.decision == "retrieve"
        assert "budget" in out.why

    def test_cap_binds_every_retrieval_intent(self):
        """C8 has no intent scoping: a SUPERLATIVE_RANKING ask-turn hits the
        same cap as a semantic one. Red while the cap checked intents."""
        u, _ = enforce_probe_budget(
            _u(intent=IntentType.SUPERLATIVE_RANKING), probe_count=2
        )
        out = turn_decision(u, UserSessionPreferences(), CFG, probe_count=2)
        assert out.decision == "retrieve"

    def test_below_cap_zero_axes_still_asks(self):
        """The cap must not swallow the normal ask path: below it, a
        ready-but-axisless turn still asks (C8's '>= 1 axis' clause)."""
        u, _ = enforce_probe_budget(_u(), probe_count=1)
        out = turn_decision(u, UserSessionPreferences(), CFG, probe_count=1)
        assert out.decision == "ask"

    def test_probe_cap_with_one_axis_retrieves(self):
        u, _ = enforce_probe_budget(_u(), probe_count=2)
        out = turn_decision(u, UserSessionPreferences(preferred_mood="funny"), CFG, probe_count=2)
        assert out.decision == "retrieve"

    def test_negation_exclusion_with_only_excluded_genre_retrieves(self):
        """C1: NEGATION_EXCLUSION labels exclusion-only turns; the excluded
        genre is an explicit filter -> retrieve, not ask."""
        u = _u(
            intent=IntentType.NEGATION_EXCLUSION,
            filters=MetadataFilterCriteria(excluded_genres=["Horror"]),
        )
        assert turn_decision(u, UserSessionPreferences(), CFG).decision == "retrieve"

    def test_runtime_max_counts_as_explicit_filter(self):
        """C6: column-shaped caveats retrieve; text don'ts do not."""
        u = _u(filters=MetadataFilterCriteria(runtime_max=90))
        assert turn_decision(u, UserSessionPreferences(), CFG).decision == "retrieve"

    def test_text_donts_alone_do_not_retrieve(self):
        u = _u(
            preference_delta=PreferenceDelta(add_donts=["nothing boring"]),
        )
        out = dispose(u, UserSessionPreferences(), CFG)
        assert turn_decision(out.understanding, out.preferences, CFG).decision == "ask"


class TestSchemaHostility:
    """The model cannot smuggle values past the schema."""

    def test_negative_runtime_max_rejected_by_type_discipline(self):
        with pytest.raises(ValidationError):
            MetadataFilterCriteria(runtime_max="ninety")  # type: ignore[arg-value]

    def test_confidence_nan_is_rejected(self):
        with pytest.raises(ValidationError):
            _u(confidence=float("nan"))

    def test_deterministic_ask_is_always_retrieval_safe(self):
        """The C12 terminal shape must ask, never retrieve with a poisoned query."""
        u = deterministic_ask("'; DROP TABLE movies; --")
        assert u.ready_to_retrieve is False
        assert turn_decision(u, UserSessionPreferences(), CFG).decision == "ask"
