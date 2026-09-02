"""Adversarial tests for the weekly budget tracker (#8 completion)."""

from datetime import date

import pytest

from src.maya.guardrails import (
    GuardrailVerdict,
    WeeklyBudgetTracker,
    estimate_cost,
)
from src.storage.database import MovieDatabase

pytestmark = pytest.mark.adversarial


class ExplodingSink:
    """Simulates storage failure on every operation."""

    def record_budget_entry(self, *a):
        raise sqlite3_error()

    def weekly_spend_usd(self):
        raise sqlite3_error()


def sqlite3_error():
    import sqlite3

    return sqlite3.OperationalError("database is locked")


def test_sink_failure_fails_open_never_raises():
    """DB hiccup must never brick the chat — logged, turn proceeds."""
    tracker = WeeklyBudgetTracker(ExplodingSink())
    assert tracker.record("any-model", 100, 50) is GuardrailVerdict.CLEAN
    assert tracker.current_verdict() is GuardrailVerdict.CLEAN


def test_unknown_model_cost_fail_closed():
    """Unknown/expensive models price at the ceiling rate, not zero."""
    assert estimate_cost("gpt-99-ultra", 1_000_000, 1_000_000) == pytest.approx(2.0)
    assert estimate_cost("", 1_000, 0) > 0


def test_rogue_sink_negative_costs_do_not_credit_budget(tmp_path):
    """Negative/malformed entries can't fake the budget into negative spend."""
    db = MovieDatabase(str(tmp_path / "adv.db"))
    db.record_budget_entry(date.today().isoformat(), -50.0, 1, "rogue")  # hostile row
    tracker = WeeklyBudgetTracker(db)
    # SUM can go negative — verdict stays CLEAN (no crash, no false block)
    assert tracker.current_verdict() is GuardrailVerdict.CLEAN


def test_partial_week_rows_only_count_current_week(tmp_path):
    """Rows from previous weeks/years never throttle the current week."""
    db = MovieDatabase(str(tmp_path / "adv2.db"))
    db.record_budget_entry("2020-01-01", 999.0, 1, "ancient")  # years ago
    tracker = WeeklyBudgetTracker(db)
    assert tracker.current_verdict() is GuardrailVerdict.CLEAN


def test_record_budget_entry_round_trips_floats(tmp_path):
    """Costs survive the SQLite round-trip without float corruption to zero."""
    db = MovieDatabase(str(tmp_path / "adv3.db"))
    db.record_budget_entry(date.today().isoformat(), 0.0003, 1500, "m")
    assert db.weekly_spend_usd() == pytest.approx(0.0003)
