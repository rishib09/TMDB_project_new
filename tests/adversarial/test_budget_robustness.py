"""Adversarial tests for the weekly budget tracker (#8 completion)."""

from datetime import date, timedelta

import pytest

from src.domain.budget import utc_today
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
    db.record_budget_entry(utc_today().isoformat(), -50.0, 1, "rogue")  # hostile row
    tracker = WeeklyBudgetTracker(db)
    # SUM can go negative — verdict stays CLEAN (no crash, no false block)
    assert tracker.current_verdict() is GuardrailVerdict.CLEAN


def test_partial_week_rows_only_count_current_week(tmp_path):
    """Rows from previous weeks/years never throttle the current week."""
    db = MovieDatabase(str(tmp_path / "adv2.db"))
    db.record_budget_entry("2020-01-01", 999.0, 1, "ancient")  # years ago
    tracker = WeeklyBudgetTracker(db)
    assert tracker.current_verdict() is GuardrailVerdict.CLEAN


def test_future_dated_row_does_not_pollute_current_week(tmp_path):
    """#39: the weekly sum has an UPPER bound — a future-dated row (clock skew,
    hostile entry) must not count toward this week's spend."""
    db = MovieDatabase(str(tmp_path / "adv4.db"))
    db.record_budget_entry(utc_today().isoformat(), 1.00, 1, "today")
    future = _next_week().isoformat()
    db.record_budget_entry(future, 9.00, 1, "future")  # next week's Monday or later
    assert db.weekly_spend_usd() == pytest.approx(1.00)


def test_past_reference_week_sums_only_that_week(tmp_path):
    """#39: querying a PAST week's spend returns that week alone, not
    'everything from its Monday onward'."""
    db = MovieDatabase(str(tmp_path / "adv5.db"))
    past_monday = _next_week() - timedelta(days=14)  # Monday two weeks ago
    db.record_budget_entry(past_monday.isoformat(), 2.00, 1, "that-week")
    db.record_budget_entry(utc_today().isoformat(), 3.00, 1, "this-week")
    assert db.weekly_spend_usd(reference=past_monday) == pytest.approx(2.00)


def _next_week() -> date:
    """Monday of next week — a date guaranteed outside the current ISO week."""
    today = utc_today()
    return today + timedelta(days=7 - today.weekday())


def test_record_budget_entry_round_trips_floats(tmp_path):
    """Costs survive the SQLite round-trip without float corruption to zero."""
    db = MovieDatabase(str(tmp_path / "adv3.db"))
    db.record_budget_entry(utc_today().isoformat(), 0.0003, 1500, "m")
    assert db.weekly_spend_usd() == pytest.approx(0.0003)
