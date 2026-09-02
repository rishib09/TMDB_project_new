"""Unit tests for the weekly budget tracker (#8 completion)."""

from datetime import date

import pytest

from src.maya.guardrails import (
    GuardrailVerdict,
    WeeklyBudgetTracker,
    estimate_cost,
)
from src.storage.database import MovieDatabase

pytestmark = pytest.mark.unit


class FakeSink:
    """In-memory BudgetSink: rows keyed by date_str."""

    def __init__(self):
        self.rows: list[tuple[str, float, int, str]] = []

    def record_budget_entry(self, date_str, cost_usd, tokens_used, model_name):
        self.rows.append((date_str, cost_usd, tokens_used, model_name))

    def weekly_spend_usd(self, reference=None):
        ref = reference or date.today()
        week_start = ref.toordinal() - ref.weekday()
        return sum(
            cost for (d, cost, _, _) in self.rows
            if date.fromisoformat(d).toordinal() - date.fromisoformat(d).weekday() == week_start
        )


# --- cost estimation ------------------------------------------------------

def test_estimate_cost_known_model():
    # llama-3.3-70b @ $0.20/1M blended: 1500 tokens → $0.0003
    assert estimate_cost("meta-llama/llama-3.3-70b-instruct", 1000, 500) == pytest.approx(0.0003)


def test_estimate_cost_unknown_model_uses_priciest_fallback():
    # Fail-closed on cost: unknown model priced at $1.00/1M
    assert estimate_cost("totally-unknown-model", 1_000_000, 0) == pytest.approx(1.00)


def test_estimate_cost_negative_tokens_clamped():
    assert estimate_cost("meta-llama/llama-3.2-3b-instruct", -100, 500) == pytest.approx(
        500 / 1_000_000 * 0.06
    )


# --- verdict thresholds ---------------------------------------------------

def test_verdict_thresholds():
    tracker = WeeklyBudgetTracker(FakeSink())
    assert tracker.verdict_for(4.0) is GuardrailVerdict.CLEAN        # < 80%
    assert tracker.verdict_for(8.0) is GuardrailVerdict.SUSPICIOUS   # ≥ 80%
    assert tracker.verdict_for(9.99) is GuardrailVerdict.SUSPICIOUS
    assert tracker.verdict_for(10.0) is GuardrailVerdict.BLOCKED     # ≥ cap


# --- tracker + real SQLite sink -------------------------------------------

@pytest.fixture
def db(tmp_path):
    return MovieDatabase(str(tmp_path / "budget.db"))


def test_record_appends_row_and_accumulates(db):
    tracker = WeeklyBudgetTracker(db)
    assert tracker.record("meta-llama/llama-3.3-70b-instruct", 10_000, 5_000) is GuardrailVerdict.CLEAN
    assert tracker.record("meta-llama/llama-3.3-70b-instruct", 10_000, 5_000) is GuardrailVerdict.CLEAN
    # 30k tokens @ $0.20/1M = $0.006 accumulated
    assert db.weekly_spend_usd() == pytest.approx(0.006)


def test_week_rollover_resets_spend(db):
    from datetime import timedelta

    old_date = (date.today() - timedelta(days=date.today().weekday() + 3)).isoformat()
    db.record_budget_entry(old_date, 9.99, 1, "old-model")  # last week
    tracker = WeeklyBudgetTracker(db)
    assert tracker.current_verdict() is GuardrailVerdict.CLEAN  # old spend ignored


def test_guard_node_blocks_at_weekly_cap():
    """Graph-level: budget exhausted → next turn refused before routing."""
    from langchain_core.messages import HumanMessage

    from src.graph.orchestrator import build_maya_graph
    from src.observability.tracer import DualModeObservabilityManager

    class CappedSink:
        def record_budget_entry(self, *a):
            pass

        def weekly_spend_usd(self):
            return 12.50  # over cap

    from src.domain.config import ExperimentConfig
    from tests.unit.test_orchestrator import FakeEngine, FakeRouter, FakeSynthesizer, _decision

    graph = build_maya_graph(
        ExperimentConfig(), FakeRouter([_decision()]), FakeEngine(movies=[]),
        FakeSynthesizer(), DualModeObservabilityManager(session_id="t"),
        budget_tracker=WeeklyBudgetTracker(CappedSink()),
    )
    from src.domain.routing import IntentType
    from tests.unit.test_orchestrator import _decision as _dec

    graph = build_maya_graph(
        ExperimentConfig(),
        FakeRouter([_dec(intent=IntentType.CAPABILITIES, requires_rag=False)]),
        FakeEngine(movies=[]), FakeSynthesizer(),
        DualModeObservabilityManager(session_id="t"),
        budget_tracker=WeeklyBudgetTracker(CappedSink()),
    )
    out = graph.invoke({"messages": [HumanMessage(content="a movie")]})
    assert "Weekly budget exhausted" in out["final_response"]


def test_record_and_verdict_at_cap_block_next_turn(db):
    tracker = WeeklyBudgetTracker(db)
    # simulate a week that's already at cap
    db.record_budget_entry(date.today().isoformat(), 10.00, 1, "any")
    assert tracker.current_verdict() is GuardrailVerdict.BLOCKED
    assert tracker.record("meta-llama/llama-3.2-3b-instruct", 10, 10) is GuardrailVerdict.BLOCKED
