"""Adversarial tests for the weekly budget tracker (#8 completion) and the
session cost meter (#123): every LLM call must move the meter."""

from datetime import date, timedelta

import pytest
from langchain_core.messages import HumanMessage

from src.domain.budget import utc_today
from src.maya.guardrails import (
    GuardrailVerdict,
    SessionCostLimiter,
    WeeklyBudgetTracker,
    estimate_cost,
)
from src.maya.v2 import MayaV2Router
from src.storage.database import MovieDatabase

pytestmark = pytest.mark.adversarial


# --- #123 fakes: minimal collaborators for graph-level metering tests ---------

def _fresh_tracer():
    from src.observability.tracer import DualModeObservabilityManager

    return DualModeObservabilityManager(session_id="adv-budget")


class _UsageSynth:
    """Synthesizer shaped like the real seam: (text, usage)."""

    def synthesize(self, query, decision, movies, history):
        from src.graph.state import SynthesisUsage

        return "reply", SynthesisUsage(
            model="fake-model", prompt_tokens=10, completion_tokens=5
        )


class _EmptyEngine:
    def retrieve(self, query, routing, top_k=8, candidate_pool=50, shown_ids=None):
        return []


class _UsageV1Router:
    """v1 router reporting its token usage like the real MayaRouter (#123)."""

    def __init__(self):
        from src.domain.usage import LLMUsage

        self.last_usage = LLMUsage(
            model="glm-5.3-flash", prompt_tokens=500, completion_tokens=200
        )

    def route(self, query, state, feedback=None):
        from src.domain.routing import IntentType, QueryRoutingDecision

        return QueryRoutingDecision(
            intent=IntentType.CAPABILITIES,
            confidence=0.9,
            standalone_query=query,
            requires_rag=False,
            reasoning="test",
        )


class _UsageV2Router(MayaV2Router):
    """Real subclass so the stack selector wires route_node_v2; understand()
    returns the post-#123 3-tuple with usage."""

    def __init__(self):
        from src.domain.config import ExperimentConfig

        # Real construction (api_key test-pinned, no network at build time).
        super().__init__(ExperimentConfig(), api_key="test-key")

    def understand(self, query, prefs, shown_titles, last_assistant, probe_count):
        from src.domain.routing import IntentType
        from src.domain.usage import LLMUsage
        from src.maya.v2 import Understanding

        return (
            Understanding(
                intent=IntentType.SEMANTIC_SEARCH,
                standalone_query=query,
                confidence=0.9,
                ready_to_retrieve=True,
            ),
            ["scripted"],
            LLMUsage(model="glm-5.3-flash", prompt_tokens=300, completion_tokens=100),
        )


def test_sub_cent_session_spend_renders_four_decimals():
    """#123: a glm turn costs ~$0.0003 — two decimals rendered $0.00 for the
    first ~30 turns, which the visitor read as a broken meter."""
    from src.ui.sidebar_lab import format_session_spend

    text = format_session_spend(0.0024, 0.10)
    assert "$0.0024" in text
    assert "$0.00 /" not in text


def test_dollar_scale_session_spend_keeps_two_decimals():
    from src.ui.sidebar_lab import format_session_spend

    assert "$0.05 / $0.10" in format_session_spend(0.05, 0.10)


def test_v1_route_node_writes_router_cost_to_session():
    """#123 adversarial: the v1 Router call was unmetered — a routed turn
    must carry session_cost_usd > 0 (no such key on current code), and the
    session limiter must reflect the route + synthesize SUM."""
    from src.domain.config import ExperimentConfig
    from src.graph.orchestrator import build_maya_graph

    expected = estimate_cost("glm-5.3-flash", 500, 200) + estimate_cost(
        "fake-model", 10, 5
    )
    limiter = SessionCostLimiter()
    graph = build_maya_graph(
        ExperimentConfig(),
        _UsageV1Router(),
        _EmptyEngine(),
        _UsageSynth(),
        _fresh_tracer(),
        limiter=limiter,
        budget_tracker=None,
    )
    out = graph.invoke({"messages": [HumanMessage(content="what can you do")]})
    assert out["session_cost_usd"] == pytest.approx(expected)
    assert out["session_cost_usd"] > 0
    assert out["session_tokens"] == 700 + 15


def test_v2_route_node_writes_understand_cost_to_session():
    """#123 adversarial: the v2 Understand call was unmetered — a v2 turn
    must carry session_cost_usd > 0 and a ``cost`` trace row."""
    from src.domain.config import ExperimentConfig
    from src.graph.orchestrator import build_maya_graph

    expected = estimate_cost("glm-5.3-flash", 300, 100)
    graph = build_maya_graph(
        ExperimentConfig(),
        _UsageV2Router(),
        _EmptyEngine(),
        _UsageSynth(),
        tracer := _fresh_tracer(),
        budget_tracker=None,
    )
    out = graph.invoke({"messages": [HumanMessage(content="feel-good comedies")]})
    assert out["session_cost_usd"] == pytest.approx(expected)
    assert out["session_cost_usd"] > 0
    assert "cost" in [t["node"] for t in tracer.traces()]


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
