"""Unit tests for the weekly budget tracker (#8 completion)."""

from datetime import date

import pytest

from src.domain.budget import utc_today
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
        ref = reference or utc_today()
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

    old_date = (utc_today() - timedelta(days=utc_today().weekday() + 3)).isoformat()
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
    db.record_budget_entry(utc_today().isoformat(), 10.00, 1, "any")
    assert tracker.current_verdict() is GuardrailVerdict.BLOCKED
    assert tracker.record("meta-llama/llama-3.2-3b-instruct", 10, 10) is GuardrailVerdict.BLOCKED


# --- session cost limiter (#39) ---------------------------------------------

def test_session_cost_limiter_thresholds():
    """Under $0.08 clean, ≥ $0.08 warn, ≥ $0.10 block ("m" prices at the $1/MTok
    fallback; +1 tokens keep the sums clear of float-equality at the lines)."""
    from src.maya.guardrails import SessionCostLimiter

    limiter = SessionCostLimiter()
    assert limiter.record("m", 50_000, 0) is GuardrailVerdict.CLEAN       # $0.05
    assert limiter.record("m", 30_001, 0) is GuardrailVerdict.SUSPICIOUS  # $0.080001
    assert limiter.record("m", 20_001, 0) is GuardrailVerdict.BLOCKED     # $0.100002


def test_session_cost_accumulates_blended_estimate():
    """Per-session cost uses the SAME blended table as the weekly tracker."""
    from src.maya.guardrails import SessionCostLimiter

    limiter = SessionCostLimiter()
    limiter.record("meta-llama/llama-3.3-70b-instruct", 1_000_000, 0)      # $0.20
    assert limiter.record("meta-llama/llama-3.3-70b-instruct", 3_000_000, 0) is GuardrailVerdict.BLOCKED


# --- week window (#39) --------------------------------------------------------

def test_week_bounds_always_monday_to_sunday():
    from datetime import date, timedelta

    from src.domain.budget import week_bounds

    monday = date(2026, 9, 21)
    for offset in range(7):
        start, end = week_bounds(monday + timedelta(days=offset))
        assert start == monday and end == date(2026, 9, 27)
        assert start.weekday() == 0 and end.weekday() == 6


def test_week_boundary_sunday_in_monday_out(db):
    """The Mon 23:59 entry belongs to this week; Mon 00:00 starts fresh."""
    from datetime import timedelta

    from src.domain.budget import utc_today

    today = utc_today()
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    next_monday = sunday + timedelta(days=1)
    db.record_budget_entry(sunday.isoformat(), 4.00, 1, "sun")
    db.record_budget_entry(next_monday.isoformat(), 6.00, 1, "next-mon")
    assert db.weekly_spend_usd(reference=monday) == pytest.approx(4.00)
    assert db.weekly_spend_usd(reference=next_monday) == pytest.approx(6.00)


def test_daily_spend_usd_counts_only_that_day(db):
    from datetime import timedelta

    from src.domain.budget import utc_today

    today = utc_today()
    db.record_budget_entry(today.isoformat(), 1.00, 1, "a")
    db.record_budget_entry(today.isoformat(), 0.50, 1, "b")
    db.record_budget_entry((today - timedelta(days=1)).isoformat(), 9.00, 1, "yesterday")
    assert db.daily_spend_usd() == pytest.approx(1.50)


def test_format_week_caption_names_window_reset_and_today():
    from datetime import date

    from src.ui.sidebar_lab import format_week_caption

    caption = format_week_caption(1.23, 10.0, date(2026, 9, 21), 0.40)
    assert "Mon 21 Sep" in caption and "Sun 27 Sep" in caption
    assert "resets Mon 28 Sep" in caption
    assert "$1.23" in caption and "$0.40" in caption


def test_estimate_cost_glm_row_pinned():
    # #97: glm-5.3-flash @ $0.25/1M blended (z.ai list 0.15 in / 0.50 out)
    assert estimate_cost("glm-5.3-flash", 1_000_000, 0) == pytest.approx(0.25)
    assert estimate_cost("z-ai/glm-5.3-flash", 500_000, 500_000) == pytest.approx(0.25)


# --- #123: the graph meters every LLM call ------------------------------------


def test_graph_turn_meters_route_and_synthesis_into_limiter_and_ledger(tmp_path):
    """#123: one turn = one route call + one synthesis call. The limiter
    accumulates the SUM (cap == sum, so BLOCKED proves exact accumulation)
    and the weekly ledger gains one row per call."""
    from langchain_core.messages import HumanMessage

    from src.domain.config import ExperimentConfig
    from src.domain.routing import IntentType, QueryRoutingDecision
    from src.domain.usage import LLMUsage
    from src.graph.orchestrator import build_maya_graph
    from src.graph.state import SynthesisUsage
    from src.maya.guardrails import SessionCostLimiter
    from src.observability.tracer import DualModeObservabilityManager

    class _Router:
        def __init__(self):
            self.last_usage = LLMUsage(
                model="glm-5.3-flash", prompt_tokens=500, completion_tokens=200
            )

        def route(self, query, state, feedback=None):
            return QueryRoutingDecision(
                intent=IntentType.CAPABILITIES,
                confidence=0.9,
                standalone_query=query,
                requires_rag=False,
                reasoning="t",
            )

    class _Synth:
        def synthesize(self, query, decision, movies, history):
            return "reply", SynthesisUsage(
                model="fake-model", prompt_tokens=10, completion_tokens=5
            )

    class _NoEngine:
        def retrieve(self, query, routing, top_k=8, candidate_pool=50, shown_ids=None, boost=None):
            return []

    route_cost = estimate_cost("glm-5.3-flash", 500, 200)
    synth_cost = estimate_cost("fake-model", 10, 5)
    db = MovieDatabase(str(tmp_path / "budget.db"))
    limiter = SessionCostLimiter(cap=route_cost + synth_cost)
    graph = build_maya_graph(
        ExperimentConfig(),
        _Router(),
        _NoEngine(),
        _Synth(),
        DualModeObservabilityManager(session_id="unit-budget"),
        limiter=limiter,
        budget_tracker=WeeklyBudgetTracker(db),
    )
    out = graph.invoke({"messages": [HumanMessage(content="what can you do")]})

    assert out["session_cost_usd"] == pytest.approx(route_cost + synth_cost)
    assert out["session_tokens"] == 715
    # limiter holds the exact SUM: at cap -> BLOCKED (under-count -> CLEAN)
    assert limiter.check_current().verdict is GuardrailVerdict.BLOCKED
    # ledger: both calls' costs summed for the week
    assert db.weekly_spend_usd() == pytest.approx(route_cost + synth_cost)


def test_meter_llm_unusable_usage_costs_nothing_but_is_recorded(tmp_path):
    """#123: stubbed clients (no usage) must not distort the ledger — and
    after the review hardening (P2-1) they leave an explicit ``unmetered``
    marker row in the Trace instead of vanishing (AGENTS.md: fail-open
    must be explicit and recorded)."""
    from langchain_core.messages import HumanMessage

    from src.domain.config import ExperimentConfig
    from src.domain.routing import IntentType, QueryRoutingDecision
    from src.graph.orchestrator import build_maya_graph
    from src.graph.state import SynthesisUsage
    from src.observability.tracer import DualModeObservabilityManager

    class _Router:  # no last_usage attr at all (old-style fake)
        def route(self, query, state, feedback=None):
            return QueryRoutingDecision(
                intent=IntentType.CAPABILITIES,
                confidence=0.9,
                standalone_query=query,
                requires_rag=False,
                reasoning="t",
            )

    class _Synth:
        def synthesize(self, query, decision, movies, history):
            return "reply", SynthesisUsage(model="fake", prompt_tokens=0, completion_tokens=0)

    class _NoEngine:
        def retrieve(self, query, routing, top_k=8, candidate_pool=50, shown_ids=None, boost=None):
            return []

    db = MovieDatabase(str(tmp_path / "budget2.db"))
    graph = build_maya_graph(
        ExperimentConfig(),
        _Router(),
        _NoEngine(),
        _Synth(),
        tracer := DualModeObservabilityManager(session_id="unit-budget2"),
        budget_tracker=WeeklyBudgetTracker(db),
    )
    out = graph.invoke({"messages": [HumanMessage(content="hi")]})
    assert out["session_cost_usd"] == 0.0
    assert db.weekly_spend_usd() == 0.0
    # P2-1: the skipped ROUTE call is VISIBLE — one unmetered marker row
    # (the synth stub reports zero-token usage, which meters normally).
    markers = [
        t
        for t in tracer.traces()
        if t["node"] == "cost" and t["payload"].get("unmetered") == "no_usage_metadata"
    ]
    assert {m["payload"]["node"] for m in markers} == {"route"}
