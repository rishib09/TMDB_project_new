"""Offline end-to-end test of the conversation-mode driver (#93).

Real compiled graph (with checkpointer) + scripted router/engine/synthesizer
— no API calls. Pins: thread per conversation, the Q12–Q14 scoring through
the real turn pipeline, the no-repeat measurement (the #80 detection), the
result envelope, and the #59 filename/delta contract for mode="conversation".
"""


import pytest

from src.domain.config import ExperimentConfig
from src.domain.movie import MovieRecord
from src.domain.routing import IntentType, MetadataFilterCriteria, QueryRoutingDecision
from src.evals.conversations import (
    ConversationSet,
    ConversationTurn,
    ExpectedConstraints,
    GoldenConversation,
    TurnExpectation,
)
from src.evals.runner import BenchmarkRunner
from src.graph.orchestrator import build_maya_graph
from src.graph.state import SynthesisUsage
from src.observability.tracer import DualModeObservabilityManager

pytestmark = pytest.mark.unit


class ScriptedRouter:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def route(self, query, state, feedback=None):
        return self.decisions.pop(0)


class PoolEngine:
    last_dense_failure = None

    def __init__(self, ids):
        self.ids = ids

    def retrieve(self, query, routing, top_k=8, candidate_pool=50, shown_ids=None):
        return [
            type("R", (), {"movie": MovieRecord(id=i, title=f"M{i}", release_year=2001),
                           "score": 1.0, "source": "sql"})()
            for i in self.ids[:top_k]
        ]


class StubSynthesizer:
    def synthesize(self, query, decision, movies, history):
        return f"{len(movies)} found.", SynthesisUsage(model="stub", prompt_tokens=1, completion_tokens=1)


def _golden() -> ConversationSet:
    def turn(n, user, intent, path, constraints=None, **expect):
        return ConversationTurn(n=n, user=user, expect=TurnExpectation(
            intent=intent, path=path,
            constraints=constraints or ExpectedConstraints(), **expect,
        ))
    return ConversationSet(
        version="test-v1",
        description="offline driver fixture",
        conversations=[
            GoldenConversation(
                id="CT1", tier="C_records", title="Driver fixture", source="authored",
                turns=[
                    turn(1, "recent movies", IntentType.SEMANTIC_SEARCH, "retrieve",
                         ExpectedConstraints(year_min=2015)),
                    turn(2, "older movies", IntentType.SEMANTIC_SEARCH, "retrieve",
                         ExpectedConstraints(year_min=1990), no_repeat=True),
                    turn(3, "tell me a joke", IntentType.OUT_OF_SCOPE, "pivot"),
                    turn(4, "even older", IntentType.SEMANTIC_SEARCH, "retrieve",
                         ExpectedConstraints(year_min=1980)),
                ],
            ),
        ],
    )


def _runner(tmp_path=None) -> BenchmarkRunner:
    router = ScriptedRouter([
        QueryRoutingDecision(intent=IntentType.SEMANTIC_SEARCH, confidence=0.95,
                             standalone_query="recent", requires_rag=True,
                             filters=MetadataFilterCriteria(year_min=2015)),
        QueryRoutingDecision(intent=IntentType.SEMANTIC_SEARCH, confidence=0.95,
                             standalone_query="older", requires_rag=True,
                             filters=MetadataFilterCriteria(year_min=1990)),
        QueryRoutingDecision(intent=IntentType.OUT_OF_SCOPE, confidence=0.99,
                             standalone_query="a joke", requires_rag=False),
        QueryRoutingDecision(intent=IntentType.SEMANTIC_SEARCH, confidence=0.95,
                             standalone_query="even older", requires_rag=True,
                             filters=MetadataFilterCriteria(year_min=1980)),
    ])
    from langgraph.checkpoint.memory import InMemorySaver
    tracer = DualModeObservabilityManager(session_id="driver-test")  # ONE tracer: graph + runner
    graph = build_maya_graph(
        ExperimentConfig(), router, PoolEngine([1, 2]), StubSynthesizer(),
        tracer, checkpointer=InMemorySaver(),
    )
    return BenchmarkRunner(ExperimentConfig(), engine=None, graph=graph, tracer=tracer)


def test_driver_scores_retrieve_ask_and_pivot(tmp_path):
    summary = _runner(tmp_path).run_conversations(_golden(), "conv-test")

    assert summary.mode == "conversation"
    assert summary.routing_stack == "v1"
    assert summary.n_conversations == 1 and summary.n_turns == 4
    assert summary.dataset_version == "test-v1"
    assert summary.intent_accuracy == 1.0  # every turn produced a reading
    assert summary.path_accuracy == 1.0    # retrieve, retrieve, pivot
    assert summary.constraint_fidelity == 1.0  # year filters reached the engine

    t1, t2, t3, t4 = summary.per_conversation[0].per_turn
    assert t1.observed_path == "retrieve" and t1.effective["year_min"] == 2015
    assert t2.observed_path == "retrieve" and t2.effective["year_min"] == 1990
    assert t3.observed_path == "pivot"
    assert t4.observed_path == "retrieve" and t4.effective["year_min"] == 1980

    # the #80 measurement: same ids returned twice with no_repeat expected
    assert t2.no_repeat is False
    assert t2.no_repeat_violation_ids == [1, 2]
    assert summary.no_repeat_rate == 0.0


def test_driver_marks_failed_turns_with_trace_slice(tmp_path):
    summary = _runner(tmp_path).run_conversations(_golden(), "conv-test")
    t2 = summary.per_conversation[0].per_turn[1]
    assert t2.constraint_detail.get("trace_slice")  # failing turn carries its slice
    assert summary.per_conversation[0].failed is True


def test_save_conversations_writes_identity_and_delta(tmp_path):
    import json

    first_runner = _runner()  # fresh scripted router per run
    first = first_runner.run_conversations(_golden(), "conv-test")
    path1 = first_runner.save_conversations(first, out_dir=tmp_path)
    assert first.mode in path1.read_text(encoding="utf-8")

    second_runner = _runner()
    second = second_runner.run_conversations(_golden(), "conv-test")
    path2 = second_runner.save_conversations(second, out_dir=tmp_path)
    payload = json.loads(path2.read_text(encoding="utf-8"))
    assert payload["delta"], "second run of the same hash+mode must carry a delta"
    assert path1.name.split("_")[1] == path2.name.split("_")[1]  # same config hash


def test_turn_error_is_isolated_and_run_continues():
    """Q19 (spec review P1): an API error on one turn must not kill the run —
    the fixed script continues; only budget-blocked and dense-loss abort."""
    class ExplodingSynthesizer(StubSynthesizer):
        calls = 0

        def synthesize(self, query, decision, movies, history):
            ExplodingSynthesizer.calls += 1
            if ExplodingSynthesizer.calls == 2:  # turn 2's synthesis dies
                raise RuntimeError("synthesis API error")
            return super().synthesize(query, decision, movies, history)

    from langgraph.checkpoint.memory import InMemorySaver
    router = ScriptedRouter([
        QueryRoutingDecision(intent=IntentType.SEMANTIC_SEARCH, confidence=0.95,
                             standalone_query="recent", requires_rag=True,
                             filters=MetadataFilterCriteria(year_min=2015)),
        QueryRoutingDecision(intent=IntentType.SEMANTIC_SEARCH, confidence=0.95,
                             standalone_query="older", requires_rag=True,
                             filters=MetadataFilterCriteria(year_min=1990)),
        QueryRoutingDecision(intent=IntentType.OUT_OF_SCOPE, confidence=0.99,
                             standalone_query="a joke", requires_rag=False),
        QueryRoutingDecision(intent=IntentType.SEMANTIC_SEARCH, confidence=0.95,
                             standalone_query="even older", requires_rag=True,
                             filters=MetadataFilterCriteria(year_min=1980)),
    ])
    tracer = DualModeObservabilityManager(session_id="err-test")
    graph = build_maya_graph(
        ExperimentConfig(), router, PoolEngine([1, 2]), ExplodingSynthesizer(),
        tracer, checkpointer=InMemorySaver(),
    )
    runner = BenchmarkRunner(ExperimentConfig(), engine=None, graph=graph, tracer=tracer)
    summary = runner.run_conversations(_golden(), "err-test")

    turns = summary.per_conversation[0].per_turn
    assert len(turns) == 4                      # all scripted turns ran
    assert turns[1].observed_path == "error"    # the failed turn is recorded
    assert turns[1].path_correct is False
    assert "synthesis API error" in turns[1].constraint_detail["error"]
    assert turns[3].observed_path == "retrieve"  # and the run continued


def test_fallback_decision_scores_intent_incorrect():
    """Q19: a fallback's coincidentally-correct label is still a miss.

    The fallback trips the bounded re-route cycle (#12), so the router is
    queried again for the same turn — the scripted fallback repeats until
    the attempt budget runs out and the turn ends on the fallback decision.
    """
    older = QueryRoutingDecision(intent=IntentType.SEMANTIC_SEARCH, confidence=0.95,
                                 standalone_query="older", requires_rag=True,
                                 filters=MetadataFilterCriteria(year_min=1990))

    class FallbackRouter(ScriptedRouter):
        def route(self, query, state, feedback=None):
            if query == "older movies":  # always a fallback — re-routes included
                return older.model_copy(update={
                    "is_fallback": True, "confidence": 0.1,
                    "fallback_reason": "api_error",
                })
            return super().route(query, state, feedback)

    from langgraph.checkpoint.memory import InMemorySaver
    router = FallbackRouter([
        QueryRoutingDecision(intent=IntentType.SEMANTIC_SEARCH, confidence=0.95,
                             standalone_query="recent", requires_rag=True,
                             filters=MetadataFilterCriteria(year_min=2015)),
        QueryRoutingDecision(intent=IntentType.OUT_OF_SCOPE, confidence=0.99,
                             standalone_query="a joke", requires_rag=False),
        QueryRoutingDecision(intent=IntentType.SEMANTIC_SEARCH, confidence=0.95,
                             standalone_query="even older", requires_rag=True,
                             filters=MetadataFilterCriteria(year_min=1980)),
    ])
    tracer = DualModeObservabilityManager(session_id="fb-test")
    graph = build_maya_graph(
        ExperimentConfig(), router, PoolEngine([1, 2]), StubSynthesizer(),
        tracer, checkpointer=InMemorySaver(),
    )
    runner = BenchmarkRunner(ExperimentConfig(), engine=None, graph=graph, tracer=tracer)
    summary = runner.run_conversations(_golden(), "fb-test")

    t2 = summary.per_conversation[0].per_turn[1]
    assert t2.observed_intent == "SEMANTIC_SEARCH"  # label matches…
    assert t2.is_fallback is True
    assert t2.intent_correct is False                # …but still a miss (Q19)


def test_message_window_is_a_config_tunable():
    """ADR 0004 (standards review P1): the window rides ExperimentConfig."""
    from src.domain.config import ExperimentConfig
    from src.domain.memory import MESSAGE_WINDOW

    cfg = ExperimentConfig()
    assert cfg.message_window == MESSAGE_WINDOW == 10
    cfg2 = ExperimentConfig(message_window=4)
    assert cfg2.message_window == 4


def test_judge_failure_is_fail_open_and_recorded():
    """Q19 (round-5 hardening): a malformed judge response must not kill the
    run — the exact crash the first baseline attempt hit (truncated judge
    JSON, the #74 class). Fail-open, recorded on the turn row."""
    class ExplodingJudge:
        def judge_faithfulness(self, query, response, movies):
            raise ValueError("Invalid JSON: EOF while parsing an object")

    # golden with a relevant-ids turn (t1) so the judge is invoked
    golden = _golden().model_copy(update={"conversations": [
        _golden().conversations[0].model_copy(update={"turns": [
            t.model_copy(update={"expect": t.expect.model_copy(
                deep=True, update={"relevant_movie_ids": [1]},
            )}) for t in _golden().conversations[0].turns
        ]}),
    ]})
    runner = _runner()
    runner.judge = ExplodingJudge()
    summary = runner.run_conversations(golden, "judge-fail-test")

    t1 = summary.per_conversation[0].per_turn[0]
    assert t1.hit_rate is not None      # IR metrics still computed
    assert "EOF while parsing" in (t1.constraint_detail.get("judge_error") or "")
    assert summary.judge_turns == 0     # no scores counted from the failure
    assert summary.n_turns == 4         # the run itself completed


def test_session_cap_can_be_disabled_for_the_harness():
    """#93: the conversation driver spans 23 scripted conversations on one
    graph — a per-user session cap would refuse every turn after the first
    few retrievals. cap=None disables it; the default stays $0.10 for the
    live app (#39)."""
    from src.maya.guardrails import GuardrailVerdict, SessionCostLimiter

    unlimited = SessionCostLimiter(cap=None)
    for _ in range(10):
        unlimited.record("m", 10_000, 0)  # 10 × $0.01 = the $0.10 cap itself
    assert unlimited.check_current().verdict is GuardrailVerdict.CLEAN

    default = SessionCostLimiter()
    for _ in range(11):
        default.record("m", 10_000, 0)  # $0.11 accumulated — 11th call clears the cap
    assert default.check_current().verdict is GuardrailVerdict.BLOCKED
