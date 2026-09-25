"""Offline end-to-end test of the conversation-mode driver (#93).

Real compiled graph (with checkpointer) + scripted router/engine/synthesizer
— no API calls. Pins: thread per conversation, the Q12–Q14 scoring through
the real turn pipeline, the no-repeat measurement (the #80 detection), the
result envelope, and the #59 filename/delta contract for mode="conversation".
"""

from pathlib import Path

import pytest

from src.domain.config import ExperimentConfig
from src.domain.movie import MovieRecord
from src.domain.routing import IntentType, MetadataFilterCriteria, QueryRoutingDecision
from src.evals.conversations import ConversationSet, ConversationTurn, ExpectedConstraints, GoldenConversation, TurnExpectation
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

    def retrieve(self, query, routing, top_k=8, candidate_pool=50):
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
