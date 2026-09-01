"""Unit tests for IR metrics, the LLM judge, and the benchmark runner (issue #6).

Offline: hand-computed metric cases, mocked judge LLM, fake engine — the
runner's summary math and delta serialization verified without network.
"""

import json

import pytest

from src.domain.config import ExperimentConfig
from src.domain.movie import MovieRecord
from src.evals.judge import FaithfulnessVerdict, RelevancyVerdict, strip_formatting
from src.evals.metrics import (
    aggregate,
    context_precision_at_k,
    hit_rate_at_k,
    mrr_at_k,
)
from src.evals.runner import BenchmarkRunner, load_dataset

pytestmark = pytest.mark.unit


# --- Hit Rate@K (hand-computed) ---

def test_hit_rate_perfect_ranking():
    assert hit_rate_at_k([5, 3, 8], [3], 5) == 1.0


def test_hit_rate_relevant_beyond_k():
    assert hit_rate_at_k([1, 2, 3, 9], [9], 3) == 0.0
    assert hit_rate_at_k([1, 2, 3, 9], [9], 4) == 1.0


def test_hit_rate_no_relevant_or_empty():
    assert hit_rate_at_k([], [1], 5) == 0.0
    assert hit_rate_at_k([1], [], 5) == 0.0
    assert hit_rate_at_k([1], [2], 0) == 0.0


# --- MRR@K (hand-computed) ---

def test_mrr_positions():
    assert mrr_at_k([7, 3, 9], [3], 5) == pytest.approx(1 / 2)
    assert mrr_at_k([3, 7, 9], [3], 5) == pytest.approx(1.0)
    assert mrr_at_k([7, 8, 3], [3], 5) == pytest.approx(1 / 3)


def test_mrr_absent_relevant_is_zero():
    assert mrr_at_k([1, 2], [3], 5) == 0.0
    assert mrr_at_k([1, 3], [3], 2 - 1) == 0.0  # relevant at position 2, k=1


# --- Context Precision@K (hand-computed, AP-style) ---

def test_context_precision_prefers_early_hits():
    # hit at rank 1: AP = 1/1 / 1 = 1.0; hit at rank 2: (1/2)/1 = 0.5
    assert context_precision_at_k([5], [5], 3) == pytest.approx(1.0)
    assert context_precision_at_k([4, 5], [5], 3) == pytest.approx(0.5)


def test_context_precision_multiple_hits():
    # ranks 1 and 3: (1/1 + 2/3) / 2 = 0.8333
    assert context_precision_at_k([5, 9, 7], [5, 7], 3) == pytest.approx((1.0 + 2 / 3) / 2)


def test_context_precision_degenerate():
    assert context_precision_at_k([1, 2], [], 3) == 0.0
    assert context_precision_at_k([], [1], 3) == 0.0


def test_aggregate_mean_and_empty():
    assert aggregate([1.0, 0.0, 0.5]) == pytest.approx(0.5)
    assert aggregate([]) == 0.0


# --- Judge (mocked LLM) ---

class FakeChain:
    def __init__(self, verdict):
        self.verdict = verdict
        self.last_input = None

    def invoke(self, prompt):
        self.last_input = prompt
        return self.verdict


@pytest.fixture
def judge(monkeypatch):
    from src.evals import judge as judge_module

    j = judge_module.MayaJudge.__new__(judge_module.MayaJudge)
    j.config = ExperimentConfig()
    return j


def test_faithfulness_score_is_code_derived(judge, monkeypatch):
    """Even if the model self-reports wildly, score = entailed/total from counts."""
    verdict = FaithfulnessVerdict(total_claims=4, entailed_claims=3, unsupported_claims=[])
    judge._faithfulness_chain = FakeChain(verdict)
    out = judge.judge_faithfulness("q", "response", [_movie()])
    assert out.score == pytest.approx(3 / 4)


def test_faithfulness_claims_repaired_when_inconsistent(judge):
    """Model reports fewer totals than unsupported claims — code repairs."""
    verdict = FaithfulnessVerdict(total_claims=1, entailed_claims=1,
                                  unsupported_claims=["made-up sequel", "wrong year"])
    judge._faithfulness_chain = FakeChain(verdict)
    out = judge.judge_faithfulness("q", "response", [_movie()])
    assert out.total_claims == 2  # max(1, len(unsupported))
    assert out.entailed_claims == 1  # min(1, 2)
    assert out.score == pytest.approx(0.5)


def test_faithfulness_empty_context_still_judges(judge):
    verdict = FaithfulnessVerdict(total_claims=2, entailed_claims=0, unsupported_claims=["x"])
    judge._faithfulness_chain = FakeChain(verdict)
    out = judge.judge_faithfulness("q", "hallucinated", [])
    assert out.score == 0.0
    assert "(empty" in judge._faithfulness_chain.last_input


def test_relevancy_clamped(judge):
    judge._relevancy_chain = FakeChain(RelevancyVerdict(score=1.7, reason="r"))
    assert judge.judge_relevancy("q", "r").score == 1.0
    judge._relevancy_chain = FakeChain(RelevancyVerdict(score=-0.3, reason="r"))
    assert judge.judge_relevancy("q", "r").score == 0.0


def test_strip_formatting_removes_markdown():
    dirty = "**Inception (2010)** — great![Poster](https://x/y.jpg) #pick"
    assert strip_formatting(dirty) == "Inception (2010) — great pick"


# --- Runner (fake engine) ---

class FakeEngine:
    def __init__(self, ranked_by_query):
        self.ranked_by_query = ranked_by_query
        self.calls = []

    def retrieve(self, query, routing, top_k=8, candidate_pool=50):
        self.calls.append(query)
        return [
            type("R", (), {"movie": type("M", (), {"id": mid})()})()
            for mid in self.ranked_by_query.get(query, [])
        ]


def _movie(mid=27205):
    return MovieRecord(id=mid, title="Inception", release_year=2010)


def _dataset(tmp_path):
    rows = [
        dict(id="X1", tier="B_retrieval", query="dream heist", expected_intent="SEMANTIC_SEARCH",
             expected_path="rrf", relevant_movie_ids=[27205], superlative=None, filters=None, notes=""),
        dict(id="X2", tier="A_guardrails", query="best movie of 1962", expected_intent="OUT_OF_SCOPE",
             expected_path="pivot", relevant_movie_ids=[], superlative=None, filters=None, notes=""),
        dict(id="X3", tier="B_retrieval", query="time loop", expected_intent="SEMANTIC_SEARCH",
             expected_path="rrf", relevant_movie_ids=[999], superlative=None, filters=None, notes=""),
    ]
    path = tmp_path / "ds.json"
    path.write_text(json.dumps({"version": "test", "queries": rows}), encoding="utf-8")
    return path


def test_runner_retrieval_mode_skips_non_retrieval_rows(tmp_path):
    engine = FakeEngine({"dream heist": [27205, 1, 2], "time loop": [3, 4, 5]})
    runner = BenchmarkRunner(ExperimentConfig(), engine)  # type: ignore[arg-type]
    summary = runner.run_retrieval(load_dataset(_dataset(tmp_path)), "unit")

    assert engine.calls == ["dream heist", "time loop"]  # pivot row skipped
    assert summary.n_queries == 2
    assert summary.hit_rate == pytest.approx(0.5)
    assert summary.mrr == pytest.approx(1 / 2)
    assert summary.per_query[0].hit_rate == 1.0


def test_runner_save_with_delta_vs_prior_run(tmp_path):
    engine = FakeEngine({"dream heist": [27205], "time loop": [1, 2]})  # first run: miss
    runner = BenchmarkRunner(ExperimentConfig(), engine)  # type: ignore[arg-type]
    rows = load_dataset(_dataset(tmp_path))
    rows = [r for r in rows if r["expected_path"] == "rrf"]

    first = runner.run_retrieval(rows, "delta-test")
    path1 = runner.save(first, tmp_path)
    payload1 = json.loads(path1.read_text(encoding="utf-8"))
    assert payload1["delta"] is None  # first run: nothing to compare

    # second run: time loop now ranks 1st -> hit_rate 1.0 (delta +0.5)
    engine.ranked_by_query["time loop"] = [999]  # second run: hit at rank 1
    second = runner.run_retrieval(rows, "delta-test")
    path2 = runner.save(second, tmp_path)
    payload2 = json.loads(path2.read_text(encoding="utf-8"))
    assert payload2["delta"]["hit_rate"] == pytest.approx(0.5)
    assert payload2["delta"]["mrr"] == pytest.approx(0.5)


def test_runner_full_mode_requires_graph_and_judge(tmp_path):
    runner = BenchmarkRunner(ExperimentConfig(), FakeEngine({}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="graph and judge"):
        runner.run_full([], "x")
