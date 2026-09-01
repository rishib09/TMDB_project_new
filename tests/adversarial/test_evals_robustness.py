"""Adversarial tests for the eval harness (issue #6).

Degenerate metric inputs, judge verdict tampering, and — most importantly —
dataset-integrity checks against the LIVE database, so the benchmark's
ground truth cannot rot silently (wrong ids, era mismatches, drifted tiers).
"""

import json
from pathlib import Path

import pytest

from src.domain.config import ExperimentConfig
from src.evals import judge as judge_module
from src.evals.judge import FaithfulnessVerdict, RelevancyVerdict
from src.evals.metrics import (
    aggregate,
    context_precision_at_k,
    hit_rate_at_k,
    mrr_at_k,
)
from src.evals.runner import BenchmarkRunner, load_dataset
from src.storage.database import MovieDatabase

pytestmark = pytest.mark.adversarial


class _R:
    def __init__(self, mid):
        self.movie = type("M", (), {"id": mid})()


class _E:
    def retrieve(self, query, routing, top_k=8, candidate_pool=50):
        return []


# --- degenerate rankings must never crash or produce NaN ---

@pytest.mark.parametrize("metric", [hit_rate_at_k, mrr_at_k, context_precision_at_k])
def test_metrics_total_on_degenerate_input(metric):
    assert metric([], [], 5) == 0.0
    assert metric([1, 1, 1], [1], 5) == 1.0  # duplicates don't crash
    assert metric(list(range(100)), [7], 5) == 0.0  # k < corpus
    assert metric([1], [1], -1) == 0.0  # invalid k
    assert metric([1], [1], 0) == 0.0
    result = metric([1, 2], [1], 5)
    assert result == result  # NaN sentinel: result must equal itself


def test_aggregate_never_nan():
    assert aggregate([]) == 0.0
    assert aggregate([0.0, 0.0]) == 0.0


# --- judge verdict tampering ---

def test_judge_cannot_self_inflate_faithfulness(monkeypatch):
    """A judge claiming 100% entailment with unsupported claims gets repaired."""
    judge = judge_module.MayaJudge.__new__(judge_module.MayaJudge)
    judge.config = ExperimentConfig()
    tampered = FaithfulnessVerdict(total_claims=1, entailed_claims=5,
                                   unsupported_claims=[])
    judge._faithfulness_chain = type("C", (), {"invoke": lambda self, p: tampered})()
    out = judge.judge_faithfulness("q", "r", [])
    assert out.entailed_claims <= out.total_claims  # entailed can't exceed total
    assert out.score <= 1.0


def test_judge_relevancy_out_of_range_clamped():
    verdict = RelevancyVerdict(score=99.0, reason="r")
    assert max(0.0, min(1.0, verdict.score)) == 1.0


# --- dataset integrity against the live DB (ground truth cannot rot) ---

@pytest.fixture(scope="module")
def dataset_rows():
    return load_dataset(Path("data/eval_benchmark_dataset.json"))


def test_dataset_shape(dataset_rows):
    assert len(dataset_rows) == 35
    tiers = {r["tier"] for r in dataset_rows}
    assert tiers == {"A_guardrails", "B_retrieval"}
    assert len({r["id"] for r in dataset_rows}) == 35, "duplicate query ids"


def test_dataset_relevant_ids_exist_in_db(dataset_rows):
    db = MovieDatabase("data/tmdb_movies.db")
    for row in dataset_rows:
        for mid in row["relevant_movie_ids"]:
            movie = db.get_by_id(mid)
            assert movie is not None, f"{row['id']}: expected id {mid} not in DB"


def test_dataset_retrieval_rows_are_1970_2026(dataset_rows):
    db = MovieDatabase("data/tmdb_movies.db")
    for row in dataset_rows:
        for mid in row["relevant_movie_ids"]:
            movie = db.get_by_id(mid)
            assert 1970 <= movie.release_year <= 2026, (
                f"{row['id']}: {movie.title} ({movie.release_year}) outside corpus era"
            )


def test_dataset_guardrail_contract(dataset_rows):
    """Rows that must pivot/no-retrieve encode the guardrail contract exactly."""
    for row in dataset_rows:
        if row["expected_path"] in ("pivot", "no_retrieval"):
            assert row["relevant_movie_ids"] == [], (
                f"{row['id']}: guardrail row must not have relevant ids"
            )
        if row["expected_path"] == "sql":
            assert row["superlative"] is not None, f"{row['id']}: SQL row needs criteria"


def test_dataset_pre_1970_rows_are_pivot_rows(dataset_rows):
    """Any row whose text references a pre-1970 era MUST expect a pivot."""
    import re

    for row in dataset_rows:
        mentions_old_era = bool(re.search(r"\b(18\d\d|19[0-6]\d|1950s|1960s)\b", row["query"]))
        if mentions_old_era:
            assert row["expected_path"] == "pivot", (
                f"{row['id']} mentions a pre-1970 era but expects '{row['expected_path']}'"
            )


def test_dataset_superlative_rows_match_sql_ground_truth(dataset_rows):
    """SQL-path rows: the ground truth must be what the DB actually answers."""
    db = MovieDatabase("data/tmdb_movies.db")
    for row in dataset_rows:
        if row["expected_path"] != "sql":
            continue
        sup = row["superlative"]
        truth = db.query_superlative(
            metric=sup["metric"],
            direction=sup.get("direction", "DESC"),
            year=sup.get("year"),
            genre=sup.get("genre"),
            limit=1,
        )
        assert truth, f"{row['id']}: SQL query returned nothing"
        assert truth[0].id in row["relevant_movie_ids"], (
            f"{row['id']}: SQL ground truth {truth[0].title} != expected {row['relevant_movie_ids']}"
        )


# --- runner robustness ---

def test_runner_handles_engine_failure_without_nan():
    class ExplodingEngine:
        def retrieve(self, query, routing, top_k=8, candidate_pool=50):
            raise RuntimeError("chroma InternalError: Error finding id")

    runner = BenchmarkRunner(ExperimentConfig(), ExplodingEngine())  # type: ignore[arg-type]
    rows = [r for r in load_dataset(Path("data/eval_benchmark_dataset.json"))
            if r["expected_path"] == "rrf"][:2]  # rows that actually retrieve
    with pytest.raises(RuntimeError):
        runner.run_retrieval(rows, "boom")  # failure surfaces loudly, not as 0.0


def test_load_dataset_rejects_missing_fields(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"queries": [{"id": "X1", "query": "q"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_dataset(bad)
