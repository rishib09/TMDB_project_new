"""Live benchmark evaluation (issue #6): real retrieval, real judge, real deltas.

OPT-IN:
    npx @dotenvx/dotenvx run -- pytest tests/integration/test_eval_live.py -v -m live

Runs a 5-query slice in retrieval mode (offline-quality IR metrics) plus a
2-query full-mode run through the real graph with the real LLM judge, then
exercises the versioned delta serialization for real.
"""

import json
import os

import pytest

from src.domain.config import ExperimentConfig
from src.evals.judge import MayaJudge
from src.evals.runner import BenchmarkRunner, load_dataset
from src.indexing.vector_store import MovieVectorStore
from src.retrieval.hybrid_engine import HybridRetrievalEngine
from src.storage.database import MovieDatabase

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set — skipping live eval tests",
    ),
]


@pytest.fixture(scope="module")
def config():
    return ExperimentConfig()


@pytest.fixture(scope="module")
def runner(config):
    engine = HybridRetrievalEngine(
        db=MovieDatabase("data/tmdb_movies.db"),
        vector_store=MovieVectorStore("data/chroma_db"),
        rag_version="v1_1_enriched",
    )
    return BenchmarkRunner(config, engine, judge=MayaJudge(config))


def test_retrieval_mode_slice_metrics_plausible(runner, tmp_path):
    rows = [r for r in load_dataset() if r["tier"] == "B_retrieval"][:5]
    summary = runner.run_retrieval(rows, "live-slice")
    assert summary.n_queries == 5
    assert 0.0 <= summary.hit_rate <= 1.0
    assert 0.0 <= summary.mrr <= 1.0
    # the 14-query golden set measured 86% hit@5; a 5-query slice should
    # land at least above coin-flip on the strongest tier
    assert summary.hit_rate >= 0.4, (
        f"hit@5 {summary.hit_rate:.2f} below sanity floor for v1_1 slice"
    )
    path = runner.save(summary, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["mode"] == "retrieval"
    assert payload["config_snapshot"]["rag_version"]


def test_full_mode_judges_real_responses(runner, tmp_path):
    """Two full turns through the graph: IR + faithfulness + relevancy."""

    from src.graph.orchestrator import build_maya_graph
    from src.maya.agent import MayaSynthesizer
    from src.maya.guardrails import SessionTokenLimiter
    from src.maya.router import MayaRouter
    from src.observability.tracer import DualModeObservabilityManager

    graph = build_maya_graph(
        runner.config,
        MayaRouter(runner.config),
        runner.engine,
        MayaSynthesizer(runner.config),
        DualModeObservabilityManager(session_id="eval-live"),
        limiter=SessionTokenLimiter(),
    )
    runner.graph = graph
    rows = [r for r in load_dataset() if r["tier"] == "B_retrieval"][:2]
    summary = runner.run_full(rows, "live-full-slice")

    assert summary.faithfulness is not None and summary.faithfulness >= 0.5, (
        f"faithfulness {summary.faithfulness} — grounded responses must stay high"
    )
    assert summary.relevancy is not None and summary.relevancy >= 0.5
    assert summary.total_tokens > 0
    path = runner.save(summary, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    judged = [q for q in payload["per_query"] if q["faithfulness"] is not None]
    assert len(judged) == 2


def test_delta_serialization_end_to_end(runner, tmp_path):
    """Run twice with different engine results; delta must reflect the change."""
    rows = [r for r in load_dataset() if r["tier"] == "B_retrieval"][:3]

    real_retrieve = runner.engine.retrieve

    summary_a = runner.run_retrieval(rows, "live-delta")
    path_a = runner.save(summary_a, tmp_path)
    payload_a = json.loads(path_a.read_text(encoding="utf-8"))

    # sabotage the engine: return nothing -> second run measures the drop
    runner.engine.retrieve = lambda query, routing, top_k=8, candidate_pool=50: []
    summary_b = runner.run_retrieval(rows, "live-delta")
    runner.engine.retrieve = real_retrieve
    path_b = runner.save(summary_b, tmp_path)

    payload_b = json.loads(path_b.read_text(encoding="utf-8"))
    assert payload_a["delta"] is None
    assert payload_b["delta"]["hit_rate"] < 0  # degraded run shows negative delta
