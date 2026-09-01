"""Benchmark runner for the Maya evaluation harness (issue #6).

Two modes:
- ``retrieval`` (offline, no LLM): replays each dataset row's ground-truth
  routing decision through ``HybridRetrievalEngine.retrieve()`` — IR metrics
  only. This is where config sweeps live (reranker, hybrid_alpha, tier).
- ``full`` (live, opt-in): drives the compiled LangGraph from #5 — IR metrics
  on retrieved ids plus judge faithfulness/relevancy on the actual response.

Output: ``evals/results/<label>.json`` with the config snapshot, per-query
rows, aggregates, and a ``delta`` block vs. the previous run of the same
label (versioned delta serialization). ``--push-langfuse`` also records the
run as a Langfuse dataset experiment (zero new dependencies).
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.domain.config import ExperimentConfig
from src.domain.routing import IntentType, QueryRoutingDecision, SuperlativeCriteria
from src.evals.judge import MayaJudge, strip_formatting
from src.evals.metrics import (
    BenchmarkSummary,
    QueryEvalResult,
    aggregate,
    context_precision_at_k,
    hit_rate_at_k,
    mrr_at_k,
)
from src.indexing.vector_store import MovieVectorStore
from src.retrieval.hybrid_engine import HybridRetrievalEngine
from src.storage.database import MovieDatabase

DEFAULT_DATASET = Path("data/eval_benchmark_dataset.json")
RESULTS_DIR = Path("evals/results")
K = 5  # benchmark reports @5 throughout (matches the #4 close-out numbers)


def load_dataset(path: Path = DEFAULT_DATASET) -> list[dict]:
    """Loads and validates the benchmark dataset (schema checked in tests)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"id", "tier", "query", "expected_intent", "expected_path", "relevant_movie_ids"}
    for row in data["queries"]:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"dataset row {row.get('id', '?')} missing fields: {missing}")
    return data["queries"]


def _routing_from_row(row: dict) -> QueryRoutingDecision:
    """Replays the dataset's ground-truth routing decision (no LLM).

    The dataset stores what the #3-validated router SHOULD decide; the
    retrieval-mode runner measures retrieval quality against that replay.
    """
    sup = row.get("superlative")
    return QueryRoutingDecision(
        intent=IntentType(row["expected_intent"]),
        confidence=1.0,
        standalone_query=row["query"],
        requires_rag=row["expected_path"] in ("sql", "rrf"),
        is_superlative=row["expected_path"] == "sql",
        superlative=SuperlativeCriteria(**sup) if sup else None,
        filters=row.get("filters"),
    )


class BenchmarkRunner:
    """Runs the benchmark dataset against the pipeline and serializes results."""

    def __init__(
        self,
        config: ExperimentConfig,
        engine: HybridRetrievalEngine,
        judge: MayaJudge | None = None,
        graph=None,  # CompiledStateGraph — required for full mode
    ) -> None:
        self.config = config
        self.engine = engine
        self.judge = judge
        self.graph = graph

    def run_retrieval(self, queries: list[dict], label: str) -> BenchmarkSummary:
        """Offline mode: IR metrics from deterministic retrieval replay.

        Rows without a retrieval expectation (pivot/no_retrieval guardrail
        rows) are skipped — they are exercised by the full/live mode and the
        adversarial suite instead.
        """
        retrievable = [q for q in queries if q["expected_path"] in ("sql", "rrf")]
        results = []
        for row in retrievable:
            routing = _routing_from_row(row)
            retrieved = self.engine.retrieve(
                query=routing.standalone_query, routing=routing,
                top_k=self.config.retrieval_top_k,
            )
            results.append(self._ir_result(row, [r.movie.id for r in retrieved]))
        return self._summarize(results, label, mode="retrieval")

    def run_full(self, queries: list[dict], label: str, cost_lookup=None) -> BenchmarkSummary:
        """Live mode: one graph turn per query + judge metrics. Requires API key."""
        if self.graph is None or self.judge is None:
            raise ValueError("full mode requires both graph and judge")
        results = []
        for row in queries:
            turn = self.graph.invoke({"messages": [HumanMessage(content=row["query"])]})
            movies = turn.get("retrieved_movies", [])
            response = turn.get("final_response", "")
            result = self._ir_result(row, [m.id for m in movies])
            result.response = response
            verdict = self.judge.judge_faithfulness(
                row["query"], strip_formatting(response), movies
            )
            result.faithfulness = verdict.score
            result.relevancy = self.judge.judge_relevancy(
                row["query"], strip_formatting(response)
            ).score
            usage = turn.get("synthesis_usage")
            result.tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
            if cost_lookup:
                result.cost_usd = cost_lookup()
            results.append(result)
        return self._summarize(results, label, mode="full")

    def _ir_result(self, row: dict, ranked_ids: list[int]) -> QueryEvalResult:
        relevant = row["relevant_movie_ids"]
        return QueryEvalResult(
            query_id=row["id"], tier=row["tier"], query=row["query"],
            expected_path=row["expected_path"], ranked_ids=ranked_ids,
            relevant_ids=relevant,
            hit_rate=hit_rate_at_k(ranked_ids, relevant, K),
            mrr=mrr_at_k(ranked_ids, relevant, K),
            context_precision=context_precision_at_k(ranked_ids, relevant, K),
        )

    def _summarize(self, results: list[QueryEvalResult], label: str, mode: str) -> BenchmarkSummary:
        snapshot = {
            **self.config.model_dump(),
            "rag_version": getattr(self.engine, "rag_version", "?"),
            "hybrid_alpha": getattr(self.engine, "hybrid_alpha", "?"),
            "reranker_enabled": getattr(self.engine, "reranker_enabled", "?"),
        }
        return BenchmarkSummary(
            label=label, mode=mode,
            config_snapshot=snapshot,
            n_queries=len(results),
            hit_rate=aggregate([r.hit_rate for r in results]),
            mrr=aggregate([r.mrr for r in results]),
            context_precision=aggregate([r.context_precision for r in results]),
            faithfulness=aggregate([r.faithfulness for r in results if r.faithfulness is not None]) or None,
            relevancy=aggregate([r.relevancy for r in results if r.relevancy is not None]) or None,
            total_tokens=sum(r.tokens for r in results),
            total_cost_usd=sum(r.cost_usd for r in results),
            per_query=results,
        )

    def save(self, summary: BenchmarkSummary, out_dir: Path = RESULTS_DIR) -> Path:
        """Writes <label>.json with a delta block vs. the previous same-label run."""
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{summary.label}.json"
        payload = summary.model_dump()
        payload["timestamp"] = datetime.now(UTC).isoformat()
        if out_path.exists():
            previous = json.loads(out_path.read_text(encoding="utf-8"))
            summary.delta = {
                metric: round(payload[metric] - previous.get(metric, 0.0), 4)
                for metric in ("hit_rate", "mrr", "context_precision", "faithfulness", "relevancy")
                if payload.get(metric) is not None and previous.get(metric) is not None
            }
            payload["delta"] = summary.delta
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_path


def _push_langfuse(summary: BenchmarkSummary, dataset_name: str = "maya-benchmark") -> None:
    """Records the run as a Langfuse dataset experiment (optional, best-effort)."""
    try:
        from langfuse import Langfuse

        lf = Langfuse()
        lf.create_dataset(name=dataset_name)
        dataset = lf.get_dataset(dataset_name)
        for row in summary.per_query:
            dataset.item(
                input=row.query, expected_output={"relevant_ids": row.relevant_ids}
            )
        run = dataset.run(name=f"{summary.label}-{summary.mode}")
        for row in summary.per_query:
            run.observe(
                input=row.query, output=row.response or row.ranked_ids,
                metadata={"hit_rate": row.hit_rate, "mrr": row.mrr},
            )
        print(f"[langfuse] experiment recorded: {run.name}")
    except Exception as exc:  # noqa: BLE001 — telemetry must never break a run
        print(f"[langfuse] skipped: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Maya benchmark runner (#6)")
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    parser.add_argument("--label", default=None, help="results file label (default: rag_version)")
    parser.add_argument("--versions", default=None, help="comma-separated rag_versions (retrieval mode)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None, help="first N queries (smoke runs)")
    parser.add_argument("--push-langfuse", action="store_true")
    args = parser.parse_args(argv)

    queries = load_dataset(args.dataset)
    if args.limit:
        queries = queries[: args.limit]

    config = ExperimentConfig()
    labels = args.versions.split(",") if args.versions else [None]
    for version in labels:
        engine = HybridRetrievalEngine(
            db=MovieDatabase("data/tmdb_movies.db"),
            vector_store=MovieVectorStore("data/chroma_db"),
            rag_version=version or "v1_1_enriched",
            hybrid_alpha=config.hybrid_alpha,
            reranker_enabled=config.reranker_enabled,
            reranker_model=config.reranker_model,
        )
        label = args.label or version or "default"
        runner = BenchmarkRunner(config, engine)
        if args.mode == "retrieval":
            summary = runner.run_retrieval(queries, label)
        else:
            from src.graph.orchestrator import build_maya_graph
            from src.maya.agent import MayaSynthesizer
            from src.maya.guardrails import SessionTokenLimiter
            from src.maya.router import MayaRouter
            from src.observability.tracer import DualModeObservabilityManager

            graph = build_maya_graph(
                config, MayaRouter(config), engine, MayaSynthesizer(config),
                DualModeObservabilityManager(session_id="benchmark"),
                limiter=SessionTokenLimiter(),
            )
            runner.graph = graph
            runner.judge = MayaJudge(config)
            summary = runner.run_full(queries, label)
        path = runner.save(summary)
        print(
            f"[{label}] {summary.mode} n={summary.n_queries} "
            f"hit@5={summary.hit_rate:.2f} mrr@5={summary.mrr:.2f} "
            f"cp@5={summary.context_precision:.2f}"
            + (f" faith={summary.faithfulness:.2f}" if summary.faithfulness is not None else "")
            + (f" rel={summary.relevancy:.2f}" if summary.relevancy is not None else "")
            + (f" tokens={summary.total_tokens}" if summary.total_tokens else "")
        )
        if summary.delta:
            print(f"[{label}] delta vs prior: {summary.delta}")
        print(f"[{label}] results: {path}")
        if args.push_langfuse:
            _push_langfuse(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
