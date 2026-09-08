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

from src.domain.config import ExperimentConfig, PresetType
from src.domain.routing import IntentType, QueryRoutingDecision, SuperlativeCriteria
from src.evals.identity import config_hash, preset_slug
from src.evals.judge import MayaJudge, strip_formatting
from src.evals.metrics import (
    BenchmarkSummary,
    QueryEvalResult,
    aggregate,
    context_precision_at_k,
    hit_rate_at_k,
    mrr_at_k,
    routing_accuracy,
)
from src.indexing.embeddings import collection_name, provider_from_profile
from src.indexing.vector_store import MovieVectorStore
from src.retrieval.hybrid_engine import HybridRetrievalEngine
from src.storage.database import MovieDatabase

DEFAULT_DATASET = Path("data/eval_benchmark_dataset.json")
RESULTS_DIR = Path("evals/results")
K = 5  # benchmark reports @5 throughout (matches the #4 close-out numbers)

#: #59 curated OFAT sweeps (#54 grill D3) — every point is Production + ONE
#: knob changed. Synthesis-side knobs are deliberately absent (on-demand only).
ADR_0008_COMBOS: list[tuple[str, str]] = [
    ("full", "gemini_embedding_2"), ("minimal", "gemini_embedding_2"),
    ("full", "nemotron_free"), ("minimal", "nemotron_free"),
    ("full", "lfm_free"), ("minimal", "lfm_free"),
]
SWEEPS: dict[str, list] = {
    "reranker": ["off", "ms-marco-MiniLM-L-12-v2", "ms-marco-TinyBERT-L-2-v2", "ce-esci-MiniLM-L12-v2"],
    "hybrid_alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
    "retrieval_top_k": [3, 5, 10],
    "embedding_combo": ADR_0008_COMBOS,
    "router_model": [
        "~google/gemini-flash-latest",
        "meta-llama/llama-3.3-70b-instruct",
        "meta-llama/llama-3.2-3b-instruct",
    ],
}


def sweep_configs(knob: str) -> list[tuple[str, ExperimentConfig]]:
    """(value-label, config) points for a knob sweep — pure, unit-tested.

    Baseline = pristine Production preset; exactly one knob varies per point.
    """
    if knob not in SWEEPS:
        raise ValueError(f"unknown sweep knob: {knob} (have: {sorted(SWEEPS)})")
    points = []
    for value in SWEEPS[knob]:
        config = ExperimentConfig().apply_preset(PresetType.PRODUCTION_HYBRID)
        if knob == "reranker":
            config.reranker_enabled = value != "off"
            if value != "off":
                config.reranker_model = value
            label = str(value)
        elif knob == "embedding_combo":
            config.column_preset, config.embedding_profile = value
            label = f"{value[0]}_{value[1]}"
        else:
            setattr(config, knob, value)
            label = str(value)
        points.append((label, config))
    return points


def load_dataset_version(path: Path = DEFAULT_DATASET) -> str:
    """Dataset build stamp for staleness flags (#54 grill D7)."""
    return str(json.loads(path.read_text(encoding="utf-8")).get("version", ""))


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
        budget_tracker=None,  # WeeklyBudgetTracker — gates live modes (#59)
        dataset_version: str = "",
    ) -> None:
        self.config = config
        self.engine = engine
        self.judge = judge
        self.graph = graph
        self.budget_tracker = budget_tracker
        self.dataset_version = dataset_version

    def _budget_check(self) -> None:
        """Aborts a live run when the weekly cap is exhausted (#54 grill D6).

        Same guardrail chat turns get — an eval sweep must not silently burn
        the $10/week budget. Checked before the run and between queries.
        """
        if self.budget_tracker is None:
            return
        spend = self.budget_tracker.weekly_spend()
        if self.budget_tracker.verdict_for(spend).value == "blocked":
            raise RuntimeError(
                f"weekly API budget exhausted (${spend:.2f}) — eval run aborted; "
                "spend resets on Monday"
            )

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
            self._budget_check()
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

    def run_routing(self, queries: list[dict], label: str, router) -> BenchmarkSummary:
        """Routing mode (#29): live router call per query vs expected_intent.

        Cheap by design — no retrieval, no synthesis, no judge. Records
        per-row confidence and fallback so the #12 Gate 1 distribution and
        the model before/after comparison come from the same run.
        """
        from src.domain.memory import ConversationState

        results = []
        for row in queries:
            self._budget_check()
            decision = router.route(row["query"], ConversationState())
            results.append(
                QueryEvalResult(
                    query_id=row["id"], tier=row["tier"], query=row["query"],
                    expected_path=row["expected_path"],
                    routed_intent=decision.intent.value,
                    intent_correct=decision.intent.value == row["expected_intent"],
                    confidence=decision.confidence,
                    is_fallback=decision.is_fallback,
                )
            )
        summary = self._summarize(results, label, mode="routing")
        per_intent: dict[str, list[bool]] = {}
        for row, result in zip(queries, results, strict=True):
            per_intent.setdefault(row["expected_intent"], []).append(
                bool(result.intent_correct)
            )
        summary.routing_accuracy = routing_accuracy(results)
        summary.routing_per_intent = {
            intent: sum(hits) / len(hits) for intent, hits in sorted(per_intent.items())
        }
        summary.fallback_count = sum(1 for r in results if r.is_fallback)
        return summary

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
            config_hash=config_hash(self.config),
            preset=preset_slug(self.config),
            dataset_version=self.dataset_version,
            collection=str(getattr(self.engine, "rag_version", "")),
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
        """Writes `{preset}_{confighash8}_{timestamp}.json` (#59 run identity).

        Delta is computed vs the most recent prior run with the SAME config
        hash and mode — never vs a different architecture wearing the same
        label. Legacy `<label>.json` artifacts don't match the glob and are
        left untouched (history, #54 grill D2).
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC)
        out_path = out_dir / f"{summary.preset}_{summary.config_hash}_{now:%Y%m%dT%H%M%SZ}.json"
        payload = summary.model_dump()
        payload["timestamp"] = now.isoformat()
        prior = sorted(out_dir.glob(f"*_{summary.config_hash}_*.json"))
        prior_same_mode = None
        for path in reversed(prior):
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if candidate.get("mode") == summary.mode:
                prior_same_mode = candidate
                break
        if prior_same_mode is not None:
            summary.delta = {
                metric: round(payload[metric] - prior_same_mode.get(metric, 0.0), 4)
                for metric in ("hit_rate", "mrr", "context_precision", "faithfulness", "relevancy", "routing_accuracy")
                if payload.get(metric) is not None and prior_same_mode.get(metric) is not None
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


def _engine_for(config: ExperimentConfig, db: MovieDatabase, store: MovieVectorStore) -> HybridRetrievalEngine:
    """#59: the dense pair derives from config — collection AND query embedder."""
    return HybridRetrievalEngine(
        db=db,
        vector_store=store,
        rag_version=collection_name(config.column_preset, config.embedding_profile),
        hybrid_alpha=config.hybrid_alpha,
        reranker_enabled=config.reranker_enabled,
        reranker_model=config.reranker_model,
        search_provider=provider_from_profile(config.embedding_profile),
    )


def _report(summary: BenchmarkSummary, path: Path) -> None:
    label = summary.label
    if summary.mode == "routing":
        print(
            f"[{label}] routing n={summary.n_queries} "
            f"accuracy={summary.routing_accuracy:.2f} fallbacks={summary.fallback_count}"
        )
        for intent, acc in (summary.routing_per_intent or {}).items():
            print(f"[{label}]   {intent}: {acc:.2f}")
    else:
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


def _run_one(
    config: ExperimentConfig, mode: str, queries: list[dict], label: str,
    dataset_version: str, db: MovieDatabase, store: MovieVectorStore,
) -> BenchmarkSummary | None:
    """One run of one config. Returns None when the collection isn't built.

    Availability guard (#30 rule): a combo without a built Chroma collection
    is skipped with a warning, never silently searched against nothing.
    """
    from src.maya.guardrails import WeeklyBudgetTracker

    tracker = WeeklyBudgetTracker(db) if mode in ("routing", "full") else None
    if mode == "routing":
        # Routing mode needs no retrieval stack — router + dataset only (#29).
        from src.maya.router import MayaRouter

        runner = BenchmarkRunner(
            config, engine=None, budget_tracker=tracker, dataset_version=dataset_version
        )
        return runner.run_routing(queries, label, MayaRouter(config))

    target = collection_name(config.column_preset, config.embedding_profile)
    if not store.has_collection(target):
        print(f"[{label}] skipped — collection `{target}` is not built", file=sys.stderr)
        return None
    engine = _engine_for(config, db, store)
    runner = BenchmarkRunner(
        config, engine, budget_tracker=tracker, dataset_version=dataset_version
    )
    if mode == "retrieval":
        return runner.run_retrieval(queries, label)

    from src.graph.orchestrator import build_maya_graph
    from src.maya.agent import MayaSynthesizer
    from src.maya.guardrails import SessionTokenLimiter
    from src.maya.router import MayaRouter
    from src.observability.tracer import DualModeObservabilityManager

    runner.graph = build_maya_graph(
        config, MayaRouter(config), engine, MayaSynthesizer(config),
        DualModeObservabilityManager(session_id="benchmark"),
        limiter=SessionTokenLimiter(),
    )
    runner.judge = MayaJudge(config)
    return runner.run_full(queries, label)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Maya benchmark runner (#6, #59)")
    parser.add_argument("--mode", choices=["retrieval", "full", "routing"], default="retrieval")
    parser.add_argument("--label", default=None, help="display label (default: preset slug)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=None, help="first N queries (smoke runs)")
    parser.add_argument("--push-langfuse", action="store_true")
    parser.add_argument(
        "--router-model", default=None,
        help="override config.router_model (routing-mode A/B, #29)",
    )
    parser.add_argument(
        "--sweep", choices=sorted(SWEEPS), default=None,
        help="OFAT sweep of one knob against the Production baseline (#59)",
    )
    parser.add_argument(
        "--combos", action="store_true",
        help="run the 6 ADR 0008 column×profile cells (= --sweep embedding_combo)",
    )
    args = parser.parse_args(argv)

    queries = load_dataset(args.dataset)
    if args.limit:
        queries = queries[: args.limit]
    dataset_version = load_dataset_version(args.dataset)
    db = MovieDatabase("data/tmdb_movies.db")
    store = MovieVectorStore("data/chroma_db")

    sweep = "embedding_combo" if args.combos else args.sweep
    if sweep:
        # Router-model sweeps are live routing runs; everything else is free
        # retrieval replay (#54 grill D3).
        mode = "routing" if sweep == "router_model" else "retrieval"
        for value_label, config in sweep_configs(sweep):
            label = f"{sweep}={value_label}"
            summary = _run_one(config, mode, queries, label, dataset_version, db, store)
            if summary is None:
                continue
            summary.sweep = {"knob": sweep, "value": value_label, "baseline": "production"}
            runner = BenchmarkRunner(config, engine=None)
            path = runner.save(summary)
            _report(summary, path)
            if args.push_langfuse:
                _push_langfuse(summary)
        return 0

    config = ExperimentConfig()
    if args.router_model:
        config = config.model_copy(update={"router_model": args.router_model})
    label = args.label or (
        f"routing_{config.router_model.split('/')[-1]}" if args.mode == "routing"
        else preset_slug(config)
    )
    summary = _run_one(config, args.mode, queries, label, dataset_version, db, store)
    if summary is None:
        return 1
    runner = BenchmarkRunner(config, engine=None)
    path = runner.save(summary)
    _report(summary, path)
    if args.push_langfuse:
        _push_langfuse(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
