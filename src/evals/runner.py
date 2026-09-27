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
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.domain.config import ExperimentConfig, PresetType
from src.domain.memory import UserSessionPreferences
from src.domain.routing import IntentType, QueryRoutingDecision, SuperlativeCriteria
from src.evals.conversations import (
    DEFAULT_CONVERSATIONS,
    ConversationSet,
    load_conversations,
)
from src.evals.conversation_metrics import (
    ConversationResult,
    ConversationRunSummary,
    ConversationTurnResult,
    composite_effective,
    observed_path_v1,
    score_constraints,
    turn_failed,
)
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
        "glm-5.3-flash",  # #97 z.ai native
        "google/gemini-3.5-flash-lite",  # #97 notch-down — the new pristine default
        "~google/gemini-flash-latest",  # former default (alias → gemini-3.8-flash), kept for A/B
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
        graph=None,  # CompiledStateGraph — required for full/conversation modes
        budget_tracker=None,  # WeeklyBudgetTracker — gates live modes (#59)
        dataset_version: str = "",
        tracer=None,  # DualModeObservabilityManager — conversation mode (#93)
    ) -> None:
        self.config = config
        self.engine = engine
        self.judge = judge
        self.graph = graph
        self.budget_tracker = budget_tracker
        self.dataset_version = dataset_version
        self.tracer = tracer

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

    def _refuse_dense_loss(self, query_id: str) -> None:
        """A hybrid run that lost dense measured BM25, not the config (#65).

        Sep 8 sweep: six embedding combos saved identical BM25-only rows
        stamped "rrf". Sparse-only configs (hybrid_alpha == 0) never ask for
        dense, so nothing to check there.
        """
        if self.config.hybrid_alpha == 0.0:
            return
        failure = getattr(self.engine, "last_dense_failure", None)
        if failure:
            raise RuntimeError(
                f"dense retrieval failed on {query_id} ({failure}) — hybrid run "
                "refused: the numbers would measure BM25-only, not this config"
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
            self._refuse_dense_loss(row["id"])
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
            self._refuse_dense_loss(row["id"])
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

    # --- conversation mode (#93, decisions #87 Q5–Q19) ------------------------

    def run_conversations(
        self, conversations: ConversationSet, label: str
    ) -> ConversationRunSummary:
        """Replays the golden conversations turn by turn and scores each turn.

        One fresh ``thread_id`` per conversation (D16); the fixed script
        never adapts to what the stack asks (G2 stack-neutrality); the budget
        gate fires between turns (Q16); usage is exact per turn from the
        UsageMetadataCallbackHandler (Q11/O1); failing turns embed their
        trace slice, captured at turn time — the ring would evict early
        turns on a 200-turn run (Q18).
        """
        if self.graph is None:
            raise ValueError("conversation mode requires the compiled graph")
        from src.observability.tracer import DualModeObservabilityManager

        tracer = self.tracer or DualModeObservabilityManager(session_id="benchmark")
        hash8 = config_hash(self.config)
        spend_before = (
            self.budget_tracker.weekly_spend() if self.budget_tracker is not None else None
        )
        convo_results: list[ConversationResult] = []
        faithfulness_scores: list[float] = []

        for convo in conversations.conversations:
            cfg = {"configurable": {"thread_id": uuid4().hex}}
            turn_rows: list[ConversationTurnResult] = []
            for turn in convo.turns:
                self._budget_check()
                ring_before = len(tracer.traces())
                trace_id = tracer.new_turn_trace()
                try:
                    shown_before = list(
                        self.graph.get_state(cfg).values.get("shown_movie_ids", [])
                    )
                    usage_handler = UsageMetadataCallbackHandler()
                    out = self.graph.invoke(
                        {"messages": [HumanMessage(content=turn.user)]},
                        {
                            **cfg,
                            "callbacks": [usage_handler, *tracer.callbacks()],
                            "metadata": {
                                **tracer.metadata(),
                                "eval_run": hash8,
                                "conversation_id": convo.id,
                                "turn": turn.n,
                            },
                        },
                    )
                except Exception as exc:  # noqa: BLE001 — Q19: the fixed script continues
                    # A per-turn failure (API error, schema crash) is recorded,
                    # never fatal. The two abort conditions are raised by OUR
                    # code OUTSIDE this try: _budget_check before the turn and
                    # _refuse_dense_loss after it.
                    turn_rows.append(ConversationTurnResult(
                        n=turn.n, user=turn.user,
                        expected_intent=turn.expect.intent.value, observed_intent="ERROR",
                        intent_correct=False,
                        expected_path=turn.expect.path, observed_path="error",
                        path_correct=False,
                        constraint_detail={"error": str(exc)[:300],
                                           "trace_slice": _bound_traces(tracer.traces()[ring_before:])},
                        trace_id=trace_id,
                    ))
                    continue
                self._refuse_dense_loss(convo.id)  # Q19/#65: never baseline BM25-only rows
                prefs = (
                    self.graph.get_state(cfg).values.get("session_preferences")
                    or UserSessionPreferences()
                )
                tokens = sum(
                    (u.get("total_tokens") or (u.get("input_tokens", 0) + u.get("output_tokens", 0)))
                    for u in usage_handler.usage_metadata.values()
                )

                expect = turn.expect
                decision = out.get("routing_decision")
                observed_intent = (
                    decision.intent.value
                    if decision is not None
                    else f"FUNNEL_{(out.get('turn_stage') or 'probe').upper()}"
                )
                # Stack-neutral (G2): v1 never routes funnel-answer turns — an
                # absent reading is None, never a wrong answer.
                intent_correct = (
                    observed_intent == expect.intent.value if decision is not None else None
                )
                # Q19: a fallback decision scores intent-incorrect even when the
                # label coincidentally matches — that is the visitor's experience.
                if intent_correct and decision is not None and decision.is_fallback:
                    intent_correct = False
                observed_path = observed_path_v1(out)
                if observed_path in ("retrieve", "ask"):
                    effective = composite_effective(out, prefs)
                    detail = score_constraints(expect.constraints, effective)
                else:
                    # Q13: constraints are the turn's business only when
                    # something must reach the engine (retrieve) or accumulate
                    # for it (ask). A pivot/converse/refuse turn deflected —
                    # the carry is not asserted by the golden row, and scoring
                    # it would conflate deflection with state pollution.
                    effective = {}
                    detail = {"fidelity": None, "keys": {}, "informational": [],
                              "violations": [], "intersection_failure": False}
                retrieved_ids = [m.id for m in out.get("retrieved_movies", [])]
                no_repeat_pass = (
                    not (set(retrieved_ids) & set(shown_before))
                    if expect.no_repeat else None
                )
                hit_rate = None
                faith = None
                judge_error = None
                if expect.relevant_movie_ids:  # Q15: judge + IR only here
                    hit_rate = hit_rate_at_k(retrieved_ids, expect.relevant_movie_ids, K)
                    if self.judge is not None:
                        try:
                            faith = self.judge.judge_faithfulness(
                                turn.user,
                                strip_formatting(out.get("final_response", "")),
                                out.get("retrieved_movies", []),
                            ).score
                            faithfulness_scores.append(faith)
                        except Exception as exc:  # noqa: BLE001 — judge failure is
                            # fail-open, never fatal (Q19): malformed judge JSON
                            # must not kill the run — recorded, turn continues.
                            # The truncation class is #74's missing max_tokens cap.
                            judge_error = str(exc)[:200]
                            print(
                                f"[judge] fail-open {convo.id} t{turn.n}: {judge_error}",
                                file=sys.stderr,
                            )

                if judge_error:  # fail-open, on the record (AGENTS.md)
                    detail = {**detail, "judge_error": judge_error}

                row = ConversationTurnResult(
                    n=turn.n,
                    user=turn.user,
                    expected_intent=expect.intent.value,
                    observed_intent=observed_intent,
                    intent_correct=intent_correct,
                    expected_path=expect.path,
                    observed_path=observed_path,
                    path_correct=observed_path == expect.path,
                    expected_constraints={
                        k: v for k, v in expect.constraints.model_dump().items()
                        if v not in (None, [], "")
                    },
                    effective=effective,
                    constraint_detail=detail,
                    fidelity=detail["fidelity"],
                    no_repeat=no_repeat_pass,
                    no_repeat_violation_ids=sorted(
                        set(retrieved_ids) & set(shown_before)
                    ) if expect.no_repeat else [],
                    reference_check=None,  # v1 has no referenced-titles mechanism (C10 lands with v2);
                                          # turns asserting the flag stay None — never a silent pass
                    ranked_ids=retrieved_ids,
                    relevant_ids=expect.relevant_movie_ids,
                    hit_rate=hit_rate,
                    tokens=tokens,
                    is_fallback=bool(decision is not None and decision.is_fallback),
                    fallback_reason=(decision.fallback_reason or "") if decision is not None else "",
                    trace_id=trace_id,
                )
                turn_rows.append(row)
                if turn_failed(row):  # Q18: slice captured at turn time, bounded
                    row.constraint_detail = {
                        **detail,
                        "trace_slice": _bound_traces(
                            tracer.traces()[ring_before:]
                        ),
                    }

            intents = [t.intent_correct for t in turn_rows if t.intent_correct is not None]
            convo_results.append(ConversationResult(
                id=convo.id, tier=convo.tier, title=convo.title,
                n_turns=len(turn_rows), per_turn=turn_rows,
                intent_accuracy=(sum(intents) / len(intents)) if intents else None,
                path_accuracy=aggregate([float(t.path_correct) for t in turn_rows]),
                fidelity=aggregate([t.fidelity for t in turn_rows if t.fidelity is not None]),
                failed=any(turn_failed(t) for t in turn_rows),
            ))

        all_turns = [t for c in convo_results for t in c.per_turn]
        intents = [t.intent_correct for t in all_turns if t.intent_correct is not None]
        no_repeat_turns = [t.no_repeat for t in all_turns if t.no_repeat is not None]
        spend_now = (
            self.budget_tracker.weekly_spend() if self.budget_tracker is not None else None
        )
        return ConversationRunSummary(
            label=label,
            config_snapshot=self.config.model_dump(),
            config_hash=hash8,
            preset=preset_slug(self.config),
            dataset_version=conversations.version,
            routing_stack=self.config.routing_stack,
            n_conversations=len(convo_results),
            n_turns=len(all_turns),
            intent_accuracy=(sum(intents) / len(intents)) if intents else None,
            path_accuracy=aggregate([float(t.path_correct) for t in all_turns]),
            constraint_fidelity=aggregate(
                [t.fidelity for t in all_turns if t.fidelity is not None]
            ),
            no_repeat_rate=(
                aggregate([float(v) for v in no_repeat_turns]) if no_repeat_turns else None
            ) or 0.0,
            intersection_failures=sum(
                1 for t in all_turns if t.constraint_detail.get("intersection_failure")
            ),
            judge_turns=len(faithfulness_scores),
            faithfulness=(
                aggregate(faithfulness_scores) if faithfulness_scores else None
            ),
            total_tokens=sum(t.tokens for t in all_turns),
            total_cost_usd=(spend_now - spend_before)
            if spend_before is not None and spend_now is not None else 0.0,
            per_conversation=convo_results,
        )

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
        return _write_run(
            summary.model_dump(),
            preset=summary.preset, config_hash8=summary.config_hash,
            mode=summary.mode,
            delta_metrics=("hit_rate", "mrr", "context_precision", "faithfulness",
                           "relevancy", "routing_accuracy"),
            out_dir=out_dir,
        )

    def save_conversations(
        self, summary: ConversationRunSummary, out_dir: Path = RESULTS_DIR
    ) -> Path:
        """Same identity/delta contract as save(), mode="conversation" (#93)."""
        return _write_run(
            summary.model_dump(),
            preset=summary.preset, config_hash8=summary.config_hash,
            mode=summary.mode,
            delta_metrics=("intent_accuracy", "path_accuracy", "constraint_fidelity",
                           "no_repeat_rate", "intersection_failures"),
            out_dir=out_dir,
        )


def _bound_traces(traces: list[dict], chars: int = 200) -> list[dict]:
    """Q18: failing-turn trace slices embedded in the result file, bounded."""
    def _bound(value):
        return value[:chars] + "…" if isinstance(value, str) and len(value) > chars else value
    return [{k: _bound(v) for k, v in trace.items()} for trace in traces]


def _write_run(
    payload: dict, *, preset: str, config_hash8: str, mode: str,
    delta_metrics: tuple[str, ...], out_dir: Path,
) -> Path:
    """Shared #59 writer: filename identity + delta vs prior same hash+mode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    out_path = out_dir / f"{preset}_{config_hash8}_{now:%Y%m%dT%H%M%SZ}.json"
    payload["timestamp"] = now.isoformat()
    prior = sorted(out_dir.glob(f"*_{config_hash8}_*.json"))
    prior_same_mode = None
    for path in reversed(prior):
        candidate = json.loads(path.read_text(encoding="utf-8"))
        if candidate.get("mode") == mode:
            prior_same_mode = candidate
            break
    if prior_same_mode is not None:
        payload["delta"] = {
            metric: round(payload[metric] - prior_same_mode.get(metric, 0.0), 4)
            for metric in delta_metrics
            if payload.get(metric) is not None and prior_same_mode.get(metric) is not None
        }
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


def _push_langfuse_conversations(
    summary: ConversationRunSummary, dataset_name: str = "maya-conversations"
) -> None:
    """Records the run in Langfuse (optional, best-effort, #93 Q18).

    Uses ONLY the installed 4.15.1 client surface: ``create_dataset`` +
    ``create_dataset_item`` + ``create_score`` attached to the turn's own
    trace id (the one the driver minted per turn) — the v2-era fluent
    dataset.item()/run().observe() API does not exist on this pin (round-4
    review finding). Baseline comparison (v1 vs v2) is Langfuse's scores/
    traces view; no custom comparison code.
    """
    try:
        from langfuse import Langfuse

        lf = Langfuse()
        lf.create_dataset(name=dataset_name)
        for convo in summary.per_conversation:
            for turn in convo.per_turn:
                lf.create_dataset_item(
                    dataset_name=dataset_name,
                    input=turn.user,
                    expected_output={"intent": turn.expected_intent, "path": turn.expected_path},
                    metadata={"conversation_id": convo.id, "turn": turn.n, "tier": convo.tier},
                )
                if not turn.trace_id:
                    continue
                scores = [("fidelity", turn.fidelity)]
                if turn.intent_correct is not None:
                    scores.append(("intent_correct", float(turn.intent_correct)))
                scores.append(("path_correct", float(turn.path_correct)))
                for name, value in scores:
                    if value is None:
                        continue
                    lf.create_score(
                        trace_id=turn.trace_id, name=name, value=value,
                        data_type="NUMERIC",
                        comment=f"{convo.id} t{turn.n} stack={summary.routing_stack}",
                    )
        print(f"[langfuse] conversation scores recorded on {summary.n_turns} turn traces")
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
    if getattr(summary, "mode", "") == "conversation":
        intent = summary.intent_accuracy
        intent_txt = f"{intent:.2f}" if intent is not None else "n/a"
        print(
            f"[{label}] conversation stack={summary.routing_stack} "
            f"convs={summary.n_conversations} turns={summary.n_turns} "
            f"intent={intent_txt} "
            f"path={summary.path_accuracy:.2f} "
            f"fidelity={summary.constraint_fidelity:.2f} "
            f"no_repeat={summary.no_repeat_rate:.2f} "
            f"intersections={summary.intersection_failures} "
            f"tokens={summary.total_tokens}"
        )
    elif summary.mode == "routing":
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
    conversations: ConversationSet | None = None,
) -> BenchmarkSummary | ConversationRunSummary | None:
    """One run of one config. Returns None when the collection isn't built.

    Availability guard (#30 rule): a combo without a built Chroma collection
    is skipped with a warning, never silently searched against nothing.
    """
    from src.maya.guardrails import WeeklyBudgetTracker

    tracker = (
        WeeklyBudgetTracker(db) if mode in ("routing", "full", "conversation") else None
    )
    if mode == "conversation":
        # #93: fresh checkpointer + thread per conversation; the tracer's
        # session IS the run (one trace per turn, Q18).
        if conversations is None:
            raise ValueError("conversation mode requires the golden conversations")
        if config.routing_stack == "v2":
            print(
                "conversation mode: the v2 routing stack does not exist yet (#83) "
                "— the baseline is v1",
                file=sys.stderr,
            )
            return None
        target = collection_name(config.column_preset, config.embedding_profile)
        if not store.has_collection(target):
            print(f"[{label}] skipped — collection `{target}` is not built", file=sys.stderr)
            return None
        from src.graph.orchestrator import build_maya_graph
        from src.maya.agent import MayaSynthesizer
        from src.maya.router import MayaRouter
        from src.maya.guardrails import SessionCostLimiter
        from src.observability.tracer import DualModeObservabilityManager

        tracer = DualModeObservabilityManager(session_id=f"eval-{config_hash(config)}")
        engine = _engine_for(config, db, store)
        runner = BenchmarkRunner(
            config, engine,
            judge=MayaJudge(config),
            graph=build_maya_graph(
                config,
                MayaRouter(config, genre_vocabulary=db.distinct_genres()),
                engine,
                MayaSynthesizer(config),
                tracer,
                # cap=None (#93): 23 scripted conversations share this graph —
                # a per-user-session cap would refuse every turn after the
                # first few retrievals. The weekly tracker gates the run.
                limiter=SessionCostLimiter(cap=None),
                budget_tracker=tracker,
                checkpointer=InMemorySaver(),
            ),
            budget_tracker=tracker,
            dataset_version=conversations.version,
            tracer=tracer,
        )
        return runner.run_conversations(conversations, label)
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
    from src.maya.guardrails import SessionCostLimiter
    from src.maya.router import MayaRouter
    from src.observability.tracer import DualModeObservabilityManager

    runner.graph = build_maya_graph(
        config, MayaRouter(config), engine, MayaSynthesizer(config),
        DualModeObservabilityManager(session_id="benchmark"),
        limiter=SessionCostLimiter(),
    )
    runner.judge = MayaJudge(config)
    return runner.run_full(queries, label)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Maya benchmark runner (#6, #59, #93)")
    parser.add_argument("--mode", choices=["retrieval", "full", "routing", "conversation"], default="retrieval")
    parser.add_argument("--label", default=None, help="display label (default: preset slug)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--conversations", type=Path, default=DEFAULT_CONVERSATIONS,
                        help="golden conversations file (conversation mode)")
    parser.add_argument("--ids", default=None,
                        help="comma-separated conversation ids (smoke runs, e.g. C01,C02)")
    parser.add_argument("--tier", default=None,
                        help="filter by golden tier (conversation mode)")
    parser.add_argument("--stack", choices=["v1", "v2"],
                        default=os.getenv("MAYA_ROUTING_STACK", "v1"),
                        help="routing stack under test (#83; v2 not yet built)")
    parser.add_argument("--limit", type=int, default=None, help="first N queries (smoke runs)")
    parser.add_argument("--push-langfuse", action="store_true")
    parser.add_argument(
        "--no-langfuse", action="store_true",
        help="strip LANGFUSE keys before anything runs: local trace ring only. "
        "A dead/slow Langfuse endpoint must never stall a baseline run "
        "(observed: OTel exporter DNS failure + rate-limit backoffs).",
    )
    parser.add_argument(
        "--router-model", default=None,
        help="override config.router_model (routing-mode A/B, #29)",
    )
    parser.add_argument("--synthesis-model", help="override config.synthesis_model (#89 sweep)")
    parser.add_argument(
        "--router-pin", action="store_true",
        help="#89: keep router_model verbatim via OpenRouter even under ZAI_API_KEY",
    )
    parser.add_argument(
        "--synthesis-pin", action="store_true",
        help="#89: keep synthesis_model verbatim via OpenRouter even under ZAI_API_KEY",
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

    if args.no_langfuse:  # before ANY tracer is constructed: local ring only
        os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
        os.environ.pop("LANGFUSE_SECRET_KEY", None)
        os.environ.pop("LANGFUSE_HOST", None)
        print("[trace] cloud disabled — local ring only (--no-langfuse)")

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
    if args.synthesis_model:
        config = config.model_copy(update={"synthesis_model": args.synthesis_model})
    sweep_pins = {
        k: True
        for k, on in (
            ("pin_router_config_id", args.router_pin),
            ("pin_synthesis_config_id", args.synthesis_pin),
        )
        if on
    }
    if sweep_pins:
        config = config.model_copy(update=sweep_pins)  # #89: explicit in the envelope
    if args.mode == "conversation":
        config = config.model_copy(update={"routing_stack": args.stack})
        conversations = load_conversations(args.conversations)
        if args.ids:
            wanted = {c.strip().upper() for c in args.ids.split(",")}
            conversations = conversations.model_copy(update={
                "conversations": [c for c in conversations.conversations if c.id.upper() in wanted]
            })
        if args.tier:
            conversations = conversations.model_copy(update={
                "conversations": [c for c in conversations.conversations if c.tier == args.tier]
            })
        if not conversations.conversations:
            print("no conversations matched the given --ids/--tier filters", file=sys.stderr)
            return 1
    else:
        conversations = None
    label = args.label or (
        f"conv-{config.routing_stack}" if args.mode == "conversation"
        else f"routing_{config.router_model.split('/')[-1]}" if args.mode == "routing"
        else preset_slug(config)
    )
    summary = _run_one(config, args.mode, queries, label, dataset_version, db, store,
                       conversations=conversations)
    if summary is None:
        return 1
    runner = BenchmarkRunner(config, engine=None)
    path = (
        runner.save_conversations(summary)
        if args.mode == "conversation" else runner.save(summary)
    )
    _report(summary, path)
    if args.push_langfuse:
        if args.mode == "conversation":
            _push_langfuse_conversations(summary)
        else:
            _push_langfuse(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
