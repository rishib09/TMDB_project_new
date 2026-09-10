"""Evals dashboard tab (issue #7, modernized by #60 / #54 grill verdict).

Reads the identity-keyed JSON files produced by `src/evals/runner.py` (#59)
— the UI never computes metrics itself; it visualizes what the harness
measured. Runs group under preset bookmarks; sweeps render via a knob
picker; the current Lab config can be evaluated in place.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.domain.config import ExperimentConfig, PresetType
from src.evals.runner import SWEEPS, sweep_configs
from src.feedback.store import FeedbackStore

RESULTS_DIR = Path("evals/results")
_METRICS = ["hit_rate", "mrr", "context_precision", "faithfulness", "relevancy"]
_METRIC_LABELS = {
    "hit_rate": "Hit Rate@5", "mrr": "MRR@5", "context_precision": "Context Precision@5",
    "faithfulness": "Faithfulness (judge)", "relevancy": "Relevancy (judge)",
}
_PRESET_TITLES = {
    "production": "Production", "fast_budget": "Fast Budget",
    "naive_baseline": "Naive Baseline",
}
_MODE_COSTS = {  # live LLM calls per golden query (#54 grill D8)
    "retrieval": 0, "routing": 1, "full": 3,
}


def _load_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_display_name(run: dict) -> str:
    """Preset bookmark, `custom @ hash` for off-preset runs, legacy marked."""
    if not run.get("config_hash"):
        return f"{run.get('label', '?')} (legacy)"
    preset = run.get("preset", "custom")
    if preset in _PRESET_TITLES:
        return _PRESET_TITLES[preset]
    return f"custom @ {run['config_hash']}"


def staleness_flags(
    run: dict, collection_exists=None, current_dataset_version: str = ""
) -> list[str]:
    """#54 grill D7: flag stale runs, never auto-delete. Pure, unit-tested."""
    flags = []
    collection = run.get("collection", "")
    if collection and collection_exists is not None and not collection_exists(collection):
        flags.append(f"collection `{collection}` no longer exists")
    stamped = run.get("dataset_version", "")
    if stamped and current_dataset_version and stamped != current_dataset_version:
        flags.append(
            f"dataset {stamped} predates the current set ({current_dataset_version})"
        )
    return flags


def sweep_rows(runs: list[dict], knob: str) -> tuple[list[dict], list[str]]:
    """Chart rows for a knob sweep + the swept values missing a run. Pure.

    Rows come from runs stamped `sweep.knob == knob` (newest per value);
    missing values power the "run this sweep" hint (#54 grill D9).
    """
    newest_by_value: dict[str, dict] = {}
    for run in sorted(runs, key=lambda r: r.get("timestamp", "")):
        sweep = run.get("sweep") or {}
        if sweep.get("knob") == knob:
            newest_by_value[str(sweep.get("value"))] = run
    rows = []
    for value, run in newest_by_value.items():
        for metric in _METRICS:
            if run.get(metric) is not None:
                rows.append(
                    {"value": value, "metric": _METRIC_LABELS[metric], "score": run[metric]}
                )
    expected = [label for label, _ in sweep_configs(knob)]
    missing = [v for v in expected if v not in newest_by_value]
    return rows, missing


def sweep_baseline_label(knob: str) -> str | None:
    """The sweep value that IS the pristine Production baseline. Pure."""
    production = ExperimentConfig().apply_preset(PresetType.PRODUCTION_HYBRID)
    for label, config in sweep_configs(knob):
        if config == production:
            return label
    return None


def render_metric_scorecards(run: dict) -> None:
    """Metric cards with delta chips vs the prior same-config-hash run."""
    cols = st.columns(5)
    for col, metric in zip(cols, _METRICS):
        if run.get(metric) is None:
            continue
        delta = (run.get("delta") or {}).get(metric)
        col.metric(
            _METRIC_LABELS[metric],
            f"{run[metric]:.2f}",
            delta=f"{delta:+.2f}" if delta is not None else None,
        )


def render_history_chart(runs: list[dict]) -> None:
    """One grouped bar per run × metric — architecture comparison at a glance."""
    rows = []
    for run in runs:
        for metric in _METRICS:
            if run.get(metric) is not None:
                rows.append({"run": run_display_name(run), "metric": _METRIC_LABELS[metric],
                             "value": run[metric]})
    if not rows:
        st.info("No runs with metrics yet — run `python -m src.evals.runner`.")
        return
    frame = pd.DataFrame(rows)
    fig = px.bar(frame, x="run", y="value", color="metric", barmode="group",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      legend_title="", yaxis_title="score")
    st.plotly_chart(fig, use_container_width=True)


def render_sweep_section(runs: list[dict]) -> None:
    """Knob picker → metric vs knob-value from matching sweep runs (D9)."""
    st.markdown("#### Knob impact")
    knob = st.selectbox("Knob", sorted(SWEEPS), format_func=lambda k: k.replace("_", " "))
    rows, missing = sweep_rows(runs, knob)
    if rows:
        frame = pd.DataFrame(rows)
        # Categorical x-axis: plotly's add_vline can't annotate string ticks,
        # so the Production baseline is marked in the tick label itself.
        baseline = sweep_baseline_label(knob)
        if baseline is not None:
            frame["value"] = frame["value"].map(
                lambda v: f"{v} (baseline)" if v == baseline else v
            )
        fig = px.bar(frame, x="value", y="score", color="metric", barmode="group",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                          legend_title="", xaxis_title=knob, yaxis_title="score")
        st.plotly_chart(fig, use_container_width=True)
    if missing:
        st.caption(
            f"No runs yet for: {', '.join(missing)} — "
            f"`python -m src.evals.runner --sweep {knob}`"
        )


def render_evaluate_current(session) -> None:
    """Evaluate the sidebar's exact config (#54 grill D8): free by default,
    live modes opt-in with the call count + remaining budget printed."""
    st.markdown("#### Evaluate current Lab config")
    mode = st.radio(
        "Mode", ["retrieval", "routing", "full"], horizontal=True,
        format_func=lambda m: f"{m} ({'free' if _MODE_COSTS[m] == 0 else f'~{_MODE_COSTS[m]} LLM calls/query'})",
    )
    caption = "Deterministic replay — no LLM calls."
    if _MODE_COSTS[mode]:
        try:
            spend = session.budget_tracker.weekly_spend()
            remaining = max(session.budget_tracker.WEEKLY_CAP_USD - spend, 0.0)
            caption = f"Live run — budget-gated. Remaining this week: ${remaining:.2f}."
        except Exception:  # sink read failure must not break the tab
            caption = "Live run — budget-gated."
    st.caption(caption)
    if not st.button("Run evaluation", type="primary"):
        return
    from src.evals.runner import (
        BenchmarkRunner, DEFAULT_DATASET, _run_one, load_dataset, load_dataset_version,
    )
    from src.indexing.vector_store import MovieVectorStore
    from src.storage.database import MovieDatabase

    queries = load_dataset(DEFAULT_DATASET)
    with st.status(f"Running {mode} evaluation ({len(queries)} queries)", expanded=False):
        try:
            summary = _run_one(
                session.config.model_copy(deep=True), mode, queries,
                label="lab", dataset_version=load_dataset_version(DEFAULT_DATASET),
                db=MovieDatabase("data/tmdb_movies.db"),
                store=MovieVectorStore("data/chroma_db"),
            )
        except RuntimeError as exc:  # budget gate abort (#59)
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 — readable failure, not a traceback
            st.error(f"Evaluation failed: {type(exc).__name__}: {exc}")
            return
    if summary is None:
        st.error("The configured collection is not built — see the Lab combo picker.")
        return
    path = BenchmarkRunner(session.config, engine=None).save(summary)
    st.toast(f"Run saved: {path.name}")
    st.rerun()


def render_feedback_section(store: FeedbackStore | None = None) -> None:
    """User feedback breakdown by RAG version (issue #9)."""
    store = store or FeedbackStore()
    stats = store.stats_by_version()
    st.markdown("#### User Feedback")
    if not stats:
        st.info(
            "No feedback yet — rate replies with thumbs in the Chat tab; "
            "ratings land here per RAG version."
        )
        return
    frame = pd.DataFrame(stats)
    left, right = st.columns([1, 2])
    with left:
        st.dataframe(frame, use_container_width=True, hide_index=True)
    with right:
        fig = px.bar(
            frame, x="rag_version", y="avg_rating", color="n",
            hover_data=["thumbs_up", "thumbs_down"],
            color_continuous_scale="RdYlGn", range_y=[-1, 1],
            labels={"avg_rating": "avg rating (+1/−1)", "n": "n ratings"},
        )
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


def render_evals(session=None, results_dir: Path = RESULTS_DIR) -> None:
    st.header("Evals")
    st.caption(
        "Benchmark runs from `src/evals/runner.py` — one artifact per config "
        "snapshot; deltas compare against the prior run of the same config."
    )
    if session is not None:
        render_evaluate_current(session)
    render_feedback_section()
    if not results_dir.exists() or not list(results_dir.glob("*.json")):
        st.info(
            "No benchmark results yet. Run:\n\n"
            "`python -m src.evals.runner --combos`"
        )
        return

    paths = sorted(results_dir.glob("*.json"))
    runs = [_load_run(p) for p in paths]
    current_dataset_version = ""
    try:
        from src.evals.runner import DEFAULT_DATASET, load_dataset_version

        current_dataset_version = load_dataset_version(DEFAULT_DATASET)
    except Exception:  # dataset absent must not break the tab
        pass
    collection_exists = None
    if session is not None:
        from src.ui.sidebar_lab import _lab_store

        collection_exists = _lab_store(session).has_collection

    names = [run_display_name(r) for r in runs]
    modern_default = [n for r, n in zip(runs, names) if r.get("config_hash")]
    selected = st.multiselect(
        "Runs to display", sorted(set(names)), default=sorted(set(modern_default))[-4:]
    )
    selected_runs = [r for r, n in zip(runs, names) if n in selected]
    if not selected_runs:
        st.warning("Select at least one run.")
        return

    render_history_chart(selected_runs)
    render_sweep_section(runs)

    st.markdown("#### Per-run scorecards")
    for run in selected_runs:
        with st.expander(
            f"**{run_display_name(run)}** — {run['mode']} · n={run['n_queries']} · "
            f"{run.get('timestamp', '?')[:19]}",
            expanded=run is selected_runs[-1],
        ):
            for flag in staleness_flags(run, collection_exists, current_dataset_version):
                st.warning(f"Stale: {flag}")
            render_metric_scorecards(run)
            snap = run.get("config_snapshot", {})
            st.caption(
                f"config {run.get('config_hash', '?')}: "
                f"collection={run.get('collection') or snap.get('rag_version', '?')} "
                f"· alpha={snap.get('hybrid_alpha', '?')} "
                f"· reranker={snap.get('reranker_enabled', '?')} · synth={snap.get('synthesis_model', '?')}"
            )
            st.dataframe(
                pd.DataFrame(run["per_query"])[
                    ["query_id", "tier", "query", "hit_rate", "mrr", "context_precision"]
                ],
                use_container_width=True, hide_index=True,
            )
