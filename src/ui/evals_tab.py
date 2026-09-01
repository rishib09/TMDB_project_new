"""Evals dashboard tab (issue #7): benchmark results, deltas, history.

Reads the versioned JSON files produced by `src/evals/runner.py` — the UI
never computes metrics itself; it visualizes what the harness measured.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RESULTS_DIR = Path("evals/results")
_METRICS = ["hit_rate", "mrr", "context_precision", "faithfulness", "relevancy"]
_METRIC_LABELS = {
    "hit_rate": "Hit Rate@5", "mrr": "MRR@5", "context_precision": "Context Precision@5",
    "faithfulness": "Faithfulness (judge)", "relevancy": "Relevancy (judge)",
}


def _load_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render_metric_scorecards(run: dict) -> None:
    """Metric cards with delta chips vs the previous same-label run (pure data)."""
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
    """One grouped bar per label × metric — version comparison at a glance."""
    rows = []
    for run in runs:
        for metric in _METRICS:
            if run.get(metric) is not None:
                rows.append({"run": run["label"], "metric": _METRIC_LABELS[metric],
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


def render_evals(results_dir: Path = RESULTS_DIR) -> None:
    st.subheader("📊 Evals Dashboard")
    st.caption(
        "Benchmark runs from `src/evals/runner.py` — deltas compare against the "
        "previous run of the same label. Sweep via the CLI, view here."
    )
    if not results_dir.exists() or not list(results_dir.glob("*.json")):
        st.info(
            "No benchmark results yet. Run:\n\n"
            "`python -m src.evals.runner --mode retrieval --versions v1_0_baseline,v1_1_enriched,v1_2_bge_hybrid`"
        )
        return

    paths = sorted(results_dir.glob("*.json"))
    runs = [_load_run(p) for p in paths]

    labels = [r["label"] for r in runs]
    selected = st.multiselect("Runs to display", labels, default=labels[-3:])
    selected_runs = [r for r in runs if r["label"] in selected]

    if not selected_runs:
        st.warning("Select at least one run.")
        return

    render_history_chart(selected_runs)

    st.markdown("#### Per-run scorecards")
    for run in selected_runs:
        with st.expander(
            f"**{run['label']}** — {run['mode']} · n={run['n_queries']} · "
            f"{run.get('timestamp', '?')[:19]}",
            expanded=run is selected_runs[-1],
        ):
            render_metric_scorecards(run)
            snap = run.get("config_snapshot", {})
            st.caption(
                f"config: rag={snap.get('rag_version', '?')} · alpha={snap.get('hybrid_alpha', '?')} "
                f"· reranker={snap.get('reranker_enabled', '?')} · synth={snap.get('synthesis_model', '?')}"
            )
            st.dataframe(
                pd.DataFrame(run["per_query"])[
                    ["query_id", "tier", "query", "hit_rate", "mrr", "context_precision"]
                ],
                use_container_width=True, hide_index=True,
            )
