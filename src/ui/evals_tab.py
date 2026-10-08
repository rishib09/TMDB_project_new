"""Evals view: a static Routing Stack v2 report (#142, verdict #141).

Reads the JSON files produced by `src/evals/runner.py`. The view does not
compute metrics and does not run an evaluation. The decision list is closed.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.evals.runner import sweep_configs

RESULTS_DIR = Path("evals/results")
_METRICS = ["hit_rate", "mrr", "context_precision", "faithfulness", "relevancy"]
#: The three retrieval scores drawn together on one chart per decision.
_CHART_METRICS = ("hit_rate", "mrr", "context_precision")
_METRIC_LABELS = {
    "hit_rate": "Hit Rate@5",
    "mrr": "MRR@5",
    "context_precision": "Context Precision@5",
    "faithfulness": "Faithfulness (judge)",
    "relevancy": "Relevancy (judge)",
}
#: #141: closed. `router_model` is a v1 routing sweep and is not a section.
REPORT_DECISIONS: tuple[str, ...] = (
    "embedding_combo",
    "hybrid_alpha",
    "retrieval_top_k",
    "reranker",
)
_MISSING = "No saved run for this decision yet."
_GUARDRAIL = (
    "Closed-world grounding is on in every saved run on this page. "
    "The result files have no on/off comparison, so this decision has no line."
)
_FOOTER = "Saved measurements only; the routing stack itself is not swept here."
_ADR_NOTE = (
    "The Lab sidebar quotes a separate ADR 0008 figure for this decision. "
    "That figure is not drawn here."
)
_DECISION_COPY = {
    "embedding_combo": ("Which embedding describes each movie", "hit rate"),
    "hybrid_alpha": ("How lexical and vector search are mixed", "hit rate"),
    "retrieval_top_k": ("How many movies are kept", "hit rate"),
    "reranker": ("Whether a reranker reorders them", "hit rate"),
}
_POINT_LABELS: dict[str, dict[str, str]] = {
    "hybrid_alpha": {
        "0.0": "Lexical only",
        "0.25": "25% dense",
        "0.5": "Half and half",
        "0.75": "75% dense",
        "1.0": "Dense only",
    },
    "retrieval_top_k": {"3": "Keep 3", "5": "Keep 5", "10": "Keep 10"},
    "reranker": {
        "off": "Off",
        "ms-marco-TinyBERT-L-2-v2": "TinyBERT",
        "ms-marco-MiniLM-L-12-v2": "MiniLM",
        "ce-esci-MiniLM-L12-v2": "ESCI MiniLM",
    },
    "embedding_combo": {
        "full_lfm_free": "LFM · full",
        "minimal_lfm_free": "LFM · minimal",
        "full_nemotron_free": "Nemotron · full",
        "minimal_nemotron_free": "Nemotron · minimal",
        "full_gemini_embedding_2": "Gemini · full",
        "minimal_gemini_embedding_2": "Gemini · minimal",
    },
}
_QUERY_TIER_LABELS = {
    "A_guardrails": "Guardrail",
    "B_retrieval": "Retrieval",
}


def _load_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _newest_by_sweep_value(runs: list[dict], knob: str) -> dict[str, dict]:
    """Newest run for each value of one sweep knob."""
    newest: dict[str, dict] = {}
    for run in sorted(runs, key=lambda item: item.get("timestamp", "")):
        sweep = run.get("sweep") or {}
        if sweep.get("knob") == knob:
            newest[str(sweep.get("value"))] = run
    return newest


def sweep_rows(runs: list[dict], knob: str) -> tuple[list[dict], list[str]]:
    """Chart rows for a knob sweep + the swept values missing a run. Pure.

    Rows come from runs stamped `sweep.knob == knob` (newest per value);
    missing values power the "run this sweep" hint (#54 grill D9).
    """
    newest_by_value = _newest_by_sweep_value(runs, knob)
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


def retrieval_chart_rows(runs: list[dict], knob: str) -> list[dict]:
    """One chart's rows: hit rate, MRR, and context precision, in sweep order.

    Faithfulness and relevancy stay off this chart. A missing score is omitted,
    not drawn as zero.
    """
    raw, _missing = sweep_rows(runs, knob)
    labels = _POINT_LABELS.get(knob, {})
    line_order = [_METRIC_LABELS[key] for key in _CHART_METRICS]
    by_value: dict[str, dict[str, float]] = {}
    for row in raw:
        if row["metric"] in line_order:
            by_value.setdefault(row["value"], {})[row["metric"]] = row["score"]
    rows = []
    for value, _config in sweep_configs(knob):
        scores = by_value.get(value)
        if not scores:
            continue
        option = labels.get(value, value)
        for line in line_order:
            score = scores.get(line)
            if score is None:
                continue
            rows.append({"option": option, "line": line, "score": round(score * 100, 1)})
    return rows


def hit_rate_points(runs: list[dict], knob: str) -> list[tuple[str, float]]:
    """Newest sweep point per value, hit rate only, in sweep order.

    Calls ``sweep_rows``. The chart draws MRR and context precision too;
    the swing sentence still uses hit rate alone.
    """
    rows, _missing = sweep_rows(runs, knob)
    scores = {
        row["value"]: row["score"] for row in rows if row["metric"] == _METRIC_LABELS["hit_rate"]
    }
    order = [label for label, _ in sweep_configs(knob)]
    return [(label, scores[label]) for label in order if label in scores]


def score_swing(scores: list[float]) -> float | None:
    """Best minus worst. None when fewer than two scores."""
    if len(scores) < 2:
        return None
    return max(scores) - min(scores)


def widest_decision(swings: dict[str, float]) -> list[str]:
    """Ids that share the largest swing, in insertion order."""
    if not swings:
        return []
    top = max(swings.values())
    return [key for key, gap in swings.items() if gap == top]


def saved_count_note(saved: int, current: int, noun: str) -> str | None:
    """Caption clause when the saved count differs from the golden file."""
    if saved == current:
        return None
    return f"Scored {saved} {noun}. The golden file now has {current}."


def percent_label(score: float) -> str:
    """One decimal percent. 0.226 -> '22.6%'."""
    return f"{score * 100:.1f}%"


def lead_sentence(
    entries: list[tuple[str, str, list[tuple[str, float]], float]],
) -> str | None:
    """The opening sentence. ``entries`` are (title, metric phrase, percent points, swing).

    Names every decision that shares the largest swing. None when no decision
    has two options.
    """
    if not entries:
        return None
    swings = {title: gap for title, _metric, _points, gap in entries}
    packed = {title: (metric, points) for title, metric, points, _gap in entries}
    names = widest_decision(swings)
    clauses = []
    for title in names:
        metric, points = packed[title]
        best = max(points, key=lambda point: point[1])
        worst = min(points, key=lambda point: point[1])
        clauses.append(
            f"{title} moved {metric} the furthest: "
            f"{best[1]:.1f}% on {best[0]}, {worst[1]:.1f}% on {worst[0]}, "
            f"{swings[title]:.1f} points"
        )
    if len(clauses) == 1:
        return clauses[0] + "."
    return "These decisions are tied for the largest swing. " + " ".join(
        clause + "." for clause in clauses
    )


def _percent_points(
    raw: list[tuple[str, float]], labels: dict[str, str]
) -> list[tuple[str, float]]:
    return [(labels.get(value, value), round(score * 100, 1)) for value, score in raw]


def _saved_on(runs: list[dict]) -> str:
    stamps = [run.get("timestamp", "") for run in runs if run.get("timestamp")]
    if not stamps:
        return ""
    try:
        moment = datetime.fromisoformat(max(stamps).replace("Z", "+00:00"))
    except ValueError:
        return max(stamps)[:10]
    return f"{moment.day} {moment.strftime('%b')} {moment.year}"


def _line_chart(points: list[tuple[str, float]]) -> None:
    frame = pd.DataFrame(
        {"option": [point[0] for point in points], "score": [point[1] for point in points]}
    )
    fig = px.line(frame, x="option", y="score", markers=True, text="score")
    fig.update_traces(textposition="top center", texttemplate="%{text:.1f}")
    fig.update_xaxes(categoryorder="array", categoryarray=frame["option"].tolist(), title="")
    fig.update_yaxes(range=[0, 100], title="")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=28, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch", config={"staticPlot": True})


_LINE_STYLE = {
    "Hit Rate@5": {"color": "#1f4e79", "dash": "solid", "symbol": "circle", "rank": 1},
    "MRR@5": {"color": "#e07a00", "dash": "dash", "symbol": "diamond", "rank": 2},
    "Context Precision@5": {"color": "#c0392b", "dash": "solid", "symbol": "square", "rank": 3},
}
_MRR_ON_PRECISION = (
    "MRR is the dashed line. It lands on context precision here: "
    "each scored query names one relevant movie, so the two scores are equal."
)


def mrr_matches_precision(rows: list[dict]) -> bool:
    """True when every option that has both scores has the same number for both."""
    by_option: dict[str, dict[str, float]] = {}
    for row in rows:
        by_option.setdefault(row["option"], {})[row["line"]] = row["score"]
    pairs = [
        scores
        for scores in by_option.values()
        if "MRR@5" in scores and "Context Precision@5" in scores
    ]
    return bool(pairs) and all(scores["MRR@5"] == scores["Context Precision@5"] for scores in pairs)


def _metric_chart(rows: list[dict]) -> None:
    """One chart, three lines: hit rate, MRR, context precision.

    MRR is drawn last, dashed, so it stays visible when it shares a point
    with context precision.
    """
    shared = {
        row["option"]
        for row in rows
        if row["line"] == "MRR@5"
        and any(
            other["option"] == row["option"]
            and other["line"] == "Context Precision@5"
            and other["score"] == row["score"]
            for other in rows
        )
    }
    labeled = []
    for row in rows:
        text = "" if row["line"] == "MRR@5" and row["option"] in shared else f"{row['score']:.1f}"
        labeled.append({**row, "text": text})
    draw_order = {"Hit Rate@5": 0, "Context Precision@5": 1, "MRR@5": 2}
    frame = pd.DataFrame(labeled)
    frame["_draw"] = frame["line"].map(draw_order)
    frame = frame.sort_values("_draw").drop(columns="_draw")
    fig = px.line(
        frame,
        x="option",
        y="score",
        color="line",
        markers=True,
        text="text",
        category_orders={"line": list(_LINE_STYLE)},
    )
    fig.update_traces(textposition="top center", texttemplate="%{text}")
    for trace in fig.data:
        style = _LINE_STYLE[trace.name]
        trace.update(
            line=dict(color=style["color"], dash=style["dash"], width=2.5),
            marker=dict(color=style["color"], symbol=style["symbol"], size=9),
            legendrank=style["rank"],
        )
    options = list(dict.fromkeys(row["option"] for row in rows))
    fig.update_xaxes(categoryorder="array", categoryarray=options, title="")
    fig.update_yaxes(range=[0, 100], title="")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=28, b=10), legend_title="")
    st.plotly_chart(fig, width="stretch", config={"staticPlot": True})


def _load_golden_sources():
    """Golden conversations and single-turn queries. None when a file will not load."""
    conversations = queries = None
    try:
        from src.evals.conversations import load_conversations

        conversations = load_conversations()
    except (OSError, ValueError):
        pass
    try:
        from src.evals.runner import load_dataset

        queries = load_dataset()
    except (OSError, ValueError):
        pass
    return conversations, queries


def _load_runs(results_dir: Path) -> list[dict]:
    if not results_dir.exists():
        return []
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            runs.append(_load_run(path))
        except (OSError, json.JSONDecodeError):
            continue
    return runs


def _newest_sweep_runs(runs: list[dict], knob: str) -> list[dict]:
    """Newest run per sweep value, in sweep order. Caption facts, not scores."""
    newest = _newest_by_sweep_value(runs, knob)
    order = [label for label, _ in sweep_configs(knob)]
    return [newest[label] for label in order if label in newest]


def _query_count_clause(saved_counts: set[int], current: int | None) -> str:
    counts = sorted(saved_counts)
    if len(counts) == 1:
        saved = counts[0]
        if current is None:
            return f"{saved} queries."
        note = saved_count_note(saved, current, "queries")
        return note if note else f"{saved} queries."
    listed = ", ".join(str(count) for count in counts)
    if current is None:
        return f"Scored {listed} queries across the options."
    return f"Scored {listed} queries across the options. The golden file now has {current}."


def _retrieval_caption(runs: list[dict], knob: str, n_queries: int | None) -> str:
    newest_runs = _newest_sweep_runs(runs, knob)
    saved_counts = {
        int(run["n_queries"]) for run in newest_runs if run.get("n_queries") is not None
    }
    parts = []
    if saved_counts:
        parts.append(_query_count_clause(saved_counts, n_queries))
    saved_on = _saved_on(newest_runs)
    if saved_on:
        parts.append(f"Saved {saved_on}.")
    if knob == "embedding_combo":
        parts.append(_ADR_NOTE)
    return " ".join(parts)


def single_turn_score_rows(runs: list[dict]) -> list[dict]:
    """One row per retrieval option, with every single-turn score on that saved run.

    Hit rate, MRR, and context precision are the lines on the retrieval chart.
    The two judge scores stay in this table.
    """
    rows = []
    for knob in REPORT_DECISIONS:
        if knob == "models":
            continue
        title, _metric = _DECISION_COPY[knob]
        labels = _POINT_LABELS.get(knob, {})
        raw, _missing = sweep_rows(runs, knob)
        by_value: dict[str, dict[str, float]] = {}
        for row in raw:
            by_value.setdefault(row["value"], {})[row["metric"]] = row["score"]
        for value, _config in sweep_configs(knob):
            scores = by_value.get(value)
            if not scores:
                continue
            record: dict[str, str | float] = {
                "decision": title,
                "option": labels.get(value, value),
            }
            for metric, label in _METRIC_LABELS.items():
                score = scores.get(label)
                if score is not None:
                    record[label] = round(score * 100, 1)
            rows.append(record)
    return rows


def golden_query_rows(queries: list[dict]) -> list[dict]:
    """One row per single-turn golden query."""
    rows = []
    for query in queries:
        ids = query.get("relevant_movie_ids") or []
        rows.append(
            {
                "id": query.get("id", ""),
                "kind": _QUERY_TIER_LABELS.get(query.get("tier", ""), query.get("tier", "")),
                "query": query.get("query", ""),
                "expected intent": query.get("expected_intent", ""),
                "expected step": query.get("expected_path", ""),
                "relevant movie ids": ", ".join(str(movie_id) for movie_id in ids),
                "notes": query.get("notes") or "",
            }
        )
    return rows


def _show_table(rows: list[dict]) -> None:
    if not rows:
        st.caption(_MISSING)
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_evals(session=None, results_dir: Path = RESULTS_DIR) -> None:
    """Static retrieval-decision report on the v1 fork (#156).

    The Routing Stack v2 conversation report died with the v2 fork; what
    stays is stack-neutral: the retrieval sweep decisions and the golden
    single-turn queries. ``session`` is accepted so ``app.py`` can keep
    passing the Lab session. This view does not read it.
    """
    del session
    st.header("Evals")
    st.caption("Static report. Saved measurements only.")
    runs = _load_runs(results_dir)
    _conversations, queries = _load_golden_sources()
    n_queries = len(queries) if queries is not None else None

    sections: list[tuple[str, str, str, list[tuple[str, float]], str, float | None]] = []
    for knob in REPORT_DECISIONS:
        title, metric = _DECISION_COPY[knob]
        points = _percent_points(hit_rate_points(runs, knob), _POINT_LABELS.get(knob, {}))
        gap = score_swing([point[1] for point in points])
        sections.append(
            (
                knob,
                title,
                metric,
                points,
                _retrieval_caption(runs, knob, n_queries) if points else "",
                None if gap is None else round(gap, 1),
            )
        )

    swings = {
        decision_id: gap
        for decision_id, _title, _metric, _points, _caption, gap in sections
        if gap is not None
    }
    marked = set(widest_decision(swings))
    lead = lead_sentence(
        [
            (title, metric, points, gap)
            for _decision_id, title, metric, points, _caption, gap in sections
            if gap is not None
        ]
    )
    if lead:
        st.markdown(lead)

    for decision_id, title, _metric, points, caption, _gap in sections:
        with st.container(border=True):
            if decision_id in marked:
                st.markdown("**Largest swing**")
            st.subheader(title)
            if not points:
                st.caption(_MISSING)
                continue
            chart_rows = retrieval_chart_rows(runs, decision_id)
            _metric_chart(chart_rows)
            if mrr_matches_precision(chart_rows):
                st.caption(_MRR_ON_PRECISION)
            if caption:
                st.caption(caption)

    with st.container(border=True):
        st.subheader("Closed-world grounding")
        st.caption(_GUARDRAIL)

    with st.expander("Single-turn scores and golden queries"):
        st.caption(
            "Hit rate, MRR, and context precision are the three lines on each retrieval chart. "
            "These rows also keep the two judge scores from those same saved runs."
        )
        _show_table(single_turn_score_rows(runs))
        st.caption("The single-turn golden queries those runs are scored against.")
        _show_table(golden_query_rows(queries or []))
    st.caption(_FOOTER)
