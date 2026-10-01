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

from src.domain.config import ExperimentConfig
from src.evals.runner import sweep_configs

RESULTS_DIR = Path("evals/results")
_METRICS = ["hit_rate", "mrr", "context_precision", "faithfulness", "relevancy"]
_METRIC_LABELS = {
    "hit_rate": "Hit Rate@5",
    "mrr": "MRR@5",
    "context_precision": "Context Precision@5",
    "faithfulness": "Faithfulness (judge)",
    "relevancy": "Relevancy (judge)",
}
#: #141: closed. `router_model` is a v1 routing sweep and is not a section.
REPORT_DECISIONS: tuple[str, ...] = (
    "models",
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
_FOOTER = "A second routing stack still exists and is left off this page."
_ADR_NOTE = (
    "The Lab sidebar quotes a separate ADR 0008 figure for this decision. "
    "That figure is not drawn here."
)
_MODEL_ORDER = (
    "google/gemma-4-31b-it",
    "google/gemini-3.5-flash-lite",
    "google/gemini-3.8-flash",
    "glm-5.3-flash",
)
_MODEL_LABELS = {
    "google/gemma-4-31b-it": "Gemma 4 31B",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "google/gemini-3.8-flash": "Gemini 3.8 Flash",
    "glm-5.3-flash": "GLM 5.3 Flash",
}
_DECISION_COPY = {
    "models": ("Which model wrote the reply", "how often the reply stayed on the retrieved movies"),
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
_UNDERSTOOD = "Understood the request"
_STEP = "Took the expected step"
_FILTERS = "Kept the expected filters"
_REPLY = "Reply stayed on the movies"
#: Run-level key, per-conversation key, label. Reply score has no per-conversation line.
_SCORE_LINES = (
    ("intent_accuracy", "intent_accuracy", _UNDERSTOOD),
    ("path_accuracy", "path_accuracy", _STEP),
    ("constraint_fidelity", "fidelity", _FILTERS),
    ("faithfulness", None, _REPLY),
)


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


def newest_full_v2_by_model(
    runs: list[dict], n_conversations: int, n_turns: int
) -> dict[str, dict]:
    """Newest Routing Stack v2 conversation run per understand-model.

    A run counts only when ``n_conversations`` and ``n_turns`` match the
    golden conversations. A shorter pilot cannot replace a full run.
    """
    newest: dict[str, dict] = {}
    for run in runs:
        if run.get("mode") != "conversation" or run.get("routing_stack") != "v2":
            continue
        if run.get("n_conversations") != n_conversations or run.get("n_turns") != n_turns:
            continue
        model = (run.get("config_snapshot") or {}).get("v2_router_model")
        if not model:
            continue
        previous = newest.get(model)
        if previous is None or run.get("timestamp", "") > previous.get("timestamp", ""):
            newest[model] = run
    return newest


def hit_rate_points(runs: list[dict], knob: str) -> list[tuple[str, float]]:
    """Newest sweep point per value, hit rate only, in sweep order.

    Calls ``sweep_rows``. MRR and the other metrics on the same run are ignored.
    """
    rows, _missing = sweep_rows(runs, knob)
    scores = {
        row["value"]: row["score"] for row in rows if row["metric"] == _METRIC_LABELS["hit_rate"]
    }
    order = [label for label, _ in sweep_configs(knob)]
    return [(label, scores[label]) for label in order if label in scores]


def reply_points(by_model: dict[str, dict]) -> list[tuple[str, float]]:
    """(understand-model id, faithfulness) for fleet runs that have a reply score.

    Known models stay in report order. Any other id follows, sorted.
    A missing faithfulness is omitted, not drawn as zero.
    """

    def sort_key(model_id: str) -> tuple:
        if model_id in _MODEL_ORDER:
            return (0, _MODEL_ORDER.index(model_id))
        return (1, model_id)

    points = []
    for model_id in sorted(by_model, key=sort_key):
        score = by_model[model_id].get("faithfulness")
        if score is None:
            continue
        points.append((model_id, score))
    return points


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


def _conversation_chart(run: dict) -> None:
    rows = []
    ids = []
    for convo in run.get("per_conversation") or []:
        ids.append(convo.get("id", ""))
        for _run_key, convo_key, label in _SCORE_LINES:
            if convo_key is None:
                continue
            score = convo.get(convo_key)
            rows.append(
                {
                    "conversation": convo.get("id", ""),
                    "line": label,
                    "score": None if score is None else round(score * 100, 1),
                }
            )
    if not rows:
        st.caption(_MISSING)
        return
    frame = pd.DataFrame(rows)
    fig = px.line(frame, x="conversation", y="score", color="line", markers=True)
    fig.update_xaxes(categoryorder="array", categoryarray=ids, title="")
    fig.update_yaxes(range=[0, 100], title="")
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=28, b=10),
        legend_title="",
    )
    st.plotly_chart(fig, width="stretch", config={"staticPlot": True})


def _golden_sizes() -> tuple[int | None, int | None, int | None]:
    n_conversations = n_turns = n_queries = None
    try:
        from src.evals.conversations import load_conversations

        conversations = load_conversations()
        n_conversations = len(conversations.conversations)
        n_turns = sum(len(convo.turns) for convo in conversations.conversations)
    except (OSError, ValueError):
        pass
    try:
        from src.evals.runner import load_dataset

        n_queries = len(load_dataset())
    except (OSError, ValueError):
        pass
    return n_conversations, n_turns, n_queries


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


def _model_caption(runs: list[dict]) -> str:
    parts = ["This line is a judge score on the turns that wrote a reply."]
    if runs:
        run = runs[0]
        tail = f"{run.get('n_conversations', 0)} conversations, {run.get('n_turns', 0)} turns"
        saved_on = _saved_on(runs)
        if saved_on:
            tail += f", saved {saved_on}"
        parts.append(tail + ".")
    return " ".join(parts)


def render_evals(session=None, results_dir: Path = RESULTS_DIR) -> None:
    """Static Routing Stack v2 report (#142).

    ``session`` is accepted so ``app.py`` can keep passing the Lab session.
    This view does not read it.
    """
    del session
    st.header("Evals")
    st.caption("Static report. Routing Stack v2. Saved measurements only.")
    runs = _load_runs(results_dir)
    n_conversations, n_turns, n_queries = _golden_sizes()
    if n_conversations is None or n_turns is None:
        by_model: dict[str, dict] = {}
    else:
        by_model = newest_full_v2_by_model(runs, n_conversations, n_turns)
    model_points = _percent_points(reply_points(by_model), _MODEL_LABELS)

    sections: list[tuple[str, str, str, list[tuple[str, float]], str, float | None]] = []
    title, metric = _DECISION_COPY["models"]
    model_gap = score_swing([point[1] for point in model_points])
    sections.append(
        (
            "models",
            title,
            metric,
            model_points,
            _model_caption(list(by_model.values())) if model_points else "",
            None if model_gap is None else round(model_gap, 1),
        )
    )
    for knob in REPORT_DECISIONS:
        if knob == "models":
            continue
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
            _line_chart(points)
            if caption:
                st.caption(caption)

    with st.container(border=True):
        st.subheader("Closed-world grounding")
        st.caption(_GUARDRAIL)

    headline_model = ExperimentConfig().v2_router_model
    headline = by_model.get(headline_model)
    st.subheader("Golden conversations")
    if headline is None:
        st.caption(_MISSING)
    else:
        model_label = _MODEL_LABELS.get(headline_model, headline_model)
        st.caption(
            f"{model_label}. The four scores count turns. "
            f"{headline.get('n_conversations', 0)} conversations, "
            f"{headline.get('n_turns', 0)} turns, saved {_saved_on([headline])}."
        )
        with st.container(horizontal=True):
            for run_key, _convo_key, label in _SCORE_LINES:
                score = headline.get(run_key)
                st.metric(
                    label,
                    percent_label(score) if score is not None else "n/a",
                    border=True,
                )
        st.caption("This chart is one point per conversation.")
        _conversation_chart(headline)
    st.caption(_FOOTER)
