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

from src.domain.config import ExperimentConfig, PresetType
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
_PRESET_TITLES = {
    "production": "Production",
    "fast_budget": "Fast Budget",
    "naive_baseline": "Naive Baseline",
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
_CONVO_LINES = (
    ("intent_accuracy", "Understood the request"),
    ("path_accuracy", "Took the expected step"),
    ("fidelity", "Kept the expected filters"),
)


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
        flags.append(f"dataset {stamped} predates the current set ({current_dataset_version})")
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


def v2_fleet_by_model(runs: list[dict], n_conversations: int, n_turns: int) -> dict[str, dict]:
    """Newest full-set Routing Stack v2 conversation run per understand-model.

    A run counts only when it covered the golden conversations
    (``n_conversations`` and ``n_turns``). A shorter pilot cannot replace it.
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


def headline_run(by_model: dict[str, dict], model_id: str) -> dict | None:
    """Fleet run for the Experiment Config understand-model default, or None."""
    return by_model.get(model_id)


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


def lead_sentence(entries: list[tuple[str, str, list[tuple[str, float]]]]) -> str | None:
    """The opening sentence. ``entries`` are (title, metric phrase, percent points).

    Names every decision that shares the largest swing. None when no decision
    has two options.
    """
    swings: dict[str, float] = {}
    packed: dict[str, tuple[str, list[tuple[str, float]]]] = {}
    for title, metric, points in entries:
        gap = score_swing([point[1] for point in points])
        if gap is None:
            continue
        swings[title] = round(gap, 1)
        packed[title] = (metric, points)
    names = widest_decision(swings)
    if not names:
        return None
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


def metric_span(runs: list[dict], key: str) -> tuple[float, float] | None:
    """Min and max of one 0–1 metric across runs, ignoring missing values."""
    values = [run[key] for run in runs if run.get(key) is not None]
    if not values:
        return None
    return min(values), max(values)


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
        for key, label in _CONVO_LINES:
            score = convo.get(key)
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


def _span_text(runs: list[dict], key: str) -> str:
    span = metric_span(runs, key)
    if span is None:
        return "n/a"
    lo, hi = round(span[0] * 100, 1), round(span[1] * 100, 1)
    if lo == hi:
        return f"{lo:.1f}%"
    return f"{lo:.1f}–{hi:.1f}%"


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


def _load_runs(results_dir: Path) -> tuple[list[dict], int]:
    if not results_dir.exists():
        return [], 0
    runs = []
    failed = 0
    for path in sorted(results_dir.glob("*.json")):
        try:
            runs.append(_load_run(path))
        except (OSError, json.JSONDecodeError):
            failed += 1
    return runs, failed


def _newest_sweep_runs(runs: list[dict], knob: str) -> list[dict]:
    """Newest run per sweep value, in sweep order. Caption facts, not scores."""
    newest: dict[str, dict] = {}
    for run in sorted(runs, key=lambda item: item.get("timestamp", "")):
        sweep = run.get("sweep") or {}
        if sweep.get("knob") == knob:
            newest[str(sweep.get("value"))] = run
    order = [label for label, _ in sweep_configs(knob)]
    return [newest[label] for label in order if label in newest]


def _retrieval_caption(runs: list[dict], knob: str, n_queries: int | None) -> str:
    winners = _newest_sweep_runs(runs, knob)
    saved_counts = {run.get("n_queries") for run in winners if run.get("n_queries") is not None}
    parts = []
    if len(saved_counts) == 1:
        saved = int(next(iter(saved_counts)))
        note = saved_count_note(saved, n_queries, "queries") if n_queries is not None else None
        parts.append(note if note else f"{saved} queries.")
    elif saved_counts:
        parts.append("Query counts differ across the options.")
    saved_on = _saved_on(winners)
    if saved_on:
        parts.append(f"Saved {saved_on}.")
    if knob == "embedding_combo":
        parts.append(_ADR_NOTE)
    return " ".join(parts)


def _model_caption(fleet: list[dict]) -> str:
    understood = _span_text(fleet, "intent_accuracy")
    step = _span_text(fleet, "path_accuracy")
    filters = _span_text(fleet, "constraint_fidelity")
    parts = [
        f"Understood the request {understood}",
        f"Took the expected step {step}",
        f"Kept the expected filters {filters}",
        "This line is a judge score on the turns that wrote a reply",
    ]
    if fleet:
        run = fleet[0]
        tail = f"{run.get('n_conversations', 0)} conversations, {run.get('n_turns', 0)} turns"
        saved_on = _saved_on(fleet)
        if saved_on:
            tail += f", saved {saved_on}"
        parts.append(tail)
    return ". ".join(parts) + "."


def render_evals(session=None, results_dir: Path = RESULTS_DIR) -> None:
    """Static Routing Stack v2 report (#142).

    ``session`` is accepted so ``app.py`` can keep passing the Lab session.
    This view does not read it.
    """
    del session
    st.header("Evals")
    st.caption("Static report. Routing Stack v2. Saved measurements only.")
    runs, failed = _load_runs(results_dir)
    if failed:
        st.caption(f"{failed} saved file(s) could not be read.")
    n_conversations, n_turns, n_queries = _golden_sizes()
    if n_conversations is None or n_turns is None:
        by_model: dict[str, dict] = {}
    else:
        by_model = v2_fleet_by_model(runs, n_conversations, n_turns)
    fleet = list(by_model.values())
    model_raw = reply_points(by_model)
    model_points = _percent_points(model_raw, _MODEL_LABELS)

    entries: list[tuple[str, str, list[tuple[str, float]]]] = []
    sections: list[tuple[str, list[tuple[str, float]], str]] = []
    title, metric = _DECISION_COPY["models"]
    entries.append((title, metric, model_points))
    sections.append(("models", model_points, _model_caption(fleet) if model_points else ""))
    for knob in REPORT_DECISIONS:
        if knob == "models":
            continue
        title, metric = _DECISION_COPY[knob]
        points = _percent_points(hit_rate_points(runs, knob), _POINT_LABELS.get(knob, {}))
        entries.append((title, metric, points))
        sections.append((knob, points, _retrieval_caption(runs, knob, n_queries) if points else ""))

    swings: dict[str, float] = {}
    for decision_id, points, _caption in sections:
        gap = score_swing([point[1] for point in points])
        if gap is not None:
            swings[decision_id] = round(gap, 1)
    marked = set(widest_decision(swings))
    lead = lead_sentence(entries)
    if lead:
        st.markdown(lead)
    else:
        st.caption(_MISSING)

    for decision_id, points, caption in sections:
        title = _DECISION_COPY[decision_id][0]
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
    headline = headline_run(by_model, headline_model)
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
            for key, label in (
                ("intent_accuracy", "Understood the request"),
                ("path_accuracy", "Took the expected step"),
                ("constraint_fidelity", "Kept the expected filters"),
                ("faithfulness", "Reply stayed on the movies"),
            ):
                score = headline.get(key)
                st.metric(
                    label,
                    percent_label(score) if score is not None else "n/a",
                    border=True,
                )
        st.caption(
            "The chart is one point per conversation. A missing score leaves a gap in that line."
        )
        _conversation_chart(headline)
    st.caption(_FOOTER)
