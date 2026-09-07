"""Trace inspector tab (issues #7, #31): full-fidelity fetch-back view.

Cloud mode (Langfuse keys set): pick a turn, fetch its observations back
from Langfuse (#31, O1 architecture) and render the nested execution tree —
prompts, completions, model params, tokens, cost, real durations, scores.
No-keys mode: the original local tracer-ring waterfall remains, unchanged.
"""

import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from src.observability.trace_fetch import ObservationNode, TraceTree, fetch_trace_tree


def _waterfall_frame(traces: list[dict]) -> pd.DataFrame:
    """Converts the tracer ring into a Plotly timeline frame (pure, testable)."""
    rows = []
    for i, trace in enumerate(traces):
        ts = datetime.fromisoformat(trace["timestamp"])
        finish = (
            datetime.fromisoformat(traces[i + 1]["timestamp"])
            if i + 1 < len(traces)
            else ts
        )
        rows.append({
            "node": trace["node"],
            "start": ts,
            "finish": max(finish, ts),
            "detail": ", ".join(f"{k}={v}" for k, v in trace["payload"].items() if v != []),
        })
    return pd.DataFrame(rows)


def render_waterfall(traces: list[dict]) -> None:
    if not traces:
        st.info("No traces yet — send a message in the Chat tab first.")
        return
    frame = _waterfall_frame(traces)
    frame["turn"] = 1
    fig = px.timeline(
        frame, x_start="start", x_end="finish", y="turn", color="node",
        hover_data={"detail": True, "turn": False},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_yaxes(visible=False)
    fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10),
                      showlegend=True, legend_title="node")
    st.plotly_chart(fig, use_container_width=True)


def routing_counters(traces: list[dict]) -> dict[str, int]:
    """Session routing-health counters from the local ring (pure, testable).

    #12/#13/#29 observability: how often the router degraded (fallbacks),
    re-routed (attempt > 1), or contradicted itself (requires_rag mismatch).
    """
    route_payloads = [t["payload"] for t in traces if t["node"] == "route"]
    return {
        "fallbacks": sum(1 for p in route_payloads if p.get("is_fallback")),
        "reroutes": sum(1 for p in route_payloads if p.get("attempt", 1) > 1),
        "mismatches": sum(
            1 for p in route_payloads if p.get("requires_rag_mismatch")
        ),
    }


def render_routing_counters(traces: list[dict]) -> None:
    counters = routing_counters(traces)
    cols = st.columns(3)
    cols[0].metric("Router fallbacks", counters["fallbacks"])
    cols[1].metric("Re-routes", counters["reroutes"])
    cols[2].metric("requires_rag mismatches", counters["mismatches"])


def render_trace_table(traces: list[dict]) -> None:
    rows = [
        {
            "node": t["node"],
            "time": datetime.fromisoformat(t["timestamp"]).strftime("%H:%M:%S.%f")[:-5],
            **t["payload"],
        }
        for t in traces
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# --- cloud fetch-back inspector (#31) ---


def _turn_options(session) -> list[tuple[str, str]]:
    """(trace_id, label) per turn, newest first, from the session turn_log."""
    options = []
    for i, row in enumerate(session.turn_log):
        if row.get("trace_id"):
            options.append((row["trace_id"], f"Turn {i + 1}: {row['query'][:60]}"))
    return list(reversed(options))


def _cached_tree(session, trace_id: str, refresh: bool) -> TraceTree | None:
    """Session-state cache by trace_id (D9); Refresh bypasses and refills."""
    cache: dict[str, TraceTree] = getattr(session, "_trace_cache", None) or {}
    if not hasattr(session, "_trace_cache"):
        session._trace_cache = cache
    if refresh or trace_id not in cache:
        tree = fetch_trace_tree(trace_id)
        if tree is not None:
            cache[trace_id] = tree
    return cache.get(trace_id)


def _flatten(nodes: list[ObservationNode], depth: int = 0):
    for node in nodes:
        yield node, depth
        yield from _flatten(node.children, depth + 1)


def _cloud_waterfall_frame(tree: TraceTree) -> pd.DataFrame:
    """Real start/end durations from fetched observations (fixes #31/F4)."""
    rows = [
        {
            "node": node.name or node.type,
            "start": node.start_time,
            "finish": node.end_time or node.start_time,
            "detail": f"{node.type} · {node.duration_ms:.0f} ms" if node.duration_ms is not None else node.type,
        }
        for node, _ in _flatten(tree.roots)
        if node.start_time
    ]
    return pd.DataFrame(rows)


def _render_io(label: str, payload) -> None:
    """Chat messages render as role blocks; everything else as JSON."""
    if payload in (None, "", [], {}):
        return
    st.markdown(f"**{label}**")
    messages = payload if isinstance(payload, list) else None
    if messages and all(isinstance(m, dict) and ("role" in m or "content" in m) for m in messages):
        for m in messages:
            role = m.get("role", m.get("type", "message"))
            content = m.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            st.markdown(f"`{role}`")
            st.code(content, language=None, wrap_lines=True)
        return
    if isinstance(payload, str):
        st.code(payload, language=None, wrap_lines=True)
    else:
        st.json(payload, expanded=False)


def _render_node_detail(node: ObservationNode) -> None:
    """Shared observation body: metadata caption + role-block I/O (#31, #35)."""
    meta = {"type": node.type, "model": node.model, "cost_usd": node.cost, **node.usage}
    st.caption(" · ".join(f"{k}={v}" for k, v in meta.items() if v not in (None, "")))
    _render_io("Input", node.input)
    _render_io("Output", node.output)


def render_observation_node(node: ObservationNode, depth: int = 0) -> None:
    """One observation as an indented expander (Streamlit forbids nesting)."""
    icon = "✦" if node.type == "GENERATION" else "•"
    bits = [f"{'│ ' * depth}{icon} {node.name or node.type}"]
    if node.duration_ms is not None:
        bits.append(f"{node.duration_ms:.0f} ms")
    if node.usage.get("total"):
        bits.append(f"{node.usage['total']} tok")
    if node.model:
        bits.append(node.model)
    with st.expander(" · ".join(bits)):
        _render_node_detail(node)


# --- Langfuse-style observations table (#35) ---


def _preview(payload, limit: int = 80) -> str:
    """One-line truncated preview of an observation payload (pure)."""
    if payload in (None, "", [], {}):
        return ""
    text = payload if isinstance(payload, str) else json.dumps(
        payload, ensure_ascii=False, default=str
    )
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def observation_rows(
    trees: dict[str, TraceTree], turn_labels: dict[str, str]
) -> list[dict]:
    """Flattens fetched trees into table rows, newest first (pure).

    Each row carries a ``node`` ref for the detail pane and a lowercase
    ``search_blob`` over name + full I/O for text filtering.
    """
    rows = []
    for trace_id, tree in trees.items():
        for node, _ in _flatten(tree.roots):
            rows.append({
                "start": node.start_time,
                "type": node.type,
                "name": node.name or node.type,
                "turn": turn_labels.get(trace_id, trace_id[:8]),
                "duration_ms": round(node.duration_ms) if node.duration_ms is not None else None,
                "tokens": node.usage.get("total"),
                "cost_usd": node.cost,
                "input": _preview(node.input),
                "output": _preview(node.output),
                "search_blob": " ".join(
                    str(v) for v in (node.name, node.input, node.output)
                ).lower(),
                "node": node,
            })
    rows.sort(key=lambda r: (r["start"] is None, r["start"]), reverse=True)
    return rows


def name_counts(rows: list[dict]) -> dict[str, int]:
    """Observation-name histogram for the Name filter (pure)."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["name"]] = counts.get(row["name"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def apply_filters(
    rows: list[dict], type_filter: str, names: list[str], search: str
) -> list[dict]:
    """Type / name / text-search filtering over table rows (pure)."""
    needle = search.strip().lower()
    return [
        r for r in rows
        if (type_filter == "All" or r["type"] == type_filter)
        and (not names or r["name"] in names)
        and (not needle or needle in r["search_blob"])
    ]


def _fetched_trees(session, refresh: bool) -> tuple[dict[str, TraceTree], dict[str, str], list[str]]:
    """All fetchable session traces via the #31 cache; returns pending labels too."""
    trees: dict[str, TraceTree] = {}
    labels: dict[str, str] = {}
    pending: list[str] = []
    for trace_id, label in _turn_options(session):
        labels[trace_id] = label
        tree = _cached_tree(session, trace_id, refresh)
        if tree is None:
            pending.append(label)
        else:
            trees[trace_id] = tree
    return trees, labels, pending


def render_observations_table(session) -> None:
    refresh = st.button("Refresh", key="refresh-table")
    trees, labels, pending = _fetched_trees(session, refresh)
    if pending:
        st.caption(
            "Still ingesting (15–30 s): " + " · ".join(pending)
        )
    rows = observation_rows(trees, labels)
    if not rows:
        st.info("No fetched observations yet — chat first, then Refresh.")
        return

    counts = name_counts(rows)
    fcols = st.columns([1, 2, 2])
    type_filter = fcols[0].segmented_control(
        "Type", ["All", "GENERATION", "SPAN"], default="All", key="obs-type"
    ) or "All"
    names = fcols[1].multiselect(
        "Name", list(counts),
        format_func=lambda n: f"{n} ({counts[n]})", key="obs-names",
    )
    search = fcols[2].text_input(
        "Search input/output", key="obs-search", placeholder="e.g. retrieved_movies"
    )
    filtered = apply_filters(rows, type_filter, names, search)

    timed = [r for r in filtered if r["start"] is not None]
    if timed:
        hist = pd.DataFrame({"start": [r["start"] for r in timed]})
        fig = px.histogram(hist, x="start", nbins=40)
        fig.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="count", xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    frame = pd.DataFrame([
        {
            "Start": r["start"].strftime("%H:%M:%S.%f")[:-3] if r["start"] else "",
            "Type": "✦ GEN" if r["type"] == "GENERATION" else r["type"],
            "Name": r["name"],
            "Turn": r["turn"],
            "ms": r["duration_ms"],
            "Tokens": r["tokens"],
            "Cost $": r["cost_usd"],
            "Input": r["input"],
            "Output": r["output"],
        }
        for r in filtered
    ])
    event = st.dataframe(
        frame, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row", key="obs-table",
    )
    selected = event.selection.rows if event and event.selection else []
    if selected:
        row = filtered[selected[0]]
        st.markdown(f"#### {row['name']} — {row['turn']}")
        _render_node_detail(row["node"])
    else:
        st.caption("Select a row to inspect full prompts and outputs.")


def render_cloud_trace(session, trace_id: str) -> None:
    refresh = st.button("Refresh", key=f"refresh-{trace_id}")
    tree = _cached_tree(session, trace_id, refresh)
    if tree is None:
        st.info(
            "Trace not available yet — Langfuse ingestion takes 15–30 s. "
            "Hit Refresh in a moment."
        )
        return
    frame = _cloud_waterfall_frame(tree)
    if not frame.empty:
        frame["turn"] = 1
        fig = px.timeline(
            frame, x_start="start", x_end="finish", y="turn", color="node",
            hover_data={"detail": True, "turn": False},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_yaxes(visible=False)
        fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10),
                          showlegend=True, legend_title="node")
        st.plotly_chart(fig, use_container_width=True)
    for score in tree.scores:
        st.caption(f"Score **{score['name']}** = {score['value']}"
                   + (f" — {score['comment']}" if score.get("comment") else ""))
    for node, depth in _flatten(tree.roots):
        render_observation_node(node, depth)


def render_traces(session) -> None:
    st.header("Traces")
    cloud = session.tracer.cloud_enabled and bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    )
    if cloud:
        st.caption(
            "Full-fidelity traces fetched back from Langfuse (#31): prompts, "
            "completions, tokens, cost, real durations. "
            "[Hosted dashboard](https://cloud.langfuse.com)."
        )
        options = _turn_options(session)
        if not options:
            st.info("No turns yet — send a message in the Chat tab first.")
            return
        view = st.segmented_control(
            "View", ["Table", "Tree"], default="Table", key="traces-view"
        ) or "Table"
        if view == "Table":
            render_observations_table(session)
            return
        labels = {tid: label for tid, label in options}
        trace_id = st.selectbox(
            "Turn", [tid for tid, _ in options],
            format_func=lambda tid: labels[tid],
        )
        render_cloud_trace(session, trace_id)
        return

    st.caption(
        "Node-level execution waterfall from the local tracer ring (always on). "
        "Set LANGFUSE_* keys for the full-fidelity cloud inspector."
    )
    traces = session.tracer.traces()
    render_routing_counters(traces)
    render_waterfall(traces)
    st.markdown(f"**Span log** ({len(traces)} spans this session, oldest first)")
    render_trace_table(traces)
