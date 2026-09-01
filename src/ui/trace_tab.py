"""Trace inspector tab (issue #7): node waterfall + payload table.

Local-first: renders the in-memory tracer ring (always available). When
Langfuse cloud keys are configured, links out to the Langfuse UI for the
full hosted view — the local waterfall is the offline fallback, not a
replacement.
"""

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st


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


def render_traces(session) -> None:
    st.subheader("🔍 Trace Inspector")
    cloud_keys = os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    st.caption(
        "Node-level execution waterfall from the local tracer ring (always on). "
        + ("Full hosted traces: [Langfuse dashboard](https://cloud.langfuse.com)."
           if cloud_keys else
           "Set LANGFUSE_* keys for full cloud tracing (local mode is always on).")
    )

    traces = session.tracer.traces()
    render_waterfall(traces)

    st.markdown(f"**Span log** ({len(traces)} spans this session, oldest first)")
    render_trace_table(traces)
