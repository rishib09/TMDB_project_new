"""
prototypes/observability_trace_inspector_prototype.py

Interactive prototype demonstrating the Dual-Mode Observability Layer:
1. Native Langfuse telemetry emission (if credentials present)
2. In-App Langfuse-Style Interactive Trace Tree & Span Waterfall Inspector for Streamlit.
"""

from __future__ import annotations
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
import streamlit as st

# ==============================================================================
# 1. OBSERVABILITY DATA STRUCTURES (Spans & Traces)
# ==============================================================================

SpanType = Literal["router", "retrieval", "rerank", "llm_synthesis", "guardrail", "tool"]

@dataclass
class SpanRecord:
    span_id: str
    parent_span_id: Optional[str]
    name: str
    span_type: SpanType
    start_time: float
    end_time: float
    latency_ms: float
    status: Literal["success", "error"]
    model: Optional[str] = None
    input_payload: Dict[str, Any] = field(default_factory=dict)
    output_payload: Dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    error_message: Optional[str] = None


@dataclass
class TraceRecord:
    trace_id: str
    name: str
    user_query: str
    agent_response: str
    intent: str
    timestamp: str
    total_latency_ms: float
    total_tokens: int
    total_cost_usd: float
    spans: List[SpanRecord] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. DUAL-MODE TELEMETRY MANAGER (Langfuse + Local Memory Store)
# ==============================================================================

class DualModeObservabilityManager:
    """
    Unified telemetry manager that logs to local session state for in-app UI
    and mirrors to Langfuse Cloud/self-hosted if API keys are configured.
    """
    def __init__(self):
        self.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.is_langfuse_enabled = bool(self.langfuse_public_key and self.langfuse_secret_key)
        self.langfuse_client = None

        if self.is_langfuse_enabled:
            try:
                from langfuse import Langfuse
                self.langfuse_client = Langfuse(
                    public_key=self.langfuse_public_key,
                    secret_key=self.langfuse_secret_key,
                    host=self.langfuse_host,
                )
            except Exception as e:
                self.is_langfuse_enabled = False

    def emit_trace(self, trace: TraceRecord):
        """Stores trace in local history and asynchronously pushes to Langfuse if enabled."""
        if "trace_history" not in st.session_state:
            st.session_state["trace_history"] = []
        st.session_state["trace_history"].insert(0, trace)

        if self.is_langfuse_enabled and self.langfuse_client:
            try:
                lf_trace = self.langfuse_client.trace(
                    id=trace.trace_id,
                    name=trace.name,
                    input={"query": trace.user_query},
                    output={"response": trace.agent_response},
                    tags=trace.tags + [f"intent:{trace.intent}"],
                    metadata=trace.metadata,
                )
                for s in trace.spans:
                    lf_trace.span(
                        id=s.span_id,
                        name=s.name,
                        start_time=datetime.fromtimestamp(s.start_time),
                        end_time=datetime.fromtimestamp(s.end_time),
                        input=s.input_payload,
                        output=s.output_payload,
                        level="DEFAULT" if s.status == "success" else "ERROR",
                        status_message=s.error_message,
                    )
                self.langfuse_client.flush()
            except Exception:
                pass


# ==============================================================================
# 3. MOCK TRACE GENERATOR FOR PROTOTYPE DEMONSTRATION
# ==============================================================================

def generate_mock_traces() -> List[TraceRecord]:
    """Creates realistic mock traces for SEMANTIC_SEARCH, SUPERLATIVE_RANKING, and GREETING."""
    t0 = time.time()

    # Trace 1: Semantic Search Query
    s1_1 = SpanRecord(
        span_id="span-101",
        parent_span_id=None,
        name="Maya Intent Router",
        span_type="router",
        start_time=t0,
        end_time=t0 + 0.22,
        latency_ms=220.0,
        status="success",
        model="meta-llama/llama-3.2-3b-instruct",
        input_payload={"user_query": "1970 movies about space travel and astronauts", "temperature": 0.0},
        output_payload={
            "intent": "SEMANTIC_SEARCH",
            "confidence": 0.98,
            "reasoning": "User is searching for narrative theme of space travel in 1970.",
            "semantic_search_query": "space travel astronauts NASA moon exploration",
            "requires_rag": True,
        },
        prompt_tokens=320,
        completion_tokens=45,
        total_tokens=365,
        estimated_cost_usd=0.000008,
    )
    s1_2 = SpanRecord(
        span_id="span-102",
        parent_span_id="span-101",
        name="Hybrid Dense + BM25 Retrieval",
        span_type="retrieval",
        start_time=t0 + 0.22,
        end_time=t0 + 0.245,
        latency_ms=25.0,
        status="success",
        model="BAAI/bge-small-en-v1.5 + FTS5",
        input_payload={"query": "space travel astronauts NASA moon exploration", "top_k": 10},
        output_payload={
            "retrieved_candidates_count": 10,
            "top_candidates": [
                {"id": 49013, "title": "Moon Zero Two", "score": 0.884},
                {"id": 14160, "title": "Marooned", "score": 0.791},
            ],
        },
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        estimated_cost_usd=0.0,
    )
    s1_3 = SpanRecord(
        span_id="span-103",
        parent_span_id="span-102",
        name="FlashRank Reranker",
        span_type="rerank",
        start_time=t0 + 0.245,
        end_time=t0 + 0.260,
        latency_ms=15.0,
        status="success",
        model="ms-marco-TinyBERT-L-2-v2",
        input_payload={"candidate_count": 10, "top_n": 3},
        output_payload={"reranked_top_ids": [49013, 14160]},
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        estimated_cost_usd=0.0,
    )
    s1_4 = SpanRecord(
        span_id="span-104",
        parent_span_id=None,
        name="Maya Grounded Response Synthesis",
        span_type="llm_synthesis",
        start_time=t0 + 0.260,
        end_time=t0 + 0.780,
        latency_ms=520.0,
        status="success",
        model="meta-llama/llama-3.3-70b-instruct",
        input_payload={"context_movie_count": 2, "temperature": 0.1},
        output_payload={"response_length_chars": 480, "cited_movies": ["Moon Zero Two (1970)"]},
        prompt_tokens=580,
        completion_tokens=140,
        total_tokens=720,
        estimated_cost_usd=0.000115,
    )

    t1 = TraceRecord(
        trace_id=str(uuid.uuid4()),
        name="Maya Chat Query",
        user_query="1970 movies about space travel and astronauts",
        agent_response="In 1970, notable space-themed releases included **Moon Zero Two** (1970), a sci-fi space western set on a lunar colony...",
        intent="SEMANTIC_SEARCH",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_latency_ms=780.0,
        total_tokens=1085,
        total_cost_usd=0.000123,
        spans=[s1_1, s1_2, s1_3, s1_4],
        tags=["rag-v1.2", "production", "fastembed"],
        metadata={"platform": "Hugging Face Spaces", "embedder": "bge-small-en-v1.5"},
    )

    # Trace 2: Superlative Ranking Query
    s2_1 = SpanRecord(
        span_id="span-201",
        parent_span_id=None,
        name="Maya Intent Router",
        span_type="router",
        start_time=t0,
        end_time=t0 + 0.190,
        latency_ms=190.0,
        status="success",
        model="meta-llama/llama-3.2-3b-instruct",
        input_payload={"user_query": "What is the longest film released in 1970?", "temperature": 0.0},
        output_payload={
            "intent": "SUPERLATIVE_RANKING",
            "superlative": {"metric": "RUNTIME", "direction": "DESC", "limit": 1},
            "requires_rag": True,
        },
        prompt_tokens=290,
        completion_tokens=38,
        total_tokens=328,
        estimated_cost_usd=0.000007,
    )
    s2_2 = SpanRecord(
        span_id="span-202",
        parent_span_id="span-201",
        name="Deterministic SQL Executor",
        span_type="tool",
        start_time=t0 + 0.190,
        end_time=t0 + 0.195,
        latency_ms=5.0,
        status="success",
        model="SQLite",
        input_payload={"sql": "SELECT title, runtime, director FROM movies WHERE runtime > 0 ORDER BY runtime DESC LIMIT 1;"},
        output_payload={"result": [{"title": "Waterloo", "runtime": 134, "director": "Sergei Bondarchuk"}]},
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        estimated_cost_usd=0.0,
    )
    t2 = TraceRecord(
        trace_id=str(uuid.uuid4()),
        name="Maya Superlative Query",
        user_query="What is the longest film released in 1970?",
        agent_response="The longest major theatrical release in 1970 was **Waterloo** directed by Sergei Bondarchuk, clocking in at 134 minutes.",
        intent="SUPERLATIVE_RANKING",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_latency_ms=195.0,
        total_tokens=328,
        total_cost_usd=0.000007,
        spans=[s2_1, s2_2],
        tags=["deterministic-sql", "zero-vector"],
        metadata={"query_type": "superlative_runtime"},
    )

    return [t1, t2]


# ==============================================================================
# 4. STREAMLIT OBSERVABILITY INSPECTOR UI COMPONENT
# ==============================================================================

def render_observability_inspector_ui():
    """Renders the comprehensive Langfuse-style trace tree and telemetry dashboard."""
    st.set_page_config(page_title="Maya Observability & Trace Inspector", page_icon="🔍", layout="wide")

    st.markdown("## 🔍 Maya Observability & Trace Inspector")
    st.caption("Langfuse-compatible telemetry, execution DAGs, span latencies, and token cost breakdown.")

    # Initialize demo data if empty
    if "trace_history" not in st.session_state or not st.session_state["trace_history"]:
        st.session_state["trace_history"] = generate_mock_traces()

    traces: List[TraceRecord] = st.session_state["trace_history"]

    # Top KPI Metrics Bar
    col1, col2, col3, col4, col5 = st.columns(5)
    total_traces = len(traces)
    avg_latency = sum(t.total_latency_ms for t in traces) / total_traces if total_traces else 0
    total_tokens = sum(t.total_tokens for t in traces)
    total_cost = sum(t.total_cost_usd for t in traces)
    langfuse_status = "🟢 Connected (Cloud)" if os.getenv("LANGFUSE_PUBLIC_KEY") else "🟡 Local In-Memory Mode"

    col1.metric("Total Traces Logged", f"{total_traces}")
    col2.metric("Avg Trace Latency", f"{avg_latency:.1f} ms")
    col3.metric("Total Tokens", f"{total_tokens:,}")
    col4.metric("Est. Total Cost", f"${total_cost:.5f}")
    col5.metric("Telemetry Destination", langfuse_status)

    st.divider()

    # Two-Column Layout: Trace Selector (Left) vs Deep Span Inspector (Right)
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("### 📋 Recent Executions")
        trace_options = [
            f"{t.timestamp[-8:]} | [{t.intent}] {t.user_query[:35]}..."
            for t in traces
        ]
        selected_idx = st.radio(
            "Select Trace to Inspect",
            range(len(traces)),
            format_func=lambda idx: trace_options[idx],
            label_visibility="collapsed",
        )
        selected_trace = traces[selected_idx]

        st.info(f"**Trace ID:** `{selected_trace.trace_id[:18]}...`\n\n**Intent:** `{selected_trace.intent}`\n\n**Cost:** `${selected_trace.total_cost_usd:.6f}`")

    with right_col:
        st.markdown(f"### 🔬 Trace Analysis: *{selected_trace.user_query}*")

        # Tab layout for Trace Details
        tab_spans, tab_waterfall, tab_json = st.tabs(["🌳 Span Tree & Payloads", "⏱️ Waterfall Timeline", "📄 Raw Trace JSON"])

        with tab_spans:
            st.markdown(f"**User Prompt:** `{selected_trace.user_query}`")
            st.markdown(f"**Maya Output:** {selected_trace.agent_response}")
            st.markdown("#### Execution Spans:")

            for idx, span in enumerate(selected_trace.spans):
                badge_type = {
                    "router": "🎯 ROUTER",
                    "retrieval": "📚 RETRIEVAL",
                    "rerank": "⚡ RERANKER",
                    "llm_synthesis": "🤖 SYNTHESIS",
                    "tool": "🛠️ SQL TOOL",
                }.get(span.span_type, "⚙️ SPAN")

                with st.expander(f"{badge_type}: **{span.name}** ({span.latency_ms:.1f} ms | {span.model or 'Internal'})", expanded=(idx == 0)):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.caption(f"**Type:** {span.span_type}")
                    c2.caption(f"**Latency:** {span.latency_ms:.1f} ms")
                    c3.caption(f"**Tokens:** {span.total_tokens}")
                    c4.caption(f"**Cost:** ${span.estimated_cost_usd:.6f}")

                    st.markdown("**Input Payload:**")
                    st.json(span.input_payload)

                    st.markdown("**Output Payload:**")
                    st.json(span.output_payload)

        with tab_waterfall:
            st.markdown("#### Span Latency Waterfall")
            for span in selected_trace.spans:
                percentage = max(5, int((span.latency_ms / max(1.0, selected_trace.total_latency_ms)) * 100))
                st.write(f"**{span.name}** (`{span.latency_ms:.1f} ms`)")
                st.progress(percentage / 100.0)

        with tab_json:
            st.json(selected_trace.to_dict())


if __name__ == "__main__":
    render_observability_inspector_ui()
