"""
prototypes/streamlit_app_layout_prototype.py

Interactive prototype for the Maya TMDB RAG & Evaluation Harness Streamlit UI.
Demonstrates:
- Tab 1: Chat with Maya (with Intent Badge, Grounded Synthesis, and Movie Poster Grid)
- Tab 2: Evals Dashboard (Version Comparison, KPI Deltas, and Decision Log)
- Tab 3: Observability & Trace Inspector (Langfuse-style Span Tree)
"""

from __future__ import annotations
import json
import os
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Maya — 1970 TMDB RAG & Eval Harness",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #6366f1; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #64748b; margin-bottom: 1.5rem; }
    .intent-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #e0e7ff;
        color: #4338ca;
        margin-bottom: 0.5rem;
    }
    .metric-delta-positive { color: #16a34a; font-weight: 600; }
    .movie-card {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 10px;
        background-color: #ffffff;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("## 🎬 **Maya Engine**")
    st.caption("1970 US TMDB RAG & Evaluation Harness")

    st.divider()
    st.markdown("### ⚙️ Pipeline Configuration")
    active_version = st.selectbox(
        "Active RAG Pipeline Version",
        ["v1.2-bge-hybrid (Production)", "v1.1-enriched-metadata", "v1.0-baseline"],
        index=0,
    )

    st.markdown("### 🔑 OpenRouter & Telemetry")
    api_key_input = st.text_input("OpenRouter API Key", type="password", placeholder="sk-or-v1-...", help="Uses OPENROUTER_API_KEY from environment if blank.")
    
    langfuse_status = "🟢 Connected" if os.getenv("LANGFUSE_PUBLIC_KEY") else "🟡 Local In-Memory"
    st.write(f"**Langfuse Telemetry:** {langfuse_status}")

    st.divider()
    st.markdown("### 📊 Database & Index Status")
    st.write("• **Movies Indexed:** 834 films (1970 US)")
    st.write("• **Storage:** SQLite + ChromaDB (Local)")
    st.write("• **Hosting Target:** Hugging Face Spaces")


# ==============================================================================
# MAIN TABS
# ==============================================================================
st.markdown('<div class="main-header">🎬 Maya: 1970 TMDB RAG & Evaluation Harness</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deterministic Intent Routing, Zero-Hallucination Grounding, Multi-Version RAG Evals, and Full Traceability.</div>', unsafe_allow_html=True)

tab_chat, tab_evals, tab_traces = st.tabs([
    "💬 Chat with Maya",
    "📈 Evaluation & Benchmark Dashboard",
    "🔍 Observability & Trace Inspector",
])

# ------------------------------------------------------------------------------
# TAB 1: CHAT WITH MAYA
# ------------------------------------------------------------------------------
with tab_chat:
    st.markdown("### 💬 Converse with Maya")
    st.caption("Ask anything about 1970 US theatrical releases (plots, actors, directors, box office, superlatives) or test out-of-scope guardrails.")

    # Sample Quick Queries
    st.markdown("**Sample Quick Prompts:**")
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("🚀 1970 Space Exploration"):
        st.session_state["sample_query"] = "1970 space exploration movie with lunar astronauts"
    if col2.button("💰 Highest Grossing Film"):
        st.session_state["sample_query"] = "What was the highest grossing movie of 1970?"
    if col3.button("😱 Horror by Dario Argento"):
        st.session_state["sample_query"] = "1970 horror movies directed by Dario Argento"
    if col4.button("🚫 Inception (2010) [Guardrail]"):
        st.session_state["sample_query"] = "Tell me about Inception (2010)"

    # Mock Chat History Rendering
    st.divider()
    
    # Assistant turn mock
    with st.chat_message("user"):
        st.write(st.session_state.get("sample_query", "1970 space exploration movie with lunar astronauts"))

    with st.chat_message("assistant", avatar="🎬"):
        st.markdown('<span class="intent-badge">🎯 Intent: SEMANTIC_SEARCH (Confidence: 98%) | Latency: 245ms</span>', unsafe_allow_html=True)
        st.write("""
In 1970, notable space-themed releases included **Moon Zero Two** (1970) and **Marooned** (1970).

1. **Moon Zero Two (1970)** [TMDB ID: 49013]  
   - **Director**: Roy Ward Baker | **Genres**: Science Fiction, Western  
   - **Rating**: ⭐ 5.2/10 | **Runtime**: 100 mins  
   - **Overview**: Set in 2021, a space salvager on a lunar colony is hired to capture a valuable asteroid.

2. **Marooned (1970 US Release)** [TMDB ID: 14160]  
   - **Director**: John Sturges | **Genres**: Drama, Science Fiction  
   - **Rating**: ⭐ 6.3/10 | **Runtime**: 134 mins  
   - **Overview**: Three American astronauts are stranded in space after their retrorockets fail.
""")
        st.markdown("#### 🎞️ Featured Movie Posters")
        p_col1, p_col2, p_col3, _ = st.columns([1, 1, 1, 1])
        with p_col1:
            st.image("https://image.tmdb.org/t/p/w500/u3vK3rM5i0YqfFvXzG4v2jVbQ2s.jpg", caption="Moon Zero Two (1970)", width=160)
        with p_col2:
            st.image("https://image.tmdb.org/t/p/w500/9E2l8VRnZc71m4l1W73J7h6pXmN.jpg", caption="Patton (1970)", width=160)

    user_input = st.chat_input("Ask Maya a question about 1970 movies...")


# ------------------------------------------------------------------------------
# TAB 2: EVALS DASHBOARD
# ------------------------------------------------------------------------------
with tab_evals:
    st.markdown("### 📈 RAG Evaluation Harness & Version Progression")
    st.caption("Benchmark comparison across RAG architectural milestones evaluated against `data/eval_benchmark_dataset.json`.")

    # Top KPI Metrics Comparison Table
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Hit Rate@5 (In-Scope)", "94.3%", "+19.1% vs Baseline")
    kpi_col2.metric("MRR@5 (Mean Reciprocal Rank)", "0.89", "+0.38 vs Baseline")
    kpi_col3.metric("Faithfulness / Zero-Hallucination", "99.2%", "+12.4% vs Baseline")
    kpi_col4.metric("Temporal Guardrail Accuracy", "100.0%", "Perfect on 2000-2026")

    st.divider()

    st.markdown("#### 📊 Comparative Metrics Matrix Across Versions")
    eval_df = pd.DataFrame({
        "Version": ["v1.0-baseline", "v1.1-enriched-metadata", "v1.2-bge-hybrid"],
        "Embedder": ["all-MiniLM-L6-v2", "all-MiniLM-L6-v2", "bge-small-en-v1.5"],
        "Chunking Strategy": ["Raw Overview", "Structured Markdown", "Multi-Field (Dense+SQL+BM25)"],
        "Hit Rate@5": ["61.5%", "82.4%", "94.3%"],
        "MRR@5": [0.51, 0.72, 0.89],
        "Faithfulness": ["86.8%", "94.1%", "99.2%"],
        "Avg Latency (ms)": [8.2, 11.5, 34.0],
        "Cost per 1k ($)": ["$0.008", "$0.009", "$0.012"],
    })
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### 📜 Architectural Decision Progression Log")
    with st.expander("🌱 **Version 1.0 (Baseline)** — Raw Text Chunks & Vector-Only Search", expanded=False):
        st.write("**Key Architecture:** `all-MiniLM-L6-v2` embedding raw movie overview strings. Pure cosine vector similarity.")
        st.write("**Outcome:** Performed moderately on plot queries (61.5% Hit Rate), but failed completely on actor, director, and superlative queries because cast/crew metadata were absent from the embedding.")

    with st.expander("🌿 **Version 1.1 (Enriched Metadata)** — Structured Markdown Templates", expanded=False):
        st.write("**Key Architecture:** Formatted movie records into structured markdown (`Title`, `Director`, `Cast`, `Genres`, `Overview`).")
        st.write("**Outcome:** **+20.9% Hit Rate jump** (to 82.4%) because actor and director queries now matched directly in semantic space.")

    with st.expander("🌳 **Version 1.2 (Hybrid Production)** — BGE-small + Maya SQL Intent Filter + FlashRank Reranker", expanded=True):
        st.write("**Key Architecture:** Replaced pure vector search with deterministic SQL routing for superlatives/filters + BGE-small dense embeddings + BM25Okapi lexical matching + FlashRank CPU reranker.")
        st.write("**Outcome:** Achieved **94.3% Hit Rate@5**, **0.89 MRR**, and **99.2% Faithfulness**, eliminating hallucinations while maintaining sub-40ms latency on CPU.")


# ------------------------------------------------------------------------------
# TAB 3: OBSERVABILITY & TRACE INSPECTOR
# ------------------------------------------------------------------------------
with tab_traces:
    # Embedded local version of the trace inspector from Ticket 05
    st.markdown("### 🔍 Live Trace Inspector (Langfuse-Compatible)")
    st.caption("Inspect every step of Maya's decision DAG (Router, SQL Tool, Vector Retrieval, FlashRank Reranker, Synthesis).")
    
    st.info("💡 Traces are mirrored to Langfuse Cloud when API credentials are provided, and recorded locally in memory for standalone Hugging Face Space operation.")
    st.json({
        "trace_id": "trace-maya-sample-101",
        "intent": "SEMANTIC_SEARCH",
        "latency_ms": 245.0,
        "spans": [
            {"name": "Maya Intent Router", "model": "meta-llama/llama-3.2-3b-instruct", "latency_ms": 195.0, "tokens": 365},
            {"name": "Hybrid Dense + BM25 Retrieval", "model": "BAAI/bge-small-en-v1.5 + SQLite FTS5", "latency_ms": 22.0},
            {"name": "FlashRank Reranker", "model": "ms-marco-TinyBERT-L-2-v2", "latency_ms": 14.0},
            {"name": "Maya Grounded Synthesis", "model": "meta-llama/llama-3.3-70b-instruct", "latency_ms": 510.0, "tokens": 720}
        ]
    })
