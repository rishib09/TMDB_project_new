Type: prototype
Status: resolved
Blocked by: 02

## Question

How should we design the dual-mode Observability layer that automatically transmits OpenRouter calls, latency, token counts, routing decisions, and retrieved documents to Langfuse Cloud/self-hosted (when `LANGFUSE_PUBLIC_KEY` & `LANGFUSE_SECRET_KEY` are provided) while simultaneously recording local span traces and rendering a rich interactive tree/DAG inspector inside Streamlit for local & standalone Hugging Face Space users?

## Answer

### 1. Unified Telemetry Data Model
- **`SpanRecord`**: Captures individual execution blocks (`span_id`, `parent_span_id`, `name`, `span_type: Literal["router", "retrieval", "rerank", "llm_synthesis", "tool"]`, `start_time`, `end_time`, `latency_ms`, `status`, `model`, `input_payload`, `output_payload`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `estimated_cost_usd`, `error_message`).
- **`TraceRecord`**: Top-level user session/query record aggregating total latency, token count, cost, intent classification, and ordered span tree.

### 2. Dual-Mode Telemetry Adapter (`DualModeObservabilityManager`)
- **Cloud Mode**: If `LANGFUSE_PUBLIC_KEY` & `LANGFUSE_SECRET_KEY` are detected in environment variables or Streamlit secrets, traces and spans are mirrored directly to Langfuse Cloud via the Langfuse Python SDK.
- **Local Standalone Mode**: If keys are omitted, the system continues running seamlessly in local mode without network dependencies or errors, storing trace histories in `st.session_state["trace_history"]`.

### 3. Streamlit In-App Trace Inspector UI Component
- **Top KPI Metrics**: Total Traces, Average Latency (ms), Total Token Consumption, Total Cost ($), and Telemetry Destination status badge.
- **Left Panel (Execution Selector)**: Time-stamped list with intent tags (`[SEMANTIC_SEARCH]`, `[SUPERLATIVE_RANKING]`, `[ATTRIBUTE_FILTER]`) and cost/latency badges.
- **Right Panel (Deep Inspection Tabs)**:
  1. **Span Tree & Payloads**: Collapsible expander per span showing latency, exact prompt/temperature, retrieved candidate scores, and structured JSON output.
  2. **Waterfall Timeline**: Visual progress/bar chart showing relative span latency across the pipeline.
  3. **Raw Trace JSON**: Formatted JSON for export or debugging.

### 4. Validated Prototype
- Implemented and verified in [`prototypes/observability_trace_inspector_prototype.py`](file:///D:/GitHub_Repo/TMDB_project_new/prototypes/observability_trace_inspector_prototype.py).
