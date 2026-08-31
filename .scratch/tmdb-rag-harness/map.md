## Destination

An observable, modular RAG & Evaluation Harness for 1970–2026 US TMDB movies with the "Maya" conversational agent (deterministic query intent routing, structured output, poster rendering, zero-hallucination constraints, multi-turn memory, security/budget guardrails, user feedback, transparent meta-prompt), versioned evaluation comparisons across embedders/chunkers, dynamic architecture experimentation control plane, and Langfuse-compatible trace inspector UI, ready to host on Hugging Face Spaces (Streamlit).

## Notes

- **Domain**: RAG, Information Retrieval Evaluation, Deterministic Agent Routing, LLM Observability, Multi-turn Conversational Memory, Security Guardrails, User Feedback Telemetry, Transparent System Prompts, Movie Knowledge Systems.
- **Skills to consult**: `codebase-design`, `domain-modeling`, `grilling`, `tdd`, `code-review`, `setup-pre-commit`, `managing-python-dependencies`, `building-data-apps`.
- **Standing preferences**:
  - LLM Provider: OpenRouter (cost-effective structured models like `meta-llama/llama-3.2-3b-instruct` for routing/reformulation, `meta-llama/llama-3.3-70b-instruct` / `google/gemini-2.0-flash` for synthesis).
  - Target Platform: Hugging Face Spaces (Streamlit SDK) with light memory footprint and standalone local fallback.
  - Architecture: Deep modules, clear seams between Ingestion, Storage, Vector Index, Maya Router, Multi-turn State, Experiment Control Plane, Retrieval Engine, Eval Runner, Security/Budget Layer, Feedback Store, and Streamlit UI.

## Decisions so far

- [[01] TMDB 1970–2026 Dataset Schema & Ingestion](issues/01-tmdb-1970-dataset-schema-and-ingestion.md): Implemented two-stage discovery+hydration via TMDB API v3; normalized `MovieRecord` Pydantic models; dual SQLite + FTS5 & JSON storage pre-bundled in `data/` for zero-dependency instant startup on Hugging Face Spaces.
- [[02] Maya Intent Taxonomy & Deterministic Routing](issues/02-maya-intent-taxonomy-and-deterministic-routing.md): Defined 7-class intent taxonomy (`GREETING`, `CAPABILITIES`, `SEMANTIC_SEARCH`, `ATTRIBUTE_FILTER`, `SUPERLATIVE_RANKING`, `NEGATION_EXCLUSION`, `OUT_OF_SCOPE`); structured JSON schema routing via `meta-llama/llama-3.2-3b-instruct` at temperature=0; closed-world `<retrieved_movies>` XML grounding with poster rendering.
- [[03] Embedder & Chunking Experimentation Matrix](issues/03-embedder-and-chunking-experimentation-matrix.md): Formulated 3 evaluation versions (`v1.0-baseline`, `v1.1-enriched-metadata`, `v1.2-bge-hybrid`) using lightweight ONNX/FastEmbed models (`all-MiniLM-L6-v2`, `bge-small-en-v1.5`), BM25Okapi lexical index, and FlashRank CPU reranker.
- [[04] Eval Harness Metrics & Ground Truth Benchmark](issues/04-eval-harness-metrics-and-ground-truth-benchmark.md): Established two-tier benchmark (`data/eval_benchmark_dataset.json`) containing 2000–2026 guardrail queries (majority) and 1970–2026 retrieval queries; formalized formulas for Hit Rate@K, MRR@K, Context Precision, Faithfulness/Hallucination LLM-as-a-Judge, and ChromaDB persistent vector storage.
- [[05] Langfuse Observability & In-App Trace Inspector](issues/05-langfuse-traceability-and-in-app-inspector.md): Architected unified `SpanRecord` / `TraceRecord` model with `DualModeObservabilityManager` (Langfuse Cloud + zero-dependency local memory store) and interactive Streamlit DAG/timeline inspector prototyped in `prototypes/observability_trace_inspector_prototype.py`.
- [[06] Streamlit UI Architecture & HF Spaces Deployment](issues/06-streamlit-ui-architecture-and-hf-spaces-deployment.md): Structured 3-tab layout (Chat with Maya & Poster Grid, Evals Dashboard with Version Deltas & Decision Log, Trace Inspector) and Hugging Face Spaces deployment metadata (`app.py`, `requirements.txt`, `README.md`) prototyped in `prototypes/streamlit_app_layout_prototype.py`.
- [[07] Multi-Turn Memory & Query Reformulation](issues/07-multi-turn-memory-and-query-reformulation.md): Established 5-layer conversational memory taxonomy (`ConversationState`), configurable 1-Step Fused vs 2-Step Dedicated LLM reformulation engine, persistent session exclusions for negations, and 3-tier fallback matrix (soft-boosted hybrid, pure vector, and clarification).
- [[08] Experimentation Control Plane & Dynamic Knobs](issues/08-experimentation-control-plane-and-dynamic-knobs.md): Defined `ExperimentConfig` schema with dynamic model selection, reasoning effort, hybrid alpha, reranker, CWA guardrails, 1-click Presets, sidebar lab, `/admin` shortcut, and live custom benchmarking.
- [[09] Security Guardrails, Token Caps & Weekly Budget Limits](issues/09-security-guardrails-token-limits-and-budget.md): Defined prompt injection sanitizer regex, off-topic deflection with film-curator persona pivots, 15,000 session token cap, and persistent $5.00/week SQLite budget ceiling.
- [[10] Streamlit In-App User Feedback & Langfuse Scoring](issues/10-streamlit-user-feedback-and-langfuse-scoring.md): Integrated native `st.feedback("thumbs")`, linking feedback to `trace_id` and RAG version, pushing scores to Langfuse Cloud, and recording satisfaction in SQLite.
- [[11] Maya Transparent System Prompt & Architectural Self-Awareness](issues/11-maya-transparent-system-prompt-and-architectural-persona.md): Constructed 8-pillar meta-system prompt explicitly detailing intent classification, query reformulation, dual retrievers, RRF ranking, memory state, and strict CWA grounding.

## Not yet specified

- Continuous evaluation pipeline for automated regressions on user chat feedback.

## Out of scope

- Real-time movie streaming, ticket booking, or live external cinema scraping.
- Full fine-tuning of embedding models on GPU clusters.
- Video playback or audio processing.
