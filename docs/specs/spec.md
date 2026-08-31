Status: ready-for-agent

# Specification: TMDB 1970–2026 RAG & Evaluation Harness ("Maya")

## Problem Statement

Building reliable Retrieval-Augmented Generation (RAG) systems on cheap, cost-effective LLMs is notoriously difficult. Developers and AI practitioners often struggle with:
1. **Model Drift & Hallucinations**: Cheap LLMs frequently invent movie facts, hallucinate box office figures, and fail on strict negative or superlative constraints.
2. **Invisible Architecture Trade-offs**: It is hard to empirically observe how changing an embedding model (e.g., MiniLM vs. BGE), chunking strategy (raw overview vs. enriched metadata), or routing approach affects retrieval precision, latency, and operational cost.
3. **Lack of Observability**: Most RAG apps hide execution details, making it impossible to see why an agent chose a specific path or retrieved a particular document.
4. **Hosting & Deployment Friction**: Many RAG systems rely on heavy GPU clusters or expensive hosted vector databases, making them difficult to deploy on lightweight, free CPU hosting tiers like Hugging Face Spaces.

## Solution

A modular, observable RAG application and Evaluation Harness specialized in US theatrical releases from **1970 to 2026**, powered by cost-effective OpenRouter models, orchestrated via **LangGraph**, and deployed on **Hugging Face Spaces (Streamlit)**:
- **"Maya" Conversational Film Curator**: Features deterministic intent routing (`GREETING`, `CAPABILITIES`, `SEMANTIC_SEARCH`, `ATTRIBUTE_FILTER`, `SUPERLATIVE_RANKING`, `NEGATION_EXCLUSION`, `OUT_OF_SCOPE`), multi-turn memory state, and zero-hallucination Closed-World Assumption (CWA) grounding with dynamic movie poster rendering.
- **Evaluation & Benchmark Dashboard**: Tracks quantitative Information Retrieval metrics (Hit Rate@K, MRR@K, Context Precision) and LLM-as-a-judge generation metrics (Faithfulness, Relevancy) across versioned milestones (`v1.0-baseline`, `v1.1-enriched-metadata`, `v1.2-bge-hybrid`).
- **Observability & Trace Inspector**: Dual-mode telemetry pushing traces to Langfuse Cloud while rendering an in-app interactive DAG waterfall and span inspector in Streamlit.
- **Live Experimentation Control Plane**: A sidebar `/admin` panel allowing live toggling of models, reasoning effort, memory strategies, hybrid alpha weights, and reranking to observe real-time latency and accuracy deltas.
- **Zero-Dependency Local Storage**: Bundled SQLite database with FTS5 lexical index and embedded ChromaDB collections running on CPU with FastEmbed ONNX runtime.

---

## User Stories

1. As a film enthusiast, I want to ask Maya about complex plot concepts from 1970 to 2026 (e.g., *"movies about space exploration and lunar colonies"*), so that I get accurate movie recommendations grounded strictly in verified TMDB records.
2. As a user, I want Maya to render high-resolution movie posters and formatted metadata cards (Title, Year, Director, Cast, Rating, Runtime), so that I can visually explore recommendations.
3. As a user, I want to ask superlative and ranking queries (e.g., *"What was the highest-grossing film of 1970?"* or *"What is the longest drama released in 1999?"*), so that I get exact, verified historical statistics without mathematical hallucinations.
4. As a user, I want to filter by complex metadata (e.g., *"1980s horror movies directed by John Carpenter starring Kurt Russell"*), so that I find exact filmographic matches.
5. As a user, I want to ask follow-up questions using pronouns (e.g., *"Who directed it?"*, *"Show me other movies starring him"*), so that I enjoy a smooth, contextual conversational experience.
6. As a user, I want to set persistent negative preferences (e.g., *"Don't show me horror movies"* or *"Action movies without Tom Cruise"*), so that all subsequent searches automatically respect my exclusions.
7. As a user, I want Maya to politely reject pre-1970 movie queries or non-film queries with helpful domain pivots, so that I understand the boundaries of the assistant.
8. As a developer/evaluator, I want to view the Evaluation Dashboard to compare Hit Rate@5, MRR@5, Context Precision, and Faithfulness across RAG version milestones (`v1.0`, `v1.1`, `v1.2`), so that I understand what architectural changes drove performance improvements.
9. As a developer, I want to read the historical Decision Log in the UI, so that I can see the exact engineering rationale behind each version jump.
10. As a developer, I want to run live benchmark runs against the curated 35-query ground-truth suite, so that I can test new parameter combinations on the fly.
11. As an AI engineer, I want to inspect the Observability Tab to see the exact execution DAG, latency waterfall, prompt payloads, token counts, and USD cost per turn, so that I can debug agent decision-making.
12. As a developer, I want all traces to mirror to Langfuse Cloud automatically when API keys are provided, so that I have centralized production monitoring.
13. As a developer without a Langfuse account, I want the in-app Streamlit trace inspector to work 100% locally from session state, so that the app is self-contained.
14. As an evaluator, I want to use the Sidebar Experimentation Lab (or `/admin` chat shortcut) to toggle models (`Llama-3.2-3b`, `Llama-3.3-70b`, `Gemini-2.0-Flash`), reasoning effort, memory strategies, and rerankers, so that I can observe the immediate impact on speed and accuracy.
15. As a deployment engineer, I want the entire application to start in < 5 seconds on Hugging Face Spaces free CPU tier, so that users experience zero setup lag or TMDB API rate-limiting.

---

## Implementation Decisions

### 1. Data Ingestion & Storage Architecture
- **Dataset**: US Theatrical Releases from 1970 to 2026 fetched via TMDB API v3 (`discover/movie` + `append_to_response=credits,keywords`).
- **Offline Bundling**: Pre-fetched records saved to `data/tmdb_movies.db` (SQLite) and `data/tmdb_movies.json` (~20-40 MB total).
- **SQLite + FTS5**: Relational tables indexed on `release_date`, `vote_average`, `runtime`, `revenue`, `director` + `movies_fts` virtual table for BM25 lexical keyword search.
- **Image Resolution**: Posters resolved to `https://image.tmdb.org/t/p/w500{poster_path}` with SVG placeholder fallbacks.

### 2. Embedding, Indexing & Hybrid Search Engine
- **Embedding Engine**: `fastembed` running INT8 quantized ONNX models (`BAAI/bge-small-en-v1.5` and `sentence-transformers/all-MiniLM-L6-v2`) on CPU.
- **Vector Store**: Embedded persistent `chromadb.PersistentClient` in `data/chroma_db/`.
- **Hybrid Retrieval Flow**:
  - `SUPERLATIVE_RANKING` & exact `ATTRIBUTE_FILTER`: Executed directly via deterministic SQL on SQLite.
  - `SEMANTIC_SEARCH`: ChromaDB Dense Vector Search + SQLite FTS5 BM25 Search.
  - **Fusion & Reranking**: Reciprocal Rank Fusion (RRF, $k=60$) over Dense + Sparse results $\rightarrow$ Top-20 candidates reranked via `flashrank` (`ms-marco-TinyBERT-L-2-v2`) on CPU $\rightarrow$ Top-5 context chunks.

### 3. Maya Conversational Agent & LangGraph Orchestration
- **Orchestration**: Built on **`langgraph.graph.StateGraph`** with conditional branching edges:
  ```
  [Route Query Node] --> Decision Branch:
    - If Superlative/Filter --> [Execute SQL Node] --> [Synthesize Node]
    - If Semantic Search    --> [Execute Hybrid RAG Node] --> [Synthesize Node]
    - If Direct/Out-of-Scope --> [Direct Response Node]
  ```
- **Deterministic Router**: OpenRouter `meta-llama/llama-3.2-3b-instruct` at temperature=0.0 using `ChatOpenAI.with_structured_output(QueryRoutingDecision)`.
- **Closed-World Assumption (CWA)**: Maya's synthesis prompt injects context via `<retrieved_movies>` XML tags and strictly bans fabricating unretrieved movies or metadata.

### 4. Multi-Turn Conversational Memory & Fallbacks
- **5-Layer Memory (`ConversationState`)**:
  1. *Short-term window*: Last 4–6 messages.
  2. *Active Entity Focus*: `focused_entity` and `focused_person`.
  3. *Session Exclusions*: Persistent negative constraints (`excluded_genres`, `excluded_actors`).
  4. *Seen Recommendation Tracker*: `shown_movie_ids` preventing duplicate recommendations.
  5. *Rolling Summary*: 2-sentence background summary for chats > 10 turns.
- **Reformulation**: Fused single-pass router (default) or dedicated 2-step rewriter LLM (configurable).
- **Fallback Matrix**: Soft-boosted hybrid search on low confidence, pure vector search on JSON failure, interactive disambiguation on pronoun conflicts.

### 5. Experimentation Control Plane (`ExperimentConfig`)
- Exposed via Sidebar Lab and `/admin` chat command with live Pydantic validation:
  - Models (`router_model`, `synthesis_model`, `reasoning_effort`)
  - Multi-turn mode (`fused_single_pass`, `dedicated_2step_llm`)
  - Memory strategy (`sliding_window_with_entity`, `pure_sliding_window`, `rolling_summarizer`)
  - Retrieval knobs (`embedding_model`, `hybrid_alpha`, `reranker_enabled`, `retrieval_top_k`, `temperature`)
  - 1-Click Presets: *Fast Budget*, *Production Hybrid*, *Naive Baseline*.

### 6. Observability & Telemetry
- **`DualModeObservabilityManager`**: Uses `langfuse.callback.CallbackHandler` when `LANGFUSE_PUBLIC_KEY` is present.
- **Local Streamlit Inspector**: In-app execution DAG viewer, span latency waterfall, token usage, and cost calculation.

---

## Testing Decisions

Following the **`tdd`** and **`codebase-design`** skills:
1. **Public Seam Testing Only**: Tests verify behavior strictly through public module interfaces (`db.query_superlative()`, `router.route()`, `engine.retrieve()`, `evals.run()`), never private methods.
2. **3-Tier Test Structure**:
   - **Tier 1 (Unit / Mock Tests)**: Uses in-memory SQLite database (`:memory:`), mock vector results, and recorded OpenRouter JSON fixtures. Runs 100% offline in < 2 seconds.
   - **Tier 2 (Adversarial Benchmark Tests)**: Tests temporal boundaries (pre-1970 out-of-scope), negation exclusions, typo handling, and empty-context anti-hallucination assertions.
   - **Tier 3 (Live Smoke Tests)**: Optional live OpenRouter end-to-end turn gated behind `@pytest.mark.skipif(not os.getenv("OPENROUTER_API_KEY"))`.

---

## Out of Scope

- Real-time movie ticket booking, cinema scraping, or third-party streaming links.
- Full embedding model fine-tuning on GPU clusters.
- Video playback, audio streaming, or multi-modal poster image generation.
- Cross-session user accounts / remote user authentication.

---

## Further Notes

- **Target Deployment**: Hugging Face Spaces (Streamlit SDK) with `app.py`, `requirements.txt`, and YAML metadata in `README.md`.
- **Python Version**: Python 3.11+.
