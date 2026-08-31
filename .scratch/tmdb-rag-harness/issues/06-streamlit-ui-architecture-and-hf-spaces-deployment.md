Type: prototype
Status: resolved
Blocked by: 01, 04, 05

## Question

How should the Streamlit application be architected with clear visual layout (Tab 1: Chat with Maya with poster gallery & intention tag; Tab 2: Evals Dashboard with version selector, metric charts, and decision progression; Tab 3: Trace & Observability Inspector) and packaged (`app.py`, `requirements.txt`, `README.md` with HF YAML metadata) for zero-friction deployment to Hugging Face Spaces?

## Answer

### 1. Multi-Tab Visual Architecture (`app.py`)
- **Sidebar**: Version selector (`v1.0-baseline`, `v1.1-enriched-metadata`, `v1.2-bge-hybrid`), OpenRouter API key entry (with fallback to environment variable), Langfuse connection status badge, and database statistics.
- **Tab 1 (Chat with Maya)**:
  - Streaming conversational UI with chat history.
  - Dynamic **Intent Badges** (e.g. `🎯 Intent: SEMANTIC_SEARCH (Confidence: 98%) | Latency: 245ms`).
  - Grounded Markdown synthesis citing TMDB IDs, directors, ratings, and runtimes.
  - **Movie Poster Grid**: Responsive horizontal poster cards (`https://image.tmdb.org/t/p/w500{poster_path}`).
  - Quick-prompt suggestions (including modern 2000–2026 out-of-scope queries to test temporal guardrails).
- **Tab 2 (Evals Dashboard)**:
  - Top KPI scorecards (Hit Rate@5, MRR@5, Faithfulness/Zero-Hallucination, Temporal Guardrail Accuracy).
  - Comparative metrics matrix across versions `v1.0`, `v1.1`, and `v1.2`.
  - Historical **Decision Log** detailing the architectural rationale for each version improvement.
- **Tab 3 (Observability & Trace Inspector)**:
  - Dual-mode Langfuse-compatible trace tree, span latencies, prompt payloads, token counts, and estimated cost breakdown.

### 2. Hugging Face Spaces Packaging Specification
- **`README.md`**: Includes required Hugging Face Space metadata:
  ```yaml
  ---
  title: Maya - TMDB 1970 RAG & Eval Harness
  emoji: 🎬
  colorFrom: indigo
  colorTo: purple
  sdk: streamlit
  sdk_version: 1.39.0
  app_file: app.py
  pinned: false
  ---
  ```
- **`requirements.txt`**: Pinned lightweight CPU dependencies (`streamlit`, `fastembed`, `chromadb`, `openai`, `pydantic`, `httpx`, `flashrank`, `pandas`, `plotly`, `langfuse`).
- **Zero-Dependency Startup**: Bundles `data/tmdb_1970.db`, `data/tmdb_1970_movies.json`, and `data/eval_benchmark_dataset.json` directly in the repository so the Space starts in < 5 seconds without external database setup or TMDB API rate-limiting.

### 3. Validated Prototype
- Prototyped and verified in [`prototypes/streamlit_app_layout_prototype.py`](file:///D:/GitHub_Repo/TMDB_project_new/prototypes/streamlit_app_layout_prototype.py).
