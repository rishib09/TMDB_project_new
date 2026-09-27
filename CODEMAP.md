# Maya (TMDB RAG & Evaluation Harness) — code map

_Generated from `.codemap/graph.json` @ f07c9eb, 2026-09-13. Regenerate with `codebase-map` after structural changes. Graph excludes `tests/`; blast-radius numbers below come from a second build with `--include-tests`._

## Purpose

Maya is a conversational film curator over 9,119 US theatrical releases (1970–2026) held in SQLite. A Streamlit page (`app.py`) owns one `MayaSession` per browser session. Each user message is one fresh LangGraph invocation: guardrails sanitize the text, the Router (temperature 0 LLM, Pydantic-validated) proposes an Intent and filters, deterministic code decides what to do with it (funnel, probe, pivot, refuse, or retrieve), retrieval is either exact SQL or dense+FTS5 fused by RRF, and the Synthesizer writes a grounded reply. Every node records into a Trace that is mirrored to Langfuse. The same graph is driven headless by the Evaluation Harness, and visitor Feedback (Rating or Report) is posted as comments on a GitHub issue.

## Components

| component (dir) | responsibility | depends on | depended on by | notes |
|---|---|---|---|---|
| `app.py` | Streamlit entry; view switch Chat / Evals / Traces / Feedback | src/ui | — | module-level script, no `main` |
| `src/ui` | Views, Experimentation Lab sidebar, `MayaSession` (builds and invokes the graph, owns turn log) | feedback, indexing, maya, domain, evals, storage, observability, graph, retrieval | app.py | only place `st.*` is called |
| `src/graph` | `build_maya_graph` wires nodes + conditional edges; `MayaGraphState` schema | domain, maya, indexing, observability, retrieval | ui, evals, maya | 518-loc closure holds all 11 node/edge functions |
| `src/maya` | Router, Synthesizer, prompts, probing/funnel policy, guardrails | domain, indexing, graph | graph, evals, ui | all LLM-facing components live here |
| `src/retrieval` | `HybridRetrievalEngine`: SQL path, dense + BM25 → RRF, optional flashrank rerank | domain, indexing, storage | graph, evals, ui | records dense loss instead of failing |
| `src/indexing` | embedding providers (fastembed local, OpenRouter HTTP), ChromaDB `MovieVectorStore`, collection naming | domain | maya, ui, scripts, evals, retrieval, graph, feedback | one collection per (embedder, version) |
| `src/storage` | SQLite `MovieDatabase`: Movie Records, FTS5 BM25, superlatives, person lookup, weekly budget ledger | domain | retrieval, ui, scripts, evals | single `_get_connection` seam |
| `src/domain` | Pydantic models only: `ExperimentConfig`, `MovieRecord`, `ConversationState`, `UserSessionPreferences`, routing decision, reducers | pydantic + stdlib | everything | ADR 0006 holds: no repo imports |
| `src/observability` | `DualModeObservabilityManager` (Langfuse `CallbackHandler` + local ring), `trace_fetch` | — | graph, ui, evals | fail-open when keys absent |
| `src/evals` | `BenchmarkRunner`, LLM judge, IR metrics, run identity, one-factor sweeps | domain, maya, indexing, retrieval, storage, graph, observability | ui | CLI `main` and Evals view share `_run_one` |
| `src/feedback` | `FeedbackStore` (SQLite), GitHub inbox comments, Langfuse scores | indexing | ui | no LLM |
| `scripts` | TMDB ingestion, vector index build, dense benchmark matrix | indexing, storage, domain | — | run by hand |
| `prototypes` | standalone Streamlit trace-inspector prototype | — | — | imported by nothing; duplicates the `DualModeObservabilityManager` name |

**Dependency cycles:** `src/maya` ↔ `src/graph`. `src/graph/orchestrator.py` imports router, agent, guardrails, probing; `src/maya/agent.py:24` imports `SynthesisUsage` from `src/graph/state.py`. Moving that one value type into `src/domain` breaks the cycle.

## Key flows

### Chat turn — entry `src/ui/session.py::MayaSession.turn`
```
MayaSession.turn
├─ ensure_graph → _build_graph → build_maya_graph        # rebuilt only when the config signature changes
├─ DualModeObservabilityManager.new_turn_trace           # per-turn trace id, later joined to Feedback
├─ graph.invoke(...)                                     # LangGraph runtime; NOT a graph edge — nodes below
│   guard_input_node → InjectionFilter.inspect, SessionTokenLimiter.check_current, WeeklyBudgetTracker.current_verdict
│   route_node       → MayaRouter.route (ChatOpenAI.invoke) → _normalize_decision | _heuristic_fallback
│   funnel_node      → next_funnel_step | handle_probe_answer | match_genre_pick   (deterministic)
│   retrieve_node    → HybridRetrievalEngine.retrieve
│                       → _use_sql_path ? _retrieve_sql : _retrieve_dense + _retrieve_bm25 → _rrf_fuse → _rerank?
│   synthesize_node  → MayaSynthesizer.synthesize (ChatOpenAI.invoke) → SessionTokenLimiter.record, WeeklyBudgetTracker.record
│   probe | pivot | refusal → deterministic text, END
├─ _build_turn_row → preference_chips                    # one atomic row from graph output only
└─ ConversationState.add_turn → merge_preferences        # preference reducer
```
Edge order among nodes is set by `route_after_guard`, `route_after_funnel`, `route_after_router` (`src/graph/orchestrator.py:507-556`); `docs/orchestrator-flow.md` has the per-scenario walkthrough.

### Evaluation run — entries `src/evals/runner.py::main` (CLI) and `src/ui/evals_tab.py::render_evaluate_current` (UI)
```
main
├─ load_dataset, load_dataset_version
├─ sweep_configs → ExperimentConfig.apply_preset [low: name shared with MayaSession.apply_preset]
├─ _run_one
│  ├─ BenchmarkRunner.run_routing   → MayaRouter.route → routing_accuracy
│  ├─ _engine_for                   → HybridRetrievalEngine(provider_from_profile, collection_name)
│  ├─ BenchmarkRunner.run_retrieval → HybridRetrievalEngine.retrieve → _refuse_dense_loss → _ir_result
│  └─ BenchmarkRunner.run_full      → build_maya_graph → graph.invoke → MayaJudge.judge_faithfulness / judge_relevancy
├─ BenchmarkRunner.save             → evals/results/*.json
└─ _push_langfuse
```

### Ingestion — entries `scripts/ingest_tmdb.py::ingest_tmdb`, then `scripts/build_vector_index.py::rebuild_index`
```
ingest_tmdb → fetch_discovery_page / hydrate_movie_details (TMDB HTTP) → MovieDatabase.upsert_movies_bulk
rebuild_index → MovieDatabase.get_all_movies → MovieVectorStore.index_movies
                 → MovieRecord.to_dense_text → EmbeddingProvider.embed [low: fastembed | OpenRouter] → chroma upsert
```

### Feedback — entries `src/ui/chat_tab.py::render_feedback` (Rating) and `render_chat` → `parse_feedback_command` (Report)
```
MayaSession.record_feedback → FeedbackStore.record [low] → format_rating_comment → post_inbox_comment (GitHub) → push_feedback_score (Langfuse)
MayaSession.record_report   → validate_report → format_report_comment → post_inbox_comment → push_report_comment
render_feedback_view        → fetch_inbox_comments → parse_inbox_comment; fetch_issue_state per promoted Report
```

## Load-bearing symbols

| symbol | direct callers | breaks if changed | test files in reach |
|---|---|---|---|
| `src/domain/memory.py::UserSessionPreferences` | 11 | funnel, router extraction, reducers, chips | 27 |
| `src/domain/config.py::ExperimentConfig` | 5 (133 transitive) | every tunable, presets, eval run identity, sidebar | 27 |
| `src/storage/database.py::MovieDatabase._get_connection` | 13 | every SQL read/write: FTS5, superlatives, upserts, budget ledger | 24 |
| `src/graph/orchestrator.py::build_maya_graph` | 2 | session turns and full-pipeline evals | 19 |
| `src/domain/memory.py::merge_preferences` | 5 | mood/genre carry-over rules | 11 |
| `src/retrieval/hybrid_engine.py::HybridRetrievalEngine.retrieve` | 3 | retrieve node, retrieval evals | 7 |
| `src/maya/router.py::MayaRouter.route` | 3 | route node, funnel signal extraction, routing evals | 7 |
| `src/indexing/embeddings.py::provider_from_profile` | 6 | embedder selection and collection naming | 6 |
| `src/observability/tracer.py::DualModeObservabilityManager.record_local` | 9 | every node's local Trace | 3 |
| `src/indexing/vector_store.py::MovieVectorStore.search` | 2 real (parser says 15) | dense retrieval and the dense benchmark | 7 |

## Boundaries (external calls)

| symbol | calls out to | kind |
|---|---|---|
| `MovieDatabase._get_connection`, `FeedbackStore.__init__` | `sqlite3.connect` on `data/tmdb_movies.db` and the feedback store | db |
| `MovieVectorStore.__init__` / `.search` / `.index_movies` | chromadb `PersistentClient`, `query`, `upsert` | db (vector) |
| `FastembedProvider`, `MovieVectorStore.get_embedder` | fastembed `TextEmbedding` (local model) | sdk |
| `OpenRouterEmbeddingProvider.embed` | OpenRouter embeddings endpoint | http |
| `MayaRouter.__init__`, `MayaSynthesizer.__init__`, `MayaJudge.__init__` | `ChatOpenAI` → `.invoke` | http (LLM) |
| `HybridRetrievalEngine._rerank` | flashrank `Ranker` (lazy) | sdk |
| `DualModeObservabilityManager.*`, `trace_fetch.fetch_trace_tree`, `langfuse_score.*`, `runner._push_langfuse` | langfuse `get_client`, `CallbackHandler` | http |
| `inbox.post_inbox_comment` / `fetch_inbox_comments` / `fetch_issue_state` | GitHub REST, needs `GITHUB_TOKEN` | http |
| `scripts/ingest_tmdb.py::fetch_discovery_page` / `hydrate_movie_details` | TMDB API | http |
| `sidebar_lab.available_model_ids` | `urlopen` model list | http |
| `build_maya_graph`, `MayaSession.turn`, `BenchmarkRunner.run_full` | langgraph `StateGraph.compile`, `graph.invoke` | sdk (dynamic dispatch) |
| `src/ui/*` | streamlit `st.*` | ui |
| `BenchmarkRunner.save`, `evals_tab._load_run` | `evals/results/*.json` | fs |

## Confidence and gaps

- Call edges (source graph, tests excluded): 275 high / 214 medium / 43 low. Low edges involve `count` (18: `MovieDatabase`, `MovieVectorStore`, `FeedbackStore`, `TokenCounter`, chroma collection), `apply_preset` (6: `ExperimentConfig` vs `MayaSession`), `record` (6: `SessionTokenLimiter`, `WeeklyBudgetTracker`, `FeedbackStore`), `embed` / `token_counter` (9: `EmbeddingProvider` protocol plus two implementations), `DualModeObservabilityManager` (4: prototype duplicate).
- False positives: the parser lists 15 callers of `MovieVectorStore.search`; 13 are `re.search` or compiled-pattern `.search` in `probing.py`, `router.py`, `guardrails.py`, `inbox.py`, `embeddings.py` (verified by grep). Real callers are `HybridRetrievalEngine._retrieve_dense` and `scripts/benchmark_dense_matrix.py::measure_cell`. Treat any fan-in figure for a name that collides with a stdlib method the same way.
- Not in the graph: LangGraph dispatch. `add_node` registration and `graph.invoke` are external calls, so there is no edge from `MayaSession.turn` or `BenchmarkRunner.run_full` to the node functions. The Chat-turn stack above splices them in from `orchestrator.py` source.
- Skipped files: 0. Entrypoint heuristic found none because `app.py` is a module-level script and every `main` runs under `if __name__`.
- Inferred, not observed: the `kind` column in Boundaries (which SDK each external name belongs to) comes from module docstrings and imports, not the graph. Test-file counts come from the tests-inclusive graph (105 files, 3,924 edges, 1,526 low: test helpers reuse names such as `_graph` and `_router`, so trust the file counts, not per-symbol test edges).
- Tests on disk: 25 unit, 15 adversarial, 15 integration (11 `*_live.py` hit real LLMs).
