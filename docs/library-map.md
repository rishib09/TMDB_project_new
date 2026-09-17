# Library map — what the installed packages already cover

Read this before the **library check** step of the ticket ritual (AGENTS.md).
Versions are the exact pins in `requirements.txt`; verify any row with an
import in `.venv` before relying on it. Rows marked *stays hand-written* are
domain rules an ADR requires in code, not gaps in the libraries.

Sources: the LangGraph and LangChain audits of 2026-09-16 (map #81, decisions
D16–D19) against docs.langchain.com and the installed packages.

## Orchestration and turn state — LangGraph 1.2.11

| Capability | Installed API | Stays hand-written |
|---|---|---|
| Carry state between turns | `graph.compile(checkpointer=InMemorySaver())` (`langgraph.checkpoint.memory`) + `config={"configurable": {"thread_id": ...}}` on every call; read back with the compiled graph's `get_state` and `get_state_history` | the per-turn scratch reset node and the history trim node |
| Accumulate lists and merge preferences | `Annotated[..., reducer]` on the state schema (`langgraph.graph`) | the merge rules themselves (`merge_preferences`) |
| Route from inside a node | return `Command(update=..., goto=...)` (`langgraph.types`) — never beside a static edge from the same node | the routing rules |
| Fan-out over a list | `Send` (`langgraph.types`) | — |
| Handle a node failure after retries | `add_node(..., error_handler=...)` (1.2+) | the schema-failure → deterministic-ask rule |
| Trim message history | `RemoveMessage` (`langchain_core.messages`), `REMOVE_ALL_MESSAGES` (`langgraph.graph.message`); the docs' `langgraph.types` import does not exist on 1.2.11 | window size (config) |
| Stream tokens | `graph.stream(..., stream_mode="messages")`; tag a model `"nostream"` to exclude it | — |
| Pause for a human | `interrupt()` / `Command(resume=)` — **not used here** (D18: a resumed thread bypasses the guards and re-runs the node) | the clarifying question as a message |
| Cross-session memory | `InMemoryStore` (`langgraph.store.memory`) — not in scope | — |

## Model calls, prompts, schemas — LangChain 1.3.18, langchain-core 1.6.1, langchain-openai 1.6.0

| Capability | Installed API | Stays hand-written |
|---|---|---|
| Build a chat client | `init_chat_model(model, model_provider="openai", base_url=, api_key=, max_tokens=, timeout=, max_retries=, reasoning_effort=)` (`langchain.chat_models`); one factory, used by router, synthesizer, judge. On `ChatOpenAI` directly the field is `request_timeout`, not `timeout` | model ids in `ExperimentConfig` (ADR 0004) |
| Closed vocabularies (mood, audience, the 19 genres, era labels) | `Literal[...]` fields and `field_validator` on the pydantic schema (pydantic 2.13.5); the model can only answer inside the set | the mapping of era labels to years (config knobs) |
| Retries on API errors | the client's `max_retries` (429, 5xx, connection) — **never a node retry policy** (D17) | — |
| Structured output | `model.with_structured_output(schema, method="json_schema", include_raw=True)` | every invariant re-derived in code (ADR 0005) |
| Prompt with history | `ChatPromptTemplate.from_messages([("system", ...), MessagesPlaceholder("history"), ("human", ...)])` (`langchain_core.prompts`); pass movie blocks as values, never inside the template literal | prompt text and its composition |
| Token usage per turn | `get_usage_metadata_callback()` / `UsageMetadataCallbackHandler` (`langchain_core.callbacks`) | cost table per model |
| Rate limiting | `rate_limiter=InMemoryRateLimiter(requests_per_second=...)` on the model (`langchain_core.rate_limiters`) | — |
| Token-aware history window | `trim_messages(strategy="last", token_counter="approximate")` (`langchain_core.messages`) | — |
| Tools with state access | `@tool`, `ToolRuntime` (`langchain.tools`); `create_retriever_tool(..., response_format="content_and_artifact")` (`langchain_core.tools`) | tool bodies: closed world, era mapping, exclusions |
| Agent loop with middleware | `create_agent`, `@before_model(can_jump_to=["end"])`, `SummarizationMiddleware`, `@dynamic_prompt` (`langchain.agents`, `langchain.agents.middleware`) — Level B, prototype only (#85) | guard rules, token cap |
| Per-node spans and usage for the Trace | `BaseCallbackHandler` (`langchain_core.callbacks`): `on_chain_start/end`, `on_llm_end` | the trace ring, the Langfuse read path |
| Test doubles | `GenericFakeChatModel`, `FakeListChatModel` (`langchain_core.language_models.fake_chat_models`); `DeterministicFakeEmbedding` (`langchain_core.embeddings`) | — |
| LLM response cache | `set_llm_cache` exists — **forbidden** without latency evidence (ADR 0009); global and cross-user. The factory sets `cache=False` per client so a global cache can never be switched on by accident | — |

## Retrieval, indexing, storage — raw libraries by decision (D19)

| Capability | Installed API | Stays hand-written |
|---|---|---|
| Dense vectors | `chromadb.PersistentClient`, `collection.query(where=...)` (chromadb 1.5.9) | the `where` clause on the dense leg is #88 |
| Sparse leg | SQLite FTS5 `bm25()` (stdlib `sqlite3`) | `rank-bm25` is pinned but unused |
| Fusion | reciprocal-rank arithmetic in `HybridRetrievalEngine` | equals LangChain's ensemble retriever, which lives in `langchain-classic` (not installed) |
| Reranking | `flashrank.Ranker` (0.2.10) | LangChain's wrapper lives in `langchain-community` (not installed, being sunset) |
| Local embeddings | `fastembed.TextEmbedding` (0.8.0) | `HuggingFaceEmbeddings` needs PyTorch (ADR 0003 says no) |
| Cloud embeddings | `openai` client against OpenRouter inside the ADR 0008 seam; `OpenAIEmbeddings(base_url=, check_embedding_ctx_length=False)` (langchain-openai) verified to return identical vectors and may carry the transport only | the packing window, token counter, window correction and free-tier backoff stay in the seam |
| SQL filters and superlatives | `sqlite3` in `MovieDatabase` | runtime and rating predicates are missing (#88) |

## Evaluation and observability

| Capability | Installed API | Stays hand-written |
|---|---|---|
| Judges | `with_structured_output` graders (as the LangSmith RAG tutorial does) | `openevals`, `agentevals` not installed |
| IR metrics | pure functions in `src/evals/metrics.py` | nothing in LangChain |
| Cloud tracing | `langfuse.langchain.CallbackHandler` (langfuse 4.15.1) | one handler, passed in `config["callbacks"]` |
| Multi-turn goldens | one `thread_id` per conversation, `graph.get_state` for scoring (#93) | — |
