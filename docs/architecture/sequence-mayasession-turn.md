# Sequence — one chat turn, entry `MayaSession.turn`

Depth-first call order from `src/ui/session.py::MayaSession.turn` to depth 4, generated from `.codemap/graph.json`. The LangGraph run itself (`graph.invoke`) is an external call and sits between steps 24 and 25; see [sequence-retrieve-node.md](sequence-retrieve-node.md) for the inside of the retrieve node.

```mermaid
sequenceDiagram
  autonumber
  participant MayaSession as MayaSession
  participant embeddings_py as embeddings.py
  participant FastembedProvider as FastembedProvider
  participant OpenRouterEmbeddingProvider as OpenRouterEmbeddingProvider
  participant HybridRetrievalEngine as HybridRetrievalEngine
  participant session_py as session.py
  participant MovieVectorStore as MovieVectorStore
  participant orchestrator_py as orchestrator.py
  participant SessionTokenLimiter as SessionTokenLimiter
  participant InjectionFilter as InjectionFilter
  participant OffTopicPivot as OffTopicPivot
  participant MayaRouter as MayaRouter
  participant MovieDatabase as MovieDatabase
  participant MayaSynthesizer as MayaSynthesizer
  participant DualModeObservabilityManager as DualModeObservabilityManager
  participant probing_py as probing.py
  participant ConversationState as ConversationState
  participant ChatMessage as ChatMessage
  participant memory_py as memory.py
  participant FocusedMovieEntity as FocusedMovieEntity
  participant UserSessionPreferences as UserSessionPreferences
  Note over MayaSession: entry: MayaSession.turn
  MayaSession->>MayaSession: ensure_graph()
  MayaSession->>MayaSession: _graph_signature()
  MayaSession->>MayaSession: _build_graph()
  MayaSession->>embeddings_py: collection_name()
  MayaSession->>embeddings_py: provider_from_profile()
  embeddings_py->>FastembedProvider: FastembedProvider()
  embeddings_py->>OpenRouterEmbeddingProvider: OpenRouterEmbeddingProvider()
  MayaSession->>HybridRetrievalEngine: HybridRetrievalEngine()
  MayaSession->>session_py: shared_vector_store()
  session_py->>MovieVectorStore: MovieVectorStore()
  MayaSession->>orchestrator_py: build_maya_graph()
  orchestrator_py->>SessionTokenLimiter: SessionTokenLimiter()
  orchestrator_py->>InjectionFilter: InjectionFilter()
  orchestrator_py->>OffTopicPivot: OffTopicPivot()
  MayaSession->>MayaRouter: MayaRouter()
  MayaSession->>MovieDatabase: distinct_genres()
  MovieDatabase->>MovieDatabase: _get_connection()
  MayaSession->>MayaSynthesizer: MayaSynthesizer()
  MayaSession->>DualModeObservabilityManager: traces()
  MayaSession->>DualModeObservabilityManager: new_turn_trace()
  DualModeObservabilityManager->>DualModeObservabilityManager: _init_cloud_handler()
  MayaSession->>DualModeObservabilityManager: record_local()
  MayaSession->>session_py: _to_lc_messages()
  MayaSession->>DualModeObservabilityManager: callbacks()
  MayaSession->>DualModeObservabilityManager: metadata()
  MayaSession->>MayaSession: _build_turn_row()
  MayaSession->>MayaSession: _path_taken()
  MayaSession->>probing_py: preference_chips()
  MayaSession->>MayaSession: _filter_chips()
  MayaSession->>session_py: slice_new_traces()
  MayaSession->>ConversationState: add_turn()
  ConversationState->>ChatMessage: ChatMessage()
  ConversationState->>memory_py: merge_unique_ids()
  ConversationState->>FocusedMovieEntity: FocusedMovieEntity()
  ConversationState->>UserSessionPreferences: UserSessionPreferences()
  ConversationState->>memory_py: merge_preferences()
  memory_py->>UserSessionPreferences: UserSessionPreferences()
```

## Reading notes

- **Steps 1–18 are the lazy graph build.** `ensure_graph` rebuilds only when `_graph_signature` (the Experiment Config) changes. A steady-state turn starts at step 19.
- **The whole LangGraph run is invisible here.** `graph.invoke` runs guard → route or funnel → retrieve → synthesize between `metadata()` (step 24) and `_build_turn_row` (step 25). The parser cannot follow `add_node` registration, so no node appears in this sequence.
- **Step 21 (`record_local`) is conditional.** It fires only for a recalled query (`recalled=True`); the diagram cannot show the branch.
- **Steps 6 and 7 are alternatives.** `provider_from_profile` constructs one embedding provider per config, not both.
- **The turn ends in the reducer.** `ConversationState.add_turn` → `merge_preferences` (steps 30–36) decides which preferences survive to the next turn. This is where mood changes drop accumulated genres.

## Confidence

- Low-confidence edges shown: 0.
- Omitted: `graph.invoke` and every node inside it; Langfuse calls behind `new_turn_trace`, `callbacks`, `metadata`; Streamlit rendering, which happens in `render_chat` after this method returns.
- Graph `git_head`: f07c9eb.
