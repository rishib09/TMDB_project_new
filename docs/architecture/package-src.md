# Package dependencies — whole repo

Import edges between top-level packages, generated from `.codemap/graph.json`; edge labels are import counts. `.` is `app.py`.

```mermaid
flowchart LR
  _["."]
  scripts["scripts"]
  src_domain["src/domain"]
  src_evals["src/evals"]
  src_feedback["src/feedback"]
  src_graph["src/graph"]
  src_indexing["src/indexing"]
  src_maya["src/maya"]
  src_observability["src/observability"]
  src_retrieval["src/retrieval"]
  src_storage["src/storage"]
  src_ui["src/ui"]
  src_maya -->|9| src_domain
  src_graph -->|7| src_domain
  _ -->|6| src_ui
  src_evals -->|6| src_domain
  src_evals -->|6| src_maya
  src_ui -->|6| src_feedback
  src_ui -->|6| src_indexing
  scripts -->|5| src_indexing
  src_graph -->|5| src_maya
  src_ui -->|5| src_maya
  scripts -->|4| src_storage
  src_maya -->|4| src_indexing
  src_ui -->|4| src_domain
  src_evals -->|3| src_indexing
  src_ui -->|3| src_evals
  src_indexing -->|2| src_domain
  src_retrieval -->|2| src_domain
  src_storage -->|2| src_domain
  src_ui -->|2| src_storage
  src_ui -->|2| src_observability
  scripts -->|1| src_domain
  src_evals -->|1| src_retrieval
  src_evals -->|1| src_storage
  src_evals -->|1| src_graph
  src_evals -->|1| src_observability
  src_feedback -->|1| src_indexing
  src_graph -->|1| src_indexing
  src_graph -->|1| src_observability
  src_graph -->|1| src_retrieval
  src_maya -->|1| src_graph
  src_retrieval -->|1| src_indexing
  src_retrieval -->|1| src_storage
  src_ui -->|1| src_graph
  src_ui -->|1| src_retrieval
  linkStyle default stroke-width:1px
  %% CYCLES (import both ways): src/graph<->src/maya
```

## Reading notes

- **One cycle: `src/graph` ↔ `src/maya`.** Five imports go graph → maya (router, agent, guardrails, probing); one goes back: `src/maya/agent.py:24` imports `SynthesisUsage` from `src/graph/state.py`. Moving that value type into `src/domain` removes the cycle.
- **`src/domain` is the sink.** Seven packages import it; it imports nothing in the repo. ADR 0006 (pure Pydantic + stdlib) holds.
- **Two composition roots.** `src/ui` imports nine of the ten packages and is the only thing `app.py` touches. `src/evals` imports seven and drives the same components headless.
- **`src/observability` imports nothing in the repo.** It is a pure adapter over Langfuse and an in-memory ring; `graph`, `ui`, and `evals` write into it.
- **`scripts` never reach `maya` or `graph`.** Ingestion and index builds touch only `indexing`, `storage`, `domain`.
- **`prototypes/` is absent.** Nothing imports it and it imports nothing from `src`; it is dead to the dependency graph.

## Confidence

- Low-confidence edges shown: 0 (import edges are resolved by path, always high).
- Omitted: third-party imports (streamlit, langgraph, langchain, chromadb, fastembed, flashrank, langfuse, pydantic) and `tests/`.
- Graph `git_head`: f07c9eb.
