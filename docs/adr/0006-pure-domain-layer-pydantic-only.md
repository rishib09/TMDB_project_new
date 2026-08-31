# 6. Pure Domain Layer: Pydantic-Only Models with Zero Framework Imports

We are enforcing that `src/domain/` contains **pure Pydantic domain models only** — no imports from LangGraph, LangChain, or any other framework. Framework-shaped adapters (state schemas carrying LangGraph reducer semantics) live next to the orchestration code that uses them, not in the domain layer.

### Why this decision was made:

- **The domain layer is the stable core.** Business concepts (movies, routing decisions, memory state, session preferences) outlive any particular orchestration framework. If the pipeline ever moves off LangGraph, `src/domain/` must remain untouched — its value is precisely that it depends on nothing but Pydantic and the standard library.
- **Dependency arrows must point inward.** Orchestration (`src/maya/`, `src/graph/`) depends on the domain; the domain depends on nothing outward. A framework import inside `src/domain/` inverts this: the stable layer becomes hostage to a volatile external library's API churn.
- **The violation already exists.** Commit `1e30f12` (issue #2) introduced `MayaGraphState` into `src/domain/memory.py` with `from langchain_core.messages import BaseMessage` and `from langgraph.graph.message import add_messages`, plus `Annotated[..., reducer]` annotations that encode LangGraph execution semantics inside type hints. This is accepted as interim state and must be relocated in issue #5, which is the natural home: the graph state schema belongs next to the graph that compiles it.
- **Nothing is lost by separating.** `MayaGraphState` shares its underlying types with the domain (`UserSessionPreferences`, `QueryRoutingDecision`, `MovieRecord`) — moving it to a framework-adjacent module (e.g. `src/graph/state.py`) duplicates no logic; only the LangGraph-shaped *view* of the domain moves.

### Rules going forward:

1. `src/domain/` may import only: `pydantic`, the standard library, and other `src/domain/` modules.
2. LangGraph reducers, `Annotated[...]` state annotations, `BaseMessage`, and any other framework types live in orchestration modules (`src/maya/`, `src/graph/`, `src/observability/`).
3. A `ruff` guard (import restrictions on `src/domain/`) should enforce rule 1 mechanically; until then, code review enforces it.
4. Domain models communicate intent through plain Pydantic validation and pure functions only.

### Consequences:

- Issue #5 must relocate `MayaGraphState`, `merge_unique_ids`, and `merge_preferences` out of `src/domain/memory.py` into the graph module, restoring the domain layer to zero framework imports (verified by the existing domain test suite plus an import-lint check).
- Future issues must not add framework imports under `src/domain/`; reviewers and agents should treat any such import as a defect.
- The domain remains portable and unit-testable in isolation, keeping the fast offline test tier (<2s) framework-free.
