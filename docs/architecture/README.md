# Architecture diagrams

Generated from `.codemap/graph.json` (git_head f07c9eb, 2026-09-13) with the `codebase-uml` skill. Every diagram is emitted by `graph_to_mermaid.py`, not hand-drawn; the reading notes under each are the added interpretation. `CODEMAP.md` at the repo root is the prose companion.

| file | view | question it answers |
|---|---|---|
| [package-src.md](package-src.md) | package | which packages import which, and where the one cycle is |
| [class-src.md](class-src.md) | class (all of `src/`, methods hidden) | which classes use which; who the composition roots are |
| [sequence-mayasession-turn.md](sequence-mayasession-turn.md) | sequence | what `MayaSession.turn` does around the LangGraph run |
| [sequence-retrieve-node.md](sequence-retrieve-node.md) | sequence | what the retrieve node does down to SQLite and ChromaDB |
| [routing-v1.md](routing-v1.md) | flow (hand-authored, cited) + sequence (generated) | how a message is routed in v1: guard, route, funnel, probe, retrieve; where filters reach SQLite and ChromaDB; the 15-gate inventory |

## Regenerate

```bash
python <codebase-map>/scripts/build_map.py .
python <codebase-uml>/scripts/graph_to_mermaid.py .codemap/graph.json package
python <codebase-uml>/scripts/graph_to_mermaid.py .codemap/graph.json class --scope src --no-methods
python <codebase-uml>/scripts/graph_to_mermaid.py .codemap/graph.json sequence MayaSession.turn --depth 4
python <codebase-uml>/scripts/graph_to_mermaid.py .codemap/graph.json sequence build_maya_graph.retrieve_node --depth 3
```

Set `PYTHONIOENCODING=utf-8` on Windows; the scripts print box-drawing characters.

## Known parser limits that affect these diagrams

- LangGraph node dispatch (`add_node`, `graph.invoke`) is not a call edge, so the turn sequence stops at the graph boundary and node-level sequences must be requested by node name.
- Name collisions produce false or fan-out edges: `re.search` resolves to `MovieVectorStore.search`; `embed` / `token_counter` / `count` / `record` / `apply_preset` fan out to every definition. Each affected file lists them under **Confidence**.
