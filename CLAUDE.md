# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**AGENTS.md is the single source of truth for how to work in this repo.** Read it and follow it fully — the ticket ritual (approval before code), the three-tier testing protocol, frozen `requirements.txt`, run commands, and the standing rule (verify existence + get permission before creating anything).

Also in force:

- **CONTEXT.md** — ubiquitous language. Use its terms exactly; avoid the listed forbidden synonyms.
- **docs/library-map.md** — which installed package API covers each capability. Read it before the library check in the ticket ritual; reimplement only with a stated reason.
- **docs/adr/** — ADRs 0004–0006 govern design: tunables live in `ExperimentConfig` (0004), "model proposes, code disposes" for every LLM-facing component (0005), `src/domain/` stays pure Pydantic + stdlib (0006).

## Orientation

Streamlit app (`app.py`) fronting a RAG pipeline over 9,119 US films (1970–2026) in SQLite (`data/tmdb_movies.db`). Flow: `src/graph/orchestrator.py` routes intent (`src/maya/`) → deterministic SQL (`src/storage/`) or hybrid FTS5+ChromaDB retrieval (`src/retrieval/`, `src/indexing/`) → CWA-grounded synthesis → tracing (`src/observability/`). Evaluation harness in `src/evals/`, UI views in `src/ui/`, ingestion scripts in `scripts/`.

Tests mirror the AGENTS.md tiers: `tests/unit/`, `tests/adversarial/`, `tests/integration/` (`*_live.py` files hit real LLMs and require the dotenvx-wrapped command from AGENTS.md).
