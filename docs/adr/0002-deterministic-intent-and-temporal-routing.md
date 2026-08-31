# 2. Deterministic Intent & Temporal Routing with Strict XML Grounding

We are routing all incoming user queries through a deterministic temperature=0.0 OpenRouter model (`meta-llama/llama-3.2-3b-instruct`) with strict Pydantic JSON schema validation, categorizing queries into a 7-class taxonomy (`GREETING`, `CAPABILITIES`, `SEMANTIC_SEARCH`, `ATTRIBUTE_FILTER`, `SUPERLATIVE_RANKING`, `NEGATION_EXCLUSION`, `OUT_OF_SCOPE`), and grounding Maya's synthesis within a `<retrieved_movies>` XML context.

### Why this decision was made:
- Separating intent routing from response generation ensures that non-RAG queries (greetings, capabilities) answer instantly (< 20ms) and superlatives/filters route to exact deterministic SQL queries rather than fuzzy vector approximations.
- Temporal detection classifies pre-1970 films (e.g., 1939 *Wizard of Oz*, 1968 *2001: A Space Odyssey*) as out-of-scope while allowing 1970–2026 queries to retrieve seamlessly.
- Enforcing the Closed-World Assumption (CWA) via `<retrieved_movies>` XML injection prevents Maya from hallucinating non-existent movies, fictional box office stats, or incorrect cast members.
