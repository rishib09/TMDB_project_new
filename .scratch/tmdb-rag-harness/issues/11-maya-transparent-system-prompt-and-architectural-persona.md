Type: grilling
Status: resolved

## Question

How should we structure Maya's master system prompt so that her persona explicitly and transparently articulates her internal architectural mechanics (interpreting user intent, query reformulation, short-term vs long-term memory, history compaction, vector embeddings vs deterministic SQL execution, RRF ranking, FlashRank reranking, and strict closed-world historical fact grounding)?

## Answer

### 1. 8-Pillar Meta-Prompt Architecture (`src/maya/prompts.py`)
Maya's master system prompt explicitly encodes 8 transparent architectural pillars:
1. **`[PERSONA & SCOPE]`**: Specialized film curator for US theatrical releases from 1970 to 2026.
2. **`[USER INTENT & INTERPRETATION]`**: Explains how the user prompt was categorized (semantic, attribute filter, superlative, negation).
3. **`[QUERY REFORMULATION & COREFERENCE]`**: Shows how conversational pronouns (*"who directed it?"*) and ellipses were resolved into a standalone query.
4. **`[RETRIEVER CONTEXT]`**: Differentiates deterministic SQLite queries (exact dates/revenues/runtimes) from dense vector embeddings (thematic plots).
5. **`[RANKING & FUSION]`**: Transparently notes Reciprocal Rank Fusion (RRF) and FlashRank cross-encoder reranking.
6. **`[HISTORICAL FACT INTEGRITY & CWA GROUNDING]`**: Strict Closed-World Assumption—never fabricates unretrieved movies, actors, or box office stats.
7. **`[MEMORY & USER PREFERENCES]`**: Details active entity focus and persistent session exclusions (*"no horror"*).
8. **`[OUTPUT FORMATTING & VISUAL POSTERS]`**: Formats metadata cards with TMDB IDs and generates markdown posters (`![Poster](url)`).
