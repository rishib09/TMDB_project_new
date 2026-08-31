Type: research
Status: resolved

## Question

What is the comprehensive intent taxonomy and Pydantic/JSON validation schema for the "Maya" agent (distinguishing `GREETING`, `CHITCHAT`, `CAPABILITIES`, `SEMANTIC_SEARCH`, `ATTRIBUTE_FILTER`, `SUPERLATIVE_RANKING`, `NEGATION_EXCLUSION`, and `OUT_OF_SCOPE`), and how should we structure the deterministic temperature=0 OpenRouter router call (e.g. via `meta-llama/llama-3.2-3b-instruct` or `google/gemini-2.0-flash-lite`) to guarantee structured outputs with zero hallucination and strict context constraints?

## Answer

### 1. 7-Class Intent Taxonomy
- `GREETING`: Pleasantries / small talk -> Instant static template response (<20ms), zero retrieval.
- `CAPABILITIES`: Inquiries about Maya's persona, features, and 1970 TMDB bounds -> Instant capabilities response.
- `SEMANTIC_SEARCH`: Concept / theme / plot queries -> Hybrid vector + BM25 search over enriched overviews.
- `ATTRIBUTE_FILTER`: Specific metadata constraints (director, actor, genre, runtime, rating) -> Deterministic SQL filter + vector re-ranking.
- `SUPERLATIVE_RANKING`: Extremas / Top-N queries (highest grossing, longest runtime, top rated) -> Direct deterministic SQL `ORDER BY ... DESC LIMIT N`.
- `NEGATION_EXCLUSION`: Exclude genres/actors/themes ("not comedies", "no gore") -> SQL `NOT LIKE` + negative vector penalty.
- `OUT_OF_SCOPE`: Non-movie questions or movies from other years -> Polite pivot explaining 1970 US TMDB boundaries.

### 2. Deterministic Router Implementation
- **Pydantic Schema (`src/maya/schemas/routing.py`)**: `QueryRoutingDecision` (fields: `intent: IntentType`, `confidence: float`, `reasoning: str`, `semantic_search_query: Optional[str]`, `filters: Optional[MetadataFilterCriteria]`, `superlative: Optional[SuperlativeCriteria]`, `negative_themes: List[str]`, `is_temporal_mismatch: bool`, `target_year_mentioned: Optional[int]`, `requires_rag: bool`).
- **Primary Model**: `meta-llama/llama-3.2-3b-instruct` ($0.02 / 1M tokens, ~200ms latency) with `strict: true` JSON schema, temperature=0.0, seed=42.
- **Fallback Model**: `google/gemini-2.0-flash-lite` with heuristic safety fallback.

### 3. Strict Grounding & Anti-Hallucination Constraints
- Maya Synthesis LLM operates under the **Closed-World Assumption (CWA)**.
- Movies are injected in `<retrieved_movies>` XML context.
- System prompt strictly bans naming any movie not present in `<retrieved_movies>`, requires exact metadata citation with TMDB IDs, and generates Markdown poster images: `![Poster](https://image.tmdb.org/t/p/w500{poster_path})`.
