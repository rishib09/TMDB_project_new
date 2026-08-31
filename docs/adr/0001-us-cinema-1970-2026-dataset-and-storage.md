# 1. US Cinema 1970–2026 Dataset & Bundled Storage

We are ingesting and structuring US theatrical releases from **1970 to 2026** via the TMDB API v3, bundling the normalized dataset into local SQLite (`data/tmdb_movies.db`) and JSON cache (`data/tmdb_movies.json`) alongside an FTS5 lexical search index.

### Why this decision was made:
- Spanning 1970–2026 covers 56 years of cinema, enabling cross-decade queries, modern blockbuster benchmarks (*Inception*, *Oppenheimer*, *The Dark Knight*), and historical trends.
- Ingesting offline and bundling the SQLite database into the repository ensures that the Streamlit application on Hugging Face Spaces boots instantly with zero external TMDB API runtime dependencies and complete immunity to 429 rate limits.
- SQLite + FTS5 indexes allow deterministic SQL queries for superlatives (highest grossing, top rated, runtime extremes) and exact actor/director filters to execute in < 5ms without vector drift.
