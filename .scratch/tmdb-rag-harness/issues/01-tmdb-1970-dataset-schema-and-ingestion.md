Type: research
Status: resolved

## Question

How should we query, filter, normalize, and cache all 1970 US release movies from the TMDB API (including movie IDs, titles, overviews, release dates, genres, directors/crew, top cast, runtime, vote stats, and poster image URLs) into an efficient, standalone local SQLite/DuckDB + JSON store that can be bundled and run seamlessly on Hugging Face Spaces without runtime API rate-limiting or latency bottlenecks?

## Answer

### 1. Ingestion Strategy & Endpoints
- **Stage 1 (Discovery)**: `GET https://api.themoviedb.org/3/discover/movie?primary_release_year=1970&region=US&sort_by=popularity.desc&include_adult=false&page={page}`. Paginates through all ~40-50 pages to collect ~800 candidate movie IDs.
- **Stage 2 (Hydration)**: `GET https://api.themoviedb.org/3/movie/{movie_id}?append_to_response=credits,keywords` with `asyncio.Semaphore(10)` concurrency. Hydrates runtime, budget, revenue, directors, top 10 billed cast, and plot keywords in a single call per movie.

### 2. Normalized Data Models & Storage
- **Pydantic Model (`src/domain/movie.py`)**: `MovieRecord` containing `id`, `title`, `overview`, `release_date`, `genres`, `runtime`, `vote_average`, `vote_count`, `popularity`, `director`, `directors`, `cast: List[CastMember]`, `budget`, `revenue`, `keywords`, `poster_path`, `backdrop_path`, `poster_url`, and `to_rag_document_text()`.
- **Relational Storage (`data/tmdb_1970.db`)**: SQLite table `movies` with indexes on `release_date`, `vote_average`, `runtime`, `revenue`, `director` + `movies_fts` virtual FTS5 table for BM25 lexical keyword matching.
- **Static Asset URLs**: Posters rendered as `https://image.tmdb.org/t/p/w500{poster_path}` (with SVG placeholder fallback for missing posters).
- **Hugging Face Spaces Zero-Dependency Bundling**: Ingestion runs offline; `data/tmdb_1970.db` (~2.5 MB) and `data/tmdb_1970_movies.json` (~1.5 MB) are committed to git, allowing HF Spaces to start instantly with zero TMDB API rate-limiting or external key dependencies.
