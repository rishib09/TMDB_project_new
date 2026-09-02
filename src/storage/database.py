"""SQLite relational storage and FTS5 full-text lexical search engine for TMDB movies."""

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from src.domain.movie import MovieRecord
from src.domain.routing import MetadataFilterCriteria


class MovieDatabase:
    """High-performance SQLite database manager with FTS5 BM25 search for 1970-2026 US movies."""

    def __init__(self, db_path: str = "data/tmdb_movies.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with WAL mode and row factory enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Initialize database schema, indexes, and FTS5 virtual table."""
        with self._get_connection() as conn:
            # 1. Main Movies Relational Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS movies (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                original_title TEXT,
                tagline TEXT,
                overview TEXT,
                release_date TEXT,
                release_year INTEGER NOT NULL,
                runtime INTEGER,
                vote_average REAL,
                vote_count INTEGER,
                popularity REAL,
                director TEXT,
                budget INTEGER,
                revenue INTEGER,
                imdb_id TEXT,
                poster_path TEXT,
                backdrop_path TEXT,
                genres_json TEXT NOT NULL DEFAULT '[]',
                cast_json TEXT NOT NULL DEFAULT '[]',
                keywords_json TEXT NOT NULL DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 2. Performance Indexes for Superlatives and Filter Criteria
            conn.execute("CREATE INDEX IF NOT EXISTS idx_movies_year ON movies(release_year);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_movies_vote_avg ON movies(vote_average DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_movies_revenue ON movies(revenue DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_movies_director ON movies(director);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_movies_popularity ON movies(popularity DESC);")

            # 3. FTS5 Virtual Table for BM25 Lexical Keyword Search
            conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS movies_fts USING fts5(
                id UNINDEXED,
                title,
                tagline,
                overview,
                director,
                genres,
                cast_names,
                keywords,
                tokenize = 'porter unicode61'
            );
            """)

            # 4. Feedback Telemetry Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                rag_version TEXT NOT NULL,
                rating INTEGER NOT NULL,
                user_comment TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 5. Weekly Budget Tracker Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS budget_tracker (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_str TEXT NOT NULL,
                cost_usd REAL NOT NULL,
                tokens_used INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.commit()

    # --- weekly budget tracker (#8): aggregate API spend -----------------

    def record_budget_entry(
        self, date_str: str, cost_usd: float, tokens_used: int, model_name: str
    ) -> None:
        """Appends one LLM call's estimated cost to the budget_tracker table."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO budget_tracker (date_str, cost_usd, tokens_used, model_name) "
                "VALUES (?, ?, ?, ?)",
                (date_str, cost_usd, tokens_used, model_name),
            )
            conn.commit()

    def weekly_spend_usd(self, reference: date | None = None) -> float:
        """Sum of cost_usd for the ISO week (Mon–Sun) containing ``reference``."""
        ref = reference or date.today()
        week_start = ref - timedelta(days=ref.weekday())
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM budget_tracker "
                "WHERE date_str >= ?",
                (week_start.isoformat(),),
            ).fetchone()
        return float(row[0])

    def upsert_movie(self, movie_data: dict[str, Any]) -> None:
        """Insert or replace a movie and synchronize the FTS5 index."""
        with self._get_connection() as conn:
            genres_list = movie_data.get("genres", [])
            cast_list = movie_data.get("cast", [])
            keywords_list = movie_data.get("keywords", [])

            genres_str = " ".join(genres_list) if isinstance(genres_list, list) else str(genres_list)
            cast_names_str = " ".join([c.get("name", "") if isinstance(c, dict) else str(c) for c in cast_list])
            keywords_str = " ".join(keywords_list) if isinstance(keywords_list, list) else str(keywords_list)

            conn.execute("""
            INSERT OR REPLACE INTO movies (
                id, title, original_title, tagline, overview, release_date,
                release_year, runtime, vote_average, vote_count, popularity,
                director, budget, revenue, imdb_id, poster_path, backdrop_path,
                genres_json, cast_json, keywords_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                movie_data["id"],
                movie_data["title"],
                movie_data.get("original_title", movie_data["title"]),
                movie_data.get("tagline", ""),
                movie_data.get("overview", ""),
                movie_data.get("release_date", ""),
                int(movie_data.get("release_year", 0)),
                movie_data.get("runtime", 0),
                float(movie_data.get("vote_average", 0.0)),
                int(movie_data.get("vote_count", 0)),
                float(movie_data.get("popularity", 0.0)),
                movie_data.get("director", ""),
                int(movie_data.get("budget", 0)),
                int(movie_data.get("revenue", 0)),
                movie_data.get("imdb_id", ""),
                movie_data.get("poster_path", ""),
                movie_data.get("backdrop_path", ""),
                json.dumps(genres_list),
                json.dumps(cast_list),
                json.dumps(keywords_list)
            ))

            # Synchronize FTS5
            conn.execute("DELETE FROM movies_fts WHERE id = ?", (str(movie_data["id"]),))
            conn.execute("""
            INSERT INTO movies_fts (id, title, tagline, overview, director, genres, cast_names, keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(movie_data["id"]),
                movie_data["title"],
                movie_data.get("tagline", ""),
                movie_data.get("overview", ""),
                movie_data.get("director", ""),
                genres_str,
                cast_names_str,
                keywords_str
            ))
            conn.commit()

    def upsert_movies_bulk(self, movies: list[dict[str, Any]]) -> int:
        """Bulk insert or replace movies for fast ingestion."""
        with self._get_connection() as conn:
            for movie_data in movies:
                genres_list = movie_data.get("genres", [])
                cast_list = movie_data.get("cast", [])
                keywords_list = movie_data.get("keywords", [])

                genres_str = " ".join(genres_list) if isinstance(genres_list, list) else str(genres_list)
                cast_names_str = " ".join([c.get("name", "") if isinstance(c, dict) else str(c) for c in cast_list])
                keywords_str = " ".join(keywords_list) if isinstance(keywords_list, list) else str(keywords_list)

                conn.execute("""
                INSERT OR REPLACE INTO movies (
                    id, title, original_title, tagline, overview, release_date,
                    release_year, runtime, vote_average, vote_count, popularity,
                    director, budget, revenue, imdb_id, poster_path, backdrop_path,
                    genres_json, cast_json, keywords_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    movie_data["id"],
                    movie_data["title"],
                    movie_data.get("original_title", movie_data["title"]),
                    movie_data.get("tagline", ""),
                    movie_data.get("overview", ""),
                    movie_data.get("release_date", ""),
                    int(movie_data.get("release_year", 0)),
                    movie_data.get("runtime", 0),
                    float(movie_data.get("vote_average", 0.0)),
                    int(movie_data.get("vote_count", 0)),
                    float(movie_data.get("popularity", 0.0)),
                    movie_data.get("director", ""),
                    int(movie_data.get("budget", 0)),
                    int(movie_data.get("revenue", 0)),
                    movie_data.get("imdb_id", ""),
                    movie_data.get("poster_path", ""),
                    movie_data.get("backdrop_path", ""),
                    json.dumps(genres_list),
                    json.dumps(cast_list),
                    json.dumps(keywords_list)
                ))

                conn.execute("DELETE FROM movies_fts WHERE id = ?", (str(movie_data["id"]),))
                conn.execute("""
                INSERT INTO movies_fts (id, title, tagline, overview, director, genres, cast_names, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(movie_data["id"]),
                    movie_data["title"],
                    movie_data.get("tagline", ""),
                    movie_data.get("overview", ""),
                    movie_data.get("director", ""),
                    genres_str,
                    cast_names_str,
                    keywords_str
                ))
            conn.commit()
        return len(movies)

    def get_by_id(self, movie_id: int) -> MovieRecord | None:
        """Fetch a single movie record by TMDB ID as a typed MovieRecord."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM movies WHERE id = ?", (movie_id,))
            row = cursor.fetchone()
            return MovieRecord.model_validate(self._row_to_dict(row)) if row else None

    def query_superlative(
        self,
        metric: str,
        direction: str = "DESC",
        year: int | None = None,
        genre: str | None = None,
        limit: int = 5
    ) -> list[MovieRecord]:
        """Deterministic SQL superlative ranking query returning typed MovieRecords."""
        metric_map = {
            "REVENUE": "revenue",
            "BUDGET": "budget",
            "RATING": "vote_average",
            "POPULARITY": "popularity",
            "RUNTIME": "runtime",
            "VOTE_COUNT": "vote_count",
        }
        col = metric_map.get(metric.upper(), "vote_average")
        order_dir = "ASC" if direction.upper() == "ASC" else "DESC"

        query = "SELECT * FROM movies WHERE 1=1"
        params: list[Any] = []

        if year:
            query += " AND release_year = ?"
            params.append(year)
        if genre:
            query += " AND genres_json LIKE ?"
            params.append(f"%{genre}%")

        # Exclude zero entries for budget/revenue/runtime superlatives
        if col in ["revenue", "budget", "runtime"]:
            query += f" AND {col} > 0"
        elif col == "vote_average":
            query += " AND vote_count >= 50"  # Exclude obscure 10/10 single-vote noise

        query += f" ORDER BY {col} {order_dir} LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [MovieRecord.model_validate(self._row_to_dict(r)) for r in cursor.fetchall()]

    def search_bm25(self, query_str: str, limit: int = 20) -> list[MovieRecord]:
        """Execute BM25 lexical full-text search against movies_fts returning typed MovieRecords."""
        clean_query = "".join([c if c.isalnum() or c.isspace() else " " for c in query_str]).strip()
        if not clean_query:
            return []

        tokens = clean_query.split()
        fts_expr = " OR ".join(tokens)

        sql = """
        SELECT m.*, bm25(movies_fts) as rank_score
        FROM movies_fts fts
        JOIN movies m ON fts.id = m.id
        WHERE movies_fts MATCH ?
        ORDER BY rank_score ASC
        LIMIT ?
        """
        with self._get_connection() as conn:
            try:
                cursor = conn.execute(sql, (fts_expr, limit))
                return [MovieRecord.model_validate(self._row_to_dict(r)) for r in cursor.fetchall()]
            except sqlite3.OperationalError:
                return []

    def search_metadata_filters(
        self,
        filters: "MetadataFilterCriteria",
        limit: int = 20
    ) -> list[MovieRecord]:
        """Deterministic SQL query for structured metadata filters (issue #4).

        Translates MetadataFilterCriteria into parameterized WHERE clauses.
        Exclusions (excluded_genres / excluded_actors) are enforced here too.
        """
        query = "SELECT * FROM movies WHERE 1=1"
        params: list[Any] = []

        if filters.exact_year is not None:
            query += " AND release_year = ?"
            params.append(filters.exact_year)
        if filters.year_min is not None:
            query += " AND release_year >= ?"
            params.append(filters.year_min)
        if filters.year_max is not None:
            query += " AND release_year <= ?"
            params.append(filters.year_max)
        for genre in filters.genres:
            query += " AND genres_json LIKE ?"
            params.append(f'%"{genre}"%')
        if filters.director:
            query += " AND director LIKE ?"
            params.append(f"%{filters.director}%")
        if filters.cast_member:
            query += " AND cast_json LIKE ?"
            params.append(f'%"{filters.cast_member}%')
        for excluded_genre in filters.excluded_genres:
            query += " AND genres_json NOT LIKE ?"
            params.append(f'%"{excluded_genre}"%')
        for excluded_actor in filters.excluded_actors:
            query += " AND cast_json NOT LIKE ?"
            params.append(f'%"{excluded_actor}"%')

        # Deterministic ordering for filter-only queries (no relevance signal)
        query += " ORDER BY vote_count DESC, release_year DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [MovieRecord.model_validate(self._row_to_dict(r)) for r in cursor.fetchall()]

    def get_all_movies(self) -> list[MovieRecord]:
        """Fetch all movies as typed MovieRecord models."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM movies ORDER BY release_year ASC, popularity DESC")
            return [MovieRecord.model_validate(self._row_to_dict(r)) for r in cursor.fetchall()]

    def count(self) -> int:
        """Count total movies in database."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM movies")
            return cursor.fetchone()[0]

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a SQLite row to a Python dict with deserialized JSON lists."""
        d = dict(row)
        d["genres"] = json.loads(d.get("genres_json") or "[]")
        d["cast"] = json.loads(d.get("cast_json") or "[]")
        d["keywords"] = json.loads(d.get("keywords_json") or "[]")
        return d
