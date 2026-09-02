"""User feedback persistence (issue #9): SQLite store for thumbs ratings.

Each assistant turn carries a trace_id (minted by the tracer); a thumb
rating is stored against it together with the RAG version and routing
intent active at turn time. Writes are UPSERTs keyed on trace_id —
changing your thumb updates the row instead of duplicating it.

Framework-free: plain sqlite3, no LangChain/LangGraph imports (ADR 0006).
One persistent connection per store: mandatory for ``:memory:`` databases
(each ``connect(":memory:")`` would create a fresh empty DB) and cheap for
file DBs; ``check_same_thread=False`` because Streamlit reruns scripts on
worker threads.
"""

import sqlite3
from pathlib import Path


class FeedbackStore:
    """Read/write access to the ``user_feedback`` table (schema in MovieDatabase)."""

    def __init__(self, db_path: str = "data/tmdb_movies.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        conn = self._conn
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                rag_version TEXT NOT NULL,
                rating INTEGER NOT NULL,
                user_comment TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # MovieDatabase's schema has no UNIQUE on trace_id; the unique index
        # makes the UPSERT work on fresh AND pre-existing tables (#9).
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_user_feedback_trace "
            "ON user_feedback(trace_id)"
        )
        try:
            conn.execute("ALTER TABLE user_feedback ADD COLUMN intent TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.commit()

    def record(self, trace_id: str, rating: int, rag_version: str,
               intent: str = "", comment: str | None = None) -> None:
        """UPSERT one rating keyed on trace_id (re-rating updates in place).

        Only +1 (thumbs up) and -1 (thumbs down) are valid st.feedback
        'thumbs' values; anything else raises ValueError before touching disk.
        """
        if rating not in (1, -1):
            raise ValueError(f"rating must be +1 or -1, got {rating!r}")
        if not trace_id:
            raise ValueError("trace_id must be non-empty")
        self._conn.execute(
            """
            INSERT INTO user_feedback (trace_id, rag_version, rating, user_comment, intent)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(trace_id) DO UPDATE SET
                rating = excluded.rating,
                rag_version = excluded.rag_version,
                user_comment = excluded.user_comment,
                intent = excluded.intent,
                timestamp = CURRENT_TIMESTAMP
            """,
            (trace_id, rag_version, rating, comment, intent),
        )
        self._conn.commit()

    def stats_by_version(self) -> list[dict]:
        """Aggregates ratings per RAG version for the Evals dashboard."""
        rows = self._conn.execute(
            """
            SELECT rag_version,
                   COUNT(*)               AS n,
                   AVG(rating)            AS avg_rating,
                   SUM(rating = 1)        AS thumbs_up,
                   SUM(rating = -1)       AS thumbs_down
            FROM user_feedback
            GROUP BY rag_version
            ORDER BY n DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def get_by_trace_id(self, trace_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM user_feedback WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM user_feedback").fetchone()[0]
