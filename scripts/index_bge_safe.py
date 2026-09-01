"""Fast parallel indexer for v1_2_bge_hybrid utilizing all CPU cores."""

import os
import sys
import time
from pathlib import Path

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import MovieDatabase
from src.indexing.vector_store import MovieVectorStore


def index_bge_fast():
    workers = os.cpu_count() or 4
    print(f"[START] Loading 9119 movies from SQLite (Using {workers} parallel CPU workers)...", flush=True)
    db = MovieDatabase("data/tmdb_movies.db")
    movies = db.get_all_movies()
    print(f"  - Loaded {len(movies)} movies.", flush=True)

    vector_store = MovieVectorStore("data/chroma_db")

    # Reset any partial collection
    try:
        vector_store.client.delete_collection("v1_2_bge_hybrid")
        print("  - Cleared partial v1_2_bge_hybrid collection.", flush=True)
    except Exception:
        pass

    print(f"[INDEXING] Starting parallel indexing for v1_2_bge_hybrid with {workers} workers...", flush=True)
    t0 = time.time()
    count = vector_store.index_movies(
        version_name="v1_1_enriched",
        movies=movies,
        batch_size=128
    )  # defaults from TIER_PROFILES: bge-small + t2_enriched + 512 tokens
    elapsed = time.time() - t0
    print(f"[DONE] Successfully indexed {count} vectors in {elapsed:.2f}s ({count / elapsed:.1f} docs/sec)!", flush=True)
    print(f"Total in v1_2_bge_hybrid: {vector_store.count('v1_2_bge_hybrid')}", flush=True)


if __name__ == "__main__":
    index_bge_fast()
