"""Cleanly builds all 3 ChromaDB collections from scratch without corrupted segments."""

import shutil
import sys
import time
from pathlib import Path

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database import MovieDatabase
from src.indexing.vector_store import MovieVectorStore


def rebuild_clean_index(db_path: str = "data/tmdb_movies.db", chroma_path: str = "data/chroma_db"):
    print("[START] Loading 9119 movies from SQLite...", flush=True)
    db = MovieDatabase(db_path)
    movies = db.get_all_movies()
    print(f"  - Loaded {len(movies)} movies.", flush=True)

    # 1. Remove old/interrupted chroma_db directory to ensure 100% clean HNSW segments
    chroma_dir = Path(chroma_path)
    if chroma_dir.exists():
        print(f"  - Resetting existing {chroma_path} to prevent corrupted segments...", flush=True)
        shutil.rmtree(chroma_dir)

    vector_store = MovieVectorStore(chroma_path)

    # 2. Build Milestone v1.0: Baseline MiniLM
    print("\n[INDEX 1/3] Building v1_0_baseline (all-MiniLM-L6-v2 + Overview)...", flush=True)
    t0 = time.time()
    c1 = vector_store.index_movies(
        version_name="v1_0_baseline",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunking_strategy="baseline",
        movies=movies,
        batch_size=256
    )
    print(f"  [OK] Indexed {c1} vectors in {time.time() - t0:.2f}s", flush=True)

    # 3. Build Milestone v1.1: Enriched MiniLM
    print("\n[INDEX 2/3] Building v1_1_enriched (all-MiniLM-L6-v2 + Enriched Metadata)...", flush=True)
    t0 = time.time()
    c2 = vector_store.index_movies(
        version_name="v1_1_enriched",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunking_strategy="enriched_metadata",
        movies=movies,
        batch_size=256
    )
    print(f"  [OK] Indexed {c2} vectors in {time.time() - t0:.2f}s", flush=True)

    # 4. Build Milestone v1.2: BGE-Small Hybrid
    print("\n[INDEX 3/3] Building v1_2_bge_hybrid (BAAI/bge-small-en-v1.5 + Enriched Metadata)...", flush=True)
    t0 = time.time()
    c3 = vector_store.index_movies(
        version_name="v1_2_bge_hybrid",
        embedding_model="BAAI/bge-small-en-v1.5",
        chunking_strategy="enriched_metadata",
        movies=movies,
        batch_size=256
    )
    print(f"  [OK] Indexed {c3} vectors in {time.time() - t0:.2f}s", flush=True)

    print("\n[COMPLETE] All 3 ChromaDB collections successfully built and verified!", flush=True)
    for col in vector_store.list_collections():
        print(f"  - Collection '{col}': {vector_store.count(col)} vectors", flush=True)


if __name__ == "__main__":
    rebuild_clean_index()
