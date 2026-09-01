"""Rebuilds the 3 tier-distinct ChromaDB collections (issue #14), resumable.

Each collection is a genuinely different embedding tier — different input
set, tokenizer-exact token budget, and embedding model — driven entirely by
MovieVectorStore.TIER_PROFILES.

Resume behavior: complete collections (count matches, metadata matches) are
skipped, so an interrupted rebuild only redoes what's missing. Pass --clean
to wipe the directory and rebuild everything from scratch.
"""

import os
import shutil
import sys
import time
from pathlib import Path

# Models are pre-cached; skip HF hub network checks so builds can't stall on
# a hung HEAD request inside headless/background runs.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.vector_store import MovieVectorStore
from src.storage.database import MovieDatabase


def rebuild_index(
    db_path: str = "data/tmdb_movies.db",
    chroma_path: str = "data/chroma_db",
    clean: bool = False,
    versions: list[str] | None = None,
):
    """Build tier-distinct ChromaDB collections.

    ``versions`` selects a subset of TIER_PROFILES keys (default: all three).
    """
    selected = versions or list(MovieVectorStore.TIER_PROFILES)
    unknown = set(selected) - set(MovieVectorStore.TIER_PROFILES)
    if unknown:
        raise SystemExit(
            f"Unknown version(s): {sorted(unknown)}. "
            f"Valid: {sorted(MovieVectorStore.TIER_PROFILES)}"
        )

    print("[START] Loading movies from SQLite...", flush=True)
    db = MovieDatabase(db_path)
    movies = db.get_all_movies()
    expected = len(movies)
    print(f"  - Loaded {expected} movies.", flush=True)

    chroma_dir = Path(chroma_path)
    if clean and chroma_dir.exists():
        print(f"  - --clean: wiping {chroma_path}...", flush=True)
        shutil.rmtree(chroma_dir)

    vector_store = MovieVectorStore(chroma_path)

    timings = {}
    for version_name, profile in MovieVectorStore.TIER_PROFILES.items():
        if version_name not in selected:
            continue

        # Resume: skip collections already complete with matching metadata.
        try:
            existing = vector_store.client.get_collection(version_name)
            meta = existing.metadata or {}
            if (
                existing.count() == expected
                and meta.get("embedding_model") == profile["embedding_model"]
                and meta.get("tier") == profile["tier"]
                and meta.get("token_budget") == profile["token_budget"]
            ):
                print(f"\n[SKIP] {version_name} already complete "
                      f"({expected} vectors, metadata matches).", flush=True)
                continue
            state = f"incomplete: {existing.count()}/{expected} vectors"
        except Exception:
            state = "missing"
        print(f"\n[INDEX] Building {version_name} ({state})...", flush=True)
        print(f"  profile: {profile}", flush=True)
        t0 = time.time()

        def heartbeat(done: int, total: int, _t0=t0):
            print(f"  {done}/{total} vectors in {time.time()-_t0:.0f}s", flush=True)

        count = vector_store.index_movies(
            version_name=version_name, movies=movies, batch_size=128, progress=heartbeat
        )
        elapsed = time.time() - t0
        timings[version_name] = elapsed
        print(f"  [OK] Indexed {count} vectors in {elapsed:.1f}s", flush=True)

    print("\n[COMPLETE] Selected tier collections verified!", flush=True)
    for col in vector_store.list_collections():
        meta = vector_store.client.get_collection(col).metadata
        print(f"  - {col}: {vector_store.count(col)} vectors | {meta}", flush=True)
    if timings:
        print("\nIndexing wall-times:", {k: f"{v:.1f}s" for k, v in timings.items()}, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build tier-distinct ChromaDB embedding collections (issue #14)."
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=sorted(MovieVectorStore.TIER_PROFILES),
        default=sorted(MovieVectorStore.TIER_PROFILES),
        help="Which collection(s) to build (default: all three)."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe the ChromaDB directory before building (default: resume)."
    )
    parser.add_argument("--db", default="data/tmdb_movies.db", help="SQLite database path.")
    parser.add_argument("--chroma", default="data/chroma_db", help="ChromaDB persist directory.")
    args = parser.parse_args()

    rebuild_index(
        db_path=args.db,
        chroma_path=args.chroma,
        clean=args.clean,
        versions=args.versions,
    )
