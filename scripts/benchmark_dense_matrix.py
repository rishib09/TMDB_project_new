"""#11 Phase 1: dense retrieval precision matrix — column presets x embedding models.

Builds one ChromaDB collection per (preset, model) cell and measures the
golden-query Hit Rate@5 / MRR@5 over it. Results feed the ADR 0008 tier
verdict: whether the legacy middle tier earns its keep is decided by this
table, not intuition.

Usage:
    python scripts/benchmark_dense_matrix.py                     # full matrix
    python scripts/benchmark_dense_matrix.py --cell minimal_lfm_free
    python scripts/benchmark_dense_matrix.py --presets minimal --models lfm_free bge_m3

Cloud cells need OPENROUTER_API_KEY (dotenvx). Local cells are free and offline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain.movie import MovieRecord  # noqa: E402
from src.indexing.embeddings import (  # noqa: E402
    BENCHMARK_PROFILES,
    MODEL_PROFILES,
    provider_from_profile,
)
from src.indexing.vector_store import MovieVectorStore  # noqa: E402
from src.storage.database import MovieDatabase  # noqa: E402
from tests.integration.test_vector_store import GOLDEN_QUERIES  # noqa: E402

PRESETS = ["minimal", "full"]


def load_movies(db: MovieDatabase) -> list[MovieRecord]:
    return db.get_all_movies()


def cell_name(preset: str, profile: str) -> str:
    return f"{preset}_{profile}"


def build_cell(store: MovieVectorStore, movies: list[MovieRecord],
               preset: str, profile: str, progress: Any = None) -> Any:
    """Indexes one (preset, model) collection from scratch; returns the provider
    (carrying truncation telemetry for the build report)."""
    provider = provider_from_profile(profile)
    store.index_movies(
        cell_name(preset, profile), movies,
        provider=provider, columns=preset, progress=progress,
    )
    return provider


def measure_cell(store: MovieVectorStore, preset: str, profile: str,
                 k: int = 5) -> dict:
    """Golden-query Hit@5 / MRR@5 against one cell."""
    provider = provider_from_profile(profile)
    hits, mrrs, misses = 0, [], []
    for query, expected_title in GOLDEN_QUERIES:
        results = store.search(
            query, version_name=cell_name(preset, profile),
            provider=provider, top_k=k,
        )
        titles = [r.movie.title for r in results]
        match_positions = [i for i, t in enumerate(titles)
                           if expected_title.lower() in t.lower()]
        if match_positions:
            hits += 1
            mrrs.append(1.0 / (match_positions[0] + 1))
        else:
            misses.append(f"  MISS '{query[:44]}' want '{expected_title}'")
    total = len(GOLDEN_QUERIES)
    return {
        "cell": cell_name(preset, profile),
        "hit@5": hits / total,
        "mrr@5": sum(mrrs) / total if mrrs else 0.0,
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", nargs="*", default=PRESETS, choices=PRESETS)
    parser.add_argument("--models", nargs="*", default=BENCHMARK_PROFILES,
                        choices=sorted(MODEL_PROFILES))
    parser.add_argument("--cells", nargs="*",
                        help="full cell names (preset_model) — overrides presets/models")
    parser.add_argument("--chroma-path", default="data/chroma_db")
    args = parser.parse_args()

    if args.cells:
        cells = []
        for cell in args.cells:
            preset, _, profile = cell.partition("_")
            if f"{preset}_{profile}" != cell or profile not in MODEL_PROFILES \
                    or preset not in PRESETS:
                parser.error(f"bad cell '{cell}' (expected <preset>_<model>)")
            cells.append((preset, profile))
    else:
        cells = [(p, m) for p in args.presets for m in args.models]

    db = MovieDatabase()
    store = MovieVectorStore(args.chroma_path)
    movies = load_movies(db)
    print(f"{len(movies)} Movie Records | {len(cells)} cells | {len(GOLDEN_QUERIES)} golden queries\n")

    results = []
    for preset, profile in cells:
        t0 = time.time()
        print(f"building {cell_name(preset, profile)} ...", flush=True)
        provider = build_cell(store, movies, preset, profile,
                              progress=lambda done, total: print(f"\r  {done}/{total}", end="", flush=True))
        build_s = time.time() - t0
        row = measure_cell(store, preset, profile)
        row["build_s"] = build_s
        results.append(row)
        print(f"\r  hit@5={row['hit@5']:.0%}  mrr@5={row['mrr@5']:.3f}  ({build_s:.0f}s build)")
        # truncation telemetry: a build must be able to SAY whether any doc
        # was cut, and by how much (#11 — silent truncation is forbidden)
        if hasattr(provider, "truncation_events"):
            print(f"  truncation: {provider.truncation_events} doc(s) cut, "
                  f"worst {provider.max_truncated_tokens} tok, "
                  f"window settled at {provider.max_tokens} tok")

    print("\n=== RESULTS (golden Hit@5 / MRR@5) ===")
    print(f"{'cell':<32} {'hit@5':>7} {'mrr@5':>8} {'build':>8}")
    for row in sorted(results, key=lambda r: -r["hit@5"]):
        print(f"{row['cell']:<32} {row['hit@5']:>6.0%} {row['mrr@5']:>8.3f} {row['build_s']:>7.0f}s")
    for row in results:
        if row["misses"]:
            print(f"\n{row['cell']} misses:")
            print("\n".join(row["misses"]))


if __name__ == "__main__":
    main()
