"""Startup download skips work when the Lab collections are already on disk."""

from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.indexing.embeddings import collection_name, lab_collection_names
from src.ui.session import ensure_runtime_files, runtime_files_missing
from src.ui.sidebar_lab import _COMBOS


def test_lab_collection_names_match_the_sidebar_combos():
    assert set(lab_collection_names()) == {
        collection_name(preset, profile) for preset, profile, _ in _COMBOS
    }
    assert len(lab_collection_names()) == 6


def _client(path: Path):
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )


def _seed(path: Path, names: tuple[str, ...], count: int = 2) -> None:
    client = _client(path)
    for name in names:
        col = client.get_or_create_collection(name)
        col.add(
            ids=[f"{name}-{i}" for i in range(count)],
            embeddings=[[float(i), 0.5, 0.25] for i in range(count)],
            documents=[f"movie {i}" for i in range(count)],
            metadatas=[{"i": i} for i in range(count)],
        )


def _catalog(data_dir: Path) -> None:
    (data_dir / "tmdb_movies.db").write_bytes(b"catalog")


def test_ensure_skips_download_when_the_lab_collections_are_present(tmp_path):
    _catalog(tmp_path)
    _seed(tmp_path / "chroma_db", lab_collection_names())
    called = []
    ensure_runtime_files(tmp_path, download=lambda repo, dest: called.append(repo))
    assert called == []
    assert runtime_files_missing(tmp_path) is False


def test_ensure_downloads_when_the_catalog_is_missing(tmp_path):
    called = []
    ensure_runtime_files(
        tmp_path,
        repo_id="rishib09/maya-data",
        download=lambda repo, dest: called.append((repo, dest)),
    )
    assert called == [("rishib09/maya-data", tmp_path)]


def test_slim_collections_copies_embeddings_into_a_fresh_directory(tmp_path):
    from scripts.publish_space import slim_collections

    names = ("full_gemini_embedding_2", "minimal_lfm_free")
    _seed(tmp_path / "source", names, count=3)
    slim_collections(tmp_path / "source", tmp_path / "dest", names)
    copied = _client(tmp_path / "dest")
    for name in names:
        col = copied.get_collection(name)
        assert col.count() == 3
        got = col.get(ids=[f"{name}-1"], include=["embeddings"])["embeddings"][0]
        assert [round(float(value), 5) for value in got] == [1.0, 0.5, 0.25]
