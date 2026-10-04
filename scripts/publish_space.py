"""Publish Maya to a Hugging Face Space (#145).

Copies the six Lab collections into a fresh Chroma directory, uploads that
directory and the movie catalog to the dataset repo, then creates the
Streamlit Space and sets its secrets from the environment.

Run with dotenvx so the keys stay out of the shell history:

    npx @dotenvx/dotenvx run -- .venv/Scripts/python.exe scripts/publish_space.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.config import Settings
from huggingface_hub import HfApi

from src.indexing.embeddings import lab_collection_names

DATA_REPO = "rishib09/maya-data"
SPACE_REPO = "rishib09/maya"
_BATCH = 256


def _client(persist_dir: Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def _chunk(values, start: int, end: int):
    if values is None:
        return None
    piece = values[start:end]
    if hasattr(piece, "tolist"):
        return piece.tolist()
    return list(piece)


def slim_collections(source_dir: Path, dest_dir: Path, names: tuple[str, ...]) -> None:
    """Copy named collections into a fresh Chroma directory.

    Calls ``chromadb.PersistentClient``, ``Collection.get``, and ``Collection.add``.
    """
    source = _client(source_dir)
    dest = _client(dest_dir)
    for name in names:
        origin = source.get_collection(name)
        if origin.count() == 0:
            raise ValueError(f"collection {name} is empty")
        target = dest.get_or_create_collection(name=name, metadata=origin.metadata)
        offset = 0
        while True:
            page = origin.get(
                include=["embeddings", "documents", "metadatas"],
                limit=_BATCH,
                offset=offset,
            )
            ids = page["ids"]
            if not ids:
                break
            end = offset + len(ids)
            metadatas = _chunk(page["metadatas"], 0, len(ids))
            if metadatas is not None:
                metadatas = [{} if item is None else item for item in metadatas]
            target.add(
                ids=list(ids),
                embeddings=_chunk(page["embeddings"], 0, len(ids)),
                documents=_chunk(page["documents"], 0, len(ids)),
                metadatas=metadatas,
            )
            offset = end
        print(f"{name}: {offset}/{origin.count()}", flush=True)


def _export_tracked(dest: Path) -> None:
    """Copy tracked files and untracked files git does not ignore.

    The Space must receive this working tree, including the Dockerfile, which
    is not committed yet. The catalog and the vector index stay in the dataset.
    """
    listing = subprocess.check_output(["git", "ls-files", "-z", "-c", "-o", "--exclude-standard"])
    skip = (".claude/", ".scratch/", "data/")
    for rel in listing.decode().split("\0"):
        if not rel or rel.startswith(skip):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rel, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish Maya to a Hugging Face Space.")
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Dataset repo is already uploaded; create the Space only.",
    )
    args = parser.parse_args()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    openrouter = os.getenv("OPENROUTER_API_KEY")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Create a write token and export it.")
    if not openrouter:
        raise SystemExit("OPENROUTER_API_KEY is not set.")

    api = HfApi(token=token)
    if not args.skip_dataset:
        root = Path(tempfile.mkdtemp(prefix="maya-data-"))
        try:
            print("copying the six Lab collections", flush=True)
            slim_collections(Path("data/chroma_db"), root / "chroma_db", lab_collection_names())
            shutil.copy2("data/tmdb_movies.db", root / "tmdb_movies.db")
            print(f"uploading dataset {DATA_REPO}", flush=True)
            api.create_repo(DATA_REPO, repo_type="dataset", exist_ok=True, private=False)
            api.upload_folder(
                repo_id=DATA_REPO,
                folder_path=str(root),
                repo_type="dataset",
                commit_message="Movie catalog and the six Lab collections",
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    with tempfile.TemporaryDirectory() as tmp:
        code = Path(tmp)
        _export_tracked(code)
        print(f"creating Space {SPACE_REPO}", flush=True)
        api.create_repo(
            SPACE_REPO,
            repo_type="space",
            exist_ok=True,
            private=False,
            space_sdk="docker",
            space_hardware="cpu-basic",
        )
        api.upload_folder(
            repo_id=SPACE_REPO,
            folder_path=str(code),
            repo_type="space",
            commit_message="Publish Maya",
        )
    api.add_space_secret(SPACE_REPO, "OPENROUTER_API_KEY", openrouter)
    zai = os.getenv("ZAI_API_KEY")
    if zai:
        api.add_space_secret(SPACE_REPO, "ZAI_API_KEY", zai)
    api.add_space_variable(SPACE_REPO, "MAYA_ROUTING_STACK", "v2")
    print(f"https://huggingface.co/spaces/{SPACE_REPO}", flush=True)


if __name__ == "__main__":
    main()
