"""Persistent vector store manager using FastEmbed ONNX embeddings and ChromaDB."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding

from src.domain.movie import MovieRecord, CastMember


class SearchResult(BaseModel):
    """Normalized search result returned from vector similarity queries."""
    id: int
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    movie: MovieRecord
    document_text: str


class MovieVectorStore:
    """Manages persistent ChromaDB vector collections and FastEmbed CPU models."""

    DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, persist_dir: str = "data/chroma_db"):
        """Initializes ChromaDB PersistentClient."""
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self._embedder_cache: Dict[str, TextEmbedding] = {}

    def get_embedder(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> TextEmbedding:
        """Retrieves or lazy-loads a cached FastEmbed ONNX runtime engine."""
        if model_name not in self._embedder_cache:
            self._embedder_cache[model_name] = TextEmbedding(model_name=model_name)
        return self._embedder_cache[model_name]

    def get_or_create_collection(
        self,
        version_name: str,
        distance_metric: str = "cosine"
    ) -> chromadb.Collection:
        """Gets or creates an isolated ChromaDB collection for a RAG version."""
        return self.client.get_or_create_collection(
            name=version_name,
            metadata={"hnsw:space": distance_metric}
        )

    def index_movies(
        self,
        version_name: str,
        embedding_model: str,
        chunking_strategy: str,
        movies: List[MovieRecord],
        batch_size: int = 128
    ) -> int:
        """Embeds and indexes movie records using parallel CPU workers."""
        collection = self.get_or_create_collection(version_name)
        embedder = self.get_embedder(embedding_model)

        total_indexed = 0
        workers = os.cpu_count() or 4

        for i in range(0, len(movies), batch_size):
            batch = movies[i:i + batch_size]
            
            # 1. Format text representations based on chunking strategy
            doc_texts = [m.to_dense_text(strategy=chunking_strategy) for m in batch]
            ids = [str(m.id) for m in batch]

            # 2. Generate embeddings via FastEmbed ONNX using parallel CPU cores
            embeddings_gen = embedder.embed(doc_texts, batch_size=batch_size, parallel=workers)
            embeddings = [emb.tolist() for emb in embeddings_gen]

            # 3. Create metadata dictionary (primitives only for ChromaDB)
            metadatas = [
                {
                    "id": m.id,
                    "title": m.title,
                    "release_year": m.release_year,
                    "director": m.director or "",
                    "vote_average": float(m.vote_average),
                    "revenue": int(m.revenue),
                    "genres_str": " ".join(m.genres),
                    "poster_path": m.poster_path or "",
                    "raw_json": m.model_dump_json(),
                }
                for m in batch
            ]

            # 4. Upsert into ChromaDB
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=doc_texts,
                metadatas=metadatas
            )
            total_indexed += len(batch)

        return total_indexed

    def search(
        self,
        query: str,
        version_name: str = "v1_2_bge_hybrid",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = 10,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Performs vector similarity search against the specified ChromaDB collection."""
        clean_query = query.strip()
        if not clean_query:
            return []

        collection = self.get_or_create_collection(version_name)
        if collection.count() == 0:
            return []

        # 1. Embed query vector on CPU
        embedder = self.get_embedder(embedding_model)
        query_embedding = list(embedder.embed([clean_query]))[0].tolist()

        # 2. Query ChromaDB HNSW index
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        search_results: List[SearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for movie_id_str, distance, doc_text, metadata in zip(ids, distances, documents, metadatas):
            similarity = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
            raw_json_str = metadata.get("raw_json")
            if raw_json_str:
                movie_obj = MovieRecord.model_validate_json(raw_json_str)
            else:
                movie_obj = MovieRecord(
                    id=int(metadata.get("id", movie_id_str)),
                    title=metadata.get("title", ""),
                    release_year=int(metadata.get("release_year", 0)),
                    director=metadata.get("director", ""),
                    vote_average=float(metadata.get("vote_average", 0.0)),
                    revenue=int(metadata.get("revenue", 0)),
                    poster_path=metadata.get("poster_path", ""),
                )

            search_results.append(SearchResult(
                id=int(movie_id_str),
                score=round(similarity, 4),
                movie=movie_obj,
                document_text=doc_text
            ))

        return search_results

    def count(self, version_name: str) -> int:
        """Returns total vector count in a collection."""
        try:
            col = self.client.get_collection(version_name)
            return col.count()
        except Exception:
            return 0

    def list_collections(self) -> List[str]:
        """Lists all existing collection names."""
        return [c.name for c in self.client.list_collections()]
