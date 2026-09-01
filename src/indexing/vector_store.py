"""Persistent vector store manager using FastEmbed ONNX embeddings and ChromaDB.

Issue #14: the three milestone collections are genuinely distinct embedding
tiers — different input sets, tokenizer-exact token budgets, and different
embedding models — with model↔collection pairing enforced at search time.
"""


import sys
from pathlib import Path
from typing import Any

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    # Windows Application Control (WDAC / Smart App Control) can block grpc's
    # cygrpc DLL at the machine-policy level. chromadb only needs it for
    # remote OTLP telemetry export, which this project never uses — so on
    # that failure we stub the exporter import path and proceed with the
    # local PersistentClient untouched.
    import types

    class _BlockedOTLPSpanExporter:  # pragma: no cover - environment shim
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "OTLP export unavailable: grpc DLL blocked by Application Control"
            )

    _stub_name = "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
    _stub = types.ModuleType(_stub_name)
    _stub.OTLPSpanExporter = _BlockedOTLPSpanExporter
    _stub.__spec__ = None
    _stub.__path__ = []
    sys.modules[_stub_name] = _stub
    import opentelemetry.exporter.otlp.proto.grpc as _otlp_grpc_pkg

    _otlp_grpc_pkg.trace_exporter = _stub

    import chromadb
    from chromadb.config import Settings

from chromadb.errors import NotFoundError
from fastembed import TextEmbedding
from pydantic import BaseModel, Field

from src.domain.movie import MovieRecord


class SearchResult(BaseModel):
    """Normalized search result returned from vector similarity queries."""
    id: int
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    movie: MovieRecord
    document_text: str


class FastembedTokenCounter:
    """TokenCounter backed by the target embedding model's real tokenizer.

    Truncation cuts on token boundaries using encoder offsets, preserving the
    original text (casing/punctuation) instead of round-tripping through
    decode — no silent truncation, no character estimates (issue #14).

    Padding matters: some fastembed tokenizers (e.g. MiniLM) pad every
    encoding to a fixed sequence length, so raw ``len(ids)`` is a constant,
    not a content length. Always count via the attention mask.
    """

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        encoded = self._tokenizer.encode(text)
        return sum(encoded.attention_mask)

    def truncate(self, text: str, max_tokens: int) -> str:
        encoded = self._tokenizer.encode(text)
        real_length = sum(encoded.attention_mask)
        if real_length <= max_tokens:
            return text
        cut = encoded.offsets[max_tokens - 1][1]
        prefix = text[:cut]
        # Trim to a clean word boundary.
        return prefix.rsplit(" ", 1)[0] if " " in prefix else prefix


class MovieVectorStore:
    """Manages persistent ChromaDB vector collections and FastEmbed CPU models."""

    DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_VERSION = "v1_2_bge_hybrid"

    #: The three benchmark milestone tiers (issue #14). Version name → tier,
    #: model and tokenizer-exact token budget. Input sets live in
    #: MovieRecord._tier_parts (t1_identity / t2_enriched / t3_exhaustive).
    TIER_PROFILES: dict[str, dict[str, Any]] = {
        "v1_0_baseline": {
            "tier": "t1_identity",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "token_budget": 128,  # MiniLM's real fastembed window (measured)
        },
        "v1_1_enriched": {
            "tier": "t2_enriched",
            "embedding_model": "snowflake/snowflake-arctic-embed-s",
            "token_budget": 512,
        },
        "v1_2_bge_hybrid": {
            "tier": "t3_exhaustive",
            "embedding_model": "jinaai/jina-embeddings-v2-base-en",
            "token_budget": 1024,
        },
    }

    def __init__(self, persist_dir: str = "data/chroma_db"):
        """Initializes ChromaDB PersistentClient."""
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self._embedder_cache: dict[str, TextEmbedding] = {}
        self._counter_cache: dict[str, FastembedTokenCounter] = {}

    # --- model & tokenizer plumbing -----------------------------------------

    def get_embedder(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> TextEmbedding:
        """Retrieves or lazy-loads a cached FastEmbed ONNX runtime engine."""
        if model_name not in self._embedder_cache:
            self._embedder_cache[model_name] = TextEmbedding(model_name=model_name)
        return self._embedder_cache[model_name]

    def get_token_counter(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> FastembedTokenCounter:
        """TokenCounter bound to the model's real tokenizer (exact packing)."""
        if model_name not in self._counter_cache:
            self._counter_cache[model_name] = FastembedTokenCounter(
                self.get_embedder(model_name).model.tokenizer
            )
        return self._counter_cache[model_name]

    # --- collection management ----------------------------------------------

    def get_collection_checked(self, version_name: str) -> chromadb.Collection:
        """Returns the collection; raises if it predates model-metadata pairing.

        Legacy collections (issue #2 era) carry no ``embedding_model`` metadata
        and cannot be searched safely — they must be rebuilt.
        """
        collection = self.client.get_collection(version_name)
        stored_model = (collection.metadata or {}).get("embedding_model")
        if not stored_model:
            raise ValueError(
                f"Collection '{version_name}' has no embedding_model metadata "
                f"(pre-issue-#14 build). Rebuild it before searching."
            )
        return collection

    def delete_collection(self, version_name: str) -> None:
        """Drops a collection if it exists (used by clean rebuilds)."""
        try:
            self.client.delete_collection(version_name)
        except (NotFoundError, ValueError):
            pass

    def index_movies(
        self,
        version_name: str,
        movies: list[MovieRecord],
        embedding_model: str | None = None,
        tier: str | None = None,
        token_budget: int | None = None,
        batch_size: int = 128,
        progress: Any = None,
    ) -> int:
        """Embeds and indexes movie records using the version's tier profile.

        All parameters default from TIER_PROFILES[version_name]; explicit
        arguments override (for factorial experiments via ExperimentConfig).
        Documents are packed with the target model's real tokenizer — stored
        text is guaranteed ≤ token_budget model tokens.

        ``progress(done, total)`` is invoked after each batch (for build logs).
        """
        profile = self.TIER_PROFILES.get(version_name, {})
        model_name = embedding_model or profile.get("embedding_model")
        if not model_name:
            raise ValueError(f"No embedding model for unknown version '{version_name}'")
        tier = tier or profile.get("tier", "t2_enriched")
        token_budget = token_budget or profile.get(
            "token_budget", MovieRecord.DEFAULT_TIER_BUDGETS.get(tier, 512)
        )

        # Recreate cleanly so collection metadata always reflects this build.
        self.delete_collection(version_name)
        collection = self.client.get_or_create_collection(
            name=version_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": model_name,
                "tier": tier,
                "token_budget": token_budget,
            },
        )
        embedder = self.get_embedder(model_name)
        counter = self.get_token_counter(model_name)

        total_indexed = 0

        # 1. Tier-shaped, tokenizer-exact text for all movies (front-loaded so
        #    documents can be length-sorted before embedding).
        doc_data = []
        for m in movies:
            meta = {
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
            doc_data.append(
                (str(m.id), m.to_dense_text(tier=tier, token_budget=token_budget, token_counter=counter), meta)
            )

        # 2. Length-sort: ONNX pads each batch to its longest member, so
        #    grouping similar lengths avoids paying 512/1024-token cost for
        #    200-token documents (~2-3x wall-time win on real corpora).
        doc_data.sort(key=lambda item: len(item[1]))

        for i in range(0, len(doc_data), batch_size):
            batch = doc_data[i:i + batch_size]
            ids = [item[0] for item in batch]
            doc_texts = [item[1] for item in batch]
            metadatas = [item[2] for item in batch]

            # 3. Generate embeddings via FastEmbed ONNX (serial: ONNX already
            #    saturates all cores via intra-op threads; fastembed's parallel=
            #    spawns worker processes per batch, measured 2.7x SLOWER)
            embeddings_gen = embedder.embed(doc_texts, batch_size=batch_size)
            embeddings = [emb.tolist() for emb in embeddings_gen]

            # 4. Upsert into ChromaDB
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=doc_texts,
                metadatas=metadatas
            )
            total_indexed += len(batch)
            if progress:
                progress(total_indexed, len(doc_data))

        return total_indexed

    def search(
        self,
        query: str,
        version_name: str = DEFAULT_VERSION,
        embedding_model: str | None = None,
        top_k: int = 10,
        where_filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """Performs vector similarity search with enforced model↔collection pairing.

        The query is embedded with the collection's stored embedding model.
        Passing an explicit ``embedding_model`` that differs from the stored
        one raises (previously this silently produced cross-space garbage).
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        collection = self.get_collection_checked(version_name)
        stored_model = collection.metadata["embedding_model"]
        if embedding_model and embedding_model != stored_model:
            raise ValueError(
                f"Embedding model mismatch: '{version_name}' was built with "
                f"'{stored_model}' but searched with '{embedding_model}'. "
                f"Cross-space queries silently return garbage and are refused."
            )
        model_name = stored_model

        if collection.count() == 0:
            return []

        # 1. Embed query vector on CPU with the paired model
        embedder = self.get_embedder(model_name)
        query_embedding = list(embedder.embed([clean_query]))[0].tolist()

        # 2. Query ChromaDB HNSW index. chroma 1.5.9 intermittently raises
        #    InternalError ("Error finding id") on where-filtered queries
        #    (observed both under concurrent indexing and idly). Fallback:
        #    retry once, then fetch unfiltered and apply the filter in Python.
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, collection.count()),
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k * 4, collection.count()),
                    include=["documents", "metadatas", "distances"]
                )
                results = self._apply_where_in_python(results, where_filter)

        search_results: list[SearchResult] = []
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

    @staticmethod
    def _apply_where_in_python(results: dict, where_filter: dict | None) -> dict:
        """Fallback where-filtering for chroma's intermittent InternalError.

        Supports the operator subset this project uses ($gte, $lte, $eq on
        numeric metadata fields).
        """
        if not where_filter:
            return results
        operators = {
            "$gte": lambda v, t: v >= t,
            "$lte": lambda v, t: v <= t,
            "$eq": lambda v, t: v == t,
        }
        keep = []
        for i, metadata in enumerate(results["metadatas"][0]):
            ok = True
            for field, conditions in where_filter.items():
                value = metadata.get(field)
                if isinstance(conditions, dict):
                    for op, target in conditions.items():
                        if value is None or not operators[op](value, target):
                            ok = False
                            break
                elif value != conditions:
                    ok = False
                    break
            if ok:
                keep.append(i)
            if len(keep) >= len(results["ids"][0]):
                break
        for key in ("ids", "documents", "metadatas", "distances"):
            if results.get(key) and results[key][0] is not None:
                results[key][0] = [results[key][0][i] for i in keep]
        return results

    def count(self, version_name: str) -> int:
        """Returns total vector count in a collection."""
        try:
            col = self.client.get_collection(version_name)
            return col.count()
        except (NotFoundError, ValueError):
            return 0

    def list_collections(self) -> list[str]:
        """Lists all existing collection names."""
        return [c.name for c in self.client.list_collections()]
