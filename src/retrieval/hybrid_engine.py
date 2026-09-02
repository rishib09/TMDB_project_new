"""Hybrid retrieval engine (issue #4): SQL superlatives, dense+BM25 RRF fusion,
FlashRank CPU cross-encoder reranking.

One entry point for the pipeline (#5): a QueryRoutingDecision in, ranked
movies with posters out. All libraries are from requirements.txt — no
custom ML code: rank fusion is arithmetic, reranking is flashrank.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from src.domain.movie import MovieRecord
from src.domain.routing import IntentType, MetadataFilterCriteria, QueryRoutingDecision
from src.indexing.vector_store import MovieVectorStore, SearchResult
from src.storage.database import MovieDatabase


class RetrievalResult(BaseModel):
    """A ranked movie with provenance for the trace inspector (#5)."""

    movie: MovieRecord
    score: float = Field(..., description="Final ranking score (path-dependent)")
    source: str = Field(..., description="sql | dense | bm25 | rrf | reranked")
    dense_rank: int | None = None
    sparse_rank: int | None = None
    document_text: str = ""


class HybridRetrievalEngine:
    """Routes superlatives to SQL, fuses dense+BM25 via RRF, optionally reranks."""

    RRF_K: ClassVar[int] = 60  # standard RRF smoothing constant

    def __init__(
        self,
        db: MovieDatabase,
        vector_store: MovieVectorStore,
        rag_version: str = "v1_1_enriched",
        hybrid_alpha: float = 0.5,
        reranker_enabled: bool = False,
        reranker_model: str = "ms-marco-TinyBERT-L-2-v2",
    ):
        # reranker_enabled defaults to OFF deliberately (live measurement
        # 2026-08-31): on the golden query both tiny cross-encoders ranked
        # obscure keyword-dense docs ABOVE Inception, while pure RRF put it
        # #1-#2 on every tier. Whether reranking nets positive is a benchmark
        # question for #6 — the knob stays for the A/B (ADR 0004).
        self.db = db
        self.vector_store = vector_store
        self.rag_version = rag_version
        self.hybrid_alpha = hybrid_alpha
        self.reranker_enabled = reranker_enabled
        self.reranker_model = reranker_model
        self._ranker: Any = None  # lazy: only loaded when reranking first runs

    # --- public entry point ---------------------------------------------------

    def retrieve(
        self,
        query: str,
        routing: QueryRoutingDecision,
        top_k: int = 8,
        candidate_pool: int = 50,
    ) -> list[RetrievalResult]:
        """Returns the final ranked movies for one router decision."""
        if not routing.requires_rag:
            return []

        routing = self._resolve_person(routing)
        if routing is None:  # named person not in the archive (#24)
            return []

        if self._use_sql_path(routing):
            return self._retrieve_sql(routing, top_k)

        dense = self._retrieve_dense(query, candidate_pool)
        sparse = self._retrieve_bm25(query, candidate_pool)
        fused = self._rrf_fuse(dense, sparse)

        # Uniform post-filtering: positive filters + exclusions on the small
        # candidate pool (BM25 has no metadata columns; this keeps one path).
        fused = [r for r in fused if self._is_allowed(r.movie, routing.filters)]

        if self.reranker_enabled and fused:
            return self._rerank(query, fused, top_k)
        return fused[:top_k]

    # --- person role resolution (#24) -------------------------------------------

    def _resolve_person(self, routing: QueryRoutingDecision) -> QueryRoutingDecision | None:
        """DB ground truth for role-less person mentions (#24).

        director-only → director filter; cast-only → cast filter; BOTH → keep
        ``person`` (the predicate OR-matches both filmographies); neither →
        None (caller turns this into the deterministic not-found response).
        """
        filters = routing.filters
        if not filters or not filters.person:
            return routing
        as_director, as_cast = self.db.classify_person(filters.person)
        if as_director and as_cast:
            return routing  # union: predicate OR-matches director + cast
        if as_director:
            return routing.model_copy(update={"filters": filters.model_copy(
                update={"person": None, "director": filters.person}
            )})
        if as_cast:
            return routing.model_copy(update={"filters": filters.model_copy(
                update={"person": None, "cast_member": filters.person}
            )})
        return None

    # --- path selection ---------------------------------------------------------

    @staticmethod
    def _use_sql_path(routing: QueryRoutingDecision) -> bool:
        """Superlatives and exact metadata go to deterministic SQL, never vectors."""
        if routing.intent == IntentType.SUPERLATIVE_RANKING and routing.superlative:
            return True
        if routing.intent == IntentType.ATTRIBUTE_FILTER and routing.filters:
            f = routing.filters
            return bool(f.exact_year or f.director or f.cast_member)
        return False

    # --- SQL path ---------------------------------------------------------------

    def _retrieve_sql(self, routing: QueryRoutingDecision, top_k: int) -> list[RetrievalResult]:
        if routing.superlative:
            s = routing.superlative
            movies = self.db.query_superlative(
                metric=s.metric,
                direction=s.direction,
                year=s.year,
                genre=s.genre,
                limit=top_k * 2,  # headroom so post-filtering can't empty the page
            )
        else:
            movies = self.db.search_metadata_filters(routing.filters, limit=top_k * 2)

        movies = [m for m in movies if self._is_allowed(m, routing.filters)]
        return [
            RetrievalResult(movie=m, score=float(rank + 1), source="sql")
            for rank, m in enumerate(movies[:top_k])
        ]

    # --- hybrid path ------------------------------------------------------------

    def _retrieve_dense(self, query: str, top_k: int) -> list[SearchResult]:
        try:
            return self.vector_store.search(
                query=query, version_name=self.rag_version, top_k=top_k
            )
        except Exception:
            # Missing/legacy collection must not kill retrieval — BM25 carries on.
            return []

    def _retrieve_bm25(self, query: str, top_k: int) -> list[MovieRecord]:
        return self.db.search_bm25(query, limit=top_k)

    def _rrf_fuse(
        self,
        dense: list[SearchResult],
        sparse: list[MovieRecord],
    ) -> list[RetrievalResult]:
        """Reciprocal Rank Fusion: score(d) = sum(w_i / (k + rank_i)).

        hybrid_alpha weights the dense list (1.0 = dense only, 0.0 = BM25 only).
        """
        w_dense, w_sparse = self.hybrid_alpha, 1.0 - self.hybrid_alpha
        entries: dict[int, dict[str, Any]] = {}

        for rank, result in enumerate(dense):
            entry = entries.setdefault(
                result.movie.id,
                {"movie": result.movie, "document_text": result.document_text,
                 "dense_rank": None, "sparse_rank": None},
            )
            entry["dense_rank"] = rank + 1
        for rank, movie in enumerate(sparse):
            entry = entries.setdefault(
                movie.id,
                {"movie": movie, "document_text": movie.overview,
                 "dense_rank": None, "sparse_rank": None},
            )
            entry["sparse_rank"] = rank + 1

        results = []
        for entry in entries.values():
            score = 0.0
            if entry["dense_rank"] is not None:
                score += w_dense / (self.RRF_K + entry["dense_rank"])
            if entry["sparse_rank"] is not None:
                score += w_sparse / (self.RRF_K + entry["sparse_rank"])
            results.append(RetrievalResult(
                movie=entry["movie"],
                score=round(score, 6),
                source="rrf",
                dense_rank=entry["dense_rank"],
                sparse_rank=entry["sparse_rank"],
                document_text=entry["document_text"],
            ))

        results.sort(key=lambda r: (-r.score, r.dense_rank or 999, r.movie.id))
        return results

    # --- reranking ----------------------------------------------------------------

    def _rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        from flashrank import Ranker, RerankRequest

        if self._ranker is None:
            self._ranker = Ranker(model_name=self.reranker_model)

        rerank_request = RerankRequest(
            query=query,
            passages=[
                {"id": str(c.movie.id), "text": c.document_text or c.movie.overview}
                for c in candidates
            ],
        )
        try:
            ranked = self._ranker.rerank(rerank_request)
        except Exception:
            # Reranker failure degrades to RRF order, never to an error page.
            return candidates[:top_k]

        by_id = {c.movie.id: c for c in candidates}
        results = []
        for position, item in enumerate(ranked[:top_k]):
            original = by_id[int(item["id"])]
            results.append(original.model_copy(
                update={"score": float(item["score"]), "source": "reranked"}
            ))
        return results

    # --- shared post-filtering -----------------------------------------------------

    def _is_allowed(
        self,
        movie: MovieRecord,
        filters: MetadataFilterCriteria | None,
    ) -> bool:
        """Single predicate for positive filters AND exclusions (query + session)."""
        if not self.matches_filters(movie, filters):
            return False
        if filters:
            excluded_genres = {g.lower() for g in filters.excluded_genres}
            excluded_actors = {a.lower() for a in filters.excluded_actors}
            if excluded_genres & {g.lower() for g in movie.genres}:
                return False
            if excluded_actors & {c.name.lower() for c in movie.cast}:
                return False
        return True

    @staticmethod
    def matches_filters(movie: MovieRecord, filters: MetadataFilterCriteria | None) -> bool:
        """True if a candidate satisfies the positive (non-exclusion) filters.

        Used to post-filter the BM25 path (FTS5 has no metadata columns).
        Dense years are already constrained by the store; this is uniform
        and cheap on a 50-candidate pool.
        """
        if filters is None:
            return True
        if filters.exact_year is not None and movie.release_year != filters.exact_year:
            return False
        if filters.year_min is not None and movie.release_year < filters.year_min:
            return False
        if filters.year_max is not None and movie.release_year > filters.year_max:
            return False
        if filters.genres:
            wanted = {g.lower() for g in filters.genres}
            have = {g.lower() for g in movie.genres}
            if filters.genre_match == "all":
                if not wanted <= have:  # intersection (#25)
                    return False
            elif not wanted & have:
                return False
        if filters.person:
            name = filters.person.lower()
            in_cast = any(name in c.name.lower() for c in movie.cast)
            if not (name in movie.director.lower() or in_cast):
                return False
        if filters.director and filters.director.lower() not in movie.director.lower():
            return False
        if filters.cast_member:
            names = {c.name.lower() for c in movie.cast}
            if filters.cast_member.lower() not in names:
                return False
        return True
