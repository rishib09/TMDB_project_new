"""Hybrid retrieval engine (issue #4): SQL superlatives, dense+BM25 RRF fusion,
FlashRank CPU cross-encoder reranking.

One entry point for the pipeline (#5): a QueryRoutingDecision in, ranked
movies with posters out. All libraries are from requirements.txt — no
custom ML code: rank fusion is arithmetic, reranking is flashrank.
"""

from collections.abc import Sequence
import math
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
    dense_failed: bool = Field(
        default=False,
        description="#65: dense search raised for this call; the list is BM25-only",
    )


def build_where_clause(
    filters: MetadataFilterCriteria | None,
    shown_ids: Sequence[int] | None,
) -> dict[str, Any] | None:
    """#88 (D5): the Chroma-expressible subset of a routed decision — pure.

    Years map to ``release_year`` range/equality operators; shown ids map to
    ``id`` ``$nin`` so the store draws top-k from a fresh pool (#80). Genres
    deliberately stay OUT: Chroma metadata holds ``genres_str`` as a joined
    string, and ``$in`` matches whole values only — pushing genres would
    silently exclude multi-genre movies ("Action Sci-Fi" fails "$in: Action").
    Genres remain in the engine's post-filter safety net until a re-index
    stores them as list metadata. Returns None when nothing is expressible.
    """
    conditions: dict[str, Any] = {}
    if filters:
        if filters.exact_year is not None:
            conditions["release_year"] = {"$eq": filters.exact_year}
        else:
            year: dict[str, int] = {}
            if filters.year_min is not None:
                year["$gte"] = filters.year_min
            if filters.year_max is not None:
                year["$lte"] = filters.year_max
            if year:
                conditions["release_year"] = year
    if shown_ids:
        conditions["id"] = {"$nin": list(shown_ids)}
    return conditions or None


class HybridRetrievalEngine:
    """Routes superlatives to SQL, fuses dense+BM25 via RRF, optionally reranks."""

    RRF_K: ClassVar[int] = 60  # standard RRF smoothing constant
    #: #137 boost normalization. RRF scores are ~0.03-magnitude sums of
    #: 1/(K+rank) terms; boost terms are normalized field scores in [0, 2]
    #: scaled by NORMALIZER so one full boost term ≈ ~30 RRF rank positions —
    #: a real tilt inside the candidate pool that never manufactures
    #: candidates out of thin air (pool membership is unchanged).
    BOOST_NORMALIZER: ClassVar[float] = 0.01
    BOOST_RUNTIME_WEIGHT: ClassVar[float] = 0.5
    BOOST_TERM_CAP: ClassVar[float] = 2.0
    def __init__(
        self,
        db: MovieDatabase,
        vector_store: MovieVectorStore,
        rag_version: str = "full_gemini_embedding_2",
        hybrid_alpha: float = 0.5,
        reranker_enabled: bool = False,
        reranker_model: str = "ms-marco-TinyBERT-L-2-v2",
        search_provider: Any | None = None,
    ):
        # reranker_enabled defaults to OFF deliberately (live measurement
        # 2026-08-31): on the golden query both tiny cross-encoders ranked
        # obscure keyword-dense docs ABOVE Inception, while pure RRF put it
        # #1-#2 on every tier. Whether reranking nets positive is a benchmark
        # question for #6 — the knob stays for the A/B (ADR 0004).
        self.db = db
        self.vector_store = vector_store
        self.rag_version = rag_version
        self.search_provider = search_provider  # #11: cloud provider for the default collection
        #: #65 (map #64 D16): repr of the exception the most recent hybrid
        #: retrieve() swallowed in the dense leg; None when dense ran. Chat
        #: keeps the BM25 fallback, the trace records it, the harness refuses.
        self.last_dense_failure: str | None = None
        #: #88 (D5): what the last retrieve() pushed into the stores — the
        #: Chroma where clause (years, shown ids) and the exclusion list — so
        #: the trace can record ``where_applied`` / ``excluded_shown``.
        self.last_where_applied: dict[str, Any] | None = None
        self.last_excluded_ids: list[int] = []
        #: #137: what the last retrieve() applied from a MoodProfile — the
        #: profile id, the floors actually enforced, and whether a floor
        #: relaxation fired (fail-open is explicit and recorded).
        self.last_profile_applied: dict[str, Any] | None = None
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
        shown_ids: Sequence[int] | None = None,
        boost: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Returns the final ranked movies for one router decision.

        #88 (D5): ``shown_ids`` and the filter's years are pushed INTO the
        stores — a Chroma where clause on the dense leg, ``NOT IN`` on the
        SQL legs — so the fetched pool is already fresh and constrained;
        genre/actor/exclusion matching stays in the post-filter safety net.

        #137: ``boost`` is a MoodProfile's soft-boost spec (see
        ``boost_spec_of``) — RRF fusion tilts by genre/runtime/popularity/
        revenue weight; the profile's FLOORS ride in ``routing.filters``
        (``vote_count_min``) and post-filter every path. If a floor empties
        an otherwise non-empty pool, the floor relaxes once and the
        relaxation is recorded (fail-open, on the record).
        """
        self.last_dense_failure = None
        self.last_profile_applied = (
            {"profile_id": boost.get("profile_id"), "floors_relaxed": False}
            if boost
            else None
        )
        where = build_where_clause(routing.filters, shown_ids)
        self.last_where_applied = where
        self.last_excluded_ids = list(shown_ids or [])
        if not routing.requires_rag:
            return []

        routing = self._resolve_person(routing)
        if routing is None:  # named person not in the archive (#24)
            return []

        if self._use_sql_path(routing):
            return self._retrieve_sql(routing, top_k, excluded_ids=self.last_excluded_ids)

        dense = self._retrieve_dense(query, candidate_pool, where)
        sparse = self._retrieve_bm25(self.sparse_query(query, routing.filters), candidate_pool)
        fused = self._rrf_fuse(dense, sparse, boost=boost)
        if self.last_dense_failure is not None:
            fused = [r.model_copy(update={"dense_failed": True}) for r in fused]

        # Uniform post-filtering: positive filters + exclusions on the small
        # candidate pool (BM25 has no metadata columns; this keeps one path).
        fused = [r for r in fused if self._is_allowed(r.movie, routing.filters)]
        if shown_ids:  # #88 review: the sparse leg and dense-failure fallback
            # bypass the store-level $nin — exclude shown ids here so the
            # hybrid path honors the fresh-pool guarantee end to end.
            excluded = set(shown_ids)
            fused = [r for r in fused if r.movie.id not in excluded]

        # #137 fail-open: profile floors must not blank a turn. If the pool
        # had candidates but the floor-filtered set is empty, retry once
        # without the mood floor and record the relaxation.
        if not fused and boost and routing.filters and routing.filters.vote_count_min is not None:
            relaxed = routing.model_copy(
                update={"filters": routing.filters.model_copy(update={"vote_count_min": None})}
            )
            dense = self._retrieve_dense(query, candidate_pool, where)
            sparse = self._retrieve_bm25(self.sparse_query(query, relaxed.filters), candidate_pool)
            fused = self._rrf_fuse(dense, sparse, boost=boost)
            fused = [r for r in fused if self._is_allowed(r.movie, relaxed.filters)]
            if shown_ids:
                excluded = set(shown_ids)
                fused = [r for r in fused if r.movie.id not in excluded]
            if self.last_profile_applied is not None:
                self.last_profile_applied["floors_relaxed"] = True

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

    def _retrieve_sql(
        self,
        routing: QueryRoutingDecision,
        top_k: int,
        excluded_ids: Sequence[int] | None = None,
    ) -> list[RetrievalResult]:
        if routing.superlative:
            s = routing.superlative
            movies = self.db.query_superlative(
                metric=s.metric,
                direction=s.direction,
                year=s.year,
                genre=s.genre,
                limit=top_k * 2,  # headroom so post-filtering can't empty the page
                excluded_ids=excluded_ids,  # #88: exclude in-query, not post-trim
            )
        else:
            movies = self.db.search_metadata_filters(
                routing.filters, limit=top_k * 2, excluded_ids=excluded_ids,
            )

        movies = [m for m in movies if self._is_allowed(m, routing.filters)]
        return [
            RetrievalResult(movie=m, score=float(rank + 1), source="sql")
            for rank, m in enumerate(movies[:top_k])
        ]

    # --- hybrid path ------------------------------------------------------------

    def _retrieve_dense(
        self, query: str, top_k: int, where_filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        try:
            return self.vector_store.search(
                query=query, version_name=self.rag_version, top_k=top_k,
                provider=self.search_provider,
                where_filter=where_filter,  # #88: years + shown ids at the store
            )
        except Exception as exc:  # noqa: BLE001 — fallback is deliberate and RECORDED
            # Missing/legacy collection or a failed provider call must not kill
            # retrieval — BM25 carries on — but the loss is never silent (#65).
            self.last_dense_failure = f"{type(exc).__name__}: {exc}"
            return []

    def _retrieve_bm25(self, query: str, top_k: int) -> list[MovieRecord]:
        return self.db.search_bm25(query, limit=top_k)

    @staticmethod
    def sparse_query(query: str, filters: MetadataFilterCriteria | None) -> str:
        """Deliberate BM25 query construction (#32).

        Excluded-entity tokens (actors, genres) must never enter FTS5 as
        positive keywords — BM25 rewards title matches on them (e.g. 'no Tom
        Cruise' retrieving 'Speed 2: Cruise Control'). Exclusions act only as
        post-retrieval filters; unrelated tokens pass through untouched.
        """
        if not filters:
            return query
        banned: set[str] = set()
        for entity in (*filters.excluded_actors, *filters.excluded_genres):
            banned.update(token.lower() for token in entity.split())
        if not banned:
            return query
        kept = [token for token in query.split() if token.lower() not in banned]
        return " ".join(kept)

    def _rrf_fuse(
        self,
        dense: list[SearchResult],
        sparse: list[MovieRecord],
        boost: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Reciprocal Rank Fusion: score(d) = sum(w_i / (k + rank_i)).

        hybrid_alpha weights the dense list (1.0 = dense only, 0.0 = BM25 only).

        #137: with a MoodProfile boost spec, each fused candidate gains
        ``scale * NORMALIZER * term`` where term = genre affinity (sum of
        matching weights) + popularity/revenue log-normalized weights + a
        fixed runtime-band term. NEGATIVE weights demote (hidden gem).
        Pool membership never changes — boosts only reorder pool members.
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

        scale = (boost or {}).get("scale", 1.0)
        genre_boosts = (boost or {}).get("genre_boosts") or {}
        rt_min = (boost or {}).get("runtime_boost_min")
        w_pop = (boost or {}).get("popularity_boost", 0.0)
        w_rev = (boost or {}).get("revenue_boost", 0.0)

        results = []
        for entry in entries.values():
            score = 0.0
            if entry["dense_rank"] is not None:
                score += w_dense / (self.RRF_K + entry["dense_rank"])
            if entry["sparse_rank"] is not None:
                score += w_sparse / (self.RRF_K + entry["sparse_rank"])
            if boost:
                movie: MovieRecord = entry["movie"]
                term = 0.0
                have = {g.lower() for g in movie.genres}
                term += sum(w for g, w in genre_boosts.items() if g.lower() in have)
                if w_pop:
                    term += w_pop * min(math.log1p(movie.popularity) / 10.0, 1.0)
                if w_rev:
                    term += w_rev * min(math.log1p(movie.revenue) / 25.0, 1.0)
                if rt_min is not None and movie.runtime >= rt_min:
                    term += self.BOOST_RUNTIME_WEIGHT
                score += scale * self.BOOST_NORMALIZER * max(-self.BOOST_TERM_CAP, min(self.BOOST_TERM_CAP, term))
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
        if filters.vote_count_min is not None and movie.vote_count < filters.vote_count_min:
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
