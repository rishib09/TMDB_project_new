"""Integration tests for the tier-distinct ChromaDB collections (issue #14).

These run against locally built collections (`python scripts/build_vector_index.py`)
and use real FastEmbed ONNX models on CPU — no network.
"""

import time

import pytest

from src.domain.movie import MovieRecord
from src.indexing.vector_store import MovieVectorStore, SearchResult


@pytest.fixture(scope="module")
def vector_store() -> MovieVectorStore:
    """Fixture providing access to the persistent ChromaDB collections."""
    return MovieVectorStore("data/chroma_db")


# --- collection integrity -----------------------------------------------------


@pytest.mark.integration
def test_vector_store_collections_populated(vector_store: MovieVectorStore):
    """All 3 benchmark milestone collections exist, populated, and tier-profiled."""
    collections = vector_store.list_collections()
    for version_name in MovieVectorStore.TIER_PROFILES:
        assert version_name in collections
        assert vector_store.count(version_name) == 9119


@pytest.mark.integration
def test_collections_declare_embedding_model(vector_store: MovieVectorStore):
    """Each collection's metadata records its embedding model, tier and budget."""
    for version_name, profile in MovieVectorStore.TIER_PROFILES.items():
        metadata = vector_store.client.get_collection(version_name).metadata
        assert metadata["embedding_model"] == profile["embedding_model"]
        assert metadata["tier"] == profile["tier"]
        assert metadata["token_budget"] == profile["token_budget"]


@pytest.mark.integration
def test_stored_documents_within_token_budget(vector_store: MovieVectorStore):
    """THE #14 invariant, verified on real stored documents with real tokenizers."""
    sample_limit = 300
    for version_name, profile in MovieVectorStore.TIER_PROFILES.items():
        collection = vector_store.client.get_collection(version_name)
        docs = collection.get(limit=sample_limit, include=["documents"])["documents"]
        counter = vector_store.get_token_counter(profile["embedding_model"])
        over = [
            (counter.count(d), profile["token_budget"])
            for d in docs
            if counter.count(d) > profile["token_budget"]
        ]
        assert not over, f"{version_name}: docs over budget: {over[:5]}"


@pytest.mark.integration
def test_tiers_serialize_distinct_documents(vector_store: MovieVectorStore):
    """The three collections must hold genuinely different document texts."""
    ids = vector_store.client.get_collection("v1_0_baseline").get(limit=50, include=[])[
        "ids"
    ]
    texts = {}
    for version_name in MovieVectorStore.TIER_PROFILES:
        got = vector_store.client.get_collection(version_name).get(
            ids=ids, include=["documents"]
        )
        texts[version_name] = dict(zip(got["ids"], got["documents"]))

    for movie_id in ids:
        t1, t2, t3 = (
            texts["v1_0_baseline"][movie_id],
            texts["v1_1_enriched"][movie_id],
            texts["v1_2_bge_hybrid"][movie_id],
        )
        assert t1 != t2 != t3, f"movie {movie_id}: tier texts must all differ"


# --- pairing enforcement ------------------------------------------------------


@pytest.mark.integration
def test_search_rejects_cross_space_model(vector_store: MovieVectorStore):
    """Searching a collection with the WRONG model must raise, not return garbage."""
    with pytest.raises(ValueError, match="mismatch"):
        vector_store.search(
            query="space exploration",
            version_name="v1_1_enriched",
            embedding_model="jinaai/jina-embeddings-v2-base-en",
        )


@pytest.mark.integration
def test_search_auto_resolves_paired_model(vector_store: MovieVectorStore):
    """Omitting the model uses the collection's stored model — always safe."""
    results = vector_store.search(query="space exploration", version_name="v1_0_baseline")
    assert len(results) > 0


@pytest.mark.integration
def test_search_legacy_collection_refused(vector_store: MovieVectorStore, tmp_path):
    """A pre-#14 collection (no embedding_model metadata) is refused, not searched."""
    legacy_store = MovieVectorStore(str(tmp_path / "legacy_db"))
    legacy_store.client.get_or_create_collection(
        name="v1_0_baseline", metadata={"hnsw:space": "cosine"}
    )
    with pytest.raises(ValueError, match="Rebuild"):
        legacy_store.search(query="anything", version_name="v1_0_baseline")


# --- semantic quality ---------------------------------------------------------


@pytest.mark.integration
def test_semantic_search_dream_heist(vector_store: MovieVectorStore):
    """Semantic plot query retrieves Inception with healthy cosine similarity."""
    results = vector_store.search(
        query="mind-bending dream heist in the subconscious",
        version_name="v1_2_bge_hybrid",
        top_k=5
    )

    assert len(results) > 0
    top_result = results[0]
    assert isinstance(top_result, SearchResult)
    assert isinstance(top_result.movie, MovieRecord)
    assert top_result.score > 0.40  # jina-v2 cosine similarities run lower than bge

    top_titles = [r.movie.title for r in results]
    assert any("Inception" in t for t in top_titles)


@pytest.mark.integration
def test_vector_search_latency_cpu_benchmark(vector_store: MovieVectorStore):
    """Query vectorization + ChromaDB search under 350ms on CPU (768d tier)."""
    vector_store.search("space exploration", version_name="v1_2_bge_hybrid", top_k=5)

    start = time.perf_counter()
    results = vector_store.search(
        query="cyberpunk detective solving murders in futuristic city",
        version_name="v1_2_bge_hybrid",
        top_k=5
    )
    latency_ms = (time.perf_counter() - start) * 1000.0

    assert len(results) == 5
    assert latency_ms < 350.0, f"Vector search took {latency_ms:.2f}ms, expected < 350ms on CPU"


@pytest.mark.integration
def test_vector_search_metadata_year_filter(vector_store: MovieVectorStore):
    """ChromaDB metadata filtering properly constrains release year."""
    results = vector_store.search(
        query="superhero action blockbusters",
        version_name="v1_2_bge_hybrid",
        top_k=5,
        where_filter={"release_year": {"$gte": 2020}}
    )

    assert len(results) > 0
    for r in results:
        assert r.movie.release_year >= 2020


@pytest.mark.integration
def test_vector_search_empty_query_safety(vector_store: MovieVectorStore):
    """Blank or whitespace queries return empty lists safely."""
    assert vector_store.search("   ") == []
    assert vector_store.search("") == []


# --- golden-query smoke eval (behavioral tier verification) --------------------

GOLDEN_QUERIES = [
    # (query, expected title substring) — dataset spans 1970-2026
    ("mind-bending dream heist inside the subconscious", "Inception"),
    ("simulated reality where humans are trapped by machines", "Matrix"),
    ("dinosaur theme park goes horribly wrong", "Jurassic Park"),
    ("toy cowboy and spaceman come alive when humans leave", "Toy Story"),
    ("lion cub heir flees then returns to claim his kingdom", "Lion King"),
    ("teenager travels through time in a DeLorean sports car", "Back to the Future"),
    ("office tower seized by terrorists on Christmas Eve", "Die Hard"),
    ("hitmen, a glowing briefcase and a diner conversation", "Pulp Fiction"),
    ("green ogre and a talkative donkey rescue a princess", "Shrek"),
    ("father fish crosses the ocean to find his lost son", "Finding Nemo"),
    ("simple-hearted man witnesses decades of American history", "Forrest Gump"),
    ("boy left home alone defends house from burglars", "Home Alone"),
    ("news reporter relives the same day over and over", "Groundhog Day"),
    ("ship sinks after striking an iceberg on its maiden voyage", "Titanic"),
]

VIBE_QUERIES = [
    "spooky haunted house atmosphere",
    "feel-good underdog sports story",
    "cozy small town romance",
    "gritty dystopian surveillance state",
    "heartwarming animal adventure for the whole family",
]

_HIT_RATE_FLOOR = 0.6  # every tier must hit >= 60% of golden queries in top-5


@pytest.mark.integration
def test_smoke_eval_all_tiers_relevant(vector_store: MovieVectorStore):
    """Every tier must retrieve the expected movie in top-5 for >= 60% of queries."""
    rates = {}
    for version_name in MovieVectorStore.TIER_PROFILES:
        hits = 0
        for query, expected in GOLDEN_QUERIES:
            results = vector_store.search(query=query, version_name=version_name, top_k=5)
            titles = [r.movie.title for r in results]
            if any(expected.lower() in t.lower() for t in titles):
                hits += 1
        rates[version_name] = hits / len(GOLDEN_QUERIES)
        assert rates[version_name] >= _HIT_RATE_FLOOR, (
            f"{version_name} hit rate {rates[version_name]:.2f} < {_HIT_RATE_FLOOR}"
        )
    print("\nGolden-query hit rates:", {k: f"{v:.0%}" for k, v in rates.items()})


@pytest.mark.integration
def test_smoke_eval_tiers_behave_differently(vector_store: MovieVectorStore):
    """The tiers must be behaviorally distinct, not just numerically distinct.

    At least half of the vibe queries should return different top-1 results
    across tiers (different embedding spaces rank differently in practice).
    """
    different_top1 = 0
    for query in VIBE_QUERIES:
        top1 = set()
        for version_name in MovieVectorStore.TIER_PROFILES:
            results = vector_store.search(query=query, version_name=version_name, top_k=1)
            top1.add(results[0].movie.id if results else None)
        if len(top1) > 1:
            different_top1 += 1

    assert different_top1 >= len(VIBE_QUERIES) // 2, (
        f"Only {different_top1}/{len(VIBE_QUERIES)} vibe queries differ across tiers — "
        f"the collections may be embedding the same space."
    )
