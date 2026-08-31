"""Live integration tests for ChromaDB vector store and FastEmbed ONNX models."""

import time
import pytest
from src.indexing.vector_store import MovieVectorStore, SearchResult
from src.domain.movie import MovieRecord


@pytest.fixture(scope="module")
def vector_store() -> MovieVectorStore:
    """Fixture providing access to the persistent ChromaDB collections."""
    return MovieVectorStore("data/chroma_db")


@pytest.mark.integration
def test_vector_store_collections_populated(vector_store: MovieVectorStore):
    """Verifies that all 3 benchmark milestone collections exist and are populated."""
    collections = vector_store.list_collections()
    assert "v1_0_baseline" in collections
    assert "v1_1_enriched" in collections
    assert "v1_2_bge_hybrid" in collections

    assert vector_store.count("v1_0_baseline") == 9119
    assert vector_store.count("v1_1_enriched") == 9119
    assert vector_store.count("v1_2_bge_hybrid") == 9119


@pytest.mark.integration
def test_semantic_search_dream_heist(vector_store: MovieVectorStore):
    """Verifies that semantic plot query retrieves Inception with high cosine similarity."""
    results = vector_store.search(
        query="mind-bending dream heist in the subconscious",
        version_name="v1_2_bge_hybrid",
        embedding_model="BAAI/bge-small-en-v1.5",
        top_k=5
    )

    assert len(results) > 0
    top_result = results[0]
    assert isinstance(top_result, SearchResult)
    assert isinstance(top_result.movie, MovieRecord)
    assert top_result.score > 0.60

    # Inception should be in top results
    top_titles = [r.movie.title for r in results]
    assert any("Inception" in t for t in top_titles)


@pytest.mark.integration
def test_vector_search_latency_cpu_benchmark(vector_store: MovieVectorStore):
    """Verifies that query vectorization + ChromaDB search runs under 350ms on CPU."""
    # Warmup
    vector_store.search("space exploration", version_name="v1_2_bge_hybrid", top_k=5)

    # Benchmark
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
    """Verifies that ChromaDB metadata filtering properly constrains release year."""
    results = vector_store.search(
        query="superhero action blockbusters",
        version_name="v1_2_bge_hybrid",
        top_k=5,
        where_filter={"release_year": {"$gte": 2020}}
    )

    assert len(results) > 0
    for r in results:
        assert r.movie.release_year >= 2020, f"Expected year >= 2020, got {r.movie.release_year} for {r.movie.title}"


@pytest.mark.integration
def test_vector_search_empty_query_safety(vector_store: MovieVectorStore):
    """Verifies that searching blank or whitespace strings returns empty list safely."""
    assert vector_store.search("   ") == []
    assert vector_store.search("") == []
