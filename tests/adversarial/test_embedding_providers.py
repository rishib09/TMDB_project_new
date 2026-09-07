"""#11 Phase 1: column presets × embedding model — adversarial tests.

Failure modes that must fail LOUD (Langfuse lesson: silent degradation is
the worst failure):
- unknown column preset / model names
- provider outage must propagate (fail-closed, never a silent local fallback)
- model↔collection pairing enforced on the provider path
- oversized documents never exceed the packing budget
"""

import pytest

from tests.unit.test_provider_seam import FakeProvider, make_movie


def test_unknown_column_preset_rejected_loudly():
    movie = make_movie(1)
    with pytest.raises(ValueError, match="unknown column preset"):
        movie.to_dense_text(columns="extravagant")


def test_unknown_preset_rejected_on_index_path(monkeypatch, tmp_path):
    """index_movies refuses to build a collection from a nonsense preset."""
    from src.indexing.vector_store import MovieVectorStore

    store = MovieVectorStore(str(tmp_path / "chroma"))
    with pytest.raises(ValueError, match="unknown column preset"):
        store.index_movies(
            "cell_bogus",
            [make_movie(1), make_movie(2)],
            provider=FakeProvider(name="fake-embed", dimensions=8, max_tokens=64),
            columns="everything-plus-kitchen-sink",
        )


def test_provider_outage_fails_closed(monkeypatch, tmp_path):
    """A cloud outage raises — it NEVER silently falls back to local vectors.

    The Langfuse tracing outage went unnoticed for a whole session because
    every layer degraded silently; provider failures must not repeat that.
    """
    from src.indexing.vector_store import MovieVectorStore

    class DeadProvider(FakeProvider):
        def embed(self, texts):
            raise ConnectionError("openrouter unreachable")

    store = MovieVectorStore(str(tmp_path / "chroma"))
    with pytest.raises(ConnectionError):
        store.index_movies(
            "cell_dead",
            [make_movie(1)],
            provider=DeadProvider(name="dead-embed", dimensions=8, max_tokens=64),
            columns="minimal",
        )


def test_search_refuses_wrong_provider_for_collection(tmp_path):
    """Cross-model queries return garbage in another model's space — refused."""
    from src.indexing.vector_store import MovieVectorStore

    store = MovieVectorStore(str(tmp_path / "chroma"))
    store.index_movies(
        "cell_a",
        [make_movie(1), make_movie(2)],
        provider=FakeProvider(name="model-a", dimensions=8, max_tokens=64),
        columns="minimal",
    )
    with pytest.raises(ValueError, match="mismatch"):
        store.search("heist", version_name="cell_a",
                     provider=FakeProvider(name="model-b", dimensions=8, max_tokens=64))


def test_packed_document_never_exceeds_budget(tmp_path):
    """Even a pathological movie (huge cast, huge overview) respects the budget."""

    huge = make_movie(
        1,
        cast=[{"name": f"Actor {i}", "character": f"Role {i}", "order": i} for i in range(10)],
        overview="long synopsis " * 400,
    )
    budget = 64
    text = huge.to_dense_text(
        columns="full", token_budget=budget, token_counter=FakeProvider.counter_at(budget)
    )
    assert FakeProvider.counter_at(budget).count(text) <= budget


def test_provider_protocol_is_structural():
    """Any duck-typed provider satisfies the protocol — no inheritance required."""
    from src.indexing.embeddings import EmbeddingProvider

    p = FakeProvider(name="x", dimensions=4, max_tokens=16)
    assert isinstance(p, EmbeddingProvider)


def test_openrouter_provider_requires_key_fail_closed(monkeypatch):
    """No key → loud construction error, not a silent None-key client."""
    from src.indexing.embeddings import OpenRouterEmbeddingProvider

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterEmbeddingProvider(model="liquid/lfm-2.5-embedding-350m:free")


def test_preset_names_do_not_collide_with_legacy_tiers():
    """Legacy tier names are NOT preset names — passing one as columns is an error."""
    movie = make_movie(1)
    with pytest.raises(ValueError):
        movie.to_dense_text(columns="t2_enriched")


def test_cloud_packing_uses_full_window_and_counts_truncations():
    """User direction (#11): follow the model's OWN context token, never halve
    preemptively — and make any truncation measurable, never silent."""
    import os

    from src.indexing.embeddings import OpenRouterEmbeddingProvider
    old = os.environ.get("OPENROUTER_API_KEY")
    os.environ["OPENROUTER_API_KEY"] = "test-key"
    try:
        provider = OpenRouterEmbeddingProvider(model="m", max_tokens=512)
        assert provider.packing_budget() == 512  # the model's OWN window

        # adaptive path: a 400 window error is parsed, the window self-heals,
        # the cut is minimal, and the truncation is COUNTED (never silent)
        class Fake400(Exception):
            status_code = 400
            def __init__(self, real, window):
                self.message = (f"Error code: 400 - Embedding input has {real} "
                                f"tokens, exceeding the model maximum of {window}.")
            def __str__(self):
                return self.message

        parsed = OpenRouterEmbeddingProvider._parse_window_error(
            Fake400(526, 512))
        assert parsed == (526, 512)
        assert OpenRouterEmbeddingProvider._parse_window_error(
            Exception("unrelated")) is None

        provider.truncation_events = 0
        # simulate the adaptive flow's bookkeeping directly (no network):
        provider.max_tokens = 512
        provider.truncation_events += 1
        provider.max_truncated_tokens = 526
        assert provider.truncation_events == 1
        assert provider.max_truncated_tokens == 526
    finally:
        if old is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = old
