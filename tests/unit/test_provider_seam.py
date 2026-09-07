"""#11 Phase 1: EmbeddingProvider seam + column presets — unit tests.

The provider seam is the ONE new seam of this ticket: a deterministic fake
(hash-derived vectors, no network, no ONNX) drives the store end-to-end,
mirroring the FakeRouter/FakeEngine pattern of the orchestrator suite.
"""

import hashlib
from pathlib import Path

from src.domain.movie import CastMember, MovieRecord

# --- deterministic fake provider ---------------------------------------------

class FakeProvider:
    """Hash-derived vectors: deterministic, offline, instant.

    Also hosts a trivial char-based counter so budget-packing tests stay
    deterministic without loading any fastembed tokenizer.
    """

    def __init__(self, name: str, dimensions: int, max_tokens: int):
        self.name = name
        self.dimensions = dimensions
        self.max_tokens = max_tokens
        self.embedded_texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = [b / 255.0 for b in digest[: self.dimensions]]
            vectors.append(vector)
        return vectors

    def token_counter(self):
        return self.counter_at(self.max_tokens)

    @staticmethod
    def counter_at(max_tokens: int):
        from src.indexing.embeddings import CharEstimateCounter

        return CharEstimateCounter(chars_per_token=4, max_tokens=max_tokens)


def make_movie(
    id: int,
    title: str = "Test Movie",
    cast: list[dict] | None = None,
    **kwargs,
) -> MovieRecord:
    defaults = dict(
        id=id,
        title=title,
        release_year=2010,
        overview="A riveting test movie about heists and redemption.",
        director="A. Director",
        genres=["Thriller", "Drama"],
        keywords=["heist", "redemption"],
        tagline="Steal big or go home.",
        cast=[
            CastMember(name=c["name"], character=c["character"], order=c["order"])
            for c in (cast or [{"name": "Star One", "character": "Lead", "order": 0}])
        ],
    )
    defaults.update(kwargs)
    return MovieRecord(**defaults)


# --- column preset invariants (pure domain) ----------------------------------

def test_minimal_preset_has_no_cast():
    movie = make_movie(1, cast=[
        {"name": f"A{i}", "character": f"R{i}", "order": i} for i in range(10)
    ])
    text = movie.to_dense_text(columns="minimal", token_budget=512)
    assert "Director: A. Director" in text
    assert "Themes: heist" in text
    assert "Cast:" not in text
    assert "Star One" not in text
    assert "Tagline:" not in text


def test_full_preset_carries_top10_cast_in_billing_order():
    movie = make_movie(1, cast=[
        {"name": f"Actor {9 - i}", "character": f"Role {9 - i}", "order": 9 - i}
        for i in range(10)  # shuffled input: order 9 first
    ])
    text = movie.to_dense_text(columns="full", token_budget=1024)
    assert "Cast: Actor 0 as Role 0" in text          # billing order 0 leads
    assert "Actor 9 as Role 9" in text
    assert "Tagline: Steal big or go home." in text


def test_full_preset_cast_stops_at_ten():
    movie = make_movie(1, cast=[
        {"name": f"A{i}", "character": f"R{i}", "order": i} for i in range(12)
    ])
    text = movie.to_dense_text(columns="full", token_budget=1024)
    assert "A9 as R9" in text
    assert "A10" not in text and "A11" not in text


def test_minimal_preset_is_pure_vibe_document():
    """Person noise absent, theme signal present — the isolating contrast."""
    movie = make_movie(1)
    minimal = movie.to_dense_text(columns="minimal", token_budget=256)
    assert "heist" in minimal and "redemption" in minimal
    assert "Director" in minimal


def test_financials_never_enter_any_preset():
    movie = make_movie(1, budget=250_000_000, revenue=900_000_000, vote_average=8.1)
    for columns in ("minimal", "full"):
        text = movie.to_dense_text(columns=columns, token_budget=1024)
        assert "Budget:" not in text
        assert "Box office:" not in text
        assert "Rating:" not in text


def test_tagline_included_when_present_only():
    missing = make_movie(1, tagline="")
    text = missing.to_dense_text(columns="full", token_budget=1024)
    assert "Tagline:" not in text


def test_empty_fields_serialize_without_crash():
    bare = MovieRecord(id=1, title="Bare", release_year=2001)
    for columns in ("minimal", "full"):
        text = bare.to_dense_text(columns=columns, token_budget=256)
        assert "Bare" in text


def test_budget_respected_for_both_presets():
    movie = make_movie(
        1,
        cast=[{"name": f"A{i}", "character": f"R{i}", "order": i} for i in range(10)],
        overview="synopsis words " * 200,
    )
    for columns in ("minimal", "full"):
        counter = FakeProvider.counter_at(80)
        text = movie.to_dense_text(columns=columns, token_budget=80, token_counter=counter)
        assert counter.count(text) <= 80


# --- provider seam through the real store (tmp chroma, no network) -----------

def test_provider_indexes_and_searches_roundtrip(tmp_path: Path):
    from src.indexing.vector_store import MovieVectorStore

    store = MovieVectorStore(str(tmp_path / "chroma"))
    provider = FakeProvider(name="fake-embed", dimensions=16, max_tokens=128)
    movies = [make_movie(1, title="Heist Alpha"), make_movie(2, title="Slow Drama")]
    n = store.index_movies("minimal_fake-embed", movies, provider=provider, columns="minimal")
    assert n == 2

    results = store.search(
        "heist redemption", version_name="minimal_fake-embed", provider=FakeProvider(
            name="fake-embed", dimensions=16, max_tokens=128
        ), top_k=2,
    )
    assert len(results) == 2
    assert {r.movie.title for r in results} == {"Heist Alpha", "Slow Drama"}
    # the same serialization reached the embedder as reached the store
    assert all("Synopsis:" in t or "Title:" in t for t in provider.embedded_texts)


def test_collection_metadata_records_both_axes(tmp_path: Path):
    from src.indexing.vector_store import MovieVectorStore

    store = MovieVectorStore(str(tmp_path / "chroma"))
    store.index_movies(
        "full_fake-embed",
        [make_movie(1)],
        provider=FakeProvider(name="fake-embed", dimensions=8, max_tokens=64),
        columns="full",
    )
    collection = store.get_collection_checked("full_fake-embed")
    assert collection.metadata["embedding_model"] == "fake-embed"
    assert collection.metadata["columns"] == "full"
    assert collection.metadata["token_budget"] == 64


def test_packing_budget_derives_from_provider_window(tmp_path: Path):
    """token_budget omitted → the provider's max_tokens governs packing."""
    from src.indexing.vector_store import MovieVectorStore

    store = MovieVectorStore(str(tmp_path / "chroma"))
    provider = FakeProvider(name="fake-embed", dimensions=8, max_tokens=48)
    store.index_movies("minimal_fake-embed", [make_movie(1)], provider=provider, columns="minimal")
    collection = store.get_collection_checked("minimal_fake-embed")
    assert collection.metadata["token_budget"] == 48


def test_legacy_tier_path_unchanged():
    """#14's tier serialization still works bit-for-bit via the tier param."""
    movie = make_movie(1)
    text = movie.to_dense_text(tier="t1_identity", token_budget=128)
    assert "Title: Test Movie (2010)" in text
    assert "Genres: Thriller, Drama" in text
    assert "Synopsis:" in text
