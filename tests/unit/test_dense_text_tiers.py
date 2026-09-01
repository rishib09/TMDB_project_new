"""Unit tests for tier-distinct dense text serialization (issue #14).

Covers the hard invariants from the ticket: tokenizer-exact budget compliance,
tier-distinct input sets, financials-as-words, and no silent truncation.
Uses a deterministic fake counter (1 word = 1 token) so no ONNX model is needed.
"""

import pytest

from src.domain.movie import CastMember, MovieRecord


class FakeTokenCounter:
    """Deterministic counter: 1 whitespace-separated word = 1 token."""

    def count(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])


@pytest.fixture
def fake_counter():
    return FakeTokenCounter()


@pytest.fixture
def bloated_movie() -> MovieRecord:
    """A movie stressing every packing dimension: huge overview, 98 keywords, full cast."""
    return MovieRecord(
        id=1,
        title="Bloated: The Movie",
        original_title="Bloated: The Movie (Director's Cut)",
        release_year=2020,
        director="Maximalist Director",
        runtime=240,
        vote_average=7.5,
        vote_count=1000,
        budget=200_000_000,
        revenue=1_500_000_000,
        overview=" ".join(f"plotpoint{i}" for i in range(400)),  # ~400 tokens
        genres=["Action", "Adventure", "Drama", "Thriller"],
        cast=[CastMember(name=f"Actor{i}", character=f"Role{i}") for i in range(10)],
        keywords=[f"kw{i}" for i in range(98)],  # corpus max observed
        tagline="Everything and the kitchen sink.",
        imdb_id="tt0000001",
    )


@pytest.mark.unit
def test_token_budget_never_exceeded_all_tiers(bloated_movie, fake_counter):
    """THE invariant: stored text <= budget, exact counter, worst-case movie."""
    for tier, budget in MovieRecord.DEFAULT_TIER_BUDGETS.items():
        text = bloated_movie.to_dense_text(tier=tier, token_counter=fake_counter)
        token_count = fake_counter.count(text)
        assert token_count <= budget, (
            f"{tier}: {token_count} tokens > budget {budget}"
        )


@pytest.mark.unit
def test_synopsis_trimmed_to_fit_exact_token_boundary(bloated_movie, fake_counter):
    """Overview longer than the budget is cut on token boundaries, never silently."""
    text = bloated_movie.to_dense_text(tier="t1_identity", token_counter=fake_counter)
    assert fake_counter.count(text) <= 256
    assert "Synopsis:" in text  # some synopsis content survives
    assert "plotpoint399" not in text  # the tail did NOT silently sneak in


@pytest.mark.unit
def test_tier_input_sets_are_distinct(bloated_movie, fake_counter):
    """The three tiers must serialize to meaningfully different documents."""
    texts = {
        tier: bloated_movie.to_dense_text(tier=tier, token_counter=fake_counter)
        for tier in MovieRecord.DEFAULT_TIER_BUDGETS
    }
    assert texts["t1_identity"] != texts["t2_enriched"] != texts["t3_exhaustive"]
    assert len(texts["t1_identity"]) < len(texts["t2_enriched"]) < len(texts["t3_exhaustive"])


@pytest.mark.unit
def test_excluded_fields_never_appear(bloated_movie, fake_counter):
    """popularity / vote_count / imdb_id are SQL-tool material, never embedded."""
    for tier in MovieRecord.DEFAULT_TIER_BUDGETS:
        text = bloated_movie.to_dense_text(tier=tier, token_counter=fake_counter)
        assert "IMDb" not in text
        assert "tt0000001" not in text
        assert "popularity" not in text.lower()
        assert "35000" not in text


@pytest.mark.unit
def test_t3_financials_as_words_not_digits(bloated_movie, fake_counter):
    """Financials spelled as words carry semantic signal; raw digits do not."""
    text = bloated_movie.to_dense_text(tier="t3_exhaustive", token_counter=fake_counter)
    assert "$200 million" in text
    assert "$1.5 billion" in text
    assert "200,000,000" not in text


@pytest.mark.unit
def test_t3_includes_original_title_when_different(bloated_movie, fake_counter):
    text = bloated_movie.to_dense_text(tier="t3_exhaustive", token_counter=fake_counter)
    assert "Original title:" in text


@pytest.mark.unit
def test_t3_omits_original_title_when_same_as_title(fake_counter):
    movie = MovieRecord(id=2, title="Same", original_title="Same", release_year=2000)
    text = movie.to_dense_text(tier="t3_exhaustive", token_counter=fake_counter)
    assert "Original title:" not in text


@pytest.mark.unit
def test_t2_keywords_capped_t3_all_keywords(bloated_movie, fake_counter):
    t2 = bloated_movie.to_dense_text(tier="t2_enriched", token_counter=fake_counter)
    t3 = bloated_movie.to_dense_text(tier="t3_exhaustive", token_counter=fake_counter)
    assert "kw11" in t2 and "kw12" not in t2  # top-12 cap
    assert "kw97" in t3  # all 98 keywords in tier 3


@pytest.mark.unit
def test_priority_dropping_under_tiny_budget(bloated_movie, fake_counter):
    """With a brutally small budget, lower-priority parts drop, title survives."""
    text = bloated_movie.to_dense_text(tier="t2_enriched", token_budget=40, token_counter=fake_counter)
    assert "Title: Bloated: The Movie (2020)" in text
    assert fake_counter.count(text) <= 40


@pytest.mark.unit
def test_missing_overview_and_tagline_safe(fake_counter):
    """19 movies have no overview, 1232 no tagline — serialization must not crash."""
    movie = MovieRecord(id=3, title="Sparse", release_year=1985)
    for tier in MovieRecord.DEFAULT_TIER_BUDGETS:
        text = movie.to_dense_text(tier=tier, token_counter=fake_counter)
        assert "Title: Sparse (1985)" in text


@pytest.mark.unit
def test_char_estimate_fallback_never_crashes(bloated_movie):
    """No-counter path (offline convenience) still produces bounded output."""
    text = bloated_movie.to_dense_text(tier="t2_enriched")
    assert len(text) <= int(512 * 3.8) + 200  # small slack for the synopsis label


@pytest.mark.unit
def test_real_tokenizer_padding_is_ignored():
    """Regression (issue #14): MiniLM's fastembed tokenizer pads every encode
    to a fixed length — raw len(ids) is a constant 128, which produced
    title-only t1 documents in the first build. Count via attention mask."""
    from fastembed import TextEmbedding

    from src.indexing.vector_store import FastembedTokenCounter

    counter = FastembedTokenCounter(
        TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2").model.tokenizer
    )
    short = counter.count("Title: Inception (2010)")
    long = counter.count("word " * 400)
    assert short != long, "padded tokenizer leaks constant counts"
    assert short < long
    assert 5 <= short <= 15
    assert long == 128  # MiniLM's real fastembed window: content truncates at 128
