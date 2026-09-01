"""Movie and Cast domain entities with computed TMDB image URLs and tiered dense text serialization."""

from typing import ClassVar, Protocol

from pydantic import BaseModel, Field, computed_field


class TokenCounter(Protocol):
    """Tokenization interface injected from the indexing layer (keeps domain pure).

    Implemented by wrapping the target embedding model's real tokenizer, so
    packing decisions are exact rather than character-estimated (issue #14).
    """

    def count(self, text: str) -> int:
        """Number of model tokens in ``text``."""
        ...

    def truncate(self, text: str, max_tokens: int) -> str:
        """Longest prefix of ``text`` using at most ``max_tokens`` tokens."""
        ...


class CastMember(BaseModel):
    """Represents a cast member in a movie."""
    name: str
    character: str = ""
    order: int = 0
    profile_path: str | None = None

    @computed_field
    @property
    def profile_url(self) -> str | None:
        """Constructs secure TMDB headshot image URL."""
        if self.profile_path:
            return f"https://image.tmdb.org/t/p/w185{self.profile_path}"
        return None


class MovieRecord(BaseModel):
    """Normalized domain entity representing a TMDB movie (1970-2026)."""
    id: int
    title: str
    original_title: str = ""
    tagline: str = ""
    overview: str = ""
    release_date: str = ""
    release_year: int
    runtime: int = 0
    vote_average: float = 0.0
    vote_count: int = 0
    popularity: float = 0.0
    director: str = ""
    budget: int = 0
    revenue: int = 0
    imdb_id: str = ""
    poster_path: str = ""
    backdrop_path: str = ""
    genres: list[str] = Field(default_factory=list)
    cast: list[CastMember] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def poster_url(self) -> str:
        """Constructs high-resolution TMDB poster URL with fallback SVG."""
        if self.poster_path:
            return f"https://image.tmdb.org/t/p/w500{self.poster_path}"
        return "https://via.placeholder.com/500x750?text=No+Poster+Available"

    @computed_field
    @property
    def backdrop_url(self) -> str | None:
        """Constructs wide backdrop image URL with fallback."""
        if self.backdrop_path:
            return f"https://image.tmdb.org/t/p/w1280{self.backdrop_path}"
        return None

    # --- tiered dense-text serialization (issue #14) -------------------------

    #: Default per-tier token budgets; each tier also carries its own input
    #: set. Budgets chosen against the 1970-2026 corpus: content tops out
    #: below 1024 tokens, so larger budgets would pad, not enrich.
    DEFAULT_TIER_BUDGETS: ClassVar[dict] = {
        "t1_identity": 128,   # MiniLM's REAL fastembed window is 128 tokens,
                              # not the model-card 256 (measured 2026-08-31)
        "t2_enriched": 512,
        "t3_exhaustive": 1024,
    }

    #: Synopsis never gets fewer tokens than this, else it is omitted whole.
    _MIN_SYNOPSIS_TOKENS: ClassVar[int] = 24

    @staticmethod
    def _money_words(amount: int) -> str:
        """Spells an amount as coarse words — raw digits are weak embedding signal."""
        if amount >= 1_000_000_000:
            billions = amount / 1_000_000_000
            return f"${billions:.1f} billion".replace(".0 ", " ")
        if amount >= 1_000_000:
            return f"${round(amount / 1_000_000)} million"
        if amount > 0:
            return f"${round(amount / 1_000)} thousand"
        return ""

    def _tier_parts(self, tier: str) -> tuple[list[str], str]:
        """Ordered (priority, highest first) document parts plus the synopsis.

        Input sets per issue #14:
        - t1_identity:   title, year, genres, overview
        - t2_enriched:   + director, top-10 cast with characters, top-12
                         keywords, tagline, runtime, rating
        - t3_exhaustive: + original_title, ALL keywords, financials as words
        popularity / vote_count / imdb_id are deliberately excluded everywhere
        (SQL-tool / filter material, not semantic signal).
        """
        genres_str = ", ".join(self.genres)
        parts = [f"Title: {self.title} ({self.release_year})"]

        if tier == "t1_identity":
            if genres_str:
                parts.append(f"Genres: {genres_str}")
            return parts, self.overview

        if self.director:
            parts.append(f"Director: {self.director}")
        if genres_str:
            parts.append(f"Genres: {genres_str}")

        cast_details = [
            f"{c.name} as {c.character}" if c.character else c.name
            for c in self.cast[:10]
        ]
        if cast_details:
            parts.append(f"Cast: {', '.join(cast_details)}")

        if tier == "t3_exhaustive":
            if self.keywords:
                parts.append(f"Themes: {', '.join(self.keywords)}")
        elif self.keywords:
            parts.append(f"Themes: {', '.join(self.keywords[:12])}")

        if self.tagline:
            parts.append(f"Tagline: {self.tagline}")
        if self.runtime:
            parts.append(f"Runtime: {self.runtime} mins")
        if self.vote_average > 0:
            parts.append(f"Rating: {self.vote_average:.1f}/10")

        if tier == "t3_exhaustive":
            if self.original_title and self.original_title != self.title:
                parts.append(f"Original title: {self.original_title}")
            if self.budget > 0:
                parts.append(f"Budget: {self._money_words(self.budget)}")
            if self.revenue > 0:
                parts.append(f"Box office: {self._money_words(self.revenue)}")

        return parts, self.overview

    def to_dense_text(
        self,
        tier: str = "t2_enriched",
        token_budget: int | None = None,
        token_counter: TokenCounter | None = None,
    ) -> str:
        """Serializes movie metadata into a tier-shaped document within a token budget.

        With ``token_counter`` (the target model's real tokenizer) packing is
        exact: the stored text never exceeds ``token_budget`` model tokens and
        the synopsis is cut on token boundaries via offsets — no silent
        truncation, no character estimates. Without one, a chars/3.8 estimate
        is used (offline convenience only; never for index builds).
        """
        budget = token_budget or self.DEFAULT_TIER_BUDGETS[tier]
        parts, overview = self._tier_parts(tier)

        if token_counter is None:
            return self._pack_by_char_estimate(parts, overview, budget)
        return self._pack_exact(parts, overview, budget, token_counter)

    def _pack_exact(
        self,
        parts: list[str],
        overview: str,
        budget: int,
        counter: TokenCounter,
    ) -> str:
        """Greedily includes highest-priority parts that fit; synopsis fills the rest."""
        included: list[str] = []
        used = 0
        for part in parts:
            cost = counter.count(part) + 1  # +1 for the joining newline
            if used + cost <= budget:
                included.append(part)
                used += cost

        remaining = budget - used
        synopsis_label = "Synopsis: "
        if (
            overview
            and remaining >= self._MIN_SYNOPSIS_TOKENS + counter.count(synopsis_label)
        ):
            synopsis_budget = remaining - counter.count(synopsis_label) - 1
            included.append(synopsis_label + counter.truncate(overview, synopsis_budget))

        return "\n".join(included)

    def _pack_by_char_estimate(
        self,
        parts: list[str],
        overview: str,
        budget: int,
    ) -> str:
        """Fallback packing at ~3.8 chars/token (offline only, never for builds)."""
        char_budget = int(budget * 3.8)
        included: list[str] = []
        used = 0
        for part in parts:
            cost = len(part) + 1
            if used + cost <= char_budget:
                included.append(part)
                used += cost

        remaining = char_budget - used
        synopsis_label = "Synopsis: "
        if overview and remaining >= len(synopsis_label) + 50:
            synopsis_budget = remaining - len(synopsis_label) - 1
            trimmed = overview[:synopsis_budget]
            if len(overview) > synopsis_budget:
                trimmed = trimmed.rsplit(" ", 1)[0] + "..."
            included.append(synopsis_label + trimmed)

        return "\n".join(included)
