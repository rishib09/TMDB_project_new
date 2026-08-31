"""Movie and Cast domain entities with computed TMDB image URLs and dynamic dense text serialization."""

from typing import List, Optional
from pydantic import BaseModel, Field, computed_field


class CastMember(BaseModel):
    """Represents a cast member in a movie."""
    name: str
    character: str = ""
    order: int = 0
    profile_path: Optional[str] = None

    @computed_field
    @property
    def profile_url(self) -> Optional[str]:
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
    genres: List[str] = Field(default_factory=list)
    cast: List[CastMember] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

    @computed_field
    @property
    def poster_url(self) -> str:
        """Constructs high-resolution TMDB poster URL with fallback SVG."""
        if self.poster_path:
            return f"https://image.tmdb.org/t/p/w500{self.poster_path}"
        return "https://via.placeholder.com/500x750?text=No+Poster+Available"

    @computed_field
    @property
    def backdrop_url(self) -> Optional[str]:
        """Constructs wide backdrop image URL."""
        if self.backdrop_path:
            return f"https://image.tmdb.org/t/p/w1280{self.backdrop_path}"
        return None

    def to_dense_text(
        self,
        strategy: str = "enriched_metadata",
        token_budget: int = 256
    ) -> str:
        """Serializes movie metadata dynamically scaled to fit within the target token budget.

        Uses a prioritized tier-packing algorithm to maximize semantic signal without truncation:
          - 256 tokens: Core identity + top 4 cast + top 6 keywords + trimmed synopsis
          - 512 tokens: Core identity + top 10 cast with roles + all keywords + tagline + full synopsis + runtime/rating
          - 1024+ tokens: Full cast (top 20) + crew + production financials + full synopsis
        """
        if strategy == "baseline":
            # Strategy A (v1.0): Overview only
            return f"{self.title} ({self.release_year}): {self.overview}".strip()

        # Target character budget (1 token ~ 3.8 characters with safety margin)
        char_budget = int(token_budget * 3.8)

        # Tier 1: Core Identity (Mandatory)
        genres_str = ", ".join(self.genres)
        core_parts = [f"Title: {self.title} ({self.release_year})"]
        if self.director:
            core_parts.append(f"Director: {self.director}")
        if genres_str:
            core_parts.append(f"Genres: {genres_str}")

        # Tier 2: High-Signal Metadata scaled to token budget
        if token_budget <= 256:
            # 256 tokens: Top 4 cast names, top 6 keywords
            cast_names = ", ".join([c.name for c in self.cast[:4]])
            keywords_str = ", ".join(self.keywords[:6])
            if cast_names:
                core_parts.append(f"Cast: {cast_names}")
            if keywords_str:
                core_parts.append(f"Themes: {keywords_str}")
            if self.tagline:
                core_parts.append(f"Tagline: {self.tagline}")
        elif token_budget <= 512:
            # 512 tokens: Top 10 cast with character roles, all keywords
            cast_details = [
                f"{c.name} as {c.character}" if c.character else c.name
                for c in self.cast[:10]
            ]
            if cast_details:
                core_parts.append(f"Cast: {', '.join(cast_details)}")
            if self.keywords:
                core_parts.append(f"Themes: {', '.join(self.keywords)}")
            if self.tagline:
                core_parts.append(f"Tagline: {self.tagline}")
            if self.runtime:
                core_parts.append(f"Runtime: {self.runtime} mins")
            if self.vote_average > 0:
                core_parts.append(f"Rating: {self.vote_average:.1f}/10 ({self.vote_count} votes)")
        else:
            # 1024+ tokens: Full cast (top 20), production details, financials
            cast_details = [
                f"{c.name} as {c.character}" if c.character else c.name
                for c in self.cast[:20]
            ]
            if cast_details:
                core_parts.append(f"Cast: {', '.join(cast_details)}")
            if self.keywords:
                core_parts.append(f"Themes: {', '.join(self.keywords)}")
            if self.tagline:
                core_parts.append(f"Tagline: {self.tagline}")
            if self.runtime:
                core_parts.append(f"Runtime: {self.runtime} mins")
            if self.budget > 0 or self.revenue > 0:
                core_parts.append(f"Financials: Budget ${self.budget:,} | Box Office ${self.revenue:,}")
            if self.imdb_id:
                core_parts.append(f"IMDb: {self.imdb_id}")

        current_header = "\n".join(core_parts)
        remaining_chars = char_budget - len(current_header) - 12  # 12 chars for "\nSynopsis: "

        # Tier 3: Synopsis / Overview fit into remaining budget
        if self.overview and remaining_chars > 50:
            if len(self.overview) <= remaining_chars:
                core_parts.append(f"Synopsis: {self.overview}")
            else:
                # Truncate synopsis cleanly at nearest whitespace boundary
                truncated_overview = self.overview[:remaining_chars].rsplit(" ", 1)[0] + "..."
                core_parts.append(f"Synopsis: {truncated_overview}")

        return "\n".join(core_parts)
