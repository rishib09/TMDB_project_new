"""Movie and Cast domain entities with computed TMDB image URLs and embedding serializers."""

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

    def to_dense_text(self, strategy: str = "enriched_metadata", token_budget: int = 256) -> str:
        """Serializes high-signal movie metadata for embedding models without token truncation."""
        ##TODO: confirm token_budget can we increased to 512,1024,2048,4096 to experimentConfig
        top_cast_names = ", ".join([c.name for c in self.cast[:10]])
        genres_str = ", ".join(self.genres)
        keywords_str = ", ".join(self.keywords[:6])

        if strategy == "baseline":
            # Strategy A (v1.0): Overview only
            return f"{self.title} ({self.release_year}): {self.overview}".strip()

        # Strategy B/C (v1.1 / v1.2): Enriched structured semantic representation
        parts = [f"Title: {self.title} ({self.release_year})"]
        if self.director:
            parts.append(f"Director: {self.director}")
        if genres_str:
            parts.append(f"Genres: {genres_str}")
        if top_cast_names:
            parts.append(f"Cast: {top_cast_names}")
        if keywords_str:
            parts.append(f"Themes: {keywords_str}")
        if self.tagline:
            parts.append(f"Tagline: {self.tagline}")
        if self.overview:
            parts.append(f"Synopsis: {self.overview}")

        return "\n".join(parts)
