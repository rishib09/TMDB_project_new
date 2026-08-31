"""Routing, Intent taxonomy, and deterministic filter criteria schemas."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """7-Class Intent Taxonomy for Maya Movie Agent."""
    GREETING = "GREETING"
    CAPABILITIES = "CAPABILITIES"
    SEMANTIC_SEARCH = "SEMANTIC_SEARCH"
    ATTRIBUTE_FILTER = "ATTRIBUTE_FILTER"
    SUPERLATIVE_RANKING = "SUPERLATIVE_RANKING"
    NEGATION_EXCLUSION = "NEGATION_EXCLUSION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class SuperlativeMetric(str, Enum):
    """Supported metrics for deterministic SQL superlative queries."""
    REVENUE = "REVENUE"
    BUDGET = "BUDGET"
    RATING = "RATING"
    POPULARITY = "POPULARITY"
    RUNTIME = "RUNTIME"
    VOTE_COUNT = "VOTE_COUNT"


class SuperlativeCriteria(BaseModel):
    """Structured constraints for superlative queries (e.g. highest grossing 1970 movie)."""
    metric: SuperlativeMetric
    direction: str = "DESC"  # "DESC" (highest/longest) or "ASC" (lowest/shortest)
    year: Optional[int] = None
    genre: Optional[str] = None
    limit: int = 5


class MetadataFilterCriteria(BaseModel):
    """Deterministic relational filters for metadata-driven queries."""
    exact_year: Optional[int] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    genres: List[str] = Field(default_factory=list)
    director: Optional[str] = None
    cast_member: Optional[str] = None
    excluded_genres: List[str] = Field(default_factory=list)
    excluded_actors: List[str] = Field(default_factory=list)


class QueryRoutingDecision(BaseModel):
    """Structured JSON output enforced on OpenRouter intent classifier."""
    intent: IntentType
    confidence: float = Field(..., ge=0.0, le=1.0, description="Routing confidence score 0.0 to 1.0")
    standalone_query: str = Field(..., description="Coreference-resolved standalone search string")
    requires_rag: bool = Field(..., description="Whether retrieval is needed (False for greetings/out-of-scope)")
    is_superlative: bool = Field(default=False, description="True if query asks for extreme ranking")
    superlative: Optional[SuperlativeCriteria] = None
    filters: Optional[MetadataFilterCriteria] = None
    reasoning: str = Field(default="", description="Chain-of-thought routing justification")
