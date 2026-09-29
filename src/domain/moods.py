"""Mood Profiles: the declarative translation of a canonical mood into
concrete retrieval signals (issue #137).

Both routing stacks classify moods into a closed vocabulary, but until this
module the classification died at retrieval as text flavor — the embedder and
BM25 alone decided what "epic" meant, which measured (2026-09-29): BM25 title
collisions (*Epic* (2013) above *Gladiator*), 0.0-vote junk surviving into the
top 5, and run-to-run variance. A mood now means a versioned, written-down
profile that CODE applies deterministically:

- ``query_phrases``   semantic anchors appended to the standalone query —
                      dilutes the single stemmed mood token in the sparse leg
                      and gives the dense leg the connotation, deterministically
- ``floors``          hard minimums merged into ``MetadataFilterCriteria`` —
                      "culturally real", not "good" (deliberately mild)
- ``boosts``          soft ranking tilts applied in RRF fusion — genre
                      affinity, runtime band, popularity/revenue weight.
                      NEGATIVE weights invert polarity (hidden gem = popular
                      demoted), which is why floors live per-profile and not
                      as a global quality gate.

Pure domain (ADR 0006): pydantic + stdlib only. The registry is curated DATA
— the Router classifies, this module interprets, the engine applies, the
Trace records, the harness grades (``epic@1`` vs ``epic@2`` under Experiment
Config). Unknown moods resolve to None and change nothing (fail-open, the
caller records it).
"""

from typing import Any, ClassVar

from pydantic import BaseModel, Field, field_validator

from src.domain.routing import MetadataFilterCriteria


class MoodFloorCriteria(BaseModel):
    """Hard minimums a profile imposes on every candidate.

    Deliberately narrow: floors encode cultural existence, not quality.
    ``min_vote_average`` stays None on most profiles — a 5.5-rated epic
    (10,000 BC) is still an epic; rating floors are a harness question.
    """

    min_vote_count: int | None = Field(
        default=None, ge=0, description="Minimum TMDB vote count (popularity-of-record floor)"
    )
    min_vote_average: float | None = Field(
        default=None, ge=0.0, le=10.0, description="Minimum TMDB vote average"
    )


class MoodProfile(BaseModel):
    """One canonical mood's deterministic retrieval translation."""

    mood: str = Field(description="Canonical mood — must be a _MOOD_VOCAB value")
    version: int = Field(default=1, ge=1, description="Registry version; the Trace pins which ran")

    query_phrases: list[str] = Field(
        default_factory=list,
        description="Connotation anchors appended verbatim to the standalone query",
    )
    genre_boosts: dict[str, float] = Field(
        default_factory=dict,
        description="Genre -> weight; soft affinity, NEVER a filter (a quiet drama-epic survives)",
    )
    floors: MoodFloorCriteria = Field(default_factory=MoodFloorCriteria)
    runtime_boost_min: int | None = Field(
        default=None, ge=0, description="Movies at/above this runtime get a fixed boost term"
    )
    popularity_boost: float = Field(
        default=0.0,
        description="Weight on normalized log popularity; NEGATIVE demotes (hidden gem)",
    )
    revenue_boost: float = Field(
        default=0.0,
        description="Weight on normalized log revenue; NEGATIVE demotes (flop signature)",
    )

    @field_validator("mood")
    @classmethod
    def _mood_canonical(cls, v: str) -> str:
        v = v.strip().casefold()
        if not v:
            raise ValueError("mood must be a non-empty canonical tag")
        return v

    @property
    def id(self) -> str:
        """Stable identity for the Trace and harness runs: ``epic@1``."""
        return f"{self.mood}@{self.version}"


#: The registry. Versioned DATA — grows by ticket, graded by the harness.
#: Scope is deliberately minimal (proof): one standard profile and one
#: polarity-inverted profile.
MOOD_PROFILES: ClassVar[dict[str, MoodProfile]] = {
    "epic": MoodProfile(
        mood="epic",
        version=1,
        query_phrases=[
            "sweeping large-scale saga",
            "mythic stakes on a grand canvas",
            "large-scale battle or historical epic",
        ],
        genre_boosts={
            "Action": 0.8,
            "Adventure": 0.8,
            "War": 0.7,
            "History": 0.6,
            "Fantasy": 0.6,
            "Science Fiction": 0.5,
            "Drama": 0.3,
        },
        floors=MoodFloorCriteria(min_vote_count=3000),
        runtime_boost_min=120,
        popularity_boost=0.2,
        revenue_boost=0.2,
    ),
    "hidden-gem": MoodProfile(
        mood="hidden-gem",
        version=1,
        query_phrases=[
            "overlooked underrated gem",
            "under-seen critically loved film",
        ],
        genre_boosts={"Drama": 0.5, "Documentary": 0.4, "Comedy": 0.3, "Romance": 0.3},
        floors=MoodFloorCriteria(min_vote_count=1500),
        popularity_boost=-0.8,  # the inversion: under-seen wins
        revenue_boost=-0.4,
    ),
}


def resolve_mood_profile(
    mood: str,
    registry: dict[str, MoodProfile] = MOOD_PROFILES,
) -> MoodProfile | None:
    """Canonical mood -> profile, or None when unmapped (fail-open contract).

    Accepts an ALREADY-canonical tag (both stacks store one in
    ``preferred_mood`` / ``decision.mood``); raw-utterance canonicalization
    stays in the probing vocabulary — the domain never imports the Router.
    """
    if not mood:
        return None
    return registry.get(mood.strip().casefold())


def expand_query_text(query: str, profile: MoodProfile) -> str:
    """Append the profile's connotation anchors to the query, deterministically.

    Same input, same output — no LLM in the loop. The anchors dilute the
    single stemmed mood token so a title collision (*Epic*) stops dominating
    the sparse leg, and hand the dense leg the vibe in words.
    """
    if not profile.query_phrases:
        return query
    return f"{query} {', '.join(profile.query_phrases)}"


def merge_profile_floors(
    filters: MetadataFilterCriteria | None,
    profile: MoodProfile,
) -> MetadataFilterCriteria:
    """Fold the profile's floors into a routing decision's filters.

    Tightening only: an explicit user filter always wins — the profile can
    narrow further, never loosen (max of the two minimums).
    """
    base = filters or MetadataFilterCriteria()
    updates: dict[str, Any] = {}
    if profile.floors.min_vote_count is not None:
        current = base.vote_count_min
        updates["vote_count_min"] = (
            profile.floors.min_vote_count if current is None else max(current, profile.floors.min_vote_count)
        )
    if profile.floors.min_vote_average is not None:
        current = base.rating_min
        updates["rating_min"] = (
            profile.floors.min_vote_average if current is None else max(current, profile.floors.min_vote_average)
        )
    if not updates:
        return base
    return base.model_copy(update=updates)


def boost_spec_of(profile: MoodProfile) -> dict[str, Any]:
    """The profile's soft-boost half as a plain dict for ``engine.retrieve``.

    Plain dict, not the model: the engine stays mechanical and the caller
    (``retrieve_node``) injects the Experiment Config ``scale`` alongside.
    """
    return {
        "profile_id": profile.id,
        "genre_boosts": dict(profile.genre_boosts),
        "runtime_boost_min": profile.runtime_boost_min,
        "popularity_boost": profile.popularity_boost,
        "revenue_boost": profile.revenue_boost,
        "scale": 1.0,
    }
