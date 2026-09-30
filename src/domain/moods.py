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
caller records it). Hard floors stay hard: an empty pool after a floor is an
empty pool.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.domain.routing import MetadataFilterCriteria

#: Single source for MoodProfile boost calibration (ADR 0004): the literals
#: live HERE only. ExperimentConfig field defaults, MoodBoostSpec field
#: defaults, and ``mood_boost_spec`` all derive from this mapping; nothing
#: else in the repo may restate the values.
MOOD_BOOST_CALIBRATION: dict[str, float] = {
    "normalizer": 0.01,
    "runtime_weight": 0.5,
    "term_cap": 2.0,
    "popularity_log_divisor": 10.0,
    "revenue_log_divisor": 25.0,
}


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


class MoodBoostSpec(BaseModel):
    """Soft-boost half of a Mood Profile, plus Experiment Config calibration.

    Built by ``mood_boost_spec`` in the retrieve seam; the engine reads this
    typed object mechanically (no Ranking ClassVars, no plain-dict reparse).
    Calibration defaults DERIVE from ``MOOD_BOOST_CALIBRATION`` — the single
    literal source shared with ExperimentConfig (ADR 0004).
    """

    profile_id: str
    genre_boosts: dict[str, float] = Field(default_factory=dict)
    runtime_boost_min: int | None = None
    popularity_boost: float = 0.0
    revenue_boost: float = 0.0
    scale: float = Field(default=1.0, ge=0.0)
    normalizer: float = Field(default=MOOD_BOOST_CALIBRATION["normalizer"], gt=0.0)
    runtime_weight: float = Field(default=MOOD_BOOST_CALIBRATION["runtime_weight"], ge=0.0)
    term_cap: float = Field(default=MOOD_BOOST_CALIBRATION["term_cap"], gt=0.0)
    popularity_log_divisor: float = Field(
        default=MOOD_BOOST_CALIBRATION["popularity_log_divisor"], gt=0.0
    )
    revenue_log_divisor: float = Field(
        default=MOOD_BOOST_CALIBRATION["revenue_log_divisor"], gt=0.0
    )


#: The registry. Versioned DATA — grows by ticket, graded by the harness.
#: Scope is deliberately minimal (proof): one standard profile and one
#: polarity-inverted profile.
MOOD_PROFILES: dict[str, MoodProfile] = {
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


def mood_boost_spec(
    profile: MoodProfile,
    *,
    scale: float = 1.0,
    normalizer: float | None = None,
    runtime_weight: float | None = None,
    term_cap: float | None = None,
    popularity_log_divisor: float | None = None,
    revenue_log_divisor: float | None = None,
) -> MoodBoostSpec:
    """Build the typed soft-boost payload for ``HybridRetrievalEngine.retrieve``.

    Calibration kwargs default to ``MOOD_BOOST_CALIBRATION`` (single source);
    the retrieve seam overrides them from the live Experiment Config (ADR 0004).
    """
    cal = MOOD_BOOST_CALIBRATION
    return MoodBoostSpec(
        profile_id=profile.id,
        genre_boosts=dict(profile.genre_boosts),
        runtime_boost_min=profile.runtime_boost_min,
        popularity_boost=profile.popularity_boost,
        revenue_boost=profile.revenue_boost,
        scale=scale,
        normalizer=cal["normalizer"] if normalizer is None else normalizer,
        runtime_weight=cal["runtime_weight"] if runtime_weight is None else runtime_weight,
        term_cap=cal["term_cap"] if term_cap is None else term_cap,
        popularity_log_divisor=(
            cal["popularity_log_divisor"] if popularity_log_divisor is None else popularity_log_divisor
        ),
        revenue_log_divisor=(
            cal["revenue_log_divisor"] if revenue_log_divisor is None else revenue_log_divisor
        ),
    )
