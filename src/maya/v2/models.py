"""The v2 response models — one structured reading of a user turn (#82).

``Understanding`` replaces the v1 routing decision in v2: the model
proposes the whole reading of the turn, code disposes (ADR 0005). The
disposer lives in ``disposer.py``; these models are pure data.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.domain.routing import IntentType, MetadataFilterCriteria
from src.maya.v2.vocabularies import Audience, Axis, Genre, Mood

#: C7: era is a LABEL only — code maps it to years through the ``era_*``
#: config knobs. The model never emits a raw year for vague era language;
#: explicit years the user states go in ``filters``.
Era = Literal["old", "recent"]


class PreferenceDelta(BaseModel):
    """C2: delta memory — only what is NEW in this message.

    The existing ``merge_preferences`` reducer merges it (reset_requested
    rides ``Understanding.reset_context``). Removals the reducer cannot
    express (``remove_genres``, ``clear_mood``, ``revoke_exclusions``) are
    disposed by the disposer after the merge.
    """

    set_mood: Mood | None = None
    clear_mood: bool = False
    set_audience: Audience | None = None
    add_genres: list[Genre] = Field(default_factory=list)
    remove_genres: list[Genre] = Field(default_factory=list)
    add_excluded_genres: list[Genre] = Field(default_factory=list)
    add_excluded_actors: list[str] = Field(default_factory=list)
    revoke_exclusions: list[str] = Field(default_factory=list)
    add_donts: list[str] = Field(default_factory=list)


class Understanding(BaseModel):
    """The single structured response of the v2 Understand call (C1–C14)."""

    intent: IntentType  # C1: the seven intents stay; filters allowed on any
    standalone_query: str  # coreference-resolved, self-contained (rule 1)
    filters: MetadataFilterCriteria | None = None  # C6: + runtime_max/rating_min
    era: Era | None = None
    decade: int | None = None
    preference_delta: PreferenceDelta = Field(default_factory=PreferenceDelta)
    reset_context: bool = False  # replaces the FRESH_START_PHRASES gate
    ready_to_retrieve: bool = False
    missing_slots: list[Axis] = Field(default_factory=list)
    clarifying_question: str | None = None  # C9: asked only if code agrees
    referenced_titles: list[str] = Field(default_factory=list)  # C10
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)  # C11: telemetry only
