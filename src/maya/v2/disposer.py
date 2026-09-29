"""The v2 disposer — the model proposes, code disposes (ADR 0005, C7–C12).

Every function here is pure and framework-free (ADR 0006): models in,
models out, ``notes`` carrying the human-readable record of every
invariant enforced. The notes become trace fields when the v2 stack wires
this module (#106) — if it matters, it appears in the Trace.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.domain.config import ExperimentConfig
from src.domain.memory import UserSessionPreferences, merge_preferences
from src.domain.routing import IntentType
from src.maya.probing import MAX_PROBE_TURNS
from src.maya.v2.models import Understanding
from src.maya.v2.vocabularies import Axis

#: C9: a clarifying question must be short; code checks the length and
#: falls back to the template on wording only. A semantic guard, not a
#: tunable — it lives in code like MAX_PROBE_TURNS does (surfacing it as an
#: Experiment Config knob is a #106 decision).
CLARIFY_MAX_CHARS = 200

#: C12: the schema-retry budget — initial attempt + ONE retry with the
#: validation error, then the deterministic ask. The #106 understand()
#: wrapper imports this datum; it never invents a literal.
MAX_SCHEMA_ATTEMPTS = 2

#: C8's ask fallback slots, shared by the deterministic ask and the ladder.
DEFAULT_MISSING_SLOTS: list[Axis] = ["mood", "genres"]

#: C12 + C9 shared fallback wording — Maya's deterministic ask.
TEMPLATE_ASK = "Tell me a mood or a genre and I'll find something good."

#: C7: the dataset boundary, enforced in code on EVERY response.
DATASET_START_YEAR = 1970


class Disposition(BaseModel):
    """C13: the record of invariant enforcement for one turn."""

    understanding: Understanding
    preferences: UserSessionPreferences
    notes: list[str] = Field(default_factory=list)


class TurnDecision(BaseModel):
    """C13: ask / retrieve / converse / pivot (the neutral path vocabulary)."""

    decision: Literal["ask", "retrieve", "converse", "pivot"]
    why: str


def known_axes(prefs: UserSessionPreferences) -> list[Axis]:
    """C8: the Narrowing Axes that currently hold a value."""
    axes: list[Axis] = []
    if prefs.preferred_mood:
        axes.append("mood")
    if prefs.audience:
        axes.append("audience")
    if prefs.preferred_genres:
        axes.append("genres")
    if prefs.exact_year or prefs.year_min or prefs.year_max:
        axes.append("era")
    return axes


def deterministic_ask(user: str) -> Understanding:
    """C12: the terminal shape after the schema budget is spent — no
    retrieval, the template question, honest zero confidence (it never
    gates)."""
    return Understanding(
        intent=IntentType.SEMANTIC_SEARCH,
        standalone_query=user,
        ready_to_retrieve=False,
        missing_slots=list(DEFAULT_MISSING_SLOTS),
        clarifying_question=TEMPLATE_ASK,
        confidence=0.0,
    )


def enforce_question(u: Understanding) -> tuple[Understanding, list[str]]:
    """C9: a model-authored question must be non-empty and short; the
    template fallback covers wording only."""
    notes: list[str] = []
    if u.clarifying_question is None:
        return u, notes
    q = u.clarifying_question.strip()
    if not q or len(q) > CLARIFY_MAX_CHARS:
        notes.append(
            f"clarifying question rejected by length check ({len(q)} chars); template used"
        )
        u = u.model_copy(update={"clarifying_question": TEMPLATE_ASK})
    return u, notes


def enforce_probe_budget(
    u: Understanding, probe_count: int
) -> tuple[Understanding, list[str]]:
    """C8: at most MAX_PROBE_TURNS asks per session, then retrieve with what
    exists — for every intent that can reach an ask (converse and pivot
    resolve earlier, so no intent check belongs here). Same bound as the
    v1 funnel — one source of truth."""
    notes: list[str] = []
    if probe_count >= MAX_PROBE_TURNS:
        notes.append(
            f"probe budget exhausted ({probe_count}/{MAX_PROBE_TURNS}); "
            "ready_to_retrieve forced"
        )
        u = u.model_copy(update={"ready_to_retrieve": True})
    return u, notes


def dispose(
    u: Understanding,
    prefs: UserSessionPreferences,
    config: ExperimentConfig,
) -> Disposition:
    """Enforce every invariant the model must not be trusted with (C7 + C2).

    The live ExperimentConfig is injected by the caller (ADR 0004 — the
    session's knobs rule, never a fresh default). Order matters: the
    out-of-scope guard rewrites the understanding BEFORE the preference
    merge, so an out-of-scope turn never poisons memory; the era mapping
    fills only year slots the user did not set explicitly.
    """
    notes: list[str] = []

    # C7: any pre-1970 reference forces OUT_OF_SCOPE in code.
    f = u.filters
    pre1970 = any(
        y is not None and y < DATASET_START_YEAR
        for y in (
            f.exact_year if f else None,
            f.year_min if f else None,
            f.year_max if f else None,
            u.decade,
        )
    )
    if pre1970:
        notes.append(f"disposition: pre-{DATASET_START_YEAR} reference -> OUT_OF_SCOPE (C7)")
        u = u.model_copy(update={
            "intent": IntentType.OUT_OF_SCOPE, "filters": None, "era": None, "decade": None,
        })

    # ONE out-of-scope invariant, whoever ruled the turn out: the model's
    # OUT_OF_SCOPE label and the code-forced pre-1970 rewrite behave
    # identically — no memory writes (matches the v1 pivot path, which
    # applies no state changes). Adversarial-pinned: the delta must not
    # survive either path.
    if u.intent is IntentType.OUT_OF_SCOPE:
        notes.append("disposition: out of scope -> no memory writes")
        return Disposition(understanding=u, preferences=prefs, notes=notes)

    # C7: era label / decade -> years through Experiment Config; explicit
    # years in filters always win.
    era_year_min: int | None = None
    era_year_max: int | None = None
    has_explicit_years = (
        f is not None
        and (f.exact_year is not None or f.year_min is not None or f.year_max is not None)
    )
    supersedes = " (era label supersedes decade)" if u.decade is not None else ""
    if not has_explicit_years:
        if u.era == "old":
            era_year_max = config.era_old_year_max
            notes.append(f"disposition: era 'old' -> year_max={config.era_old_year_max}{supersedes}")
        elif u.era == "recent":
            era_year_min = config.era_recent_year_min
            notes.append(f"disposition: era 'recent' -> year_min={config.era_recent_year_min}{supersedes}")
        elif u.decade is not None:
            era_year_min, era_year_max = u.decade, u.decade + 9
            notes.append(f"disposition: decade {u.decade}s -> {u.decade}-{u.decade + 9}")

    # C2: delta -> UserSessionPreferences update -> the existing reducer.
    d = u.preference_delta
    # #121a: an ATTRIBUTE_FILTER director scope becomes standing session
    # scope (the C06 snapshot discipline, applied to directors). Record
    # questions never do: "who directed X" (SEMANTIC_SEARCH) scopes this
    # turn's retrieval only.
    director_scope = (
        f.director
        if u.intent is IntentType.ATTRIBUTE_FILTER and f is not None and f.director
        else None
    )
    incoming = UserSessionPreferences(
        preferred_mood=d.set_mood or "",
        audience=d.set_audience or "",
        preferred_genres=list(d.add_genres),
        excluded_genres=list(d.add_excluded_genres),
        excluded_actors=list(d.add_excluded_actors),
        noted_donts=list(d.add_donts),
        preferred_directors=[director_scope] if director_scope else [],
        exact_year=f.exact_year if f else None,
        year_min=f.year_min if f and f.year_min is not None else era_year_min,
        year_max=f.year_max if f and f.year_max is not None else era_year_max,
        reset_requested=u.reset_context,
    )
    if director_scope:
        notes.append(f"disposition: director scope persists: {director_scope}")
    merged = merge_preferences(prefs, incoming)

    # Delta removals the reducer cannot express (prototype #85, proven).
    if d.remove_genres:
        merged = merged.model_copy(update={
            "preferred_genres": [g for g in merged.preferred_genres if g not in d.remove_genres],
        })
        notes.append(f"disposition: removed genres {list(d.remove_genres)}")
    if d.clear_mood:
        merged = merged.model_copy(update={"preferred_mood": ""})
        notes.append("disposition: mood cleared")
    if d.revoke_exclusions:
        revoked = {r.lower() for r in d.revoke_exclusions}
        merged = merged.model_copy(update={
            "excluded_genres": [g for g in merged.excluded_genres if g.lower() not in revoked],
            "excluded_actors": [a for a in merged.excluded_actors if a.lower() not in revoked],
        })
        notes.append(f"disposition: revoked exclusions {d.revoke_exclusions}")
    if u.reset_context:
        notes.append("disposition: reset_context -> clean slate")

    return Disposition(understanding=u, preferences=merged, notes=notes)


def turn_decision(
    u: Understanding,
    prefs: UserSessionPreferences,
    config: ExperimentConfig,
    probe_count: int = 0,
) -> TurnDecision:
    """C8: ask / retrieve / converse / pivot — deterministic, in order.
    The live config is injected (ADR 0004); ``probe_count`` rides in so the
    cap is enforceable HERE, not only via the ready flag: at
    MAX_PROBE_TURNS the ladder retrieves with what exists, axes or not —
    the conversation can never dead-end in an ask loop."""
    if u.intent in {IntentType.GREETING, IntentType.CAPABILITIES}:
        return TurnDecision(decision="converse", why="non-retrieval intent")
    if u.intent is IntentType.OUT_OF_SCOPE:
        return TurnDecision(decision="pivot", why="out of scope")
    if probe_count >= MAX_PROBE_TURNS:
        return TurnDecision(
            decision="retrieve",
            why=(f"probe budget exhausted ({probe_count}/{MAX_PROBE_TURNS}); "
                 "retrieving with what exists"),
        )
    f = u.filters
    if f is not None and any([
        f.exact_year, f.year_min, f.year_max, f.genres, f.director,
        f.cast_member, f.person, f.excluded_genres, f.excluded_actors,
        f.runtime_max, f.rating_min,
    ]):
        return TurnDecision(decision="retrieve", why="explicit filters present")
    axes = known_axes(prefs)
    if u.ready_to_retrieve and len(axes) >= 1:
        return TurnDecision(decision="retrieve", why="ready + at least one axis known")
    if len(axes) >= config.funnel_retrieve_axes:
        return TurnDecision(
            decision="retrieve", why=f"axes {axes} >= {config.funnel_retrieve_axes}"
        )
    return TurnDecision(
        decision="ask",
        why=f"missing slots {u.missing_slots or list(DEFAULT_MISSING_SLOTS)}",
    )
