"""v2 package: the Understanding contract beside the gated v1 stack.

Map #81. Pure domain only (ADR 0006) — the LLM call, prompt and graph
wiring land with the v2 stack ticket (#106).
"""

from src.maya.v2.disposer import (
    Disposition,
    TurnDecision,
    deterministic_ask,
    dispose,
    enforce_probe_budget,
    enforce_question,
    known_axes,
    turn_decision,
)
from src.maya.v2.models import PreferenceDelta, Understanding
from src.maya.v2.vocabularies import (
    AUDIENCES,
    GENRES,
    MOOD_HINTS,
    Audience,
    Axis,
    Genre,
    Mood,
)

__all__ = [
    "AUDIENCES",
    "GENRES",
    "MOOD_HINTS",
    "Audience",
    "Axis",
    "Disposition",
    "Genre",
    "Mood",
    "PreferenceDelta",
    "TurnDecision",
    "Understanding",
    "deterministic_ask",
    "dispose",
    "enforce_probe_budget",
    "enforce_question",
    "known_axes",
    "turn_decision",
]
