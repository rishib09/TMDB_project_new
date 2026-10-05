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
from src.maya.v2.models import PreferenceDelta, SubmitUnderstanding, Understanding
from src.maya.v2.notices import build_filter_carryover_notice, injected_genres
from src.maya.v2.project import project_understanding
from src.maya.v2.prompt import SYSTEM_PROMPT_V2, build_state_block
from src.maya.v2.router import MayaV2Router
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
    "SYSTEM_PROMPT_V2",
    "SubmitUnderstanding",
    "TurnDecision",
    "Understanding",
    "MayaV2Router",
    "build_filter_carryover_notice",
    "build_state_block",
    "deterministic_ask",
    "dispose",
    "enforce_probe_budget",
    "enforce_question",
    "injected_genres",
    "known_axes",
    "project_understanding",
    "turn_decision",
]
