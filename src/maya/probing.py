"""Shared deterministic vocabulary (v2 stack).

Trimmed to the four symbols the v2 stack consumes (#156 hard split):
``is_fresh_start`` (guard node choke point), ``extract_probe_answers`` +
``MAX_PROBE_TURNS`` (the v2 disposer's fallback vocabulary and ask budget),
and ``preference_chips`` (UI narrowing chips). The v1 funnel state machine
this module used to host died with the v1 stack; the vocabulary itself is
stack-neutral and exact — no fuzzy matching, so a hostile or garbled query
can never invent fields.
"""

import re
from typing import ClassVar

from src.domain.memory import UserSessionPreferences

#: Hard cap on probe/ask turns per session — never an interrogation (#22,
#: consumed by the v2 disposer's probe-budget invariant).
MAX_PROBE_TURNS: ClassVar[int] = 2

#: Phrases that wipe accumulated preferences for a clean start (#26-E/L).
#: Deterministic and matched by substring on the lowered query — checked in
#: the guard node, the single choke point every turn passes.
FRESH_START_PHRASES: ClassVar[tuple[str, ...]] = (
    "something completely different",
    "something different",
    "start fresh",
    "start over",
    "fresh start",
    "fresh search",
    "start with a fresh",
    "watch something else",
    "clear my preferences",
    "clear the filters",
    "clear all filters",
    "remove all the filters",
    "remove all filters",
    "remove the filters",
    "reset filters",
    "reset my preferences",
    "no filters",
)


def is_fresh_start(query: str) -> bool:
    """True when the user asks to drop accumulated preferences (#26-E).

    Deterministic escape hatch offered by the carry-over announcement —
    checked in the guard node, the single choke point every turn passes.
    """
    lowered = query.lower()
    if any(phrase in lowered for phrase in FRESH_START_PHRASES):
        return True
    # #26-L: "remove all the filters"-style verb+object patterns the fixed
    # list can't cover ("clear those filters", "reset my filters", ...).
    return bool(re.search(
        r"\b(remove|clear|reset|drop|wipe)\b[^.!?]{0,24}\bfilters?\b", lowered
    ))


#: Exact-vocabulary extraction for the scalar axes (#22). Deliberately
#: small and literal — the v2 Understand model extracts genres/directors/
#: don'ts via MetadataFilterCriteria; this covers what the schema does not.
_MOOD_VOCAB: ClassVar[dict[str, str]] = {
    "edge of the seat": "edge-of-your-seat",
    "edge of your seat": "edge-of-your-seat",
    "edge-of-your-seat": "edge-of-your-seat",
    "on the edge of my seat": "edge-of-your-seat",
    "funny": "funny",
    "hilarious": "funny",
    "comedy": "funny",
    "feel-good": "feel-good",
    "heartwarming": "feel-good",
    "scary": "scary",
    "spooky": "scary",
    "horror": "scary",  # #26-D: the domain's own genre word maps to the mood
    "creepy": "scary",
    "frightening": "scary",
    "terrifying": "scary",
    "romantic": "romantic",
    "romance": "romantic",
    "thrilling": "thrilling",
    "thriller": "thrilling",
    "intense": "thrilling",
    "gripping": "thrilling",
    "suspense": "thrilling",
    "sad": "tearjerker",
    "cry": "tearjerker",
    "epic": "epic",
    "hidden gem": "hidden-gem",  # #137: profile-backed mood (inverted polarity)
    "underrated": "hidden-gem",
    "underappreciated": "hidden-gem",
    "overlooked": "hidden-gem",
}

_AUDIENCE_VOCAB: ClassVar[dict[str, str]] = {
    "kids": "kids",
    "kid": "kids",
    "children": "kids",
    "family": "family",
    "date night": "date night",
    "date": "date night",
    "adults": "adults",
    "grown-ups": "adults",
    "solo": "solo",
    # #27-P: every phrasing the audience probe itself invites must extract
    "alone": "solo",
    "just me": "solo",
    "just for me": "solo",
    "by myself": "solo",
    "on my own": "solo",
}

#: Window checked before a keyword hit for a negation token (#22).
_NEGATION_PREFIX_RE = re.compile(r"\b(no|not|without|never|nothing)[\s-]+$")


def _is_negated(text: str, keyword_start: int) -> bool:
    prefix = text[max(0, keyword_start - 16):keyword_start]
    return bool(_NEGATION_PREFIX_RE.search(prefix))


def extract_probe_answers(query: str) -> UserSessionPreferences:
    """Exact-vocabulary keyword extraction of mood/audience from free text.

    Word-boundary matches only ("cry" must not fire inside "cryogenic").
    Negated mentions ("no kids", "not funny") are skipped — the naive
    extractor must not record the *opposite* of what the user asked for.
    Returns an incremental UserSessionPreferences (only matched fields set);
    the graph's merge_preferences reducer combines it with session state.
    """
    lowered = query.lower()

    def first_match(vocab: dict[str, str]) -> str:
        for keyword, value in vocab.items():
            match = re.search(rf"\b{re.escape(keyword)}\b", lowered)
            if match and not _is_negated(lowered, match.start()):
                return value
        return ""

    mood = first_match(_MOOD_VOCAB)
    audience = first_match(_AUDIENCE_VOCAB)
    if not mood and not audience:
        return UserSessionPreferences()
    return UserSessionPreferences(preferred_mood=mood, audience=audience)


def preference_chips(prefs: UserSessionPreferences) -> list[str]:
    """Human-readable chips for the active preferences (shared by UI + notice)."""
    chips: list[str] = []
    if prefs.preferred_mood:
        chips.append(f"mood: {prefs.preferred_mood}")
    if prefs.audience:
        chips.append(f"audience: {prefs.audience}")
    chips.extend(f"no {d}" for d in prefs.noted_donts)
    if prefs.preferred_genres:
        chips.append("genres: " + ", ".join(prefs.preferred_genres))
    chips.extend(f"dir. {d}" for d in prefs.preferred_directors)
    # #27-Q: carried year constraints are visible like any other filter.
    if prefs.exact_year:
        chips.append(f"year: {prefs.exact_year}")
    elif prefs.year_min or prefs.year_max:
        lo = prefs.year_min or "…"
        hi = prefs.year_max or "…"
        chips.append(f"years: {lo}-{hi}")
    return chips
