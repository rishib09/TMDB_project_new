"""Deterministic probing policy for guided narrowing (issue #22).

Model proposes, code disposes (ADR 0005): whether Maya probes is decided
here in code, never by prompt vibes. Probe turns are fully deterministic —
zero LLM cost — and bounded by MAX_PROBE_TURNS so the conversation always
moves forward. Answer extraction is exact-vocabulary keyword matching:
no fuzzy matching, so a hostile or garbled query can never invent fields.
"""

import re
from typing import ClassVar

from pydantic import BaseModel

from src.domain.memory import UserSessionPreferences
from src.domain.routing import QueryRoutingDecision

#: Queries longer than this are treated as specific enough to answer directly.
BROAD_QUERY_WORD_LIMIT: ClassVar[int] = 5
#: Hard cap on probe turns per session — never an interrogation.
MAX_PROBE_TURNS: ClassVar[int] = 2
#: Probing only makes sense while at least this many axes are unanswered.
MIN_UNANSWERED_AXES: ClassVar[int] = 2

#: Echo sanitization — identical contract to the #21 empty-retrieval response.
_SMUGGLED_MARKUP_RE = re.compile(r"</?\s*\w+\s*/?>|```.*?```", re.DOTALL)
_ECHO_CAP: ClassVar[int] = 120


class ProbeQuestion(BaseModel):
    """One narrowing question, phrased in Maya's voice."""

    axis: str  # mood | audience | donts | genres | directors
    question: str


PROBE_FUNNEL: ClassVar[list[ProbeQuestion]] = [
    ProbeQuestion(
        axis="mood",
        question="First things first — what mood are we in? Feel-good, "
        "edge-of-your-seat, laugh-out-loud, something that'll make you cry?",
    ),
    ProbeQuestion(
        axis="audience",
        question="Who's watching — just you, a date night, or the whole "
        "family with kids in tow?",
    ),
    ProbeQuestion(
        axis="donts",
        question="Any hard passes? Tell me what to keep off the shelf — "
        "horror, heavy drama, anything with clowns...",
    ),
    ProbeQuestion(
        axis="genres",
        question="Any genres you're craving — or curious to try?",
    ),
    ProbeQuestion(
        axis="directors",
        question="Got a favorite director? I'll happily dig through their whole shelf.",
    ),
]

#: Exact-vocabulary extraction for the scalar probe axes (#22). Deliberately
#: small and literal — the router already extracts genres/directors/don'ts
#: via MetadataFilterCriteria; this covers what the router schema does not.
_MOOD_VOCAB: ClassVar[dict[str, str]] = {
    "funny": "funny",
    "hilarious": "funny",
    "comedy": "funny",
    "feel-good": "feel-good",
    "heartwarming": "feel-good",
    "scary": "scary",
    "spooky": "scary",
    "romantic": "romantic",
    "romance": "romantic",
    "thrilling": "thrilling",
    "intense": "thrilling",
    "sad": "tearjerker",
    "cry": "tearjerker",
    "epic": "epic",
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
}


def should_probe(
    decision: QueryRoutingDecision,
    prefs: UserSessionPreferences,
    probe_count: int,
) -> bool:
    """True only for broad, filterless, non-superlative RAG requests.

    Deterministic guards, in order:
    - superlative or filtered queries are specific — answer directly
    - the probe cap is absolute (never an interrogation)
    - probing stops once enough narrowing signal exists
    - long queries carry their own signal — don't stall them
    """
    if not decision.requires_rag:
        return False
    if decision.is_superlative:
        return False
    filters = decision.filters
    if filters and (
        filters.genres
        or filters.director
        or filters.excluded_genres
        or filters.exact_year
        or filters.year_min
        or filters.year_max
    ):
        return False
    if probe_count >= MAX_PROBE_TURNS:
        return False
    if len(prefs.answered_axes()) >= 2:
        return False
    return len((decision.standalone_query or "").split()) <= BROAD_QUERY_WORD_LIMIT


def next_probe_question(prefs: UserSessionPreferences) -> ProbeQuestion | None:
    """First unanswered funnel question, or None when the funnel is exhausted."""
    answered = set(prefs.answered_axes())
    return next((q for q in PROBE_FUNNEL if q.axis not in answered), None)


def build_probe_response(prefs: UserSessionPreferences, query: str = "") -> str:
    """Deterministic Maya-voiced probe turn — grounded, inject-safe, no titles."""
    question = next_probe_question(prefs)
    if question is None:  # caller should have checked should_probe; stay safe
        return "I've got enough to work with — what are you in the mood for?"
    echo = _SMUGGLED_MARKUP_RE.sub(" ", query)
    echo = re.sub(r"\s{2,}", " ", echo).strip()
    if len(echo) > _ECHO_CAP:
        echo = echo[:_ECHO_CAP].rstrip() + "…"
    opener = f'Ooh, "{echo}" — I can work with that! ' if echo else ""
    trail_items = [
        f"a {prefs.preferred_mood} mood" if prefs.preferred_mood else "",
        f"for {prefs.audience}" if prefs.audience else "",
        *(f"no {d}" for d in prefs.noted_donts),
        *(prefs.preferred_genres or []),
        *(f"{d}'s films" for d in prefs.preferred_directors),
    ]
    trail = (
        "So far I've noted: " + ", ".join(t for t in trail_items if t) + ". "
        if any(trail_items)
        else ""
    )
    return f"{opener}But before I start pulling films, let me narrow it down. {trail}{question.question}"


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


#: Window checked before a keyword hit for a negation token (#22).
_NEGATION_PREFIX_RE = re.compile(r"\b(no|not|without|never|nothing)[\s-]+$")


def _is_negated(text: str, keyword_start: int) -> bool:
    prefix = text[max(0, keyword_start - 16):keyword_start]
    return bool(_NEGATION_PREFIX_RE.search(prefix))
