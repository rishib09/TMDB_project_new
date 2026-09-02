"""Deterministic probing policy for guided narrowing (issue #22).

Model proposes, code disposes (ADR 0005): whether Maya probes is decided
here in code, never by prompt vibes. Probe turns are fully deterministic —
zero LLM cost — and bounded by MAX_PROBE_TURNS so the conversation always
moves forward. Answer extraction is exact-vocabulary keyword matching:
no fuzzy matching, so a hostile or garbled query can never invent fields.
"""

import re
from typing import ClassVar

from pydantic import BaseModel, Field

from src.domain.memory import UserSessionPreferences, merge_preferences
from src.domain.routing import QueryRoutingDecision

#: Queries longer than this are treated as specific enough to answer directly.
BROAD_QUERY_WORD_LIMIT: ClassVar[int] = 5
#: Hard cap on probe turns per session — never an interrogation.
MAX_PROBE_TURNS: ClassVar[int] = 2
#: Probing only makes sense while at least this many axes are unanswered.
MIN_UNANSWERED_AXES: ClassVar[int] = 2
#: Once this many axes are answered, confirm before retrieving (#23).
CONFIRM_THRESHOLD: ClassVar[int] = 2
#: Phrases that end the funnel and trigger retrieval immediately (#23).
RETRIEVE_CONFIRMATIONS: ClassVar[tuple[str, ...]] = (
    "go ahead",
    "show me",
    "pull the films",
    "pull them up",
    "just show",
    "no more questions",
    "that's all",
    "thats all",
    "good enough",
)

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


# --- funnel state machine (#23): the turn after a probe is OURS -----------
#
# Walkthrough defect (#23): probe answers like "edge of the seat" confused
# the router (GREETING) and topical follow-ups pivoted OUT_OF_SCOPE. Fix:
# when a probe was just asked, the funnel handles the reply deterministically
# and the router only sees queries the funnel can't own.


class FunnelOutcome(BaseModel):
    """What the funnel decides to do with a post-probe user message."""

    action: str  # probe | confirm | confirm_genres | retrieve | fallthrough
    response: str | None = None  # deterministic response for probe/confirm
    prefs_update: UserSessionPreferences | None = None  # MERGED prefs (idempotent under the reducer)
    offered_genre_options: list[str] = Field(default_factory=list)  # #25 confirm_genres


def _is_confirmation(query: str) -> bool:
    lowered = query.lower()
    return any(phrase in lowered for phrase in RETRIEVE_CONFIRMATIONS)


def build_confirm_response(prefs: UserSessionPreferences) -> str:
    """Deterministic confirm-before-retrieve turn (#23)."""
    trail_items = [
        f"a {prefs.preferred_mood} mood" if prefs.preferred_mood else "",
        f"for {prefs.audience}" if prefs.audience else "",
        *(f"no {d}" for d in prefs.noted_donts),
        *(prefs.preferred_genres or []),
        *(f"{d}'s films" for d in prefs.preferred_directors),
    ]
    trail = ", ".join(t for t in trail_items if t)
    return (
        f"Got it — {trail}. Want to add anything else — a year, a director, "
        "a genre? Or shall I pull the films now?"
    )


def build_funnel_query(prefs: UserSessionPreferences) -> str:
    """Natural-language retrieval query synthesized from funnel answers.

    Embeddings handle this fluently; genres/directors keep flowing through
    the router's SQL filters when the user states them explicitly.
    """
    parts = []
    if prefs.preferred_mood:
        parts.append(prefs.preferred_mood)
    parts.append("movies")
    if prefs.preferred_genres:
        parts.append(" ".join(prefs.preferred_genres))
    if prefs.audience:
        parts.append(f"for {prefs.audience}")
    if prefs.preferred_directors:
        parts.append(f"directed by {' and '.join(prefs.preferred_directors)}")
    return " ".join(parts).strip() or "good movies"


def funnel_axes(prefs: UserSessionPreferences) -> list[str]:
    """Axes the USER directly expressed, for the confirm threshold (#25).

    A mood and its mapped genre are ONE signal, not two: "something funny"
    yielding Comedy must not jump straight to confirmation. Genres count as
    their own axis only when confirmed OUTSIDE the mood map (user picked
    them from a candidate list for an unmapped/absent mood).
    """
    axes = [a for a in prefs.answered_axes() if a != "genres"]
    mood_covered = (
        prefs.preferred_mood
        and prefs.preferred_mood.casefold() in MOOD_GENRE_MAP
    )
    if prefs.preferred_genres and prefs.genre_confirmation_done and not mood_covered:
        axes.append("genres")
    return axes


def handle_probe_answer(
    query: str, prefs: UserSessionPreferences, probe_count: int,
    prefs_update: UserSessionPreferences | None = None,
) -> FunnelOutcome:
    """Funnel decision for the message following a probe (#23).

    Order matters: confirmations beat extraction, extraction beats probing,
    and anything the funnel can't own falls through to normal routing.
    ``prefs_update`` carries the extractor's findings (LLM per #24, with the
    deterministic vocab as fallback) — merging and stage progression are
    pure functions of that input.
    """
    if _is_confirmation(query):
        return FunnelOutcome(action="retrieve", prefs_update=prefs_update)
    if prefs_update is None:
        prefs_update = extract_probe_answers(query)
    if prefs_update.answered_axes():
        merged = merge_preferences(prefs, prefs_update)
        return next_funnel_step(merged, probe_count, query)
    return FunnelOutcome(action="fallthrough")


def next_funnel_step(
    prefs: UserSessionPreferences, probe_count: int, query: str = ""
) -> FunnelOutcome:
    """Pure funnel progression from the CURRENT merged preferences (#25).

    Stage order: genre confirmation (mood just learned) → enough-axes
    confirm → next probe → retrieval. Single-candidate mood maps auto-accept
    their genre without wasting a turn ("funny" IS comedy, no need to ask).
    """
    merged = prefs
    pending = UserSessionPreferences()
    if prefs.preferred_mood and not prefs.genre_confirmation_done:
        candidates = MOOD_GENRE_MAP.get(prefs.preferred_mood.casefold(), [])
        have = {g.casefold() for g in prefs.preferred_genres}
        remaining = [g for g in candidates if g.casefold() not in have]
        if len(remaining) == 1:
            pending = UserSessionPreferences(
                preferred_genres=remaining, genre_confirmation_done=True
            )
            merged = merge_preferences(prefs, pending)
        elif remaining:
            return FunnelOutcome(
                action="confirm_genres",
                response=build_genre_confirm_response(prefs, remaining),
                prefs_update=merged,
                offered_genre_options=remaining,
            )
        else:
            pending = UserSessionPreferences(genre_confirmation_done=True)
            merged = merge_preferences(prefs, pending)

    if len(funnel_axes(merged)) >= CONFIRM_THRESHOLD:
        return FunnelOutcome(
            action="confirm", response=build_confirm_response(merged),
            prefs_update=merged,
        )
    question = next_probe_question(merged)
    if question and probe_count < MAX_PROBE_TURNS:
        return FunnelOutcome(
            action="probe", response=build_probe_response(merged, query),
            prefs_update=merged,
        )
    return FunnelOutcome(action="retrieve", prefs_update=merged)


# --- mood → genre mapping (#25): close the open-vocabulary loop -------------

#: Mood values (from the vocab or LLM extraction) → candidate genres for the
#: confirmation turn. Curated DATA, not prompts — the LLM proposes the mood,
#: this map proposes the genres, the USER confirms. Unmapped moods skip the
#: stage (flavor-only) so the loop can never dead-end.
MOOD_GENRE_MAP: ClassVar[dict[str, list[str]]] = {
    "edge-of-your-seat": ["Thriller", "Sci-Fi", "Horror", "Drama"],
    "thrilling": ["Thriller", "Action", "Crime"],
    "funny": ["Comedy"],
    "feel-good": ["Comedy", "Drama", "Family", "Romance"],
    "scary": ["Horror", "Thriller"],
    "romantic": ["Romance", "Drama"],
    "tearjerker": ["Drama", "Romance"],
    "epic": ["Action", "Adventure", "Fantasy", "History"],
}

#: Phrases accepting the ENTIRE offered candidate set (#25).
_CONFIRM_ALL_RE = re.compile(
    r"\b(all of them|all|everything|any of them|both|either)\b"
)


def build_genre_confirm_response(
    prefs: UserSessionPreferences, candidates: list[str]
) -> str:
    """Deterministic genre-confirmation turn (#25).

    Framing adapts to provenance: with an explicit genre already confirmed
    the question narrows WITHIN it ("within sci-fi…"); otherwise it's a
    plain candidate list.
    """
    mood = prefs.preferred_mood
    have = {g.casefold() for g in prefs.preferred_genres}
    options = ", ".join(candidates)
    if have:
        established = ", ".join(g for g in prefs.preferred_genres if g.casefold() in have)
        return (
            f'Good taste — "{mood}" runs right through {established}. Within '
            f"{established}, do you also want the {options} side? Pick any "
            '(or say "all of them").'
        )
    return (
        f'"{mood}" can mean a few things on my shelves: {options}. '
        'Which of those are you in the mood for? (or say "all of them")'
    )


def match_genre_pick(query: str, options: list[str]) -> list[str] | None:
    """Match a user reply against the offered genre candidates (#25).

    Deterministic against the KNOWN candidate list — no LLM needed for a
    multiple-choice question. Returns the picks, or None when the reply
    isn't a genre answer (caller falls through to normal handling).
    Negated mentions ("no horror") never count as picks.
    """
    lowered = re.sub(r"\bsci fi\b", "sci-fi", query.lower())
    if _CONFIRM_ALL_RE.search(lowered):
        return list(options)
    picks = []
    for option in options:
        match = re.search(rf"\b{re.escape(option.lower())}\b", lowered)
        if match and not _is_negated(lowered, match.start()):
            picks.append(option)
    return picks or None
