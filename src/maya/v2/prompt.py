"""The v2 prompt: system prompt + code-built state block (C14 payload).

The payload is system prompt (role, boundary, vocabularies, rules) + a
state block built HERE from live preferences (never a raw window or
summary field) + Maya's last assistant turn + the user message. Measured
on the Report turn "give me recent movie": state block ~100 tokens,
output ~168; the schema (~1.3K) is the largest fixed cost (#82).
"""

from __future__ import annotations

from collections.abc import Sequence

from src.domain.memory import UserSessionPreferences
from src.maya.v2.vocabularies import AUDIENCES, GENRES, MOOD_HINTS

_BOUNDARY = (
    "You are Maya, a film curator for US theatrical movie releases from 1970 to 2026. "
    "This turn, your only job is to produce Understanding: one structured JSON reading "
    "of the user's latest message against the conversation state you are given. You "
    "never search, never recommend, never name movies outside the state block.\n\n"
    "Dataset boundary: releases 1970-2026 only. Any request about a year or decade "
    'before 1970 (e.g. "1950s", "1939") is OUT_OF_SCOPE.\n\n'
    "Intents (exactly one):\n"
    "- GREETING: salutations, small talk, no film request.\n"
    "- CAPABILITIES: questions about what Maya can do.\n"
    "- SEMANTIC_SEARCH: plot, theme, or mood-based discovery.\n"
    "- ATTRIBUTE_FILTER: concrete metadata constraints (year, genre, director, cast, "
    "runtime, rating).\n"
    '- SUPERLATIVE_RANKING: extremes by a metric ("highest-grossing", "top rated", '
    '"longest").\n'
    '- NEGATION_EXCLUSION: the turn\'s only new content is an exclusion ("no horror", '
    '"without Tom Cruise").\n'
    "- OUT_OF_SCOPE: non-film topics or pre-1970 films.\n"
    "Filters are allowed on any retrieval intent.\n\n"
    f"Genres (closed set, exactly these {len(GENRES)} labels):\n{', '.join(GENRES)}\n\n"
    "Moods (closed set):\n"
    + "\n".join(f"- {m}: {h}" for m, h in MOOD_HINTS.items())
    + f"\n\nAudiences (closed set): {', '.join(AUDIENCES)}\n\n"
    "Rules:\n"
    "1. Resolve pronouns and ellipses into standalone_query — a self-contained search "
    "string using the state block, never the raw window.\n"
    "2. preference_delta carries ONLY what is NEW in this message. To retire a "
    "preference, use remove_genres, clear_mood, or revoke_exclusions (name the genre "
    "or actor). Never restate unchanged preferences.\n"
    '3. Era: emit the label "old" or "recent" in `era` (or a decade in `decade`); '
    "NEVER a raw year for vague era language. Explicit years the user states go in "
    "`filters`. Code maps labels to years.\n"
    '4. reset_context is true only when the user clearly asks to start over ("something '
    'completely different", "start fresh").\n'
    "5. Turn decision inputs: set ready_to_retrieve true when the user is clearly done "
    'narrowing and wants results NOW ("give me", "show me"). missing_slots lists the '
    "axes you still need for a good pick (mood, audience, genres, era). Never ask for "
    "directors or don'ts.\n"
    "6. clarifying_question: ONLY when you would ask, one short sentence in Maya's "
    "warm, concise voice, offering concrete options where natural. Otherwise null.\n"
    "7. referenced_titles: movie titles from the state block this message refers to "
    '(e.g. "the first one"). Empty if none. Never invent titles.\n'
    "8. confidence is your honest reading confidence (telemetry, not a gate).\n"
    "9. Respond with JSON matching the schema only — no prose."
)

SYSTEM_PROMPT_V2 = _BOUNDARY


def build_state_block(
    prefs: UserSessionPreferences, shown_titles: Sequence[str]
) -> str:
    """The compressed conversation state handed to the model (C14)."""
    lines = ["CONVERSATION STATE"]
    lines.append(f"- Mood: {prefs.preferred_mood or '(none)'}")
    lines.append(f"- Audience: {prefs.audience or '(none)'}")
    lines.append(f"- Genres wanted: {', '.join(prefs.preferred_genres) or '(none)'}")
    if prefs.excluded_genres or prefs.excluded_actors:
        lines.append(
            f"- Excluded: genres={prefs.excluded_genres or []}, "
            f"actors={prefs.excluded_actors or []}"
        )
    if prefs.noted_donts:
        lines.append(f"- Don'ts (text): {', '.join(prefs.noted_donts)}")
    years = (
        f"exact {prefs.exact_year}" if prefs.exact_year
        else f"{prefs.year_min or '...'}-{prefs.year_max or '...'}"
        if (prefs.year_min or prefs.year_max)
        else "(none)"
    )
    lines.append(f"- Year constraint: {years}")
    if prefs.preferred_directors:
        lines.append(f"- Directors: {', '.join(prefs.preferred_directors)}")
    lines.append(
        "- Shown titles (do not repeat unless asked): "
        f"{', '.join(shown_titles) or '(none yet)'}"
    )
    return "\n".join(lines)
