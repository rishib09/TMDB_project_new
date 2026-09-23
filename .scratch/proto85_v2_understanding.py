"""PROTOTYPE #85 — THROWAWAY. v2 Understand call vs three documented conversations.

Question: does one structured LLM response per turn (contract #82, C1-C14)
read the Report conversation (#78/#79/#80) and the #56/#26 walkthroughs
correctly, with a minimal code disposer, before any graph wiring?

No orchestrator changes. Deletes with .scratch/; captured on a throwaway
branch per the prototype skill. Run:

    npx @dotenvx/dotenvx run -- ./.venv/Scripts/python.exe .scratch/proto85_v2_understanding.py

Env: OPENROUTER_API_KEY (required), PROTO85_MODEL (default z-ai/glm-5.3-flash,
the D15 baseline candidate), PROTO85_REASONING (default low).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Literal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from src.domain.config import ExperimentConfig
from src.domain.memory import UserSessionPreferences, merge_preferences
from src.domain.routing import IntentType, MetadataFilterCriteria
from src.storage.database import MovieDatabase

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = os.getenv("PROTO85_MODEL", "z-ai/glm-5.3-flash")
REASONING = os.getenv("PROTO85_REASONING", "low")
MAX_PROBE_TURNS = 2  # src/maya/probing.py ProbeStateMachine.MAX_PROBE_TURNS
CLARIFY_MAX_CHARS = 200  # C9: question must be short; template fallback on wording
DATASET_START_YEAR = 1970

CFG = ExperimentConfig()

# --- closed vocabularies (C3, C4, C5) -----------------------------------------

MOOD_HINTS = {
    "edge-of-your-seat": "tense, suspenseful, gripping",
    "thrilling": "exciting, action-packed, adrenaline",
    "funny": "comedy, laugh-out-loud",
    "feel-good": "warm, uplifting, comforting",
    "scary": "horror, frightening, chilling",
    "romantic": "love stories, romance-forward",
    "tearjerker": "emotional, moving, sad",
    "epic": "grand scale, sweeping, monumental",
}
AUDIENCES = ("solo", "date night", "family", "kids", "adults")

GENRES = sorted(MovieDatabase().distinct_genres())
Genre = Literal[tuple(GENRES)]  # type: ignore[valid-type]
Mood = Literal[tuple(sorted(MOOD_HINTS))]  # type: ignore[valid-type]
Audience = Literal[tuple(AUDIENCES)]  # type: ignore[valid-type]
Axis = Literal["mood", "audience", "genres", "era"]


# --- response models (contract #82 C14 sketch, prototype-local) ---------------

class Filters(MetadataFilterCriteria):
    """MetadataFilterCriteria + the two column-shaped caveats (C6)."""

    runtime_max: int | None = None
    rating_min: float | None = None


class PreferenceDelta(BaseModel):
    """C2: delta memory; the merge_preferences reducer merges it."""

    set_mood: Mood | None = None
    clear_mood: bool = False
    set_audience: Audience | None = None
    add_genres: list[Genre] = []
    remove_genres: list[Genre] = []
    add_excluded_genres: list[Genre] = []
    add_excluded_actors: list[str] = []
    revoke_exclusions: list[str] = []
    add_donts: list[str] = []


class Understanding(BaseModel):
    """The single structured reading of one user turn (CONTEXT.md, v2)."""

    intent: IntentType
    standalone_query: str
    filters: Filters | None = None
    era: Literal["old", "recent"] | None = None
    decade: int | None = None
    preference_delta: PreferenceDelta
    reset_context: bool = False
    ready_to_retrieve: bool = False
    missing_slots: list[Axis] = []
    clarifying_question: str | None = None
    referenced_titles: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)


# dynamically-built Literals (runtime vocabularies) need an explicit rebuild
PreferenceDelta.model_rebuild()
Understanding.model_rebuild()


# --- prompt (C14: system prompt v2 + state block + last assistant turn + user) --

SYSTEM_PROMPT = f"""You are Maya, a film curator for US theatrical movie releases from 1970 to 2026. \
This turn, your only job is to produce Understanding: one structured JSON reading of the user's \
latest message against the conversation state you are given. You never search, never recommend, \
never name movies outside the state block.

Dataset boundary: releases 1970-2026 only. Any request about a year or decade before 1970 \
(e.g. "1950s", "1939") is OUT_OF_SCOPE.

Intents (exactly one):
- GREETING: salutations, small talk, no film request.
- CAPABILITIES: questions about what Maya can do.
- SEMANTIC_SEARCH: plot, theme, or mood-based discovery.
- ATTRIBUTE_FILTER: concrete metadata constraints (year, genre, director, cast, runtime, rating).
- SUPERLATIVE_RANKING: extremes by a metric ("highest-grossing", "top rated", "longest").
- NEGATION_EXCLUSION: the turn's only new content is an exclusion ("no horror", "without Tom Cruise").
- OUT_OF_SCOPE: non-film topics or pre-1970 films.
Filters are allowed on any retrieval intent.

Genres (closed set, exactly these {len(GENRES)} labels):
{", ".join(GENRES)}

Moods (closed set):
{chr(10).join(f"- {m}: {h}" for m, h in MOOD_HINTS.items())}

Audiences (closed set): {", ".join(AUDIENCES)}

Rules:
1. Resolve pronouns and ellipses into standalone_query — a self-contained search string using \
the state block, never the raw window.
2. preference_delta carries ONLY what is NEW in this message. To retire a preference, use \
remove_genres, clear_mood, or revoke_exclusions (name the genre or actor). Never restate \
unchanged preferences.
3. Era: emit the label "old" or "recent" in `era` (or a decade in `decade`); NEVER a raw year \
for vague era language. Explicit years the user states go in `filters`. Code maps labels to years.
4. reset_context is true only when the user clearly asks to start over ("something completely \
different", "start fresh").
5. Turn decision inputs: set ready_to_retrieve true when the user is clearly done narrowing and \
wants results NOW ("give me", "show me"). missing_slots lists the axes you still need for a good \
pick (mood, audience, genres, era). Never ask for directors or don'ts.
6. clarifying_question: ONLY when you would ask, one short sentence in Maya's warm, concise voice, \
offering concrete options where natural. Otherwise null.
7. referenced_titles: movie titles from the state block this message refers to (e.g. "the first \
one"). Empty if none. Never invent titles.
8. confidence is your honest reading confidence (telemetry, not a gate).
9. Respond with JSON matching the schema only — no prose."""


def build_state_block(prefs: UserSessionPreferences, shown_titles: list[str]) -> str:
    """C14: code-built state block — the compressed history handed to the model."""
    lines = ["CONVERSATION STATE"]
    mood = prefs.preferred_mood or "(none)"
    aud = prefs.audience or "(none)"
    lines.append(f"- Mood: {mood}")
    lines.append(f"- Audience: {aud}")
    lines.append(f"- Genres wanted: {', '.join(prefs.preferred_genres) or '(none)'}")
    if prefs.excluded_genres or prefs.excluded_actors:
        lines.append(
            f"- Excluded: genres={prefs.excluded_genres or []}, actors={prefs.excluded_actors or []}"
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
    lines.append(f"- Shown titles (do not repeat unless asked): "
                 f"{', '.join(shown_titles) or '(none yet)'}")
    return "\n".join(lines)


# --- the Understand call (D17: retries in the client, C12: one schema retry) ---

def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        temperature=0.0,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_tokens=2048,
        reasoning_effort=REASONING,  # #79: control hidden reasoning
    )


def understand(
    llm: ChatOpenAI,
    user: str,
    prefs: UserSessionPreferences,
    shown_titles: list[str],
    last_assistant: str | None,
    probe_count: int,
) -> tuple[Understanding, dict, list[str]]:
    """One Understand call. Returns (understanding, usage, notes). C12: one retry
    with the validation error, then the deterministic ask."""
    chain = llm.with_structured_output(Understanding, include_raw=True)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("system", build_state_block(prefs, shown_titles)),
    ]
    if last_assistant:
        messages.append(("system", f"MAYA'S LAST REPLY (for reference): {last_assistant}"))
    messages.append(("human", user))

    notes: list[str] = []
    result = chain.invoke(messages)
    usage = dict(result["raw"].response_metadata.get("token_usage", {}))
    parsed = result["parsed"]

    if parsed is None or isinstance(parsed, ValidationError):
        notes.append(f"schema failure on attempt 1: {parsed!r:.200}")
        retry = list(messages)
        retry.insert(2, ("system", f"Your previous JSON failed schema validation: {parsed!r:.400}. "
                                   "Return valid JSON matching the schema exactly."))
        result = chain.invoke(retry)
        usage = dict(result["raw"].response_metadata.get("token_usage", {}))
        parsed = result["parsed"]
        if parsed is None or isinstance(parsed, ValidationError):
            notes.append("schema failure on attempt 2 -> deterministic ask (C12)")
            return (
                Understanding(
                    intent=IntentType.SEMANTIC_SEARCH,
                    standalone_query=user,
                    preference_delta=PreferenceDelta(),
                    ready_to_retrieve=False,
                    missing_slots=["mood", "genres"],
                    clarifying_question="Tell me a mood or a genre and I'll find something good.",
                    confidence=0.0,
                ),
                usage,
                notes,
            )

    u = parsed
    # C9: code checks non-empty and short; template fallback on wording only.
    if u.clarifying_question:
        q = u.clarifying_question.strip()
        if not q or len(q) > CLARIFY_MAX_CHARS:
            notes.append(f"clarifying question rejected by length check ({len(q)} chars); template used")
            u = u.model_copy(
                update={"clarifying_question": "Tell me a mood or a genre and I'll find something good."}
            )
    if probe_count >= MAX_PROBE_TURNS and u.intent in {IntentType.SEMANTIC_SEARCH, IntentType.ATTRIBUTE_FILTER}:
        notes.append(f"probe budget exhausted ({probe_count}/{MAX_PROBE_TURNS}); ready_to_retrieve forced")
        u = u.model_copy(update={"ready_to_retrieve": True})
    return u, usage, notes


# --- disposer (C7, C2, C10): the model proposes, code disposes ----------------

def known_axes(prefs: UserSessionPreferences) -> list[str]:
    """C8 Narrowing Axes with a value: mood, audience, genres, era."""
    axes = []
    if prefs.preferred_mood:
        axes.append("mood")
    if prefs.audience:
        axes.append("audience")
    if prefs.preferred_genres:
        axes.append("genres")
    if prefs.exact_year or prefs.year_min or prefs.year_max:
        axes.append("era")
    return axes


def dispose(
    u: Understanding, prefs: UserSessionPreferences
) -> tuple[Understanding, UserSessionPreferences, list[str]]:
    notes: list[str] = []

    # C7: pre-1970 forces OUT_OF_SCOPE in code.
    def _pre1970(f: Filters | None) -> bool:
        if f is None:
            return False
        return any(
            y is not None and y < DATASET_START_YEAR
            for y in (f.exact_year, f.year_min, f.year_max, u.decade)
        )

    if _pre1970(u.filters) or (u.decade is not None and u.decade < 1970):
        notes.append("disposition: pre-1970 reference -> OUT_OF_SCOPE (C7)")
        u = u.model_copy(
            update={"intent": IntentType.OUT_OF_SCOPE, "filters": None, "era": None, "decade": None}
        )

    # C7: era label -> years through Experiment Config; explicit years win.
    year_fields = dict(exact_year=None, year_min=None, year_max=None)
    filters = u.filters
    has_explicit_years = filters is not None and any(
        getattr(filters, k) is not None for k in ("exact_year", "year_min", "year_max")
    )
    if not has_explicit_years:
        if u.era == "old":
            year_fields["year_max"] = CFG.era_old_year_max
            notes.append(f"disposition: era 'old' -> year_max={CFG.era_old_year_max}")
        elif u.era == "recent":
            year_fields["year_min"] = CFG.era_recent_year_min
            notes.append(f"disposition: era 'recent' -> year_min={CFG.era_recent_year_min}")
        elif u.decade is not None:
            year_fields["year_min"] = u.decade
            year_fields["year_max"] = u.decade + 9
            notes.append(f"disposition: decade {u.decade}s -> {u.decade}-{u.decade + 9}")

    # C2: delta -> UserSessionPreferences update -> merge_preferences.
    d = u.preference_delta
    incoming = UserSessionPreferences(
        preferred_mood=d.set_mood or "",
        audience=d.set_audience or "",
        preferred_genres=list(d.add_genres),
        excluded_genres=list(d.add_excluded_genres),
        excluded_actors=list(d.add_excluded_actors),
        noted_donts=list(d.add_donts),
        exact_year=filters.exact_year if filters else year_fields["exact_year"],
        year_min=(filters.year_min if filters and filters.year_min is not None
                  else year_fields["year_min"]),
        year_max=(filters.year_max if filters and filters.year_max is not None
                  else year_fields["year_max"]),
        reset_requested=u.reset_context,
    )
    merged = merge_preferences(prefs, incoming)

    # Delta removals the reducer does not know (prototype disposing logic).
    if d.remove_genres:
        merged = merged.model_copy(update={
            "preferred_genres": [g for g in merged.preferred_genres if g not in d.remove_genres]
        })
        notes.append(f"disposition: removed genres {list(d.remove_genres)}")
    if d.clear_mood:
        merged = merged.model_copy(update={"preferred_mood": ""})
        notes.append("disposition: mood cleared")
    if d.revoke_exclusions:
        rev = {r.lower() for r in d.revoke_exclusions}
        merged = merged.model_copy(update={
            "excluded_genres": [g for g in merged.excluded_genres if g.lower() not in rev],
            "excluded_actors": [a for a in merged.excluded_actors if a.lower() not in rev],
        })
        notes.append(f"disposition: revoked exclusions {d.revoke_exclusions}")
    if u.reset_context:
        notes.append("disposition: reset_context -> clean slate")
    return u, merged, notes


def turn_decision(u: Understanding, prefs: UserSessionPreferences) -> tuple[str, str]:
    """C8: ask / retrieve / converse / pivot. Returns (decision, why)."""
    if u.intent in {IntentType.GREETING, IntentType.CAPABILITIES}:
        return "converse", "non-retrieval intent"
    if u.intent is IntentType.OUT_OF_SCOPE:
        return "pivot", "out of scope"
    f = u.filters
    explicit = f is not None and any([
        f.exact_year, f.year_min, f.year_max, f.genres, f.director,
        f.cast_member, f.person, f.excluded_genres, f.excluded_actors,
        f.runtime_max, f.rating_min,
    ])
    if explicit:
        return "retrieve", "explicit filters present"
    if u.ready_to_retrieve and len(known_axes(prefs)) >= 1:
        return "retrieve", "ready + at least one axis known"
    if len(known_axes(prefs)) >= CFG.funnel_retrieve_axes:
        return "retrieve", f"axes {known_axes(prefs)} >= {CFG.funnel_retrieve_axes}"
    return "ask", f"missing slots {u.missing_slots or ['mood', 'genres']}"


# --- the three conversations, v1 behavior documented in the issues ------------

CONVERSATIONS = [
    {
        "name": "A - Report conversation (#78/#79/#80, trace 0839ce6c)",
        "initial_prefs": UserSessionPreferences(),
        "turns": [
            ("show me some movies",
             "v1: funnel entry, mood probe asked (no era captured; n/a here)"),
            ("feel good",
             "v1: mood=feel-good set via probe; genre candidates confirm stage"),
            ("date night",
             "v1: audience=date night; retrieved 5 incl. Date Movie 2006, Holidate 2020, Date Night 2010"),
            ("give me recent movie",
             "v1: router failed TWICE (length limit, 979 hidden reasoning tokens), fallback conf 0.1 "
             "no filters; prefs year_min=2015 never applied by retrieve; 3/5 posters repeated from turn 3"),
        ],
    },
    {
        "name": "B - #56 walkthrough ('show me old classic' -> 'just me')",
        "initial_prefs": UserSessionPreferences(),
        "turns": [
            ("show me old classic",
             "v1: era NOT captured on funnel entry (extract_era ran only in funnel_node) -> no year_max"),
            ("feel good",
             "v1: mood probe answered; genre candidates Comedy/Drama/Family/Romance offered"),
            ("all of them",
             "v1: 4-genre INTERSECTION (genre_match=all) -> exactly 1 movie"),
            ("just me",
             "v1: audience=solo; retrieved Hannah Montana: The Movie (2009) - era never applied, "
             "single-genre-deep result"),
        ],
    },
    {
        "name": "C - #26 walkthrough (mid-session: mood=scary, audience=kids)",
        "initial_prefs": UserSessionPreferences(preferred_mood="scary", audience="kids"),
        "turns": [
            ("suggest me horror movies",
             "v1: chip OUT_OF_SCOPE 1.00 + movie cards shown (Scream 1996, Zombie Kids 2013) - incoherent"),
            ("show me horror movies for kids",
             "v1: chip SEMANTIC_SEARCH 1.00, 5 movies retrieved, but pivot text 'outside my reel' shown"),
            ("scary movies for kids",
             "v1: chip OUT_OF_SCOPE 1.00, 0 movies, pivot text"),
            ("horror movies",
             "v1: ATTRIBUTE_FILTER 1.00, 5 grounded cards - correct"),
            ("show me horror movies for kids",
             "v1: ATTRIBUTE_FILTER 1.00, 0 movies, pivot text - incoherent"),
        ],
    },
]


def main() -> None:
    llm = make_llm()
    transcript: list[str] = [
        f"# Prototype #85 — v2 Understand vs v1 on three documented conversations",
        "",
        f"Model: `{MODEL}` (reasoning `{REASONING}`) · temperature 0.0 · knobs: "
        f"era_old_year_max={CFG.era_old_year_max}, era_recent_year_min={CFG.era_recent_year_min}, "
        f"funnel_retrieve_axes={CFG.funnel_retrieve_axes}, MAX_PROBE_TURNS={MAX_PROBE_TURNS}",
        "",
    ]
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    failures = 0

    for conv in CONVERSATIONS:
        transcript += [f"## {conv['name']}", ""]
        prefs = conv["initial_prefs"].model_copy(deep=True)
        last_assistant: str | None = None
        shown_titles: list[str] = []
        probe_count = 0

        for i, (user, v1_note) in enumerate(conv["turns"], 1):
            u, usage, notes = understand(
                llm, user, prefs, shown_titles, last_assistant, probe_count
            )
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)
            u, prefs, dnotes = dispose(u, prefs)
            decision, why = turn_decision(u, prefs)
            if decision == "ask":
                probe_count += 1
                last_assistant = u.clarifying_question
            if decision == "retrieve" and not u.referenced_titles:
                pass  # prototype does not retrieve; shown titles stay as seeded

            turn_fail = bool(
                [n for n in notes if "deterministic ask" in n or "attempt 2" in n]
            )
            failures += turn_fail
            transcript += [
                f"### Turn {i}: `{user}`",
                "",
                f"**v1 did:** {v1_note}",
                "",
                f"```json",
                json.dumps(json.loads(u.model_dump_json()), indent=2),
                "```",
                f"- disposition: {'; '.join(dnotes) or 'no changes'}",
                f"- turn decision: **{decision}** ({why})",
                f"- merged prefs: mood={prefs.preferred_mood or '-'}, audience={prefs.audience or '-'}, "
                f"genres={prefs.preferred_genres or '-'}, years="
                f"{prefs.exact_year or (prefs.year_min, prefs.year_max)}",
                f"- clarifying question: {u.clarifying_question or '-'}",
                f"- usage: in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')}"
                + (f" (reasoning={usage.get('completion_tokens_details', {}).get('reasoning_tokens')})"
                   if usage.get("completion_tokens_details") else ""),
                *(f"- NOTE: {n}" for n in notes),
                "",
            ]
            print(f"[{conv['name']}] turn {i}: {u.intent.value} -> {decision} "
                  f"(conf {u.confidence:.2f})", flush=True)

    transcript += [
        "## Totals",
        "",
        f"- schema-failure fallbacks (C12): {failures}",
        f"- tokens: in={total_usage['prompt_tokens']}, out={total_usage['completion_tokens']}",
    ]
    out = ".scratch/proto85_transcript.md"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(transcript))
    print(f"\nTranscript: {out} ({failures} fallbacks, "
          f"{total_usage['prompt_tokens']} in / {total_usage['completion_tokens']} out)")


if __name__ == "__main__":
    sys.exit(main())
