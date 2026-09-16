"""Draft the multi-turn golden source (#86, map #81 G7–G9).

One structured LLM call per authored coverage item; the recorded
conversations (inbox Reports, walkthrough tickets) are constants, never
generated. Output: a draft JSON validated against the row shape of record,
plus a markdown table for the review comment on the ticket.

Run (needs OPENROUTER_API_KEY):
    npx @dotenvx/dotenvx run -- ./.venv/Scripts/python.exe scripts/draft_conversations.py \
        --out data/eval_conversations.draft.json --markdown data/eval_conversations.draft.md

One-off tooling: never imported by the app, the harness, or CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_openai import ChatOpenAI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.domain.config import ExperimentConfig  # noqa: E402
from src.domain.routing import IntentType  # noqa: E402
from src.evals.conversations import (  # noqa: E402
    ConversationSet,
    ConversationTurn,
    ExpectedConstraints,
    GoldenConversation,
    Tier,
    TurnExpectation,
    render_markdown,
)
from src.maya.probing import MOOD_GENRE_MAP  # noqa: E402
from src.maya.router import OPENROUTER_BASE_URL  # noqa: E402
from src.storage.database import MovieDatabase  # noqa: E402

AUDIENCES = ["solo", "date night", "family", "kids", "adults"]


class CoverageItem(BaseModel):
    id: str
    tier: Tier
    requirement: str
    min_turns: int
    max_turns: int


#: G7 situations not already covered by a recorded conversation.
COVERAGE: list[CoverageItem] = [
    CoverageItem(
        id="C06", tier="C_memory", min_turns=10, max_turns=10,
        requirement=(
            "An exclusion stated on turn 1 ('no horror' or 'nothing with Tom Cruise') that "
            "must still be honored on turns 8-10 after several refinements (mood, audience, "
            "era, a bare 'yes'). Include one turn that revokes nothing and one that adds a "
            "second exclusion mid-way."
        ),
    ),
    CoverageItem(
        id="C07", tier="C_memory", min_turns=6, max_turns=7,
        requirement=(
            "Column-shaped caveats: 'nothing over two hours' (runtime_max=120) stated early "
            "and 'only well-rated ones' (rating_min=7.0) added later; both must persist to "
            "the last turn. Start with a genre or mood so retrieval happens by turn 2."
        ),
    ),
    CoverageItem(
        id="C08", tier="C_memory", min_turns=6, max_turns=7,
        requirement=(
            "Seed from a real visitor report: turn 1 is exactly 'sci-fi movies from 1999 "
            "about virtual reality' (exact_year=1999, genres=['Science Fiction']). Later "
            "turns refine (audience, a caveat) and the exact year must persist; then "
            "'actually anything from the 90s' replaces it with year_min=1990, year_max=1999."
        ),
    ),
    CoverageItem(
        id="C09", tier="C_narrowing", min_turns=6, max_turns=7,
        requirement=(
            "Cold broad start ('show me something') that must ask; the user answers one "
            "axis per turn; a bare 'go ahead' after one axis must retrieve; then one "
            "off-topic turn ('what's the weather like') that pivots WITHOUT wiping the "
            "constraints; the next film turn shows the constraints intact."
        ),
    ),
    CoverageItem(
        id="C10", tier="C_refinement", min_turns=10, max_turns=10,
        requirement=(
            "Era change and pivots: start with a mood and 'something old' (year_max=2000), "
            "later 'something recent instead' (year_min=2015 and NO year_max), then an "
            "explicit genre pivot ('forget that, action movies') that retires the earlier "
            "mood, then 'something completely different' that resets everything, then a "
            "fresh start that asks again."
        ),
    ),
    CoverageItem(
        id="C11", tier="C_refinement", min_turns=8, max_turns=8,
        requirement=(
            "A superlative mid-conversation: after two retrieval turns with a genre, the "
            "user asks 'highest grossing action movie of 2012' (SUPERLATIVE_RANKING, "
            "constraints empty for that turn), then returns to the earlier constraints "
            "('ok back to the funny ones') which must still be present."
        ),
    ),
    CoverageItem(
        id="C12", tier="C_reference", min_turns=10, max_turns=10,
        requirement=(
            "Ordinal and title references: after a retrieval, 'who directed the second one' "
            "(converse, referenced_titles_from_shown), 'more like the first one' (retrieve, "
            "referenced, no_repeat), 'not that one, the other one', and a title named "
            "back ('something like Holidate but scarier')."
        ),
    ),
]


def _t(n: int, user: str, intent: str, path: str, notes: str = "", **kw) -> ConversationTurn:
    flags = {k: kw.pop(k) for k in ("no_repeat", "referenced_titles_from_shown") if k in kw}
    return ConversationTurn(
        n=n, user=user,
        expect=TurnExpectation(
            intent=IntentType(intent), path=path, constraints=ExpectedConstraints(**kw),
            notes=notes, **flags,
        ),
    )


S = "SEMANTIC_SEARCH"

#: Verbatim records. Expectations state the intended behavior, not what shipped.
RECORDS: list[GoldenConversation] = [
    GoldenConversation(
        id="C01", tier="C_records", title="recent date night (inbox Report)", source="report#75",
        turns=[
            _t(1, "show me some movies", S, "ask"),
            _t(2, "feel good", S, "ask", mood="feel-good"),
            _t(3, "date night", S, "retrieve", mood="feel-good", audience="date night"),
            _t(4, "give me recent movie", S, "retrieve", mood="feel-good", audience="date night",
               year_min=2015, no_repeat=True,
               notes="era_recent_year_min default; five fresh titles, none from turn 3 (#78, #80)"),
        ],
    ),
    GoldenConversation(
        id="C02", tier="C_records", title="old classic, all of them, just me", source="walkthrough#56",
        turns=[
            _t(1, "show me old classic", S, "ask", year_max=2000,
               notes="era captured on the entry turn (#56 F1)"),
            _t(2, "feel good", S, "ask", year_max=2000, mood="feel-good"),
            _t(3, "all of them", S, "retrieve", year_max=2000, mood="feel-good",
               notes="answers a genre offer in v1 (union, not intersection, #56 F2); mood+era known"),
            _t(4, "just me", S, "retrieve", year_max=2000, mood="feel-good", audience="solo",
               no_repeat=True),
        ],
    ),
    GoldenConversation(
        id="C03", tier="C_records", title="horror for kids, five ways", source="walkthrough#26",
        turns=[
            _t(1, "suggest me horror movies", S, "retrieve", genres=["Horror"],
               notes="a genre word is never OUT_OF_SCOPE (#26-B)"),
            _t(2, "show me horror movies for kids", S, "retrieve", genres=["Horror"], audience="kids",
               no_repeat=True),
            _t(3, "scary movies for kids", S, "retrieve", genres=["Horror"], audience="kids",
               mood="scary", no_repeat=True, notes="a refinement never pivots (#26-C)"),
            _t(4, "horror movies", S, "retrieve", genres=["Horror"], audience="kids", mood="scary",
               no_repeat=True, notes="no new constraint; state persists"),
            _t(5, "show me horror movies for kids", S, "retrieve", genres=["Horror"], audience="kids",
               mood="scary", no_repeat=True),
        ],
    ),
    GoldenConversation(
        id="C04", tier="C_records", title="make me cry, then start over", source="walkthrough#27",
        turns=[
            _t(1, "make me cry", S, "ask", mood="tearjerker"),
            _t(2, "lets say drama", S, "retrieve", mood="tearjerker", genres=["Drama"],
               notes="mood + genre known; v1 asks audience here by design"),
            _t(3, "alone just for me", S, "retrieve", mood="tearjerker", genres=["Drama"],
               audience="solo", no_repeat=True, notes="a direct answer never leaves the funnel (#27-P)"),
            _t(4, "sad, drama, years from 2000 - 2026", S, "retrieve", mood="tearjerker",
               genres=["Drama"], audience="solo", year_min=2000, year_max=2026, no_repeat=True,
               notes="stated years become filters (#27-Q)"),
            _t(5, "clear out the previous filter and start a new search", "CAPABILITIES", "converse",
               notes="reset; intent label is cosmetic, the empty constraint set is the check"),
            _t(6, "show me some movies", S, "ask", notes="post-reset cold start asks again"),
        ],
    ),
    GoldenConversation(
        id="C05", tier="C_records", title="help me with the mood, maybe old", source="walkthrough#42",
        turns=[
            _t(1, "show me a movie", S, "ask"),
            _t(2, "help me out with the mood. feel good or sad movie", S, "ask",
               notes="the user asks Maya to choose; no mood is settled yet"),
            _t(3, "alone just me", S, "ask", audience="solo"),
            _t(4, "may be an old movie", S, "retrieve", audience="solo", year_max=2000,
               notes="mid-conversation era must not escape (#42)"),
        ],
    ),
]


class ConversationDraft(BaseModel):
    """What the model returns; id, tier and source are assigned by the script."""

    title: str
    turns: list[ConversationTurn]


def build_brief(item: CoverageItem, genres: list[str]) -> str:
    moods = ", ".join(MOOD_GENRE_MAP)
    return f"""You write test conversations for Maya, a film curator over US theatrical releases 1970-2026.
Produce ONE conversation of {item.min_turns} to {item.max_turns} user turns that exercises this situation:
{item.requirement}

Rules for the user turns:
- Plain, casual, sometimes terse, as a real visitor types (typos allowed once or twice).
- Each user turn must stand on its own whatever Maya asked: never quote Maya, never say "the option you listed".
- Ordinal references ("the second one") are allowed only where the situation asks for them.

Rules for each turn's expectation (what must be true AFTER the turn):
- intent: one of {[i.value for i in IntentType]}. Film requests and refinements are SEMANTIC_SEARCH; "highest/most/best of <year>" is SUPERLATIVE_RANKING; non-film turns are OUT_OF_SCOPE; a reset request is CAPABILITIES.
- path: ask (Maya needs one more narrowing axis), retrieve (results are due), converse (a non-search reply, e.g. answering who directed a shown title or acknowledging a reset), pivot (off-topic deflection), refuse (never use here).
- constraints: the FULL set active after the turn, cumulative from earlier turns, using only these keys:
  mood (one of: {moods}), audience (one of: {", ".join(AUDIENCES)}), genres (from: {", ".join(genres)}),
  exact_year, year_min, year_max, excluded_genres, excluded_actors, runtime_max (minutes), rating_min (0-10).
  A reset empties the set. An era word maps to year_max=2000 for "old" and year_min=2015 for "recent"; a decade to its ten-year range.
- Retrieval is due when explicit filters are present, or the user asks for results, or two narrowing axes (mood, audience, genres, era) are known.
- no_repeat=true on a retrieval turn that follows another retrieval turn.
- referenced_titles_from_shown=true when the user points at a shown title.
- notes: one short clause on what the turn tests, or empty.
Number turns 1..N consecutively. Give the conversation a five-word title."""


def draft_one(llm: ChatOpenAI, item: CoverageItem, genres: list[str]) -> GoldenConversation:
    chain = llm.with_structured_output(ConversationDraft)
    draft: ConversationDraft = chain.invoke(build_brief(item, genres))
    return GoldenConversation(
        id=item.id, tier=item.tier, title=draft.title, source="authored", turns=draft.turns
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=Path("data/eval_conversations.draft.json"))
    parser.add_argument("--markdown", type=Path, default=None, help="also write the review table")
    parser.add_argument("--model", default=ExperimentConfig().synthesis_model)
    parser.add_argument("--only", nargs="*", default=None, help="coverage ids to (re)draft")
    args = parser.parse_args(argv)

    genres = MovieDatabase().distinct_genres()
    llm = ChatOpenAI(
        model=args.model, base_url=OPENROUTER_BASE_URL,
        api_key=os.getenv("OPENROUTER_API_KEY"), temperature=0.7,
    )
    authored: list[GoldenConversation] = []
    for item in COVERAGE:
        if args.only and item.id not in args.only:
            continue
        print(f"drafting {item.id} ({item.tier}) ...", file=sys.stderr)
        for attempt in (1, 2):
            try:
                authored.append(draft_one(llm, item, genres))
                break
            except Exception as exc:  # noqa: BLE001 — one retry, then surface it
                print(f"  attempt {attempt} failed: {exc}", file=sys.stderr)
                if attempt == 2:
                    return 1

    conversations = ConversationSet(
        version=date.today().isoformat(),
        description=(
            "Maya multi-turn golden source, DRAFT for review on #86 (map #81 G1-G9): "
            "five recorded conversations plus LLM-drafted coverage of exclusions, caveats, "
            "narrowing, refinement, and references."
        ),
        conversations=[*RECORDS, *authored],
    )
    args.out.write_text(conversations.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({len(conversations.conversations)} conversations)", file=sys.stderr)
    if args.markdown:
        args.markdown.write_text(render_markdown(conversations), encoding="utf-8")
        print(f"wrote {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
