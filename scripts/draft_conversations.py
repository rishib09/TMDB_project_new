"""Draft the multi-turn golden source (#86, map #81 G7–G9).

One structured LLM call per coverage item. Recorded conversations (inbox
Reports, walkthrough tickets) are SEEDS: their opening turns are verbatim
constants and the model continues them into a longer conversation. Output: a
draft JSON validated against the row shape of record, plus a markdown table
for the review comment on the ticket.

Run (needs OPENROUTER_API_KEY):
    npx @dotenvx/dotenvx run -- ./.venv/Scripts/python.exe scripts/draft_conversations.py \
        --out data/eval_conversations.draft.json --markdown data/eval_conversations.draft.md
    ... --only C12 C15      # redraft some ids, keep the rest of the existing draft

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
from pydantic import BaseModel, Field  # noqa: E402

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
S = "SEMANTIC_SEARCH"


def _t(n: int, user: str, intent: str, path: str, notes: str = "", **kw) -> ConversationTurn:
    flags = {k: kw.pop(k) for k in ("no_repeat", "referenced_titles_from_shown") if k in kw}
    return ConversationTurn(
        n=n, user=user,
        expect=TurnExpectation(
            intent=IntentType(intent), path=path, constraints=ExpectedConstraints(**kw),
            notes=notes, **flags,
        ),
    )


class CoverageItem(BaseModel):
    id: str
    tier: Tier
    source: str = "authored"
    requirement: str
    min_turns: int
    max_turns: int
    #: Verbatim opening turns (a record). The model continues after them.
    seed: list[ConversationTurn] = Field(default_factory=list)


#: Seeds: real conversations, verbatim, with expectations stating the intended
#: behavior (not what shipped). The model continues each into a long one.
COVERAGE: list[CoverageItem] = [
    CoverageItem(
        id="C01", tier="C_records", source="report#75+authored", min_turns=9, max_turns=10,
        requirement=(
            "Continue the recent date-night conversation: a runtime caveat, an actor "
            "exclusion, a bare 'yes', an ordinal question about a shown title, and a late "
            "era change back to 'something older' that must drop year_min and set year_max."
        ),
        seed=[
            _t(1, "show me some movies", S, "ask"),
            _t(2, "feel good", S, "ask", mood="feel-good"),
            _t(3, "date night", S, "retrieve", mood="feel-good", audience="date night"),
            _t(4, "give me recent movie", S, "retrieve", mood="feel-good", audience="date night",
               year_min=2015, no_repeat=True,
               notes="era_recent_year_min default; five fresh titles, none from turn 3 (#78, #80)"),
        ],
    ),
    CoverageItem(
        id="C02", tier="C_records", source="walkthrough#56+authored", min_turns=9, max_turns=10,
        requirement=(
            "Continue the old-classic conversation: a decade narrowing ('70s only'), a genre "
            "exclusion, two consecutive 'show me others' turns (no_repeat both), then a "
            "contradicting 'actually something recent' that retires the old ceiling."
        ),
        seed=[
            _t(1, "show me old classic", S, "ask", year_max=2000,
               notes="era captured on the entry turn (#56 F1)"),
            _t(2, "feel good", S, "ask", year_max=2000, mood="feel-good"),
            _t(3, "all of them", S, "retrieve", year_max=2000, mood="feel-good",
               notes="answers a genre offer in v1 (union, not intersection, #56 F2); mood+era known"),
            _t(4, "just me", S, "retrieve", year_max=2000, mood="feel-good", audience="solo",
               no_repeat=True),
        ],
    ),
    CoverageItem(
        id="C03", tier="C_records", source="walkthrough#26+authored", min_turns=9, max_turns=10,
        requirement=(
            "Continue the horror-for-kids conversation: a rating floor, a director filter "
            "('anything by Tim Burton'), a question about who directed a shown title, then "
            "'forget the director' which removes only that filter."
        ),
        seed=[
            _t(1, "suggest me horror movies", S, "retrieve", genres=["Horror"],
               notes="a genre word is never OUT_OF_SCOPE (#26-B)"),
            _t(2, "show me horror movies for kids", S, "retrieve", genres=["Horror"], audience="kids",
               no_repeat=True),
            _t(3, "scary movies for kids", S, "retrieve", genres=["Horror"], audience="kids",
               mood="scary", no_repeat=True, notes="a refinement never pivots (#26-C)"),
        ],
    ),
    CoverageItem(
        id="C04", tier="C_records", source="walkthrough#27+authored", min_turns=9, max_turns=10,
        requirement=(
            "Continue after the reset: the post-reset cold start must ask; then a genre and "
            "'for the family' (retrieve), an actor exclusion, a superlative detour "
            "('longest one of those') with empty constraints, and a return to the family set."
        ),
        seed=[
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
    CoverageItem(
        id="C05", tier="C_records", source="walkthrough#42+authored", min_turns=9, max_turns=10,
        requirement=(
            "Continue the maybe-old conversation: the user finally picks a mood, adds a cast "
            "member ('with Al Pacino'), excludes a genre, asks for 'more like the first one', "
            "and ends with a decade that must replace the old ceiling ('80s actually')."
        ),
        seed=[
            _t(1, "show me a movie", S, "ask"),
            _t(2, "help me out with the mood. feel good or sad movie", S, "ask",
               notes="the user asks Maya to choose; no mood is settled yet"),
            _t(3, "alone just me", S, "ask", audience="solo"),
            _t(4, "may be an old movie", S, "retrieve", audience="solo", year_max=2000,
               notes="mid-conversation era must not escape (#42)"),
        ],
    ),
    # --- authored coverage (G7) ---------------------------------------------
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
            "referenced, no_repeat), 'not that one, the other one', then a genre exclusion "
            "that is later revoked ('romance is fine again' puts Romance back in genres)."
        ),
    ),
    CoverageItem(
        id="C13", tier="C_memory", min_turns=10, max_turns=10,
        requirement=(
            "Person filters across ten turns: 'movies with Tom Hanks' (cast_member), later "
            "'directed by Spielberg' (director, both kept), then 'without Tom Hanks' which "
            "moves him from cast_member to excluded_actors, then 'any director is fine' which "
            "drops only the director. Mix in a mood and a runtime caveat that must persist."
        ),
    ),
    CoverageItem(
        id="C14", tier="C_narrowing", min_turns=8, max_turns=8,
        requirement=(
            "Two axes answered in ONE turn ('funny, for the kids') so retrieval is due at "
            "turn 1; then the audience changes twice (kids -> adults -> family), each "
            "replacing the previous; a bare 'nah' that changes nothing (retrieve, no_repeat); "
            "and a final 'ok kids again'."
        ),
    ),
    CoverageItem(
        id="C15", tier="C_refinement", min_turns=10, max_turns=10,
        requirement=(
            "Fresh top-k pressure: after one retrieval, four consecutive 'more', 'others', "
            "'different ones', 'keep going' turns, each retrieve with no_repeat and identical "
            "constraints; then one constraint tweak (a decade); then two more 'more' turns."
        ),
    ),
    CoverageItem(
        id="C16", tier="C_reference", min_turns=10, max_turns=10,
        requirement=(
            "Long-range reference: a retrieval on turn 1, five refinement turns, then 'the "
            "first one you showed at the very start, who directed it' (converse, referenced), "
            "'more like that one' (retrieve, referenced, no_repeat), and 'what year was the "
            "third one' (converse, referenced)."
        ),
    ),
    CoverageItem(
        id="C17", tier="C_memory", min_turns=8, max_turns=8,
        requirement=(
            "Contradicting eras: 'old movies' (year_max=2000), then 'from 2019' (exact_year "
            "replaces the range), then 'no wait, the 90s' (range replaces the exact year), "
            "then 'recent' (year_min=2015 only). A genre and an exclusion stated early must "
            "survive every era change."
        ),
    ),
    CoverageItem(
        id="C18", tier="C_refinement", min_turns=8, max_turns=8,
        requirement=(
            "A pre-1970 request mid-conversation ('1950s westerns') must pivot (OUT_OF_SCOPE) "
            "and leave the constraints intact; 'ok 70s then' sets year_min=1970, "
            "year_max=1979 and keeps the rest; a second off-topic turn ('recommend a book') "
            "pivots again with state intact."
        ),
    ),
    CoverageItem(
        id="C19", tier="C_narrowing", min_turns=8, max_turns=8,
        requirement=(
            "Terse and misspelled turns: 'sumthing funy 4 kids', 'yeh', 'nah not that', "
            "'moar', 'no horor pls', 'ok recent 1s'. Expectations use the canonical values "
            "(mood funny, audience kids, excluded_genres Horror, year_min 2015)."
        ),
    ),
    CoverageItem(
        id="C20", tier="C_refinement", min_turns=10, max_turns=10,
        requirement=(
            "Genre union and intersection: 'sci-fi or fantasy' (genres both, union), later "
            "'only ones that are also thrillers' (Thriller added), 'drop fantasy' (removed), "
            "a rating floor, a reset, and a rebuild from scratch ending with two genres again."
        ),
    ),
]


class ConversationDraft(BaseModel):
    """What the model returns; id, tier and source are assigned by the script."""

    title: str
    turns: list[ConversationTurn]


def _rules(genres: list[str]) -> str:
    moods = ", ".join(MOOD_GENRE_MAP)
    return f"""Rules for the user turns:
- Plain, casual, sometimes terse, as a real visitor types (a typo once or twice is fine).
- Each user turn must stand on its own whatever Maya asked: never quote Maya, never say "the option you listed".
- Ordinal references ("the second one") only where the situation asks for them. Never name a movie Maya did not show.

Rules for each turn's expectation (what must be true AFTER the turn):
- intent: one of {[i.value for i in IntentType]}. Mood, audience, genre words, confirmations, 'more', and
  questions about a shown title are SEMANTIC_SEARCH; a year, decade, runtime, rating, director or cast refinement
  is ATTRIBUTE_FILTER; "no X" / "without X" is NEGATION_EXCLUSION; "highest/most/longest of <year>" is
  SUPERLATIVE_RANKING; non-film turns and pre-1970 requests are OUT_OF_SCOPE; a reset request is CAPABILITIES.
- path: ask (Maya needs one more narrowing axis), retrieve (results are due), converse (a non-search reply:
  answering who directed a shown title, acknowledging a reset), pivot (off-topic or pre-1970 deflection), refuse (never here).
- constraints: the FULL set active after the turn, cumulative from earlier turns, using only these keys:
  mood (one of: {moods}), audience (one of: {", ".join(AUDIENCES)}), genres (from: {", ".join(genres)}),
  exact_year, year_min, year_max, director, cast_member, excluded_genres, excluded_actors,
  runtime_max (minutes), rating_min (0-10).
  A reset empties the set. "old" -> year_max=2000; "recent" -> year_min=2015 (and no year_max); a decade -> its
  ten-year range; an exact year replaces a range and a range replaces an exact year. A superlative turn has empty
  constraints. A pivot turn keeps the previous set unchanged.
- Retrieval is due when explicit filters are present (a genre, a year, a person), or the user asks for results,
  or two narrowing axes (mood, audience, genres, era) are known. A bare cold start with nothing known asks.
- no_repeat=true on a retrieval turn that follows another retrieval turn.
- referenced_titles_from_shown=true when the user points at a shown title.
- notes: one short clause on what the turn tests, or empty."""


def build_brief(item: CoverageItem, genres: list[str]) -> str:
    head = (
        "You write test conversations for Maya, a film curator over US theatrical releases 1970-2026.\n"
    )
    if item.seed:
        seed_json = json.dumps([t.model_dump(mode="json") for t in item.seed], indent=1)
        head += (
            f"The conversation ALREADY has these {len(item.seed)} turns, verbatim from a real session "
            f"(do not repeat or alter them):\n{seed_json}\n\n"
            f"Continue it with turns {len(item.seed) + 1} to between {item.min_turns} and {item.max_turns} "
            f"(total), numbered from {len(item.seed) + 1}, so that it exercises this situation:\n"
        )
    else:
        head += (
            f"Produce ONE conversation of {item.min_turns} to {item.max_turns} user turns that exercises "
            "this situation:\n"
        )
    head += item.requirement + "\n\n"
    return head + _rules(genres) + "\nGive the conversation a five-word title."


def draft_one(llm: ChatOpenAI, item: CoverageItem, genres: list[str]) -> GoldenConversation:
    chain = llm.with_structured_output(ConversationDraft)
    draft: ConversationDraft = chain.invoke(build_brief(item, genres))
    turns = [*item.seed, *draft.turns]
    turns = [t.model_copy(update={"n": i}) for i, t in enumerate(turns, start=1)]
    return GoldenConversation(
        id=item.id, tier=item.tier, title=draft.title, source=item.source, turns=turns
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=Path("data/eval_conversations.draft.json"))
    parser.add_argument("--markdown", type=Path, default=None, help="also write the review table")
    parser.add_argument("--model", default=ExperimentConfig().synthesis_model)
    parser.add_argument("--only", nargs="*", default=None,
                        help="ids to (re)draft; the rest are kept from the existing --out file")
    args = parser.parse_args(argv)

    kept: dict[str, GoldenConversation] = {}
    if args.only and args.out.exists():
        existing = ConversationSet.model_validate_json(args.out.read_text(encoding="utf-8"))
        kept = {c.id: c for c in existing.conversations if c.id not in args.only}

    genres = MovieDatabase().distinct_genres()
    llm = ChatOpenAI(
        model=args.model, base_url=OPENROUTER_BASE_URL,
        api_key=os.getenv("OPENROUTER_API_KEY"), temperature=0.7,
    )
    drafted: dict[str, GoldenConversation] = {}
    for item in COVERAGE:
        if args.only and item.id not in args.only:
            continue
        print(f"drafting {item.id} ({item.tier}) ...", file=sys.stderr)
        for attempt in (1, 2):
            try:
                drafted[item.id] = draft_one(llm, item, genres)
                break
            except Exception as exc:  # noqa: BLE001 — one retry, then surface it
                print(f"  attempt {attempt} failed: {exc}", file=sys.stderr)
                if attempt == 2:
                    return 1

    ordered = [drafted.get(i.id) or kept.get(i.id) for i in COVERAGE]
    conversations = ConversationSet(
        version=date.today().isoformat(),
        description=(
            "Maya multi-turn golden source, DRAFT for review on #86 (map #81 G1-G9): "
            "five recorded conversations continued by the model plus authored coverage of "
            "exclusions, caveats, persons, narrowing, refinement, eras, and references."
        ),
        conversations=[c for c in ordered if c is not None],
    )
    args.out.write_text(conversations.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({len(conversations.conversations)} conversations)", file=sys.stderr)
    if args.markdown:
        args.markdown.write_text(render_markdown(conversations), encoding="utf-8")
        print(f"wrote {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
