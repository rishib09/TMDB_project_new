"""Multi-turn golden source: the row shape of record and its loader (#86, map #81).

One conversation is an ordered list of scripted user turns. Each turn states
what must be true after it: the Intent, the neutral path, and the FULL set
of constraints that must reach the engine on that turn (G2: cumulative,
never a delta, so a Routing Stack with no delta can be scored the same way).

Pure Pydantic + stdlib. The harness mode that runs these rows is #87.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.domain.routing import IntentType

#: Neutral path vocabulary (G3). The harness maps v1 probe/confirm to "ask".
Path_ = Literal["ask", "retrieve", "converse", "pivot", "refuse"]
Tier = Literal["C_records", "C_memory", "C_narrowing", "C_refinement", "C_reference"]

DEFAULT_CONVERSATIONS = Path("data/eval_conversations.json")
MIN_TURNS = 4
MAX_TURNS = 10


class ExpectedConstraints(BaseModel):
    """What must reach the engine on a turn. Keys are the engine's, nothing else."""

    model_config = ConfigDict(extra="forbid")

    mood: str | None = None
    audience: str | None = None
    genres: list[str] = Field(default_factory=list)
    exact_year: int | None = None
    year_min: int | None = None
    year_max: int | None = None
    director: str | None = None
    cast_member: str | None = None
    excluded_genres: list[str] = Field(default_factory=list)
    excluded_actors: list[str] = Field(default_factory=list)
    runtime_max: int | None = None
    rating_min: float | None = None

    def summary(self) -> str:
        """Compact `key=value` line for the review table; 'none' when empty."""
        parts = []
        for key, value in self.model_dump().items():
            if value in (None, [], ""):
                continue
            parts.append(f"{key}={','.join(value) if isinstance(value, list) else value}")
        return ", ".join(parts) or "none"


class TurnExpectation(BaseModel):
    intent: IntentType
    path: Path_
    constraints: ExpectedConstraints = Field(default_factory=ExpectedConstraints)
    #: G6: consecutive retrieval turns must not repeat ids.
    no_repeat: bool = False
    #: G4: the user points at a shown title; scored against runtime state.
    referenced_titles_from_shown: bool = False
    #: G5: optional, only on semantic turns with an obvious answer.
    relevant_movie_ids: list[int] = Field(default_factory=list)
    notes: str = ""


class ConversationTurn(BaseModel):
    n: int
    user: str
    expect: TurnExpectation


class GoldenConversation(BaseModel):
    id: str
    tier: Tier
    title: str
    source: str  # "report#75" | "walkthrough#56" | ... | "authored"
    turns: list[ConversationTurn]

    @field_validator("turns")
    @classmethod
    def _turns_consecutive_within_band(cls, turns: list[ConversationTurn]) -> list[ConversationTurn]:
        if not MIN_TURNS <= len(turns) <= MAX_TURNS:
            raise ValueError(f"{len(turns)} turns; a conversation has {MIN_TURNS}–{MAX_TURNS}")
        numbers = [t.n for t in turns]
        if numbers != list(range(1, len(turns) + 1)):
            raise ValueError(f"turn numbers must run 1..{len(turns)}, got {numbers}")
        return turns


class ConversationSet(BaseModel):
    version: str
    description: str
    conversations: list[GoldenConversation]

    @field_validator("conversations")
    @classmethod
    def _ids_unique(cls, conversations: list[GoldenConversation]) -> list[GoldenConversation]:
        ids = [c.id for c in conversations]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate conversation ids: {duplicates}")
        return conversations


def load_conversations(path: Path = DEFAULT_CONVERSATIONS) -> ConversationSet:
    """Loads and validates the golden source; names the offending row on failure."""
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        return ConversationSet.model_validate(data)
    except ValidationError as exc:
        rows = data.get("conversations", []) if isinstance(data, dict) else []
        offending = sorted(
            {
                str(rows[e["loc"][1]].get("id", "?"))
                for e in exc.errors()
                if len(e["loc"]) > 1 and e["loc"][0] == "conversations"
                and isinstance(e["loc"][1], int) and e["loc"][1] < len(rows)
            }
        )
        raise ValueError(f"golden conversations invalid (rows {offending or ['?']}): {exc}") from exc


def render_markdown(conversations: ConversationSet) -> str:
    """One table per conversation, for the review comment on the ticket (G9)."""
    lines = [f"Golden conversations `{conversations.version}`: {conversations.description}", ""]
    for convo in conversations.conversations:
        lines.append(f"### {convo.id} · {convo.title}")
        lines.append(f"tier `{convo.tier}` · source `{convo.source}` · {len(convo.turns)} turns")
        lines.append("")
        lines.append("| n | user says | intent | path | constraints after the turn | flags | notes |")
        lines.append("|---|---|---|---|---|---|---|")
        for turn in convo.turns:
            e = turn.expect
            flags = ", ".join(
                f for f, on in (
                    ("no_repeat", e.no_repeat),
                    ("referenced_titles", e.referenced_titles_from_shown),
                    (f"ids={e.relevant_movie_ids}", bool(e.relevant_movie_ids)),
                ) if on
            )
            lines.append(
                f"| {turn.n} | {turn.user} | {e.intent.value} | {e.path} | "
                f"{e.constraints.summary()} | {flags} | {e.notes} |"
            )
        lines.append("")
    return "\n".join(lines)
