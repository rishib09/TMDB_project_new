"""Conversation-mode scoring (#93). Decisions of record: #87 Q12–Q14.

Pure functions + result models, no framework imports (ADR 0006) — the
driver feeds them two plain dicts per turn:

- ``effective``  — the composite constraint view: carried narrowing axes
  (mood/audience, and on ask turns the persisted year/exclusion constraints)
  ∪ the FINAL filters the retrieve node passed the engine (``filters_applied``).
- ``out``        — the graph's turn output (path observation, movies, fallback).

The rule (worked on real data, #87 Q13):
- fidelity = expected keys correct / (expected keys + violations);
- unexpected non-empty keys follow KEY_POLICY: ``genres`` is informational
  (v1's mood→genre candidates are a stack-dependent extra), everything else
  is a violation — the stale-carry class (#33 mood persisting after pivot,
  #26-E resets);
- ``genre_match="all"`` with empty expected genres is a failure
  (``over_constraint_intersection``) — #56-F2's signature.
"""

from typing import Literal

from pydantic import BaseModel, Field

from src.domain.memory import UserSessionPreferences
from src.domain.routing import IntentType
from src.evals.conversations import ExpectedConstraints
from src.maya.guardrails import GuardrailVerdict

#: One row per constraint key: is an unexpected value informational or a
#: violation? Additions are one row; policy flips are one cell (#87 Q13).
KEY_POLICY: dict[str, Literal["informational", "violation"]] = {
    "genres": "informational",  # candidate-list extras; set-equality applies when expected
    "mood": "violation",
    "audience": "violation",
    "exact_year": "violation",
    "year_min": "violation",
    "year_max": "violation",
    "director": "violation",
    "cast_member": "violation",
    "person": "violation",
    "excluded_genres": "violation",
    "excluded_actors": "violation",
    "runtime_max": "violation",
    "rating_min": "violation",
}

#: Engine-key fields lifted from ``filters_applied`` (genre_match rides along
#: for the match rule, never compared as a constraint).
_FILTER_KEYS = (
    "exact_year", "year_min", "year_max", "genres", "director",
    "cast_member", "person", "excluded_genres", "excluded_actors",
    "runtime_max", "rating_min",
)


def _present(value) -> bool:
    return value not in (None, [], "")


def _values_match(expected_value, observed_value) -> bool:
    if isinstance(expected_value, list) or isinstance(observed_value, list):
        return set(expected_value or []) == set(observed_value or [])
    return expected_value == observed_value


def observed_path_v1(out: dict) -> str:
    """Q12 mapping for the v1 stack — the harness stays stack-blind.

    ``filters_applied`` is the engine-invoked signal: non-None means the
    retrieve node ran, **even with 0 rows** (#21: a 0-row retrieve must not
    read as an ask). ``turn_stage`` is checked BEFORE the decision branches:
    a routed turn that ends in a probe (decision set, stage "probe") is an
    ask, not a retrieve — the smoke run's C01 t1 false failure.
    """
    guard = out.get("guardrail_result")
    if guard is not None and getattr(guard, "verdict", None) is GuardrailVerdict.BLOCKED:
        return "refuse"
    if out.get("filters_applied") is not None:
        return "retrieve"
    if out.get("turn_stage") in {"probe", "confirm", "confirm_genres"}:
        return "ask"  # deterministic funnel stage ended the turn
        # ("fallthrough" is absent: routing continues past the funnel, so the
        # decision branches below decide that turn)
    decision = out.get("routing_decision")
    if decision is None:
        return "ask"  # funnel-owned turn, router never ran
    if decision.intent in {IntentType.GREETING, IntentType.CAPABILITIES}:
        return "converse"
    if decision.intent is IntentType.OUT_OF_SCOPE:
        return "pivot"
    if not decision.requires_rag:
        return "converse"
    return "retrieve"  # defensive: requires_rag turns always record filters_applied


def composite_effective(out: dict, prefs: UserSessionPreferences) -> dict:
    """The turn's effective constraint view (Q13).

    Retrieve turns: engine truth only (``filters_applied``) plus the
    mood/audience axes — scoring prefs years here would HIDE the #78 bug
    (prefs had 2015, the engine never saw it). Ask turns: the carried
    state, years and exclusions included (C02 t1 expects year_max on an
    ask turn).
    """
    effective: dict = {}
    if prefs.preferred_mood:
        effective["mood"] = prefs.preferred_mood
    if prefs.audience:
        effective["audience"] = prefs.audience
    filters_applied = out.get("filters_applied")
    if filters_applied is not None:
        for key in _FILTER_KEYS:
            value = filters_applied.get(key)
            if _present(value):
                effective[key] = value
        if filters_applied.get("genre_match"):
            effective["genre_match"] = filters_applied["genre_match"]
    else:
        if prefs.exact_year is not None:
            effective["exact_year"] = prefs.exact_year
        if prefs.year_min is not None:
            effective["year_min"] = prefs.year_min
        if prefs.year_max is not None:
            effective["year_max"] = prefs.year_max
        if prefs.excluded_genres:
            effective["excluded_genres"] = prefs.excluded_genres
        if prefs.excluded_actors:
            effective["excluded_actors"] = prefs.excluded_actors
    return effective


def score_constraints(expected: ExpectedConstraints, effective: dict) -> dict:
    """The Q13 rule. Returns fidelity + the full per-key detail."""
    expected_fields = {
        key: value
        for key, value in expected.model_dump().items()
        if _present(value)
    }
    keys_ok: dict[str, bool] = {}
    for key, want in expected_fields.items():
        got = effective.get(key)
        keys_ok[key] = _present(got) and _values_match(want, got)
    unexpected = [
        key for key, value in effective.items()
        if key not in expected_fields and key != "genre_match" and _present(value)
    ]
    informational = [k for k in unexpected if KEY_POLICY.get(k) == "informational"]
    violations = [k for k in unexpected if KEY_POLICY.get(k) == "violation"]
    correct = sum(1 for ok in keys_ok.values() if ok)
    denominator = len(expected_fields) + len(violations)
    fidelity = (correct / denominator) if denominator else 1.0
    return {
        "fidelity": round(fidelity, 4),
        "keys": keys_ok,
        "informational": informational,
        "violations": violations,
        "intersection_failure": match_mode_rule(expected, effective),
    }


def match_mode_rule(expected: ExpectedConstraints, effective: dict) -> bool:
    """True when candidate acceptance silently became an intersection (#56-F2):
    ``genre_match="all"`` observed while the golden row expects NO genres."""
    if expected.genres:
        return False
    return effective.get("genre_match") == "all" and _present(effective.get("genres"))


def turn_failed(turn: ConversationTurnResult) -> bool:
    """One failure predicate, shared by the runner (row slice + conversation
    roll-up) and the Evals tab (failed-turn tables) — #93 review round 2."""
    detail = turn.constraint_detail or {}
    return bool(
        turn.intent_correct is False
        or not turn.path_correct
        or detail.get("intersection_failure")
        or detail.get("error")
        or any(not ok for ok in detail.get("keys", {}).values())
        or detail.get("violations")
        or turn.no_repeat_violation_ids
    )


# --- result models (Q14) ------------------------------------------------------

class ConversationTurnResult(BaseModel):
    n: int
    user: str
    expected_intent: str
    observed_intent: str
    #: None = the stack produced no reading this turn (v1 funnel-answer
    #: turns never route, G2 stack-neutrality) — excluded from aggregates.
    intent_correct: bool | None = None
    expected_path: str
    observed_path: str
    path_correct: bool
    expected_constraints: dict = Field(default_factory=dict)
    effective: dict = Field(default_factory=dict)
    constraint_detail: dict = Field(default_factory=dict)
    fidelity: float | None = None
    no_repeat: bool | None = None  # None = not expected this turn
    no_repeat_violation_ids: list[int] = Field(default_factory=list)
    reference_check: bool | None = None  # v1 has no referenced-titles mechanism (C10 is v2)
    ranked_ids: list[int] = Field(default_factory=list)
    relevant_ids: list[int] = Field(default_factory=list)
    hit_rate: float | None = None
    tokens: int = 0
    is_fallback: bool = False
    fallback_reason: str = ""
    trace_id: str = ""


class ConversationResult(BaseModel):
    id: str
    tier: str
    title: str
    n_turns: int
    per_turn: list[ConversationTurnResult] = Field(default_factory=list)
    #: None when the stack produced no readings at all (all turns funnel-owned)
    intent_accuracy: float | None = None
    path_accuracy: float = 0.0
    fidelity: float = 0.0
    failed: bool = False


class ConversationRunSummary(BaseModel):
    """mode="conversation" run envelope; filename/delta reuse the #59 machinery."""

    label: str
    mode: str = "conversation"
    config_snapshot: dict = Field(default_factory=dict)
    config_hash: str = ""
    preset: str = "custom"
    dataset_version: str = ""  # the golden conversations version
    routing_stack: str = "v1"
    n_conversations: int = 0
    n_turns: int = 0
    #: None when no turn produced a reading (all funnel-owned) — never a
    #: vacuous 1.0 (#93 review round 2)
    intent_accuracy: float | None = None
    path_accuracy: float = 0.0
    constraint_fidelity: float = 0.0
    no_repeat_rate: float = 0.0
    referenced_title_rate: float | None = None  # None until a stack scores references
    intersection_failures: int = 0
    judge_turns: int = 0
    faithfulness: float | None = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    per_conversation: list[ConversationResult] = Field(default_factory=list)
    delta: dict | None = None
