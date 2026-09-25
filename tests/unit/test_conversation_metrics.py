"""Unit tests for conversation-mode scoring (#93).

The four #87-Q13 worked examples are pinned verbatim — they were validated
on real data (the #85 v2 prototype response and the v1 trace 0839ce6c
behind #78/#80) before the rule was accepted.
"""

import pytest

from src.domain.memory import UserSessionPreferences
from src.domain.routing import IntentType, QueryRoutingDecision
from src.evals.conversations import ExpectedConstraints
from src.evals.conversation_metrics import (
    composite_effective,
    match_mode_rule,
    observed_path_v1,
    score_constraints,
)
from src.maya.guardrails import GuardrailResult, GuardrailVerdict

pytestmark = pytest.mark.unit


def _decision(intent=IntentType.SEMANTIC_SEARCH, *, requires_rag=True, **kw):
    return QueryRoutingDecision(
        intent=intent, confidence=0.9, standalone_query="q",
        requires_rag=requires_rag, **kw
    )


# --- Example 1: C01 t4 read correctly (v2-style, from the #85 prototype) ------

def test_c01_t4_clean_pass_scores_full_fidelity():
    expected = ExpectedConstraints(
        mood="feel-good", audience="date night", year_min=2015
    )
    effective = {
        "mood": "feel-good", "audience": "date night", "year_min": 2015,
    }
    detail = score_constraints(expected, effective)
    assert detail["fidelity"] == 1.0
    assert detail["violations"] == [] and detail["informational"] == []
    assert detail["intersection_failure"] is False


# --- Example 2: the #78/#80 regression, as v1 actually ran it -----------------

def test_v1_year_filter_miss_is_caught_and_genres_are_informational():
    expected = ExpectedConstraints(
        mood="feel-good", audience="date night", year_min=2015
    )
    effective = {  # engine saw mood/audience + candidate genres, NOT the year
        "mood": "feel-good", "audience": "date night",
        "genres": ["Comedy", "Drama", "Family", "Romance"],
        "genre_match": "any",
    }
    detail = score_constraints(expected, effective)
    assert detail["fidelity"] == pytest.approx(2 / 3, abs=1e-3)
    assert detail["keys"]["year_min"] is False
    assert detail["informational"] == ["genres"]  # union is legitimate, not penalized
    assert detail["violations"] == []


# --- Example 3: #33-class stale mood is a violation, not telemetry ------------

def test_stale_mood_after_pivot_lowers_fidelity():
    expected = ExpectedConstraints(genres=["Action"])  # mood retired by the pivot
    effective = {"mood": "funny", "genres": ["Action"]}
    detail = score_constraints(expected, effective)
    assert detail["fidelity"] == 0.5  # 1 correct / (1 expected + 1 violation)
    assert detail["violations"] == ["mood"]


def test_stale_genre_carry_fails_set_equality():
    expected = ExpectedConstraints(genres=["Action"])
    effective = {"genres": ["Action", "Comedy"]}  # Comedy survived the pivot (#33)
    detail = score_constraints(expected, effective)
    assert detail["keys"]["genres"] is False
    assert detail["fidelity"] == 0.0


# --- Example 4: #56-F2 — union vs intersection --------------------------------

def test_candidate_union_is_informational_but_intersection_fails():
    expected = ExpectedConstraints(mood="feel-good", year_max=2000)  # no genres demanded
    union = {"mood": "feel-good", "year_max": 2000,
             "genres": ["Comedy", "Drama", "Family", "Romance"], "genre_match": "any"}
    intersection = {**union, "genre_match": "all"}

    detail_union = score_constraints(expected, union)
    assert detail_union["fidelity"] == 1.0
    assert detail_union["intersection_failure"] is False

    detail_intersection = score_constraints(expected, intersection)
    assert detail_intersection["intersection_failure"] is True  # the failure flag
    assert detail_intersection["fidelity"] == 1.0  # flag is separate from fidelity


def test_match_mode_rule_only_fires_on_empty_expected_genres():
    expected = ExpectedConstraints(genres=["Sci-Fi", "Horror"])  # explicit conjunction
    assert match_mode_rule(expected, {"genres": ["Sci-Fi", "Horror"], "genre_match": "all"}) is False


# --- empty expectations --------------------------------------------------------

def test_no_constraints_expected_full_fidelity_unless_violations():
    detail = score_constraints(ExpectedConstraints(), {"mood": "scary"})
    assert detail["fidelity"] == 0.0  # unexpected mood = violation
    assert score_constraints(ExpectedConstraints(), {})["fidelity"] == 1.0


# --- path observation (Q12) ----------------------------------------------------

def test_observed_path_v1_mapping():
    assert observed_path_v1({
        "filters_applied": {"year_min": 2015}, "routing_decision": _decision(),
        "retrieved_movies": [],
    }) == "retrieve"
    # 0-row retrieve is still a retrieve (#21 amendment)
    assert observed_path_v1({
        "filters_applied": {}, "routing_decision": _decision(), "retrieved_movies": [],
    }) == "retrieve"
    # funnel probe turn: no decision, no engine
    assert observed_path_v1({
        "routing_decision": None, "turn_stage": "probe", "final_response": "What mood?",
    }) == "ask"
    assert observed_path_v1({
        "routing_decision": _decision(IntentType.OUT_OF_SCOPE, requires_rag=False),
    }) == "pivot"
    assert observed_path_v1({
        "routing_decision": _decision(IntentType.GREETING, requires_rag=False),
    }) == "converse"
    blocked = GuardrailResult(verdict=GuardrailVerdict.BLOCKED, sanitized_query="", reason="budget")
    assert observed_path_v1({"guardrail_result": blocked}) == "refuse"


# --- composite effective view (Q13) --------------------------------------------

def test_composite_prefers_engine_truth_on_retrieve_turns():
    """The #78 catch: prefs year must NOT mask the engine's missing year."""
    out = {"filters_applied": {"genres": ["Comedy"]}}
    prefs = UserSessionPreferences(
        preferred_mood="feel-good", audience="date night", year_min=2015,
    )
    effective = composite_effective(out, prefs)
    assert effective == {"mood": "feel-good", "audience": "date night", "genres": ["Comedy"]}


def test_composite_carries_prefs_years_on_ask_turns():
    """C02 t1/t2: era constraints are scored on ask turns from carried state."""
    out = {"filters_applied": None}
    prefs = UserSessionPreferences(preferred_mood="feel-good", year_max=2000)
    assert composite_effective(out, prefs) == {"mood": "feel-good", "year_max": 2000}
