"""Adversarial: #113 — the Feedback Window path label on v2 turns (v2 fork, #156).

`_build_turn_row` derived the path from v1's `route` node traces only; v2
records `route_v2` (#106 funnel collapse), so every v2 routed turn was
labeled "refusal" — a visitor filed it as "keeps going to route refusal"
(#75 comment 5875796867). These tests fail on the pre-fix code.
"""

from types import SimpleNamespace

from src.domain.routing import IntentType
from src.ui.session import MayaSession


def _v2_out(stage: str) -> dict:
    return {
        "routing_decision": SimpleNamespace(
            intent=IntentType.SEMANTIC_SEARCH, confidence=0.9, filters=None
        ),
        "turn_stage": stage,
        "final_response": "reply",
        "session_tokens": 100,
        "session_cost_usd": 0.01,
        "retrieved_movies": [101, 102],
        "session_preferences": None,
    }


def _build(new_traces: list[dict], out: dict) -> dict:
    return MayaSession._build_turn_row(
        out,
        query="feel good movies",
        trace_id="trace-1",
        rag_version="test",
        new_traces=new_traces,
        prev_tokens=0,
        prev_cost=0.0,
    )


def test_v2_routed_retrieve_turn_is_not_labeled_refusal():
    row = _build([{"node": "route_v2", "data": {}}], _v2_out("retrieve"))
    assert row["path"] != "refusal"
    assert row["path"] == "retrieve"


def test_v2_ask_turn_labels_ask():
    row = _build([{"node": "route_v2", "data": {}}], _v2_out("ask"))
    assert row["path"] == "ask"


def test_v2_clean_ask_without_notes_still_labels_ask():
    """Live finding (#113): a clean glm ask emits zero guard notes — the
    unconditional route_v2 passage marker is the only trace evidence."""
    row = _build([{"node": "route_v2", "data": {"turn_decision": "ask"}}], _v2_out("ask"))
    assert row["path"] == "ask"


def test_v2_clean_retrieve_resets_stage_but_labels_retrieve():
    """v2 retrieves reset turn_stage to \"\" (per-turn state reset) — the
    passage marker must carry the label, not the stale stage."""
    out = _v2_out("")
    row = _build(
        [{"node": "route_v2", "data": {"turn_decision": "retrieve"}}], out
    )
    assert row["path"] == "retrieve"
    assert row["path"] != "refusal"


def test_v2_decision_present_without_evidence_is_retrieve_never_refusal():
    """Live finding, C06 t2: a full retrieval turn (5 movies synthesized)
    arrived with decision present, stage '', and no route evidence — the
    old fallthrough labeled it 'refusal'. Decision-present turns are never
    refusals; refusals are guard-diverted decisionless."""
    out = _v2_out("")
    row = _build([], out)
    assert row["path"] == "retrieve"


def test_refusal_node_trace_labels_refusal_even_with_decision():
    out = _v2_out("")
    row = _build([{"node": "refusal", "data": {}}], out)
    assert row["path"] == "refusal"


def test_pivot_node_trace_labels_pivot():
    out = _v2_out("")
    row = _build([{"node": "pivot", "data": {}}], out)
    assert row["path"] == "pivot"
