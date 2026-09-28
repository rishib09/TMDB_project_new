"""Adversarial: #113 — the Feedback Window path label on v2 turns.

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


def test_v1_single_route_label_unchanged():
    out = _v2_out("synthesize")
    row = _build([{"node": "route", "data": {}}], out)
    assert row["path"] == "single-route"


def test_v1_reroute_label_unchanged():
    out = _v2_out("synthesize")
    row = _build(
        [{"node": "route", "data": {}}, {"node": "route", "data": {}}], out
    )
    assert row["path"] == "reroute"


def test_v1_genuine_refusal_still_labels_refusal():
    # v1 refusal: guard diverts before any route node runs.
    out = _v2_out("refuse")
    out["routing_decision"] = None
    row = _build([], out)
    assert row["path"] == "funnel"  # decision-less turn keeps funnel semantics
    assert row["intent"] == "FUNNEL_REFUSE"


def test_v1_funnel_owned_retrieve_still_labels_funnel():
    out = _v2_out("retrieve")
    row = _build([], out)  # v1: stage=retrieve, no route_v2 traces
    assert row["path"] == "funnel"
