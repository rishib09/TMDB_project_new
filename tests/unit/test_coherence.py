"""Unit tests for #26 coherence fixes: atomic turn rows,
fresh-start reset, inline metadata (v2 stack, #156).

The v1 funnel coherence coverage (genre guard, funnel turn rows, v1
carry-over notice) died with the v1 fork; the v2 notice has its own suite
(test_v2_carryover_notice.py) and the C7 code guard the v2 disposer's.
"""

import pytest
from langchain_core.messages import HumanMessage

from src.domain.config import ExperimentConfig
from src.domain.memory import UserSessionPreferences, merge_preferences
from src.domain.routing import IntentType, MetadataFilterCriteria, QueryRoutingDecision
from src.graph.orchestrator import build_maya_graph
from src.maya.guardrails import SessionCostLimiter
from src.maya.probing import extract_probe_answers, is_fresh_start, preference_chips
from src.observability.tracer import DualModeObservabilityManager
from src.ui.chat_tab import intent_badge_text
from tests.unit.test_orchestrator import (
    FakeEngine,
    FakeSynthesizer,
    ScriptedV2,
    _movie,
)

pytestmark = pytest.mark.unit


# --- A: atomic turn rows (#26-A) -------------------------------------------

def _row(out, **kwargs):
    from src.ui.session import MayaSession

    defaults = dict(
        query="scary movies", trace_id="t1", rag_version="v1_1_enriched",
        new_traces=[], prev_tokens=0,
    )
    defaults.update(kwargs)
    return MayaSession._build_turn_row(out, **defaults)


def _decision(filters=None):
    return QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH, confidence=0.9,
        standalone_query="stub query", requires_rag=True, filters=filters,
    )


def test_refusal_turn_row_is_complete_without_decision():
    """On the v2 fork a missing decision means the guard refused (#156):
    the row must stay complete — intent REFUSAL, path refusal."""
    row = _row({
        "final_response": "I can't help with that.",
        "turn_stage": "",
        "retrieved_movies": [],
    })
    assert row["intent"] == "REFUSAL"
    assert row["confidence"] == 1.0
    assert row["path"] == "refusal"
    assert row["response"] == "I can't help with that."


def test_ask_turn_row_labels_the_stage():
    row = _row({
        "final_response": "Solo or family night?",
        "turn_stage": "ask",
        "routing_decision": _decision(),
        "retrieved_movies": [],
        "new_traces": [{"node": "route_v2", "payload": {}}],
    })
    assert row["path"] == "ask"
    assert row["intent"] == "SEMANTIC_SEARCH"


def test_row_includes_active_filter_chips():
    decision = _decision(MetadataFilterCriteria(
        genres=["Horror", "Thriller"], genre_match="all", year_min=2000,
    ))
    row = _row({
        "final_response": "r", "routing_decision": decision,
        "retrieved_movies": [], "session_preferences": UserSessionPreferences(),
    })
    assert "genres: Horror, Thriller (all)" in row["filters"]
    assert "2000–…" in row["filters"]


# --- D: mood vocab fallback (#26-D) ------------------------------------------

def test_horror_maps_to_scary_mood():
    assert extract_probe_answers("horror movies").preferred_mood == "scary"


# --- E: fresh-start reset (#26-E) ---------------------------------------------

def test_reset_requested_wipes_preferences_through_the_reducer():
    populated = UserSessionPreferences(
        preferred_mood="scary", audience="kids", preferred_genres=["Horror"],
    )
    merged = merge_preferences(populated, UserSessionPreferences(reset_requested=True))
    assert merged == UserSessionPreferences()


@pytest.mark.parametrize("phrase", [
    "something completely different", "let's start fresh",
    "I want to start over", "watch something else",
])
def test_is_fresh_start_vocabulary(phrase):
    assert is_fresh_start(phrase)


def test_is_fresh_start_ignores_normal_queries():
    assert not is_fresh_start("scary movies for kids")


def _guard_reset_graph():
    """The escape hatch works at the ONE choke point every turn passes."""
    from src.maya.v2 import Understanding

    u = Understanding(
        intent=IntentType.GREETING, standalone_query="something completely different",
        confidence=0.9,
    )
    graph = build_maya_graph(
        ExperimentConfig(),
        ScriptedV2([(u, [])]),
        FakeEngine(),
        FakeSynthesizer(),
        DualModeObservabilityManager(session_id="t"),
        limiter=SessionCostLimiter(),
    )
    return graph


def test_guard_reset_wipes_preferences_in_thread():
    graph = _guard_reset_graph()
    out = graph.invoke({
        "messages": [HumanMessage(content="something completely different")],
        "session_preferences": UserSessionPreferences(
            preferred_mood="scary", audience="kids",
        ),
    })
    assert out["session_preferences"] == UserSessionPreferences()


# --- F: inline metadata line (#26-F) -----------------------------------------

def test_intent_badge_carries_narrowing_and_filters_inline():
    row = {
        "intent": "ATTRIBUTE_FILTER", "confidence": 0.9, "path": "retrieve",
        "attempts": 0, "n_movies": 5, "tokens": 100,
        "narrowing": ["mood: scary", "audience: kids"],
        "filters": ["genres: Horror, Thriller (all)"],
    }
    text = intent_badge_text(row)
    assert "Narrowing by: mood: scary · audience: kids" in text
    assert "Filters: genres: Horror, Thriller (all)" in text
    assert "INTENT: ATTRIBUTE_FILTER" in text


def test_intent_badge_omits_empty_sections():
    text = intent_badge_text({
        "intent": "GREETING", "confidence": 1.0, "path": "retrieve",
        "attempts": 0, "n_movies": 0, "tokens": 0,
    })
    assert "Narrowing" not in text and "Filters" not in text


def test_preference_chips_format():
    chips = preference_chips(UserSessionPreferences(
        preferred_mood="scary", audience="kids", noted_donts=["clowns"],
        preferred_genres=["Horror"], preferred_directors=["Cronenberg"],
    ))
    assert chips == [
        "mood: scary", "audience: kids", "no clowns",
        "genres: Horror", "dir. Cronenberg",
    ]
