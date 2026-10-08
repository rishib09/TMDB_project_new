"""Unit tests for #153: the v2 carry-over notice + transparency chips.

Two layers:
- pure helpers in ``src/maya/v2/notices.py`` (the deliberate copy of v1's
  #26-E notice, plus the injection diff) — no graph, no LLM;
- the ``carryover_notice`` node's graph behavior on the v2 stack: fires
  exactly on silent-join turns, once per session, never on v1.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.domain.config import ExperimentConfig
from src.domain.memory import UserSessionPreferences
from src.domain.routing import IntentType, MetadataFilterCriteria
from src.graph.orchestrator import build_maya_graph
from src.maya.v2 import (
    MayaV2Router,
    PreferenceDelta,
    Understanding,
    build_filter_carryover_notice,
    injected_genres,
)
from src.observability.tracer import DualModeObservabilityManager
from src.ui.session import MayaSession
from tests.unit.test_orchestrator import (
    FakeEngine,
    FakeSynthesizer,
    _movie,
)

pytestmark = pytest.mark.unit


# --- pure helpers (src/maya/v2/notices.py) -----------------------------------

def test_v2_notice_announces_exactly_the_injected_genres():
    """Review 2026-10-04: the line scopes to what actually joined — not every
    remembered chip (v1's all-chips version stays in probing.py for v1)."""
    notice = build_filter_carryover_notice(["Action"])
    assert notice == (
        "\n\n---\nStill filtering by genres: Action — want to continue "
        "with these, or watch something completely different?"
    )
    assert "genres: Action, Comedy" in build_filter_carryover_notice(
        ["Action", "Comedy"]
    )


def test_v2_notice_empty_without_injected_genres():
    assert build_filter_carryover_notice([]) == ""


def test_injected_genres_is_effective_minus_declared():
    assert injected_genres({"genres": ["Action"]}, None) == ["Action"]
    assert injected_genres({"genres": ["Action"]}, MetadataFilterCriteria(
        genres=["Action"]
    )) == []  # the turn declared it — nothing joined silently
    assert injected_genres({"genres": ["Action"]}, MetadataFilterCriteria(
        genres=["action"]
    )) == []  # model casing must not resurrect a declared genre
    assert injected_genres(
        {"genres": ["Action", "Comedy"]}, MetadataFilterCriteria(genres=["Comedy"])
    ) == ["Action"]


def test_injected_genres_empty_without_effective_filters():
    assert injected_genres(None, None) == []
    assert injected_genres({}, MetadataFilterCriteria()) == []
    assert injected_genres({"genres": []}, None) == []


# --- the node on the v2 graph ------------------------------------------------


class ScriptedV2(MayaV2Router):
    """Real constructor (no network); scripted understand() results."""

    def __init__(self, results):
        super().__init__(ExperimentConfig(routing_stack="v2"), api_key="test-key")
        self.results = list(results)

    def understand(self, query, prefs, shown_titles, last_assistant, probe_count):
        result = self.results.pop(0)
        if len(result) == 2:
            return result[0], result[1], None
        return result


def _v2u(**kw) -> Understanding:
    defaults = dict(
        intent=IntentType.SEMANTIC_SEARCH,
        standalone_query="q", confidence=0.9,
    )
    defaults.update(kw)
    return Understanding(**defaults)


def _v2graph(router, engine, session_id="unit-153", tracer=None):
    return build_maya_graph(
        ExperimentConfig(routing_stack="v2"), router, engine, FakeSynthesizer(),
        tracer or DualModeObservabilityManager(session_id=session_id),
        checkpointer=InMemorySaver(),
    )


def _cfg(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


def test_notice_fires_when_session_genres_join_the_sql():
    """Turn 1 declares Action (no injection, no notice); turn 2 declares no
    genres so code folds the remembered Action in — the notice must announce
    it, once, inside the SAME assistant message (no duplicate bubble)."""
    u1 = _v2u(
        filters=MetadataFilterCriteria(genres=["Action"]),
        preference_delta=PreferenceDelta(add_genres=["Action"]),
    )
    u2 = _v2u(ready_to_retrieve=True)
    router = ScriptedV2([(u1, []), (u2, [])])
    tracer = DualModeObservabilityManager(session_id="unit-153-join")
    graph = _v2graph(router, FakeEngine(movies=[_movie()]), tracer=tracer)
    cfg = _cfg("notice-join")

    out1 = graph.invoke({"messages": [HumanMessage(content="action movies")]}, cfg)
    assert out1["final_response"] == "Here is what I found."  # declared → silent
    assert "Still filtering by" not in out1["final_response"]

    out2 = graph.invoke({"messages": [HumanMessage(content="more, please")]}, cfg)
    assert "Still filtering by genres: Action" in out2["final_response"]
    assert "completely different?" in out2["final_response"]  # escape hatch
    # replacement, not duplication: exactly one assistant message carries it
    noticed = [
        m for m in out2["messages"]
        if isinstance(m, AIMessage) and "Still filtering by" in m.content
    ]
    assert len(noticed) == 1
    assert noticed[0].content == out2["final_response"]
    # once-per-session flag persisted in the thread
    assert graph.get_state(cfg).values["carryover_notice_shown"] is True
    # explicit in the trace (telemetry rule): WHICH genres joined silently, and
    # whether the transcript was replaced — the no-id fail-open must be
    # distinguishable in the payload (review 2026-10-04)
    fired = [t for t in tracer.traces() if t["node"] == "carryover_notice"]
    assert fired and fired[0]["payload"]["injected_genres"] == ["Action"]
    assert fired[0]["payload"]["fired"] is True
    assert fired[0]["payload"]["transcript_replaced"] is True


def test_notice_never_overclaims_what_did_not_join():
    """Review 2026-10-04: the trigger is genre-scoped (#25 injects genres
    only), so a remembered DIRECTOR must not be announced as 'still
    filtering' — the old all-chips line claimed filters that never ran."""
    u1 = _v2u(
        filters=MetadataFilterCriteria(genres=["Action"]),
        preference_delta=PreferenceDelta(add_genres=["Action"]),
    )
    u2 = _v2u(ready_to_retrieve=True)
    router = ScriptedV2([(u1, []), (u2, [])])
    graph = _v2graph(router, FakeEngine(movies=[_movie()]),
                     session_id="unit-153-scope")
    cfg = _cfg("notice-scope")
    graph.invoke(
        {
            "messages": [HumanMessage(content="action movies")],
            "session_preferences": UserSessionPreferences(
                preferred_directors=["Christopher Nolan"],
            ),
        },
        cfg,
    )
    out2 = graph.invoke({"messages": [HumanMessage(content="more")]}, cfg)
    assert "Still filtering by genres: Action" in out2["final_response"]
    assert "dir. Christopher Nolan" not in out2["final_response"]


def test_notice_fires_once_per_session():
    u1 = _v2u(
        filters=MetadataFilterCriteria(genres=["Action"]),
        preference_delta=PreferenceDelta(add_genres=["Action"]),
    )
    u2 = _v2u(ready_to_retrieve=True)
    u3 = _v2u(ready_to_retrieve=True)
    router = ScriptedV2([(u1, []), (u2, []), (u3, [])])
    graph = _v2graph(router, FakeEngine(movies=[_movie()]))
    cfg = _cfg("notice-once")

    graph.invoke({"messages": [HumanMessage(content="action movies")]}, cfg)
    out2 = graph.invoke({"messages": [HumanMessage(content="more")]}, cfg)
    out3 = graph.invoke({"messages": [HumanMessage(content="again")]}, cfg)

    assert "Still filtering by" in out2["final_response"]
    assert "Still filtering by" not in out3["final_response"]  # debounce holds


def test_notice_skips_zero_retrieval_turns():
    """Injection may fire, but with no movies shown there is nothing carried —
    the deterministic refinement question must not be followed by a nag."""
    u1 = _v2u(intent=IntentType.SEMANTIC_SEARCH, ready_to_retrieve=True)
    router = ScriptedV2([(u1, [])])
    graph = _v2graph(router, FakeEngine(movies=[]), session_id="unit-153-empty")
    out = graph.invoke(
        {
            "messages": [HumanMessage(content="some movie")],
            "session_preferences": UserSessionPreferences(preferred_genres=["Action"]),
        },
        _cfg("notice-empty"),
    )
    assert out["retrieved_movies"] == []
    assert "Still filtering by" not in out["final_response"]
    assert out.get("carryover_notice_shown", False) is False  # never fired


def test_current_filter_chip_uses_effective_set_not_declared():
    """The Nolan/Action repro: the turn declared only director+years, code
    injected Action — the chip must show Action (the old code hid it)."""
    from src.domain.routing import QueryRoutingDecision

    decision = QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH, confidence=0.9,
        standalone_query="nolan", requires_rag=True,
        filters=MetadataFilterCriteria(director="Christopher Nolan", year_min=2017),
    )
    applied = {"genres": ["Action"], "genre_match": "any",
               "director": "Christopher Nolan", "year_min": 2017}
    chips = MayaSession._effective_filter_chips(applied, decision)
    assert "genres: Action" in chips
    assert "dir. Christopher Nolan" in chips
    assert "2017–…" in chips


def test_current_filter_chip_falls_back_to_decision_without_engine_run():
    from src.domain.routing import QueryRoutingDecision

    decision = QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH, confidence=0.9,
        standalone_query="horror", requires_rag=True,
        filters=MetadataFilterCriteria(genres=["Horror", "Thriller"], genre_match="all"),
    )
    assert MayaSession._effective_filter_chips(None, decision) == (
        MayaSession._filter_chips(decision)
    )
    assert "genres: Horror, Thriller (all)" in (
        MayaSession._effective_filter_chips(None, decision)
    )