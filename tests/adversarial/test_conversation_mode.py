"""Adversarial tests for conversation mode (#93): checkpointer, begin_turn,
trim window, filters_applied.

Pins the D16 contract: one thread per conversation, callers send ONLY the new
message, the graph remembers everything else. Each test fails on the
pre-#93 graph (no checkpointer -> get_state raises; state hand-copy -> carry
lost; no begin_turn -> stale scratch; no trim -> unbounded window).
"""

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.domain.config import ExperimentConfig
from src.domain.movie import MovieRecord
from src.domain.routing import IntentType, MetadataFilterCriteria, QueryRoutingDecision
from src.graph.orchestrator import build_maya_graph
from src.graph.state import SynthesisUsage
from src.observability.tracer import DualModeObservabilityManager

pytestmark = pytest.mark.adversarial


def _movie(mid: int, title: str, year: int = 2001) -> MovieRecord:
    return MovieRecord(id=mid, title=title, release_year=year)


class ScriptedRouter:
    """Pops one decision per route call; records the state it was given."""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.seen_states = []

    def route(self, query, state, feedback=None):
        self.seen_states.append(state)
        return self.decisions.pop(0)


class RecordingEngine:
    """Deterministic movie pool; records every engine call's routing filters."""

    def __init__(self, movies=None):
        self.movies = movies or []
        self.calls = []

    def retrieve(self, query, routing, top_k=8, candidate_pool=50):
        self.calls.append({"query": query, "routing": routing, "top_k": top_k})
        return [
            type("R", (), {"movie": m, "score": 1.0, "source": "dense"})()
            for m in self.movies[:top_k]
        ]


class StubSynthesizer:
    def __init__(self):
        self.calls = []

    def synthesize(self, query, decision, movies, history):
        self.calls.append((query, len(movies)))
        return f"Here are {len(movies)} movies.", SynthesisUsage(
            model="stub", prompt_tokens=3, completion_tokens=5
        )


def _decision(intent=IntentType.SEMANTIC_SEARCH, *, requires_rag=True,
              filters=None, mood="", audience=""):
    return QueryRoutingDecision(
        intent=intent,
        confidence=0.95,
        standalone_query="stub query",
        requires_rag=requires_rag,
        filters=filters,
        mood=mood,
        audience=audience,
    )


def _graph(router, engine, synthesizer=None):
    return build_maya_graph(
        ExperimentConfig(),
        router,
        engine,
        synthesizer or StubSynthesizer(),
        DualModeObservabilityManager(session_id="adv-93"),
        checkpointer=InMemorySaver(),
    )


def _cfg(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


def test_thread_carries_preferences_and_probe_count_without_resent_state():
    """D16: turn 2 sends ONLY the new message; prefs/probe_count ride the thread."""
    router = ScriptedRouter([
        _decision(mood="scary", filters=MetadataFilterCriteria(year_min=2015)),  # retrieve, mood merged
        _decision(filters=MetadataFilterCriteria(year_min=1990, genres=["Horror"])),  # retrieve
    ])
    graph = _graph(router, RecordingEngine())
    cfg = _cfg("carry-1")

    graph.invoke({"messages": [HumanMessage(content="something scary")]}, cfg)
    second = graph.invoke({"messages": [HumanMessage(content="horror movies")]}, cfg)

    values = graph.get_state(cfg).values
    assert values["session_preferences"].preferred_mood == "scary"  # turn 1's mood rides the thread
    assert values.get("probe_count", 0) == 0  # both turns retrieved — never probed (absent channel = 0)
    # v1 semantics: explicit decision genres ride the decision (engine
    # filters), not session_preferences — they must still be visible to the
    # turn that stated them.
    assert second["filters_applied"]["genres"] == ["Horror"]


def test_threads_are_isolated():
    router = ScriptedRouter([_decision(mood="scary"), _decision()])
    graph = _graph(router, RecordingEngine())
    a, b = _cfg("iso-a"), _cfg("iso-b")

    graph.invoke({"messages": [HumanMessage(content="something scary")]}, a)
    graph.invoke({"messages": [HumanMessage(content="hello again")]}, b)

    assert graph.get_state(a).values["session_preferences"].preferred_mood == "scary"
    assert graph.get_state(b).values["session_preferences"].preferred_mood == ""


def test_begin_turn_resets_scratch_between_turns():
    """A pivot turn after a retrieve turn must not see the previous turn's
    routing_decision / retrieved_movies / final_response."""
    router = ScriptedRouter([
        _decision(filters=MetadataFilterCriteria(year_min=2000)),  # retrieve
        _decision(intent=IntentType.OUT_OF_SCOPE, requires_rag=False),  # pivot
    ])
    graph = _graph(router, RecordingEngine([_movie(1, "A"), _movie(2, "B")]))
    cfg = _cfg("scratch-1")

    first = graph.invoke({"messages": [HumanMessage(content="horror movies")]}, cfg)
    assert first["retrieved_movies"]  # turn 1 retrieved

    second = graph.invoke({"messages": [HumanMessage(content="tell me a joke")]}, cfg)
    assert second["retrieved_movies"] == []          # stale movies gone
    assert second["routing_decision"].intent is IntentType.OUT_OF_SCOPE
    assert "curator" in second["final_response"]     # pivot text, not turn 1's


def test_message_window_capped_at_ten():
    """The trim node keeps the 10-message window inside the thread."""
    router = ScriptedRouter([
        _decision(intent=IntentType.OUT_OF_SCOPE, requires_rag=False)
    ] * 8)
    graph = _graph(router, RecordingEngine())
    cfg = _cfg("window-1")

    for i in range(8):  # 8 pivot turns = 16 messages if untrimmed
        graph.invoke({"messages": [HumanMessage(content=f"off topic {i}")]}, cfg)

    assert len(graph.get_state(cfg).values["messages"]) == 10


def test_shown_movie_ids_accumulate_across_retrieve_turns():
    """#80 by construction: the union of shown ids survives turn boundaries —
    the state the caller used to forget to pass now rides the checkpoint."""
    router = ScriptedRouter([
        _decision(filters=MetadataFilterCriteria(year_min=2000)),
        _decision(filters=MetadataFilterCriteria(year_min=1990)),
    ])
    engine = RecordingEngine([_movie(1, "A"), _movie(2, "B"), _movie(3, "C")])
    graph = _graph(router, engine)
    cfg = _cfg("shown-1")

    graph.invoke({"messages": [HumanMessage(content="recent movies")]}, cfg)
    assert graph.get_state(cfg).values["shown_movie_ids"] == [1, 2, 3]

    graph.invoke({"messages": [HumanMessage(content="older movies")]}, cfg)
    # union — no reset between turns, no duplicates
    assert graph.get_state(cfg).values["shown_movie_ids"] == [1, 2, 3]


def test_retrieve_records_filters_applied_and_engine_invoked_signal():
    """Q3 O1: the retrieve node records the FINAL effective filters; None
    means the engine never ran this turn ({} = ran, zero filters)."""
    router = ScriptedRouter([
        _decision(filters=MetadataFilterCriteria(
            year_min=2015, genres=["Comedy"], genre_match="all",
        )),
        _decision(intent=IntentType.OUT_OF_SCOPE, requires_rag=False),
    ])
    engine = RecordingEngine([_movie(1, "A")])
    graph = _graph(router, engine)
    cfg = _cfg("filters-1")

    out = graph.invoke({"messages": [HumanMessage(content="recent comedies")]}, cfg)
    assert out["filters_applied"] == {
        "exact_year": None, "year_min": 2015, "year_max": None,
        "genres": ["Comedy"], "genre_match": "all", "director": None,
        "cast_member": None, "person": None,
        "excluded_genres": [], "excluded_actors": [],
    }
    # the engine saw exactly those filters
    assert engine.calls[0]["routing"].filters.year_min == 2015

    out2 = graph.invoke({"messages": [HumanMessage(content="a joke")]}, cfg)
    assert out2["filters_applied"] is None  # pivot turn — engine never ran


def test_zero_result_retrieve_still_signals_engine_invoked():
    """#21 amendment (Q12): a 0-row retrieve is a retrieve, not an ask."""
    router = ScriptedRouter([
        _decision(filters=MetadataFilterCriteria(year_min=2026)),
    ])
    graph = _graph(router, RecordingEngine([]))  # empty pool -> 0 rows
    cfg = _cfg("zero-1")

    out = graph.invoke({"messages": [HumanMessage(content="future films")]}, cfg)
    # 0 rows, but the engine RAN with the stated constraint — the signal is
    # non-None filters_applied, not movie count (Q12: 0-row retrieve ≠ ask).
    assert out["filters_applied"]["year_min"] == 2026
    assert out["retrieved_movies"] == []
