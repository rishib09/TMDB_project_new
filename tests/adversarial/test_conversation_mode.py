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

    def retrieve(self, query, routing, top_k=8, candidate_pool=50, shown_ids=None, boost=None):
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


def test_persistent_exclusions_accumulate_in_thread():
    """D16 seam fix: decision exclusions must persist in-graph — the old
    session-side add_turn merge no longer runs to feed them back."""
    router = ScriptedRouter([
        _decision(filters=MetadataFilterCriteria(excluded_genres=["Horror"])),
        _decision(filters=MetadataFilterCriteria(year_min=1990)),
    ])
    graph = _graph(router, RecordingEngine([_movie(1, "A")]))
    cfg = _cfg("excl-1")

    graph.invoke({"messages": [HumanMessage(content="funny but no horror")]}, cfg)
    graph.invoke({"messages": [HumanMessage(content="older movies")]}, cfg)

    assert graph.get_state(cfg).values["session_preferences"].excluded_genres == ["Horror"]


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
        "runtime_max": None, "rating_min": None,  # #82 C6: additive v2 fields
        "vote_count_min": None,  # #137: additive mood-floor field
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


def test_relaxation_retry_records_final_match_mode():
    """#93 review round 3 (spec P1): the all->any relaxation retry must leave
    `decision` on the RELAXED routing — filters_applied records what the
    engine actually ran, never a phantom intersection."""
    class MatchSensitiveEngine:
        """Empty on intersection ("all"), results on union ("any")."""
        last_dense_failure = None

        def retrieve(self, query, routing, top_k=8, candidate_pool=50, shown_ids=None, boost=None):
            if routing.filters.genre_match == "all":
                return []
            return [
                type("R", (), {"movie": _movie(1, "A"), "score": 1.0, "source": "sql"})()
            ]

    router = ScriptedRouter([
        _decision(filters=MetadataFilterCriteria(
            genres=["Comedy", "Romance"], genre_match="all", year_min=2000,
        )),
    ])
    graph = _graph(router, MatchSensitiveEngine())
    out = graph.invoke(
        {"messages": [HumanMessage(content="comedy romance hybrids")]},
        _cfg("relax-1"),
    )
    assert out["filters_applied"]["genre_match"] == "any"  # the FINAL routing
    assert out["filters_applied"]["genres"] == ["Comedy", "Romance"]
    assert len(out["retrieved_movies"]) == 1


# --- #80: fresh top-k — shown ids ride the thread and reset on fresh start ---


class PooledEngine:
    """Pops one scripted result set per retrieve; records the shown_ids it
    was handed (the #88 orchestrator handoff)."""

    def __init__(self, pools):
        self.pools = [list(p) for p in pools]
        self.shown_seen = []

    def _results(self, movies):
        return [type("R", (), {"movie": m, "score": 1.0, "source": "dense"})() for m in movies]

    def retrieve(self, query, routing, top_k=8, candidate_pool=50, shown_ids=None, boost=None):
        self.shown_seen.append(list(shown_ids or []))
        return self._results(self.pools.pop(0)[:top_k])


class HonoringEngine(PooledEngine):
    """Same contract as the real engine: excludes shown ids before top-k."""

    def retrieve(self, query, routing, top_k=8, candidate_pool=50, shown_ids=None, boost=None):
        shown = set(shown_ids or [])
        self.shown_seen.append(list(shown_ids or []))
        fresh = [m for m in self.pools.pop(0) if m.id not in shown]
        return self._results(fresh[:top_k])


def test_fresh_start_clears_shown_ids_in_thread():
    """"something completely different" must wipe the shown slate (#80).

    Red on current code: the union reducer kept every id forever — the guard
    reset preferences but could not express clearing the id accumulator, so
    earlier titles stayed excluded for the whole thread.
    """
    engine = PooledEngine([
        [_movie(1, "A"), _movie(2, "B"), _movie(3, "C")],
        [_movie(7, "G"), _movie(8, "H")],
    ])
    router = ScriptedRouter([
        _decision(filters=MetadataFilterCriteria(year_min=2000)),
        _decision(filters=MetadataFilterCriteria(year_min=1990)),
    ])
    graph = _graph(router, engine)
    cfg = _cfg("fresh-clear")
    graph.invoke({"messages": [HumanMessage(content="scary movies")]}, cfg)
    assert graph.get_state(cfg).values["shown_movie_ids"] == [1, 2, 3]

    graph.invoke(
        {"messages": [HumanMessage(content="something completely different")]}, cfg
    )
    # reset reached retrieve: turn 2 was offered an empty slate, not the old ids
    assert engine.shown_seen[1] == []
    # and the accumulator holds only the new turn's ids
    assert graph.get_state(cfg).values["shown_movie_ids"] == [7, 8]


def test_second_turn_never_repeats_first_turn_ids():
    """Regression guard (green post-#88, was red before): the orchestrator
    hands the thread's shown ids to the engine, and a refinement turn's
    retrieved set contains none of them."""
    engine = HonoringEngine([
        [_movie(1, "A"), _movie(2, "B"), _movie(3, "C")],
        [_movie(3, "C"), _movie(4, "D"), _movie(5, "E")],
    ])
    router = ScriptedRouter([
        _decision(filters=MetadataFilterCriteria(year_min=2000)),
        _decision(filters=MetadataFilterCriteria(year_min=1990)),
    ])
    graph = _graph(router, engine)
    cfg = _cfg("no-repeat")
    graph.invoke({"messages": [HumanMessage(content="scary movies")]}, cfg)
    out2 = graph.invoke({"messages": [HumanMessage(content="different ones")]}, cfg)

    assert engine.shown_seen[1] == [1, 2, 3]  # handoff: thread -> engine
    ids2 = [m.id for m in out2["retrieved_movies"]]
    assert ids2 == [4, 5]  # id 3 was shown in turn 1 — it must not return


# --- #106: the v2 stack wired beside v1 ---------------------------------------

from src.maya.v2 import MayaV2Router, PreferenceDelta, Understanding


class ScriptedV2(MayaV2Router):
    """Real constructor (no network — chain is bound but never invoked);
    understand() pops scripted (Understanding, notes, usage) results
    (3-tuple since #123)."""

    def __init__(self, results):
        super().__init__(ExperimentConfig(routing_stack="v2"), api_key="test-key")
        self.results = list(results)
        self.calls = []

    def understand(self, query, prefs, shown_titles, last_assistant, probe_count):
        self.calls.append({
            "query": query, "prefs": prefs, "shown_titles": list(shown_titles),
            "last_assistant": last_assistant, "probe_count": probe_count,
        })
        result = self.results.pop(0)
        if len(result) == 2:  # scripted as (u, notes) — metered usage is None
            return result[0], result[1], None
        return result


def _v2u(**kw) -> Understanding:
    defaults = dict(intent=IntentType.SEMANTIC_SEARCH, standalone_query="q", confidence=0.9)
    defaults.update(kw)
    return Understanding(**defaults)


def _v2graph(router, engine):
    return build_maya_graph(
        ExperimentConfig(routing_stack="v2"), router, engine,
        StubSynthesizer(), DualModeObservabilityManager(session_id="adv-106"),
        checkpointer=InMemorySaver(),
    )


def test_v2_ask_turn_answers_inline_and_skips_the_funnel():
    """C8/C9: the model-authored question IS the reply — probe/funnel nodes
    never run; the ask is bounded by turn_decision, not node topology."""
    u = _v2u(ready_to_retrieve=False, missing_slots=["mood"],
             clarifying_question="Solo or family night?")
    router = ScriptedV2([(u, [])])
    engine = RecordingEngine([_movie(1, "A")])
    graph = _v2graph(router, engine)
    cfg = _cfg("v2-ask")
    out = graph.invoke({"messages": [HumanMessage(content="scary movies")]}, cfg)
    assert out["final_response"] == "Solo or family night?"
    assert out["turn_stage"] == "ask" and out["probe_count"] == 1
    assert out["retrieved_movies"] == []
    values = graph.get_state(cfg).values
    assert values.get("funnel_active", False) is False
    assert values.get("offered_genre_options", []) == []


def test_v2_preferences_snapshot_rides_verbatim_including_removals():
    """The resurrect regression, at graph level: a delta removal must survive
    the reducer — a plain update would union the snapshot with stale state."""
    u1 = _v2u(filters=MetadataFilterCriteria(genres=["Comedy", "Drama"]),
              preference_delta=PreferenceDelta(add_genres=["Comedy", "Drama"]))
    u2 = _v2u(ready_to_retrieve=True,
              preference_delta=PreferenceDelta(remove_genres=["Comedy"]))
    router = ScriptedV2([(u1, []), (u2, [])])
    engine = RecordingEngine([_movie(1, "A"), _movie(2, "B")])
    graph = _v2graph(router, engine)
    cfg = _cfg("v2-snapshot")
    graph.invoke({"messages": [HumanMessage(content="comedies and dramas")]}, cfg)
    graph.invoke({"messages": [HumanMessage(content="drop the comedies")]}, cfg)
    prefs = graph.get_state(cfg).values["session_preferences"]
    assert prefs.preferred_genres == ["Drama"]  # union would resurrect Comedy


def test_v2_fresh_start_clears_shown_ids_and_titles():
    """Turn 2 (fresh start, no other axis) asks — reset beats the delta per
    C2, leaving zero axes. Turn 3 then retrieves on a CLEAN slate: the ids
    and titles were wiped, so nothing from turn 1 is excluded or offered."""
    pool = [_movie(1, "Alpha"), _movie(2, "Beta"), _movie(3, "Gamma")]
    router = ScriptedV2([
        (_v2u(ready_to_retrieve=True,
              preference_delta=PreferenceDelta(set_mood="scary")), []),
        (_v2u(reset_context=True), []),
        (_v2u(ready_to_retrieve=True,
              preference_delta=PreferenceDelta(set_mood="funny")), []),
    ])
    engine = PooledEngine([pool, pool, pool])
    graph = _v2graph(router, engine)
    cfg = _cfg("v2-fresh")
    graph.invoke({"messages": [HumanMessage(content="scary movies")]}, cfg)
    out2 = graph.invoke(
        {"messages": [HumanMessage(content="something completely different")]}, cfg
    )
    assert out2["turn_stage"] == "ask"  # clean slate, zero axes: ask (C8)
    graph.invoke({"messages": [HumanMessage(content="funny movies please")]}, cfg)
    values = graph.get_state(cfg).values
    assert values["shown_movie_ids"] == [1, 2, 3]
    assert values["shown_movie_titles"] == ["Alpha", "Beta", "Gamma"]
    assert engine.shown_seen[1] == []  # turn 3 is the 2nd engine call (turn 2 asked)


def test_v2_shown_titles_reach_the_state_block():
    router = ScriptedV2([
        (_v2u(ready_to_retrieve=True,
              preference_delta=PreferenceDelta(set_mood="scary")), []),
        (_v2u(ready_to_retrieve=True,
              preference_delta=PreferenceDelta(set_audience="solo")), []),
    ])
    engine = PooledEngine([[_movie(1, "Alpha")], [_movie(2, "Beta")]])
    graph = _v2graph(router, engine)
    cfg = _cfg("v2-titles")
    graph.invoke({"messages": [HumanMessage(content="space movies")]}, cfg)
    graph.invoke({"messages": [HumanMessage(content="different ones")]}, cfg)
    assert router.calls[1]["shown_titles"] == ["Alpha"]


def test_v2_pivot_goes_to_the_shared_pivot_node():
    u = _v2u(intent=IntentType.OUT_OF_SCOPE)
    router = ScriptedV2([(u, [])])
    engine = RecordingEngine([_movie(1, "A")])
    graph = _v2graph(router, engine)
    out = graph.invoke({"messages": [HumanMessage(content="tell me a joke")]}, _cfg("v2-pivot"))
    assert out["retrieved_movies"] == []
    assert out["final_response"]  # deterministic pivot text, no LLM


def test_v1_stack_ignores_v2_fields():
    """Selector sanity: a v1 router on a v2-flagged config still routes v1 —
    the isinstance check, not the flag alone, picks the node."""
    router = ScriptedRouter([_decision(filters=MetadataFilterCriteria(year_min=2000))])
    engine = RecordingEngine([_movie(1, "A")])
    graph = build_maya_graph(
        ExperimentConfig(routing_stack="v1"), router, engine,
        StubSynthesizer(), DualModeObservabilityManager(session_id="adv-106-v1"),
        checkpointer=InMemorySaver(),
    )
    out = graph.invoke({"messages": [HumanMessage(content="recent movies")]}, _cfg("v1-guard"))
    assert out["routing_decision"] is not None  # v1 decision shape, v1 node ran
