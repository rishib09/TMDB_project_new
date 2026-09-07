"""Unit tests for the fetch-back trace tree (issue #31) — pure logic only."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.observability.trace_fetch import TraceTree, build_tree
from src.ui.trace_tab import _cloud_waterfall_frame, _flatten, _turn_options

pytestmark = pytest.mark.unit


def _obs(id, parent=None, name="", type="SPAN", start=None, end=None, **extra):
    return {
        "id": id,
        "parent_observation_id": parent,
        "name": name,
        "type": type,
        "start_time": start,
        "end_time": end,
        **extra,
    }


def _t(seconds: float) -> datetime:
    return datetime(2026, 9, 7, 12, 0, int(seconds), int((seconds % 1) * 1e6), tzinfo=UTC)


class TestBuildTree:
    def test_nests_children_under_parent(self):
        roots = build_tree([
            _obs("root", name="LangGraph"),
            _obs("route", parent="root", name="route"),
            _obs("gen", parent="route", name="router-llm", type="GENERATION"),
        ])
        assert len(roots) == 1
        assert roots[0].children[0].name == "route"
        assert roots[0].children[0].children[0].type == "GENERATION"

    def test_siblings_sorted_by_start_time(self):
        roots = build_tree([
            _obs("root"),
            _obs("b", parent="root", name="second", start=_t(2)),
            _obs("a", parent="root", name="first", start=_t(1)),
            _obs("c", parent="root", name="untimed"),  # None start sorts last
        ])
        assert [c.name for c in roots[0].children] == ["first", "second", "untimed"]

    def test_duration_from_real_timestamps(self):
        roots = build_tree([_obs("x", start=_t(1), end=_t(1.25))])
        assert roots[0].duration_ms == pytest.approx(250.0)

    def test_json_string_io_is_decoded(self):
        # Live-verified defect (#31): the API returns io as JSON strings.
        node = build_tree([_obs(
            "g", type="GENERATION",
            input='[{"role": "system", "content": "You are Maya."}]',
            output='"pong"',
        )])[0]
        assert node.input == [{"role": "system", "content": "You are Maya."}]
        assert node.output == "pong"

    def test_plain_string_output_passes_through(self):
        node = build_tree([_obs("g", output="not json {")])[0]
        assert node.output == "not json {"

    def test_normalizes_sdk_model_objects(self):
        raw = SimpleNamespace(
            id="g1", parent_observation_id=None, name="synth", type="GENERATION",
            start_time=_t(0), end_time=_t(1),
            input=[{"role": "system", "content": "You are Maya."}],
            output={"content": "Here are three films."},
            model="llama-3.3-70b",
            usage_details={"input": 900, "output": 120, "total": 1020},
            total_cost=0.0004,
        )
        node = build_tree([raw])[0]
        assert node.model == "llama-3.3-70b"
        assert node.usage["total"] == 1020
        assert node.cost == pytest.approx(0.0004)
        assert node.input[0]["role"] == "system"


class TestTraceTabHelpers:
    def test_turn_options_newest_first_with_labels(self):
        session = SimpleNamespace(turn_log=[
            {"trace_id": "t1", "query": "sci-fi from the 90s"},
            {"trace_id": "", "query": "no trace id row"},
            {"trace_id": "t3", "query": "x" * 100},
        ])
        options = _turn_options(session)
        assert [tid for tid, _ in options] == ["t3", "t1"]
        assert options[0][1].startswith("Turn 3: ")
        assert len(options[0][1]) <= len("Turn 3: ") + 60

    def test_waterfall_frame_uses_real_spans(self):
        tree = TraceTree(trace_id="t", roots=build_tree([
            _obs("a", name="route", start=_t(0), end=_t(0.5)),
            _obs("b", name="retrieve", start=_t(0.5), end=_t(1.5)),
            _obs("c", name="untimed"),  # no start -> excluded
        ]))
        frame = _cloud_waterfall_frame(tree)
        assert list(frame["node"]) == ["route", "retrieve"]
        assert (frame["finish"] >= frame["start"]).all()

    def test_flatten_is_depth_first_with_depths(self):
        tree = build_tree([
            _obs("root", name="r", start=_t(0)),
            _obs("child", parent="root", name="c", start=_t(1)),
        ])
        flat = list(_flatten(tree))
        assert [(n.name, d) for n, d in flat] == [("r", 0), ("c", 1)]
