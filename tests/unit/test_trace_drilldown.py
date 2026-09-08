"""Unit tests for the Langfuse-style drill-down (#31) — pure helpers."""

from datetime import UTC, datetime

import pytest

from src.observability.trace_fetch import TraceTree, build_tree
from src.ui.trace_tab import (
    _flatten,
    bar_fraction,
    find_root,
    node_header,
    node_metadata,
)

pytestmark = pytest.mark.unit


def _t(seconds: float) -> datetime:
    return datetime(2026, 9, 7, 12, 0, int(seconds), tzinfo=UTC)


@pytest.fixture
def tree() -> TraceTree:
    roots = build_tree([
        {"id": "root", "name": "LangGraph", "type": "SPAN",
         "start_time": _t(0), "end_time": _t(10)},
        {"id": "gen", "parent_observation_id": "root", "name": "ChatOpenAI",
         "type": "GENERATION", "start_time": _t(1), "end_time": _t(5),
         "model": "gpt-4o", "usage": {"total": 500}, "total_cost": 0.00023},
        {"id": "leaf", "parent_observation_id": "gen", "name": "provider attempt 1",
         "type": "SPAN", "start_time": _t(1), "end_time": _t(2)},
        {"id": "route", "parent_observation_id": "root", "name": "route",
         "type": "SPAN"},
    ])
    return TraceTree(trace_id="t1", roots=roots)


class TestFlatten:
    def test_depth_first_with_depths(self, tree):
        out = [(n.id, d) for n, d in _flatten(tree.roots)]
        assert out == [("root", 0), ("gen", 1), ("leaf", 2), ("route", 1)]


class TestNodeHeader:
    def test_full_fields(self, tree):
        gen = tree.roots[0].children[0]
        header = node_header(gen)
        assert "ChatOpenAI" in header and "GENERATION" in header
        assert "4000 ms" in header and "gpt-4o" in header
        assert "500 tok" in header and "$0.000230" in header

    def test_missing_fields_skipped(self, tree):
        route = tree.roots[0].children[1]
        assert node_header(route) == "route · SPAN"


class TestBarFraction:
    def test_proportional(self, tree):
        root = tree.roots[0]
        assert bar_fraction(root.children[0], root) == pytest.approx(0.4)
        assert bar_fraction(root, root) == 1.0

    def test_missing_duration_is_zero(self, tree):
        root = tree.roots[0]
        assert bar_fraction(root.children[1], root) == 0.0

    def test_zero_duration_root(self, tree):
        route = tree.roots[0].children[1]
        assert bar_fraction(tree.roots[0], route) == 0.0

    def test_clamped_to_one(self, tree):
        root = tree.roots[0]
        assert bar_fraction(root, root.children[0].children[0]) == 1.0


class TestNodeMetadata:
    def test_blanks_dropped_and_usage_flattened(self, tree):
        meta = node_metadata(tree.roots[0].children[0])
        assert meta["usage.total"] == 500
        assert meta["duration_ms"] == 4000
        assert "parent_id" in meta

    def test_untimed_node(self, tree):
        meta = node_metadata(tree.roots[0].children[1])
        assert "duration_ms" not in meta and "model" not in meta


class TestFindRoot:
    def test_finds_containing_root(self, tree):
        assert find_root(tree, "leaf").id == "root"

    def test_unknown_id_falls_back_to_first_root(self, tree):
        assert find_root(tree, "nope").id == "root"

    def test_empty_tree(self):
        assert find_root(TraceTree(trace_id="x", roots=[]), "a") is None
