"""Adversarial tests (issue #31): malformed traces must render, never crash."""

import pytest

from src.observability.trace_fetch import build_tree, fetch_trace_tree

pytestmark = pytest.mark.adversarial


class TestMalformedTraces:
    def test_empty_trace_yields_no_roots(self):
        assert build_tree([]) == []

    def test_orphan_parent_becomes_root(self):
        roots = build_tree([
            {"id": "a", "parent_observation_id": "ghost", "name": "orphan"},
        ])
        assert len(roots) == 1
        assert roots[0].name == "orphan"

    def test_self_parent_becomes_root(self):
        roots = build_tree([{"id": "a", "parent_observation_id": "a", "name": "self"}])
        assert len(roots) == 1
        assert roots[0].children == []

    def test_two_node_cycle_breaks_flat(self):
        roots = build_tree([
            {"id": "a", "parent_observation_id": "b"},
            {"id": "b", "parent_observation_id": "a"},
        ])
        # one link survives, the cycle-forming one is demoted to root
        assert len(roots) >= 1
        flat_ids = set()

        def walk(nodes, seen):
            for n in nodes:
                assert n.id not in seen, "cycle survived into the tree"
                seen.add(n.id)
                flat_ids.add(n.id)
                walk(n.children, seen)

        walk(roots, set())
        assert flat_ids == {"a", "b"}

    def test_missing_fields_default_cleanly(self):
        node = build_tree([{"id": "bare"}])[0]
        assert node.type == "SPAN"
        assert node.input is None and node.output is None
        assert node.usage == {} and node.cost is None
        assert node.duration_ms is None

    def test_non_numeric_usage_and_cost_dropped(self):
        node = build_tree([{
            "id": "x",
            "usage": {"total": "NaN-ish", "input": 5},
            "total_cost": "free",
        }])[0]
        assert node.usage == {"input": 5}
        assert node.cost is None


class TestFetchFailOpen:
    def test_empty_trace_id_returns_none(self):
        assert fetch_trace_tree("") is None

    def test_sdk_failure_returns_none(self, monkeypatch):
        import langfuse

        def boom():
            raise RuntimeError("network down")

        monkeypatch.setattr(langfuse, "get_client", boom)
        assert fetch_trace_tree("deadbeef") is None
