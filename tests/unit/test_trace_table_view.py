"""Unit tests for the Langfuse-style observations table (#35) — pure helpers."""

from datetime import UTC, datetime

import pytest

from src.observability.trace_fetch import TraceTree, build_tree
from src.ui.trace_tab import _preview, apply_filters, name_counts, observation_rows

pytestmark = pytest.mark.unit


def _t(seconds: float) -> datetime:
    return datetime(2026, 9, 7, 12, 0, int(seconds), tzinfo=UTC)


def _tree(trace_id: str, observations: list[dict]) -> TraceTree:
    return TraceTree(trace_id=trace_id, roots=build_tree(observations))


@pytest.fixture
def rows():
    trees = {
        "t1": _tree("t1", [
            {"id": "a", "name": "LangGraph", "type": "SPAN",
             "start_time": _t(0), "end_time": _t(2)},
            {"id": "b", "parent_observation_id": "a", "name": "ChatOpenAI",
             "type": "GENERATION", "start_time": _t(1), "end_time": _t(2),
             "input": [{"role": "system", "content": "You are Maya."}],
             "output": "here are films", "usage": {"total": 500}},
        ]),
        "t2": _tree("t2", [
            {"id": "c", "name": "route", "type": "SPAN", "start_time": _t(5)},
        ]),
    }
    labels = {"t1": "Turn 1: hello", "t2": "Turn 2: funny"}
    return observation_rows(trees, labels)


class TestObservationRows:
    def test_newest_first_with_turn_labels(self, rows):
        assert [r["name"] for r in rows] == ["route", "ChatOpenAI", "LangGraph"]
        assert rows[0]["turn"] == "Turn 2: funny"
        assert rows[1]["turn"] == "Turn 1: hello"

    def test_row_fields(self, rows):
        gen = next(r for r in rows if r["type"] == "GENERATION")
        assert gen["duration_ms"] == 1000
        assert gen["tokens"] == 500
        assert "maya" in gen["search_blob"]
        assert gen["input"].startswith('[{"role"')


class TestFilters:
    def test_type_filter(self, rows):
        assert all(r["type"] == "GENERATION"
                   for r in apply_filters(rows, "GENERATION", [], ""))
        assert len(apply_filters(rows, "All", [], "")) == 3

    def test_name_filter(self, rows):
        out = apply_filters(rows, "All", ["route"], "")
        assert [r["name"] for r in out] == ["route"]

    def test_text_search_hits_full_io_not_just_preview(self, rows):
        assert [r["name"] for r in apply_filters(rows, "All", [], "MAYA")] == ["ChatOpenAI"]
        assert apply_filters(rows, "All", [], "no-such-text") == []


class TestNameCounts:
    def test_counts_sorted_desc(self, rows):
        counts = name_counts(rows + rows)
        assert counts["ChatOpenAI"] == 2
        assert list(counts.values()) == sorted(counts.values(), reverse=True)


class TestPreview:
    def test_truncates_and_collapses_whitespace(self):
        text = "word " * 50
        out = _preview(text, limit=20)
        assert len(out) == 20 and out.endswith("…")
        assert "  " not in out

    def test_empty_payloads(self):
        assert _preview(None) == ""
        assert _preview([]) == ""

    def test_json_payload(self):
        assert _preview({"a": 1}) == '{"a": 1}'
