"""#48 recall (ADR 0009): pure helper filters + the recalled turn-row flag."""

from src.ui.chat_tab import recall_queries
from src.ui.session import MayaSession


def _row(query: str, path: str = "hybrid", intent: str = "RECOMMEND") -> dict:
    return {"query": query, "path": path, "intent": intent}


def test_recall_newest_first_and_deduplicated():
    log = [_row("funny movies"), _row("space operas"), _row("funny movies")]
    assert recall_queries(log) == ["funny movies", "space operas"]


def test_recall_excludes_funnel_turns():
    log = [
        _row("something scary"),
        _row("yes", path="funnel", intent="FUNNEL_CONFIRM"),
        _row("family", path="funnel", intent="RECOMMEND"),
        _row("ok", path="direct", intent="FUNNEL_PROBE"),
    ]
    assert recall_queries(log) == ["something scary"]


def test_recall_caps_at_limit():
    log = [_row(f"query {i}") for i in range(15)]
    assert recall_queries(log, limit=10) == [f"query {i}" for i in range(14, 4, -1)]


def test_recall_skips_empty_queries():
    assert recall_queries([_row(""), {}]) == []


def test_turn_row_carries_no_recalled_flag_by_default():
    row = MayaSession._build_turn_row(
        {"final_response": "hi", "turn_stage": "probe"},
        query="q", trace_id="t", rag_version="v", new_traces=[], prev_tokens=0,
    )
    # The flag is stamped by turn(), not the pure builder — the builder stays
    # graph-output-only (#26-A). Recalled-ness is caller context.
    assert "recalled" not in row
