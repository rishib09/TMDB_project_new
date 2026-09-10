"""Issue #76 feedback inbox: pure guards, command parsing, templates.

Offline: no network, no Streamlit runtime.
"""

import pytest

from src.feedback.inbox import (
    EXCERPT_CHARS,
    REPORT_MAX_CHARS,
    excerpt,
    fence,
    format_rating_comment,
    format_report_comment,
    parse_feedback_action,
    parse_feedback_command,
    parse_inbox_comment,
    validate_report,
)
from src.ui.feedback_tab import feedback_action_label, kind_label


def _row(n: int = 1) -> dict:
    return {
        "trace_id": f"trace-{n}", "rag_version": "v1_1", "intent": "SEMANTIC_SEARCH",
        "path": "single-route", "query": f"query {n}", "response": f"reply {n}",
    }


# --- command -----------------------------------------------------------------


@pytest.mark.parametrize("query,expected", [
    ("/feedback the year filter was wrong", "the year filter was wrong"),
    ("  /FEEDBACK   spaced  ", "spaced"),
    ("/feedback", ""),
    ("/feedbackx", None),
    ("hello /feedback", None),
    ("what movies", None),
])
def test_parse_feedback_command(query, expected):
    assert parse_feedback_command(query) == expected


# --- guards ------------------------------------------------------------------


def test_validate_report_accepts_and_collapses_whitespace():
    assert validate_report("  the  year filter\n\nignored 1999 ") == "the year filter ignored 1999"


@pytest.mark.parametrize("text", [
    "", "too short", "onewordthatislongenough", "1234567890 1234567890",
    "x" * (REPORT_MAX_CHARS + 1),
])
def test_validate_report_rejects_trivial_or_oversize(text):
    assert validate_report(text) is None


def test_fence_neutralises_nested_fences():
    fenced = fence("a ``` b")
    assert fenced.count("```") == 2
    assert fenced.startswith("```text\n") and fenced.endswith("\n```")


def test_excerpt_caps_length():
    assert len(excerpt("w " * 500)) <= EXCERPT_CHARS


# --- templates + round trip ---------------------------------------------------


def test_rating_comment_round_trips_through_parser():
    body = format_rating_comment(-1, _row())
    parsed = parse_inbox_comment({"body": body, "created_at": "2026-09-10T12:00:00Z", "html_url": "u"})
    assert parsed["kind"] == "rating" and parsed["rating"] == "down"
    assert parsed["version"] == "v1_1" and parsed["trace"] == "trace-1"
    assert parsed["date"] == "2026-09-10" and parsed["action"] == "received"


def test_report_comment_carries_window_and_fenced_text():
    window = [_row(n) for n in range(1, 6)]
    body = format_report_comment("the year filter ignored 1999", window)
    parsed = parse_inbox_comment({"body": body, "created_at": "2026-09-10T00:00:00Z"})
    assert parsed["kind"] == "report" and parsed["trace"] == "trace-5"
    assert "last 5 user turns" in body
    for n in range(1, 6):
        assert f"trace `trace-{n}`" in body
    assert "```text\nthe year filter ignored 1999\n```" in body


def test_parse_feedback_action_reads_issue_line():
    assert parse_feedback_action("x\nIssue: #77\n") == ("issue", 77)
    assert parse_feedback_action("Issue: #77 is unrelated prose") == ("received", None)
    assert parse_feedback_action("nothing") == ("received", None)


def test_developer_replies_without_header_are_ignored():
    assert parse_inbox_comment({"body": "Thanks, looking into it."}) is None


# --- view labels -------------------------------------------------------------


def test_action_labels():
    received = {"action": "received", "issue": None}
    promoted = {"action": "issue", "issue": 77}
    assert feedback_action_label(received, None) == "received"
    assert feedback_action_label(promoted, "open") == "issue #77 open"
    assert feedback_action_label(promoted, "closed") == "fixed (#77)"
    assert feedback_action_label(promoted, None) == "issue #77 open"


def test_kind_labels():
    assert kind_label({"kind": "report", "rating": ""}) == "Report"
    assert kind_label({"kind": "rating", "rating": "up"}) == "Rating: thumbs up"
    assert kind_label({"kind": "rating", "rating": "down"}) == "Rating: thumbs down"
