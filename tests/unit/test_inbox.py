"""Issue #76 feedback inbox: pure guards, command parsing, templates.

Offline: no network, no Streamlit runtime.
"""

import pytest

from src.feedback.inbox import (
    EXCERPT_CHARS,
    REPO,
    REPORT_MAX_CHARS,
    excerpt,
    fence,
    format_rating_comment,
    format_report_comment,
    parse_feedback_action,
    parse_feedback_command,
    parse_inbox_comment,
    report_text,
    validate_report,
)
from src.ui.feedback_tab import (
    VIEW_EXCERPT_CHARS,
    action_cell,
    feedback_action_label,
    feedback_cell,
    issue_url,
    kind_label,
)


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
    assert kind_label({"kind": "report", "rating": "", "text": "the year filter ignored 1999"}) == (
        "Report: the year filter ignored 1999"
    )
    assert kind_label({"kind": "rating", "rating": "up", "text": ""}) == "Rating: thumbs up"
    assert kind_label({"kind": "rating", "rating": "down", "text": ""}) == "Rating: thumbs down"


def test_report_label_caps_visitor_text():
    long = "word " * 60
    label = kind_label({"kind": "report", "rating": "", "text": long})
    assert label.startswith("Report: word word")
    assert len(label) <= len("Report: ") + VIEW_EXCERPT_CHARS


# --- report text + links ------------------------------------------------------


def test_report_text_is_first_fence_only():
    window = [_row(n) for n in range(1, 3)]
    body = format_report_comment("the year filter ignored 1999", window)
    assert report_text(body) == "the year filter ignored 1999"
    assert report_text(format_rating_comment(1, _row())) == "Q: query 1\nA: reply 1"
    assert report_text("Thanks, looking into it.") == ""


def test_parsed_report_row_carries_text_and_url():
    body = format_report_comment("the year filter ignored 1999", [_row()])
    parsed = parse_inbox_comment({"body": body, "created_at": "2026-09-13T00:00:00Z", "html_url": "u"})
    assert parsed["text"] == "the year filter ignored 1999" and parsed["url"] == "u"
    rating = parse_inbox_comment({"body": format_rating_comment(1, _row()), "html_url": "r"})
    assert rating["text"] == ""


def test_issue_url():
    assert issue_url({"action": "issue", "issue": 77}) == f"https://github.com/{REPO}/issues/77"
    assert issue_url({"action": "received", "issue": None}) == ""


# --- table cells: the label IS the link ----------------------------------------


def test_feedback_cell_links_label_to_inbox_comment():
    row = {"kind": "report", "rating": "", "text": "the year filter ignored 1999", "url": "https://g/c1"}
    assert feedback_cell(row) == "[Report: the year filter ignored 1999](https://g/c1)"
    rating = {"kind": "rating", "rating": "down", "text": "", "url": "https://g/c2"}
    assert feedback_cell(rating) == "[Rating: thumbs down](https://g/c2)"
    assert feedback_cell({**row, "url": ""}) == "Report: the year filter ignored 1999"


def test_feedback_cell_neutralises_markdown_in_visitor_text():
    hostile = {"kind": "report", "rating": "", "url": "https://g/c1",
               "text": "a] (http://evil) | b [c"}
    cell = feedback_cell(hostile)
    assert cell.endswith("](https://g/c1)")
    assert "http://evil" not in cell.split("](")[-1]  # visitor text cannot become the href
    assert cell.count("|") == 0  # cannot break the table row


def test_action_cell_links_only_when_promoted():
    promoted = {"action": "issue", "issue": 77}
    assert action_cell(promoted, "issue #77 open") == (
        f"[issue #77 open](https://github.com/{REPO}/issues/77)"
    )
    assert action_cell({"action": "received", "issue": None}, "received") == "received"
