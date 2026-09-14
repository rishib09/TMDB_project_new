"""Issue #76 feedback inbox: failure isolation, injection, caps.

Every network path fails open (D12); visitor text never renders (D11);
the per-tab cap and the guards log nothing when tripped (D9).
"""

import httpx
import pytest

from src.feedback import inbox
from src.feedback.inbox import (
    REPORTS_PER_SESSION,
    ReportResult,
    fetch_inbox_comments,
    fetch_issue_state,
    format_rating_comment,
    format_report_comment,
    post_inbox_comment,
)
from src.feedback.store import FeedbackStore
from src.ui.session import MayaSession


def _session(turns: int = 6) -> MayaSession:
    session = MayaSession.__new__(MayaSession)
    session.feedback_store = FeedbackStore(":memory:")
    session.feedback_log = {}
    session.report_count = 0
    session.report_receipts = {}
    session.turn_log = [
        {"trace_id": f"t{n}", "rag_version": "v", "intent": "SEMANTIC_SEARCH",
         "path": "single-route", "query": f"q{n}", "response": f"a{n}"}
        for n in range(turns)
    ]
    return session


# --- fail-open ---------------------------------------------------------------


def test_post_without_token_returns_none_without_network(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def explode(*a, **k):
        raise AssertionError("network must not be touched without a token")

    monkeypatch.setattr(httpx, "post", explode)
    assert post_inbox_comment("body") is None


def test_post_network_failure_returns_none(monkeypatch):
    def explode(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", explode)
    assert post_inbox_comment("body", token="tok") is None


def test_post_http_error_returns_none(monkeypatch):
    class Resp:
        def raise_for_status(self):
            raise httpx.HTTPStatusError("401", request=None, response=None)

    monkeypatch.setattr(httpx, "post", lambda *a, **k: Resp())
    assert post_inbox_comment("body", token="revoked") is None


def test_fetch_failures_return_empty(monkeypatch):
    def explode(*a, **k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", explode)
    assert fetch_inbox_comments() == []
    assert fetch_issue_state(77) is None


def test_inbox_read_follows_pagination_newest_first(monkeypatch):
    """GitHub serves issue comments oldest first, 100 per page (review R2).

    A one-page read returned the oldest 100 and dropped the newest; the
    read must follow ``Link: rel="next"`` until the inbox is exhausted.
    """
    def comment(n: int) -> dict:
        row = {"trace_id": f"t{n}", "rag_version": "v", "intent": "GREETING",
               "query": f"q{n}", "response": f"a{n}"}
        return {"body": format_rating_comment(1, row), "created_at": "2026-09-13T00:00:00Z",
                "html_url": f"u{n}"}

    base = "https://api.github.com/repos/rishib09/TMDB_project_new/issues/75/comments"
    calls = []

    class Page:
        def __init__(self, items, next_url):
            self._items, self.links = items, ({"next": {"url": next_url}} if next_url else {})

        def raise_for_status(self):
            pass

        def json(self):
            return self._items

    def fake_get(url, *a, **k):
        calls.append(url)
        if "page=2" in url:
            return Page([comment(n) for n in range(101, 106)], None)
        return Page([comment(n) for n in range(1, 101)], f"{base}?per_page=100&page=2")

    monkeypatch.setattr(httpx, "get", fake_get)
    rows = fetch_inbox_comments()
    assert len(calls) == 2
    assert len(rows) == 105
    assert rows[0]["trace"] == "t105" and rows[-1]["trace"] == "t1"


def test_report_undelivered_when_both_backends_fail(monkeypatch):
    """Neither GitHub nor Langfuse accepted the Report (review R3).

    The old code returned True here, so the visitor saw "Feedback recorded"
    while nothing was stored anywhere. The attempt must not consume the cap.
    """
    import src.ui.session as session_mod

    monkeypatch.setattr(session_mod, "post_inbox_comment", lambda body: None)
    monkeypatch.setattr(session_mod, "push_report_comment", lambda tid, text: False)
    session = _session()
    assert session.record_report("the year filter ignored 1999") == ReportResult.UNDELIVERED
    assert session.report_count == 0
    assert session.report_receipts == {}


# --- Feedback Receipt (#76 amendment) --------------------------------------------


def test_receipt_keyed_by_reported_reply_with_inbox_url(monkeypatch):
    """A RECORDED Report leaves a receipt on the reply it was about (D9: the last one)."""
    import src.ui.session as session_mod

    monkeypatch.setattr(session_mod, "post_inbox_comment", lambda body: "https://github.com/x/issues/75#c1")
    session = _session()
    assert session.record_report("the year filter ignored 1999") == ReportResult.RECORDED
    assert session.report_receipts == {"t5": "https://github.com/x/issues/75#c1"}


def test_receipt_without_url_on_langfuse_fallback(monkeypatch):
    import src.ui.session as session_mod

    monkeypatch.setattr(session_mod, "post_inbox_comment", lambda body: None)
    monkeypatch.setattr(session_mod, "push_report_comment", lambda tid, text: True)
    session = _session()
    assert session.record_report("the year filter ignored 1999") == ReportResult.RECORDED
    assert session.report_receipts == {"t5": None}


def test_rejected_report_leaves_no_receipt(monkeypatch):
    import src.ui.session as session_mod

    monkeypatch.setattr(session_mod, "post_inbox_comment", lambda body: pytest.fail("posted"))
    session = _session()
    assert session.record_report("bad") == ReportResult.REJECTED
    assert session.report_receipts == {}


def test_report_falls_back_to_langfuse_when_inbox_unreachable(monkeypatch):
    posted, fallback = [], []
    monkeypatch.setattr(inbox, "post_inbox_comment", lambda body: posted.append(body) or None)
    import src.ui.session as session_mod

    monkeypatch.setattr(session_mod, "post_inbox_comment", lambda body: posted.append(body) or None)
    monkeypatch.setattr(session_mod, "push_report_comment", lambda tid, text: fallback.append((tid, text)) or True)
    session = _session()
    assert session.record_report("the year filter ignored 1999") == ReportResult.RECORDED
    assert len(posted) == 1
    assert fallback == [("t5", "the year filter ignored 1999")]


# --- guards + cap ------------------------------------------------------------


def test_trivial_report_logs_nothing(monkeypatch):
    import src.ui.session as session_mod

    calls = []
    monkeypatch.setattr(session_mod, "post_inbox_comment", lambda body: calls.append(body) or "u")
    monkeypatch.setattr(session_mod, "push_report_comment", lambda *a: calls.append(a) or True)
    session = _session()
    assert session.record_report("bad") == ReportResult.REJECTED
    assert session.record_report("") == ReportResult.REJECTED
    assert calls == [] and session.report_count == 0


def test_report_without_turns_logs_nothing(monkeypatch):
    import src.ui.session as session_mod

    monkeypatch.setattr(session_mod, "post_inbox_comment", lambda body: pytest.fail("posted"))
    session = _session(turns=0)
    assert session.record_report("the year filter ignored 1999") == ReportResult.REJECTED


def test_per_tab_cap_blocks_fourth_report(monkeypatch):
    import src.ui.session as session_mod

    posted = []
    monkeypatch.setattr(session_mod, "post_inbox_comment", lambda body: posted.append(body) or "u")
    session = _session()
    for _ in range(REPORTS_PER_SESSION):
        assert session.record_report("the year filter ignored 1999") == ReportResult.RECORDED
    assert session.record_report("the year filter ignored 1999 again") == ReportResult.REJECTED
    assert len(posted) == REPORTS_PER_SESSION


def test_window_is_last_five_turns(monkeypatch):
    import src.ui.session as session_mod

    posted = []
    monkeypatch.setattr(session_mod, "post_inbox_comment", lambda body: posted.append(body) or "u")
    session = _session(turns=8)
    session.record_report("the year filter ignored 1999")
    body = posted[0]
    assert "trace `t2`" not in body
    for n in range(3, 8):
        assert f"trace `t{n}`" in body


# --- injection ---------------------------------------------------------------


def test_visitor_markdown_never_renders():
    """Mentions, links, headings, and fences from visitor text stay inside a code fence."""
    hostile = "@rishib09 [click](http://evil.example) # heading ``` ```python\nimport os"
    row = {"trace_id": "t", "rag_version": "v", "intent": "I", "path": "p",
           "query": hostile, "response": hostile}
    body = format_report_comment(hostile + " enough words here", [row])
    inside = False
    for line in body.splitlines():
        if line.startswith("```"):
            inside = not inside
            continue
        if not inside:
            assert "@rishib09" not in line and "http://evil" not in line and "# heading" not in line
    assert not inside, "unbalanced fence"
    assert "```python" not in body
