"""Live feedback inbox tests (issue #76): real GitHub API.

Read path needs no token (public repo). Write path posts one real comment
on the inbox issue and is skipped without ``GITHUB_TOKEN``.
"""

import os
import time

import pytest

from src.feedback.inbox import (
    fetch_inbox_comments,
    fetch_issue_state,
    format_rating_comment,
    post_inbox_comment,
)

pytestmark = [pytest.mark.live]


def test_public_read_paths_reach_github():
    assert fetch_issue_state(72) == "closed"  # the grill that produced this loop
    assert isinstance(fetch_inbox_comments(), list)


@pytest.mark.skipif(not os.getenv("GITHUB_TOKEN"), reason="GITHUB_TOKEN not configured")
def test_rating_comment_lands_on_inbox():
    row = {"trace_id": "live-test-trace", "rag_version": "live-test", "intent": "GREETING",
           "query": "live test query", "response": "live test reply"}
    url = post_inbox_comment(format_rating_comment(1, row))
    assert url and url.startswith("https://github.com/")
    for _ in range(5):  # GitHub lists lag a few seconds behind a write
        if any(r["trace"] == "live-test-trace" for r in fetch_inbox_comments()):
            return
        time.sleep(3)
    pytest.fail("posted comment never appeared in the inbox listing")
