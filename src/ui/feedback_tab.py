"""Feedback view (issue #76): the loop, at a glance.

Reads the GitHub feedback inbox back through the public API (no token)
and shows one high-level row per Feedback with its Feedback Action:
received, promoted to an issue, or fixed. Cached for five minutes —
the unauthenticated GitHub limit is 60 requests per hour per address.
"""

import pandas as pd
import streamlit as st

from src.feedback.inbox import INBOX_ISSUE, REPO, fetch_inbox_comments, fetch_issue_state

INBOX_URL = f"https://github.com/{REPO}/issues/{INBOX_ISSUE}"
_CACHE_TTL_S = 300


@st.cache_data(ttl=_CACHE_TTL_S, show_spinner=False)
def _load_inbox() -> list[dict]:
    return fetch_inbox_comments()


@st.cache_data(ttl=_CACHE_TTL_S, show_spinner=False)
def _issue_state(number: int) -> str | None:
    return fetch_issue_state(number)


def feedback_action_label(row: dict, state: str | None) -> str:
    """Pure: one visitor-facing phrase per Feedback Action."""
    if row["action"] != "issue":
        return "received"
    if state == "closed":
        return f"fixed (#{row['issue']})"
    return f"issue #{row['issue']} open"


def kind_label(row: dict) -> str:
    if row["kind"] == "report":
        return "Report"
    return "Rating: thumbs up" if row["rating"] == "up" else "Rating: thumbs down"


def render_feedback_view(session=None) -> None:
    st.header("Feedback")
    st.caption(
        "Every thumb and every `/feedback` report in the chat becomes a comment on the "
        f"[feedback inbox]({INBOX_URL}), linked to its trace. Genuine reports are promoted "
        "to issues by hand and closed by the PR that fixes them."
    )
    rows = _load_inbox()
    if not rows:
        st.info("No feedback yet. Rate a reply or type `/feedback <what went wrong>` in Chat.")
        return
    labels = [feedback_action_label(r, _issue_state(r["issue"]) if r["issue"] else None) for r in rows]
    cols = st.columns(4)
    cols[0].metric("Ratings", sum(r["kind"] == "rating" for r in rows))
    cols[1].metric("Reports", sum(r["kind"] == "report" for r in rows))
    cols[2].metric("Issues opened", sum(r["action"] == "issue" for r in rows))
    cols[3].metric("Fixed", sum(label.startswith("fixed") for label in labels))
    frame = pd.DataFrame({
        "date": [r["date"] for r in rows],
        "feedback": [kind_label(r) for r in rows],
        "version": [r["version"] for r in rows],
        "action": labels,
    })
    st.dataframe(frame, use_container_width=True, hide_index=True)
