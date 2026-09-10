"""GitHub feedback inbox (issue #76, verdict on #72).

System of record for visitor Feedback: one always-open GitHub issue (the
inbox) receives one comment per Rating (thumb) or Report (``/feedback``
text). SQLite on the Space is ephemeral, so GitHub holds the durable copy
and the Feedback view reads it back through the public API.

Guards are abuse limits, not Experiment Config knobs (ADR 0004 covers
behaviour tunables; these bound what a stranger can write to a public
repo). No LLM touches this path: every comment is a template. All network
calls fail open — a missing or revoked token never breaks the chat.
"""

import logging
import os
import re

import httpx

logger = logging.getLogger(__name__)

REPO = "rishib09/TMDB_project_new"
INBOX_ISSUE = 75
API = "https://api.github.com"

FEEDBACK_COMMAND = "/feedback"
REPORT_MIN_CHARS = 15
REPORT_MIN_WORDS = 2
REPORT_MAX_CHARS = 500
REPORTS_PER_SESSION = 3
EXCERPT_CHARS = 200
WINDOW_TURNS = 5
TIMEOUT_S = 10.0

_HEADER_RE = re.compile(r"<!--\s*feedback\s+(.*?)\s*-->")
_ISSUE_LINE_RE = re.compile(r"^Issue:\s*#(\d+)\s*$", re.MULTILINE)


# --- command + guards (pure) -------------------------------------------------


def parse_feedback_command(query: str) -> str | None:
    """Text after ``/feedback``, or None when the query is not the command."""
    stripped = query.strip()
    head = stripped[: len(FEEDBACK_COMMAND)].lower()
    rest = stripped[len(FEEDBACK_COMMAND):]
    if head != FEEDBACK_COMMAND or (rest and not rest[0].isspace()):
        return None
    return rest.strip()


def validate_report(text: str) -> str | None:
    """Whitespace-collapsed report text, or None when it fails a guard.

    Trivial input (too short, one word, no letters) logs nothing — the
    grill's D9. Oversize input is rejected rather than truncated so a
    visitor never sees a silently cut report.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) < REPORT_MIN_CHARS or len(cleaned) > REPORT_MAX_CHARS:
        return None
    if len(cleaned.split()) < REPORT_MIN_WORDS:
        return None
    if not any(ch.isalpha() for ch in cleaned):
        return None
    return cleaned


def fence(text: str) -> str:
    """Visitor text as a plain code fence: no Markdown, links, or @mentions render."""
    safe = text.replace("```", "` ` `")
    return f"```text\n{safe}\n```"


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# --- comment templates (pure) --------------------------------------------------


def _header(**fields: str) -> str:
    body = " ".join(f"{k}={v}" for k, v in fields.items() if v)
    return f"<!-- feedback {body} -->"


def format_rating_comment(rating: int, row: dict) -> str:
    """One Rating comment from a turn_log row (thumb + version + intent + trace)."""
    thumb = "up" if rating == 1 else "down"
    return "\n".join([
        _header(kind="rating", rating=thumb, version=row.get("rag_version", ""),
                intent=row.get("intent", ""), trace=row.get("trace_id", "")),
        f"**Rating: thumbs {thumb}** · version `{row.get('rag_version', '')}` "
        f"· intent `{row.get('intent', '')}` · trace `{row.get('trace_id', '')}`",
        "",
        fence(f"Q: {excerpt(row.get('query', ''))}\nA: {excerpt(row.get('response', ''))}"),
    ])


def format_report_comment(text: str, window: list[dict]) -> str:
    """One Report comment: fenced visitor text plus the Feedback Window."""
    last = window[-1] if window else {}
    lines = [
        _header(kind="report", version=last.get("rag_version", ""),
                intent=last.get("intent", ""), trace=last.get("trace_id", "")),
        f"**Report** · version `{last.get('rag_version', '')}` "
        f"· trace `{last.get('trace_id', '')}`",
        "",
        fence(text),
        "",
        f"**Feedback Window** (last {len(window)} user turns, oldest first)",
    ]
    for n, row in enumerate(window, 1):
        lines.append(
            f"{n}. `{row.get('intent', '')}` via {row.get('path', '')} "
            f"· trace `{row.get('trace_id', '')}`"
        )
        lines.append(fence(
            f"Q: {excerpt(row.get('query', ''))}\nA: {excerpt(row.get('response', ''))}"
        ))
    return "\n".join(lines)


def parse_inbox_comment(comment: dict) -> dict | None:
    """Machine header of one GitHub comment → flat row for the Feedback view.

    Returns None for comments without a feedback header (developer replies).
    """
    body = comment.get("body") or ""
    match = _HEADER_RE.search(body)
    if not match:
        return None
    fields = dict(pair.split("=", 1) for pair in match.group(1).split() if "=" in pair)
    action, issue = parse_feedback_action(body)
    return {
        "date": (comment.get("created_at") or "")[:10],
        "kind": fields.get("kind", ""),
        "rating": fields.get("rating", ""),
        "version": fields.get("version", ""),
        "intent": fields.get("intent", ""),
        "trace": fields.get("trace", ""),
        "action": action,
        "issue": issue,
        "url": comment.get("html_url", ""),
    }


def parse_feedback_action(comment_body: str) -> tuple[str, int | None]:
    """("issue", N) when the developer appended ``Issue: #N``; else ("received", None)."""
    match = _ISSUE_LINE_RE.search(comment_body)
    return ("issue", int(match.group(1))) if match else ("received", None)


# --- GitHub I/O (fail-open) ----------------------------------------------------


def _headers(token: str | None = None) -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def post_inbox_comment(body: str, *, token: str | None = None) -> str | None:
    """Appends one comment to the inbox issue; comment URL, or None on any failure."""
    token = token or os.getenv("GITHUB_TOKEN")
    if not token:
        logger.info("GITHUB_TOKEN absent — feedback kept in Langfuse only")
        return None
    try:
        resp = httpx.post(
            f"{API}/repos/{REPO}/issues/{INBOX_ISSUE}/comments",
            json={"body": body}, headers=_headers(token), timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        return resp.json().get("html_url")
    except Exception:  # noqa: BLE001 — telemetry must never break a turn
        logger.exception("Failed to post feedback comment to GitHub")
        return None


def fetch_inbox_comments() -> list[dict]:
    """Parsed feedback rows from the inbox, newest first; [] on any failure."""
    try:
        resp = httpx.get(
            f"{API}/repos/{REPO}/issues/{INBOX_ISSUE}/comments",
            params={"per_page": 100}, headers=_headers(), timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        rows = [parse_inbox_comment(c) for c in resp.json()]
        return [r for r in reversed(rows) if r]
    except Exception:  # noqa: BLE001
        logger.exception("Failed to fetch feedback inbox")
        return []


def fetch_issue_state(number: int) -> str | None:
    """'open' | 'closed' for a promoted issue; None on any failure."""
    try:
        resp = httpx.get(
            f"{API}/repos/{REPO}/issues/{number}", headers=_headers(), timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        return resp.json().get("state")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to fetch issue %s", number)
        return None
