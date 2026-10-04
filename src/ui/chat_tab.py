"""Chat view (issue #7, #147): one thread, three labeled regions per reply.

The page chrome matches the other views (theme background, ``st.header``,
``st.caption``, bordered containers). Feedback thumbs are captured per
assistant turn and linked to that turn's trace row (Langfuse scoring +
SQLite persistence land with issue #9).
"""

import re

import streamlit as st

from src.feedback.inbox import (
    REPORT_MAX_CHARS,
    REPORT_MIN_CHARS,
    REPORTS_PER_SESSION,
    ReportResult,
    parse_feedback_command,
)
from src.ui.session import MayaSession

MAYA_AVATAR = ":material/movie:"
USER_AVATAR = ":material/person:"

#: FORMAT_RULE card header: ``**Title (Year)**`` at the start of a line.
_CARD_LINE = re.compile(r"^\*\*[^*\n]+\(\d{4}\)\*\*")


def intent_badge_text(log_row: dict) -> str:
    """Pure helper (unit-tested): ONE inline metadata line per turn (#26-F).

    The intent chip, the narrowing trail and the active SQL filters share a
    single caption — the line is the turn's single source of pipeline truth.
    """
    path = log_row.get("path", "?")
    attempts = log_row.get("attempts", 1)
    path_label = f"{path} (x{attempts})" if attempts > 1 else path
    text = (
        f"INTENT: {log_row.get('intent', '?')} — confidence {log_row.get('confidence', 0):.2f} "
        f"— route: {path_label} — {log_row.get('n_movies', 0)} movies "
        f"— {log_row.get('tokens', 0)} tokens"
    )
    narrowing = log_row.get("narrowing") or []
    if narrowing:
        text += " · Narrowing by: " + " · ".join(narrowing)
    filters = log_row.get("filters") or []
    if filters:
        text += " · Filters: " + " · ".join(filters)
    return text


def render_intent_badge(log_row: dict) -> None:
    st.caption(intent_badge_text(log_row))


def conversation_prose(response: str) -> str:
    """Visible reply with FORMAT_RULE movie cards removed (#147).

    A card is a line starting with ``**Title (Year)**`` plus the lines under
    it until the next blank line. Card-free text (greetings, probes) is
    returned unchanged. A reply that is only cards returns "" so the view
    can show Retrieval instead of the text list. The stored response is
    untouched.
    """
    if not response:
        return ""
    kept: list[str] = []
    skipping = False
    for line in response.splitlines():
        if _CARD_LINE.match(line.strip()):
            skipping = True
            continue
        if skipping:
            if not line.strip():
                skipping = False
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def metadata_fields(log_row: dict) -> list[tuple[str, str]]:
    """Labeled turn-record fields (#147). Empty narrowing and filters are omitted."""
    path = log_row.get("path", "?")
    attempts = log_row.get("attempts", 1) or 1
    path_label = f"{path} (x{attempts})" if attempts > 1 else str(path)
    confidence = log_row.get("confidence", 0)
    try:
        confidence_text = f"{float(confidence):.2f}"
    except (TypeError, ValueError):
        confidence_text = str(confidence)
    fields = [
        ("Intent", str(log_row.get("intent", "?"))),
        ("Confidence", confidence_text),
        ("Route", path_label),
        ("Movies", str(log_row.get("n_movies", 0))),
        ("Tokens", str(log_row.get("tokens", 0))),
    ]
    narrowing = log_row.get("narrowing") or []
    if narrowing:
        fields.append(("Narrowing", " · ".join(narrowing)))
    filters = log_row.get("filters") or []
    if filters:
        fields.append(("Filters", " · ".join(filters)))
    return fields


def render_metadata(log_row: dict) -> None:
    for label, value in metadata_fields(log_row):
        st.markdown(f"**{label}** · {value}")


def _widget_rating_to_canonical(value: int) -> int:
    """Pure boundary mapping: st.feedback 1/0 → canonical +1/-1 (unit-tested)."""
    return 1 if value == 1 else -1


def render_feedback(session: MayaSession, turn_index: int) -> None:
    """Thumbs up/down per assistant turn; persisted + pushed on change (#9).

    st.feedback('thumbs') yields 1 = up, 0 = down, None = unset; stored
    canonically as ±1 at the UI boundary (0 → -1).
    """
    value = st.feedback("thumbs", key=f"feedback_{turn_index}")
    if value is None:
        return
    rating = _widget_rating_to_canonical(value)
    if rating != session.feedback_log.get(turn_index):
        session.record_feedback(turn_index, rating)


def render_report_receipt(session: MayaSession, row: dict) -> None:
    """Persistent Feedback Receipt under a reported reply (#76 amendment).

    No-op when no Report exists for this reply. Links the inbox comment when
    GitHub accepted it; the Langfuse-only fallback (D12) says so without a link.
    """
    if row["trace_id"] not in session.report_receipts:
        return
    url = session.report_receipts[row["trace_id"]]
    where = f"[inbox comment]({url})" if url else "kept in telemetry"
    st.caption(f"Feedback recorded for this reply ({where}). See the Feedback page in the sidebar.")


def render_poster_grid(movies, cols: int = 4) -> None:
    """Poster cards for one turn's retrieval. The section title is the caller's."""
    if not movies:
        return
    for start in range(0, len(movies), cols):
        chunk = movies[start : start + cols]
        for col, movie in zip(st.columns(cols), chunk):
            with col, st.container(border=True):
                st.image(movie.poster_url)
                st.markdown(f"**{movie.title}** ({movie.release_year})")
                st.caption(f"{movie.vote_average:.1f} / 10 — " + ", ".join(movie.genres[:3]))


def recall_queries(turn_log: list[dict], limit: int = 10) -> list[str]:
    """Pure helper (#48): recallable queries, newest first, deduplicated.

    Funnel-owned turns are excluded — their queries ("yes", "family") only
    made sense mid-funnel and would route to nonsense replayed cold.
    """
    seen: set[str] = set()
    out: list[str] = []
    for row in reversed(turn_log):
        query = row.get("query", "")
        if not query or row.get("path") == "funnel":
            continue
        if str(row.get("intent", "")).startswith("FUNNEL_"):
            continue
        if query in seen:
            continue
        seen.add(query)
        out.append(query)
        if len(out) == limit:
            break
    return out


def render_recall(session: MayaSession) -> str | None:
    """Recent-queries popover (#48, ADR 0009). Not mounted (#147).

    Returns a query to replay this rerun, or None. Recall replays INPUT only:
    the returned query becomes an ordinary turn under current state.
    """
    queries = recall_queries(session.turn_log)
    if not queries:
        return None
    picked: str | None = None
    with st.popover(":material/history: Recent queries"):
        for i, query in enumerate(queries):
            cols = st.columns([5, 1, 1])
            cols[0].markdown(query)
            if cols[1].button("Resend", key=f"recall_send_{i}"):
                picked = query
            if cols[2].button("Edit", key=f"recall_edit_{i}"):
                st.session_state["recall_draft"] = query
        draft = st.session_state.get("recall_draft")
        if draft:
            edited = st.text_area("Edit and resend", value=draft, key=f"recall_area_{hash(draft)}")
            if st.button("Send", key="recall_send_edited") and edited.strip():
                st.session_state.pop("recall_draft", None)
                picked = edited.strip()
    return picked


def resolve_turn_row(session: MayaSession, ui_index: int) -> dict | None:
    """Maps the UI's assistant-message counter to its turn row (#26-K).

    Identity join through the stamped ``turn_ref`` — the message window
    slides while turn_log grows, so raw index arithmetic silently points at
    the wrong row past 5 assistant turns (the badge/feedback misalignment
    from the walkthrough). Falls back to the raw index for legacy rows.
    """
    assistants = [m for m in session.conversation.messages if m.role == "assistant"]
    if 0 <= ui_index < len(assistants):
        ref = assistants[ui_index].turn_ref
        if ref is not None and 0 <= ref < len(session.turn_log):
            return session.turn_log[ref]
    return None


_AUTO_SCROLL_JS = """
<script>
(function() {
  const d = window.parent.document;
  const el = d.querySelector('section.stMain')
          || d.querySelector('section.main')
          || d.querySelector('[data-testid="stAppViewContainer"]');
  if (el) { el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' }); }
})();
</script>
"""


def scroll_to_newest() -> None:
    """Scrolls the conversation to the newest message (#27-R, zero height).

    Streamlit resets the viewport to the top on every rerun, so the fresh
    response renders below the fold after every turn. The JS runs in a
    zero-height same-origin component and scrolls the app's main container.
    Called ONLY on the fresh-turn path — thumb-click reruns keep the user's
    scroll position. st.html replaces the retired components-v1 html embed
    (#119); unsafe_allow_javascript=True is required — the default (False)
    would silently strip the script and kill the auto-scroll.
    """
    st.html(
        f"<div style='height:0px'></div>{_AUTO_SCROLL_JS}",
        unsafe_allow_javascript=True,
    )


def render_assistant_turn(
    session: MayaSession, content: str, row: dict | None, turn_index: int
) -> None:
    """Conversation, Retrieval, and Metadata inside one assistant message (#147).

    Uses bordered containers so the regions pick up the same theme as the
    other views. Movie-card markdown stays in the stored reply and is omitted
    here. Posters come from this turn's row, not from a list under the thread.
    """
    movies = list((row or {}).get("movies") or [])
    prose = conversation_prose(content)
    if prose or not movies:
        with st.container(border=True):
            st.markdown("**Conversation**")
            st.markdown(prose or content)
    if movies:
        with st.container(border=True):
            st.markdown("**Retrieval**")
            st.caption(f"{len(movies)} movies in Maya's closed world this turn")
            render_poster_grid(movies)
    if row is not None:
        with st.container(border=True):
            st.markdown("**Metadata**")
            render_metadata(row)
            render_feedback(session, turn_index)
            render_report_receipt(session, row)
    else:
        render_feedback(session, turn_index)


def render_chat(session: MayaSession) -> None:
    st.header("Maya")
    st.caption(
        "Conversational film curator for US theatrical releases, 1970–2026 — "
        "deterministic routing, closed-world grounding, full trace observability."
    )

    messages = session.conversation.messages
    turn_index = -1
    for i, msg in enumerate(messages):
        role = "user" if msg.role == "user" else "assistant"
        avatar = USER_AVATAR if msg.role == "user" else MAYA_AVATAR
        with st.chat_message(role, avatar=avatar):
            if msg.role != "assistant":
                st.markdown(msg.content)
            else:
                turn_index += 1
                row = resolve_turn_row(session, turn_index)  # #26-K identity join
                render_assistant_turn(session, msg.content, row, turn_index)
        if msg.role == "user":
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            if nxt is not None and nxt.role == "assistant":
                st.space("medium")  # #147: air between the visitor and the reply
        elif i + 1 < len(messages):
            st.space("small")

    # #147: recent-query recall is not mounted. The current thread stays.
    query = st.chat_input("Ask Maya about movies")
    if not query:
        return
    report = parse_feedback_command(query)
    if report is not None:  # #76: Report on the last reply, never a turn
        result = session.record_report(report)
        if result == ReportResult.RECORDED:
            st.rerun()  # the Feedback Receipt renders under the reported reply
        elif result == ReportResult.UNDELIVERED:
            st.toast("Feedback could not be saved right now. Please try again in a moment.")
        else:
            st.toast(
                f"Usage: /feedback <what went wrong>, {REPORT_MIN_CHARS}–{REPORT_MAX_CHARS} "
                f"characters, at most {REPORTS_PER_SESSION} per session."
            )
        return

    if messages:
        st.space("small")
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(query)
    st.space("medium")
    assistant = st.chat_message("assistant", avatar=MAYA_AVATAR)
    try:
        with assistant, st.status("Working through the pipeline", expanded=False):
            session.turn(query)
    except Exception as exc:  # noqa: BLE001 — surface a readable failure, never a traceback
        st.error(
            "Maya could not complete this turn. Check that the app was started with "
            "the encrypted environment loaded:  \n"
            "`npx @dotenvx/dotenvx run -- streamlit run app.py`  \n"
            f"Details: {type(exc).__name__}: {exc}"
        )
        return
    idx = len(session.turn_log) - 1
    with assistant:
        last = resolve_turn_row(session, idx) or session.turn_log[idx]
        render_assistant_turn(session, last.get("response", ""), last, idx)
    scroll_to_newest()  # #27-R: land on the fresh response, not the page top
