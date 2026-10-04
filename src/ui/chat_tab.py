"""Chat view (issue #7, #147): prototype A, in the app's white / red / grey.

Each turn is the stacked layout from the chat prototype: a right-aligned
visitor line, then one panel whose Conversation, Retrieval, and Metadata
regions stack. Colors are the theme hexes in ``.streamlit/config.toml``:
``#FFFFFF``, ``#D7263D``, ``#F3F4F6``, with ``#1B1B24`` text.
Feedback thumbs stay per assistant turn (issue #9).
"""

import html
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

#: FORMAT_RULE card header: ``**Title (Year)**`` at the start of a line.
_CARD_LINE = re.compile(r"^\*\*[^*\n]+\(\d{4}\)\*\*")

# Theme hexes from .streamlit/config.toml — white, red, grey, text.
_WHITE = "#FFFFFF"
_RED = "#D7263D"
_GREY = "#F3F4F6"
_INK = "#1B1B24"

_LAYOUT_CSS = f"""
<style>
[data-testid="stMainBlockContainer"] {{
  max-width: 760px !important;
  margin-left: auto;
  margin-right: auto;
}}
[data-testid="stFeedback"] {{
  margin: 8px 0 16px;
}}
.maya-turn {{ margin: 0 0 8px; }}
.maya-user {{ display: flex; justify-content: flex-end; }}
.maya-user-copy {{
  max-width: 68%;
  margin: 0;
  background: {_RED};
  color: {_WHITE};
  padding: 12px 16px;
  border-radius: 16px 16px 4px 16px;
  font-size: 15px;
  line-height: 1.45;
  white-space: pre-wrap;
}}
.maya-assistant {{
  margin-top: 40px;
  background: {_WHITE};
  border: 1px solid color-mix(in srgb, {_INK} 16%, {_WHITE});
  border-radius: 16px;
  overflow: hidden;
}}
.maya-sec {{ padding: 18px 20px 16px; }}
.maya-sec + .maya-sec {{
  border-top: 1px solid color-mix(in srgb, {_INK} 16%, {_WHITE});
}}
.maya-retrieval {{ background: {_GREY}; }}
.maya-kicker {{
  margin: 0 0 8px;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-weight: 700;
  color: {_RED};
}}
.maya-prose {{
  margin: 0;
  font-size: 16px;
  line-height: 1.6;
  color: {_INK};
  white-space: pre-wrap;
}}
.maya-strip {{
  display: flex;
  gap: 10px;
  overflow-x: auto;
  padding-bottom: 4px;
}}
.maya-poster {{ width: 112px; flex: 0 0 112px; margin: 0; }}
.maya-poster img {{
  width: 112px;
  height: 168px;
  object-fit: cover;
  border-radius: 8px;
  display: block;
  background: {_GREY};
}}
.maya-poster figcaption {{
  margin-top: 6px;
  font-size: 12px;
  line-height: 1.35;
  color: {_INK};
}}
.maya-chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.maya-chip {{
  background: {_GREY};
  color: {_INK};
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
}}
</style>
"""


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


def user_bubble_html(text: str) -> str:
    """Right-aligned visitor line (prototype A)."""
    return (
        '<div class="maya-user">'
        f'<p class="maya-user-copy">{html.escape(text)}</p>'
        "</div>"
    )


def _poster_strip_html(movies: list) -> str:
    cards = []
    for movie in movies:
        title = html.escape(getattr(movie, "title", ""))
        year = html.escape(str(getattr(movie, "release_year", "")))
        score = f"{float(getattr(movie, 'vote_average', 0) or 0):.1f}"
        genres = html.escape(", ".join(getattr(movie, "genres", [])[:3]))
        src = html.escape(getattr(movie, "poster_url", "") or "", quote=True)
        cards.append(
            '<figure class="maya-poster">'
            f'<img src="{src}" alt="{title}">'
            f"<figcaption><b>{title}</b> ({year})<br>{score} / 10"
            + (f" — {genres}" if genres else "")
            + "</figcaption></figure>"
        )
    return f'<div class="maya-strip">{"".join(cards)}</div>'


def _chips_html(fields: list[tuple[str, str]]) -> str:
    chips = "".join(
        f'<span class="maya-chip">{html.escape(label)} · {html.escape(value)}</span>'
        for label, value in fields
    )
    return f'<div class="maya-chips">{chips}</div>'


def assistant_panel_html(response: str, row: dict | None) -> str:
    """One assistant panel: Conversation, Retrieval, Metadata (prototype A)."""
    movies = list((row or {}).get("movies") or [])
    prose = conversation_prose(response)
    if not prose and not movies:
        prose = (response or "").strip()
    sections: list[str] = []
    if prose:
        sections.append(
            '<section class="maya-sec maya-conversation">'
            '<h2 class="maya-kicker">Conversation</h2>'
            f'<p class="maya-prose">{html.escape(prose)}</p>'
            "</section>"
        )
    if movies:
        sections.append(
            '<section class="maya-sec maya-retrieval">'
            '<h2 class="maya-kicker">Retrieval</h2>'
            f"{_poster_strip_html(movies)}"
            "</section>"
        )
    if row is not None:
        sections.append(
            '<section class="maya-sec maya-metadata">'
            '<h2 class="maya-kicker">Metadata</h2>'
            f"{_chips_html(metadata_fields(row))}"
            "</section>"
        )
    return f'<div class="maya-assistant">{"".join(sections)}</div>'


def stacked_turn_html(user: str, response: str, row: dict | None) -> str:
    """Visitor bubble, 40px, then the assistant panel."""
    return (
        '<div class="maya-turn">'
        f"{user_bubble_html(user)}"
        f"{assistant_panel_html(response, row)}"
        "</div>"
    )


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
    session: MayaSession, user: str, content: str, row: dict | None, turn_index: int
) -> None:
    """Prototype A turn: one HTML panel, then the live thumb control (#147)."""
    st.markdown(stacked_turn_html(user, content, row), unsafe_allow_html=True)
    render_feedback(session, turn_index)
    if row is not None:
        render_report_receipt(session, row)


def render_chat(session: MayaSession) -> None:
    st.markdown(_LAYOUT_CSS, unsafe_allow_html=True)
    st.header("Maya")
    st.caption(
        "Conversational film curator for US theatrical releases, 1970–2026 — "
        "deterministic routing, closed-world grounding, full trace observability."
    )

    messages = session.conversation.messages
    turn_index = -1
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.role != "user":
            i += 1
            continue
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        if nxt is not None and nxt.role == "assistant":
            turn_index += 1
            row = resolve_turn_row(session, turn_index)  # #26-K identity join
            render_assistant_turn(session, msg.content, nxt.content, row, turn_index)
            i += 2
        else:
            st.markdown(
                f'<div class="maya-turn">{user_bubble_html(msg.content)}</div>',
                unsafe_allow_html=True,
            )
            i += 1

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

    try:
        with st.spinner("Working through the pipeline"):
            session.turn(query)
    except Exception as exc:  # noqa: BLE001 — surface a readable failure, never a traceback
        st.markdown(
            f'<div class="maya-turn">{user_bubble_html(query)}</div>',
            unsafe_allow_html=True,
        )
        st.error(
            "Maya could not complete this turn. Check that the app was started with "
            "the encrypted environment loaded:  \n"
            "`npx @dotenvx/dotenvx run -- streamlit run app.py`  \n"
            f"Details: {type(exc).__name__}: {exc}"
        )
        return
    idx = len(session.turn_log) - 1
    last = resolve_turn_row(session, idx) or session.turn_log[idx]
    render_assistant_turn(session, query, last.get("response", ""), last, idx)
    scroll_to_newest()  # #27-R: land on the fresh response, not the page top
