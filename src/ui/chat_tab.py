"""Chat view (issue #7): conversation, routing chip, poster gallery, feedback.

Professional chrome: no emojis, plain-text separators. Feedback thumbs are
captured per assistant turn and linked to that turn's trace row (Langfuse
scoring + SQLite persistence land with issue #9).
"""

import streamlit as st
from streamlit.components.v1 import html as _components_html

from src.ui.session import MayaSession

MAYA_AVATAR = ":material/movie:"
USER_AVATAR = ":material/person:"

_CHAT_CSS = """
<style>
[data-testid="stHeader"] { display: none; }
[data-testid="stChatMessage"] { padding-top: 0.3rem; padding-bottom: 0.3rem; }
[data-testid="stChatMessage"] p { margin-bottom: 0.25rem; }
[data-testid="stVerticalBlock"] { gap: 0.35rem; }
.maya-sticky-header {
  position: sticky; top: 0; z-index: 1000;
  background: #ffffff; padding: 0.5rem 0 0.6rem;
  border-bottom: 1px solid #ececec; margin-bottom: 0.5rem;
}
.maya-sticky-header .maya-title {
  font-size: 1.7rem; font-weight: 700; line-height: 1.25; color: #1b1b24;
}
.maya-sticky-header .maya-subtitle { font-size: 0.85rem; color: #5a5a66; }
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


def render_poster_grid(movies, cols: int = 4) -> None:
    """Retrieved-context gallery: bordered cards, uniform poster width."""
    if not movies:
        return
    st.markdown(
        f"**Retrieved context** — Maya's closed world this turn ({len(movies)} movies)"
    )
    for start in range(0, len(movies), cols):
        chunk = movies[start : start + cols]
        for col, movie in zip(st.columns(cols), chunk):
            with col, st.container(border=True):
                st.image(movie.poster_url)
                st.markdown(f"**{movie.title}** ({movie.release_year})")
                st.caption(f"{movie.vote_average:.1f} / 10 — " + ", ".join(movie.genres[:3]))


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
    scroll position.
    """
    _components_html(f"<div style='height:0px'></div>{_AUTO_SCROLL_JS}", height=0)


def render_chat(session: MayaSession) -> None:
    st.markdown(_CHAT_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="maya-sticky-header">'
        '<div class="maya-title">Maya</div>'
        '<div class="maya-subtitle">Conversational film curator for US theatrical '
        "releases, 1970\u20132026 — deterministic routing, closed-world grounding, "
        "full trace observability. Type /admin for the Experimentation Lab."
        "</div></div>",
        unsafe_allow_html=True,
    )

    # history: routing chip + thumbs inside each assistant bubble
    turn_index = -1
    for msg in session.conversation.messages:
        role = "user" if msg.role == "user" else "assistant"
        avatar = USER_AVATAR if msg.role == "user" else MAYA_AVATAR
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg.content)
            if msg.role != "assistant":
                continue
            turn_index += 1
            row = resolve_turn_row(session, turn_index)  # #26-K identity join
            if row is not None:
                render_intent_badge(row)
            render_feedback(session, turn_index)

    render_poster_grid(session.last_movies)

    query = st.chat_input("Ask Maya about movies")
    if not query:
        return
    if session.is_admin_command(query):
        st.toast("The Experimentation Lab lives in the collapsible sidebar.")
        return

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(query)
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
        st.markdown(last["response"])
        render_intent_badge(last)
        render_feedback(session, idx)
    render_poster_grid(session.last_movies)
    scroll_to_newest()  # #27-R: land on the fresh response, not the page top
