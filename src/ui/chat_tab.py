"""Chat view (issue #7): conversation, routing chip, poster gallery, feedback.

Professional chrome: no emojis, plain-text separators. Feedback thumbs are
captured per assistant turn and linked to that turn's trace row (Langfuse
scoring + SQLite persistence land with issue #9).
"""

import streamlit as st

from src.ui.session import ADMIN_COMMAND, MayaSession


def intent_badge_text(log_row: dict) -> str:
    """Pure helper (unit-tested): one-line routing chip for a turn."""
    path = log_row.get("path", "?")
    attempts = log_row.get("attempts", 1)
    path_label = f"{path} (x{attempts})" if attempts > 1 else path
    return (
        f"INTENT: {log_row.get('intent', '?')} — confidence {log_row.get('confidence', 0):.2f} "
        f"— route: {path_label} — {log_row.get('n_movies', 0)} movies "
        f"— {log_row.get('tokens', 0)} tokens"
    )


def render_intent_badge(log_row: dict) -> None:
    st.caption(intent_badge_text(log_row))


def render_feedback(session: MayaSession, turn_index: int) -> None:
    """Thumbs up/down per assistant turn; value persists in the session log."""
    value = st.feedback("thumbs", key=f"feedback_{turn_index}")
    if value is not None:
        session.feedback_log[turn_index] = value


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


def render_chat(session: MayaSession) -> None:
    st.title("Maya")
    st.caption(
        "Conversational film curator for US theatrical releases, 1970–2026. "
        "Deterministic routing, closed-world grounding, full trace observability. "
        f"Type {ADMIN_COMMAND} for the Experimentation Lab."
    )

    # history: routing chip + thumbs inside each assistant bubble
    turn_index = -1
    for msg in session.conversation.messages:
        role = "user" if msg.role == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)
            if msg.role != "assistant":
                continue
            turn_index += 1
            if turn_index < len(session.turn_log):
                render_intent_badge(session.turn_log[turn_index])
            render_feedback(session, turn_index)

    render_poster_grid(session.last_movies)

    query = st.chat_input("Ask Maya about movies")
    if not query:
        return
    if session.is_admin_command(query):
        st.toast("The Experimentation Lab lives in the collapsible sidebar.")
        return

    with st.chat_message("user"):
        st.markdown(query)
    assistant = st.chat_message("assistant")
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
        last = session.turn_log[idx]
        st.markdown(last["response"])
        render_intent_badge(last)
        render_feedback(session, idx)
    render_poster_grid(session.last_movies)
