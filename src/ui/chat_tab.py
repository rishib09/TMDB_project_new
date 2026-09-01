"""Chat tab (issue #7): conversation, intent badges, poster grid."""

import streamlit as st

from src.domain.routing import IntentType
from src.ui.session import ADMIN_COMMAND, MayaSession

#: Intent → badge color (Streamlit native st.badge palette).
_INTENT_COLORS = {
    IntentType.GREETING: "blue",
    IntentType.CAPABILITIES: "blue",
    IntentType.SEMANTIC_SEARCH: "green",
    IntentType.ATTRIBUTE_FILTER: "green",
    IntentType.SUPERLATIVE_RANKING: "orange",
    IntentType.NEGATION_EXCLUSION: "red",
    IntentType.OUT_OF_SCOPE: "gray",
}


def intent_badge_text(log_row: dict) -> str:
    """Pure helper (unit-tested): one-line transparency chip for a turn."""
    path = log_row.get("path", "?")
    attempts = log_row.get("attempts", 1)
    path_label = f"{path} ×{attempts}" if attempts > 1 else path
    return (
        f"{log_row.get('intent', '?')} · conf {log_row.get('confidence', 0):.2f} "
        f"· {path_label} · {log_row.get('n_movies', 0)} movies · {log_row.get('tokens', 0)} tok"
    )


def render_intent_badge(log_row: dict) -> None:
    st.caption("🔎 " + intent_badge_text(log_row))


def render_poster_grid(movies, cols: int = 3) -> None:
    """Top-k poster gallery below a RAG reply (native columns + images)."""
    if not movies:
        return
    st.caption(f"🎬 Retrieved context ({len(movies)} movies — Maya's closed world this turn)")
    for start in range(0, len(movies), cols):
        chunk = movies[start : start + cols]
        columns = st.columns(cols)
        for col, movie in zip(columns, chunk):
            with col:
                st.image(movie.poster_url, use_container_width=True)
                st.markdown(
                    f"**{movie.title}** ({movie.release_year})\n\n"
                    f"⭐ {movie.vote_average:.1f} · {', '.join(movie.genres[:3])}"
                )


def render_chat(session: MayaSession) -> None:
    st.subheader("💬 Chat with Maya")
    st.caption(
        "Ask about movies 1970–2026 — plots, moods, superlatives, exclusions. "
        f"Type `{ADMIN_COMMAND}` to open the Experimentation Lab."
    )

    # history
    for msg, log_row in zip(session.conversation.messages, _pair_log(session)):
        role = "user" if msg.role == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)
            if role == "assistant" and log_row is not None:
                render_intent_badge(log_row)

    # last turn's retrieved posters
    render_poster_grid(session.last_movies)

    query = st.chat_input("Ask Maya about movies…")
    if not query:
        return
    if session.is_admin_command(query):
        session.admin_mode = True
        st.rerun()

    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"), st.spinner("Maya is thinking…"):
        session.turn(query)
        last = session.turn_log[-1]
        st.markdown(last["response"])
        render_intent_badge(last)
        render_poster_grid(session.last_movies)


def _pair_log(session: MayaSession):
    """Aligns history messages with turn-log rows for badge rendering."""
    pairs = []
    log_iter = iter(session.turn_log)
    for msg in session.conversation.messages:
        pairs.append(next(log_iter) if msg.role == "assistant" else None)
    return pairs
