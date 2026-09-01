"""Maya — Conversational Film Curator (Streamlit entrypoint).

Runs locally:  streamlit run app.py
Deploys to Hugging Face Spaces (Streamlit SDK — see README.md metadata).

Layout: sidebar = Experimentation Lab; main = Chat | Evals | Traces tabs.
`/admin` in chat flips the main area into the Lab.
"""

import streamlit as st

from src.ui.chat_tab import render_chat
from src.ui.evals_tab import render_evals
from src.ui.session import get_session
from src.ui.sidebar_lab import render_lab
from src.ui.trace_tab import render_traces

st.set_page_config(
    page_title="Maya — Film Curator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

session = get_session()
render_lab(session)

st.title("🎬 Maya")
st.caption("Your film curator for everything 1970–2026 — grounded, traced, measurable.")

if session.admin_mode:
    st.info("Experimentation Lab — press *Back to Chat* when done.")
    if st.button("← Back to Chat"):
        session.admin_mode = False
        st.rerun()
    # Lab inline in the main area: config editor + live traces side by side
    left, right = st.columns([3, 2])
    with left:
        render_lab(session)
    with right:
        render_traces(session)
else:
    tab_chat, tab_evals, tab_trace = st.tabs(["💬 Chat", "📊 Evals", "🔍 Traces"])
    with tab_chat:
        render_chat(session)
    with tab_evals:
        render_evals()
    with tab_trace:
        render_traces(session)
