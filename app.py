"""Maya — Conversational Film Curator (Streamlit entrypoint).

Runs locally:  streamlit run app.py
Deploys to Hugging Face Spaces (Streamlit SDK — see README.md metadata).

Layout: collapsible sidebar = view navigation (Chat | Evals | Traces) plus
the Experimentation Lab; the main area renders the selected view.
"""

import streamlit as st

from src.ui.chat_tab import render_chat
from src.ui.evals_tab import render_evals
from src.ui.session import get_session
from src.ui.sidebar_lab import render_lab
from src.ui.trace_tab import render_traces

st.set_page_config(
    page_title="Maya — Film Curator",
    page_icon=":material/movie:",
    layout="wide",
    initial_sidebar_state="expanded",
)

session = get_session()

with st.sidebar:
    views = ["Chat", "Evals", "Traces"]
    view_icons = {"Chat": ":material/chat:", "Evals": ":material/monitoring:", "Traces": ":material/timeline:"}
    selection = st.segmented_control(
        "Pages",
        views,
        format_func=lambda v: f"{view_icons[v]} {v}",
        default=session.view,
        width="stretch",
        help="Switch between the Chat, Evals, and Traces pages.",
    )
    if selection:
        session.view = selection
    st.divider()
    render_lab(session)

if session.view == "Chat":
    render_chat(session)
elif session.view == "Evals":
    render_evals(session)
else:
    render_traces(session)
