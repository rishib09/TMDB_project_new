"""Smoke tests for the Streamlit UI layer (issue #7).

Deliberately minimal: only pure helpers are unit-tested — Streamlit widget
code cannot execute outside the Streamlit runtime, and the real acceptance
is the manual look-and-feel walkthrough. These tests catch import errors
and pure-logic regressions cheaply.
"""


from src.ui.chat_tab import intent_badge_text
from src.ui.session import ADMIN_COMMAND, MayaSession


def test_modules_import():
    """UI modules import (catches syntax/bad-import errors).

    Note: app.py itself is excluded — executing it bare would boot the whole
    pipeline (DB + chroma + graph) outside the Streamlit runtime; it is
    verified by `streamlit run app.py` in the manual walkthrough instead.
    """
    import importlib

    for module in ("src.ui.session", "src.ui.chat_tab", "src.ui.sidebar_lab",
                   "src.ui.trace_tab", "src.ui.evals_tab"):
        importlib.import_module(module)


def test_admin_command_detection():
    assert MayaSession.is_admin_command("/admin")
    assert MayaSession.is_admin_command("  /ADMIN ")
    assert not MayaSession.is_admin_command("/admin what movies")
    assert not MayaSession.is_admin_command("hello")
    assert ADMIN_COMMAND == "/admin"


def test_intent_badge_text_formatting():
    row = {"intent": "SEMANTIC_SEARCH", "confidence": 0.92, "path": "single-route",
           "attempts": 1, "n_movies": 5, "tokens": 210}
    text = intent_badge_text(row)
    assert text == "SEMANTIC_SEARCH · conf 0.92 · single-route · 5 movies · 210 tok"


def test_intent_badge_shows_reroute_attempts():
    row = {"intent": "SEMANTIC_SEARCH", "confidence": 0.4, "path": "reroute",
           "attempts": 2, "n_movies": 0, "tokens": 0}
    assert "reroute ×2" in intent_badge_text(row)


def test_waterfall_frame_handles_empty_and_single():
    from src.ui.trace_tab import _waterfall_frame

    assert _waterfall_frame([]).empty
    frame = _waterfall_frame([{"node": "guard_input", "timestamp": "2026-09-01T10:00:00",
                               "payload": {"verdict": "clean"}}])
    assert len(frame) == 1 and frame.iloc[0]["node"] == "guard_input"
