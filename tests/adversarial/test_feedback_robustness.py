"""Adversarial tests for the feedback loop (issue #9)."""

import sqlite3

import pytest

from src.feedback.langfuse_score import push_feedback_score
from src.feedback.store import FeedbackStore

pytestmark = pytest.mark.adversarial


@pytest.fixture
def store() -> FeedbackStore:
    return FeedbackStore(":memory:")


# --- input validation: only ±1 survives ---------------------------------

@pytest.mark.parametrize("bad", [0, None, "up", 5, 1.5])
def test_invalid_ratings_rejected_before_disk(store, bad):
    with pytest.raises(ValueError):
        store.record("trace-1", bad, "v1_0_baseline")
    assert store.count() == 0


def test_empty_trace_id_rejected(store):
    with pytest.raises(ValueError):
        store.record("", 1, "v1_0_baseline")
    assert store.count() == 0


def test_widget_boundary_mapping():
    """st.feedback('thumbs') yields 0 for down; UI must map to -1 (#9)."""
    from src.ui.chat_tab import _widget_rating_to_canonical

    assert _widget_rating_to_canonical(1) == 1  # thumbs up
    assert _widget_rating_to_canonical(0) == -1  # thumbs down → canonical ±1


# --- failure isolation: telemetry must never break the chat --------------

def test_langfuse_client_failure_returns_false_not_raises(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

    class ExplodingClient:
        def create_score(self, **kwargs):
            raise RuntimeError("network down")

    import langfuse

    monkeypatch.setattr(langfuse, "get_client", lambda: ExplodingClient())
    assert push_feedback_score("trace-1", 1) is False  # swallowed + logged


def test_langfuse_import_failure_returns_false(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    import sys

    monkeypatch.setitem(sys.modules, "langfuse", None)  # import langfuse → None-ish failure
    assert push_feedback_score("trace-1", -1) is False


# --- persistence robustness ----------------------------------------------

def test_malformed_preexisting_table_still_upserts(tmp_path):
    """Pre-#9 tables lack the UNIQUE index; the store must migrate in place."""
    db = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE user_feedback (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               trace_id TEXT NOT NULL, rag_version TEXT NOT NULL,
               rating INTEGER NOT NULL, user_comment TEXT,
               timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.execute(
        "INSERT INTO user_feedback (trace_id, rag_version, rating) VALUES ('old', 'v9', 1)"
    )
    conn.commit()
    conn.close()

    store = FeedbackStore(db)
    store.record("old", -1, "v1_1_enriched")  # UPSERT over the legacy row
    assert store.count() == 1
    assert store.get_by_trace_id("old")["rating"] == -1


def test_rapid_rerating_last_value_wins(store):
    for rating in (1, -1, 1, -1):
        store.record("trace-1", rating, "v1_1_enriched")
    assert store.count() == 1
    assert store.get_by_trace_id("trace-1")["rating"] == -1


def test_session_record_feedback_validates_inputs():
    """session.record_feedback raises before touching store or cloud."""
    from src.ui.session import MayaSession

    session = MayaSession.__new__(MayaSession)
    session.turn_log = []
    session.feedback_log = {}
    session.feedback_store = FeedbackStore(":memory:")

    with pytest.raises(ValueError):
        session.record_feedback(0, 0)
    with pytest.raises(IndexError):
        session.record_feedback(99, 1)
    assert session.feedback_store.count() == 0
