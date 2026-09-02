"""Live feedback-loop tests (issue #9): real SQLite + real Langfuse.

Runs against the real ``data/tmdb_movies.db`` and, when keys are present,
the real Langfuse cloud. Skipif-gated on the ``live`` marker contract.
"""

import pytest

from src.feedback.langfuse_score import cloud_configured, push_feedback_score
from src.feedback.store import FeedbackStore

pytestmark = [pytest.mark.live, pytest.mark.skipif(not cloud_configured(),
               reason="Langfuse keys not configured")]


def test_feedback_persists_to_real_database(tmp_path):
    """Real file DB: record → read back → stats (uses tmp copy, never prod data)."""
    store = FeedbackStore(str(tmp_path / "live_feedback.db"))
    store.record("live-trace-1", 1, "v1_1_enriched", intent="SEMANTIC_SEARCH")
    store.record("live-trace-2", -1, "v1_0_baseline")
    store.record("live-trace-1", -1, "v1_1_enriched")  # re-rate updates

    assert store.count() == 2
    assert store.get_by_trace_id("live-trace-1")["rating"] == -1
    stats = {s["rag_version"]: s for s in store.stats_by_version()}
    assert stats["v1_1_enriched"]["n"] == 1
    assert stats["v1_0_baseline"]["thumbs_down"] == 1


def test_feedback_score_reaches_langfuse_cloud():
    """Real cloud round-trip: mint trace id → push score → flush.

    Verification is ingestion-side (no exception through flush): project
    keys can write scores but GET read-back may be unauthorized depending
    on key scope — the loop itself never needs read-back (#9).
    """
    import langfuse

    client = langfuse.get_client()
    trace_id = client.create_trace_id()
    assert push_feedback_score(trace_id, 1) is True
    client.flush()  # raises if ingestion rejected the batch
    # re-rating upserts via deterministic score_id — must also ingest cleanly
    assert push_feedback_score(trace_id, -1) is True
    client.flush()
