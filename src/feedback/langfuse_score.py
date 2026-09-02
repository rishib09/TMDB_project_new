"""Langfuse score push for user feedback (issue #9).

Cloud half of the feedback loop: pushes the thumb rating as a NUMERIC
score on the exact per-turn trace. Strictly optional — without keys or
when the client fails this returns False and logs; the UI must never
break because telemetry is down.
"""

import logging
import os

logger = logging.getLogger(__name__)

FEEDBACK_SCORE_NAME = "user-feedback"


def cloud_configured() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def push_feedback_score(trace_id: str, rating: int, comment: str | None = None) -> bool:
    """Pushes one ±1 rating as a NUMERIC score; True on success.

    Never raises: telemetry failure must not break the chat (issue #9,
    adversarial requirement).
    """
    if rating not in (1, -1):
        raise ValueError(f"rating must be +1 or -1, got {rating!r}")
    if not cloud_configured():
        logger.info("Langfuse not configured — feedback kept in SQLite only")
        return False
    try:
        import langfuse

        langfuse.get_client().create_score(
            trace_id=trace_id,
            name=FEEDBACK_SCORE_NAME,
            value=float(rating),
            data_type="NUMERIC",
            comment=comment,
            # deterministic id → re-rating UPSERTS the score cloud-side too
            score_id=f"{trace_id}:{FEEDBACK_SCORE_NAME}",
        )
        return True
    except Exception:
        logger.exception("Failed to push feedback score to Langfuse")
        return False
