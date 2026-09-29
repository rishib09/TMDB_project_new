"""Pure date/budget primitives shared by guardrails, storage and the UI (#39).

``src/domain/`` stays stdlib-only (ADR 0006), which is what lets storage and
UI both import these without a layering violation: the tracker records in
``src/maya/guardrails.py``, the window is queried in ``src/storage/database.py``,
the caption renders in ``src/ui/sidebar_lab.py`` — all three must agree on
"which week is it".
"""

from datetime import UTC, date, datetime, timedelta


def utc_today() -> date:
    """Today's date in UTC — the one clock every budget date goes through.

    The app deploys to HF Spaces (server clock = UTC); pinning here keeps the
    write side (record), the read side (window query) and the caption on the
    same boundary, independent of any local timezone.
    """
    return datetime.now(UTC).date()


def week_bounds(reference: date) -> tuple[date, date]:
    """The ISO week (Monday, Sunday) containing ``reference`` — inclusive."""
    week_start = reference - timedelta(days=reference.weekday())
    return week_start, week_start + timedelta(days=6)
