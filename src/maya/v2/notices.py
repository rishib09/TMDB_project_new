"""v2 filter transparency (#153): the carry-over notice, owned by v2.

DELIBERATE COPY of ``src/maya/probing.py``'s ``preference_chips`` and
``build_filter_carryover_notice`` (#26-E). The v2 stack must not import v1's
modules — the two routing stacks will be split into separate branches — so
these helpers are duplicated here, byte-for-byte in behavior, with the new
injection-diff helper this stack needs. When the split lands, this module
travels with v2 and probing.py stays with v1 untouched (per #153 decision).

The transparency line itself is deterministic code, never a model call: the
filter injection it explains is a code disposal (the #25 genre merge in the
shared retrieve node), so the explanation is code's job too (ADR 0005).
"""

from src.domain.memory import UserSessionPreferences
from src.domain.routing import MetadataFilterCriteria


def preference_chips(prefs: UserSessionPreferences) -> list[str]:
    """Human-readable chips for the active preferences (shared by UI + notice).

    Copy of ``src/maya/probing.preference_chips`` — see module docstring.
    """
    chips: list[str] = []
    if prefs.preferred_mood:
        chips.append(f"mood: {prefs.preferred_mood}")
    if prefs.audience:
        chips.append(f"audience: {prefs.audience}")
    chips.extend(f"no {d}" for d in prefs.noted_donts)
    if prefs.preferred_genres:
        chips.append("genres: " + ", ".join(prefs.preferred_genres))
    chips.extend(f"dir. {d}" for d in prefs.preferred_directors)
    # #27-Q: carried year constraints are visible like any other filter.
    if prefs.exact_year:
        chips.append(f"year: {prefs.exact_year}")
    elif prefs.year_min or prefs.year_max:
        lo = prefs.year_min or "…"
        hi = prefs.year_max or "…"
        chips.append(f"years: {lo}-{hi}")
    return chips


def build_filter_carryover_notice(prefs: UserSessionPreferences) -> str:
    """Post-retrieval transparency line (#153, pattern #26-E): what is still
    filtering, with the deterministic escape hatch.

    Copy of ``src/maya/probing.build_filter_carryover_notice`` — see module
    docstring. Empty when nothing is remembered (nothing to announce).
    """
    chips = preference_chips(prefs)
    if not chips:
        return ""  # no prefs carried → nothing to announce
    return (
        "\n\n---\nStill filtering by " + " · ".join(chips) + " — want to "
        'continue with these, or watch something completely different?'
    )


def injected_genres(
    filters_applied: dict | None,
    declared_filters: MetadataFilterCriteria | None,
) -> list[str]:
    """Genres code folded INTO the effective SQL that this turn never declared.

    The shared retrieve node injects the session's preferred genres when the
    turn's own filters carry none (#25). ``routing_decision`` stays
    pre-injection in graph state (retrieve never writes it back), so the
    difference between the effective set (``filters_applied``) and the
    declared set is exactly what joined silently — the event the transparency
    notice announces. Empty when nothing was injected.

    Case-insensitive on purpose: the declared list comes from a model
    utterance, the effective list from the merged criteria.
    """
    if not filters_applied:
        return []
    declared = {
        g.casefold() for g in (declared_filters.genres if declared_filters else [])
    }
    return [
        g for g in (filters_applied.get("genres") or [])
        if g.casefold() not in declared
    ]