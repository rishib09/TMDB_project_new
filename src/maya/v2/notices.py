"""v2 filter transparency (#153): the carry-over notice, owned by v2.

v2 never imports v1's modules — the two routing stacks will be split into
separate branches — so the transparency line is v2-local. It deliberately
DIVERGES from v1's ``src/maya/probing.build_filter_carryover_notice``: v1
announces every remembered chip (its funnel owns the whole narrowing), while
v2 announces exactly the genres code folded into this turn's SQL (#25) —
review decision 2026-10-04: the line must not over-claim filters that did
not run. v1's all-chips version stays in probing.py for v1's funnel path.

The transparency line itself is deterministic code, never a model call: the
filter injection it explains is a code disposal (the #25 genre merge in the
shared retrieve node), so the explanation is code's job too (ADR 0005).
"""

from src.domain.routing import MetadataFilterCriteria


def build_filter_carryover_notice(injected: list[str]) -> str:
    """Post-retrieval transparency line (#153): the genres that silently
    joined this turn's SQL, with the deterministic escape hatch.

    Empty when nothing joined (nothing to announce).
    """
    if not injected:
        return ""
    return (
        "\n\n---\nStill filtering by genres: " + ", ".join(injected) + " — want to "
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