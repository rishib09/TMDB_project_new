"""Fetch-back trace inspection (issue #31): Langfuse observations → local tree.

O1 architecture (grilled 2026-09-07): the app never duplicates prompt capture.
The stock LangChain handler ships everything to Langfuse Cloud; this module
fetches it back per trace_id via ``api.observations.get_many`` and rebuilds
the execution tree client-side from ``parent_observation_id`` links.
``api.trace.get`` is deprecated (Cloud removal 2026-11-16) and must not be
used. All network failures fail open: callers receive None and the UI shows
a "still ingesting" status (Langfuse ingestion lag is 15-30s).
"""

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

#: Field groups requested from the observations endpoint — everything the
#: inspector renders (D2): io, model params, usage/cost, real timings.
OBSERVATION_FIELDS = "core,basic,time,io,model,usage,metrics"

#: Per-trace observation ceiling. A turn yields ~8-12 observations; 100
#: leaves ample headroom without pagination.
FETCH_LIMIT = 100


class ObservationNode(BaseModel):
    """One fetched observation with its children (pure, framework-free)."""

    id: str
    parent_id: str | None = None
    name: str = ""
    type: str = "SPAN"  # SPAN | GENERATION | EVENT
    start_time: datetime | None = None
    end_time: datetime | None = None
    input: Any = None
    output: Any = None
    model: str | None = None
    usage: dict = Field(default_factory=dict)
    cost: float | None = None
    children: list["ObservationNode"] = Field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        """Real span duration (fixes the old gap-to-next-span estimate)."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None


class TraceTree(BaseModel):
    """A complete fetched trace: root observations plus attached scores."""

    trace_id: str
    roots: list[ObservationNode]
    scores: list[dict] = Field(default_factory=list)


def _decode_io(value: Any) -> Any:
    """The observations API returns input/output as JSON strings (live-verified
    2026-09-07); decode so message lists render as role blocks, not one blob.
    Non-JSON strings (plain completions) pass through unchanged."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _as_node(raw: Any) -> ObservationNode:
    """Normalizes one SDK observation (model or dict) into an ObservationNode."""
    get = raw.get if isinstance(raw, dict) else lambda k, d=None: getattr(raw, k, d)
    usage = get("usage_details") or get("usage") or {}
    if not isinstance(usage, dict):
        usage = getattr(usage, "__dict__", {}) or {}
    cost = get("total_cost", None)
    if cost is None:
        cost_details = get("cost_details") or {}
        if isinstance(cost_details, dict):
            cost = cost_details.get("total")
    return ObservationNode(
        id=str(get("id", "")),
        parent_id=get("parent_observation_id") or None,
        name=get("name") or "",
        type=get("type") or "SPAN",
        start_time=get("start_time"),
        end_time=get("end_time"),
        input=_decode_io(get("input")),
        output=_decode_io(get("output")),
        model=get("model"),
        usage={k: v for k, v in usage.items() if isinstance(v, int | float)},
        cost=cost if isinstance(cost, int | float) else None,
    )


def _creates_cycle(node_id: str, parent_id: str, by_id: dict[str, ObservationNode]) -> bool:
    """True when attaching node under parent would loop back to node."""
    seen = set()
    current: str | None = parent_id
    while current is not None and current not in seen:
        if current == node_id:
            return True
        seen.add(current)
        parent = by_id.get(current)
        current = parent.parent_id if parent else None
    return current is not None  # a pre-existing loop above: treat as cyclic


def build_tree(observations: list[dict]) -> list[ObservationNode]:
    """Pure: rebuilds the nested tree from parent_observation_id links.

    Orphans (missing parent) and cycle-forming links become roots — a
    malformed trace renders flat rather than crashing the inspector.
    Siblings sort by start_time (None last) for a stable waterfall order.
    """
    nodes = [_as_node(raw) for raw in observations]
    by_id = {n.id: n for n in nodes if n.id}
    roots: list[ObservationNode] = []
    for node in nodes:
        pid = node.parent_id
        if pid and pid in by_id and pid != node.id and not _creates_cycle(node.id, pid, by_id):
            by_id[pid].children.append(node)
        else:
            roots.append(node)

    def sort_key(n: ObservationNode):
        return (n.start_time is None, n.start_time or datetime.min, n.id)

    def sort_rec(items: list[ObservationNode]) -> None:
        items.sort(key=sort_key)
        for item in items:
            sort_rec(item.children)

    sort_rec(roots)
    return roots


def _fetch_scores(client, trace_id: str) -> list[dict]:
    """Trace-level scores (thumbs feedback from #9); [] on any failure."""
    try:
        response = client.api.scores_v3.get_many_v3(trace_id=trace_id)
        return [
            {
                "name": getattr(s, "name", ""),
                "value": getattr(s, "value", None),
                "comment": getattr(s, "comment", None),
            }
            for s in getattr(response, "data", []) or []
        ]
    except Exception:
        return []


def fetch_trace_tree(trace_id: str) -> TraceTree | None:
    """Fetches one trace's observations + scores from Langfuse Cloud.

    Returns None when the trace is not yet ingested (15-30s lag), keys are
    absent, or any network/SDK error occurs — fail-open by contract, and
    the failure is deliberate UI state ("still ingesting"), not silence.
    """
    if not trace_id:
        return None
    try:
        from langfuse import get_client

        client = get_client()
        response = client.api.observations.get_many(
            trace_id=trace_id, fields=OBSERVATION_FIELDS, limit=FETCH_LIMIT
        )
        raw = list(getattr(response, "data", []) or [])
        if not raw:
            return None
        return TraceTree(
            trace_id=trace_id,
            roots=build_tree(raw),
            scores=_fetch_scores(client, trace_id),
        )
    except Exception:
        return None
