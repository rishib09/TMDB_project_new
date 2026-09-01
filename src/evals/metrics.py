"""Information-Retrieval metrics for the Maya benchmark (issue #6).

Pure functions over ranked-movie-id lists — textbook definitions, no
framework. Every function is total: degenerate inputs (empty lists, no
relevant ids, k > corpus) return 0.0 instead of raising, because a metric
that crashes mid-run destroys the rest of the report.
"""

from pydantic import BaseModel, Field


def hit_rate_at_k(ranked_ids: list[int], relevant_ids: list[int], k: int) -> float:
    """1.0 if any relevant id appears in the top-k, else 0.0."""
    if not relevant_ids or k <= 0:
        return 0.0
    top = ranked_ids[:k]
    return 1.0 if any(mid in relevant_ids for mid in top) else 0.0


def mrr_at_k(ranked_ids: list[int], relevant_ids: list[int], k: int) -> float:
    """Reciprocal rank of the first relevant id within the top-k (0 if absent)."""
    if not relevant_ids or k <= 0:
        return 0.0
    for rank, mid in enumerate(ranked_ids[:k], start=1):
        if mid in relevant_ids:
            return 1.0 / rank
    return 0.0


def context_precision_at_k(ranked_ids: list[int], relevant_ids: list[int], k: int) -> float:
    """Average precision over the top-k (ragas-style Context Precision).

    AP@k = sum(precision@i * rel(i)) / number of relevant hits in top-k,
    rewarding rankings that place relevant ids earlier.
    """
    if not relevant_ids or k <= 0:
        return 0.0
    hits = 0
    score = 0.0
    for rank, mid in enumerate(ranked_ids[:k], start=1):
        if mid in relevant_ids:
            hits += 1
            score += hits / rank
    return score / hits if hits else 0.0


def aggregate(metrics: list[float]) -> float:
    """Mean of per-query metric values; 0.0 for an empty run."""
    return sum(metrics) / len(metrics) if metrics else 0.0


class QueryEvalResult(BaseModel):
    """Outcome of one benchmark query."""

    query_id: str
    tier: str
    query: str
    expected_path: str
    ranked_ids: list[int] = Field(default_factory=list)
    relevant_ids: list[int] = Field(default_factory=list)
    hit_rate: float = 0.0
    mrr: float = 0.0
    context_precision: float = 0.0
    # full-mode extras
    response: str = ""
    faithfulness: float | None = None
    relevancy: float | None = None
    tokens: int = 0
    cost_usd: float = 0.0


class BenchmarkSummary(BaseModel):
    """Aggregated benchmark run with config snapshot and optional delta."""

    label: str
    mode: str
    config_snapshot: dict = Field(default_factory=dict)
    n_queries: int = 0
    hit_rate: float = 0.0
    mrr: float = 0.0
    context_precision: float = 0.0
    faithfulness: float | None = None
    relevancy: float | None = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    per_query: list[QueryEvalResult] = Field(default_factory=list)
    delta: dict | None = None
