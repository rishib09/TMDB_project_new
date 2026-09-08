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


def routing_accuracy(results: list["QueryEvalResult"]) -> float:
    """Share of queries whose routed intent matched the expected intent.

    Only rows that were actually routed count (intent_correct is not None);
    an empty run returns 0.0 like every other metric here.
    """
    scored = [r for r in results if r.intent_correct is not None]
    if not scored:
        return 0.0
    return sum(1.0 for r in scored if r.intent_correct) / len(scored)


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
    # routing-mode extras (#29): live router vs expected_intent
    routed_intent: str = ""
    intent_correct: bool | None = None
    confidence: float | None = None
    is_fallback: bool = False


class BenchmarkSummary(BaseModel):
    """Aggregated benchmark run with config snapshot and optional delta."""

    label: str
    mode: str
    config_snapshot: dict = Field(default_factory=dict)
    # #59 run identity + staleness stamps (#54 grill D1/D7)
    config_hash: str = ""
    preset: str = "custom"
    dataset_version: str = ""
    collection: str = ""
    # #59 sweep attribution (#54 grill D3): set when this run is one point of
    # a one-factor-at-a-time sweep against the Production baseline.
    sweep: dict | None = None
    n_queries: int = 0
    hit_rate: float = 0.0
    mrr: float = 0.0
    context_precision: float = 0.0
    faithfulness: float | None = None
    relevancy: float | None = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    # routing-mode aggregates (#29)
    routing_accuracy: float | None = None
    routing_per_intent: dict[str, float] | None = None
    fallback_count: int = 0
    per_query: list[QueryEvalResult] = Field(default_factory=list)
    delta: dict | None = None
