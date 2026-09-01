"""Evaluation harness (issue #6): IR metrics, LLM-as-a-judge, benchmark runner."""

from src.evals.judge import MayaJudge
from src.evals.metrics import (
    BenchmarkSummary,
    QueryEvalResult,
    context_precision_at_k,
    hit_rate_at_k,
    mrr_at_k,
)
from src.evals.runner import BenchmarkRunner, load_dataset

__all__ = [
    "MayaJudge",
    "BenchmarkSummary",
    "QueryEvalResult",
    "context_precision_at_k",
    "hit_rate_at_k",
    "mrr_at_k",
    "BenchmarkRunner",
    "load_dataset",
]
