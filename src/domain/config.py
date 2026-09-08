"""Dynamic Architecture Experimentation Control Plane Configuration."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PresetType(StrEnum):
    """Predefined Architecture Presets."""
    FAST_BUDGET = "FAST_BUDGET"
    PRODUCTION_HYBRID = "PRODUCTION_HYBRID"
    NAIVE_BASELINE = "NAIVE_BASELINE"
    CUSTOM = "CUSTOM"


class ExperimentConfig(BaseModel):
    """Configuration state for live experimentation control plane and evaluation runs."""
    # Model Selection & Inference
    router_model: str = Field(
        default="~google/gemini-flash-latest",
        description="Router LLM model ID (#29: measured 91% vs 3B's 66% routing accuracy; "
        "the ~ alias always resolves to the newest Flash on OpenRouter)",
    )
    synthesis_model: str = Field(
        default="~google/gemini-flash-latest",
        description="Synthesis LLM model ID (#30: Flash default — cheap, strong, "
        "same always-newest alias as the router)",
    )
    reasoning_effort: str = Field(default="low", description="Reasoning effort: none, low, medium, high")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")

    # Retrieval & Indexing Knobs
    #: #11/#30: the two decoupled dense axes. Together they name the Chroma
    #: collection ({column_preset}_{embedding_profile}) AND the provider that
    #: embeds user queries — one knob pair, structurally consistent.
    embedding_profile: str = Field(
        default="gemini_embedding_2",
        description="Cloud embedding profile (#11): lfm_free | nemotron_free | "
        "gemini_embedding_2 (ADR 0008 production winner)",
    )
    column_preset: Literal["minimal", "full"] = Field(
        default="full",
        description="Serialized column set for dense documents (#11): minimal = "
        "overview+keywords+genres+title+director; full = minimal + cast + tagline",
    )
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", description="Legacy local embedding model (v1_* eval collections)")
    token_budget: int = Field(
        default=256,
        description="Max token budget for dense text serialization (256, 512, 1024, 2048, 4096)"
    )
    chunking_strategy: str = Field(default="enriched_metadata", description="baseline | enriched_metadata")
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="0.0 = BM25 sparse, 1.0 = Dense vector")
    reranker_enabled: bool = Field(
        default=False,
        description="Enable FlashRank CPU cross-encoder reranking (A/B: off — hurts hit-rate)"
    )
    reranker_model: str = Field(
        default="ms-marco-MiniLM-L-12-v2",
        description="FlashRank model when enabled (best of 4 measured: 71% vs RRF 86%)"
    )
    retrieval_top_k: int = Field(default=5, ge=1, le=20, description="Number of final context movies")

    # Memory & Guardrails
    multi_turn_mode: str = Field(default="fused_single_pass", description="fused_single_pass | dedicated_2step_llm")
    route_max_attempts: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Bounded re-route cycle (#5): max routing attempts when the "
        "router signals a heuristic fallback (low confidence / API error) — "
        "measured: iterative re-routing resolves a share of routing failures (#12)",
    )
    memory_strategy: str = Field(default="sliding_window_with_entity", description="Memory retention strategy")
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Router low-confidence fallback threshold (#12): decisions "
        "below this confidence degrade to the heuristic fallback",
    )
    cwa_guardrail_enabled: bool = Field(default=True, description="Enforce Closed-World Assumption XML grounding")
    judge_model: str = Field(
        default="meta-llama/llama-3.3-70b-instruct",
        description="LLM-as-a-judge model for eval faithfulness/relevancy (#6)",
    )

    @model_validator(mode="after")
    def validate_token_budget_against_model(self) -> "ExperimentConfig":
        """Clamps token budget if it exceeds the active embedding model's context ceiling to prevent silent truncation."""
        model_ceilings = {
            "sentence-transformers/all-MiniLM-L6-v2": 256,
            "BAAI/bge-small-en-v1.5": 512,
            "BAAI/bge-base-en-v1.5": 512,
            "snowflake/snowflake-arctic-embed-s": 512,
            "jinaai/jina-embeddings-v2-base-en": 8192,
            "nomic-ai/nomic-embed-text-v1.5": 8192,
            "snowflake/snowflake-arctic-embed-m": 512,
            "text-embedding-3-small": 8192,
        }
        max_allowed = model_ceilings.get(self.embedding_model, 512)
        if self.token_budget > max_allowed:
            self.token_budget = max_allowed
        return self

    def apply_preset(self, preset: PresetType) -> "ExperimentConfig":
        """Reconfigures the embedding combo + retrieval knobs (#30 grilling).

        Presets own the embedding axis and retrieval shape ONLY — router and
        synthesis models are independent knobs (no vendor coupling; the user
        picks chat models freely). Mapping per ADR 0008: each model's best
        measured column preset.
        """
        if preset == PresetType.FAST_BUDGET:
            self.embedding_profile = "nemotron_free"  # free, 71% hit@5
            self.column_preset = "full"  # nemotron's best cell (MRR .631 vs .560)
            self.hybrid_alpha = 1.0  # Dense only
            self.reranker_enabled = False
            self.retrieval_top_k = 3
        elif preset == PresetType.PRODUCTION_HYBRID:
            self.embedding_profile = "gemini_embedding_2"  # 100% hit@5, MRR .964
            self.column_preset = "full"
            self.hybrid_alpha = 0.5  # 50/50 Dense + Sparse RRF
            # Reranker stays OFF even in the quality preset: measured 71% vs
            # pure-RRF 86% hit@5 on golden queries (2026-08-31 A/B, issue #4).
            self.reranker_enabled = False
            self.retrieval_top_k = 5
        elif preset == PresetType.NAIVE_BASELINE:
            self.embedding_profile = "lfm_free"  # the measured floor (43%)
            self.column_preset = "minimal"  # lfm's best cell (full collapses to .262)
            self.hybrid_alpha = 1.0
            self.reranker_enabled = False
            self.retrieval_top_k = 5
        return self
