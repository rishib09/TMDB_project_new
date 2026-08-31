"""Dynamic Architecture Experimentation Control Plane Configuration."""

from enum import Enum
from pydantic import BaseModel, Field, model_validator


class PresetType(str, Enum):
    """Predefined Architecture Presets."""
    FAST_BUDGET = "FAST_BUDGET"
    PRODUCTION_HYBRID = "PRODUCTION_HYBRID"
    NAIVE_BASELINE = "NAIVE_BASELINE"
    CUSTOM = "CUSTOM"


class ExperimentConfig(BaseModel):
    """Configuration state for live experimentation control plane and evaluation runs."""
    # Model Selection & Inference
    router_model: str = Field(default="meta-llama/llama-3.2-3b-instruct", description="Router LLM model ID")
    synthesis_model: str = Field(default="meta-llama/llama-3.3-70b-instruct", description="Synthesis LLM model ID")
    reasoning_effort: str = Field(default="low", description="Reasoning effort: none, low, medium, high")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")

    # Retrieval & Indexing Knobs
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", description="Embedding model name")
    token_budget: int = Field(
        default=256,
        description="Max token budget for dense text serialization (256, 512, 1024, 2048, 4096)"
    )
    chunking_strategy: str = Field(default="enriched_metadata", description="baseline | enriched_metadata")
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="0.0 = BM25 sparse, 1.0 = Dense vector")
    reranker_enabled: bool = Field(default=True, description="Enable FlashRank CPU reranker")
    retrieval_top_k: int = Field(default=5, ge=1, le=20, description="Number of final context movies")

    # Memory & Guardrails
    multi_turn_mode: str = Field(default="fused_single_pass", description="fused_single_pass | dedicated_2step_llm")
    memory_strategy: str = Field(default="sliding_window_with_entity", description="Memory retention strategy")
    cwa_guardrail_enabled: bool = Field(default=True, description="Enforce Closed-World Assumption XML grounding")

    @model_validator(mode="after")
    def validate_token_budget_against_model(self) -> "ExperimentConfig":
        """Clamps token budget if it exceeds the active embedding model's context ceiling to prevent silent truncation."""
        model_ceilings = {
            "sentence-transformers/all-MiniLM-L6-v2": 256,
            "BAAI/bge-small-en-v1.5": 512,
            "BAAI/bge-base-en-v1.5": 512,
            "nomic-ai/nomic-embed-text-v1.5": 8192,
            "text-embedding-3-small": 8192,
        }
        max_allowed = model_ceilings.get(self.embedding_model, 512)
        if self.token_budget > max_allowed:
            self.token_budget = max_allowed
        return self

    def apply_preset(self, preset: PresetType) -> "ExperimentConfig":
        """Reconfigures parameters to standard predefined benchmark baselines."""
        if preset == PresetType.FAST_BUDGET:
            self.router_model = "meta-llama/llama-3.2-3b-instruct"
            self.synthesis_model = "google/gemini-2.0-flash-lite"
            self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            self.token_budget = 256
            self.chunking_strategy = "baseline"
            self.hybrid_alpha = 1.0  # Dense only
            self.reranker_enabled = False
            self.retrieval_top_k = 3
        elif preset == PresetType.PRODUCTION_HYBRID:
            self.router_model = "meta-llama/llama-3.2-3b-instruct"
            self.synthesis_model = "meta-llama/llama-3.3-70b-instruct"
            self.embedding_model = "BAAI/bge-small-en-v1.5"
            self.token_budget = 512
            self.chunking_strategy = "enriched_metadata"
            self.hybrid_alpha = 0.5  # 50/50 Dense + Sparse RRF
            self.reranker_enabled = True
            self.retrieval_top_k = 5
        elif preset == PresetType.NAIVE_BASELINE:
            self.router_model = "meta-llama/llama-3.2-3b-instruct"
            self.synthesis_model = "meta-llama/llama-3.2-3b-instruct"
            self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            self.token_budget = 256
            self.chunking_strategy = "baseline"
            self.hybrid_alpha = 1.0
            self.reranker_enabled = False
            self.retrieval_top_k = 5
        return self
