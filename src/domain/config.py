"""Dynamic Architecture Experimentation Control Plane Configuration."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.domain.memory import MESSAGE_WINDOW  # window default lives in domain memory
from src.domain.moods import MOOD_BOOST_CALIBRATION  # boost calibration single source


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
        default="google/gemini-3.5-flash-lite",
        description="Router LLM model ID (#29: measured 91% vs 3B's 66% routing accuracy). "
        "One notch below the ~flash-latest alias (serves gemini-3.8-flash); dated id = "
        "frozen, reproducible (#97). With ZAI_API_KEY set, google-gemini ids resolve "
        "to zai_model on z.ai; without, verbatim to OpenRouter.",
    )
    synthesis_model: str = Field(
        default="google/gemini-3.5-flash-lite",
        description="Synthesis LLM model ID (#30: Flash default — cheap, strong). "
        "Same z.ai-first resolution as the router (#97).",
    )
    zai_model: str = Field(
        default="glm-5.3-flash",
        description="Model used on z.ai when ZAI_API_KEY is present (#97): "
        "~google/-prefixed router/synthesis ids resolve to this. GLM-5.3-flash, "
        "proven in the #85 prototype (13/13 turns, 0 schema failures).",
    )
    pin_router_config_id: bool = Field(
        default=False,
        description="#89 sweep isolation: router keeps the config id verbatim via "
        "OpenRouter even under ZAI_API_KEY (google-family candidates must not "
        "silently become glm). Recorded in the run envelope = explicit identity.",
    )
    pin_synthesis_config_id: bool = Field(
        default=False,
        description="#89 sweep isolation: synthesizer keeps the config id verbatim "
        "(vary the router ONLY). Explicit in the experiment identity.",
    )
    reasoning_effort: str = Field(default="low", description="Reasoning effort: none, low, medium, high")
    #: #106/D15: the v2 Understand model (baseline candidate); #107 sweeps it.
    v2_router_model: str = Field(default="glm-5.3-flash")
    #: #113: secondary Understand model — one attempt when the primary call
    #: fails (transport, or schema after the C12 budget). Traced when fired.
    v2_router_fallback_model: str = Field(default="gemini-3.5-flash-lite")
    #: #107 sweep isolation: pin exact config id, no provider swap (v1's
    #: pin_router_config_id pattern).
    pin_v2_router_config_id: bool = Field(default=False)
    #: #150: how the v2 Understand call returns structure. tool_call = the
    #: schema bound as a forced submit-tool (bind_tools + tool_choice) with
    #: strict arg validation — z.ai honors the force (live-probed);
    #: structured_output = with_structured_output(function_calling,
    #: include_raw=True) with the SAME strict validation on raw args (the
    #: wrapper's parsed is never trusted — pydantic silently ignores extras);
    #: prompt_json = the pre-#150 prose path (fenced JSON), kept as the
    #: escape hatch. ADR 0004 tunable; default flipped only on live evidence.
    v2_understand_transport: Literal[
        "prompt_json", "tool_call", "structured_output"
    ] = Field(default="tool_call")
    routing_stack: Literal["v1", "v2"] = Field(
        default="v1",
        description="Routing Stack in force (#83): v1 = gated router (production); "
        "v2 = LLM Understanding (#106). Swept by the evaluation "
        "harness (--stack), flippable locally via MAYA_ROUTING_STACK — never a Lab knob.",
    )
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
    #: #137: MoodProfile translation (canonical mood -> floors + boosts +
    #: phrasebook). Off = pre-#137 flavor-only behavior for A/B runs.
    mood_profiles_enabled: bool = Field(
        default=True,
        description="Apply MoodProfile floors/boosts/phrasebook at retrieval (#137)",
    )
    mood_boost_scale: float = Field(
        default=1.0, ge=0.0, le=3.0,
        description="Global multiplier on MoodProfile boost terms (#137; 0 = floors only)",
    )
    #: #137: RRF boost calibration (ADR 0004). One full boost term ≈ NORMALIZER
    #: * term_cap in score units — a real tilt inside the candidate pool that
    #: never manufactures candidates (pool membership is unchanged). Defaults
    #: derive from MOOD_BOOST_CALIBRATION (single literal source).
    mood_boost_normalizer: float = Field(
        default=MOOD_BOOST_CALIBRATION["normalizer"], gt=0.0,
        description="Scales MoodBoostSpec terms into RRF score units (#137)",
    )
    mood_boost_runtime_weight: float = Field(
        default=MOOD_BOOST_CALIBRATION["runtime_weight"], ge=0.0,
        description="Fixed boost term when runtime >= profile runtime_boost_min (#137)",
    )
    mood_boost_term_cap: float = Field(
        default=MOOD_BOOST_CALIBRATION["term_cap"], gt=0.0,
        description="Absolute cap on the pre-normalizer boost term (#137)",
    )
    mood_boost_popularity_log_divisor: float = Field(
        default=MOOD_BOOST_CALIBRATION["popularity_log_divisor"], gt=0.0,
        description="log1p(popularity) divisor before clamping to [0, 1] (#137)",
    )
    mood_boost_revenue_log_divisor: float = Field(
        default=MOOD_BOOST_CALIBRATION["revenue_log_divisor"], gt=0.0,
        description="log1p(revenue) divisor before clamping to [0, 1] (#137)",
    )

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
    #: #93/D16: the in-thread message window (trim node + read model).
    message_window: int = Field(
        default=MESSAGE_WINDOW, ge=2,
        description="Conversation message window kept in the LangGraph thread "
        "(trim node, #93/D16); the read model trims to the same size",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Router low-confidence fallback threshold (#12): decisions "
        "below this confidence degrade to the heuristic fallback",
    )
    cwa_guardrail_enabled: bool = Field(default=True, description="Enforce Closed-World Assumption XML grounding")
    funnel_retrieve_axes: int = Field(
        default=2,
        ge=1,
        le=5,
        description="#53: answered narrowing axes at which the funnel retrieves "
        "immediately (no confirm-before-retrieve turn)",
    )
    era_old_year_max: int = Field(
        default=2000,
        ge=1970,
        le=2026,
        description="#42: what a vague 'old/classic movie' means — deterministic "
        "year_max applied when the era vocabulary fires mid-funnel",
    )
    era_recent_year_min: int = Field(
        default=2015,
        ge=1970,
        le=2026,
        description="#42: what a vague 'recent/latest movie' means — deterministic "
        "year_min applied when the era vocabulary fires mid-funnel",
    )
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


def matching_preset(config: ExperimentConfig) -> PresetType | None:
    """The preset the config matches EXACTLY, else None (custom state).

    A preset attribution must never lie (#30 grilling Q3, reused by #59 run
    identity): the match is against a PRISTINE baseline (defaults + preset),
    so ANY manual knob edit clears it.
    """
    for preset in (
        PresetType.PRODUCTION_HYBRID,
        PresetType.FAST_BUDGET,
        PresetType.NAIVE_BASELINE,
    ):
        if ExperimentConfig().apply_preset(preset) == config:
            return preset
    return None
