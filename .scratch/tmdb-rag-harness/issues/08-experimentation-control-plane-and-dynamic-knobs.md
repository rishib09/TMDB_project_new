Type: grilling
Status: resolved

## Question

How should the Architecture Experimentation Control Plane (`/admin` panel or expandable Sidebar Lab) be structured, what is the exact Pydantic schema for dynamic configuration knobs (`ExperimentConfig` covering Models, Reasoning Effort, Memory Strategy, Reformulation Engine, Fallback Mode, Embedder Model, Hybrid Fusion Alpha, Chunking Representation, Reranker, Context Format, and Guardrails), how do live parameter updates propagate to Maya's pipeline, and how are resulting metric/latency deltas recorded in the Eval Dashboard and Trace Inspector?

## Answer

### 1. The Complete `ExperimentConfig` Schema (`src/domain/config.py`)
```python
from typing import Literal
from pydantic import BaseModel, Field

class ExperimentConfig(BaseModel):
    # Model Selection & Reasoning
    router_model: str = "meta-llama/llama-3.2-3b-instruct"
    synthesis_model: str = "meta-llama/llama-3.3-70b-instruct"
    reasoning_effort: Literal["none", "low", "medium"] = "none"

    # Multi-Turn & Memory
    reformulation_mode: Literal["fused_single_pass", "dedicated_2step_llm", "raw_passthrough"] = "fused_single_pass"
    memory_strategy: Literal["sliding_window_with_entity", "pure_sliding_window", "rolling_summarizer"] = "sliding_window_with_entity"
    fallback_mode: Literal["soft_boosted_hybrid", "pure_vector_fallback", "interactive_clarify"] = "soft_boosted_hybrid"

    # Retrieval & Indexing
    embedding_model: Literal["bge-small-en-v1.5", "all-MiniLM-L6-v2"] = "bge-small-en-v1.5"
    chunk_representation: Literal["multi_field_hybrid", "structured_markdown", "raw_overview"] = "multi_field_hybrid"
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)  # 0.0=Vector, 1.0=BM25
    reranker_enabled: bool = True                             # FlashRank CPU ON/OFF
    retrieval_top_k: int = Field(default=5, ge=1, le=15)

    # Prompting & Guardrails
    context_format: Literal["xml_tags", "structured_json", "markdown_cards"] = "xml_tags"
    strict_cwa_mode: bool = True                              # Closed-World Assumption ON/OFF
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    enable_streaming: bool = True
    max_generation_tokens: int = Field(default=500, ge=150, le=1000)
```

### 2. Curated 1-Click Preset Profiles
- ⚡ **"Fast Budget (~180ms)"**: `MiniLM` + `Fused Router` + `FlashRank OFF` + `Top-K=3` + `Llama-3.2-3b` + `xml_tags`.
- 🏆 **"Production Hybrid (Top Accuracy)"**: `BGE-Small` + `Dedicated 2-Step Rewriter` + `Entity Memory` + `FlashRank ON` + `Top-K=5` + `Llama-3.3-70b`.
- ⚠️ **"Naive Baseline (Flawed / Educational)"**: `MiniLM` + `Raw Overview` + `No Entity Memory` + `FlashRank OFF` (shows failure on cast/director queries).
- 🛠️ **"Custom Lab"**: Granular access to all sliders and selectors.

### 3. Dual UI Control Plane Placement in Streamlit
- **Sidebar Accordion**: Always-accessible "🧪 Architecture Lab" with live model selector, preset buttons, and parameter sliders.
- **Chat Slash Command (`/admin` or `/config`)**: Typing `/admin` inside Maya's chat window renders an interactive configuration dialog with live telemetry gauges.
- **URL Parameter (`?mode=admin`)**: Opens advanced developer diagnostics on load.

### 4. Real-Time Telemetry & Eval Dashboard Propagation
- Toggling parameters dynamically updates `st.session_state["active_config"]`.
- The **Trace Inspector (Tab 3)** instantly renders the modified span execution DAG (e.g. adding/removing `QueryRewriter` or `FlashRank` spans).
- The **Evals Dashboard (Tab 2)** includes a **"🚀 Benchmark Current Configuration"** button that executes the 35-query benchmark against the live configuration and appends a `[Custom Config]` comparison column against frozen baseline milestones.
