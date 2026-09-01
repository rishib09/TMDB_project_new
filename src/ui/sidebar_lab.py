"""Experimentation Lab (issue #7): live config knobs, presets, budget meter.

Streamlit widgets edit an ExperimentConfig copy; on change the session
rebuilds the graph (next turn runs on the new architecture — no hidden
state). /admin in chat flips the session into this panel.
"""

import streamlit as st

from src.domain.config import ExperimentConfig, PresetType
from src.maya.guardrails import SessionTokenLimiter

_ROUTER_MODELS = [
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.0-flash-lite",
]
_SYNTH_MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.2-3b-instruct",
    "google/gemini-2.0-flash-lite",
]
_RERANKERS = ["ms-marco-MiniLM-L-12-v2", "ms-marco-TinyBERT-L-2-v2", "ce-esci-MiniLM-L12-v2"]


def knob_editor(config: ExperimentConfig) -> ExperimentConfig | None:
    """Renders grouped knobs; returns an edited config copy or None if unchanged.

    Pure-ish (unit-testable via the return contract): the widgets are
    Streamlit-bound, but the change-detection contract is simple.
    """
    edited = config.model_copy(deep=True)
    changed = False

    with st.expander("🤖 Models", expanded=True):
        router = st.selectbox("Router model", _ROUTER_MODELS,
                              index=_ROUTER_MODELS.index(config.router_model)
                              if config.router_model in _ROUTER_MODELS else 0)
        synth = st.selectbox("Synthesis model", _SYNTH_MODELS,
                             index=_SYNTH_MODELS.index(config.synthesis_model)
                             if config.synthesis_model in _SYNTH_MODELS else 0)
        temp = st.slider("Temperature", 0.0, 1.0, config.temperature, 0.1)
        effort = st.select_slider("Reasoning effort",
                                  ["none", "low", "medium", "high"], config.reasoning_effort)
        for name, old, new in [("router", config.router_model, router),
                               ("synthesis", config.synthesis_model, synth),
                               ("temperature", config.temperature, temp),
                               ("effort", config.reasoning_effort, effort)]:
            if old != new:
                changed = True
        edited.router_model, edited.synthesis_model = router, synth
        edited.temperature, edited.reasoning_effort = temp, effort

    with st.expander("🔎 Retrieval", expanded=True):
        top_k = st.slider("Retrieval top-K", 1, 20, config.retrieval_top_k)
        alpha = st.slider("hybrid_alpha (0=BM25, 1=Dense)", 0.0, 1.0, config.hybrid_alpha, 0.05)
        reranker = st.checkbox("Reranker (flashrank — measured slower & weaker than RRF)",
                               config.reranker_enabled)
        reranker_model = st.selectbox("Reranker model", _RERANKERS,
                                      disabled=not reranker)
        for old, new in [(config.retrieval_top_k, top_k), (config.hybrid_alpha, alpha),
                         (config.reranker_enabled, reranker),
                         (config.reranker_model, reranker_model)]:
            if old != new:
                changed = True
        edited.retrieval_top_k, edited.hybrid_alpha = top_k, alpha
        edited.reranker_enabled, edited.reranker_model = reranker, reranker_model

    with st.expander("🛡️ Routing & Guardrails", expanded=True):
        attempts = st.slider("Route max attempts (bounded re-route cycle)", 1, 5,
                             config.route_max_attempts)
        cwa = st.checkbox("CWA grounding enforcement", config.cwa_guardrail_enabled)
        for old, new in [(config.route_max_attempts, attempts),
                         (config.cwa_guardrail_enabled, cwa)]:
            if old != new:
                changed = True
        edited.route_max_attempts, edited.cwa_guardrail_enabled = attempts, cwa

    return edited if changed else None


def render_budget_meter(session) -> None:
    """Session token pressure vs the 15k cap (#8) — visible, no surprises."""
    used = session.conversation.session_tokens
    ratio = min(used / SessionTokenLimiter.SESSION_CAP, 1.0)
    st.progress(ratio, text=f"Session tokens: {used:,} / {SessionTokenLimiter.SESSION_CAP:,}")
    if session.limiter.check_current().verdict.value == "suspicious":
        st.warning("Near token cap — wrap up this session soon.")


def render_lab(session) -> None:
    st.subheader("🧪 Experimentation Lab")
    st.caption(
        "Live architecture knobs (ADR 0004): changes rebuild the pipeline and "
        "apply from the NEXT message. Type `/admin` in chat to jump here."
    )

    st.markdown("**1-click presets**")
    preset_cols = st.columns(3)
    preset_map = [
        (preset_cols[0], PresetType.FAST_BUDGET, "Fast/Budget"),
        (preset_cols[1], PresetType.PRODUCTION_HYBRID, "Production"),
        (preset_cols[2], PresetType.NAIVE_BASELINE, "Naive baseline"),
    ]
    for col, preset, label in preset_map:
        if col.button(label, use_container_width=True):
            session.apply_preset(preset)
            st.toast(f"Preset applied: {label}", icon="✅")

    edited = knob_editor(session.config)
    if edited is not None:
        session.replace_config(edited)
        st.toast("Config updated — pipeline rebuilt", icon="🔧")

    st.divider()
    render_budget_meter(session)

    st.divider()
    st.markdown("**Evaluation entry points**")
    st.markdown("📊 Metrics live in the **Evals** tab; traces in the **Traces** tab.")
    st.markdown(
        "CLI sweeps: `python -m src.evals.runner --mode retrieval --versions v1_1_enriched`"
    )
