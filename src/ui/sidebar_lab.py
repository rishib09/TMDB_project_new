"""Experimentation Lab (issue #7): live config knobs, presets, budget meter.

Lives in the collapsible sidebar. Streamlit widgets edit an
ExperimentConfig copy; on change the session rebuilds the graph (next turn
runs on the new architecture — no hidden state).
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


def knob_editor(config: ExperimentConfig, version: int) -> ExperimentConfig | None:
    """Renders grouped knobs; returns an edited config copy or None if unchanged.

    Widget keys embed ``version`` (bumped when a preset is applied) so the
    widgets remount with the new config instead of forcing their stale
    Streamlit-held values back — that was the preset-button bug.
    """
    v = version
    edited = config.model_copy(deep=True)
    changed = False

    with st.expander("Models", expanded=True):
        router = st.selectbox(
            "Router model",
            _ROUTER_MODELS,
            index=_ROUTER_MODELS.index(config.router_model)
            if config.router_model in _ROUTER_MODELS else 0,
            key=f"router_model_{v}",
        )
        synth = st.selectbox(
            "Synthesis model",
            _SYNTH_MODELS,
            index=_SYNTH_MODELS.index(config.synthesis_model)
            if config.synthesis_model in _SYNTH_MODELS else 0,
            key=f"synth_model_{v}",
        )
        temp = st.slider("Temperature", 0.0, 1.0, config.temperature, 0.1, key=f"temp_{v}")
        effort = st.select_slider(
            "Reasoning effort", ["none", "low", "medium", "high"], config.reasoning_effort,
            key=f"effort_{v}",
        )
        for old, new in [
            (config.router_model, router), (config.synthesis_model, synth),
            (config.temperature, temp), (config.reasoning_effort, effort),
        ]:
            if old != new:
                changed = True
        edited.router_model, edited.synthesis_model = router, synth
        edited.temperature, edited.reasoning_effort = temp, effort

    with st.expander("Retrieval", expanded=True):
        top_k = st.slider("Retrieval top-K", 1, 20, config.retrieval_top_k, key=f"topk_{v}")
        alpha = st.slider(
            "hybrid_alpha (0 = lexical, 1 = dense)", 0.0, 1.0, config.hybrid_alpha, 0.05,
            key=f"alpha_{v}",
        )
        reranker = st.checkbox(
            "Reranker (measured slower and weaker than RRF fusion)", config.reranker_enabled,
            key=f"reranker_{v}",
        )
        reranker_model = st.selectbox(
            "Reranker model", _RERANKERS, disabled=not reranker, key=f"reranker_model_{v}"
        )
        for old, new in [
            (config.retrieval_top_k, top_k), (config.hybrid_alpha, alpha),
            (config.reranker_enabled, reranker), (config.reranker_model, reranker_model),
        ]:
            if old != new:
                changed = True
        edited.retrieval_top_k, edited.hybrid_alpha = top_k, alpha
        edited.reranker_enabled, edited.reranker_model = reranker, reranker_model

    with st.expander("Routing and Guardrails", expanded=True):
        attempts = st.slider(
            "Route max attempts (bounded re-route cycle)", 1, 5, config.route_max_attempts,
            key=f"attempts_{v}",
        )
        cwa = st.checkbox(
            "Closed-world-assumption grounding enforcement", config.cwa_guardrail_enabled,
            key=f"cwa_{v}",
        )
        for old, new in [
            (config.route_max_attempts, attempts), (config.cwa_guardrail_enabled, cwa)
        ]:
            if old != new:
                changed = True
        edited.route_max_attempts, edited.cwa_guardrail_enabled = attempts, cwa

    return edited if changed else None


def render_budget_meter(session) -> None:
    """Session tokens vs the 15k cap + weekly $ spend vs the $10 cap (#8)."""
    used = session.conversation.session_tokens
    ratio = min(used / SessionTokenLimiter.SESSION_CAP, 1.0)
    st.progress(ratio, text=f"Session tokens: {used:,} / {SessionTokenLimiter.SESSION_CAP:,}")
    if session.limiter.check_current().verdict.value == "suspicious":
        st.warning("Near the session token cap — wrap up this session soon.")

    tracker = session.budget_tracker
    try:
        weekly_spend = tracker.weekly_spend()
    except Exception:  # sink read failure must not break the sidebar
        return
    spend_ratio = min(weekly_spend / tracker.WEEKLY_CAP_USD, 1.0)
    st.progress(
        spend_ratio,
        text=f"Weekly API spend: ${weekly_spend:.2f} / ${tracker.WEEKLY_CAP_USD:.2f}",
    )
    weekly_verdict = tracker.verdict_for(weekly_spend)
    if weekly_verdict.value == "suspicious":
        st.warning("Weekly API budget nearing its cap — Maya will pause when it's exhausted.")
    elif weekly_verdict.value == "blocked":
        st.error("Weekly API budget exhausted — spend resets on Monday.")


def render_lab(session) -> None:
    st.markdown("**Experimentation Lab**")
    st.caption("Live architecture knobs — applied from the next message.")

    preset_cols = st.columns(3)
    preset_map = [
        (preset_cols[0], PresetType.FAST_BUDGET, "Fast Budget"),
        (preset_cols[1], PresetType.PRODUCTION_HYBRID, "Production"),
        (preset_cols[2], PresetType.NAIVE_BASELINE, "Naive Baseline"),
    ]
    for col, preset, label in preset_map:
        if col.button(label, use_container_width=True):
            session.apply_preset(preset)
            st.toast(f"Preset applied: {label}")

    st.caption(
        "Fast Budget: 3B router + Flash-Lite synthesis, dense-only, small context. "
        "Production: 3B router + 70B synthesis, 50/50 dense-lexical RRF. "
        "Naive Baseline: 3B end-to-end, dense-only."
    )

    edited = knob_editor(session.config, session.config_version)
    if edited is not None:
        session.replace_config(edited)
        st.toast("Configuration updated — pipeline rebuilt")

    st.caption("Evaluation sweeps: python -m src.evals.runner --mode retrieval --versions v1_1_enriched")
    st.divider()
    render_budget_meter(session)
