"""Experimentation Lab (issue #7, #30): live config knobs, presets, budget meter.

Lives in the collapsible sidebar. Streamlit widgets edit an
ExperimentConfig copy; on change the session rebuilds the graph (next turn
runs on the new architecture — no hidden state).

#30 grilled design: preset buttons own the EMBEDDING axis only (combo =
column preset x embedding profile, per ADR 0008's best-cell mapping); chat
models are independent knobs. The active preset renders red (theme primary)
only while the config matches it exactly — any manual edit clears it.
"""

import json
import urllib.request
from functools import lru_cache

import streamlit as st

from src.domain.config import ExperimentConfig, PresetType
from src.indexing.embeddings import MODEL_PROFILES, collection_name
from src.maya.guardrails import SessionTokenLimiter

_ROUTER_MODELS = [
    "~google/gemini-flash-latest",  # #29 upgrade (~ = OpenRouter newest-Flash alias)
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.2-3b-instruct",
]
_SYNTH_MODELS = [
    "~google/gemini-flash-latest",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.2-3b-instruct",
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]
_RERANKERS = ["ms-marco-MiniLM-L-12-v2", "ms-marco-TinyBERT-L-2-v2", "ce-esci-MiniLM-L12-v2"]

#: The 6 dense combos the Lab exposes (#30), best-first, with the ADR 0008
#: measured numbers inline — bad combos warn by information, not lockout.
_COMBOS: list[tuple[str, str, str]] = [
    ("full", "gemini_embedding_2", "hit@5 100% · MRR .96"),
    ("minimal", "gemini_embedding_2", "hit@5 100% · MRR .95"),
    ("full", "nemotron_free", "hit@5 71% · MRR .63 · free"),
    ("minimal", "nemotron_free", "hit@5 71% · MRR .56 · free"),
    ("minimal", "lfm_free", "hit@5 43% · MRR .43 · free"),
    ("full", "lfm_free", "hit@5 43% · MRR .26 · free"),
]
_PROFILE_LABELS = {"gemini_embedding_2": "Gemini", "nemotron_free": "Nemotron", "lfm_free": "LFM"}

_MINIMAL_COLUMNS = "overview + keywords + genres + title + director"
_FULL_COLUMNS = _MINIMAL_COLUMNS + " + top-10 cast + tagline"

_PRESET_HELP = {
    PresetType.PRODUCTION_HYBRID: (
        "Embeddings: Gemini · full (100% hit@5, MRR .96 — ADR 0008 winner). "
        f"Full columns: {_FULL_COLUMNS}. Retrieval: 50/50 dense+lexical RRF, top-5."
    ),
    PresetType.FAST_BUDGET: (
        "Embeddings: Nemotron · full — free tier, 71% hit@5, MRR .63 (its best "
        f"column set). Full columns: {_FULL_COLUMNS}. Retrieval: dense-only, top-3."
    ),
    PresetType.NAIVE_BASELINE: (
        "Embeddings: LFM · minimal — the measured floor (43% hit@5; the full "
        f"columns collapse it to MRR .26). Minimal columns: {_MINIMAL_COLUMNS}. "
        "Retrieval: dense-only, top-5."
    ),
}


@lru_cache(maxsize=1)
def available_model_ids() -> frozenset[str] | None:
    """OpenRouter's live model ids, fetched once per process (#30 grilling).

    Fail-open by contract: None means "list unavailable — assume everything
    works" (a downed catalog must not empty the Lab's dropdowns). Motivated
    by gemini-2.0-flash-lite silently delisting while hardcoded here.
    """
    try:
        with urllib.request.urlopen(
            "https://openrouter.ai/api/v1/models", timeout=4
        ) as response:
            payload = json.load(response)
        return frozenset(m["id"] for m in payload["data"])
    except Exception:
        return None


def usable_models(candidates: list[str], current: str) -> list[str]:
    """Drops delisted slugs from a dropdown; the active model always stays."""
    available = available_model_ids()
    if available is None:
        return candidates
    return [m for m in candidates if m in available or m == current]


def matching_preset(config: ExperimentConfig) -> PresetType | None:
    """The preset the config matches EXACTLY, else None (custom state).

    A highlighted preset button must never lie (#30 grilling Q3): the match
    is against a PRISTINE baseline (defaults + preset), so ANY manual knob
    edit — preset-owned or not — clears the highlight.
    """
    for preset in (
        PresetType.PRODUCTION_HYBRID,
        PresetType.FAST_BUDGET,
        PresetType.NAIVE_BASELINE,
    ):
        if ExperimentConfig().apply_preset(preset) == config:
            return preset
    return None


def resolve_combo(
    chosen: tuple[str, str], current: tuple[str, str], collection_exists=None
) -> tuple[tuple[str, str], str | None]:
    """Availability guard (#30): a combo without a built collection never wins.

    Returns (combo_to_apply, warning). Pure — the streamlit layer only renders
    the warning; the decision is testable without a widget runtime.
    """
    if chosen == current:
        return current, None
    target = collection_name(chosen[0], chosen[1])
    if collection_exists is not None and not collection_exists(target):
        return current, (
            f"Collection `{target}` is not built — keeping the current combo. "
            "Build it with scripts/benchmark_dense_matrix.py."
        )
    return chosen, None


def knob_editor(
    config: ExperimentConfig, version: int, collection_exists=None
) -> ExperimentConfig | None:
    """Renders grouped knobs; returns an edited config copy or None if unchanged.

    Widget keys embed ``version`` (bumped when a preset is applied) so the
    widgets remount with the new config instead of forcing their stale
    Streamlit-held values back — that was the preset-button bug.
    ``collection_exists``: availability guard (#30) — a combo whose Chroma
    collection is missing warns and never reaches the engine.
    """
    v = version
    edited = config.model_copy(deep=True)
    changed = False

    with st.expander("Models", expanded=False):
        router_options = usable_models(_ROUTER_MODELS, config.router_model)
        router = st.selectbox(
            "Router model",
            router_options,
            index=router_options.index(config.router_model)
            if config.router_model in router_options else 0,
            key=f"router_model_{v}",
        )
        synth_options = usable_models(_SYNTH_MODELS, config.synthesis_model)
        synth = st.selectbox(
            "Synthesis model",
            synth_options,
            index=synth_options.index(config.synthesis_model)
            if config.synthesis_model in synth_options else 0,
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

    with st.expander("Embeddings", expanded=False):
        combo_ids = [(preset, profile) for preset, profile, _ in _COMBOS]
        labels = {
            (preset, profile): f"{_PROFILE_LABELS[profile]} · {preset} — {stats}"
            for preset, profile, stats in _COMBOS
        }
        current = (config.column_preset, config.embedding_profile)
        combo = st.selectbox(
            "Columns × embedding model",
            combo_ids,
            index=combo_ids.index(current) if current in combo_ids else 0,
            format_func=lambda c: labels[c],
            key=f"embed_combo_{v}",
            help=(
                f"minimal = {_MINIMAL_COLUMNS}. full = minimal + top-10 cast + "
                "tagline. Numbers are the ADR 0008 golden-query measurements. "
                "The same model embeds your queries at ask time."
            ),
        )
        combo, warning = resolve_combo(combo, current, collection_exists)
        if warning:
            st.warning(warning)
        if combo != current:
            changed = True
        edited.column_preset, edited.embedding_profile = combo
        # #30 transparency: user queries are embedded by the collection's model.
        st.text_input(
            "User queries embedded by",
            value=str(MODEL_PROFILES[combo[1]]["model"]),
            disabled=True,
            key=f"query_embedder_{v}",
        )

    with st.expander("Retrieval", expanded=False):
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

    with st.expander("Routing and Guardrails", expanded=False):
        attempts = st.slider(
            "Route max attempts (bounded re-route cycle)", 1, 5, config.route_max_attempts,
            key=f"attempts_{v}",
        )
        threshold = st.slider(
            "Router confidence threshold (below → heuristic fallback, #12)",
            0.0, 1.0, config.confidence_threshold, 0.05,
            key=f"conf_threshold_{v}",
        )
        cwa = st.checkbox(
            "Closed-world-assumption grounding enforcement", config.cwa_guardrail_enabled,
            key=f"cwa_{v}",
        )
        retrieve_axes = st.slider(
            "Funnel retrieve threshold (answered axes → retrieve, #53)",
            1, 5, config.funnel_retrieve_axes,
            key=f"retrieve_axes_{v}",
        )
        era_old = st.number_input(
            "'Old/classic movie' year cutoff (year_max, #42)",
            1970, 2026, config.era_old_year_max,
            key=f"era_old_{v}",
        )
        era_recent = st.number_input(
            "'Recent/latest movie' year cutoff (year_min, #42)",
            1970, 2026, config.era_recent_year_min,
            key=f"era_recent_{v}",
        )
        for old, new in [
            (config.route_max_attempts, attempts), (config.cwa_guardrail_enabled, cwa),
            (config.confidence_threshold, threshold),
            (config.funnel_retrieve_axes, retrieve_axes),
            (config.era_old_year_max, era_old),
            (config.era_recent_year_min, era_recent),
        ]:
            if old != new:
                changed = True
        edited.route_max_attempts, edited.cwa_guardrail_enabled = attempts, cwa
        edited.confidence_threshold = threshold
        edited.funnel_retrieve_axes = retrieve_axes
        edited.era_old_year_max, edited.era_recent_year_min = era_old, era_recent

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


def _lab_store(session):
    """One MovieVectorStore per session for the availability guard (cached)."""
    store = getattr(session, "_lab_vector_store", None)
    if store is None:
        from src.indexing.vector_store import MovieVectorStore

        store = MovieVectorStore("data/chroma_db")
        session._lab_vector_store = store
    return store


def render_lab(session) -> None:
    st.markdown("**Experimentation Lab**")
    st.caption("Live architecture knobs — applied from the next message.")

    active = matching_preset(session.config)
    if active is not None:
        # Highlight matching the nav's selected-segment look: light primary
        # fill, primary border and text (see Pages control in app.py).
        st.markdown(
            f"<style>.st-key-preset_{active.value} button {{"
            "background-color: rgba(215, 38, 61, 0.1);"
            "border-color: #D7263D; color: #D7263D;"
            "}</style>",
            unsafe_allow_html=True,
        )
    preset_cols = st.columns(3)
    preset_map = [
        (preset_cols[0], PresetType.FAST_BUDGET, "Fast Budget"),
        (preset_cols[1], PresetType.PRODUCTION_HYBRID, "Production"),
        (preset_cols[2], PresetType.NAIVE_BASELINE, "Naive Baseline"),
    ]
    for col, preset, label in preset_map:
        if col.button(
            label,
            use_container_width=True,
            key=f"preset_{preset.value}",
            help=_PRESET_HELP[preset],
        ):
            session.apply_preset(preset)
            st.toast(f"Preset applied: {label}")
            st.rerun()  # repaint the highlight + remounted knobs immediately

    edited = knob_editor(
        session.config, session.config_version,
        collection_exists=_lab_store(session).has_collection,
    )
    if edited is not None:
        session.replace_config(edited)
        st.toast("Configuration updated — pipeline rebuilt")

    st.divider()
    render_budget_meter(session)
