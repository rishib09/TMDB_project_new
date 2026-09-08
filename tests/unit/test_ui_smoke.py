"""Smoke tests for the Streamlit UI layer (issue #7).

Deliberately minimal: only pure helpers are unit-tested — Streamlit widget
code cannot execute outside the Streamlit runtime, and the real acceptance
is the manual look-and-feel walkthrough. These tests catch import errors
and pure-logic regressions cheaply.
"""


from src.ui.chat_tab import intent_badge_text
from src.ui.session import ADMIN_COMMAND, MayaSession


def test_modules_import():
    """UI modules import (catches syntax/bad-import errors).

    Note: app.py itself is excluded — executing it bare would boot the whole
    pipeline (DB + chroma + graph) outside the Streamlit runtime; it is
    verified by `streamlit run app.py` in the manual walkthrough instead.
    """
    import importlib

    for module in ("src.ui.session", "src.ui.chat_tab", "src.ui.sidebar_lab",
                   "src.ui.trace_tab", "src.ui.evals_tab"):
        importlib.import_module(module)


def test_admin_command_detection():
    assert MayaSession.is_admin_command("/admin")
    assert MayaSession.is_admin_command("  /ADMIN ")
    assert not MayaSession.is_admin_command("/admin what movies")
    assert not MayaSession.is_admin_command("hello")
    assert ADMIN_COMMAND == "/admin"


def test_intent_badge_text_formatting():
    row = {"intent": "SEMANTIC_SEARCH", "confidence": 0.92, "path": "single-route",
           "attempts": 1, "n_movies": 5, "tokens": 210}
    text = intent_badge_text(row)
    assert text == ("INTENT: SEMANTIC_SEARCH — confidence 0.92 — route: single-route "
                    "— 5 movies — 210 tokens")


def test_intent_badge_shows_reroute_attempts():
    row = {"intent": "SEMANTIC_SEARCH", "confidence": 0.4, "path": "reroute",
           "attempts": 2, "n_movies": 0, "tokens": 0}
    assert "route: reroute (x2)" in intent_badge_text(row)


def test_waterfall_frame_handles_empty_and_single():
    from src.ui.trace_tab import _waterfall_frame

    assert _waterfall_frame([]).empty
    frame = _waterfall_frame([{"node": "guard_input", "timestamp": "2026-09-01T10:00:00",
                               "payload": {"verdict": "clean"}}])
    assert len(frame) == 1 and frame.iloc[0]["node"] == "guard_input"


def test_slice_new_traces_only_current_turn():
    """Issue #18: route counts come from this turn's ring slice, not the whole ring."""
    from src.ui.session import slice_new_traces

    ring = [
        {"node": "route", "payload": {}},
        {"node": "retrieve", "payload": {}},
    ]
    assert slice_new_traces(2, ring) == []  # nothing new this turn
    new_ring = ring + [{"node": "route", "payload": {}}, {"node": "route", "payload": {}}]
    assert len(slice_new_traces(2, new_ring)) == 2


def test_auto_scroll_script_targets_streamlit_containers():
    """Issue #27-R: the scroll snippet must target Streamlit's main scroller
    with fallbacks — a selector miss would silently leave the viewport stale."""
    from src.ui.chat_tab import _AUTO_SCROLL_JS

    assert "section.stMain" in _AUTO_SCROLL_JS
    assert "section.main" in _AUTO_SCROLL_JS          # legacy fallback
    assert "stAppViewContainer" in _AUTO_SCROLL_JS    # container fallback
    assert "scrollHeight" in _AUTO_SCROLL_JS          # bottom-of-conversation


def test_scroll_to_newest_renders_without_runtime():
    """The zero-height component call must not raise outside a Streamlit run."""
    from src.ui.chat_tab import scroll_to_newest

    scroll_to_newest()  # smoke: import-time wiring correct


def test_routing_counters_aggregate_route_spans_only():
    """#29 observability: fallback / re-route / mismatch chips from the ring."""
    from src.ui.trace_tab import routing_counters

    traces = [
        {"node": "route", "payload": {"attempt": 1, "is_fallback": True,
                                      "requires_rag_mismatch": False}},
        {"node": "route", "payload": {"attempt": 2, "is_fallback": False,
                                      "requires_rag_mismatch": True}},
        {"node": "retrieve", "payload": {"is_fallback": True}},  # not a route span
    ]
    counters = routing_counters(traces)
    assert counters == {"fallbacks": 1, "reroutes": 1, "mismatches": 1}


def test_routing_counters_empty_ring():
    from src.ui.trace_tab import routing_counters

    assert routing_counters([]) == {"fallbacks": 0, "reroutes": 0, "mismatches": 0}


# --- #30: Lab embedding knobs -------------------------------------------------

def test_matching_preset_exact_or_none():
    """Red highlight semantics: pristine preset match only; any edit clears."""
    from src.domain.config import ExperimentConfig, PresetType
    from src.ui.sidebar_lab import matching_preset

    assert matching_preset(ExperimentConfig()) is PresetType.PRODUCTION_HYBRID
    fast = ExperimentConfig().apply_preset(PresetType.FAST_BUDGET)
    assert matching_preset(fast) is PresetType.FAST_BUDGET
    fast.temperature = 0.3  # a NON-preset knob still clears the highlight
    assert matching_preset(fast) is None
    naive = ExperimentConfig().apply_preset(PresetType.NAIVE_BASELINE)
    naive.embedding_profile = "gemini_embedding_2"
    assert matching_preset(naive) is None


def test_usable_models_drops_delisted_keeps_active(monkeypatch):
    """Availability guard: delisted slugs vanish; the active model survives."""
    from src.ui import sidebar_lab

    monkeypatch.setattr(
        sidebar_lab, "available_model_ids", lambda: frozenset({"model/a"})
    )
    assert sidebar_lab.usable_models(["model/a", "model/dead"], "model/a") == ["model/a"]
    # the currently-selected model never disappears from its own dropdown
    assert sidebar_lab.usable_models(["model/a", "model/dead"], "model/dead") == [
        "model/a", "model/dead",
    ]


def test_usable_models_fails_open(monkeypatch):
    """Catalog down -> assume everything works (dropdowns must not empty)."""
    from src.ui import sidebar_lab

    monkeypatch.setattr(sidebar_lab, "available_model_ids", lambda: None)
    assert sidebar_lab.usable_models(["x", "y"], "x") == ["x", "y"]


def test_collection_name_single_source():
    from src.indexing.embeddings import collection_name

    assert collection_name("full", "gemini_embedding_2") == "full_gemini_embedding_2"
    assert collection_name("minimal", "lfm_free") == "minimal_lfm_free"


def test_has_collection_semantics():
    """Empty or missing collections both count as unavailable (#30 guard)."""
    from src.indexing.vector_store import MovieVectorStore

    class Stub:
        def __init__(self, n):
            self._n = n

        def count(self, name):
            return self._n

    assert MovieVectorStore.has_collection(Stub(9119), "full_gemini_embedding_2")
    assert not MovieVectorStore.has_collection(Stub(0), "anything")


def test_lab_combos_cover_all_six_and_carry_measurements():
    from src.ui.sidebar_lab import _COMBOS

    assert len(_COMBOS) == 6
    assert {(p, m) for p, m, _ in _COMBOS} == {
        (preset, model)
        for preset in ("minimal", "full")
        for model in ("lfm_free", "nemotron_free", "gemini_embedding_2")
    }
    assert all("hit@5" in stats for _, _, stats in _COMBOS)
