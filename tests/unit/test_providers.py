"""Unit tests for the provider endpoint seam (#97) — pure resolution matrix."""

import pytest

from src.maya.providers import (
    DEFAULT_ZAI_BASE_URL,
    OPENROUTER_BASE_URL,
    ProviderEndpoint,
    resolve_chat_endpoint,
)


def test_openrouter_default_when_no_zai_key():
    """No ZAI key: the ~google alias passes through verbatim."""
    ep = resolve_chat_endpoint("~google/gemini-flash-latest")
    assert ep == ProviderEndpoint(
        OPENROUTER_BASE_URL, None, "~google/gemini-flash-latest", "openrouter"
    )


def test_zai_key_swaps_google_family_to_glm():
    """ZAI key present: ~google ids resolve to the z.ai model on the z.ai endpoint."""
    ep = resolve_chat_endpoint(
        "~google/gemini-flash-latest", zai_model="glm-5.3-flash", zai_api_key="zk1"
    )
    assert (ep.base_url, ep.api_key, ep.wire_model, ep.provider) == (
        DEFAULT_ZAI_BASE_URL, "zk1", "glm-5.3-flash", "zai"
    )


def test_zai_key_swaps_dated_gemini_ids_too():
    """The dated flash-lite default is in the swappable family as well."""
    ep = resolve_chat_endpoint(
        "google/gemini-3.5-flash-lite", zai_model="glm-5.3-flash", zai_api_key="zk1"
    )
    assert (ep.provider, ep.wire_model) == ("zai", "glm-5.3-flash")


def test_explicit_glm_id_goes_to_zai():
    ep = resolve_chat_endpoint("glm-5.3-flash", zai_api_key="zk1")
    assert (ep.provider, ep.wire_model) == ("zai", "glm-5.3-flash")


def test_glm_id_falls_back_to_openrouter_namespace():
    """No ZAI key: glm ids route to OpenRouter under z-ai/."""
    ep = resolve_chat_endpoint(
        "glm-5.3-flash", zai_api_key=None, openrouter_api_key="ok1"
    )
    assert (ep.base_url, ep.api_key, ep.wire_model, ep.provider) == (
        OPENROUTER_BASE_URL, "ok1", "z-ai/glm-5.3-flash", "openrouter"
    )


def test_judge_model_immune_to_zai_swap():
    """The llama judge id never swaps, even with a z.ai key (comparability guard)."""
    ep = resolve_chat_endpoint(
        "meta-llama/llama-3.3-70b-instruct", zai_api_key="zk1"
    )
    assert (ep.provider, ep.wire_model) == ("openrouter", "meta-llama/llama-3.3-70b-instruct")


def test_empty_zai_key_counts_as_absent():
    """A ZAI var set to empty must not flip the provider."""
    ep = resolve_chat_endpoint("~google/gemini-flash-latest", zai_api_key="")
    assert ep.provider == "openrouter"


def test_coding_plan_endpoint_override_respected():
    ep = resolve_chat_endpoint(
        "glm-5.3-flash",
        zai_api_key="zk1",
        zai_base_url="https://api.z.ai/api/coding/paas/v4",
    )
    assert ep.base_url == "https://api.z.ai/api/coding/paas/v4"


def test_explicit_openrouter_key_pins_openrouter():
    """Callers that pass api_key explicitly (tests/tools) pin OpenRouter."""
    ep = resolve_chat_endpoint(
        "google/gemini-3.5-flash-lite",
        zai_api_key=None,  # constructors pass None when api_key was explicit
        openrouter_api_key="explicit",
    )
    assert (ep.provider, ep.api_key) == ("openrouter", "explicit")


def test_endpoint_is_frozen():
    ep = resolve_chat_endpoint("glm-5.3-flash", zai_api_key="zk1")
    with pytest.raises(Exception):
        ep.wire_model = "tampered"  # type: ignore[misc]


def test_synthesis_usage_reports_wire_model(monkeypatch):
    """#97 defect: budget attribution must carry the WIRE model, not the config id.

    Under a z.ai key the config id is google/gemini-3.5-flash-lite but the call
    is served by glm-5.3-flash — pricing (estimate_cost) keys off this string.
    """
    import os
    import types

    from src.domain.config import ExperimentConfig
    from src.maya.agent import MayaSynthesizer

    monkeypatch.setenv("ZAI_API_KEY", "zk1")
    synth = MayaSynthesizer(ExperimentConfig())  # env-driven: ZAI key set, no explicit pin
    assert synth._endpoint.wire_model == "glm-5.3-flash"  # swapped, not the config id

    fake_response = types.SimpleNamespace(
        usage_metadata={"input_tokens": 10, "output_tokens": 5}, text="ok"
    )
    monkeypatch.setattr(type(synth._llm), "invoke", lambda self, _messages: fake_response)
    from src.domain.routing import QueryRoutingDecision

    decision = QueryRoutingDecision(
        intent="SEMANTIC_SEARCH", confidence=0.9,
        standalone_query="a sci-fi movie", requires_rag=True,
    )
    _, usage = synth.synthesize("a sci-fi movie", decision, [], [])
    assert usage.model == "glm-5.3-flash"


def test_allow_swap_false_pins_google_family_verbatim():
    """#97 review P1: the judge must be able to opt out of the family swap —
    a Gemini judge_model id stays verbatim on OpenRouter even under a z.ai key."""
    for mid in ("google/gemini-3.5-flash-lite", "~google/gemini-flash-latest"):
        ep = resolve_chat_endpoint(mid, zai_api_key="zk1", allow_swap=False)
        assert (ep.provider, ep.wire_model) == ("openrouter", mid)


def test_allow_swap_does_not_affect_glm_provider_choice():
    """allow_swap gates the family MODEL swap only; glm ids still route by key."""
    zai = resolve_chat_endpoint("glm-5.3-flash", zai_api_key="zk1", allow_swap=False)
    assert zai.provider == "zai"
    fallback = resolve_chat_endpoint("glm-5.3-flash", allow_swap=False)
    assert (fallback.provider, fallback.wire_model) == ("openrouter", "z-ai/glm-5.3-flash")


def test_judge_with_gemini_id_stays_openrouter_under_zai_key(monkeypatch):
    """The guard, enforced at the judge call site: even a Gemini judge_model
    must not silently swap while ZAI_API_KEY is set."""
    from src.domain.config import ExperimentConfig
    from src.evals.judge import MayaJudge

    monkeypatch.setenv("ZAI_API_KEY", "zk1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")  # client construction needs a key
    cfg = ExperimentConfig().model_copy(
        update={"judge_model": "google/gemini-3.5-flash-lite"}
    )
    judge = MayaJudge(cfg)
    assert (judge._endpoint.provider, judge._endpoint.wire_model) == (
        "openrouter", "google/gemini-3.5-flash-lite",
    )


def test_sweep_pin_keeps_synthesis_on_config_id(monkeypatch):
    """#89 sweep isolation: pin_synthesis_config_id=True means the synthesizer
    keeps the config id verbatim via OpenRouter even under a z.ai key."""
    from src.domain.config import ExperimentConfig
    from src.maya.agent import MayaSynthesizer

    monkeypatch.setenv("ZAI_API_KEY", "zk1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    cfg = ExperimentConfig().model_copy(update={"pin_synthesis_config_id": True})
    synth = MayaSynthesizer(cfg)
    assert (synth._endpoint.provider, synth._endpoint.wire_model) == (
        "openrouter", "google/gemini-3.5-flash-lite",
    )


def test_sweep_pin_keeps_router_on_config_id(monkeypatch):
    """#89 sweep isolation, router side: a google-family router candidate must
    not silently become glm under a z.ai key."""
    from src.domain.config import ExperimentConfig
    from src.maya.router import MayaRouter

    monkeypatch.setenv("ZAI_API_KEY", "zk1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    cfg = ExperimentConfig().model_copy(update={
        "router_model": "google/gemini-3.5-flash-lite",
        "pin_router_config_id": True,
    })
    router = MayaRouter(cfg)
    assert (router._endpoint.provider, router._endpoint.wire_model) == (
        "openrouter", "google/gemini-3.5-flash-lite",
    )


def test_sweep_unpinned_router_still_swaps(monkeypatch):
    """Without the pin the family rule still applies (the glm candidate path)."""
    from src.domain.config import ExperimentConfig
    from src.maya.router import MayaRouter

    monkeypatch.setenv("ZAI_API_KEY", "zk1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    router = MayaRouter(ExperimentConfig())
    assert (router._endpoint.provider, router._endpoint.wire_model) == ("zai", "glm-5.3-flash")


def test_judge_client_has_max_tokens_cap():
    """#74/#89: the judge client caps completions (no more 16K runaway JSON)."""
    from src.domain.config import ExperimentConfig
    from src.evals.judge import MayaJudge

    judge = MayaJudge(ExperimentConfig(), api_key="or-key")
    assert judge._llm.max_tokens == 1024
