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
