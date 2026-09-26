"""Provider endpoint resolution: z.ai keys first, OpenRouter fallback (#97).

Pure module — no I/O, no client construction. The three model-client
constructors (router, synthesizer, judge) call ``resolve_chat_endpoint``
and feed the result to ``ChatOpenAI`` (langchain-openai; both providers
speak the OpenAI Chat Completions protocol).

Resolution rules (user decision #97):
- ``glm-*`` model ids go to z.ai when ``ZAI_API_KEY`` is set, otherwise to
  OpenRouter under the ``z-ai/`` namespace.
- ``~google/...`` and ``google/gemini...`` ids (the router/synthesis
  defaults) are the swappable family: under a z.ai key they resolve to
  ``zai_model`` (GLM-5.3-flash), because z.ai does not serve Gemini.
  Without a z.ai key they pass through to OpenRouter verbatim — OpenRouter
  resolves the ``~`` alias itself (note: the ``~`` alias family only has a
  ``-latest`` member; the flash-lite notch-down is the dated
  ``google/gemini-3.5-flash-lite``, verified 2026-09-26).
- Everything else (e.g. the judge's ``meta-llama/...``) always goes to
  OpenRouter verbatim, z.ai key or not — the judge comparability guard
  (#97 requirement 4) is structural, not conventional: the judge call site
  passes ``allow_swap=False``, so even a Gemini judge id can never swap.

An explicitly passed ``openrouter_api_key`` (the constructors' pre-existing
``api_key`` argument, used by tests and tools) pins OpenRouter: callers
only offer the z.ai key when no explicit key was given.
"""

from dataclasses import dataclass
from typing import Literal

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"


@dataclass(frozen=True)
class ProviderEndpoint:
    """Where a model call goes, under which key, with which wire model id."""

    base_url: str
    api_key: str | None
    wire_model: str
    provider: Literal["zai", "openrouter"]  # budget attribution


def resolve_chat_endpoint(
    model_id: str,
    *,
    zai_model: str = "glm-5.3-flash",
    zai_api_key: str | None = None,
    openrouter_api_key: str | None = None,
    zai_base_url: str = DEFAULT_ZAI_BASE_URL,
    allow_swap: bool = True,  # False: config id passes through verbatim (judge guard)
) -> ProviderEndpoint:
    """Resolve one model call to (base_url, api_key, wire_model, provider).

    ``zai_api_key=""`` (an env var set to empty) counts as absent.
    """
    if model_id.startswith("glm-"):
        if zai_api_key:
            return ProviderEndpoint(zai_base_url, zai_api_key, model_id, "zai")
        return ProviderEndpoint(
            OPENROUTER_BASE_URL, openrouter_api_key, f"z-ai/{model_id}", "openrouter"
        )
    if (
        allow_swap
        and zai_api_key
        and (
            model_id.startswith("~google/") or model_id.startswith("google/gemini")
        )
    ):
        return ProviderEndpoint(zai_base_url, zai_api_key, zai_model, "zai")
    return ProviderEndpoint(
        OPENROUTER_BASE_URL, openrouter_api_key, model_id, "openrouter"
    )
