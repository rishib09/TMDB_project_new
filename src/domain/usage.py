"""Token-usage value types for budget accounting (#8, #123).

ADR 0006: pure domain — no repo imports. ``from_response`` is duck-typed on
purpose (``getattr`` only) so it adapts any langchain-shaped response object
without importing the framework into the domain layer.
"""

from pydantic import BaseModel


class LLMUsage(BaseModel):
    """Token usage of one LLM call, for budget accounting (#8, #123).

    One taxonomy for every stack call — v1 route, v2 understand, synthesis.
    The former synthesis-only type (``SynthesisUsage``) is an alias of this.
    """

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @classmethod
    def from_response(cls, response: object) -> "LLMUsage | None":
        """Extracts usage from a langchain response (``usage_metadata`` dict).

        Returns ``None`` when the response carries no usage (stubbed clients,
        provider omission) — callers meter nothing rather than guess.
        """
        meta = getattr(response, "usage_metadata", None)
        if not isinstance(meta, dict):
            return None
        try:
            return cls(
                model="",
                prompt_tokens=int(meta.get("input_tokens") or 0),
                completion_tokens=int(meta.get("output_tokens") or 0),
            )
        except (TypeError, ValueError):
            return None
