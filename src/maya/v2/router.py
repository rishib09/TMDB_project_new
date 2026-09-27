"""The v2 Understand router: one structured LLM call per turn (#82 C12–C14).

Pattern-matched to the v1 ``MayaRouter`` (endpoint seam, D17 client, bound
chain) but shares no v1 code — v1 is never modified (map #81 invariant).
Failure degradation is explicit and recorded: schema double-failure and
API errors both land in ``deterministic_ask`` with a note (C12), never a
raised exception and never a silent guess.
"""

from __future__ import annotations

import os
from typing import Sequence

from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from src.domain.config import ExperimentConfig
from src.domain.memory import UserSessionPreferences
from src.maya.providers import DEFAULT_ZAI_BASE_URL, resolve_chat_endpoint
from src.maya.v2.disposer import (
    MAX_SCHEMA_ATTEMPTS,
    deterministic_ask,
    enforce_probe_budget,
    enforce_question,
)
from src.maya.v2.models import Understanding
from src.maya.v2.prompt import SYSTEM_PROMPT_V2, build_state_block


class MayaV2Router:
    """The v2 Understand call beside the gated v1 stack (#83 S1–S10)."""

    def __init__(self, config: ExperimentConfig, api_key: str | None = None) -> None:
        self.config = config
        self._endpoint = resolve_chat_endpoint(
            config.v2_router_model,
            zai_model=config.zai_model,
            zai_api_key=os.getenv("ZAI_API_KEY") if api_key is None else None,
            openrouter_api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            zai_base_url=os.getenv("ZAI_BASE_URL") or DEFAULT_ZAI_BASE_URL,
            allow_swap=not config.pin_v2_router_config_id,  # #107 sweep isolation
        )
        self._llm = ChatOpenAI(
            model=self._endpoint.wire_model,
            temperature=config.temperature,
            base_url=self._endpoint.base_url,
            api_key=self._endpoint.api_key,
            max_tokens=2048,  # C14: schema + verbose questions on flash models
            reasoning_effort=config.reasoning_effort,  # #79: bound hidden reasoning
            request_timeout=120,  # D17: retries/timeout live in the client
            max_retries=1,
        )
        # Bound once at construction; tests stub this attribute directly.
        self._chain = self._llm.with_structured_output(Understanding, include_raw=True)

    # --- public seam ---------------------------------------------------------

    def understand(
        self,
        query: str,
        prefs: UserSessionPreferences,
        shown_titles: Sequence[str],
        last_assistant: str | None,
        probe_count: int,
    ) -> tuple[Understanding, list[str]]:
        """One Understand call: C14 payload -> structured response -> guards.

        Returns ``(understanding, notes)``; the notes carry every
        disposition for the trace (telemetry rule). Schema failures get
        ONE retry with the validation error, then the deterministic ask
        (C12, ``MAX_SCHEMA_ATTEMPTS``). API errors degrade the same way,
        recorded — never raised.
        """
        messages: list[tuple[str, str]] = [
            ("system", SYSTEM_PROMPT_V2),
            ("system", build_state_block(prefs, shown_titles)),
        ]
        if last_assistant:
            messages.append(("system", f"MAYA'S LAST REPLY (for reference): {last_assistant}"))
        messages.append(("human", query))

        notes: list[str] = []
        parsed, api_error = self._try_chain(messages)
        attempts = 1
        while (
            parsed is None
            and api_error is None
            and attempts < MAX_SCHEMA_ATTEMPTS
        ):
            attempts += 1
            notes.append(f"schema failure on attempt {attempts - 1}; retrying with the error")
            parsed, api_error = self._try_chain(messages)
        if parsed is None:
            if api_error is not None:
                notes.append(f"understand api_error={api_error} -> deterministic ask (C12)")
            else:
                notes.append(
                    f"schema failures on all {attempts} attempts -> deterministic ask (C12)"
                )
            return deterministic_ask(query), notes

        u, guard_notes = enforce_question(parsed)
        notes.extend(guard_notes)
        u, budget_notes = enforce_probe_budget(u, probe_count)
        notes.extend(budget_notes)
        return u, notes

    # --- private -------------------------------------------------------------

    def _try_chain(self, messages) -> tuple[Understanding | None, str | None]:
        """One structured call; ``(None, error)`` on API failure, ``(None, None)``
        on schema-invalid output."""
        try:
            result = self._chain.invoke(messages)
        except Exception as exc:  # noqa: BLE001 — D17 client exhausted; degrade (C12)
            return None, f"{type(exc).__name__}: {exc}"
        parsed = result.get("parsed") if isinstance(result, dict) else result
        if parsed is None or isinstance(parsed, ValidationError):
            return None, None
        return parsed, None
