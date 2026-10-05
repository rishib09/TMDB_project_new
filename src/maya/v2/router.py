"""The v2 Understand router: one structured LLM call per turn (#82 C12–C14).

Pattern-matched to the v1 ``MayaRouter`` (endpoint seam, D17 client, bound
chain) but shares no v1 code — v1 is never modified (map #81 invariant).
Failure degradation is explicit and recorded: schema double-failure and
API errors both land in ``deterministic_ask`` with a note (C12), never a
raised exception and never a silent guess.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from src.domain.config import ExperimentConfig
from src.domain.memory import UserSessionPreferences
from src.domain.routing import MetadataFilterCriteria
from src.domain.usage import LLMUsage
from src.maya.providers import DEFAULT_ZAI_BASE_URL, resolve_chat_endpoint
from src.maya.v2.disposer import (
    MAX_SCHEMA_ATTEMPTS,
    deterministic_ask,
    enforce_probe_budget,
    enforce_question,
)
from src.maya.v2.models import PreferenceDelta, SubmitUnderstanding, Understanding
from src.maya.v2.prompt import SYSTEM_PROMPT_V2, build_state_block

#: Strict key sets (#150): derived from the schemas, never hardcoded — the
#: schema IS the contract. Unknown keys are the #149 failure mode: pydantic's
#: default policy silently ignores them, so they must be caught BEFORE
#: validation. The dict-shaped slots of Understanding are exactly these
#: three levels; every other field is a scalar or a list of enums/strings.
_TOP_KEYS = frozenset(Understanding.model_fields)
_FILTER_KEYS = frozenset(MetadataFilterCriteria.model_fields)
_DELTA_KEYS = frozenset(PreferenceDelta.model_fields)


def strict_understanding(args) -> tuple[Understanding | None, str | None]:
    """Pure (#150): validate submit-tool args against the Understanding
    schema, rejecting keys the schema does not define at every dict-shaped
    level (top, ``filters``, ``preference_delta``) BEFORE pydantic's default
    silently ignores them (#149: five live Bruce Willis turns invented
    ``filters.cast`` and the actor vanished — silently, confidence 0.97).

    Returns ``(understanding, None)`` or ``(None, reason)``; the reason
    names the offending keys so the trace shows why (telemetry rule).
    Validates with ``Understanding`` itself — never the ``SubmitUnderstanding``
    subclass — so the result stays class-equal across the stack (pydantic
    equality is class-strict; ``deterministic_ask`` comparisons rely on it).
    """
    if not isinstance(args, dict):
        return None, f"args not a dict: {type(args).__name__}"
    unknown = sorted(set(args) - _TOP_KEYS)
    if unknown:
        return None, f"unexpected key(s): {', '.join(unknown)}"
    filters = args.get("filters")
    if isinstance(filters, dict):
        unknown = sorted(set(filters) - _FILTER_KEYS)
        if unknown:
            return None, f"unexpected filters key(s): {', '.join(unknown)}"
    delta = args.get("preference_delta")
    if isinstance(delta, dict):
        unknown = sorted(set(delta) - _DELTA_KEYS)
        if unknown:
            return None, f"unexpected preference_delta key(s): {', '.join(unknown)}"
    try:
        return Understanding.model_validate(args), None
    except ValidationError as exc:
        parts = [
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
            for err in exc.errors()[:3]
        ]
        return None, f"invalid args: {'; '.join(parts)}"


def _tool_args(resp) -> tuple[dict | None, str | None]:
    """Pure (#150): the args dict of the first tool call on a response.
    ``(None, reason)`` when the model answered without a usable tool call —
    the forced-tool transports treat prose as a schema failure, never feed
    it to a second parser (D19)."""
    calls = getattr(resp, "tool_calls", None) or []
    if not calls:
        return None, "no tool call"
    first = calls[0]
    args = first.get("args") if isinstance(first, dict) else getattr(first, "args", None)
    if not isinstance(args, dict):
        return None, "tool call args not a dict"
    return args, None


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
        # D19: langchain owns the client/transport; the JSON parse is owned
        # HERE — with_structured_output raises on malformed JSON before the
        # raw text can be recovered (and pydantic truncates it in str(exc)),
        # which made fence recovery impossible on the z.ai coding endpoint
        # (live finding, #106 smoke). The C14 prompt carries the contract;
        # pydantic validates; C12 budgets the retries. Tests stub _llm.
        self._fallback_llm = None  # #113: resolved lazily by _resolve_fallback

    def _resolve_fallback(self):
        """#113: the secondary Understand client, resolved once and cached.
        Same endpoint wiring as the primary; when the fallback is configured
        equal to the primary the primary client is reused (no double call).
        Returns ``None`` when the fallback client cannot be built (no key,
        unknown model) — the understand() path then degrades exactly as
        before this feature, with the reason recorded in the notes."""
        if self._fallback_llm is not None:
            return self._fallback_llm
        if self.config.v2_router_fallback_model == self.config.v2_router_model:
            self._fallback_llm = self._llm
            return self._llm
        try:
            endpoint = resolve_chat_endpoint(
                self.config.v2_router_fallback_model,
                zai_model=self.config.zai_model,
                zai_api_key=os.getenv("ZAI_API_KEY"),
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
                zai_base_url=os.getenv("ZAI_BASE_URL") or DEFAULT_ZAI_BASE_URL,
                allow_swap=not self.config.pin_v2_router_config_id,
            )
            self._fallback_llm = ChatOpenAI(
                model=endpoint.wire_model,
                temperature=self.config.temperature,
                base_url=endpoint.base_url,
                api_key=endpoint.api_key,
                max_tokens=2048,
                reasoning_effort=self.config.reasoning_effort,
                request_timeout=120,
                max_retries=1,
            )
        except Exception as exc:  # noqa: BLE001 — degrade to pre-#113 behavior
            self._fallback_unavailable = f"{type(exc).__name__}: {exc}"
            return None
        return self._fallback_llm

    # --- public seam ---------------------------------------------------------

    def understand(
        self,
        query: str,
        prefs: UserSessionPreferences,
        shown_titles: Sequence[str],
        last_assistant: str | None,
        probe_count: int,
    ) -> tuple[Understanding, list[str], LLMUsage | None]:
        """One Understand call: C14 payload -> structured response -> guards.

        Returns ``(understanding, notes, usage)``; the notes carry every
        disposition for the trace (telemetry rule) and ``usage`` is the
        token total of ALL attempts this call made (schema retries and the
        #113 fallback included), for budget metering (#123). Schema failures
        get ONE retry with the validation error, then the deterministic ask
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
        total_prompt = 0
        total_completion = 0

        if self.config.v2_understand_transport != "prompt_json":
            notes.append(f"understand transport={self.config.v2_understand_transport} (#150)")

        def _accrue(usage: LLMUsage | None) -> None:
            nonlocal total_prompt, total_completion
            if usage is not None:
                total_prompt += usage.prompt_tokens
                total_completion += usage.completion_tokens

        parsed, api_error, usage = self._try_chain(messages, notes=notes)
        _accrue(usage)
        attempts = 1
        while (
            parsed is None
            and api_error is None
            and attempts < MAX_SCHEMA_ATTEMPTS
        ):
            attempts += 1
            notes.append(f"schema failure on attempt {attempts - 1}; retrying with the error")
            parsed, api_error, usage = self._try_chain(messages, notes=notes)
            _accrue(usage)
        if parsed is None:
            # #113: one attempt on the fallback model before the ask — the
            # sweep winner (glm) must not degrade a whole turn when a spare
            # model is configured. Fired-or-not lands in the trace (notes).
            fb_llm = self._resolve_fallback()
            if fb_llm is not None:
                fb_parsed, fb_error, usage = self._try_chain(messages, llm=fb_llm, notes=notes)
                _accrue(usage)  # D3: priced at the primary rate — see _summed_usage
                if fb_parsed is not None:
                    notes.append(
                        f"understand fallback fired: {self.config.v2_router_fallback_model} "
                        f"(primary: {api_error or f'schema failures x{attempts}'})"
                    )
                    parsed = fb_parsed
                elif fb_error is not None:
                    notes.append(f"understand fallback error={fb_error}")
            else:
                reason = getattr(self, "_fallback_unavailable", "not resolvable")
                notes.append(f"understand fallback unavailable: {reason}")
        if parsed is None:
            if api_error is not None:
                notes.append(f"understand api_error={api_error} -> deterministic ask (C12)")
            else:
                notes.append(
                    f"schema failures on all {attempts} attempts -> deterministic ask (C12)"
                )
            return deterministic_ask(query), notes, self._summed_usage(total_prompt, total_completion)

        u, guard_notes = enforce_question(parsed)
        notes.extend(guard_notes)
        u, budget_notes = enforce_probe_budget(u, probe_count)
        notes.extend(budget_notes)
        return u, notes, self._summed_usage(total_prompt, total_completion)

    def _summed_usage(self, total_prompt: int, total_completion: int) -> LLMUsage | None:
        """One LLMUsage for the whole understand() attempt chain (#123).

        Token counts are exact (summed across schema retries and the #113
        fallback attempt — token-linear). The model is attributed to the
        PRIMARY config model (user decision D3): fallback-attempt tokens are
        priced at the primary rate — conservative over-pricing, up to ~2x
        on those tokens (glm-5.3-flash 0.25 vs flash-lite 0.12 $/MTok) but
        bounded to a fraction of a cent per turn — disclosed here rather
        than hidden. Returns ``None`` when no attempt consumed tokens
        (stubbed clients).
        """
        if total_prompt + total_completion == 0:
            return None
        return LLMUsage(
            model=self.config.v2_router_model,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
        )

    # --- private -------------------------------------------------------------

    def _try_chain(
        self, messages, llm=None, notes=None
    ) -> tuple[Understanding | None, str | None, LLMUsage | None]:
        """One client call under the configured transport (#150 knob):
        ``(None, error, None)`` on API failure, ``(None, None, usage)`` on an
        unusable reading (tokens were still spent — metered, #123). ``llm``
        overrides the client (#113 fallback); ``notes`` collects the
        per-attempt disposition when provided."""
        transport = self.config.v2_understand_transport
        if transport == "tool_call":
            return self._try_chain_tool_call(messages, llm, notes)
        if transport == "structured_output":
            return self._try_chain_structured(messages, llm, notes)
        return self._try_chain_prose(messages, llm)

    def _try_chain_prose(
        self, messages, llm=None
    ) -> tuple[Understanding | None, str | None, LLMUsage | None]:
        """Pre-#150 prose path (``prompt_json`` escape hatch): the C14 prompt
        carries the contract; the fence is stripped and pydantic validates
        HERE — within the SAME attempt, before any C12 retry is spent. Kept
        byte-compatible: it is the pinned baseline of the transport
        comparison and the fallback if a provider cannot honor tool calls."""
        try:
            resp = (llm or self._llm).invoke(messages)
        except Exception as exc:  # noqa: BLE001 — D17 client exhausted; degrade (C12)
            return None, f"{type(exc).__name__}: {exc}", None
        usage = LLMUsage.from_response(resp)
        text = getattr(resp, "content", None)
        if isinstance(text, list):  # content blocks -> joined text
            text = "".join(getattr(b, "text", "") for b in text)
        if not isinstance(text, str) or not text.strip():
            return None, None, usage
        stripped = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.DOTALL
        )
        try:
            return Understanding.model_validate_json(stripped), None, usage
        except ValidationError:
            return None, None, usage

    def _try_chain_tool_call(
        self, messages, llm=None, notes=None
    ) -> tuple[Understanding | None, str | None, LLMUsage | None]:
        """``tool_call`` transport (#150, option A): the schema bound as the
        single tool with ``tool_choice`` forced — the model's reading arrives
        as the tool-call arguments: structured JSON, exact keys, no fence to
        strip (live-probed on z.ai glm-5.3-flash: the force is honored and
        invented keys in prose JSON disappear). Args pass the strict
        validator BEFORE pydantic (#149); a prose answer despite the force
        is a schema failure — never a second parser (D19)."""
        client = llm or self._llm
        try:
            resp = client.bind_tools(
                [SubmitUnderstanding], tool_choice="SubmitUnderstanding"
            ).invoke(messages)
        except Exception as exc:  # noqa: BLE001 — D17 client exhausted; degrade (C12)
            return None, f"{type(exc).__name__}: {exc}", None
        usage = LLMUsage.from_response(resp)
        calls = getattr(resp, "tool_calls", None) or []
        if len(calls) > 1 and notes is not None:
            notes.append(f"two tool calls — used first ({len(calls)} seen)")
        args, reason = _tool_args(resp)
        if args is None:
            if reason is not None and notes is not None:
                notes.append(f"understand tool_call: {reason}")
            return None, None, usage
        parsed, strict_reason = strict_understanding(args)
        if parsed is None and notes is not None:
            notes.append(f"understand strict rejection: {strict_reason}")
        return parsed, strict_reason, usage

    def _try_chain_structured(
        self, messages, llm=None, notes=None
    ) -> tuple[Understanding | None, str | None, LLMUsage | None]:
        """``structured_output`` transport (#150, option B): LangChain's
        wrapper — ``with_structured_output(method='function_calling',
        include_raw=True)`` with ``tool_choice`` forced via kwargs
        passthrough (verified on langchain-openai 1.6.0: the bind kwargs
        carry it). The wrapper's ``parsed`` is NEVER trusted:
        PydanticToolsParser validates with the schema whose nested models
        silently ignore extras — the exact invented-key path of #149. The
        raw AIMessage's tool args get the same strict validation, and usage
        is metered from ``raw`` — token granularity is kept under this
        transport (#123)."""
        client = llm or self._llm
        try:
            structured = client.with_structured_output(
                SubmitUnderstanding,
                method="function_calling",
                include_raw=True,
                tool_choice="SubmitUnderstanding",
            )
            result = structured.invoke(messages)
        except Exception as exc:  # noqa: BLE001 — D17 client exhausted; degrade (C12)
            return None, f"{type(exc).__name__}: {exc}", None
        if not isinstance(result, dict):  # include_raw=False shape would mean a re-design
            return None, None, None
        raw = result.get("raw")
        usage = LLMUsage.from_response(raw)
        if result.get("parsing_error") is not None:
            if notes is not None:
                notes.append(f"understand parsing_error: {result['parsing_error']}")
            return None, None, usage
        calls = getattr(raw, "tool_calls", None) or []
        if len(calls) > 1 and notes is not None:
            notes.append(f"two tool calls — used first ({len(calls)} seen)")
        args, reason = _tool_args(raw)
        if args is None:
            if reason is not None and notes is not None:
                notes.append(f"understand structured_output: {reason}")
            return None, None, usage
        parsed, strict_reason = strict_understanding(args)
        if parsed is None and notes is not None:
            notes.append(f"understand strict rejection: {strict_reason}")
        return parsed, strict_reason, usage
