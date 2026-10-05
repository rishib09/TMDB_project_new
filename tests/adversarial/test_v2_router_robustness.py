"""Adversarial robustness tests for the v2 Understand transports (#150).

The defect (#149): the Understand call received JSON in prose, and pydantic
silently IGNORES keys the schema does not define — the model invented
``filters.cast``/``add_cast``/``add_actor_filters`` on five live Bruce Willis
turns, ``cast_member`` stayed null, the SQL path never fired, and nobody was
told. The fix: the schema is bound as a forced submit-tool and every arg dict
passes a strict validator that REJECTS unknown keys (top level, ``filters``,
``preference_delta``) before pydantic sees them.

These tests are written against the pre-#150 code: they run red there (the
transport knob does not exist; prose JSON with invented keys silently
validates) and must run green after the fix.

Both tool transports get the strict backstop:
- ``tool_call``           — bind_tools + forced tool_choice, args read directly.
- ``structured_output``   — with_structured_output(function_calling,
  include_raw=True); the wrapper's ``parsed`` is NOT trusted (pydantic's
  default would silently drop the same keys again) — raw tool args are
  validated with the same strict helper.
"""

import json

import pytest
from langchain_core.messages import AIMessage

from src.domain.config import ExperimentConfig
from src.domain.memory import UserSessionPreferences
from src.maya.v2 import MayaV2Router, SubmitUnderstanding, deterministic_ask
from src.maya.v2.models import Understanding

BASE_ARGS = {"intent": "SEMANTIC_SEARCH", "standalone_query": "q", "confidence": 0.9}


def _tool_resp(args: dict, *, usage: dict | None = None, extra_call: dict | None = None):
    """An AIMessage that calls the submit tool (optionally twice)."""
    calls = [{"name": "SubmitUnderstanding", "args": args, "id": "c1", "type": "tool_call"}]
    if extra_call is not None:
        calls.append({**extra_call, "id": "c2", "type": "tool_call"})
    kw = {"usage_metadata": usage} if usage else {}
    return AIMessage(content="", tool_calls=calls, **kw)


def _structured_result(raw, *, parsed=None, parsing_error=None):
    """The dict ``with_structured_output(..., include_raw=True)`` returns."""
    return {"raw": raw, "parsed": parsed, "parsing_error": parsing_error}


def _parsed_u(args: dict) -> Understanding:
    """What B-as-LangChain-intends would hand back: extras silently dropped."""
    return Understanding.model_validate(args)


class _ToolLLM:
    """Fake client for the ``tool_call`` transport: bind_tools scripts invoke."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.bind_seen = None

    def bind_tools(self, tools, **kwargs):
        self.bind_seen = {"tools": tools, "kwargs": kwargs}
        return self

    def invoke(self, messages):
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _Run:
    """The runnable ``with_structured_output`` returns (real seam: .invoke)."""

    def __init__(self, fn):
        self._fn = fn

    def invoke(self, messages):
        return self._fn(messages)


class _StructuredLLM:
    """Fake client for the ``structured_output`` transport."""

    def __init__(self, results):
        self._results = list(results)
        self.seen = {}

    def with_structured_output(self, schema, **kwargs):
        self.seen = {"schema": schema, "kwargs": kwargs}

        def _run(messages):
            item = self._results.pop(0)
            if isinstance(item, Exception):
                raise item
            return item  # dict {'raw','parsed','parsing_error'}

        return _Run(_run)

        return _run


def _router(responses, *, transport="tool_call", llm=None):
    cfg = ExperimentConfig(v2_understand_transport=transport)
    r = MayaV2Router(cfg, api_key="test-key")
    r._llm = llm if llm is not None else _ToolLLM(responses)
    if llm is None:
        r._llm = _ToolLLM(responses)
    return r, r._llm


def _structured_router(results):
    r, llm = _router([], transport="structured_output", llm=_StructuredLLM(results))
    return r, llm


# --- #149: invented keys must be rejected, never silently dropped --------------


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        ({**BASE_ARGS, "lead_actor": "Bruce Willis"}, "lead_actor"),
        ({**BASE_ARGS, "filters": {"cast": "Bruce Willis"}}, "cast"),
        ({**BASE_ARGS, "preference_delta": {"add_cast": ["Bruce Willis"]}}, "add_cast"),
    ],
)
def test_invented_keys_rejected_at_every_level(needle, args):
    """The five live Bruce failures invented keys at top level AND inside
    filters. Each must surface as a schema failure with the key named —
    not a silently-pruned Understanding with confidence 0.9."""
    r, _ = _router([_tool_resp(args), _tool_resp(args)])  # retry invents again
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out == deterministic_ask("q")
    assert any(needle in n for n in notes), notes
    assert any("C12" in n for n in notes)


def test_invented_key_never_lands_in_parsed_filters():
    """The pin: with the invented key present, understand() must not return
    an Understanding that carries filters at all (pre-#150 it did, with
    ``cast`` silently dropped and the actor lost on the next rewrite)."""
    args = {**BASE_ARGS, "filters": {"cast": "Bruce Willis", "genres": ["Action"]}}
    r, _ = _router([_tool_resp(args)])
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out.filters is None  # deterministic ask carries no filters


# --- forced tool: prose answers are schema failures, not a second parser -------


def test_prose_instead_of_tool_call_is_schema_failure():
    """The tool is FORCED: a model that answers in prose (even valid JSON)
    must not be parsed by a fallback prose path — it is a schema failure
    (tokens spent), retried, then the deterministic ask."""
    prose = AIMessage(content=json.dumps({**BASE_ARGS, "ready_to_retrieve": True}))
    r, _ = _router([prose, prose])
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out == deterministic_ask("q")
    assert any("no tool call" in n for n in notes), notes


def test_two_tool_calls_uses_first_and_records():
    """Pathological: two tool calls in one response — use the first, say so."""
    first = {**BASE_ARGS, "standalone_query": "first"}
    r, _ = _router(
        [_tool_resp(first, extra_call={"name": "SubmitUnderstanding", "args": {"junk": 1}})]
    )
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out.standalone_query == "first"
    assert any("two tool calls" in n for n in notes), notes


def test_wrong_enum_value_rejected_with_reason():
    """An intent outside the closed taxonomy is a schema failure whose
    reason names the field — telemetry must show WHY, not just that."""
    r, _ = _router([_tool_resp({**BASE_ARGS, "intent": "MOVIE_SEARCH"})] * 2)
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out == deterministic_ask("q")
    assert any("intent" in n for n in notes), notes


def test_forced_tool_call_and_schema_reach_the_client():
    """The bind must carry the schema as the tool AND force tool_choice —
    z.ai honors the force (live probe #150); without it glm-4.7 answers in
    prose."""
    r, llm = _router([_tool_resp(BASE_ARGS)])
    r.understand("q", UserSessionPreferences(), [], None, 0)
    assert llm.bind_seen is not None
    assert llm.bind_seen["tools"] == [SubmitUnderstanding]
    assert llm.bind_seen["kwargs"].get("tool_choice") == "SubmitUnderstanding"


def test_tool_transport_meters_usage_from_response():
    """#123: tokens spent on the tool call are metered exactly as prose."""
    usage = {"input_tokens": 40, "output_tokens": 10, "total_tokens": 50}
    r, _ = _router([_tool_resp(BASE_ARGS, usage=usage)])
    _, _, usage_out = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert usage_out is not None
    assert usage_out.prompt_tokens == 40 and usage_out.completion_tokens == 10


# --- structured_output transport: the wrapper's parsed is never trusted --------


def test_structured_parsed_silent_drop_is_caught():
    """B-as-LangChain-intends would hand back ``parsed`` with the invented
    key already silently dropped (pydantic default). The raw args say
    ``cast``; the router must reject on the RAW args — this test fails on
    any implementation that trusts ``result['parsed']``."""
    args = {**BASE_ARGS, "filters": {"cast": "Bruce Willis"}}
    raw = _tool_resp(args)
    r, _ = _structured_router([_structured_result(raw, parsed=_parsed_u(args))])
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out == deterministic_ask("q")
    assert any("cast" in n for n in notes), notes


def test_structured_parsing_error_is_schema_failure_not_crash():
    """The wrapper parks parser exceptions in ``parsing_error`` (include_raw).
    It is a schema failure: retried, then the deterministic ask — never a
    raised exception (C12)."""
    boom = _structured_result(_tool_resp(BASE_ARGS), parsing_error="no tool calls found")
    good = _structured_result(_tool_resp(BASE_ARGS))
    r, _ = _structured_router([boom, good])
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out.standalone_query == "q"


def test_structured_usage_granularity_kept_from_raw():
    """The answer to 'does B lose token granularity?': no — usage rides on
    result['raw'] (the AIMessage) and must be metered (#123)."""
    usage = {"input_tokens": 55, "output_tokens": 21, "total_tokens": 76}
    raw = _tool_resp(BASE_ARGS, usage=usage)
    r, _ = _structured_router([_structured_result(raw, parsed=_parsed_u(BASE_ARGS))])
    _, _, usage_out = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert usage_out is not None
    assert usage_out.prompt_tokens == 55 and usage_out.completion_tokens == 21


def test_structured_wrapper_forces_tool_choice():
    """with_structured_output does NOT force tool_choice by default — the
    bind kwargs must carry it (kwargs passthrough to bind_tools)."""
    r, llm = _structured_router([_structured_result(_tool_resp(BASE_ARGS))])
    r.understand("q", UserSessionPreferences(), [], None, 0)
    assert llm.seen["schema"] is SubmitUnderstanding
    assert llm.seen["kwargs"].get("tool_choice") == "SubmitUnderstanding"


# --- #113 fallback: the spare client gets the same forced tool -----------------


def test_fallback_binds_same_forced_tool():
    """Primary API-fails twice; the fallback attempt must bind the same
    schema with the same tool_choice and land a parsed Understanding."""
    r, _ = _router([RuntimeError("primary down"), RuntimeError("primary down")])
    fb = _ToolLLM([_tool_resp(BASE_ARGS)])
    r._fallback_llm = fb
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out.standalone_query == "q"
    assert any("fallback fired" in n for n in notes), notes
    assert fb.bind_seen["kwargs"].get("tool_choice") == "SubmitUnderstanding"