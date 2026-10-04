"""Unit tests for the v2 stack pieces (#106): Understand router (stubbed
client — no LLM), projection, prompt/state-block builders."""

import json

from src.domain.config import ExperimentConfig
from src.domain.memory import (
    PreferencesUpdate,
    UserSessionPreferences,
    merge_preferences,
)
from src.domain.routing import IntentType, MetadataFilterCriteria
from src.maya.v2 import (
    MayaV2Router,
    Understanding,
    build_state_block,
    deterministic_ask,
    project_understanding,
)
from src.maya.v2.router import _tool_args, strict_understanding

CFG = ExperimentConfig()
#: The prose transport these ladder tests were written against — the knob
#: default is the #150 tool_call transport, so the fence/retry/degradation
#: tests pin ``prompt_json`` explicitly (ADR 0004 tunable).
CFG_PROSE = ExperimentConfig(v2_understand_transport="prompt_json")


def _u(**kw) -> Understanding:
    defaults = dict(intent=IntentType.SEMANTIC_SEARCH, standalone_query="q", confidence=0.9)
    defaults.update(kw)
    return Understanding(**defaults)


def _content(u: Understanding | None, *, fenced: bool = False) -> str:
    """A model response body for u: fenced JSON (the glm tic) or invalid text."""
    if u is None:
        return "I cannot produce JSON today, sorry."
    body = json.dumps(u.model_dump(mode="json"))
    return f"```json\n{body}\n```" if fenced else body


def _router_with(responses) -> MayaV2Router:
    """Router whose client pops scripted response bodies (str) or raises.
    Pinned to ``prompt_json``: these tests exercise the prose-path fence and
    the C12 ladder, not the #150 tool transports."""
    r = MayaV2Router(CFG_PROSE, api_key="test-key")
    scripted = list(responses)

    class _Resp:
        def __init__(self, content):
            self.content = content

    class _LLM:
        def invoke(self, messages):
            item = scripted.pop(0)
            if isinstance(item, Exception):
                raise item
            return _Resp(item)

    r._llm = _LLM()
    return r


# --- Understand call: C12 retry + degradation ---------------------------------


def test_understand_happy_path_applies_guards():
    u = _u(clarifying_question="x" * 500)  # guard must template it
    r = _router_with([_content(u)])
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out.clarifying_question == "Tell me a mood or a genre and I'll find something good."
    assert notes and "length check" in notes[0]


def test_understand_strips_fences_in_the_same_attempt():
    """Live smoke finding (2026-09-27): glm wraps JSON in a ```json fence.
    A formatting tic — stripped and validated within ONE attempt, so the
    C12 retry budget stays unspent."""
    r = _router_with([_content(_u(), fenced=True)])
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out.standalone_query == "q"
    assert notes == []


def test_understand_schema_retry_then_success():
    r = _router_with(["not json at all", _content(_u())])
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out.standalone_query == "q"
    assert any("retrying" in n for n in notes)


def test_understand_double_schema_failure_lands_deterministic_ask():
    r = _router_with(["nope", "still nope"])
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out == deterministic_ask("q")
    assert any("C12" in n for n in notes)


def test_understand_api_error_skips_retry_and_degrades_recorded():
    r = _router_with([RuntimeError("endpoint down")])
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out.ready_to_retrieve is False
    assert any("api_error" in n for n in notes)


def test_understand_builds_c14_payload():
    seen = {}

    class _Resp:
        content = _content(_u())

    class _LLM:
        def invoke(self, messages):
            seen["messages"] = messages
            return _Resp()

    r = MayaV2Router(CFG_PROSE, api_key="test-key")
    r._llm = _LLM()
    prefs = UserSessionPreferences(preferred_mood="scary", excluded_genres=["Horror"])
    r.understand("the query", prefs, ["Inception"], "last reply", 1)
    roles = [m[0] for m in seen["messages"]]
    assert roles == ["system", "system", "system", "human"]
    assert "Mood: scary" in seen["messages"][1][1]
    assert "Inception" in seen["messages"][1][1]
    assert "Horror" in seen["messages"][1][1]
    assert seen["messages"][2][1].startswith("MAYA'S LAST REPLY")


# --- state block -----------------------------------------------------------------


def test_state_block_empty_state():
    block = build_state_block(UserSessionPreferences(), [])
    assert "(none)" in block and "(none yet)" in block


# --- projection: the one seam ------------------------------------------------------


def test_projection_maps_retrieval_intents():
    for intent in (IntentType.SEMANTIC_SEARCH, IntentType.ATTRIBUTE_FILTER,
                   IntentType.SUPERLATIVE_RANKING, IntentType.NEGATION_EXCLUSION):
        d = project_understanding(_u(intent=intent), UserSessionPreferences())
        assert d.requires_rag is True


def test_projection_non_retrieval_intents():
    for intent in (IntentType.GREETING, IntentType.CAPABILITIES, IntentType.OUT_OF_SCOPE):
        d = project_understanding(_u(intent=intent), UserSessionPreferences())
        assert d.requires_rag is False


def test_projection_carries_filters_including_c6():
    f = MetadataFilterCriteria(runtime_max=90, rating_min=7.5)
    d = project_understanding(_u(filters=f), UserSessionPreferences())
    assert d.filters.runtime_max == 90 and d.filters.rating_min == 7.5


def test_projection_mood_audience_from_snapshot_not_delta():
    prefs = UserSessionPreferences(preferred_mood="scary", audience="kids")
    d = project_understanding(_u(), prefs)
    assert d.mood == "scary" and d.audience == "kids"
    assert d.is_fallback is False


# --- PreferencesUpdate: snapshot rides verbatim (#106 resurrect guard) -------------


def test_replace_snapshot_never_resurrects_removed_genres():
    cur = UserSessionPreferences(preferred_genres=["Comedy", "Drama"])
    snap = UserSessionPreferences(preferred_genres=["Drama"])
    out = merge_preferences(cur, PreferencesUpdate(prefs=snap, replace=True))
    assert out.preferred_genres == ["Drama"]  # union would resurrect Comedy


def test_non_replace_update_delegates_to_merge():
    cur = UserSessionPreferences(preferred_mood="scary")
    out = merge_preferences(cur, PreferencesUpdate(prefs=UserSessionPreferences(preferred_genres=["Horror"])))
    assert out.preferred_mood == "scary"
    assert out.preferred_genres == ["Horror"]


# --- #113: Understand fallback model ------------------------------------------


def test_fallback_fires_when_primary_api_fails(monkeypatch):
    class _FB:
        def invoke(self, messages):
            return type("R", (), {"content": _content(_u())})()

    r = _router_with([RuntimeError("primary down")])
    monkeypatch.setattr(r, "_fallback_llm", _FB())
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out is not None
    assert any(
        "fallback fired" in n and "gemini-3.5-flash-lite" in n for n in notes
    )


def test_fallback_skips_when_primary_succeeds(monkeypatch):
    calls = []

    class _Primary:
        def invoke(self, messages):
            calls.append("primary")
            return type("R", (), {"content": _content(_u())})()

    r = _router_with([])
    r._llm = _Primary()
    monkeypatch.setattr(r, "_fallback_llm", _Primary())
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out is not None
    assert calls == ["primary"]
    assert not any("fallback" in n for n in notes)


def test_fallback_error_also_recorded_then_ask(monkeypatch):
    class _Dead:
        def invoke(self, messages):
            raise RuntimeError("fallback down too")

    r = _router_with([RuntimeError("primary down")])
    monkeypatch.setattr(r, "_fallback_llm", _Dead())
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out is not None  # deterministic ask still lands
    assert any("fallback error" in n for n in notes)
    assert any("deterministic ask" in n for n in notes)


def test_fallback_default_config_points_at_flashlite():
    assert ExperimentConfig().v2_router_fallback_model == "gemini-3.5-flash-lite"


# --- #123: usage reporting -----------------------------------------------------


def _resp_with_usage(content: str, input_tokens: int, output_tokens: int):
    """A langchain-shaped response carrying usage_metadata (#123)."""

    class _Resp:
        def __init__(self):
            self.content = content
            self.usage_metadata = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

    return _Resp()


def test_understand_sums_usage_across_schema_retry():
    """#123: the C12 retry consumed tokens too — reported usage is the SUM."""
    r = _router_with([])
    scripted = [
        _resp_with_usage("not json", 40, 10),
        _resp_with_usage(_content(_u()), 50, 20),
    ]

    class _LLM:
        def invoke(self, messages):
            return scripted.pop(0)

    r._llm = _LLM()
    out, notes, usage = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out.standalone_query == "q"
    assert usage is not None
    assert usage.prompt_tokens == 90
    assert usage.completion_tokens == 30
    assert usage.model == CFG.v2_router_model


def test_understand_fallback_tokens_price_at_primary_rate():
    """#123 D3: fallback-attempt tokens land in ONE usage attributed to the
    PRIMARY model (bounded pricing error, disclosed in _summed_usage)."""
    r = _router_with([RuntimeError("primary down")])

    class _FB:
        def invoke(self, messages):
            return _resp_with_usage(_content(_u()), 70, 30)

    r._fallback_llm = _FB()
    out, notes, usage = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out is not None
    assert any("fallback fired" in n for n in notes)
    assert usage is not None
    assert usage.prompt_tokens == 70 and usage.completion_tokens == 30
    assert usage.model == CFG.v2_router_model  # primary, not the fallback model


def test_understand_stub_without_usage_meters_none():
    """Stubs without usage_metadata (all existing tests) meter nothing."""
    r = _router_with([_content(_u())])
    out, notes, usage = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out is not None
    assert usage is None


# --- #150: strict arg validation (pure) ----------------------------------------

BASE_ARGS = {"intent": IntentType.SEMANTIC_SEARCH, "standalone_query": "q", "confidence": 0.9}


def test_strict_understanding_passes_exact_keys():
    out, reason = strict_understanding(BASE_ARGS)
    assert reason is None
    assert out is not None and out.standalone_query == "q"
    assert type(out) is Understanding  # class-strict equality across the stack


def test_strict_understanding_rejects_unknown_at_every_level():
    for bad, needle in (
        ({**BASE_ARGS, "lead_actor": "Bruce Willis"}, "lead_actor"),
        ({**BASE_ARGS, "filters": {"cast": "Bruce Willis"}}, "cast"),
        ({**BASE_ARGS, "preference_delta": {"add_cast": ["x"]}}, "add_cast"),
        ({**BASE_ARGS, "intent": "MOVIE_SEARCH"}, "intent"),  # outside the taxonomy
    ):
        out, reason = strict_understanding(bad)
        assert out is None
        assert needle in reason


def test_strict_understanding_valid_filters_survive():
    f = {"cast_member": "Bruce Willis", "genres": ["Action"]}
    out, reason = strict_understanding({**BASE_ARGS, "filters": f})
    assert reason is None
    assert out.filters.cast_member == "Bruce Willis"
    assert out.filters.genres == ["Action"]


def test_strict_understanding_non_dict_args():
    out, reason = strict_understanding("not a dict")
    assert out is None and "not a dict" in reason


def test_tool_args_extraction_shapes():
    resp = type("R", (), {"tool_calls": [{"name": "SubmitUnderstanding", "args": BASE_ARGS}]})()
    args, reason = _tool_args(resp)
    assert reason is None and args == BASE_ARGS

    args, reason = _tool_args(type("R", (), {"tool_calls": []})())
    assert args is None and reason == "no tool call"

    args, reason = _tool_args(type("R", (), {"tool_calls": None})())
    assert args is None and reason == "no tool call"

    ns_call = type("C", (), {"args": BASE_ARGS})()
    args, reason = _tool_args(type("R", (), {"tool_calls": [ns_call]})())
    assert reason is None and args == BASE_ARGS

    args, reason = _tool_args(type("R", (), {"tool_calls": [{"args": "junk"}]})())
    assert args is None and reason == "tool call args not a dict"


def test_config_defaults_tool_call_transport():
    assert CFG.v2_understand_transport == "tool_call"  # #150 default (option A)


def test_understand_tool_transport_notes_the_transport():
    """Telemetry rule: the transport in force is visible in the trace."""
    r = _router_with([_content(_u())])
    r.config = ExperimentConfig(v2_understand_transport="tool_call")
    r._llm = _ToolCallLLM([_tool_msg(BASE_ARGS)])
    out, notes, _ = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out.standalone_query == "q"
    assert notes[0].startswith("understand transport=tool_call")


class _ToolCallLLM:
    """Minimal tool_call-transport client: bind_tools scripts one response."""

    def __init__(self, responses):
        self._responses = list(responses)

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        return self._responses.pop(0)


def _tool_msg(args: dict):
    from langchain_core.messages import AIMessage

    return AIMessage(
        content="",
        tool_calls=[{"name": "SubmitUnderstanding", "args": args, "id": "c1", "type": "tool_call"}],
    )
