"""Unit tests for the v2 stack pieces (#106): Understand router (stubbed
client — no LLM), projection, prompt/state-block builders."""

import json

import pytest

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

CFG = ExperimentConfig()


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
    """Router whose client pops scripted response bodies (str) or raises."""
    r = MayaV2Router(CFG, api_key="test-key")
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
    out, notes = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out.clarifying_question == "Tell me a mood or a genre and I'll find something good."
    assert notes and "length check" in notes[0]


def test_understand_strips_fences_in_the_same_attempt():
    """Live smoke finding (2026-09-27): glm wraps JSON in a ```json fence.
    A formatting tic — stripped and validated within ONE attempt, so the
    C12 retry budget stays unspent."""
    r = _router_with([_content(_u(), fenced=True)])
    out, notes = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out.standalone_query == "q"
    assert notes == []


def test_understand_schema_retry_then_success():
    r = _router_with(["not json at all", _content(_u())])
    out, notes = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out.standalone_query == "q"
    assert any("retrying" in n for n in notes)


def test_understand_double_schema_failure_lands_deterministic_ask():
    r = _router_with(["nope", "still nope"])
    out, notes = r.understand("q", UserSessionPreferences(), [], None, 0)
    assert out == deterministic_ask("q")
    assert any("C12" in n for n in notes)


def test_understand_api_error_skips_retry_and_degrades_recorded():
    r = _router_with([RuntimeError("endpoint down")])
    out, notes = r.understand("q", UserSessionPreferences(), [], None, 0)
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

    r = MayaV2Router(CFG, api_key="test-key")
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
