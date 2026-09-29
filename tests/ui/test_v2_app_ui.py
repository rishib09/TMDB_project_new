"""#113: the UI tier — the real Streamlit app, in-process via AppTest.

The golden set grades routing at the graph layer; the Feedback Window
label is a UI-layer transform (`session._build_turn_row`) that no graph
test executes. That gap let the v2 "via refusal" mislabel ship. These
tests drive `app.py` through `streamlit.testing.v1.AppTest` (1.62.0):
`chat_input` sends visitor turns, `session_state` exposes the turn log.

- Offline (`integration`): fake router/synthesizer/engine injected at
  construction — asserts the wiring and the v2 path labels, CI-safe.
- Live (`live`, opt-in): replays 9 tier-representative golden
  conversations on the v2 stack (glm primary, flash-lite fallback),
  asserting no routed turn is ever labeled "refusal".
"""

from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from src.domain.routing import IntentType
from src.maya.v2 import MayaV2Router, PreferenceDelta, Understanding

APP = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "app.py")
NINE_CONVERSATIONS = ["C01", "C05", "C06", "C09", "C12", "C15", "C18", "C21", "C23"]


# --- fakes (constructor-compatible with the real collaborators) --------------
class _FakeRouter(MayaV2Router):
    """Scripted Understand: first call asks, every later call retrieves.

    Subclasses the real router so the orchestrator's isinstance stack
    selector wires the v2 route node; only ``understand`` is scripted."""

    def __init__(self, config, api_key: str | None = None):
        super().__init__(config, api_key="test-key")
        self.calls = 0

    def understand(self, query, prefs, shown_titles, last_assistant, probe_count):
        self.calls += 1
        if self.calls == 1:
            return (
                Understanding(
                    intent=IntentType.SEMANTIC_SEARCH,
                    standalone_query=query,
                    confidence=0.9,
                    clarifying_question="What mood are you after?",
                ),
                ["scripted ask"],
                None,  # #123 usage — stubbed client meters nothing
            )
        return (
            Understanding(
                intent=IntentType.SEMANTIC_SEARCH,
                standalone_query=query,
                confidence=0.9,
                ready_to_retrieve=True,
                preference_delta=PreferenceDelta(set_mood="feel-good"),
            ),
            ["scripted retrieve"],
            None,
        )


class _FakeSynthesizer:
    def __init__(self, config, api_key: str | None = None):
        self.config = config

    def synthesize(self, query, decision, movies, history):
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, model="fake")
        return f"Fake reply to: {query}", usage


class _FakeEngine:
    def __init__(self, **kwargs):
        pass

    def retrieve(self, query, routing, top_k=8, candidate_pool=50, shown_ids=None):
        return []


def _boot_app(monkeypatch) -> AppTest:
    """The real app, in-process, with the heavy collaborators faked."""
    from src.ui import session as session_module

    monkeypatch.setenv("MAYA_ROUTING_STACK", "v2")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(session_module, "MayaV2Router", _FakeRouter)
    monkeypatch.setattr(session_module, "MayaSynthesizer", _FakeSynthesizer)
    monkeypatch.setattr(session_module, "HybridRetrievalEngine", _FakeEngine)
    monkeypatch.setattr(session_module, "shared_vector_store", lambda *a, **k: None)
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


# --- offline: the wiring + the label -----------------------------------------
pytestmark = pytest.mark.integration


def test_app_boots_on_v2_and_labels_routed_turns(monkeypatch):
    at = _boot_app(monkeypatch)
    assert not at.exception
    session = at.session_state["maya_session"]
    assert session.config.routing_stack == "v2"

    for query in ["show me some movies", "feel good", "under two hours"]:
        at.chat_input[0].set_value(query).run()
        assert not at.exception

    rows = session.turn_log
    assert [r["path"] for r in rows] == ["ask", "retrieve", "retrieve"]
    assert all(r["path"] != "refusal" for r in rows)
    assert all(r["intent"] == "SEMANTIC_SEARCH" for r in rows)


def test_admin_input_routes_as_ordinary_turn(monkeypatch):
    """#91: /admin is dead — the Lab is public, so typing it must process
    as an ordinary message (one turn in the log), not a swallowed branch."""
    at = _boot_app(monkeypatch)
    at.chat_input[0].set_value("/admin").run()
    assert not at.exception
    session = at.session_state["maya_session"]
    assert len(session.turn_log) == 1  # the old admin branch swallowed the input


def test_app_rerun_does_not_double_turn(monkeypatch):
    """chat_tab draws standing-grid replies on idle reruns only (#80)."""
    at = _boot_app(monkeypatch)
    at.chat_input[0].set_value("show me some movies").run()
    n_after_turn = len(at.session_state["maya_session"].turn_log)
    at.run()  # idle rerun — no new input
    assert len(at.session_state["maya_session"].turn_log) == n_after_turn


# --- live: 9 golden conversations through the real v2 stack ------------------
def _load_conversations():
    from src.evals.conversations import load_conversations

    return {c.id: c for c in load_conversations().conversations}


@pytest.mark.live
@pytest.mark.skipif(
    not __import__("os").getenv("OPENROUTER_API_KEY"),
    reason="live: requires OPENROUTER_API_KEY",
)
def test_nine_conversations_through_the_ui_never_mislabel_refusal():
    """v2/glm live replay; the #113 acceptance on the real stack."""
    import os

    os.environ["MAYA_ROUTING_STACK"] = "v2"  # read at session construction
    convos = _load_conversations()
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    session = at.session_state["maya_session"]
    assert session.config.routing_stack == "v2"

    for cid in NINE_CONVERSATIONS:
        convo = convos[cid]
        for turn in convo.turns:
            at.chat_input[0].set_value(turn.user).run()
            assert not at.exception, f"{cid} t{turn.n} crashed the app"
            row = session.turn_log[-1]
            assert row["query"] == turn.user
            assert row["path"] != "refusal" or row["intent"] in (
                "OUT_OF_SCOPE",
                "GREETING",
                "CAPABILITIES",
            ), f"{cid} t{turn.n}: routed turn labeled refusal ({row})"
        print(f"{cid}: {len(convo.turns)} turns OK")
