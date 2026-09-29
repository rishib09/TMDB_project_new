"""#119: forward-compat — checkpoint msgpack allowlist + st.components.v1.html.

The LangGraph checkpointer (langgraph-checkpoint 4.2.0) logs a deprecation
warning for every unregistered domain type riding msgpack and will block them
once strict. `src.ui.session` owns the allowlisted `JsonPlusSerializer`
(`_CHECKPOINT_SERDE`); these tests prove the allowlist is complete and
load-bearing without booting the UI.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SERDE_LOGGER = "langgraph.checkpoint.serde.jsonplus"


def _payload() -> dict:
    from src.domain.memory import UserSessionPreferences
    from src.domain.routing import IntentType, QueryRoutingDecision
    from src.maya.guardrails import GuardrailResult, GuardrailVerdict

    return {
        "guardrail": GuardrailResult(
            verdict=GuardrailVerdict.CLEAN, sanitized_query="x",
            matched_patterns=[], reason="r",
        ),
        "enum": IntentType.SEMANTIC_SEARCH,
        "decision": QueryRoutingDecision(
            intent=IntentType.SEMANTIC_SEARCH, confidence=0.9,
            standalone_query="x", requires_rag=True,
        ),
        "prefs": UserSessionPreferences(),
    }


def test_default_serde_warns_unregistered_for_all_five_types(caplog):
    """Negative control: the default serde warns for each of the five types —
    the exact behavior the session allowlist closes."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    with caplog.at_level(logging.WARNING, logger=SERDE_LOGGER):
        blob = JsonPlusSerializer().dumps_typed(_payload())
        JsonPlusSerializer().loads_typed(blob)
    warned = [ln for ln in caplog.text.splitlines() if "unregistered type" in ln]
    assert len(warned) == 5


def test_session_serde_round_trips_domain_types_without_warnings(caplog):
    """#119 adversarial: the session's allowlisted serde round-trips the five
    domain types with value equality and ZERO unregistered warnings.
    Fails on current code (no _CHECKPOINT_SERDE; bare InMemorySaver)."""
    from src.domain.memory import UserSessionPreferences
    from src.domain.routing import IntentType, QueryRoutingDecision
    from src.maya.guardrails import GuardrailVerdict
    from src.ui.session import _CHECKPOINT_SERDE

    payload = _payload()
    with caplog.at_level(logging.WARNING, logger=SERDE_LOGGER):
        out = _CHECKPOINT_SERDE.loads_typed(_CHECKPOINT_SERDE.dumps_typed(payload))
    assert out["enum"] == IntentType.SEMANTIC_SEARCH
    assert out["decision"].requires_rag is True
    assert out["prefs"] == UserSessionPreferences()
    assert out["guardrail"].verdict == GuardrailVerdict.CLEAN
    assert "unregistered type" not in caplog.text


def test_allowlist_missing_one_type_warns_only_for_that_type(caplog):
    """The allowlist is load-bearing: omitting one type blocks exactly it —
    the model silently degrades to a plain dict on restore."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from src.domain.memory import UserSessionPreferences
    from src.domain.routing import IntentType, QueryRoutingDecision
    from src.maya.guardrails import GuardrailVerdict, GuardrailResult

    partial = JsonPlusSerializer(
        allowed_msgpack_modules=[
            GuardrailVerdict, GuardrailResult, IntentType, QueryRoutingDecision,
        ]
    )  # UserSessionPreferences deliberately missing
    with caplog.at_level(logging.WARNING, logger=SERDE_LOGGER):
        out = partial.loads_typed(partial.dumps_typed({"u": UserSessionPreferences()}))
    assert not isinstance(out["u"], UserSessionPreferences)  # degraded to dict
    blocked = [ln for ln in caplog.text.splitlines() if "Blocked deserialization" in ln]
    assert len(blocked) == 1 and "UserSessionPreferences" in blocked[0]


def test_chat_tab_has_no_components_v1_reference():
    """#119: st.components.v1.html is past its removal horizon — the import
    and the call must be gone from the chat view."""
    src = (REPO / "src" / "ui" / "chat_tab.py").read_text(encoding="utf-8")
    assert "components.v1" not in src
    assert "streamlit.components" not in src


def test_strict_msgpack_with_session_serde_survives_subprocess():
    """CI-only strict-mode proof: with LANGGRAPH_STRICT_MSGPACK=true (read at
    import — hence subprocess), our allowlisted serde must round-trip the five
    types WITHOUT raising. Never set this env in production."""
    code = (
        "from src.ui.session import _CHECKPOINT_SERDE as serde\n"
        "from src.domain.routing import IntentType, QueryRoutingDecision\n"
        "from src.domain.memory import UserSessionPreferences\n"
        "from src.maya.guardrails import GuardrailVerdict, GuardrailResult\n"
        "payload = {\n"
        "    'e': IntentType.SEMANTIC_SEARCH,\n"
        "    'd': QueryRoutingDecision(intent=IntentType.SEMANTIC_SEARCH,\n"
        "                              confidence=0.9, standalone_query='x', requires_rag=True),\n"
        "    'p': UserSessionPreferences(),\n"
        "    'g': GuardrailResult(verdict=GuardrailVerdict.CLEAN, sanitized_query='x',\n"
        "                         matched_patterns=[], reason='r'),\n"
        "}\n"
        "out = serde.loads_typed(serde.dumps_typed(payload))\n"
        "assert out['e'] == IntentType.SEMANTIC_SEARCH\n"
        "assert isinstance(out['p'], UserSessionPreferences)\n"
        "assert isinstance(out['d'], QueryRoutingDecision)\n"
        "assert isinstance(out['g'], GuardrailResult)\n"
        "print('STRICT-OK')\n"
    )
    env = {**os.environ, "LANGGRAPH_STRICT_MSGPACK": "true",
           "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(REPO)}
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(REPO),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"strict mode rejected our allowlist:\n{r.stderr[-2000:]}"
    assert "STRICT-OK" in r.stdout
