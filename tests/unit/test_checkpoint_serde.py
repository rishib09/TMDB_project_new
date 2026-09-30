"""#119: forward-compat — checkpoint msgpack allowlist + st.components.v1.html.

The LangGraph checkpointer (langgraph-checkpoint 4.2.0) logs a deprecation
warning for every unregistered domain type riding msgpack and will block them
once strict. `src.ui.session` owns the allowlisted `JsonPlusSerializer`
(`_CHECKPOINT_SERDE`); these tests prove the allowlist is complete and
load-bearing without booting the UI.

After #123, ``LLMUsage`` also rides the checkpoint — six types, not five.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SERDE_LOGGER = "langgraph.checkpoint.serde.jsonplus"
#: Every domain type the session allowlist must cover (warn count / strict).
_ALLOWED_COUNT = 6


def _payload() -> dict:
    from src.domain.memory import UserSessionPreferences
    from src.domain.routing import IntentType, QueryRoutingDecision
    from src.domain.usage import LLMUsage
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
        "usage": LLMUsage(model="fake", prompt_tokens=1, completion_tokens=1),
    }


def test_default_serde_warns_unregistered_for_all_six_types():
    """Negative control: the default serde warns for each allowlisted type —
    the exact behavior the session allowlist closes.

    Subprocess-isolated: langgraph dedups these warnings once per (module,
    qualname) per process (_warned_unregistered_types), so in a full battery
    any earlier default-checkpointer test burns the budget and caplog sees
    zero lines. A fresh interpreter makes the count deterministic."""
    import textwrap

    code = textwrap.dedent(
        """
        import logging
        logging.basicConfig(level=logging.WARNING)
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        from src.domain.memory import UserSessionPreferences
        from src.domain.routing import IntentType, QueryRoutingDecision
        from src.domain.usage import LLMUsage
        from src.maya.guardrails import GuardrailResult, GuardrailVerdict
        payload = {
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
            "usage": LLMUsage(model="fake", prompt_tokens=1, completion_tokens=1),
        }
        blob = JsonPlusSerializer().dumps_typed(payload)
        JsonPlusSerializer().loads_typed(blob)
        """
    )
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(REPO)}
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(REPO),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"negative control crashed:\n{r.stderr[-2000:]}"
    warned = [ln for ln in r.stderr.splitlines() if "unregistered type" in ln]
    expected = {
        "GuardrailVerdict", "GuardrailResult", "IntentType",
        "QueryRoutingDecision", "UserSessionPreferences", "LLMUsage",
    }
    named = {t for t in expected if any(t in ln for ln in warned)}
    assert len(warned) == _ALLOWED_COUNT, f"expected {_ALLOWED_COUNT} warnings:\n{warned}"
    assert named == expected, f"missing warnings for: {expected - named}"


def test_session_serde_round_trips_domain_types_without_warnings(caplog):
    """#119 adversarial: the session's allowlisted serde round-trips the
    domain types with value equality and ZERO unregistered warnings."""
    from src.domain.memory import UserSessionPreferences
    from src.domain.routing import IntentType, QueryRoutingDecision
    from src.domain.usage import LLMUsage
    from src.maya.guardrails import GuardrailVerdict
    from src.ui.session import _CHECKPOINT_SERDE

    payload = _payload()
    with caplog.at_level(logging.WARNING, logger=SERDE_LOGGER):
        out = _CHECKPOINT_SERDE.loads_typed(_CHECKPOINT_SERDE.dumps_typed(payload))
    assert out["enum"] == IntentType.SEMANTIC_SEARCH
    assert out["decision"].requires_rag is True
    assert out["prefs"] == UserSessionPreferences()
    assert out["guardrail"].verdict == GuardrailVerdict.CLEAN
    assert isinstance(out["usage"], LLMUsage)
    assert out["usage"].prompt_tokens == 1
    assert "unregistered type" not in caplog.text


def test_allowlist_missing_one_type_warns_only_for_that_type(caplog):
    """The allowlist is load-bearing: omitting one type blocks exactly it —
    the value silently degrades to a plain dict on restore."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from src.domain.memory import UserSessionPreferences
    from src.domain.routing import IntentType, QueryRoutingDecision
    from src.domain.usage import LLMUsage
    from src.maya.guardrails import GuardrailResult, GuardrailVerdict

    partial = JsonPlusSerializer(
        allowed_msgpack_modules=[
            GuardrailVerdict, GuardrailResult, IntentType, QueryRoutingDecision,
            LLMUsage,
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
    import — hence subprocess), our allowlisted serde must round-trip every
    registered type WITHOUT raising. Never set this env in production."""
    code = (
        "from src.ui.session import _CHECKPOINT_SERDE as serde\n"
        "from src.domain.routing import IntentType, QueryRoutingDecision\n"
        "from src.domain.memory import UserSessionPreferences\n"
        "from src.domain.usage import LLMUsage\n"
        "from src.maya.guardrails import GuardrailVerdict, GuardrailResult\n"
        "payload = {\n"
        "    'e': IntentType.SEMANTIC_SEARCH,\n"
        "    'd': QueryRoutingDecision(intent=IntentType.SEMANTIC_SEARCH,\n"
        "                              confidence=0.9, standalone_query='x', requires_rag=True),\n"
        "    'p': UserSessionPreferences(),\n"
        "    'g': GuardrailResult(verdict=GuardrailVerdict.CLEAN, sanitized_query='x',\n"
        "                         matched_patterns=[], reason='r'),\n"
        "    'u': LLMUsage(model='fake', prompt_tokens=1, completion_tokens=1),\n"
        "}\n"
        "out = serde.loads_typed(serde.dumps_typed(payload))\n"
        "assert out['e'] == IntentType.SEMANTIC_SEARCH\n"
        "assert isinstance(out['p'], UserSessionPreferences)\n"
        "assert isinstance(out['d'], QueryRoutingDecision)\n"
        "assert isinstance(out['g'], GuardrailResult)\n"
        "assert isinstance(out['u'], LLMUsage)\n"
        "print('STRICT-OK')\n"
    )
    env = {**os.environ, "LANGGRAPH_STRICT_MSGPACK": "true",
           "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(REPO)}
    r = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(REPO),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"strict mode rejected our allowlist:\n{r.stderr[-2000:]}"
    assert "STRICT-OK" in r.stdout
