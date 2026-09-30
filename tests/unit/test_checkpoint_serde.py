"""#119: forward-compat — checkpoint msgpack allowlist + st.components.v1.html.

The LangGraph checkpointer (langgraph-checkpoint 4.2.0) logs a deprecation
warning for every unregistered domain type riding msgpack and will block them
once strict. `src.ui.session` owns the allowlisted `JsonPlusSerializer`
(`_CHECKPOINT_SERDE`, types in `_CHECKPOINT_ALLOWLIST`); these tests prove the
allowlist is complete and load-bearing without booting the UI.

After #123, ``LLMUsage`` also rides the checkpoint — six types, not five.
Review: the payload is ONE source snippet (`_PAYLOAD_SOURCE`) exec'd both
in-process and in subprocesses, and every expectation (names, counts, partial
allowlist, strict isinstance checks) derives from `_CHECKPOINT_ALLOWLIST` —
a newly checkpointed type is one edit in session.py plus one line here.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SERDE_LOGGER = "langgraph.checkpoint.serde.jsonplus"

#: Shared payload source (keys are the type names): exec'd in-process by
#: `_payload()` and spliced into every subprocess script, so the two can
#: never drift. GuardrailVerdict also rides top-level so the drift guard
#: stays a plain set comparison; its nested occurrence inside GuardrailResult
#: is deduplicated by langgraph's once-per-type warning budget.
_PAYLOAD_SOURCE = """\
from src.domain.memory import UserSessionPreferences
from src.domain.routing import IntentType, QueryRoutingDecision
from src.domain.usage import LLMUsage
from src.maya.guardrails import GuardrailResult, GuardrailVerdict

payload = {
    "GuardrailVerdict": GuardrailVerdict.CLEAN,
    "GuardrailResult": GuardrailResult(
        verdict=GuardrailVerdict.CLEAN, sanitized_query="x",
        matched_patterns=[], reason="r",
    ),
    "IntentType": IntentType.SEMANTIC_SEARCH,
    "QueryRoutingDecision": QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH, confidence=0.9,
        standalone_query="x", requires_rag=True,
    ),
    "UserSessionPreferences": UserSessionPreferences(),
    "LLMUsage": LLMUsage(model="fake", prompt_tokens=1, completion_tokens=1),
}
"""


def _allowlist() -> tuple[type, ...]:
    from src.ui.session import _CHECKPOINT_ALLOWLIST

    return _CHECKPOINT_ALLOWLIST


def _payload() -> dict:
    """The shared payload, built in-process from the same source."""
    ns: dict = {}
    exec(_PAYLOAD_SOURCE, ns)  # noqa: S102 - trusted repo-local snippet
    return ns["payload"]


def _run_subprocess(code: str, **extra_env) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(REPO), **extra_env}
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=str(REPO),
                          capture_output=True, text=True, timeout=300)


def test_payload_covers_every_allowlisted_type():
    """Drift guard: the shared payload exercises every allowlisted type —
    a type added to `_CHECKPOINT_ALLOWLIST` without a payload line fails here
    instead of silently shrinking coverage."""
    assert set(_payload()) == {t.__name__ for t in _allowlist()}


def test_default_serde_warns_unregistered_for_all_six_types():
    """Negative control: the default serde warns for each allowlisted type —
    the exact behavior the session allowlist closes.

    subprocess-isolated: langgraph dedups these warnings once per (module,
    qualname) per process (_warned_unregistered_types), so in a full battery
    any earlier default-checkpointer test burns the budget and caplog sees
    zero lines. A fresh interpreter makes the count deterministic."""
    code = (
        "import logging\n"
        "logging.basicConfig(level=logging.WARNING)\n"
        "from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer\n"
        f"{_PAYLOAD_SOURCE}"
        "blob = JsonPlusSerializer().dumps_typed(payload)\n"
        "JsonPlusSerializer().loads_typed(blob)\n"
    )
    r = _run_subprocess(code)
    assert r.returncode == 0, f"negative control crashed:\n{r.stderr[-2000:]}"
    warned = [ln for ln in r.stderr.splitlines() if "unregistered type" in ln]
    expected = {t.__name__ for t in _allowlist()}
    named = {t for t in expected if any(t in ln for ln in warned)}
    assert len(warned) == len(expected), f"expected {len(expected)} warnings:\n{warned}"
    assert named == expected, f"missing warnings for: {expected - named}"


def test_session_serde_round_trips_domain_types_without_warnings(caplog):
    """#119 adversarial: the session's allowlisted serde round-trips the
    domain types with value equality and ZERO unregistered warnings."""
    import logging

    from src.ui.session import _CHECKPOINT_SERDE

    payload = _payload()
    with caplog.at_level(logging.WARNING, logger=SERDE_LOGGER):
        out = _CHECKPOINT_SERDE.loads_typed(_CHECKPOINT_SERDE.dumps_typed(payload))
    assert out["IntentType"] == payload["IntentType"]
    assert out["QueryRoutingDecision"].requires_rag is True
    assert out["UserSessionPreferences"] == payload["UserSessionPreferences"]
    assert out["GuardrailVerdict"] == payload["GuardrailVerdict"]
    assert out["GuardrailResult"].verdict == payload["GuardrailVerdict"]
    assert isinstance(out["LLMUsage"], type(payload["LLMUsage"]))
    assert out["LLMUsage"].prompt_tokens == 1
    assert "unregistered type" not in caplog.text


def test_allowlist_missing_one_type_warns_only_for_that_type(caplog):
    """The allowlist is load-bearing: omitting one type blocks exactly it —
    the value silently degrades to a plain dict on restore."""
    import logging

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from src.domain.memory import UserSessionPreferences

    partial = JsonPlusSerializer(
        allowed_msgpack_modules=[
            t for t in _allowlist() if t is not UserSessionPreferences
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
    registered type WITHOUT raising, and every restored value keeps its type.
    Never set this env in production."""
    code = (
        "from src.ui.session import _CHECKPOINT_SERDE as serde, _CHECKPOINT_ALLOWLIST\n"
        f"{_PAYLOAD_SOURCE}"
        "out = serde.loads_typed(serde.dumps_typed(payload))\n"
        "assert set(payload) == {t.__name__ for t in _CHECKPOINT_ALLOWLIST}\n"
        "by_name = {t.__name__: t for t in _CHECKPOINT_ALLOWLIST}\n"
        "assert all(type(out[name]) is t for name, t in by_name.items())\n"
        "print('STRICT-OK')\n"
    )
    r = _run_subprocess(code, LANGGRAPH_STRICT_MSGPACK="true")
    assert r.returncode == 0, f"strict mode rejected our allowlist:\n{r.stderr[-2000:]}"
    assert "STRICT-OK" in r.stdout
