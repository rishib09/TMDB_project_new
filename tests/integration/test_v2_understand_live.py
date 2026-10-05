"""Live z.ai integration tests for the v2 Understand transports (#150).

These hit the real z.ai API and are OPT-IN:
    npx @dotenvx/dotenvx run -- ./.venv/Scripts/python.exe -m pytest \\
        tests/integration/test_v2_understand_live.py -m live -v -s
    (requires ZAI_API_KEY in env; PYTHONIOENCODING=utf-8)

They reproduce the #149 window — five live Bruce Willis turns where the
model invented ``filters.cast`` in prose JSON and the actor vanished —
against BOTH #150 tool transports:

- ``tool_call``         — bind_tools + forced tool_choice (option A)
- ``structured_output`` — with_structured_output(function_calling,
  include_raw=True) + the same strict backstop (option B)

A final report test prints the per-transport token/notes comparison used to
pick the config default (ADR 0004: evidence, not taste).
"""

import os

import pytest

from src.domain.config import ExperimentConfig
from src.domain.memory import UserSessionPreferences
from src.maya.v2 import MayaV2Router

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.getenv("ZAI_API_KEY"),
        reason="ZAI_API_KEY not set — skipping live z.ai tests",
    ),
]

#: The #149 conversation window, one self-contained turn per row (the
#: multi-turn memory threading is the orchestrator's job — offline-tested).
BRUCE_TURNS = [
    ("show me some Bruce Willis movies", {"cast": "Bruce Willis"}),
    ("action movies starring Bruce Willis", {"cast": "Bruce Willis", "genres": "Action"}),
    ("recent Bruce Willis movies", {"cast": "Bruce Willis"}),
]

RESULTS: list[dict] = []


def _router_for(transport: str) -> MayaV2Router:
    return MayaV2Router(
        ExperimentConfig(v2_understand_transport=transport), api_key=None
    )


def _run_turn(router: MayaV2Router, query: str) -> tuple:
    out, notes, usage = router.understand(query, UserSessionPreferences(), [], None, 0)
    return out, notes, usage


def _assert_cast_lands(out, notes) -> None:
    """The #149 core: the named actor MUST reach ``filters.cast_member``."""
    assert out.filters is not None, f"filters lost entirely; notes={notes}"
    assert out.filters.cast_member == "Bruce Willis", (
        f"cast_member={out.filters.cast_member!r}; notes={notes}"
    )


@pytest.mark.live
@pytest.mark.parametrize("transport", ["tool_call", "structured_output"])
def test_live_named_cast_lands_in_filters(transport):
    """Turn 1 — the exact silent-drop of #149, now loud if it fails."""
    out, notes, usage = _run_turn(_router_for(transport), BRUCE_TURNS[0][0])
    RESULTS.append(
        {"transport": transport, "turn": 1, "notes": notes, "usage": _usage(usage)}
    )
    assert not any("strict rejection" in n or "parsing_error" in n for n in notes), notes
    _assert_cast_lands(out, notes)


@pytest.mark.live
@pytest.mark.parametrize("transport", ["tool_call", "structured_output"])
def test_live_genre_and_cast_together(transport):
    """Turn 2 — genre + actor in one reading; genres hit the closed vocab."""
    out, notes, usage = _run_turn(_router_for(transport), BRUCE_TURNS[1][0])
    RESULTS.append(
        {"transport": transport, "turn": 2, "notes": notes, "usage": _usage(usage)}
    )
    assert not any("strict rejection" in n or "parsing_error" in n for n in notes), notes
    _assert_cast_lands(out, notes)
    got = (out.filters.genres or []) + (out.preference_delta.add_genres or [])
    assert any("Action" in g for g in got), f"genres={out.filters.genres!r}; notes={notes}"


@pytest.mark.live
@pytest.mark.parametrize("transport", ["tool_call", "structured_output"])
def test_live_era_signal_survives(transport):
    """Turn 3 — vague era language maps to the era label or concrete years,
    never an invented key."""
    out, notes, usage = _run_turn(_router_for(transport), BRUCE_TURNS[2][0])
    RESULTS.append(
        {"transport": transport, "turn": 3, "notes": notes, "usage": _usage(usage)}
    )
    assert not any("strict rejection" in n or "parsing_error" in n for n in notes), notes
    _assert_cast_lands(out, notes)
    era_hit = out.era == "recent" or out.decade or (
        out.filters is not None and out.filters.year_min
    )
    assert era_hit, f"era={out.era!r} decade={out.decade!r}; notes={notes}"


def _usage(usage) -> dict:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0}
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
    }


@pytest.mark.live
def test_v150_transport_comparison_report():
    """Runs last (file order): prints the A-vs-B comparison table the config
    default decision is made from (ADR 0004). Asserts the runs happened."""
    assert len(RESULTS) >= 6, "transport comparison needs both transports x 3 turns"
    print("\n\n#150 transport comparison (z.ai glm-5.3-flash, live):")
    print(f"{'transport':<20}{'turn':<6}{'prompt':<8}{'completion':<12}notes")
    for row in RESULTS:
        u = row["usage"]
        print(
            f"{row['transport']:<20}{row['turn']:<6}{u['prompt_tokens']:<8}"
            f"{u['completion_tokens']:<12}{row['notes'] or ['(clean)']}"
        )
    for transport in ("tool_call", "structured_output"):
        rows = [r for r in RESULTS if r["transport"] == transport]
        pt = sum(r["usage"]["prompt_tokens"] for r in rows)
        ct = sum(r["usage"]["completion_tokens"] for r in rows)
        print(
            f"{transport:<20}{'TOTAL':<6}{pt:<8}{ct:<12}"
            f"retries={sum(1 for r in rows for n in r['notes'] if 'retrying' in n)}"
        )