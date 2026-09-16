"""Adversarial tests for the multi-turn golden source schema (#86, map #81 G1–G9).

A malformed golden row must be rejected at load time, never at scoring time:
a path outside the neutral vocabulary, a constraint the engine cannot honor,
a turn numbering gap, or a conversation outside the 4–10 turn band.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.evals.conversations import (
    ConversationSet,
    ConversationTurn,
    ExpectedConstraints,
    GoldenConversation,
    TurnExpectation,
    load_conversations,
)


def _turn(n: int, path: str = "retrieve", **constraints) -> ConversationTurn:
    return ConversationTurn(
        n=n,
        user=f"turn {n}",
        expect=TurnExpectation(
            intent="SEMANTIC_SEARCH",
            path=path,
            constraints=ExpectedConstraints(**constraints),
        ),
    )


def _conversation(turns: list[ConversationTurn], cid: str = "C01") -> GoldenConversation:
    return GoldenConversation(
        id=cid, tier="C_memory", title="t", source="authored", turns=turns
    )


def test_path_outside_neutral_vocabulary_is_rejected():
    with pytest.raises(ValidationError):
        _turn(1, path="probe")  # v1 word; goldens speak the neutral five only


def test_constraint_key_outside_engine_vocabulary_is_rejected():
    with pytest.raises(ValidationError):
        ExpectedConstraints(director_mood="tense")


def test_turn_numbers_must_be_consecutive_from_one():
    with pytest.raises(ValidationError):
        _conversation([_turn(1), _turn(2), _turn(4), _turn(5)])
    with pytest.raises(ValidationError):
        _conversation([_turn(2), _turn(3), _turn(4), _turn(5)])


@pytest.mark.parametrize("count", [3, 11])
def test_conversation_length_outside_four_to_ten_is_rejected(count):
    with pytest.raises(ValidationError):
        _conversation([_turn(i) for i in range(1, count + 1)])


def test_duplicate_conversation_ids_are_rejected():
    convo = _conversation([_turn(i) for i in range(1, 5)])
    with pytest.raises(ValidationError):
        ConversationSet(version="v", description="d", conversations=[convo, convo])


def test_loader_reports_the_offending_row(tmp_path: Path):
    bad = {
        "version": "2026-09-16",
        "description": "d",
        "conversations": [
            {
                "id": "C99",
                "tier": "C_memory",
                "title": "t",
                "source": "authored",
                "turns": [
                    {"n": 1, "user": "u", "expect": {"intent": "SEMANTIC_SEARCH", "path": "teleport"}}
                ],
            }
        ],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="C99"):
        load_conversations(path)
