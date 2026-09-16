"""Unit tests for the multi-turn golden source: loader, renderer, records (#86)."""

import json
from pathlib import Path

from src.evals.conversations import (
    ConversationSet,
    ConversationTurn,
    ExpectedConstraints,
    GoldenConversation,
    TurnExpectation,
    load_conversations,
    render_markdown,
)

ROW_OF_RECORD = {
    "version": "2026-09-16",
    "description": "test set",
    "conversations": [
        {
            "id": "C03",
            "tier": "C_memory",
            "title": "no horror, said once",
            "source": "authored",
            "turns": [
                {
                    "n": 1,
                    "user": "something for the kids, no horror please",
                    "expect": {
                        "intent": "SEMANTIC_SEARCH",
                        "path": "ask",
                        "constraints": {"audience": "kids", "excluded_genres": ["Horror"]},
                    },
                },
                {
                    "n": 2,
                    "user": "funny",
                    "expect": {
                        "intent": "SEMANTIC_SEARCH",
                        "path": "retrieve",
                        "constraints": {
                            "audience": "kids",
                            "mood": "funny",
                            "excluded_genres": ["Horror"],
                        },
                    },
                },
                {
                    "n": 3,
                    "user": "yes",
                    "expect": {"intent": "SEMANTIC_SEARCH", "path": "retrieve",
                               "constraints": {"audience": "kids", "mood": "funny",
                                               "excluded_genres": ["Horror"]},
                               "no_repeat": True},
                },
                {
                    "n": 4,
                    "user": "ok something scarier then",
                    "expect": {
                        "intent": "SEMANTIC_SEARCH",
                        "path": "retrieve",
                        "constraints": {
                            "audience": "kids",
                            "mood": "scary",
                            "excluded_genres": ["Horror"],
                        },
                        "no_repeat": True,
                        "notes": "scary mood, Horror still excluded",
                    },
                },
            ],
        }
    ],
}


def test_loader_round_trips_the_row_of_record(tmp_path: Path):
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps(ROW_OF_RECORD), encoding="utf-8")
    loaded = load_conversations(path)
    assert loaded.version == "2026-09-16"
    convo = loaded.conversations[0]
    assert convo.id == "C03"
    assert [t.n for t in convo.turns] == [1, 2, 3, 4]
    assert convo.turns[3].expect.constraints.excluded_genres == ["Horror"]
    assert convo.turns[3].expect.no_repeat is True
    # defaults fill what the row omits
    assert convo.turns[0].expect.no_repeat is False
    assert convo.turns[0].expect.relevant_movie_ids == []


def test_renderer_lists_every_turn_with_its_expectation():
    conversations = ConversationSet.model_validate(ROW_OF_RECORD)
    text = render_markdown(conversations)
    assert "### C03" in text and "no horror, said once" in text
    for turn in conversations.conversations[0].turns:
        assert turn.user in text
    assert "retrieve" in text and "excluded_genres=Horror" in text
    assert "no_repeat" in text  # the flag is visible to the reviewer


def test_constraints_render_compactly_and_skip_empty_fields():
    expectation = TurnExpectation(
        intent="SEMANTIC_SEARCH",
        path="retrieve",
        constraints=ExpectedConstraints(mood="funny", year_min=2015),
    )
    assert expectation.constraints.summary() == "mood=funny, year_min=2015"
    assert ExpectedConstraints().summary() == "none"


def test_turn_numbers_consecutive_is_accepted_for_ten_turns():
    turns = [
        ConversationTurn(
            n=i, user=f"u{i}",
            expect=TurnExpectation(intent="SEMANTIC_SEARCH", path="retrieve"),
        )
        for i in range(1, 11)
    ]
    convo = GoldenConversation(id="C10", tier="C_reference", title="t", source="authored", turns=turns)
    assert len(convo.turns) == 10


def test_golden_file_loads_with_the_reviewed_counts():
    """The promoted golden source (#86) is valid and complete: 23 conversations, six tiers."""
    conversations = load_conversations()
    assert len(conversations.conversations) == 23
    assert sum(len(c.turns) for c in conversations.conversations) == 200
    assert {c.tier for c in conversations.conversations} == {
        "C_records", "C_memory", "C_narrowing", "C_refinement", "C_reference", "C_plot"
    }
    assert sum(1 for c in conversations.conversations if len(c.turns) == 10) == 8
