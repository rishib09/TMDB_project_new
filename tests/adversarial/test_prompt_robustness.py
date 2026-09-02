"""Adversarial tests for the prompt layer (issue #10).

The prompt itself is an attack surface: a static section that names famous
movies poisons the closed world (the model could "recall" them on empty
retrieval — the #21 vector, reintroduced by our own prompt).
"""

import re

import pytest

from src.maya import prompts
from src.maya.prompts import build_system_prompt

pytestmark = pytest.mark.adversarial


def test_no_concrete_movie_titles_in_static_sections():
    """Static prompt must never name films — titles only come from retrieval."""
    title_pattern = re.compile(r"\*\*(.+?)\s*\(\d{4}\)\*\*")  # Maya's own card format
    static_text = "\n".join([
        prompts.MAYA_PERSONA, prompts.MAYA_ARCHITECTURE,
        prompts.CWA_RETRIEVAL_RULE, prompts.CWA_NO_RETRIEVAL_RULE,
        prompts.SUPERLATIVE_RULE, prompts.CONVERSATION_ETHOS, prompts.FORMAT_RULE,
    ])
    matches = title_pattern.findall(static_text)
    assert matches == [], f"static prompt names movies: {matches}"


def test_architecture_section_has_no_tool_imperatives():
    """Descriptive first-person only — no instruction surface for exploits."""
    arch = prompts.MAYA_ARCHITECTURE.lower()
    for imperative in ("run sql", "execute", "invoke", "call the", "you can access"):
        assert imperative not in arch


def test_architecture_section_has_no_xml_tag_examples():
    """No XML fragments the model could echo as if they were CWA context."""
    xml_pattern = re.compile(r"</\s*\w+\s*>|<\s*(?!retrieved_movies\b)\w+\s*>")
    leaked = xml_pattern.findall(prompts.MAYA_ARCHITECTURE)
    assert leaked == [], f"architecture section contains XML tags: {leaked}"


def test_all_compositions_still_guarded():
    """No flag combination leaves the model without CWA hard rules."""
    for retrieval in (True, False):
        prompt = build_system_prompt(has_retrieval=retrieval, is_superlative=True)
        assert "HARD RULES" in prompt
        assert "override VOICE" in prompt
        assert "NEVER" in prompt


def test_cwa_rule_cannot_be_disabled_by_flag_combination():
    """Both CWA variants forbid naming movies outside grounded context."""
    assert "NEVER" in prompts.CWA_RETRIEVAL_RULE
    assert "NEVER" in prompts.CWA_NO_RETRIEVAL_RULE
    assert "name any movie outside the block" in prompts.CWA_RETRIEVAL_RULE
    assert "name specific movies" in prompts.CWA_NO_RETRIEVAL_RULE
