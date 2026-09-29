"""Adversarial tests for the #114 voice contract (ADR 0011 enforcement).

Offline, canned replies, no LLM: the checker must catch a ranked reply
that never states its ranking basis, and the CWA verifier must keep
catching excluded/non-retrieved titles presented as results (the
"Epic Movie was disqualified" incident). Prompt-layer assertions pin the
contract text so a regression fails here before any live turn.
"""

import pytest

from src.domain.movie import MovieRecord
from src.domain.routing import (
    IntentType,
    MetadataFilterCriteria,
    QueryRoutingDecision,
    SuperlativeCriteria,
    SuperlativeMetric,
)
from src.maya.agent import MayaSynthesizer
from src.maya.prompts import build_system_prompt

pytestmark = pytest.mark.adversarial


def _decision(**overrides) -> QueryRoutingDecision:
    defaults = dict(
        intent=IntentType.SUPERLATIVE_RANKING,
        confidence=0.9,
        standalone_query="highest grossing movie",
        requires_rag=True,
        is_superlative=True,
        superlative=SuperlativeCriteria(metric=SuperlativeMetric.POPULARITY),
        filters=MetadataFilterCriteria(),
    )
    defaults.update(overrides)
    return QueryRoutingDecision(**defaults)


def _movie(title: str = "Interstellar") -> MovieRecord:
    return MovieRecord(
        id=1, title=title, release_year=2014, genres=["Sci-Fi"],
        director="Christopher Nolan", cast=[], runtime=169, budget=165_000_000,
        revenue=677_471_339, vote_average=8.1, vote_count=30_000,
        overview="space", keywords=[], poster_path="/x.jpg",
    )


def test_checker_exists():
    """The #114 checker ships at all (fails by absence on pre-change code)."""
    assert hasattr(MayaSynthesizer, "voice_contract_violations"), (
        "voice_contract_violations missing — ADR 0011 is not enforced"
    )


def test_ranked_reply_without_basis_clause_is_flagged(synthesizer_factory=None):
    """A ranked reply that never states the basis violates the contract."""
    synth = MayaSynthesizer.__new__(MayaSynthesizer)  # checker is pure; skip __init__
    reply = (
        "The crown goes to **Interstellar (2014)** — a cosmic gut-punch "
        "that no other film comes close to."
    )
    violations = synth.voice_contract_violations(
        reply, [_movie()], ranked=True, basis="popularity, highest first"
    )
    assert any("basis" in v for v in violations), violations


def test_ranked_reply_with_basis_clause_is_clean():
    synth = MayaSynthesizer.__new__(MayaSynthesizer)
    reply = (
        "Ranked by popularity, highest first: **Interstellar (2014)** "
        "takes the crown."
    )
    assert synth.voice_contract_violations(
        reply, [_movie()], ranked=True, basis="popularity, highest first"
    ) == []


def test_excluded_title_presented_as_result_is_flagged():
    """The 'Epic Movie disqualified' class: a non-retrieved title named as
    a result stays flagged (trace-only on retrieval turns, per D1)."""
    synth = MayaSynthesizer.__new__(MayaSynthesizer)
    retrieved = [_movie("Date Night")]
    reply = "**Date Night (2010)** delivered — **Epic Movie (2007)** was disqualified."
    violations = synth.cwa_violations(reply, retrieved)
    assert any(v.mentioned_title == "Epic Movie" for v in violations)


def test_superlative_prompt_demands_basis_disclosure():
    """The prompt layer must instruct the disclosure (fails pre-change)."""
    prompt = build_system_prompt(has_retrieval=True, is_superlative=True)
    assert "<ranking_basis>" in prompt, "superlative rule does not reference the basis block"
    assert "ranking basis" in prompt.lower()


def test_retrieval_rule_forbids_excluded_title_as_result():
    """ADR 0011's grounding contract is quoted in the retrieval hard rules."""
    from src.maya import prompts
    assert "excluded" in prompts.CWA_RETRIEVAL_RULE.lower()
    assert "invented variety" in prompts.CWA_RETRIEVAL_RULE.lower()


# --- ranking_basis_for derivation ------------------------------------------

def test_v1_basis_names_metric_and_direction():
    from src.maya.agent import ranking_basis_for
    basis = ranking_basis_for(_decision())
    assert basis == "popularity, highest first"


def test_v2_projected_ranked_turn_discloses_honest_basis():
    """v2 Understandings project intent without superlative criteria — the
    honest disclosure is 'relevance-ranked retrieval', never a fake metric."""
    from src.maya.agent import ranking_basis_for
    basis = ranking_basis_for(
        _decision(is_superlative=False, superlative=None)
    )
    assert basis == "relevance-ranked retrieval"


def test_plain_turn_has_no_basis():
    from src.maya.agent import ranking_basis_for
    assert ranking_basis_for(
        _decision(intent=IntentType.SEMANTIC_SEARCH, is_superlative=False,
                  superlative=None)
    ) is None
