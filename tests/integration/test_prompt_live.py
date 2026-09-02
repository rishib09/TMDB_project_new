"""Live tests for the Maya prompt layer (issue #10) — real LLM synthesis."""

import pytest

from src.domain.config import ExperimentConfig
from src.domain.routing import IntentType, QueryRoutingDecision
from src.maya.agent import MayaSynthesizer

pytestmark = pytest.mark.live


@pytest.fixture
def synthesizer():
    return MayaSynthesizer(ExperimentConfig())


def _decision(intent=IntentType.SEMANTIC_SEARCH, query="best 1970s sci-fi",
              is_superlative=True):
    return QueryRoutingDecision(
        intent=intent, confidence=0.95, standalone_query=query,
        requires_rag=True, is_superlative=is_superlative,
    )


@pytest.fixture
def sci_movies():
    from src.domain.movie import CastMember, MovieRecord

    return [
        MovieRecord(id=348, title="Alien", release_year=1979, genres=["Sci-Fi", "Horror"],
                    director="Ridley Scott", vote_average=8.5, revenue=104_931_801,
                    popularity=100.0, vote_count=10000, runtime=117,
                    budget=11_000_000, overview="A crew answers a distress signal.",
                    poster_path="/alien.jpg", cast=[CastMember(name="Sigourney Weaver", character="Ripley")]),
        MovieRecord(id=667, title="Star Trek: The Motion Picture", release_year=1979,
                    genres=["Sci-Fi"], director="Robert Wise", vote_average=6.4,
                    revenue=139_000_000, popularity=30.0, vote_count=1200, runtime=132,
                    budget=35_000_000, overview="A mysterious cloud heads for Earth.",
                    poster_path="/sttmp.jpg", cast=[CastMember(name="William Shatner", character="Kirk")]),
    ]


def test_superlative_leads_with_winner_and_metric(synthesizer, sci_movies):
    """#10 FORMAT + RANKING contracts hold against a real model."""
    response, _ = synthesizer.synthesize(
        "best 1970s sci-fi", _decision(), sci_movies, []
    )
    assert "**Alien (1979)**" in response  # highest-rated of the two
    assert "8.5" in response  # metric value verbatim from the record
    assert synthesizer.cwa_violations(response, sci_movies) == []


def test_architecture_question_answered_in_character(synthesizer):
    """'How do you work?' → first-person pipeline explanation, zero titles."""
    decision = QueryRoutingDecision(
        intent=IntentType.CAPABILITIES, confidence=0.9,
        standalone_query="how do you work", requires_rag=False,
    )
    response, _ = synthesizer.synthesize("how do you work", decision, [], [])
    assert synthesizer.cwa_violations(response, []) == []  # no invented titles
    lowered = response.lower()
    assert any(k in lowered for k in ("retriev", "record", "archiv", "search")), (
        f"expected pipeline language, got: {response[:200]}"
    )


def test_no_retrieval_turn_names_no_movies(synthesizer):
    """Non-RAG turn: the prompt's no-retrieval rule holds against a real model."""
    decision = QueryRoutingDecision(
        intent=IntentType.GREETING, confidence=0.9,
        standalone_query="hey there", requires_rag=False,
    )
    response, _ = synthesizer.synthesize("hey there", decision, [], [])
    assert synthesizer.cwa_violations(response, []) == []
