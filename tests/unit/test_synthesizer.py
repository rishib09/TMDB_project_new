"""Unit tests for the MayaSynthesizer (issue #5): CWA grounding contract.

The LLM is mocked (FakeMessage/fake invoke) — these tests verify the
code-side contract: XML context construction, prompt content, usage
accounting, and the ``cwa_violations`` verifier (model proposes, code
disposes — ADR 0005).
"""

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from src.domain.config import ExperimentConfig
from src.domain.movie import CastMember, MovieRecord
from src.domain.routing import (
    IntentType,
    QueryRoutingDecision,
    SuperlativeCriteria,
)
from src.maya.agent import MayaSynthesizer

pytestmark = pytest.mark.unit


class FakeLLM:
    """Stands in for ChatOpenAI; returns a canned AIMessage with usage."""

    def __init__(self, text="Sure — here are my picks."):
        self.text_response = text
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = list(messages)
        return AIMessage(
            content=self.text_response,
            usage_metadata={"input_tokens": 120, "output_tokens": 45, "total_tokens": 165},
        )


def _decision(query="spacetime epic"):
    return QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH,
        confidence=0.9,
        standalone_query=query,
        requires_rag=True,
    )


def _inception():
    return MovieRecord(
        id=27205,
        title="Inception",
        release_year=2010,
        director="Christopher Nolan",
        genres=["Sci-Fi", "Thriller"],
        vote_average=8.4,
        runtime=148,
        overview="A thief enters dreams to steal secrets.",
        poster_path="/poster.jpg",
        cast=[CastMember(name="Leonardo DiCaprio", character="Cobb")],
    )


@pytest.fixture
def synthesizer(monkeypatch):
    fake = FakeLLM()
    synth = MayaSynthesizer.__new__(MayaSynthesizer)
    synth.config = ExperimentConfig()
    synth._llm = fake
    return synth


def test_movie_xml_contains_cwa_fields(synthesizer):
    xml = synthesizer._movie_xml(_inception())
    assert '<movie_record id="27205">' in xml
    assert "<title>Inception</title>" in xml
    assert "<year>2010</year>" in xml
    assert "<director>Christopher Nolan</director>" in xml
    assert "<poster_path>/poster.jpg</poster_path>" in xml
    assert "Leonardo DiCaprio" in xml


def test_user_message_wraps_retrieved_movies_block(synthesizer):
    message = synthesizer._build_user_message("q", _decision(), [_inception()])
    assert "<retrieved_movies>" in message and "</retrieved_movies>" in message
    assert "<user_query>q</user_query>" in message


def test_user_message_empty_retrieval_has_no_block(synthesizer):
    message = synthesizer._build_user_message("q", _decision(), [])
    assert "<retrieved_movies>" not in message


def test_system_prompt_enforces_cwa_with_retrieval(synthesizer):
    prompt = synthesizer._build_system_prompt(has_retrieval=True)
    assert "CLOSED-WORLD ASSUMPTION" in prompt
    assert "<retrieved_movies>" in prompt
    assert "NEVER invent" in prompt
    # Poster images live ONLY in the UI grid (issue #18) — never inline markdown
    assert "![Poster]" not in prompt
    assert "Do NOT insert images" in prompt


def test_system_prompt_superlative_answers_directly(synthesizer):
    prompt = synthesizer._build_system_prompt(has_retrieval=True, is_superlative=True)
    assert "SUPERLATIVE" in prompt
    assert "<ranking_criteria>" in prompt
    assert "Never hedge" in prompt


def test_user_message_includes_ranking_criteria_for_superlative(synthesizer):
    decision = QueryRoutingDecision(
        intent=IntentType.SUPERLATIVE_RANKING,
        confidence=1.0,
        standalone_query="best movie of 2026",
        requires_rag=True,
        is_superlative=True,
        superlative=SuperlativeCriteria(metric="RATING", direction="DESC", year=2026, limit=5),
    )
    message = synthesizer._build_user_message("best movie of 2026", decision, [])
    assert "<ranking_criteria>" in message
    assert "metric=RATING; direction=DESC; year=2026; max_results=5" in message


def test_system_prompt_no_retrieval_forbids_recommendations(synthesizer):
    prompt = synthesizer._build_system_prompt(has_retrieval=False)
    assert "no-retrieval turn" in prompt
    assert "NEVER recommend or name specific movies" in prompt


def test_synthesize_returns_text_and_usage(synthesizer):
    text, usage = synthesizer.synthesize(
        "q", _decision(), [_inception()], history=[]
    )
    assert text == "Sure — here are my picks."
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 45
    assert usage.model == synthesizer.config.synthesis_model


def test_synthesize_includes_history_and_system_order(synthesizer):
    prior = AIMessage(content="earlier")
    synthesizer.synthesize("q", _decision(), [_inception()], history=[prior])
    roles = [
        m.type if isinstance(m, BaseMessage) else m[0]
        for m in synthesizer._llm.last_messages
    ]
    assert roles == ["ai", "system", "human"]  # history first, then grounding


# --- the verification half of the CWA contract ---

def test_cwa_violations_flags_foreign_titles(synthesizer):
    response = "**Inception (2010)** — dir. Nolan\n**Interstellar (2014)** — dir. Nolan"
    violations = synthesizer.cwa_violations(response, [_inception()])
    assert [v.mentioned_title for v in violations] == ["Interstellar"]


def test_cwa_violations_clean_when_all_titles_in_context(synthesizer):
    response = "**Inception (2010)** — dir. Nolan"
    assert synthesizer.cwa_violations(response, [_inception()]) == []


def test_cwa_violations_case_insensitive(synthesizer):
    response = "**INCEPTION (2010)**"
    assert synthesizer.cwa_violations(response, [_inception()]) == []


def test_cwa_violations_ignores_non_title_bold_text(synthesizer):
    response = "Here's why **this fits perfectly**: mind-bending structure."
    assert synthesizer.cwa_violations(response, [_inception()]) == []


def test_movie_xml_exposes_ranking_facts(synthesizer):
    """Issue #19: the closed world must include the metric values (revenue etc.),
    otherwise the model cannot cite them and hedges superlative answers."""
    xml = synthesizer._movie_xml(_inception())
    for tag in ("<revenue>", "<budget>", "<popularity>", "<vote_count>"):
        assert tag in xml


def test_system_prompt_forbids_hedging_when_values_present(synthesizer):
    prompt = synthesizer._build_system_prompt(has_retrieval=True, is_superlative=True)
    assert "verbatim" in prompt
    assert "never hedge" in prompt
    assert "Conversationally frame every answer" in prompt
