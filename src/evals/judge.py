"""LLM-as-a-judge evaluator for Maya responses (issue #6).

Lean judge: structured-output verdicts via the same ``ChatOpenAI.with_structured_output``
pattern as the router (#3) — no ragas dependency (HF Spaces boot budget),
but with ragas-compatible metric definitions so scores stay comparable.

Model proposes, code disposes (ADR 0005): the judge model classifies claims
and scores relevancy, but the code re-derives the faithfulness score from
claim counts and clamps relevancy — the judge never self-reports its own
arithmetic.
"""

import os
import re

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.domain.config import ExperimentConfig
from src.domain.movie import MovieRecord
from src.maya.providers import DEFAULT_ZAI_BASE_URL, resolve_chat_endpoint

_FAITHFULNESS_PROMPT = """You are a strict evaluation judge. Given a USER QUERY, an
ASSISTANT RESPONSE about movies, and the RETRIEVED CONTEXT (the only movies the
assistant was allowed to use), decompose the response into atomic factual claims
and classify each claim:

- "entailed": directly supported by the retrieved context
- "unsupported": not found in the context (invented, recalled from memory, or
  contradicting it)

Count only claims about movies, people, dates, or plot facts. Ignore pleasantries,
hedging, and recommendations phrasing. Be strict: if a claim needs context that is
absent, it is unsupported."""

_RELEVANCY_PROMPT = """You are a strict evaluation judge. Rate how well the ASSISTANT
RESPONSE answers the USER QUERY on this scale:

- 1.0: directly answers the request with relevant movie recommendations
- 0.5: partially answers (relevant but incomplete, or generic)
- 0.0: does not answer the query at all (irrelevant, deflection, refusal without
  justification)

Return the score and a one-sentence reason."""


class FaithfulnessVerdict(BaseModel):
    """Judge's claim-level classification; score is code-derived."""

    total_claims: int = Field(ge=0, description="Atomic factual claims found")
    entailed_claims: int = Field(ge=0, description="Claims supported by context")
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str = ""

    @property
    def score(self) -> float:
        """Code-derived: entailed / total (0.0 when no claims were found)."""
        return self.entailed_claims / self.total_claims if self.total_claims else 0.0


class RelevancyVerdict(BaseModel):
    """Judge's relevancy score; clamped by code."""

    score: float = Field(description="0.0 (irrelevant) to 1.0 (fully relevant)")
    reason: str = ""


class MayaJudge:
    """LLM-as-a-judge for faithfulness and relevancy of Maya's responses."""

    def __init__(self, config: ExperimentConfig, api_key: str | None = None) -> None:
        self.config = config
        # the judge goes through the same seam with allow_swap=False: NO model
        # substitution, ever (#97 requirement 4, review P1 — structural now,
        # not conventional). glm ids still route to z.ai for cheaper transport.
        self._endpoint = resolve_chat_endpoint(
            config.judge_model,
            zai_api_key=os.getenv("ZAI_API_KEY") if api_key is None else None,
            openrouter_api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            zai_base_url=os.getenv("ZAI_BASE_URL") or DEFAULT_ZAI_BASE_URL,
            allow_swap=False,
        )
        self._llm = ChatOpenAI(
            model=self._endpoint.wire_model,
            base_url=self._endpoint.base_url,
            api_key=self._endpoint.api_key,
            temperature=0.0,
            # #93: bounded calls — see the router's note; D17 client ownership.
            request_timeout=120,
            max_retries=1,
        )
        self._faithfulness_chain = self._llm.with_structured_output(FaithfulnessVerdict)
        self._relevancy_chain = self._llm.with_structured_output(RelevancyVerdict)

    def judge_faithfulness(
        self, query: str, response: str, context_movies: list[MovieRecord]
    ) -> FaithfulnessVerdict:
        """Classifies response claims against the retrieved context."""
        context = "\n".join(
            f"- {m.title} ({m.release_year}), dir. {m.director}: {m.overview}"
            for m in context_movies
        ) or "(empty — the assistant was given no movies)"
        verdict = self._faithfulness_chain.invoke(
            f"{_FAITHFULNESS_PROMPT}\n\nUSER QUERY: {query}\n\n"
            f"ASSISTANT RESPONSE:\n{response}\n\nRETRIEVED CONTEXT:\n{context}"
        )
        # Code disposes: consistency repairs the model cannot be trusted with.
        verdict.total_claims = max(verdict.total_claims, len(verdict.unsupported_claims))
        verdict.entailed_claims = min(verdict.entailed_claims, verdict.total_claims)
        return verdict

    def judge_relevancy(self, query: str, response: str) -> RelevancyVerdict:
        """Scores how well the response answers the query."""
        verdict = self._relevancy_chain.invoke(
            f"{_RELEVANCY_PROMPT}\n\nUSER QUERY: {query}\n\nASSISTANT RESPONSE:\n{response}"
        )
        verdict.score = max(0.0, min(1.0, verdict.score))
        return verdict


#: Poster/markdown tokens are formatting, not claims — strip before counting.
_MARKDOWN_NOISE = re.compile(r"!\[[^\]]*\]\([^)]*\)|\*\*|\[|\]|#+")


def strip_formatting(text: str) -> str:
    """Removes markdown noise so the judge reads clean prose."""
    return _MARKDOWN_NOISE.sub("", text).strip()
