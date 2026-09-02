"""Maya synthesis engine (issue #5): CWA-grounded response generation.

Model proposes, code disposes (ADR 0005): the LLM writes the conversational
response, but its world is *closed* — it may only discuss movies present in
the injected <retrieved_movies> XML block, and the code verifies the output
(``cwa_violations``) instead of trusting the prompt.

Library-first: the LLM is the stock ``langchain_openai.ChatOpenAI`` pointed
at OpenRouter (same pattern as the router, #3) — no custom clients, no
custom JSON handling.
"""

import os
import re
from collections.abc import Sequence

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.domain.config import ExperimentConfig
from src.domain.movie import MovieRecord
from src.domain.routing import QueryRoutingDecision
from src.graph.state import SynthesisUsage
from src.maya.prompts import build_system_prompt
from src.maya.router import OPENROUTER_BASE_URL

TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w500"
_CWA_TITLE_PATTERN = re.compile(r"\*\*(.+?)\s*\(\d{4}\)\*\*")

#: Movie block line whose title is missing bold markers, e.g.
#: ``Avatar (2009) — dir. James Cameron With a staggering revenue…``
#: (issue #20: the model emits the block format inconsistently).
_UNBOLDED_BLOCK = re.compile(r"^([A-Z][^*\n]*?\(\d{4}\))\s+—\s+dir\.\s*(.*)$")
#: Bold movie block line (the contract format) — used to keep blocks apart.
_BOLD_BLOCK = re.compile(r"^\*\*[^*]+\(\d{4}\)\*\*")


def normalize_movie_markdown(text: str) -> str:
    """Normalizes synthesizer output to the contract block format (issue #20).

    The model sometimes drops the ``**`` bold markers and the blank lines
    between movie blocks; markdown then collapses everything into one
    paragraph (looks like a CSS glitch) and unbolded titles escape the CWA
    verifier. This is deterministic repair: model proposes, code disposes.
    """
    lines = text.split("\n")
    fixed: list[str] = []
    for line in lines:
        stripped = line.strip()
        match = _UNBOLDED_BLOCK.match(stripped)
        if match:
            title, rest = match.groups()
            line = f"**{title}** — dir. {rest}".rstrip()
        if (
            fixed
            and fixed[-1].strip()
            and (_BOLD_BLOCK.match(line.strip()) or _UNBOLDED_BLOCK.match(stripped))
        ):
            fixed.append("")  # blocks always start their own paragraph
        fixed.append(line)
    return "\n".join(fixed)


class CwaViolation(BaseModel):
    """One Closed-World-Assumption breach detected in a synthesis response."""

    mentioned_title: str
    reason: str = "Title referenced outside the retrieved-movie context"


class MayaSynthesizer:
    """Generates the user-facing response under the Closed-World Assumption."""

    def __init__(self, config: ExperimentConfig, api_key: str | None = None) -> None:
        self.config = config
        self._llm = ChatOpenAI(
            model=config.synthesis_model,
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            temperature=config.temperature,
        )

    def synthesize(
        self,
        query: str,
        decision: QueryRoutingDecision,
        movies: list[MovieRecord],
        history: Sequence[BaseMessage],
    ) -> tuple[str, SynthesisUsage]:
        """Returns (response_text, usage) for one grounded turn."""
        messages = [
            *history,
            ("system", self._build_system_prompt(
                has_retrieval=bool(movies), is_superlative=decision.is_superlative
            )),
            ("human", self._build_user_message(query, decision, movies)),
        ]
        response = self._llm.invoke(messages)
        usage_meta = response.usage_metadata or {}
        usage = SynthesisUsage(
            model=self.config.synthesis_model,
            prompt_tokens=usage_meta.get("input_tokens", 0),
            completion_tokens=usage_meta.get("output_tokens", 0),
        )
        # Deterministic output normalization (issue #20) BEFORE the CWA check:
        # unbolded titles get bolded and blocks get paragraph separation, so the
        # verifier sees every mentioned title.
        return normalize_movie_markdown(response.text), usage

    def _build_system_prompt(self, has_retrieval: bool, is_superlative: bool = False) -> str:
        """Delegates to the prompt layer (issue #10) — this class owns behavior."""
        return build_system_prompt(has_retrieval=has_retrieval, is_superlative=is_superlative)

    def _build_user_message(
        self,
        query: str,
        decision: QueryRoutingDecision,
        movies: list[MovieRecord],
    ) -> str:
        parts = [f"<user_query>{query}</user_query>"]
        if movies:
            parts.append(
                "<retrieved_movies>\n"
                + "\n".join(self._movie_xml(m) for m in movies)
                + "\n</retrieved_movies>"
            )
        if decision.is_superlative and decision.superlative:
            c = decision.superlative
            parts.append(
                "<ranking_criteria>\n"
                f"metric={c.metric.value}; direction={c.direction}; "
                f"year={c.year if c.year else 'any'}; max_results={c.limit}\n"
                "</ranking_criteria>"
            )
        if decision.filters and (
            decision.filters.excluded_genres or decision.filters.excluded_actors
        ):
            parts.append(
                f"<session_constraints>Respect these exclusions: "
                f"genres={decision.filters.excluded_genres}, "
                f"actors={decision.filters.excluded_actors}</session_constraints>"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _movie_xml(movie: MovieRecord) -> str:
        """One <movie_record> element — the closed world the model may speak of.

        Includes the deterministic ranking facts (revenue, budget, popularity,
        vote_count, issue #19): without them the model cannot cite metric
        values and hedges superlative answers (CWA correctly blocks guessing).
        """
        cast = ", ".join(c.name for c in movie.cast[:5])
        return (
            f'  <movie_record id="{movie.id}">\n'
            f"    <title>{movie.title}</title>\n"
            f"    <year>{movie.release_year}</year>\n"
            f"    <director>{movie.director}</director>\n"
            f"    <genres>{', '.join(movie.genres)}</genres>\n"
            f"    <rating>{movie.vote_average:.1f}</rating>\n"
            f"    <runtime>{movie.runtime} min</runtime>\n"
            f"    <revenue>{movie.revenue}</revenue>\n"
            f"    <budget>{movie.budget}</budget>\n"
            f"    <popularity>{movie.popularity:.1f}</popularity>\n"
            f"    <vote_count>{movie.vote_count}</vote_count>\n"
            f"    <overview>{movie.overview}</overview>\n"
            f"    <poster_path>{movie.poster_path}</poster_path>\n"
            f"    <cast>{cast}</cast>\n"
            f"  </movie_record>"
        )

    def cwa_violations(self, response_text: str, movies: list[MovieRecord]) -> list[CwaViolation]:
        """Code-side CWA verification: bolded movie mentions must be in context.

        Model proposes, code disposes — this is the verification half of the
        grounding contract; #6 logs violations for eval (never silently fixed).
        """
        allowed = {m.title.casefold() for m in movies}
        mentioned = _CWA_TITLE_PATTERN.findall(response_text)
        return [
            CwaViolation(mentioned_title=title)
            for title in mentioned
            if title.casefold() not in allowed
        ]
