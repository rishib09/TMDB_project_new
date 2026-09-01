"""5-Layer Conversational Memory State and LangGraph State Reducers."""

import operator
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.domain.movie import MovieRecord
from src.domain.routing import QueryRoutingDecision


class ChatMessage(BaseModel):
    """Represents a single message turn in the conversation history."""
    role: str  # "user" | "assistant" | "system"
    content: str
    intent: str | None = None
    retrieved_movie_ids: list[int] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class FocusedMovieEntity(BaseModel):
    """Active movie entity currently in conversational focus for coreference resolution."""
    id: int
    title: str
    release_year: int
    director: str = ""


class UserSessionPreferences(BaseModel):
    """Persistent session-level constraints and preferences across turns."""
    excluded_genres: list[str] = Field(default_factory=list)
    excluded_actors: list[str] = Field(default_factory=list)
    preferred_genres: list[str] = Field(default_factory=list)


# --- LangGraph Functional Reducers ---

def merge_unique_ids(left: list[int], right: list[int]) -> list[int]:
    """Reducer that appends newly shown movie IDs while preserving uniqueness."""
    return list(dict.fromkeys(left + right))


def merge_preferences(
    current: UserSessionPreferences,
    incoming: UserSessionPreferences | None
) -> UserSessionPreferences:
    """Reducer that merges session-level preferences and persistent exclusions."""
    if not incoming:
        return current
    return UserSessionPreferences(
        excluded_genres=list(dict.fromkeys(current.excluded_genres + incoming.excluded_genres)),
        excluded_actors=list(dict.fromkeys(current.excluded_actors + incoming.excluded_actors)),
        preferred_genres=list(dict.fromkeys(current.preferred_genres + incoming.preferred_genres)),
    )


class MayaGraphState(BaseModel):
    """LangGraph 5-Layer Agent State with functional reducers.

    Nodes return simple dict updates (e.g. {'messages': [AIMessage(...)], 'shown_movie_ids': [27205]}),
    and LangGraph reducers handle deduplication, merging, and thread persistence automatically.
    """
    # 1. Message History (Sliding window managed via LangGraph add_messages)
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(default_factory=list)

    # 2. Entity Focus Layer
    focused_entity: FocusedMovieEntity | None = None
    focused_person: str | None = None

    # 3. Seen Recommendations Tracker (Unique Reducer)
    shown_movie_ids: Annotated[list[int], merge_unique_ids] = Field(default_factory=list)

    # 4. Persistent User Preferences & Exclusions (Merge Reducer)
    session_preferences: Annotated[UserSessionPreferences, merge_preferences] = Field(
        default_factory=UserSessionPreferences
    )

    # 5. Session Metrics & Summary
    rolling_summary: str = ""
    session_tokens: Annotated[int, operator.add] = 0

    # Transient per-turn pipeline artifacts
    routing_decision: QueryRoutingDecision | None = None
    retrieved_movies: list[MovieRecord] = Field(default_factory=list)


class ConversationState(BaseModel):
    """5-Layer Conversational Memory State for UI session state."""
    messages: list[ChatMessage] = Field(default_factory=list)
    focused_entity: FocusedMovieEntity | None = None
    focused_person: str | None = None
    shown_movie_ids: list[int] = Field(default_factory=list)
    session_preferences: UserSessionPreferences = Field(default_factory=UserSessionPreferences)
    rolling_summary: str = ""
    session_tokens: int = 0

    def add_turn(
        self,
        user_query: str,
        assistant_response: str,
        retrieved_movies: list[MovieRecord],
        decision: QueryRoutingDecision | None = None,
        tokens_used: int = 0,
    ) -> None:
        """Updates conversational state with new turn, entity focus, and shown IDs."""
        user_intent_str = decision.intent.value if decision else None
        retrieved_ids = [m.id for m in retrieved_movies]

        self.messages.append(ChatMessage(
            role="user",
            content=user_query,
            intent=user_intent_str
        ))
        self.messages.append(ChatMessage(
            role="assistant",
            content=assistant_response,
            retrieved_movie_ids=retrieved_ids
        ))

        # Use functional reducers for state updates
        self.shown_movie_ids = merge_unique_ids(self.shown_movie_ids, retrieved_ids)

        if retrieved_movies:
            top_movie = retrieved_movies[0]
            self.focused_entity = FocusedMovieEntity(
                id=top_movie.id,
                title=top_movie.title,
                release_year=top_movie.release_year,
                director=top_movie.director,
            )

        if decision and decision.filters:
            incoming = UserSessionPreferences(
                excluded_genres=decision.filters.excluded_genres,
                excluded_actors=decision.filters.excluded_actors,
            )
            self.session_preferences = merge_preferences(self.session_preferences, incoming)

        self.session_tokens += tokens_used

        if len(self.messages) > 10:
            self.messages = self.messages[-10:]

    def clear(self) -> None:
        """Reset conversation state to a clean slate."""
        self.messages.clear()
        self.focused_entity = None
        self.focused_person = None
        self.shown_movie_ids.clear()
        self.session_preferences = UserSessionPreferences()
        self.rolling_summary = ""
        self.session_tokens = 0
