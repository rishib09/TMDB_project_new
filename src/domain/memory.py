"""5-Layer Conversational Memory and Session State Models."""

from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field

from src.domain.movie import MovieRecord
from src.domain.routing import QueryRoutingDecision


class ChatMessage(BaseModel):
    """Represents a single message turn in the conversation history."""
    role: str  # "user" | "assistant" | "system"
    content: str
    intent: Optional[str] = None
    retrieved_movie_ids: List[int] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FocusedMovieEntity(BaseModel):
    """Active movie entity currently in conversational focus for coreference resolution."""
    id: int
    title: str
    release_year: int
    director: str = ""


class UserSessionPreferences(BaseModel):
    """Persistent session-level constraints and preferences across turns."""
    excluded_genres: List[str] = Field(default_factory=list)
    excluded_actors: List[str] = Field(default_factory=list)
    preferred_genres: List[str] = Field(default_factory=list)


class ConversationState(BaseModel):
    """5-Layer Conversational Memory State for Maya Agent."""
    messages: List[ChatMessage] = Field(default_factory=list)
    focused_entity: Optional[FocusedMovieEntity] = None
    focused_person: Optional[str] = None
    shown_movie_ids: List[int] = Field(default_factory=list)
    session_preferences: UserSessionPreferences = Field(default_factory=UserSessionPreferences)
    rolling_summary: str = ""
    session_tokens: int = 0

    #TODO: Check if add_turn can be Leaned out with Lang chain or LanGraph 
    def add_turn(
        self,
        user_query: str,
        assistant_response: str,
        retrieved_movies: List[MovieRecord],
        decision: Optional[QueryRoutingDecision] = None,
        tokens_used: int = 0,
    ) -> None:
        """Updates conversational state with new turn, entity focus, and shown IDs."""
        # 1. Append user and assistant messages
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

        # 2. Update shown movie IDs (preventing repeat recommendations)
        for m in retrieved_movies:
            if m.id not in self.shown_movie_ids:
                self.shown_movie_ids.append(m.id)

        # 3. Update active focused entity (top recommended movie)
        if retrieved_movies:
            top_movie = retrieved_movies[0]
            self.focused_entity = FocusedMovieEntity(
                id=top_movie.id,
                title=top_movie.title,
                release_year=top_movie.release_year,
                director=top_movie.director,
            )

        # 4. Update session exclusions if negation criteria present
        if decision and decision.filters:
            for genre in decision.filters.excluded_genres:
                if genre not in self.session_preferences.excluded_genres:
                    self.session_preferences.excluded_genres.append(genre)
            for actor in decision.filters.excluded_actors:
                if actor not in self.session_preferences.excluded_actors:
                    self.session_preferences.excluded_actors.append(actor)

        # 5. Track tokens
        self.session_tokens += tokens_used

        # 6. Trim sliding message window to last 10 messages
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
