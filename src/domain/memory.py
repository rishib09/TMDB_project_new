"""Pure conversational memory models (5-Layer Conversational Memory State).

ADR 0006: this module is pure Pydantic — zero framework imports. The
LangGraph-shaped state view (``MayaGraphState``) and its reducer wiring live
in ``src/graph/state.py``; the pure merge functions below are domain logic
shared by both ``ConversationState`` (direct calls) and the graph reducers.
"""

from datetime import UTC, datetime

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
    """Persistent session-level constraints and preferences across turns.

    Populated two ways (issue #22): the router's MetadataFilterCriteria
    extraction (genres/directors/exclusions) and deterministic keyword
    extraction of probe answers (mood/audience/don'ts).
    """

    excluded_genres: list[str] = Field(default_factory=list)
    excluded_actors: list[str] = Field(default_factory=list)
    preferred_genres: list[str] = Field(default_factory=list)
    # Probing axes (#22) — scalar answers are last-wins, lists accumulate.
    preferred_mood: str = ""
    audience: str = ""
    preferred_directors: list[str] = Field(default_factory=list)
    noted_donts: list[str] = Field(default_factory=list)
    #: #25: mood→genre confirmation settled (never re-asked for this mood).
    genre_confirmation_done: bool = False
    #: #26-E: one-shot signal to WIPE accumulated preferences. Must ride the
    #: reducer (an empty update would otherwise be a no-op merge).
    reset_requested: bool = False

    def answered_axes(self) -> list[str]:
        """Ordered narrowing axes with a value — drives the probe funnel."""
        axes = []
        if self.preferred_mood:
            axes.append("mood")
        if self.audience:
            axes.append("audience")
        if self.noted_donts:
            axes.append("donts")
        if self.preferred_genres:
            axes.append("genres")
        if self.preferred_directors:
            axes.append("directors")
        return axes


# --- Pure Merge Logic (domain semantics; graph reducers reuse these) ---

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
    if incoming.reset_requested:  # #26-E: clean slate beats any merge
        return UserSessionPreferences()
    return UserSessionPreferences(
        excluded_genres=list(dict.fromkeys(current.excluded_genres + incoming.excluded_genres)),
        excluded_actors=list(dict.fromkeys(current.excluded_actors + incoming.excluded_actors)),
        preferred_genres=list(dict.fromkeys(current.preferred_genres + incoming.preferred_genres)),
        preferred_mood=incoming.preferred_mood or current.preferred_mood,
        audience=incoming.audience or current.audience,
        preferred_directors=list(
            dict.fromkeys(current.preferred_directors + incoming.preferred_directors)
        ),
        noted_donts=list(dict.fromkeys(current.noted_donts + incoming.noted_donts)),
        # Mood change reopens genre confirmation (#25): the new mood may map
        # to different candidate genres. Same mood → confirmation stays settled.
        genre_confirmation_done=(
            incoming.genre_confirmation_done
            if incoming.preferred_mood and incoming.preferred_mood != current.preferred_mood
            else incoming.genre_confirmation_done or current.genre_confirmation_done
        ),
    )


class ConversationState(BaseModel):
    """5-Layer Conversational Memory State for UI session state."""
    messages: list[ChatMessage] = Field(default_factory=list)
    focused_entity: FocusedMovieEntity | None = None
    focused_person: str | None = None
    shown_movie_ids: list[int] = Field(default_factory=list)
    session_preferences: UserSessionPreferences = Field(default_factory=UserSessionPreferences)
    rolling_summary: str = ""
    session_tokens: int = 0
    probe_count: int = 0  # guided narrowing (#22): persists across turns, caps probing
    funnel_active: bool = False  # #23: next message belongs to the funnel
    offered_genre_options: list[str] = Field(default_factory=list)  # #25 pending genre picks

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
