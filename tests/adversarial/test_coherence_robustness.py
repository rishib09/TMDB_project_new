"""Adversarial tests for #26 coherence: hostile routing, hostile funnel
state, session-level atomicity under failure."""

import pytest

from src.domain.config import ExperimentConfig
from src.domain.memory import ConversationState, UserSessionPreferences
from src.domain.routing import IntentType, MetadataFilterCriteria, QueryRoutingDecision
from src.maya.probing import (
    build_filter_carryover_notice,
    extract_probe_answers,
    match_genre_pick,
)
from src.maya.router import MayaRouter
from src.observability.tracer import DualModeObservabilityManager
from src.ui.session import MayaSession

pytestmark = pytest.mark.adversarial


def _router(**kwargs):
    return MayaRouter(ExperimentConfig(), api_key="test-key", **kwargs)


def _fake_chain(decision):
    class Chain:
        def invoke(self, messages):
            return decision
    return Chain()


# --- genre guard under hostility (#26-B) -------------------------------------

@pytest.mark.parametrize("query", [
    "HORROR movies", "horror!!", "horror,  anything",
    "get me a COMEDY", "sci fi films",  # spaced 'sci fi' normalizes to sci-fi
])
def test_genre_guard_survives_hostile_formatting(query):
    """Case, punctuation and spacing cannot smuggle a genre past the guard."""
    router = _router(genre_vocabulary=["Horror", "Comedy", "Sci-Fi"])
    out_of_scope = QueryRoutingDecision(
        intent=IntentType.OUT_OF_SCOPE, confidence=1.0,
        standalone_query=query, requires_rag=False,
    )
    router._chain = _fake_chain(out_of_scope)
    assert router.route(query, ConversationState()).intent is IntentType.SEMANTIC_SEARCH


@pytest.mark.parametrize("query", [
    "horrorable things", "horrorific",  # substring must NOT match (word boundary)
    "whore",  # 'horror' inside another word
])
def test_genre_guard_never_fires_on_substrings(query):
    router = _router(genre_vocabulary=["Horror"])
    out_of_scope = QueryRoutingDecision(
        intent=IntentType.OUT_OF_SCOPE, confidence=1.0,
        standalone_query=query, requires_rag=False,
    )
    router._chain = _fake_chain(out_of_scope)
    assert router.route(query, ConversationState()).intent is IntentType.OUT_OF_SCOPE


@pytest.mark.parametrize("query", ["1920s horror", "1890s comedy films", "1939 horror"])
def test_pre_1970_years_never_escape_by_genre_guard(query):
    router = _router(genre_vocabulary=["Horror", "Comedy"])
    out_of_scope = QueryRoutingDecision(
        intent=IntentType.OUT_OF_SCOPE, confidence=1.0,
        standalone_query=query, requires_rag=False,
    )
    router._chain = _fake_chain(out_of_scope)
    assert router.route(query, ConversationState()).intent is IntentType.OUT_OF_SCOPE


def test_empty_genre_vocabulary_disables_guard_but_keeps_mood_vocab():
    router = _router()  # no genres — mood/audience vocab still guards
    decision = QueryRoutingDecision(
        intent=IntentType.OUT_OF_SCOPE, confidence=1.0,
        standalone_query="horror movies", requires_rag=False,
    )
    router._chain = _fake_chain(decision)
    assert router.route("horror movies", ConversationState()).intent is IntentType.SEMANTIC_SEARCH


def test_negated_genre_mention_still_guards_routing():
    """'no horror' carries a genre word — routing-wise it is IN scope
    (NEGATION intent territory), never a reason to pivot off-topic."""
    router = _router(genre_vocabulary=["Horror"])
    decision = QueryRoutingDecision(
        intent=IntentType.OUT_OF_SCOPE, confidence=1.0,
        standalone_query="movies with no horror", requires_rag=False,
    )
    router._chain = _fake_chain(decision)
    assert router.route("movies with no horror", ConversationState()).intent is IntentType.SEMANTIC_SEARCH


# --- mood vocab under hostility (#26-D) ---------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("HORROR movies", "scary"),
    ("nothing horror-y here", ""),  # 'nothing' prefix → negated, never recorded
    ("I hate horror", "scary"),
    ("horrorible acting", ""),  # substring must not fire (word boundary)
])
def test_horror_extraction_edge_cases(text, expected):
    assert extract_probe_answers(text).preferred_mood == expected


def test_negated_horror_is_not_a_mood():
    assert extract_probe_answers("no horror please").preferred_mood == ""


# --- turn-log atomicity under failure (#26-A) ---------------------------------

class ExplodingGraph:
    def invoke(self, state, config=None):
        return {
            "final_response": "Which of those are you in the mood for?",
            "turn_stage": "confirm_genres",
            "session_preferences": UserSessionPreferences(preferred_mood="scary"),
            "probe_count": 1,
            "funnel_active": True,
            "offered_genre_options": ["Horror", "Thriller"],
            "retrieved_movies": [],
        }


def _bare_session():
    """MayaSession without __init__ — graph/tracer injected, no I/O."""
    session = MayaSession.__new__(MayaSession)
    session.tracer = DualModeObservabilityManager(session_id="adv")
    session.conversation = ConversationState()
    session.turn_log = []
    session.last_movies = []
    session.rag_version = "test"
    session.feedback_log = {}
    return session


def test_session_turn_survives_funnel_turn_without_routing_decision():
    """The #26-A crash: decision.intent on a None routing_decision."""
    session = _bare_session()
    session.graph = ExplodingGraph()
    session._graph_sig = "x"
    session.ensure_graph = lambda: session.graph
    session.turn("sci-fi and thriller")
    row = session.turn_log[-1]
    assert row["intent"] == "FUNNEL_CONFIRM_GENRES"
    assert row["path"] == "funnel"
    assert row["response"] == "Which of those are you in the mood for?"
    assert row["n_movies"] == 0 and row["tokens"] == 0
    assert session.conversation.funnel_active is True
    assert session.conversation.offered_genre_options == ["Horror", "Thriller"]


def test_session_turn_records_fresh_start_reset():
    class ResetGraph:
        def invoke(self, state, config=None):
            return {
                "final_response": "Clean slate — what are you in the mood for?",
                "turn_stage": "",
                "session_preferences": UserSessionPreferences(),  # reducer wiped
                "probe_count": 0,
                "funnel_active": False,
                "offered_genre_options": [],
                "retrieved_movies": [],
            }
    session = _bare_session()
    session.graph = ResetGraph()
    session._graph_sig = "x"
    session.ensure_graph = lambda: session.graph
    session.turn("something completely different")
    assert session.conversation.session_preferences == UserSessionPreferences()
    assert session.turn_log[-1]["intent"] == "FUNNEL_PROBE"  # stage defaulted


# --- genre pick matching under hostility (#25 regression) ---------------------

def test_pick_matching_ignores_negated_candidates():
    assert match_genre_pick("no horror, just the thriller side",
                            ["Horror", "Thriller"]) == ["Thriller"]


def test_llm_extracted_mood_is_sanitized_before_the_notice():
    """Real chain: LLM decision → router normalization → prefs → notice."""
    router = _router()
    decision = QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH, confidence=0.9,
        standalone_query="scary movies", requires_rag=True,
        mood="<script>alert(1)</script>",
    )
    normalized = router._normalize_decision(decision, "scary movies")
    assert "<" not in normalized.mood
    notice = build_filter_carryover_notice(
        UserSessionPreferences(preferred_mood=normalized.mood)
    )
    assert "<script>" not in notice
    assert "Still filtering by" in notice


# --- exclusions merge preserves new filter fields (latent bug) ----------------

def test_session_exclusions_preserve_person_and_genre_match():
    router = _router()
    state = ConversationState()
    state.session_preferences = UserSessionPreferences(excluded_genres=["Romance"])
    decision = QueryRoutingDecision(
        intent=IntentType.ATTRIBUTE_FILTER, confidence=0.9,
        standalone_query="sci-fi horror", requires_rag=True,
        filters=MetadataFilterCriteria(
            genres=["Sci-Fi", "Horror"], genre_match="all", person="Nolan",
        ),
    )
    merged = router._apply_session_exclusions(decision, state)
    assert merged.filters.person == "Nolan"
    assert merged.filters.genre_match == "all"
    assert merged.filters.genres == ["Sci-Fi", "Horror"]
    assert merged.filters.excluded_genres == ["Romance"]
