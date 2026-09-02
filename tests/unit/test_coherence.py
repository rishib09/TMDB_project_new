"""Unit tests for #26 coherence fixes: atomic turn rows, genre guard,
fresh-start reset, carry-over notice, inline metadata."""

import pytest
from langchain_core.messages import HumanMessage

from src.domain.config import ExperimentConfig
from src.domain.memory import (
    ConversationState,
    UserSessionPreferences,
    merge_preferences,
)
from src.domain.routing import IntentType, MetadataFilterCriteria, QueryRoutingDecision
from src.graph.orchestrator import build_maya_graph
from src.maya.guardrails import SessionTokenLimiter
from src.maya.probing import (
    build_filter_carryover_notice,
    extract_probe_answers,
    is_fresh_start,
    preference_chips,
)
from src.maya.router import MayaRouter
from src.observability.tracer import DualModeObservabilityManager
from src.ui.chat_tab import intent_badge_text
from src.ui.session import MayaSession
from tests.unit.test_orchestrator import (
    FakeEngine,
    FakeRouter,
    FakeSynthesizer,
    _decision,
    _movie,
)

pytestmark = pytest.mark.unit


# --- A: atomic turn rows (#26-A) -------------------------------------------

def _row(out, **kwargs):
    defaults = dict(
        query="scary movies", trace_id="t1", rag_version="v1_1_enriched",
        new_traces=[], prev_tokens=0,
    )
    defaults.update(kwargs)
    return MayaSession._build_turn_row(out, **defaults)


def test_funnel_probe_turn_row_is_complete_without_router():
    """routing_decision None must never crash or blank the row."""
    row = _row({
        "final_response": "What mood are we in?",
        "turn_stage": "probe",
        "session_preferences": UserSessionPreferences(preferred_mood="scary"),
        "retrieved_movies": [],
    })
    assert row["intent"] == "FUNNEL_PROBE"
    assert row["confidence"] == 1.0
    assert row["path"] == "funnel"
    assert row["response"] == "What mood are we in?"
    assert row["narrowing"] == ["mood: scary"]


def test_funnel_confirm_genres_row_labels_the_stage():
    row = _row({
        "final_response": "Which of those are you in the mood for?",
        "turn_stage": "confirm_genres",
        "session_preferences": UserSessionPreferences(),
        "retrieved_movies": [],
    })
    assert row["intent"] == "FUNNEL_CONFIRM_GENRES"
    assert row["path"] == "funnel"


def test_funnel_retrieve_turn_is_path_funnel_with_synthetic_decision():
    decision = _decision()
    row = _row({
        "final_response": "here you go",
        "turn_stage": "retrieve",
        "routing_decision": decision,
        "retrieved_movies": [_movie()],
        "session_preferences": UserSessionPreferences(preferred_mood="scary"),
    })
    assert row["path"] == "funnel"  # no route trace ran — synthetic decision
    assert row["intent"] == "SEMANTIC_SEARCH"
    assert row["n_movies"] == 1


def test_row_includes_active_filter_chips():
    decision = _decision()
    decision = decision.model_copy(update={"filters": MetadataFilterCriteria(
        genres=["Horror", "Thriller"], genre_match="all", year_min=2000,
    )})
    row = _row({
        "final_response": "r", "routing_decision": decision,
        "retrieved_movies": [], "session_preferences": UserSessionPreferences(),
    })
    assert "genres: Horror, Thriller (all)" in row["filters"]
    assert "2000–…" in row["filters"]


# --- B: genre guard (#26-B) --------------------------------------------------

def _router(**kwargs):
    return MayaRouter(ExperimentConfig(), api_key="test-key", **kwargs)


def _fake_chain(decision):
    class Chain:
        def invoke(self, messages):
            return decision
    return Chain()


def test_genre_guard_overrides_out_of_scope_misclassification():
    router = _router(genre_vocabulary=["Horror", "Sci-Fi", "Comedy"])
    out_of_scope = QueryRoutingDecision(
        intent=IntentType.OUT_OF_SCOPE, confidence=1.0,
        standalone_query="suggest me horror movies", requires_rag=False,
    )
    router._chain = _fake_chain(out_of_scope)
    decision = router.route("suggest me horror movies", ConversationState())
    assert decision.intent is IntentType.SEMANTIC_SEARCH
    assert decision.requires_rag is True


def test_mood_vocabulary_alone_beats_out_of_scope():
    """#26-C: 'scary movies for kids' is a refinement, never off-topic."""
    router = _router()  # EMPTY genre vocabulary — mood vocab must suffice
    out_of_scope = QueryRoutingDecision(
        intent=IntentType.OUT_OF_SCOPE, confidence=1.0,
        standalone_query="scary movies for kids", requires_rag=False,
    )
    router._chain = _fake_chain(out_of_scope)
    decision = router.route("scary movies for kids", ConversationState())
    assert decision.intent is IntentType.SEMANTIC_SEARCH


def test_pre_1970_text_beats_genre_guard():
    router = _router(genre_vocabulary=["Horror"])
    out_of_scope = QueryRoutingDecision(
        intent=IntentType.OUT_OF_SCOPE, confidence=1.0,
        standalone_query="1950s horror movies", requires_rag=False,
    )
    router._chain = _fake_chain(out_of_scope)
    decision = router.route("1950s horror movies", ConversationState())
    assert decision.intent is IntentType.OUT_OF_SCOPE


def test_genuinely_off_topic_still_pivots():
    router = _router(genre_vocabulary=["Horror"])
    out_of_scope = QueryRoutingDecision(
        intent=IntentType.OUT_OF_SCOPE, confidence=1.0,
        standalone_query="why is the sky blue", requires_rag=False,
    )
    router._chain = _fake_chain(out_of_scope)
    decision = router.route("why is the sky blue", ConversationState())
    assert decision.intent is IntentType.OUT_OF_SCOPE


def test_genre_word_in_standalone_out_of_scope_pivots():
    """Only the QUERY is guarded — a legit off-topic classification of
    unrelated text stays put even when history discussed genres."""
    router = _router(genre_vocabulary=["Horror"])
    out_of_scope = QueryRoutingDecision(
        intent=IntentType.OUT_OF_SCOPE, confidence=1.0,
        standalone_query="what is the capital of France", requires_rag=False,
    )
    router._chain = _fake_chain(out_of_scope)
    decision = router.route("what is the capital of France", ConversationState())
    assert decision.intent is IntentType.OUT_OF_SCOPE


# --- D: mood vocab fallback + genre synonyms (#26-D) -------------------------

def test_horror_maps_to_scary_mood():
    assert extract_probe_answers("horror movies").preferred_mood == "scary"


def test_vocab_fallback_fills_missing_llm_extraction():
    router = _router()
    decision = QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH, confidence=0.9,
        standalone_query="horror movies", requires_rag=True, mood="",
    )
    normalized = router._normalize_decision(decision, "horror movies")
    assert normalized.mood == "scary"


# --- E: fresh-start reset + carry-over notice (#26-E) ------------------------

def test_reset_requested_wipes_preferences_through_the_reducer():
    populated = UserSessionPreferences(
        preferred_mood="scary", audience="kids", preferred_genres=["Horror"],
    )
    merged = merge_preferences(populated, UserSessionPreferences(reset_requested=True))
    assert merged == UserSessionPreferences()


@pytest.mark.parametrize("phrase", [
    "something completely different", "let's start fresh",
    "I want to start over", "watch something else",
])
def test_is_fresh_start_vocabulary(phrase):
    assert is_fresh_start(phrase)


def test_is_fresh_start_ignores_normal_queries():
    assert not is_fresh_start("scary movies for kids")


def test_carryover_notice_lists_preferences_and_escape_hatch():
    notice = build_filter_carryover_notice(UserSessionPreferences(
        preferred_mood="scary", audience="kids", preferred_genres=["Horror"],
    ))
    assert "mood: scary" in notice
    assert "audience: kids" in notice
    assert "genres: Horror" in notice
    assert "something completely different" in notice


def test_carryover_notice_empty_without_preferences():
    assert build_filter_carryover_notice(UserSessionPreferences()) == ""


def test_guard_resets_preferences_even_mid_funnel():
    """The escape hatch works at the ONE choke point — funnel armed or not."""
    config = ExperimentConfig()
    graph = build_maya_graph(
        config,
        FakeRouter([_decision(intent=IntentType.GREETING, requires_rag=False,
                              query="something completely different")]),
        FakeEngine(),
        FakeSynthesizer(),
        DualModeObservabilityManager(session_id="t"),
        limiter=SessionTokenLimiter(),
    )
    out = graph.invoke({
        "messages": [HumanMessage(content="something completely different")],
        "session_preferences": UserSessionPreferences(
            preferred_mood="scary", audience="kids",
        ),
        "funnel_active": True,
    })
    assert out["session_preferences"] == UserSessionPreferences()
    assert out["funnel_active"] is False
    # funnel disarmed + clean prefs → the turn routed normally (synthesize)


def test_first_funnel_recommendation_announces_carried_filters():
    config = ExperimentConfig()
    graph = build_maya_graph(
        config,
        FakeRouter([]),  # never consulted — funnel owns the turn
        FakeEngine([_movie()]),
        FakeSynthesizer(response="Here are five films."),
        DualModeObservabilityManager(session_id="t"),
        limiter=SessionTokenLimiter(),
    )
    out = graph.invoke({
        "messages": [HumanMessage(content="go ahead")],
        "session_preferences": UserSessionPreferences(
            preferred_mood="scary", audience="kids",
        ),
        "funnel_active": True,
    })
    assert out["retrieved_movies"]
    assert "Still filtering by" in out["final_response"]
    assert "mood: scary" in out["final_response"]
    assert "audience: kids" in out["final_response"]


# --- F: inline metadata line (#26-F) -----------------------------------------

def test_intent_badge_carries_narrowing_and_filters_inline():
    row = {
        "intent": "ATTRIBUTE_FILTER", "confidence": 0.9, "path": "single-route",
        "attempts": 1, "n_movies": 5, "tokens": 100,
        "narrowing": ["mood: scary", "audience: kids"],
        "filters": ["genres: Horror, Thriller (all)"],
    }
    text = intent_badge_text(row)
    assert "Narrowing by: mood: scary · audience: kids" in text
    assert "Filters: genres: Horror, Thriller (all)" in text
    assert "INTENT: ATTRIBUTE_FILTER" in text


def test_intent_badge_omits_empty_sections():
    text = intent_badge_text({
        "intent": "GREETING", "confidence": 1.0, "path": "single-route",
        "attempts": 1, "n_movies": 0, "tokens": 0,
    })
    assert "Narrowing" not in text and "Filters" not in text


def test_preference_chips_format():
    chips = preference_chips(UserSessionPreferences(
        preferred_mood="scary", audience="kids", noted_donts=["clowns"],
        preferred_genres=["Horror"], preferred_directors=["Cronenberg"],
    ))
    assert chips == [
        "mood: scary", "audience: kids", "no clowns",
        "genres: Horror", "dir. Cronenberg",
    ]
