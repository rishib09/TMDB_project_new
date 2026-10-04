"""Chat layout helpers (#147): prose without movie cards, labeled metadata."""

from src.ui.chat_tab import conversation_prose, metadata_fields


def test_conversation_prose_drops_movie_cards():
    response = (
        "Rainy and tense — three films sit in that weather.\n"
        "\n"
        "**Se7en (1995)** — dir. David Fincher\n"
        "The bleakest of the set.\n"
        "\n"
        "**Heat (1995)** — dir. Michael Mann\n"
        "A city at night.\n"
    )
    assert conversation_prose(response) == (
        "Rainy and tense — three films sit in that weather."
    )


def test_conversation_prose_keeps_card_free_replies():
    text = "Which of those are you in the mood for?"
    assert conversation_prose(text) == text


def test_conversation_prose_keeps_a_title_mentioned_in_a_sentence():
    text = "Curtis Hanson directed **L.A. Confidential (1997)**, filed as Crime."
    assert conversation_prose(text) == text


def test_conversation_prose_of_cards_only_is_empty():
    response = (
        "**Se7en (1995)** — dir. David Fincher\n"
        "Bleak.\n"
        "\n"
        "**Heat (1995)** — dir. Michael Mann\n"
        "Night.\n"
    )
    assert conversation_prose(response) == ""


def test_metadata_fields_are_labeled_and_skip_empty_sections():
    fields = dict(metadata_fields({
        "intent": "SEMANTIC_SEARCH",
        "confidence": 0.86,
        "path": "retrieve",
        "attempts": 1,
        "n_movies": 4,
        "tokens": 640,
    }))
    assert fields["Intent"] == "SEMANTIC_SEARCH"
    assert fields["Confidence"] == "0.86"
    assert fields["Route"] == "retrieve"
    assert fields["Movies"] == "4"
    assert fields["Tokens"] == "640"
    assert "Narrowing" not in fields
    assert "Filters" not in fields


def test_metadata_fields_include_narrowing_filters_and_reroute():
    fields = dict(metadata_fields({
        "intent": "ATTRIBUTE_FILTER",
        "confidence": 0.9,
        "path": "reroute",
        "attempts": 2,
        "n_movies": 5,
        "tokens": 100,
        "narrowing": ["mood: scary", "audience: kids"],
        "filters": ["genres: Horror, Thriller (all)"],
    }))
    assert fields["Route"] == "reroute (x2)"
    assert fields["Narrowing"] == "mood: scary · audience: kids"
    assert fields["Filters"] == "genres: Horror, Thriller (all)"


def test_turn_stores_movies_for_the_retrieval_section():
    from types import SimpleNamespace

    from src.domain.config import ExperimentConfig
    from src.domain.memory import ConversationState
    from src.domain.movie import MovieRecord
    from src.domain.routing import IntentType, QueryRoutingDecision
    from src.observability.tracer import DualModeObservabilityManager
    from src.ui.session import MayaSession

    movie = MovieRecord(id=7, title="Heat", release_year=1995, genres=["Crime"])
    session = MayaSession.__new__(MayaSession)
    session.config = ExperimentConfig()
    session.conversation = ConversationState()
    session.tracer = DualModeObservabilityManager(session_id="layout")
    session.rag_version = "test"
    session.turn_log = []
    session.last_movies = []
    session._thread_id = "layout"
    session._graph_sig = session._graph_signature()
    decision = QueryRoutingDecision(
        intent=IntentType.SEMANTIC_SEARCH,
        confidence=0.8,
        standalone_query="rain",
        requires_rag=True,
    )
    session.graph = SimpleNamespace(invoke=lambda *_a, **_k: {
        "final_response": "Rainy.\n\n**Heat (1995)** — dir. Michael Mann\nNight.",
        "retrieved_movies": [movie],
        "routing_decision": decision,
        "turn_stage": "retrieve",
        "session_tokens": 10,
        "session_cost_usd": 0.0,
    })
    session.turn("something rainy")
    assert session.turn_log[0]["movies"][0].title == "Heat"
    assert conversation_prose(session.turn_log[0]["response"]) == "Rainy."


def test_stacked_turn_is_the_prototype_panel():
    from types import SimpleNamespace

    from src.ui.chat_tab import stacked_turn_html, user_bubble_html

    movie = SimpleNamespace(
        title="Heat",
        release_year=1995,
        vote_average=8.3,
        genres=["Crime", "Drama"],
        poster_url="https://image.tmdb.org/t/p/w500/heat.jpg",
    )
    row = {
        "intent": "SEMANTIC_SEARCH",
        "confidence": 0.9,
        "path": "retrieve",
        "attempts": 1,
        "n_movies": 1,
        "tokens": 100,
        "movies": [movie],
    }
    response = (
        "A city at night.\n\n"
        "**Heat (1995)** — dir. Michael Mann\n"
        "The one to watch.\n"
    )
    html_turn = stacked_turn_html("something rainy", response, row)
    assert 'class="maya-user"' in html_turn
    assert 'class="maya-assistant"' in html_turn
    assert "Conversation" in html_turn and "A city at night." in html_turn
    assert "**Heat (1995)**" not in html_turn
    assert "Retrieval" in html_turn and "heat.jpg" in html_turn
    assert "Metadata" in html_turn and "maya-chip" in html_turn
    assert "SEMANTIC_SEARCH" in html_turn
    assert "<h2" not in html_turn
    assert 'class="maya-kicker"' in html_turn
    assert "<script>" not in user_bubble_html('<script>alert("x")</script>')


def test_retrieval_strip_is_a_full_width_horizontal_scroller():
    from src.ui.chat_tab import _LAYOUT_CSS

    assert "grid-auto-flow: column" in _LAYOUT_CSS
    assert "grid-auto-columns: minmax(160px, 1fr)" in _LAYOUT_CSS
    assert "overflow-x: auto" in _LAYOUT_CSS
    assert "width: 112px" not in _LAYOUT_CSS

