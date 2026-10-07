"""Pure helpers for the static Evals report."""

from src.ui.evals_tab import sweep_rows


def test_sweep_rows_newest_per_value_and_missing_hint():
    runs = [
        {
            "sweep": {"knob": "retrieval_top_k", "value": "3"},
            "timestamp": "2026-09-01",
            "hit_rate": 0.5,
            "mrr": 0.4,
            "context_precision": 0.3,
        },
        {
            "sweep": {"knob": "retrieval_top_k", "value": "3"},
            "timestamp": "2026-09-08",
            "hit_rate": 0.8,
            "mrr": 0.6,
            "context_precision": 0.5,
        },
        {
            "sweep": {"knob": "hybrid_alpha", "value": "0.5"},
            "timestamp": "2026-09-08",
            "hit_rate": 0.9,
        },
    ]
    rows, missing = sweep_rows(runs, "retrieval_top_k")
    top3 = [r for r in rows if r["value"] == "3" and r["metric"] == "Hit Rate@5"]
    assert top3 == [{"value": "3", "metric": "Hit Rate@5", "score": 0.8}]  # newest wins
    assert missing == ["5", "10"]


def test_hit_rate_points_follow_sweep_order_and_ignore_mrr():
    from src.ui.evals_tab import REPORT_DECISIONS, hit_rate_points

    runs = [
        {
            "sweep": {"knob": "retrieval_top_k", "value": "10"},
            "timestamp": "2026-09-02",
            "hit_rate": 0.9,
            "mrr": 0.1,
        },
        {
            "sweep": {"knob": "retrieval_top_k", "value": "3"},
            "timestamp": "2026-09-02",
            "hit_rate": 0.5,
            "mrr": 0.4,
        },
        {"sweep": {"knob": "retrieval_top_k", "value": "5"}, "timestamp": "2026-09-02", "mrr": 0.8},
        {
            "sweep": {"knob": "router_model", "value": "meta-llama/llama-3.2-3b-instruct"},
            "timestamp": "2026-09-02",
            "hit_rate": 0.2,
        },
    ]
    assert hit_rate_points(runs, "retrieval_top_k") == [("3", 0.5), ("10", 0.9)]
    assert "router_model" not in REPORT_DECISIONS


def test_retrieval_chart_rows_keep_hit_rate_mrr_and_precision_on_one_chart():
    from src.ui.evals_tab import retrieval_chart_rows

    runs = [
        {
            "sweep": {"knob": "retrieval_top_k", "value": "10"},
            "timestamp": "2026-09-02",
            "hit_rate": 0.9,
            "mrr": 0.4,
            "context_precision": 0.3,
            "faithfulness": 0.2,
        },
        {
            "sweep": {"knob": "retrieval_top_k", "value": "3"},
            "timestamp": "2026-09-02",
            "hit_rate": 0.5,
            "mrr": 0.2,
        },
    ]
    assert retrieval_chart_rows(runs, "retrieval_top_k") == [
        {"option": "Keep 3", "line": "Hit Rate@5", "score": 50.0},
        {"option": "Keep 3", "line": "MRR@5", "score": 20.0},
        {"option": "Keep 10", "line": "Hit Rate@5", "score": 90.0},
        {"option": "Keep 10", "line": "MRR@5", "score": 40.0},
        {"option": "Keep 10", "line": "Context Precision@5", "score": 30.0},
    ]


def test_mrr_matches_precision_only_when_every_shared_point_is_equal():
    from src.ui.evals_tab import mrr_matches_precision

    same = [
        {"option": "Keep 5", "line": "MRR@5", "score": 72.1},
        {"option": "Keep 5", "line": "Context Precision@5", "score": 72.1},
    ]
    different = [
        {"option": "Keep 5", "line": "MRR@5", "score": 72.1},
        {"option": "Keep 5", "line": "Context Precision@5", "score": 40.0},
    ]
    assert mrr_matches_precision(same)
    assert not mrr_matches_precision(different)


def test_swing_tie_count_note_and_percent():
    from src.ui.evals_tab import (
        lead_sentence,
        percent_label,
        saved_count_note,
        score_swing,
        widest_decision,
    )

    assert score_swing([89.7, 55.2]) == 34.5
    assert score_swing([1.0]) is None
    assert widest_decision({"reranker": 34.5, "hybrid_alpha": 34.5, "models": 10.1}) == [
        "reranker",
        "hybrid_alpha",
    ]
    assert saved_count_note(29, 35, "queries") == ("Scored 29 queries. The golden file now has 35.")
    assert saved_count_note(23, 23, "conversations") is None
    assert percent_label(0.226) == "22.6%"
    sentence = lead_sentence(
        [
            (
                "Whether a reranker reorders them",
                "hit rate",
                [("Off", 89.7), ("ESCI MiniLM", 55.2)],
                34.5,
            ),
        ]
    )
    assert sentence == (
        "Whether a reranker reorders them moved hit rate the furthest: "
        "89.7% on Off, 55.2% on ESCI MiniLM, 34.5 points."
    )


def test_retrieval_caption_states_every_saved_count_against_the_golden_file():
    from src.ui.evals_tab import _retrieval_caption

    runs = [
        {
            "sweep": {"knob": "hybrid_alpha", "value": "0.0"},
            "timestamp": "2026-09-10T00:00:00+00:00",
            "n_queries": 29,
            "hit_rate": 0.7,
        },
        {
            "sweep": {"knob": "hybrid_alpha", "value": "1.0"},
            "timestamp": "2026-09-10T00:00:00+00:00",
            "n_queries": 35,
            "hit_rate": 0.9,
        },
    ]
    text = _retrieval_caption(runs, "hybrid_alpha", 35)
    assert text.startswith("Scored 29, 35 queries across the options. The golden file now has 35.")


def test_single_turn_score_rows_keep_precision_and_mrr():
    from src.ui.evals_tab import single_turn_score_rows

    runs = [
        {
            "sweep": {"knob": "retrieval_top_k", "value": "5"},
            "timestamp": "2026-09-10",
            "hit_rate": 0.897,
            "mrr": 0.7,
            "context_precision": 0.6,
            "faithfulness": 0.5,
            "relevancy": 0.4,
        }
    ]
    rows = single_turn_score_rows(runs)
    assert rows == [
        {
            "decision": "How many movies are kept",
            "option": "Keep 5",
            "Hit Rate@5": 89.7,
            "MRR@5": 70.0,
            "Context Precision@5": 60.0,
            "Faithfulness (judge)": 50.0,
            "Relevancy (judge)": 40.0,
        }
    ]
