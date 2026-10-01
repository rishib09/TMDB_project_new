"""#60 Evals tab pure helpers (#54 grill D1/D7/D9)."""

from src.ui.evals_tab import (
    run_display_name,
    staleness_flags,
    sweep_baseline_label,
    sweep_rows,
)


def test_display_name_preset_custom_and_legacy():
    assert run_display_name({"config_hash": "ab12cd34", "preset": "production"}) == "Production"
    assert run_display_name({"config_hash": "ab12cd34", "preset": "custom"}) == "custom @ ab12cd34"
    assert run_display_name({"label": "v1_1_enriched"}) == "v1_1_enriched (legacy)"


def test_staleness_missing_collection_and_old_dataset():
    run = {"collection": "full_gemini_embedding_2", "dataset_version": "2026-08-01"}
    flags = staleness_flags(
        run, collection_exists=lambda name: False, current_dataset_version="2026-09-01"
    )
    assert len(flags) == 2
    assert "no longer exists" in flags[0]
    assert "predates" in flags[1]


def test_staleness_clean_run_has_no_flags():
    run = {"collection": "full_gemini_embedding_2", "dataset_version": "2026-09-01"}
    assert staleness_flags(run, lambda name: True, "2026-09-01") == []


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


def test_sweep_baseline_labels_match_production():
    assert sweep_baseline_label("hybrid_alpha") == "0.5"
    assert sweep_baseline_label("retrieval_top_k") == "5"
    assert sweep_baseline_label("reranker") == "off"
    assert sweep_baseline_label("embedding_combo") == "full_gemini_embedding_2"
    assert sweep_baseline_label("router_model") == "google/gemini-3.5-flash-lite"


def _fleet(model: str, stamp: str, n_turns: int = 200, stack: str = "v2", **scores) -> dict:
    return {
        "mode": "conversation",
        "routing_stack": stack,
        "n_conversations": 23,
        "n_turns": n_turns,
        "timestamp": stamp,
        "config_snapshot": {"v2_router_model": model},
        **scores,
    }


def test_v2_fleet_ignores_pilots_and_v1_and_keeps_the_newest():
    from src.ui.evals_tab import headline_run, reply_points, v2_fleet_by_model

    runs = [
        _fleet("glm-5.3-flash", "2026-09-27T01:00:00", faithfulness=0.9),
        _fleet("glm-5.3-flash", "2026-09-27T02:00:00", faithfulness=0.6),  # newer, lower
        _fleet("glm-5.3-flash", "2026-09-27T03:00:00", n_turns=15, faithfulness=0.99),
        _fleet("glm-5.3-flash", "2026-09-27T04:00:00", stack="v1", faithfulness=0.99),
        _fleet("google/gemma-4-31b-it", "2026-09-27T02:00:00", faithfulness=None),
        _fleet("other/model", "2026-09-27T02:00:00", faithfulness=0.5),
    ]
    by_model = v2_fleet_by_model(runs, n_conversations=23, n_turns=200)
    assert set(by_model) == {"glm-5.3-flash", "google/gemma-4-31b-it", "other/model"}
    assert by_model["glm-5.3-flash"]["faithfulness"] == 0.6
    assert headline_run(by_model, "glm-5.3-flash")["timestamp"] == "2026-09-27T02:00:00"
    assert headline_run(by_model, "missing") is None
    assert reply_points(by_model) == [
        ("glm-5.3-flash", 0.6),
        ("other/model", 0.5),
    ]


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
            ),
        ]
    )
    assert sentence == (
        "Whether a reranker reorders them moved hit rate the furthest: "
        "89.7% on Off, 55.2% on ESCI MiniLM, 34.5 points."
    )
