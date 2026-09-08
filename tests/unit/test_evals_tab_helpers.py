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
    flags = staleness_flags(run, collection_exists=lambda name: False,
                            current_dataset_version="2026-09-01")
    assert len(flags) == 2
    assert "no longer exists" in flags[0]
    assert "predates" in flags[1]


def test_staleness_clean_run_has_no_flags():
    run = {"collection": "full_gemini_embedding_2", "dataset_version": "2026-09-01"}
    assert staleness_flags(run, lambda name: True, "2026-09-01") == []


def test_sweep_rows_newest_per_value_and_missing_hint():
    runs = [
        {"sweep": {"knob": "retrieval_top_k", "value": "3"}, "timestamp": "2026-09-01",
         "hit_rate": 0.5, "mrr": 0.4, "context_precision": 0.3},
        {"sweep": {"knob": "retrieval_top_k", "value": "3"}, "timestamp": "2026-09-08",
         "hit_rate": 0.8, "mrr": 0.6, "context_precision": 0.5},
        {"sweep": {"knob": "hybrid_alpha", "value": "0.5"}, "timestamp": "2026-09-08",
         "hit_rate": 0.9},
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
    assert sweep_baseline_label("router_model") == "~google/gemini-flash-latest"
