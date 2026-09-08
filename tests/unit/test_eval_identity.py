"""#59 run identity, sweeps, and budget gate (#54 grill D1/D3/D6/D7)."""

import pytest

from src.domain.config import ExperimentConfig, PresetType
from src.evals.identity import config_hash, preset_slug, run_filename
from src.evals.runner import BenchmarkRunner, sweep_configs


def test_config_hash_stable_and_knob_sensitive():
    a, b = ExperimentConfig(), ExperimentConfig()
    assert config_hash(a) == config_hash(b)
    b.hybrid_alpha = 0.75
    assert config_hash(a) != config_hash(b)


def test_preset_slug_exact_match_and_custom():
    production = ExperimentConfig().apply_preset(PresetType.PRODUCTION_HYBRID)
    assert preset_slug(production) == "production"
    production.retrieval_top_k = 7  # any manual edit -> custom
    assert preset_slug(production) == "custom"


def test_run_filename_scheme():
    config = ExperimentConfig().apply_preset(PresetType.FAST_BUDGET)
    name = run_filename(config)
    assert name.startswith(f"fast_budget_{config_hash(config)}_")
    assert name.endswith("Z.json")


def test_sweep_is_one_factor_at_a_time():
    baseline = ExperimentConfig().apply_preset(PresetType.PRODUCTION_HYBRID)
    for label, config in sweep_configs("hybrid_alpha"):
        diff = {
            k for k, v in config.model_dump().items()
            if baseline.model_dump()[k] != v
        }
        assert diff <= {"hybrid_alpha"}, f"{label} changed extra knobs: {diff}"


def test_sweep_reranker_variants():
    points = dict(sweep_configs("reranker"))
    assert points["off"].reranker_enabled is False
    assert points["ms-marco-MiniLM-L-12-v2"].reranker_enabled is True
    assert points["ms-marco-MiniLM-L-12-v2"].reranker_model == "ms-marco-MiniLM-L-12-v2"


def test_sweep_embedding_combo_sets_both_axes():
    points = dict(sweep_configs("embedding_combo"))
    combo = points["minimal_lfm_free"]
    assert (combo.column_preset, combo.embedding_profile) == ("minimal", "lfm_free")


def test_sweep_unknown_knob_raises():
    with pytest.raises(ValueError, match="unknown sweep knob"):
        sweep_configs("temperature")


class _BlockedTracker:
    def weekly_spend(self) -> float:
        return 10.0

    def verdict_for(self, spend):
        class V:
            value = "blocked"
        return V()


def test_budget_gate_aborts_live_run():
    runner = BenchmarkRunner(
        ExperimentConfig(), engine=None, budget_tracker=_BlockedTracker()
    )
    with pytest.raises(RuntimeError, match="budget exhausted"):
        runner._budget_check()


def test_summary_carries_identity_and_stamps():
    runner = BenchmarkRunner(
        ExperimentConfig(), engine=None, dataset_version="2026-09-01"
    )
    summary = runner._summarize([], "unit", mode="retrieval")
    assert summary.config_hash == config_hash(runner.config)
    # Defaults ARE the pristine Production preset (ADR 0008 verdict), so the
    # attribution rule must bookmark them as such.
    assert summary.preset == "production"
    assert summary.dataset_version == "2026-09-01"
