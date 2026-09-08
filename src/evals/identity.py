"""Eval run identity (#59, #54 grill D1): config-hash keyed artifacts.

Pure helpers — a run is identified by the hash of its ExperimentConfig
snapshot; presets are presentation-layer bookmarks resolved by exact match
against pristine preset configs (same rule as the sidebar highlight, #30).
"""

import hashlib
import json
from datetime import UTC, datetime

from src.domain.config import ExperimentConfig, PresetType, matching_preset

_PRESET_SLUGS = {
    PresetType.PRODUCTION_HYBRID: "production",
    PresetType.FAST_BUDGET: "fast_budget",
    PresetType.NAIVE_BASELINE: "naive_baseline",
}


def config_hash(config: ExperimentConfig) -> str:
    """First 8 hex chars of the sha256 over the canonical config dump.

    Canonical = sorted keys, compact separators — the same config always
    hashes identically regardless of field declaration order.
    """
    canonical = json.dumps(config.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def preset_slug(config: ExperimentConfig) -> str:
    """Preset bookmark for a config: exact pristine match, else "custom"."""
    preset = matching_preset(config)
    return _PRESET_SLUGS.get(preset, "custom")


def run_filename(config: ExperimentConfig, when: datetime | None = None) -> str:
    """`{preset-or-custom}_{confighash8}_{timestamp}.json` (#54 grill D7)."""
    when = when or datetime.now(UTC)
    return f"{preset_slug(config)}_{config_hash(config)}_{when:%Y%m%dT%H%M%SZ}.json"
