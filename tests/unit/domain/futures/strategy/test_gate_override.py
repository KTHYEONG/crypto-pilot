"""Tests for apply_tf_gate_overrides helper."""
from __future__ import annotations

from src.domain.futures.strategy.config import (
    CandidateStrategyConfig,
    apply_tf_gate_overrides,
)


def _base_cfg() -> CandidateStrategyConfig:
    return CandidateStrategyConfig(
        l1_pair_min_effective_obs=5.0,
        l1_min_sym_count=6,
        l1_min_fold_ratio=0.50,
        l1_min_realized_match_ratio=0.90,
    )


# ─── Scenario 5: Per-TF gate override ──────────────────────────────────────

def test_gate_override_applies_for_matching_tf() -> None:
    """Scenario 5: apply_tf_gate_overrides(cfg, '1h') returns cfg with 1h overrides."""
    cfg = _base_cfg()
    cfg = CandidateStrategyConfig(
        l1_pair_min_effective_obs=5.0,
        l1_min_sym_count=6,
        l1_min_fold_ratio=0.50,
        l1_min_realized_match_ratio=0.90,
        per_tf_gate_overrides={
            "1h": {
                "l1_pair_min_effective_obs": 3.0,
                "l1_min_sym_count": 4,
            },
        },
        per_tf_gate_enabled=True,
    )
    result = apply_tf_gate_overrides(cfg, "1h")
    assert result.l1_pair_min_effective_obs == 3.0
    assert result.l1_min_sym_count == 4
    # Non-overridden fields unchanged
    assert result.l1_min_fold_ratio == 0.50
    assert result.l1_min_realized_match_ratio == 0.90


# ─── Scenario 6: Gate override for non-overridden TF ───────────────────────

def test_gate_override_non_overridden_tf_returns_original() -> None:
    """Scenario 6: apply_tf_gate_overrides(cfg, '4h') returns same cfg when no 4h override."""
    cfg = CandidateStrategyConfig(
        l1_pair_min_effective_obs=5.0,
        l1_min_sym_count=6,
        per_tf_gate_overrides={"1h": {"l1_pair_min_effective_obs": 3.0}},
        per_tf_gate_enabled=True,
    )
    result = apply_tf_gate_overrides(cfg, "4h")
    # Same object (no copy) when no override exists
    assert result is cfg
    assert result.l1_pair_min_effective_obs == 5.0


def test_gate_override_non_overridden_tf_identity() -> None:
    """No override for '4h' → returns original cfg unchanged."""
    cfg = CandidateStrategyConfig(
        l1_pair_min_effective_obs=5.0,
        l1_min_sym_count=6,
        per_tf_gate_overrides={
            "1h": {"l1_pair_min_effective_obs": 3.0},
        },
    )
    result = apply_tf_gate_overrides(cfg, "4h")
    assert result is cfg


def test_gate_override_none_overrides_returns_original() -> None:
    """per_tf_gate_overrides=None → returns original cfg unchanged."""
    cfg = CandidateStrategyConfig(l1_pair_min_effective_obs=5.0)
    assert cfg.per_tf_gate_overrides is None
    result = apply_tf_gate_overrides(cfg, "1h")
    assert result is cfg
