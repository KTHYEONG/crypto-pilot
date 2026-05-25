from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.diagnostics import (
    alpha_gate_diagnostics,
    build_quality_report,
    gross_return_diagnostics,
    ndcg_proxy_at_k,
    passes_ic_gate,
    passes_quality_gate,
    passes_signal_preservation_gate,
)


def test_ndcg_proxy_at_k_range() -> None:
    score = np.array([[0.9, 0.5, 0.1, -0.1, -0.5]], dtype=np.float64)
    rel = np.array([[4.0, 3.0, 2.0, 1.0, 0.0]], dtype=np.float64)
    val = ndcg_proxy_at_k(score, rel, k=3)
    assert 0.0 <= val <= 1.0
    assert val > 0.95


def test_passes_ic_gate_returns_true_when_all_thresholds_met() -> None:
    """Track 3: passes_ic_gate must return True when all metrics exceed thresholds."""
    # Arrange — ic_summary() output format
    summary = {"mean_ic": 0.05, "t_stat": 3.0, "hit_ratio": 0.55}

    # Act + Assert
    assert passes_ic_gate(summary, min_mean_ic=0.02, min_t_stat=2.0, min_hit_ratio=0.45) is True


def test_passes_ic_gate_returns_false_when_mean_ic_below_threshold() -> None:
    """Track 3: passes_ic_gate must return False when mean_ic < min_mean_ic."""
    summary = {"mean_ic": 0.005, "t_stat": 3.0, "hit_ratio": 0.55}

    assert passes_ic_gate(summary, min_mean_ic=0.02, min_t_stat=2.0, min_hit_ratio=0.45) is False


def test_passes_ic_gate_returns_false_when_t_stat_below_threshold() -> None:
    """Track 3: passes_ic_gate must return False when t_stat < min_t_stat."""
    summary = {"mean_ic": 0.05, "t_stat": 1.0, "hit_ratio": 0.55}

    assert passes_ic_gate(summary, min_mean_ic=0.02, min_t_stat=2.0, min_hit_ratio=0.45) is False


def test_passes_ic_gate_returns_false_when_hit_ratio_below_threshold() -> None:
    """Track 3: passes_ic_gate must return False when hit_ratio < min_hit_ratio."""
    summary = {"mean_ic": 0.05, "t_stat": 3.0, "hit_ratio": 0.40}

    assert passes_ic_gate(summary, min_mean_ic=0.02, min_t_stat=2.0, min_hit_ratio=0.45) is False


def test_passes_ic_gate_accepts_quality_report_key_aliases() -> None:
    """Track 3: passes_ic_gate must accept build_quality_report() key format."""
    # Arrange — quality_report output format (aliased keys)
    report = {
        "spearman_rank_ic": 0.03,
        "ic_t_stat": 2.5,
        "ic_hit_ratio": 0.50,
    }

    assert passes_ic_gate(report, min_mean_ic=0.02, min_t_stat=2.0, min_hit_ratio=0.45) is True


def test_passes_ic_gate_boundary_values_at_exactly_threshold() -> None:
    """Track 3: passes_ic_gate must return True at exactly the threshold (inclusive)."""
    summary = {"mean_ic": 0.02, "t_stat": 2.0, "hit_ratio": 0.45}

    assert passes_ic_gate(summary, min_mean_ic=0.02, min_t_stat=2.0, min_hit_ratio=0.45) is True


def test_passes_ic_gate_warn_only_config_uses_relaxed_thresholds() -> None:
    """Track 3: StrategyMLConfig ic_gate_warn_only=True uses relaxed thresholds.

    ic_gate_warn_only is handled by ml_builder (caller). passes_ic_gate itself
    only returns bool — no side effects. This test verifies config default values
    and that a failing summary returns False (caller decides warn vs raise).
    """
    # Arrange
    cfg = StrategyMLConfig(ic_gate_warn_only=True)
    report = {"mean_ic": 0.0, "t_stat": 0.0, "hit_ratio": 0.0}

    # Act
    result = passes_ic_gate(
        report,
        min_mean_ic=cfg.ic_gate_min_mean_ic,
        min_t_stat=cfg.ic_gate_min_t_stat,
        min_hit_ratio=cfg.ic_gate_min_hit_ratio,
    )

    # Assert — gate fails; ml_builder (not passes_ic_gate) emits warning
    assert result is False
    assert cfg.ic_gate_warn_only is True


def test_build_quality_report_and_gate_pass() -> None:
    t, n, f = 8, 6, 4
    rng = np.random.default_rng(42)
    feature_values = rng.normal(size=(t, n, f)).astype(np.float32)
    feature_valid_mask = np.ones((t, n), dtype=bool)
    label_eligible_mask = np.ones((t, n), dtype=bool)
    signed_ret = rng.normal(scale=1e-3, size=(t, n)).astype(np.float64)
    score = signed_ret + rng.normal(scale=1e-5, size=(t, n)).astype(np.float64)
    relevance = np.tile(np.array([4, 3, 2, 1, 0, 2], dtype=np.float64), (t, 1))
    q50 = score.copy()
    q10 = q50 - 2e-4
    q90 = q50 + 2e-4
    alpha_long = np.maximum(score, 0.0)
    alpha_short = np.maximum(-score, 0.0)
    cost = np.full((t, n), 1e-4, dtype=np.float64)

    report = build_quality_report(
        feature_values=feature_values,
        feature_valid_mask=feature_valid_mask,
        label_eligible_mask=label_eligible_mask,
        score_2d=score,
        signed_ret_2d=signed_ret,
        relevance_2d=relevance,
        q10_2d=q10,
        q50_2d=q50,
        q90_2d=q90,
        alpha_long_2d=alpha_long,
        alpha_short_2d=alpha_short,
        cost_2d=cost,
    )

    assert report["feature_finite_ratio"] == 1.0
    assert report["label_valid_ratio"] == 1.0
    assert report["ranker_valid_ndcg_at_5"] > 0.0
    assert "ev_cost_ratio_proxy" in report
    assert passes_quality_gate(report) is True


# ---------------------------------------------------------------------------
# gross_return_diagnostics tests
# ---------------------------------------------------------------------------


def test_gross_return_diagnostics_returns_one_when_no_residualization() -> None:
    """gross_return_diagnostics must return variance_retention_ratio=1.0 for identical inputs."""
    # Arrange
    rng = np.random.default_rng(0)
    gross = rng.normal(scale=0.01, size=(10, 6)).astype(np.float32)
    eligible = np.ones((10, 6), dtype=bool)

    # Act — pass same array for both gross and resid
    result = gross_return_diagnostics(gross, gross, eligible, min_symbols=5)

    # Assert
    assert result["variance_retention_ratio"] == pytest.approx(1.0, rel=1e-6)
    assert result["n_timesteps"] == pytest.approx(10.0)


def test_gross_return_diagnostics_returns_less_than_one_after_shrinkage() -> None:
    """gross_return_diagnostics must reflect variance shrinkage when resid is scaled down."""
    # Arrange
    rng = np.random.default_rng(1)
    gross = rng.normal(scale=0.02, size=(10, 6)).astype(np.float32)
    resid = (gross * 0.5).astype(np.float32)
    eligible = np.ones((10, 6), dtype=bool)

    # Act
    result = gross_return_diagnostics(gross, resid, eligible, min_symbols=5)

    # Assert — std scales linearly, so ratio should be ~0.5
    assert result["variance_retention_ratio"] == pytest.approx(0.5, rel=1e-4)


def test_gross_return_diagnostics_skips_rows_below_min_symbols() -> None:
    """gross_return_diagnostics must skip rows when fewer than min_symbols valid symbols exist."""
    # Arrange — only 3 symbols, min_symbols=5 → no eligible rows
    rng = np.random.default_rng(2)
    gross = rng.normal(scale=0.01, size=(8, 3)).astype(np.float32)
    resid = gross * 0.8
    eligible = np.ones((8, 3), dtype=bool)

    # Act
    result = gross_return_diagnostics(gross, resid, eligible, min_symbols=5)

    # Assert — all metrics zero, no eligible timesteps
    assert result["n_timesteps"] == 0.0
    assert result["raw_cs_std_mean"] == 0.0
    assert result["resid_cs_std_mean"] == 0.0
    assert result["variance_retention_ratio"] == 0.0
    assert result["raw_nonzero_ratio"] == 0.0
    assert result["resid_nonzero_ratio"] == 0.0


def test_gross_return_diagnostics_handles_all_nan_gracefully() -> None:
    """gross_return_diagnostics must not raise and must return all-zero dict on full NaN input."""
    # Arrange
    gross = np.full((6, 6), np.nan, dtype=np.float32)
    resid = np.full((6, 6), np.nan, dtype=np.float32)
    eligible = np.ones((6, 6), dtype=bool)

    # Act — must not raise
    result = gross_return_diagnostics(gross, resid, eligible, min_symbols=5)

    # Assert
    assert result["n_timesteps"] == 0.0
    assert result["variance_retention_ratio"] == 0.0


def test_passes_signal_preservation_gate_returns_true_when_both_ratios_meet_threshold() -> None:
    """passes_signal_preservation_gate returns True when both ratios exceed thresholds."""
    # Arrange
    summary = {
        "xs_long_preservation_ratio": 0.75,
        "xs_short_preservation_ratio": 0.80,
    }

    # Act
    result = passes_signal_preservation_gate(
        summary,
        min_long_preservation_ratio=0.70,
        min_short_preservation_ratio=0.70,
    )

    # Assert
    assert result is True


def test_passes_signal_preservation_gate_returns_false_when_long_ratio_below_threshold() -> None:
    """passes_signal_preservation_gate returns False when long ratio is below threshold."""
    # Arrange
    summary = {
        "xs_long_preservation_ratio": 0.50,
        "xs_short_preservation_ratio": 0.80,
    }

    # Act
    result = passes_signal_preservation_gate(
        summary,
        min_long_preservation_ratio=0.70,
        min_short_preservation_ratio=0.60,
    )

    # Assert
    assert result is False


def test_passes_signal_preservation_gate_returns_false_when_short_ratio_below_threshold() -> None:
    """passes_signal_preservation_gate returns False when short ratio is below threshold."""
    # Arrange
    summary = {
        "xs_long_preservation_ratio": 0.80,
        "xs_short_preservation_ratio": 0.50,
    }

    # Act
    result = passes_signal_preservation_gate(
        summary,
        min_long_preservation_ratio=0.70,
        min_short_preservation_ratio=0.70,
    )

    # Assert
    assert result is False


def test_alpha_gate_diagnostics_exposes_fail_reasons() -> None:
    result = alpha_gate_diagnostics(
        alpha_p95_bps=5.0,
        friction_bps=12.0,
        hurdle_bps=10.0,
        long_nz=0.1,
        short_nz=0.2,
        xs_long_preservation_ratio=0.3,
        xs_short_preservation_ratio=0.4,
        min_long_nz=0.5,
        min_short_nz=0.5,
        min_xs_preservation=0.8,
    )
    assert result["alpha_gate_pass"] is False
    reasons = result["alpha_gate_fail_reasons"]
    assert isinstance(reasons, list)
    assert len(reasons) >= 1
