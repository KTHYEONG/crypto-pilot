"""Tests for the MHS evidence-gate statistical calibration primitives."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.calibration import (
    calibrate_max_share_null,
    sharpe_lower_confidence_bound,
    stationary_block_bootstrap,
)
from src.mhs.params import NULL_BOOTSTRAP_MIN_ROWS


def _ar1_series(n_rows: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    innovations = rng.standard_normal(n_rows)
    series = np.empty(n_rows, dtype="float64")
    series[0] = innovations[0]
    scale = float(np.sqrt(1.0 - phi**2))
    for i in range(1, n_rows):
        series[i] = phi * series[i - 1] + scale * innovations[i]
    return series


def _lag1_autocorr(series: np.ndarray) -> float:
    return float(np.corrcoef(series[:-1], series[1:])[0, 1])


# SCENARIO_MHS_BLOCK_BOOTSTRAP_PRESERVES_AUTOCORRELATION
def test_block_bootstrap_preserves_autocorrelation() -> None:
    source = _ar1_series(4000, 0.5, seed=7)
    resampled = stationary_block_bootstrap(
        source, 1000, np.random.default_rng(11), mean_block_days=20,
    )
    assert len(resampled) == 1000
    assert set(np.unique(resampled)).issubset(set(np.unique(source)))
    assert _lag1_autocorr(resampled) >= 0.25
    # iid 재표집이면 0에 수렴하므로 대조군으로도 자기상관 보존을 검증한다.
    iid_resample = source[np.random.default_rng(12).integers(0, len(source), 1000)]
    assert _lag1_autocorr(iid_resample) < _lag1_autocorr(resampled)


# SCENARIO_MHS_BLOCK_BOOTSTRAP_PRESERVES_AUTOCORRELATION
def test_block_bootstrap_rejects_degenerate_inputs() -> None:
    source = np.array([0.1, -0.2, 0.3])
    with pytest.raises(ValueError, match="size"):
        stationary_block_bootstrap(source, 0, np.random.default_rng(3))
    with pytest.raises(ValueError, match="mean_block_days"):
        stationary_block_bootstrap(
            source, 10, np.random.default_rng(3), mean_block_days=0,
        )
    with pytest.raises(ValueError, match="non-empty"):
        stationary_block_bootstrap(np.empty(0), 10, np.random.default_rng(3))


# SCENARIO_MHS_NULL_CALIBRATION_IS_DETERMINISTIC
def test_null_calibration_deterministic_and_global_state_untouched() -> None:
    pooled = np.random.default_rng(42).normal(0.0004, 0.02, 1429)
    global_state_before = np.random.get_state()[1][0]
    first = calibrate_max_share_null(pooled, 4, 357, 0.30, trials=400)
    second = calibrate_max_share_null(pooled, 4, 357, 0.30, trials=400)
    assert first.threshold == second.threshold
    assert first.observed_percentile == second.observed_percentile
    assert first.alpha == 0.05
    assert (first.n_folds, first.fold_days, first.trials) == (4, 357, 400)
    assert np.random.get_state()[1][0] == global_state_before


# SCENARIO_MHS_NULL_CALIBRATION_THRESHOLD_RANGE
def test_null_calibration_threshold_range_and_alpha_monotonicity() -> None:
    # 일간 vol 2% 정규 + 실측과 같은 강한 양의 드리프트(연율 Sharpe ~2):
    # 무드리프트 표본은 fold 로그성장이 음수가 빈번해 share null이 발산한다.
    pooled = np.random.default_rng(2026).normal(0.00227, 0.02, 1429)
    base = calibrate_max_share_null(pooled, 4, 357, 1.0 / 4, alpha=0.05, trials=800)
    relaxed = calibrate_max_share_null(pooled, 4, 357, 1.0 / 4, alpha=0.20, trials=800)
    assert 0.40 < base.threshold < 0.90
    assert base.threshold > 1.0 / 4
    # alpha 상향(더 관대한 오차율)은 임계값을 단조 감소시킨다.
    assert relaxed.threshold < base.threshold
    # 균등몫(1/n_folds) 관측치는 null 하위권에 위치한다.
    assert base.observed_percentile < 25.0


# SCENARIO_MHS_NULL_CALIBRATION_FAILS_CLOSED
def test_null_calibration_fails_closed_on_insufficient_rows() -> None:
    pooled = np.full(NULL_BOOTSTRAP_MIN_ROWS - 1, 0.001)
    with pytest.raises(DataIntegrityError, match=str(NULL_BOOTSTRAP_MIN_ROWS - 1)):
        calibrate_max_share_null(pooled, 4, 60, 0.4)


# SCENARIO_MHS_NULL_CALIBRATION_FAILS_CLOSED
def test_null_calibration_fails_closed_on_too_few_usable_trials() -> None:
    pooled = np.random.default_rng(5).normal(-0.05, 0.02, 600)
    with pytest.raises(DataIntegrityError):
        calibrate_max_share_null(pooled, 4, 60, 0.4, trials=200)


# SCENARIO_MHS_SHARPE_LCB_IS_CONSERVATIVE_AND_FAILS_CLOSED
def test_sharpe_lower_confidence_bound_below_point_and_convergent() -> None:
    short_sample = np.random.default_rng(99).normal(0.0006, 0.02, 357)
    long_sample = np.random.default_rng(98).normal(0.0006, 0.02, 1429)

    def _point_annualized(r: np.ndarray) -> float:
        return float(r.mean() / r.std(ddof=1) * np.sqrt(365.0))

    lcb_short = sharpe_lower_confidence_bound(short_sample)
    lcb_long = sharpe_lower_confidence_bound(long_sample)
    assert lcb_short < _point_annualized(short_sample)
    assert lcb_long < _point_annualized(long_sample)
    # 긴 표본일수록 하한이 점추정에 가까워진다(갭 단조 감소).
    assert (
        _point_annualized(long_sample) - lcb_long
        < _point_annualized(short_sample) - lcb_short
    )


# SCENARIO_MHS_SHARPE_LCB_IS_CONSERVATIVE_AND_FAILS_CLOSED
def test_sharpe_lower_confidence_bound_degenerate_inputs_return_neg_inf() -> None:
    degenerate_inputs = (
        np.array([0.01]),
        np.empty(0),
        np.full(100, 0.01),
    )
    for returns in degenerate_inputs:
        value = sharpe_lower_confidence_bound(returns)
        assert value == float("-inf")
