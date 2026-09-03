"""RGR: fold-concentration gate re-specification tests.

Covers the bounded gross-share statistic, its fold-count-aware derived
threshold, and the calibration property that a known-good process is not
false-rejected (RGR-01 through RGR-05).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.baseline.backtest import BacktestResult
from src.research.evaluation.reliability import (
    derive_fold_concentration_threshold,
    compute_equal_duration_fold_distribution,
    compute_fold_distribution,
)


def _year_contribution_result(contributions: list[float], start_year: int = 2022) -> BacktestResult:
    """BacktestResult whose yearly log contributions equal ``contributions``.

    One equity mark per calendar year (YS frequency) makes each inter-year log
    return land in its own year bucket, so the contributions telescope exactly.
    """
    n = len(contributions)
    idx = pd.date_range(f"{start_year}-01-01", periods=n + 1, freq="YS", tz="UTC")
    values = [100.0]
    for contribution in contributions:
        values.append(values[-1] * np.exp(contribution))
    equity = pd.Series(values, index=idx)
    trades = pd.DataFrame({
        "entry_bar": list(range(n)),
        "exit_bar": list(range(1, n + 1)),
        "pnl": [1.0] * n,
        "return_pct": [0.001] * n,
    })
    return BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame(index=idx))


class TestFoldStatisticIsBounded:
    """RGR-01-FOLD-STATISTIC-IS-BOUNDED"""

    def test_legacy_statistic_is_unbounded_but_new_statistic_is_bounded(self) -> None:
        result = _year_contribution_result([0.30, 0.30, 0.30, -0.29])
        r = compute_fold_distribution(result)

        # legacy max|v| / |sum v| diverges as net approaches zero (~0.49 here)
        assert r.max_period_contribution > 0.40
        assert abs(r.max_period_contribution - (0.30 / 0.61)) < 1e-9

        # corrected max|v| / sum|v| is bounded in [1/n_folds, 1]
        assert abs(r.fold_concentration - (0.30 / 1.19)) < 1e-9
        assert 1.0 / 4.0 <= r.fold_concentration <= 1.0

    def test_second_fixture_matches_measured_legacy_and_corrected_values(self) -> None:
        result = _year_contribution_result([0.45, 0.45, 0.45, -0.35])
        r = compute_fold_distribution(result)

        assert abs(r.max_period_contribution - 0.450) < 1e-9
        assert abs(r.fold_concentration - (0.45 / 1.70)) < 1e-9


class TestThresholdIsDeterministicAndFoldCountAware:
    """RGR-02-THRESHOLD-IS-DETERMINISTIC-AND-FOLD-COUNT-AWARE"""

    def test_threshold_is_bit_identical_across_calls(self) -> None:
        a = derive_fold_concentration_threshold(5, 0.987)
        b = derive_fold_concentration_threshold(5, 0.987)
        assert a == b

    def test_threshold_strictly_decreases_with_fold_count(self) -> None:
        t4 = derive_fold_concentration_threshold(4, 0.987)
        t5 = derive_fold_concentration_threshold(5, 0.987)
        t8 = derive_fold_concentration_threshold(8, 0.987)
        assert t4 > t5 > t8

    def test_threshold_respects_uniform_baseline_floor(self) -> None:
        for n_folds in (2, 3, 4, 5, 8):
            assert derive_fold_concentration_threshold(n_folds, 0.987) >= 1.0 / n_folds

    def test_threshold_rejects_invalid_arguments(self) -> None:
        with pytest.raises(ValueError, match="n_folds"):
            derive_fold_concentration_threshold(1, 0.987)
        with pytest.raises(ValueError, match="false_rejection_rate"):
            derive_fold_concentration_threshold(5, 0.987, false_rejection_rate=0.0)
        with pytest.raises(ValueError, match="false_rejection_rate"):
            derive_fold_concentration_threshold(5, 0.987, false_rejection_rate=0.5)
        with pytest.raises(ValueError, match="draws"):
            derive_fold_concentration_threshold(5, 0.987, draws=999)


class TestKnownGoodProcessIsNotFalseRejected:
    """RGR-03-KNOWN-GOOD-PROCESS-IS-NOT-FALSE-REJECTED"""

    def test_calibration_rejection_rate_matches_false_rejection_target(self) -> None:
        rng = np.random.default_rng(2024)
        reference_sharpe = 1.3
        threshold = derive_fold_concentration_threshold(
            4, reference_sharpe, false_rejection_rate=0.10,
        )
        draws = rng.normal(reference_sharpe, 1.0, size=(400, 4))
        abs_values = np.abs(draws)
        concentration = abs_values.max(axis=1) / abs_values.sum(axis=1)
        rejection_rate = float(np.mean(concentration > threshold))
        assert abs(rejection_rate - 0.10) <= 0.05


class TestFoldGateUsesBoundedStatistic:
    """RGR-04-FOLD-GATE-USES-BOUNDED-STATISTIC"""

    def test_known_good_pattern_passes_with_bounded_statistic(self) -> None:
        result = _year_contribution_result([0.45, 0.45, 0.45, -0.35])
        r = compute_fold_distribution(result)

        # legacy diagnostic still reported unchanged
        assert abs(r.max_period_contribution - 0.450) < 1e-9
        # bounded statistic clears the derived threshold
        assert r.fold_concentration < r.fold_concentration_threshold
        assert r.gate_pass is True
        assert r.fold_concentration_threshold > 0.0
        assert r.fold_reference_sharpe > 0.0

    def test_zero_trade_early_return_still_passes(self) -> None:
        idx = pd.date_range("2022-01-01", periods=5, freq="YS", tz="UTC")
        empty = BacktestResult(
            equity=pd.Series([100.0] * 5, index=idx),
            trades=pd.DataFrame(columns=["entry_bar", "pnl", "return_pct"]),
            signals=pd.DataFrame(index=idx),
        )
        r = compute_fold_distribution(empty)
        assert r.n_folds == 0
        assert r.gate_pass is True


class TestEqualDurationVariantStaysConsistent:
    """RGR-05-EQUAL-DURATION-VARIANT-STAYS-CONSISTENT"""

    def test_equal_duration_and_annual_folds_agree_on_pass_fail(self) -> None:
        idx = pd.date_range("2023-01-01", periods=731, freq="D", tz="UTC")
        values = np.full(731, 1000.0)
        values[365:] = 2000.0
        equity = pd.Series(values, index=idx)
        trades = pd.DataFrame({
            "entry_bar": [100],
            "exit_bar": [700],
            "pnl": [1.0],
            "return_pct": [0.001],
        })
        result = BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame(index=idx))

        annual = compute_fold_distribution(result)
        equal_duration = compute_equal_duration_fold_distribution(equity)

        # both use the bounded statistic and a derived threshold, never the constant
        assert annual.gate_pass == (annual.fold_concentration <= annual.fold_concentration_threshold)
        assert (
            equal_duration.gate_pass
            == (equal_duration.fold_concentration <= equal_duration.fold_concentration_threshold)
        )
        assert annual.gate_pass == equal_duration.gate_pass
