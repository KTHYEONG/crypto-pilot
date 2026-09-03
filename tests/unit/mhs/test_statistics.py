"""Tests for the MHS statistical evidence helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.mhs import statistics


def test_xs_rank_ic_empty_on_no_overlap() -> None:
    """Disjoint signal/forward-return indices yield an empty result dict."""
    idx_a = pd.date_range("2022-01-01", periods=5, freq="h", tz="UTC")
    idx_b = pd.date_range("2023-01-01", periods=5, freq="h", tz="UTC")
    signal = pd.DataFrame({"A": 1.0}, index=idx_a)
    opens = pd.DataFrame({"A": 1.0}, index=idx_b)

    assert statistics._xs_rank_ic(signal, opens, forward_bars=1) == {}


def test_xs_rank_ic_reports_finite_stats_on_synthetic_signal() -> None:
    """A well-formed signal/forward-return panel returns finite IC stats."""
    idx = pd.date_range("2022-01-01", periods=50, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    signal = pd.DataFrame(
        rng.normal(size=(50, 6)), index=idx, columns=list("ABCDEF"),
    )
    opens = pd.DataFrame(
        100.0 + np.cumsum(rng.normal(size=(50, 6)), axis=0), index=idx, columns=list("ABCDEF"),
    )

    result = statistics._xs_rank_ic(signal, opens, forward_bars=2)

    assert result
    assert np.isfinite(result["mean_ic"])
    assert result["forward_bars"] == 2


def test_annualized_1h_sharpe_none_on_short_series() -> None:
    """Fewer than 2 observations returns None, never NaN."""
    assert statistics._annualized_1h_sharpe(pd.Series([0.01])) is None


def test_annualized_1h_sharpe_none_on_zero_std() -> None:
    """A constant series has zero std and returns None explicitly."""
    net = pd.Series([0.01, 0.01, 0.01])
    assert statistics._annualized_1h_sharpe(net) is None


def test_annualized_1h_sharpe_finite_value() -> None:
    """A varying series returns a finite annualized Sharpe."""
    net = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01, 0.015])
    result = statistics._annualized_1h_sharpe(net)
    assert result is not None
    assert np.isfinite(result)


def test_finite_or_none_passes_finite_value() -> None:
    """A finite float passes through unchanged."""
    assert statistics._finite_or_none(1.5) == 1.5


def test_finite_or_none_coerces_nan_and_inf_to_none() -> None:
    """Non-finite values (NaN, inf) are coerced to an explicit None."""
    assert statistics._finite_or_none(float("nan")) is None
    assert statistics._finite_or_none(float("inf")) is None


def test_date_clustered_ols_empty_result_on_no_overlap() -> None:
    """Disjoint indices/columns yield the degenerate all-NaN result dict."""
    idx_a = pd.date_range("2022-01-01", periods=5, freq="h", tz="UTC")
    idx_b = pd.date_range("2023-01-01", periods=5, freq="h", tz="UTC")
    past = pd.DataFrame({"A": 1.0}, index=idx_a)
    opens = pd.DataFrame({"A": 1.0}, index=idx_b)

    result = statistics._date_clustered_ols(past, opens, forward_bars=1)

    assert result["n"] == 0
    assert result["n_dates"] == 0
    assert np.isnan(result["past_beta"])


def test_date_clustered_ols_below_min_sample_returns_nan_beta() -> None:
    """Fewer than 10 valid observations returns NaN beta without raising."""
    idx = pd.date_range("2022-01-01", periods=5, freq="h", tz="UTC")
    past = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    opens = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=idx)

    result = statistics._date_clustered_ols(past, opens, forward_bars=1)

    assert result["n"] < 10
    assert np.isnan(result["past_beta"])


def test_mean_ann_empty_series_returns_nan() -> None:
    """An empty series returns NaN rather than raising on an empty mean."""
    assert np.isnan(statistics._mean_ann(pd.Series([], dtype="float64"), 365.0))


def test_mean_ann_scales_by_periods_per_year() -> None:
    """The annualized mean is the per-bar mean scaled by periods_per_year."""
    series = pd.Series([0.01, 0.02, 0.03])
    assert statistics._mean_ann(series, 100.0) == series.mean() * 100.0


def test_geometric_cagr_nan_on_empty_or_nonpositive() -> None:
    """An empty series or a non-positive endpoint returns NaN."""
    assert np.isnan(statistics._geometric_cagr(pd.Series([], dtype="float64")))
    assert np.isnan(statistics._geometric_cagr(pd.Series([0.0, 1.0])))
    assert np.isnan(statistics._geometric_cagr(pd.Series([1.0, -1.0])))


def test_geometric_cagr_finite_on_growing_equity() -> None:
    """A monotonically growing equity curve returns a finite positive CAGR."""
    equity = pd.Series([100.0, 110.0, 121.0])
    result = statistics._geometric_cagr(equity)
    assert np.isfinite(result)
    assert result > 0


def test_mdd_nan_on_empty_series() -> None:
    """An empty equity series returns NaN rather than raising."""
    assert np.isnan(statistics._mdd(pd.Series([], dtype="float64")))


def test_mdd_zero_on_monotonic_series() -> None:
    """A monotonically increasing equity series has zero max drawdown."""
    equity = pd.Series([100.0, 110.0, 120.0])
    assert statistics._mdd(equity) == 0.0


def test_mdd_negative_on_drawdown() -> None:
    """A peak-to-trough decline reports a negative max drawdown."""
    equity = pd.Series([100.0, 120.0, 90.0, 110.0])
    result = statistics._mdd(equity)
    assert result < 0
    assert np.isclose(result, 90.0 / 120.0 - 1.0)


def test_causal_lag1_autocorr_nan_on_short_or_constant_window() -> None:
    """Fewer than 3 points, or a zero-variance half-window, yields NaN."""
    assert np.isnan(statistics._causal_lag1_autocorr(np.array([1.0, 2.0])))
    assert np.isnan(statistics._causal_lag1_autocorr(np.array([1.0, 1.0, 1.0])))


def test_causal_lag1_autocorr_finite_on_varying_window() -> None:
    """A varying window returns a finite lag-1 Pearson correlation."""
    x = np.array([1.0, 2.0, 1.5, 3.0, 2.5])
    result = statistics._causal_lag1_autocorr(x)
    assert np.isfinite(result)
    assert -1.0 <= result <= 1.0


def test_per_observation_sharpe_nan_on_short_or_zero_variance() -> None:
    """Fewer than 2 observations, or zero variance, yields NaN."""
    assert np.isnan(statistics._per_observation_sharpe(pd.Series([0.01])))
    assert np.isnan(statistics._per_observation_sharpe(pd.Series([0.01, 0.01, 0.01])))


def test_per_observation_sharpe_matches_mean_over_std() -> None:
    """The per-observation Sharpe is exactly mean/std with no annualization scale."""
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, -0.01])
    result = statistics._per_observation_sharpe(returns)
    expected = float(returns.mean() / returns.std(ddof=1))
    assert np.isclose(result, expected)


def test_hourly_ledger_series_resamples_and_aligns_turnover() -> None:
    """Sub-hourly equity/turnover series resample to hourly with aligned indices."""
    idx = pd.date_range("2022-01-01", periods=180, freq="20min", tz="UTC")
    equity = pd.Series(np.linspace(100.0, 110.0, len(idx)), index=idx)
    turnover = pd.Series(0.01, index=idx)

    equity_1h, net_returns_1h, turnover_1h = statistics._hourly_ledger_series(equity, turnover)

    assert equity_1h.index.freqstr in (None, "h", "H")
    assert net_returns_1h.index.equals(turnover_1h.index)
    assert (turnover_1h > 0).all()


def test_bootstrap_ci_degenerate_single_observation() -> None:
    """A single-observation series returns that value as both CI bounds."""
    net = pd.Series([0.05])
    lo, hi = statistics._bootstrap_ci(net, n_replicates=10, mean_block=168, seed=1)
    assert lo == hi == 0.05


def test_bootstrap_ci_empty_series_returns_nan_pair() -> None:
    """An empty series returns a (NaN, NaN) confidence interval."""
    net = pd.Series([], dtype="float64")
    lo, hi = statistics._bootstrap_ci(net, n_replicates=10, mean_block=168, seed=1)
    assert np.isnan(lo)
    assert np.isnan(hi)


def test_bootstrap_ci_bounds_are_ordered_and_finite() -> None:
    """A well-formed return series yields a finite, ordered (lo, hi) CI."""
    rng = np.random.default_rng(7)
    net = pd.Series(rng.normal(0.001, 0.01, 300))

    lo, hi = statistics._bootstrap_ci(net, n_replicates=50, mean_block=20, seed=7)

    assert np.isfinite(lo)
    assert np.isfinite(hi)
    assert lo <= hi
