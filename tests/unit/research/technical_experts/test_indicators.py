from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.technical_experts.indicators import ema, regression_slope_tstat, supertrend


def test_ema_is_causal_and_preserves_index() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    close = pd.Series([100.0, 101.0, 99.0, 102.0], index=index)

    result = ema(close, 2)

    assert result.index.equals(index)
    assert result.iloc[0] == close.iloc[0]
    assert result.iloc[-1] != close.iloc[-1]


class TestRegressionSlopeTstat:
    def test_tsi_02_regression_tstat_sign_and_nan(self) -> None:
        # TSI-02: the OLS intercept must use the window sum of the relative x grid; a
        # wrong intercept collapses SSE to zero and the t-stat never turns
        # negative. On a seeded random walk the t-stat is negative on a
        # material fraction of bars and NaN only across the pre-warmup prefix.
        rng = np.random.default_rng(7)
        idx = pd.date_range("2024-01-01", periods=400, freq="4h", tz="UTC")
        walk = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 400))), index=idx)
        tstat = regression_slope_tstat(walk, 63)
        assert float(tstat.isna().mean()) < 0.20
        assert float((tstat < 0).mean()) > 0.20

        decline = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(-0.01, 0.002, 200))),
            index=pd.date_range("2024-01-01", periods=200, freq="4h", tz="UTC"),
        )
        assert float(regression_slope_tstat(decline, 63).iloc[-1]) < 0.0

        with pytest.raises(ValueError, match="period must be >= 2"):
            regression_slope_tstat(walk, 1)


class TestSupertrend:
    def test_tsi_03_supertrend_state_persists(self) -> None:
        # TSI-03: SuperTrend is a state machine: up flip on close > prior final upper,
        # down flip on close < prior final lower, otherwise the prior state is
        # carried unchanged. The returned line is the lower band while long and
        # the upper band while short.
        idx = pd.date_range("2024-01-01", periods=300, freq="4h", tz="UTC")
        rally = pd.Series(100.0 * np.exp(np.cumsum(np.full(300, 0.004))), index=idx)
        line, long_trend = supertrend(rally * 1.005, rally * 0.995, rally, 10, 3.0)
        assert float(long_trend.mean()) > 0.90
        assert bool((line[long_trend] < rally[long_trend]).all())

        decline = pd.Series(100.0 * np.exp(np.cumsum(np.full(300, -0.004))), index=idx)
        line2, long_trend2 = supertrend(decline * 1.005, decline * 0.995, decline, 10, 3.0)
        assert float(long_trend2.mean()) < 0.20
        assert bool((line2[~long_trend2] > decline[~long_trend2]).all())

        path = pd.Series(np.concatenate([
            np.linspace(300.0, 120.0, 150),
            np.linspace(120.0, 320.0, 150),
        ]), index=idx)
        _line3, long_trend3 = supertrend(path * 1.002, path * 0.998, path, 10, 3.0)
        flips = np.diff(long_trend3.astype(int))
        assert int((flips > 0).sum()) >= 1
        assert int((flips < 0).sum()) >= 1
        assert not bool(long_trend3.iloc[100])
        assert bool(long_trend3.iloc[-1])
