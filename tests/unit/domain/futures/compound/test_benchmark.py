from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.compound.benchmark import (
    DailyMarketReturns,
    aggregate_1h_close_to_daily_last,
    build_causal_l2_benchmark,
    build_daily_market_returns,
)
from src.domain.futures.compound.config import L2BenchmarkConfig
from src.domain.futures.compound.contracts import CausalityError


class TestAggregate1hCloseToDailyLast:
    def test_aggregate_1h_close_to_daily_last_returns_final_close_not_mean(self) -> None:
        ts = np.arange(48, dtype=np.int64) * 3_600_000_000_000
        close = np.linspace(100.0, 147.0, 48).reshape(-1, 1)
        daily_ts, daily_close = aggregate_1h_close_to_daily_last(ts, close)
        assert len(daily_ts) == 2
        assert daily_close[0, 0] == pytest.approx(123.0)
        assert daily_close[1, 0] == pytest.approx(147.0)
        mean_0 = float(np.mean(close[:24, 0]))
        assert daily_close[0, 0] != pytest.approx(mean_0)


class TestBuildDailyMarketReturns:
    def test_returns_from_consecutive_closes(self) -> None:
        ts = np.array([0, 86400_000_000_000], dtype=np.int64)
        close = np.array([[100.0], [110.0]], dtype=np.float64)
        dmr = build_daily_market_returns(timestamps_ns=ts, close_2d=close, symbols=("TEST",))
        assert dmr.timestamps_ns.shape == (1,)
        assert dmr.returns_2d.shape == (1, 1)
        assert dmr.returns_2d[0, 0] == pytest.approx(0.10)


class TestBuildCausalL2Benchmark:
    def test_build_causal_l2_benchmark_trailing_window_matches_window_days(self) -> None:
        n = 100
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.01, (n, 2))
        ns_per_day = np.int64(24 * 3600 * 10**9)
        ts = np.arange(n, dtype=np.int64) * ns_per_day
        dmr = DailyMarketReturns(timestamps_ns=ts, returns_2d=returns.astype(np.float64), symbols=("BTCUSDT", "ETHUSDT"))
        window_start = n // 2
        window_ts = ts[window_start:window_start + 30]
        config = L2BenchmarkConfig(volatility_lookback_days=20, target_ann_vol=0.15)
        result = build_causal_l2_benchmark(daily_market_returns=dmr, window_timestamps_ns=window_ts, config=config)
        assert result.timestamps_ns.shape == (30,)
        np.testing.assert_array_equal(result.timestamps_ns, window_ts)

    def test_build_causal_l2_benchmark_unaligned_window_raises_causality_error(self) -> None:
        n = 50
        ts = np.arange(n, dtype=np.int64) * np.int64(24 * 3600 * 10**9)
        dmr = DailyMarketReturns(timestamps_ns=ts, returns_2d=np.zeros((n, 2)), symbols=("BTCUSDT", "ETHUSDT"))
        window_ts = np.arange(30, dtype=np.int64) * np.int64(24 * 3600 * 10**9) + np.int64(12 * 3600 * 10**9)
        config = L2BenchmarkConfig(volatility_lookback_days=20, target_ann_vol=0.15)
        with pytest.raises(CausalityError):
            build_causal_l2_benchmark(daily_market_returns=dmr, window_timestamps_ns=window_ts, config=config)

    def test_build_causal_l2_benchmark_primed_scale_tracks_target_volatility(self) -> None:
        n = 200
        ts = np.arange(n, dtype=np.int64) * np.int64(24 * 3600 * 10**9)
        rng = np.random.default_rng(42)
        high_vol = rng.normal(0.0, 0.03, (n, 2))
        dmr = DailyMarketReturns(timestamps_ns=ts, returns_2d=high_vol, symbols=("BTCUSDT", "ETHUSDT"))
        config = L2BenchmarkConfig(volatility_lookback_days=60, target_ann_vol=0.15)
        result = build_causal_l2_benchmark(daily_market_returns=dmr, window_timestamps_ns=ts, config=config)
        first_scale = result.causal_scale_1d[62]
        assert first_scale < 1.0
