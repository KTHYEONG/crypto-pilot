from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.domain.futures.compound.allocator import (
    apply_beta_hedge_overlay,
    derive_mdd_parity_scale,
)
from src.domain.futures.compound.benchmark import (
    assert_contemporaneous_alignment,
    causal_beta_series,
)
from src.domain.futures.compound.config import L2GateConfig
from src.domain.futures.compound.contracts import (
    CausalityError,
    L2BenchmarkSeries,
    L2GateVerdict,
)
from src.domain.futures.data_lake.run_windows import QuarterlyWindowConfig

_RNG = np.random.default_rng(42)


# ── T1: _daily_timestamps_from_4h matches aggregation grid ─────────────────────
class TestDailyTimestampsMatchAggregationGrid:
    def test_daily_timestamps_match_aggregation_grid(self) -> None:
        from src.domain.futures.compound.validation import (
            _daily_timestamps_from_4h,
            aggregate_returns_to_utc_days,
        )

        ns_per_4h = 4 * 3600 * 10**9
        n_4h = 60
        timestamps = np.arange(n_4h, dtype=np.int64) * ns_per_4h
        returns = _RNG.normal(0.0, 0.001, n_4h).astype(np.float64)

        daily_ret = aggregate_returns_to_utc_days(timestamps, returns)
        daily_ts = _daily_timestamps_from_4h(timestamps)

        assert len(daily_ts) == len(daily_ret)
        if len(daily_ts) > 0:
            ns_per_day = 6 * ns_per_4h
            first_day_start = timestamps[0] - (timestamps[0] % ns_per_day)
            expected_first_ts = first_day_start
            assert daily_ts[0] == expected_first_ts, (
                f"expected {expected_first_ts}, got {daily_ts[0]}"
            )

    def test_no_24h_offset(self) -> None:
        from src.domain.futures.compound.validation import (
            _daily_timestamps_from_4h,
            aggregate_returns_to_utc_days,
        )

        ns_per_4h = 4 * 3600 * 10**9
        n_4h = 12
        timestamps = np.arange(n_4h, dtype=np.int64) * ns_per_4h
        returns = np.zeros(n_4h, dtype=np.float64)

        daily_ret = aggregate_returns_to_utc_days(timestamps, returns)
        daily_ts = _daily_timestamps_from_4h(timestamps)

        assert len(daily_ts) == len(daily_ret)
        if len(daily_ts) > 0:
            expected_ts = timestamps[0] - (timestamps[0] % (6 * ns_per_4h))
            assert daily_ts[0] == expected_ts
            assert daily_ts[0] < timestamps[-1]


# ── T2: causal_beta_series uses only past ─────────────────────────────────────
class TestCausalBetaUsesOnlyPast:
    def test_causal_beta_uses_only_past(self) -> None:
        n = 200
        lookback = 60
        bench = _RNG.normal(0.0, 0.02, n).astype(np.float64)
        strat = (0.64 * bench + _RNG.normal(0.0, 0.004, n)).astype(np.float64)

        beta = causal_beta_series(
            strat, bench,
            lookback_days=lookback, min_obs=30, beta_clip=(-1.0, 3.0),
        )

        beta_up_to_k = beta[:100].copy()
        bench_modified = bench.copy()
        bench_modified[120:] *= 10
        beta2 = causal_beta_series(
            strat, bench_modified,
            lookback_days=lookback, min_obs=30, beta_clip=(-1.0, 3.0),
        )

        np.testing.assert_array_equal(beta_up_to_k, beta2[:100])


# ── T3: causal_beta_series recovers known beta ────────────────────────────────
class TestCausalBetaRecoversKnownBeta:
    def test_causal_beta_recovers_known_beta(self) -> None:
        n = 400
        lookback = 120
        true_beta = 0.64
        bench = _RNG.normal(0.0, 0.02, n).astype(np.float64)
        strat = (true_beta * bench + _RNG.normal(0.0, 0.004, n)).astype(np.float64)

        beta = causal_beta_series(
            strat, bench,
            lookback_days=lookback, min_obs=30, beta_clip=(-1.0, 3.0),
        )

        warmup_beta = beta[:lookback]
        assert np.all(warmup_beta == 0.0), "warmup period should be zero"

        post_warmup = beta[lookback:]
        mean_beta = float(np.mean(post_warmup))
        assert abs(mean_beta - true_beta) < 0.05, (
            f"mean beta {mean_beta:.4f} != {true_beta}"
        )


# ── T4: causal_beta_series warmup and clip ────────────────────────────────────
class TestCausalBetaWarmupAndClip:
    def test_causal_beta_warmup_and_clip(self) -> None:
        n = 100
        bench = _RNG.normal(0.0, 0.02, n).astype(np.float64)
        strat = np.full(n, 0.001, dtype=np.float64)

        beta = causal_beta_series(
            strat, bench,
            lookback_days=50, min_obs=30, beta_clip=(-1.0, 3.0),
        )

        assert np.all(beta[:50] == 0.0), "min_obs period should be zero"

    def test_clip_extreme_beta(self) -> None:
        n = 200
        lookback = 60
        bench = _RNG.normal(0.0, 0.02, n).astype(np.float64)
        strat = bench * 10.0 + _RNG.normal(0.0, 0.01, n)

        beta = causal_beta_series(
            strat, bench,
            lookback_days=lookback, min_obs=30, beta_clip=(-1.0, 3.0),
        )

        post_warmup = beta[lookback:]
        assert np.all(post_warmup <= 3.0 + 1e-10)
        assert np.all(post_warmup >= -1.0 - 1e-10)


# ── T5: assert_contemporaneous_alignment ──────────────────────────────────────
class TestAssertContemporaneousAlignment:
    def test_raises_on_shift(self) -> None:
        n = 200
        sig = _RNG.normal(0.0, 0.01, n).astype(np.float64)
        shifted = np.concatenate([sig[1:], np.zeros(1)])

        with pytest.raises(CausalityError):
            assert_contemporaneous_alignment(sig, shifted, max_lag=1)

    def test_passes_when_aligned(self) -> None:
        n = 200
        sig = _RNG.normal(0.0, 0.01, n).astype(np.float64)
        bench = sig * 0.5 + _RNG.normal(0.0, 0.005, n).astype(np.float64)

        assert_contemporaneous_alignment(sig, bench, max_lag=1)

    def test_short_series_does_not_raise(self) -> None:
        sig = np.array([0.001, 0.002], dtype=np.float64)
        bench = np.array([0.0015, 0.0025], dtype=np.float64)

        assert_contemporaneous_alignment(sig, bench, max_lag=1)


# ── T6: excess degenerates to absolute when beta=0/1 ─────────────────────────
class TestExcessDegenerates:
    def _compute_excess(self, daily_returns, benchmark_returns, beta_1d):
        return np.log1p(daily_returns) - beta_1d * np.log1p(benchmark_returns)

    def test_beta_zero_equals_absolute(self) -> None:
        n = 100
        strat = _RNG.normal(0.0, 0.01, n).astype(np.float64)
        bench = _RNG.normal(0.0, 0.01, n).astype(np.float64)

        beta_zero = np.zeros(n, dtype=np.float64)
        excess = self._compute_excess(strat, bench, beta_zero)
        expected = np.log1p(strat)
        np.testing.assert_array_almost_equal(excess, expected)

    def test_beta_one_equals_raw_difference(self) -> None:
        n = 100
        strat = _RNG.normal(0.0, 0.01, n).astype(np.float64)
        bench = _RNG.normal(0.0, 0.01, n).astype(np.float64)

        beta_one = np.ones(n, dtype=np.float64)
        excess = self._compute_excess(strat, bench, beta_one)
        expected = np.log1p(strat) - np.log1p(bench)
        np.testing.assert_array_almost_equal(excess, expected)


# ── T7: hedge overlay reduces benchmark columns ──────────────────────────────
class TestHedgeOverlayReducesBenchmarkColumns:
    def test_hedge_overlay_reduces_benchmark_columns(self) -> None:
        n_bars, n_syms = 10, 51
        symbols = tuple(f"SYM{i}" for i in range(51))
        btc_idx = 0
        eth_idx = 1
        symbols_list = list(symbols)
        symbols_list[btc_idx] = "BTCUSDT"
        symbols_list[eth_idx] = "ETHUSDT"
        symbols = tuple(symbols_list)

        weights = np.full((n_bars, n_syms), 1.0 / 51, dtype=np.float64)
        beta = np.full(n_bars, 0.6, dtype=np.float64)
        scale = np.ones(n_bars, dtype=np.float64)

        result = apply_beta_hedge_overlay(
            weights,
            symbols=symbols, beta_per_bar_1d=beta,
            benchmark_symbols=("BTCUSDT", "ETHUSDT"),
            benchmark_weights=(0.5, 0.5),
            benchmark_scale_1d=scale, gross_cap=2.0,
        )

        expected_btc = 1.0 / 51 + (-0.6 * 1.0 * 0.5)
        expected_eth = 1.0 / 51 + (-0.6 * 1.0 * 0.5)
        assert result[0, btc_idx] == pytest.approx(expected_btc)
        assert result[0, eth_idx] == pytest.approx(expected_eth)

        for j in range(2, 51):
            np.testing.assert_array_equal(result[:, j], weights[:, j])


# ── T8: hedge overlay respects gross cap ──────────────────────────────────────
class TestHedgeOverlayRespectsGrossCap:
    def test_hedge_overlay_respects_gross_cap(self) -> None:
        n_bars, n_syms = 10, 51
        symbols = tuple(f"SYM{i}" for i in range(51))
        symbols_list = list(symbols)
        symbols_list[0] = "BTCUSDT"
        symbols_list[1] = "ETHUSDT"
        symbols = tuple(symbols_list)

        zero_weights = np.zeros((n_bars, n_syms), dtype=np.float64)
        zero_weights[:, 2:] = 2.0 / 49
        weights = zero_weights
        beta = np.full(n_bars, 0.6, dtype=np.float64)
        scale = np.ones(n_bars, dtype=np.float64)
        gross_cap = 1.0

        result = apply_beta_hedge_overlay(
            weights,
            symbols=symbols, beta_per_bar_1d=beta,
            benchmark_symbols=("BTCUSDT", "ETHUSDT"),
            benchmark_weights=(0.5, 0.5),
            benchmark_scale_1d=scale, gross_cap=gross_cap,
        )

        gross = np.sum(np.abs(result[0]))
        assert gross <= gross_cap + 1e-10, f"gross={gross} > cap={gross_cap}"

        relative_ratios = result[0, 2:] / weights[0, 2:]
        assert np.allclose(relative_ratios, relative_ratios[0])


# ── T9: MDD parity scale ──────────────────────────────────────────────────────
class TestMddParityScale:
    def test_causal_and_clipped(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        returns = rng.normal(0.0, 0.01, n).astype(np.float64)

        scale = derive_mdd_parity_scale(returns, mdd_budget=0.10, max_scale=3.0)

        assert 1.0 <= scale <= 3.0

    def test_high_mdd_lower_scale(self) -> None:
        n = 500
        high_dd = np.full(n, -0.01, dtype=np.float64)
        high_dd[100:120] = -0.05
        high_dd[200:220] = -0.03

        scale_high = derive_mdd_parity_scale(high_dd, mdd_budget=0.10, max_scale=3.0)

        low_dd = np.full(n, 0.001, dtype=np.float64)
        low_dd[100:120] = -0.005
        low_dd[200:220] = -0.003

        scale_low = derive_mdd_parity_scale(low_dd, mdd_budget=0.10, max_scale=3.0)

        assert scale_high < scale_low, "higher MDD should give lower scale"

    def test_zero_mdd_returns_max_scale(self) -> None:
        returns = np.full(100, 0.001, dtype=np.float64)
        scale = derive_mdd_parity_scale(returns, mdd_budget=0.10, max_scale=3.0)
        assert scale == 3.0

    def test_negative_mdd_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="mdd_budget"):
            derive_mdd_parity_scale(np.zeros(10), mdd_budget=-0.01, max_scale=3.0)


# ── T10: min_oos_days raised, thresholds unchanged ──────────────────────────
class TestGateConfigThresholds:
    def test_min_oos_days_raised_not_relaxed(self) -> None:
        cfg = L2GateConfig()
        assert cfg.min_oos_days == 500, f"expected 500, got {cfg.min_oos_days}"
        assert cfg.min_excess_growth_probability == 0.90
        assert cfg.min_deflated_sharpe_probability == 0.90
        assert cfg.min_bootstrap_sharpe_probability == 0.90
        assert cfg.max_spa_pvalue == 0.10


# ── T11: quarterly window fits available axis ────────────────────────────────
class TestQuarterlyWindowFitsAxis:
    def test_quarterly_window_fits_available_axis(self) -> None:
        cfg = QuarterlyWindowConfig()
        total = cfg.warmup_days + cfg.l1_days + cfg.l2_days + cfg.l3_days
        assert total == 907, f"total days = {total}, expected 907"
        assert total <= 910, f"total days {total} exceeds axis budget 910"

    def test_boundaries_are_strictly_increasing(self) -> None:
        cfg = QuarterlyWindowConfig()
        from src.domain.futures.data_lake.run_windows import resolve_completed_quarter_window
        from datetime import date

        window = resolve_completed_quarter_window(date(2026, 7, 26), cfg)
        assert window.acquisition_start_ns < window.l1_start_ns
        assert window.l1_start_ns < window.l2_start_ns
        assert window.l2_start_ns < window.l3_start_ns
        assert window.l3_start_ns < window.cutoff_exclusive_ns


# ── T12: engine wires two-pass beta hedge ─────────────────────────────────────
class TestEngineTwoPassBetaHedge:
    def test_engine_wires_two_pass_beta_hedge(self, tmp_path: Path) -> None:
        from src.domain.futures.compound.config import CompoundEngineConfig, L2GateConfig
        from src.domain.futures.compound.contracts import (
            CompoundEngineResult,
            MarketFeatureCube,
            SealedHoldoutManifest,
        )
        from src.domain.futures.compound.engine import run_multiscale_compound_engine
        from src.domain.futures.compound.holdout_store import SealedHoldoutStore

        _NS_PER_HOUR = 3_600_000_000_000
        n_bars = 6000
        n_syms = 51
        symbols = tuple(f"SYM{i}" for i in range(51))
        symbols_list = list(symbols)
        symbols_list[0] = "BTCUSDT"
        symbols_list[1] = "ETHUSDT"
        symbols = tuple(symbols_list)

        close = np.column_stack(tuple(
            np.linspace(100, 110 + i * 0.1, n_bars) for i in range(n_syms)
        )).astype(np.float64)
        arr_f32 = close.astype(np.float32)

        cube = MarketFeatureCube(
            timestamps_ns=np.arange(n_bars, dtype=np.int64) * _NS_PER_HOUR,
            symbols=symbols,
            fields_2d={
                "open": arr_f32 * 0.9995,
                "high": arr_f32 * 1.005,
                "low": arr_f32 * 0.995,
                "close": arr_f32,
                "quote_volume": np.ones((n_bars, n_syms), dtype=np.float32) * 50_000_000,
                "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
                "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
                "mark": arr_f32.copy(),
                "index": arr_f32.copy(),
                "taker_buy_quote": np.ones((n_bars, n_syms), dtype=np.float32) * 25_000_000,
            },
            available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
            eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
            entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            capacity_usdt_2d=np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64),
            execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
            data_manifest_hash="h2",
        )

        universe = type("Universe", (), {"symbols": symbols, "snapshots": ()})()

        store = SealedHoldoutStore(tmp_path / "engine_beta_hedge.sqlite3")
        ns_per_day = 24 * _NS_PER_HOUR
        holdout_start_ns = int(cube.timestamps_ns[-1]) - 90 * ns_per_day
        manifest = SealedHoldoutManifest(
            holdout_id="beta-hedge-test",
            start_time_ns=holdout_start_ns,
            end_time_ns=int(cube.timestamps_ns[-1]),
            holdout_days=90,
            model_version="v1",
            data_manifest_hash="h2",
        )
        store.create(manifest)

        l2_gate_config = L2GateConfig(min_oos_days=30)
        cfg = CompoundEngineConfig(l2_gate=l2_gate_config)
        result = run_multiscale_compound_engine(
            market=cube,
            universe=universe,
            holdout_store=store,
            holdout_id="beta-hedge-test",
            config=cfg,
        )

        assert isinstance(result, CompoundEngineResult)
        assert result.l2 is not None
        assert result.l2.verdict in (
            L2GateVerdict.PASS, L2GateVerdict.FAIL, L2GateVerdict.NO_EVIDENCE,
        )


# ── T13: gate alignment invariant blocks regression ──────────────────────────
class TestGateAlignmentInvariant:
    def _make_ledger_with_shift(self, shift_days: int = 1):
        from src.domain.futures.compound.contracts import ExecutionLedger, L2BenchmarkSeries

        ns_per_4h = 4 * 3600 * 10**9
        n_4h = 360
        timestamps = np.arange(n_4h, dtype=np.int64) * ns_per_4h
        returns = _RNG.normal(0.0, 0.001, n_4h).astype(np.float64)

        ledger = ExecutionLedger(
            timestamps_ns=timestamps,
            net_returns_1d=returns,
            equity_1d=np.cumprod(1.0 + returns).astype(np.float64),
            target_weights_2d=np.zeros((n_4h, 2), dtype=np.float32),
            fee_returns_1d=np.zeros(n_4h, dtype=np.float64),
            slippage_returns_1d=np.zeros(n_4h, dtype=np.float64),
            impact_returns_1d=np.zeros(n_4h, dtype=np.float64),
            funding_returns_1d=np.zeros(n_4h, dtype=np.float64),
            integrity_ok=True,
            integrity_reasons=(),
        )

        n_daily = n_4h // 6
        daily_ts = np.arange(n_daily, dtype=np.int64) * (6 * ns_per_4h)

        if shift_days > 0:
            daily_ts = daily_ts + shift_days * (6 * ns_per_4h)

        benchmark = L2BenchmarkSeries(
            benchmark_id="shift_test",
            timestamps_ns=daily_ts[:n_daily],
            daily_returns_1d=np.zeros(n_daily, dtype=np.float64),
            causal_scale_1d=np.ones(n_daily, dtype=np.float64),
        )

        return ledger, benchmark

    def test_gate_alignment_invariant_blocks_regression(self) -> None:
        from src.domain.futures.compound.config import L2GateConfig
        from src.domain.futures.compound.multiplicity import TrialMultiplicity
        from src.domain.futures.compound.validation import evaluate_l2_walk_forward

        ns_per_4h = 4 * 3600 * 10**9
        n_4h = 600
        timestamps = np.arange(n_4h, dtype=np.int64) * ns_per_4h

        rng = np.random.default_rng(42)
        n_daily = n_4h // 6

        common_daily = rng.normal(0.0, 0.01, n_daily).astype(np.float64)
        strategy_ret_4h = np.repeat(common_daily, 6)[:n_4h] / 6 + rng.normal(0.0, 0.0005, n_4h).astype(np.float64)

        daily_ts = np.arange(n_daily, dtype=np.int64) * (6 * ns_per_4h)

        benchmark_shifted = L2BenchmarkSeries(
            benchmark_id="shifted",
            timestamps_ns=daily_ts,
            daily_returns_1d=np.concatenate([common_daily[1:], common_daily[:1]]).copy(),
            causal_scale_1d=np.ones(n_daily, dtype=np.float64),
        )

        ledger = _ledger(strategy_ret_4h, timestamps, n_weights_cols=2)

        beta_ones = np.ones(n_daily, dtype=np.float64)
        with pytest.raises(CausalityError):
            evaluate_l2_walk_forward(
                ledger=ledger, fold_ids_1d=np.zeros(n_4h, dtype=np.int16),
                benchmark=benchmark_shifted,
                trial_multiplicity=TrialMultiplicity(10, 10.0, 1.0),
                config=L2GateConfig(min_oos_days=50, min_rebalances=1, min_active_days_ratio=0.001),
                bootstrap_seed=42,
                beta_1d=beta_ones,
            )


# ── helpers ──────────────────────────────────────────────────────────────────
def _ledger(returns: np.ndarray, timestamps: np.ndarray | None = None, *, integrity_ok: bool = True, n_weights_cols: int = 2):
    from src.domain.futures.compound.contracts import ExecutionLedger

    equity = np.concatenate((np.array([1.0]), np.cumprod(1.0 + returns)))
    n = returns.size
    if timestamps is None:
        ns_per_4h = 4 * 3_600_000_000_000
        timestamps = np.arange(n, dtype=np.int64) * ns_per_4h
    rng_w = np.random.default_rng(123)
    weights = rng_w.uniform(-0.1, 0.1, (n, n_weights_cols)).astype(np.float32)
    return ExecutionLedger(
        timestamps_ns=timestamps,
        net_returns_1d=returns.astype(np.float64),
        equity_1d=equity.astype(np.float64),
        target_weights_2d=weights,
        fee_returns_1d=np.zeros(n, dtype=np.float64),
        slippage_returns_1d=np.zeros(n, dtype=np.float64),
        impact_returns_1d=np.zeros(n, dtype=np.float64),
        funding_returns_1d=np.zeros(n, dtype=np.float64),
        integrity_ok=integrity_ok,
        integrity_reasons=() if integrity_ok else ("execution_integrity",),
    )


def _benchmark_series(daily_timestamps: np.ndarray):
    from src.domain.futures.compound.contracts import L2BenchmarkSeries

    n = len(daily_timestamps)
    return L2BenchmarkSeries(
        benchmark_id="test",
        timestamps_ns=daily_timestamps,
        daily_returns_1d=np.zeros(n, dtype=np.float64),
        causal_scale_1d=np.ones(n, dtype=np.float64),
    )
