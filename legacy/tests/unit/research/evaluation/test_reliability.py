from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.contracts import CostModel, StrategySpec
from src.market_data.storage.loaders import load_ohlcv_4h
from src.research.baseline.backtest import BacktestResult
from src.research.evaluation.reliability import (
    FoldDistributionResult,
    ReliabilityGateConfig,
    ReliabilityGateResult,
    _equal_duration_fold_labels,
    _year_log_return_contributions,
    compute_equal_duration_fold_distribution,
    compute_equity_reliability_gate,
    compute_fold_distribution,
    compute_portfolio_reliability_gate,
    compute_reliability_gate,
    compute_stress_test_gate,
    compute_turnover_fold_upper_bound,
    count_closed_trades,
    derive_block_size,
    derive_cost_multiple_hurdle_rate,
    derive_realized_weights_cost_total,
    equity_span_years,
    split_holdout_segment,
)

BTC_PATH = Path("data/futures/ohlcv/1h/BTCUSDT.parquet")


class TestReliabilityGateConfig:
    def test_reliability_gate_config_defaults(self) -> None:
        c = ReliabilityGateConfig()
        assert (c.hurdle_rate, c.hurdle_cost_multiple, c.block_size,
                c.n_bootstrap, c.seed, c.min_trades,
                c.mdd_floor, c.t_stat_floor, c.max_period_contribution) == (
            0.15, 2.0, None, 3000, 0, 30, -0.25, 2.0, 0.40,
        )

    def test_reliability_gate_result_fields(self) -> None:
        assert {f.name for f in dataclasses.fields(ReliabilityGateResult)} == {
            "lcb90_cagr", "lcb95_cagr", "p_negative", "point_cagr",
            "t_stat", "trade_count", "block_size_used", "verdict",
        }
        assert {f.name for f in dataclasses.fields(FoldDistributionResult)} == {
            "n_folds", "median_fold_cagr", "worst_fold_cagr",
            "median_fold_calmar", "max_period_contribution", "gate_pass",
            "fold_concentration", "fold_concentration_threshold", "fold_reference_sharpe",
        }

    def test_config_validation(self) -> None:
        with pytest.raises(ValueError, match="hurdle_rate"):
            ReliabilityGateConfig(hurdle_rate=-0.01)
        with pytest.raises(ValueError, match="block_size"):
            ReliabilityGateConfig(block_size=0)
        with pytest.raises(ValueError, match="n_bootstrap"):
            ReliabilityGateConfig(n_bootstrap=10)
        with pytest.raises(ValueError, match="min_trades"):
            ReliabilityGateConfig(min_trades=0)
        with pytest.raises(ValueError, match="mdd_floor"):
            ReliabilityGateConfig(mdd_floor=0.0)
        with pytest.raises(ValueError, match="t_stat_floor"):
            ReliabilityGateConfig(t_stat_floor=-0.1)
        with pytest.raises(ValueError, match="lcb_confidence"):
            ReliabilityGateConfig(lcb_confidence=1.0)
        with pytest.raises(ValueError, match="max_period_contribution"):
            ReliabilityGateConfig(max_period_contribution=0.0)
        # SCENARIO_HURDLE_COST_MULTIPLE_CONTRACT_LOCK: default 2.0 is locked by
        # the defaults assertion above; a negative override must fail closed.
        with pytest.raises(ValueError, match="hurdle_cost_multiple"):
            ReliabilityGateConfig(hurdle_cost_multiple=-1.0)


class TestDeriveBlockSize:
    def test_derive_block_size_white_noise_and_ar1(self) -> None:
        rng = np.random.default_rng(3)
        white_noise = rng.normal(0, 0.02, 200)
        assert derive_block_size(white_noise) == 1

        n = 200
        rng2 = np.random.default_rng(7)
        noise = rng2.normal(0, 0.02, n)
        ar = np.zeros(n)
        for t in range(n):
            ar[t] = 0.8 * ar[t - 3] + noise[t] if t >= 3 else noise[t]
        block = derive_block_size(ar)
        assert block >= 3, "lag-3 dependence must be detected"
        assert block <= max(1, n // 5), "block must not exceed the 20%-of-sample cap"

    def test_derive_block_size_tiny_sample_returns_one(self) -> None:
        assert derive_block_size(np.array([0.01, -0.01, 0.02])) == 1
        assert derive_block_size(np.zeros(50)) == 1


def _strongly_positive_trades(n: int = 50, mean: float = 0.02, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = np.abs(rng.normal(mean, 0.01, n))
    return pd.DataFrame({"return_pct": rets})


def test_reliability_gate_execution() -> None:
    trades = _strongly_positive_trades(seed=2)
    result = compute_reliability_gate(trades, years=2.0, sharpe=1.5, mdd=-0.05)
    assert isinstance(result, ReliabilityGateResult)
    assert result.verdict == "PASS"
    assert result.trade_count == len(trades)
    assert result.lcb90_cagr > 0.15


class TestCountClosedTrades:
    """SCENARIO_RELIABILITY_RELOCATE_01: relocated closed-trade count proxy."""

    def test_scenario_reliability_relocate_01_counts_transitions(self) -> None:
        # SCENARIO_RELIABILITY_RELOCATE_01
        weights = pd.DataFrame(
            {"A": [0.0, 0.0, 0.5, 0.5], "B": [0.0, 0.0, 0.0, -0.5]},
        )
        assert count_closed_trades(weights) == 2

    def test_scenario_reliability_relocate_01_zero_and_empty_frames(self) -> None:
        assert count_closed_trades(pd.DataFrame({"A": [0.0, 0.0, 0.0]})) == 0
        assert count_closed_trades(pd.DataFrame({"A": [0.5, 0.5]})) == 0
        assert count_closed_trades(pd.DataFrame()) == 0

    def test_scenario_reliability_relocate_01_matches_legacy_proxy_formula(self) -> None:
        rng = np.random.default_rng(0)
        weights = pd.DataFrame(rng.normal(0.0, 0.2, (40, 5)))
        arr = weights.to_numpy(dtype=np.float64)
        expected = int(np.count_nonzero(np.abs(np.diff(arr, axis=0)) > 1e-12))
        assert count_closed_trades(weights) == expected


class TestComputeReliabilityGate:
    def test_compute_reliability_gate_verdicts(self) -> None:
        strong = _strongly_positive_trades()
        r = compute_reliability_gate(strong, years=2.0, sharpe=1.5, mdd=-0.05)
        assert r.verdict == "PASS"
        assert r.p_negative == 0.0
        assert r.trade_count == 50
        assert r.lcb90_cagr > 0.15

        # SC-GATE-03: all-negative trades -> FAIL, LCB below zero
        neg = pd.DataFrame({"return_pct": np.full(50, -0.01)})
        r_neg = compute_reliability_gate(neg, years=2.0, sharpe=-1.0, mdd=-0.05)
        assert r_neg.lcb90_cagr < 0
        assert r_neg.verdict == "FAIL"

        # SC-GATE-04: too few trades -> PENDING, never a silent PASS
        thin = pd.DataFrame({"return_pct": np.full(10, 0.05)})
        r_thin = compute_reliability_gate(thin, years=2.0, sharpe=1.5, mdd=-0.05)
        assert r_thin.verdict == "PENDING"
        assert r_thin.lcb90_cagr > 0.15

        # SC-GATE-05: mdd below the floor -> FAIL despite strong LCB
        r_mdd = compute_reliability_gate(strong, years=2.0, sharpe=1.5, mdd=-0.30)
        assert r_mdd.verdict == "FAIL"

        # SC-GATE-06: t_stat below floor -> FAIL
        r_t = compute_reliability_gate(strong, years=2.0, sharpe=1.0, mdd=-0.05)
        assert r_t.verdict == "FAIL"

        # SC-GATE-07: statistically positive but below hurdle_rate -> FAIL (not PASS)
        weak = pd.DataFrame({"return_pct": np.full(50, 0.003)})
        r_weak = compute_reliability_gate(weak, years=2.0, sharpe=1.5, mdd=-0.05)
        assert r_weak.lcb90_cagr > 0
        assert r_weak.lcb90_cagr < 0.15
        assert r_weak.verdict == "FAIL"

    def test_compute_reliability_gate_determinism(self) -> None:
        trades = _strongly_positive_trades(seed=5)
        a = compute_reliability_gate(trades, years=2.0, sharpe=1.5, mdd=-0.05)
        b = compute_reliability_gate(trades, years=2.0, sharpe=1.5, mdd=-0.05)
        assert a == b
        assert a.block_size_used == b.block_size_used

    def test_compute_reliability_gate_empty_returns_pending(self) -> None:
        empty = pd.DataFrame({"return_pct": []})
        r = compute_reliability_gate(empty, years=2.0, sharpe=1.5, mdd=-0.05)
        assert r.verdict == "PENDING"
        assert r.trade_count == 0
        assert r.lcb90_cagr == 0.0
        assert r.point_cagr == 0.0

    def test_compute_reliability_gate_input_validation(self) -> None:
        with pytest.raises(ValueError, match="return_pct"):
            compute_reliability_gate(pd.DataFrame({"pnl": [1.0]}), years=2.0, sharpe=1.0, mdd=-0.1)
        with pytest.raises(ValueError, match="years"):
            compute_reliability_gate(
                pd.DataFrame({"return_pct": [0.01]}), years=0.0, sharpe=1.0, mdd=-0.1,
            )


class TestComputeEquityReliabilityGate:
    """SC-GATE-EQ-01/02: the canonical promotion gate on the marked equity stream."""

    def test_equity_reliability_gate_is_canonical_for_single_and_portfolio_ledgers(self) -> None:
        # SC-GATE-EQ-01: a valid but zero-return ledger cannot pass a 15% LCB and
        # the canonical gate is byte-identical to the former portfolio gate.
        flat = pd.Series(
            [100.0, 100.0],
            index=pd.date_range("2025-01-01", periods=2, freq="4h", tz="UTC"),
        )
        assert compute_equity_reliability_gate(flat, 30).verdict == "FAIL"
        assert compute_equity_reliability_gate(flat, 30).lcb90_cagr == 0.0

        idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
        rising = pd.Series([100.0, 110.0, 121.0, 133.1], index=idx)
        canonical = compute_equity_reliability_gate(rising, closed_trade_count=50)
        delegator = compute_portfolio_reliability_gate(rising, closed_trade_count=50)
        assert canonical == delegator, "the portfolio delegator must reuse the canonical gate"
        assert canonical.verdict == "PASS"

    def test_equity_gate_pending_below_min_closed_trades(self) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
        rising = pd.Series([100.0, 110.0, 121.0, 133.1], index=idx)
        assert compute_equity_reliability_gate(rising, closed_trade_count=10).verdict == "PENDING"

    def test_equity_gate_determinism(self) -> None:
        idx = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
        equity = pd.Series([100.0, 103.0, 101.5, 106.0, 109.0, 111.0], index=idx)
        a = compute_equity_reliability_gate(equity, closed_trade_count=40)
        b = compute_equity_reliability_gate(equity, closed_trade_count=40)
        assert a == b

    def test_equity_gate_rejects_malformed_ledger(self) -> None:
        # SC-GATE-EQ-02: no malformed ledger can be silently promoted.
        base_idx = pd.date_range("2025-01-01", periods=3, freq="4h", tz="UTC")
        with pytest.raises(ValueError, match="monotonic"):
            compute_equity_reliability_gate(
                pd.Series([100.0, 110.0, 90.0], index=base_idx[::-1]), 30,
            )
        with pytest.raises(ValueError, match="strictly positive"):
            compute_equity_reliability_gate(
                pd.Series([100.0, 0.0, 110.0], index=base_idx), 30,
            )
        with pytest.raises(ValueError, match="finite"):
            compute_equity_reliability_gate(
                pd.Series([100.0, np.nan], index=base_idx[:2]), 30,
            )
        with pytest.raises(ValueError, match="at least 2"):
            compute_equity_reliability_gate(
                pd.Series([100.0], index=base_idx[:1]), 30,
            )
        with pytest.raises(ValueError, match="closed_trade_count"):
            compute_equity_reliability_gate(
                pd.Series([100.0, 110.0, 90.0], index=base_idx), -1,
            )


class TestEquitySpanYears:
    def test_equity_span_years_matches_existing_gate_internal_formula(self) -> None:
        # The extracted helper reproduces the gate's inline formula exactly:
        # 365.25 days apart is 1.0 year, and the gate delegates to it.
        idx = pd.date_range("2024-01-01", periods=2, freq="365.25D", tz="UTC")
        equity = pd.Series([100.0, 110.0], index=idx)
        assert equity_span_years(equity) == pytest.approx(1.0, abs=1e-9)
        gate = compute_equity_reliability_gate(equity, closed_trade_count=40)
        assert gate.verdict in ("PASS", "FAIL", "PENDING")

    def test_equity_span_years_rejects_short_and_malformed_ledgers(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            equity_span_years(pd.Series([100.0], index=pd.date_range("2024-01-01", periods=1, tz="UTC")))
        with pytest.raises(ValueError, match="DatetimeIndex"):
            equity_span_years(pd.Series([100.0, 110.0]))
        empty_range = pd.date_range("2024-01-01", periods=2, freq="365.25D", tz="UTC")
        with pytest.raises(ValueError, match="positive time range"):
            equity_span_years(pd.Series([100.0, 110.0], index=empty_range[::-1]))


class TestDeriveCostMultipleHurdleRate:
    def test_derive_cost_multiple_hurdle_rate_scales_with_measured_cost(self) -> None:
        # The hurdle is the annualized measured allocation cost scaled by the
        # margin: 0.02 total cost over 2 years * 2x = 0.02 annualized hurdle.
        assert derive_cost_multiple_hurdle_rate(0.02, 2.0, 2.0) == pytest.approx(0.02)
        assert derive_cost_multiple_hurdle_rate(0.0, 2.0, 2.0) == 0.0
        assert derive_cost_multiple_hurdle_rate(0.06, 3.0, 3.0) == pytest.approx(0.06)

    def test_derive_cost_multiple_hurdle_rate_rejects_non_positive_years(self) -> None:
        with pytest.raises(ValueError, match="years"):
            derive_cost_multiple_hurdle_rate(0.02, 0.0, 2.0)
        with pytest.raises(ValueError, match="allocation_cost_total"):
            derive_cost_multiple_hurdle_rate(-0.01, 2.0, 2.0)
        with pytest.raises(ValueError, match="cost_multiple"):
            derive_cost_multiple_hurdle_rate(0.02, 2.0, -1.0)


class TestDeriveRealizedWeightsCostTotal:
    # SCENARIO_COST_TOTAL_MATCHES_LEDGER_CONVENTION
    def test_derive_realized_weights_cost_total_matches_ledger_convention(self) -> None:
        # 3-row 2-column weight path: deltas per column sum to (|1.0-0.0| +
        # |-1.0-0.0|) = 2.0 on bar 1 and (|0.5-1.0| + |-0.5-(-1.0)|) = 1.0 on
        # bar 2, total turnover 3.0 -- the exact sum-of-abs-changes convention
        # run_xs_composite_ledger and evaluate_xs_admission's breakeven_cost
        # charge, with no 0.5 factor.
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        w = pd.DataFrame({"A": [0.0, 1.0, 0.5], "B": [0.0, -1.0, -0.5]}, index=idx)
        assert abs(derive_realized_weights_cost_total(w, 0.0008) - (2.0 + 1.0) * 0.0008) < 1e-12

    # SCENARIO_COST_TOTAL_ZERO_TURNOVER
    def test_derive_realized_weights_cost_total_zero_turnover(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        w = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [-0.5, -0.5, -0.5]}, index=idx)
        assert derive_realized_weights_cost_total(w, 0.0008) == 0.0

    def test_derive_realized_weights_cost_total_fewer_than_two_rows_fail_closed(self) -> None:
        idx = pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC")
        single = pd.DataFrame({"A": [1.0], "B": [-1.0]}, index=idx)
        assert derive_realized_weights_cost_total(single, 0.0008) == 0.0

    # SCENARIO_COST_TOTAL_NEGATIVE_RATE_RAISES
    def test_derive_realized_weights_cost_total_rejects_negative_rate(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        w = pd.DataFrame({"A": [0.0, 1.0, 0.5]}, index=idx)
        with pytest.raises(ValueError, match="round_trip_cost_rate"):
            derive_realized_weights_cost_total(w, -0.0001)


class TestSplitHoldoutSegment:
    def _result(self, entry_bars: list[int], exit_bars: list[int]) -> BacktestResult:
        idx = pd.date_range("2025-06-01", periods=20, freq="4D", tz="UTC")
        equity = pd.Series(np.linspace(10000, 12000, 20), index=idx)
        trades = pd.DataFrame({
            "entry_bar": entry_bars,
            "exit_bar": exit_bars,
            "pnl": [10.0] * len(entry_bars),
            "return_pct": [0.01] * len(entry_bars),
        })
        return BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame(index=idx))

    def test_split_holdout_segment_rebasing_and_exit_classification(self) -> None:
        cutoff = pd.Timestamp("2025-07-01", tz="UTC")
        # exits at bar 5 -> 2025-06-21 (pre-cutoff) and bar 12 -> 2025-07-17 (post-cutoff)
        seg = split_holdout_segment(self._result([2, 8], [5, 12]), cutoff)

        assert abs(seg.holdout_equity.iloc[0] - 1.0) < 1e-9
        assert seg.observation_trades["exit_bar"].tolist() == [5]
        assert seg.holdout_trades["exit_bar"].tolist() == [12]
        assert seg.holdout_mdd <= 0.0
        assert seg.holdout_years > 0

    def test_split_holdout_counts_crossing_trade_at_exit_not_entry(self) -> None:
        # SC-HOLDOUT-EXIT-01: a trade entered before the cutoff but closed after it
        # must leave observation evidence and be counted in the holdout.
        cutoff = pd.Timestamp("2025-07-01", tz="UTC")
        seg = split_holdout_segment(self._result([2], [12]), cutoff)
        assert len(seg.observation_trades) == 0
        assert len(seg.holdout_trades) == 1

        closed_pre_cutoff = split_holdout_segment(self._result([2], [5]), cutoff)
        assert len(closed_pre_cutoff.observation_trades) == 1
        assert len(closed_pre_cutoff.holdout_trades) == 0

    def test_split_holdout_empty_trades_has_empty_segments(self) -> None:
        cutoff = pd.Timestamp("2025-07-01", tz="UTC")
        seg = split_holdout_segment(self._result([], []), cutoff)
        assert seg.observation_trades.empty
        assert seg.holdout_trades.empty

    def test_split_holdout_uses_portfolio_exit_time_when_present(self) -> None:
        cutoff = pd.Timestamp("2025-07-01", tz="UTC")
        result = self._result([2], [5])
        result.trades["exit_time"] = [pd.Timestamp("2025-06-21", tz="UTC")]
        seg = split_holdout_segment(result, cutoff)
        assert len(seg.observation_trades) == 1
        assert seg.holdout_trades.empty

    def test_split_holdout_segment_requires_exit_timing(self) -> None:
        idx = pd.date_range("2025-06-01", periods=20, freq="4D", tz="UTC")
        equity = pd.Series(np.linspace(10000, 12000, 20), index=idx)
        trades = pd.DataFrame({
            "entry_bar": [5],
            "pnl": [10.0],
            "return_pct": [0.01],
        })
        result = BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame(index=idx))
        with pytest.raises(ValueError, match="exit_bar"):
            split_holdout_segment(result, pd.Timestamp("2025-07-01", tz="UTC"))

    def test_split_holdout_segment_validation(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            split_holdout_segment(
                self._result([2], [5]), pd.Timestamp("2030-01-01", tz="UTC"),
            )
        with pytest.raises(ValueError, match="tz-naive"):
            split_holdout_segment(
                self._result([2], [5]), pd.Timestamp("2025-07-01"),
            )


def _fold_equity_fixture(equity_values: list[float], n_trades: int = 4) -> BacktestResult:
    idx = pd.date_range("2022-01-01", periods=len(equity_values), freq="YS", tz="UTC")
    equity = pd.Series(equity_values, index=idx)
    trades = pd.DataFrame({
        "entry_bar": list(range(n_trades)),
        "exit_bar": list(range(1, n_trades + 1)),
        "pnl": [0.0] * n_trades,
        "return_pct": [0.0] * n_trades,
    })
    return BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame(index=idx))


class TestYearLogReturnContributions:
    """SC-FOLD-MTM-01: marked log returns are attributed by mark timestamp, not entry."""

    def test_year_log_return_contributions_attributes_by_mark_timestamp(self) -> None:
        equity = pd.Series(
            [100.0, 110.0, 121.0],
            index=pd.DatetimeIndex([
                "2024-12-31 20:00:00+00:00",
                "2025-01-01 00:00:00+00:00",
                "2025-01-01 04:00:00+00:00",
            ]),
        )
        contrib = _year_log_return_contributions(equity)
        # the entire two-bar gain is attributed to 2025 marks even though the
        # first mark's predecessor was in 2024
        assert abs(contrib[2025] - np.log(1.21)) < 1e-12

    def test_year_log_return_contributions_validation(self) -> None:
        idx = pd.DatetimeIndex(["2025-01-01 00:00:00+00:00", "2025-01-01 04:00:00+00:00"])
        with pytest.raises(ValueError, match="monotonic"):
            _year_log_return_contributions(pd.Series([100.0, 110.0], index=idx[::-1]))
        with pytest.raises(ValueError, match="strictly positive"):
            _year_log_return_contributions(pd.Series([100.0, -1.0], index=idx))
        with pytest.raises(ValueError, match="finite"):
            _year_log_return_contributions(pd.Series([100.0, np.nan], index=idx))
        with pytest.raises(ValueError, match="at least 2"):
            _year_log_return_contributions(pd.Series([100.0], index=idx[:1]))
        with pytest.raises(ValueError, match="DatetimeIndex"):
            _year_log_return_contributions(pd.Series([100.0, 110.0], index=[1, 2]))


class TestComputeFoldDistribution:
    def test_fold_distribution_uses_marked_cross_year_log_returns(self) -> None:
        # a trade entered in 2024 and spanning the year boundary contributes its
        # post-cutoff marked gains entirely to 2025
        equity = pd.Series(
            [100.0, 110.0, 121.0],
            index=pd.DatetimeIndex([
                "2024-12-31 20:00:00+00:00",
                "2025-01-01 00:00:00+00:00",
                "2025-01-01 04:00:00+00:00",
            ]),
        )
        trades = pd.DataFrame({
            "entry_bar": [0],
            "exit_bar": [2],
            "pnl": [100.0],
            "return_pct": [1.0],
        })
        result = BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame(index=equity.index))
        r = compute_fold_distribution(result)
        assert abs(r.max_period_contribution - 1.0) < 1e-12
        assert r.gate_pass is False

    def test_fold_concentration_boundary_passes_at_or_below_limit(self) -> None:
        # SC-FOLD-MTM-02: balanced multi-year growth -> <= .40 passes
        balanced = compute_fold_distribution(_fold_equity_fixture([100.0, 200.0, 400.0, 800.0]))
        assert abs(balanced.max_period_contribution - (1.0 / 3.0)) < 1e-9
        assert balanced.gate_pass is True

        # one dominant year -> > .40 fails
        dominant = compute_fold_distribution(_fold_equity_fixture([100.0, 100.0, 100.0, 1000.0]))
        assert abs(dominant.max_period_contribution - 1.0) < 1e-9
        assert dominant.gate_pass is False

    def test_compute_fold_distribution_matches_measured_v1_concentration(self) -> None:
        # Pinned to the original 2022-04-01 data floor: this locks in the v1
        # measured concentration figure and must stay stable regardless of
        # how far back the shared local data lake is later backfilled.
        df = load_ohlcv_4h(BTC_PATH, start="2022-04-01", end="2025-12-31 23:59:59")
        from src.research.baseline.backtest import run_backtest

        spec, costs = StrategySpec(), CostModel()
        result = run_backtest(df, spec, costs)
        r = compute_fold_distribution(result)
        assert r.n_folds == 4
        assert abs(r.max_period_contribution - 0.8179) < 1e-3
        assert r.gate_pass is False

    def test_fold_distribution_empty_trades_and_short_equity(self) -> None:
        empty_trades = pd.DataFrame(columns=["entry_bar", "pnl", "return_pct"])
        idx = pd.date_range("2022-01-01", periods=4, freq="YS", tz="UTC")
        equity = pd.Series(
            [10000, 10000, 13000, 14000, 14200],
            index=pd.date_range("2022-01-01", periods=5, freq="YS", tz="UTC"),
        )
        empty = BacktestResult(
            equity=equity, trades=empty_trades, signals=pd.DataFrame(index=idx),
        )
        r = compute_fold_distribution(empty)
        assert r.n_folds == 0
        assert r.gate_pass is True

        idx = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
        equity = pd.Series(np.linspace(10000, 11000, 50), index=idx)
        trades = pd.DataFrame({"entry_bar": [10], "pnl": [5.0], "return_pct": [0.001]})
        short = BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame(index=idx))
        with pytest.raises(ValueError, match="fewer than 2"):
            compute_fold_distribution(short)


class TestEqualDurationFoldLabelsAndTurnoverBound:
    def test_equal_duration_fold_labels_extraction_is_behavior_preserving(
        self,
    ) -> None:
        # RELIAB-01: the private helper extracted from
        # compute_equal_duration_fold_distribution's inline block must assign
        # fold codes exactly as the original inline pd.cut did -- right-closed
        # calendar month-start intervals. A monthly point exactly on an
        # internal 6-month boundary (2022-07-01) belongs to the fold ENDING
        # there (fold 0), and the first point sitting on the left bin edge is
        # left unfolded (NaN).
        idx = pd.date_range("2022-01-01", periods=13, freq="MS", tz="UTC")
        labels = _equal_duration_fold_labels(idx, "6MS")
        assert len(labels) == 13
        assert labels[0] != labels[0]  # left edge of the first bin: no fold
        assert labels[1] == 0
        assert labels[6] == 0  # internal boundary point -> preceding fold
        assert labels[7] == 1
        assert labels[12] == 1  # final bin edge -> last fold

        # The public routine built on the helper still admits the same folds --
        # same window, same fold count as the pre-refactor inline construction.
        daily = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")
        equity = pd.Series(np.cumprod(1.0 + np.full(400, 0.001)), index=daily)
        assert compute_equal_duration_fold_distribution(equity).n_folds >= 2

    def test_equal_duration_fold_labels_rejects_malformed_and_short_span(
        self,
    ) -> None:
        # RELIAB-01: identical validation to the original inline block.
        idx = pd.date_range("2022-01-01", periods=6, freq="2MS", tz="UTC")
        with pytest.raises(ValueError, match="month-start frequency"):
            _equal_duration_fold_labels(idx, "6ME")
        short = pd.date_range("2022-01-01", periods=2, freq="4h", tz="UTC")
        with pytest.raises(ValueError, match="does not admit"):
            _equal_duration_fold_labels(short, "6MS")

    def test_turnover_fold_upper_bound_targets_worst_regime(self) -> None:
        # RELIAB-02: on a two-regime turnover series (first half near-zero,
        # second half materially higher, spanning >= 12 months) the worst
        # 6-month fold's annualized mean dominates the full-period pooled mean
        # and lands close to the high-regime fold's own annualized level.
        idx = pd.date_range("2022-01-01", periods=8760, freq="4h", tz="UTC")
        values = np.concatenate([np.full(4380, 0.001), np.full(4380, 0.05)])
        turnover = pd.Series(values, index=idx)
        bound = compute_turnover_fold_upper_bound(turnover, 2190)
        pooled = float(turnover.mean()) * 2190
        assert bound > pooled
        assert abs(bound - 0.05 * 2190) < 1.0

    def test_turnover_fold_upper_bound_fail_closed_when_span_too_short(
        self,
    ) -> None:
        # RELIAB-02: a single-fold (or shorter) span cannot define a worst-of-
        # several-folds statistic -- fail closed to +inf, never a finite value.
        too_short = pd.date_range("2022-01-01", periods=100, freq="4h", tz="UTC")
        assert (
            compute_turnover_fold_upper_bound(
                pd.Series(np.full(100, 0.01), index=too_short), 2190,
            )
            == float("inf")
        )
        # Single-fold span: the last bar sits exactly one fold-duration after
        # the first, so the helper admits exactly one fold -- still not enough
        # to define a worst-of-folds statistic.
        single_fold = pd.date_range("2022-01-01", periods=182, freq="D", tz="UTC")
        assert (
            compute_turnover_fold_upper_bound(
                pd.Series(np.full(182, 0.01), index=single_fold), 2190,
            )
            == float("inf")
        )
        # A single bar cannot define any fold statistic either.
        one_bar = pd.date_range("2022-01-01", periods=1, freq="4h", tz="UTC")
        assert (
            compute_turnover_fold_upper_bound(
                pd.Series([0.01], index=one_bar), 2190,
            )
            == float("inf")
        )

    def test_turnover_fold_upper_bound_is_deterministic(self) -> None:
        # RELIAB-03: identical input yields the exact same float every call.
        rng = np.random.default_rng(0)
        idx = pd.date_range("2022-01-01", periods=8760, freq="4h", tz="UTC")
        turnover = pd.Series(np.abs(rng.normal(0.02, 0.01, 8760)), index=idx)
        a = compute_turnover_fold_upper_bound(turnover, 2190)
        b = compute_turnover_fold_upper_bound(turnover, 2190)
        assert a == b
        with pytest.raises(ValueError, match="DatetimeIndex"):
            compute_turnover_fold_upper_bound(pd.Series([0.01, 0.02]), 2190)


@pytest.mark.slow
class TestFixedSleeveFoldThreshold:
    def test_fixed_sleeve_fold_threshold_is_unchanged(self) -> None:
        # SC-SGV2-08: the fold threshold stays frozen at 0.40, and the current
        # 5-sleeve annual ledger (measured max_period_contribution=0.489891)
        # remains a fold FAIL. The candidate cannot pass through a changed
        # threshold, only through genuinely distributed annual returns.
        from src.research.contracts import CostModel
        from src.research.sleeve_blend.fixed import (
            run_fixed_sleeve_portfolio_calibrated,
        )

        assert ReliabilityGateConfig().max_period_contribution == 0.40

        symbols = ("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT")
        result, _lev = run_fixed_sleeve_portfolio_calibrated(
            symbols, None, "2025-12-31 23:59:59", CostModel(), mdd_budget_fraction=0.85,
        )
        r = compute_fold_distribution(result)
        assert r.n_folds == 6
        assert abs(r.max_period_contribution - 0.489891) < 1e-3
        assert r.gate_pass is False


@pytest.mark.slow
class TestStressTestGate:
    def test_compute_stress_test_gate_matches_measured_v1_stress_survival(self) -> None:
        df = load_ohlcv_4h(BTC_PATH, end="2025-12-31 23:59:59")
        r = compute_stress_test_gate(df, StrategySpec(), CostModel())
        assert r.verdict == "PASS"
        assert r.point_cagr > 0
        assert r.trade_count > 0
