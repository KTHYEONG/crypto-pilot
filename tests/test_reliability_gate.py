from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import CostModel, StrategySpec
from src.data.loader import load_ohlcv_4h
from src.engine import BacktestResult
from src.reliability_gate import (
    FoldDistributionResult,
    ReliabilityGateConfig,
    ReliabilityGateResult,
    compute_fold_distribution,
    compute_reliability_gate,
    compute_stress_test_gate,
    derive_block_size,
    split_holdout_segment,
)

BTC_PATH = Path("data/futures/ohlcv/1h/BTCUSDT.parquet")


class TestReliabilityGateConfig:
    def test_reliability_gate_config_defaults(self) -> None:
        c = ReliabilityGateConfig()
        assert (c.hurdle_rate, c.block_size, c.n_bootstrap, c.seed, c.min_trades,
                c.mdd_floor, c.t_stat_floor, c.max_period_contribution) == (
            0.15, None, 3000, 0, 30, -0.25, 2.0, 0.40,
        )

    def test_reliability_gate_result_fields(self) -> None:
        assert {f.name for f in dataclasses.fields(ReliabilityGateResult)} == {
            "lcb90_cagr", "lcb95_cagr", "p_negative", "point_cagr",
            "t_stat", "trade_count", "block_size_used", "verdict",
        }
        assert {f.name for f in dataclasses.fields(FoldDistributionResult)} == {
            "n_folds", "median_fold_cagr", "worst_fold_cagr",
            "median_fold_calmar", "max_period_contribution", "gate_pass",
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


class TestSplitHoldoutSegment:
    def _result(self, entry_bars: list[int]) -> BacktestResult:
        idx = pd.date_range("2025-06-01", periods=20, freq="4D", tz="UTC")
        equity = pd.Series(np.linspace(10000, 12000, 20), index=idx)
        trades = pd.DataFrame({
            "entry_bar": entry_bars,
            "pnl": [10.0] * len(entry_bars),
            "return_pct": [0.01] * len(entry_bars),
        })
        return BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame(index=idx))

    def test_split_holdout_segment_rebasing_and_entry_classification(self) -> None:
        cutoff = pd.Timestamp("2025-07-01", tz="UTC")
        # bar 5 -> 2025-06-21 (pre-cutoff), bar 12 -> 2025-07-17 (post-cutoff)
        seg = self._result([5, 12])
        seg = split_holdout_segment(seg, cutoff)

        assert abs(seg.holdout_equity.iloc[0] - 1.0) < 1e-9
        assert seg.observation_trades["entry_bar"].tolist() == [5]
        assert seg.holdout_trades["entry_bar"].tolist() == [12]
        assert seg.holdout_mdd <= 0.0
        assert seg.holdout_years > 0

    def test_trade_opened_pre_cutoff_classified_as_observation(self) -> None:
        cutoff = pd.Timestamp("2025-07-01", tz="UTC")
        # a trade entered at bar 5 (pre-cutoff) that exits far later is observation
        seg = split_holdout_segment(self._result([5]), cutoff)
        assert len(seg.observation_trades) == 1
        assert len(seg.holdout_trades) == 0

    def test_split_holdout_segment_validation(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            split_holdout_segment(
                self._result([5]), pd.Timestamp("2030-01-01", tz="UTC"),
            )
        with pytest.raises(ValueError, match="tz-naive"):
            split_holdout_segment(
                self._result([5]), pd.Timestamp("2025-07-01"),
            )


def _fold_fixture(pnls: list[float]) -> BacktestResult:
    idx = pd.date_range("2022-01-01", periods=4, freq="YS", tz="UTC")
    trades = pd.DataFrame({
        "entry_bar": [0, 1, 2, 3],
        "pnl": pnls,
        "return_pct": [0.0] * 4,
    })
    equity = pd.Series(
        [10000, 10000, 13000, 14000, 14200],
        index=pd.date_range("2022-01-01", periods=5, freq="YS", tz="UTC"),
    )
    return BacktestResult(equity=equity, trades=trades, signals=pd.DataFrame(index=idx))


class TestComputeFoldDistribution:
    def test_fold_distribution_concentration_gates(self) -> None:
        # SC-FOLD-01: one fold carrying ~76% of net pnl -> hard gate fails
        concentrated = compute_fold_distribution(_fold_fixture([-100.0, 1000.0, 200.0, 200.0]))
        assert abs(concentrated.max_period_contribution - (1000.0 / 1300.0)) < 1e-9
        assert concentrated.gate_pass is False

        # SC-FOLD-02: pnl evenly split -> 25% concentration -> gate passes
        even = compute_fold_distribution(_fold_fixture([250.0, 250.0, 250.0, 250.0]))
        assert abs(even.max_period_contribution - 0.25) < 1e-9
        assert even.gate_pass is True

    def test_compute_fold_distribution_matches_measured_v1_concentration(self) -> None:
        df = load_ohlcv_4h(BTC_PATH, end="2025-12-31 23:59:59")
        from src.engine import run_backtest

        spec, costs = StrategySpec(), CostModel()
        result = run_backtest(df, spec, costs)
        r = compute_fold_distribution(result)
        assert r.n_folds == 4
        assert abs(r.max_period_contribution - 0.7586) < 1e-3
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


@pytest.mark.slow
class TestStressTestGate:
    def test_compute_stress_test_gate_matches_measured_v1_stress_survival(self) -> None:
        df = load_ohlcv_4h(BTC_PATH, end="2025-12-31 23:59:59")
        r = compute_stress_test_gate(df, StrategySpec(), CostModel())
        assert r.verdict == "PASS"
        assert r.point_cagr > 0
        assert r.trade_count > 0
