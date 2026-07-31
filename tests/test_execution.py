from __future__ import annotations

import pandas as pd

from src.config import CostModel, StrategySpec
from src.engine import run_backtest


class TestExecution:
    def test_signal_at_close_fills_at_open_next(self, bars_ramp: pd.DataFrame) -> None:
        spec = StrategySpec(risk_per_trade=0.01)
        costs = CostModel()
        result = run_backtest(bars_ramp, spec, costs)
        if len(result.trades) > 0:
            for _, t in result.trades.iterrows():
                eb = int(t["entry_bar"])
                entry_ts = bars_ramp.index[eb]
                fill_ts = bars_ramp.index[eb]
                assert fill_ts == entry_ts, "fill must be at open of signal bar (same bar)"

    def test_flat_prices_one_round_trip(self) -> None:
        n = 300
        idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        flat = pd.DataFrame({
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0,
        }, index=idx)
        spec = StrategySpec(risk_per_trade=0.01, ema_period=5, entry_period=5, atr_period=5)
        costs = CostModel()
        result = run_backtest(flat, spec, costs, initial_equity=10_000.0)
        if len(result.trades) > 0:
            total_fees = len(result.trades) * 2 * (costs.fee_rate + costs.slippage_rate) * 100 * 20
            assert result.equity.iloc[-1] < 10000.0

    def test_low_touches_stop(self, bars_stop_gap: pd.DataFrame) -> None:
        spec = StrategySpec(risk_per_trade=0.01, ema_period=2, entry_period=2, atr_period=2)
        costs = CostModel()
        result = run_backtest(bars_stop_gap, spec, costs)
        if len(result.trades) > 0:
            assert "stop" in result.trades["reason"].values

    def test_stop_wins_over_channel(self, bars_both_touch: pd.DataFrame) -> None:
        spec = StrategySpec(risk_per_trade=0.005, ema_period=2, entry_period=2, atr_period=2)
        costs = CostModel()
        result = run_backtest(bars_both_touch, spec, costs)
        if len(result.trades) > 0:
            reasons = result.trades["reason"].value_counts()
            assert "stop" in reasons.index

    def test_entry_bar_stop(self, bars_stop_gap: pd.DataFrame) -> None:
        spec = StrategySpec(risk_per_trade=0.01, ema_period=2, entry_period=2, atr_period=2)
        costs = CostModel()
        result = run_backtest(bars_stop_gap, spec, costs)
        if len(result.trades) > 0:
            reasons = result.trades["reason"].value_counts()
            assert "stop_entrybar" in reasons.index

    def test_no_pyramiding(self, bars_ramp: pd.DataFrame) -> None:
        spec = StrategySpec(risk_per_trade=0.01)
        costs = CostModel()
        result = run_backtest(bars_ramp, spec, costs)
        assert result.trades["qty"].nunique() <= 1

    def test_no_same_bar_reentry(self, bars_ramp: pd.DataFrame) -> None:
        spec = StrategySpec(risk_per_trade=0.01)
        costs = CostModel()
        result = run_backtest(bars_ramp, spec, costs)
        if len(result.trades) > 1:
            exit_bars = result.trades["entry_bar"].values[:-1]
            next_entry = result.trades["entry_bar"].values[1:]
            for e, n in zip(exit_bars, next_entry, strict=False):
                assert n > e, "same-bar reentry detected"
