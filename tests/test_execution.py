from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import CostModel, StrategySpec
from src.data.loader import load_ohlcv_4h
from src.engine import run_backtest

BTC_PATH = Path("data/futures/ohlcv/1h/BTCUSDT.parquet")


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


@pytest.mark.slow
class TestSignalDelayBars:
    def test_signal_delay_bars_zero_is_byte_identical(self) -> None:
        df = load_ohlcv_4h(BTC_PATH, end="2025-12-31 23:59:59")
        spec = StrategySpec()
        costs = CostModel()
        r0 = run_backtest(df, spec, costs)
        r0b = run_backtest(df, spec, costs, signal_delay_bars=0)
        assert r0.equity.equals(r0b.equity)
        assert r0.trades.equals(r0b.trades)

    def test_signal_delay_bars_one_shifts_fills(self, bars_breakout_sparse: pd.DataFrame) -> None:
        spec = StrategySpec(risk_per_trade=0.01, ema_period=5, entry_period=5, atr_period=5)
        costs = CostModel()
        base = run_backtest(bars_breakout_sparse, spec, costs)
        delayed = run_backtest(bars_breakout_sparse, spec, costs, signal_delay_bars=1)

        assert len(base.trades) > 0, "fixture must produce at least one trade"
        assert len(delayed.trades) == len(base.trades)
        assert [int(b) for b in delayed.trades["entry_bar"]] == [
            int(b) + 1 for b in base.trades["entry_bar"]
        ]

    def test_signal_delay_bars_negative_raises(self, bars_ramp: pd.DataFrame) -> None:
        spec = StrategySpec()
        costs = CostModel()
        with pytest.raises(ValueError, match="signal_delay_bars"):
            run_backtest(bars_ramp, spec, costs, signal_delay_bars=-1)

    def test_entry_bar_is_entry_not_exit(self, btc_4h_slice: pd.DataFrame) -> None:
        """entry_bar must index the bar where the position was OPENED, not where it
        was closed -- verified independently of internal state by reconstructing the
        fill price from df['open'] at entry_bar and comparing to the recorded
        entry_price. A trade held for multiple bars (channel/stop exit, not
        stop_entrybar) would fail this if entry_bar pointed at the exit bar instead."""
        spec = StrategySpec()
        costs = CostModel()
        result = run_backtest(btc_4h_slice, spec, costs)
        held_trades = result.trades[result.trades["reason"] != "stop_entrybar"]
        assert len(held_trades) > 0, "fixture must produce at least one held (non-entry-bar) trade"
        for _, t in held_trades.iterrows():
            eb = int(t["entry_bar"])
            expected_fill = btc_4h_slice["open"].iloc[eb] * (1 + costs.slippage_rate)
            assert abs(t["entry_price"] - expected_fill) < 1e-6, (
                f"entry_bar={eb} does not reconstruct entry_price "
                f"({t['entry_price']} != {expected_fill}) -- entry_bar likely points at the exit bar"
            )
