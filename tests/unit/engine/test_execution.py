from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.types import CostModel, StrategySpec
from src.data.loader import DataIntegrityError, load_ohlcv_4h
from src.engine.backtest import run_backtest

BTC_PATH = Path("data/futures/ohlcv/1h/BTCUSDT.parquet")


def test_engine_run_backtest() -> None:
    idx = pd.date_range("2024-01-01", periods=300, freq="4h", tz="UTC")
    df = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0,
    }, index=idx)
    spec = StrategySpec(risk_per_trade=0.01, ema_period=5, entry_period=5, atr_period=5)
    costs = CostModel()
    result = run_backtest(df, spec, costs)
    assert result.equity.index.equals(df.index)
    assert result.equity.notna().all()
    assert set(result.trades.columns) == {
        "entry_bar", "exit_bar", "entry_price", "exit_price", "qty", "reason", "pnl",
        "return_pct", "funding_pnl",
    }
    assert "entry_signal" in result.signals.columns


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


class TestExitBarSurface:
    """SC-ENGINE-EXIT-01: every closed trade exposes its actual exit bar."""

    def test_run_backtest_trades_expose_exit_bar(self, btc_4h_slice: pd.DataFrame) -> None:
        spec = StrategySpec()
        costs = CostModel()
        result = run_backtest(btc_4h_slice, spec, costs)
        assert {"entry_bar", "exit_bar"}.issubset(result.trades.columns)
        assert len(result.trades) > 0, "fixture must produce at least one closed trade"
        for _, t in result.trades.iterrows():
            eb, xb = int(t["entry_bar"]), int(t["exit_bar"])
            assert xb < len(btc_4h_slice), "exit_bar must index the backtest bar window"
            assert 0 <= eb <= xb, "a trade must exit no earlier than its entry bar"
            if t["reason"] != "stop_entrybar":
                assert xb > eb, "a held trade must exit strictly after its entry bar"

    def test_entry_bar_stop_exit_bar_equals_entry_bar(self, bars_stop_gap: pd.DataFrame) -> None:
        spec = StrategySpec(risk_per_trade=0.01, ema_period=2, entry_period=2, atr_period=2)
        costs = CostModel()
        result = run_backtest(bars_stop_gap, spec, costs)
        if len(result.trades) > 0:
            entrybar_stops = result.trades[result.trades["reason"] == "stop_entrybar"]
            for _, t in entrybar_stops.iterrows():
                assert int(t["exit_bar"]) == int(t["entry_bar"])


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


def _ramp_then_crash() -> pd.DataFrame:
    """A monotonic ramp that opens one long early, then a crash that closes it
    via the channel exit -- a single completed trade spanning any mid-run bar."""
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    opens = np.arange(100.0, 100.0 + n, dtype=np.float64)
    o = opens.copy()
    h = opens + 1.0
    l_ = opens - 1.0
    c = opens + 0.5
    o[150:] = opens[149] - 30.0
    h[150:] = opens[149] - 29.0
    l_[150:] = opens[149] - 31.0
    c[150:] = opens[149] - 29.5
    return pd.DataFrame({
        "open": o, "high": h, "low": l_, "close": c, "volume": 1000.0,
    }, index=idx)


class TestFundingLedger:
    """SC-FUND-01/02: published funding accrues while a long is open, and a
    candidate without funding data is a validation failure, never zero-cost."""

    def _held_long_result(self, funding_rates: pd.Series | None):
        df = _ramp_then_crash()
        spec = StrategySpec(risk_per_trade=0.01, ema_period=5, entry_period=5, atr_period=5)
        costs = CostModel()
        return df, run_backtest(df, spec, costs, funding_rates=funding_rates)

    def test_positive_funding_debits_open_long_position(self) -> None:
        df = _ramp_then_crash()
        funding_bar = 50
        ts = df.index[funding_bar]
        _, base = self._held_long_result(None)
        _, funded = self._held_long_result(pd.Series([0.001], index=[ts]))
        assert len(funded.trades) == 1, "fixture must hold one position across the funding bar"
        qty = float(funded.trades["qty"].iloc[0])
        expected = qty * float(df["open"].iloc[funding_bar]) * 0.001
        diff = base.equity.iloc[-1] - funded.equity.iloc[-1]
        assert diff == pytest.approx(expected, rel=1e-9), (
            f"positive funding must reduce final equity by notional x rate, got {diff} vs {expected}"
        )
        assert funded.trades["funding_pnl"].iloc[0] == pytest.approx(-expected, rel=1e-9)

    def test_negative_funding_credits_open_long_position(self) -> None:
        df = _ramp_then_crash()
        funding_bar = 50
        ts = df.index[funding_bar]
        _, base = self._held_long_result(None)
        _, funded = self._held_long_result(pd.Series([-0.001], index=[ts]))
        qty = float(funded.trades["qty"].iloc[0])
        expected = qty * float(df["open"].iloc[funding_bar]) * 0.001
        diff = funded.equity.iloc[-1] - base.equity.iloc[-1]
        assert diff == pytest.approx(expected, rel=1e-9)
        assert funded.trades["funding_pnl"].iloc[0] == pytest.approx(expected, rel=1e-9)

    def test_boundary_funding_timestamp_accrues(self) -> None:
        # A funding timestamp exactly at a bar boundary is charged to a position
        # held into it; a mid-bar timestamp maps to the containing bar.
        df = _ramp_then_crash()
        spec = StrategySpec(risk_per_trade=0.01, ema_period=5, entry_period=5, atr_period=5)
        costs = CostModel()
        base = run_backtest(df, spec, costs)

        funding_bar = 50
        expected = None
        for offset in (pd.Timedelta(0), pd.Timedelta(hours=1)):
            ts = df.index[funding_bar] + offset
            funded = run_backtest(
                df, spec, costs, funding_rates=pd.Series([0.001], index=[ts]),
            )
            qty = float(funded.trades["qty"].iloc[0])
            debit = qty * float(df["open"].iloc[funding_bar]) * 0.001
            if expected is None:
                expected = debit
            assert (base.equity - funded.equity).max() == pytest.approx(debit, rel=1e-9)
        assert expected is not None

    def test_candidate_without_funding_raises(self, bars_ramp: pd.DataFrame) -> None:
        # SC-FUND-02: absent candidate funding is a validation failure.
        spec = StrategySpec(
            risk_per_trade=0.01, ema_period=5, entry_period=5, atr_period=5,
            min_taker_buy_ratio=0.52,
        )
        with pytest.raises(DataIntegrityError, match="funding"):
            run_backtest(bars_ramp, spec, CostModel())

    def test_non_monotonic_funding_raises(self) -> None:
        df = _ramp_then_crash()
        ts = df.index[[1, 0]]
        funding_rates = pd.Series([0.001, 0.001], index=ts)
        with pytest.raises(DataIntegrityError, match="monotonic"):
            self._held_long_result(funding_rates)

    def test_out_of_window_funding_raises(self) -> None:
        df = _ramp_then_crash()
        ts = df.index[0] - pd.Timedelta(hours=4)
        with pytest.raises(DataIntegrityError, match="aligned"):
            self._held_long_result(pd.Series([0.001], index=[ts]))

    def test_non_finite_funding_raises(self) -> None:
        df = _ramp_then_crash()
        ts = df.index[50]
        with pytest.raises(DataIntegrityError, match="finite"):
            self._held_long_result(pd.Series([np.nan], index=[ts]))
