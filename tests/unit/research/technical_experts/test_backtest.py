from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.research.contracts import CostModel
from src.research.technical_experts.backtest import run_technical_expert_backtest
from src.research.technical_experts.catalog import resolve_technical_candidate

ZERO_COSTS = CostModel(fee_rate=0.0, slippage_rate=0.0)


def execution_fixture() -> pd.DataFrame:
    """Flat 220-bar 4h grid on which a crafted long->cash->short event stream runs."""
    index = pd.date_range("2024-01-01", periods=220, freq="4h", tz="UTC")
    return pd.DataFrame({
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1000.0,
    }, index=index)


def funding_fixture(frame: pd.DataFrame | None = None) -> pd.Series:
    """A single positive funding settlement inside the short's holding window."""
    grid = execution_fixture() if frame is None else frame
    return pd.Series([0.001], index=pd.DatetimeIndex([grid.index[4]]))


def _long_cash_short_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Decision stream: long entry (bar0), long exit (bar1), short entry (bar2), short exit (bar3)."""
    n = len(frame)
    data = {col: [False] * n for col in ("long_entry", "short_entry", "long_exit", "short_exit")}
    data["long_entry"][0] = True
    data["long_exit"][1] = True
    data["short_entry"][2] = True
    data["short_exit"][3] = True
    return pd.DataFrame(data, index=frame.index)


def _short_hold_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Decision stream that keeps a short open after the next-bar fill."""
    n = len(frame)
    data = {col: [False] * n for col in ("long_entry", "short_entry", "long_exit", "short_exit")}
    data["short_entry"][0] = True
    return pd.DataFrame(data, index=frame.index)


def _long_entry_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Decision stream: a single long entry at bar 0, held forever after."""
    n = len(frame)
    data = {col: [False] * n for col in ("long_entry", "short_entry", "long_exit", "short_exit")}
    data["long_entry"][0] = True
    return pd.DataFrame(data, index=frame.index)


def _long_window_events(
    frame: pd.DataFrame, entry_bar: int, exit_bar: int,
) -> pd.DataFrame:
    """Decision stream: one long entry and one long exit at explicit bars."""
    n = len(frame)
    data = {col: [False] * n for col in ("long_entry", "short_entry", "long_exit", "short_exit")}
    data["long_entry"][entry_bar] = True
    data["long_exit"][exit_bar] = True
    return pd.DataFrame(data, index=frame.index)


def _stop_fixture(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build a 210-bar fixture from crafted OHLC rows padded with flat bars."""
    n = 210
    columns = ("open", "high", "low", "close")
    data = {col: [0.0] * n for col in (*columns, "volume")}
    for idx, (o, h, low, c) in enumerate(rows):
        data["open"][idx] = o
        data["high"][idx] = h
        data["low"][idx] = low
        data["close"][idx] = c
    flat = (100.0, 101.0, 99.0, 100.0)
    for idx in range(len(rows), n):
        data["open"][idx], data["high"][idx], data["low"][idx], data["close"][idx] = flat
    data["volume"] = [1000.0] * n
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(data, index=index)


class TestTechnicalExpertBacktest:
    def test_target_transition_is_next_open_and_costed(self, monkeypatch) -> None:
        frame = execution_fixture()
        monkeypatch.setattr(
            "src.research.technical_experts.backtest.generate_signal_events",
            lambda f, candidate: _long_cash_short_events(f),
        )
        candidate = resolve_technical_candidate("technical_macd_histogram_regime_long_v1")
        zero = run_technical_expert_backtest(frame, candidate, ZERO_COSTS, funding_fixture(frame))
        costed = run_technical_expert_backtest(frame, candidate, CostModel(), funding_fixture(frame))

        # Each decision at close[t] is filled at open[t+1].
        assert costed.equity.name == "equity"
        assert costed.signals["target"].iloc[0] == 1
        assert costed.signals["target"].iloc[1] == 0
        assert costed.signals["target"].iloc[2] == -1
        assert len(costed.trades) == 2
        assert costed.trades.iloc[0]["entry_bar"] == 1
        assert costed.trades.iloc[0]["exit_bar"] == 2
        assert costed.trades.iloc[1]["entry_bar"] == 3
        assert costed.trades.iloc[1]["exit_bar"] == 4
        # The short leg receives positive funding while held at settlement.
        assert costed.trades.iloc[1]["side"] == "short"
        assert costed.trades.iloc[1]["funding_pnl"] > 0.0
        # Fee/slippage is charged once per transition.
        assert costed.equity.iloc[-1] < zero.equity.iloc[-1]

    def test_signal_delay_shifts_every_fill(self, monkeypatch) -> None:
        frame = execution_fixture()
        monkeypatch.setattr(
            "src.research.technical_experts.backtest.generate_signal_events",
            lambda f, candidate: _long_cash_short_events(f),
        )
        candidate = resolve_technical_candidate("technical_macd_histogram_regime_long_v1")
        base = run_technical_expert_backtest(frame, candidate, ZERO_COSTS, funding_fixture(frame))
        delayed = run_technical_expert_backtest(
            frame, candidate, ZERO_COSTS, funding_fixture(frame), signal_delay_bars=1,
        )
        assert base.trades.iloc[0]["entry_bar"] == 1
        assert delayed.trades.iloc[0]["entry_bar"] == 2

    def test_no_events_leaves_cash(self) -> None:
        frame = execution_fixture()
        candidate = resolve_technical_candidate("technical_macd_histogram_regime_long_v1")
        result = run_technical_expert_backtest(frame, candidate, CostModel(), funding_fixture(frame))
        assert len(result.trades) == 0
        assert np.allclose(result.equity.to_numpy(), 10_000.0)

    def test_rejects_negative_delay_and_nonpositive_equity(self) -> None:
        frame = execution_fixture()
        candidate = resolve_technical_candidate("technical_macd_histogram_regime_long_v1")
        with pytest.raises(ValueError, match="signal_delay_bars"):
            run_technical_expert_backtest(
                frame, candidate, CostModel(), funding_fixture(frame), signal_delay_bars=-1,
            )
        with pytest.raises(ValueError, match="initial_equity"):
            run_technical_expert_backtest(
                frame, candidate, CostModel(), funding_fixture(frame), initial_equity=0.0,
            )

    def test_missing_funding_fails_closed(self) -> None:
        frame = execution_fixture()
        candidate = resolve_technical_candidate("technical_macd_histogram_regime_long_v1")
        with pytest.raises(DataIntegrityError, match="funding"):
            run_technical_expert_backtest(frame, candidate, CostModel(), pd.Series(dtype=float))

    def test_short_equity_exhaustion_fails_closed(self, monkeypatch) -> None:
        frame = execution_fixture()
        frame.loc[frame.index[2], "close"] = 250.0
        monkeypatch.setattr(
            "src.research.technical_experts.backtest.generate_signal_events",
            lambda f, candidate: _short_hold_events(f),
        )
        candidate = resolve_technical_candidate("technical_macd_histogram_regime_long_v1")

        with pytest.raises(DataIntegrityError, match="equity exhausted"):
            run_technical_expert_backtest(frame, candidate, ZERO_COSTS, funding_fixture(frame))


class TestStopLossEngine:
    def test_ter_01_default_none_is_byte_identical(self, monkeypatch) -> None:
        """TER-01-DEFAULT-NONE-BYTE-IDENTICAL: explicit defaults match the no-arg call."""
        frame = execution_fixture()
        monkeypatch.setattr(
            "src.research.technical_experts.backtest.generate_signal_events",
            lambda f, candidate: _long_cash_short_events(f),
        )
        candidate = resolve_technical_candidate("technical_macd_histogram_regime_long_v1")
        base = run_technical_expert_backtest(
            frame, candidate, ZERO_COSTS, funding_fixture(frame), signal_delay_bars=1,
        )
        explicit = run_technical_expert_backtest(
            frame, candidate, ZERO_COSTS, funding_fixture(frame), signal_delay_bars=1,
            stop_loss_mode=None, stop_loss_value=None, atr_period=14, trailing_stop=False,
        )
        pd.testing.assert_series_equal(base.equity, explicit.equity)
        pd.testing.assert_frame_equal(base.trades, explicit.trades)

    def test_ter_02_fixed_pct_static_long_triggers_on_low_breach(self, monkeypatch) -> None:
        """TER-02-FIXED-PCT-STATIC-LONG-STOP-TRIGGERS-ON-LOW-BREACH: static fixed-% stop."""
        frame = _stop_fixture([
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 94.0, 100.0),
        ])
        monkeypatch.setattr(
            "src.research.technical_experts.backtest.generate_signal_events",
            lambda f, candidate: _long_entry_events(f),
        )
        candidate = resolve_technical_candidate("technical_macd_histogram_regime_long_v1")
        result = run_technical_expert_backtest(
            frame, candidate, ZERO_COSTS, funding_fixture(frame),
            stop_loss_mode="fixed_pct", stop_loss_value=0.05,
        )
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["entry_bar"] == 1
        assert trade["entry_price"] == 100.0
        assert trade["exit_bar"] == 2
        assert trade["reason"] == "stop_loss"
        assert trade["side"] == "long"
        assert trade["exit_price"] == 95.0

    def test_ter_03_fixed_pct_trailing_long_ratchets_up(self, monkeypatch) -> None:
        """TER-03-FIXED-PCT-TRAILING-LONG-STOP-RATCHETS-UP: trailing stop raises the stop price."""
        frame = _stop_fixture([
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 110.0, 108.0, 109.0),
            (109.0, 110.0, 104.0, 105.0),
        ])
        monkeypatch.setattr(
            "src.research.technical_experts.backtest.generate_signal_events",
            lambda f, candidate: _long_window_events(f, entry_bar=0, exit_bar=2),
        )
        candidate = resolve_technical_candidate("technical_macd_histogram_regime_long_v1")
        result = run_technical_expert_backtest(
            frame, candidate, ZERO_COSTS, funding_fixture(frame),
            stop_loss_mode="fixed_pct", stop_loss_value=0.05, trailing_stop=True,
        )
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["reason"] == "stop_loss"
        assert trade["exit_bar"] == 3
        # The static entry-anchored stop (95.0) never fired here (low 104 > 95);
        # the exit at the ratcheted 105.0 proves the trailing stop moved up.
        assert trade["exit_price"] == 105.0

    def test_ter_04_atr_multiple_stop_uses_causal_atr(self, monkeypatch) -> None:
        """TER-04-ATR-MULTIPLE-STOP-USES-CAUSAL-ATR: stop distance uses the shifted, causal ATR."""
        rows = [
            (100.0, 110.0, 90.0, 105.0),
            (105.0, 115.0, 100.0, 110.0),
            (110.0, 120.0, 105.0, 115.0),
            (110.0, 115.0, 108.0, 112.0),
            (112.0, 115.0, 70.0, 80.0),
        ]
        frame = _stop_fixture(rows)
        monkeypatch.setattr(
            "src.research.technical_experts.backtest.generate_signal_events",
            lambda f, candidate: _long_window_events(f, entry_bar=2, exit_bar=3),
        )
        candidate = resolve_technical_candidate("technical_macd_histogram_regime_long_v1")
        result = run_technical_expert_backtest(
            frame, candidate, ZERO_COSTS, funding_fixture(frame),
            stop_loss_mode="atr_multiple", stop_loss_value=2.0, atr_period=2,
        )
        # Reference causal ATR(2): TR = [20, 15, 15, 7]; the rolling(2) window
        # at bar 2 is mean(TR[1], TR[2]) = 15, and the entry bar 3 uses that
        # ATR[2] (shifted by one bar -> no same-bar lookahead, and bar 3's own
        # TR=7 is excluded).
        expected_atr = (15.0 + 15.0) / 2.0
        expected_stop_distance = 2.0 * expected_atr
        expected_exit = 110.0 - expected_stop_distance
        assert len(result.trades) == 1
        trade = result.trades.iloc[0]
        assert trade["entry_bar"] == 3
        assert trade["entry_price"] == 110.0
        assert trade["reason"] == "stop_loss"
        assert trade["exit_bar"] == 4
        assert np.isclose(trade["exit_price"], expected_exit)

    def test_ter_05_invalid_stop_loss_params_fail_closed(self, monkeypatch) -> None:
        """TER-05-INVALID-STOP-LOSS-PARAMS-FAIL-CLOSED: bad values raise; mode=None accepts any combo."""
        frame = execution_fixture()
        monkeypatch.setattr(
            "src.research.technical_experts.backtest.generate_signal_events",
            lambda f, candidate: _long_cash_short_events(f),
        )
        candidate = resolve_technical_candidate("technical_macd_histogram_regime_long_v1")
        with pytest.raises(ValueError, match="stop_loss_value"):
            run_technical_expert_backtest(
                frame, candidate, ZERO_COSTS, funding_fixture(frame),
                stop_loss_mode="fixed_pct", stop_loss_value=None,
            )
        with pytest.raises(ValueError, match="stop_loss_value"):
            run_technical_expert_backtest(
                frame, candidate, ZERO_COSTS, funding_fixture(frame),
                stop_loss_mode="fixed_pct", stop_loss_value=1.0,
            )
        with pytest.raises(ValueError, match="atr_period"):
            run_technical_expert_backtest(
                frame, candidate, ZERO_COSTS, funding_fixture(frame), atr_period=0,
            )
        # The master switch (None) accepts any stop_loss_value/atr_period/trailing_stop combo.
        accepted = run_technical_expert_backtest(
            frame, candidate, ZERO_COSTS, funding_fixture(frame),
            stop_loss_mode=None, stop_loss_value=0.05, atr_period=1, trailing_stop=True,
        )
        assert len(accepted.trades) == 2


def test_backtest_signature_is_frozen() -> None:
    from inspect import signature

    params = signature(run_technical_expert_backtest).parameters
    assert list(params) == [
        "frame", "candidate", "costs", "funding_rates",
        "initial_equity", "signal_delay_bars",
        "stop_loss_mode", "stop_loss_value", "atr_period", "trailing_stop",
    ]
    assert params["signal_delay_bars"].default == 0
    assert params["stop_loss_mode"].default is None
    assert params["stop_loss_value"].default is None
    assert params["atr_period"].default == 14
    assert params["trailing_stop"].default is False
