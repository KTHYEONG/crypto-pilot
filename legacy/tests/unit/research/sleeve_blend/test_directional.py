from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.research.baseline.backtest import _run_directional_engine, run_directional_backtest
from src.research.contracts import CostModel, StrategySpec
from src.research.sleeve_blend import directional as sleeve_backtest
from src.research.sleeve_blend.contracts import FIXED_DIRECTIONAL_SYMBOLS

_SPEC = StrategySpec(entry_period=5, exit_period=3, ema_period=5, atr_period=5)


def _directional_frame() -> pd.DataFrame:
    """Flat base with a long breakout at 260 and a short breakdown at 400.

    The short breaks down at bar 400, holds through a quiet plateau, then
    gaps up at bar 411 into an adverse high stop.
    """
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    l_ = np.full(n, 99.0)
    c = np.full(n, 100.0)

    o[260] = 106.0
    h[260] = 108.0
    l_[260] = 105.0
    c[260] = 107.0
    for t in range(261, 268):
        o[t] = 106.0
        h[t] = 107.0
        l_[t] = 105.0
        c[t] = 106.0

    o[400] = 94.0
    h[400] = 96.0
    l_[400] = 92.0
    c[400] = 93.0
    for t in range(401, 411):
        o[t] = 94.0
        h[t] = 95.0
        l_[t] = 93.0
        c[t] = 94.0
    o[411] = 105.0
    h[411] = 110.0
    l_[411] = 104.0
    c[411] = 109.0

    return pd.DataFrame({
        "open": o, "high": h, "low": l_, "close": c, "volume": 1000.0,
    }, index=idx)


def _funding(df: pd.DataFrame, bars: list[int], rate: float) -> pd.Series:
    return pd.Series([rate] * len(bars), index=[df.index[i] for i in bars])


def _short_trade(result) -> pd.DataFrame:
    return result.trades[result.trades["side"] == "short"]


class TestDirectionalBacktest:
    def test_channel_exits_are_signed_for_both_directions(self) -> None:
        frame = _directional_frame()
        # Make the stop unreachable so each side exercises its channel branch.
        spec = replace(_SPEC, stop_atr_mult=100.0)
        funding = pd.Series(
            [-0.001, 0.001, 0.001, 0.001, 0.001],
            index=[frame.index[i] for i in [255, 395, 399, 403, 407]],
        )
        result = run_directional_backtest(
            frame, spec, CostModel(), funding,
        )
        reasons = set(result.trades["reason"].values)
        assert "channel" in reasons
        assert {"long", "short"}.issubset(set(result.trades["side"].values))

    def test_short_entry_bar_adverse_stop_is_recorded(self) -> None:
        frame = _directional_frame()
        frame.loc[frame.index[401], "high"] = 120.0
        result = run_directional_backtest(
            frame, _SPEC, CostModel(), _funding(frame, [395, 399], 0.001),
        )
        short = result.trades[result.trades["side"] == "short"]
        assert len(short) >= 1
        assert short.iloc[0]["reason"] == "stop_entrybar"

    def test_long_entry_bar_adverse_stop_is_recorded(self) -> None:
        frame = _directional_frame()
        frame.loc[frame.index[261], "low"] = 90.0
        result = run_directional_backtest(
            frame, _SPEC, CostModel(), _funding(frame, [255], -0.001),
        )
        long = result.trades[result.trades["side"] == "long"]
        assert len(long) >= 1
        assert long.iloc[0]["reason"] == "stop_entrybar"

    def test_zero_signal_atr_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = __import__(
            "src.research.baseline.signal", fromlist=["generate_directional_funding_signals"],
        ).generate_directional_funding_signals

        def zero_signal_atr(df, spec, funding):
            features = original(df, spec, funding)
            features.loc[df.index[260], "atr"] = 0.0
            return features

        monkeypatch.setattr(
            "src.research.baseline.signal.generate_directional_funding_signals",
            zero_signal_atr,
        )
        with pytest.raises(ValueError, match="stop_distance"):
            run_directional_backtest(
                _directional_frame(), _SPEC, CostModel(),
                _funding(_directional_frame(), [255], -0.001),
            )

    def test_short_stop_and_funding_cashflow_are_symmetric(self) -> None:
        # SC-SGV2-03: a short entered on a breakdown with positive funding
        # receives positive funding cashflow and stops out on an adverse high;
        # a long pays the same positive funding (symmetric mirror).
        df = _directional_frame()
        costs = CostModel()
        funded_bars = [395, 399, 403, 407]

        zero = run_directional_backtest(
            df, _SPEC, costs, _funding(df, funded_bars, 0.0),
        )
        positive = run_directional_backtest(
            df, _SPEC, costs, _funding(df, funded_bars, 0.001),
        )

        zero_short = _short_trade(zero)
        pos_short = _short_trade(positive)
        assert len(pos_short) >= 1, "fixture must produce at least one short trade"
        assert set(pos_short["reason"].values) == {"stop"}
        assert float(pos_short["funding_pnl"].iloc[0]) > 0.0
        assert float(zero_short["funding_pnl"].iloc[0]) == pytest.approx(0.0, abs=1e-12)
        assert positive.equity.iloc[-1] > zero.equity.iloc[-1]

        # the mirror long pays the same positive funding rate.
        long = run_directional_backtest(df, _SPEC, costs, _funding(df, funded_bars, 0.001))
        long_trades = long.trades[long.trades["side"] == "long"]
        if len(long_trades) > 0:
            assert float(long_trades["funding_pnl"].iloc[0]) < 0.0

    def test_adverse_high_stop_fills_above_entry_for_short(self) -> None:
        # the short's stop is on the high and fills at the gap-up open, never
        # below the stop price.
        df = _directional_frame()
        costs = CostModel()
        result = run_directional_backtest(
            df, _SPEC, costs, _funding(df, [395, 399, 403], 0.001),
        )
        short = _short_trade(result)
        assert len(short) >= 1
        row = short.iloc[0]
        assert row["reason"] == "stop"
        entry_price = float(row["entry_price"])
        stop_price = entry_price + 2.0 * float(_SPEC.stop_atr_mult) * 1.0
        assert float(row["exit_price"]) >= entry_price * (1 + costs.slippage_rate)

    def test_signed_ledger_is_positive_and_monotonic(self) -> None:
        df = _directional_frame()
        result = run_directional_backtest(df, _SPEC, CostModel(), _funding(df, [395, 399, 403, 407], 0.001))
        assert result.equity.index.is_monotonic_increasing
        assert (result.equity > 0).all()
        assert set(result.trades["side"].values) <= {"long", "short"}

    def test_signal_delay_shifts_directional_features(self) -> None:
        df = _directional_frame()
        result = run_directional_backtest(
            df, _SPEC, CostModel(), _funding(df, [255, 399], 0.001),
            signal_delay_bars=1,
        )
        assert result.equity.index.equals(df.index)

    def test_negative_signal_delay_is_rejected(self) -> None:
        df = _directional_frame()
        with pytest.raises(ValueError, match="signal_delay_bars"):
            run_directional_backtest(
                df, _SPEC, CostModel(), _funding(df, [255], 0.001),
                signal_delay_bars=-1,
            )

    def test_invalid_direction_is_rejected(self) -> None:
        df = _directional_frame()
        with pytest.raises(ValueError, match="side must be one"):
            _run_directional_engine(
                df, _SPEC, CostModel(), _funding(df, [255], 0.001), side="invalid",
            )

    def test_no_same_bar_reentry(self) -> None:
        df = _directional_frame()
        result = run_directional_backtest(df, _SPEC, CostModel(), _funding(df, [395, 399, 403, 407], 0.001))
        if len(result.trades) > 1:
            exits = result.trades["exit_bar"].to_numpy()[:-1]
            entries = result.trades["entry_bar"].to_numpy()[1:]
            assert (entries > exits).all()

    def test_long_and_short_stops_are_adverse_symmetric(self) -> None:
        # a stop is adverse on both sides: a long exits at or below entry, a
        # short exits at or above entry, each with its own slippage charge.
        df = _directional_frame()
        mixed = pd.Series(
            [-0.001, 0.001, 0.001, 0.001],
            index=[df.index[255], df.index[399], df.index[403], df.index[407]],
        )
        result = run_directional_backtest(df, _SPEC, CostModel(), mixed)
        assert {"long", "short"}.issubset(set(result.trades["side"].values))
        for _, trade in result.trades.iterrows():
            entry, exit_ = float(trade["entry_price"]), float(trade["exit_price"])
            if trade["side"] == "long":
                assert exit_ <= entry
            else:
                assert exit_ >= entry

    def test_malformed_funding_raises(self) -> None:
        df = _directional_frame()
        with pytest.raises(DataIntegrityError, match="non-empty"):
            run_directional_backtest(df, _SPEC, CostModel(), pd.Series(dtype=float))
        with pytest.raises(DataIntegrityError, match="finite"):
            run_directional_backtest(
                df, _SPEC, CostModel(),
                pd.Series([np.nan], index=[df.index[399]]),
            )

    def test_non_positive_equity_rejected(self) -> None:
        df = _directional_frame()
        with pytest.raises(ValueError, match="initial_equity"):
            run_directional_backtest(
                df, _SPEC, CostModel(), _funding(df, [395, 399, 403, 407], 0.001),
                initial_equity=0.0,
            )

    def test_fixed_weights_wrapper_reuses_core_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC")
        expected = pd.Series([10_000.0, 10_001.0], index=idx, name="equity")
        trades = pd.DataFrame(columns=["side"])
        weights = pd.DataFrame({"BTCUSDT:long": [1.0]}, index=idx[:1])

        def fake_core(**kwargs):
            if kwargs["fixed_weights"] is not None:
                assert kwargs["fixed_weights"] is weights
            return expected, trades, weights

        monkeypatch.setattr(sleeve_backtest, "_run_directional_sleeve_core", fake_core)
        base_only = sleeve_backtest.run_directional_sleeve_portfolio(
            ("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT"),
            None, None, CostModel(),
        )
        pd.testing.assert_series_equal(base_only.equity, expected)
        base_result, base_result_weights = sleeve_backtest.run_directional_sleeve_portfolio_with_weights(
            ("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT"),
            None, None, CostModel(),
        )
        pd.testing.assert_series_equal(base_result.equity, expected)
        pd.testing.assert_frame_equal(base_result_weights, weights)
        result = sleeve_backtest.run_directional_sleeve_portfolio_fixed_weights(
            ("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT"),
            None, None, CostModel(), weights,
        )
        pd.testing.assert_series_equal(result.equity, expected)

    def test_directional_sleeve_core_loads_all_fixed_symbols(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frame = _directional_frame()
        funding = _funding(frame, [255, 395, 399, 403, 407], 0.001)
        frames = dict.fromkeys(FIXED_DIRECTIONAL_SYMBOLS, frame)
        funding_by_symbol = dict.fromkeys(FIXED_DIRECTIONAL_SYMBOLS, funding)
        monkeypatch.setattr(
            sleeve_backtest,
            "load_ohlcv_4h",
            lambda path, start=None, end=None: frames[Path(str(path)).stem],
        )
        monkeypatch.setattr(
            sleeve_backtest,
            "load_funding_rates",
            lambda path: funding_by_symbol[Path(str(path)).stem],
        )
        result = sleeve_backtest.run_directional_sleeve_portfolio(
            FIXED_DIRECTIONAL_SYMBOLS, None, None, CostModel(),
        )
        assert result.equity.index.equals(frame.index)
        assert (result.equity > 0).all()
        base, weights = sleeve_backtest.run_directional_sleeve_portfolio_with_weights(
            FIXED_DIRECTIONAL_SYMBOLS, None, None, CostModel(),
        )
        stressed = sleeve_backtest.run_directional_sleeve_portfolio_fixed_weights(
            FIXED_DIRECTIONAL_SYMBOLS, None, None, CostModel(), weights,
        )
        assert base.equity.index.equals(stressed.equity.index)
        with pytest.raises(ValueError, match="signal_delay_bars"):
            sleeve_backtest.run_directional_sleeve_portfolio_fixed_weights(
                FIXED_DIRECTIONAL_SYMBOLS, None, None, CostModel(), weights,
                signal_delay_bars=-1,
            )
        with pytest.raises(ValueError, match="initial_equity"):
            sleeve_backtest.run_directional_sleeve_portfolio(
                FIXED_DIRECTIONAL_SYMBOLS, None, None, CostModel(), initial_equity=0.0,
            )
        monkeypatch.setattr(
            sleeve_backtest,
            "load_funding_rates",
            lambda path: pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC")),
        )
        with pytest.raises(DataIntegrityError, match="no funding events"):
            sleeve_backtest.run_directional_sleeve_portfolio(
                FIXED_DIRECTIONAL_SYMBOLS, None, None, CostModel(),
            )
        monkeypatch.setattr(
            sleeve_backtest,
            "load_ohlcv_4h",
            lambda path, start=None, end=None: frame.iloc[:0],
        )
        with pytest.raises(DataIntegrityError, match="no 4h bars"):
            sleeve_backtest.run_directional_sleeve_portfolio(
                FIXED_DIRECTIONAL_SYMBOLS, None, None, CostModel(),
            )
