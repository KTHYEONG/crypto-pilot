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


def test_backtest_signature_is_frozen() -> None:
    from inspect import signature

    params = signature(run_technical_expert_backtest).parameters
    assert list(params) == [
        "frame", "candidate", "costs", "funding_rates",
        "initial_equity", "signal_delay_bars",
    ]
    assert params["signal_delay_bars"].default == 0
