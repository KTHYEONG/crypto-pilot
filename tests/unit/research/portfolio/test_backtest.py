from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.contracts import CostModel, PortfolioSpec, StrategySpec
from src.common.errors import DataIntegrityError
from src.research.portfolio.backtest import run_portfolio_backtest


def _breakout_frame(jump: float, signal_bar: int = 260, crash_bar: int = 275) -> pd.DataFrame:
    """Flat base that breaks out at ``signal_bar``, holds, then crashes at ``crash_bar``."""
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    l_ = np.full(n, 99.0)
    c = np.full(n, 100.0)
    c[signal_bar] = 100.0 + jump
    h[signal_bar] = 100.0 + jump + 1.0
    l_[signal_bar] = 100.0 + jump - 1.0
    o[signal_bar + 1:crash_bar] = 100.0 + jump
    h[signal_bar + 1:crash_bar] = 100.0 + jump + 1.0
    l_[signal_bar + 1:crash_bar] = 100.0 + jump - 1.0
    c[signal_bar + 1:crash_bar] = 100.0 + jump
    o[crash_bar:] = 90.0
    h[crash_bar:] = 91.0
    l_[crash_bar:] = 89.0
    c[crash_bar:] = 90.0
    return pd.DataFrame({
        "open": o, "high": h, "low": l_, "close": c, "quote_vol": 1000.0,
    }, index=idx)


def _funding(frame: pd.DataFrame, rates: dict[int, float] | None = None) -> pd.Series:
    """A non-empty, aligned funding series: zero everywhere unless ``rates`` override."""
    if rates is None:
        return pd.Series([0.0], index=[frame.index[0]])
    return pd.Series(list(rates.values()), index=[frame.index[k] for k in rates])


@pytest.fixture
def portfolio_spec() -> PortfolioSpec:
    return PortfolioSpec()


@pytest.fixture
def six_signal_portfolio() -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], StrategySpec, PortfolioSpec]:
    """Six symbols with distinct breakouts on the same bar; five slots."""
    jumps = {"A": 6.0, "B": 7.0, "C": 8.0, "D": 9.0, "E": 10.0, "F": 11.0}
    frames = {symbol: _breakout_frame(jump) for symbol, jump in jumps.items()}
    funding = {symbol: _funding(frame) for symbol, frame in frames.items()}
    spec = StrategySpec(risk_per_trade=0.005, ema_period=5, entry_period=5, atr_period=5)
    pspec = PortfolioSpec(universe_size=6, max_positions=5)
    return frames, funding, spec, pspec


class TestPortfolioExecution:
    def test_same_bar_entries_are_ranked_then_limited_to_five_slots(
        self, six_signal_portfolio: tuple[
            dict[str, pd.DataFrame], dict[str, pd.Series], StrategySpec, PortfolioSpec,
        ],
    ) -> None:
        # SC-PORT-03: the five largest ATR-normalized breakouts fill the slots on
        # the same bar; the weakest breakout is denied.
        frames, funding, spec, pspec = six_signal_portfolio
        result = run_portfolio_backtest(frames, funding, spec, pspec, CostModel())
        assert len(result.trades) == 5
        assert result.trades["entry_time"].nunique() == 1
        entered = set(result.trades["symbol"])
        assert entered == {"B", "C", "D", "E", "F"}
        assert "A" not in entered
        assert result.equity.notna().all()
        assert set(result.trades.columns) >= {
            "symbol", "entry_time", "exit_time", "portfolio_equity_before_entry", "return_pct",
        }

    def test_tie_break_by_symbol_is_deterministic(self) -> None:
        # SC-PORT-03: equal ATR-normalized strength resolves lexicographically.
        jumps = {"A": 6.0, "B": 8.0, "C": 8.0}
        frames = {symbol: _breakout_frame(jump) for symbol, jump in jumps.items()}
        funding = {symbol: _funding(frame) for symbol, frame in frames.items()}
        spec = StrategySpec(risk_per_trade=0.005, ema_period=5, entry_period=5, atr_period=5)
        pspec = PortfolioSpec(universe_size=3, max_positions=2)
        result = run_portfolio_backtest(frames, funding, spec, pspec, CostModel())
        entered = list(result.trades["symbol"])
        assert set(entered) == {"B", "C"}
        assert entered[0] == "B"

    def test_symbol_missing_funding_never_enters(
        self, six_signal_portfolio: tuple[
            dict[str, pd.DataFrame], dict[str, pd.Series], StrategySpec, PortfolioSpec,
        ],
    ) -> None:
        # SC-PORT-02: a symbol without an aligned funding series is ineligible; the
        # strongest signal without funding must never produce a zero-cost assumption.
        frames, funding, spec, pspec = six_signal_portfolio
        no_funding = {symbol: rates for symbol, rates in funding.items() if symbol != "F"}
        result = run_portfolio_backtest(frames, no_funding, spec, pspec, CostModel())
        assert "F" not in result.trades["symbol"].values
        assert len(result.trades) == 5

    def test_out_of_window_funding_raises(
        self, six_signal_portfolio: tuple[
            dict[str, pd.DataFrame], dict[str, pd.Series], StrategySpec, PortfolioSpec,
        ],
    ) -> None:
        frames, funding, spec, pspec = six_signal_portfolio
        bad = dict(funding)
        bad["A"] = pd.Series([0.001], index=[frames["A"].index[0] - pd.Timedelta(hours=4)])
        with pytest.raises(DataIntegrityError, match="aligned"):
            run_portfolio_backtest(frames, bad, spec, pspec, CostModel())

    def test_total_initial_risk_is_capped_by_total_equity(self) -> None:
        # SC-PORT-04: with 1% risk per trade, three same-bar signals fit only two
        # positions because sum(initial_risk) must never exceed 2.5% of pre-entry
        # total equity.
        jumps = {"A": 6.0, "B": 7.0, "C": 8.0}
        frames = {symbol: _breakout_frame(jump) for symbol, jump in jumps.items()}
        funding = {symbol: _funding(frame) for symbol, frame in frames.items()}
        spec = StrategySpec(risk_per_trade=0.01, ema_period=5, entry_period=5, atr_period=5)
        pspec = PortfolioSpec(universe_size=3, max_positions=3)
        result = run_portfolio_backtest(frames, funding, spec, pspec, CostModel())
        assert len(result.trades) == 2
        assert "A" not in result.trades["symbol"].values
        equity_before = result.trades["portfolio_equity_before_entry"].iloc[0]
        total_risk = float(result.trades["initial_risk"].sum())
        assert total_risk <= 0.025 * equity_before
        assert total_risk == pytest.approx(2 * 0.01 * equity_before, rel=1e-6)

    def test_positive_negative_funding_attributed_to_symbol_trades(self) -> None:
        # SC-PORT-05: positive funding debits the long and negative funding credits
        # it, each accrued into the correct symbol trade and the single ledger.
        jumps = {"A": 6.0, "B": 7.0}
        frames = {symbol: _breakout_frame(jump, crash_bar=280) for symbol, jump in jumps.items()}
        spec = StrategySpec(risk_per_trade=0.005, ema_period=5, entry_period=5, atr_period=5)
        pspec = PortfolioSpec(universe_size=2, max_positions=2)
        idx = frames["A"].index
        funding = {
            "A": _funding(frames["A"], {270: 0.001}),
            "B": _funding(frames["B"], {270: -0.001}),
        }
        result = run_portfolio_backtest(frames, funding, spec, pspec, CostModel())
        assert len(result.trades) == 2
        trade_a = result.trades[result.trades["symbol"] == "A"].iloc[0]
        trade_b = result.trades[result.trades["symbol"] == "B"].iloc[0]
        expected_a = -float(trade_a["qty"]) * frames["A"]["open"].iloc[270] * 0.001
        expected_b = float(trade_b["qty"]) * frames["B"]["open"].iloc[270] * 0.001
        assert trade_a["funding_pnl"] == pytest.approx(expected_a, rel=1e-9)
        assert trade_b["funding_pnl"] == pytest.approx(expected_b, rel=1e-9)

        base = run_portfolio_backtest(
            frames,
            {symbol: _funding(frame) for symbol, frame in frames.items()},
            spec, pspec, CostModel(),
        )
        assert result.equity.iloc[-1] - base.equity.iloc[-1] == pytest.approx(
            expected_a + expected_b, rel=1e-9,
        )
