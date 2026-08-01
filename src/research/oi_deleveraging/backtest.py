from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.research.baseline.backtest import BacktestResult, _align_funding_rates
from src.research.contracts import CostModel
from src.research.oi_deleveraging.contracts import OIDeleveragingMarketData
from src.research.oi_deleveraging.market_data import validate_oi_deleveraging_market_data

_logger = logging.getLogger("OIDeleveragingBacktest")

_INITIAL_EQUITY = 10_000.0

_TRADE_COLUMNS = (
    "entry_bar",
    "exit_bar",
    "entry_price",
    "exit_price",
    "qty",
    "reason",
    "pnl",
    "return_pct",
    "funding_pnl",
    "side",
)


def run_open_interest_deleveraging_screen(
    market_data: OIDeleveragingMarketData,
    costs: CostModel,
    signal_delay_bars: int = 1,
) -> BacktestResult:
    """Run the fixed-sign, one-day open-interest deleveraging screen.

    At each 4h decision timestamp a symbol is short during the next bar only
    when its completed 24h mark return is non-positive AND its completed daily
    ``sum_open_interest_value`` change is non-positive; every other state is
    cash. The decision is formed from the as-of joined metrics and executed at
    the next bar's open (plus ``signal_delay_bars``), charging the existing
    fee/slippage model and crediting funding to the short leg. A missing metric
    yields no signal, never an imputed feature. No parameter is optimized and
    no catalog mutation occurs.
    """
    validate_oi_deleveraging_market_data(market_data)
    if signal_delay_bars < 0:
        raise ValueError(f"signal_delay_bars must be >= 0, got {signal_delay_bars}")

    grid = market_data.bars.index
    n = len(grid)
    period = grid[1] - grid[0]
    window_end = grid[-1] + period
    funding_scope = market_data.funding[
        (market_data.funding.index >= grid[0]) & (market_data.funding.index < window_end)
    ]
    bar_funding = _align_funding_rates(funding_scope, grid)

    open_arr = pd.to_numeric(market_data.bars["open"], errors="coerce").to_numpy(dtype=np.float64)
    close_arr = pd.to_numeric(market_data.bars["close"], errors="coerce").to_numpy(dtype=np.float64)
    mark_return = market_data.joined["mark_return_24h"].to_numpy(dtype=np.float64)
    oi_change = market_data.joined["feature_oi_value_change"].to_numpy(dtype=np.float64)
    signal_short = (mark_return <= 0.0) & (oi_change <= 0.0)

    equity_arr = np.full(n, np.nan, dtype=np.float64)
    cash = _INITIAL_EQUITY
    position_qty = 0.0
    side_sign = 0  # 0 flat, -1 short
    entry_price = 0.0
    entry_bar_idx = -1
    trade_funding_pnl = 0.0
    trade_rows: list[dict[str, object]] = []

    # Each completed decision timestamp maps deterministically to one future
    # execution bar, so the target schedule is precomputed instead of holding a
    # single overwritten pending slot.
    schedule: list[str | None] = [None] * n
    for t in range(n):
        target = "SHORT" if bool(signal_short[t]) else "CASH"
        apply_bar = t + 1 + signal_delay_bars
        if apply_bar < n:
            schedule[apply_bar] = target

    def _close(reason: str, exit_price: float, t: int) -> None:
        nonlocal cash, position_qty, side_sign, trade_funding_pnl
        pnl = position_qty * side_sign * (exit_price - entry_price)
        exit_fee = costs.fee_rate * abs(position_qty * exit_price)
        cash -= position_qty * exit_price + exit_fee
        pnl -= exit_fee
        pnl += trade_funding_pnl
        trade_rows.append({
            "entry_bar": entry_bar_idx,
            "exit_bar": t,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "qty": position_qty,
            "reason": reason,
            "pnl": pnl,
            "return_pct": pnl / (cash - pnl) if cash != pnl else 0.0,
            "funding_pnl": trade_funding_pnl,
            "side": "short",
        })
        _logger.info(
            "bar=%d reason=%s entry=%.4f exit=%.4f qty=%.6f pnl=%.4f",
            t, reason, entry_price, exit_price, position_qty, pnl,
        )
        position_qty = 0.0
        side_sign = 0
        trade_funding_pnl = 0.0

    for t in range(n):
        o = open_arr[t]
        c = close_arr[t]

        # funding accrual for a short held into this bar: positive funding credits.
        if position_qty > 0 and bar_funding[t] != 0.0:
            funding_pnl = -side_sign * position_qty * o * bar_funding[t]
            cash += funding_pnl
            trade_funding_pnl += funding_pnl

        # apply the scheduled next-bar target at this bar's open.
        scheduled = schedule[t]
        if scheduled is not None:
            if scheduled == "SHORT" and position_qty == 0:
                fill = o * (1 - costs.slippage_rate)
                if fill > 0:
                    qty = cash / fill
                    if qty > 0:
                        entry_fee = costs.fee_rate * qty * fill
                        cash = cash + qty * fill - entry_fee
                        position_qty = qty
                        side_sign = -1
                        entry_price = fill
                        entry_bar_idx = t
                        trade_funding_pnl = 0.0
                        _logger.info(
                            "bar=%d SHORT fill=%.4f qty=%.6f", t, fill, qty,
                        )
            elif scheduled == "CASH" and position_qty > 0:
                _close("signal", o * (1 + costs.slippage_rate), t)

        # mark total equity at the bar close.
        equity_arr[t] = cash + side_sign * position_qty * c if position_qty > 0 else cash

    equity_series = pd.Series(equity_arr, index=grid, name="equity", dtype=np.float64)
    trades_df = (
        pd.DataFrame(trade_rows)
        if trade_rows
        else pd.DataFrame(columns=list(_TRADE_COLUMNS))
    )
    signals = pd.DataFrame(
        {"target": ["SHORT" if bool(s) else "CASH" for s in signal_short]},
        index=grid,
    )
    return BacktestResult(equity=equity_series, trades=trades_df, signals=signals)


def _check_contract() -> None:
    """Executable assertions locking the frozen OI screen surface."""
    from inspect import signature

    assert run_open_interest_deleveraging_screen.__name__ == "run_open_interest_deleveraging_screen"
    params = signature(run_open_interest_deleveraging_screen).parameters
    assert list(params) == ["market_data", "costs", "signal_delay_bars"]
    assert params["signal_delay_bars"].default == 1
    assert params["market_data"].annotation == "OIDeleveragingMarketData"
    assert params["costs"].annotation == "CostModel"


_check_contract()
