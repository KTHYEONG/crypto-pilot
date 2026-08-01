"""Generic signed-target execution engine for the frozen technical candidates.

A decision formed at close[t] produces a signed target in {-1, 0, 1} that is
executed no earlier than open[t + 1 + signal_delay_bars]. Every position
transition charges the existing fee and slippage on the changed notional, and
positive funding debits a long while crediting a short, only when the position
exists at the settlement timestamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.research.baseline.backtest import BacktestResult, _align_funding_rates
from src.research.contracts import CostModel
from src.research.technical_experts.contracts import TechnicalCandidate
from src.research.technical_experts.signals import _validate_ohlcv_frame, generate_signal_events

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


def run_technical_expert_backtest(
    frame: pd.DataFrame,
    candidate: TechnicalCandidate,
    costs: CostModel,
    funding_rates: pd.Series,
    initial_equity: float = 10_000.0,
    signal_delay_bars: int = 0,
) -> BacktestResult:
    """Run one directional technical candidate on the completed-bar grid.

    The candidate's events are converted into a persistent signed target and a
    deterministic future-open execution schedule; each transition charges the
    existing fee and slippage on the changed notional and funding accrues only
    when a position exists at the settlement timestamp. The equity series is
    the single marked ledger and the closed-trade records carry the exact
    ``BacktestResult`` shape used by the shared component panel.
    """
    _validate_ohlcv_frame(frame, candidate.min_history_bars)
    if signal_delay_bars < 0:
        raise ValueError(f"signal_delay_bars must be >= 0, got {signal_delay_bars}")
    if initial_equity <= 0:
        raise ValueError(f"initial_equity must be > 0, got {initial_equity}")
    if funding_rates is None or len(funding_rates) == 0:
        raise DataIntegrityError(
            "technical expert mode requires a non-empty funding_rates series"
        )

    events = generate_signal_events(frame, candidate)
    grid = frame.index
    bar_funding = _align_funding_rates(funding_rates, grid)

    open_arr = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=np.float64)
    close_arr = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(grid)

    schedule: list[int | None] = [None] * n
    target_series: list[int] = []
    current_target = 0
    for t in range(n):
        if bool(events["long_entry"].iloc[t]):
            current_target = 1
        elif bool(events["short_entry"].iloc[t]):
            current_target = -1
        elif bool(events["long_exit"].iloc[t]) or bool(events["short_exit"].iloc[t]):
            current_target = 0
        target_series.append(current_target)
        apply_bar = t + 1 + signal_delay_bars
        if apply_bar < n:
            schedule[apply_bar] = current_target

    equity_series, trade_rows = _execute_target_schedule(
        open_arr, close_arr, bar_funding, schedule, costs, initial_equity, grid,
    )
    trades_df = (
        pd.DataFrame(trade_rows)
        if trade_rows
        else pd.DataFrame(columns=list(_TRADE_COLUMNS))
    )
    signals = pd.DataFrame({"target": target_series}, index=grid)
    return BacktestResult(equity=equity_series, trades=trades_df, signals=signals)


def _execute_target_schedule(
    open_arr: np.ndarray,
    close_arr: np.ndarray,
    bar_funding: np.ndarray,
    schedule: list[int | None],
    costs: CostModel,
    initial_equity: float,
    grid: pd.DatetimeIndex,
) -> tuple[pd.Series, list[dict[str, object]]]:
    """Execute a per-bar signed target schedule at each bar's open.

    A scheduled entry opens a full-notional position (1.0 gross exposure) in the
    target direction, a scheduled zero closes it, and a direct reversal closes
    the current position before opening the opposite direction. Fee/slippage is
    charged on every changed notional and positive funding is debited from a
    long while credited to a short, only while the position exists at the
    settlement timestamp.
    """
    n = len(open_arr)
    cash = initial_equity
    position_qty = 0.0
    side_sign = 0  # 1 long, -1 short, 0 flat
    entry_price = 0.0
    entry_bar_idx = -1
    trade_funding_pnl = 0.0
    equity_arr = np.full(n, np.nan, dtype=np.float64)
    trade_rows: list[dict[str, object]] = []

    def _close(reason: str, exit_price: float, t: int) -> None:
        nonlocal cash, position_qty, side_sign, trade_funding_pnl
        pnl = position_qty * side_sign * (exit_price - entry_price)
        exit_fee = costs.fee_rate * abs(position_qty * exit_price)
        if side_sign == 1:
            cash += position_qty * exit_price - exit_fee
        else:
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
            "side": "long" if side_sign == 1 else "short",
        })
        position_qty = 0.0
        side_sign = 0
        trade_funding_pnl = 0.0

    def _open(direction: int, fill: float, t: int) -> None:
        nonlocal cash, position_qty, side_sign, entry_price, entry_bar_idx, trade_funding_pnl
        qty = cash / fill
        if qty <= 0.0:
            return
        entry_fee = costs.fee_rate * qty * fill
        if direction == 1:
            cash -= qty * fill + entry_fee
        else:
            cash += qty * fill - entry_fee
        position_qty = qty
        side_sign = direction
        entry_price = fill
        entry_bar_idx = t
        trade_funding_pnl = 0.0

    for t in range(n):
        o = open_arr[t]
        c = close_arr[t]

        # Funding accrual at this bar's published timestamp. Positive funding
        # debits a long and credits a short, only for a position already held
        # into the settlement bar.
        if position_qty > 0 and bar_funding[t] != 0.0:
            funding_pnl = -side_sign * position_qty * o * bar_funding[t]
            cash += funding_pnl
            trade_funding_pnl += funding_pnl

        scheduled = schedule[t]
        if scheduled is not None:
            if scheduled == 0 and position_qty > 0:
                close_fill = (
                    o * (1.0 - costs.slippage_rate)
                    if side_sign == 1
                    else o * (1.0 + costs.slippage_rate)
                )
                _close("signal", close_fill, t)
            elif scheduled != 0 and position_qty > 0 and scheduled != side_sign:
                close_fill = (
                    o * (1.0 - costs.slippage_rate)
                    if side_sign == 1
                    else o * (1.0 + costs.slippage_rate)
                )
                _close("reversal", close_fill, t)
                open_fill = (
                    o * (1.0 + costs.slippage_rate)
                    if scheduled == 1
                    else o * (1.0 - costs.slippage_rate)
                )
                _open(scheduled, open_fill, t)
            elif scheduled != 0 and position_qty == 0:
                open_fill = (
                    o * (1.0 + costs.slippage_rate)
                    if scheduled == 1
                    else o * (1.0 - costs.slippage_rate)
                )
                _open(scheduled, open_fill, t)

        marked_equity = cash + side_sign * position_qty * c if position_qty > 0 else cash
        if marked_equity <= 0.0:
            raise DataIntegrityError(
                f"technical expert equity exhausted at {grid[t].isoformat()}"
            )
        equity_arr[t] = marked_equity

    equity_series = pd.Series(equity_arr, index=grid, name="equity", dtype=np.float64)
    return equity_series, trade_rows


def _check_contract() -> None:
    """Executable assertions locking the frozen technical backtest surface."""
    from inspect import signature

    assert run_technical_expert_backtest.__name__ == "run_technical_expert_backtest"
    params = signature(run_technical_expert_backtest).parameters
    assert list(params) == [
        "frame", "candidate", "costs", "funding_rates",
        "initial_equity", "signal_delay_bars",
    ]
    assert params["initial_equity"].default == 10_000.0
    assert params["signal_delay_bars"].default == 0


_check_contract()
