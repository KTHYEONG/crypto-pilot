"""Generic signed-target execution engine for the frozen technical candidates.

A decision formed at close[t] produces a signed target in {-1, 0, 1} that is
executed no earlier than open[t + 1 + signal_delay_bars]. Every position
transition charges the existing fee and slippage on the changed notional, and
positive funding debits a long while crediting a short, only when the position
exists at the settlement timestamp.
"""

from __future__ import annotations

from typing import Literal

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
    stop_loss_mode: Literal["fixed_pct", "atr_multiple"] | None = None,
    stop_loss_value: float | None = None,
    atr_period: int = 14,
    trailing_stop: bool = False,
    execution_start: pd.Timestamp | None = None,
) -> BacktestResult:
    """Run one directional technical candidate on the completed-bar grid.

    The candidate's events are converted into a persistent signed target and a
    deterministic future-open execution schedule; each transition charges the
    existing fee and slippage on the changed notional and funding accrues only
    when a position exists at the settlement timestamp. The equity series is
    the single marked ledger and the closed-trade records carry the exact
    ``BacktestResult`` shape used by the shared component panel.

    When ``execution_start`` is supplied, signals are still generated on the
    entire supplied frame (indicator warm-up) but cash, positions, funding,
    equity marks, and trade rows begin at the first bar at or after
    ``execution_start`` with fresh ``initial_equity``. Earlier bars carry the
    flat initial-equity ledger so no pre-start PnL or bankruptcy can invalidate
    the returned evaluation window. When omitted the behavior is byte-identical
    to prior releases.
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
    if stop_loss_mode is not None:
        if stop_loss_value is None or stop_loss_value <= 0.0:
            raise ValueError(
                f"stop_loss_value must be > 0.0 when stop_loss_mode is set, got "
                f"{stop_loss_value}"
            )
        if stop_loss_mode == "fixed_pct" and stop_loss_value >= 1.0:
            raise ValueError(
                f"fixed_pct stop_loss_value must be < 1.0, got {stop_loss_value}"
            )
    if atr_period < 1:
        raise ValueError(f"atr_period must be >= 1, got {atr_period}")

    events = generate_signal_events(frame, candidate)
    grid = frame.index
    bar_funding = _align_funding_rates(funding_rates, grid)

    execution_start_idx: int | None = None
    if execution_start is not None:
        start_ts = execution_start
        if not isinstance(start_ts, pd.Timestamp):
            start_ts = pd.Timestamp(start_ts)
        if start_ts.tzinfo is None and grid.tz is not None:
            start_ts = start_ts.tz_localize(grid.tz)
        if start_ts.tzinfo is not None and grid.tz is None:
            start_ts = start_ts.tz_localize(None)
        if start_ts < grid[0] or start_ts > grid[-1]:
            raise ValueError(
                f"execution_start {start_ts.isoformat()} is outside the bar grid "
                f"[{grid[0].isoformat()}, {grid[-1].isoformat()}]"
            )
        execution_start_idx = int(grid.searchsorted(start_ts, side="left"))


    open_arr = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=np.float64)
    high_arr = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=np.float64)
    low_arr = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=np.float64)
    close_arr = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(grid)

    atr_arr: np.ndarray | None = None
    if stop_loss_mode == "atr_multiple":
        prev_close = np.full(n, np.nan, dtype=np.float64)
        prev_close[1:] = close_arr[:-1]
        true_range = np.maximum.reduce([
            high_arr - low_arr,
            np.abs(high_arr - prev_close),
            np.abs(low_arr - prev_close),
        ])
        true_range[0] = high_arr[0] - low_arr[0]
        atr = (
            pd.Series(true_range)
            .rolling(window=atr_period, min_periods=atr_period)
            .mean()
            .to_numpy()
        )
        atr_arr = np.full(n, np.nan, dtype=np.float64)
        atr_arr[1:] = atr[:-1]

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
        open_arr, close_arr, high_arr, low_arr, bar_funding, schedule, costs,
        initial_equity, grid,
        stop_loss_mode=stop_loss_mode, stop_loss_value=stop_loss_value,
        atr_arr=atr_arr, trailing_stop=trailing_stop,
        execution_start_idx=execution_start_idx,
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
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    bar_funding: np.ndarray,
    schedule: list[int | None],
    costs: CostModel,
    initial_equity: float,
    grid: pd.DatetimeIndex,
    stop_loss_mode: Literal["fixed_pct", "atr_multiple"] | None = None,
    stop_loss_value: float | None = None,
    atr_arr: np.ndarray | None = None,
    trailing_stop: bool = False,
    execution_start_idx: int | None = None,
) -> tuple[pd.Series, list[dict[str, object]]]:
    """Execute a per-bar signed target schedule at each bar's open.

    A scheduled entry opens a full-notional position (1.0 gross exposure) in the
    target direction, a scheduled zero closes it, and a direct reversal closes
    the current position before opening the opposite direction. Fee/slippage is
    charged on every changed notional and positive funding is debited from a
    long while credited to a short, only while the position exists at the
    settlement timestamp.

    When ``stop_loss_mode`` is set, an intrabar stop check runs before the
    scheduled-target block for a position carried in from a prior bar. The stop
    distance is fixed at entry (fixed fraction of the fill price, or an ATR
    multiple evaluated on causally shifted bars); ``trailing_stop`` instead
    anchors the stop to the favorable extreme seen while the position is open.

    When ``execution_start_idx`` is supplied, bars strictly before it are
    indicator-only warm-up: they carry the flat ``initial_equity`` mark, never
    open a position, accrue no funding, and produce no trade rows. Execution
    begins at that bar with fresh capital.
    """
    n = len(open_arr)
    cash = initial_equity
    position_qty = 0.0
    side_sign = 0  # 1 long, -1 short, 0 flat
    entry_price = 0.0
    entry_bar_idx = -1
    stop_distance: float | None = None
    favorable_extreme = 0.0
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
        nonlocal cash, position_qty, side_sign, entry_price, entry_bar_idx
        nonlocal trade_funding_pnl, stop_distance, favorable_extreme
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
        stop_distance = None
        favorable_extreme = fill
        if stop_loss_mode == "fixed_pct":
            assert stop_loss_value is not None
            stop_distance = fill * stop_loss_value
        elif stop_loss_mode == "atr_multiple" and atr_arr is not None:
            entry_atr = atr_arr[t]
            if not np.isnan(entry_atr):
                stop_distance = stop_loss_value * entry_atr

    for t in range(n):
        o = open_arr[t]
        c = close_arr[t]

        # Indicator-only warm-up: before execution_start the ledger is flat at
        # initial_equity and no position, funding, or trade row exists.
        if execution_start_idx is not None and t < execution_start_idx:
            equity_arr[t] = initial_equity
            continue

        # Funding accrual at this bar's published timestamp. Positive funding
        # debits a long and credits a short, only for a position already held
        # into the settlement bar.
        if position_qty > 0 and bar_funding[t] != 0.0:
            funding_pnl = -side_sign * position_qty * o * bar_funding[t]
            cash += funding_pnl
            trade_funding_pnl += funding_pnl

        if stop_loss_mode is not None and position_qty > 0 and stop_distance is not None:
            if trailing_stop:
                if side_sign == 1:
                    favorable_extreme = max(favorable_extreme, high_arr[t])
                    stop_price = favorable_extreme - stop_distance
                else:
                    favorable_extreme = min(favorable_extreme, low_arr[t])
                    stop_price = favorable_extreme + stop_distance
            elif side_sign == 1:
                stop_price = entry_price - stop_distance
            else:
                stop_price = entry_price + stop_distance
            stop_hit = (
                (side_sign == 1 and low_arr[t] <= stop_price)
                or (side_sign == -1 and high_arr[t] >= stop_price)
            )
            if stop_hit:
                stop_fill = (
                    stop_price * (1.0 - costs.slippage_rate)
                    if side_sign == 1
                    else stop_price * (1.0 + costs.slippage_rate)
                )
                _close("stop_loss", stop_fill, t)

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
        "stop_loss_mode", "stop_loss_value", "atr_period", "trailing_stop",
        "execution_start",
    ]
    assert params["initial_equity"].default == 10_000.0
    assert params["signal_delay_bars"].default == 0
    assert params["stop_loss_mode"].default is None
    assert params["stop_loss_value"].default is None
    assert params["atr_period"].default == 14
    assert params["trailing_stop"].default is False
    assert params["execution_start"].default is None


_check_contract()
