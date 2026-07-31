from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.logging_setup import setup_logger
from src.core.types import CarryCostModel, CashCarrySpec
from src.data.carry_data import CarryMarketData, validate_carry_market_data
from src.data.loader import DataIntegrityError
from src.engine.backtest import BacktestResult, _align_funding_rates
from src.strategy.cash_carry import generate_cash_carry_target

_logger = setup_logger("CashCarryBacktest")

_TRADE_COLUMNS = (
    "entry_bar",
    "exit_bar",
    "entry_time",
    "exit_time",
    "spot_entry",
    "spot_exit",
    "perp_entry",
    "perp_exit",
    "qty",
    "reason",
    "pnl",
    "return_pct",
    "funding_pnl",
    "equity_before_entry",
)


@dataclass
class _CarryTrade:
    entry_bar: int
    exit_bar: int
    spot_entry: float
    spot_exit: float
    perp_entry: float
    perp_exit: float
    qty: float
    reason: str
    pnl: float
    return_pct: float
    funding_pnl: float
    equity_before_entry: float


def run_cash_carry_backtest(
    data: CarryMarketData,
    spec: CashCarrySpec,
    costs: CarryCostModel,
    initial_equity: float = 10000.0,
    signal_delay_bars: int = 0,
) -> BacktestResult:
    """Same-asset cash-and-carry total-equity ledger.

    One spot long and one equal-base-quantity perpetual short are marked in a
    single total-equity ledger: ``equity = cash + spot_qty*spot_close +
    perp_qty*(perp_entry - perp_close)``, so equal spot/perpetual price moves
    cancel before financing. Positive funding credits the short leg at each
    actual settlement event; the per-bar quote-cash borrow rate finances the
    spot leg; both legs incur their fees and slippage on entry and exit.
    ``signal_delay_bars`` pushes every target execution one bar later for the
    frozen stress perturbation. A maintenance-buffer violation force-closes at
    the adverse executable mark and records ``margin_liquidation``; the returned
    equity stays finite. Inputs are never mutated.
    """
    validate_carry_market_data(data)
    if spec.symbol != data.symbol:
        raise DataIntegrityError(
            f"spec symbol {spec.symbol} does not match data symbol {data.symbol}"
        )
    if initial_equity <= 0:
        raise ValueError(f"initial_equity must be > 0, got {initial_equity}")
    if signal_delay_bars < 0:
        raise ValueError(f"signal_delay_bars must be >= 0, got {signal_delay_bars}")

    spot = data.spot
    perp = data.perp
    grid = spot.index
    n = len(grid)
    period = grid[1] - grid[0]
    window_end = grid[-1] + period
    funding_scope = data.funding[(data.funding.index >= grid[0]) & (data.funding.index < window_end)]
    bar_funding = _align_funding_rates(funding_scope, grid)
    borrow_arr = pd.to_numeric(data.borrow, errors="coerce").to_numpy(dtype=np.float64)

    spot_open = spot["open"].to_numpy(dtype=np.float64)
    spot_close = spot["close"].to_numpy(dtype=np.float64)
    perp_open = perp["open"].to_numpy(dtype=np.float64)
    perp_close = perp["close"].to_numpy(dtype=np.float64)

    equity_arr = np.full(n, np.nan, dtype=np.float64)
    cash = float(initial_equity)
    equity_prev = float(initial_equity)
    spot_qty = 0.0
    perp_qty = 0.0
    spot_entry = 0.0
    perp_entry = 0.0
    spot_efee = 0.0
    perp_efee = 0.0
    margin_reserved = 0.0
    trade_funding_pnl = 0.0
    entry_bar_idx = -1
    pending_target = "HOLD"
    pending_bar = -1
    trades: list[_CarryTrade] = []
    target_log: list[str] = []

    def record_close(t: int, spot_exit: float, perp_exit: float, reason: str) -> None:
        nonlocal cash, spot_qty, perp_qty, margin_reserved, trade_funding_pnl
        spot_xfee = costs.spot_fee_rate * spot_qty * spot_exit
        perp_xfee = costs.perp_fee_rate * perp_qty * perp_exit
        spot_leg = spot_qty * (spot_exit - spot_entry)
        perp_leg = perp_qty * (perp_entry - perp_exit)
        cash += spot_qty * spot_exit - spot_xfee + perp_leg - perp_xfee
        pnl = spot_leg + perp_leg - spot_efee - perp_efee - spot_xfee - perp_xfee + trade_funding_pnl
        return_pct = pnl / equity_prev if equity_prev != 0 else 0.0
        trades.append(_CarryTrade(
            entry_bar=entry_bar_idx, exit_bar=t,
            spot_entry=spot_entry, spot_exit=spot_exit,
            perp_entry=perp_entry, perp_exit=perp_exit,
            qty=spot_qty, reason=reason, pnl=pnl,
            return_pct=return_pct, funding_pnl=trade_funding_pnl,
            equity_before_entry=equity_prev,
        ))
        _logger.info(
            "bar=%d reason=%s spot_entry=%.4f spot_exit=%.4f perp_entry=%.4f "
            "perp_exit=%.4f qty=%.6f pnl=%.4f funding_pnl=%.4f",
            t, reason, spot_entry, spot_exit, perp_entry, perp_exit,
            spot_qty, pnl, trade_funding_pnl, extra={"tag": "ALGO"},
        )
        spot_qty = 0.0
        perp_qty = 0.0
        margin_reserved = 0.0
        trade_funding_pnl = 0.0

    for t in range(n):
        # 0. funding and borrow accrual for a position held into this bar.
        if perp_qty > 0 and bar_funding[t] != 0.0:
            funding_pnl = perp_qty * perp_open[t] * bar_funding[t]
            cash += funding_pnl
            trade_funding_pnl += funding_pnl
        if spot_qty > 0 and borrow_arr[t] != 0.0:
            borrow_cost = spot_qty * spot_open[t] * borrow_arr[t]
            cash -= borrow_cost
            trade_funding_pnl -= borrow_cost

        # 1. apply the scheduled target at this bar's open.
        if t == pending_bar and pending_target != "HOLD":
            if pending_target == "OPEN" and spot_qty == 0:
                fill_spot = spot_open[t] * (1 + costs.slippage_rate)
                fill_perp = perp_open[t] * (1 - costs.slippage_rate)
                denom = fill_spot * (1 + costs.spot_fee_rate) + fill_perp * (
                    costs.perp_fee_rate + spec.initial_margin_rate
                )
                if denom <= 0:
                    raise DataIntegrityError(f"entry denominator <= 0 at bar {t}")
                qty = equity_prev / denom
                spot_efee = costs.spot_fee_rate * qty * fill_spot
                perp_efee = costs.perp_fee_rate * qty * fill_perp
                margin_reserved = spec.initial_margin_rate * qty * fill_perp
                cash = cash - qty * fill_spot - spot_efee - perp_efee
                spot_qty = qty
                perp_qty = qty
                spot_entry = fill_spot
                perp_entry = fill_perp
                entry_bar_idx = t
                trade_funding_pnl = 0.0
                _logger.info(
                    "bar=%d OPEN fill_spot=%.4f fill_perp=%.4f qty=%.6f margin=%.4f",
                    t, fill_spot, fill_perp, qty, margin_reserved, extra={"tag": "ALGO"},
                )
            elif pending_target == "CLOSE" and spot_qty > 0:
                spot_exit = spot_open[t] * (1 - costs.slippage_rate)
                perp_exit = perp_open[t] * (1 + costs.slippage_rate)
                record_close(t, spot_exit, perp_exit, "carry_close")
            pending_target = "HOLD"
            pending_bar = -1

        # 2. maintenance margin check at this bar's mark.
        if perp_qty > 0:
            margin_available = margin_reserved + perp_qty * (perp_entry - perp_close[t])
            maintenance_req = spec.maintenance_margin_rate * perp_qty * perp_close[t]
            if margin_available < maintenance_req:
                spot_exit = spot_close[t] * (1 - costs.slippage_rate)
                perp_exit = perp_close[t] * (1 + costs.slippage_rate)
                record_close(t, spot_exit, perp_exit, "margin_liquidation")
                _logger.info(
                    "bar=%d margin_liquidation available=%.4f maintenance=%.4f",
                    t, margin_available, maintenance_req, extra={"tag": "ALGO"},
                )

        # 3. mark equity at the bar close.
        equity = (
            cash + spot_qty * spot_close[t] + perp_qty * (perp_entry - perp_close[t])
            if perp_qty > 0
            else cash
        )
        equity_arr[t] = equity
        equity_prev = equity

        # 4. form the next target from the completed decision timestamp. A
        # pending OPEN/CLOSE is locked once scheduled so ``signal_delay_bars``
        # delays that exact decision; only an idle or HOLD slot is refreshed.
        if pending_bar == -1 or pending_target == "HOLD":
            target = generate_cash_carry_target(data, grid[t], is_open=spot_qty > 0)
            apply_bar = t + 1 + signal_delay_bars
            if apply_bar < n:
                pending_target = target
                pending_bar = apply_bar
        target_log.append(pending_target)

    equity_series = pd.Series(equity_arr, index=grid, name="equity", dtype=np.float64)
    trades_df = (
        pd.DataFrame([{
            "entry_bar": tr.entry_bar,
            "exit_bar": tr.exit_bar,
            "entry_time": grid[tr.entry_bar],
            "exit_time": grid[tr.exit_bar],
            "spot_entry": tr.spot_entry,
            "spot_exit": tr.spot_exit,
            "perp_entry": tr.perp_entry,
            "perp_exit": tr.perp_exit,
            "qty": tr.qty,
            "reason": tr.reason,
            "pnl": tr.pnl,
            "return_pct": tr.return_pct,
            "funding_pnl": tr.funding_pnl,
            "equity_before_entry": tr.equity_before_entry,
        } for tr in trades])
        if trades
        else pd.DataFrame(columns=list(_TRADE_COLUMNS))
    )
    signals = pd.DataFrame({"target": target_log}, index=grid)
    return BacktestResult(equity=equity_series, trades=trades_df, signals=signals)


def _check_contract() -> None:
    """Executable assertions locking the frozen carry ledger surface."""
    assert run_cash_carry_backtest.__name__ == "run_cash_carry_backtest"
    from inspect import signature  # noqa: PLC0415

    params = signature(run_cash_carry_backtest).parameters
    assert list(params) == [
        "data", "spec", "costs", "initial_equity", "signal_delay_bars",
    ]


_check_contract()
