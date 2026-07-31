from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.logging_setup import setup_logger
from src.core.types import CostModel, StrategySpec

_logger = setup_logger("Backtest")


@dataclass
class TradeRecord:
    entry_bar: int
    entry_price: float
    exit_price: float
    qty: float
    reason: str
    pnl: float
    return_pct: float


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    signals: pd.DataFrame


def calculate_position_size(
    *,
    equity: float,
    risk_fraction: float,
    entry_price: float,
    stop_price: float,
    max_leverage: float,
) -> float:
    if equity <= 0:
        raise ValueError(f"equity must be > 0, got {equity}")
    if not 0 < risk_fraction <= 1:
        raise ValueError(f"risk_fraction must be in (0, 1], got {risk_fraction}")
    if entry_price <= 0:
        raise ValueError(f"entry_price must be > 0, got {entry_price}")
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        raise ValueError(f"stop_distance must be > 0, got {stop_distance}")

    risk_qty = equity * risk_fraction / stop_distance
    leverage_qty = max_leverage * equity / entry_price
    return min(risk_qty, leverage_qty)


def run_backtest(
    df: pd.DataFrame,
    spec: StrategySpec,
    costs: CostModel,
    initial_equity: float = 10_000.0,
    signal_delay_bars: int = 0,
) -> BacktestResult:
    from src.strategy.donchian import generate_signals

    if signal_delay_bars < 0:
        raise ValueError(f"signal_delay_bars must be >= 0, got {signal_delay_bars}")

    feat = generate_signals(df, spec)
    if signal_delay_bars > 0:
        feat["entry_signal"] = feat["entry_signal"].shift(signal_delay_bars).fillna(False).astype(bool)
        feat["exit_lower"] = feat["exit_lower"].shift(signal_delay_bars)

    warmup = max(spec.ema_period, spec.entry_period, spec.atr_period) + 1 + signal_delay_bars
    atr_arr = feat["atr"].to_numpy(dtype=np.float64)

    equity_arr = np.full(len(feat), np.nan, dtype=np.float64)
    cash = initial_equity
    position_qty = 0.0
    entry_price = 0.0
    stop_price = 0.0
    entry_bar_idx = -1
    pending_entry = False
    pending_exit = False
    exited_this_bar = False
    trades: list[TradeRecord] = []

    for t in range(len(feat)):
        row = feat.iloc[t]
        o, l_, c = row["open"], row["low"], row["close"]

        if t < warmup:
            equity_arr[t] = cash
            pending_entry = bool(row["entry_signal"])
            pending_exit = False
            continue

        # 1. stop check (intrabar)
        if position_qty > 0 and l_ <= stop_price:
            exit_price = min(o, stop_price) * (1 - costs.slippage_rate)
            reason = "stop"
        # 3. channel exit
        elif position_qty > 0 and pending_exit:
            exit_price = o * (1 - costs.slippage_rate)
            reason = "channel"
        else:
            exit_price = None
            reason = ""

        if exit_price is not None:
            pnl = position_qty * (exit_price - entry_price)
            exit_fee = costs.fee_rate * abs(position_qty * exit_price)
            cash += position_qty * exit_price - exit_fee
            pnl -= exit_fee
            trades.append(TradeRecord(
                entry_bar=entry_bar_idx, entry_price=entry_price,
                exit_price=exit_price, qty=position_qty,
                reason=reason, pnl=pnl,
                return_pct=pnl / (cash - pnl) if cash != pnl else 0.0,
            ))
            _logger.info(
                "bar=%d reason=%s entry=%.4f exit=%.4f qty=%.6f pnl=%.4f",
                t, reason, entry_price, exit_price, position_qty, pnl,
                extra={"tag": "ALGO"},
            )
            position_qty = 0.0
            exited_this_bar = True

        # 5. entry (only if flat and not exited this bar)
        if position_qty == 0 and not exited_this_bar and pending_entry:
            # signal-bar ATR (t-1): the entry signal was formed at close[t-1],
            # so the stop distance must be causal at that point, not use atr[t]
            # (which requires high[t]/low[t]/close[t], unknown at open[t]).
            stop_distance = spec.stop_atr_mult * atr_arr[t - 1]
            if stop_distance <= 0:
                raise ValueError(f"stop_distance <= 0 at bar {t}: {stop_distance}")
            fill = o * (1 + costs.slippage_rate)
            qty = calculate_position_size(
                equity=cash,
                risk_fraction=spec.risk_per_trade,
                entry_price=fill,
                stop_price=fill - stop_distance,
                max_leverage=spec.max_leverage,
            )
            notional = qty * fill
            if notional < 5.0:
                pass
            else:
                entry_fee = costs.fee_rate * notional
                cash = cash - notional - entry_fee
                position_qty = qty
                entry_price = fill
                stop_price = fill - stop_distance
                entry_bar_idx = t

                # 7b. entry-bar stop check
                if l_ <= stop_price:
                    exit_price = min(o, stop_price) * (1 - costs.slippage_rate)
                    pnl = position_qty * (exit_price - entry_price) - entry_fee
                    exit_fee = costs.fee_rate * abs(position_qty * exit_price)
                    cash += position_qty * exit_price - exit_fee
                    pnl -= exit_fee
                    trades.append(TradeRecord(
                        entry_bar=entry_bar_idx, entry_price=entry_price,
                        exit_price=exit_price, qty=qty,
                        reason="stop_entrybar", pnl=pnl,
                        return_pct=pnl / (cash - pnl) if cash != pnl else 0.0,
                    ))
                    _logger.info(
                        "bar=%d reason=stop_entrybar entry=%.4f exit=%.4f qty=%.6f pnl=%.4f",
                        t, entry_price, exit_price, qty, pnl,
                        extra={"tag": "ALGO"},
                    )
                    position_qty = 0.0
                    exited_this_bar = True

        # 8. mark equity = cash + market value of position
        equity_arr[t] = cash + position_qty * c if position_qty > 0 else cash

        # 9. form pending signals for next bar
        pending_entry = (
            position_qty == 0
            and not exited_this_bar
            and bool(row["entry_signal"])
        )
        pending_exit = position_qty > 0 and bool(c < row["exit_lower"])
        if exited_this_bar and position_qty == 0:
            pending_entry = False
        exited_this_bar = False

    equity_series = pd.Series(
        equity_arr,
        index=feat.index,
        name="equity",
        dtype=np.float64,
    )
    trades_df = pd.DataFrame([{
        "entry_bar": t.entry_bar,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "qty": t.qty,
        "reason": t.reason,
        "pnl": t.pnl,
        "return_pct": t.return_pct,
    } for t in trades]) if trades else pd.DataFrame(columns=[
        "entry_bar", "entry_price", "exit_price", "qty", "reason", "pnl", "return_pct",
    ])

    return BacktestResult(equity=equity_series, trades=trades_df, signals=feat)
