from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.logging_setup import setup_logger
from src.core.types import CostModel, StrategySpec
from src.data.loader import DataIntegrityError

_logger = setup_logger("Backtest")


@dataclass
class TradeRecord:
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    qty: float
    reason: str
    pnl: float
    return_pct: float
    funding_pnl: float = 0.0


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


def _align_funding_rates(
    funding_rates: pd.Series,
    bar_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Map a published-funding series onto per-bar accrued rates.

    Returns an array aligned to ``bar_index`` where each entry is the sum of
    funding rates published inside that bar's window. Raises
    ``DataIntegrityError`` for non-finite rates, a non-monotonic index, or
    timestamps outside the bar window: missing funding is never a zero-cost
    assumption.
    """
    if not isinstance(bar_index, pd.DatetimeIndex) or len(bar_index) < 2:
        raise DataIntegrityError("bar index must be a DatetimeIndex of length >= 2")
    ts = pd.DatetimeIndex(pd.to_datetime(funding_rates.index, utc=True, errors="coerce"))
    if ts.hasnans:
        raise DataIntegrityError("funding_rates index must contain datetimes")
    rates = pd.to_numeric(funding_rates, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(rates).all():
        raise DataIntegrityError("funding_rates must be finite")
    if not ts.is_monotonic_increasing:
        raise DataIntegrityError("funding_rates must be monotonic in time")

    series = pd.Series(rates, index=ts)
    series = series[~series.index.duplicated(keep="last")].sort_index()

    bar_period = bar_index[1] - bar_index[0]
    window_end = bar_index[-1] + bar_period
    inside = (series.index >= bar_index[0]) & (series.index < window_end)
    if not inside.all():
        raise DataIntegrityError(
            "funding_rates timestamps are not aligned with the bar window"
        )

    pos = bar_index.searchsorted(series.index, side="right") - 1
    bar_funding = np.zeros(len(bar_index), dtype=np.float64)
    np.add.at(bar_funding, pos, series.to_numpy(dtype=np.float64))
    return bar_funding


def run_backtest(
    df: pd.DataFrame,
    spec: StrategySpec,
    costs: CostModel,
    initial_equity: float = 10_000.0,
    signal_delay_bars: int = 0,
    funding_rates: pd.Series | None = None,
) -> BacktestResult:
    from src.strategy.donchian import generate_signals

    if signal_delay_bars < 0:
        raise ValueError(f"signal_delay_bars must be >= 0, got {signal_delay_bars}")
    if spec.min_taker_buy_ratio is not None and (
        funding_rates is None or len(funding_rates) == 0
    ):
        raise DataIntegrityError(
            "candidate mode (min_taker_buy_ratio set) requires a non-empty, "
            "aligned funding_rates series"
        )

    feat = generate_signals(df, spec)
    if signal_delay_bars > 0:
        feat["entry_signal"] = feat["entry_signal"].shift(signal_delay_bars).fillna(False).astype(bool)
        feat["exit_lower"] = feat["exit_lower"].shift(signal_delay_bars)

    warmup = max(spec.ema_period, spec.entry_period, spec.atr_period) + 1 + signal_delay_bars
    atr_arr = feat["atr"].to_numpy(dtype=np.float64)

    bar_funding = (
        _align_funding_rates(funding_rates, feat.index)
        if funding_rates is not None
        else np.zeros(len(feat), dtype=np.float64)
    )

    equity_arr = np.full(len(feat), np.nan, dtype=np.float64)
    cash = initial_equity
    position_qty = 0.0
    entry_price = 0.0
    stop_price = 0.0
    entry_bar_idx = -1
    pending_entry = False
    pending_exit = False
    exited_this_bar = False
    trade_funding_pnl = 0.0
    trades: list[TradeRecord] = []

    for t in range(len(feat)):
        row = feat.iloc[t]
        o, l_, c = row["open"], row["low"], row["close"]

        if t < warmup:
            equity_arr[t] = cash
            pending_entry = bool(row["entry_signal"])
            pending_exit = False
            continue

        # 0. funding accrual at this bar's published timestamp (bar-open aligned).
        # A long held into the timestamp pays notional x rate for positive funding
        # and is credited for negative funding. Positions opened later in this same
        # bar are not charged for a timestamp that precedes their entry.
        if position_qty > 0 and bar_funding[t] != 0.0:
            funding_pnl = -position_qty * o * bar_funding[t]
            cash += funding_pnl
            trade_funding_pnl += funding_pnl

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
            pnl += trade_funding_pnl
            trades.append(TradeRecord(
                entry_bar=entry_bar_idx, exit_bar=t, entry_price=entry_price,
                exit_price=exit_price, qty=position_qty,
                reason=reason, pnl=pnl,
                return_pct=pnl / (cash - pnl) if cash != pnl else 0.0,
                funding_pnl=trade_funding_pnl,
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
                trade_funding_pnl = 0.0

                # 7b. entry-bar stop check
                if l_ <= stop_price:
                    exit_price = min(o, stop_price) * (1 - costs.slippage_rate)
                    pnl = position_qty * (exit_price - entry_price) - entry_fee
                    exit_fee = costs.fee_rate * abs(position_qty * exit_price)
                    cash += position_qty * exit_price - exit_fee
                    pnl -= exit_fee
                    pnl += trade_funding_pnl
                    trades.append(TradeRecord(
                        entry_bar=entry_bar_idx, exit_bar=t, entry_price=entry_price,
                        exit_price=exit_price, qty=qty,
                        reason="stop_entrybar", pnl=pnl,
                        return_pct=pnl / (cash - pnl) if cash != pnl else 0.0,
                        funding_pnl=trade_funding_pnl,
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
        "exit_bar": t.exit_bar,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "qty": t.qty,
        "reason": t.reason,
        "pnl": t.pnl,
        "return_pct": t.return_pct,
        "funding_pnl": t.funding_pnl,
    } for t in trades]) if trades else pd.DataFrame(columns=[
        "entry_bar", "exit_bar", "entry_price", "exit_price", "qty", "reason", "pnl",
        "return_pct", "funding_pnl",
    ])

    return BacktestResult(equity=equity_series, trades=trades_df, signals=feat)
