"""
Shared-cash spot portfolio: one balance, bar-aligned symbols, optional top-N concurrent slots.

When max_concurrent_positions == 1, only one symbol holds a position at a time (capital rotation).
Exit/entry rules match engine_spot.backtest_loop_numba_spot per active slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import SPOT_INITIAL_BALANCE, SPOT_SLIPPAGE_RATE, UPBIT_SPOT_TAKER_FEE_RATE


@dataclass
class SharedCashResult:
    equity_curve: np.ndarray
    final_balance: float
    total_trades: int


@dataclass
class _SlotState:
    sym_idx: int = -1
    in_position: bool = False
    entry_price: float = 0.0
    entry_idx: int = 0
    amount: float = 0.0
    entry_fee_stored: float = 0.0
    stop_price: float = 0.0
    tp_price: float = 0.0
    highest: float = 0.0
    scale_done: bool = False


def run_shared_cash_multi_symbol(
    symbol_arrays: Dict[str, Dict[str, np.ndarray]],
    symbols_ordered: List[str],
    params: Dict[str, object],
    *,
    initial_balance: float = SPOT_INITIAL_BALANCE,
    max_concurrent_positions: int = 3,
    rank_scores: Optional[Dict[str, np.ndarray]] = None,
    warmup_bars: int = 0,
    execution_start_idx: int = 0,
) -> SharedCashResult:
    """
    Shared balance; per bar: exit all slots first, then enter ranked candidates into free slots.
    All symbol arrays must have length n.
    """
    if not symbols_ordered:
        return SharedCashResult(equity_curve=np.array([]), final_balance=initial_balance, total_trades=0)

    n = int(len(symbol_arrays[symbols_ordered[0]]["close"]))
    for s in symbols_ordered:
        if len(symbol_arrays[s]["close"]) != n:
            raise ValueError("Shared-cash requires identical bar counts per symbol.")

    fee_rate = float(UPBIT_SPOT_TAKER_FEE_RATE)
    slippage_rate = float(SPOT_SLIPPAGE_RATE)
    risk_per_trade = float(params.get("RISK_PER_TRADE", 0.015))
    max_position_pct = float(params.get("MAX_POSITION_PCT", 0.25))
    long_atr_mult = float(params.get("LONG_ATR_MULT", 3.0))
    long_trail_mult = float(params.get("LONG_TRAIL_MULT", 3.0))
    long_tp_mult = float(params.get("LONG_TP_MULT", 5.0))
    long_scale_atr_mult = float(params.get("LONG_SCALE_ATR_MULT", 0.0))
    scale_out_pct = float(params.get("SCALE_OUT_PCT", 0.0))
    time_stop_bars = int(params.get("TIME_STOP_BARS", 0))

    max_slots = max(1, min(int(max_concurrent_positions), len(symbols_ordered)))
    slots: List[_SlotState] = [_SlotState() for _ in range(max_slots)]
    balance = float(initial_balance)
    equity_curve = np.zeros(n, dtype=np.float64)
    total_trades = 0

    for i in range(n):
        # --- exits and in-position management for each occupied slot ---
        for slot in slots:
            if slot.sym_idx < 0 or not slot.in_position:
                continue
            sym = symbols_ordered[slot.sym_idx]
            arr = symbol_arrays[sym]
            slot, balance, td = _process_in_position_bar(
                i=i,
                n=n,
                close=arr["close"],
                high=arr["high"],
                low=arr["low"],
                open_p=arr["open"],
                atr=arr["atr"],
                slot=slot,
                balance=balance,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                long_trail_mult=long_trail_mult,
                long_tp_mult=long_tp_mult,
                long_scale_atr_mult=long_scale_atr_mult,
                scale_out_pct=scale_out_pct,
                time_stop_bars=time_stop_bars,
                warmup_bars=warmup_bars,
                execution_start_idx=execution_start_idx,
            )
            total_trades += td
            if not slot.in_position:
                slot.sym_idx = -1

        # --- entries ---
        if i < max(warmup_bars, execution_start_idx):
            equity_curve[i] = _portfolio_equity(balance, slots, symbol_arrays, symbols_ordered, i)
            continue

        prev_i = i - 1 if i > 0 else 0
        free_slots = [j for j, sl in enumerate(slots) if sl.sym_idx < 0]
        if not free_slots:
            equity_curve[i] = _portfolio_equity(balance, slots, symbol_arrays, symbols_ordered, i)
            continue

        candidates: List[int] = []
        for si, sym in enumerate(symbols_ordered):
            if any(sl.sym_idx == si and sl.in_position for sl in slots):
                continue
            arr = symbol_arrays[sym]
            les = float(arr["long_entry_signal"][prev_i])
            if les < 0.5 or np.isnan(les):
                continue
            c_high = float(arr["high"][i])
            if c_high > float(arr["entry_upper"][prev_i]):
                candidates.append(si)

        if not candidates:
            equity_curve[i] = _portfolio_equity(balance, slots, symbol_arrays, symbols_ordered, i)
            continue

        def rank_key(si: int) -> float:
            if rank_scores is None:
                return float(-si)
            sym = symbols_ordered[si]
            rs = rank_scores.get(sym)
            if rs is None or len(rs) <= prev_i:
                return 0.0
            v = float(rs[prev_i])
            return v if np.isfinite(v) else 0.0

        candidates.sort(key=rank_key, reverse=True)

        for si in candidates:
            if not free_slots:
                break
            sym = symbols_ordered[si]
            arr = symbol_arrays[sym]
            j = free_slots[0]
            rr = arr.get("regime_risk_mult")
            st, balance, opened, td = _try_open_long(
                i=i,
                n=n,
                close=arr["close"],
                high=arr["high"],
                low=arr["low"],
                open_p=arr["open"],
                entry_upper=arr["entry_upper"],
                long_entry_signal=arr["long_entry_signal"],
                atr=arr["atr"],
                balance=balance,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                risk_per_trade=risk_per_trade,
                max_position_pct=max_position_pct,
                long_atr_mult=long_atr_mult,
                long_tp_mult=long_tp_mult,
                warmup_bars=warmup_bars,
                execution_start_idx=execution_start_idx,
                regime_risk_mult=rr,
            )
            total_trades += td
            if opened:
                st.sym_idx = si
                slots[j] = st
                free_slots.pop(0)

        equity_curve[i] = _portfolio_equity(balance, slots, symbol_arrays, symbols_ordered, i)

    # Force-close at last bar (match numba)
    last_idx = n - 1
    if n > 0:
        for slot in slots:
            if slot.sym_idx < 0 or not slot.in_position:
                continue
            sym = symbols_ordered[slot.sym_idx]
            arr = symbol_arrays[sym]
            c_last = float(arr["close"][last_idx])
            exit_price = c_last * (1.0 - slippage_rate)
            pnl = (exit_price - slot.entry_price) * slot.amount
            pnl -= slot.amount * exit_price * fee_rate
            balance += (slot.amount * slot.entry_price) + pnl
            total_trades += 1
            slot.in_position = False
            slot.sym_idx = -1
        equity_curve[last_idx] = balance

    return SharedCashResult(equity_curve=equity_curve, final_balance=float(balance), total_trades=total_trades)


def _portfolio_equity(
    balance: float,
    slots: List[_SlotState],
    symbol_arrays: Dict[str, Dict[str, np.ndarray]],
    symbols_ordered: List[str],
    i: int,
) -> float:
    eq = balance
    for slot in slots:
        if slot.sym_idx < 0 or not slot.in_position:
            continue
        sym = symbols_ordered[slot.sym_idx]
        c_price = float(symbol_arrays[sym]["close"][i])
        unrealized = (c_price - slot.entry_price) * slot.amount
        eq += slot.amount * slot.entry_price + unrealized
    return eq


def _process_in_position_bar(
    *,
    i: int,
    n: int,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_p: np.ndarray,
    atr: np.ndarray,
    slot: _SlotState,
    balance: float,
    fee_rate: float,
    slippage_rate: float,
    long_trail_mult: float,
    long_tp_mult: float,
    long_scale_atr_mult: float,
    scale_out_pct: float,
    time_stop_bars: int,
    warmup_bars: int,
    execution_start_idx: int,
) -> Tuple[_SlotState, float, int]:
    trades_delta = 0
    if i < warmup_bars or i < execution_start_idx:
        return slot, balance, 0

    c_open = float(open_p[i])
    c_high = float(high[i])
    c_low = float(low[i])
    trail_atr = float(atr[i])
    if trail_atr <= 0.0 or np.isnan(trail_atr):
        trail_atr = float(atr[slot.entry_idx]) if slot.entry_idx < n else 1e-9

    exit_triggered = False
    exit_price = 0.0

    if c_high > slot.highest:
        slot.highest = c_high

    if c_open <= slot.stop_price:
        exit_price = c_open * (1.0 - slippage_rate)
        exit_triggered = True
    elif c_low <= slot.stop_price:
        exit_price = slot.stop_price * (1.0 - slippage_rate)
        exit_triggered = True
    elif long_tp_mult > 0.0:
        if c_open >= slot.tp_price:
            exit_price = c_open * (1.0 - slippage_rate)
            exit_triggered = True
        elif c_high >= slot.tp_price:
            exit_price = slot.tp_price * (1.0 - slippage_rate)
            exit_triggered = True

    if (
        not exit_triggered
        and (not slot.scale_done)
        and long_scale_atr_mult > 0.0
        and scale_out_pct > 1e-12
        and scale_out_pct < 1.0 - 1e-12
    ):
        trig_px = slot.entry_price + long_scale_atr_mult * trail_atr
        if c_high >= trig_px:
            exit_px = trig_px * (1.0 - slippage_rate)
            partial_amt = slot.amount * scale_out_pct
            exit_fee_p = partial_amt * exit_px * fee_rate
            pnl_p = (exit_px - slot.entry_price) * partial_amt - exit_fee_p
            balance += partial_amt * slot.entry_price + pnl_p
            ef_part = slot.entry_fee_stored * (partial_amt / slot.amount)
            slot.entry_fee_stored -= ef_part
            trades_delta += 1
            slot.amount -= partial_amt
            slot.scale_done = True

    if slot.in_position and not exit_triggered:
        new_stop = slot.highest - (trail_atr * long_trail_mult)
        if new_stop > slot.stop_price:
            slot.stop_price = new_stop

    if slot.in_position and not exit_triggered:
        if time_stop_bars > 0 and (i - slot.entry_idx) >= time_stop_bars:
            exit_price = c_open * (1.0 - slippage_rate)
            exit_triggered = True

    if exit_triggered:
        pnl = (exit_price - slot.entry_price) * slot.amount
        pnl -= slot.amount * exit_price * fee_rate
        balance += (slot.amount * slot.entry_price) + pnl
        trades_delta += 1
        slot.in_position = False
        return slot, balance, trades_delta

    return slot, balance, trades_delta


def _try_open_long(
    *,
    i: int,
    n: int,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_p: np.ndarray,
    entry_upper: np.ndarray,
    long_entry_signal: np.ndarray,
    atr: np.ndarray,
    balance: float,
    fee_rate: float,
    slippage_rate: float,
    risk_per_trade: float,
    max_position_pct: float,
    long_atr_mult: float,
    long_tp_mult: float,
    warmup_bars: int,
    execution_start_idx: int,
    regime_risk_mult: Optional[np.ndarray] = None,
) -> Tuple[_SlotState, float, bool, int]:
    trades_delta = 0
    slot = _SlotState()
    if i < warmup_bars or i < execution_start_idx:
        return slot, balance, False, 0

    prev_i = i - 1 if i > 0 else 0
    if long_entry_signal[prev_i] < 0.5 or np.isnan(long_entry_signal[prev_i]):
        return slot, balance, False, 0

    c_open = float(open_p[i])
    c_high = float(high[i])
    c_low = float(low[i])

    if c_high <= float(entry_upper[prev_i]):
        return slot, balance, False, 0

    fill_price = max(c_open, float(entry_upper[prev_i])) * (1.0 + slippage_rate)
    prev_atr = float(atr[prev_i])
    if np.isnan(prev_atr) or prev_atr <= 0.0:
        return slot, balance, False, 0

    stop_price = fill_price - (prev_atr * long_atr_mult)
    stop_distance = abs(fill_price - stop_price)
    if stop_distance <= 1e-18:
        return slot, balance, False, 0

    rm = 1.0
    if regime_risk_mult is not None and len(regime_risk_mult) > prev_i:
        rm = float(regime_risk_mult[prev_i])
    if not np.isfinite(rm) or rm <= 0.0:
        rm = 1.0
    rm = float(np.clip(rm, 0.05, 1.0))

    risk_budget = balance * risk_per_trade * rm
    amount_from_risk = risk_budget / stop_distance
    max_notional = balance * max_position_pct
    amount_pos_cap = max_notional / fill_price
    max_affordable = (balance * 0.99) / (fill_price * (1.0 + fee_rate))
    amt = min(amount_from_risk, amount_pos_cap, max_affordable)

    if amt <= 0:
        return slot, balance, False, 0

    required_capital = amt * fill_price
    entry_fee = required_capital * fee_rate
    if balance < required_capital + entry_fee:
        return slot, balance, False, 0

    balance -= required_capital + entry_fee
    slot.entry_fee_stored = entry_fee
    slot.in_position = True
    slot.entry_price = fill_price
    slot.entry_idx = i
    slot.highest = fill_price
    slot.amount = amt
    slot.stop_price = stop_price
    slot.scale_done = False
    if long_tp_mult > 0.0:
        slot.tp_price = fill_price + (prev_atr * long_tp_mult)
    else:
        slot.tp_price = 0.0

    if c_low <= stop_price:
        intra_exit_price = stop_price * (1.0 - slippage_rate)
        pnl = (intra_exit_price - slot.entry_price) * slot.amount
        pnl -= slot.amount * intra_exit_price * fee_rate
        balance += (slot.amount * slot.entry_price) + pnl
        trades_delta += 1
        slot.in_position = False
        return slot, balance, False, trades_delta

    return slot, balance, True, trades_delta
