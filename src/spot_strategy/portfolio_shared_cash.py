"""
Shared-cash spot portfolio: one balance, bar-aligned symbols, optional top-N concurrent slots.

When max_concurrent_positions == 1, only one symbol holds a position at a time (capital rotation).
Exit/entry rules match engine_spot.backtest_loop_numba_spot per active slot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.opt_config import SLIPPAGE_GAMMA_BASE, SLIPPAGE_REFERENCE_ADV_KRW
from config.settings import SPOT_INITIAL_BALANCE, SPOT_SLIPPAGE_RATE, UPBIT_SPOT_TAKER_FEE_RATE

_logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class SharedCashResult:
    equity_curve: np.ndarray
    final_balance: float
    total_trades: int
    pnl_array: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    per_symbol_trades: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    per_symbol_wins: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    per_symbol_pnl: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))


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
    fractal_scale_done: bool = False


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
    allow_python_fallback: bool = True,
    concurrency_penalty_scale: float = 1.0,
) -> SharedCashResult:
    """
    Shared balance; per bar: exit all slots first, then enter ranked candidates into free slots.
    All symbol arrays must have length n.
    """
    if not symbols_ordered:
        return SharedCashResult(equity_curve=np.array([]), final_balance=initial_balance, total_trades=0)

    try:
        from src.spot_strategy.portfolio_shared_cash_numba import (
            run_packed_from_symbol_arrays,
            use_numba_shared_cash,
        )

        if use_numba_shared_cash():
            return run_packed_from_symbol_arrays(
                symbol_arrays,
                symbols_ordered,
                params,
                initial_balance=initial_balance,
                max_concurrent_positions=max_concurrent_positions,
                rank_scores=rank_scores,
                warmup_bars=warmup_bars,
                execution_start_idx=execution_start_idx,
                concurrency_penalty_scale=float(concurrency_penalty_scale),
            )
    except Exception as exc:
        if not allow_python_fallback:
            raise
        _logger.warning("Shared-cash Numba path failed; using Python loop: %s", exc, exc_info=True)

    n = int(len(symbol_arrays[symbols_ordered[0]]["close"]))
    for s in symbols_ordered:
        if len(symbol_arrays[s]["close"]) != n:
            raise ValueError("Shared-cash requires identical bar counts per symbol.")

    fee_rate = float(UPBIT_SPOT_TAKER_FEE_RATE)
    slippage_rate = float(SPOT_SLIPPAGE_RATE)
    risk_per_trade = float(params.get("RISK_PER_TRADE", 0.015))
    max_position_pct = float(params.get("MAX_POSITION_PCT", 0.25))
    cap_coin = params.get("MAX_CAP_PER_COIN")
    if cap_coin is not None:
        max_position_pct = min(max_position_pct, float(cap_coin))
    long_atr_mult = float(params.get("LONG_ATR_MULT", 3.0))
    long_trail_mult = float(params.get("LONG_TRAIL_MULT", 3.0))
    long_tp_mult = float(params.get("LONG_TP_MULT", 5.0))
    long_scale_atr_mult = float(params.get("LONG_SCALE_ATR_MULT", 0.0))
    scale_out_pct = float(params.get("SCALE_OUT_PCT", 0.0))
    fractal_scale_out_ratio = float(params.get("SCALE_OUT_RATIO", 0.5))
    time_stop_bars = int(params.get("TIME_STOP_BARS", 0))
    tp_lock_mult = float(params.get("TP_LOCK_ATR_MULT", 3.0))
    long_trail_lock_mult = float(params.get("LONG_TRAIL_LOCK_MULT", 1.5))

    max_slots = max(1, min(int(max_concurrent_positions), len(symbols_ordered)))
    slots: List[_SlotState] = [_SlotState() for _ in range(max_slots)]
    balance = float(initial_balance)
    equity_curve = np.zeros(n, dtype=np.float64)
    total_trades = 0

    n_sym = len(symbols_ordered)
    per_symbol_trades = np.zeros(n_sym, dtype=np.int32)
    per_symbol_wins = np.zeros(n_sym, dtype=np.int32)
    per_symbol_pnl = np.zeros(n_sym, dtype=np.float64)
    pnl_list = []

    n_sym = len(symbols_ordered)
    sym_cooldown = np.zeros(n_sym, dtype=np.int32)
    sym_cooldown_skip = np.zeros(n_sym, dtype=np.bool_)
    kill_cd_bars = int(params.get("KILL_COOLDOWN_BARS", 6))
    delta_gate = float(params.get("DELTA_GATE", 0.08))
    last_risk_pct_sym = [0.0] * n_sym
    gamma_base = float(params.get("SLIPPAGE_GAMMA_BASE", SLIPPAGE_GAMMA_BASE))
    ref_adv_py = max(float(params.get("SLIPPAGE_REFERENCE_ADV_KRW", SLIPPAGE_REFERENCE_ADV_KRW)), 1.0)
    pen_scale_py = float(concurrency_penalty_scale)
    adv_by_si = [0.0] * n_sym
    for si_a, sym_a in enumerate(symbols_ordered):
        a0 = symbol_arrays[sym_a]
        v0 = a0["volume"]
        c0 = a0["close"]
        lb0 = min(6, len(c0))
        adv_by_si[si_a] = float(np.mean(v0[-lb0:] * c0[-lb0:])) if lb0 > 0 else 1.0

    for i in range(n):
        # --- exits and in-position management for each occupied slot ---
        for slot in slots:
            if slot.sym_idx < 0 or not slot.in_position:
                continue
            sym = symbols_ordered[slot.sym_idx]
            arr = symbol_arrays[sym]
            ks = arr.get("kill_signal")
            fh = arr.get("fractal_high_flag")
            slot, balance, td, spnl = _process_in_position_bar(
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
                long_trail_lock_mult=long_trail_lock_mult,
                tp_lock_mult=tp_lock_mult,
                long_tp_mult=long_tp_mult,
                long_scale_atr_mult=long_scale_atr_mult,
                scale_out_pct=scale_out_pct,
                fractal_scale_out_ratio=fractal_scale_out_ratio,
                fractal_high_flag=fh,
                time_stop_bars=time_stop_bars,
                warmup_bars=warmup_bars,
                execution_start_idx=execution_start_idx,
                kill_signal=ks,
                sym_idx=slot.sym_idx,
                sym_cooldown=sym_cooldown,
                sym_cooldown_skip=sym_cooldown_skip,
                kill_cooldown_bars=kill_cd_bars,
                bb_upper=arr.get("bb_upper"),
                trail_tighten=arr.get("trail_tighten_flag"),
                regime_risk_mult=arr.get("regime_risk_mult"),
            )
            total_trades += td
            if td > 0:
                ssi = slot.sym_idx
                per_symbol_trades[ssi] += td
                per_symbol_pnl[ssi] += spnl
                if spnl > 0:
                    per_symbol_wins[ssi] += 1
                pnl_list.append(spnl)
            if not slot.in_position:
                done_si = int(slot.sym_idx)
                if 0 <= done_si < n_sym:
                    last_risk_pct_sym[done_si] = 0.0
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
            if sym_cooldown[si] > 0:
                continue
            if any(sl.sym_idx == si and sl.in_position for sl in slots):
                continue
            arr = symbol_arrays[sym]
            les = float(arr["long_entry_signal"][prev_i])
            if les < 0.5 or np.isnan(les):
                continue
            c_high = float(arr["high"][i])
            eu_prev = float(arr["entry_upper"][prev_i])
            if eu_prev < 1.0:
                candidates.append(si)
            elif c_high > eu_prev:
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

        n_new_e = min(len(candidates), len(free_slots))
        excess_e = max(0, n_new_e - 1)
        excess_pow = float(excess_e) ** 1.5

        for si in candidates:
            if not free_slots:
                break
                
            # CIRCUIT BREAKER: Stop new entries if capital drops below 30% (Wealth Preservation).
            if balance < initial_balance * 0.3:
                 break
                 
            sym = symbols_ordered[si]
            arr = symbol_arrays[sym]
            j = free_slots[0]
            rr = arr.get("regime_risk_mult")
            gk = arr.get("garch_kelly_f")
            adv_r = max(adv_by_si[si] / ref_adv_py, 0.01)
            gamma_sym = gamma_base / float(np.sqrt(adv_r))
            adj_slip = slippage_rate * (1.0 + pen_scale_py * gamma_sym * excess_pow)
            st, balance, opened, td, spnl = _try_open_long(
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
                slippage_rate=adj_slip,
                risk_per_trade=risk_per_trade,
                max_position_pct=max_position_pct,
                long_atr_mult=long_atr_mult,
                long_tp_mult=long_tp_mult,
                warmup_bars=warmup_bars,
                execution_start_idx=execution_start_idx,
                regime_risk_mult=rr,
                garch_kelly_f=gk,
                delta_gate=delta_gate,
                last_risk_pct_ref=last_risk_pct_sym,
                sym_idx=si,
            )
            if td > 0:
                total_trades += td
                per_symbol_trades[si] += td
                per_symbol_pnl[si] += spnl
                if spnl > 0:
                    per_symbol_wins[si] += 1
                pnl_list.append(spnl)
                
                if not opened:
                    # Intra-SL: Churning Prevention
                    sym_cooldown[si] = 1 
                    sym_cooldown_skip[si] = True
            if opened:
                st.sym_idx = si
                slots[j] = st
                free_slots.pop(0)

        for sii in range(n_sym):
            if sym_cooldown_skip[sii]:
                sym_cooldown_skip[sii] = False
            elif sym_cooldown[sii] > 0:
                sym_cooldown[sii] -= 1

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
            done_si = slot.sym_idx
            per_symbol_trades[done_si] += 1
            per_symbol_pnl[done_si] += pnl
            if pnl > 0:
                per_symbol_wins[done_si] += 1
            pnl_list.append(pnl)
            
            slot.in_position = False
            slot.sym_idx = -1
        equity_curve[last_idx] = balance

    return SharedCashResult(
        equity_curve=equity_curve, 
        final_balance=float(balance), 
        total_trades=total_trades,
        pnl_array=np.array(pnl_list, dtype=np.float64),
        per_symbol_trades=per_symbol_trades,
        per_symbol_wins=per_symbol_wins,
        per_symbol_pnl=per_symbol_pnl,
    )


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
    long_trail_lock_mult: float,
    tp_lock_mult: float,
    long_tp_mult: float,
    long_scale_atr_mult: float,
    scale_out_pct: float,
    fractal_scale_out_ratio: float,
    fractal_high_flag: Optional[np.ndarray],
    time_stop_bars: int,
    warmup_bars: int,
    execution_start_idx: int,
    kill_signal: Optional[np.ndarray] = None,
    sym_idx: int = -1,
    sym_cooldown: Optional[np.ndarray] = None,
    sym_cooldown_skip: Optional[np.ndarray] = None,
    kill_cooldown_bars: int = 6,
    bb_upper: Optional[np.ndarray] = None,
    trail_tighten: Optional[np.ndarray] = None,
    regime_risk_mult: Optional[np.ndarray] = None,
) -> Tuple[_SlotState, float, int, float]:
    trades_delta = 0
    pnl_out = 0.0
    if i < warmup_bars or i < execution_start_idx:
        return slot, balance, 0, 0.0

    c_open = float(open_p[i])
    c_high = float(high[i])
    c_low = float(low[i])

    if (
        i > 0
        and kill_signal is not None
        and len(kill_signal) > (i - 1)
        and float(kill_signal[i - 1]) > 0.5
        and slot.in_position
    ):
        exit_price = c_open * (1.0 - slippage_rate)
        pnl = (exit_price - slot.entry_price) * slot.amount
        pnl -= slot.amount * exit_price * fee_rate
        balance += slot.amount * slot.entry_price + pnl
        trades_delta += 1
        pnl_out = pnl
        slot.in_position = False
        if sym_cooldown is not None and sym_cooldown_skip is not None and 0 <= sym_idx < len(sym_cooldown):
            sym_cooldown[sym_idx] = int(kill_cooldown_bars)
            sym_cooldown_skip[sym_idx] = True
            if last_risk_pct_sym is not None and 0 <= sym_idx < len(last_risk_pct_sym):
                last_risk_pct_sym[sym_idx] = 0.0
        return slot, balance, trades_delta, pnl_out

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
        (not exit_triggered)
        and bb_upper is not None
        and len(bb_upper) > i
        and np.isfinite(bb_upper[i])
        and float(bb_upper[i]) < 1e18
        and c_high >= float(bb_upper[i])
    ):
        exit_price = float(bb_upper[i]) * (1.0 - slippage_rate)
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
            pnl_out += pnl_p
            slot.amount -= partial_amt
            slot.scale_done = True

    if (
        not exit_triggered
        and slot.in_position
        and (not slot.fractal_scale_done)
        and fractal_high_flag is not None
        and len(fractal_high_flag) > i
        and float(fractal_high_flag[i]) > 0.5
        and slot.amount > 0.0
        and fractal_scale_out_ratio > 1e-12
        and fractal_scale_out_ratio < 1.0 - 1e-12
    ):
        cap_amt = slot.amount * 0.99
        partial_f = slot.amount * fractal_scale_out_ratio
        if partial_f > cap_amt:
            partial_f = cap_amt
        if partial_f > 0.0:
            exit_px_f = c_open * (1.0 - slippage_rate)
            exit_fee_f = partial_f * exit_px_f * fee_rate
            pnl_f = (exit_px_f - slot.entry_price) * partial_f - exit_fee_f
            balance += partial_f * slot.entry_price + pnl_f
            ef_part_f = slot.entry_fee_stored * (partial_f / slot.amount)
            slot.entry_fee_stored -= ef_part_f
            slot.amount -= partial_f
            slot.fractal_scale_done = True
            pnl_out += pnl_f
        if slot.amount < 1e-12 and slot.in_position:
            dust_px = c_open * (1.0 - slippage_rate)
            pnl_d = (dust_px - slot.entry_price) * slot.amount
            pnl_d -= slot.amount * dust_px * fee_rate
            balance += slot.amount * slot.entry_price + pnl_d
            trades_delta += 1
            pnl_out += pnl_d
            slot.in_position = False
            return slot, balance, trades_delta, pnl_out

    if slot.in_position and not exit_triggered:
        ei2 = int(slot.entry_idx)
        pos_atr = float(atr[ei2]) if ei2 < n else trail_atr
        if pos_atr <= 0.0 or np.isnan(pos_atr):
            pos_atr = trail_atr
        dist = float(slot.highest - slot.entry_price)
        current_trail_mult = long_trail_mult
        if (
            trail_tighten is not None
            and len(trail_tighten) > i
            and float(trail_tighten[i]) > 0.5
        ):
            current_trail_mult = long_trail_lock_mult
        elif dist > pos_atr * tp_lock_mult:
            current_trail_mult = long_trail_lock_mult
        new_stop = float(slot.highest) - pos_atr * current_trail_mult
        if new_stop > slot.stop_price:
            slot.stop_price = new_stop

    if slot.in_position and not exit_triggered:
        # Adaptive Time-Stop: Shorten if regime is bearish/choppy
        adaptive_stop = time_stop_bars
        if regime_risk_mult is not None and len(regime_risk_mult) > i:
            rrm = float(regime_risk_mult[i])
            if rrm < 0.4:
                adaptive_stop = max(1, time_stop_bars // 2)
        
        if time_stop_bars > 0 and (i - slot.entry_idx) >= adaptive_stop:
            exit_price = c_open * (1.0 - slippage_rate)
            exit_triggered = True

    if exit_triggered:
        pnl = (exit_price - slot.entry_price) * slot.amount
        pnl -= slot.amount * exit_price * fee_rate
        balance += (slot.amount * slot.entry_price) + pnl
        trades_delta += 1
        pnl_out += pnl
        slot.in_position = False
        return slot, balance, trades_delta, pnl_out

    return slot, balance, trades_delta, pnl_out


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
    garch_kelly_f: Optional[np.ndarray] = None,
    delta_gate: float = 0.08,
    last_risk_pct_ref: Optional[List[float]] = None,
    sym_idx: int = 0,
) -> Tuple[_SlotState, float, bool, int, float]:
    trades_delta = 0
    pnl_out = 0.0
    slot = _SlotState()
    if i < warmup_bars or i < execution_start_idx:
        return slot, balance, False, 0, 0.0

    prev_i = i - 1 if i > 0 else 0
    if long_entry_signal[prev_i] < 0.5 or np.isnan(long_entry_signal[prev_i]):
        return slot, balance, False, 0, 0.0

    c_open = float(open_p[i])
    c_high = float(high[i])
    c_low = float(low[i])

    eu_prev = float(entry_upper[prev_i])
    if eu_prev < 1.0:
        fill_price = c_open * (1.0 + slippage_rate)
    else:
        if c_high <= eu_prev:
            return slot, balance, False, 0, 0.0
        fill_price = max(c_open, eu_prev) * (1.0 + slippage_rate)
    prev_atr = float(atr[prev_i])
    if np.isnan(prev_atr) or prev_atr <= 0.0:
        return slot, balance, False, 0, 0.0

    stop_price = fill_price - (prev_atr * long_atr_mult)
    stop_distance = abs(fill_price - stop_price)
    if stop_distance <= 1e-18:
        return slot, balance, False, 0, 0.0

    rm = 1.0
    if regime_risk_mult is not None and len(regime_risk_mult) > prev_i:
        rm = float(regime_risk_mult[prev_i])
    if not np.isfinite(rm):
        return slot, balance, False, 0, 0.0
    rm = float(np.clip(rm, 0.05, 1.0))

    gk = 1.0
    if garch_kelly_f is not None and len(garch_kelly_f) > prev_i:
        gk = float(garch_kelly_f[prev_i])
    if not np.isfinite(gk) or gk <= 0.0:
        gk = 1.0
    gk = float(np.clip(gk, 0.05, 1.0))

    eff = float(np.clip(rm * gk, 0.05, 1.0))
    new_risk_pct = risk_per_trade * eff
    if (
        last_risk_pct_ref is not None
        and 0 <= sym_idx < len(last_risk_pct_ref)
        and last_risk_pct_ref[sym_idx] > 1e-12
        and abs(new_risk_pct - last_risk_pct_ref[sym_idx]) < delta_gate
    ):
        return slot, balance, False, 0, 0.0

    risk_budget = balance * new_risk_pct
    amount_from_risk = risk_budget / stop_distance
    max_notional = balance * max_position_pct
    amount_pos_cap = max_notional / fill_price
    max_affordable = (balance * 0.99) / (fill_price * (1.0 + fee_rate))
    amt = min(amount_from_risk, amount_pos_cap, max_affordable)

    if amt <= 0:
        return slot, balance, False, 0, 0.0

    required_capital = amt * fill_price
    entry_fee = required_capital * fee_rate
    if balance < required_capital + entry_fee:
        return slot, balance, False, 0, 0.0

    balance -= required_capital + entry_fee
    slot.entry_fee_stored = entry_fee
    slot.in_position = True
    slot.entry_price = fill_price
    slot.entry_idx = i
    slot.highest = fill_price
    slot.amount = amt
    slot.stop_price = stop_price
    slot.fractal_scale_done = False
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
        pnl_out = pnl
        slot.in_position = False
        return slot, balance, False, trades_delta, pnl_out

    if last_risk_pct_ref is not None and 0 <= sym_idx < len(last_risk_pct_ref):
        last_risk_pct_ref[sym_idx] = new_risk_pct

    return slot, balance, True, trades_delta, pnl_out


def _warn_if_numba_shared_cash_smoke_fails() -> None:
    """One-time import check: warn if Numba is expected but JIT compilation fails."""
    try:
        from src.spot_strategy.portfolio_shared_cash_numba import use_numba_shared_cash

        if not use_numba_shared_cash():
            _logger.warning(
                "OPT_SPOT_SHARED_CASH_NUMBA is disabled; shared-cash may use slow Python loop."
            )
            return
        from numba import njit

        @njit(cache=False)
        def _smoke_add(x: float) -> float:
            return x + 1.0

        _ = float(_smoke_add(1.0))
    except Exception as exc:
        _logger.warning(
            "Shared-cash Numba JIT smoke test failed; optimization may fall back to Python loop: %s",
            exc,
            exc_info=True,
        )


_warn_if_numba_shared_cash_smoke_fails()
