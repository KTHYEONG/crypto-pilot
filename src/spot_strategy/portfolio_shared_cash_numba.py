"""
Numba-accelerated shared-cash portfolio loop (packed 2D arrays).
Mirrors run_shared_cash_multi_symbol in portfolio_shared_cash.py.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit

from config.opt_config import SLIPPAGE_GAMMA_BASE, SLIPPAGE_REFERENCE_ADV_KRW
from config.settings import SPOT_INITIAL_BALANCE, SPOT_SLIPPAGE_RATE, UPBIT_SPOT_TAKER_FEE_RATE


@njit(cache=True)
def _equity_at_bar(
    balance: float,
    slot_sym: np.ndarray,
    slot_in: np.ndarray,
    slot_entry_price: np.ndarray,
    slot_amount: np.ndarray,
    close: np.ndarray,
    i: int,
    max_slots: int,
) -> float:
    eq = balance
    for sj in range(max_slots):
        if slot_sym[sj] < 0 or not slot_in[sj]:
            continue
        si = slot_sym[sj]
        c_price = close[si, i]
        unreal = (c_price - slot_entry_price[sj]) * slot_amount[sj]
        eq += slot_amount[sj] * slot_entry_price[sj] + unreal
    return eq


@njit(cache=True)
def _run_shared_cash_packed_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_p: np.ndarray,
    atr: np.ndarray,
    long_entry_signal: np.ndarray,
    entry_upper: np.ndarray,
    regime_risk: np.ndarray,
    garch_kelly: np.ndarray,
    kill_signal: np.ndarray,
    rank_scores: np.ndarray,
    fractal_high_flag: np.ndarray,
    bb_upper: np.ndarray,
    trail_tighten: np.ndarray,
    adv_symbol_krw: np.ndarray,
    *,
    initial_balance: float,
    fee_rate: float,
    slippage_rate: float,
    risk_per_trade: float,
    max_position_pct: float,
    long_atr_mult: float,
    long_trail_mult: float,
    long_trail_lock_mult: float,
    long_tp_mult: float,
    tp_lock_mult: float,
    long_scale_atr_mult: float,
    scale_out_pct: float,
    fractal_scale_out_ratio: float,
    time_stop_bars: int,
    warmup_bars: int,
    execution_start_idx: int,
    kill_cd_bars: int,
    delta_gate: float,
    max_slots: int,
    slippage_gamma_base: float,
    slippage_ref_adv_krw: float,
    concurrency_penalty_scale: float,
) -> Tuple[np.ndarray, float, int]:
    n_sym, n = close.shape
    balance = initial_balance
    equity_curve = np.zeros(n, dtype=np.float64)
    total_trades = 0

    slot_sym = np.full(max_slots, -1, dtype=np.int32)
    slot_in = np.zeros(max_slots, dtype=np.bool_)
    slot_entry_price = np.zeros(max_slots, dtype=np.float64)
    slot_entry_idx = np.zeros(max_slots, dtype=np.int32)
    slot_amount = np.zeros(max_slots, dtype=np.float64)
    slot_entry_fee = np.zeros(max_slots, dtype=np.float64)
    slot_stop = np.zeros(max_slots, dtype=np.float64)
    slot_tp = np.zeros(max_slots, dtype=np.float64)
    slot_highest = np.zeros(max_slots, dtype=np.float64)
    slot_scale_done = np.zeros(max_slots, dtype=np.bool_)
    slot_fractal_scale_done = np.zeros(max_slots, dtype=np.bool_)

    sym_cooldown = np.zeros(n_sym, dtype=np.int32)
    sym_cooldown_skip = np.zeros(n_sym, dtype=np.bool_)
    last_risk_pct_sym = np.zeros(n_sym, dtype=np.float64)

    wu = warmup_bars
    if execution_start_idx > wu:
        wu = execution_start_idx

    for i in range(n):
        if i >= wu:
            for sj in range(max_slots):
                if slot_sym[sj] < 0 or not slot_in[sj]:
                    continue
                si = slot_sym[sj]
                ks = kill_signal[si, i - 1] if i > 0 else 0.0

                c_open = open_p[si, i]
                c_high = high[si, i]
                c_low = low[si, i]

                if ks > 0.5 and slot_in[sj]:
                    exit_price = c_open * (1.0 - slippage_rate)
                    pnl = (exit_price - slot_entry_price[sj]) * slot_amount[sj]
                    pnl -= slot_amount[sj] * exit_price * fee_rate
                    balance += slot_amount[sj] * slot_entry_price[sj] + pnl
                    total_trades += 1
                    slot_in[sj] = False
                    if 0 <= si < n_sym:
                        sym_cooldown[si] = kill_cd_bars
                        sym_cooldown_skip[si] = True
                    slot_sym[sj] = -1
                    continue

                trail_atr = atr[si, i]
                if trail_atr <= 0.0 or np.isnan(trail_atr):
                    ei0 = slot_entry_idx[sj]
                    if ei0 < n:
                        trail_atr = atr[si, ei0]
                    else:
                        trail_atr = 1e-9

                exit_triggered = False
                ex_px = 0.0

                if c_high > slot_highest[sj]:
                    slot_highest[sj] = c_high

                if c_open <= slot_stop[sj]:
                    ex_px = c_open * (1.0 - slippage_rate)
                    exit_triggered = True
                elif c_low <= slot_stop[sj]:
                    ex_px = slot_stop[sj] * (1.0 - slippage_rate)
                    exit_triggered = True
                elif long_tp_mult > 0.0:
                    if c_open >= slot_tp[sj]:
                        ex_px = c_open * (1.0 - slippage_rate)
                        exit_triggered = True
                    elif c_high >= slot_tp[sj]:
                        ex_px = slot_tp[sj] * (1.0 - slippage_rate)
                        exit_triggered = True
                bub = bb_upper[si, i]
                if (
                    (not exit_triggered)
                    and np.isfinite(bub)
                    and bub < 1e18
                    and c_high >= bub
                ):
                    ex_px = bub * (1.0 - slippage_rate)
                    exit_triggered = True

                if (
                    (not exit_triggered)
                    and (not slot_scale_done[sj])
                    and long_scale_atr_mult > 0.0
                    and scale_out_pct > 1e-12
                    and scale_out_pct < 1.0 - 1e-12
                ):
                    trig_px = slot_entry_price[sj] + long_scale_atr_mult * trail_atr
                    if c_high >= trig_px:
                        exit_px = trig_px * (1.0 - slippage_rate)
                        partial_amt = slot_amount[sj] * scale_out_pct
                        exit_fee_p = partial_amt * exit_px * fee_rate
                        pnl_p = (exit_px - slot_entry_price[sj]) * partial_amt - exit_fee_p
                        balance += partial_amt * slot_entry_price[sj] + pnl_p
                        ef_part = slot_entry_fee[sj] * (partial_amt / slot_amount[sj])
                        slot_entry_fee[sj] -= ef_part
                        total_trades += 1
                        slot_amount[sj] -= partial_amt
                        slot_scale_done[sj] = True

                if (
                    (not exit_triggered)
                    and slot_in[sj]
                    and (not slot_fractal_scale_done[sj])
                    and fractal_high_flag[si, i] > 0.5
                    and slot_amount[sj] > 0.0
                    and fractal_scale_out_ratio > 1e-12
                    and fractal_scale_out_ratio < 1.0 - 1e-12
                ):
                    cap_amt = slot_amount[sj] * 0.99
                    partial_f = slot_amount[sj] * fractal_scale_out_ratio
                    if partial_f > cap_amt:
                        partial_f = cap_amt
                    if partial_f > 0.0:
                        exit_px_f = c_open * (1.0 - slippage_rate)
                        exit_fee_f = partial_f * exit_px_f * fee_rate
                        pnl_f = (exit_px_f - slot_entry_price[sj]) * partial_f - exit_fee_f
                        balance += partial_f * slot_entry_price[sj] + pnl_f
                        ef_part_f = slot_entry_fee[sj] * (partial_f / slot_amount[sj])
                        slot_entry_fee[sj] -= ef_part_f
                        slot_amount[sj] -= partial_f
                        slot_fractal_scale_done[sj] = True
                    if slot_amount[sj] < 1e-12 and slot_in[sj]:
                        dust_px = c_open * (1.0 - slippage_rate)
                        pnl = (dust_px - slot_entry_price[sj]) * slot_amount[sj]
                        pnl -= slot_amount[sj] * dust_px * fee_rate
                        balance += slot_amount[sj] * slot_entry_price[sj] + pnl
                        total_trades += 1
                        slot_in[sj] = False
                        if 0 <= si < n_sym:
                            last_risk_pct_sym[si] = 0.0
                        slot_sym[sj] = -1
                        continue

                if slot_in[sj] and (not exit_triggered):
                    ei2 = slot_entry_idx[sj]
                    pos_atr = atr[si, ei2] if ei2 < n else trail_atr
                    if pos_atr <= 0.0 or np.isnan(pos_atr):
                        pos_atr = trail_atr
                    dist = slot_highest[sj] - slot_entry_price[sj]
                    cur_tm = long_trail_mult
                    if trail_tighten[si, i] > 0.5:
                        cur_tm = long_trail_lock_mult
                    elif dist > pos_atr * tp_lock_mult:
                        cur_tm = long_trail_lock_mult
                    new_stop = slot_highest[sj] - pos_atr * cur_tm
                    if new_stop > slot_stop[sj]:
                        slot_stop[sj] = new_stop

                if slot_in[sj] and (not exit_triggered):
                    if time_stop_bars > 0 and (i - slot_entry_idx[sj]) >= time_stop_bars:
                        ex_px = c_open * (1.0 - slippage_rate)
                        exit_triggered = True

                if exit_triggered:
                    pnl = (ex_px - slot_entry_price[sj]) * slot_amount[sj]
                    pnl -= slot_amount[sj] * ex_px * fee_rate
                    balance += slot_amount[sj] * slot_entry_price[sj] + pnl
                    total_trades += 1
                    slot_in[sj] = False
                    if 0 <= si < n_sym:
                        last_risk_pct_sym[si] = 0.0
                    slot_sym[sj] = -1

        if i < wu:
            equity_curve[i] = _equity_at_bar(
                balance, slot_sym, slot_in, slot_entry_price, slot_amount, close, i, max_slots
            )
            continue

        prev_i = i - 1
        if prev_i < 0:
            prev_i = 0

        n_free = 0
        free_idx = np.zeros(max_slots, dtype=np.int32)
        for sj in range(max_slots):
            if slot_sym[sj] < 0:
                free_idx[n_free] = sj
                n_free += 1

        if n_free == 0:
            equity_curve[i] = _equity_at_bar(
                balance, slot_sym, slot_in, slot_entry_price, slot_amount, close, i, max_slots
            )
            for sii in range(n_sym):
                if sym_cooldown_skip[sii]:
                    sym_cooldown_skip[sii] = False
                elif sym_cooldown[sii] > 0:
                    sym_cooldown[sii] -= 1
            continue

        n_cand = 0
        cand_si = np.zeros(n_sym, dtype=np.int32)
        for si in range(n_sym):
            if sym_cooldown[si] > 0:
                continue
            occupied = False
            for sj in range(max_slots):
                if slot_sym[sj] == si and slot_in[sj]:
                    occupied = True
                    break
            if occupied:
                continue
            les = long_entry_signal[si, prev_i]
            if les < 0.5 or np.isnan(les):
                continue
            eu_prev = entry_upper[si, prev_i]
            if eu_prev < 1.0:
                cand_si[n_cand] = si
                n_cand += 1
            elif high[si, i] > eu_prev:
                cand_si[n_cand] = si
                n_cand += 1

        if n_cand == 0:
            equity_curve[i] = _equity_at_bar(
                balance, slot_sym, slot_in, slot_entry_price, slot_amount, close, i, max_slots
            )
            for sii in range(n_sym):
                if sym_cooldown_skip[sii]:
                    sym_cooldown_skip[sii] = False
                elif sym_cooldown[sii] > 0:
                    sym_cooldown[sii] -= 1
            continue

        for a in range(n_cand - 1):
            best = a
            best_v = rank_scores[cand_si[a], prev_i]
            if not np.isfinite(best_v):
                best_v = 0.0
            for b in range(a + 1, n_cand):
                v = rank_scores[cand_si[b], prev_i]
                if not np.isfinite(v):
                    v = 0.0
                if v > best_v:
                    best = b
                    best_v = v
            if best != a:
                tmp = cand_si[a]
                cand_si[a] = cand_si[best]
                cand_si[best] = tmp

        n_new_entries = n_cand
        if n_new_entries > n_free:
            n_new_entries = n_free
        excess_entries = n_new_entries - 1
        if excess_entries < 0:
            excess_entries = 0
        excess_f = float(excess_entries)
        ref_adv = slippage_ref_adv_krw
        if ref_adv < 1.0:
            ref_adv = 1.0
        gamma0 = slippage_gamma_base
        pen_scale = concurrency_penalty_scale

        fp = 0
        free_ptr = 0
        while fp < n_cand and free_ptr < n_free:
            si = cand_si[fp]
            fp += 1
            sj = free_idx[free_ptr]

            c_open = open_p[si, i]
            c_high = high[si, i]
            c_low = low[si, i]
            prev_i2 = prev_i
            if long_entry_signal[si, prev_i2] < 0.5 or np.isnan(long_entry_signal[si, prev_i2]):
                continue
            eu_p = entry_upper[si, prev_i2]
            adv_r = adv_symbol_krw[si] / ref_adv
            if adv_r < 0.01:
                adv_r = 0.01
            gamma_sym = gamma0 / np.sqrt(adv_r)
            adj_slip = slippage_rate * (1.0 + pen_scale * gamma_sym * (excess_f**1.5))
            if eu_p < 1.0:
                fill_price = c_open * (1.0 + adj_slip)
            else:
                if c_high <= eu_p:
                    continue
                fill_price = max(c_open, eu_p) * (1.0 + adj_slip)
            prev_atr = atr[si, prev_i2]
            if np.isnan(prev_atr) or prev_atr <= 0.0:
                continue
            stop_price = fill_price - prev_atr * long_atr_mult
            stop_dist = abs(fill_price - stop_price)
            if stop_dist <= 1e-18:
                continue
            rm = regime_risk[si, prev_i2]
            if not np.isfinite(rm):
                continue
            if rm < 0.05:
                rm = 0.05
            if rm > 1.0:
                rm = 1.0
            gk = garch_kelly[si, prev_i2]
            if not np.isfinite(gk) or gk <= 0.0:
                gk = 1.0
            if gk < 0.05:
                gk = 0.05
            if gk > 1.0:
                gk = 1.0
            eff = rm * gk
            if eff < 0.05:
                eff = 0.05
            if eff > 1.0:
                eff = 1.0
            new_risk_pct = risk_per_trade * eff
            if last_risk_pct_sym[si] > 1e-12 and abs(new_risk_pct - last_risk_pct_sym[si]) < delta_gate:
                continue

            risk_budget = balance * new_risk_pct
            amt_from_risk = risk_budget / stop_dist
            max_notional = balance * max_position_pct
            amt_pos_cap = max_notional / fill_price
            max_aff = balance * 0.99 / (fill_price * (1.0 + fee_rate))
            amt = amt_from_risk
            if amt_pos_cap < amt:
                amt = amt_pos_cap
            if max_aff < amt:
                amt = max_aff
            if amt <= 0:
                continue
            req_cap = amt * fill_price
            if req_cap < 5000.0:
                continue
            entry_fee = req_cap * fee_rate
            total_required = req_cap + entry_fee
            if total_required <= 0.0:
                continue
            # CIRCUIT BREAKER: If balance drops below 30% of initial capital, stop opening new positions (Wealth Preservation).
            if balance < initial_balance * 0.3:
                break
                
            if balance < total_required:
                if balance <= 0.0:
                    continue
                scale = balance / total_required
                amt *= scale
                req_cap = amt * fill_price
                entry_fee = req_cap * fee_rate
                total_required = req_cap + entry_fee
            if amt <= 0.0 or req_cap < 5000.0:
                continue
            balance -= total_required
            slot_entry_fee[sj] = entry_fee
            slot_in[sj] = True
            slot_entry_price[sj] = fill_price
            slot_entry_idx[sj] = i
            slot_highest[sj] = fill_price
            slot_amount[sj] = amt
            slot_stop[sj] = stop_price
            slot_scale_done[sj] = False
            slot_fractal_scale_done[sj] = False
            if long_tp_mult > 0.0:
                slot_tp[sj] = fill_price + prev_atr * long_tp_mult
            else:
                slot_tp[sj] = 0.0
            slot_sym[sj] = si
            last_risk_pct_sym[si] = new_risk_pct

            if c_low <= stop_price:
                intra_exit = stop_price * (1.0 - slippage_rate)
                pnl = (intra_exit - slot_entry_price[sj]) * slot_amount[sj]
                pnl -= slot_amount[sj] * intra_exit * fee_rate
                balance += slot_amount[sj] * slot_entry_price[sj] + pnl
                total_trades += 1
                slot_in[sj] = False
                slot_sym[sj] = -1
            else:
                free_ptr += 1

        equity_curve[i] = _equity_at_bar(
            balance, slot_sym, slot_in, slot_entry_price, slot_amount, close, i, max_slots
        )

        for sii in range(n_sym):
            if sym_cooldown_skip[sii]:
                sym_cooldown_skip[sii] = False
            elif sym_cooldown[sii] > 0:
                sym_cooldown[sii] -= 1

    last_idx = n - 1
    if n > 0:
        for sj in range(max_slots):
            if slot_sym[sj] < 0 or not slot_in[sj]:
                continue
            si = slot_sym[sj]
            c_last = close[si, last_idx]
            exit_price = c_last * (1.0 - slippage_rate)
            pnl = (exit_price - slot_entry_price[sj]) * slot_amount[sj]
            pnl -= slot_amount[sj] * exit_price * fee_rate
            balance += slot_amount[sj] * slot_entry_price[sj] + pnl
            total_trades += 1
            slot_in[sj] = False
            slot_sym[sj] = -1
        equity_curve[last_idx] = balance

    return equity_curve, balance, total_trades


def run_packed_from_symbol_arrays(
    symbol_arrays: Dict[str, Dict[str, np.ndarray]],
    symbols_ordered: List[str],
    params: Dict[str, object],
    *,
    initial_balance: float,
    max_concurrent_positions: int,
    rank_scores: Optional[Dict[str, np.ndarray]],
    warmup_bars: int,
    execution_start_idx: int,
    concurrency_penalty_scale: float = 1.0,
) -> Tuple[np.ndarray, float, int]:
    """Build packed arrays and run numba kernel."""
    n = len(symbol_arrays[symbols_ordered[0]]["close"])
    n_sym = len(symbols_ordered)
    close = np.empty((n_sym, n), dtype=np.float64)
    high = np.empty((n_sym, n), dtype=np.float64)
    low = np.empty((n_sym, n), dtype=np.float64)
    open_p = np.empty((n_sym, n), dtype=np.float64)
    atr = np.empty((n_sym, n), dtype=np.float64)
    long_entry_signal = np.empty((n_sym, n), dtype=np.float64)
    entry_upper = np.empty((n_sym, n), dtype=np.float64)
    regime_risk = np.ones((n_sym, n), dtype=np.float64)
    garch_kelly = np.ones((n_sym, n), dtype=np.float64)
    kill_signal = np.zeros((n_sym, n), dtype=np.float64)
    rank_scores_m = np.zeros((n_sym, n), dtype=np.float64)
    fractal_high_m = np.zeros((n_sym, n), dtype=np.float64)
    bb_upper_m = np.full((n_sym, n), np.inf, dtype=np.float64)
    trail_m = np.zeros((n_sym, n), dtype=np.float64)
    adv_krw = np.zeros(n_sym, dtype=np.float64)

    for si, sym in enumerate(symbols_ordered):
        arr = symbol_arrays[sym]
        close[si, :] = arr["close"]
        high[si, :] = arr["high"]
        low[si, :] = arr["low"]
        open_p[si, :] = arr["open"]
        atr[si, :] = arr["atr"]
        long_entry_signal[si, :] = arr["long_entry_signal"]
        entry_upper[si, :] = arr["entry_upper"]
        if "regime_risk_mult" in arr:
            regime_risk[si, :] = arr["regime_risk_mult"]
        if "garch_kelly_f" in arr:
            garch_kelly[si, :] = arr["garch_kelly_f"]
        if "kill_signal" in arr:
            kill_signal[si, :] = arr["kill_signal"]
        if "fractal_high_flag" in arr:
            fractal_high_m[si, :] = arr["fractal_high_flag"]
        if rank_scores is not None and sym in rank_scores:
            rank_scores_m[si, :] = rank_scores[sym]
        else:
            for j in range(n):
                rank_scores_m[si, j] = float(-si)
        if "bb_upper" in arr:
            bb_upper_m[si, :] = arr["bb_upper"]
        if "trail_tighten_flag" in arr:
            trail_m[si, :] = arr["trail_tighten_flag"]
        c = arr["close"]
        if "volume" in arr:
            v = arr["volume"]
        else:
            # Fallback when arrays omit volume (caller should pass volume for realistic ADV).
            v = np.ones(n, dtype=np.float64)
        lb = min(6, n)
        if lb > 0:
            adv_krw[si] = float(np.mean(v[n - lb :] * c[n - lb :]))
        else:
            adv_krw[si] = 1.0

    fee_rate = float(UPBIT_SPOT_TAKER_FEE_RATE)
    slippage_rate = float(SPOT_SLIPPAGE_RATE)
    risk_per_trade = float(params.get("RISK_PER_TRADE", 0.015))
    max_position_pct = float(params.get("MAX_POSITION_PCT", 0.25))
    cap_coin = params.get("MAX_CAP_PER_COIN")
    if cap_coin is not None:
        max_position_pct = min(max_position_pct, float(cap_coin))
    long_atr_mult = float(params.get("LONG_ATR_MULT", 3.0))
    long_trail_mult = float(params.get("LONG_TRAIL_MULT", 3.0))
    long_trail_lock_mult = float(params.get("LONG_TRAIL_LOCK_MULT", 1.5))
    long_tp_mult = float(params.get("LONG_TP_MULT", 5.0))
    tp_lock_mult = float(params.get("TP_LOCK_ATR_MULT", 3.0))
    long_scale_atr_mult = float(params.get("LONG_SCALE_ATR_MULT", 0.0))
    scale_out_pct = float(params.get("SCALE_OUT_PCT", 0.0))
    fractal_scale_out_ratio = float(params.get("SCALE_OUT_RATIO", 0.5))
    time_stop_bars = int(params.get("TIME_STOP_BARS", 0))
    kill_cd_bars = int(params.get("KILL_COOLDOWN_BARS", 6))
    delta_gate = float(params.get("DELTA_GATE", 0.08))
    max_slots = max(1, min(int(max_concurrent_positions), n_sym))
    gamma_base = float(params.get("SLIPPAGE_GAMMA_BASE", SLIPPAGE_GAMMA_BASE))
    ref_adv = float(params.get("SLIPPAGE_REFERENCE_ADV_KRW", SLIPPAGE_REFERENCE_ADV_KRW))
    pen_scale = float(concurrency_penalty_scale)

    return _run_shared_cash_packed_numba(
        close,
        high,
        low,
        open_p,
        atr,
        long_entry_signal,
        entry_upper,
        regime_risk,
        garch_kelly,
        kill_signal,
        rank_scores_m,
        fractal_high_m,
        bb_upper_m,
        trail_m,
        adv_krw,
        initial_balance=initial_balance,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        risk_per_trade=risk_per_trade,
        max_position_pct=max_position_pct,
        long_atr_mult=long_atr_mult,
        long_trail_mult=long_trail_mult,
        long_trail_lock_mult=long_trail_lock_mult,
        long_tp_mult=long_tp_mult,
        tp_lock_mult=tp_lock_mult,
        long_scale_atr_mult=long_scale_atr_mult,
        scale_out_pct=scale_out_pct,
        fractal_scale_out_ratio=fractal_scale_out_ratio,
        time_stop_bars=time_stop_bars,
        warmup_bars=warmup_bars,
        execution_start_idx=execution_start_idx,
        kill_cd_bars=kill_cd_bars,
        delta_gate=delta_gate,
        max_slots=max_slots,
        slippage_gamma_base=gamma_base,
        slippage_ref_adv_krw=ref_adv,
        concurrency_penalty_scale=pen_scale,
    )


def use_numba_shared_cash() -> bool:
    return os.getenv("OPT_SPOT_SHARED_CASH_NUMBA", "1").strip().lower() not in ("0", "false", "no")
