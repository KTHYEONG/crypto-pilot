from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import FUNDING_FEE_RATE, SLIPPAGE_RATE, TRADING_FEE_RATE

_logger = logging.getLogger(__name__)


# =============================================================================
# CORE EXECUTION LOGIC (Numba-accelerated)
# =============================================================================

@njit(inline="always")
def process_long_scale_out(
    c_open: float,
    c_high: float,
    entry_price: float,
    pos_atr: float,
    l_scale_atr: float,
    amount: float,
    fee_rate: float,
) -> tuple[bool, float, float, float, float]:
    scale_target = entry_price + (pos_atr * l_scale_atr)
    if c_high >= scale_target:
        sc_price = c_open if c_open >= scale_target else scale_target
        sc_amount = amount / 2.0
        pnl = (sc_price - entry_price) * sc_amount
        fee = sc_amount * sc_price * fee_rate
        return True, sc_price, sc_amount, pnl, fee
    return False, 0.0, 0.0, 0.0, 0.0


@njit(inline="always")
def process_short_scale_out(
    c_open: float,
    c_low: float,
    entry_price: float,
    pos_atr: float,
    s_tp_mult: float,
    amount: float,
    fee_rate: float,
) -> tuple[bool, float, float, float, float]:
    tp_price = entry_price - (pos_atr * s_tp_mult)
    if c_open <= tp_price or c_low <= tp_price:
        sc_price = c_open if c_open <= tp_price else tp_price
        sc_amount = amount / 2.0
        pnl = (entry_price - sc_price) * sc_amount
        fee = sc_amount * sc_price * fee_rate
        return True, sc_price, sc_amount, pnl, fee
    return False, 0.0, 0.0, 0.0, 0.0


@njit(inline="always")
def check_long_exit(
    c_open: float,
    c_low: float,
    highest: float,
    pos_atr: float,
    stop_price: float,
    l_trail_mult: float,
    slippage_rate: float,
) -> tuple[bool, float, float]:
    if c_open <= stop_price:
        return True, c_open * (1.0 - slippage_rate), stop_price
    elif c_low <= stop_price:
        return True, stop_price * (1.0 - slippage_rate), stop_price

    new_stop = highest - (pos_atr * l_trail_mult)
    if new_stop > stop_price:
        stop_price = new_stop
    return False, 0.0, stop_price


@njit(inline="always")
def check_short_exit(
    c_open: float,
    c_high: float,
    lowest: float,
    pos_atr: float,
    stop_price: float,
    s_trail_mult: float,
    slippage_rate: float,
) -> tuple[bool, float, float]:
    if c_open >= stop_price:
        return True, c_open * (1.0 + slippage_rate), stop_price
    elif c_high >= stop_price:
        return True, stop_price * (1.0 + slippage_rate), stop_price

    new_stop = lowest + (pos_atr * s_trail_mult)
    if new_stop < stop_price:
        stop_price = new_stop

    return False, 0.0, stop_price


@njit(inline="always")
def calculate_position_size(
    fill_price: float,
    asset_atr_pct: float,        # 코인의 내재 변동성 (ATR / Price)
    current_equity_for_risk: float,
    available_margin: float,
    risk_per_trade: float,       # 포트폴리오 타겟 변동성 (기존 risk_per_trade 활용)
    leverage: float,
    sf: float,                   # Confidence multiplier (Z-Score driven Alpha multiplier)
    gk: float,
    max_exposure_per_coin: float = 1.5,
) -> float:
    """[RE-ENGINEERED] Alpha-Driven Kelly & Dynamic Portfolio Scaling.

    Fuses ML Z-Score Alpha (sf) with Target Volatility Sizing.
    """
    # 0. NaN Protection
    if np.isnan(asset_atr_pct) or np.isnan(current_equity_for_risk) or np.isnan(fill_price):
        return 0.0
    
    # 1. Target Volatility 기반 명목 자본 할당 (Notional Allocation)
    vol_scalar = risk_per_trade / max(asset_atr_pct, 0.001)
    
    # Alpha-Driven Confidence mapping
    conf_mult = max(min(sf, 1.0), 0.0)
    
    # Garch-Kelly inhibitor (optional but kept for robustness)
    gk_use = max(min(gk, 1.0), 0.0)
    
    target_notional = current_equity_for_risk * vol_scalar * conf_mult * gk_use
    
    # 2. [ROBUST MARGIN PROTECTION]
    max_safe_by_equity = max(current_equity_for_risk, 0.0) * leverage * 0.70
    max_safe_by_margin = max(available_margin, 0.0) * leverage * 0.80
    
    target_notional = min(target_notional, min(max_safe_by_equity, max_safe_by_margin))

    # 3. 명목 한도 캡 (Max Exposure / Anti-Gap Protection)
    max_qty_by_exposure = (current_equity_for_risk * max_exposure_per_coin) / fill_price
    target_qty = min(target_notional / fill_price, max_qty_by_exposure)

    # 4. 가용 증거금 실질 한도 캡 (Margin Constraint) - 수수료 예비분 3% 제외
    max_qty_by_margin = (available_margin * 0.97 * leverage) / fill_price
    if max_qty_by_margin < 0:
        max_qty_by_margin = 0.0
    target_qty = min(target_qty, max_qty_by_margin)

    # 5. 소액 계좌 최소 먼지(Dust) 한도 보정 ($6.0 보장)
    if target_qty > 0.0 and (target_qty * fill_price) < 6.0:
        min_qty = 6.0 / fill_price
        if min_qty <= max_qty_by_margin:
            target_qty = min_qty
        else:
            target_qty = 0.0
    
    return target_qty


@njit(inline="always")
def check_intra_bar_stop(
    pos_side: int,
    c_high: float,
    c_low: float,
    stop_price: float,
    entry_price: float,
    amount: float,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[bool, float, float, float]:
    if pos_side == 1 and c_low <= stop_price:
        intra_exit_price = stop_price * (1.0 - slippage_rate)
        pnl = (intra_exit_price - entry_price) * amount
        exit_fee = amount * intra_exit_price * fee_rate
        return True, intra_exit_price, pnl, exit_fee
    elif pos_side == -1 and c_high >= stop_price:
        intra_exit_price = stop_price * (1.0 + slippage_rate)
        pnl = (entry_price - intra_exit_price) * amount
        exit_fee = amount * intra_exit_price * fee_rate
        return True, intra_exit_price, pnl, exit_fee
    return False, 0.0, 0.0, 0.0


# =============================================================================
# PORTFOLIO UTILITIES (Numba-accelerated)
# =============================================================================

@njit(nogil=True, cache=True)
def _recompute_cs_dirs_numba(
    prev_i: int,
    n_syms: int,
    xs_long: np.ndarray,
    xs_short: np.ndarray,
    hmm_crisis: np.ndarray,
    hmm_mod_long: np.ndarray,
    hmm_mod_short: np.ndarray,
    crisis_gamma: float,
    k_rank: int,
    cs_z_threshold: float,
    cs_z_exit_threshold: float,
    computed_dir: np.ndarray,
    prev_computed_dir: np.ndarray,
) -> None:
    """Hard top-K CS dirs: binary top-K, crisis (1-p)^gamma, |mag| in [0.05,1].
    
    [IMPROVED] Added Hysteresis (Schmitt Trigger) and Absolute Alpha check.
    """
    gam_val = crisis_gamma if crisis_gamma > 1e-9 else 1e-9
    fk = float(k_rank)

    # 1. Compute cross-sectional mean and std for Z-scoring
    mean_l, mean_s = 0.0, 0.0
    count_l, count_s = 0, 0
    for s in range(n_syms):
        vl = float(xs_long[prev_i, s])
        if np.isfinite(vl):
            mean_l += vl
            count_l += 1
        vs = float(xs_short[prev_i, s])
        if np.isfinite(vs):
            mean_s += vs
            count_s += 1
    if count_l > 0:
        mean_l /= count_l
    if count_s > 0:
        mean_s /= count_s

    std_l, std_s = 0.0, 0.0
    for s in range(n_syms):
        vl = float(xs_long[prev_i, s])
        if np.isfinite(vl):
            std_l += (vl - mean_l)**2
        vs = float(xs_short[prev_i, s])
        if np.isfinite(vs):
            std_s += (vs - mean_s)**2
    if count_l > 1:
        std_l = np.sqrt(std_l / (count_l - 1))
    if count_s > 1:
        std_s = np.sqrt(std_s / (count_s - 1))
    std_l = std_l if std_l > 1e-9 else 1e-9
    std_s = std_s if std_s > 1e-9 else 1e-9

    for s in range(n_syms):
        sl = float(xs_long[prev_i, s])
        ss = float(xs_short[prev_i, s])
        if not np.isfinite(sl) or not np.isfinite(ss):
            computed_dir[s] = 0.0
            continue

        mod_l = float(hmm_mod_long[prev_i, s])
        mod_s = float(hmm_mod_short[prev_i, s])

        c_raw = float(hmm_crisis[prev_i, s])
        if not np.isfinite(c_raw):
            c_raw = 0.0
        c_raw = max(0.0, min(1.0, c_raw))

        rank_l = 1
        for t in range(n_syms):
            if t == s:
                continue
            vt = xs_long[prev_i, t]
            if np.isfinite(vt) and vt > sl:
                rank_l += 1
        rank_s = 1
        for t in range(n_syms):
            if t == s:
                continue
            vt = xs_short[prev_i, t]
            if np.isfinite(vt) and vt < ss:
                rank_s += 1

        z_l = (sl - mean_l) / std_l
        z_s = (mean_s - ss) / std_s

        c_prob = float(hmm_crisis[prev_i, s])
        if c_prob > 0.5:
            computed_dir[s] = 0.0
            continue

        # Hysteresis (Schmitt Trigger)
        prev_side = 1.0 if prev_computed_dir[s] > 0.0 else (-1.0 if prev_computed_dir[s] < 0.0 else 0.0)
        
        eff_z_l_thr = cs_z_exit_threshold if prev_side == 1.0 else cs_z_threshold
        eff_z_s_thr = cs_z_exit_threshold if prev_side == -1.0 else cs_z_threshold

        # Absolute Alpha Check: sl >= 0.55 for long, ss <= 0.45 for short (assuming 0.5 neutral)
        # Note: In some pipelines xs_long/xs_short are already normalized or directional.
        # We add a safety check for conviction.
        binary_l = 1.0 if (float(rank_l) <= fk and mod_l >= 0.1 and z_l >= eff_z_l_thr and sl >= 0.55) else 0.0
        binary_s = 1.0 if (float(rank_s) <= fk and mod_s >= 0.1 and z_s >= eff_z_s_thr and ss <= 0.45) else 0.0

        exposure_discount = (1.0 - c_prob) ** gam_val

        mag_l = ((z_l - eff_z_l_thr) / (3.0 - eff_z_l_thr)) ** 1.5 if z_l >= eff_z_l_thr else 0.0
        mag_s = ((z_s - eff_z_s_thr) / (3.0 - eff_z_s_thr)) ** 1.5 if z_s >= eff_z_s_thr else 0.0

        # Rank-based Sizing: weighting allocations by rank conviction
        rank_mult_l = (fk - float(rank_l) + 1.0) / fk if float(rank_l) <= fk else 0.0
        rank_mult_s = (fk - float(rank_s) + 1.0) / fk if float(rank_s) <= fk else 0.0

        long_mag = binary_l * min(max(mag_l * rank_mult_l, 0.1), 1.0) * exposure_discount
        short_mag = binary_s * min(max(mag_s * rank_mult_s, 0.1), 1.0) * exposure_discount

        if long_mag >= short_mag and long_mag > 0.0:
            computed_dir[s] = 1.0 * long_mag
        elif short_mag > long_mag and short_mag > 0.0:
            computed_dir[s] = -1.0 * short_mag
        else:
            computed_dir[s] = 0.0


# =============================================================================
# BACKTEST LOOPS (Numba-accelerated)
# =============================================================================

@njit(nogil=True, cache=True)
def backtest_loop_single_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_prices: np.ndarray,
    entry_upper: np.ndarray,
    entry_lower: np.ndarray,
    trend_dir: np.ndarray,
    strength_filter: np.ndarray,
    atr: np.ndarray,
    macro_ema_arr: np.ndarray,
    garch_kelly_f: np.ndarray,
    initial_balance: float,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
    risk_per_trade: float,
    timestamps: np.ndarray,
    funding_rate_sums: np.ndarray,
    atr_mult: float,
    trail_mult: float,
    l_atr_mult_unused: float,
    s_tp_mult: float,
    l_scale_atr: float,
    s_trail_mult_unused: float,
    warmup_bars: int,
    execution_start_idx: int,
    use_compounding: bool,
    max_capital_usage: float,
    max_exp_per_coin: float,
    dd_scaling_threshold: float,
    hmm_crisis: np.ndarray,
    hmm_mod_long: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    funding_paid_total = 0.0
    n = len(close)
    balance = initial_balance
    peak_equity = initial_balance
    equity_curve = np.zeros(n)

    in_position = False
    pos_side = 0
    entry_price = 0.0
    entry_idx = 0
    amount = 0.0
    entry_fee_stored = 0.0

    stop_price = 0.0
    highest = 0.0
    has_scaled_out = False
    lowest = 0.0

    max_trades = 30000
    trades = np.zeros((max_trades, 8))
    trade_count = 0

    for i in range(n):
        if i < warmup_bars or i < execution_start_idx:
            equity_curve[i] = initial_balance
            continue

        c_open = open_prices[i]
        c_price = close[i]
        c_high = high[i]
        c_low = low[i]

        current_dd = (peak_equity - equity_curve[i-1]) / peak_equity if peak_equity > 0 else 0.0
        dd_factor = 1.0
        if current_dd > dd_scaling_threshold:
            dd_factor = max(0.1, 1.0 - (current_dd / 0.40))
        effective_risk = risk_per_trade * dd_factor

        bar_processed = False

        current_eq_check = balance
        if in_position:
            current_eq_check = (
                balance
                + (amount * entry_price) / leverage
                + (c_open - entry_price) * amount * pos_side
            )
        if current_eq_check <= 0:
            equity_curve[i] = current_eq_check
            if in_position:
                exit_price = c_open
                pnl = (exit_price - entry_price) * amount * pos_side
                exit_fee = amount * exit_price * fee_rate
                if trade_count < max_trades:
                    trades[trade_count] = [entry_idx, i, pos_side, entry_price, exit_price, pnl - exit_fee, amount, entry_fee_stored]
                    trade_count += 1
                in_position = False
                balance = 0.0
            break

        if in_position:
            funding_rate_sum = funding_rate_sums[i]
            if not np.isnan(funding_rate_sum) and funding_rate_sum != 0.0:
                funding_cost = (amount * c_open) * funding_rate_sum * pos_side
                balance -= funding_cost
                funding_paid_total += funding_cost
                funding_eq_check = balance + (amount * entry_price) / leverage + (c_open - entry_price) * amount * pos_side
                if funding_eq_check <= 0:
                    exit_price = c_open
                    pnl = (exit_price - entry_price) * amount * pos_side
                    exit_fee = amount * exit_price * fee_rate
                    if trade_count < max_trades:
                        trades[trade_count] = [entry_idx, i, pos_side, entry_price, exit_price, pnl - exit_fee, amount, entry_fee_stored]
                        trade_count += 1
                    in_position = False; balance = 0.0; break

            exit_triggered, exit_price = False, 0.0

            if pos_side == 1:
                if c_high > highest: highest = c_high
                pos_atr = atr[entry_idx]
                if not has_scaled_out:
                    triggered, sc_price, sc_amount, pnl_scale, exit_fee_scale = process_long_scale_out(c_open, c_high, entry_price, pos_atr, l_scale_atr, amount, fee_rate)
                    if triggered:
                        balance += (sc_amount * entry_price) / leverage + pnl_scale - exit_fee_scale
                        if trade_count < max_trades:
                            trades[trade_count] = [entry_idx, i, pos_side, entry_price, sc_price, pnl_scale - exit_fee_scale, sc_amount, entry_fee_stored / 2.0]
                            trade_count += 1
                        amount -= sc_amount; entry_fee_stored -= entry_fee_stored / 2.0; has_scaled_out = True
                exit_triggered, exit_price, stop_price = check_long_exit(c_open, c_low, highest, pos_atr, stop_price, trail_mult, slippage_rate)
            elif pos_side == -1:
                if c_low < lowest: lowest = c_low
                pos_atr = atr[entry_idx]
                if not has_scaled_out:
                    triggered, sc_price, sc_amount, pnl_scale, exit_fee_scale = process_short_scale_out(c_open, c_low, entry_price, pos_atr, s_tp_mult, amount, fee_rate)
                    if triggered:
                        balance += (sc_amount * entry_price) / leverage + pnl_scale - exit_fee_scale
                        if trade_count < max_trades:
                            trades[trade_count] = [entry_idx, i, pos_side, entry_price, sc_price, pnl_scale - exit_fee_scale, sc_amount, entry_fee_stored / 2.0]
                            trade_count += 1
                        amount -= sc_amount; entry_fee_stored -= entry_fee_stored / 2.0; has_scaled_out = True; stop_price = entry_price - (entry_price * fee_rate * 2.0)
                exit_triggered, exit_price, stop_price = check_short_exit(c_open, c_high, lowest, pos_atr, stop_price, trail_mult, slippage_rate)

            if exit_triggered:
                pnl = (exit_price - entry_price) * amount * pos_side
                exit_fee = amount * exit_price * fee_rate
                pnl -= exit_fee
                margin = (amount * entry_price) / leverage
                balance += margin + pnl
                if trade_count < max_trades:
                    trades[trade_count] = [entry_idx, i, pos_side, entry_price, exit_price, pnl, amount, entry_fee_stored]
                    trade_count += 1
                in_position = False; bar_processed = True

        if not in_position and not bar_processed:
            prev_i = i - 1 if i > 0 else 0
            sf_raw = strength_filter[prev_i]
            if sf_raw <= 0.0 or np.isnan(sf_raw):
                equity_curve[i] = balance; continue

            do_entry, fill_price, pending_side = False, 0.0, 0
            if trend_dir[prev_i] == 1:
                if c_high > entry_upper[prev_i]:
                    fill_price = max(c_open, entry_upper[prev_i]) * (1 + slippage_rate)
                    pending_side = 1; do_entry = True
            elif trend_dir[prev_i] == -1:
                if c_low < entry_lower[prev_i]:
                    fill_price = min(c_open, entry_lower[prev_i]) * (1 - slippage_rate)
                    pending_side = -1; do_entry = True

            if do_entry:
                prev_atr = atr[prev_i]
                if np.isnan(prev_atr) or prev_atr <= 0.0:
                    equity_curve[i] = balance; continue
                stop_mult_val = atr_mult
                if pending_side == 1:
                    if hmm_crisis[prev_i] > 0.2: stop_mult_val *= 0.6
                    elif hmm_mod_long[prev_i] < 0.7: stop_mult_val *= 0.8
                    stop_price = fill_price - (prev_atr * stop_mult_val)
                else:
                    stop_price = fill_price + (prev_atr * atr_mult)

                stop_distance = abs(fill_price - stop_price)
                if stop_distance > 0:
                    current_equity = max_capital_usage if use_compounding and balance > max_capital_usage else balance
                    amount = calculate_position_size(fill_price, stop_distance/fill_price, current_equity, current_equity, effective_risk, leverage, sf_raw, garch_kelly_f[prev_i], max_exposure_per_coin=max_exp_per_coin)
                    required_margin = (amount * fill_price) / leverage
                    entry_fee = amount * fill_price * fee_rate
                    if balance >= required_margin + entry_fee:
                        balance -= required_margin + entry_fee; entry_fee_stored = entry_fee
                        in_position = True; pos_side = pending_side; entry_price = fill_price; entry_idx = i; highest = fill_price; lowest = fill_price; has_scaled_out = False
                        triggered, intra_exit_price, pnl_intra, exit_fee_intra = check_intra_bar_stop(pos_side, c_high, c_low, stop_price, entry_price, amount, fee_rate, slippage_rate)
                        if triggered:
                            pnl_intra -= exit_fee_intra; balance += (amount * entry_price) / leverage + pnl_intra
                            if trade_count < max_trades:
                                trades[trade_count] = [entry_idx, i, pos_side, entry_price, intra_exit_price, pnl_intra, amount, entry_fee_stored]
                                trade_count += 1
                            in_position = False

        if in_position:
            margin = (amount * entry_price) / leverage
            unrealized = (c_price - entry_price) * amount * pos_side
            equity_curve[i] = balance + margin + unrealized
        else: equity_curve[i] = balance
        if equity_curve[i] > peak_equity: peak_equity = equity_curve[i]

    if in_position and n > 0:
        last_idx = n - 1; last_close = close[last_idx]
        exit_price = last_close * (1 - slippage_rate * pos_side)
        pnl = (exit_price - entry_price) * amount * pos_side - (amount * exit_price * fee_rate)
        balance += (amount * entry_price) / leverage + pnl
        if trade_count < max_trades:
            trades[trade_count] = [entry_idx, last_idx, pos_side, entry_price, exit_price, pnl, amount, entry_fee_stored]
            trade_count += 1

    return trades[:trade_count], balance, equity_curve, funding_paid_total


@njit(nogil=True, cache=True)
def backtest_portfolio_numba(
    close_2d: np.ndarray,
    high_2d: np.ndarray,
    low_2d: np.ndarray,
    open_2d: np.ndarray,
    entry_upper: np.ndarray,
    entry_lower: np.ndarray,
    trend_dir: np.ndarray,
    strength_filter_raw: np.ndarray,
    atr_2d: np.ndarray,
    garch_kelly_f: np.ndarray,
    kill_signal: np.ndarray,
    funding_rate: np.ndarray,
    slot_rank_score: np.ndarray,
    xs_long: np.ndarray,
    xs_short: np.ndarray,
    hmm_crisis: np.ndarray,
    hmm_hard_state: np.ndarray,
    hmm_mod_long: np.ndarray,
    hmm_mod_short: np.ndarray,
    initial_balance: float,
    lev_2d: np.ndarray,
    fee_rate: float,
    slippage_rate: float,
    risk_per_trade: float,
    atr_mult: float,
    trail_mult: float,
    l_atr_mult_unused: float,
    s_tp_mult: float,
    l_scale_atr: float,
    s_trail_mult_unused: float,
    max_concurrent: int,
    max_exposure: float,
    max_exp_per_coin: float,
    dd_scaling_threshold: float,
    k_rank_long: int,
    k_rank_short: int,
    cs_z_threshold: float,
    rebalance_bars: int,
    rebalance_turnover_threshold: float,
    crisis_gamma: float,
    use_cs_rank: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    n_bars, n_syms = close_2d.shape
    computed_dir = np.zeros(n_syms, dtype=np.float64)
    next_dir = np.zeros(n_syms, dtype=np.float64)
    prev_rebalance_bucket = -999999
    dust_skip_cnt, margin_fail_cnt, t_dir_zero_cnt, p_side_zero_cnt, mod_skip_cnt = 0, 0, 0, 0, 0
    # [REFINED] Minimum Conviction Floor: Notional value must be >= 2% of current equity to avoid micro-churn
    min_notional_floor_pct = 0.02

    balance = initial_balance
    equity_curve = np.zeros(n_bars, dtype=np.float64)
    equity_curve[0] = initial_balance
    hwm = initial_balance

    in_pos = np.zeros(n_syms, dtype=np.bool_)
    pos_side = np.zeros(n_syms, dtype=np.int8)
    entry_p = np.zeros(n_syms, dtype=np.float64)
    entry_idx = np.zeros(n_syms, dtype=np.int32)
    amount = np.zeros(n_syms, dtype=np.float64)
    entry_fee_stored = np.zeros(n_syms, dtype=np.float64)
    fund_fee_stored = np.zeros(n_syms, dtype=np.float64)

    stop_p = np.zeros(n_syms, dtype=np.float64)
    highest, lowest = np.zeros(n_syms, dtype=np.float64), np.zeros(n_syms, dtype=np.float64)
    has_scaled = np.zeros(n_syms, dtype=np.bool_)
    just_exited = np.zeros(n_syms, dtype=np.bool_)
    candidate_pool = np.zeros((n_syms, 6), dtype=np.float64)
    entry_lev = np.ones(n_syms, dtype=np.float64)

    max_trades = 50000
    trades: np.ndarray = np.zeros((max_trades, 10), dtype=np.float64)
    t_count = 0

    for i in range(1, n_bars):
        unrealized_total, used_margin_total, num_open_pos = 0.0, 0.0, 0
        just_exited[:] = False
        for s in range(n_syms):
            if in_pos[s]:
                if np.isnan(close_2d[i, s]): continue
                cur_p = close_2d[i, s]
                used_margin_total += (amount[s] * cur_p) / entry_lev[s]
                num_open_pos += 1
                unrealized_total += (cur_p - entry_p[s]) * amount[s] * pos_side[s]
                fund_fee = amount[s] * cur_p * funding_rate[i, s] * pos_side[s]
                fund_fee_stored[s] += fund_fee

        current_equity = balance + used_margin_total + unrealized_total
        equity_curve[i] = current_equity
        if current_equity > hwm: hwm = current_equity

        dd = (hwm - current_equity) / hwm if hwm > 1e-9 else 0.0
        dd_scale = max(0.0, 1.0 - (dd / dd_scaling_threshold)) if dd_scaling_threshold > 1e-9 else 1.0
        c_prob = float(hmm_crisis[i, 0])
        h_state = int(hmm_hard_state[i, 0])
        
        # [NEW] HMM-based Exposure Scaling (Go/No-Go)
        regime_exp_mult = 1.0
        if c_prob > 0.5 or h_state == 4: # REGIME_CRISIS
            regime_exp_mult = 0.0
        elif h_state == 2: # REGIME_BEAR
            regime_exp_mult = 0.5
        elif h_state == 3: # REGIME_CHOP
            regime_exp_mult = 0.7
            
        max_exp_scaled = max_exposure * regime_exp_mult
        crisis_scale = (1.0 - c_prob) ** crisis_gamma
        effective_risk_per_trade = risk_per_trade * dd_scale * crisis_scale

        prev_i = i - 1
        n_cands = 0
        if use_cs_rank != 0 and prev_i >= 0:
            # 1. Potential next_dir check (with hysteresis: maintenance=0.7x entry)
            cs_z_maint = cs_z_threshold * 0.7
            _recompute_cs_dirs_numba(prev_i, n_syms, xs_long, xs_short, hmm_crisis, hmm_mod_long, hmm_mod_short, crisis_gamma, k_rank_long, cs_z_threshold, cs_z_maint, next_dir, computed_dir)

            # 2. Rebalance Trigger: Fixed Bars OR Event-Driven (Turnover > 15%)
            bucket = prev_i // rebalance_bars if rebalance_bars > 0 else prev_i
            
            turnover = 0.0
            for s_idx in range(n_syms):
                turnover += abs(next_dir[s_idx] - computed_dir[s_idx])
            
            do_rebalance = False
            if bucket != prev_rebalance_bucket:
                do_rebalance = True
            elif turnover > rebalance_turnover_threshold: # Event-driven turnover threshold
                do_rebalance = True
            
            if do_rebalance:
                prev_rebalance_bucket = bucket
                for s_idx in range(n_syms):
                    computed_dir[s_idx] = next_dir[s_idx]

        for s in range(n_syms):
            if not in_pos[s]: continue
            c_open, c_high, c_low = open_2d[i, s], high_2d[i, s], low_2d[i, s]
            pos_atr = atr_2d[entry_idx[s], s]
            exit_triggered, exit_price = False, 0.0
            
            # 3. Alpha Conviction Exit & Hard Kill
            if kill_signal[i-1, s] > 0.5: 
                exit_triggered, exit_price = True, c_open * (1.0 - slippage_rate * pos_side[s])
            elif use_cs_rank != 0 and prev_i >= 0:
                t_dir = computed_dir[s]
                # [FIX] Soft Alpha Exit: Only force-exit if the signal REVERSES.
                # If signal is neutral (0.0), hold and let Trailing Stop manage.
                if pos_side[s] == 1 and t_dir < 0.0:
                    exit_triggered, exit_price = True, c_open * (1.0 - slippage_rate * pos_side[s])
                elif pos_side[s] == -1 and t_dir > 0.0:
                    exit_triggered, exit_price = True, c_open * (1.0 - slippage_rate * pos_side[s])

            if not exit_triggered:
                if pos_side[s] == 1:
                    if c_high > highest[s]: highest[s] = c_high
                    exit_triggered, exit_price, stop_p[s] = check_long_exit(c_open, c_low, highest[s], pos_atr, stop_p[s], trail_mult, slippage_rate)
                    if not exit_triggered and not has_scaled[s]:
                        tr, sc_p, sc_a, pnl_s, fee_s = process_long_scale_out(c_open, c_high, entry_p[s], pos_atr, l_scale_atr, amount[s], fee_rate)
                        if tr:
                            sc_f = fund_fee_stored[s] / 2.0; balance += (sc_a * entry_p[s]) / entry_lev[s] + (pnl_s - fee_s)
                            if t_count < max_trades:
                                trades[t_count] = [float(s), float(entry_idx[s]), float(i), 1.0, entry_p[s], sc_p, pnl_s - fee_s - sc_f, sc_a, entry_fee_stored[s]/2.0, sc_f]
                                t_count += 1
                            amount[s] -= sc_a; entry_fee_stored[s] /= 2.0; fund_fee_stored[s] /= 2.0; has_scaled[s] = True
                else:
                    if c_low < lowest[s]: lowest[s] = c_low
                    exit_triggered, exit_price, stop_p[s] = check_short_exit(c_open, c_high, lowest[s], pos_atr, stop_p[s], trail_mult, slippage_rate)
                    if not exit_triggered and not has_scaled[s]:
                        tr, sc_p, sc_a, pnl_s, fee_s = process_short_scale_out(c_open, c_low, entry_p[s], pos_atr, s_tp_mult, amount[s], fee_rate)
                        if tr:
                            sc_f = fund_fee_stored[s] / 2.0; balance += (sc_a * entry_p[s]) / entry_lev[s] + (pnl_s - fee_s)
                            if t_count < max_trades:
                                trades[t_count] = [float(s), float(entry_idx[s]), float(i), -1.0, entry_p[s], sc_p, pnl_s - fee_s - sc_f, sc_a, entry_fee_stored[s]/2.0, sc_f]
                                t_count += 1
                            amount[s] -= sc_a; entry_fee_stored[s] /= 2.0; fund_fee_stored[s] /= 2.0; has_scaled[s] = True; stop_p[s] = entry_p[s] - (entry_p[s]*fee_rate*2.0)

            if exit_triggered:
                pnl = (exit_price - entry_p[s]) * amount[s] * pos_side[s]; fee = amount[s] * exit_price * fee_rate
                balance += ((amount[s] * entry_p[s]) / entry_lev[s]) + (pnl - fee)
                if t_count < max_trades:
                    trades[t_count] = [float(s), float(entry_idx[s]), float(i), float(pos_side[s]), entry_p[s], exit_price, pnl - fee - fund_fee_stored[s], amount[s], entry_fee_stored[s], fund_fee_stored[s]]
                    t_count += 1
                in_pos[s], just_exited[s] = False, True; num_open_pos -= 1

        free_margin = current_equity - used_margin_total
        if num_open_pos < max_concurrent:
            for s in range(n_syms):
                if in_pos[s] or just_exited[s] or np.isnan(open_2d[i, s]): continue
                sf = 1.0
                if use_cs_rank != 0:
                    t_dir = computed_dir[s]; dir_abs = abs(float(t_dir)); sf = min(dir_abs, 1.0)
                else:
                    sf_raw = strength_filter_raw[prev_i, s]
                    if sf_raw < 0.5: continue
                    t_dir = trend_dir[prev_i, s]
                if t_dir == 0.0:
                    if use_cs_rank != 0: t_dir_zero_cnt += 1
                    continue
                if use_cs_rank != 0:
                    # Relaxed HMM gate from 0.5 to 0.1
                    if (t_dir > 0 and hmm_mod_long[prev_i, s] < 0.1) or (t_dir < 0 and hmm_mod_short[prev_i, s] < 0.1):
                        mod_skip_cnt += 1; continue

                c_open, p_side, fill_p = open_2d[i, s], 0, 0.0
                if t_dir > 0.0:
                    if entry_upper[prev_i, s] <= 1e-9 or high_2d[i, s] > entry_upper[prev_i, s]:
                        base_p = c_open if entry_upper[prev_i, s] <= 1e-9 else max(c_open, entry_upper[prev_i, s])
                        fill_p = base_p * (1.0 + slippage_rate); p_side = 1
                elif t_dir < 0.0:
                    if entry_lower[prev_i, s] <= 1e-9 or entry_lower[prev_i, s] >= 999998.0 or low_2d[i, s] < entry_lower[prev_i, s]:
                        base_p = c_open if (entry_lower[prev_i, s] <= 1e-9 or entry_lower[prev_i, s] >= 999998.0) else min(c_open, entry_lower[prev_i, s])
                        fill_p = base_p * (1.0 - slippage_rate); p_side = -1
                if p_side == 0:
                    if use_cs_rank != 0: p_side_zero_cnt += 1
                    continue

                atr_p = atr_2d[prev_i, s] / max(close_2d[prev_i, s], 1e-12)
                le_ent = max(1.0, float(lev_2d[i, s]))
                gk_use = garch_kelly_f[prev_i, s] * (hmm_mod_long[prev_i, s] if p_side == 1 else hmm_mod_short[prev_i, s])
                if not np.isfinite(gk_use) or gk_use < 0.0: gk_use = 0.0
                stop_mult = atr_mult
                if p_side == 1:
                    if hmm_crisis[prev_i, s] > 0.2: stop_mult *= 0.6
                    elif hmm_mod_long[prev_i, s] < 0.7: stop_mult *= 0.8

                target_qty = calculate_position_size(fill_p, atr_p, current_equity, free_margin, effective_risk_per_trade, le_ent, sf, gk_use, max_exp_per_coin)
                if target_qty > 0:
                    sort_key = abs(float(t_dir)) if use_cs_rank != 0 else abs(float(slot_rank_score[prev_i, s]))
                    candidate_pool[n_cands] = [sort_key, float(s), float(p_side), fill_p, target_qty, atr_2d[prev_i, s] * stop_mult]
                    n_cands += 1

            if n_cands > 0:
                for c1 in range(n_cands):
                    for c2 in range(c1 + 1, n_cands):
                        if candidate_pool[c1, 0] < candidate_pool[c2, 0]:
                            for k in range(6):
                                tmp = candidate_pool[c1, k]; candidate_pool[c1, k] = candidate_pool[c2, k]; candidate_pool[c2, k] = tmp
                n_sel = min(n_cands, max_concurrent - num_open_pos)
                total_req = 0.0
                total_new_notional = 0.0
                for idx in range(n_sel):
                    s_i = int(candidate_pool[idx, 1]); le_i = max(1.0, float(lev_2d[i, s_i]))
                    notional_i = candidate_pool[idx, 4] * candidate_pool[idx, 3]
                    total_req += notional_i / le_i
                    total_new_notional += notional_i
                
                # [NEW] Cap total exposure by max_exp_scaled (HMM-aware)
                current_notional = 0.0
                for s_in in range(n_syms):
                    if in_pos[s_in]:
                        current_notional += amount[s_in] * close_2d[i, s_in]
                
                new_notional_cap = max(0.0, (current_equity * max_exp_scaled) - current_notional)
                
                scale = min(1.0, (free_margin * 0.96) / total_req) if total_req > 0 else 1.0
                if total_new_notional > 1e-9:
                    scale = min(scale, new_notional_cap / total_new_notional)

                for idx in range(n_sel):
                    _, s_f, p_side_f, fill_p, target_qty, stop_dist = candidate_pool[idx]
                    s, p_side, final_qty = int(s_f), int(p_side_f), target_qty * scale
                    
                    # [REFINED] Minimum Conviction Floor
                    # Skip entry if notional value is less than 2% of current equity
                    min_notional = max(6.0, current_equity * min_notional_floor_pct)
                    if final_qty * fill_p < min_notional: 
                        dust_skip_cnt += 1; continue
                    
                    le_ent = max(1.0, float(lev_2d[i, s]))
                    req_m = (final_qty * fill_p) / le_ent; e_fee = final_qty * fill_p * fee_rate
                    if free_margin >= (req_m + e_fee):
                        balance -= (req_m + e_fee); in_pos[s], pos_side[s], entry_p[s], entry_idx[s] = True, p_side, fill_p, i
                        entry_lev[s] = le_ent
                        amount[s], entry_fee_stored[s], fund_fee_stored[s], highest[s], lowest[s], has_scaled[s], stop_p[s] = final_qty, e_fee, 0.0, fill_p, fill_p, False, fill_p - (stop_dist * p_side)
                        free_margin -= (req_m + e_fee)
                    else: margin_fail_cnt += 1

    if n_bars > 0:
        last_idx = n_bars - 1
        for s in range(n_syms):
            if in_pos[s]:
                cur_p = close_2d[last_idx, s] if not np.isnan(close_2d[last_idx, s]) else entry_p[s]
                pnl = (cur_p - entry_p[s]) * amount[s] * pos_side[s] - (amount[s] * cur_p * fee_rate)
                balance += ((amount[s] * entry_p[s]) / entry_lev[s]) + pnl
                if t_count < max_trades:
                    trades[t_count] = [float(s), float(entry_idx[s]), float(last_idx), float(pos_side[s]), entry_p[s], cur_p, pnl - fund_fee_stored[s], amount[s], entry_fee_stored[s], fund_fee_stored[s]]
                    t_count += 1

    diag_out = np.array([dust_skip_cnt, margin_fail_cnt, t_dir_zero_cnt, p_side_zero_cnt, mod_skip_cnt], dtype=np.int64)
    return trades[:t_count], balance, equity_curve, diag_out


# =============================================================================
# ENGINE CLASSES
# =============================================================================

class SingleSymbolEngine:
    """Consolidated Single-symbol Futures Backtest Engine."""

    _REQUIRED_INDICATOR_COLS = frozenset({"entry_upper", "entry_lower", "trend_direction", "strength_filter", "atr", "macro_ema"})
    _OPTIONAL_MERGE_COLS = frozenset({"garch_kelly_f"})

    def __init__(
        self,
        hourly_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        strategy: Any,
        initial_balance: float = 1_000_000,
        precomputed_daily_df: pd.DataFrame | None = None,
        warmup_bars: int | None = None,
        execution_start_idx: int = 0,
    ) -> None:
        self.hourly_df = hourly_df.copy(deep=False)
        self.daily_df = daily_df.copy(deep=False)
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self._precomputed_daily_df = precomputed_daily_df
        self._warmup_bars_override = warmup_bars
        self._execution_start_idx = max(0, int(execution_start_idx))

        self.leverage: float = self.strategy.params.get("LEVERAGE", 1.0)
        self.risk_per_trade: float = self.strategy.params.get("RISK_PER_TRADE", 0.015)
        self.fee_rate = TRADING_FEE_RATE
        self.slippage_rate = SLIPPAGE_RATE

        self._prepare_data()

    def _prepare_data(self) -> None:
        exclude_cols = {"date_key", "datetime", "date", "open", "high", "low", "close", "volume", "timestamp"}
        if all(c in self.hourly_df.columns for c in self._REQUIRED_INDICATOR_COLS):
            signal_df = self.hourly_df
        else:
            signal_df = self.strategy.generate_signals(self.hourly_df.copy(deep=True))

        merge_keys = self._REQUIRED_INDICATOR_COLS | self._OPTIONAL_MERGE_COLS
        indicator_cols = [c for c in signal_df.columns if c not in exclude_cols and c in merge_keys]
        self.merged_df = self.hourly_df.copy(deep=False)
        for col in indicator_cols:
            self.merged_df[f"daily_{col}"] = signal_df[col].values

    def run(self) -> dict[str, Any]:
        df = self.merged_df
        n = len(df)
        open_prices, close, high, low = df["open"].values, df["close"].values, df["high"].values, df["low"].values
        entry_upper, entry_lower = df["daily_entry_upper"].values, df["daily_entry_lower"].values
        trend_dir, strength_filter = df["daily_trend_direction"].values, df["daily_strength_filter"].values
        atr, macro_ema = df["daily_atr"].values, df["daily_macro_ema"].values
        
        garch_kelly_f = df["daily_garch_kelly_f"].values if "daily_garch_kelly_f" in df.columns else np.ones(n)
        garch_kelly_f = np.nan_to_num(garch_kelly_f, nan=1.0)

        atr_mult = float(self.strategy.params.get("ATR_MULT", 3.0))
        trail_mult = float(self.strategy.params.get("TRAIL_MULT", 3.0))
        short_tp_mult = float(self.strategy.params.get("SHORT_TP_MULT", 3.0))
        long_scale_atr_mult = float(self.strategy.params.get("LONG_SCALE_ATR_MULT", 3.0))
        
        timestamps = df["timestamp"].values
        funding_rate_sums = df["funding_rate_sum"].values if "funding_rate_sum" in df.columns else np.full(n, FUNDING_FEE_RATE/3.0)

        warmup_bars = self._warmup_bars_override if self._warmup_bars_override is not None else int(getattr(df, "attrs", {}).get("warmup_bars", self.strategy.get_required_warmup(freq=self.strategy.params.get("TIMEFRAME", "1h"))))
        self._warmup_bars = warmup_bars
        self._effective_start_idx = max(warmup_bars, self._execution_start_idx)

        hmm_crisis = df["hmm_prob_crisis"].values if "hmm_prob_crisis" in df.columns else np.zeros(n)
        hmm_mod_long = df["hmm_modulator_long"].values if "hmm_modulator_long" in df.columns else np.ones(n)
        long_mod_floor = float(self.strategy.params.get("LONG_MOD_FLOOR", 0.70))
        hmm_mod_long = np.maximum(hmm_mod_long, long_mod_floor)

        trades_raw, final_balance, equity_curve, funding_total = backtest_loop_single_numba(
            close, high, low, open_prices, entry_upper, entry_lower, trend_dir, strength_filter, atr, macro_ema, garch_kelly_f,
            self.initial_balance, self.leverage, self.fee_rate, self.slippage_rate, self.risk_per_trade, timestamps, funding_rate_sums,
            atr_mult, trail_mult, atr_mult, short_tp_mult, long_scale_atr_mult, trail_mult, warmup_bars, self._execution_start_idx,
            bool(self.strategy.params.get("USE_COMPOUNDING", True)), float(self.strategy.params.get("MAX_CAPITAL_USAGE", 1e12)),
            float(self.strategy.params.get("MAX_EXPOSURE_PER_COIN", 1.5)), float(self.strategy.params.get("DD_SCALING_THRESHOLD", 0.15)),
            hmm_crisis, hmm_mod_long
        )

        self.balance = final_balance
        self._equity_curve = equity_curve
        self._total_funding_paid = funding_total
        
        datetime_vals = df["datetime"].values
        self.trades = []
        for i in range(len(trades_raw)):
            e_idx, x_idx = int(trades_raw[i][0]), int(trades_raw[i][1])
            self.trades.append({
                "entry_time": datetime_vals[e_idx], "exit_time": datetime_vals[x_idx],
                "side": "LONG" if trades_raw[i][2] == 1 else "SHORT",
                "entry_price": trades_raw[i][3], "exit_price": trades_raw[i][4],
                "pnl": trades_raw[i][5], "amount": trades_raw[i][6], "entry_fee": trades_raw[i][7]
            })
        return self.get_results()

    def get_results(self) -> dict[str, Any]:
        if not np.isfinite(self.balance) or not self.trades:
            return self._empty_result()

        total_return_pct = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        pnl_arr = np.array([t["pnl"] for t in self.trades])
        win_trades = int(np.sum(pnl_arr > 0))
        
        equity = self._equity_curve[self._effective_start_idx:] if len(self._equity_curve) > self._effective_start_idx else self._equity_curve
        if len(equity) > 0:
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / np.where(running_max == 0, 1e-9, running_max) * 100
            mdd = float(drawdown.min())
        else: mdd = 0.0

        return {
            "total_trades": len(self.trades), "win_trades": win_trades, "loss_trades": len(self.trades) - win_trades,
            "win_rate": (win_trades / len(self.trades)) * 100, "total_return_pct": total_return_pct,
            "final_balance": self.balance, "mdd_pct": mdd, "trades_df": pd.DataFrame(self.trades),
            "equity_curve": equity, "total_funding_paid": float(self._total_funding_paid),
            "gross_pnl_abs": float(np.sum(np.abs(pnl_arr)))
        }

    def _empty_result(self) -> dict[str, Any]:
        return {"total_trades": 0, "win_trades": 0, "loss_trades": 0, "win_rate": 0, "total_return_pct": 0, "final_balance": self.initial_balance, "mdd_pct": 0, "trades_df": pd.DataFrame(), "equity_curve": np.array([]), "total_funding_paid": 0.0, "gross_pnl_abs": 0.0}


class MultiSymbolEngine:
    """Consolidated Multi-symbol Portfolio Futures Backtest Engine."""

    def __init__(
        self,
        aligned_data: dict[str, np.ndarray],
        symbol_names: list[str],
        strategy_params: dict[str, Any],
        initial_balance: float = 1_000_000,
        fee_rate: float = 0.0004,
        slippage_rate: float = 0.001,
    ) -> None:
        self.data = aligned_data
        self.symbols = symbol_names
        self.params = strategy_params
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.leverage = float(self.params.get("LEVERAGE", 1.0))
        self.risk_per_trade = float(self.params.get("RISK_PER_TRADE", 0.02))
        self.max_exposure = float(self.params.get("MAX_EXPOSURE", 0.8))
        self.max_concurrent_positions = int(OPT_FUTURES_CONFIG.get("FUTURES_MAX_CONCURRENT_POSITIONS", 2))

    def run(self) -> tuple[pd.DataFrame, np.ndarray, float, np.ndarray]:
        d = self.data; c2d = d["close"]
        sf_raw = d.get("ml_calib_prob", d.get("strength_filter", np.zeros_like(c2d)))
        hmm_mod_l = np.maximum(d.get("hmm_modulator_long", np.ones_like(c2d)), float(self.params.get("LONG_MOD_FLOOR", 0.70)))
        
        lev_2d = d.get("dyn_leverage", np.full(c2d.shape, self.leverage))
        
        trades_arr, final_bal, equity, diag = backtest_portfolio_numba(
            c2d, d["high"], d["low"], d["open"], d["entry_upper"], d["entry_lower"], d["trend_direction"],
            sf_raw, d["atr"], d["garch_kelly_f"], d.get("kill_signal", np.zeros_like(c2d)),
            d.get("funding_rate_sum", np.zeros_like(c2d)), d["slot_rank_score"],
            d.get("xs_score_long", np.zeros_like(c2d)), d.get("xs_score_short", np.zeros_like(c2d)),
            d.get("hmm_prob_crisis", np.zeros_like(c2d)), 
            d.get("hmm_hard_state", np.zeros_like(c2d)),
            hmm_mod_l, d.get("hmm_modulator_short", np.ones_like(c2d)),
            self.initial_balance, lev_2d, self.fee_rate, self.slippage_rate, self.risk_per_trade,
            float(self.params.get("ATR_MULT", 3.0)), float(self.params.get("TRAIL_MULT", 3.0)), float(self.params.get("ATR_MULT", 3.0)),
            float(self.params.get("SHORT_TP_MULT", 3.0)), float(self.params.get("LONG_SCALE_ATR_MULT", 3.0)), float(self.params.get("TRAIL_MULT", 3.0)),
            self.max_concurrent_positions, self.max_exposure, float(self.params.get("MAX_EXPOSURE_PER_COIN", 1.5)),
            float(self.params.get("DD_SCALING_THRESHOLD", 0.15)), int(self.params.get("K_RANK", 2)), int(self.params.get("K_RANK", 2)),
            float(self.params.get("CS_Z_SCORE_THRESHOLD", 0.0)), max(1, int(self.params.get("REBALANCE_BARS", 6))),
            float(self.params.get("REBALANCE_TURNOVER_THRESHOLD", 0.15)),
            float(self.params.get("CRISIS_GAMMA", 1.0)), 1 if bool(self.params.get("USE_CS_RANK_ENGINE", True)) else 0
        )

        if trades_arr.size == 0: return pd.DataFrame(), equity, final_bal, diag
        df = pd.DataFrame(trades_arr, columns=["sym_idx", "entry_idx", "exit_idx", "side_val", "entry_price", "exit_price", "pnl", "amount", "entry_fee", "funding_fee"])
        df["symbol"] = [self.symbols[int(i)] for i in df["sym_idx"]]
        df["side"] = np.where(df["side_val"] == 1.0, "LONG", "SHORT")
        return df[["symbol", "entry_idx", "exit_idx", "side", "entry_price", "exit_price", "pnl", "amount", "entry_fee", "funding_fee"]], equity, final_bal, diag


class FuturesBacktestEngine:
    """Unified entry point for Futures Backtesting."""

    @staticmethod
    def run_single(hourly_df: pd.DataFrame, daily_df: pd.DataFrame, strategy: Any, **kwargs) -> dict[str, Any]:
        engine = SingleSymbolEngine(hourly_df, daily_df, strategy, **kwargs)
        return engine.run()

    @staticmethod
    def run_multi(aligned_data: dict[str, np.ndarray], symbol_names: list[str], strategy_params: dict[str, Any], **kwargs) -> tuple[pd.DataFrame, np.ndarray, float, np.ndarray]:
        engine = MultiSymbolEngine(aligned_data, symbol_names, strategy_params, **kwargs)
        return engine.run()
