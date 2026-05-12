"""Numba execution: fees, stops, scale-out, portfolio/single backtest loops."""

from __future__ import annotations

import numpy as np
from numba import njit



@njit(inline="always")
def process_long_scale_out(
    c_open: float,
    c_high: float,
    entry_price: float,
    pos_atr: float,
    l_scale_atr: float,
    amount: float,
    maker_fee: float,
    taker_fee: float,
) -> tuple[bool, float, float, float, float]:
    scale_target = entry_price + (pos_atr * l_scale_atr)
    if c_high >= scale_target:
        # Realistic: If high > target, assume Maker fill (someone else took our limit)
        is_maker = c_high > scale_target
        sc_price = scale_target
        sc_amount = amount / 2.0
        pnl = (sc_price - entry_price) * sc_amount
        fee_rate = maker_fee if is_maker else taker_fee
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
    maker_fee: float,
    taker_fee: float,
) -> tuple[bool, float, float, float, float]:
    tp_price = entry_price - (pos_atr * s_tp_mult)
    if c_open <= tp_price or c_low <= tp_price:
        # Realistic: If low < tp_price, assume Maker fill
        is_maker = c_low < tp_price or c_open < tp_price
        sc_price = tp_price
        sc_amount = amount / 2.0
        pnl = (entry_price - sc_price) * sc_amount
        fee_rate = maker_fee if is_maker else taker_fee
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
    # Stop loss is always Taker
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
    # Stop loss is always Taker
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
    risk_per_trade: float,       # Fractional Kelly Lambda (Shrinkage)
    leverage: float,
    p_win: float,                # Calibrated Win Probability p
    b_ratio: float,              # Estimated Win/Loss Ratio (Profit Factor)
    gk: float,
    max_exposure_per_coin: float = 1.5,
) -> float:
    """[RE-ENGINEERED] Fractional Kelly Sizing.

    f* = lambda * (p - (1-p)/b)
    """
    # 0. NaN Protection
    if np.isnan(asset_atr_pct) or np.isnan(current_equity_for_risk) or np.isnan(fill_price):
        return 0.0
    
    # 1. Fractional Kelly Calculation
    # f* = p - (1-p)/b
    p = max(min(p_win, 1.0), 0.0)
    b = max(b_ratio, 0.01) # Avoid division by zero
    
    f_star = p - (1.0 - p) / b
    f_star = max(f_star, 0.0) # No negative bets (shorts are handled by side, not negative f)
    
    # Garch-Kelly inhibitor
    gk_use = max(min(gk, 1.0), 0.0)
    
    # Final Kelly Fraction f = lambda * f*
    kelly_f = risk_per_trade * f_star * gk_use
    
    target_notional = current_equity_for_risk * kelly_f
    
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

    # 5. 소액 계좌 최소 먼지(Dust) 한도 보정 ($0.01 보장)
    if target_qty > 0.0 and (target_qty * fill_price) < 0.01:
        min_qty = 0.01 / fill_price
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
    maker_fee: float,
    taker_fee: float,
    slippage_rate: float,
    smart_offset: float,
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
    estimated_b: float = 1.05,
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
                exit_fee = amount * exit_price * taker_fee
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
                    exit_fee = amount * exit_price * taker_fee
                    if trade_count < max_trades:
                        trades[trade_count] = [entry_idx, i, pos_side, entry_price, exit_price, pnl - exit_fee, amount, entry_fee_stored]
                        trade_count += 1
                    in_position = False; balance = 0.0; break

            exit_triggered, exit_price = False, 0.0

            if pos_side == 1:
                if c_high > highest:
                    highest = c_high
                pos_atr = atr[entry_idx]
                if not has_scaled_out:
                    triggered, sc_price, sc_amount, pnl_scale, exit_fee_scale = process_long_scale_out(
                        c_open, c_high, entry_price, pos_atr, l_scale_atr, amount, maker_fee, taker_fee
                    )
                    if triggered:
                        amt_bef = amount
                        frac_sc = sc_amount / amt_bef if amt_bef > 1e-12 else 0.5
                        fee_alloc = entry_fee_stored * frac_sc
                        balance += (sc_amount * entry_price) / leverage + pnl_scale - exit_fee_scale
                        if trade_count < max_trades:
                            trades[trade_count] = [
                                entry_idx,
                                i,
                                pos_side,
                                entry_price,
                                sc_price,
                                pnl_scale - exit_fee_scale,
                                sc_amount,
                                fee_alloc,
                            ]
                            trade_count += 1
                        amount -= sc_amount
                        entry_fee_stored -= fee_alloc
                        has_scaled_out = True
                exit_triggered, exit_price, stop_price = check_long_exit(
                    c_open, c_low, highest, pos_atr, stop_price, trail_mult, slippage_rate
                )
            elif pos_side == -1:
                if c_low < lowest:
                    lowest = c_low
                pos_atr = atr[entry_idx]
                if not has_scaled_out:
                    triggered, sc_price, sc_amount, pnl_scale, exit_fee_scale = process_short_scale_out(
                        c_open, c_low, entry_price, pos_atr, s_tp_mult, amount, maker_fee, taker_fee
                    )
                    if triggered:
                        amt_bef = amount
                        frac_sc = sc_amount / amt_bef if amt_bef > 1e-12 else 0.5
                        fee_alloc = entry_fee_stored * frac_sc
                        balance += (sc_amount * entry_price) / leverage + pnl_scale - exit_fee_scale
                        if trade_count < max_trades:
                            trades[trade_count] = [
                                entry_idx,
                                i,
                                pos_side,
                                entry_price,
                                sc_price,
                                pnl_scale - exit_fee_scale,
                                sc_amount,
                                fee_alloc,
                            ]
                            trade_count += 1
                        amount -= sc_amount
                        entry_fee_stored -= fee_alloc
                        has_scaled_out = True
                        stop_price = entry_price - (entry_price * taker_fee * 2.0)
                exit_triggered, exit_price, stop_price = check_short_exit(
                    c_open, c_high, lowest, pos_atr, stop_price, trail_mult, slippage_rate
                )

            if exit_triggered:
                pnl = (exit_price - entry_price) * amount * pos_side
                exit_fee = amount * exit_price * taker_fee
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

            do_entry, fill_price, pending_side, entry_fee_rate = False, 0.0, 0, taker_fee
            if trend_dir[prev_i] == 1:
                if c_high > entry_upper[prev_i]:
                    # Realistic Entry: If low <= open, assume Maker fill at (open - offset)
                    if c_low <= c_open:
                        fill_price = c_open * (1.0 - smart_offset)
                        entry_fee_rate = maker_fee
                    else:
                        fill_price = max(c_open, entry_upper[prev_i]) * (1.0 + slippage_rate)
                        entry_fee_rate = taker_fee
                    pending_side = 1; do_entry = True
            elif trend_dir[prev_i] == -1:
                if c_low < entry_lower[prev_i]:
                    # Realistic Entry: If high >= open, assume Maker fill at (open + offset)
                    if c_high >= c_open:
                        fill_price = c_open * (1.0 + smart_offset)
                        entry_fee_rate = maker_fee
                    else:
                        fill_price = min(c_open, entry_lower[prev_i]) * (1.0 - slippage_rate)
                        entry_fee_rate = taker_fee
                    pending_side = -1; do_entry = True

            if do_entry:
                prev_atr = atr[prev_i]
                if np.isnan(prev_atr) or prev_atr <= 0.0:
                    equity_curve[i] = balance; continue
                stop_mult_val = atr_mult
                if hmm_crisis[prev_i] > 0.2:
                    stop_mult_val *= 0.6
                elif hmm_mod_long[prev_i] < 0.7:
                    stop_mult_val *= 0.8
                if pending_side == 1:
                    stop_price = fill_price - (prev_atr * stop_mult_val)
                else:
                    stop_price = fill_price + (prev_atr * stop_mult_val)

                stop_distance = abs(fill_price - stop_price)
                if stop_distance > 0:
                    current_equity = max_capital_usage if use_compounding and balance > max_capital_usage else balance
                    amount = calculate_position_size(fill_price, stop_distance/fill_price, current_equity, current_equity, effective_risk, leverage, sf_raw, garch_kelly_f[prev_i], max_exposure_per_coin=max_exp_per_coin)
                    required_margin = (amount * fill_price) / leverage
                    entry_fee = amount * fill_price * entry_fee_rate
                    if balance >= required_margin + entry_fee:
                        balance -= required_margin + entry_fee; entry_fee_stored = entry_fee
                        in_position = True; pos_side = pending_side; entry_price = fill_price; entry_idx = i; highest = fill_price; lowest = fill_price; has_scaled_out = False
                        triggered, intra_exit_price, pnl_intra, exit_fee_intra = check_intra_bar_stop(pos_side, c_high, c_low, stop_price, entry_price, amount, taker_fee, slippage_rate)
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
        pnl = (exit_price - entry_price) * amount * pos_side - (amount * exit_price * taker_fee)
        balance += (amount * entry_price) / leverage + pnl
        if trade_count < max_trades:
            trades[trade_count] = [entry_idx, last_idx, pos_side, entry_price, exit_price, pnl, amount, entry_fee_stored]
            trade_count += 1

    return trades[:trade_count], balance, equity_curve, funding_paid_total

@njit(nogil=True, cache=True)
def backtest_target_weights_numba(
    close_2d: np.ndarray,
    high_2d: np.ndarray,
    low_2d: np.ndarray,
    open_2d: np.ndarray,
    funding_rate: np.ndarray,
    kill_signal: np.ndarray,
    target_weights: np.ndarray,
    initial_balance: float,
    lev_2d: np.ndarray,
    maker_fee: float,
    taker_fee: float,
    slippage_rate: float,
    rebalance_bars: int,
    max_hold_bars: int,
    short_borrow_daily: float,
    atr_2d: np.ndarray,
    atr_mult: float,
    trail_mult: float,
    use_simple_atr_stop: int,
    max_concurrent: int,
    max_exposure: float,
    max_exp_per_coin: float,
    dd_scaling_threshold: float,
    volume_2d: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Execute toward signed equity fractions `target_weights` (rebalance at open); no CS ranking path.

    Scale-out omitted (v1). Rows `target_weights[i]` used only when ``i > 0`` and
    ``(i % rebalance_bars) == 0``. ``maker_fee`` reserved for API parity; rebalance uses taker/slip fills.
    """
    _ = maker_fee
    n_bars, n_syms = close_2d.shape
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
    highest = np.zeros(n_syms, dtype=np.float64)
    lowest = np.zeros(n_syms, dtype=np.float64)
    entry_lev = np.ones(n_syms, dtype=np.float64)

    dust_skip_cnt, margin_fail_cnt = 0, 0
    spare_a, spare_b = 0, 0
    min_notional_floor_pct = 0.0001

    max_trades = 50000
    trades: np.ndarray = np.zeros((max_trades, 10), dtype=np.float64)
    t_count = 0

    rb = rebalance_bars if rebalance_bars > 0 else 999999999

    for i in range(1, n_bars):
        prev_i = i - 1

        if (i % rb) == 0:
            tw_work = np.zeros(n_syms, dtype=np.float64)
            idx_ord = np.zeros(n_syms, dtype=np.int64)
            for s in range(n_syms):
                idx_ord[s] = s
                tw = float(target_weights[i, s])
                if not np.isfinite(tw):
                    tw = 0.0
                if tw > 1.0:
                    tw = 1.0
                elif tw < -1.0:
                    tw = -1.0
                tw_work[s] = tw

            gross = 0.0
            for s in range(n_syms):
                gross += abs(tw_work[s])
            if gross > max_exposure + 1e-12:
                sg = max_exposure / gross
                for s in range(n_syms):
                    tw_work[s] *= sg

            for a in range(n_syms):
                for b in range(a + 1, n_syms):
                    ia = idx_ord[a]
                    ib = idx_ord[b]
                    if abs(tw_work[ia]) < abs(tw_work[ib]):
                        tmp = idx_ord[a]
                        idx_ord[a] = idx_ord[b]
                        idx_ord[b] = tmp

            mc = max_concurrent if max_concurrent > 0 else n_syms
            for k in range(mc, n_syms):
                sym_z = idx_ord[k]
                tw_work[sym_z] = 0.0

            eq_snap = balance
            for s in range(n_syms):
                if in_pos[s]:
                    opx = open_2d[i, s]
                    if np.isnan(opx):
                        continue
                    eq_snap += (amount[s] * entry_p[s]) / entry_lev[s] + (opx - entry_p[s]) * amount[s] * pos_side[s]

            for s in range(n_syms):
                if not in_pos[s]:
                    continue
                op = open_2d[i, s]
                if np.isnan(op):
                    continue
                tw_s = tw_work[s]
                tgt_notional = eq_snap * tw_s
                ts = 0
                if tgt_notional > 1e-12:
                    ts = 1
                elif tgt_notional < -1e-12:
                    ts = -1
                desired_amt = abs(tgt_notional) / op if ts != 0 else 0.0

                need_exit = False
                if ts == 0:
                    need_exit = True
                elif ts != pos_side[s]:
                    need_exit = True
                elif abs(amount[s] - desired_amt) * op > max(0.01, eq_snap * min_notional_floor_pct):
                    need_exit = True

                if need_exit:
                    # [Non-linear Friction] Slippage = Constant * (ATR/Close) * sqrt(OrderSize / Volume)
                    eff_slip = slippage_rate
                    if volume_2d is not None:
                        vol = volume_2d[i, s]
                        if vol > 0:
                            order_val = amount[s] * op
                            atr_v = atr_2d[i, s]
                            # Heuristic constant 0.1 for market impact
                            impact = 0.1 * (atr_v / op) * np.sqrt(order_val / vol)
                            eff_slip += impact

                    exit_price = op * (1.0 - eff_slip * pos_side[s])
                    pnl = (exit_price - entry_p[s]) * amount[s] * pos_side[s]
                    fee_x = amount[s] * exit_price * taker_fee
                    balance += ((amount[s] * entry_p[s]) / entry_lev[s]) + (pnl - fee_x)
                    if t_count < max_trades:
                        trades[t_count] = [
                            float(s),
                            float(entry_idx[s]),
                            float(i),
                            float(pos_side[s]),
                            entry_p[s],
                            exit_price,
                            pnl - fee_x - fund_fee_stored[s],
                            amount[s],
                            entry_fee_stored[s],
                            fund_fee_stored[s],
                        ]
                        t_count += 1
                    in_pos[s] = False

            eq_snap2 = balance
            used_open = 0.0
            for s in range(n_syms):
                if in_pos[s]:
                    op = open_2d[i, s]
                    if np.isnan(op):
                        continue
                    eq_snap2 += (amount[s] * entry_p[s]) / entry_lev[s] + (op - entry_p[s]) * amount[s] * pos_side[s]
                    used_open += (amount[s] * op) / entry_lev[s]

            free_margin = eq_snap2 - used_open

            for s in range(n_syms):
                if in_pos[s]:
                    continue
                op = open_2d[i, s]
                if np.isnan(op):
                    continue
                tw_s = tw_work[s]
                tgt_notional = eq_snap2 * tw_s
                ts = 0
                if tgt_notional > 1e-12:
                    ts = 1
                elif tgt_notional < -1e-12:
                    ts = -1
                if ts == 0:
                    continue

                # [Non-linear Friction]
                eff_slip = slippage_rate
                if volume_2d is not None:
                    vol = volume_2d[i, s]
                    if vol > 0:
                        order_val = abs(tgt_notional)
                        atr_v = atr_2d[i, s]
                        impact = 0.1 * (atr_v / op) * np.sqrt(order_val / vol)
                        eff_slip += impact

                fill_p = op * (1.0 + eff_slip * float(ts))
                atr_prev = atr_2d[prev_i, s]
                if np.isnan(atr_prev) or atr_prev <= 0.0:
                    continue
                stop_mult = atr_mult
                stop_dist = atr_prev * stop_mult
                abs_tgt = abs(tgt_notional)
                desired_amt = abs_tgt / fill_p
                max_qty_exp = (eq_snap2 * max_exp_per_coin) / fill_p
                if max_qty_exp > 0.0:
                    desired_amt = min(desired_amt, max_qty_exp)

                le_ent = max(1.0, float(lev_2d[i, s]))
                req_m = (desired_amt * fill_p) / le_ent
                e_fee = desired_amt * fill_p * taker_fee
                min_notional = max(0.01, eq_snap2 * min_notional_floor_pct)
                if desired_amt * fill_p < min_notional:
                    dust_skip_cnt += 1
                    continue

                max_qty_margin = (free_margin * 0.97 * le_ent) / fill_p
                if max_qty_margin < 0.0:
                    max_qty_margin = 0.0
                final_qty = min(desired_amt, max_qty_margin)

                if final_qty <= 1e-12:
                    margin_fail_cnt += 1
                    continue

                req_m = (final_qty * fill_p) / le_ent
                e_fee = final_qty * fill_p * taker_fee
                if free_margin < req_m + e_fee:
                    margin_fail_cnt += 1
                    continue

                balance -= req_m + e_fee
                free_margin -= req_m + e_fee
                in_pos[s] = True
                pos_side[s] = ts
                entry_p[s] = fill_p
                entry_idx[s] = i
                entry_lev[s] = le_ent
                amount[s] = final_qty
                entry_fee_stored[s] = e_fee
                fund_fee_stored[s] = 0.0
                highest[s] = fill_p
                lowest[s] = fill_p
                stop_p[s] = fill_p - (stop_dist * float(ts))

        unrealized_total = 0.0
        used_margin_total = 0.0

        for s in range(n_syms):
            if not in_pos[s]:
                continue
            if np.isnan(close_2d[i, s]):
                continue
            cur_p = close_2d[i, s]
            used_margin_total += (amount[s] * cur_p) / entry_lev[s]
            unrealized_total += (cur_p - entry_p[s]) * amount[s] * pos_side[s]
            fund_fee = amount[s] * cur_p * funding_rate[i, s] * pos_side[s]
            fund_fee_stored[s] += fund_fee
            if short_borrow_daily > 0.0 and pos_side[s] == -1:
                balance -= amount[s] * cur_p * (short_borrow_daily / 24.0)

        current_equity = balance + used_margin_total + unrealized_total
        equity_curve[i] = current_equity
        if current_equity > hwm:
            hwm = current_equity

        for s in range(n_syms):
            if not in_pos[s]:
                continue
            c_open = open_2d[i, s]
            c_high = high_2d[i, s]
            c_low = low_2d[i, s]
            if np.isnan(c_open):
                continue

            pos_atr = atr_2d[entry_idx[s], s]
            exit_triggered = False
            exit_price = 0.0

            if kill_signal[prev_i, s] > 0.5:
                exit_triggered = True
                exit_price = c_open * (1.0 - slippage_rate * pos_side[s])

            if not exit_triggered and max_hold_bars > 0 and (i - entry_idx[s]) >= max_hold_bars:
                exit_triggered = True
                exit_price = c_open * (1.0 - slippage_rate * pos_side[s])

            if not exit_triggered:
                if use_simple_atr_stop != 0:
                    if pos_side[s] == 1:
                        if c_open <= stop_p[s]:
                            exit_triggered = True
                            exit_price = c_open * (1.0 - slippage_rate)
                        elif c_low <= stop_p[s]:
                            exit_triggered = True
                            exit_price = stop_p[s] * (1.0 - slippage_rate)
                    else:
                        if c_open >= stop_p[s]:
                            exit_triggered = True
                            exit_price = c_open * (1.0 + slippage_rate)
                        elif c_high >= stop_p[s]:
                            exit_triggered = True
                            exit_price = stop_p[s] * (1.0 + slippage_rate)
                else:
                    if pos_side[s] == 1:
                        if c_high > highest[s]:
                            highest[s] = c_high
                        exit_triggered, exit_price, stop_p[s] = check_long_exit(
                            c_open, c_low, highest[s], pos_atr, stop_p[s], trail_mult, slippage_rate
                        )
                    else:
                        if c_low < lowest[s]:
                            lowest[s] = c_low
                        exit_triggered, exit_price, stop_p[s] = check_short_exit(
                            c_open, c_high, lowest[s], pos_atr, stop_p[s], trail_mult, slippage_rate
                        )

            if exit_triggered:
                pnl_x = (exit_price - entry_p[s]) * amount[s] * pos_side[s]
                fee_x = amount[s] * exit_price * taker_fee
                balance += ((amount[s] * entry_p[s]) / entry_lev[s]) + (pnl_x - fee_x)
                if t_count < max_trades:
                    trades[t_count] = [
                        float(s),
                        float(entry_idx[s]),
                        float(i),
                        float(pos_side[s]),
                        entry_p[s],
                        exit_price,
                        pnl_x - fee_x - fund_fee_stored[s],
                        amount[s],
                        entry_fee_stored[s],
                        fund_fee_stored[s],
                    ]
                    t_count += 1
                in_pos[s] = False

    if n_bars > 0:
        last_idx = n_bars - 1
        for s in range(n_syms):
            if in_pos[s]:
                cur_p = close_2d[last_idx, s]
                if np.isnan(cur_p):
                    cur_p = entry_p[s]
                pnl = (cur_p - entry_p[s]) * amount[s] * pos_side[s] - (amount[s] * cur_p * taker_fee)
                balance += ((amount[s] * entry_p[s]) / entry_lev[s]) + (pnl - fund_fee_stored[s])
                if t_count < max_trades:
                    trades[t_count] = [
                        float(s),
                        float(entry_idx[s]),
                        float(last_idx),
                        float(pos_side[s]),
                        entry_p[s],
                        cur_p,
                        pnl - fund_fee_stored[s],
                        amount[s],
                        entry_fee_stored[s],
                        fund_fee_stored[s],
                    ]
                    t_count += 1

    diag_out = np.array([dust_skip_cnt, margin_fail_cnt, spare_a, spare_b, 0], dtype=np.int64)
    return trades[:t_count], balance, equity_curve, diag_out
