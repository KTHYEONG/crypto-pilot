"""Numba execution: fees, stops, portfolio backtest loop (target-weight driven)."""

from __future__ import annotations

import numpy as np
from numba import njit


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
    # maker_fee: reserved for future limit-order support; rebalance fills are Taker-only
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

            # DD scaling: drawdown 초과 시 전체 가중치 축소
            if dd_scaling_threshold > 0.0 and hwm > 1e-9:
                current_dd = (hwm - eq_snap) / hwm
                if current_dd > dd_scaling_threshold:
                    dd_factor = max(0.1, 1.0 - (current_dd / 0.40))
                    for s in range(n_syms):
                        tw_work[s] *= dd_factor

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
                    balance += (
                        (amount[s] * entry_p[s]) / entry_lev[s]
                    ) + (pnl - fee_x - fund_fee_stored[s])
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
            fr = funding_rate[i, s]
            if not np.isnan(fr):
                fund_fee = amount[s] * cur_p * fr * pos_side[s]
                if np.isfinite(fund_fee):
                    fund_fee_stored[s] += fund_fee
            if short_borrow_daily > 0.0 and pos_side[s] == -1:
                balance -= amount[s] * cur_p * (short_borrow_daily / 24.0)

        current_equity = balance + used_margin_total + unrealized_total
        equity_curve[i] = current_equity
        if current_equity > hwm:
            hwm = current_equity

        # Liquidation guard: 순자산이 0 이하이면 전 포지션 강제청산 후 종료
        if current_equity <= 0.0:
            for s in range(n_syms):
                if not in_pos[s]:
                    continue
                cur_p = close_2d[i, s]
                if np.isnan(cur_p):
                    cur_p = entry_p[s]
                pnl_x = (cur_p - entry_p[s]) * amount[s] * float(pos_side[s])
                fee_x = amount[s] * cur_p * taker_fee
                if t_count < max_trades:
                    trades[t_count] = [
                        float(s), float(entry_idx[s]), float(i),
                        float(pos_side[s]), entry_p[s], cur_p,
                        pnl_x - fee_x - fund_fee_stored[s],
                        amount[s], entry_fee_stored[s], fund_fee_stored[s],
                    ]
                    t_count += 1
                in_pos[s] = False
            equity_curve[i] = 0.0
            break

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
                balance += (
                    (amount[s] * entry_p[s]) / entry_lev[s]
                ) + (pnl_x - fee_x - fund_fee_stored[s])
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
