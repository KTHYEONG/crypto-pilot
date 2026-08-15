"""Numba execution: fees, stops, portfolio backtest loop (target-weight driven)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

import numpy as np
from numba import njit

F = TypeVar("F", bound=Callable[..., Any])


def _typed_njit(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for numba.njit to keep strict mypy signatures on decorated funcs."""

    def _decorator(func: F) -> F:
        return cast(F, njit(*args, **kwargs)(func))

    return _decorator


@_typed_njit(inline="always")
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


@_typed_njit(inline="always")
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


@_typed_njit(nogil=True, cache=True)
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
    bar_hours: float,
    atr_2d: np.ndarray,
    atr_mult: float,
    trail_mult: float,
    use_simple_atr_stop: int,
    max_concurrent: int,
    max_exposure: float,
    max_exp_per_coin: float,
    dd_scaling_threshold: float,
    volume_2d: np.ndarray | None = None,
    candidate_stop_atr_mult: np.ndarray | None = None,
    candidate_take_profit_atr_mult: np.ndarray | None = None,
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
    take_p = np.zeros(n_syms, dtype=np.float64)
    highest = np.zeros(n_syms, dtype=np.float64)
    lowest = np.zeros(n_syms, dtype=np.float64)
    entry_lev = np.ones(n_syms, dtype=np.float64)

    dust_skip_cnt, margin_fail_cnt = 0, 0
    spare_a, spare_b = 0, 0
    min_notional_floor_pct = 0.0001

    liq_p = np.zeros(n_syms, dtype=np.float64)
    maint_margin_rate = 0.005  # Binance USDT-M 기본 유지증거금율 0.5%

    max_trades = max(50_000, n_bars * n_syms * 3)
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
                if (
                    ts == 0
                    or ts != pos_side[s]
                    or abs(amount[s] - desired_amt) * op > max(0.01, eq_snap * min_notional_floor_pct)
                ):
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
                    balance += ((amount[s] * entry_p[s]) / entry_lev[s]) + (pnl - fee_x - fund_fee_stored[s])
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
                stop_mult = atr_mult
                tp_mult = 0.0
                if candidate_stop_atr_mult is not None:
                    raw_stop_mult = float(candidate_stop_atr_mult[i, s])
                    if np.isfinite(raw_stop_mult) and raw_stop_mult > 0.0:
                        stop_mult = raw_stop_mult
                if candidate_take_profit_atr_mult is not None:
                    raw_tp_mult = float(candidate_take_profit_atr_mult[i, s])
                    if np.isfinite(raw_tp_mult) and raw_tp_mult > 0.0:
                        tp_mult = raw_tp_mult
                stop_dist = atr_prev * stop_mult
                tp_dist = atr_prev * tp_mult
                stop_p[s] = fill_p - (stop_dist * float(ts))
                take_p[s] = fill_p + (tp_dist * float(ts)) if tp_mult > 0.0 else 0.0
                # Isolated margin liquidation price
                # Long: entry*(1 - 1/lev + MMR), Short: entry*(1 + 1/lev - MMR)
                liq_p[s] = fill_p * (1.0 - (1.0 / le_ent) * float(ts) + maint_margin_rate * float(ts))

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
                # bar_hours로 스케일: 4h봉=4/24, 1h봉=1/24. 일별 borrow를 bar당 비용으로 환산.
                balance -= amount[s] * cur_p * (short_borrow_daily * bar_hours / 24.0)

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
                        float(s),
                        float(entry_idx[s]),
                        float(i),
                        float(pos_side[s]),
                        entry_p[s],
                        cur_p,
                        pnl_x - fee_x - fund_fee_stored[s],
                        amount[s],
                        entry_fee_stored[s],
                        fund_fee_stored[s],
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

            # Liquidation check (isolated margin model): priority over stop-loss
            if not exit_triggered and liq_p[s] > 0.0:
                if pos_side[s] == 1 and c_low <= liq_p[s]:
                    exit_triggered = True
                    exit_price = liq_p[s] * (1.0 - slippage_rate)
                elif pos_side[s] == -1 and c_high >= liq_p[s]:
                    exit_triggered = True
                    exit_price = liq_p[s] * (1.0 + slippage_rate)

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
                    if not exit_triggered and take_p[s] > 0.0:
                        if pos_side[s] == 1:
                            if c_open >= take_p[s]:
                                exit_triggered = True
                                exit_price = c_open * (1.0 - slippage_rate)
                            elif c_high >= take_p[s]:
                                exit_triggered = True
                                exit_price = take_p[s] * (1.0 - slippage_rate)
                        else:
                            if c_open <= take_p[s]:
                                exit_triggered = True
                                exit_price = c_open * (1.0 + slippage_rate)
                            elif c_low <= take_p[s]:
                                exit_triggered = True
                                exit_price = take_p[s] * (1.0 + slippage_rate)
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
                    if not exit_triggered and take_p[s] > 0.0:
                        if pos_side[s] == 1:
                            if c_open >= take_p[s]:
                                exit_triggered = True
                                exit_price = c_open * (1.0 - slippage_rate)
                            elif c_high >= take_p[s]:
                                exit_triggered = True
                                exit_price = take_p[s] * (1.0 - slippage_rate)
                        else:
                            if c_open <= take_p[s]:
                                exit_triggered = True
                                exit_price = c_open * (1.0 + slippage_rate)
                            elif c_low <= take_p[s]:
                                exit_triggered = True
                                exit_price = take_p[s] * (1.0 + slippage_rate)

            if exit_triggered:
                pnl_x = (exit_price - entry_p[s]) * amount[s] * pos_side[s]
                fee_x = amount[s] * exit_price * taker_fee
                balance += ((amount[s] * entry_p[s]) / entry_lev[s]) + (pnl_x - fee_x - fund_fee_stored[s])
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
                liq_p[s] = 0.0
                take_p[s] = 0.0

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


@_typed_njit(nogil=True, cache=True)
def backtest_target_weights_intrabar_numba(
    decision_close_2d: np.ndarray,
    decision_high_2d: np.ndarray,
    decision_low_2d: np.ndarray,
    decision_open_2d: np.ndarray,
    target_weights: np.ndarray,
    lev_2d: np.ndarray,
    atr_2d: np.ndarray,
    kill_signal_2d: np.ndarray,
    path_open_2d: np.ndarray,
    path_high_2d: np.ndarray,
    path_low_2d: np.ndarray,
    path_close_2d: np.ndarray,
    decision_start_1m_idx: np.ndarray,
    decision_end_1m_idx: np.ndarray,
    initial_balance: float,
    maker_fee: float,
    taker_fee: float,
    slippage_rate: float,
    rebalance_bars: int,
    max_hold_bars: int,
    short_borrow_daily: float,
    atr_mult: float,
    trail_mult: float,
    use_simple_atr_stop: int,
    max_concurrent: int,
    max_exposure: float,
    max_exp_per_coin: float,
    dd_scaling_threshold: float,
    funding_event_mask_1m: np.ndarray | None = None,
    funding_rate_1m: np.ndarray | None = None,
    volume_1m_2d: np.ndarray | None = None,
    candidate_stop_atr_mult: np.ndarray | None = None,
    candidate_take_profit_atr_mult: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Intrabar execution with 1m path scan and decision-level rebalancing."""
    _ = maker_fee
    _ = decision_high_2d
    _ = decision_low_2d
    n_decisions, n_syms = decision_close_2d.shape
    n_path, _ = path_close_2d.shape

    balance = initial_balance
    equity_curve = np.zeros(n_decisions, dtype=np.float64)
    if n_decisions > 0:
        equity_curve[0] = initial_balance
    hwm = initial_balance

    in_pos = np.zeros(n_syms, dtype=np.bool_)
    pos_side = np.zeros(n_syms, dtype=np.int8)
    entry_p = np.zeros(n_syms, dtype=np.float64)
    entry_decision_idx = np.zeros(n_syms, dtype=np.int32)
    amount = np.zeros(n_syms, dtype=np.float64)
    entry_fee_stored = np.zeros(n_syms, dtype=np.float64)
    fund_fee_stored = np.zeros(n_syms, dtype=np.float64)
    stop_p = np.zeros(n_syms, dtype=np.float64)
    take_p = np.zeros(n_syms, dtype=np.float64)
    highest = np.zeros(n_syms, dtype=np.float64)
    lowest = np.zeros(n_syms, dtype=np.float64)
    entry_lev = np.ones(n_syms, dtype=np.float64)
    liq_p = np.zeros(n_syms, dtype=np.float64)
    maint_margin_rate = 0.005  # Binance USDT-M 기본 유지증거금율 0.5%

    dust_skip_cnt, margin_fail_cnt = 0, 0
    spare_a, spare_b = 0, 0
    min_notional_floor_pct = 0.0001

    max_trades = max(50_000, n_decisions * n_syms * 3)
    trades = np.zeros((max_trades, 10), dtype=np.float64)
    t_count = 0

    rb = rebalance_bars if rebalance_bars > 0 else 999999999

    for i in range(1, n_decisions):
        prev_i = i - 1
        start_m = int(decision_start_1m_idx[i])
        end_m = int(decision_end_1m_idx[i])
        if start_m < 0:
            start_m = 0
        if end_m > n_path:
            end_m = n_path
        if end_m <= start_m:
            equity_curve[i] = equity_curve[prev_i]
            continue

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
                tw_work[idx_ord[k]] = 0.0

            eq_snap = balance
            for s in range(n_syms):
                if not in_pos[s]:
                    continue
                opx = path_open_2d[start_m, s]
                if np.isnan(opx):
                    continue
                eq_snap += (amount[s] * entry_p[s]) / entry_lev[s] + (opx - entry_p[s]) * amount[s] * pos_side[s]

            if dd_scaling_threshold > 0.0 and hwm > 1e-9:
                current_dd = (hwm - eq_snap) / hwm
                if current_dd > dd_scaling_threshold:
                    dd_factor = max(0.1, 1.0 - (current_dd / 0.40))
                    for s in range(n_syms):
                        tw_work[s] *= dd_factor

            for s in range(n_syms):
                if not in_pos[s]:
                    continue
                op = path_open_2d[start_m, s]
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
                if (
                    ts == 0
                    or ts != pos_side[s]
                    or abs(amount[s] - desired_amt) * op > max(0.01, eq_snap * min_notional_floor_pct)
                ):
                    need_exit = True

                if need_exit:
                    eff_slip = slippage_rate
                    if volume_1m_2d is not None:
                        vol = volume_1m_2d[start_m, s]
                        if vol > 0.0:
                            order_val = amount[s] * op
                            atr_v = atr_2d[i, s]
                            impact = 0.1 * (atr_v / op) * np.sqrt(order_val / vol)
                            eff_slip += impact

                    exit_price = op * (1.0 - eff_slip * pos_side[s])
                    pnl = (exit_price - entry_p[s]) * amount[s] * pos_side[s]
                    fee_x = amount[s] * exit_price * taker_fee
                    balance += ((amount[s] * entry_p[s]) / entry_lev[s]) + (pnl - fee_x - fund_fee_stored[s])
                    if t_count < max_trades:
                        trades[t_count] = [
                            float(s),
                            float(entry_decision_idx[s]),
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
                if not in_pos[s]:
                    continue
                op = path_open_2d[start_m, s]
                if np.isnan(op):
                    continue
                eq_snap2 += (amount[s] * entry_p[s]) / entry_lev[s] + (op - entry_p[s]) * amount[s] * pos_side[s]
                used_open += (amount[s] * op) / entry_lev[s]
            free_margin = eq_snap2 - used_open

            for s in range(n_syms):
                if in_pos[s]:
                    continue
                op = path_open_2d[start_m, s]
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

                eff_slip = slippage_rate
                if volume_1m_2d is not None:
                    vol = volume_1m_2d[start_m, s]
                    if vol > 0.0:
                        order_val = abs(tgt_notional)
                        atr_v = atr_2d[i, s]
                        impact = 0.1 * (atr_v / op) * np.sqrt(order_val / vol)
                        eff_slip += impact

                fill_p = op * (1.0 + eff_slip * float(ts))
                atr_prev = atr_2d[prev_i, s]
                if np.isnan(atr_prev) or atr_prev <= 0.0:
                    continue
                stop_mult = atr_mult
                tp_mult = 0.0
                if candidate_stop_atr_mult is not None:
                    raw_stop_mult = float(candidate_stop_atr_mult[i, s])
                    if np.isfinite(raw_stop_mult) and raw_stop_mult > 0.0:
                        stop_mult = raw_stop_mult
                if candidate_take_profit_atr_mult is not None:
                    raw_tp_mult = float(candidate_take_profit_atr_mult[i, s])
                    if np.isfinite(raw_tp_mult) and raw_tp_mult > 0.0:
                        tp_mult = raw_tp_mult
                stop_dist = atr_prev * stop_mult
                tp_dist = atr_prev * tp_mult
                abs_tgt = abs(tgt_notional)
                desired_amt = abs_tgt / fill_p
                max_qty_exp = (eq_snap2 * max_exp_per_coin) / fill_p
                if max_qty_exp > 0.0:
                    desired_amt = min(desired_amt, max_qty_exp)

                le_ent = max(1.0, float(lev_2d[i, s]))
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
                entry_decision_idx[s] = i
                entry_lev[s] = le_ent
                amount[s] = final_qty
                entry_fee_stored[s] = e_fee
                fund_fee_stored[s] = 0.0
                highest[s] = fill_p
                lowest[s] = fill_p
                stop_p[s] = fill_p - (stop_dist * float(ts))
                take_p[s] = fill_p + (tp_dist * float(ts)) if tp_mult > 0.0 else 0.0
                # Isolated margin liquidation price
                liq_p[s] = fill_p * (1.0 - (1.0 / le_ent) * float(ts) + maint_margin_rate * float(ts))

        window_liq = False
        liq_m = start_m
        for m in range(start_m, end_m):
            unrealized_total = 0.0
            used_margin_total = 0.0

            for s in range(n_syms):
                if not in_pos[s]:
                    continue
                cur_p = path_close_2d[m, s]
                if np.isnan(cur_p):
                    continue
                used_margin_total += (amount[s] * cur_p) / entry_lev[s]
                unrealized_total += (cur_p - entry_p[s]) * amount[s] * pos_side[s]

                if (
                    funding_event_mask_1m is not None
                    and funding_rate_1m is not None
                    and funding_event_mask_1m[m, s] > 0.5
                ):
                    fr = funding_rate_1m[m, s]
                    if not np.isnan(fr):
                        fund_fee = amount[s] * cur_p * fr * pos_side[s]
                        if np.isfinite(fund_fee):
                            fund_fee_stored[s] += fund_fee

                if short_borrow_daily > 0.0 and pos_side[s] == -1:
                    balance -= amount[s] * cur_p * (short_borrow_daily / 1440.0)

            current_equity = balance + used_margin_total + unrealized_total
            if current_equity > hwm:
                hwm = current_equity
            if current_equity <= 0.0:
                window_liq = True
                liq_m = m
                break

            for s in range(n_syms):
                if not in_pos[s]:
                    continue
                c_open = path_open_2d[m, s]
                c_high = path_high_2d[m, s]
                c_low = path_low_2d[m, s]
                if np.isnan(c_open):
                    continue

                pos_atr = atr_2d[entry_decision_idx[s], s]
                exit_triggered = False
                exit_price = 0.0

                if kill_signal_2d[prev_i, s] > 0.5:
                    exit_triggered = True
                    exit_price = c_open * (1.0 - slippage_rate * pos_side[s])

                if not exit_triggered and max_hold_bars > 0 and (i - entry_decision_idx[s]) >= max_hold_bars:
                    exit_triggered = True
                    exit_price = c_open * (1.0 - slippage_rate * pos_side[s])

                # Liquidation check (isolated margin model): priority over stop-loss
                if not exit_triggered and liq_p[s] > 0.0:
                    if pos_side[s] == 1 and c_low <= liq_p[s]:
                        exit_triggered = True
                        exit_price = liq_p[s] * (1.0 - slippage_rate)
                    elif pos_side[s] == -1 and c_high >= liq_p[s]:
                        exit_triggered = True
                        exit_price = liq_p[s] * (1.0 + slippage_rate)

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
                        if not exit_triggered and take_p[s] > 0.0:
                            if pos_side[s] == 1:
                                if c_open >= take_p[s]:
                                    exit_triggered = True
                                    exit_price = c_open * (1.0 - slippage_rate)
                                elif c_high >= take_p[s]:
                                    exit_triggered = True
                                    exit_price = take_p[s] * (1.0 - slippage_rate)
                            else:
                                if c_open <= take_p[s]:
                                    exit_triggered = True
                                    exit_price = c_open * (1.0 + slippage_rate)
                                elif c_low <= take_p[s]:
                                    exit_triggered = True
                                    exit_price = take_p[s] * (1.0 + slippage_rate)
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
                        if not exit_triggered and take_p[s] > 0.0:
                            if pos_side[s] == 1:
                                if c_open >= take_p[s]:
                                    exit_triggered = True
                                    exit_price = c_open * (1.0 - slippage_rate)
                                elif c_high >= take_p[s]:
                                    exit_triggered = True
                                    exit_price = take_p[s] * (1.0 - slippage_rate)
                            else:
                                if c_open <= take_p[s]:
                                    exit_triggered = True
                                    exit_price = c_open * (1.0 + slippage_rate)
                                elif c_low <= take_p[s]:
                                    exit_triggered = True
                                    exit_price = take_p[s] * (1.0 + slippage_rate)

                if exit_triggered:
                    pnl_x = (exit_price - entry_p[s]) * amount[s] * pos_side[s]
                    fee_x = amount[s] * exit_price * taker_fee
                    balance += ((amount[s] * entry_p[s]) / entry_lev[s]) + (pnl_x - fee_x - fund_fee_stored[s])
                    if t_count < max_trades:
                        trades[t_count] = [
                            float(s),
                            float(entry_decision_idx[s]),
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
                    liq_p[s] = 0.0
                    take_p[s] = 0.0

        if window_liq:
            for s in range(n_syms):
                if not in_pos[s]:
                    continue
                cur_p = path_close_2d[liq_m, s]
                if np.isnan(cur_p):
                    cur_p = entry_p[s]
                pnl_x = (cur_p - entry_p[s]) * amount[s] * float(pos_side[s])
                fee_x = amount[s] * cur_p * taker_fee
                if t_count < max_trades:
                    trades[t_count] = [
                        float(s),
                        float(entry_decision_idx[s]),
                        float(i),
                        float(pos_side[s]),
                        entry_p[s],
                        cur_p,
                        pnl_x - fee_x - fund_fee_stored[s],
                        amount[s],
                        entry_fee_stored[s],
                        fund_fee_stored[s],
                    ]
                    t_count += 1
                in_pos[s] = False
            equity_curve[i] = 0.0
            break

        mark_m = end_m - 1
        unrealized = 0.0
        used_margin = 0.0
        for s in range(n_syms):
            if not in_pos[s]:
                continue
            cp = path_close_2d[mark_m, s]
            if np.isnan(cp):
                continue
            used_margin += (amount[s] * cp) / entry_lev[s]
            unrealized += (cp - entry_p[s]) * amount[s] * pos_side[s]
        eq = balance + used_margin + unrealized
        equity_curve[i] = eq
        if eq > hwm:
            hwm = eq

    if n_decisions > 0:
        last_decision = n_decisions - 1
        for s in range(n_syms):
            if in_pos[s]:
                m_last = int(decision_end_1m_idx[last_decision]) - 1
                if m_last < 0:
                    m_last = 0
                if m_last >= n_path:
                    m_last = n_path - 1
                cur_p = path_close_2d[m_last, s]
                if np.isnan(cur_p):
                    cur_p = entry_p[s]
                pnl = (cur_p - entry_p[s]) * amount[s] * pos_side[s] - (amount[s] * cur_p * taker_fee)
                balance += ((amount[s] * entry_p[s]) / entry_lev[s]) + (pnl - fund_fee_stored[s])
                if t_count < max_trades:
                    trades[t_count] = [
                        float(s),
                        float(entry_decision_idx[s]),
                        float(last_decision),
                        float(pos_side[s]),
                        entry_p[s],
                        cur_p,
                        pnl - fund_fee_stored[s],
                        amount[s],
                        entry_fee_stored[s],
                        fund_fee_stored[s],
                    ]
                    t_count += 1

    diag_out = np.array([dust_skip_cnt, margin_fail_cnt, spare_a, spare_b, 1], dtype=np.int64)
    return trades[:t_count], balance, equity_curve, diag_out


def backtest_target_weights_intrabar(
    decision_close_2d: np.ndarray,
    decision_high_2d: np.ndarray,
    decision_low_2d: np.ndarray,
    decision_open_2d: np.ndarray,
    target_weights: np.ndarray,
    lev_2d: np.ndarray,
    atr_2d: np.ndarray,
    kill_signal_2d: np.ndarray,
    path_open_2d: np.ndarray,
    path_high_2d: np.ndarray,
    path_low_2d: np.ndarray,
    path_close_2d: np.ndarray,
    decision_start_1m_idx: np.ndarray,
    decision_end_1m_idx: np.ndarray,
    initial_balance: float,
    maker_fee: float,
    taker_fee: float,
    slippage_rate: float,
    rebalance_bars: int,
    max_hold_bars: int,
    short_borrow_daily: float,
    atr_mult: float,
    trail_mult: float,
    use_simple_atr_stop: int,
    max_concurrent: int,
    max_exposure: float,
    max_exp_per_coin: float,
    dd_scaling_threshold: float,
    funding_event_mask_1m: np.ndarray | None = None,
    funding_rate_1m: np.ndarray | None = None,
    volume_1m_2d: np.ndarray | None = None,
    mark_price_1m: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Python wrapper for backtest_target_weights_intrabar_numba.

    mark_price_1m 파라미터를 처리하여 Numba 함수에 전달한다.
    mark_price_1m이 제공되면 청산 판정 기준으로 mark_price를 사용하기 위해
    path_low_2d (Long 청산)와 path_high_2d (Short 청산)를 mark_price로 교체한다.

    Args:
        mark_price_1m: shape [B_1m, N]. None이면 기존 exec_low/high 사용.

    Returns:
        (trades, final_balance, equity_curve, diag) — Numba 함수와 동일 형식.

    """
    if mark_price_1m is not None:
        mark_arr = np.asarray(mark_price_1m, dtype=np.float64)
        # Long 청산 판정에 사용되는 path_low → mark_price로 교체
        # Short 청산 판정에 사용되는 path_high → mark_price로 교체
        eff_path_low = mark_arr
        eff_path_high = mark_arr
    else:
        eff_path_low = path_low_2d
        eff_path_high = path_high_2d

    return backtest_target_weights_intrabar_numba(
        decision_close_2d=decision_close_2d,
        decision_high_2d=decision_high_2d,
        decision_low_2d=decision_low_2d,
        decision_open_2d=decision_open_2d,
        target_weights=target_weights,
        lev_2d=lev_2d,
        atr_2d=atr_2d,
        kill_signal_2d=kill_signal_2d,
        path_open_2d=path_open_2d,
        path_high_2d=eff_path_high,
        path_low_2d=eff_path_low,
        path_close_2d=path_close_2d,
        decision_start_1m_idx=decision_start_1m_idx,
        decision_end_1m_idx=decision_end_1m_idx,
        initial_balance=float(initial_balance),
        maker_fee=float(maker_fee),
        taker_fee=float(taker_fee),
        slippage_rate=float(slippage_rate),
        rebalance_bars=int(rebalance_bars),
        max_hold_bars=int(max_hold_bars),
        short_borrow_daily=float(short_borrow_daily),
        atr_mult=float(atr_mult),
        trail_mult=float(trail_mult),
        use_simple_atr_stop=int(use_simple_atr_stop),
        max_concurrent=int(max_concurrent),
        max_exposure=float(max_exposure),
        max_exp_per_coin=float(max_exp_per_coin),
        dd_scaling_threshold=float(dd_scaling_threshold),
        funding_event_mask_1m=funding_event_mask_1m,
        funding_rate_1m=funding_rate_1m,
        volume_1m_2d=volume_1m_2d,
    )
