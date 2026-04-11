"""
2D Portfolio backtest engine: single global balance, per-symbol state arrays.
For use when mode=multi (aligned Time x Symbol matrix).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from numba import njit

from config.opt_config import OPT_FUTURES_CONFIG
from src.domain.futures.engine_logic_futures import (
    calculate_position_size,
    check_intra_bar_stop,
    check_long_exit,
    check_short_exit,
    process_long_scale_out,
    process_short_scale_out,
)

_logger: logging.Logger = logging.getLogger(__name__)


class PortfolioBacktestEngineFast:
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

        cfg = OPT_FUTURES_CONFIG
        self.max_concurrent_positions = int(cfg.get("FUTURES_MAX_CONCURRENT_POSITIONS", 2))

    def run(
        self,
    ) -> tuple[pd.DataFrame, np.ndarray, float]:
        close_2d = self.data["close"]
        high_2d = self.data["high"]
        low_2d = self.data["low"]
        open_2d = self.data["open"]

        entry_upper = self.data["entry_upper"]
        entry_lower = self.data["entry_lower"]
        trend_dir = self.data["trend_direction"]
        strength_filter = self.data["strength_filter"]
        atr_2d = self.data["atr"]
        garch_kelly_f = self.data["garch_kelly_f"]
        funding_rate = self.data["funding_rate_sum"]
        slot_rank_score = self.data["slot_rank_score"]

        _logger.debug(
            f"Engine Multi: symbols={self.symbols}, max_concurrent={self.max_concurrent_positions}"
        )

        l_atr_mult = float(self.params.get("LONG_ATR_MULT", 3.0))
        l_trail_mult = float(self.params.get("LONG_TRAIL_MULT", 3.0))
        s_atr_mult = float(self.params.get("SHORT_ATR_MULT", 2.0))
        s_tp_mult = float(self.params.get("SHORT_TP_MULT", 3.0))
        l_scale_atr = float(self.params.get("LONG_SCALE_ATR_MULT", 3.0))
        s_trail_mult = float(self.params.get("SHORT_TRAIL_MULT", 3.0))

        # --- Dynamic Asset Management Params ---
        max_exp_per_coin = float(self.params.get("MAX_EXPOSURE_PER_COIN", 1.5))
        dd_scaling_threshold = float(self.params.get("DD_SCALING_THRESHOLD", 0.15))

        trades_arr, final_balance, equity_curve = backtest_portfolio_numba(
            close_2d,
            high_2d,
            low_2d,
            open_2d,
            entry_upper,
            entry_lower,
            trend_dir,
            strength_filter,
            atr_2d,
            garch_kelly_f,
            funding_rate,
            slot_rank_score,
            self.initial_balance,
            self.leverage,
            self.fee_rate,
            self.slippage_rate,
            self.risk_per_trade,
            l_atr_mult,
            l_trail_mult,
            s_atr_mult,
            s_tp_mult,
            l_scale_atr,
            s_trail_mult,
            self.max_concurrent_positions,
            self.max_exposure,
            max_exp_per_coin,
            dd_scaling_threshold,
        )

        trades_list: list[dict[str, Any]] = []
        for t in trades_arr:
            sym_idx = int(t[0])
            trades_list.append(
                {
                    "symbol": self.symbols[sym_idx],
                    "entry_idx": int(t[1]),
                    "exit_idx": int(t[2]),
                    "side": "LONG" if t[3] == 1 else "SHORT",
                    "entry_price": float(t[4]),
                    "exit_price": float(t[5]),
                    "pnl": float(t[6]),
                    "amount": float(t[7]),
                    "entry_fee": float(t[8]),
                    "funding_fee": float(t[9]),
                }
            )

        _logger.debug(f"Engine Finished. Trades: {len(trades_list)}")
        return pd.DataFrame(trades_list), equity_curve, final_balance


@njit(nogil=True, cache=True)
def backtest_portfolio_numba(
    close_2d: np.ndarray,
    high_2d: np.ndarray,
    low_2d: np.ndarray,
    open_2d: np.ndarray,
    entry_upper: np.ndarray,
    entry_lower: np.ndarray,
    trend_dir: np.ndarray,
    strength_filter: np.ndarray,
    atr_2d: np.ndarray,
    garch_kelly_f: np.ndarray,
    funding_rate: np.ndarray,
    slot_rank_score: np.ndarray,
    initial_balance: float,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
    risk_per_trade: float,
    l_atr_mult: float,
    l_trail_mult: float,
    s_atr_mult: float,
    s_tp_mult: float,
    l_scale_atr: float,
    s_trail_mult: float,
    max_concurrent: int,
    max_exposure: float,
    max_exp_per_coin: float,
    dd_scaling_threshold: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    n_bars = close_2d.shape[0]
    n_syms = close_2d.shape[1]

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
    has_scaled = np.zeros(n_syms, dtype=np.bool_)

    max_trades = 50000
    trades = np.zeros((max_trades, 10), dtype=np.float64)
    t_count = 0

    for i in range(1, n_bars):
        unrealized_total = 0.0
        just_exited = np.zeros(n_syms, dtype=np.bool_)
        used_margin_total = 0.0
        num_open_pos = 0

        for s in range(n_syms):
            if in_pos[s]:
                num_open_pos += 1
                if np.isnan(close_2d[i, s]):
                    continue

                # Apply funding fee
                fund_fee = amount[s] * close_2d[i, s] * funding_rate[i, s] * pos_side[s]
                fund_fee_stored[s] += fund_fee
                balance -= fund_fee

                u_pnl = (close_2d[i, s] - entry_p[s]) * amount[s] * pos_side[s]
                unrealized_total += u_pnl
                used_margin_total += (amount[s] * entry_p[s]) / leverage

        current_equity = balance + used_margin_total + unrealized_total
        equity_curve[i] = current_equity
        if current_equity > hwm:
            hwm = current_equity

        if current_equity <= 0:
            break

        # --- Drawdown-Dependent Risk Scaling (Anti-Martingale) ---
        current_dd = (hwm - current_equity) / hwm if hwm > 0 else 0.0
        dd_scaling_factor = 1.0
        if current_dd > dd_scaling_threshold:
            # Linear scaling: 40% MDD -> risk halved (example logic)
            dd_scaling_factor = max(0.1, 1.0 - (current_dd / 0.40))
        
        effective_risk_per_trade = risk_per_trade * dd_scaling_factor

        free_margin = current_equity - used_margin_total
        allowed_margin = max(0.0, (current_equity * max_exposure) - used_margin_total)
        free_margin = min(free_margin, allowed_margin)

        # --- Exit logic ---
        for s in range(n_syms):
            if not in_pos[s]:
                continue
            if np.isnan(close_2d[i, s]):
                continue

            c_open = open_2d[i, s]
            c_high = high_2d[i, s]
            c_low = low_2d[i, s]
            pos_atr = atr_2d[entry_idx[s], s]
            exit_triggered = False
            exit_price = 0.0

            if pos_side[s] == 1:
                if c_high > highest[s]:
                    highest[s] = c_high

                if not has_scaled[s]:
                    triggered, sc_price, sc_amount, pnl_scale, exit_fee_scale = (
                        process_long_scale_out(
                            c_open, c_high, entry_p[s], pos_atr, l_scale_atr, amount[s], fee_rate
                        )
                    )
                    if triggered:
                        sc_fund = fund_fee_stored[s] / 2.0
                        balance += (sc_amount * entry_p[s]) / leverage + (
                            pnl_scale - exit_fee_scale
                        )

                        trades[t_count] = [
                            s,
                            entry_idx[s],
                            i,
                            1.0,
                            entry_p[s],
                            sc_price,
                            pnl_scale - exit_fee_scale - sc_fund,
                            sc_amount,
                            entry_fee_stored[s] / 2.0,
                            sc_fund,
                        ]
                        t_count += 1

                        amount[s] -= sc_amount
                        entry_fee_stored[s] = entry_fee_stored[s] / 2.0
                        fund_fee_stored[s] = fund_fee_stored[s] / 2.0
                        has_scaled[s] = True

                exit_triggered, exit_price, stop_p[s] = check_long_exit(
                    c_open, c_low, highest[s], pos_atr, stop_p[s], l_trail_mult, slippage_rate
                )

            elif pos_side[s] == -1:
                if c_low < lowest[s]:
                    lowest[s] = c_low

                if not has_scaled[s]:
                    triggered, sc_price, sc_amount, pnl_scale, exit_fee_scale = (
                        process_short_scale_out(
                            c_open, c_low, entry_p[s], pos_atr, s_tp_mult, amount[s], fee_rate
                        )
                    )
                    if triggered:
                        sc_fund = fund_fee_stored[s] / 2.0
                        balance += (sc_amount * entry_p[s]) / leverage + (
                            pnl_scale - exit_fee_scale
                        )

                        trades[t_count] = [
                            s,
                            entry_idx[s],
                            i,
                            -1.0,
                            entry_p[s],
                            sc_price,
                            pnl_scale - exit_fee_scale - sc_fund,
                            sc_amount,
                            entry_fee_stored[s] / 2.0,
                            sc_fund,
                        ]
                        t_count += 1

                        amount[s] -= sc_amount
                        entry_fee_stored[s] = entry_fee_stored[s] / 2.0
                        fund_fee_stored[s] = fund_fee_stored[s] / 2.0
                        has_scaled[s] = True
                        stop_p[s] = entry_p[s] - (entry_p[s] * fee_rate * 2.0)
                else:
                    exit_triggered, exit_price, stop_p[s] = check_short_exit(
                        c_open, c_high, lowest[s], pos_atr, stop_p[s], s_trail_mult, slippage_rate
                    )

            if exit_triggered:
                if pos_side[s] == 1:
                    pnl = (exit_price - entry_p[s]) * amount[s]
                else:
                    pnl = (entry_p[s] - exit_price) * amount[s]
                fee = amount[s] * exit_price * fee_rate

                balance += ((amount[s] * entry_p[s]) / leverage) + (pnl - fee)

                trades[t_count] = [
                    s,
                    entry_idx[s],
                    i,
                    float(pos_side[s]),
                    entry_p[s],
                    exit_price,
                    pnl - fee - fund_fee_stored[s],
                    amount[s],
                    entry_fee_stored[s],
                    fund_fee_stored[s],
                ]
                t_count += 1
                in_pos[s] = False
                just_exited[s] = True
                num_open_pos -= 1

        unrealized_total = 0.0
        used_margin_total = 0.0
        for s in range(n_syms):
            if in_pos[s] and not np.isnan(close_2d[i, s]):
                unrealized_total += (close_2d[i, s] - entry_p[s]) * amount[s] * pos_side[s]
                used_margin_total += (amount[s] * entry_p[s]) / leverage
        current_equity = balance + used_margin_total + unrealized_total
        free_margin = current_equity - used_margin_total

        # --- Entry logic with Concurrency limit and Rank Priority ---
        if num_open_pos < max_concurrent:
            prev_i = i - 1

            # 1. Collect potential candidates
            candidates = []
            for s in range(n_syms):
                if in_pos[s] or just_exited[s]:
                    continue
                if np.isnan(open_2d[i, s]) or np.isnan(strength_filter[prev_i, s]):
                    continue

                sf = strength_filter[prev_i, s]
                if sf > 0.0 and not np.isnan(sf):
                    c_open = open_2d[i, s]
                    p_side = 0
                    fill_p = 0.0

                    if trend_dir[prev_i, s] == 1 and high_2d[i, s] > entry_upper[prev_i, s]:
                        fill_p = max(c_open, entry_upper[prev_i, s]) * (1.0 + slippage_rate)
                        p_side = 1
                    elif trend_dir[prev_i, s] == -1 and low_2d[i, s] < entry_lower[prev_i, s]:
                        fill_p = min(c_open, entry_lower[prev_i, s]) * (1.0 - slippage_rate)
                        p_side = -1

                    if p_side != 0:
                        # Store rank score (slot_rank_score) and other info
                        candidates.append((slot_rank_score[prev_i, s], s, p_side, fill_p, sf))

            if candidates:
                # 2. Sort candidates by rank_score descending (using simple sort for Numba compatibility)
                for c1 in range(len(candidates)):
                    for c2 in range(c1 + 1, len(candidates)):
                        if candidates[c1][0] < candidates[c2][0]:
                            tmp = candidates[c1]
                            candidates[c1] = candidates[c2]
                            candidates[c2] = tmp

                # 3. Try to enter up to max_concurrent
                for cand in candidates:
                    if num_open_pos >= max_concurrent:
                        break

                    _rs_val, s, p_side, fill_p, sf = cand
                    pos_atr = atr_2d[prev_i, s]
                    stop_dist = (pos_atr * l_atr_mult) if p_side == 1 else (pos_atr * s_atr_mult)

                    if stop_dist > 0:
                        target_qty = calculate_position_size(
                            fill_p,
                            stop_dist,
                            current_equity,
                            free_margin,
                            effective_risk_per_trade,
                            leverage,
                            sf,
                            garch_kelly_f[prev_i, s],
                            max_exposure_per_coin=max_exp_per_coin,
                        )

                        max_margin_per_coin = current_equity / float(max_concurrent)
                        max_qty_by_cap = (max_margin_per_coin * leverage) / fill_p
                        if target_qty > max_qty_by_cap:
                            target_qty = max_qty_by_cap

                        req_margin = (target_qty * fill_p) / leverage
                        entry_fee = target_qty * fill_p * fee_rate

                        if target_qty > 0 and free_margin >= (req_margin + entry_fee):
                            balance -= req_margin + entry_fee
                            free_margin -= req_margin + entry_fee

                            in_pos[s] = True
                            pos_side[s] = p_side
                            entry_p[s] = fill_p
                            entry_idx[s] = i
                            amount[s] = target_qty
                            entry_fee_stored[s] = entry_fee
                            fund_fee_stored[s] = 0.0
                            highest[s] = fill_p
                            lowest[s] = fill_p
                            has_scaled[s] = False
                            stop_p[s] = fill_p - stop_dist if p_side == 1 else fill_p + stop_dist

                            triggered, intra_exit_price, pnl_intra, exit_fee_intra = (
                                check_intra_bar_stop(
                                    p_side,
                                    high_2d[i, s],
                                    low_2d[i, s],
                                    stop_p[s],
                                    fill_p,
                                    target_qty,
                                    fee_rate,
                                    slippage_rate,
                                )
                            )
                            if triggered:
                                pnl_intra -= exit_fee_intra
                                balance += (target_qty * fill_p) / leverage + pnl_intra
                                free_margin += (target_qty * fill_p) / leverage + pnl_intra
                                trades[t_count] = [
                                    s,
                                    i,
                                    i,
                                    float(p_side),
                                    fill_p,
                                    intra_exit_price,
                                    pnl_intra - fund_fee_stored[s],
                                    target_qty,
                                    entry_fee,
                                    fund_fee_stored[s],
                                ]
                                t_count += 1
                                in_pos[s] = False
                            else:
                                num_open_pos += 1

    if n_bars > 0:
        last_idx = n_bars - 1
        for s in range(n_syms):
            if in_pos[s]:
                c_last = close_2d[last_idx, s]
                if pos_side[s] == 1:
                    exit_price = c_last * (1.0 - slippage_rate)
                    pnl = (exit_price - entry_p[s]) * amount[s]
                else:
                    exit_price = c_last * (1.0 + slippage_rate)
                    pnl = (entry_p[s] - exit_price) * amount[s]

                fee = amount[s] * exit_price * fee_rate
                balance += ((amount[s] * entry_p[s]) / leverage) + (pnl - fee)

                trades[t_count] = [
                    s,
                    entry_idx[s],
                    last_idx,
                    float(pos_side[s]),
                    entry_p[s],
                    exit_price,
                    pnl - fee - fund_fee_stored[s],
                    amount[s],
                    entry_fee_stored[s],
                    fund_fee_stored[s],
                ]
                t_count += 1
                in_pos[s] = False

    return trades[:t_count], balance, equity_curve
