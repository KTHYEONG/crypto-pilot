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
    ) -> tuple[pd.DataFrame, np.ndarray, float, np.ndarray]:
        close_2d = self.data["close"]
        high_2d = self.data["high"]
        low_2d = self.data["low"]
        open_2d = self.data["open"]

        entry_upper = self.data["entry_upper"]
        entry_lower = self.data["entry_lower"]
        trend_dir = self.data["trend_direction"]
        # In optimized design, strength_filter is the raw probability
        strength_filter_raw = self.data.get(
            "ml_calib_prob", self.data.get("strength_filter", np.zeros_like(close_2d))
        )
        atr_2d = self.data["atr"]
        garch_kelly_f = self.data["garch_kelly_f"]
        kill_signal = self.data.get("kill_signal", np.zeros_like(close_2d))
        funding_rate = self.data.get("funding_rate_sum", np.zeros_like(close_2d))
        slot_rank_score = self.data["slot_rank_score"]
        xs_long = self.data.get("xs_score_long")
        xs_short = self.data.get("xs_score_short")
        hmm_crisis = self.data.get("hmm_prob_crisis")
        hmm_mod_l = self.data.get("hmm_modulator_long")
        hmm_mod_s = self.data.get("hmm_modulator_short")
        if xs_long is None or xs_long.shape != close_2d.shape:
            xs_long = np.zeros_like(close_2d, dtype=np.float64)
        if xs_short is None or xs_short.shape != close_2d.shape:
            xs_short = np.zeros_like(close_2d, dtype=np.float64)
        if hmm_crisis is None or hmm_crisis.shape != close_2d.shape:
            hmm_crisis = np.zeros_like(close_2d, dtype=np.float64)
        if hmm_mod_l is None or hmm_mod_l.shape != close_2d.shape:
            hmm_mod_l = np.ones_like(close_2d, dtype=np.float64)
        if hmm_mod_s is None or hmm_mod_s.shape != close_2d.shape:
            hmm_mod_s = np.ones_like(close_2d, dtype=np.float64)

        l_atr_mult = float(self.params.get("LONG_ATR_MULT", 3.0))
        l_trail_mult = float(self.params.get("LONG_TRAIL_MULT", 3.0))
        s_atr_mult = float(self.params.get("SHORT_ATR_MULT", 2.0))
        s_tp_mult = float(self.params.get("SHORT_TP_MULT", 3.0))
        l_scale_atr = float(self.params.get("LONG_SCALE_ATR_MULT", 3.0))
        s_trail_mult = float(self.params.get("SHORT_TRAIL_MULT", 3.0))

        max_exp_per_coin = float(self.params.get("MAX_EXPOSURE_PER_COIN", 1.5))
        dd_scaling_threshold = float(self.params.get("DD_SCALING_THRESHOLD", 0.15))
        k_long = int(self.params.get("K_LONG", 2))
        k_short = int(self.params.get("K_SHORT", 2))
        rebalance_bars = max(1, int(self.params.get("REBALANCE_BARS", 6)))
        crisis_gamma = float(
            self.params.get("CRISIS_GAMMA", self.params.get("CRISIS_GATE_PROB", 1.0))
        )
        use_cs_rank = 1 if bool(self.params.get("USE_CS_RANK_ENGINE", True)) else 0

        lev_2d = self.data.get("dyn_leverage")
        if lev_2d is None or lev_2d.shape != close_2d.shape:
            lev_2d = np.full(close_2d.shape, self.leverage, dtype=np.float64)
        else:
            lev_2d = np.ascontiguousarray(np.maximum(lev_2d, 1.0), dtype=np.float64)

        trades_arr, final_balance, equity_curve, bt_diag = backtest_portfolio_numba(
            close_2d,
            high_2d,
            low_2d,
            open_2d,
            entry_upper,
            entry_lower,
            trend_dir,
            strength_filter_raw,
            atr_2d,
            garch_kelly_f,
            kill_signal,
            funding_rate,
            slot_rank_score,
            np.ascontiguousarray(xs_long, dtype=np.float64),
            np.ascontiguousarray(xs_short, dtype=np.float64),
            np.ascontiguousarray(hmm_crisis, dtype=np.float64),
            np.ascontiguousarray(hmm_mod_l, dtype=np.float64),
            np.ascontiguousarray(hmm_mod_s, dtype=np.float64),
            self.initial_balance,
            lev_2d,
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
            k_long,
            k_short,
            rebalance_bars,
            crisis_gamma,
            use_cs_rank,
        )

        _logger.debug(
            "[BT_DIAG] dust_skip=%d margin_fail=%d t_dir_zero=%d p_side_zero=%d",
            int(bt_diag[0]),
            int(bt_diag[1]),
            int(bt_diag[2]),
            int(bt_diag[3]),
        )

        if trades_arr.size == 0:
            return pd.DataFrame(), equity_curve, final_balance, bt_diag

        df_trades = pd.DataFrame(
            trades_arr,
            columns=[
                "sym_idx", "entry_idx", "exit_idx", "side_val", "entry_price",
                "exit_price", "pnl", "amount", "entry_fee", "funding_fee"
            ]
        )
        df_trades["symbol"] = [self.symbols[int(i)] for i in df_trades["sym_idx"]]
        df_trades["side"] = np.where(df_trades["side_val"] == 1.0, "LONG", "SHORT")

        final_cols = [
            "symbol", "entry_idx", "exit_idx", "side", "entry_price",
            "exit_price", "pnl", "amount", "entry_fee", "funding_fee"
        ]
        return df_trades[final_cols], equity_curve, final_balance, bt_diag


@njit(nogil=True, cache=True)  # type: ignore[untyped-decorator]
def _recompute_cs_dirs_numba(
    prev_i: int,
    n_syms: int,
    xs_long: np.ndarray,
    xs_short: np.ndarray,
    hmm_crisis: np.ndarray,
    crisis_gamma: float,
    k_long: int,
    k_short: int,
    computed_dir: np.ndarray,
) -> None:
    """Hard top-K CS dirs: binary top-K, crisis (1-p)^gamma, |mag| in [0.05,1]."""
    computed_dir[:] = 0.0
    gam = crisis_gamma if crisis_gamma > 1e-9 else 1e-9
    fk = float(k_long)
    fks = float(k_short)

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
    if count_l > 0: mean_l /= count_l
    if count_s > 0: mean_s /= count_s

    std_l, std_s = 0.0, 0.0
    for s in range(n_syms):
        vl = float(xs_long[prev_i, s])
        if np.isfinite(vl): std_l += (vl - mean_l)**2
        vs = float(xs_short[prev_i, s])
        if np.isfinite(vs): std_s += (vs - mean_s)**2
    if count_l > 1: std_l = np.sqrt(std_l / (count_l - 1))
    if count_s > 1: std_s = np.sqrt(std_s / (count_s - 1))
    std_l = std_l if std_l > 1e-9 else 1e-9
    std_s = std_s if std_s > 1e-9 else 1e-9

    for s in range(n_syms):
        sl = float(xs_long[prev_i, s])
        ss = float(xs_short[prev_i, s])
        if not np.isfinite(sl) or not np.isfinite(ss):
            continue
        c_raw = float(hmm_crisis[prev_i, s])
        if not np.isfinite(c_raw):
            c_raw = 0.0
        if c_raw < 0.0:
            c_raw = 0.0
        if c_raw > 1.0:
            c_raw = 1.0
        crisis_w = (1.0 - c_raw) ** gam

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

        binary_l = 1.0 if float(rank_l) <= fk else 0.0
        binary_s = 1.0 if float(rank_s) <= fks else 0.0

        # Z-score magnitudes: clip between 0 and 3, then scale to max 1.0
        z_l = (sl - mean_l) / std_l
        z_s = (mean_s - ss) / std_s  # Lower ss is better, so mean - ss

        mag_l = z_l / 3.0 if z_l > 0 else 0.0
        mag_s = z_s / 3.0 if z_s > 0 else 0.0

        long_mag = binary_l * min(max(mag_l, 0.05), 1.0) * crisis_w
        short_mag = binary_s * min(max(mag_s, 0.05), 1.0) * crisis_w

        if long_mag >= short_mag and long_mag > 0.0:
            computed_dir[s] = 1.0 * long_mag
        elif short_mag > long_mag and short_mag > 0.0:
            computed_dir[s] = -1.0 * short_mag
        else:
            computed_dir[s] = 0.0


@njit(nogil=True, cache=True)  # type: ignore[untyped-decorator]
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
    hmm_mod_long: np.ndarray,
    hmm_mod_short: np.ndarray,
    initial_balance: float,
    lev_2d: np.ndarray,
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
    k_long: int,
    k_short: int,
    rebalance_bars: int,
    crisis_gamma: float,
    use_cs_rank: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    n_bars = close_2d.shape[0]
    n_syms = close_2d.shape[1]

    computed_dir = np.zeros(n_syms, dtype=np.float64)
    prev_rebalance_bucket = -999999
    dust_skip_cnt = 0
    margin_fail_cnt = 0
    t_dir_zero_cnt = 0
    p_side_zero_cnt = 0
    min_notional = max(5.0, 0.0005 * initial_balance) if initial_balance > 0.0 else 5.0

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

    just_exited = np.zeros(n_syms, dtype=np.bool_)
    candidate_pool = np.zeros((n_syms, 6), dtype=np.float64)
    entry_lev = np.ones(n_syms, dtype=np.float64)

    max_trades = 50000
    trades: np.ndarray = np.zeros((max_trades, 10), dtype=np.float64)
    t_count = 0

    for i in range(1, n_bars):
        unrealized_total = 0.0
        just_exited[:] = False
        used_margin_total = 0.0
        num_open_pos = 0

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
        risk_scale = max(0.0, 1.0 - (dd / dd_scaling_threshold)) if dd_scaling_threshold > 1e-9 else 1.0
        effective_risk_per_trade = risk_per_trade * risk_scale

        for s in range(n_syms):
            if not in_pos[s]: continue
            c_open, c_high, c_low = open_2d[i, s], high_2d[i, s], low_2d[i, s]
            pos_atr = atr_2d[entry_idx[s], s]
            exit_triggered, exit_price = False, 0.0

            if kill_signal[i-1, s] > 0.5:
                exit_triggered, exit_price = True, c_open * (1.0 - slippage_rate * pos_side[s])
            
            if not exit_triggered:
                if pos_side[s] == 1:
                    if c_high > highest[s]: highest[s] = c_high
                    
                    # [REFACTORED] Pessimistic Execution: Check Exit (SL) FIRST before Scale-out (TP)
                    exit_triggered, exit_price, stop_p[s] = check_long_exit(c_open, c_low, highest[s], pos_atr, stop_p[s], l_trail_mult, slippage_rate)
                    
                    if not exit_triggered and not has_scaled[s]:
                        tr, sc_p, sc_a, pnl_s, fee_s = process_long_scale_out(c_open, c_high, entry_p[s], pos_atr, l_scale_atr, amount[s], fee_rate)
                        if tr:
                            sc_f = fund_fee_stored[s] / 2.0
                            balance += (sc_a * entry_p[s]) / entry_lev[s] + (pnl_s - fee_s)
                            if t_count < max_trades:
                                trades[t_count] = [float(s), float(entry_idx[s]), float(i), 1.0, entry_p[s], sc_p, pnl_s - fee_s - sc_f, sc_a, entry_fee_stored[s]/2.0, sc_f]
                                t_count += 1
                            amount[s] -= sc_a
                            entry_fee_stored[s] /= 2.0; fund_fee_stored[s] /= 2.0; has_scaled[s] = True
                else:
                    if c_low < lowest[s]: lowest[s] = c_low
                    
                    # [REFACTORED] Pessimistic Execution: Check Exit (SL) FIRST before Scale-out (TP)
                    exit_triggered, exit_price, stop_p[s] = check_short_exit(c_open, c_high, lowest[s], pos_atr, stop_p[s], s_trail_mult, slippage_rate)
                    
                    if not exit_triggered and not has_scaled[s]:
                        tr, sc_p, sc_a, pnl_s, fee_s = process_short_scale_out(c_open, c_low, entry_p[s], pos_atr, s_tp_mult, amount[s], fee_rate)
                        if tr:
                            sc_f = fund_fee_stored[s] / 2.0
                            balance += (sc_a * entry_p[s]) / entry_lev[s] + (pnl_s - fee_s)
                            if t_count < max_trades:
                                trades[t_count] = [float(s), float(entry_idx[s]), float(i), -1.0, entry_p[s], sc_p, pnl_s - fee_s - sc_f, sc_a, entry_fee_stored[s]/2.0, sc_f]
                                t_count += 1
                            amount[s] -= sc_a
                            entry_fee_stored[s] /= 2.0; fund_fee_stored[s] /= 2.0; has_scaled[s] = True; stop_p[s] = entry_p[s] - (entry_p[s]*fee_rate*2.0)

            if exit_triggered:
                pnl = (exit_price - entry_p[s]) * amount[s] * pos_side[s]
                fee = amount[s] * exit_price * fee_rate
                balance += ((amount[s] * entry_p[s]) / entry_lev[s]) + (pnl - fee)
                if t_count < max_trades:
                    trades[t_count] = [float(s), float(entry_idx[s]), float(i), float(pos_side[s]), entry_p[s], exit_price, pnl - fee - fund_fee_stored[s], amount[s], entry_fee_stored[s], fund_fee_stored[s]]
                    t_count += 1
                in_pos[s], just_exited[s] = False, True
                num_open_pos -= 1

        free_margin = current_equity - used_margin_total
        if num_open_pos < max_concurrent:
            prev_i, n_cands = i - 1, 0
            if use_cs_rank != 0 and prev_i >= 0:
                bucket = prev_i // rebalance_bars if rebalance_bars > 0 else prev_i
                if bucket != prev_rebalance_bucket:
                    prev_rebalance_bucket = bucket
                    _recompute_cs_dirs_numba(
                        prev_i,
                        n_syms,
                        xs_long,
                        xs_short,
                        hmm_crisis,
                        crisis_gamma,
                        k_long,
                        k_short,
                        computed_dir,
                    )

            for s in range(n_syms):
                if in_pos[s] or just_exited[s] or np.isnan(open_2d[i, s]): continue

                sf = 1.0
                if use_cs_rank != 0:
                    t_dir = computed_dir[s]
                    dir_abs = abs(float(t_dir))
                    sf = dir_abs if dir_abs < 1.0 else 1.0
                else:
                    sf_raw = strength_filter_raw[prev_i, s]
                    if sf_raw < 0.5:
                        continue
                    t_dir = trend_dir[prev_i, s]
                    sf = 1.0
                if t_dir == 0.0:
                    if use_cs_rank != 0:
                        t_dir_zero_cnt += 1
                    continue

                c_open, p_side, fill_p = open_2d[i, s], 0, 0.0
                if t_dir > 0.0:
                    if entry_upper[prev_i, s] <= 1e-9 or high_2d[i, s] > entry_upper[prev_i, s]:
                        base_p = c_open if entry_upper[prev_i, s] <= 1e-9 else max(c_open, entry_upper[prev_i, s])
                        fill_p = base_p * (1.0 + slippage_rate)
                        p_side = 1
                elif t_dir < 0.0:
                    if entry_lower[prev_i, s] <= 1e-9 or entry_lower[prev_i, s] >= 999998.0 or low_2d[i, s] < entry_lower[prev_i, s]:
                        base_p = c_open if (entry_lower[prev_i, s] <= 1e-9 or entry_lower[prev_i, s] >= 999998.0) else min(c_open, entry_lower[prev_i, s])
                        fill_p = base_p * (1.0 - slippage_rate)
                        p_side = -1

                if p_side == 0:
                    if use_cs_rank != 0:
                        p_side_zero_cnt += 1
                    continue

                atr_p = atr_2d[prev_i, s] / max(close_2d[prev_i, s], 1e-12)
                lev_entry = float(lev_2d[i, s])
                if lev_entry < 1.0:
                    lev_entry = 1.0
                gk0 = garch_kelly_f[prev_i, s]
                if p_side == 1:
                    gk_use = gk0 * hmm_mod_long[prev_i, s]
                else:
                    gk_use = gk0 * hmm_mod_short[prev_i, s]
                if not np.isfinite(gk_use) or gk_use < 0.0:
                    gk_use = 0.0
                target_qty = calculate_position_size(
                    fill_p,
                    atr_p,
                    current_equity,
                    free_margin,
                    effective_risk_per_trade,
                    lev_entry,
                    sf,
                    gk_use,
                    max_exp_per_coin,
                )
                if target_qty > 0:
                    sort_key = abs(float(t_dir)) if use_cs_rank != 0 else abs(
                        float(slot_rank_score[prev_i, s])
                    )
                    candidate_pool[n_cands] = [
                        sort_key,
                        float(s),
                        float(p_side),
                        fill_p,
                        target_qty,
                        atr_2d[prev_i, s] * (l_atr_mult if p_side == 1 else s_atr_mult),
                    ]
                    n_cands += 1

            if n_cands > 0:
                for c1 in range(n_cands):
                    for c2 in range(c1 + 1, n_cands):
                        if candidate_pool[c1, 0] < candidate_pool[c2, 0]:
                            for k in range(6):
                                tmp = candidate_pool[c1, k]; candidate_pool[c1, k] = candidate_pool[c2, k]; candidate_pool[c2, k] = tmp
                n_to_select = min(n_cands, max_concurrent - num_open_pos)
                total_req = 0.0
                for idx in range(n_to_select):
                    s_i = int(candidate_pool[idx, 1])
                    le_i = float(lev_2d[i, s_i])
                    if le_i < 1.0:
                        le_i = 1.0
                    total_req += (candidate_pool[idx, 4] * candidate_pool[idx, 3]) / le_i
                scale = (free_margin * 0.96) / total_req if total_req > (free_margin * 0.96) and total_req > 0 else 1.0
                for idx in range(n_to_select):
                    _, s_f, p_side_f, fill_p, target_qty, stop_dist = candidate_pool[idx]
                    s, p_side, final_qty = int(s_f), int(p_side_f), target_qty * scale
                    if final_qty * fill_p < min_notional:
                        dust_skip_cnt += 1
                        continue
                    le_ent = float(lev_2d[i, s])
                    if le_ent < 1.0:
                        le_ent = 1.0
                    entry_lev[s] = le_ent
                    req_m = (final_qty * fill_p) / le_ent
                    e_fee = final_qty * fill_p * fee_rate
                    if free_margin >= (req_m + e_fee):
                        balance -= (req_m + e_fee)
                        in_pos[s], pos_side[s], entry_p[s], entry_idx[s] = True, p_side, fill_p, i
                        amount[s], entry_fee_stored[s], fund_fee_stored[s], highest[s], lowest[s], has_scaled[s], stop_p[s] = (
                            final_qty,
                            e_fee,
                            0.0,
                            fill_p,
                            fill_p,
                            False,
                            fill_p - (stop_dist * p_side),
                        )
                    else:
                        margin_fail_cnt += 1
    
    # Force close all positions at end
    if n_bars > 0:
        last_idx = n_bars - 1
        for s in range(n_syms):
            if in_pos[s]:
                cur_p = close_2d[last_idx, s]
                if np.isnan(cur_p): cur_p = entry_p[s]
                pnl = (cur_p - entry_p[s]) * amount[s] * pos_side[s]
                fee = amount[s] * cur_p * fee_rate
                balance += ((amount[s] * entry_p[s]) / entry_lev[s]) + (pnl - fee)
                if t_count < max_trades:
                    trades[t_count] = [float(s), float(entry_idx[s]), float(last_idx), float(pos_side[s]), entry_p[s], cur_p, pnl - fee - fund_fee_stored[s], amount[s], entry_fee_stored[s], fund_fee_stored[s]]
                    t_count += 1
                in_pos[s] = False

    diag_out = np.empty(4, dtype=np.int64)
    diag_out[0] = dust_skip_cnt
    diag_out[1] = margin_fail_cnt
    diag_out[2] = t_dir_zero_cnt
    diag_out[3] = p_side_zero_cnt
    return trades[:t_count], balance, equity_curve, diag_out
