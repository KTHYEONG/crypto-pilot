"""
Futures Optuna objective: CPCV paths, Kelly-CVaR scalar, disk+memory signal cache.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import (
    FUTURES_INITIAL_BALANCE,
)
from src.domain.futures.engine_multi_futures import PortfolioBacktestEngineFast
from src.domain.futures.opt_futures_utils.cv_utils import (
    build_cpcv_test_paths_with_fallback,
)
from src.domain.futures.opt_futures_utils.metrics import (
    _log_tw_from_ret_pct,
    calc_mdd_from_equity,
    calc_profit_factor_from_pnl,
)
from src.domain.futures.opt_futures_utils.opt_params import suggest_params_futures
from src.domain.futures.strategies_futures import UltimateStrategy

from .data_utils import (
    _build_aligned_2d_from_prebuilt,
    _dataframe_to_symbol_arrays,
    _segment_with_context,
)
from .oos_evaluator import evaluate_symbol_fold
from .signal_cache import (
    _build_signal_cache_key,
    _dataset_fingerprint_from_df,
    get_or_compute_signals,
)

_logger: logging.Logger = logging.getLogger("opt_futures")


def compute_embargo_bars(tf: str, longest_indicator_period: int = 150) -> int:
    fixed_min: Dict[str, int] = {"1h": 24, "4h": 12}
    ratio_map: Dict[str, float] = {"1h": 0.08, "4h": 0.05}
    ratio: float = ratio_map.get(tf, 0.03)
    return max(fixed_min.get(tf, 2), int(longest_indicator_period * ratio))


EMBARGO_BARS: Dict[str, int] = {
    "1h": compute_embargo_bars("1h"),
    "4h": compute_embargo_bars("4h"),
}


def objective_futures(
    trial: optuna.Trial,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf_target: str,
    *,
    space: Dict[str, Dict[str, Any]],
    mode: str = "single",
    project_root: Optional[str] = None,
    prebuilt_cpcv_bundle: Optional[Tuple[List[List[Tuple[int, int]]], int, int]] = None,
    signal_disk_cache_root: Optional[Path] = None,
    relaxed_constraints: bool = False,
) -> float:
    params: Dict[str, Any] = suggest_params_futures(trial, space, tf_target)
    tf: str = tf_target
    strategy: UltimateStrategy = UltimateStrategy(name="OptFutures", params=params)

    ref_sym = symbols[0]
    ref_df: Optional[pd.DataFrame] = data_maps.get(ref_sym, {}).get(tf)
    if ref_df is None or ref_df.empty:
        trial.set_user_attr("prune_reason", "no_ref_data")
        raise optuna.TrialPruned()

    is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
    ref_len = len(ref_df) - is_off
    if ref_len < 200:
        trial.set_user_attr("prune_reason", f"ref_len_too_short:{ref_len}")
        raise optuna.TrialPruned()

    embargo = int(EMBARGO_BARS.get(tf, 0))
    if prebuilt_cpcv_bundle is not None:
        cpcv_paths, _, _ = prebuilt_cpcv_bundle
    else:
        cpcv_paths, _, _ = build_cpcv_test_paths_with_fallback(ref_len, embargo=embargo)
    if not cpcv_paths:
        trial.set_user_attr("prune_reason", "no_cpcv_paths")
        raise optuna.TrialPruned()

    cfg: Dict[str, Any] = OPT_FUTURES_CONFIG
    min_seg_trades = int(cfg.get("FUTURES_MIN_TRADES_PER_CPCV_SEGMENT", 5))

    # --- Hard Risk Thresholds ---
    cvar_alpha = float(cfg.get("FUTURES_CPCV_CVAR_ALPHA", 0.10))
    cvar_thr_log = float(cfg.get("FUTURES_CPCV_CVAR_THRESHOLD_LOG", -0.05))  # Max log-loss allowed

    cache_root = signal_disk_cache_root
    if cache_root is None and project_root is not None:
        cache_root = Path(project_root) / "cache_futures"

    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        target_df_full = data_maps.get(sym, {}).get(tf)
        if target_df_full is None or target_df_full.empty:
            continue
        fp = _dataset_fingerprint_from_df(target_df_full)
        cache_key = _build_signal_cache_key(params, sym, tf, len(target_df_full), fp)
        full_signal_dfs[sym] = get_or_compute_signals(
            cache_key, target_df_full, strategy, disk_cache_root=cache_root
        )

    if len(full_signal_dfs) != len(symbols):
        if not relaxed_constraints:
            raise optuna.TrialPruned()
        missing = [s for s in symbols if s not in full_signal_dfs]
        _logger.debug("relaxed: signal missing for %s, continuing with subset", missing)
        symbols = [s for s in symbols if s in full_signal_dfs]
        if not symbols:
            trial.set_user_attr("prune_reason", "signal_fail:all_symbols")
            raise optuna.TrialPruned()
        trial.set_user_attr("prune_reason", f"signal_partial:{len(missing)}_missing")

    prebuilt_full_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    if mode == "multi":
        # [DATETIME ALIGNMENT] Align all symbols to the INTERSECTION of IS periods.
        # Without this, symbols listed later (e.g. GALA, ID, ROSE) have shorter arrays
        # than the ref symbol (e.g. ETC), causing slice_end > arr.shape[0] in
        # _build_aligned_2d_from_prebuilt → every segment returns None → no valid paths.
        is_start_dts_per_sym: Dict[str, Any] = {}
        for sym in symbols:
            sym_df = full_signal_dfs[sym]
            sym_is_off = int(data_maps[sym].get(f"is_start_idx_{tf}", 0))
            if len(sym_df) > sym_is_off and "datetime" in sym_df.columns:
                is_start_dts_per_sym[sym] = sym_df["datetime"].iloc[sym_is_off]

        if is_start_dts_per_sym:
            # Latest IS-start across all symbols = common intersection start
            common_is_start_dt = max(is_start_dts_per_sym.values())
            trimmed_dfs: Dict[str, pd.DataFrame] = {}
            for sym in symbols:
                sym_df = full_signal_dfs[sym]
                if "datetime" in sym_df.columns:
                    m = sym_df["datetime"] >= common_is_start_dt
                    if m.any():
                        trimmed_dfs[sym] = sym_df[m].reset_index(drop=True)

            if not trimmed_dfs:
                trial.set_user_attr("prune_reason", "no_common_is_period")
                raise optuna.TrialPruned()

            if len(trimmed_dfs) < len(symbols):
                symbols = [s for s in symbols if s in trimmed_dfs]

            for sym in symbols:
                prebuilt_full_arrays[sym] = _dataframe_to_symbol_arrays(trimmed_dfs[sym])

            # Recompute IS length and CPCV paths from the aligned (intersection) data
            eff_ref_len = min(v["close"].shape[0] for v in prebuilt_full_arrays.values())
            if eff_ref_len < 200:
                trial.set_user_attr("prune_reason", f"aligned_is_too_short:{eff_ref_len}")
                raise optuna.TrialPruned()

            cpcv_paths, _, _ = build_cpcv_test_paths_with_fallback(eff_ref_len, embargo=embargo)
            if not cpcv_paths:
                trial.set_user_attr("prune_reason", "no_cpcv_paths_aligned")
                raise optuna.TrialPruned()

            # Data already trimmed to IS start → offset = 0
            is_off = 0
        else:
            # Fallback when no datetime column: bar-index alignment
            for sym in symbols:
                prebuilt_full_arrays[sym] = _dataframe_to_symbol_arrays(full_signal_dfs[sym])

    liq_mdd_thr = float(cfg.get("FUTURES_MAX_MDD", 25.0))
    if relaxed_constraints:
        liq_mdd_thr = 100.0  # Bypass MDD constraint for robust mapping
        min_seg_trades = 1   # Bypass trade count constraint

    path_compound_raw_log_tw: List[float] = []
    path_pfs: List[float] = []
    path_wins: List[float] = []
    path_trades: List[float] = []
    path_mdds: List[float] = []
    all_seg_long: int = 0
    all_seg_short: int = 0
    all_long_pnls: List[float] = []
    all_short_pnls: List[float] = []
    funding_ratios: List[float] = []
    sym_pf_accum: Dict[str, List[float]] = {s: [] for s in symbols}
    path_terminal_wealth_ratios: List[float] = []

    for path_idx, path in enumerate(cpcv_paths):
        seg_raw_logs: List[float] = []
        seg_mdds: List[float] = []
        seg_pfs: List[float] = []
        seg_wins: List[float] = []
        seg_trades: List[float] = []
        seg_fund_ratio: List[float] = []

        running_balance = float(FUTURES_INITIAL_BALANCE)

        for test_start, test_end in path:
            if mode == "multi":
                abs_start = is_off + int(test_start)
                abs_end = is_off + int(test_end)
                slice_start = max(0, abs_start - 1)
                slice_end = min(len(ref_df), abs_end)
                if slice_end - slice_start < 2:
                    continue
                aligned_data = _build_aligned_2d_from_prebuilt(
                    prebuilt_full_arrays,
                    symbols,
                    slice_start,
                    slice_end,
                )
                if not aligned_data:
                    if relaxed_constraints:
                        trial.set_user_attr("prune_reason", "aligned_data_fail")
                        continue  # Skip this segment; don't abort entire trial
                    raise optuna.TrialPruned()

                segment_initial = max(running_balance, 1e-9)
                try:
                    from config.settings import SLIPPAGE_RATE, TRADING_FEE_RATE

                    engine = PortfolioBacktestEngineFast(
                        aligned_data=aligned_data,
                        symbol_names=symbols,
                        strategy_params=params,
                        initial_balance=segment_initial,
                        fee_rate=TRADING_FEE_RATE,
                        slippage_rate=SLIPPAGE_RATE,
                    )
                    trades_df, equity_curve, final_balance = engine.run()
                except Exception as exc:
                    _logger.warning("Portfolio engine CPCV error: %s", exc, exc_info=True)
                    if relaxed_constraints:
                        return -5.0
                    raise optuna.TrialPruned() from exc

                if trades_df is None or trades_df.empty:
                    if relaxed_constraints:
                        return -5.0
                    raise optuna.TrialPruned()

                # 심볼별 최소 트레이드 수 체크
                if "symbol" in trades_df.columns and len(symbols) > 1:
                    for _sym in symbols:
                        if int((trades_df["symbol"] == _sym).sum()) == 0:
                            if relaxed_constraints:
                                pass # Allow 0 trades on some symbols in relaxed mode
                            else:
                                raise optuna.TrialPruned()

                ret_pct = float((final_balance / segment_initial - 1.0) * 100.0)
                raw_log = _log_tw_from_ret_pct(ret_pct)
                running_balance = max(float(final_balance), 1e-9)
                mdd_seg = (
                    float(calc_mdd_from_equity(equity_curve)) if equity_curve.size > 0 else 0.0
                )

                if mdd_seg >= liq_mdd_thr:
                    if relaxed_constraints:
                        pass
                    else:
                        raise optuna.TrialPruned()

                true_pnl = trades_df["pnl"] - trades_df["entry_fee"]
                pf_s = calc_profit_factor_from_pnl(true_pnl)
                long_pnl_s = true_pnl[trades_df["side"] == "LONG"]
                short_pnl_s = true_pnl[trades_df["side"] == "SHORT"]
                all_long_pnls.extend(long_pnl_s.tolist())
                all_short_pnls.extend(short_pnl_s.tolist())

                win_rate_s = (
                    float((len(trades_df[true_pnl > 0]) / len(trades_df)) * 100)
                    if len(trades_df) > 0
                    else 0.0
                )
                n_tr = len(trades_df)
                lc = len(trades_df[trades_df["side"] == "LONG"])
                sc = len(trades_df[trades_df["side"] == "SHORT"])
                all_seg_long += lc
                all_seg_short += sc
                seg_raw_logs.append(raw_log)
                seg_mdds.append(mdd_seg)
                seg_pfs.append(pf_s)
                seg_wins.append(win_rate_s)
                seg_trades.append(float(n_tr))

                sum_ff = (
                    float(trades_df["funding_fee"].sum())
                    if "funding_fee" in trades_df.columns
                    else 0.0
                )
                sum_pnl_abs = float(trades_df["pnl"].abs().sum())
                funding_ratios.append(sum_ff / max(sum_pnl_abs, 1e-9))

                if "symbol" in trades_df.columns:
                    for _sx in symbols:
                        _sx_pnl = float(trades_df.loc[trades_df["symbol"] == _sx, "pnl"].sum())
                        sym_pf_accum[_sx].append(_sx_pnl)

                if n_tr < min_seg_trades:
                    if relaxed_constraints:
                        pass
                    else:
                        raise optuna.TrialPruned()

                continue

            # --- Single Mode (Legacy/Fallback) ---
            sym_logs, sym_mdds, sym_pfs, sym_wins, sym_trades, sym_fund_r = [], [], [], [], [], []
            for sym in symbols:
                target_df = data_maps[sym][tf]
                daily_df = data_maps[sym]["1d"]
                merge_idx = data_maps[sym][f"merge_idx_{tf}"]
                is_start_idx = int(data_maps[sym].get(f"is_start_idx_{tf}", 0))
                adj_s, adj_e = test_start + is_start_idx, test_end + is_start_idx
                seg, ex_idx = _segment_with_context(full_signal_dfs[sym], adj_s, adj_e)
                _cagr, ret_pct, mdd_pct, ntr, wr, pf, lc, sc, _eq, fpaid, gross = (
                    evaluate_symbol_fold(
                        strategy,
                        params,
                        sym,
                        tf,
                        target_df,
                        daily_df,
                        merge_idx,
                        None,
                        adj_s,
                        adj_e,
                        seg,
                        ex_idx,
                    )
                )
                if mdd_pct >= liq_mdd_thr:
                    if relaxed_constraints:
                        pass
                    else:
                        raise optuna.TrialPruned()
                denom = max(abs(gross), 1e-9)
                sym_fund_r.append(float(fpaid) / denom)
                sym_logs.append(_log_tw_from_ret_pct(ret_pct))
                sym_mdds.append(mdd_pct)
                sym_pfs.append(pf)
                sym_wins.append(wr)
                sym_trades.append(float(ntr))
                all_seg_long, all_seg_short = all_seg_long + lc, all_seg_short + sc
                sym_pf_accum[sym].append(pf)

            if np.sum(sym_trades) < min_seg_trades:
                if relaxed_constraints:
                    pass
                else:
                    raise optuna.TrialPruned()
            seg_raw_logs.append(float(np.mean(sym_logs)) if sym_logs else -10.0)
            seg_mdds.append(float(np.mean(sym_mdds)) if sym_mdds else 0.0)
            seg_pfs.append(float(np.mean(sym_pfs)) if sym_pfs else 1.0)
            seg_wins.append(float(np.mean(sym_wins)) if sym_wins else 0.0)
            seg_trades.append(np.sum(sym_trades) / max(len(sym_logs), 1))
            seg_fund_ratio.append(float(np.mean(sym_fund_r)) if sym_fund_r else 0.0)

        if mode == "multi":
            path_terminal_wealth_ratios.append(
                float(running_balance / float(FUTURES_INITIAL_BALANCE))
            )

        if not seg_raw_logs:
            if relaxed_constraints:
                continue  # Skip path with no valid segments; don't abort entire trial
            raise optuna.TrialPruned()
        path_compound_raw_log_tw.append(float(np.sum(seg_raw_logs)))
        path_pfs.append(float(np.mean(seg_pfs)))
        path_wins.append(float(np.mean(seg_wins)))
        path_trades.append(float(np.mean(seg_trades)))
        path_mdds.append(float(np.max(seg_mdds)) if seg_mdds else 0.0)
        funding_ratios.extend(seg_fund_ratio)

        if path_idx >= 4 and path_compound_raw_log_tw:
            trial.report(float(np.mean(path_compound_raw_log_tw)), step=path_idx)
            if trial.should_prune() and not relaxed_constraints:
                raise optuna.TrialPruned()

    # --- Hybrid Constrained Optimization: P10 GMGR as Core Objective ---
    if not path_compound_raw_log_tw:
        trial.set_user_attr("prune_reason", "no_valid_paths")
        raise optuna.TrialPruned()
    path_arr = np.asarray(path_compound_raw_log_tw, dtype=np.float64)
    p10_log_tw = (
        float(np.percentile(path_arr, 10.0)) if path_arr.size >= 10 else float(np.mean(path_arr))
    )
    p10_gmgr = float(np.expm1(p10_log_tw))

    # --- Multiplicative Desirability Functions (0.0 to 1.0) ---
    # 1. Funding Drag (Target <= 15%)
    mean_fund_r = float(np.mean(funding_ratios)) if funding_ratios else 0.0
    d_fund = float(np.clip(1.0 - (max(0.0, abs(mean_fund_r) - 0.15) / 0.10), 0.0, 1.0))

    # 2. Long/Short Balance (Target imbalance <= 20%)
    tot_l, tot_s = float(all_seg_long), float(all_seg_short)
    ls_imbalance = abs(tot_l - tot_s) / (tot_l + tot_s + 1e-9)
    d_balance = float(np.clip(1.0 - (max(0.0, ls_imbalance - 0.20) / 0.30), 0.0, 1.0))

    # 3. Path Stability (Coefficient of Variation)
    mu_paths, sd_paths = (
        float(np.mean(path_arr)),
        float(np.std(path_arr, ddof=1)) if path_arr.size > 1 else 1.0,
    )
    cv_paths = sd_paths / (abs(mu_paths) + 1e-6)
    d_stability = float(np.clip(1.0 - (max(0.0, cv_paths - 1.5) / 1.0), 0.0, 1.0))

    # 4. Temporal Decay (Recent performance consistency)
    d_temporal = 1.0
    if len(path_compound_raw_log_tw) >= 4:
        path_center_times = [float(np.mean([(s + e) / 2.0 for s, e in p])) for p in cpcv_paths]
        time_sort_idx = np.argsort(path_center_times)
        tw_sorted = np.asarray(path_compound_raw_log_tw, dtype=np.float64)[time_sort_idx]
        recent_mean = float(np.mean(tw_sorted[-(len(tw_sorted) // 4) :]))
        d_temporal = float(
            np.clip(1.0 - (max(0.0, mu_paths - recent_mean) / (abs(mu_paths) + 1e-6)), 0.0, 1.0)
        )

    # 5. CVaR Log-Return (Tail Risk protection)
    sorted_rtns = np.sort(path_arr)
    k_worst = max(2, int(sorted_rtns.size * cvar_alpha))
    cvar_log = float(np.mean(sorted_rtns[:k_worst])) if sorted_rtns.size > 0 else -1.0
    d_cvar = float(np.clip(1.0 - (max(0.0, cvar_thr_log - cvar_log) / 0.10), 0.0, 1.0))

    # 6. MDD Soft Penalty (Method C modified)
    # [EXPLOSIVE GROWTH] P90 MDD 기반 지수 패널티 적용
    if path_mdds:
        p90_mdd = float(np.percentile(path_mdds, 90.0))
        if p90_mdd > 20.0:
            d_mdd = float(np.clip(np.exp(-(p90_mdd - 20.0) / 4.0), 0.01, 1.0))
        else:
            d_mdd = 1.0
    else:
        d_mdd = 1.0

    # 7. Directional PF (Min 1.05 for Long & Short)
    long_pf = calc_profit_factor_from_pnl(np.array(all_long_pnls)) if all_long_pnls else 1.5
    short_pf = calc_profit_factor_from_pnl(np.array(all_short_pnls)) if all_short_pnls else 1.5
    min_dir_pf = min(long_pf, short_pf)
    d_directional = float(np.clip(1.0 - (max(0.0, 1.10 - min_dir_pf) / 0.10), 0.01, 1.0))

    stability_bonus = 0.0
    if path_arr.size >= 2:
        neg_only = path_arr[path_arr < 0]
        ddev = float(np.std(neg_only, ddof=1)) if neg_only.size > 1 else sd_paths
        psort = float(mu_paths / (ddev + 1e-12)) if ddev > 0 else 0.0
        if psort > 1.5:
            stability_bonus += min(0.1, (psort - 1.5) * 0.05)

    objective_final = (
        p10_gmgr * (d_fund * d_balance * d_stability * d_temporal * d_cvar * d_mdd * d_directional)
        + stability_bonus
    )

    # --- Export Metrics & Constraints ---
    avg_pf = float(np.mean(path_pfs)) if path_pfs else 1.0
    avg_mdd = float(np.mean(path_mdds)) if path_mdds else 0.0
    avg_trades = float(np.mean(path_trades)) if path_trades else 0.0
    avg_ret_trade = (np.expm1(path_arr).mean() / avg_trades) if avg_trades > 0 else 0.0
    from config.settings import SLIPPAGE_RATE, TRADING_FEE_RATE

    ev_cost_is = (
        (avg_ret_trade / ((TRADING_FEE_RATE * 2.0) + (SLIPPAGE_RATE * 2.0)))
        if avg_trades > 0
        else 0.0
    )

    trial.set_user_attr("gate1_p10_gmgr", p10_gmgr)
    trial.set_user_attr("ev_cost_ratio", ev_cost_is)
    trial.set_user_attr("avg_pf", avg_pf)
    trial.set_user_attr("avg_mdd", avg_mdd)
    trial.set_user_attr("avg_trades", avg_trades)
    trial.set_user_attr("long_short_ratio", (min(tot_l, tot_s) / max(tot_l + tot_s, 1.0)))
    trial.set_user_attr("mean_funding_drag_ratio_pct", float(mean_fund_r * 100.0))
    trial.set_user_attr("cpcv_path_oos_log_tw", [float(x) for x in path_compound_raw_log_tw])

    tw_ratios = (
        np.asarray(path_terminal_wealth_ratios, dtype=np.float64)
        if mode == "multi"
        else np.exp(np.clip(path_arr, -50.0, 50.0))
    )
    trial.set_user_attr(
        "min_path_terminal_wealth_ratio", float(np.min(tw_ratios)) if tw_ratios.size else 0.0
    )
    trial.set_user_attr(
        "mean_path_terminal_wealth_ratio", float(np.mean(tw_ratios)) if tw_ratios.size else 0.0
    )

    if path_arr.size >= 2:
        m_pt, s_pt = float(np.mean(path_arr)), float(np.std(path_arr, ddof=1))
        sharpe = m_pt / (s_pt + 1e-12)
        trial.set_user_attr("gate1_sqn", float(math.sqrt(float(path_arr.size)) * sharpe))
        neg_only = path_arr[path_arr < 0]
        ddev = float(np.std(neg_only, ddof=1)) if neg_only.size > 1 else s_pt
        trial.set_user_attr(
            "gate1_path_sortino", float(m_pt / (ddev + 1e-12)) if ddev > 0 else 999.0
        )
        pos_only = path_arr[path_arr > 0]
        neg_mean = float(np.mean(neg_only)) if neg_only.size else -1e-9
        tr_val = (
            float(np.mean(pos_only) / (abs(neg_mean) + 1e-12))
            if pos_only.size and neg_only.size
            else (10.0 if pos_only.size else 0.0)
        )
        trial.set_user_attr("gate1_tail_ratio", tr_val)
        psr_est = float(min(0.99, max(0.0, 0.5 + 0.15 * sharpe)))
        trial.set_user_attr("gate1_psr", psr_est)
        trial.set_user_attr("psr_paths", psr_est)
        trial.set_user_attr("gate1_dsr", psr_est * 0.95)

    trial.set_user_attr("kelly_score_pct", float(objective_final * 100.0))
    trial.set_user_attr("growth_score", float(objective_final))
    return objective_final
