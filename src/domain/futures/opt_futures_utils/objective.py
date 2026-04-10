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
    SLIPPAGE_RATE,
    TRADING_FEE_RATE,
)
from src.domain.futures.engine_portfolio_futures import PortfolioBacktestEngineFast
from src.domain.futures.opt_futures_utils.cv_utils import (
    build_cpcv_test_paths_with_fallback,
)
from src.domain.futures.opt_futures_utils.metrics import (
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
    fixed_min: Dict[str, int] = {"1h": 24, "4h": 6}
    ratio_map: Dict[str, float] = {"1h": 0.08, "4h": 0.05}
    ratio: float = ratio_map.get(tf, 0.03)
    return max(fixed_min.get(tf, 2), int(longest_indicator_period * ratio))


EMBARGO_BARS: Dict[str, int] = {
    "1h": compute_embargo_bars("1h"),
    "4h": compute_embargo_bars("4h"),
}



































def calc_tail_ratio_from_equity(equity: np.ndarray) -> float:
    """95th percentile return / abs(5th percentile return)."""
    if equity.size < 2:
        return 1.0
    r = np.diff(equity) / np.clip(equity[:-1], 1e-12, None)
    if r.size < 5:
        return 1.0
    val95 = float(np.percentile(r, 95.0))
    val5 = float(np.percentile(r, 5.0))
    if abs(val5) < 1e-12:
        return 5.0 if val95 > 0 else 1.0
    return float(val95 / abs(val5))


def _log_tw_from_ret_pct(ret_pct: float) -> float:
    r = 1.0 + float(ret_pct) / 100.0
    if r <= 0.0 or not math.isfinite(r):
        return -10.0
    return float(math.log(max(r, 1e-9)))


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
) -> float:
    params: Dict[str, Any] = suggest_params_futures(trial, space, tf_target)
    tf: str = tf_target
    strategy: UltimateStrategy = UltimateStrategy(name="OptFutures", params=params)

    ref_sym = symbols[0]
    ref_df: Optional[pd.DataFrame] = data_maps.get(ref_sym, {}).get(tf)
    if ref_df is None or ref_df.empty:
        raise optuna.TrialPruned()

    is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
    ref_len = len(ref_df) - is_off
    if ref_len < 200:
        raise optuna.TrialPruned()

    embargo = int(EMBARGO_BARS.get(tf, 0))
    if prebuilt_cpcv_bundle is not None:
        cpcv_paths, _, _ = prebuilt_cpcv_bundle
    else:
        cpcv_paths, _, _ = build_cpcv_test_paths_with_fallback(ref_len, embargo=embargo)
    if not cpcv_paths:
        raise optuna.TrialPruned()

    cfg: Dict[str, Any] = OPT_FUTURES_CONFIG
    min_seg_trades = int(cfg.get("FUTURES_MIN_TRADES_PER_CPCV_SEGMENT", 5))
    min_pf_trades_dynamic = max(40, len(symbols) * 8)
    
    w_mean = float(np.clip(float(cfg.get("FUTURES_OBJECTIVE_W_MEAN_LOG_TW", 0.7)), 0.0, 1.0))
    w_p10 = 1.0 - w_mean
    cvar_alpha = float(cfg.get("FUTURES_CPCV_CVAR_ALPHA", 0.10))
    cvar_thr = float(cfg.get("FUTURES_CPCV_CVAR_THRESHOLD", 0.05))
    cvar_weight = float(cfg.get("FUTURES_CPCV_CVAR_WEIGHT", 0.80))
    temporal_lambda = float(cfg.get("FUTURES_CPCV_TEMPORAL_LAMBDA", 1.5))

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
        raise optuna.TrialPruned()

    prebuilt_full_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    if mode == "multi":
        for sym in symbols:
            prebuilt_full_arrays[sym] = _dataframe_to_symbol_arrays(full_signal_dfs[sym])

    leverage = float(params.get("LEVERAGE", 20.0))
    liq_mdd_thr = float(cfg.get("FUTURES_MAX_MDD", 25.0))

    path_compound_raw_log_tw: List[float] = []
    path_pfs: List[float] = []
    path_wins: List[float] = []
    path_trades: List[float] = []
    path_mdds: List[float] = []
    all_seg_long: int = 0
    all_seg_short: int = 0
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
                    raise optuna.TrialPruned()
                    
                segment_initial = max(running_balance, 1e-9)
                try:
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
                    raise optuna.TrialPruned()

                if trades_df is None or trades_df.empty:
                    raise optuna.TrialPruned()

                ret_pct = float((final_balance / segment_initial - 1.0) * 100.0)
                raw_log = _log_tw_from_ret_pct(ret_pct)
                running_balance = max(float(final_balance), 1e-9)
                mdd_seg = float(calc_mdd_from_equity(equity_curve)) if equity_curve.size > 0 else 0.0
                true_pnl = trades_df["pnl"] - trades_df["entry_fee"]
                pf_s = calc_profit_factor_from_pnl(true_pnl)
                win_rate_s = (
                    float((len(trades_df[true_pnl > 0]) / len(trades_df)) * 100)
                    if len(trades_df) > 0
                    else 0.0
                )
                n_tr = int(len(trades_df))
                lc = int(len(trades_df[trades_df["side"] == "LONG"]))
                sc = int(len(trades_df[trades_df["side"] == "SHORT"]))
                all_seg_long += lc
                all_seg_short += sc
                seg_raw_logs.append(raw_log)
                seg_mdds.append(mdd_seg)
                seg_pfs.append(pf_s)
                seg_wins.append(win_rate_s)
                seg_trades.append(float(n_tr))
                _fp = float(trades_df["funding_fee"].sum()) if "funding_fee" in trades_df.columns else 0.0
                _gr = float(trades_df["pnl"].abs().sum())
                funding_ratios.append(_fp / max(_gr, 1e-9))

                if mdd_seg >= liq_mdd_thr:
                    raise optuna.TrialPruned()

                if n_tr < min_seg_trades:
                    raise optuna.TrialPruned()

                continue

            sym_logs: List[float] = []
            sym_mdds: List[float] = []
            sym_pfs: List[float] = []
            sym_wins: List[float] = []
            sym_trades: List[float] = []
            sym_fund_r: List[float] = []

            for sym in symbols:
                target_df = data_maps[sym][tf]
                daily_df = data_maps[sym]["1d"]
                merge_idx = data_maps[sym][f"merge_idx_{tf}"]
                is_start_idx = int(data_maps[sym].get(f"is_start_idx_{tf}", 0))
                adj_s = test_start + is_start_idx
                adj_e = test_end + is_start_idx
                seg, ex_idx = _segment_with_context(full_signal_dfs[sym], adj_s, adj_e)
                _cagr, ret_pct, mdd_pct, ntr, wr, pf, lc, sc, _eq, fpaid, gross = evaluate_symbol_fold(
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
                    precomputed_signal_df=seg,
                    execution_start_idx=ex_idx,
                )
                lg = _log_tw_from_ret_pct(ret_pct)
                if mdd_pct >= liq_mdd_thr:
                    lg -= 1e9
                denom = max(abs(gross), 1e-9)
                sym_fund_r.append(float(fpaid) / denom)
                sym_logs.append(lg)
                sym_mdds.append(mdd_pct)
                sym_pfs.append(pf)
                sym_wins.append(wr)
                sym_trades.append(float(ntr))
                all_seg_long += lc
                all_seg_short += sc
                sym_pf_accum[sym].append(pf)

            n_sym = max(len(sym_logs), 1)
            total_trades_seg = float(np.sum(sym_trades))
            if total_trades_seg < float(min_seg_trades):
                raise optuna.TrialPruned()

            seg_raw_logs.append(float(np.mean(sym_logs)) if sym_logs else -10.0)
            seg_mdds.append(float(np.mean(sym_mdds)) if sym_mdds else 0.0)
            seg_pfs.append(float(np.mean(sym_pfs)) if sym_pfs else 1.0)
            seg_wins.append(float(np.mean(sym_wins)) if sym_wins else 0.0)
            seg_trades.append(total_trades_seg / n_sym)
            seg_fund_ratio.append(float(np.mean(sym_fund_r)) if sym_fund_r else 0.0)

        if mode == "multi":
            path_terminal_wealth_ratios.append(
                float(running_balance / float(FUTURES_INITIAL_BALANCE))
            )

        if not seg_raw_logs:
            raise optuna.TrialPruned()

        path_compound_raw_log_tw.append(float(np.sum(seg_raw_logs)))
        path_pfs.append(float(np.mean(seg_pfs)))
        path_wins.append(float(np.mean(seg_wins)))
        path_trades.append(float(np.mean(seg_trades)))
        path_mdds.append(float(np.max(seg_mdds)) if seg_mdds else 0.0)
        funding_ratios.extend(seg_fund_ratio)

        if path_idx >= 4 and path_compound_raw_log_tw:
            trial.report(float(np.mean(path_compound_raw_log_tw)), step=path_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    path_arr = np.asarray(path_compound_raw_log_tw, dtype=np.float64)

    mean_log_tw_k = float(np.mean(path_arr)) if path_arr.size > 0 else 0.0
    p10_log_tw_path = (
        float(np.percentile(path_arr, 10.0)) if path_arr.size >= 10 else mean_log_tw_k
    )
    kelly_obj = w_mean * mean_log_tw_k + w_p10 * p10_log_tw_path

    sorted_rtns = np.sort(path_arr)
    n_paths_log = int(sorted_rtns.size)
    if n_paths_log > 0:
        k_worst = max(2, int(n_paths_log * cvar_alpha))
        k_worst = min(k_worst, n_paths_log)
        cvar_val = float(-np.mean(sorted_rtns[:k_worst]))
    else:
        cvar_val = 0.0
    cvar_pen = max(0.0, cvar_val - cvar_thr) * cvar_weight

    n_p_raw = len(path_compound_raw_log_tw)
    temporal_decay_pen = 0.0
    if n_p_raw >= 4 and temporal_lambda > 0.0:
        tw_raw_arr = np.asarray(path_compound_raw_log_tw, dtype=np.float64)
        k_recent = max(2, n_p_raw // 4)
        all_mean_tw = float(np.mean(tw_raw_arr))
        recent_mean_tw = float(np.mean(tw_raw_arr[-k_recent:]))
        temporal_decay_pen = max(0.0, all_mean_tw - recent_mean_tw) * temporal_lambda

    mu_paths = float(np.mean(path_arr)) if path_arr.size > 0 else -10.0
    sd_paths = float(np.std(path_arr, ddof=1)) if path_arr.size > 1 else 10.0
    cv_paths = sd_paths / (abs(mu_paths) + 1e-6) if path_arr.size > 1 else 10.0
    concentration_pen = max(0.0, cv_paths - 1.5) * 0.15

    mean_fund_r = float(np.mean(funding_ratios)) if funding_ratios else 0.0
    # Penalize both excessive funding cost (>15%) and excessive funding income (< -15%, strong penalty to prevent yield-farming overfit)
    funding_drag_pen = max(0.0, mean_fund_r - 0.15) * 2.0 + max(0.0, -mean_fund_r - 0.15) * 4.0

    min_reg_cfg = float(cfg.get("FUTURES_MIN_REGIME_ON_RATE", 0.0))
    w_reg_cfg = float(cfg.get("FUTURES_REGIME_ON_PENALTY_WEIGHT", 0.0))
    regime_on_rate = 0.5
    ref_sig_df = full_signal_dfs.get(ref_sym)
    if ref_sig_df is not None and "regime_risk_mult" in ref_sig_df.columns:
        rrm = ref_sig_df["regime_risk_mult"].to_numpy(dtype=np.float64)
        if rrm.size > is_off:
            rrm_is = rrm[is_off:]
            regime_on_rate = float(np.mean(rrm_is > 0.5)) if rrm_is.size else 0.5

    avg_trades = float(np.mean(path_trades)) if path_trades else 0.0
    
    trade_penalty = 0.0
    if avg_trades < min_pf_trades_dynamic:
        trade_penalty = ((min_pf_trades_dynamic - avg_trades) / min_pf_trades_dynamic) ** 2 * 5.0
        
    sortino_bonus = 0.0
    cagr_bonus = 0.0
    if path_arr.size >= 2:
        m_pt = float(np.mean(path_arr))
        s_pt = float(np.std(path_arr, ddof=1))
        neg_only = path_arr[path_arr < 0]
        ddev = float(np.std(neg_only, ddof=1)) if neg_only.size > 1 else s_pt
        psort = float(m_pt / (ddev + 1e-12)) if ddev > 0 else 0.0
        pos_only = path_arr[path_arr > 0]
        neg_mean = float(np.mean(neg_only)) if neg_only.size else -1e-9
        tr_val = 0.0
        if pos_only.size and neg_only.size:
             tr_val = float(np.mean(pos_only) / (abs(neg_mean) + 1e-12))
        elif pos_only.size and not neg_only.size:
             tr_val = 5.0
             
        if psort > 1.5:
             sortino_bonus += min(0.5, (psort - 1.5) * 0.1)
        if tr_val > 1.5:
             sortino_bonus += min(0.5, (tr_val - 1.5) * 0.1)
             
        # Add CAGR bonus to force TPE to seek growth (up to 30%)
        mean_path_ret_pct = float(np.mean(np.expm1(path_arr) * 100.0)) if path_arr.size else 0.0
        if mean_path_ret_pct > 0:
            cagr_bonus = min(1.0, mean_path_ret_pct / 30.0)

    objective_final = float(
        kelly_obj
        - cvar_pen
        - funding_drag_pen
        - concentration_pen
        - temporal_decay_pen
        - trade_penalty
        + sortino_bonus
        + cagr_bonus
    )
    if min_reg_cfg > 0.0 and w_reg_cfg > 0.0:
        objective_final -= max(0.0, min_reg_cfg - regime_on_rate) * w_reg_cfg

    _opt_meta = data_maps.get(ref_sym, {}).get("_futures_opt_meta")
    if isinstance(_opt_meta, dict):
        _obj_floor = _opt_meta.get("objective_floor_strict")
        if _obj_floor is not None and objective_final < float(_obj_floor):
            raise optuna.TrialPruned()

    avg_pf = float(np.mean(path_pfs)) if path_pfs else 1.0
    avg_win_rate = float(np.mean(path_wins)) if path_wins else 0.0
    avg_mdd = float(np.mean(path_mdds)) if path_mdds else 0.0

    if mode == "multi":
        min_sym_pf = avg_pf
    else:
        per_sym_mean_pf = [
            float(np.mean(sym_pf_accum[s])) if sym_pf_accum.get(s) else avg_pf for s in symbols
        ]
        min_sym_pf = float(np.min(per_sym_mean_pf)) if per_sym_mean_pf else avg_pf

    tot_l = float(all_seg_long)
    tot_s = float(all_seg_short)
    minority = float(min(tot_l, tot_s))
    majority = float(max(tot_l + tot_s, 1.0))
    ls_ratio = minority / majority

    n_cpcv_paths = int(len(cpcv_paths))
    worst_seg_mdd_pct = float(np.max(path_mdds)) if path_mdds else 0.0
    mean_path_ret_pct = (
        float(np.mean(np.expm1(path_arr) * 100.0)) if path_arr.size else 0.0
    )

    if path_arr.size >= 2:
        m_pt = float(np.mean(path_arr))
        s_pt = float(np.std(path_arr, ddof=1))
        sharpe_paths = m_pt / (s_pt + 1e-12)
        gate1_sqn = float(math.sqrt(float(path_arr.size)) * sharpe_paths)
        neg_only = path_arr[path_arr < 0]
        ddev = float(np.std(neg_only, ddof=1)) if neg_only.size > 1 else s_pt
        gate1_path_sortino = float(m_pt / (ddev + 1e-12)) if ddev > 0 else 999.0
        pos_only = path_arr[path_arr > 0]
        neg_mean = float(np.mean(neg_only)) if neg_only.size else -1e-9
        if pos_only.size and neg_only.size:
            gate1_tail_ratio = float(np.mean(pos_only) / (abs(neg_mean) + 1e-12))
        elif pos_only.size and not neg_only.size:
            gate1_tail_ratio = 10.0
        else:
            gate1_tail_ratio = 0.0
        gate1_psr = float(min(0.99, max(0.0, 0.5 + 0.12 * sharpe_paths)))
        gate1_dsr = float(
            min(
                0.99,
                max(
                    0.0,
                    gate1_psr * (1.0 - 0.08 * min(1.0, 1.0 / float(path_arr.size))),
                ),
            )
        )
    else:
        gate1_sqn = 0.0
        gate1_path_sortino = 0.0
        gate1_tail_ratio = 0.0
        gate1_psr = 0.0
        gate1_dsr = 0.0

    gate1_p10_gmgr = float(p10_log_tw_path)
    funding_drag_pct_stat = float(mean_fund_r * 100.0)

    if mode == "multi":
        min_path_terminal_wealth_ratio = (
            float(np.min(np.asarray(path_terminal_wealth_ratios, dtype=np.float64)))
            if path_terminal_wealth_ratios
            else 0.0
        )
        mean_path_terminal_wealth_ratio = (
            float(np.mean(np.asarray(path_terminal_wealth_ratios, dtype=np.float64)))
            if path_terminal_wealth_ratios
            else 0.0
        )
    else:
        tw_path_arr = np.asarray(
            [float(np.exp(np.clip(x, -50.0, 50.0))) for x in path_compound_raw_log_tw],
            dtype=np.float64,
        )
        min_path_terminal_wealth_ratio = (
            float(np.min(tw_path_arr)) if tw_path_arr.size else 0.0
        )
        mean_path_terminal_wealth_ratio = (
            float(np.mean(tw_path_arr)) if tw_path_arr.size else 0.0
        )

    trial.set_user_attr("avg_cagr", float(np.mean(path_compound_raw_log_tw)))
    trial.set_user_attr("avg_mdd", avg_mdd)
    trial.set_user_attr("avg_trades", avg_trades)
    trial.set_user_attr("avg_pf", avg_pf)
    trial.set_user_attr("avg_win_rate", avg_win_rate)
    trial.set_user_attr("min_sym_pf", min_sym_pf)
    trial.set_user_attr("long_short_ratio", ls_ratio)
    trial.set_user_attr("objective_final", objective_final)
    trial.set_user_attr("kelly_score_pct", float(mean_log_tw_k * 100.0))
    trial.set_user_attr("gate1_sqn", gate1_sqn)
    trial.set_user_attr("gate1_path_sortino", gate1_path_sortino)
    trial.set_user_attr("gate1_tail_ratio", gate1_tail_ratio)
    trial.set_user_attr("gate1_psr", gate1_psr)
    trial.set_user_attr("regime_on_rate", float(regime_on_rate))
    trial.set_user_attr("gate1_dsr", gate1_dsr)
    trial.set_user_attr("gate1_p10_gmgr", gate1_p10_gmgr)
    trial.set_user_attr("cpcv_mean_path_return_pct", mean_path_ret_pct)
    trial.set_user_attr("cpcv_worst_segment_mdd_pct", worst_seg_mdd_pct)
    trial.set_user_attr("n_cpcv_paths", n_cpcv_paths)
    trial.set_user_attr("mean_funding_drag_ratio_pct", funding_drag_pct_stat)
    trial.set_user_attr("cpcv_path_oos_log_tw", path_compound_raw_log_tw)
    trial.set_user_attr("min_path_terminal_wealth_ratio", float(min_path_terminal_wealth_ratio))
    trial.set_user_attr("mean_path_terminal_wealth_ratio", float(mean_path_terminal_wealth_ratio))
    trial.set_user_attr("psr_paths", float(gate1_psr))
    trial.set_user_attr("growth_score", float(objective_final))

    return objective_final










