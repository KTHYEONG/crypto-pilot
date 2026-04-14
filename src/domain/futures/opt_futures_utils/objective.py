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
from src.domain.futures.opt_futures_utils.alpha_evaluator import (
    calculate_spearman_ic,
    compute_vol_adj_forward_returns,
)
from src.domain.futures.opt_futures_utils.cv_utils import (
    build_cpcv_test_paths_with_fallback,
    list_cpcv_block_ranges,
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
    _dataset_fingerprint_from_df,
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


def _merge_futures_params_fixed_then_suggest(
    trial: optuna.Trial,
    space: Dict[str, Dict[str, Any]],
    tf: str,
    fixed_param_overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Stage 2: locked signal params first, then current-trial suggestions (portfolio) win."""
    suggested = suggest_params_futures(trial, space, tf)
    if not fixed_param_overrides:
        return suggested
    merged = dict(fixed_param_overrides)
    merged.update(suggested)
    return merged

# Cross-sectional momentum lookback periods precomputed for all trials.
_CS_MOM_LOOKBACKS: List[int] = [12, 24, 36, 48, 60, 72]


def inject_cs_momentum_ranks(
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf: str,
    lookbacks: Optional[List[int]] = None,
) -> None:
    """Inject cross-sectional momentum rank columns into data_maps[sym][tf].

    Ranking at time t uses only pct_change(periods=lb) which is backward-looking
    — no temporal look-ahead bias.  Safe to call on IS-only or full (IS+OOS) data.
    """
    if lookbacks is None:
        lookbacks = _CS_MOM_LOOKBACKS
    if len(symbols) < 2:
        return

    rets_series: Dict[str, pd.Series] = {}
    for sym in symbols:
        df = data_maps.get(sym, {}).get(tf)
        if df is not None and not df.empty:
            rets_series[sym] = df["close"]

    if len(rets_series) < 2:
        return

    for lb in lookbacks:
        all_rets = {s: r.pct_change(periods=lb) for s, r in rets_series.items()}
        ranks_df = pd.DataFrame(all_rets).rank(axis=1, pct=True)
        for sym in rets_series:
            if sym in ranks_df.columns:
                col_name = f"cs_mom_rank_{lb}"
                data_maps[sym][tf][col_name] = ranks_df[sym].astype(np.float64)


def compute_multi_alignment_info(
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf: str,
    embargo: int,
) -> Optional[Dict[str, Any]]:
    """Precompute alignment, fingerprints, and CSM ranks to avoid per-trial overhead."""
    is_start_dts_per_sym: Dict[str, Any] = {}
    fingerprints: Dict[str, int] = {}

    for sym in symbols:
        sym_df = data_maps[sym].get(tf)
        if sym_df is None or sym_df.empty:
            continue

        # [OPTIMIZATION] Precompute fingerprint once per symbol
        fingerprints[sym] = _dataset_fingerprint_from_df(sym_df)

        is_off = int(data_maps[sym].get(f"is_start_idx_{tf}", 0))
        if len(sym_df) > is_off and "datetime" in sym_df.columns:
            is_start_dts_per_sym[sym] = sym_df["datetime"].iloc[is_off]

    if not is_start_dts_per_sym:
        return None

    common_is_start_dt = max(is_start_dts_per_sym.values())

    alignment_offsets: Dict[str, int] = {}
    eff_ref_lens: List[int] = []

    for sym in symbols:
        sym_df = data_maps[sym].get(tf)
        if sym_df is None or sym_df.empty or "datetime" not in sym_df.columns:
            continue
        # [OPTIMIZATION] Using searchsorted for O(log N) instead of O(N) mask
        start_idx = sym_df["datetime"].searchsorted(common_is_start_dt)
        alignment_offsets[sym] = int(start_idx)
        eff_ref_lens.append(len(sym_df) - int(start_idx))

    if not eff_ref_lens:
        return None

    eff_ref_len = min(eff_ref_lens)
    if eff_ref_len < 200:
        return None

    cpcv_bundle = build_cpcv_test_paths_with_fallback(eff_ref_len, embargo=embargo)
    if not cpcv_bundle[0]:
        return None

    # [OPTIMIZATION] Pre-calculate and Inject CS Momentum Ranks into data_maps
    inject_cs_momentum_ranks(data_maps, symbols, tf)

    return {
        "common_is_start_dt": common_is_start_dt,
        "alignment_offsets": alignment_offsets,
        "eff_ref_len": eff_ref_len,
        "cpcv_bundle": cpcv_bundle,
        "fingerprints": fingerprints,
    }


def objective_futures(
    trial: optuna.Trial,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf: str,
    *,
    space: Dict[str, Dict[str, Any]],
    mode: str = "single",
    project_root: Optional[str] = None,
    prebuilt_cpcv_bundle: Optional[Tuple[List[List[Tuple[int, int]]], int, int]] = None,
    multi_alignment_info: Optional[Dict[str, Any]] = None,
    signal_disk_cache_root: Optional[Path] = None,
    relaxed_constraints: bool = False,
    fixed_param_overrides: Optional[Dict[str, Any]] = None,
) -> float:
    params: Dict[str, Any] = _merge_futures_params_fixed_then_suggest(
        trial, space, tf, fixed_param_overrides
    )

    # [Institutional Quant] Inject Phase C Ensemble & Weighting configs
    winning_configs = space.get("_winning_configs")
    if isinstance(winning_configs, list):
        # If multiple signals were selected in Phase C, set up Ensemble
        if len(winning_configs) > 1:
            params["ENSEMBLE_SIGNALS"] = [
                {"name": w["name"], "params": w["params"]} for w in winning_configs
            ]

        # Inject the primary signal's discovered best regime weights
        # (Assuming all winning signals share similar regime profiles or we use Primary's)
        params["REGIME_WEIGHTS"] = winning_configs[0].get("regime_weights")

    # [OPTIMIZATION] CS_MOMENTUM column mapping to avoid per-trial DF copies
    # H5: Guard CSM_LOOKBACK to precomputed _CS_MOM_LOOKBACKS to prevent KeyError.
    if params.get("SIGNAL_TYPE") == "CS_MOMENTUM" and len(symbols) > 1:
        lb = int(params.get("CSM_LOOKBACK", 24))
        if lb not in _CS_MOM_LOOKBACKS:
            lb = min(_CS_MOM_LOOKBACKS, key=lambda x: abs(x - lb))
        params["CSM_RANK_COL"] = f"cs_mom_rank_{lb}"

    strategy: UltimateStrategy = UltimateStrategy(name="OptFutures", params=params)

    ref_sym = symbols[0]
    ref_df: Optional[pd.DataFrame] = data_maps.get(ref_sym, {}).get(tf)
    if ref_df is None or ref_df.empty:
        trial.set_user_attr("prune_reason", "no_ref_data")
        raise optuna.TrialPruned()

    is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
    ref_len = len(ref_df) - is_off

    # [OPTIMIZATION] Reuse precomputed alignment info if available
    if multi_alignment_info:
        ref_len = multi_alignment_info["eff_ref_len"]
        cpcv_paths = multi_alignment_info["cpcv_bundle"][0]
    else:
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
    cvar_thr_log = float(cfg.get("FUTURES_CPCV_CVAR_THRESHOLD_LOG", -0.05))

    cache_root = signal_disk_cache_root
    if cache_root is None and project_root is not None:
        cache_root = Path(project_root) / "cache_futures"

    from .signal_cache import get_tiered_signals

    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        target_df_raw = data_maps.get(sym, {}).get(tf)
        if target_df_raw is None or target_df_raw.empty:
            continue

        # [OPTIMIZATION] Use tiered caching instead of monolithic
        full_signal_dfs[sym] = get_tiered_signals(params, sym, tf, target_df_raw, strategy)

    if len(full_signal_dfs) != len(symbols):
        n_ok, n_tot = len(full_signal_dfs), len(symbols)
        _reason = (
            "signal_incomplete:no_bars"
            if n_ok == 0
            else f"signal_incomplete:{n_tot - n_ok}_symbols"
        )
        trial.set_user_attr("prune_reason", _reason)
        if not relaxed_constraints:
            raise optuna.TrialPruned()
        missing = [s for s in symbols if s not in full_signal_dfs]
        symbols = [s for s in symbols if s in full_signal_dfs]
        if not symbols:
            trial.set_user_attr("prune_reason", "signal_fail:all_symbols")
            raise optuna.TrialPruned()
        trial.set_user_attr("prune_reason", f"signal_partial:{len(missing)}_missing")

    # Tracks effective IS length after datetime alignment; used for annualization downstream
    _stat_ref_len: int = ref_len

    prebuilt_full_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    if mode == "multi":
        if multi_alignment_info is not None:
            # [OPTIMIZED PATH] Use precomputed common IS start and offsets
            eff_ref_len = multi_alignment_info["eff_ref_len"]
            cpcv_paths = multi_alignment_info["cpcv_bundle"][0]
            alignment_offsets = multi_alignment_info["alignment_offsets"]

            for sym in symbols:
                if sym not in alignment_offsets:
                    continue
                start_idx = alignment_offsets[sym]
                sym_df = full_signal_dfs[sym].iloc[start_idx:]
                prebuilt_full_arrays[sym] = _dataframe_to_symbol_arrays(sym_df)

            is_off = 0
            _stat_ref_len = eff_ref_len
        else:
            # [LEGACY PATH] Align all symbols to the INTERSECTION of IS periods.
            is_start_dts_per_sym: Dict[str, Any] = {}
            for sym in symbols:
                sym_df = full_signal_dfs[sym]
                sym_is_off = int(data_maps[sym].get(f"is_start_idx_{tf}", 0))
                if len(sym_df) > sym_is_off and "datetime" in sym_df.columns:
                    is_start_dts_per_sym[sym] = sym_df["datetime"].iloc[sym_is_off]

            if is_start_dts_per_sym:
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

                eff_ref_len = min(v["close"].shape[0] for v in prebuilt_full_arrays.values())
                if eff_ref_len < 200:
                    trial.set_user_attr("prune_reason", f"aligned_is_too_short:{eff_ref_len}")
                    raise optuna.TrialPruned()

                cpcv_paths, _, _ = build_cpcv_test_paths_with_fallback(
                    eff_ref_len, embargo=int(EMBARGO_BARS.get(tf, 0))
                )
                if not cpcv_paths:
                    trial.set_user_attr("prune_reason", "no_cpcv_paths_aligned")
                    raise optuna.TrialPruned()

                is_off = 0
                _stat_ref_len = eff_ref_len
            else:
                for sym in symbols:
                    prebuilt_full_arrays[sym] = _dataframe_to_symbol_arrays(full_signal_dfs[sym])

    liq_mdd_thr = float(cfg.get("FUTURES_MAX_MDD", 25.0))
    if relaxed_constraints:
        liq_mdd_thr = 100.0  # Bypass MDD constraint for robust mapping
        min_seg_trades = 1  # Bypass trade count constraint

    # --- CPCV Block Memoization (Priority 2) ---
    embargo = int(EMBARGO_BARS.get(tf, 0))
    n_blocks_cpcv = 8
    if multi_alignment_info:
        n_blocks_cpcv = multi_alignment_info["cpcv_bundle"][1]
    elif prebuilt_cpcv_bundle:
        n_blocks_cpcv = prebuilt_cpcv_bundle[1]

    unique_blocks = list_cpcv_block_ranges(_stat_ref_len, n_blocks_cpcv, embargo=embargo)

    block_results: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for b_start, b_end in unique_blocks:
        if mode == "multi":
            abs_start, abs_end = is_off + b_start, is_off + b_end
            slice_start, slice_end = max(0, abs_start - 1), min(len(ref_df), abs_end)
            if slice_end - slice_start < 2:
                continue

            aligned_data = _build_aligned_2d_from_prebuilt(
                prebuilt_full_arrays, symbols, slice_start, slice_end
            )
            if not aligned_data:
                continue

            try:
                from config.settings import SLIPPAGE_RATE, TRADING_FEE_RATE

                engine = PortfolioBacktestEngineFast(
                    aligned_data=aligned_data,
                    symbol_names=symbols,
                    strategy_params=params,
                    initial_balance=float(FUTURES_INITIAL_BALANCE),
                    fee_rate=TRADING_FEE_RATE,
                    slippage_rate=SLIPPAGE_RATE,
                )
                b_trades_df, equity_curve, final_balance = engine.run()
            except Exception as exc:
                _logger.warning("Block backtest error: %s", exc)
                continue

            if b_trades_df is None or b_trades_df.empty:
                block_results[(b_start, b_end)] = {"log_ret": -1.0, "mdd": 100.0, "trades": 0}
                continue

            p_arr = b_trades_df["pnl"].to_numpy(dtype=np.float64, copy=False)
            f_arr = b_trades_df["entry_fee"].to_numpy(dtype=np.float64, copy=False)
            true_pnl_arr = p_arr - f_arr
            side_arr = b_trades_df["side"].to_numpy(copy=False)

            block_results[(b_start, b_end)] = {
                "log_ret": _log_tw_from_ret_pct(
                    float((final_balance / FUTURES_INITIAL_BALANCE - 1.0) * 100.0)
                ),
                "mdd": float(calc_mdd_from_equity(equity_curve)) if equity_curve.size > 0 else 0.0,
                "pf_info": (
                    float(true_pnl_arr[true_pnl_arr > 0].sum()),
                    float(abs(true_pnl_arr[true_pnl_arr < 0].sum())),
                ),
                "trades": int(true_pnl_arr.size),
                "long_trades": int((side_arr == "LONG").sum()),
                "short_trades": int((side_arr == "SHORT").sum()),
                "long_pnls": true_pnl_arr[side_arr == "LONG"].tolist(),
                "short_pnls": true_pnl_arr[side_arr == "SHORT"].tolist(),
                "funding_ratio": float(
                    b_trades_df["funding_fee"].sum() / max(np.abs(p_arr).sum(), 1e-9)
                )
                if "funding_fee" in b_trades_df.columns
                else 0.0,
                # Store full trades slice for per-symbol Min-Max evaluation
                "trades_slice": b_trades_df,
            }
        else:
            # Single mode memoization
            b_logs, b_mdds, b_ntrs, b_lc, b_sc, b_fund = [], [], [], [], [], []
            for sym in symbols:
                target_df, daily_df = data_maps[sym][tf], data_maps[sym]["1d"]
                merge_idx = data_maps[sym][f"merge_idx_{tf}"]
                sym_is_off = int(data_maps[sym].get(f"is_start_idx_{tf}", 0))
                adj_s, adj_e = b_start + sym_is_off, b_end + sym_is_off
                seg, ex_idx = _segment_with_context(full_signal_dfs[sym], adj_s, adj_e)
                res_is = evaluate_symbol_fold(
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
                ret_pct, mdd_pct, ntr = res_is[1], res_is[2], res_is[3]
                lc, sc, fpaid, gross = res_is[6], res_is[7], res_is[9], res_is[10]

                b_logs.append(_log_tw_from_ret_pct(ret_pct))
                b_mdds.append(mdd_pct)
                b_ntrs.append(ntr)
                b_lc.append(lc)
                b_sc.append(sc)
                b_fund.append(float(fpaid) / max(abs(gross), 1e-9))

            block_results[(b_start, b_end)] = {
                "log_ret": float(np.mean(b_logs)),
                "mdd": float(np.mean(b_mdds)),
                "trades": int(np.sum(b_ntrs)),
                "long_trades": int(np.sum(b_lc)),
                "short_trades": int(np.sum(b_sc)),
                "funding_ratio": float(np.mean(b_fund)),
            }

    path_compound_raw_log_tw: List[float] = []
    successful_path_center_times: List[float] = []
    path_pfs, path_trades, path_mdds = [], [], []
    all_seg_long, all_seg_short = 0, 0
    all_long_pnls, all_short_pnls, funding_ratios = [], [], []

    for path_idx, path in enumerate(cpcv_paths):
        # ... (중략: 루프 내부 로직)
        p_log_ret, p_mdd, p_trades = 0.0, 0.0, 0
        p_l_prof, p_l_loss = 0.0, 0.0

        valid_path = True
        for b_key in path:
            key = tuple(b_key) if not isinstance(b_key, tuple) else b_key
            res = block_results.get(key)
            if res is None:
                valid_path = False
                break
            if res.get("mdd", 0) >= liq_mdd_thr or res.get("trades", 0) < min_seg_trades:
                valid_path = False
                break

            p_log_ret += res["log_ret"]
            p_mdd = max(p_mdd, res["mdd"])
            p_trades += res["trades"]
            all_seg_long += res.get("long_trades", 0)
            all_seg_short += res.get("short_trades", 0)
            if "pf_info" in res:
                p_l_prof += res["pf_info"][0]
                p_l_loss += res["pf_info"][1]
            if "long_pnls" in res:
                all_long_pnls.extend(res["long_pnls"])
            if "short_pnls" in res:
                all_short_pnls.extend(res["short_pnls"])
            funding_ratios.append(res.get("funding_ratio", 0.0))

        if not valid_path:
            if relaxed_constraints:
                continue
            raise optuna.TrialPruned()

        path_compound_raw_log_tw.append(p_log_ret)
        path_mdds.append(p_mdd)
        path_trades.append(float(p_trades))
        path_pfs.append(p_l_prof / max(p_l_loss, 1e-9) if p_l_loss > 0 else 1.5)
        # Collect center time only for valid paths to match length of path_compound_raw_log_tw
        successful_path_center_times.append(float(np.mean([(s + e) / 2.0 for s, e in path])))

        if path_idx >= 4:
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
    mean_fund_r = float(np.mean(funding_ratios)) if funding_ratios else 0.0
    d_fund = float(np.clip(1.0 - (max(0.0, abs(mean_fund_r) - 0.15) / 0.10), 0.0, 1.0))

    tot_l, tot_s = float(all_seg_long), float(all_seg_short)
    ls_imbalance = abs(tot_l - tot_s) / (tot_l + tot_s + 1e-9)
    d_balance = float(np.clip(1.0 - (max(0.0, ls_imbalance - 0.20) / 0.30), 0.0, 1.0))

    mu_paths, sd_paths = (
        float(np.mean(path_arr)),
        float(np.std(path_arr, ddof=1)) if path_arr.size > 1 else 1.0,
    )
    cv_paths = sd_paths / (abs(mu_paths) + 1e-6)
    d_stability = float(np.clip(1.0 - (max(0.0, cv_paths - 1.5) / 1.0), 0.0, 1.0))

    d_temporal = 1.0
    if len(path_compound_raw_log_tw) >= 4:
        # [FIX] Use successful_path_center_times to match length with path_compound_raw_log_tw
        time_sort_idx = np.argsort(successful_path_center_times)
        tw_sorted = np.asarray(path_compound_raw_log_tw, dtype=np.float64)[time_sort_idx]
        recent_mean = float(np.mean(tw_sorted[-(len(tw_sorted) // 4) :]))
        d_temporal = float(
            np.clip(1.0 - (max(0.0, mu_paths - recent_mean) / (abs(mu_paths) + 1e-6)), 0.0, 1.0)
        )

    sorted_rtns = np.sort(path_arr)
    k_worst = max(2, int(sorted_rtns.size * cvar_alpha))
    cvar_log = float(np.mean(sorted_rtns[:k_worst])) if sorted_rtns.size > 0 else -1.0
    d_cvar = float(np.clip(1.0 - (max(0.0, cvar_thr_log - cvar_log) / 0.10), 0.0, 1.0))

    if path_mdds:
        p90_mdd = float(np.percentile(path_mdds, 90.0))
        if p90_mdd > 20.0:
            d_mdd = float(np.clip(np.exp(-(p90_mdd - 20.0) / 4.0), 0.01, 1.0))
        else:
            d_mdd = 1.0
    else:
        d_mdd = 1.0

    l_pf = calc_profit_factor_from_pnl(np.array(all_long_pnls)) if all_long_pnls else 1.5
    s_pf = calc_profit_factor_from_pnl(np.array(all_short_pnls)) if all_short_pnls else 1.5
    min_dir_pf = min(l_pf, s_pf)
    d_directional = float(np.clip(1.0 - (max(0.0, 1.15 - min_dir_pf) / 0.15), 0.01, 1.0))

    # --- [New V5] Min-Max Symbol Penalty ---
    # Ensure no single symbol is holding back the portfolio or being 'sacrificed'
    d_min_sym_pf = 1.0
    if mode == "multi":
        sym_pfs = []
        # Get all unique trades across memoized blocks
        all_unique_trades_list = [
            res["trades_slice"] for res in block_results.values() if "trades_slice" in res
        ]
        if all_unique_trades_list:
            all_is_trades = pd.concat(all_unique_trades_list)
            for s_eval in symbols:
                s_trades = all_is_trades[all_is_trades["symbol"] == s_eval]
                if s_trades.empty:
                    sym_pfs.append(0.0)
                    continue
                s_pnl = s_trades["pnl"].to_numpy() - s_trades["entry_fee"].to_numpy()
                s_pf_val = calc_profit_factor_from_pnl(s_pnl)
                sym_pfs.append(s_pf_val)

            min_s_pf = min(sym_pfs)
            # Penalty starts if any individual symbol PF < 1.10
            d_min_sym_pf = float(np.clip(min_s_pf / 1.10, 0.0, 1.0))

    stability_bonus = 0.0
    if path_arr.size >= 2:
        neg_only = path_arr[path_arr < 0]
        ddev = float(np.std(neg_only, ddof=1)) if neg_only.size > 1 else sd_paths
        psort = float(mu_paths / (ddev + 1e-12)) if ddev > 0 else 0.0
        if psort > 1.5:
            stability_bonus += min(0.1, (psort - 1.5) * 0.05)

    objective_final = (
        p10_gmgr
        * (
            d_fund
            * d_balance
            * d_stability
            * d_temporal
            * d_cvar
            * d_mdd
            * d_directional
            * d_min_sym_pf
        )
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
    trial.set_user_attr("gate1_eff_ref_len", _stat_ref_len)
    trial.set_user_attr("ev_cost_ratio", ev_cost_is)
    trial.set_user_attr("avg_pf", avg_pf)
    trial.set_user_attr("avg_mdd", avg_mdd)
    trial.set_user_attr("avg_trades", avg_trades)
    trial.set_user_attr("long_short_ratio", (min(tot_l, tot_s) / max(tot_l + tot_s, 1.0)))
    trial.set_user_attr("mean_funding_drag_ratio_pct", float(mean_fund_r * 100.0))
    trial.set_user_attr("cpcv_path_oos_log_tw", [float(x) for x in path_compound_raw_log_tw])

    tw_ratios = np.exp(np.clip(path_arr, -50.0, 50.0))
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

        # Deflated Sharpe Ratio (Bailey & Lopez-de-Prado, 2014) — scipy-free.
        # Adjusts SR for multiple testing across n_paths CPCV paths.
        # SR_b ~ sqrt(2*log(n)) from Gumbel extreme-value approximation.
        n_paths_f = float(path_arr.size)
        sk = float(np.mean(((path_arr - m_pt) / (s_pt + 1e-12)) ** 3))
        ex_kurt = float(np.mean(((path_arr - m_pt) / (s_pt + 1e-12)) ** 4)) - 3.0
        sr_var_denom = max(
            1.0 - sk * sharpe + ((ex_kurt + 3.0 - 1.0) / 4.0) * sharpe**2,
            1e-12,
        )
        sr_bench = math.sqrt(2.0 * math.log(max(n_paths_f, 2.0)))
        z_dsr = (sharpe - sr_bench) * math.sqrt(max(n_paths_f - 1.0, 1.0)) / math.sqrt(sr_var_denom)
        dsr_val = float(0.5 * (1.0 + math.erf(z_dsr / math.sqrt(2.0))))
        trial.set_user_attr("gate1_dsr", float(min(0.99, max(0.0, dsr_val))))

    trial.set_user_attr("kelly_score_pct", float(objective_final * 100.0))
    trial.set_user_attr("growth_score", float(objective_final))
    return objective_final


def objective_futures_discovery(
    trial: optuna.Trial,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf: str,
    *,
    space: Dict[str, Dict[str, Any]],
    signal_disk_cache_root: Optional[Path] = None,
    project_root: Optional[str] = None,
    prebuilt_cpcv_bundle: Optional[Tuple[List[List[Tuple[int, int]]], int, int]] = None,
) -> float:
    """Stage 1: IC-based discovery objective (single allocation, signal params only)."""
    params: Dict[str, Any] = suggest_params_futures(trial, space, tf)

    winning_configs = space.get("_winning_configs")
    if isinstance(winning_configs, list):
        if len(winning_configs) > 1:
            params["ENSEMBLE_SIGNALS"] = [
                {"name": w["name"], "params": w["params"]} for w in winning_configs
            ]
        params["REGIME_WEIGHTS"] = winning_configs[0].get("regime_weights")

    if params.get("SIGNAL_TYPE") == "CS_MOMENTUM" and len(symbols) > 1:
        lb = int(params.get("CSM_LOOKBACK", 24))
        if lb not in _CS_MOM_LOOKBACKS:
            lb = min(_CS_MOM_LOOKBACKS, key=lambda x: abs(x - lb))
        params["CSM_RANK_COL"] = f"cs_mom_rank_{lb}"

    strategy = UltimateStrategy(name="DiscoveryFutures", params=params)

    cache_root = signal_disk_cache_root
    if cache_root is None and project_root is not None:
        cache_root = Path(project_root) / "cache_futures"

    from .signal_cache import get_tiered_signals

    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        target_df_raw = data_maps.get(sym, {}).get(tf)
        if target_df_raw is None or target_df_raw.empty:
            continue
        full_signal_dfs[sym] = get_tiered_signals(params, sym, tf, target_df_raw, strategy)

    if len(full_signal_dfs) != len(symbols):
        trial.set_user_attr("prune_reason", "discovery_signal_fail")
        raise optuna.TrialPruned()

    ref_df = data_maps[symbols[0]][tf]
    is_off = int(data_maps[symbols[0]].get(f"is_start_idx_{tf}", 0))
    ref_len = len(ref_df) - is_off
    embargo = int(EMBARGO_BARS.get(tf, 0))
    if prebuilt_cpcv_bundle:
        cpcv_paths, n_blocks, _ = prebuilt_cpcv_bundle
    else:
        cpcv_paths, n_blocks, _ = build_cpcv_test_paths_with_fallback(ref_len, embargo=embargo)

    if not cpcv_paths:
        trial.set_user_attr("prune_reason", "no_cpcv_paths")
        raise optuna.TrialPruned()

    unique_blocks = list_cpcv_block_ranges(ref_len, n_blocks, embargo=embargo)
    horizon = int(OPT_FUTURES_CONFIG.get("FUTURES_STAGE1_IC_LOOKFORWARD_BARS", 12))
    horizons_list = [horizon]

    block_ic_results: Dict[Tuple[int, int], Dict[str, float]] = {}
    for b_start, b_end in unique_blocks:
        block_ics: List[float] = []
        for sym in symbols:
            sig_df = full_signal_dfs[sym]
            adj_s, adj_e = b_start + is_off, b_end + is_off
            seg = sig_df.iloc[max(0, adj_s) : adj_e]
            if len(seg) < horizon + 2:
                continue
            if "slot_rank_score" not in seg.columns:
                continue
            _ohlc = ("open", "high", "low", "close")
            if not all(c in seg.columns for c in _ohlc):
                continue
            vol_map = compute_vol_adj_forward_returns(seg, horizons_list)
            vol_adj_fwd = vol_map[horizon]
            signal_arr = seg["slot_rank_score"].to_numpy(dtype=np.float64)
            min_len = min(len(signal_arr), len(vol_adj_fwd))
            if min_len < 50:
                continue
            ic = calculate_spearman_ic(signal_arr[:min_len], vol_adj_fwd[:min_len])
            block_ics.append(ic)

        if block_ics:
            block_ic_results[(b_start, b_end)] = {
                "ic": float(np.mean(block_ics)),
                "ic_std": float(np.std(block_ics)) if len(block_ics) > 1 else 0.0,
            }

    path_ics: List[float] = []
    for path in cpcv_paths:
        p_ics: List[float] = []
        for b in path:
            b_key = (int(b[0]), int(b[1]))
            br = block_ic_results.get(b_key)
            if br is not None:
                p_ics.append(br["ic"])
        if len(p_ics) >= 2:
            path_ics.append(float(np.mean(p_ics)))

    if not path_ics:
        trial.set_user_attr("prune_reason", "discovery_no_path_ics")
        raise optuna.TrialPruned()

    path_ic_arr = np.asarray(path_ics, dtype=np.float64)

    p10_ic = (
        float(np.percentile(path_ic_arr, 10.0))
        if path_ic_arr.size >= 10
        else float(np.mean(path_ic_arr))
    )

    ic_mean = float(np.mean(path_ic_arr))
    ic_std = float(np.std(path_ic_arr, ddof=1)) if path_ic_arr.size > 1 else 0.0
    ic_cv = ic_std / (abs(ic_mean) + 1e-6)
    d_ic_stability = float(np.clip(1.0 - max(0.0, ic_cv - 1.0) / 1.0, 0.0, 1.0))

    k_worst = max(2, int(path_ic_arr.size * 0.10))
    cvar_ic = float(np.mean(np.sort(path_ic_arr)[:k_worst]))
    d_cvar_ic = float(np.clip(1.0 - max(0.0, -0.02 - cvar_ic) / 0.05, 0.0, 1.0))

    objective_discovery = p10_ic * d_ic_stability * d_cvar_ic

    trial.set_user_attr("p10_ic", p10_ic)
    trial.set_user_attr("mean_ic", ic_mean)
    trial.set_user_attr("ic_stability", d_ic_stability)
    trial.set_user_attr("cvar_ic", cvar_ic)
    trial.set_user_attr("kelly_score_pct", float(objective_discovery * 100.0))

    return float(objective_discovery)
