"""NSGA-II Phase D objectives for ML pipeline (CPCV paths, O(1) slicing)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.trial import FrozenTrial

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import FUTURES_INITIAL_BALANCE, SLIPPAGE_RATE, TRADING_FEE_RATE
from src.domain.futures.engine_multi_futures import (
    backtest_portfolio_numba,
)
from src.domain.futures.opt_futures_utils.cv_utils import (
    build_cpcv_test_paths_with_fallback,
    list_cpcv_block_ranges,
)
from src.domain.futures.opt_futures_utils.data_utils import (
    _build_aligned_2d_from_prebuilt,
    _dataframe_to_symbol_arrays,
)
from src.domain.futures.opt_futures_utils.metrics import _log_tw_from_ret_pct, calc_mdd_from_equity
from src.domain.futures.opt_futures_utils.objective import (
    EMBARGO_BARS,
    compute_multi_alignment_info,
)
from src.domain.futures.opt_futures_utils.signal_cache import get_tiered_signals
from src.domain.futures.strategies_futures import UltimateStrategy

_logger = logging.getLogger(__name__)

ML_PHASE_D_PARAM_SPACE: Dict[str, Any] = {
    "ENTRY_THRESHOLD": {"type": "float", "low": 0.05, "high": 0.50},
    "TRAILING_ACTIVATION_ATR": {"type": "float", "low": 0.3, "high": 1.2},
    "BAYESIAN_C": {"type": "float", "low": 1.0, "high": 30.0, "log": True},
    "KELLY_SHRINKAGE": {"type": "float", "low": 0.05, "high": 0.4},
}


@dataclass
class MLPhaseDContext:
    data_maps: Dict[str, Dict[str, Any]]
    symbols: List[str]
    tf: str
    seed: int = 42
    # [OPTIMIZATION] Pre-aligned global matrices for O(1) slicing
    cpcv_block_slices: Optional[List[Dict[str, Any]]] = None
    holdout_slice: Optional[Dict[str, np.ndarray]] = None
    multi_alignment_info: Optional[Dict[str, Any]] = None


def _inject_dyn_leverage_trimmed(trimmed_sig: pd.DataFrame, raw_full: pd.DataFrame) -> None:
    """P3.B: HMM argmax state → dynamic leverage (2..LEVERAGE cap) for Numba margin."""
    hmm_cols = sorted(
        (c for c in raw_full.columns if str(c).startswith("hmm_prob_")),
        key=lambda x: int(str(x).split("_")[-1]),
    )
    base_lev = float(OPT_FUTURES_CONFIG.get("FUTURES_DISCOVERY_LEVERAGE", 5))
    if not hmm_cols:
        trimmed_sig["dyn_leverage"] = float(base_lev)
        return
    p_mat = (
        raw_full[hmm_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0 / float(len(hmm_cols)))
        .to_numpy(dtype=np.float64)
    )
    state_id = np.argmax(p_mat, axis=1).astype(np.float64)
    k = len(hmm_cols)
    lev_min = 2.0
    lev_max = max(base_lev, lev_min)
    if k <= 1:
        levs: np.ndarray = np.full(len(state_id), lev_max, dtype=np.float64)
    else:
        levs = np.asarray(
            lev_min + (lev_max - lev_min) * (state_id / float(k - 1)),
            dtype=np.float64,
        )
    trimmed_sig["dyn_leverage"] = levs


def precompute_ml_optimization_context(ctx: MLPhaseDContext) -> None:
    """Pre-align and pre-slice all data before Optuna starts to eliminate trial overhead."""
    # 1. Alignment & Baseline Signals
    pre_ml = {
        "ENTRY_THRESHOLD": 0.5,
        "TRAILING_ACTIVATION_ATR": 1.0,
        "BAYESIAN_C": 10.0,
        "KELLY_SHRINKAGE": 0.3,
    }
    params = _base_engine_params(pre_ml, ctx.tf)
    strategy = UltimateStrategy(name="Precompute", params=params)

    emb = int(EMBARGO_BARS.get(ctx.tf, 12))
    info = compute_multi_alignment_info(ctx.data_maps, ctx.symbols, ctx.tf, emb)
    if info is None:
        return
    ctx.multi_alignment_info = info
    
    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    for sym in ctx.symbols:
        raw = ctx.data_maps[sym][ctx.tf]
        full_signal_dfs[sym] = get_tiered_signals(params, sym, ctx.tf, raw, strategy)

    # 2. Build Global Aligned Arrays
    prebuilt_full: Dict[str, Dict[str, np.ndarray]] = {}
    eff_len = int(info["eff_ref_len"])
    for sym in ctx.symbols:
        start_idx = int(info["alignment_offsets"][sym])
        trimmed_sig = full_signal_dfs[sym].iloc[start_idx:start_idx+eff_len].copy()
        
        # [DYNAMIC FIX] Re-calculate trend_direction and strength WITHOUT threshold bias
        # This allows Numba to apply the threshold dynamically.
        raw_full = ctx.data_maps[sym][ctx.tf].iloc[start_idx:start_idx+eff_len]
        if "funding_rate_sum" in raw_full.columns:
            trimmed_sig["funding_rate_sum"] = raw_full["funding_rate_sum"].to_numpy(
                dtype=np.float64, copy=False
            )
        p_long = raw_full["ml_calib_prob_long"].values.astype(np.float64)
        p_short = raw_full["ml_calib_prob_short"].values.astype(np.float64)
        
        # Override signals with threshold-agnostic versions
        trimmed_sig["ml_calib_prob"] = np.maximum(p_long, p_short)
        td = np.where(p_long >= p_short, 1.0, -1.0)
        trimmed_sig["trend_direction"] = td
        
        # [DYNAMIC FIX] Ensure entry bounds allow trading regardless of precompute threshold
        # For Long (td=1): upper=0 (market entry), For Short (td=-1): lower=inf (market entry)
        trimmed_sig["entry_upper"] = np.where(td == 1.0, 0.0, 999999.0)
        trimmed_sig["entry_lower"] = np.where(td == -1.0, 999999.0, 0.0)

        _inject_dyn_leverage_trimmed(trimmed_sig, raw_full)

        prebuilt_full[sym] = _dataframe_to_symbol_arrays(trimmed_sig)

    # 3. CPCV Zone & Holdout Slicing
    _holdout_ratio = 0.20
    cpcv_zone_len = max(200, int(eff_len * (1.0 - _holdout_ratio)))
    embargo = int(EMBARGO_BARS.get(ctx.tf, 12))
    cpcv_bundle = build_cpcv_test_paths_with_fallback(cpcv_zone_len, embargo=embargo)
    unique_blocks = list_cpcv_block_ranges(cpcv_zone_len, cpcv_bundle[1], embargo=embargo)
    
    ctx.cpcv_block_slices = []
    for b_start, b_end in unique_blocks:
        s_start, s_end = max(0, b_start - 1), min(eff_len, b_end)
        aligned = _build_aligned_2d_from_prebuilt(prebuilt_full, ctx.symbols, s_start, s_end)
        ctx.cpcv_block_slices.append({"range": (b_start, b_end), "data": aligned})
    
    # Holdout Slice
    h_start, h_end = max(0, cpcv_zone_len - 1), eff_len
    ctx.holdout_slice = _build_aligned_2d_from_prebuilt(prebuilt_full, ctx.symbols, h_start, h_end)
    ctx.multi_alignment_info["cpcv_paths"] = cpcv_bundle[0]


def _suggest_ml_phase_d(trial: optuna.Trial) -> Dict[str, Any]:
    entry = float(trial.suggest_float("ENTRY_THRESHOLD", 0.10, 0.70))
    trail = float(trial.suggest_float("TRAILING_ACTIVATION_ATR", 0.3, 1.2))
    bayes_c = float(trial.suggest_float("BAYESIAN_C", 1.0, 30.0, log=True))
    kelly_s = float(trial.suggest_float("KELLY_SHRINKAGE", 0.05, 0.4))
    return {
        "ENTRY_THRESHOLD": entry,
        "TRAILING_ACTIVATION_ATR": trail,
        "BAYESIAN_C": bayes_c,
        "KELLY_SHRINKAGE": kelly_s,
    }


def build_ml_phase_d_params(trial_params: Dict[str, Any], tf: str) -> Dict[str, Any]:
    return _base_engine_params(trial_params, tf)


def _base_engine_params(ml: Dict[str, Any], tf: str) -> Dict[str, Any]:
    ks, bc = float(ml["KELLY_SHRINKAGE"]), float(ml["BAYESIAN_C"])
    fk_frac = float(np.clip(0.35 * ks * (1.0 + 1.0 / bc), 0.05, 0.6))
    lev = float(OPT_FUTURES_CONFIG.get("FUTURES_DISCOVERY_LEVERAGE", 5))
    rpt = 0.02
    if rpt * lev > 0.10:
        rpt = 0.10 / max(lev, 1e-9)
    return {
        "TIMEFRAME": tf, "SIGNAL_TYPE": "ML_CALIB_PROB", "REGIME_TYPE": "NONE",
        "SIZING_METHOD": "vol_target", "ENTRY_THRESHOLD": ml["ENTRY_THRESHOLD"],
        "LONG_TRAIL_MULT": float(ml["TRAILING_ACTIVATION_ATR"]) * 3.0,
        "SHORT_TRAIL_MULT": float(ml["TRAILING_ACTIVATION_ATR"]) * 3.0,
        "FK_FRACTION": fk_frac, "FK_EWMA_LAMBDA": 0.94, "FK_TARGET_VOL": 0.02,
        "FK_MAX_SIZE": 1.0, "FK_WINDOW": 60, "ATR_PERIOD": 14, "LONG_ATR_MULT": 2.5,
        "SHORT_ATR_MULT": 2.0, "LONG_SCALE_ATR_MULT": 2.5, "SHORT_TP_MULT": 2.0,
        "RISK_PER_TRADE": float(rpt), "MAX_EXPOSURE_PER_COIN": 1.0, "DD_SCALING_THRESHOLD": 0.15,
        "USE_COMPOUNDING": True,
        "LEVERAGE": int(lev),
    }


def objective_ml_phase_d(trial: optuna.Trial, ctx: MLPhaseDContext) -> tuple[float, float, float]:
    if ctx.cpcv_block_slices is None:
        precompute_ml_optimization_context(ctx)
    block_slices = ctx.cpcv_block_slices
    mai = ctx.multi_alignment_info
    if block_slices is None or mai is None:
        return 1e9, 1e9, 1e9

    ml = _suggest_ml_phase_d(trial)
    params = _base_engine_params(ml, ctx.tf)
    entry_thr = float(ml["ENTRY_THRESHOLD"])
    
    cfg = OPT_FUTURES_CONFIG
    min_seg_trades = int(cfg.get("FUTURES_MIN_TRADES_PER_CPCV_SEGMENT", 5))
    liq_mdd_thr = float(cfg.get("FUTURES_MAX_MDD", 25.0))

    block_results: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for block in block_slices:
        b_range, aligned = block["range"], block["data"]
        if not aligned:
            continue

        zkill = aligned.get("kill_signal", np.zeros_like(aligned["close"]))
        zfund = aligned.get("funding_rate_sum", np.zeros_like(aligned["close"]))
        lev_blk = aligned.get("dyn_leverage")
        if lev_blk is None or lev_blk.shape != aligned["close"].shape:
            lev_blk = np.full_like(
                aligned["close"], float(params["LEVERAGE"]), dtype=np.float64
            )
        else:
            lev_blk = np.maximum(lev_blk.astype(np.float64, copy=False), 1.0)
        b_trades_raw, b_bal, b_equity = backtest_portfolio_numba(
            aligned["close"],
            aligned["high"],
            aligned["low"],
            aligned["open"],
            aligned["entry_upper"],
            aligned["entry_lower"],
            aligned["trend_direction"],
            aligned["ml_calib_prob"],
            aligned["atr"],
            aligned["garch_kelly_f"],
            zkill,
            zfund,
            aligned["slot_rank_score"],
            float(FUTURES_INITIAL_BALANCE),
            lev_blk,
            TRADING_FEE_RATE,
            SLIPPAGE_RATE,
            float(params["RISK_PER_TRADE"]),
            float(params["LONG_ATR_MULT"]),
            float(params["LONG_TRAIL_MULT"]),
            float(params["SHORT_ATR_MULT"]),
            float(params["SHORT_TP_MULT"]),
            float(params["LONG_SCALE_ATR_MULT"]),
            float(params["SHORT_TRAIL_MULT"]),
            int(params.get("MAX_CONCURRENT_POSITIONS", 2)),
            float(params.get("MAX_EXPOSURE", 0.8)),
            float(params["MAX_EXPOSURE_PER_COIN"]),
            float(params["DD_SCALING_THRESHOLD"]),
            entry_thr,
        )
        
        n_tr = b_trades_raw.shape[0]
        if n_tr < min_seg_trades:
            block_results[b_range] = {"log_ret": -1.0, "mdd": 100.0, "trades": 0}
            continue

        mdd = float(calc_mdd_from_equity(b_equity))
        block_results[b_range] = {
            "log_ret": _log_tw_from_ret_pct(float((b_bal / FUTURES_INITIAL_BALANCE - 1.0) * 100.0)),
            "mdd": mdd, "trades": n_tr,
            "long_trades": int(np.sum(b_trades_raw[:, 3] == 1.0)),
            "short_trades": int(np.sum(b_trades_raw[:, 3] == -1.0)),
        }

    path_compound_raw_log_tw: List[float] = []
    path_seg_lists: List[List[float]] = []
    path_mdds: List[float] = []
    cpcv_paths = mai["cpcv_paths"]

    for path in cpcv_paths:
        p_log_ret, p_mdd, segs, valid_path = 0.0, 0.0, [], True
        for b_key in path:
            res = block_results.get(tuple(b_key))
            if res is None or res["mdd"] >= liq_mdd_thr or res["trades"] < min_seg_trades:
                valid_path = False
                break
            p_log_ret += res["log_ret"]
            segs.append(res["log_ret"])
            p_mdd = max(p_mdd, res["mdd"])
        if valid_path:
            path_compound_raw_log_tw.append(p_log_ret)
            path_seg_lists.append(segs)
            path_mdds.append(p_mdd)

    if not path_compound_raw_log_tw:
        return 1e9, 1e9, 1e9

    path_arr = np.asarray(path_compound_raw_log_tw, dtype=np.float64)
    worst_mdd = float(np.max(path_mdds))
    mean_log_growth = float(np.mean(path_arr))
    std_log_growth = float(np.std(path_arr))

    # Holdout (diagnostic only; not a primary NSGA objective)
    holdout_log_ret = 0.0
    if ctx.holdout_slice:
        h = ctx.holdout_slice
        hz = np.zeros_like(h["close"])
        lev_h = h.get("dyn_leverage")
        if lev_h is None or lev_h.shape != h["close"].shape:
            lev_h = np.full_like(h["close"], float(params["LEVERAGE"]), dtype=np.float64)
        else:
            lev_h = np.maximum(lev_h.astype(np.float64, copy=False), 1.0)
        _, h_bal, _ = backtest_portfolio_numba(
            h["close"],
            h["high"],
            h["low"],
            h["open"],
            h["entry_upper"],
            h["entry_lower"],
            h["trend_direction"],
            h["ml_calib_prob"],
            h["atr"],
            h["garch_kelly_f"],
            h.get("kill_signal", hz),
            h.get("funding_rate_sum", hz),
            h["slot_rank_score"],
            float(FUTURES_INITIAL_BALANCE),
            lev_h,
            TRADING_FEE_RATE,
            SLIPPAGE_RATE,
            float(params["RISK_PER_TRADE"]),
            float(params["LONG_ATR_MULT"]),
            float(params["LONG_TRAIL_MULT"]),
            float(params["SHORT_ATR_MULT"]),
            float(params["SHORT_TP_MULT"]),
            float(params["LONG_SCALE_ATR_MULT"]),
            float(params["SHORT_TRAIL_MULT"]),
            int(params.get("MAX_CONCURRENT_POSITIONS", 2)),
            float(params.get("MAX_EXPOSURE", 0.8)),
            float(params["MAX_EXPOSURE_PER_COIN"]),
            float(params["DD_SCALING_THRESHOLD"]),
            entry_thr,
        )
        h_ret_pct = float((h_bal / FUTURES_INITIAL_BALANCE - 1.0) * 100.0)
        holdout_log_ret = _log_tw_from_ret_pct(h_ret_pct)

    total_long = sum(r.get("long_trades", 0) for r in block_results.values())
    total_short = sum(r.get("short_trades", 0) for r in block_results.values())
    total_dir = total_long + total_short
    minority = float(min(total_long, total_short) / total_dir) if total_dir > 0 else 0.0

    sortinos = []
    for segs in path_seg_lists:
        a = np.asarray(segs)
        if a.size < 2:
            sortinos.append(float(np.mean(a)))
            continue
        neg = a[a < 0]
        ddev = float(np.std(neg, ddof=1)) if neg.size > 1 else float(np.std(a, ddof=1))
        sortinos.append(float(np.mean(a) / (ddev + 1e-9)))

    eff_ref_len = mai["eff_ref_len"]
    trial.set_user_attr("ml_mean_log_growth_cpcv", mean_log_growth)
    trial.set_user_attr("ml_worst_mdd_cpcv", worst_mdd)
    trial.set_user_attr("ml_std_log_growth_cpcv", std_log_growth)
    trial.set_user_attr("ml_holdout_log_ret", holdout_log_ret)
    trial.set_user_attr("ml_ls_minority_frac", minority)
    trial.set_user_attr("ml_mean_sortino_seg", float(np.mean(sortinos)) if sortinos else 0.0)
    trial.set_user_attr("gate1_eff_ref_len", eff_ref_len)
    # Minimize all three: (-E[log growth]) proxy, worst MDD, dispersion across CPCV paths
    return (-mean_log_growth, worst_mdd, std_log_growth)


def topsis_select_best(pareto_trials: List[FrozenTrial]) -> FrozenTrial:
    if not pareto_trials:
        raise ValueError("empty pareto_trials")
    if len(pareto_trials) == 1:
        return pareto_trials[0]
    n_dim = int(max(len(t.values) for t in pareto_trials))
    vals_list: list[list[float]] = []
    for t in pareto_trials:
        row = [float(x) for x in t.values]
        while len(row) < n_dim:
            row.append(0.0)
        vals_list.append(row[:n_dim])
    vals = np.asarray(vals_list, dtype=np.float64)
    vmin, vmax = vals.min(axis=0), vals.max(axis=0)
    norm = (vals - vmin) / np.where(vmax - vmin < 1e-12, 1.0, vmax - vmin)
    ideal: np.ndarray = np.zeros(n_dim, dtype=np.float64)
    nadir: np.ndarray = np.ones(n_dim, dtype=np.float64)
    d_pos = np.linalg.norm(norm - ideal, axis=1)
    d_neg = np.linalg.norm(norm - nadir, axis=1)
    return pareto_trials[int(np.argmax(d_neg / (d_pos + d_neg + 1e-12)))]


def check_hard_gates_ml(
    oos_result: Dict[str, Any],
    pbo_val: float,
    dsr_val: float,
    is_precision: float,
) -> bool:
    cfg = OPT_FUTURES_CONFIG
    pbo_ok = pbo_val < float(cfg.get("FUTURES_PBO_MAX", 0.45))
    dsr_ok = dsr_val > float(cfg.get("FUTURES_OBJECTIVE_DSR_TARGET", 1.5))
    wr = float(oos_result.get("win_rate", 0.0))
    wr_ok = wr >= is_precision * 0.85
    mdd_ok = abs(float(oos_result.get("mdd", 100.0))) < 20.0
    return bool(pbo_ok and dsr_ok and wr_ok and mdd_ok)
