"""NSGA-II Phase D objectives for ML pipeline (CPCV paths, O(1) slicing)."""

from __future__ import annotations

import logging
import math
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
from src.domain.futures.ml_pipeline.feature_engineering import HMM_SEMANTIC_PROB_COLUMNS
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
from src.domain.futures.signals.ml_calib_prob_futures import gate_ml_calib_prob_matrix
from src.domain.futures.strategies_futures import UltimateStrategy

_logger = logging.getLogger(__name__)

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


def _hmm_columns_for_dyn_leverage(df: pd.DataFrame) -> list[str]:
    sem = [c for c in HMM_SEMANTIC_PROB_COLUMNS if c in df.columns]
    if sem:
        return sem
    leg = sorted(
        (c for c in df.columns if str(c).startswith("hmm_prob_")),
        key=lambda x: int(str(x).split("_")[-1]),
    )
    return leg


def _inject_dyn_leverage_trimmed(trimmed_sig: pd.DataFrame, raw_full: pd.DataFrame) -> None:
    """Expected Kelly quality x entropy discount + crisis de-risk (tmp.md Layer 2.2)."""
    cfg = OPT_FUTURES_CONFIG
    hmm_cols = _hmm_columns_for_dyn_leverage(raw_full)
    base_lev = float(cfg.get("FUTURES_DISCOVERY_LEVERAGE", 5))
    crisis_thr = float(cfg.get("FUTURES_HMM_CRISIS_THRESHOLD", 0.6))
    if not hmm_cols:
        trimmed_sig["dyn_leverage"] = float(base_lev)
        return
    k = len(hmm_cols)
    p_mat = (
        raw_full[hmm_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0 / float(k))
        .to_numpy(dtype=np.float64)
    )
    close = raw_full["close"].astype(np.float64)
    r = np.log(close / close.shift(1).clip(lower=1e-12)).fillna(0.0).to_numpy(dtype=np.float64)
    g = (
        raw_full["gp_alpha_00"].fillna(0.0).to_numpy(dtype=np.float64)
        if "gp_alpha_00" in raw_full.columns
        else np.zeros(len(raw_full), dtype=np.float64)
    )
    factor_ret = g * r
    state_hard = np.argmax(p_mat, axis=1).astype(np.int64)
    k_vec: np.ndarray = np.zeros(k, dtype=np.float64)
    for s in range(k):
        sel = state_hard == s
        rr = factor_ret[sel]
        if np.sum(sel) > 30:
            k_vec[s] = float(np.clip(np.mean(rr) / (np.var(rr, ddof=1) + 1e-12), -1.0, 1.0))
    k_min, k_max = float(np.min(k_vec)), float(np.max(k_vec))
    k_qual = (k_vec - k_min) / (k_max - k_min + 1e-12)
    log_k = float(np.log(max(k, 2)))
    ent = -np.sum(p_mat * np.log(np.clip(p_mat, 1e-12, 1.0)), axis=1)
    ent_disc = np.clip(1.0 - (ent / log_k), 0.0, 1.0)
    exp_q = p_mat @ k_qual
    lev_min = 2.0
    lev_max = max(base_lev, lev_min)
    levs = lev_min + (lev_max - lev_min) * exp_q * ent_disc
    levs = np.clip(levs, lev_min, lev_max)
    if "hmm_prob_crisis" in raw_full.columns:
        pc = raw_full["hmm_prob_crisis"].fillna(0.0).to_numpy(dtype=np.float64)
        levs = np.where(pc > crisis_thr, 1.0, levs)
    trimmed_sig["dyn_leverage"] = levs.astype(np.float64, copy=False)


def precompute_ml_optimization_context(ctx: MLPhaseDContext) -> None:
    """Pre-align and pre-slice all data before Optuna starts to eliminate trial overhead."""
    # 1. Alignment & Baseline Signals
    pre_ml = {
        "ENTRY_THRESHOLD": 0.90,
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
    ctx.multi_alignment_info["cpcv_all_block_ranges"] = unique_blocks


def _gate1_dsr_from_path_log_tw(path_arr: np.ndarray, tf: str, stat_ref_len: float) -> float:
    """Deflated Sharpe (Bailey & López de Prado) on CPCV path log-TWR samples."""
    cfg = OPT_FUTURES_CONFIG
    if path_arr.size < 2:
        return 0.0
    m_pt = float(np.mean(path_arr))
    s_pt = float(np.std(path_arr, ddof=1))
    sharpe = m_pt / (s_pt + 1e-12)
    n_trials_opt = float(cfg.get("total_trials", 1000))
    hrs = int(tf.replace("h", "")) if tf.endswith("h") else 4
    t_samples = float(stat_ref_len) / (24.0 / float(hrs))
    sk = float(np.mean(((path_arr - m_pt) / (s_pt + 1e-12)) ** 3))
    ex_kurt = float(np.mean(((path_arr - m_pt) / (s_pt + 1e-12)) ** 4)) - 3.0
    sr_var_denom = max(
        1.0 - sk * sharpe + ((ex_kurt + 2.0) / 4.0) * sharpe**2,
        1e-12,
    )
    sr_bench = math.sqrt(2.0 * math.log(max(n_trials_opt, 2.0)))
    z_dsr = (sharpe - sr_bench) * math.sqrt(max(t_samples - 1.0, 1.0)) / math.sqrt(sr_var_denom)
    dsr_val = float(0.5 * (1.0 + math.erf(z_dsr / math.sqrt(2.0))))
    return float(min(0.99, max(0.0, dsr_val)))


def _suggest_ml_phase_d(trial: optuna.Trial) -> Dict[str, Any]:
    entry = float(trial.suggest_float("ENTRY_THRESHOLD", 0.80, 0.98))
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


def _pf_and_ev_cost_from_trades(all_trades: np.ndarray) -> tuple[float, float]:
    """PF = gross_win / |gross_loss|; EV/cost = |sum(pnl)| / sum(entry_fee + funding_fee)."""
    if all_trades.size == 0:
        return 1.0, 0.0
    pnl: np.ndarray = all_trades[:, 6].astype(np.float64, copy=False)
    gross_win = float(np.sum(pnl[pnl > 0.0]))
    gross_loss = float(np.sum(np.abs(pnl[pnl < 0.0])))
    avg_pf = gross_win / max(abs(gross_loss), 1e-9) if gross_loss != 0.0 else 1.0
    net_pnl = float(np.sum(pnl))
    fees = all_trades[:, 8].astype(np.float64, copy=False) + all_trades[:, 9].astype(
        np.float64, copy=False
    )
    total_fee = float(np.sum(fees))
    ev_cost_ratio = abs(net_pnl) / max(total_fee, 1e-9)
    return avg_pf, ev_cost_ratio


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
        "ENTRY_QUANTILE_WINDOW": int(OPT_FUTURES_CONFIG.get("ENTRY_QUANTILE_WINDOW", 240)),
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
    entry_q = float(ml["ENTRY_THRESHOLD"])
    cfg = OPT_FUTURES_CONFIG
    win_q = int(cfg.get("ENTRY_QUANTILE_WINDOW", 240))
    entry_numba = float(cfg.get("FUTURES_ENTRY_NUMBA_THRESHOLD", 0.5))

    min_seg_trades = int(cfg.get("FUTURES_MIN_TRADES_PER_CPCV_SEGMENT", 5))
    liq_mdd_thr = float(cfg.get("FUTURES_MAX_MDD", 25.0))
    min_trades_target = int(cfg.get("FUTURES_MIN_TRADES_TARGET", 25))

    block_results: Dict[Tuple[int, int], Dict[str, Any]] = {}
    all_trades_chunks: List[np.ndarray] = []
    first_bt_done = False
    for block in block_slices:
        b_range, aligned = block["range"], block["data"]
        if not aligned:
            continue

        if "kill_signal_cached" not in aligned:
            zkill = aligned.get("kill_signal")
            if zkill is None:
                zkill = np.zeros_like(aligned["close"])
            aligned["kill_signal_cached"] = zkill
        zkill = aligned["kill_signal_cached"]

        if "funding_rate_sum_cached" not in aligned:
            zfund = aligned.get("funding_rate_sum")
            if zfund is None:
                zfund = np.zeros_like(aligned["close"])
            aligned["funding_rate_sum_cached"] = zfund
        zfund = aligned["funding_rate_sum_cached"]

        if "dyn_leverage_cached" not in aligned:
            lev_blk = aligned.get("dyn_leverage")
            if lev_blk is None or lev_blk.shape != aligned["close"].shape:
                lev_blk = np.full_like(
                    aligned["close"], float(params["LEVERAGE"]), dtype=np.float64
                )
            else:
                lev_blk = np.maximum(lev_blk.astype(np.float64, copy=False), 1.0)
            aligned["dyn_leverage_cached"] = lev_blk
        lev_blk = aligned["dyn_leverage_cached"]

        strength_g = gate_ml_calib_prob_matrix(aligned["ml_calib_prob"], entry_q, win_q)
        b_trades_raw, b_bal, b_equity = backtest_portfolio_numba(
            aligned["close"],
            aligned["high"],
            aligned["low"],
            aligned["open"],
            aligned["entry_upper"],
            aligned["entry_lower"],
            aligned["trend_direction"],
            strength_g,
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
            entry_numba,
        )

        n_tr = int(b_trades_raw.shape[0])
        all_trades_chunks.append(b_trades_raw)
        if not first_bt_done:
            first_bt_done = True
            if n_tr == 0:
                raise optuna.TrialPruned()
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

    all_trades = (
        np.vstack(all_trades_chunks)
        if all_trades_chunks
        else np.zeros((0, 10), dtype=np.float64)
    )
    avg_pf_agg, ev_cost_ratio_agg = _pf_and_ev_cost_from_trades(all_trades)
    trade_counts = [float(v["trades"]) for v in block_results.values()]
    avg_trades_agg = float(np.mean(trade_counts)) if trade_counts else 0.0
    worst_mdd_blocks = float(max((r["mdd"] for r in block_results.values()), default=100.0))

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

    path_arr = (
        np.asarray(path_compound_raw_log_tw, dtype=np.float64)
        if path_compound_raw_log_tw
        else np.asarray([], dtype=np.float64)
    )
    worst_mdd = float(np.max(path_mdds)) if path_mdds else worst_mdd_blocks
    mean_log_growth = float(np.mean(path_arr)) if path_arr.size > 0 else 0.0
    if path_arr.size > 1:
        std_log_growth = float(np.std(path_arr))
    elif path_arr.size == 1:
        std_log_growth = 0.0
    else:
        std_log_growth = 1e9

    eff_ref = float(mai["eff_ref_len"])
    trial.set_user_attr(
        "cpcv_path_oos_log_tw",
        [float(x) for x in path_compound_raw_log_tw],
    )
    trial.set_user_attr("gate1_dsr", _gate1_dsr_from_path_log_tw(path_arr, ctx.tf, eff_ref))

    # Holdout (diagnostic only; not a primary NSGA objective)
    holdout_log_ret = 0.0
    if ctx.holdout_slice:
        h = ctx.holdout_slice

        if "kill_signal_cached" not in h:
            hz = h.get("kill_signal")
            if hz is None:
                hz = np.zeros_like(h["close"])
            h["kill_signal_cached"] = hz
        hz_cached = h["kill_signal_cached"]

        if "funding_rate_sum_cached" not in h:
            zfund_h = h.get("funding_rate_sum")
            if zfund_h is None:
                zfund_h = np.zeros_like(h["close"])
            h["funding_rate_sum_cached"] = zfund_h
        zfund_cached = h["funding_rate_sum_cached"]

        if "dyn_leverage_cached" not in h:
            lev_h = h.get("dyn_leverage")
            if lev_h is None or lev_h.shape != h["close"].shape:
                lev_h = np.full_like(h["close"], float(params["LEVERAGE"]), dtype=np.float64)
            else:
                lev_h = np.maximum(lev_h.astype(np.float64, copy=False), 1.0)
            h["dyn_leverage_cached"] = lev_h
        lev_h_cached = h["dyn_leverage_cached"]

        h_g = gate_ml_calib_prob_matrix(h["ml_calib_prob"], entry_q, win_q)
        _, h_bal, _ = backtest_portfolio_numba(
            h["close"],
            h["high"],
            h["low"],
            h["open"],
            h["entry_upper"],
            h["entry_lower"],
            h["trend_direction"],
            h_g,
            h["atr"],
            h["garch_kelly_f"],
            hz_cached,
            zfund_cached,
            h["slot_rank_score"],
            float(FUTURES_INITIAL_BALANCE),
            lev_h_cached,
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
            entry_numba,
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

    trade_shortfall = max(0.0, float(min_trades_target) - avg_trades_agg) / float(min_trades_target)
    obj0_penalized = -mean_log_growth + 2.0 * trade_shortfall
    if not path_compound_raw_log_tw:
        obj0_penalized = 1e9 + 2.0 * trade_shortfall
        std_log_growth = 1e9

    trial.set_user_attr("ml_mean_log_growth_cpcv", mean_log_growth)
    trial.set_user_attr("ml_worst_mdd_cpcv", worst_mdd)
    trial.set_user_attr("ml_std_log_growth_cpcv", std_log_growth)
    trial.set_user_attr("ml_holdout_log_ret", holdout_log_ret)
    trial.set_user_attr("ml_ls_minority_frac", minority)
    trial.set_user_attr("ml_mean_sortino_seg", float(np.mean(sortinos)) if sortinos else 0.0)
    trial.set_user_attr("gate1_eff_ref_len", eff_ref_len)
    trial.set_user_attr("avg_trades", avg_trades_agg)
    trial.set_user_attr("avg_pf", avg_pf_agg)
    trial.set_user_attr("avg_mdd", worst_mdd)
    trial.set_user_attr("long_short_ratio", minority)
    trial.set_user_attr("ev_cost_ratio", ev_cost_ratio_agg)

    return (obj0_penalized, worst_mdd, std_log_growth)


def select_best_trial_by_holdout_log_ret(trials: List[FrozenTrial]) -> FrozenTrial:
    """Fallback when constraint-feasible set is empty: maximize ml_holdout_log_ret."""
    if not trials:
        raise ValueError("empty trials")
    return max(
        trials,
        key=lambda t: float(t.user_attrs.get("ml_holdout_log_ret", float("-inf"))),
    )


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
    dsr_floor = float(cfg.get("FUTURES_ML_GATE1_DSR_MIN", 0.20))
    dsr_ok = dsr_val >= dsr_floor
    wr_pct = float(oos_result.get("win_rate_pct", oos_result.get("win_rate", 0.0)))
    wr_frac = wr_pct / 100.0 if wr_pct > 1.0 else wr_pct
    wr_ok = wr_frac >= is_precision * 0.85
    mdd_v = float(oos_result.get("mdd_pct", oos_result.get("mdd", 100.0)))
    mdd_ok = abs(mdd_v) < 20.0
    return bool(pbo_ok and dsr_ok and wr_ok and mdd_ok)
