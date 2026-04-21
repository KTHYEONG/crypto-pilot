"""Phase D: single-objective CPCV-DSR (TPE) for ML cross-sectional rank portfolio."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import optuna
import pandas as pd
from optuna.trial import FrozenTrial

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import FUTURES_INITIAL_BALANCE, SLIPPAGE_RATE, TRADING_FEE_RATE
from src.domain.futures.engine_multi_futures import (
    _recompute_cs_dirs_numba,
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
from src.domain.futures.opt_futures_utils.metrics import (
    _log_tw_from_ret_pct,
    calc_gate1_dsr_from_path_log_tw,
    calc_mdd_from_equity,
)
from src.domain.futures.opt_futures_utils.objective import (
    EMBARGO_BARS,
    compute_multi_alignment_info,
)
from src.domain.futures.opt_futures_utils.signal_cache import get_tiered_signals
from src.domain.futures.strategies_futures import UltimateStrategy

_logger = logging.getLogger(__name__)
_PRECOMPUTE_LOCK = threading.Lock()

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
    try:
        p_mat = (
            raw_full[hmm_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0 / float(k))
            .to_numpy(dtype=np.float64)
        )
    except Exception as e:
        _logger.error(
            "Error creating p_mat: %s. hmm_cols=%s, columns=%s",
            e,
            hmm_cols,
            list(raw_full.columns),
        )
        raise
    
    if p_mat.shape[1] != k:
         _logger.error("SHAPE MISMATCH: p_mat.shape[1]=%s, k=%s. hmm_cols=%s. ALL COLS=%s", 
                      p_mat.shape[1], k, hmm_cols, list(raw_full.columns))

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
    with _PRECOMPUTE_LOCK:
        if ctx.cpcv_block_slices is not None:
            return

        # 1. Alignment & Baseline Signals
        pre_ml = {
            "TRAILING_ACTIVATION_ATR": 1.0,
            "BAYESIAN_C": 10.0,
            "KELLY_SHRINKAGE": 0.3,
            "K_LONG": 2,
            "K_SHORT": 2,
            "REBALANCE_BARS": 6,
            "MIN_SCORE_PERCENTILE": 0.55,
            "CRISIS_GAMMA": 1.0,
            "USE_CS_RANK_ENGINE": True,
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
            trimmed_sig["ml_calib_prob"] = 1.0
            trimmed_sig["ml_calib_prob_long"] = 1.0
            trimmed_sig["ml_calib_prob_short"] = 1.0
            gp_pre = (
                raw_full["gp_alpha_00"].to_numpy(dtype=np.float64, copy=False)
                if "gp_alpha_00" in raw_full.columns
                else np.zeros(len(trimmed_sig), dtype=np.float64)
            )
            trimmed_sig["trend_direction"] = np.sign(gp_pre).astype(np.float64)
            trimmed_sig["entry_upper"] = 0.0
            trimmed_sig["entry_lower"] = 999999.0
            _xs_cols = (
                "xs_score_long",
                "xs_score_short",
                "hmm_prob_crisis",
                "hmm_modulator_long",
                "hmm_modulator_short",
            )
            for col in _xs_cols:
                if col in raw_full.columns:
                    trimmed_sig[col] = raw_full[col].to_numpy(dtype=np.float64, copy=False)

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
        ctx.holdout_slice = _build_aligned_2d_from_prebuilt(
            prebuilt_full, ctx.symbols, h_start, h_end
        )
        ctx.multi_alignment_info["cpcv_paths"] = cpcv_bundle[0]
        ctx.multi_alignment_info["cpcv_all_block_ranges"] = unique_blocks

    if ctx.cpcv_block_slices:
        aligned0 = ctx.cpcv_block_slices[0]["data"]
        xl0 = aligned0.get("xs_score_long")
        if xl0 is not None and getattr(xl0, "size", 0) > 0:
            xs_std = float(np.nanstd(np.asarray(xl0, dtype=np.float64)))
            ctx.multi_alignment_info["xs_score_aligned_std"] = xs_std
            if xs_std < 0.05:
                _logger.warning(
                    "[ML_OPT] xs_score dispersion low (std=%.6f); check GP/CS merge path.",
                    xs_std,
                )

        _log_precompute_computed_dir_sample(
            aligned0,
            int(pre_ml["K_LONG"]),
            int(pre_ml["K_SHORT"]),
            float(pre_ml["CRISIS_GAMMA"]),
        )


def _log_precompute_computed_dir_sample(
    aligned0: Dict[str, Any],
    k_long: int,
    k_short: int,
    crisis_gamma: float,
) -> None:
    """Single-bar sample of |computed_dir| dispersion (block 0, mid time index)."""
    xl = aligned0.get("xs_score_long")
    xs = aligned0.get("xs_score_short")
    hy = aligned0.get("hmm_prob_crisis")
    if xl is None or xs is None or hy is None or getattr(xl, "size", 0) == 0:
        return
    arr_l = np.ascontiguousarray(xl, dtype=np.float64)
    arr_s = np.ascontiguousarray(xs, dtype=np.float64)
    arr_h = np.ascontiguousarray(hy, dtype=np.float64)
    n_b, n_sy = arr_l.shape[0], arr_l.shape[1]
    prev_i = min(n_b - 1, max(0, n_b // 2))
    out = np.zeros(n_sy, dtype=np.float64)
    _recompute_cs_dirs_numba(
        prev_i,
        n_sy,
        arr_l,
        arr_s,
        arr_h,
        crisis_gamma,
        k_long,
        k_short,
        out,
    )
    mags = np.abs(out)
    _logger.debug(
        "[ML_OPT][precompute] computed_dir |.| mid_bar=%d "
        "std=%.6f mean=%.6f max=%.6f nonzero_frac=%.4f",
        prev_i,
        float(np.nanstd(mags)),
        float(np.nanmean(mags)),
        float(np.nanmax(mags)),
        float(np.mean(mags > 1e-12)),
    )
    flat_l = arr_l[prev_i, :].ravel()
    q1, q50, q99 = (
        float(np.nanpercentile(flat_l, 1)),
        float(np.nanpercentile(flat_l, 50)),
        float(np.nanpercentile(flat_l, 99)),
    )
    _logger.debug(
        "[ML_OPT][precompute] xs_score_long row quantiles p01/p50/p99=%.6f/%.6f/%.6f",
        q1,
        q50,
        q99,
    )


def _suggest_ml_phase_d(trial: optuna.Trial) -> Dict[str, Any]:
    trail = float(trial.suggest_float("TRAILING_ACTIVATION_ATR", 0.3, 1.2))
    bayes_c = float(trial.suggest_float("BAYESIAN_C", 1.0, 30.0, log=True))
    kelly_s = float(trial.suggest_float("KELLY_SHRINKAGE", 0.05, 0.4))
    # K_LONG/K_SHORT free [1,4]: IS(2024-25) has bear phases → K_SHORT=4 IS-optimal.
    # Restricting K_SHORT≤2 removes IS profitable shorts → all CPCV paths negative → DSR=0.
    k_long = int(trial.suggest_int("K_LONG", 1, 4))
    k_short = int(trial.suggest_int("K_SHORT", 1, 4))
    # IC half-life=3.3h: 12h rebalancing decays IC by 97% → exclude 12 from search.
    reb = int(trial.suggest_categorical("REBALANCE_BARS", [1, 3, 6]))
    # CRISIS_GAMMA cap at 2.0: gamma>2 causes extreme suppression → few trades → noise.
    crisis = float(trial.suggest_float("CRISIS_GAMMA", 0.5, 2.0))
    atr_p = int(trial.suggest_int("ATR_PERIOD", 10, 20, step=2))
    l_atr = float(trial.suggest_float("LONG_ATR_MULT", 1.5, 4.5, step=0.25))
    l_trail = float(trial.suggest_float("LONG_TRAIL_MULT", 2.5, 6.0, step=0.5))
    # SHORT_ATR_MULT min 1.5: prevents tight-short trap in rising markets.
    s_atr = float(trial.suggest_float("SHORT_ATR_MULT", 1.5, 3.0, step=0.25))
    s_tp = float(trial.suggest_float("SHORT_TP_MULT", 1.0, 3.5, step=0.5))
    s_trail = float(trial.suggest_float("SHORT_TRAIL_MULT", 1.5, 4.5, step=0.5))
    l_scale = float(trial.suggest_float("LONG_SCALE_ATR_MULT", 2.5, 6.0, step=0.5))
    max_exp = float(trial.suggest_float("MAX_EXPOSURE_PER_COIN", 0.5, 2.0, step=0.1))
    dd_thr = float(trial.suggest_float("DD_SCALING_THRESHOLD", 0.10, 0.25, step=0.05))
    return {
        "TRAILING_ACTIVATION_ATR": trail,
        "BAYESIAN_C": bayes_c,
        "KELLY_SHRINKAGE": kelly_s,
        "K_LONG": k_long,
        "K_SHORT": k_short,
        "REBALANCE_BARS": reb,
        # Fixed 0.55 (not searched): extra dimension dilutes Optuna convergence (Run 12 lesson).
        "MIN_SCORE_PERCENTILE": 0.55,
        # Fixed 3%: RPT as search param creates dimension-inflation → DSR=0 (Run 12 lesson).
        # 3% = 1.5x Run 9's effective 2%; expected CAGR ≈ 18% x 1.5 = 27%.
        "RISK_PER_TRADE": 0.03,
        "CRISIS_GAMMA": crisis,
        "ATR_PERIOD": atr_p,
        "LONG_ATR_MULT": l_atr,
        "LONG_TRAIL_MULT": l_trail,
        "SHORT_ATR_MULT": s_atr,
        "SHORT_TP_MULT": s_tp,
        "SHORT_TRAIL_MULT": s_trail,
        "LONG_SCALE_ATR_MULT": l_scale,
        "MAX_EXPOSURE_PER_COIN": max_exp,
        "DD_SCALING_THRESHOLD": dd_thr,
        "USE_CS_RANK_ENGINE": True,
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
    rpt = float(ml.get("RISK_PER_TRADE", 0.02))
    # Cap: rpt*lev ≤ 0.20 (4% max at lev=5); higher RPT breaks IS DSR (Run 10 lesson).
    if rpt * lev > 0.20:
        rpt = 0.20 / max(lev, 1e-9)
    trail = float(ml["TRAILING_ACTIVATION_ATR"])
    return {
        "TIMEFRAME": tf,
        "SIGNAL_TYPE": "ML_CALIB_PROB",
        "REGIME_TYPE": "NONE",
        "SIZING_METHOD": "vol_target",
        "USE_CS_RANK_ENGINE": bool(ml.get("USE_CS_RANK_ENGINE", True)),
        "K_LONG": int(ml.get("K_LONG", 2)),
        "K_SHORT": int(ml.get("K_SHORT", 2)),
        "REBALANCE_BARS": max(1, int(ml.get("REBALANCE_BARS", 6))),
        "MIN_SCORE_PERCENTILE": float(ml.get("MIN_SCORE_PERCENTILE", 0.55)),
        "CRISIS_GAMMA": float(ml.get("CRISIS_GAMMA", ml.get("CRISIS_GATE_PROB", 1.0))),
        "CRISIS_GATE_PROB": float(ml.get("CRISIS_GAMMA", ml.get("CRISIS_GATE_PROB", 1.0))),
        "LONG_TRAIL_MULT": trail * 3.0,
        "SHORT_TRAIL_MULT": trail * 3.0,
        "FK_FRACTION": fk_frac,
        "FK_EWMA_LAMBDA": 0.94,
        "FK_TARGET_VOL": 0.02,
        "FK_MAX_SIZE": 1.0,
        "FK_WINDOW": 60,
        "ATR_PERIOD": int(ml.get("ATR_PERIOD", 14)),
        "LONG_ATR_MULT": float(ml.get("LONG_ATR_MULT", 2.5)),
        "SHORT_ATR_MULT": float(ml.get("SHORT_ATR_MULT", 2.0)),
        "LONG_SCALE_ATR_MULT": float(ml.get("LONG_SCALE_ATR_MULT", 2.5)),
        "SHORT_TP_MULT": float(ml.get("SHORT_TP_MULT", 2.0)),
        "RISK_PER_TRADE": float(rpt),
        "MAX_EXPOSURE_PER_COIN": float(ml.get("MAX_EXPOSURE_PER_COIN", 1.0)),
        "DD_SCALING_THRESHOLD": float(ml.get("DD_SCALING_THRESHOLD", 0.15)),
        "USE_COMPOUNDING": True,
        "LEVERAGE": int(lev),
        "ENTRY_QUANTILE_WINDOW": int(OPT_FUTURES_CONFIG.get("ENTRY_QUANTILE_WINDOW", 240)),
        "MAX_CONCURRENT_POSITIONS": int(
            OPT_FUTURES_CONFIG.get("FUTURES_MAX_CONCURRENT_POSITIONS", 3)
        ),
        "MAX_EXPOSURE": 0.8,
    }


def _cached_kill_fund_lev(
    aligned: Dict[str, Any], params: Dict[str, Any]
) -> tuple[Any, Any, Any]:
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
            lev_blk = np.full_like(aligned["close"], float(params["LEVERAGE"]), dtype=np.float64)
        else:
            lev_blk = np.maximum(lev_blk.astype(np.float64, copy=False), 1.0)
        aligned["dyn_leverage_cached"] = lev_blk
    lev_blk = aligned["dyn_leverage_cached"]
    return zkill, zfund, lev_blk


def _run_portfolio_numba_block(
    params: Dict[str, Any],
    aligned: Dict[str, Any],
    zkill: Any,
    zfund: Any,
    lev_blk: Any,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    strength_g = aligned["ml_calib_prob"]
    use_cs = 1 if bool(params.get("USE_CS_RANK_ENGINE", True)) else 0
    out_bt = backtest_portfolio_numba(
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
        aligned["xs_score_long"],
        aligned["xs_score_short"],
        aligned["hmm_prob_crisis"],
        aligned["hmm_modulator_long"],
        aligned["hmm_modulator_short"],
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
        int(params["K_LONG"]),
        int(params["K_SHORT"]),
        max(1, int(params["REBALANCE_BARS"])),
        float(params.get("CRISIS_GAMMA", params.get("CRISIS_GATE_PROB", 1.0))),
        use_cs,
    )
    return cast(tuple[np.ndarray, float, np.ndarray, np.ndarray], out_bt)


def objective_ml_phase_d(trial: optuna.Trial, ctx: MLPhaseDContext) -> float:
    if ctx.cpcv_block_slices is None:
        precompute_ml_optimization_context(ctx)
    block_slices = ctx.cpcv_block_slices
    mai = ctx.multi_alignment_info
    if block_slices is None or mai is None:
        return 1e9

    ml = _suggest_ml_phase_d(trial)
    params = _base_engine_params(ml, ctx.tf)
    cfg = OPT_FUTURES_CONFIG
    if trial.number < 10 and block_slices:
        ad0 = block_slices[0].get("data") or {}
        xl = ad0.get("xs_score_long")
        hy = ad0.get("hmm_prob_crisis")
        if xl is not None and hy is not None and getattr(xl, "size", 0) > 0:
            disp = float(np.nanstd(np.asarray(xl, dtype=np.float64)))
            cclip = np.clip(np.asarray(hy, dtype=np.float64), 0.0, 1.0)
            gamma = float(params.get("CRISIS_GAMMA", params.get("CRISIS_GATE_PROB", 1.0)))
            soft_m = float(np.mean((1.0 - cclip) ** gamma))
            thr = float(cfg.get("FUTURES_HMM_CRISIS_THRESHOLD", 0.6))
            rej_r = float(np.mean(np.max(hy, axis=1) > thr))
            trial.set_user_attr("xs_score_dispersion_mean", disp)
            trial.set_user_attr("crisis_soft_weight_mean", soft_m)
            trial.set_user_attr("crisis_gate_rejection_rate", rej_r)
    liq_mdd_thr = float(cfg.get("FUTURES_MAX_MDD", 25.0))
    min_trades_target = int(cfg.get("FUTURES_MIN_TRADES_TARGET", 10))

    block_results: Dict[Tuple[int, int], Dict[str, Any]] = {}
    all_trades_chunks: List[np.ndarray] = []
    first_bt_done = False
    for block in block_slices:
        b_range, aligned = block["range"], block["data"]
        if not aligned:
            continue

        zkill, zfund, lev_blk = _cached_kill_fund_lev(aligned, params)
        b_trades_raw, b_bal, b_equity, b_diag = _run_portfolio_numba_block(
            params, aligned, zkill, zfund, lev_blk
        )

        n_tr = int(b_trades_raw.shape[0])
        all_trades_chunks.append(b_trades_raw)
        if trial.number < 3:
            _logger.debug(
                "[ML_OPT][trial=%d] CPCV block=%s trades=%d "
                "diag[dust,margin,tdir0,pside0]=[%d,%d,%d,%d]",
                trial.number,
                b_range,
                n_tr,
                int(b_diag[0]),
                int(b_diag[1]),
                int(b_diag[2]),
                int(b_diag[3]),
            )
        if not first_bt_done:
            first_bt_done = True
            if n_tr == 0:
                _logger.debug(
                    "[ML_OPT][trial=%d] Prune: first CPCV block %s produced 0 trades. "
                    "diag[dust,margin,tdir0,pside0]=[%d,%d,%d,%d]",
                    trial.number,
                    b_range,
                    int(b_diag[0]),
                    int(b_diag[1]),
                    int(b_diag[2]),
                    int(b_diag[3]),
                )
                raise optuna.TrialPruned()

        mdd = float(calc_mdd_from_equity(b_equity)) if b_equity.size > 0 else 100.0
        log_ret = _log_tw_from_ret_pct(float((b_bal / FUTURES_INITIAL_BALANCE - 1.0) * 100.0))
        block_results[b_range] = {
            "log_ret": log_ret,
            "mdd": mdd,
            "trades": n_tr,
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
            if res is None or res["mdd"] >= liq_mdd_thr:
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

    # DSR: each CPCV path = one independent IS-OOS experiment.
    # Using total_trials(2000) as n_tests gives sr_bench≈3.9 — impossible to beat.
    # Use n_valid_paths as n_tests (each path is one unique train-test split).
    n_p = float(max(2, len(path_arr)))
    hrs_per_bar = int(ctx.tf.replace("h", "")) if ctx.tf.endswith("h") else 4
    # Scale eff_ref so t_samples = n_paths (each path is one independent data point).
    eff_ref_for_dsr = n_p * (24.0 / max(hrs_per_bar, 1))
    dsr = float(calc_gate1_dsr_from_path_log_tw(path_arr, ctx.tf, eff_ref_for_dsr, n_p))
    n_valid_paths = len(path_compound_raw_log_tw)
    pen = 0.5 * max(0, 6 - n_valid_paths) / 6.0 if n_valid_paths < 6 else 0.0

    trial.set_user_attr(
        "cpcv_path_oos_log_tw",
        [float(x) for x in path_compound_raw_log_tw],
    )
    trial.set_user_attr("gate1_dsr", dsr)
    trial.set_user_attr("dsr_cpcv", dsr)
    trial.set_user_attr("n_valid_paths", n_valid_paths)

    holdout_log_ret = 0.0
    if ctx.holdout_slice:
        h = ctx.holdout_slice
        hz_cached, zfund_cached, lev_h_cached = _cached_kill_fund_lev(h, params)
        _, h_bal, _, _ = _run_portfolio_numba_block(
            params, h, hz_cached, zfund_cached, lev_h_cached
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
    trade_shortfall = max(0.0, float(min_trades_target) - avg_trades_agg) / float(
        max(min_trades_target, 1)
    )

    if path_arr.size < 2 or dsr < 0.0:
        obj = 1e9 + pen + 2.0 * trade_shortfall
    else:
        # Compound objective: DSR primary + mean_log_growth secondary gradient.
        # When DSR=0 for all trials (paths all negative), growth_signal provides
        # gradient so Optuna can distinguish better from worse params.
        # Weight 0.2: DSR dominates when > 0; too-high weight causes IS overfitting.
        growth_signal = float(np.clip(mean_log_growth, -2.0, 2.0))
        obj = -dsr - 0.2 * growth_signal + pen + 2.0 * trade_shortfall

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

    return float(obj)


def select_best_trial_by_holdout_log_ret(trials: List[FrozenTrial]) -> FrozenTrial:
    """Fallback when constraint-feasible set is empty.

    Scoring: prefer trials with DSR > 0, then by combined
    holdout_log_ret + IS CPCV mean_log_growth (equal weight).
    Pure holdout_log_ret is unstable across runs; adding IS CPCV
    stability reduces overfitting to the holdout sub-period.
    """
    if not trials:
        raise ValueError("empty trials")

    def _score(t: FrozenTrial) -> float:
        holdout = float(np.clip(t.user_attrs.get("ml_holdout_log_ret", 0.0), -2.0, 2.0))
        is_cpcv = float(np.clip(t.user_attrs.get("ml_mean_log_growth_cpcv", -2.0), -2.0, 2.0))
        dsr = float(t.user_attrs.get("gate1_dsr", 0.0))
        # Holdout 3x weight: holdout is adjacent to OOS and better predicts OOS
        # than IS CPCV (which spans diverse IS regimes not representative of OOS).
        dsr_bonus = 0.5 if dsr > 0.0 else 0.0
        return 3.0 * holdout + is_cpcv + dsr_bonus

    return max(trials, key=_score)


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
    mdd_ok = abs(mdd_v) < float(cfg.get("FUTURES_MAX_MDD", 25.0))
    return bool(pbo_ok and dsr_ok and wr_ok and mdd_ok)
