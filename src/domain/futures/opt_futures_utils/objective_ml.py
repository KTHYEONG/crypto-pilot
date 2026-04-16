"""NSGA-II Phase D objectives for ML pipeline (CPCV stability vs quality)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.trial import FrozenTrial

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import FUTURES_INITIAL_BALANCE, SLIPPAGE_RATE, TRADING_FEE_RATE
from src.domain.futures.engine_multi_futures import PortfolioBacktestEngineFast
from src.domain.futures.opt_futures_utils.cv_utils import list_cpcv_block_ranges
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

ML_PHASE_D_PARAM_SPACE: Dict[str, Dict[str, Any]] = {
    "ENTRY_THRESHOLD": {"type": "float", "low": 0.70, "high": 0.95},
    "TRAILING_ACTIVATION_ATR": {"type": "float", "low": 0.5, "high": 1.5},
    "BAYESIAN_C": {"type": "float", "low": 1.0, "high": 50.0, "log": True},
    "KELLY_SHRINKAGE": {"type": "float", "low": 0.1, "high": 0.5},
}


@dataclass
class MLPhaseDContext:
    data_maps: Dict[str, Dict[str, Any]]
    symbols: List[str]
    tf: str
    seed: int = 42


def _suggest_ml_phase_d(trial: optuna.Trial) -> Dict[str, Any]:
    entry = float(
        trial.suggest_float("ENTRY_THRESHOLD", 0.70, 0.95),
    )
    trail = float(trial.suggest_float("TRAILING_ACTIVATION_ATR", 0.5, 1.5))
    bayes_c = float(trial.suggest_float("BAYESIAN_C", 1.0, 50.0, log=True))
    kelly_s = float(trial.suggest_float("KELLY_SHRINKAGE", 0.1, 0.5))
    return {
        "ENTRY_THRESHOLD": entry,
        "TRAILING_ACTIVATION_ATR": trail,
        "BAYESIAN_C": bayes_c,
        "KELLY_SHRINKAGE": kelly_s,
    }


def build_ml_phase_d_params(trial_params: Dict[str, Any], tf: str) -> Dict[str, Any]:
    """Full engine params from NSGA-II trial params (ENTRY_THRESHOLD, TRAILING_*, ...)."""
    return _base_engine_params(trial_params, tf)


def _base_engine_params(ml: Dict[str, Any], tf: str) -> Dict[str, Any]:
    ks, bc = float(ml["KELLY_SHRINKAGE"]), float(ml["BAYESIAN_C"])
    fk_frac = float(np.clip(0.35 * ks * (1.0 + 1.0 / bc), 0.05, 0.6))
    return {
        "TIMEFRAME": tf,
        "SIGNAL_TYPE": "ML_CALIB_PROB",
        "REGIME_TYPE": "EMA_ATR",
        "SIZING_METHOD": "fk_dynamic",
        "ENTRY_THRESHOLD": ml["ENTRY_THRESHOLD"],
        "LONG_TRAIL_MULT": float(ml["TRAILING_ACTIVATION_ATR"]) * 3.0,
        "SHORT_TRAIL_MULT": float(ml["TRAILING_ACTIVATION_ATR"]) * 3.0,
        "FK_FRACTION": fk_frac,
        "FK_EWMA_LAMBDA": 0.94,
        "FK_TARGET_VOL": 0.02,
        "FK_MAX_SIZE": 1.0,
        "FK_WINDOW": 60,
        "ATR_PERIOD": 14,
        "LONG_ATR_MULT": 2.5,
        "SHORT_ATR_MULT": 2.0,
        "LONG_SCALE_ATR_MULT": 2.5,
        "SHORT_TP_MULT": 2.0,
        "RISK_PER_TRADE": 0.02,
        "MAX_EXPOSURE_PER_COIN": 1.0,
        "DD_SCALING_THRESHOLD": 0.15,
        "USE_COMPOUNDING": True,
        "LEVERAGE": int(OPT_FUTURES_CONFIG.get("FUTURES_DISCOVERY_LEVERAGE", 5)),
    }


def objective_ml_phase_d(trial: optuna.Trial, ctx: MLPhaseDContext) -> tuple[float, float]:
    """
    Returns (-f1_stability, -f2_quality) for Optuna minimize both objectives.

    f1_stability = CPCV_P25_Log_TW / max(Worst_Segment_MDD, 0.001)
    f2_quality = mean(path Sortino proxy on CPCV path log-TW segments)
    """
    ml = _suggest_ml_phase_d(trial)
    params = _base_engine_params(ml, ctx.tf)
    strategy = UltimateStrategy(name="MLPhaseD", params=params)

    multi_alignment_info = compute_multi_alignment_info(
        ctx.data_maps,
        ctx.symbols,
        ctx.tf,
        int(EMBARGO_BARS.get(ctx.tf, 12)),
    )
    if multi_alignment_info is None:
        return 1e9, 1e9

    full_signal_dfs: Dict[str, pd.DataFrame] = {}
    for sym in ctx.symbols:
        raw = ctx.data_maps.get(sym, {}).get(ctx.tf)
        if raw is None or raw.empty:
            continue
        if "ml_calib_prob" not in raw.columns:
            _logger.warning("Missing ml_calib_prob for %s; objective degraded.", sym)
            continue
        full_signal_dfs[sym] = get_tiered_signals(params, sym, ctx.tf, raw, strategy)

    if len(full_signal_dfs) != len(ctx.symbols):
        return 1e9, 1e9

    ref_sym = ctx.symbols[0]
    ref_df = ctx.data_maps[ref_sym][ctx.tf]
    is_off = int(multi_alignment_info["alignment_offsets"].get(ref_sym, 0))
    prebuilt_full_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for sym in ctx.symbols:
        start_idx = int(multi_alignment_info["alignment_offsets"][sym])
        sym_df = full_signal_dfs[sym].iloc[start_idx:]
        prebuilt_full_arrays[sym] = _dataframe_to_symbol_arrays(sym_df)

    eff_ref_len = int(multi_alignment_info["eff_ref_len"])
    cpcv_paths = multi_alignment_info["cpcv_bundle"][0]
    n_blocks_cpcv = multi_alignment_info["cpcv_bundle"][1]
    embargo = int(EMBARGO_BARS.get(ctx.tf, 12))
    unique_blocks = list_cpcv_block_ranges(eff_ref_len, n_blocks_cpcv, embargo=embargo)

    cfg = OPT_FUTURES_CONFIG
    min_seg_trades = int(cfg.get("FUTURES_MIN_TRADES_PER_CPCV_SEGMENT", 5))
    liq_mdd_thr = float(cfg.get("FUTURES_MAX_MDD", 25.0))

    block_results: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for b_start, b_end in unique_blocks:
        abs_start, abs_end = is_off + b_start, is_off + b_end
        slice_start, slice_end = max(0, abs_start - 1), min(len(ref_df), abs_end)
        aligned_data = _build_aligned_2d_from_prebuilt(
            prebuilt_full_arrays, ctx.symbols, slice_start, slice_end
        )
        if not aligned_data:
            continue
        try:
            engine = PortfolioBacktestEngineFast(
                aligned_data=aligned_data,
                symbol_names=ctx.symbols,
                strategy_params=params,
                initial_balance=float(FUTURES_INITIAL_BALANCE),
                fee_rate=TRADING_FEE_RATE,
                slippage_rate=SLIPPAGE_RATE,
            )
            b_trades_df, equity_curve, final_balance = engine.run()
        except Exception as exc:
            _logger.debug("ML Phase D block BT error: %s", exc)
            continue

        if b_trades_df is None or b_trades_df.empty:
            block_results[(b_start, b_end)] = {"log_ret": -1.0, "mdd": 100.0, "trades": 0}
            continue

        p_arr = b_trades_df["pnl"].to_numpy(dtype=np.float64, copy=False)
        f_arr = b_trades_df["entry_fee"].to_numpy(dtype=np.float64, copy=False)
        true_pnl_arr = p_arr - f_arr
        block_results[(b_start, b_end)] = {
            "log_ret": _log_tw_from_ret_pct(
                float((final_balance / FUTURES_INITIAL_BALANCE - 1.0) * 100.0)
            ),
            "mdd": float(calc_mdd_from_equity(equity_curve)) if equity_curve.size > 0 else 0.0,
            "trades": int(true_pnl_arr.size),
        }

    path_compound_raw_log_tw: List[float] = []
    path_seg_lists: List[List[float]] = []
    path_mdds: List[float] = []

    for path in cpcv_paths:
        p_log_ret = 0.0
        p_mdd = 0.0
        segs: List[float] = []
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
            lr = float(res["log_ret"])
            p_log_ret += lr
            segs.append(lr)
            p_mdd = max(p_mdd, float(res["mdd"]))

        if not valid_path:
            continue
        path_compound_raw_log_tw.append(p_log_ret)
        path_seg_lists.append(segs)
        path_mdds.append(p_mdd)

    if not path_compound_raw_log_tw:
        return 1e9, 1e9

    path_arr = np.asarray(path_compound_raw_log_tw, dtype=np.float64)
    p25 = float(np.percentile(path_arr, 25.0)) if path_arr.size else 0.0
    worst_mdd = float(np.max(path_mdds)) if path_mdds else 1.0
    f1 = p25 / max(worst_mdd, 0.001)

    sortinos: List[float] = []
    for segs in path_seg_lists:
        a = np.asarray(segs, dtype=np.float64)
        if a.size < 2:
            sortinos.append(float(np.mean(a)))
            continue
        neg = a[a < 0]
        ddev = float(np.std(neg, ddof=1)) if neg.size > 1 else float(np.std(a, ddof=1))
        sortinos.append(float(np.mean(a) / (ddev + 1e-9)))

    f2 = float(np.mean(sortinos)) if sortinos else 0.0

    trial.set_user_attr("ml_f1_stability", f1)
    trial.set_user_attr("ml_f2_quality", f2)
    trial.set_user_attr("ml_p25_log_tw", p25)
    trial.set_user_attr("ml_worst_seg_mdd", worst_mdd)

    return (-f1, -f2)


def topsis_select_best(pareto_trials: List[FrozenTrial]) -> FrozenTrial:
    """Min-max normalize two objectives (already negated for minimize); pick highest closeness."""
    if not pareto_trials:
        raise ValueError("empty pareto_trials")
    if len(pareto_trials) == 1:
        return pareto_trials[0]

    vals = np.array(
        [[float(t.values[0]), float(t.values[1])] for t in pareto_trials],
        dtype=np.float64,
    )
    vmin = vals.min(axis=0)
    vmax = vals.max(axis=0)
    span = np.where(vmax - vmin < 1e-12, 1.0, vmax - vmin)
    norm = (vals - vmin) / span
    ideal = np.array([0.0, 0.0], dtype=np.float64)
    nadir = np.array([1.0, 1.0], dtype=np.float64)
    d_pos = np.linalg.norm(norm - ideal, axis=1)
    d_neg = np.linalg.norm(norm - nadir, axis=1)
    score = d_neg / (d_pos + d_neg + 1e-12)
    best_idx = int(np.argmax(score))
    return pareto_trials[best_idx]


def check_hard_gates_ml(
    oos_result: Dict[str, Any],
    pbo_val: float,
    dsr_val: float,
    is_precision: float,
) -> bool:
    cfg = OPT_FUTURES_CONFIG
    pbo_max = float(cfg.get("FUTURES_PBO_MAX", 0.45))
    dsr_tgt = float(cfg.get("FUTURES_OBJECTIVE_DSR_TARGET", 1.5))
    oos_wr = float(oos_result.get("win_rate", 0.0) or 0.0)
    oos_mdd = float(oos_result.get("mdd", 100.0) or 100.0)
    g1 = pbo_val < pbo_max
    g2 = dsr_val > dsr_tgt
    g3 = oos_wr >= is_precision * 0.85
    g4 = abs(oos_mdd) < 20.0
    return bool(g1 and g2 and g3 and g4)
