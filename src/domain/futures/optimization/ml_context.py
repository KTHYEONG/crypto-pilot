from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.indicators.numpy_ops_futures import compute_atr_numpy
from src.core.settings import LOG_DIR
from src.domain.futures.optimization.data_aligner import (
    _build_aligned_2d_from_prebuilt,
    _dataframe_to_symbol_arrays,
    merge_effective_membership_constraints,
)
from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG
from src.domain.futures.optimization.validation import build_anchored_wf_legs
from src.domain.futures.portfolio.portfolio_constructor import (
    cov_lookback_bars,
    precompute_rolling_covariances,
)
from src.domain.futures.portfolio.signal_composer import (
    composer_sigma_lookback_bars,
    rolling_per_bar_return_std,
)
from src.domain.futures.strategy_runtime.bridge import FuturesMLStrategy
from src.domain.futures.universe.membership import build_membership_mask_bundle

if TYPE_CHECKING:
    from src.domain.futures.optimization.common import SignalCalibrator
    from src.domain.futures.strategy.config import StrategyConfig
    from src.domain.futures.validation.gates import PurgeBarsRegistry

_logger = logging.getLogger(__name__)
_PRECOMPUTE_LOCK = threading.Lock()
_MEMBERSHIP_STATS_PATH = LOG_DIR / "futures/optimization/membership_mask_stats.parquet"
AlphaOverrides = dict[str, tuple[np.ndarray, np.ndarray]]


@dataclass
class MLPhaseDContext:
    """Shared precomputed context passed to each Optuna trial in Phase D."""

    data_maps: dict[str, dict[str, Any]]
    symbols: list[str]
    tf: str
    seed: int = 42
    registry: PurgeBarsRegistry | None = None
    awf_leg_slices: list[dict[str, Any]] | None = None
    is_slice: dict[str, Any] | None = None
    holdout_slice: dict[str, np.ndarray] | None = None
    multi_alignment_info: dict[str, Any] | None = None
    # SOTA: Mathematical Policy Components
    calibrator: SignalCalibrator | None = None
    calibrator_short: SignalCalibrator | None = None
    estimated_b: float = 1.05
    kelly_ic_upper: float = 0.5  # T3-B: IC EWMA-based Kelly upper bound
    # Effective Bonferroni count = n_seeds x trials_per_seed (multi-seed studies).

    effective_total_trials: int | None = None
    # Coordinate ascent: "A"/"B"/"C"; frozen holds completed phases' Optuna-param dict slices.
    coordinate_phase: str | None = None
    coordinate_frozen_params: dict[str, Any] | None = None
    coordinate_shrunk_ranges: dict[str, tuple[Any, Any]] | None = None
    phase_ranges: dict[str, tuple[Any, Any]] | None = None
    ml_pipeline_fetch_start: str | None = None
    ml_pipeline_end: str | None = None
    ml_pipeline_is_start: str | None = None
    ml_pipeline_workers: int | None = None
    # Per-execution identifier for run-level trial filtering in shared Optuna DB.
    run_id: str | None = None
    strategy_mode: bool = False
    # AWF leg refit 시 ML pipeline에 전달할 전략 설정 (strategy/strategy-smoke 모드 전용)
    strategy_cfg: StrategyConfig | None = None
    universe_timeline: dict[date, frozenset[str] | set[str]] | None = None
    warmup_bars_required: int = 0
    data_sufficiency_report: dict[str, Any] | None = None


def build_universe_membership_arrays(
    *,
    symbol: str,
    datetimes: pd.Series,
    timeline: dict[date, frozenset[str] | set[str]] | None,
    warmup_bars_required: int,
    raw_kill_signal: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build per-bar membership arrays for one symbol."""
    if not timeline:
        n = len(datetimes)
        z = np.zeros(n, dtype=np.float64)
        o = np.ones(n, dtype=np.float64)
        raw = z if raw_kill_signal is None else np.asarray(raw_kill_signal, dtype=np.float64)
        return {
            "universe_active_mask": o,
            "universe_entry_warm_mask": o,
            "membership_kill_signal": z,
            "entry_block_mask": z,
            "kill_signal": raw,
        }
    bundle = build_membership_mask_bundle(
        datetimes=datetimes,
        symbol=symbol,
        timeline=timeline,
        warmup_bars_required=max(int(warmup_bars_required), 1),
        raw_kill_signal=raw_kill_signal,
    )
    return {
        "universe_active_mask": bundle.universe_active_mask,
        "universe_entry_warm_mask": bundle.universe_entry_warm_mask,
        "membership_kill_signal": bundle.membership_kill_signal,
        "entry_block_mask": bundle.entry_block_mask,
        "kill_signal": bundle.kill_signal,
    }


def merge_membership_constraints_into_aligned(
    aligned_data: dict[str, Any],
    *,
    persist_stats: bool = False,
) -> None:
    """Apply effective membership kill/entry constraints into aligned 2D arrays."""
    stats = merge_effective_membership_constraints(aligned_data, clamp_target_weights=False)
    if not persist_stats:
        pass
    rows = stats.get("rows", [])
    if not rows:
        pass
    if bool(aligned_data.get("_membership_stats_persisted", False)):
        pass
    _MEMBERSHIP_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(_MEMBERSHIP_STATS_PATH, index=False)
    aligned_data["_membership_stats_persisted"] = True


def _build_alpha_overrides_from_panel(
    panel: pd.DataFrame,
    *,
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    info: dict[str, Any],
) -> AlphaOverrides:
    """Build aligned alpha arrays (eff_ref_len) from panel output without mutating data_maps."""
    out: AlphaOverrides = {}
    if panel.empty:
        return out
    by_sym = panel.reset_index().groupby("symbol", sort=False)
    eff_len = int(info["eff_ref_len"])
    for sym in symbols:
        if sym not in data_maps or tf not in data_maps[sym]:
            continue
        long_arr = np.zeros(eff_len, dtype=np.float64)
        short_arr = np.zeros(eff_len, dtype=np.float64)
        try:
            sym_rows = by_sym.get_group(sym)
        except KeyError:
            out[sym] = (long_arr, short_arr)
            continue
        start_idx = int(info["alignment_offsets"][sym])
        raw = data_maps[sym][tf].iloc[start_idx : start_idx + eff_len]
        left = pd.DataFrame({"datetime": raw["datetime"]})
        right = sym_rows[["datetime", "alpha_long", "alpha_short"]].copy()
        left["_merge_datetime"] = pd.to_datetime(left["datetime"], utc=True).dt.tz_localize(None)
        right["_merge_datetime"] = pd.to_datetime(right["datetime"], utc=True).dt.tz_localize(None)
        merged = left.merge(
            right[["_merge_datetime", "alpha_long", "alpha_short"]],
            on="_merge_datetime",
            how="left",
        )
        long_arr[:] = merged["alpha_long"].fillna(0.0).to_numpy(dtype=np.float64)
        short_arr[:] = merged["alpha_short"].fillna(0.0).to_numpy(dtype=np.float64)
        out[sym] = (long_arr, short_arr)
    return out


def _fit_oos_platt_calibrators_from_maps(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    info: dict[str, Any],
    *,
    window_lo: int,
    window_hi_excl: int,
    oos_pool_start: int | None = None,
    alpha_overrides: AlphaOverrides | None = None,
) -> tuple[SignalCalibrator | None, SignalCalibrator | None, float]:
    """Platt scaling fit **only** on OOS bars (aligned eff_ref_len indices)."""
    from src.domain.futures.optimization.common import SignalCalibrator

    eff_len = int(info["eff_ref_len"])
    lo = int(np.clip(int(window_lo), 0, eff_len))
    hi = int(np.clip(int(window_hi_excl), 0, eff_len))
    min_bars = int(OPT_FUTURES_CONFIG.get("FUTURES_CALIB_PLATT_MIN_OOS_BARS", 80))
    widen = bool(OPT_FUTURES_CONFIG.get("FUTURES_CALIB_PLATT_OOS_WIDEN_TO_POOL", True))
    pool = int(oos_pool_start) if oos_pool_start is not None else lo

    def _collect(
        a0: int, a1: int
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        al, rl, ash, rsh = [], [], [], []
        for sym in symbols:
            if sym not in data_maps or tf not in data_maps[sym]:
                continue
            start_idx = int(info["alignment_offsets"][sym])
            sym_df = data_maps[sym][tf]
            raw = sym_df.iloc[start_idx : start_idx + eff_len]
            fwd_ret = raw["close"].pct_change(12).shift(-12).to_numpy(dtype=np.float64)
            i0, i1 = int(np.clip(a0, 0, eff_len)), int(np.clip(a1, 0, eff_len))
            if i1 <= i0:
                continue
            over = alpha_overrides.get(sym) if alpha_overrides else None
            if over is not None:
                alpha_long = np.asarray(over[0], dtype=np.float64)[i0:i1]
                alpha_short = np.asarray(over[1], dtype=np.float64)[i0:i1]
            else:
                alpha_long = (
                    raw["alpha_long"].to_numpy(dtype=np.float64)[i0:i1]
                    if "alpha_long" in raw.columns
                    else None
                )
                alpha_short = (
                    raw["alpha_short"].to_numpy(dtype=np.float64)[i0:i1]
                    if "alpha_short" in raw.columns
                    else None
                )
            if alpha_long is not None:
                r_t = fwd_ret[i0:i1]
                mask = ~np.isnan(alpha_long) & ~np.isnan(r_t)
                if mask.any():
                    al.append(alpha_long[mask])
                    rl.append(r_t[mask])
            if alpha_short is not None:
                r_ts = fwd_ret[i0:i1]
                mask_s = ~np.isnan(alpha_short) & ~np.isnan(r_ts)
                if mask_s.any():
                    ash.append(alpha_short[mask_s])
                    rsh.append(r_ts[mask_s])
        return al, rl, ash, rsh

    all_alphas, all_returns, all_alphas_short, all_returns_short = _collect(lo, hi)
    n_l = int(sum(a.size for a in all_alphas))
    n_s = int(sum(a.size for a in all_alphas_short))

    if widen and oos_pool_start is not None and pool < lo and (n_l < min_bars or n_s < min_bars):
        _logger.info(
            "[SignalCalibrator] OOS Platt widening window start %d \u2192 %d (pool start)",
            lo,
            pool,
        )
        lo = int(np.clip(pool, 0, eff_len))
        all_alphas, all_returns, all_alphas_short, all_returns_short = _collect(lo, hi)
        n_l = int(sum(a.size for a in all_alphas))
        n_s = int(sum(a.size for a in all_alphas_short))

    if n_l < min_bars and n_s < min_bars:
        _logger.warning(
            "[SignalCalibrator] OOS Platt skipped: only %d long / %d short samples in [%d,%d) "
            "(min %d). Using uncalibrated scores in prebuilt.",
            n_l,
            n_s,
            lo,
            hi,
            min_bars,
        )
        return None, None, 1.05

    calib: SignalCalibrator | None = None
    calib_s: SignalCalibrator | None = None
    est_b = 1.05
    if all_alphas and n_l >= min(30, min_bars):
        calib = SignalCalibrator()
        calib.fit(np.concatenate(all_alphas), np.concatenate(all_returns))
        est_b = calib.mean_b
    if all_alphas_short and n_s >= min(30, min_bars):
        calib_s = SignalCalibrator()
        calib_s.fit(np.concatenate(all_alphas_short), np.concatenate(all_returns_short))
    return calib, calib_s, est_b


def _build_prebuilt_full_arrays(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    info: dict[str, Any],
    *,
    calibrator: SignalCalibrator | None,
    calibrator_short: SignalCalibrator | None,
    alpha_overrides: AlphaOverrides | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Build vector bundles for ``_build_aligned_2d_from_prebuilt``.

    One row = eff_ref_len bar.
    """
    prebuilt_full: dict[str, dict[str, np.ndarray]] = {}
    eff_len = int(info["eff_ref_len"])
    for sym in symbols:
        if sym not in data_maps or tf not in data_maps[sym]:
            continue
        start_idx = int(info["alignment_offsets"][sym])
        raw_full = data_maps[sym][tf].iloc[start_idx : start_idx + eff_len]

        trimmed_sig = pd.DataFrame(index=raw_full.index)
        trimmed_sig["close"] = raw_full["close"].to_numpy(dtype=np.float64, copy=False)
        trimmed_sig["high"] = raw_full["high"].to_numpy(dtype=np.float64, copy=False)
        trimmed_sig["low"] = raw_full["low"].to_numpy(dtype=np.float64, copy=False)
        trimmed_sig["open"] = raw_full["open"].to_numpy(dtype=np.float64, copy=False)
        trimmed_sig["volume"] = (
            raw_full["volume"].to_numpy(dtype=np.float64, copy=False)
            if "volume" in raw_full.columns
            else np.ones(len(raw_full))
        )
        _atr_col = (
            raw_full["atr"].to_numpy(dtype=np.float64, copy=False)
            if "atr" in raw_full.columns
            else None
        )
        if _atr_col is None or not np.any(_atr_col > 0):
            _atr_period_fb = int(OPT_FUTURES_CONFIG.get("FUTURES_ATR_PERIOD_FIXED", 30))
            _atr_col = compute_atr_numpy(
                raw_full["high"].to_numpy(dtype=np.float64),
                raw_full["low"].to_numpy(dtype=np.float64),
                raw_full["close"].to_numpy(dtype=np.float64),
                _atr_period_fb,
            )
        trimmed_sig["atr"] = np.where(
            np.isfinite(_atr_col) & (_atr_col > 0),
            _atr_col,
            raw_full["close"].to_numpy(dtype=np.float64) * 0.01,
        )
        trimmed_sig["garch_kelly_f"] = (
            raw_full["garch_kelly_f"].to_numpy(dtype=np.float64, copy=False)
            if "garch_kelly_f" in raw_full.columns
            else np.ones(len(raw_full))
        )
        trimmed_sig["slot_rank_score"] = (
            raw_full["slot_rank_score"].to_numpy(dtype=np.float64, copy=False)
            if "slot_rank_score" in raw_full.columns
            else np.zeros(len(raw_full))
        )

        if "funding_rate_sum" in raw_full.columns:
            trimmed_sig["funding_rate_sum"] = raw_full["funding_rate_sum"].to_numpy(
                dtype=np.float64, copy=False
            )
        if "kill_signal" in raw_full.columns:
            trimmed_sig["kill_signal"] = raw_full["kill_signal"].to_numpy(
                dtype=np.float64,
                copy=False,
            )
        if "membership_kill_signal" in raw_full.columns:
            trimmed_sig["membership_kill_signal"] = raw_full["membership_kill_signal"].to_numpy(
                dtype=np.float64,
                copy=False,
            )
        if "entry_block_mask" in raw_full.columns:
            trimmed_sig["entry_block_mask"] = raw_full["entry_block_mask"].to_numpy(
                dtype=np.float64,
                copy=False,
            )

        over = alpha_overrides.get(sym) if alpha_overrides else None
        if over is not None:
            alpha_long_full = np.asarray(over[0], dtype=np.float64)
            alpha_short_full = np.asarray(over[1], dtype=np.float64)
        else:
            alpha_long_full = (
                raw_full["alpha_long"].to_numpy(dtype=np.float64, copy=False)
                if "alpha_long" in raw_full.columns
                else None
            )
            alpha_short_full = (
                raw_full["alpha_short"].to_numpy(dtype=np.float64, copy=False)
                if "alpha_short" in raw_full.columns
                else None
            )
        # Prefer current strategy alpha path. Keep legacy fallback for non-strategy runs.
        gp_base = (
            alpha_long_full
            if alpha_long_full is not None
            else (
                raw_full["alpha_long_00"].to_numpy(dtype=np.float64, copy=False)
                if "alpha_long_00" in raw_full.columns
                else np.zeros(len(raw_full), dtype=np.float64)
            )
        )
        if alpha_long_full is not None:
            trimmed_sig["alpha_long"] = alpha_long_full
        if alpha_short_full is not None:
            trimmed_sig["alpha_short"] = alpha_short_full
        if calibrator:
            p_base = calibrator.predict_prob(gp_base)
            trimmed_sig["ml_calib_prob"] = p_base

            if alpha_long_full is not None:
                p_l = calibrator.predict_prob(alpha_long_full)
                trimmed_sig["ml_calib_prob_long"] = p_l
            else:
                trimmed_sig["ml_calib_prob_long"] = trimmed_sig["ml_calib_prob"]

            if alpha_short_full is not None:
                calib_s = calibrator_short or calibrator
                p_s = calib_s.predict_prob(alpha_short_full)
                trimmed_sig["ml_calib_prob_short"] = p_s
            else:
                trimmed_sig["ml_calib_prob_short"] = trimmed_sig["ml_calib_prob"]
        else:
            trimmed_sig["ml_calib_prob"] = raw_full.get("ml_calib_prob", 0.5)
            trimmed_sig["ml_calib_prob_long"] = raw_full.get("ml_calib_prob_long", 0.5)
            trimmed_sig["ml_calib_prob_short"] = raw_full.get("ml_calib_prob_short", 0.5)

        gp_centered = gp_base - 0.5
        trimmed_sig["trend_direction"] = np.where(
            np.abs(gp_centered) > 0.01, np.sign(gp_centered), 0.0
        ).astype(np.float64)
        trimmed_sig["entry_upper"] = 0.0
        trimmed_sig["entry_lower"] = 999999.0
        _xs_cols = (
            "xs_score_long",
            "xs_score_short",
            "expected_variance",
            "target_variance",
            "btc_trend_vol_adj_24h",
        )
        for col in _xs_cols:
            if col in raw_full.columns:
                trimmed_sig[col] = raw_full[col].to_numpy(dtype=np.float64, copy=False)

        _sig_win = composer_sigma_lookback_bars(tf, OPT_FUTURES_CONFIG)
        trimmed_sig["composer_sigma_bar"] = rolling_per_bar_return_std(
            raw_full["close"].to_numpy(dtype=np.float64, copy=False),
            _sig_win,
        )

        _inject_dyn_leverage_trimmed(trimmed_sig, raw_full)

        prebuilt_full[sym] = _dataframe_to_symbol_arrays(trimmed_sig)

    return prebuilt_full


def _inject_dyn_leverage_trimmed(trimmed_sig: pd.DataFrame, raw_full: pd.DataFrame) -> None:
    """Apply default static leverage to trimmed signal, bypassing HMM dyn leverage."""
    cfg = OPT_FUTURES_CONFIG
    base_lev = float(cfg.get("FUTURES_DISCOVERY_LEVERAGE", 5))
    trimmed_sig["dyn_leverage"] = float(base_lev)


def compute_multi_alignment_info(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    embargo: int,
) -> dict[str, Any] | None:
    """Precompute alignment, fingerprints, and CSM ranks to avoid per-trial overhead."""
    is_start_dts_per_sym: dict[str, Any] = {}

    for sym in symbols:
        sym_df = data_maps[sym].get(tf)
        if sym_df is None or sym_df.empty:
            continue

        is_off = int(data_maps[sym].get(f"is_start_idx_{tf}", 0))
        if len(sym_df) > is_off and "datetime" in sym_df.columns:
            is_start_dts_per_sym[sym] = sym_df["datetime"].iloc[is_off]

    if not is_start_dts_per_sym:
        return None

    common_is_start_dt = max(is_start_dts_per_sym.values())

    alignment_offsets: dict[str, int] = {}
    eff_ref_lens: list[int] = []

    for sym in symbols:
        sym_df = data_maps[sym].get(tf)
        if sym_df is None or sym_df.empty or "datetime" not in sym_df.columns:
            continue
        start_idx = sym_df["datetime"].searchsorted(common_is_start_dt)
        alignment_offsets[sym] = int(start_idx)
        eff_ref_lens.append(len(sym_df) - int(start_idx))

    if not eff_ref_lens:
        return None

    eff_ref_len = min(eff_ref_lens)
    if eff_ref_len < 200:
        return None

    return {
        "common_is_start_dt": common_is_start_dt,
        "alignment_offsets": alignment_offsets,
        "eff_ref_len": eff_ref_len,
    }


def _base_engine_params(ml: dict[str, Any], tf: str) -> dict[str, Any]:
    ann_vol = float(ml.get("TARGET_ANN_VOL", 0.20))
    kelly_lambda = float(ml.get("KELLY_LAMBDA", 0.20))
    lev = float(OPT_FUTURES_CONFIG.get("FUTURES_DISCOVERY_LEVERAGE", 5))
    cfg = OPT_FUTURES_CONFIG
    atm = float(ml.get("ATR_MULT", cfg.get("FUTURES_ATR_STOP_MULT", 2.5)))
    atr_period_fixed = int(cfg.get("FUTURES_ATR_PERIOD_FIXED", 30))

    return {
        "TIMEFRAME": tf,
        "SIGNAL_TYPE": "ML_CALIB_PROB",
        "REGIME_TYPE": "EMA_ATR",
        "SIZING_METHOD": "profit_factor_kelly",
        "USE_CS_RANK_ENGINE": False,
        "K_LONG": int(ml.get("K_LONG", 2)),
        "K_SHORT": int(ml.get("K_SHORT", 2)),
        "REBALANCE_BARS": int(ml.get("REBALANCE_BARS", 1)),
        "REBALANCE_TURNOVER_THRESHOLD": float(ml.get("REBALANCE_TURNOVER_THRESHOLD", 0.15)),
        "MIN_SCORE_PERCENTILE": float(ml.get("MIN_SCORE_PERCENTILE", 0.50)),
        "CRISIS_GAMMA": float(ml.get("CRISIS_GAMMA", 1.0)),
        "TRAIL_MULT": float(ml.get("TRAIL_MULT", atm)),
        "ATR_MULT": atm,
        "ATR_PERIOD": int(ml.get("ATR_PERIOD", atr_period_fixed)),
        "SHORT_TP_MULT": float(ml.get("SHORT_TP_MULT", 2.0)),
        "LONG_SCALE_ATR_MULT": float(ml.get("LONG_SCALE_ATR_MULT", 3.0)),
        "RISK_PER_TRADE": kelly_lambda,
        "MAX_EXPOSURE_PER_COIN": float(ml.get("MAX_EXPOSURE_PER_COIN", 1.0)),
        "MAX_EXPOSURE": float(ml.get("MAX_EXPOSURE", 1.0)),
        "DD_SCALING_THRESHOLD": float(ml.get("DD_SCALING_THRESHOLD", 0.0)),
        "CS_Z_SCORE_THRESHOLD": float(ml.get("CS_Z_SCORE_THRESHOLD", 1.0)),
        "LONG_CS_Z_ENTRY": float(ml.get("LONG_CS_Z_ENTRY", ml.get("CS_Z_SCORE_THRESHOLD", 1.0))),
        "SHORT_CS_Z_ENTRY": float(ml.get("SHORT_CS_Z_ENTRY", ml.get("CS_Z_SCORE_THRESHOLD", 1.0))),
        "HYSTERESIS_GAP": float(ml.get("HYSTERESIS_GAP", 0.3)),
        "DYNAMIC_RA_CRISIS_COEF": float(ml.get("DYNAMIC_RA_CRISIS_COEF", 3.0)),
        "DYNAMIC_RA_BEAR_COEF": float(ml.get("DYNAMIC_RA_BEAR_COEF", 1.5)),
        "NORM_VAR_CONSTANT": float(ml.get("NORM_VAR_CONSTANT", 0.5)),
        "CRISIS_LONG_Z_BOOST": float(ml.get("CRISIS_LONG_Z_BOOST", 0.0)),
        "CRISIS_LONG_MAG_SUPPRESS": float(
            ml.get(
                "CRISIS_LONG_MAG_SUPPRESS",
                cfg.get("FUTURES_CRISIS_LONG_MAG_SUPPRESS", 1.0),
            )
        ),
        "TARGET_ANN_VOL": ann_vol,
        "KELLY_LAMBDA": kelly_lambda,
        "USE_COMPOUNDING": True,
        "LEVERAGE": int(lev),
        "BETA_ALPHA": float(ml.get("BETA_ALPHA", cfg.get("FUTURES_DEFAULT_BETA_ALPHA", 1.0))),
        "EV_HURDLE_BPS": float(
            ml.get("EV_HURDLE_BPS", cfg.get("FUTURES_DEFAULT_EV_HURDLE_BPS", 5.0))
        ),
        "SLIPPAGE_BPS_BUFFER_MULT": float(
            ml.get("SLIPPAGE_BPS_BUFFER_MULT", cfg.get("SLIPPAGE_BPS_BUFFER_MULT", 1.0))
        ),
        "TIME_BARRIER_H": float(
            ml.get("TIME_BARRIER_H", cfg.get("FUTURES_DEFAULT_TIME_BARRIER_H", 0.0))
        ),
        "PORTFOLIO_KAPPA": float(
            ml.get("PORTFOLIO_KAPPA", cfg.get("FUTURES_PORTFOLIO_KAPPA", 0.35))
        ),
        "FUTURES_EXECUTION_MODE": str(ml.get("FUTURES_EXECUTION_MODE", "coarse")),
        "STRATEGY_MODE": bool(ml.get("STRATEGY_MODE", False)),
    }


def precompute_ml_optimization_context(ctx: MLPhaseDContext) -> None:
    """Pre-align and pre-slice all data before Optuna starts to eliminate trial overhead."""
    from src.domain.futures.optimization.common import EMBARGO_BARS

    with _PRECOMPUTE_LOCK:
        t_precompute_total = time.perf_counter()
        t_align_total = 0.0
        t_cov_total = 0.0
        t_awf_refit_total = 0.0
        t_calibrator_total = 0.0
        t_prebuilt_total = 0.0
        if ctx.awf_leg_slices is not None:
            pass

        # 1. Alignment & Baseline Signals
        fc_pre = OPT_FUTURES_CONFIG
        atm0 = float(fc_pre.get("FUTURES_ATR_STOP_MULT", 2.5))
        pre_ml: dict[str, Any] = {
            "TRAILING_ACTIVATION_ATR": 1.0,
            "BAYESIAN_C": 10.0,
            "KELLY_SHRINKAGE": 0.3,
            "K_LONG": 2,
            "K_SHORT": 2,
            "REBALANCE_BARS": 6,
            "MIN_SCORE_PERCENTILE": 0.55,
            "CRISIS_GAMMA": 1.0,
            "ATR_PERIOD": 30,
            "ATR_MULT": atm0,
            "TRAIL_MULT": atm0,
            "PORTFOLIO_KAPPA": float(fc_pre.get("FUTURES_PORTFOLIO_KAPPA", 0.35)),
        }
        params = _base_engine_params(pre_ml, ctx.tf)
        FuturesMLStrategy(name="Precompute", params=params)

        emb = int(EMBARGO_BARS.get(ctx.tf, 12))
        t_align = time.perf_counter()
        info = compute_multi_alignment_info(ctx.data_maps, ctx.symbols, ctx.tf, emb)
        t_align_total += time.perf_counter() - t_align
        if info is None:
            pass
        ctx.multi_alignment_info = info

        eff_len = int(info["eff_ref_len"])
        embargo = int(EMBARGO_BARS.get(ctx.tf, 12))
        k_legs = int(OPT_FUTURES_CONFIG.get("FUTURES_AWF_K_LEGS", 6))
        is_pool = float(OPT_FUTURES_CONFIG.get("FUTURES_AWF_IS_POOL_FRAC", 0.70))
        awf_legs = build_anchored_wf_legs(eff_len, k=k_legs, embargo=embargo, is_pool_frac=is_pool)

        lookback = cov_lookback_bars(ctx.tf, OPT_FUTURES_CONFIG)
        close_2d_full = np.zeros((eff_len, len(ctx.symbols)), dtype=np.float64)
        for s_idx, sym in enumerate(ctx.symbols):
            if sym in ctx.data_maps and ctx.tf in ctx.data_maps[sym]:
                start_idx = info["alignment_offsets"][sym]
                close_2d_full[:, s_idx] = (
                    ctx.data_maps[sym][ctx.tf]["close"]
                    .iloc[start_idx : start_idx + eff_len]
                    .to_numpy(dtype=np.float64)
                )
        t_cov = time.perf_counter()
        sigma_3d_full = precompute_rolling_covariances(close_2d_full, lookback)
        t_cov_total += time.perf_counter() - t_cov

        first_awf_anchor = int(awf_legs[0][1]) if awf_legs else max(1, int(eff_len * is_pool))

        dates_ok = bool(ctx.ml_pipeline_fetch_start and ctx.ml_pipeline_end)
        use_full_leg_ml = dates_ok and bool(awf_legs)

        wrk = ctx.ml_pipeline_workers or max(1, min(8, len(ctx.symbols)))

        leg_refit_slices: list[dict[str, Any]] | None = None
        last_calib: SignalCalibrator | None = None
        last_calib_short: SignalCalibrator | None = None
        last_est_b = 1.05

        if use_full_leg_ml:
            from src.domain.futures.strategy.ml_builder import precompute_anchored_ml_panels
            from src.domain.futures.strategy_runtime.bridge import run_ml_pipeline_for_universe

            _logger.debug(
                "[ML_OPT] AWF full ML leg refit - %dx universe pipeline "
                "(cross-sectional alpha + systemic HMM + fusion). Expect long precompute.",
                len(awf_legs),
            )
            ref_sym = next(
                (s for s in ctx.symbols if s in ctx.data_maps and ctx.tf in ctx.data_maps[s]),
                None,
            )
            if ref_sym is None:
                use_full_leg_ml = False
            else:
                if ctx.strategy_cfg is None:
                    _logger.warning(
                        "[ML_OPT] strategy_cfg missing; disable per-leg ML refit cache path."
                    )
                    use_full_leg_ml = False
                if use_full_leg_ml:
                    sym_df_ref = ctx.data_maps[ref_sym][ctx.tf]
                    start_idx_ref = int(info["alignment_offsets"][ref_sym])
                    t_panel_cache = time.perf_counter()
                    precomputed_panels = precompute_anchored_ml_panels(
                        ctx.data_maps,
                        list(ctx.symbols),
                        ctx.tf,
                        ctx.strategy_cfg,
                    )
                    _logger.info(
                        (
                            "[ML-PANEL-CACHE] scope=awf_precompute hit=true "
                            "build=%.2fs rows=%d symbols=%d features=%d"
                        ),
                        time.perf_counter() - t_panel_cache,
                        int(precomputed_panels.features.values.shape[0]),
                        int(precomputed_panels.features.values.shape[1]),
                        int(precomputed_panels.features.values.shape[2]),
                    )
                    tmp_slices: list[dict[str, Any]] = []
                    failed = False
                    for leg_i, (_train_s, anchor, test_s, test_e) in enumerate(awf_legs):
                        leg_t0 = time.perf_counter()
                        idx_row = start_idx_ref + int(anchor)
                        if idx_row >= len(sym_df_ref):
                            _logger.error(
                                "[ML_OPT] AWF leg %d anchor idx %d out of range (len=%d).",
                                leg_i,
                                idx_row,
                                len(sym_df_ref),
                            )
                            failed = True
                            break
                        cutoff_dt = pd.to_datetime(sym_df_ref["datetime"].iloc[idx_row], utc=True)
                        is_end_str = cutoff_dt.isoformat()

                        _logger.debug(
                            "[ML_OPT] AWF leg %d/%d: is_end=%s train=[0,%d) test=[%d,%d)",
                            leg_i + 1,
                            len(awf_legs),
                            is_end_str,
                            int(anchor),
                            int(test_s),
                            int(test_e),
                        )

                        ml_out = run_ml_pipeline_for_universe(
                            list(ctx.symbols),
                            ctx.tf,
                            ctx.ml_pipeline_fetch_start,
                            ctx.ml_pipeline_end,
                            dict(OPT_FUTURES_CONFIG),
                            workers=wrk,
                            n_jobs=wrk,
                            is_end_date=is_end_str,
                            is_start_date=ctx.ml_pipeline_is_start,
                            gp_only=False,
                            preloaded_data_maps=ctx.data_maps,
                            seed=ctx.seed,
                            strategy_cfg=ctx.strategy_cfg,
                            anchor_end_idx=int(anchor),
                            target_start_idx=int(test_s),
                            target_end_idx=int(test_e),
                            precomputed_panels=precomputed_panels,
                        )
                        # alpha_panel 기준으로 empty check (meta_feature_frame은 bridge에서 미사용)
                        _alpha_panel = getattr(ml_out, "alpha_panel", None)
                        if _alpha_panel is None or _alpha_panel.empty:
                            _logger.error(
                                "[ML_OPT] AWF leg %d ML pipeline returned empty output.", leg_i
                            )
                            failed = True
                            break

                        alpha_overrides = _build_alpha_overrides_from_panel(
                            _alpha_panel,
                            data_maps=ctx.data_maps,
                            symbols=ctx.symbols,
                            tf=ctx.tf,
                            info=info,
                        )

                        t_calib = time.perf_counter()
                        calib_leg, calib_s_leg, est_b_leg = _fit_oos_platt_calibrators_from_maps(
                            ctx.data_maps,
                            ctx.symbols,
                            ctx.tf,
                            info,
                            window_lo=int(anchor),
                            window_hi_excl=int(test_s),
                            oos_pool_start=first_awf_anchor,
                            alpha_overrides=alpha_overrides,
                        )
                        last_calib, last_calib_short, last_est_b = (
                            calib_leg,
                            calib_s_leg,
                            est_b_leg,
                        )
                        t_calibrator_total += time.perf_counter() - t_calib

                        # [ALPHA-ALIGN] per-leg alpha residual diagnostic (first 3 legs only)
                        if leg_i < 3:
                            _al_nz_list: list[float] = []
                            _as_nz_list: list[float] = []
                            # Slice-scoped nonzero ratio measured strictly within the leg's
                            # backtest evaluation window [test_s, test_e). A near-zero value
                            # here while the full-frame ratio is positive proves that the ML
                            # alpha coverage does not overlap this leg (root cause of
                            # zero_trades_first_leg pruning).
                            _al_nz_slice_list: list[float] = []
                            _as_nz_slice_list: list[float] = []
                            _bars_leg = int(test_e) - int(test_s)
                            for _sym in ctx.symbols:
                                over = alpha_overrides.get(_sym)
                                if over is None:
                                    continue
                                _al_full = np.asarray(over[0], dtype=np.float64)
                                _as_full = np.asarray(over[1], dtype=np.float64)
                                _al_nz_list.append(float((_al_full != 0).mean()))
                                _as_nz_list.append(float((_as_full != 0).mean()))
                                _al_slice = _al_full[int(test_s) : int(test_e)]
                                _as_slice = _as_full[int(test_s) : int(test_e)]
                                if _al_slice.size > 0:
                                    _al_nz_slice_list.append(float((_al_slice != 0).mean()))
                                if _as_slice.size > 0:
                                    _as_nz_slice_list.append(float((_as_slice != 0).mean()))
                            _al_nz = float(np.mean(_al_nz_list)) if _al_nz_list else 0.0
                            _as_nz = float(np.mean(_as_nz_list)) if _as_nz_list else 0.0
                            _al_nz_slice = (
                                float(np.mean(_al_nz_slice_list)) if _al_nz_slice_list else 0.0
                            )
                            _as_nz_slice = (
                                float(np.mean(_as_nz_slice_list)) if _as_nz_slice_list else 0.0
                            )
                            _logger.info(
                                "[ALPHA-ALIGN] leg=%d range=[%d,%d) bars=%d "
                                "full_long_nz=%.3f full_short_nz=%.3f "
                                "leg_long_nz=%.3f leg_short_nz=%.3f",
                                leg_i,
                                int(test_s),
                                int(test_e),
                                _bars_leg,
                                _al_nz,
                                _as_nz,
                                _al_nz_slice,
                                _as_nz_slice,
                            )

                        t_prebuilt = time.perf_counter()
                        prebuilt_leg = _build_prebuilt_full_arrays(
                            ctx.data_maps,
                            ctx.symbols,
                            ctx.tf,
                            info,
                            calibrator=calib_leg,
                            calibrator_short=calib_s_leg,
                            alpha_overrides=alpha_overrides,
                        )
                        aligned_leg = _build_aligned_2d_from_prebuilt(
                            prebuilt_leg, ctx.symbols, test_s, test_e, sigma_3d_full=sigma_3d_full
                        )
                        tmp_slices.append({"leg_range": (test_s, test_e), "data": aligned_leg})
                        t_prebuilt_total += time.perf_counter() - t_prebuilt
                        leg_elapsed = time.perf_counter() - leg_t0
                        t_awf_refit_total += leg_elapsed
                        _logger.info(
                            (
                                "[AWF-REFIT-PROF] leg=%d/%d total=%.2fs bars=%d "
                                "train_end=%d test=[%d,%d)"
                            ),
                            leg_i + 1,
                            len(awf_legs),
                            leg_elapsed,
                            int(test_e) - int(test_s),
                            int(anchor),
                            int(test_s),
                            int(test_e),
                        )

                    if failed or len(tmp_slices) != len(awf_legs):
                        _logger.debug(
                            "[ML_OPT] Per-leg ML refit incomplete (%d/%d legs); "
                            "falling back to single global ML merge for AWF.",
                            len(tmp_slices),
                            len(awf_legs),
                        )
                    else:
                        leg_refit_slices = tmp_slices

        if leg_refit_slices is not None:
            ctx.awf_leg_slices = leg_refit_slices
            ctx.calibrator = last_calib
            ctx.calibrator_short = last_calib_short
            ctx.estimated_b = last_est_b

            t_prebuilt = time.perf_counter()
            prebuilt_full_is = _build_prebuilt_full_arrays(
                ctx.data_maps,
                ctx.symbols,
                ctx.tf,
                info,
                calibrator=ctx.calibrator,
                calibrator_short=ctx.calibrator_short,
            )
            ctx.is_slice = _build_aligned_2d_from_prebuilt(
                prebuilt_full_is, ctx.symbols, 0, eff_len, sigma_3d_full=sigma_3d_full
            )
            t_prebuilt_total += time.perf_counter() - t_prebuilt
        else:
            calib_min_bars = int(OPT_FUTURES_CONFIG.get("FUTURES_CALIB_PLATT_MIN_OOS_BARS", 80))
            calib_hi = max(0, first_awf_anchor - embargo)
            if calib_hi < calib_min_bars:
                _logger.warning(
                    "[CALIB] IS-only calibration window too small (%d bars < min %d); "
                    "falling back to OOS-pool window [%d, %d) to avoid data starvation.",
                    calib_hi,
                    calib_min_bars,
                    first_awf_anchor,
                    eff_len,
                )
                t_calib = time.perf_counter()
                c0, c0s, eb0 = _fit_oos_platt_calibrators_from_maps(
                    ctx.data_maps,
                    ctx.symbols,
                    ctx.tf,
                    info,
                    window_lo=first_awf_anchor,
                    window_hi_excl=eff_len,
                    oos_pool_start=first_awf_anchor,
                )
            else:
                t_calib = time.perf_counter()
                c0, c0s, eb0 = _fit_oos_platt_calibrators_from_maps(
                    ctx.data_maps,
                    ctx.symbols,
                    ctx.tf,
                    info,
                    window_lo=0,
                    window_hi_excl=calib_hi,
                    oos_pool_start=None,
                )
            t_calibrator_total += time.perf_counter() - t_calib
            ctx.calibrator = c0
            ctx.calibrator_short = c0s
            ctx.estimated_b = eb0

            t_prebuilt = time.perf_counter()
            prebuilt_full = _build_prebuilt_full_arrays(
                ctx.data_maps,
                ctx.symbols,
                ctx.tf,
                info,
                calibrator=ctx.calibrator,
                calibrator_short=ctx.calibrator_short,
            )

            ctx.is_slice = _build_aligned_2d_from_prebuilt(
                prebuilt_full, ctx.symbols, 0, eff_len, sigma_3d_full=sigma_3d_full
            )
            t_prebuilt_total += time.perf_counter() - t_prebuilt

            ctx.awf_leg_slices = []
            for _leg_i, (_train_s, _train_e, test_s, test_e) in enumerate(awf_legs):
                # [ALPHA-ALIGN] per-leg alpha residual diagnostic (first 3 legs only)
                if _leg_i < 3:
                    _al_nz2_list: list[float] = []
                    _as_nz2_list: list[float] = []
                    # Slice-scoped nonzero ratio within the leg window [test_s, test_e).
                    _al_nz2_slice_list: list[float] = []
                    _as_nz2_slice_list: list[float] = []
                    _offsets2 = info["alignment_offsets"] if info else {}
                    _bars_leg2 = int(test_e) - int(test_s)
                    for _sym2 in ctx.symbols:
                        if _sym2 not in ctx.data_maps or ctx.tf not in ctx.data_maps[_sym2]:
                            continue
                        _ldf2 = ctx.data_maps[_sym2][ctx.tf]
                        _off2 = int(_offsets2.get(_sym2, 0))
                        _lo2 = _off2 + int(test_s)
                        _hi2 = _off2 + int(test_e)
                        if "alpha_long" in _ldf2.columns:
                            _al_nz2_list.append(float((_ldf2["alpha_long"] != 0).mean()))
                            _al2_slice = _ldf2["alpha_long"].to_numpy()[_lo2:_hi2]
                            if _al2_slice.size > 0:
                                _al_nz2_slice_list.append(float((_al2_slice != 0).mean()))
                        if "alpha_short" in _ldf2.columns:
                            _as_nz2_list.append(float((_ldf2["alpha_short"] != 0).mean()))
                            _as2_slice = _ldf2["alpha_short"].to_numpy()[_lo2:_hi2]
                            if _as2_slice.size > 0:
                                _as_nz2_slice_list.append(float((_as2_slice != 0).mean()))
                    _al_nz2 = float(np.mean(_al_nz2_list)) if _al_nz2_list else 0.0
                    _as_nz2 = float(np.mean(_as_nz2_list)) if _as_nz2_list else 0.0
                    _al_nz2_slice = (
                        float(np.mean(_al_nz2_slice_list)) if _al_nz2_slice_list else 0.0
                    )
                    _as_nz2_slice = (
                        float(np.mean(_as_nz2_slice_list)) if _as_nz2_slice_list else 0.0
                    )
                    _logger.info(
                        "[ALPHA-ALIGN] leg=%d range=[%d,%d) bars=%d "
                        "full_long_nz=%.3f full_short_nz=%.3f "
                        "leg_long_nz=%.3f leg_short_nz=%.3f",
                        _leg_i,
                        int(test_s),
                        int(test_e),
                        _bars_leg2,
                        _al_nz2,
                        _as_nz2,
                        _al_nz2_slice,
                        _as_nz2_slice,
                    )
                aligned = _build_aligned_2d_from_prebuilt(
                    prebuilt_full, ctx.symbols, test_s, test_e, sigma_3d_full=sigma_3d_full
                )
                ctx.awf_leg_slices.append({"leg_range": (test_s, test_e), "data": aligned})

        ctx.holdout_slice = None

        if ctx.is_slice is not None:
            xl_is = ctx.is_slice.get("xs_score_long")
            cl_is = ctx.is_slice.get("close")
            if xl_is is not None and cl_is is not None:
                xl_arr = np.asarray(xl_is, dtype=np.float64)
                cl_arr = np.asarray(cl_is, dtype=np.float64)
                if xl_arr.ndim == 2:
                    xl_arr = np.nanmean(xl_arr, axis=1)
                if cl_arr.ndim == 2:
                    cl_arr = np.nanmean(cl_arr, axis=1)
                fwd = np.log(np.clip(cl_arr[1:], 1e-12, None) / np.clip(cl_arr[:-1], 1e-12, None))
                sig = xl_arr[:-1]
                mask = np.isfinite(sig) & np.isfinite(fwd)
                if int(np.sum(mask)) > 30:
                    from scipy.stats import spearmanr as _spearmanr

                    _ic_raw, _ = _spearmanr(sig[mask], fwd[mask])
                    ctx.kelly_ic_upper = float(np.clip(abs(_ic_raw) * 10.0, 0.05, 0.5))
                    _logger.debug(
                        "[T3-B] IC\u2192Kelly upper: %.4f (Spearman IC=%.4f)",
                        ctx.kelly_ic_upper,
                        _ic_raw,
                    )

        ctx.multi_alignment_info["awf_legs"] = awf_legs

        if ctx.awf_leg_slices:
            aligned0 = ctx.awf_leg_slices[0].get("data") or {}
            xl0 = aligned0.get("xs_score_long")
            if xl0 is not None and getattr(xl0, "size", 0) > 0:
                xs_std = float(np.nanstd(np.asarray(xl0, dtype=np.float64)))
                ctx.multi_alignment_info["xs_score_aligned_std"] = xs_std
                if xs_std < 0.05:
                    _logger.debug(
                        "[ML_OPT] xs_score dispersion low (std=%.6f); check GP/CS merge path.",
                        xs_std,
                    )

            _log_precompute_xs_dispersion(
                aligned0,
                k_long=int(pre_ml["K_LONG"]),
            )
        precompute_total = time.perf_counter() - t_precompute_total
        precompute_profile = {
            "total": float(precompute_total),
            "align": float(t_align_total),
            "covariance": float(t_cov_total),
            "awf_refit_total": float(t_awf_refit_total),
            "calibrator_total": float(t_calibrator_total),
            "prebuilt_total": float(t_prebuilt_total),
            "awf_legs": len(awf_legs),
        }
        ctx.precompute_profile = precompute_profile
        _logger.info(
            (
                "[RUN-PROF] step=ml_precompute total=%.2fs align=%.2fs "
                "covariance=%.2fs awf_refit=%.2fs calibrator=%.2fs "
                "prebuilt=%.2fs legs=%d"
            ),
            precompute_profile["total"],
            precompute_profile["align"],
            precompute_profile["covariance"],
            precompute_profile["awf_refit_total"],
            precompute_profile["calibrator_total"],
            precompute_profile["prebuilt_total"],
            precompute_profile["awf_legs"],
        )


def _log_precompute_xs_dispersion(aligned0: dict[str, Any], *, k_long: int) -> None:
    """Lightweight precompute diagnostic (linear xs scores; no CS rank engine)."""
    _ = k_long
    xl = aligned0.get("xs_score_long")
    if xl is None or getattr(xl, "size", 0) == 0:
        pass
    arr_l = np.ascontiguousarray(xl, dtype=np.float64)
    n_b = arr_l.shape[0]
    prev_i = min(n_b - 1, max(0, n_b // 2))
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


def rerun_precompute_for_ctx(ctx: MLPhaseDContext) -> None:
    """Force regeneration of anchored AWF caches (different seed/calibration path)."""
    ctx.awf_leg_slices = None
    ctx.multi_alignment_info = None
    ctx.calibrator = None
    ctx.calibrator_short = None
    precompute_ml_optimization_context(ctx)
