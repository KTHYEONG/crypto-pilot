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
from src.core.settings import LOG_DIR, TAKER_FEE_BPS
from src.domain.futures.optimization.common import SignalCalibrator
from src.domain.futures.optimization.data_aligner import (
    _build_aligned_2d_from_prebuilt,
    _dataframe_to_symbol_arrays,
    merge_effective_membership_constraints,
)
from src.domain.futures.optimization.opt_config import (
    OPT_FUTURES_CONFIG,
    default_ev_hurdle_bps,
)
from src.domain.futures.optimization.validation import build_anchored_wf_legs
from src.domain.futures.portfolio.portfolio_constructor import (
    RiskSnapshot,
    cov_lookback_bars,
    precompute_rolling_covariances,
)
from src.domain.futures.portfolio.signal_composer import (
    composer_sigma_lookback_bars,
    rolling_per_bar_return_std,
)
from src.domain.futures.universe.membership import build_membership_mask_bundle

if TYPE_CHECKING:
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
    # AWF leg refit 시 ML pipeline에 전달할 전략 설정 (strategy/alpha 전용)
    strategy_cfg: StrategyConfig | None = None
    universe_timeline: dict[date, frozenset[str] | set[str]] | None = None
    warmup_bars_required: int = 0
    data_sufficiency_report: dict[str, Any] | None = None
    precompute_profile: dict[str, float] | None = None


def _build_beta_2d_full(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    alignment_info: dict[str, Any],
    eff_len: int,
) -> np.ndarray | None:
    """Build aligned beta matrix [T, N] when source beta column is present."""
    out = np.zeros((eff_len, len(symbols)), dtype=np.float64)
    has_any_beta = False
    for s_idx, sym in enumerate(symbols):
        if sym not in data_maps or tf not in data_maps[sym]:
            continue
        start_idx = int(alignment_info["alignment_offsets"][sym])
        raw = data_maps[sym][tf].iloc[start_idx : start_idx + eff_len]
        if "beta" not in raw.columns:
            continue
        beta_arr = raw["beta"].to_numpy(dtype=np.float64)
        if beta_arr.shape[0] != eff_len:
            continue
        out[:, s_idx] = np.nan_to_num(beta_arr, nan=0.0, posinf=0.0, neginf=0.0)
        has_any_beta = True
    if has_any_beta:
        return out
    return _build_trailing_btc_beta_fallback(
        close_2d=out,
        data_maps=data_maps,
        symbols=symbols,
        tf=tf,
        alignment_info=alignment_info,
        eff_len=eff_len,
    )


def _build_trailing_btc_beta_fallback(
    *,
    close_2d: np.ndarray,
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    alignment_info: dict[str, Any],
    eff_len: int,
) -> np.ndarray | None:
    """Build causal trailing BTC-beta fallback when source beta column is absent."""
    n_syms = len(symbols)
    if eff_len <= 1 or n_syms == 0:
        return None
    filled = np.asarray(close_2d, dtype=np.float64).copy()
    for s_idx, sym in enumerate(symbols):
        if sym not in data_maps or tf not in data_maps[sym]:
            continue
        start_idx = int(alignment_info["alignment_offsets"][sym])
        raw = data_maps[sym][tf].iloc[start_idx : start_idx + eff_len]
        if "close" not in raw.columns:
            continue
        c = np.asarray(raw["close"].to_numpy(dtype=np.float64), dtype=np.float64)
        if c.shape[0] == eff_len:
            filled[:, s_idx] = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    btc_idx = next((idx for idx, sym in enumerate(symbols) if "BTC" in sym.upper()), None)
    if btc_idx is None:
        return None
    ret = np.zeros((eff_len, n_syms), dtype=np.float64)
    prev = np.maximum(np.abs(filled[:-1, :]), 1e-12)
    ret[1:, :] = (filled[1:, :] - filled[:-1, :]) / prev
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    btc_ret = ret[:, btc_idx]
    lookback = max(20, int(OPT_FUTURES_CONFIG.get("FUTURES_PORTFOLIO_COV_LOOKBACK", 180)))
    beta = np.zeros((eff_len, n_syms), dtype=np.float64)
    for t in range(1, eff_len):
        st = max(1, t - lookback + 1)
        x = btc_ret[st : t + 1]
        var_x = float(np.var(x, ddof=1)) if x.size > 1 else 0.0
        if not np.isfinite(var_x) or var_x <= 1e-12:
            continue
        x_m = float(np.mean(x))
        for j in range(n_syms):
            y = ret[st : t + 1, j]
            if y.size != x.size or y.size <= 1:
                continue
            cov_xy = float(np.mean((x - x_m) * (y - float(np.mean(y)))))
            b = cov_xy / var_x
            beta[t, j] = float(np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0))
    return beta


def _attach_risk_snapshot_slice(
    aligned: dict[str, Any] | None,
    sigma_3d_full: np.ndarray,
    beta_2d_full: np.ndarray | None,
    slice_start: int,
    slice_end: int,
    *,
    close_2d_full: np.ndarray | None = None,
    symbols: list[str] | None = None,
    lookback: int = 60,
    vol_lookback: int = 20,
) -> None:
    """Attach factor-lite risk snapshot payload to aligned slice.

    Args:
        aligned: Target aligned dict to mutate.
        sigma_3d_full: Precomputed full-length covariance [T, N, N].
        beta_2d_full: Precomputed full-length BTC beta [T, N] or None.
        slice_start: Start index of the slice.
        slice_end: End index of the slice.
        close_2d_full: Full close price array [T, N] for residual variance computation.
        symbols: Symbol name list for BTC detection.
        lookback: Rolling window for residual variance.
        vol_lookback: Rolling window for residual variance std.

    """
    if aligned is None:
        return
    sigma_slice = np.asarray(sigma_3d_full[slice_start:slice_end], dtype=np.float64)
    beta_slice: np.ndarray | None = None
    beta_source = "unavailable"

    if beta_2d_full is not None:
        beta_slice = np.asarray(beta_2d_full[slice_start:slice_end], dtype=np.float64)
        aligned["btc_beta_2d"] = beta_slice
        beta_source = "trailing_btc"

    residual_var_slice: np.ndarray | None = None
    if close_2d_full is not None and symbols is not None:
        try:
            from src.domain.futures.forecast.risk import build_risk_forecast

            rf = build_risk_forecast(
                close_2d=close_2d_full[slice_start:slice_end],
                symbols=symbols,
                tf="",
                cfg={},
                lookback=lookback,
                vol_lookback=vol_lookback,
            )
            residual_var_slice = rf.residual_var_2d
            beta_source = rf.beta_source
            if rf.beta_2d is not None and beta_slice is None:
                beta_slice = rf.beta_2d
                aligned["btc_beta_2d"] = beta_slice
        except Exception as exc:
            _logger.debug("_attach_risk_snapshot_slice: residual var computation failed: %s", exc)

    aligned["risk_snapshot"] = RiskSnapshot(
        covariance_3d=sigma_slice,
        beta_2d=beta_slice,
        residual_var_2d=residual_var_slice,
    )
    if residual_var_slice is not None:
        aligned["residual_var_2d"] = residual_var_slice
    aligned["_beta_source"] = beta_source


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
        return
    rows = stats.get("rows", [])
    if not rows:
        return
    if bool(aligned_data.get("_membership_stats_persisted", False)):
        return
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

    def _collect(a0: int, a1: int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
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
                alpha_long = None
                if "alpha_long" in raw.columns:
                    alpha_long_arr = np.asarray(
                        raw["alpha_long"].to_numpy(dtype=np.float64),
                        dtype=np.float64,
                    )
                    alpha_long = alpha_long_arr[i0:i1]
                alpha_short = None
                if "alpha_short" in raw.columns:
                    alpha_short_arr = np.asarray(
                        raw["alpha_short"].to_numpy(dtype=np.float64),
                        dtype=np.float64,
                    )
                    alpha_short = alpha_short_arr[i0:i1]
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
    calibrator: SignalCalibrator | None = None,
    calibrator_short: SignalCalibrator | None = None,
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
        _atr_col = raw_full["atr"].to_numpy(dtype=np.float64, copy=False) if "atr" in raw_full.columns else None
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

        if "funding_rate_sum" in raw_full.columns:
            trimmed_sig["funding_rate_sum"] = raw_full["funding_rate_sum"].to_numpy(dtype=np.float64, copy=False)
        if "execution_cost_bps" in raw_full.columns:
            trimmed_sig["execution_cost_bps"] = raw_full["execution_cost_bps"].to_numpy(
                dtype=np.float64,
                copy=False,
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
            alpha_long_full: np.ndarray | None = np.asarray(over[0], dtype=np.float64)
            alpha_short_full: np.ndarray | None = np.asarray(over[1], dtype=np.float64)
        else:
            alpha_long_full = None
            if "alpha_long" in raw_full.columns:
                alpha_long_full = np.asarray(
                    raw_full["alpha_long"].to_numpy(dtype=np.float64, copy=False),
                    dtype=np.float64,
                )
            alpha_short_full = None
            if "alpha_short" in raw_full.columns:
                alpha_short_full = np.asarray(
                    raw_full["alpha_short"].to_numpy(dtype=np.float64, copy=False),
                    dtype=np.float64,
                )
        gp_base: np.ndarray = (
            alpha_long_full if alpha_long_full is not None else np.zeros(len(raw_full), dtype=np.float64)
        )
        if alpha_long_full is not None:
            trimmed_sig["alpha_long"] = alpha_long_full
        if alpha_short_full is not None:
            trimmed_sig["alpha_short"] = alpha_short_full
        # Legacy calibration payload is removed from active candidate path.
        _ = calibrator
        _ = calibrator_short

        gp_centered = gp_base - 0.5
        trimmed_sig["trend_direction"] = np.where(np.abs(gp_centered) > 0.01, np.sign(gp_centered), 0.0).astype(
            np.float64
        )
        trimmed_sig["entry_upper"] = 0.0
        trimmed_sig["entry_lower"] = 999999.0
        _xs_cols = (
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


def _attach_execution_cost_bps_2d(
    *,
    aligned: dict[str, Any] | None,
    prebuilt_arrays: dict[str, dict[str, np.ndarray]],
    symbols: list[str],
    slice_start: int,
    slice_end: int,
) -> None:
    """Attach optional execution cost tensor to aligned slice.

    Always attach a static per-symbol cost tensor as baseline and precompute
    a dynamic forecast candidate tensor for trial-level Phase2 A/B toggling.
    """
    if aligned is None:
        return
    cost_cols: list[np.ndarray] = []
    for sym in symbols:
        sym_arrs = prebuilt_arrays.get(sym)
        if sym_arrs is None:
            return
        cost_arr = sym_arrs.get("execution_cost_bps")
        if cost_arr is None or slice_end > int(cost_arr.shape[0]):
            return
        cost_cols.append(np.asarray(cost_arr[slice_start:slice_end], dtype=np.float64))
    if not cost_cols:
        return
    static_cost_bps_2d = np.ascontiguousarray(np.column_stack(cost_cols))
    static_cost_frac_2d = static_cost_bps_2d / 10000.0
    aligned["execution_cost_bps_2d_static"] = static_cost_bps_2d
    aligned["execution_cost_fraction_2d_static"] = static_cost_frac_2d
    aligned["execution_cost_bps_2d"] = static_cost_bps_2d
    aligned["execution_cost_fraction_2d"] = static_cost_frac_2d
    aligned["_cost_forecast_source_static"] = "universe_static"
    aligned["_cost_forecast_dynamic"] = 0.0

    try:
        from src.domain.futures.forecast.cost import CostModelConfig, build_cost_forecast

        close_2d = np.asarray(aligned.get("close"), dtype=np.float64)
        if close_2d.shape != static_cost_bps_2d.shape:
            return
        high_2d = aligned.get("high")
        low_2d = aligned.get("low")
        volume_2d = np.asarray(aligned.get("volume"), dtype=np.float64)
        funding_2d_raw = aligned.get("funding_rate_sum")
        funding_2d = (
            np.asarray(funding_2d_raw, dtype=np.float64)
            if funding_2d_raw is not None and np.asarray(funding_2d_raw).shape == close_2d.shape
            else None
        )
        cfg = CostModelConfig(
            taker_fee_bps=float(OPT_FUTURES_CONFIG.get("FUTURES_COST_TAKER_FEE_BPS", TAKER_FEE_BPS)),
            latency_buffer_bps=float(OPT_FUTURES_CONFIG.get("FUTURES_COST_LATENCY_BUFFER_BPS", 0.5)),
            impact_coef=float(OPT_FUTURES_CONFIG.get("FUTURES_COST_IMPACT_COEF", 0.5)),
            vol_buffer_coef=float(OPT_FUTURES_CONFIG.get("FUTURES_COST_VOL_BUFFER_COEF", 0.0)),
            funding_event_buffer_bps=float(OPT_FUTURES_CONFIG.get("FUTURES_COST_FUNDING_EVENT_BUFFER_BPS", 0.0)),
            adv_lookback=int(OPT_FUTURES_CONFIG.get("FUTURES_COST_ADV_LOOKBACK", 30)),
            vol_lookback=int(OPT_FUTURES_CONFIG.get("FUTURES_COST_VOL_LOOKBACK", 20)),
            estimated_order_notional=float(OPT_FUTURES_CONFIG.get("FUTURES_COST_ORDER_NOTIONAL_USDT", 0.0)),
            uncertainty_ratio=float(OPT_FUTURES_CONFIG.get("FUTURES_COST_UNCERTAINTY_RATIO", 0.1)),
            enable_dynamic_components=True,
        )
        cf = build_cost_forecast(
            close_2d=close_2d,
            high_2d=np.asarray(high_2d, dtype=np.float64)
            if high_2d is not None and np.asarray(high_2d).shape == close_2d.shape
            else None,
            low_2d=np.asarray(low_2d, dtype=np.float64)
            if low_2d is not None and np.asarray(low_2d).shape == close_2d.shape
            else None,
            volume_2d=volume_2d,
            funding_2d=funding_2d,
            adv_usdt_2d=None,
            universe_cost_bps_2d=static_cost_bps_2d,
            cfg=cfg,
            shape=close_2d.shape,
        )
        dynamic_bps_2d = np.ascontiguousarray(cf.execution_cost_bps_2d)
        dynamic_frac_2d = np.ascontiguousarray(cf.execution_cost_fraction_2d)
        aligned["execution_cost_bps_2d_dynamic"] = dynamic_bps_2d
        aligned["execution_cost_fraction_2d_dynamic"] = dynamic_frac_2d
        if cf.capacity_notional_2d is not None:
            aligned["capacity_notional_2d_dynamic"] = np.ascontiguousarray(cf.capacity_notional_2d)
        aligned["_cost_forecast_source_dynamic"] = str(cf.source)

        if bool(OPT_FUTURES_CONFIG.get("COST_FORECAST_DYNAMIC", False)):
            aligned["execution_cost_bps_2d"] = dynamic_bps_2d
            aligned["execution_cost_fraction_2d"] = dynamic_frac_2d
            if cf.capacity_notional_2d is not None:
                aligned["capacity_notional_2d"] = np.ascontiguousarray(cf.capacity_notional_2d)
            aligned["_cost_forecast_dynamic"] = 1.0
    except Exception as exc:
        _logger.debug("_attach_execution_cost_bps_2d: dynamic forecast fallback to static: %s", exc)


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
        if sym_df is None or sym_df.empty or "datetime" not in sym_df.columns:
            continue
        # 첫 bar(= fetch_start 포함 warmup 시작점)을 anchor로 사용.
        # is_start_idx 기반 anchor는 warmup 기간을 panel에서 제외하여
        # total_months < needed → adj_train + folds=1 붕괴를 유발함.
        is_start_dts_per_sym[sym] = sym_df["datetime"].iloc[0]

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
        "EV_HURDLE_BPS": float(ml.get("EV_HURDLE_BPS", default_ev_hurdle_bps(cfg))),
        "SLIPPAGE_BPS_BUFFER_MULT": float(ml.get("SLIPPAGE_BPS_BUFFER_MULT", cfg.get("SLIPPAGE_BPS_BUFFER_MULT", 1.0))),
        "TIME_BARRIER_H": float(ml.get("TIME_BARRIER_H", cfg.get("FUTURES_DEFAULT_TIME_BARRIER_H", 0.0))),
        "PORTFOLIO_KAPPA": float(ml.get("PORTFOLIO_KAPPA", cfg.get("FUTURES_PORTFOLIO_KAPPA", 0.35))),
        "FUTURES_EXECUTION_MODE": str(
            ml.get("FUTURES_EXECUTION_MODE") or OPT_FUTURES_CONFIG.get("FUTURES_EXECUTION_MODE", "coarse")
        ),
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
        emb = int(EMBARGO_BARS.get(ctx.tf, 12))
        t_align = time.perf_counter()
        info = compute_multi_alignment_info(ctx.data_maps, ctx.symbols, ctx.tf, emb)
        t_align_total += time.perf_counter() - t_align
        _logger.debug(
            "[PROF] compute_multi_alignment_info elapsed_s=%.4f",
            time.perf_counter() - t_align,
        )
        if info is None:
            raise RuntimeError("multi alignment info unavailable")
        alignment_info: dict[str, Any] = info
        ctx.multi_alignment_info = alignment_info

        eff_len = int(alignment_info["eff_ref_len"])
        embargo = int(EMBARGO_BARS.get(ctx.tf, 12))
        k_legs = int(OPT_FUTURES_CONFIG.get("FUTURES_AWF_K_LEGS", 6))
        is_pool = float(OPT_FUTURES_CONFIG.get("FUTURES_AWF_IS_POOL_FRAC", 0.70))
        awf_legs = build_anchored_wf_legs(eff_len, k=k_legs, embargo=embargo, is_pool_frac=is_pool)

        lookback = cov_lookback_bars(ctx.tf, OPT_FUTURES_CONFIG)
        close_2d_full = np.zeros((eff_len, len(ctx.symbols)), dtype=np.float64)
        for s_idx, sym in enumerate(ctx.symbols):
            if sym in ctx.data_maps and ctx.tf in ctx.data_maps[sym]:
                start_idx = alignment_info["alignment_offsets"][sym]
                close_2d_full[:, s_idx] = (
                    ctx.data_maps[sym][ctx.tf]["close"].iloc[start_idx : start_idx + eff_len].to_numpy(dtype=np.float64)
                )
        t_cov = time.perf_counter()
        sigma_3d_full = precompute_rolling_covariances(close_2d_full, lookback)
        t_cov_total += time.perf_counter() - t_cov
        _logger.debug(
            "[PROF] precompute_rolling_covariances elapsed_s=%.4f",
            time.perf_counter() - t_cov,
        )
        beta_2d_full = _build_beta_2d_full(
            ctx.data_maps,
            ctx.symbols,
            ctx.tf,
            alignment_info,
            eff_len,
        )

        int(awf_legs[0][1]) if awf_legs else max(1, int(eff_len * is_pool))

        dates_ok = bool(ctx.ml_pipeline_fetch_start and ctx.ml_pipeline_end)
        use_full_leg_ml = dates_ok and bool(awf_legs)

        ctx.ml_pipeline_workers or max(1, min(8, len(ctx.symbols)))

        leg_refit_slices: list[dict[str, Any]] | None = None
        last_est_b = 1.05

        if use_full_leg_ml:
            # Legacy anchored ML refit path removed.
            # (Currently, candidate_ml does not use precomputed panel cache during optimization precompute)
            use_full_leg_ml = False

        if leg_refit_slices is not None:
            ctx.awf_leg_slices = leg_refit_slices
            ctx.estimated_b = last_est_b

            t_prebuilt = time.perf_counter()
            prebuilt_full_is = _build_prebuilt_full_arrays(
                ctx.data_maps,
                ctx.symbols,
                ctx.tf,
                alignment_info,
            )
            ctx.is_slice = _build_aligned_2d_from_prebuilt(
                prebuilt_full_is, ctx.symbols, 0, eff_len, sigma_3d_full=sigma_3d_full
            )
            _attach_risk_snapshot_slice(
                ctx.is_slice,
                sigma_3d_full,
                beta_2d_full,
                0,
                eff_len,
                close_2d_full=close_2d_full,
                symbols=list(ctx.symbols),
            )
            _attach_execution_cost_bps_2d(
                aligned=ctx.is_slice,
                prebuilt_arrays=prebuilt_full_is,
                symbols=ctx.symbols,
                slice_start=0,
                slice_end=eff_len,
            )
            t_prebuilt_total += time.perf_counter() - t_prebuilt
        else:
            ctx.estimated_b = 1.05

            t_prebuilt = time.perf_counter()
            prebuilt_full = _build_prebuilt_full_arrays(
                ctx.data_maps,
                ctx.symbols,
                ctx.tf,
                alignment_info,
            )

            ctx.is_slice = _build_aligned_2d_from_prebuilt(
                prebuilt_full, ctx.symbols, 0, eff_len, sigma_3d_full=sigma_3d_full
            )
            _attach_risk_snapshot_slice(
                ctx.is_slice,
                sigma_3d_full,
                beta_2d_full,
                0,
                eff_len,
                close_2d_full=close_2d_full,
                symbols=list(ctx.symbols),
            )
            _attach_execution_cost_bps_2d(
                aligned=ctx.is_slice,
                prebuilt_arrays=prebuilt_full,
                symbols=ctx.symbols,
                slice_start=0,
                slice_end=eff_len,
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
                    _al_nz2_slice = float(np.mean(_al_nz2_slice_list)) if _al_nz2_slice_list else 0.0
                    _as_nz2_slice = float(np.mean(_as_nz2_slice_list)) if _as_nz2_slice_list else 0.0
                    _logger.debug(
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
                _attach_risk_snapshot_slice(
                    aligned,
                    sigma_3d_full,
                    beta_2d_full,
                    test_s,
                    test_e,
                    close_2d_full=close_2d_full,
                    symbols=list(ctx.symbols),
                )
                _attach_execution_cost_bps_2d(
                    aligned=aligned,
                    prebuilt_arrays=prebuilt_full,
                    symbols=ctx.symbols,
                    slice_start=test_s,
                    slice_end=test_e,
                )
                ctx.awf_leg_slices.append({"leg_range": (test_s, test_e), "data": aligned})

        ctx.holdout_slice = None

        alignment_info["awf_legs"] = awf_legs

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


def rerun_precompute_for_ctx(ctx: MLPhaseDContext) -> None:
    """Force regeneration of anchored AWF caches (different seed/calibration path)."""
    ctx.awf_leg_slices = None
    ctx.multi_alignment_info = None
    ctx.calibrator = None
    ctx.calibrator_short = None
    precompute_ml_optimization_context(ctx)
