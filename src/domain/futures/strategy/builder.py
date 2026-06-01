from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

from src.domain.futures.legacy.strategy_sleev.combine import blend_sleeves
from src.domain.futures.legacy.strategy_sleev.normalize import (
    to_return_units,
    winsorized_cs_zscore,
)
from src.domain.futures.legacy.strategy_sleev.sleeves.carry import CarrySleeve
from src.domain.futures.legacy.strategy_sleev.sleeves.ts_momentum import TSMomentumSleeve
from src.domain.futures.legacy.strategy_sleev.sleeves.xs_reversal import XSReversalSleeve
from src.domain.futures.optimization.optimizer import compute_multi_alignment_info
from src.domain.futures.strategy.config import StrategyConfig
from src.domain.futures.strategy.diagnostics import ic_summary, passes_ic_gate, rolling_ic

_logger = logging.getLogger(__name__)


def _assert_no_legacy_imports() -> None:
    """Guard against deprecated heavy modules in runtime."""
    if any(name.startswith("src.domain.futures.alpha_factory") for name in sys.modules):
        raise RuntimeError("alpha_factory import forbidden in strategy module")


def build_strategy_alpha(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    """Build long-format alpha panel from aligned multi-sleeve expected return signals.

    Args:
        data_maps: Dictionary containing historical data per symbol.
        symbols: List of target symbols.
        tf: Timeframe string.
        cfg: Top-level StrategyConfig.

    Returns:
        pd.DataFrame sorted by (datetime, symbol) with columns ["alpha_long", "alpha_short"].

    """
    _assert_no_legacy_imports()
    if cfg.name in {"candidate_ml", "rule_baseline"}:
        from src.domain.futures.strategy_runtime.bridge import run_candidate_strategy_for_universe
        res = run_candidate_strategy_for_universe(
            symbols=symbols,
            tf=tf,
            strategy_cfg=cfg,
            preloaded_data_maps=data_maps,
        )
        return res.alpha_panel

    # 1. Align price panels (using compute_multi_alignment_info base)
    info = compute_multi_alignment_info(data_maps, symbols, tf, embargo=0)
    if info is None:
        return pd.DataFrame(columns=["alpha_long", "alpha_short"])

    eff_len = int(info["eff_ref_len"])
    offsets: dict[str, int] = info["alignment_offsets"]
    valid_symbols = [
        sym for sym in symbols if sym in offsets and sym in data_maps and tf in data_maps[sym]
    ]

    min_syms = cfg.blend.min_symbols
    if len(valid_symbols) < min_syms:
        raise ValueError(f"strategy needs >= {min_syms} symbols, got {len(valid_symbols)}")

    close_2d = np.zeros((eff_len, len(valid_symbols)), dtype=np.float64)
    funding_2d = np.zeros((eff_len, len(valid_symbols)), dtype=np.float64)
    datetimes: np.ndarray | None = None

    for col_idx, sym in enumerate(valid_symbols):
        df = data_maps[sym][tf]
        start_idx = offsets[sym]
        end_idx = start_idx + eff_len
        close_2d[:, col_idx] = df["close"].iloc[start_idx:end_idx].to_numpy(dtype=np.float64)
        if "funding_rate_sum" in df.columns:
            funding_2d[:, col_idx] = (
                df["funding_rate_sum"].iloc[start_idx:end_idx].to_numpy(dtype=np.float64)
            )
        elif "funding_rate" in df.columns:
            funding_2d[:, col_idx] = (
                df["funding_rate"].iloc[start_idx:end_idx].to_numpy(dtype=np.float64)
            )
        else:
            funding_2d[:, col_idx] = 0.0

        if datetimes is None:
            datetimes = df["datetime"].iloc[start_idx:end_idx].to_numpy()

    if datetimes is None:
        return pd.DataFrame(columns=["alpha_long", "alpha_short"])

    aux = {"funding_2d": funding_2d}

    # 2. Instantiate and compute active sleeves
    sleeves: list[Any] = []
    if cfg.name == "xs_reversal":
        sleeves.append(XSReversalSleeve(lookback_bars=cfg.sleeves.reversal_lookback))
    else:
        if cfg.sleeves.reversal_enabled:
            sleeves.append(XSReversalSleeve(lookback_bars=cfg.sleeves.reversal_lookback))
        if cfg.sleeves.ts_momentum_enabled:
            sleeves.append(
                TSMomentumSleeve(
                    lookback_bars=cfg.sleeves.ts_momentum_lookback,
                    skip_bars=cfg.sleeves.ts_momentum_skip,
                )
            )
        if cfg.sleeves.carry_enabled:
            sleeves.append(CarrySleeve(smooth_bars=cfg.sleeves.carry_smooth))

    if not sleeves:
        # Fallback to XS Reversal if no sleeve is enabled
        sleeves.append(XSReversalSleeve(lookback_bars=cfg.sleeves.reversal_lookback))

    z_by_sleeve: dict[str, np.ndarray] = {}
    for slv in sleeves:
        raw_sig = slv.compute_raw(close_2d, aux)
        z_by_sleeve[slv.name] = winsorized_cs_zscore(
            raw_sig, clip_z=cfg.blend.clip_z, min_symbols=min_syms
        )

    # 3. Calculate forward returns for IC evaluations
    fwd_ret_2d = np.full_like(close_2d, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd_ret_2d[:-1] = close_2d[1:] / np.maximum(close_2d[:-1], 1e-12) - 1.0

    # Precompute rolling IC for each active sleeve
    ic_by_sleeve: dict[str, np.ndarray] = {}
    for name, z_sig in z_by_sleeve.items():
        ic_by_sleeve[name] = rolling_ic(z_sig, fwd_ret_2d, method="spearman")

    # 4. Dynamic Blending and Selection Loop (No Look-Ahead)
    blended_scores = np.zeros_like(close_2d)
    w_bars = cfg.blend.ic_window_bars
    sleeve_names = list(z_by_sleeve.keys())

    # Standard warning counter to avoid logging spam
    fallback_warning_triggered = False

    for t in range(eff_len):
        # If t < w_bars, fallback to equal-weight blend due to short history.
        if t < w_bars:
            equal_weights = dict.fromkeys(sleeve_names, 1.0)
            t_z = {name: z_by_sleeve[name][t : t + 1] for name in sleeve_names}
            blended_scores[t] = blend_sleeves(
                t_z, equal_weights, clip_z=cfg.blend.clip_z, min_symbols=min_syms
            )[0]
            continue

        # Valid sliding window: [t - w_bars, t - 1]
        active_weights: dict[str, float] = {}
        passes_count = 0
        summaries: dict[str, dict[str, float]] = {}

        for name in sleeve_names:
            window_ic = ic_by_sleeve[name][t - w_bars : t]
            summary = ic_summary(window_ic)
            summaries[name] = summary

            if passes_ic_gate(
                summary,
                min_mean_ic=cfg.blend.min_mean_ic,
                min_t_stat=cfg.blend.min_t_stat,
                min_hit_ratio=cfg.blend.min_hit_ratio,
            ):
                active_weights[name] = cfg.blend.ic_shrinkage * summary["mean_ic"]
                passes_count += 1

        if passes_count == 0:
            # Fallback to single highest absolute mean_ic sleeve
            best_sleeve = sleeve_names[0]
            best_abs_mean = -1.0
            for name in sleeve_names:
                abs_mean = abs(summaries[name]["mean_ic"])
                if abs_mean > best_abs_mean:
                    best_abs_mean = abs_mean
                    best_sleeve = name

            active_weights = {best_sleeve: 1.0}
            if not fallback_warning_triggered:
                details = ", ".join(
                    f"{n}(mean={s['mean_ic']:.4f}, t={s['t_stat']:.1f}, hit={s['hit_ratio']:.2f})"
                    for n, s in summaries.items()
                )
                _logger.warning(
                    "[strategy] All sleeves failed IC selection gates "
                    "at t=%d; fallback to best sleeve: %s. Details: %s",
                    t,
                    best_sleeve,
                    details,
                )
                fallback_warning_triggered = True

        t_z = {name: z_by_sleeve[name][t : t + 1] for name in sleeve_names}
        blended_scores[t] = blend_sleeves(
            t_z, active_weights, clip_z=cfg.blend.clip_z, min_symbols=min_syms
        )[0]

    # 5. Compute blended score lagged IC (2-pass structure)
    ic_blended = rolling_ic(blended_scores, fwd_ret_2d, method="spearman")
    ic_blended_lagged = np.zeros(eff_len, dtype=np.float64)

    # Initialize first W bars with default hurdle fallback to avoid zero scaling
    ic_blended_lagged[:w_bars] = cfg.blend.min_mean_ic

    for t in range(w_bars, eff_len):
        window_ic = ic_blended[t - w_bars : t]
        mean_ic = np.nanmean(window_ic)
        ic_blended_lagged[t] = mean_ic if np.isfinite(mean_ic) else cfg.blend.min_mean_ic

    # 6. Expected forward volatility computation: per-bar return std
    rets = np.zeros_like(close_2d)
    with np.errstate(divide="ignore", invalid="ignore"):
        rets[1:] = close_2d[1:] / np.maximum(close_2d[:-1], 1e-12) - 1.0

    sigma_fwd = np.zeros_like(close_2d)
    sigma_lb = cfg.blend.sigma_lookback

    for t in range(eff_len):
        if t < sigma_lb:
            available = rets[1 : t + 1]
            if len(available) > 1:
                sigma_fwd[t] = np.nanstd(available, axis=0)
            else:
                sigma_fwd[t] = 0.02
        else:
            sigma_fwd[t] = np.nanstd(rets[t - sigma_lb + 1 : t + 1], axis=0)

    # Apply clipping to avoid division by zero or extreme volatility spikes
    sigma_fwd = np.clip(sigma_fwd, 1e-6, 1.0)

    # 7. Grinold expected return calibration
    alpha_hat = to_return_units(blended_scores, sigma_fwd, ic_blended_lagged)

    # 8. Split into directional components
    alpha_long = np.maximum(alpha_hat, 0.0)
    alpha_short = np.maximum(-alpha_hat, 0.0)

    # 9. Format output DataFrame and run strict checks
    idx = pd.MultiIndex.from_product([datetimes, valid_symbols], names=["datetime", "symbol"])
    panel = pd.DataFrame(
        {
            "alpha_long": alpha_long.reshape(-1),
            "alpha_short": alpha_short.reshape(-1),
        },
        index=idx,
    ).sort_index()

    # Guard: NaN/Inf Check
    vals = panel[["alpha_long", "alpha_short"]].to_numpy(dtype=np.float64)
    bad_mask = ~np.isfinite(vals)
    if bad_mask.any():
        bad_row = int(np.argwhere(bad_mask)[0][0])
        raise RuntimeError(f"alpha_panel contains NaN/Inf at idx={bad_row}")

    # Guard: Sorted MultiIndex Check
    if not panel.index.is_monotonic_increasing:
        raise RuntimeError("alpha_panel must be sorted by (datetime, symbol)")

    # Save meta information
    panel.attrs["strategy_name"] = cfg.name
    panel.attrs["active_sleeves"] = sleeve_names

    # [ALPHA-BUILD] diagnostic log
    _al_vals = alpha_long.ravel()
    _as_vals = alpha_short.ravel()
    _al_nz = float(np.count_nonzero(np.abs(_al_vals) > 1e-12) / max(_al_vals.size, 1))
    _as_nz = float(np.count_nonzero(np.abs(_as_vals) > 1e-12) / max(_as_vals.size, 1))
    _al_finite = _al_vals[np.isfinite(_al_vals)]
    _al_p50 = float(np.nanpercentile(_al_finite, 50)) * 10000.0 if _al_finite.size > 0 else 0.0
    _al_p95 = float(np.nanpercentile(_al_finite, 95)) * 10000.0 if _al_finite.size > 0 else 0.0
    _ic_bl_finite = ic_blended_lagged[np.isfinite(ic_blended_lagged)]
    _ic_lag_mean = float(np.nanmean(_ic_bl_finite)) if _ic_bl_finite.size > 0 else 0.0
    _ic_neg_ratio = (
        float(np.sum(_ic_bl_finite < 0.0) / max(_ic_bl_finite.size, 1))
        if _ic_bl_finite.size > 0
        else 0.0
    )
    n_symbols = len(valid_symbols)
    n_bars_log = int(alpha_long.shape[0])
    _logger.info(
        "[ALPHA-BUILD] sleeves=%s syms=%d bars=%d"
        " long_nz=%.3f short_nz=%.3f long_p50=%.1fbps long_p95=%.1fbps"
        " ic_lag_mean=%.4f ic_neg_ratio=%.3f fallback=%s",
        sleeve_names,
        n_symbols,
        n_bars_log,
        _al_nz,
        _as_nz,
        _al_p50,
        _al_p95,
        _ic_lag_mean,
        _ic_neg_ratio,
        fallback_warning_triggered,
    )

    return panel
