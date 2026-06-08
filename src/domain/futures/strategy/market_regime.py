from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from src.domain.futures.strategy.common.alignment import AlignedMarketData

if TYPE_CHECKING:
    from src.domain.futures.strategy.config import RegimeConfig

_REGIME_NAMES = (
    "bull_quiet",
    "bull_volatile",
    "bear_quiet",
    "bear_volatile",
    "transition",
    "crash",
)
_DEFAULT_BARS_PER_YEAR = 365.0 * 6.0
_EPS = 1e-12


def _ema_1d(values: NDArray[np.float64], span: int) -> NDArray[np.float64]:
    alpha = 2.0 / (float(span) + 1.0)
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for idx in range(1, values.shape[0]):
        cur = values[idx]
        prev = out[idx - 1]
        out[idx] = cur if not np.isfinite(prev) else (alpha * cur) + ((1.0 - alpha) * prev)
    return out


def _rolling_std_1d(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for idx in range(values.shape[0]):
        start = max(0, idx - window + 1)
        finite = values[start : idx + 1]
        finite = finite[np.isfinite(finite)]
        if finite.size > 0:
            out[idx] = float(np.std(finite, ddof=0))
    return out


def _zscore_1d(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    mean = np.full(values.shape[0], np.nan, dtype=np.float64)
    std = np.full(values.shape[0], np.nan, dtype=np.float64)
    for idx in range(values.shape[0]):
        start = max(0, idx - window + 1)
        finite = values[start : idx + 1]
        finite = finite[np.isfinite(finite)]
        if finite.size > 0:
            mean[idx] = float(np.mean(finite))
            std[idx] = float(np.std(finite, ddof=0))
    return (values - mean) / np.maximum(std, _EPS)


def _expanding_quantile_causal(values: NDArray[np.float64], q: float) -> NDArray[np.float64]:
    """Causal expanding q-th quantile — no lookahead. q in [0, 1].

    Args:
        values: 1-D float64 array of input values.
        q: Quantile to compute, in [0, 1]. q=0.5 is equivalent to the median.

    Returns:
        Array of same shape where out[i] = q-th percentile of values[:i+1]
        ignoring non-finite. Entries remain NaN if no finite value has been seen.

    Time complexity: O(T²·logT) — expanding prefix sort per step.
    Space complexity: O(T).
    """
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for idx in range(values.shape[0]):
        sample = values[: idx + 1]
        finite = sample[np.isfinite(sample)]
        if finite.size > 0:
            out[idx] = float(np.percentile(finite, q * 100.0))
    return out


def _infer_bars_per_year(datetimes: NDArray[np.datetime64]) -> float:
    if datetimes.shape[0] < 2:
        return _DEFAULT_BARS_PER_YEAR
    seconds = datetimes.astype("datetime64[s]").astype(np.int64)
    diffs = np.diff(seconds)
    finite = diffs[diffs > 0]
    if finite.size == 0:
        return _DEFAULT_BARS_PER_YEAR
    median_seconds = float(np.median(finite))
    if not np.isfinite(median_seconds) or median_seconds <= 0.0:
        return _DEFAULT_BARS_PER_YEAR
    return (365.0 * 24.0 * 60.0 * 60.0) / median_seconds


def _btc_index(symbols: tuple[str, ...]) -> int:
    for idx, symbol in enumerate(symbols):
        if "BTC" in symbol.upper():
            return idx
    return 0


def _btc_log_returns(aligned: AlignedMarketData) -> NDArray[np.float64]:
    close = np.asarray(aligned.close_2d, dtype=np.float64)
    btc_close = np.maximum(close[:, _btc_index(aligned.symbols)], _EPS)
    returns = np.zeros(btc_close.shape[0], dtype=np.float64)
    returns[1:] = np.diff(np.log(btc_close))
    return returns


def _expanding_robust_location_scale(
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    location = np.zeros(values.shape[0], dtype=np.float64)
    scale = np.ones(values.shape[0], dtype=np.float64)
    for idx in range(values.shape[0]):
        sample = values[: idx + 1]
        finite = sample[np.isfinite(sample)]
        if finite.size == 0:
            continue
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        location[idx] = median
        scale[idx] = max(1.4826 * mad, _EPS)
    return location, scale


def _cusum_thresholds(target_arl_bars: int) -> tuple[float, float, int]:
    tail_z = float(norm.isf(1.0 / (2.0 * max(target_arl_bars, 2))))
    drift = max(0.25 * tail_z, 0.1)
    threshold = max(1.8 * tail_z, 1.0)
    hold_bars = max(6, round(math.log1p(float(target_arl_bars))))
    return drift, threshold, hold_bars


def _compute_dispersion_z(aligned: AlignedMarketData, window: int) -> NDArray[np.float64]:
    close = np.asarray(aligned.close_2d, dtype=np.float64)
    log_ret = np.zeros_like(close, dtype=np.float64)
    log_ret[1:] = np.diff(np.log(np.maximum(close, _EPS)), axis=0)
    dispersion = np.nanstd(log_ret, axis=1, ddof=0)
    return _zscore_1d(dispersion, window)


def _continuous_regime_codes(
    *,
    trend_snr: NDArray[np.float64],
    vol_scale: NDArray[np.float64],
    crisis_active: NDArray[np.bool_],
    trend_band_arr: NDArray[np.float64] | None = None,
    vol_threshold: NDArray[np.float64] | None = None,
) -> NDArray[np.int8]:
    """Assign 6-state discrete regime codes from continuous overlay signals.

    Args:
        trend_snr: Trend signal-to-noise ratio array [T].
        vol_scale: Volatility scaling factor array [T].
        crisis_active: Boolean crisis flag array [T].
        trend_band_arr: Per-bar transition half-width array [T]. Bars where
            ``|trend_snr[t]| < trend_band_arr[t]`` are assigned transition (code 4).
            If None, falls back to fixed 0.5 for all bars (conservative default).
        vol_threshold: Per-bar adaptive threshold for quiet/volatile split [T].
            If None, falls back to fixed value 1.0 (original behaviour).

    Returns:
        int8 array of regime codes [T]:
            0=bull_quiet, 1=bull_volatile, 2=bear_quiet, 3=bear_volatile,
            4=transition, 5=crash.

    Time complexity: O(T). Space complexity: O(T).
    """
    code = np.full(trend_snr.shape[0], 4, dtype=np.int8)  # transition = default
    finite = np.isfinite(trend_snr) & np.isfinite(vol_scale)
    tb: NDArray[np.float64] = (
        trend_band_arr
        if trend_band_arr is not None
        else np.full(trend_snr.shape[0], 0.5, dtype=np.float64)
    )
    decisive = finite & (np.abs(trend_snr) >= tb)  # band 밖만 방향 확정
    bull = trend_snr >= 0.0
    # adaptive threshold: vol_threshold 제공 시 사용, 아니면 1.0 고정
    vt: NDArray[np.float64] = (
        vol_threshold if vol_threshold is not None else np.ones_like(vol_scale)
    )
    quiet = vol_scale >= vt
    code[decisive & bull & quiet] = 0
    code[decisive & bull & ~quiet] = 1
    code[decisive & ~bull & quiet] = 2
    code[decisive & ~bull & ~quiet] = 3
    # |trend_snr| < trend_band_arr인 finite 구간은 code 4(transition) 유지
    code[crisis_active] = 5
    return code


def _dwell_median(code_1d: NDArray[np.int8]) -> float:
    if code_1d.size == 0:
        return 0.0
    dwell: list[int] = []
    run = 1
    for idx in range(1, code_1d.shape[0]):
        if int(code_1d[idx]) == int(code_1d[idx - 1]):
            run += 1
            continue
        dwell.append(run)
        run = 1
    dwell.append(run)
    return float(np.median(np.asarray(dwell, dtype=np.float64)))


def _weighted_tstat(diff: NDArray[np.float64], min_n_eff: int) -> float:
    finite = diff[np.isfinite(diff)]
    n_eff = finite.shape[0]
    if n_eff < min_n_eff or n_eff < 2:
        return 0.0
    std = float(np.std(finite, ddof=1))
    if std <= _EPS:
        return 0.0
    return float(np.mean(finite) / (std / math.sqrt(float(n_eff))))


def _safe_regime_cfg(cfg: RegimeConfig | None) -> RegimeConfig:
    if cfg is not None:
        return cfg
    from src.domain.futures.strategy.config import RegimeConfig as RuntimeRegimeConfig

    return RuntimeRegimeConfig()


def _clone_aligned_with_close(
    aligned: AlignedMarketData,
    close_2d: NDArray[np.float64],
) -> AlignedMarketData:
    return AlignedMarketData(
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        open_2d=aligned.open_2d,
        high_2d=aligned.high_2d,
        low_2d=aligned.low_2d,
        close_2d=close_2d,
        volume_2d=aligned.volume_2d,
        funding_2d=aligned.funding_2d,
        active_mask=aligned.active_mask,
        warm_mask=aligned.warm_mask,
        entry_block_mask=aligned.entry_block_mask,
        kill_mask=aligned.kill_mask,
        basis_2d=aligned.basis_2d,
        oi_2d=aligned.oi_2d,
        adv_usdt_2d=aligned.adv_usdt_2d,
        execution_cost_bps_2d=aligned.execution_cost_bps_2d,
        inference_active_mask=aligned.inference_active_mask,
        inference_entry_warm_mask=aligned.inference_entry_warm_mask,
        vol_30d_1d=aligned.vol_30d_1d,
        friction_score_1d=aligned.friction_score_1d,
        alpha_capacity_score_1d=aligned.alpha_capacity_score_1d,
        diversification_score_1d=aligned.diversification_score_1d,
        tradeable_score_1d=aligned.tradeable_score_1d,
        cluster_id_1d=aligned.cluster_id_1d,
        beta_vs_market_1d=aligned.beta_vs_market_1d,
        cluster_size_1d=aligned.cluster_size_1d,
        anchor_cluster_1d=aligned.anchor_cluster_1d,
        symbol_meta=aligned.symbol_meta,
    )


@dataclass(slots=True, frozen=True)
class RiskOverlayContext:
    vol_scale_1d: NDArray[np.float64]
    trend_scale_1d: NDArray[np.float64]
    crisis_active_1d: NDArray[np.bool_]
    overlay_mult_1d: NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class MarketRegimeContext:
    code_1d: NDArray[np.int8]
    name_by_code: tuple[str, ...]
    trend_score_1d: NDArray[np.float64]
    vol_z_1d: NDArray[np.float64]
    dispersion_z_1d: NDArray[np.float64]

    def names(self) -> NDArray[np.object_]:
        return np.asarray([self.name_by_code[int(code)] for code in self.code_1d], dtype=object)


@dataclass(slots=True, frozen=True)
class RegimeQualityReport:
    persistence_dwell: float
    leakage_ok: bool
    overlay_lift_bps: float
    overlay_lift_tstat: float
    crisis_precision_ok: bool
    passed: bool
    reasons: tuple[str, ...]


def compute_risk_overlay(
    *,
    aligned: AlignedMarketData,
    cfg: RegimeConfig | None = None,
) -> RiskOverlayContext:
    """Return a causal, self-calibrating continuous risk overlay."""
    regime_cfg = _safe_regime_cfg(cfg)
    btc_log_ret = _btc_log_returns(aligned)
    bars_per_year = _infer_bars_per_year(aligned.datetimes)

    vol_mean = _ema_1d(btc_log_ret, regime_cfg.overlay_vol_ewma_span)
    vol_var = _ema_1d(np.square(btc_log_ret - vol_mean), regime_cfg.overlay_vol_ewma_span)
    realized_vol = np.sqrt(np.maximum(vol_var, 0.0) * bars_per_year)
    realized_vol = np.maximum(realized_vol, _EPS)
    vol_scale = np.clip(
        regime_cfg.overlay_target_vol_ann / realized_vol,
        regime_cfg.overlay_vol_scale_clip[0],
        regime_cfg.overlay_vol_scale_clip[1],
    )

    btc_close = np.maximum(aligned.close_2d[:, _btc_index(aligned.symbols)], _EPS)
    trend_anchor = _ema_1d(np.log(btc_close), regime_cfg.overlay_trend_snr_span)
    trend_score = np.log(btc_close) - trend_anchor
    trend_std = _rolling_std_1d(trend_score, regime_cfg.overlay_trend_snr_span)
    trend_snr = trend_score / np.maximum(trend_std, _EPS)
    trend_scale = 0.5 * (1.0 + np.tanh(np.nan_to_num(trend_snr, nan=0.0, posinf=6.0, neginf=-6.0)))

    robust_mu, robust_sigma = _expanding_robust_location_scale(btc_log_ret)
    standardized = (btc_log_ret - robust_mu) / np.maximum(robust_sigma, _EPS)
    standardized = np.clip(np.nan_to_num(standardized, nan=0.0), -10.0, 10.0)
    drift, threshold, hold_bars = _cusum_thresholds(regime_cfg.crisis_target_arl_bars)

    pos_cusum = 0.0
    neg_cusum = 0.0
    cooldown = 0
    crisis_active = np.zeros(standardized.shape[0], dtype=bool)
    for idx, residual in enumerate(standardized):
        pos_cusum = max(0.0, pos_cusum + residual - drift)
        neg_cusum = max(0.0, neg_cusum - residual - drift)
        if pos_cusum > threshold or neg_cusum > threshold:
            cooldown = hold_bars
            pos_cusum = 0.0
            neg_cusum = 0.0
        if cooldown > 0:
            crisis_active[idx] = True
            cooldown -= 1

    overlay_raw = vol_scale * trend_scale
    overlay_mult = np.where(
        crisis_active,
        regime_cfg.crisis_gross_floor,
        overlay_raw,
    )
    return RiskOverlayContext(
        vol_scale_1d=vol_scale.astype(np.float64, copy=False),
        trend_scale_1d=trend_scale.astype(np.float64, copy=False),
        crisis_active_1d=crisis_active.astype(np.bool_, copy=False),
        overlay_mult_1d=overlay_mult.astype(np.float64, copy=False),
    )


def compute_market_regime_context(
    *,
    aligned: AlignedMarketData,
    cfg: RegimeConfig | None = None,
) -> MarketRegimeContext:
    close = np.asarray(aligned.close_2d, dtype=np.float64)
    if close.ndim != 2 or close.shape[0] == 0:
        raise ValueError("aligned.close_2d must be non-empty 2D array")

    regime_cfg = _safe_regime_cfg(cfg)
    overlay = compute_risk_overlay(aligned=aligned, cfg=regime_cfg)
    btc_close = np.maximum(close[:, _btc_index(aligned.symbols)], _EPS)
    trend_anchor = _ema_1d(np.log(btc_close), regime_cfg.overlay_trend_snr_span)
    trend_score = np.log(btc_close) - trend_anchor
    trend_std = _rolling_std_1d(trend_score, regime_cfg.overlay_trend_snr_span)
    trend_snr = trend_score / np.maximum(trend_std, _EPS)

    btc_log_ret = _btc_log_returns(aligned)
    bars_per_year = _infer_bars_per_year(aligned.datetimes)
    vol_mean = _ema_1d(btc_log_ret, regime_cfg.overlay_vol_ewma_span)
    vol_var = _ema_1d(np.square(btc_log_ret - vol_mean), regime_cfg.overlay_vol_ewma_span)
    realized_vol = np.sqrt(np.maximum(vol_var, 0.0) * bars_per_year)
    vol_z = _zscore_1d(np.log(np.maximum(realized_vol, _EPS)), regime_cfg.overlay_trend_snr_span)
    dispersion_z = _compute_dispersion_z(aligned, regime_cfg.overlay_trend_snr_span)

    # causal expanding median of vol_scale as adaptive threshold for quiet/volatile split
    vol_median = _expanding_quantile_causal(overlay.vol_scale_1d, 0.5)
    # fallback to 1.0 where insufficient data (< regime_min_n_eff bars)
    min_n = regime_cfg.regime_min_n_eff
    vol_threshold = np.where(
        np.arange(vol_median.shape[0]) >= min_n,
        vol_median,
        np.ones_like(vol_median),
    ).astype(np.float64)

    # causal per-bar transition band (percentile of |trend_snr|)
    abs_snr = np.abs(np.nan_to_num(trend_snr, nan=0.0))
    raw_band = _expanding_quantile_causal(abs_snr, regime_cfg.regime_transition_occupancy)
    trend_band_arr = np.where(
        np.arange(raw_band.shape[0]) >= min_n,
        raw_band,
        np.full_like(raw_band, 0.5),
    ).astype(np.float64)

    code = _continuous_regime_codes(
        trend_snr=np.nan_to_num(trend_snr, nan=0.0),
        vol_scale=overlay.vol_scale_1d,
        crisis_active=overlay.crisis_active_1d,
        trend_band_arr=trend_band_arr,
        vol_threshold=vol_threshold,
    )

    return MarketRegimeContext(
        code_1d=code,
        name_by_code=_REGIME_NAMES,
        trend_score_1d=trend_snr.astype(np.float64, copy=False),
        vol_z_1d=vol_z.astype(np.float64, copy=False),
        dispersion_z_1d=dispersion_z.astype(np.float64, copy=False),
    )

def evaluate_regime_quality(
    *,
    aligned: AlignedMarketData,
    cfg: RegimeConfig | None = None,
    base_edge_1d: NDArray[np.float64],
    cal_eval_mask: NDArray[np.bool_],
    is_mask: NDArray[np.bool_] | None = None,
) -> RegimeQualityReport:
    """Evaluate regime overlay quality on the cal-eval slice only."""
    del is_mask
    regime_cfg = _safe_regime_cfg(cfg)
    overlay = compute_risk_overlay(aligned=aligned, cfg=regime_cfg)
    regime_ctx = compute_market_regime_context(aligned=aligned, cfg=regime_cfg)

    cal_mask = np.asarray(cal_eval_mask, dtype=bool)
    if cal_mask.shape[0] != base_edge_1d.shape[0]:
        raise ValueError("cal_eval_mask and base_edge_1d must have identical length")

    base = np.asarray(base_edge_1d, dtype=np.float64)
    base_cal = base[cal_mask]
    overlaid_cal = base_cal * overlay.overlay_mult_1d[cal_mask]

    def _safe_sharpe(arr: NDArray[np.float64]) -> float:
        """Compute mean/std Sharpe; returns 0.0 if insufficient finite data."""
        finite = arr[np.isfinite(arr)]
        if finite.size < 2:
            return 0.0
        return float(np.mean(finite) / max(float(np.std(finite, ddof=1)), _EPS))

    sharpe_base = _safe_sharpe(base_cal)
    sharpe_overlaid = _safe_sharpe(overlaid_cal)
    # risk-adjusted lift: Sharpe(overlaid) - Sharpe(base), scaled to bps for reporting
    overlay_lift_bps = (sharpe_overlaid - sharpe_base) * 1e4
    # tstat: directional significance of (overlaid - base) difference series
    overlay_diff = overlaid_cal - base_cal
    overlay_lift_tstat = _weighted_tstat(overlay_diff, regime_cfg.regime_min_n_eff)

    crisis_mask = cal_mask & overlay.crisis_active_1d
    normal_mask = cal_mask & ~overlay.crisis_active_1d
    crisis_mean = float(np.nanmean(base[crisis_mask])) if np.any(crisis_mask) else 0.0
    normal_mean = float(np.nanmean(base[normal_mask])) if np.any(normal_mask) else 0.0
    crisis_precision_ok = crisis_mean < normal_mean

    pivot = max(1, base.shape[0] // 2)
    perturbed_close = np.array(aligned.close_2d, dtype=np.float64, copy=True)
    perturbed_close[pivot + 1 :, :] *= 1.15
    perturbed = _clone_aligned_with_close(aligned, perturbed_close)
    leak_check = compute_risk_overlay(aligned=perturbed, cfg=regime_cfg)
    leakage_ok = bool(
        np.allclose(
            overlay.overlay_mult_1d[: pivot + 1],
            leak_check.overlay_mult_1d[: pivot + 1],
            atol=1e-12,
            rtol=0.0,
            equal_nan=True,
        )
    )

    regime_dwell = _dwell_median(regime_ctx.code_1d[cal_mask])
    crisis_dwell = _dwell_median(overlay.crisis_active_1d[cal_mask].astype(np.int8, copy=False))
    persistence_dwell = max(regime_dwell, crisis_dwell)
    reasons: list[str] = []
    if persistence_dwell < 6.0:
        reasons.append("dwell_below_threshold")
    if not leakage_ok:
        reasons.append("leakage_detected")
    if overlay_lift_bps <= 0.0:
        reasons.append("overlay_lift_non_positive")
    if overlay_lift_tstat < regime_cfg.regime_overlay_min_lift_tstat:
        reasons.append("overlay_lift_below_threshold")
    if not crisis_precision_ok:
        reasons.append("crisis_precision_failed")
    if np.isfinite(base[cal_mask]).sum() < regime_cfg.regime_min_n_eff:
        reasons.append("insufficient_cal_eval_obs")

    passed = not reasons if regime_cfg.regime_quality_gate_enabled else True
    report = RegimeQualityReport(
        persistence_dwell=persistence_dwell,
        leakage_ok=leakage_ok,
        overlay_lift_bps=overlay_lift_bps,
        overlay_lift_tstat=overlay_lift_tstat,
        crisis_precision_ok=crisis_precision_ok,
        passed=passed,
        reasons=tuple(reasons),
    )
    from src.domain.futures.strategy.rule_diagnostics import log_regime_quality_report

    log_regime_quality_report(report)
    return report
