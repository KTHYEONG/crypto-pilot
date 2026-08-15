"""CostForecast builder — Phase 1/2 with static floor and optional dynamic components."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.core.settings import TAKER_FEE_BPS, round_trip_cost_bps
from src.domain.futures.forecast.contracts import CostForecast
from src.domain.futures.portfolio.friction_model import compute_impact_bps

_logger = logging.getLogger(__name__)
_BPS = 1.0 / 10000.0


@dataclass(frozen=True)
class CostModelConfig:
    """Configuration for parametric cost forecast.

    Phase 1 defaults produce the same result as the existing static model.
    Phase 2 activates vol_buffer_coef, funding_event_buffer_bps.

    Attributes:
        taker_fee_bps: Taker fee in basis points.
        latency_buffer_bps: Latency slippage buffer.
        impact_coef: Market impact coefficient.
        vol_buffer_coef: Vol-scaled buffer (Phase 2, default 0.0).
        funding_event_buffer_bps: Funding event buffer (Phase 2, default 0.0).
        adv_lookback: ADV rolling lookback in bars.
        vol_lookback: Volatility rolling lookback in bars.
        estimated_order_notional: Estimated order size in USDT (for impact).
        uncertainty_ratio: Uncertainty as fraction of base cost.

    """

    taker_fee_bps: float = TAKER_FEE_BPS
    latency_buffer_bps: float = 0.5
    impact_coef: float = 0.5
    vol_buffer_coef: float = 0.0
    funding_event_buffer_bps: float = 0.0
    adv_lookback: int = 30
    vol_lookback: int = 20
    estimated_order_notional: float = 0.0
    uncertainty_ratio: float = 0.1
    enable_dynamic_components: bool = False


def _rolling_std_2d(arr_2d: np.ndarray, lookback: int) -> np.ndarray:
    """Rolling std per column using a strict-causal window [start:i]."""
    t, n = arr_2d.shape
    out = np.zeros((t, n), dtype=np.float64)
    lb = max(3, int(lookback))
    for i in range(1, t):
        start = max(0, i - lb)
        w = arr_2d[start:i, :]
        if w.shape[0] >= 2:
            out[i, :] = np.nanstd(w, axis=0, ddof=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _rolling_adv_usdt(
    close_2d: np.ndarray,
    volume_2d: np.ndarray,
    lookback: int,
) -> np.ndarray:
    """Approximate ADV(USDT) from rolling notional turnover."""
    notional = np.maximum(np.asarray(close_2d, dtype=np.float64), 0.0) * np.maximum(
        np.asarray(volume_2d, dtype=np.float64), 0.0
    )
    t, n = notional.shape
    out = np.zeros((t, n), dtype=np.float64)
    lb = max(3, int(lookback))
    for i in range(1, t):
        start = max(0, i - lb)
        w = notional[start:i, :]
        if w.size > 0:
            out[i, :] = np.nanmean(w, axis=0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def build_cost_forecast(
    close_2d: np.ndarray,
    high_2d: np.ndarray | None,
    low_2d: np.ndarray | None,
    volume_2d: np.ndarray,
    funding_2d: np.ndarray | None,
    adv_usdt_2d: np.ndarray | None,
    universe_cost_bps_2d: np.ndarray | None,
    cfg: CostModelConfig,
    *,
    shape: tuple[int, int],
) -> CostForecast:
    """Build per-bar cost forecast.

    Phase 1: static floor behaviour is preserved by default config.
    Phase 2: optional dynamic buffers can be enabled via cfg.

    Args:
        close_2d: Close prices [T, N].
        high_2d: High prices [T, N] (optional, used for spread proxy in Phase 2).
        low_2d: Low prices [T, N] (optional).
        volume_2d: Volume [T, N].
        funding_2d: Funding rate [T, N] (optional, used in Phase 2).
        adv_usdt_2d: 30-day ADV in USDT [T, N] (optional, for impact).
        universe_cost_bps_2d: Per-symbol static cost from universe selection [T, N].
        cfg: Cost model configuration.
        shape: Output (T, N) shape.

    Returns:
        Typed CostForecast.

    """
    t, n = shape
    fallback_bps = float(round_trip_cost_bps())

    if universe_cost_bps_2d is not None and np.asarray(universe_cost_bps_2d).shape == (t, n):
        raw = np.asarray(universe_cost_bps_2d, dtype=np.float64)
        valid = np.isfinite(raw) & (raw > 0.0)
        floor_bps_2d = np.where(valid, raw, fallback_bps)
        base_source = "universe_static"
    else:
        floor_bps_2d = np.full((t, n), fallback_bps, dtype=np.float64)
        base_source = "fallback_global"
        _logger.debug("CostForecast: no per-symbol cost, fallback=%.1fbps", fallback_bps)

    close = np.asarray(close_2d, dtype=np.float64)
    vol = np.asarray(volume_2d, dtype=np.float64)
    high = np.asarray(high_2d, dtype=np.float64) if high_2d is not None else None
    low = np.asarray(low_2d, dtype=np.float64) if low_2d is not None else None

    dyn_spread_bps = np.zeros((t, n), dtype=np.float64)
    if (
        cfg.enable_dynamic_components
        and high is not None
        and low is not None
        and high.shape == (t, n)
        and low.shape == (t, n)
    ):
        # Use a conservative half-spread proxy from range/close.
        dyn_spread_bps = 0.5 * np.maximum(high - low, 0.0) / np.maximum(np.abs(close), 1e-12) * 10000.0
        dyn_spread_bps = np.nan_to_num(dyn_spread_bps, nan=0.0, posinf=0.0, neginf=0.0)

    ret = np.zeros((t, n), dtype=np.float64)
    ret[1:, :] = (close[1:, :] - close[:-1, :]) / np.maximum(np.abs(close[:-1, :]), 1e-12)
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    sigma_2d = _rolling_std_2d(ret, cfg.vol_lookback)

    vol_buffer_bps = np.zeros((t, n), dtype=np.float64)
    if cfg.enable_dynamic_components:
        vol_buffer_bps = np.maximum(float(cfg.vol_buffer_coef), 0.0) * sigma_2d * 10000.0

    adv_2d = (
        np.asarray(adv_usdt_2d, dtype=np.float64)
        if adv_usdt_2d is not None and np.asarray(adv_usdt_2d).shape == (t, n)
        else _rolling_adv_usdt(close, vol, cfg.adv_lookback)
    )
    adv_2d = np.maximum(np.nan_to_num(adv_2d, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    impact_bps = np.zeros((t, n), dtype=np.float64)
    order_notional = float(max(cfg.estimated_order_notional, 0.0))
    if cfg.enable_dynamic_components and order_notional > 0.0:
        # Scalar helper reused per cell to preserve consistency with friction_model.
        for i in range(t):
            for j in range(n):
                impact_bps[i, j] = compute_impact_bps(
                    sigma_1d=float(max(sigma_2d[i, j], 0.0)),
                    order_notional=order_notional,
                    adv_30d=float(max(adv_2d[i, j], 0.0)),
                    k=float(cfg.impact_coef),
                )

    funding_buffer_bps = np.zeros((t, n), dtype=np.float64)
    if cfg.enable_dynamic_components and funding_2d is not None and np.asarray(funding_2d).shape == (t, n):
        f2d = np.asarray(funding_2d, dtype=np.float64)
        funding_buffer_bps = np.abs(np.nan_to_num(f2d, nan=0.0, posinf=0.0, neginf=0.0)) * 10000.0
        funding_buffer_bps = funding_buffer_bps + float(max(cfg.funding_event_buffer_bps, 0.0))

    latency_bps = np.zeros((t, n), dtype=np.float64)
    if cfg.enable_dynamic_components:
        latency_bps = np.full((t, n), float(max(cfg.latency_buffer_bps, 0.0)), dtype=np.float64)

    dynamic_total_bps = dyn_spread_bps + vol_buffer_bps + impact_bps + funding_buffer_bps + latency_bps
    dynamic_total_bps = np.nan_to_num(dynamic_total_bps, nan=0.0, posinf=0.0, neginf=0.0)

    taker_fee_bps_2d = np.full((t, n), float(max(cfg.taker_fee_bps, 0.0)), dtype=np.float64)
    parametric_total_bps = taker_fee_bps_2d + dynamic_total_bps
    parametric_total_bps = np.nan_to_num(parametric_total_bps, nan=0.0, posinf=0.0, neginf=0.0)

    if cfg.enable_dynamic_components:
        bps_2d = np.maximum(floor_bps_2d, parametric_total_bps)
        source = "parametric_dynamic"
    else:
        bps_2d = floor_bps_2d
        source = base_source

    frac_2d = bps_2d * _BPS
    uncertainty_bps_2d = np.maximum(
        0.0,
        _rolling_std_2d(dynamic_total_bps, cfg.vol_lookback) * max(float(cfg.uncertainty_ratio), 0.0),
    )
    if not np.any(uncertainty_bps_2d > 0.0):
        uncertainty_bps_2d = bps_2d * max(float(cfg.uncertainty_ratio), 0.0)

    capacity_notional_2d: np.ndarray | None = None
    if cfg.enable_dynamic_components and order_notional > 0.0:
        capacity_notional_2d = adv_2d * 0.01

    return CostForecast(
        execution_cost_bps_2d=bps_2d,
        execution_cost_fraction_2d=frac_2d,
        uncertainty_bps_2d=uncertainty_bps_2d,
        capacity_notional_2d=capacity_notional_2d,
        source=source,
        components={
            "base_bps": np.asarray(universe_cost_bps_2d, dtype=np.float64)
            if universe_cost_bps_2d is not None and np.asarray(universe_cost_bps_2d).shape == (t, n)
            else np.full((t, n), fallback_bps, dtype=np.float64),
            "taker_fee_bps": taker_fee_bps_2d,
            "parametric_total_bps": parametric_total_bps,
            "spread_proxy_bps": dyn_spread_bps,
            "vol_buffer_bps": vol_buffer_bps,
            "impact_bps": impact_bps,
            "funding_buffer_bps": funding_buffer_bps,
            "latency_bps": latency_bps,
        },
    )
