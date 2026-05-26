"""Friction and transaction cost models for futures trading."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.core.settings import TAKER_FEE_BPS, round_trip_cost_bps

_logger = logging.getLogger("FrictionModel")

_BPS_TO_FRACTION: float = 1.0 / 10000.0


@dataclass(frozen=True)
class CostSnapshot:
    """Canonical cost snapshot used across label/objective compose paths.

    Attributes:
        execution_cost_bps_2d: Effective round-trip execution cost in bps.
        execution_cost_fraction_2d: Same cost tensor in fraction units.
        round_trip_cost_bps_fallback: Global fallback round-trip cost in bps.
        execution_cost_bps_source: Cost source label (per_symbol or fallback_global).

    """

    execution_cost_bps_2d: NDArray[np.float64]
    execution_cost_fraction_2d: NDArray[np.float64]
    round_trip_cost_bps_fallback: float
    execution_cost_bps_source: str


def resolve_cost_snapshot(
    *,
    execution_cost_bps_2d: NDArray[np.float64] | None,
    shape: tuple[int, int],
) -> CostSnapshot:
    """Build canonical cost snapshot with explicit source selection.

    Args:
        execution_cost_bps_2d: Optional per-symbol execution cost tensor in bps.
        shape: Target (T, N) shape for the resolved tensor.

    Returns:
        CostSnapshot with bps and fraction views plus source metadata.

    """
    fallback_bps = float(round_trip_cost_bps())
    use_fallback = (
        execution_cost_bps_2d is None
        or execution_cost_bps_2d.shape != shape
        or not np.any(np.isfinite(execution_cost_bps_2d))
    )
    if use_fallback:
        bps_2d = np.full(shape, fallback_bps, dtype=np.float64)
        source = "fallback_global"
    else:
        bps_2d = np.asarray(execution_cost_bps_2d, dtype=np.float64)
        source = "per_symbol"
    frac_2d = bps_2d * _BPS_TO_FRACTION
    return CostSnapshot(
        execution_cost_bps_2d=bps_2d,
        execution_cost_fraction_2d=frac_2d,
        round_trip_cost_bps_fallback=fallback_bps,
        execution_cost_bps_source=source,
    )


@dataclass(frozen=True)
class FrictionConfig:
    """Configuration for transaction friction modeling.

    Attributes:
        taker_fee_bps: Taker fee in basis points (canonical source: core/settings.TAKER_FEE_BPS).
        maker_share: Proportion of trades executed as maker (0.0 to 1.0).
            실행 시뮬레이터는 양 leg 모두 Taker로 체결하므로 기본값 0.0 (Taker-only 현실 반영).
        maker_rebate_bps: Maker rebate/fee in basis points.
        latency_buffer_bps: Buffer for execution latency slippage.
        k_impact: Market impact scaling factor.
        tick_cost_bps: Fixed tick size cost approximation.

    """

    taker_fee_bps: float = TAKER_FEE_BPS  # canonical source: core/settings.py (5.0bps = 0.05%)
    maker_share: float = 0.0  # execution sim은 양 leg 모두 Taker → 0.0으로 보수적 precharge
    maker_rebate_bps: float = -2.0
    latency_buffer_bps: float = 0.5
    k_impact: float = 0.5
    tick_cost_bps: float = 0.0


def compute_coarse_precharge_bps(
    spread_bps: float,
    impact_bps: float,
    funding_proxy_bps: float,
    cfg: FrictionConfig | None = None,
) -> float:
    """Compute overall transaction friction pre-charge in basis points.

    Args:
        spread_bps: Half-spread of the book depth in bps.
        impact_bps: Estimated price impact in bps.
        funding_proxy_bps: Funding rate drag proxy in bps.
        cfg: Configuration parameters for friction.

    Returns:
        Total friction pre-charge in bps.

    """
    if cfg is None:
        cfg = FrictionConfig()
    fee_bps = cfg.taker_fee_bps + cfg.maker_share * (cfg.maker_rebate_bps - cfg.taker_fee_bps)
    total_bps = (
        fee_bps
        + spread_bps
        + impact_bps
        + cfg.tick_cost_bps
        + cfg.latency_buffer_bps
        + funding_proxy_bps
    )
    return float(total_bps)


def compute_impact_bps(
    sigma_1d: float,
    order_notional: float,
    adv_30d: float,
    k: float = 0.5,
) -> float:
    """Compute square-root market impact in basis points.

    Args:
        sigma_1d: Daily volatility of the asset (standard deviation of daily returns).
        order_notional: Notional size of the order in USDT.
        adv_30d: 30-day average daily volume (ADV) in USDT.
        k: Scaling parameter for impact.

    Returns:
        Estimated market impact in bps.

    """
    if adv_30d <= 0.0 or order_notional <= 0.0:
        return 0.0
    impact_bps = k * sigma_1d * np.sqrt(order_notional / adv_30d) * 10000.0
    return float(impact_bps)
