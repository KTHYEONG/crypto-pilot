"""Friction and transaction cost models for futures trading."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.core.settings import TAKER_FEE_BPS

_logger = logging.getLogger("FrictionModel")


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
