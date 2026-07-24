from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import RiskOverlayConfig
from src.domain.futures.compound.contracts import PortfolioDecision, RiskOverlayResult

_logger = logging.getLogger(__name__)

_EPSILON = 1e-12

_BARS_PER_YEAR: float = 2190.0


def apply_fractional_kelly_scaling(
    weights: NDArray[np.float64],
    portfolio_variance: float,
    *,
    f: float = 0.25,
    sigma_target: float = 0.25,
    p: float = 1.0,
    cvar_vol_multiple: float = 2.0,
    cvar_regime_active: bool = False,
) -> NDArray[np.float64]:
    ann_portfolio_vol = np.sqrt(max(portfolio_variance, _EPSILON)) * np.sqrt(_BARS_PER_YEAR)
    vol_scale = sigma_target / max(ann_portfolio_vol, sigma_target)
    scale = f * (vol_scale ** p)
    if cvar_regime_active:
        scale /= cvar_vol_multiple
    result: NDArray[np.float64] = weights * scale
    return result


def apply_risk_overlay(
    *,
    decision: PortfolioDecision,
    equity_1d: NDArray[np.float64],
    cooldown_remaining: int,
    config: RiskOverlayConfig,
) -> RiskOverlayResult:
    w = decision.target_weights_1d.copy()

    if cooldown_remaining > 0:
        return RiskOverlayResult(
            target_weights_1d=np.zeros_like(w),
            risk_scale=0.0,
            drawdown_scale=0.0,
            volatility_scale=0.0,
            cooldown_remaining=cooldown_remaining - 1,
            hard_block_reason="cooldown",
        )

    peak_equity = np.maximum.accumulate(equity_1d)
    dd = (peak_equity[-1] - equity_1d[-1]) / max(peak_equity[-1], _EPSILON)

    if dd >= config.hard_drawdown:
        return RiskOverlayResult(
            target_weights_1d=np.zeros_like(w),
            risk_scale=0.0,
            drawdown_scale=0.0,
            volatility_scale=0.0,
            cooldown_remaining=config.hard_drawdown_cooldown_bars,
            hard_block_reason="hard_drawdown",
        )

    if dd <= config.soft_drawdown_start:
        dd_scale = 1.0
    elif dd < config.drawdown_second_knot:
        dd_scale = 1.0 - (dd - config.soft_drawdown_start) / (config.drawdown_second_knot - config.soft_drawdown_start) * 0.5
    else:
        dd_scale = 0.5 - (dd - config.drawdown_second_knot) / (config.hard_drawdown - config.drawdown_second_knot) * 0.5
        dd_scale = max(dd_scale, 0.0)

    ann_vol = max(decision.forecast_ann_vol, _EPSILON)
    target_ann_vol = 0.15
    vol_scale = min(1.0, target_ann_vol / ann_vol)

    combined_scale = dd_scale * vol_scale
    w_scaled = w * combined_scale

    return RiskOverlayResult(
        target_weights_1d=w_scaled,
        risk_scale=combined_scale,
        drawdown_scale=dd_scale,
        volatility_scale=vol_scale,
        cooldown_remaining=0,
        hard_block_reason="",
    )
