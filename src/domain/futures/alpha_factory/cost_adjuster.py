"""Cost/confidence adjustment for AlphaFactoryV1 outputs."""

from __future__ import annotations

import numpy as np

from src.domain.futures.alpha_factory.config import CostAdjusterConfig
from src.domain.futures.alpha_factory.contracts import clamp01


def adjust_alpha_for_cost_and_confidence(
    raw_alpha: float,
    confidence: float,
    gross_exposure: float,
    turnover: float,
    cfg: CostAdjusterConfig,
) -> tuple[float, float, float]:
    """Apply transaction-cost penalty and confidence shrinkage.

    Returns:
        adjusted_alpha, turnover_penalty, cost_penalty

    """
    c = max(clamp01(confidence), cfg.confidence_floor)
    t = float(np.clip(turnover, 0.0, 5.0))

    turnover_penalty = t / max(cfg.turnover_ref, 1e-12)
    linear_cost = (cfg.fee_bps + cfg.slippage_bps) * 1e-4 * gross_exposure
    cost_penalty = linear_cost * (1.0 + turnover_penalty)

    shrunk = float(np.clip(raw_alpha, -cfg.alpha_clip_abs, cfg.alpha_clip_abs)) * c
    adjusted = shrunk - cost_penalty
    return adjusted, float(turnover_penalty), float(cost_penalty)
