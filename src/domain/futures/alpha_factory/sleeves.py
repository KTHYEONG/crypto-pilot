"""Sleeve score computation for AlphaFactoryV1."""

from __future__ import annotations

import numpy as np

from src.domain.futures.alpha_factory.config import SleeveConfig
from src.domain.futures.alpha_factory.contracts import SleeveScores


def _clip_score(x: float, clip_abs: float) -> float:
    return float(np.clip(x, -clip_abs, clip_abs))


def compute_sleeve_scores(features: dict[str, float], cfg: SleeveConfig) -> SleeveScores:
    """Compute 5 sleeve scores from normalized feature dictionary."""
    trend = (
        0.70 * features.get("ret_momentum", 0.0)
        + 0.20 * features.get("flow_pressure", 0.0)
        + 0.10 * features.get("vol_guard", 0.0)
    )
    reversal = (
        0.60 * features.get("ret_reversal", 0.0)
        + 0.25 * (1.0 - abs(features.get("ret_momentum", 0.0)))
        + 0.15 * features.get("quality", 0.0)
    )
    carry = (
        0.75 * features.get("carry_pressure", 0.0)
        + 0.25 * features.get("vol_guard", 0.0)
    )
    flow = (
        0.80 * features.get("flow_pressure", 0.0)
        + 0.20 * features.get("quality", 0.0)
    )
    idio = (
        0.85 * features.get("idio_edge", 0.0)
        + 0.15 * features.get("vol_guard", 0.0)
    )

    return SleeveScores(
        trend=_clip_score(trend, cfg.score_clip_abs),
        reversal=_clip_score(reversal, cfg.score_clip_abs),
        carry=_clip_score(carry, cfg.score_clip_abs),
        flow=_clip_score(flow, cfg.score_clip_abs),
        idio=_clip_score(idio, cfg.score_clip_abs),
    )


def blend_raw_alpha(scores: SleeveScores, cfg: SleeveConfig) -> float:
    """Blend sleeves into single raw alpha score."""
    raw = (
        cfg.trend_weight * scores.trend
        + cfg.reversal_weight * scores.reversal
        + cfg.carry_weight * scores.carry
        + cfg.flow_weight * scores.flow
        + cfg.idio_weight * scores.idio
    )
    return float(np.clip(raw, -cfg.score_clip_abs, cfg.score_clip_abs))
