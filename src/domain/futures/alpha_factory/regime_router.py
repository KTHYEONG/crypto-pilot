"""HMM posterior based regime routing for AlphaFactoryV1."""

from __future__ import annotations

import numpy as np

from src.domain.futures.alpha_factory.config import RegimeRouterConfig, SleeveConfig
from src.domain.futures.alpha_factory.contracts import (
    RegimeDecision,
    RegimePosterior,
    SleeveWeights,
    clamp01,
)


def _normalize_probs(bull: float, bear: float, chop: float, crisis: float) -> RegimePosterior:
    arr = np.array([bull, bear, chop, crisis], dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    s = float(arr.sum())
    if s <= 1e-12:
        arr[:] = 0.25
    else:
        arr /= s
    return RegimePosterior(
        bull=float(arr[0]),
        bear=float(arr[1]),
        chop=float(arr[2]),
        crisis=float(arr[3]),
    )


def route_by_regime(
    posterior: RegimePosterior,
    sleeve_cfg: SleeveConfig,
    regime_cfg: RegimeRouterConfig,
) -> RegimeDecision:
    """Map posterior to sleeve weights and gross exposure multiplier."""
    post = _normalize_probs(posterior.bull, posterior.bear, posterior.chop, posterior.crisis)

    trend_w = sleeve_cfg.trend_weight * (1.0 + regime_cfg.bull_trend_boost * post.bull)
    reversal_w = sleeve_cfg.reversal_weight * (
        1.0
        + regime_cfg.chop_reversal_boost * post.chop
        + regime_cfg.bear_reversal_boost * post.bear
    )
    carry_w = sleeve_cfg.carry_weight * (1.0 - 0.40 * post.crisis)
    flow_w = sleeve_cfg.flow_weight * (1.0 - 0.25 * post.crisis + 0.15 * post.bull)
    idio_w = sleeve_cfg.idio_weight * (1.0 - 0.30 * post.crisis + 0.10 * post.bear)

    vec = np.array([trend_w, reversal_w, carry_w, flow_w, idio_w], dtype=np.float64)
    vec = np.clip(vec, 0.0, None)
    denom = float(vec.sum())
    if denom <= 1e-12:
        vec[:] = 0.2
        denom = 1.0
    vec /= denom

    confidence = clamp01(post.bull + post.bear + 0.5 * post.chop)
    confidence = max(confidence, regime_cfg.min_confidence)

    risk_off = post.bear + post.crisis
    raw_exposure = (
        confidence
        * (1.0 - regime_cfg.crisis_defense * post.crisis)
        * (1.0 - 0.35 * risk_off)
    )
    gross_exposure = float(np.clip(raw_exposure, regime_cfg.min_exposure, regime_cfg.max_exposure))

    return RegimeDecision(
        weights=SleeveWeights(
            trend=float(vec[0]),
            reversal=float(vec[1]),
            carry=float(vec[2]),
            flow=float(vec[3]),
            idio=float(vec[4]),
        ),
        gross_exposure=gross_exposure,
        confidence=confidence,
    )
