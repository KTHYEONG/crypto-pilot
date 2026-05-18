"""Sleeve score computation for AlphaFactoryV1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from src.domain.futures.alpha_factory.config import SleeveConfig
from src.domain.futures.alpha_factory.contracts import SleeveScores

SLEEVE_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "trend": ("ret_momentum", "flow_pressure", "vol_guard"),
    "reversal": ("ret_reversal", "ret_momentum", "quality"),
    "carry": ("carry_pressure", "vol_guard"),
    "flow": ("flow_pressure", "quality"),
    "idio": ("idio_edge", "vol_guard"),
}


@dataclass(frozen=True, slots=True)
class SleeveRidgeModels:
    """Optional Step1 Ridge models for per-sleeve scorers."""

    models: dict[str, Ridge]


@dataclass(frozen=True, slots=True)
class SleeveBlendWeights:
    """Blend weights for 5 sleeves."""

    trend: float
    reversal: float
    carry: float
    flow: float
    idio: float


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
    raw = blend_raw_alpha_with_weights(
        scores,
        SleeveBlendWeights(
            trend=cfg.trend_weight,
            reversal=cfg.reversal_weight,
            carry=cfg.carry_weight,
            flow=cfg.flow_weight,
            idio=cfg.idio_weight,
        ),
        cfg.score_clip_abs,
    )
    return float(raw)


def blend_raw_alpha_with_weights(
    scores: SleeveScores, weights: SleeveBlendWeights, clip_abs: float
) -> float:
    """Blend sleeves with explicit weights."""
    raw = (
        weights.trend * scores.trend
        + weights.reversal * scores.reversal
        + weights.carry * scores.carry
        + weights.flow * scores.flow
        + weights.idio * scores.idio
    )
    return float(np.clip(raw, -clip_abs, clip_abs))


def fit_ridge_sleeve_models(
    features_seq: list[dict[str, float]],
    target: np.ndarray,
    is_mask: np.ndarray,
    *,
    alpha: float,
    min_samples: int,
) -> SleeveRidgeModels:
    """Fit per-sleeve Ridge models on IS rows only."""
    models: dict[str, Ridge] = {}
    y = np.asarray(target, dtype=np.float64)
    fit_mask_base = np.asarray(is_mask, dtype=bool) & np.isfinite(y)
    if int(fit_mask_base.sum()) < min_samples:
        return SleeveRidgeModels(models={})

    for sleeve, cols in SLEEVE_FEATURE_GROUPS.items():
        x = np.array(
            [[float(feat.get(col, 0.0)) for col in cols] for feat in features_seq],
            dtype=np.float64,
        )
        fit_mask = fit_mask_base & np.all(np.isfinite(x), axis=1)
        if int(fit_mask.sum()) < min_samples:
            continue
        mdl = Ridge(alpha=float(alpha))
        mdl.fit(x[fit_mask], y[fit_mask])
        models[sleeve] = mdl
    return SleeveRidgeModels(models=models)


def compute_sleeve_scores_with_models(
    features: dict[str, float],
    cfg: SleeveConfig,
    ridge_models: SleeveRidgeModels | None,
) -> SleeveScores:
    """Compute sleeve scores using fitted Ridge models when available."""
    if ridge_models is None or not ridge_models.models:
        return compute_sleeve_scores(features, cfg)

    base = compute_sleeve_scores(features, cfg)
    predicted: dict[str, float] = {
        "trend": base.trend,
        "reversal": base.reversal,
        "carry": base.carry,
        "flow": base.flow,
        "idio": base.idio,
    }
    for sleeve, cols in SLEEVE_FEATURE_GROUPS.items():
        mdl = ridge_models.models.get(sleeve)
        if mdl is None:
            continue
        x = np.array([[float(features.get(col, 0.0)) for col in cols]], dtype=np.float64)
        v = float(mdl.predict(x)[0])
        predicted[sleeve] = _clip_score(v, cfg.score_clip_abs)

    return SleeveScores(
        trend=predicted["trend"],
        reversal=predicted["reversal"],
        carry=predicted["carry"],
        flow=predicted["flow"],
        idio=predicted["idio"],
    )
