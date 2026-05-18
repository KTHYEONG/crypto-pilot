"""Config dataclasses for AlphaFactoryV1 (4h, non-ML)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FeatureNormConfig:
    """Feature normalization constraints."""

    clip_abs: float = 5.0
    eps: float = 1e-12


@dataclass(frozen=True, slots=True)
class SleeveConfig:
    """Sleeve blend and scaling parameters."""

    trend_weight: float = 0.30
    reversal_weight: float = 0.20
    carry_weight: float = 0.15
    flow_weight: float = 0.20
    idio_weight: float = 0.15
    score_clip_abs: float = 3.0


@dataclass(frozen=True, slots=True)
class RegimeRouterConfig:
    """Posterior-to-routing mapping parameters."""

    bull_trend_boost: float = 0.20
    bear_reversal_boost: float = 0.10
    chop_reversal_boost: float = 0.25
    crisis_defense: float = 0.65
    min_confidence: float = 0.25
    min_exposure: float = 0.15
    max_exposure: float = 1.20


@dataclass(frozen=True, slots=True)
class CostAdjusterConfig:
    """Cost and confidence shrinkage parameters."""

    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    turnover_ref: float = 0.75
    confidence_floor: float = 0.10
    alpha_clip_abs: float = 3.0


@dataclass(frozen=True, slots=True)
class AlphaFactoryConfig:
    """Top-level config for 4h non-ML AlphaFactoryV1."""

    timeframe: str = "4h"
    norm: FeatureNormConfig = field(default_factory=FeatureNormConfig)
    sleeves: SleeveConfig = field(default_factory=SleeveConfig)
    regime: RegimeRouterConfig = field(default_factory=RegimeRouterConfig)
    cost: CostAdjusterConfig = field(default_factory=CostAdjusterConfig)
