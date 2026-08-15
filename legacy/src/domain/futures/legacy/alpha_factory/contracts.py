"""Typed contracts for AlphaFactoryV1 (4h, non-ML)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SleeveScores:
    """Per-sleeve raw scores in [-inf, +inf] before blending."""

    trend: float
    reversal: float
    carry: float
    flow: float
    idio: float


@dataclass(frozen=True, slots=True)
class SleeveWeights:
    """Per-sleeve blend weights (sum ~= 1)."""

    trend: float
    reversal: float
    carry: float
    flow: float
    idio: float


@dataclass(frozen=True, slots=True)
class RegimePosterior:
    """Semantic 4-bucket posterior used by routing logic."""

    bull: float
    bear: float
    chop: float
    crisis: float


@dataclass(frozen=True, slots=True)
class RegimeDecision:
    """Routing output: sleeve weights + gross exposure multiplier."""

    weights: SleeveWeights
    gross_exposure: float
    confidence: float


@dataclass(frozen=True, slots=True)
class AlphaFactoryResult:
    """Final AlphaFactoryV1 output for one timestamp/symbol."""

    raw_alpha: float
    adjusted_alpha: float
    confidence: float
    gross_exposure: float
    turnover_penalty: float
    cost_penalty: float
    sleeves: SleeveScores
    regime: RegimePosterior


def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def safe_div(numer: float, denom: float, eps: float = 1e-12) -> float:
    return float(numer / (denom if abs(denom) > eps else eps))
