"""IC shrinkage ensemble helpers for AlphaFactoryV1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class ShrinkageConfig:
    """Configuration for conservative-prior IC shrinkage."""

    prior_ic: float = 0.0
    prior_strength: float = 12.0
    min_weight: float = 0.0
    max_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class EnsembleOutput:
    """Canonical ensemble output contract for downstream consumers."""

    alpha_long: np.ndarray
    alpha_short: np.ndarray
    alpha_net: np.ndarray
    confidence: np.ndarray
    turnover_hint: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True, slots=True)
class SleeveICStats:
    """Per-sleeve IC summary."""

    mu: float
    sigma: float
    n_folds: int


def compute_sleeve_shrinkage_weights(
    ic_by_sleeve: dict[str, list[float]],
    *,
    prior_mean: float,
    prior_strength: float,
    min_folds: int,
    eps: float = _EPS,
) -> tuple[dict[str, float], dict[str, SleeveICStats]]:
    """Compute non-negative normalized shrinkage weights from OOS fold ICs."""
    sleeves = ("trend", "reversal", "carry", "flow", "idio")
    raw_edges: dict[str, float] = {}
    stats: dict[str, SleeveICStats] = {}
    for sleeve in sleeves:
        vals = np.asarray(ic_by_sleeve.get(sleeve, []), dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        n_folds = int(vals.size)
        mu = float(np.mean(vals)) if n_folds > 0 else 0.0
        sigma = float(np.std(vals)) if n_folds > 0 else 0.0
        signal = mu / max(sigma, float(eps)) if n_folds > 0 else 0.0
        post = ((n_folds * signal) + (prior_strength * prior_mean)) / (
            n_folds + prior_strength + float(eps)
        )
        raw_edges[sleeve] = max(float(post), 0.0) if n_folds >= min_folds else 0.0
        stats[sleeve] = SleeveICStats(mu=mu, sigma=sigma, n_folds=n_folds)

    edge_sum = float(sum(raw_edges.values()))
    if edge_sum <= float(eps):
        return {}, stats

    weights = {k: (v / edge_sum) for k, v in raw_edges.items()}
    return weights, stats


def _safe_clip_01(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.nan_to_num(values, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return cast(np.ndarray, clipped)


def compute_ic_shrinkage_weights(
    ic_series: np.ndarray,
    sample_sizes: np.ndarray,
    config: ShrinkageConfig | None = None,
) -> np.ndarray:
    """Compute conservative-prior adaptive shrinkage weights from fold ICs."""
    cfg = config or ShrinkageConfig()
    ic = np.nan_to_num(np.asarray(ic_series, dtype=np.float64), nan=cfg.prior_ic)
    n = np.clip(np.nan_to_num(np.asarray(sample_sizes, dtype=np.float64), nan=0.0), 0.0, None)

    post_ic = (n * ic + cfg.prior_strength * cfg.prior_ic) / (n + cfg.prior_strength + _EPS)
    positive_edge = np.clip(post_ic, 0.0, None)

    if float(positive_edge.sum()) <= _EPS:
        return np.full_like(positive_edge, 1.0 / float(len(positive_edge)))

    weights = positive_edge / (positive_edge.sum() + _EPS)
    clipped = np.clip(weights, cfg.min_weight, cfg.max_weight)
    return cast(np.ndarray, clipped)


def build_ensemble(
    alpha_frame: pd.DataFrame,
    weights: np.ndarray,
) -> EnsembleOutput:
    """Build alpha ensemble output adhering to AlphaFactory contract."""
    long_cols = [c for c in alpha_frame.columns if c.startswith("alpha_long_")]
    short_cols = [c for c in alpha_frame.columns if c.startswith("alpha_short_")]
    if not long_cols:
        raise ValueError("alpha_frame must contain alpha_long_* columns")

    long_values = _safe_clip_01(alpha_frame[long_cols].to_numpy(dtype=np.float64, copy=False))
    short_values: np.ndarray
    if short_cols:
        short_values = _safe_clip_01(alpha_frame[short_cols].to_numpy(dtype=np.float64, copy=False))
    else:
        short_values = 1.0 - long_values

    if long_values.shape[1] != len(weights):
        raise ValueError(
            f"weight size mismatch: n_features={long_values.shape[1]} weights={len(weights)}"
        )

    w = np.asarray(weights, dtype=np.float64)
    w = w / (float(w.sum()) + _EPS)

    alpha_long = _safe_clip_01(long_values @ w)
    alpha_short = _safe_clip_01(short_values @ w)
    alpha_net = np.clip(alpha_long - alpha_short, -1.0, 1.0)

    disagreement = np.mean(np.abs(long_values - alpha_long[:, None]), axis=1)
    confidence = np.clip(1.0 - 2.0 * disagreement, 0.0, 1.0)

    turnover_hint = np.zeros_like(alpha_long)
    if len(alpha_long) > 1:
        turnover_hint[1:] = np.abs(np.diff(alpha_net))
        turnover_hint[0] = turnover_hint[1]
    turnover_hint = np.clip(turnover_hint, 0.0, 1.0)

    return EnsembleOutput(
        alpha_long=alpha_long,
        alpha_short=alpha_short,
        alpha_net=alpha_net,
        confidence=confidence,
        turnover_hint=turnover_hint,
        weights=w,
    )
