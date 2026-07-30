from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import L1LegConfig
from src.domain.futures.compound.contracts import (
    CausalFold,
    LegBook,
    LegEvidence,
)
from src.domain.futures.compound.l1_concept_bank import compute_lagged_gross_returns

_logger = logging.getLogger(__name__)


def compute_leg_sizing_score(
    evidence: LegEvidence, cost_bps: float, config: L1LegConfig,
) -> float:
    if config.bars_per_year <= 0:
        raise ValueError(f"bars_per_year must be > 0, got {config.bars_per_year}")
    if evidence.evidence_weight <= 0.0:
        return 0.0
    alpha_net = evidence.alpha_ann - cost_bps * 1e-4 * evidence.mean_turnover_per_bar * config.bars_per_year
    alpha_net = max(alpha_net, 0.0)
    var_ann = (evidence.alpha_ann / max(evidence.alpha_sharpe, 1e-12)) ** 2
    fold_consistency = evidence.positive_folds / max(evidence.n_folds, 1)
    return alpha_net / max(var_ann, 1e-12) * fold_consistency

def compute_evidence_weight(
    evidence: LegEvidence,
    cost_bps: float,
    config: L1LegConfig,
) -> float:
    if evidence.mean_turnover_per_bar <= config.min_turnover_per_bar:
        return 0.0
    if evidence.breakeven_cost_bps <= cost_bps * config.cost_safety_margin:
        return 0.0
    if evidence.n_folds > 0 and evidence.positive_folds / evidence.n_folds < config.min_positive_fold_ratio:
        return 0.0
    return 1.0


def normalise_leg_weights(
    raw_weights_1d: NDArray[np.float64], max_leg_weight: float,
) -> NDArray[np.float64]:
    if not (0.0 < max_leg_weight <= 1.0):
        raise ValueError(f"max_leg_weight must be in (0, 1], got {max_leg_weight}")
    k = len(raw_weights_1d)
    if k > 0 and max_leg_weight <= 1.0 / k:
        raise ValueError(
            f"degenerate cap: max_leg_weight={max_leg_weight} <= 1/K={1.0/k} "
            f"for K={k}; caps must exceed 1/K (RULE-06)"
        )
    w = raw_weights_1d.copy()
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return np.zeros_like(w)
    w = w / w_sum
    for _ in range(8):
        exceed = w > max_leg_weight
        if not np.any(exceed):
            break
        excess = float(np.sum(w[exceed] - max_leg_weight))
        n_exceed = int(np.sum(exceed))
        w[exceed] = max_leg_weight
        unsaturated = ~exceed
        unsaturated_sum = float(np.sum(w[unsaturated]))
        if unsaturated_sum <= 1e-12:
            w[exceed] = max_leg_weight + excess / n_exceed
            break
        w[unsaturated] = w[unsaturated] + excess * w[unsaturated] / unsaturated_sum
    return w


def accumulate_prequential_leg_weights(
    legs: tuple[LegBook, ...],
    market_1d: NDArray[np.float64],
    folds: tuple[CausalFold, ...],
    cost_bps: float,
    config: L1LegConfig,
) -> NDArray[np.float64]:
    k_ = len(legs)
    n_t = legs[0].book_2d.shape[0]
    weights = np.zeros((n_t, k_), dtype=np.float64)
    oos_slices = [slice(f.oos_start, f.oos_end_exclusive) for f in folds]
    for i in range(config.warmup_folds, len(folds)):
        prev_evidence: list[LegEvidence] = []
        for k in range(k_):
            ev = evaluate_leg_alpha_on_slices(
                legs[k], market_1d, tuple(oos_slices[:i]), cost_bps, config,
            )
            prev_evidence.append(ev)
        w = np.zeros(k_, dtype=np.float64)
        for k in range(k_):
            w[k] = compute_leg_sizing_score(prev_evidence[k], cost_bps, config)
        w = normalise_leg_weights(w, config.max_leg_weight)
        sl = oos_slices[i]
        weights[sl] = w[np.newaxis, :]
    last_stop = oos_slices[-1].stop if oos_slices else 0
    if last_stop < n_t and last_stop > 0:
        weights[last_stop:] = weights[last_stop - 1:last_stop]
    return weights


def evaluate_leg_alpha_on_slices(
    leg: LegBook,
    market_1d: NDArray[np.float64],
    oos_slices: tuple[slice, ...],
    cost_bps: float,
    config: L1LegConfig,
) -> LegEvidence:
    from src.domain.futures.compound.l1_leg_evaluation import evaluate_leg_alpha
    return evaluate_leg_alpha(leg, market_1d, oos_slices, cost_bps, config)


def combine_leg_books(
    legs: tuple[LegBook, ...],
    leg_weights_2d: NDArray[np.float64],
) -> NDArray[np.float64]:
    k_ = len(legs)
    n_t, n_s = legs[0].book_2d.shape
    combined = np.zeros((n_t, n_s), dtype=np.float64)
    for k in range(k_):
        w = leg_weights_2d[:, k:k + 1]
        combined += w * legs[k].book_2d
    return combined


def evaluate_portfolio_admission(
    combined_2d: NDArray[np.float64],
    asset_return_2d: NDArray[np.float64],
    folds: tuple[CausalFold, ...],
    cost_bps: float,
    config: L1LegConfig,
    *,
    admission_end_exclusive: int,
) -> tuple[bool, tuple[str, ...], float]:
    if admission_end_exclusive <= 0:
        raise ValueError(f"admission_end_exclusive must be > 0, got {admission_end_exclusive}")
    n_t, _ = combined_2d.shape
    gross_returns = compute_lagged_gross_returns(combined_2d, asset_return_2d)
    turnovers = np.zeros(n_t, dtype=np.float64)
    for t in range(1, n_t):
        prev_w = combined_2d[t - 1]
        curr_w = combined_2d[t]
        turnovers[t] = float(np.sum(np.abs(curr_w - prev_w)))
    net_returns = gross_returns - cost_bps * 1e-4 * turnovers
    stressed_cost_bps = cost_bps * config.stress_cost_multiplier
    stressed_returns = gross_returns - stressed_cost_bps * 1e-4 * turnovers

    oos_slices = [slice(f.oos_start, f.oos_end_exclusive) for f in folds]
    traded_slices = [
        sl for f, sl in zip(folds, oos_slices, strict=False)
        if f.oos_end_exclusive <= admission_end_exclusive
    ][config.warmup_folds:]
    traded_parts = [net_returns[sl] for sl in traded_slices]
    traded_parts = [p for p in traded_parts if p.shape[0] > 0]
    if not traded_parts:
        return False, ("no_prequential_admission_folds",), 0.0
    traded_returns = np.concatenate(traded_parts)
    traded_stressed = np.concatenate([stressed_returns[sl] for sl in traded_slices if stressed_returns[sl].shape[0] > 0])
    n_traded = len(traded_returns)
    if n_traded < 10:
        return False, ("insufficient_traded_bars",), 0.0
    net_ann = float(np.mean(traded_returns)) * config.bars_per_year
    stressed_ann = float(np.mean(traded_stressed)) * config.bars_per_year
    rng = np.random.default_rng(42)
    boot = rng.choice(traded_returns, size=(config.n_bootstrap, n_traded), replace=True)
    boot_mean = np.mean(boot, axis=1)
    posterior = float(np.mean(boot_mean > 0.0))
    if n_traded > 0:
        fold_returns = []
        for sl in traded_slices:
            fr = net_returns[sl]
            fold_returns.append(float(np.mean(fr)))
        positive_folds = sum(1 for fv in fold_returns if fv > 0)
        n_traded_folds = len(fold_returns)
    else:
        positive_folds = 0
        n_traded_folds = 0
    reasons: list[str] = []
    if posterior < config.min_growth_posterior_probability:
        reasons.append(f"posterior_{posterior:.3f}_below_{config.min_growth_posterior_probability}")
    if n_traded_folds > 0 and positive_folds / n_traded_folds < config.min_positive_fold_ratio:
        reasons.append(f"positive_folds_{positive_folds}/{n_traded_folds}_below_{config.min_positive_fold_ratio}")
    if stressed_ann <= 0:
        reasons.append(f"stressed_net_ann_{stressed_ann:.4f}_not_positive")
    admitted = len(reasons) == 0
    return admitted, tuple(reasons), net_ann
