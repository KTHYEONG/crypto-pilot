from __future__ import annotations

import dataclasses
import logging

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm as sp_norm

from src.domain.futures.compound.bootstrap import circular_stationary_bootstrap_growth, politis_white_block_length
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
    net_ann = max(evidence.net_alpha_ann, 0.0)
    net_var_ann = (evidence.net_alpha_ann / max(evidence.net_alpha_sharpe, 1e-12)) ** 2 if evidence.net_alpha_sharpe > 0 else 1.0
    fold_consistency = evidence.positive_folds / max(evidence.n_folds, 1)
    return net_ann / max(net_var_ann, 1e-12) * fold_consistency


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


def screen_leg_evidence(
    evidence: LegEvidence,
    cost_bps: float,
    config: L1LegConfig,
    n_tested_hypotheses: int,
) -> LegEvidence:
    if n_tested_hypotheses < 1:
        raise ValueError(f"n_tested_hypotheses must be >= 1, got {n_tested_hypotheses}")
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError(f"cost_bps must be finite and non-negative, got {cost_bps}")
    reasons: list[str] = []
    if evidence.n_folds < config.warmup_folds:
        reasons.append(f"insufficient_folds:{evidence.n_folds}_below_{config.warmup_folds}")
    if n_tested_hypotheses > 1:
        critical_t = float(sp_norm.isf(config.familywise_error_rate / n_tested_hypotheses))
    else:
        critical_t = 0.0
    if evidence.t_net_alpha_newey_west < critical_t:
        reasons.append(
            f"net_t_below_familywise_threshold:"
            f"{evidence.t_net_alpha_newey_west:.3f}_below_{critical_t:.3f}_K={n_tested_hypotheses}"
        )
    if evidence.mean_turnover_per_bar <= config.min_turnover_per_bar:
        reasons.append(
            f"turnover_below_floor:{evidence.mean_turnover_per_bar:.6f}_below_{config.min_turnover_per_bar}"
        )
    if evidence.breakeven_cost_bps <= cost_bps * config.cost_safety_margin:
        reasons.append(
            f"cost_headroom:BE_{evidence.breakeven_cost_bps:.2f}bps_<={cost_bps * config.cost_safety_margin:.2f}bps"
        )
    if evidence.n_folds > 0 and evidence.positive_folds / evidence.n_folds < config.min_positive_fold_ratio:
        reasons.append(
            f"fold_instability:{evidence.positive_folds}/{evidence.n_folds}_"
            f"below_{config.min_positive_fold_ratio}"
        )
    evidence_weight = 0.0 if reasons else 1.0
    return dataclasses.replace(evidence, evidence_weight=evidence_weight, reasons=tuple(reasons))


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
    w = np.clip(w, 0.0, None)
    w = w / w_sum
    w = np.minimum(w, max_leg_weight)
    if np.any(~np.isfinite(w)):
        return np.zeros_like(w)
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
                n_tested_hypotheses=k_,
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
    *,
    n_tested_hypotheses: int = 1,
) -> LegEvidence:
    from src.domain.futures.compound.l1_leg_evaluation import evaluate_leg_alpha
    return evaluate_leg_alpha(leg, market_1d, oos_slices, cost_bps, config, n_tested_hypotheses=n_tested_hypotheses)


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
    pw_block = 5.0
    if n_traded >= 30:
        try:
            pw_block = politis_white_block_length(traded_returns)
        except ValueError:
            pw_block = 5.0
    _, _, posterior = circular_stationary_bootstrap_growth(
        traded_returns, config.bars_per_year,
        n_bootstrap=config.n_bootstrap, block_size=pw_block, seed=42,
    )
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
