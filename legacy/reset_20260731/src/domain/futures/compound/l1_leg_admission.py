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
    L1AttributionReport,
    LegBook,
    LegEvidence,
    LegScreenDecision,
    PortfolioAdmissionEvidence,
)
from src.domain.futures.compound.l1_concept_bank import compute_lagged_gross_returns

_logger = logging.getLogger(__name__)


def compute_leg_prior_weights(
    leg_net_returns_2d: NDArray[np.float64],
    end_idx: int,
    config: L1LegConfig,
) -> NDArray[np.float64]:
    if end_idx <= 0:
        raise ValueError(f"end_idx must be > 0, got {end_idx}")
    if leg_net_returns_2d.ndim != 2:
        raise ValueError(f"leg_net_returns_2d must be 2-D, got {leg_net_returns_2d.ndim}")
    k_ = leg_net_returns_2d.shape[1]
    lookback = config.leg_prior_lookback_bars
    start = max(0, end_idx - lookback)
    w = np.zeros(k_, dtype=np.float64)
    for k in range(k_):
        seg = leg_net_returns_2d[start:end_idx, k]
        seg = seg[np.isfinite(seg)]
        vol = float(np.std(seg, ddof=1)) if len(seg) > 1 else 0.0
        if vol > 1e-12:
            w[k] = 1.0 / vol
        else:
            w[k] = 0.0
    if float(np.sum(w)) <= 1e-12:
        return np.full(k_, 1.0 / k_, dtype=np.float64)
    return normalise_leg_weights(w, config.max_leg_weight)


def compute_leg_tilt_scores(
    evidence: tuple[LegEvidence, ...],
    cost_bps: float,
    config: L1LegConfig,
) -> NDArray[np.float64]:
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError(f"cost_bps must be finite and non-negative, got {cost_bps}")
    k_ = len(evidence)
    scores = np.zeros(k_, dtype=np.float64)
    for k, ev in enumerate(evidence):
        if ev.mean_turnover_per_bar <= config.min_turnover_per_bar:
            continue
        if ev.breakeven_cost_bps <= cost_bps * config.cost_safety_margin:
            continue
        if ev.net_alpha_ann <= 0:
            continue
        var_ann = (ev.net_alpha_ann / max(ev.net_alpha_sharpe, 1e-12)) ** 2 if ev.net_alpha_sharpe > 0 else 1.0
        if not np.isfinite(var_ann) or var_ann <= 1e-12:
            continue
        scores[k] = ev.net_alpha_ann / var_ann
    return scores


def compute_shrinkage_weights(
    prior_1d: NDArray[np.float64],
    tilt_raw_1d: NDArray[np.float64],
    n_obs: int,
    config: L1LegConfig,
) -> NDArray[np.float64]:
    if prior_1d.shape != tilt_raw_1d.shape:
        raise ValueError(f"prior shape {prior_1d.shape} != tilt shape {tilt_raw_1d.shape}")
    if n_obs < 0:
        raise ValueError(f"n_obs must be >= 0, got {n_obs}")
    if not np.all(np.isfinite(prior_1d)):
        return normalise_leg_weights(np.full_like(prior_1d, 1.0 / len(prior_1d)), config.max_leg_weight)
    tilt_sum = float(np.sum(tilt_raw_1d))
    if not np.all(np.isfinite(tilt_raw_1d)) or tilt_sum <= 1e-12:
        return normalise_leg_weights(prior_1d.copy(), config.max_leg_weight)
    lam = n_obs / (n_obs + config.shrinkage_prior_obs)
    tilt_norm = tilt_raw_1d / tilt_sum
    w = (1.0 - lam) * prior_1d + lam * tilt_norm
    return normalise_leg_weights(w, config.max_leg_weight)


def compute_handoff_scale(
    posterior_positive: float,
    config: L1LegConfig,
) -> float:
    if config.handoff_posterior_floor >= config.min_growth_posterior_probability:
        raise ValueError(
            f"handoff_posterior_floor={config.handoff_posterior_floor} must be < "
            f"min_growth_posterior_probability={config.min_growth_posterior_probability}"
        )
    if not np.isfinite(posterior_positive):
        return 0.0
    phi = (posterior_positive - config.handoff_posterior_floor) / (
        config.min_growth_posterior_probability - config.handoff_posterior_floor
    )
    return max(0.0, min(1.0, float(phi)))


def classify_leg_evidence(
    evidence: LegEvidence,
    cost_bps: float,
    config: L1LegConfig,
    *,
    n_tested_hypotheses: int = 1,
) -> LegScreenDecision:
    if n_tested_hypotheses < 1:
        raise ValueError(f"n_tested_hypotheses must be >= 1, got {n_tested_hypotheses}")
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError(f"cost_bps must be finite and non-negative, got {cost_bps}")

    economic_reasons: list[str] = []
    if evidence.n_folds < config.warmup_folds:
        economic_reasons.append(f"insufficient_folds:{evidence.n_folds}_below_{config.warmup_folds}")
    if evidence.net_alpha_ann <= 0:
        economic_reasons.append(f"net_alpha_ann_not_positive:{evidence.net_alpha_ann:.6f}")
    if evidence.mean_turnover_per_bar <= config.min_turnover_per_bar:
        economic_reasons.append(
            f"turnover_below_floor:{evidence.mean_turnover_per_bar:.6f}_below_{config.min_turnover_per_bar}"
        )
    if evidence.breakeven_cost_bps <= cost_bps * config.cost_safety_margin:
        economic_reasons.append(
            f"cost_headroom:BE_{evidence.breakeven_cost_bps:.2f}bps_<={cost_bps * config.cost_safety_margin:.2f}bps"
        )
    if evidence.n_folds > 0 and evidence.positive_folds / evidence.n_folds < config.min_positive_fold_ratio:
        economic_reasons.append(
            f"fold_instability:{evidence.positive_folds}/{evidence.n_folds}_"
            f"below_{config.min_positive_fold_ratio}"
        )
    economic_eligible = len(economic_reasons) == 0

    familywise_reasons: list[str] = []
    if n_tested_hypotheses > 1:
        critical_t = float(sp_norm.isf(config.familywise_error_rate / n_tested_hypotheses))
    else:
        critical_t = 0.0
    if evidence.t_net_alpha_newey_west < critical_t:
        familywise_reasons.append(
            f"net_t_below_familywise_threshold:"
            f"{evidence.t_net_alpha_newey_west:.3f}_below_{critical_t:.3f}_K={n_tested_hypotheses}"
        )
    familywise_supported = len(familywise_reasons) == 0

    capital_eligible = economic_eligible

    return LegScreenDecision(
        economic_eligible=economic_eligible,
        familywise_supported=familywise_supported,
        capital_eligible=capital_eligible,
        economic_reasons=tuple(economic_reasons),
        familywise_reasons=tuple(familywise_reasons),
        critical_t=critical_t,
        n_tested_hypotheses=n_tested_hypotheses,
    )


def screen_leg_evidence(
    evidence: LegEvidence,
    cost_bps: float,
    config: L1LegConfig,
    n_tested_hypotheses: int,
) -> LegEvidence:
    decision = classify_leg_evidence(evidence, cost_bps, config, n_tested_hypotheses=n_tested_hypotheses)
    reasons = list(decision.economic_reasons) + list(decision.familywise_reasons)
    evidence_weight = 1.0 if decision.capital_eligible else 0.0
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
    w = np.maximum(w, 0.0)
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return np.zeros_like(w)
    w = w / w_sum
    w = np.minimum(w, max_leg_weight)
    current_sum = float(np.sum(w))
    if current_sum >= 1.0 - 1e-12:
        return w
    if np.any(~np.isfinite(w)):
        return np.zeros_like(w)
    for _ in range(16):
        unsaturated = w < max_leg_weight
        if not np.any(unsaturated):
            break
        gap = 1.0 - float(np.sum(w))
        if gap <= 1e-12:
            break
        total_unsaturated_capacity = float(np.sum(max_leg_weight - w[unsaturated]))
        fill = min(gap, total_unsaturated_capacity)
        w[unsaturated] += (max_leg_weight - w[unsaturated]) / max(total_unsaturated_capacity, 1e-12) * fill
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
    leg_net_2d = np.column_stack([leg.gross_return_1d - cost_bps * 1e-4 * leg.turnover_1d for leg in legs])
    for i in range(len(folds)):
        if i < config.prior_only_folds:
            prior = compute_leg_prior_weights(leg_net_2d, folds[i].fit_end_exclusive, config)
            w = prior
            _logger.debug("[ALGO] leg=prior fold=%d w=%s", i, np.array2string(w, precision=4))
        else:
            prior = compute_leg_prior_weights(leg_net_2d, folds[i].fit_end_exclusive, config)
            prev_evidence: list[LegEvidence] = []
            for k in range(k_):
                ev = evaluate_leg_alpha_on_slices(
                    legs[k], market_1d, tuple(oos_slices[:i]), cost_bps, config,
                    n_tested_hypotheses=1,
                )
                prev_evidence.append(ev)
            tilt_raw = compute_leg_tilt_scores(tuple(prev_evidence), cost_bps, config)
            n_obs = sum(
                oos_slices[j].stop - oos_slices[j].start
                for j in range(i)
            )
            w = compute_shrinkage_weights(prior, tilt_raw, n_obs, config)
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
    evidence = evaluate_portfolio_evidence(
        combined_2d, asset_return_2d, folds, cost_bps, config,
        admission_end_exclusive=admission_end_exclusive,
    )
    return evidence.admitted, evidence.reasons, evidence.net_alpha_ann


def accumulate_prequential_shadow_weights(
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
        raw_scores = np.zeros(k_, dtype=np.float64)
        for k in range(k_):
            ev = evaluate_leg_alpha_on_slices(
                legs[k], market_1d, tuple(oos_slices[:i]), cost_bps, config,
                n_tested_hypotheses=1,
            )
            decision = classify_leg_evidence(ev, cost_bps, config, n_tested_hypotheses=1)
            if decision.economic_eligible:
                raw_scores[k] = 1.0
        w = normalise_leg_weights(raw_scores, config.max_leg_weight)
        sl = oos_slices[i]
        weights[sl] = w[np.newaxis, :]
    last_stop = oos_slices[-1].stop if oos_slices else 0
    if last_stop < n_t and last_stop > 0:
        weights[last_stop:] = weights[last_stop - 1:last_stop]
    return weights


def evaluate_portfolio_evidence(
    combined_2d: NDArray[np.float64],
    asset_return_2d: NDArray[np.float64],
    folds: tuple[CausalFold, ...],
    cost_bps: float,
    config: L1LegConfig,
    *,
    admission_end_exclusive: int,
) -> PortfolioAdmissionEvidence:
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
    ]
    traded_parts = [net_returns[sl] for sl in traded_slices]
    traded_parts = [p for p in traded_parts if p.shape[0] > 0]
    if not traded_parts:
        return PortfolioAdmissionEvidence(
            admitted=False, reasons=("no_prequential_admission_folds",),
            net_alpha_ann=0.0, stressed_net_alpha_ann=0.0,
            posterior_positive=0.0, positive_folds=0, n_folds=0, n_traded_bars=0,
        )
    traded_returns = np.concatenate(traded_parts)
    traded_stressed = np.concatenate([stressed_returns[sl] for sl in traded_slices if stressed_returns[sl].shape[0] > 0])
    n_traded = len(traded_returns)
    if n_traded < 10:
        return PortfolioAdmissionEvidence(
            admitted=False, reasons=("insufficient_traded_bars",),
            net_alpha_ann=0.0, stressed_net_alpha_ann=0.0,
            posterior_positive=0.0, positive_folds=0, n_folds=0, n_traded_bars=n_traded,
        )
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
    fold_ratio_ok = not (n_traded_folds > 0 and positive_folds / n_traded_folds < config.min_positive_fold_ratio)
    reasons: list[str] = []
    if float(posterior) < config.min_growth_posterior_probability:
        reasons.append(f"posterior_{posterior:.3f}_below_{config.min_growth_posterior_probability}")
    if not fold_ratio_ok:
        reasons.append(f"positive_folds_{positive_folds}/{n_traded_folds}_below_{config.min_positive_fold_ratio}")
    if stressed_ann <= 0:
        reasons.append(f"stressed_net_ann_{stressed_ann:.4f}_not_positive")
    admitted = len(reasons) == 0
    hard_gates_pass = fold_ratio_ok and stressed_ann > 0.0
    handoff = compute_handoff_scale(float(posterior), config) if hard_gates_pass else 0.0
    return PortfolioAdmissionEvidence(
        admitted=admitted, reasons=tuple(reasons),
        net_alpha_ann=net_ann, stressed_net_alpha_ann=stressed_ann,
        posterior_positive=float(posterior),
        positive_folds=positive_folds, n_folds=n_traded_folds,
        n_traded_bars=n_traded, handoff_scale=handoff,
    )


def classify_l1_bottleneck(
    production: PortfolioAdmissionEvidence,
    shadow: PortfolioAdmissionEvidence,
    economic_candidate_count: int,
    capital_candidate_count: int,
    shadow_available: bool,
) -> L1AttributionReport:
    if not shadow_available:
        return L1AttributionReport(
            production=production, shadow=shadow,
            economic_candidate_count=economic_candidate_count,
            capital_candidate_count=capital_candidate_count,
            bottleneck_code="diagnostic_unavailable",
            shadow_available=False,
            production_weights_unchanged=True,
        )
    if production.handoff_scale == 1.0:
        code = "deployable"
    elif 0 < production.handoff_scale < 1.0:
        code = "partial_evidence_sized"
    elif economic_candidate_count == 0:
        code = "signal_economics_absent"
    elif shadow.admitted:
        code = "familywise_power_limited"
    else:
        code = "signal_generalization_failed"
    return L1AttributionReport(
        production=production, shadow=shadow,
        economic_candidate_count=economic_candidate_count,
        capital_candidate_count=capital_candidate_count,
        bottleneck_code=code,
        shadow_available=True,
        production_weights_unchanged=True,
    )
