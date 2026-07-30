from __future__ import annotations

import dataclasses
import logging

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import L1LegConfig
from src.domain.futures.compound.contracts import (
    LegBook,
    LegEvidence,
)
from src.domain.futures.compound.l1_diagnostics import L1AdmissionRecorder
from src.domain.futures.compound.l1_leg_admission import compute_evidence_weight

_logger = logging.getLogger(__name__)


def compute_equal_weight_market_returns(
    close_2d: NDArray[np.float32],
    eligible_2d: NDArray[np.bool_],
) -> NDArray[np.float64]:
    n_t, _ = close_2d.shape
    close_f64 = close_2d.astype(np.float64)
    ret = np.zeros(n_t, dtype=np.float64)
    for t in range(1, n_t):
        prev = close_f64[t - 1]
        curr = close_f64[t]
        mask = eligible_2d[t] & (prev > 0) & np.isfinite(prev) & (curr > 0) & np.isfinite(curr)
        n_valid = int(np.sum(mask))
        if n_valid < 2:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.log(curr / prev)
        ret[t] = float(np.nanmean(np.where(mask, log_ret, np.nan)))
    return ret


def compute_breakeven_cost_bps(
    alpha_ann: float,
    mean_turnover_per_bar: float,
    bars_per_year: float,
) -> float:
    if mean_turnover_per_bar <= 0.0 or bars_per_year <= 0.0:
        return 0.0
    return alpha_ann / (mean_turnover_per_bar * bars_per_year) * 1e4


def _newey_west_variance(
    resid: NDArray[np.float64],
    max_lag: int,
) -> float:
    n = resid.shape[0]
    var_0 = float(np.var(resid, ddof=1))
    if var_0 <= 0.0:
        return 0.0
    autocov = 0.0
    for lag in range(1, min(max_lag + 1, n)):
        c = float(np.cov(resid[:n - lag], resid[lag:], ddof=1)[0, 1])
        weight = 1.0 - lag / (max_lag + 1.0)
        autocov += 2.0 * weight * c
    nw_var = var_0 + autocov / n
    return float(max(nw_var, 1e-12))


def evaluate_leg_alpha(
    leg: LegBook,
    market_1d: NDArray[np.float64],
    oos_slices: tuple[slice, ...],
    cost_bps: float,
    config: L1LegConfig,
) -> LegEvidence:
    g = leg.gross_return_1d
    t = leg.turnover_1d
    all_alpha: list[float] = []
    all_beta: list[float] = []
    positive_folds = 0
    n_possible = 0
    for sl in oos_slices:
        g_oos = g[sl]
        m_oos = market_1d[sl]
        n = g_oos.shape[0]
        if n < 5:
            continue
        n_possible += 1
        var_m = float(np.var(m_oos, ddof=1))
        beta = 0.0 if var_m < 1e-12 else float(np.cov(m_oos, g_oos, ddof=1)[0, 1]) / var_m
        alpha = float(np.mean(g_oos)) - beta * float(np.mean(m_oos))
        if alpha > 0:
            positive_folds += 1
        all_alpha.append(alpha)
        all_beta.append(beta)
    recorder = L1AdmissionRecorder()
    if n_possible == 0:
        evidence = LegEvidence(
            concept_id=leg.spec.concept_id, mode=leg.spec.mode,
            n_oos_bars=0, alpha_ann=0.0, beta_market=0.0,
            alpha_sharpe=0.0, t_alpha_newey_west=0.0,
            breakeven_cost_bps=0.0, mean_turnover_per_bar=0.0,
            positive_folds=0, n_folds=0, posterior_positive=0.0,
            evidence_weight=0.0,
            reasons=("no_oos_folds",),
        )
        evidence = dataclasses.replace(
            evidence, evidence_weight=compute_evidence_weight(evidence, cost_bps, config),
        )
        if recorder.enabled:
            recorder.record_leg(
                concept_id=evidence.concept_id, mode=evidence.mode,
                alpha_ann=evidence.alpha_ann, beta_market=evidence.beta_market,
                alpha_sharpe=evidence.alpha_sharpe, t_alpha=evidence.t_alpha_newey_west,
                breakeven_cost_bps=evidence.breakeven_cost_bps,
                mean_turnover_per_bar=evidence.mean_turnover_per_bar,
                positive_folds=evidence.positive_folds, n_folds=evidence.n_folds,
                posterior_positive=evidence.posterior_positive,
                evidence_weight=evidence.evidence_weight, reasons=evidence.reasons,
            )
        return evidence
    pooled_alpha = float(np.mean(all_alpha))
    pooled_beta = float(np.mean(all_beta))
    all_g = np.concatenate([g[sl] for sl in oos_slices])
    all_m = np.concatenate([market_1d[sl] for sl in oos_slices])
    all_t = np.concatenate([t[sl] for sl in oos_slices])
    n_total = all_g.shape[0]
    alpha_resid = all_g - pooled_beta * all_m - pooled_alpha
    nw_lag_total = max(1, int(4.0 * (n_total / 100.0) ** (2.0 / 9.0)))
    nw_var_total = _newey_west_variance(alpha_resid, nw_lag_total)
    se_total = np.sqrt(nw_var_total / n_total) if n_total > 0 else 0.0
    t_alpha = pooled_alpha / max(se_total, 1e-12)
    alpha_vol = float(np.std(alpha_resid, ddof=1))
    alpha_sharpe = pooled_alpha / max(alpha_vol, 1e-12) * np.sqrt(config.bars_per_year)
    alpha_ann = pooled_alpha * config.bars_per_year
    mean_turn = float(np.mean(all_t))
    be_cost = compute_breakeven_cost_bps(alpha_ann, mean_turn, config.bars_per_year)
    if n_possible > 0:
        boot_draws = np.random.default_rng(42).choice(
            all_alpha, size=(config.n_bootstrap, n_possible), replace=True,
        )
        boot_mean = np.mean(boot_draws, axis=1)
        posterior = float(np.mean(boot_mean > 0.0))
    else:
        posterior = 0.0
    evidence = LegEvidence(
        concept_id=leg.spec.concept_id, mode=leg.spec.mode,
        n_oos_bars=n_total,
        alpha_ann=alpha_ann, beta_market=pooled_beta,
        alpha_sharpe=alpha_sharpe, t_alpha_newey_west=t_alpha,
        breakeven_cost_bps=be_cost,
        mean_turnover_per_bar=mean_turn,
        positive_folds=positive_folds, n_folds=n_possible,
        posterior_positive=posterior, evidence_weight=0.0,
        reasons=(),
    )
    evidence = dataclasses.replace(
        evidence, evidence_weight=compute_evidence_weight(evidence, cost_bps, config),
    )
    if recorder.enabled:
        recorder.record_leg(
            concept_id=evidence.concept_id, mode=evidence.mode,
            alpha_ann=evidence.alpha_ann, beta_market=evidence.beta_market,
            alpha_sharpe=evidence.alpha_sharpe, t_alpha=evidence.t_alpha_newey_west,
            breakeven_cost_bps=evidence.breakeven_cost_bps,
            mean_turnover_per_bar=evidence.mean_turnover_per_bar,
            positive_folds=evidence.positive_folds, n_folds=evidence.n_folds,
            posterior_positive=evidence.posterior_positive,
            evidence_weight=evidence.evidence_weight, reasons=evidence.reasons,
        )
    return evidence
