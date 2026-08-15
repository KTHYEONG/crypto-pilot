from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm as sp_norm

from src.domain.futures.compound.config import L1LegConfig
from src.domain.futures.compound.contracts import (
    LegBook,
    LegEvidence,
)
from src.domain.futures.compound.l1_diagnostics import L1AdmissionRecorder
from src.domain.futures.compound.l1_leg_admission import screen_leg_evidence

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
    *,
    n_tested_hypotheses: int = 1,
) -> LegEvidence:
    g = leg.gross_return_1d
    t = leg.turnover_1d
    n_all_alpha_gross: list[float] = []
    n_all_alpha_net: list[float] = []
    n_possible = 0
    positive_folds_net = 0
    all_beta: list[float] = []
    for sl in oos_slices:
        g_oos = g[sl]
        m_oos = market_1d[sl]
        t_oos = t[sl]
        n = g_oos.shape[0]
        if n < 5:
            continue
        n_possible += 1
        var_m = float(np.var(m_oos, ddof=1))
        beta = 0.0 if var_m < 1e-12 else float(np.cov(m_oos, g_oos, ddof=1)[0, 1]) / var_m
        gross_alpha_fold = float(np.mean(g_oos)) - beta * float(np.mean(m_oos))
        net_alpha_fold = float(np.mean(g_oos - cost_bps * 1e-4 * t_oos)) - beta * float(np.mean(m_oos))
        if net_alpha_fold > 0:
            positive_folds_net += 1
        n_all_alpha_gross.append(gross_alpha_fold)
        n_all_alpha_net.append(net_alpha_fold)
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
            net_alpha_ann=0.0, net_alpha_sharpe=0.0,
            t_net_alpha_newey_west=0.0,
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
    pooled_beta = float(np.mean(all_beta))
    all_g = np.concatenate([g[sl] for sl in oos_slices])
    all_m = np.concatenate([market_1d[sl] for sl in oos_slices])
    all_t = np.concatenate([t[sl] for sl in oos_slices])
    all_n = all_g - cost_bps * 1e-4 * all_t
    n_total = all_g.shape[0]
    pooled_gross_alpha = float(np.mean(n_all_alpha_gross))
    pooled_net_alpha = float(np.mean(n_all_alpha_net))
    gross_resid = all_g - pooled_beta * all_m - pooled_gross_alpha
    net_resid = all_n - pooled_beta * all_m - pooled_net_alpha
    nw_lag_total = max(1, int(4.0 * (n_total / 100.0) ** (2.0 / 9.0)))
    nw_var_gross = _newey_west_variance(gross_resid, nw_lag_total)
    nw_var_net = _newey_west_variance(net_resid, nw_lag_total)
    se_gross = np.sqrt(nw_var_gross / n_total) if n_total > 0 else 0.0
    se_net = np.sqrt(nw_var_net / n_total) if n_total > 0 else 0.0
    t_gross_alpha = pooled_gross_alpha / max(se_gross, 1e-12)
    t_net_alpha = pooled_net_alpha / max(se_net, 1e-12)
    gross_vol = float(np.std(gross_resid, ddof=1))
    net_vol = float(np.std(net_resid, ddof=1))
    gross_sharpe = pooled_gross_alpha / max(gross_vol, 1e-12) * np.sqrt(config.bars_per_year)
    net_sharpe = pooled_net_alpha / max(net_vol, 1e-12) * np.sqrt(config.bars_per_year)
    gross_alpha_ann = pooled_gross_alpha * config.bars_per_year
    net_alpha_ann = pooled_net_alpha * config.bars_per_year
    mean_turn = float(np.mean(all_t))
    be_cost = compute_breakeven_cost_bps(gross_alpha_ann, mean_turn, config.bars_per_year)
    if n_possible > 0:
        boot_draws = np.random.default_rng(42).choice(
            n_all_alpha_net, size=(config.n_bootstrap, n_possible), replace=True,
        )
        boot_mean = np.mean(boot_draws, axis=1)
        posterior = float(np.mean(boot_mean > 0.0))
    else:
        posterior = 0.0
    evidence = LegEvidence(
        concept_id=leg.spec.concept_id, mode=leg.spec.mode,
        n_oos_bars=n_total,
        alpha_ann=gross_alpha_ann, beta_market=pooled_beta,
        alpha_sharpe=gross_sharpe, t_alpha_newey_west=t_gross_alpha,
        breakeven_cost_bps=be_cost,
        mean_turnover_per_bar=mean_turn,
        positive_folds=positive_folds_net, n_folds=n_possible,
        posterior_positive=posterior, evidence_weight=0.0,
        reasons=(),
        net_alpha_ann=net_alpha_ann,
        net_alpha_sharpe=net_sharpe,
        t_net_alpha_newey_west=t_net_alpha,
    )
    evidence = screen_leg_evidence(evidence, cost_bps, config, n_tested_hypotheses)
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
            net_alpha_ann=evidence.net_alpha_ann,
            net_alpha_sharpe=evidence.net_alpha_sharpe,
            t_net_alpha=evidence.t_net_alpha_newey_west,
            critical_t=sp_norm.isf(config.familywise_error_rate / n_tested_hypotheses),
            n_tested_hypotheses=n_tested_hypotheses,
        )
    return evidence
