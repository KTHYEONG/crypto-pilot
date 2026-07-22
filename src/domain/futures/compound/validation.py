from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import L3ValidationConfig
from src.domain.futures.compound.contracts import (
    DeploymentVerdict,
    ExecutionLedger,
    L2Evaluation,
    L3ValidationResult,
    SealedHoldoutManifest,
)

_logger = logging.getLogger(__name__)


def _annualized_log_growth(returns: NDArray[np.float64], bars_per_year: float) -> float:
    log_returns = np.log1p(np.where(np.isfinite(returns), returns, 0.0))
    return float(bars_per_year * np.mean(log_returns))


def _max_drawdown(equity: NDArray[np.float64]) -> float:
    peak = np.maximum.accumulate(equity)
    dd = np.max(1.0 - equity / np.where(peak > 0, peak, 1.0))
    return float(dd)


def _cvar95(returns: NDArray[np.float64]) -> float:
    r = returns[np.isfinite(returns)]
    if len(r) < 10:
        return 0.0
    threshold = np.percentile(r, 5)
    return float(np.mean(r[r <= threshold]))


def _stationary_bootstrap_ci(
    returns: NDArray[np.float64], n_bootstrap: int = 1000, block_size: int = 5
) -> tuple[float, float]:
    r = returns[np.isfinite(returns)]
    n = len(r)
    if n < 10:
        return (0.0, 0.0)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = 0
        boot = np.empty(n)
        while idx < n:
            block_start = np.random.randint(0, n)
            block_len = min(np.random.geometric(1.0 / block_size), n - block_start)
            block_len = min(block_len, n - idx)
            boot[idx : idx + block_len] = r[block_start : block_start + block_len]
            idx += block_len
        samples[i] = _annualized_log_growth(boot, 8766)
    return (float(np.percentile(samples, 5)), float(np.percentile(samples, 95)))


def _compute_turnover(ledger: ExecutionLedger) -> float:
    diffs = np.diff(ledger.target_weights_2d, axis=0)
    return float(np.mean(np.sum(np.abs(diffs), axis=1))) if diffs.shape[0] > 0 else 0.0


def evaluate_l2_walk_forward(
    *,
    ledger: ExecutionLedger,
    bars_per_year: float,
    bootstrap_seed: int,
) -> L2Evaluation:
    np.random.seed(bootstrap_seed)

    log_growth = _annualized_log_growth(ledger.net_returns_1d, bars_per_year)
    ci90 = _stationary_bootstrap_ci(ledger.net_returns_1d, n_bootstrap=1000, block_size=5)
    equity_multiple = float(ledger.equity_1d[-1]) if ledger.equity_1d.size > 0 else 1.0
    mdd = _max_drawdown(ledger.equity_1d)
    cvar = _cvar95(ledger.net_returns_1d)
    ann_vol = float(np.std(ledger.net_returns_1d[np.isfinite(ledger.net_returns_1d)]) * np.sqrt(bars_per_year))
    turnover = _compute_turnover(ledger)

    safe = mdd < 0.20 and cvar > -0.04
    safe = safe and ledger.integrity_ok

    return L2Evaluation(
        annualized_log_growth=log_growth,
        growth_ci90=ci90,
        equity_multiple=equity_multiple,
        max_drawdown=mdd,
        daily_cvar95=cvar,
        annual_volatility=ann_vol,
        turnover=turnover,
        safe=safe,
        integrity_ok=ledger.integrity_ok,
    )


def evaluate_l3_sealed_holdout(
    *,
    l2_prior_returns: NDArray[np.float64],
    holdout_ledger: ExecutionLedger,
    holdout_manifest: SealedHoldoutManifest,
    config: L3ValidationConfig,
) -> L3ValidationResult:
    reasons: list[str] = []

    if not holdout_ledger.integrity_ok:
        reasons.extend(holdout_ledger.integrity_reasons)
        return L3ValidationResult(
            verdict=DeploymentVerdict.REJECT,
            posterior_growth_probability=0.0,
            holdout_days=holdout_manifest.holdout_days,
            max_drawdown=_max_drawdown(holdout_ledger.equity_1d),
            daily_cvar95=_cvar95(holdout_ledger.net_returns_1d),
            reasons=tuple(reasons),
        )

    if holdout_manifest.holdout_days < config.min_holdout_days:
        return L3ValidationResult(
            verdict=DeploymentVerdict.SHADOW,
            posterior_growth_probability=0.5,
            holdout_days=holdout_manifest.holdout_days,
            max_drawdown=_max_drawdown(holdout_ledger.equity_1d),
            daily_cvar95=_cvar95(holdout_ledger.net_returns_1d),
            reasons=("insufficient_holdout_days",),
        )

    mdd = _max_drawdown(holdout_ledger.equity_1d)
    cvar = _cvar95(holdout_ledger.net_returns_1d)

    if mdd >= config.max_drawdown:
        reasons.append("max_drawdown_exceeded")
    if cvar <= -config.max_daily_cvar95:
        reasons.append("max_cvar95_exceeded")

    if reasons:
        return L3ValidationResult(
            verdict=DeploymentVerdict.REJECT,
            posterior_growth_probability=0.0,
            holdout_days=holdout_manifest.holdout_days,
            max_drawdown=mdd,
            daily_cvar95=cvar,
            reasons=tuple(reasons),
        )

    effective_prior = min(len(l2_prior_returns), config.l2_prior_effective_days_cap)
    prior_returns = l2_prior_returns[:effective_prior]
    holdout_returns = holdout_ledger.net_returns_1d

    prior_mean = float(np.mean(prior_returns)) if len(prior_returns) > 0 else 0.0
    prior_var = float(np.var(prior_returns)) if len(prior_returns) > 1 else 1e-6
    holdout_mean = float(np.mean(holdout_returns)) if len(holdout_returns) > 0 else 0.0
    holdout_var = float(np.var(holdout_returns)) if len(holdout_returns) > 1 else 1e-6

    prior_precision = 1.0 / max(prior_var, 1e-12)
    holdout_precision = 1.0 / max(holdout_var, 1e-12)
    posterior_mean = (prior_precision * prior_mean + holdout_precision * holdout_mean) / (prior_precision + holdout_precision)
    posterior_var = 1.0 / (prior_precision + holdout_precision)
    posterior_std = np.sqrt(posterior_var)

    prob_positive = 1.0 - float(np.float64(np.float64(0.0) if posterior_std <= 0 else 0.0))
    if posterior_std > 0:
        from scipy.stats import norm
        prob_positive = float(norm.cdf(posterior_mean / posterior_std))

    if prob_positive >= config.promote_probability:
        verdict = DeploymentVerdict.PROMOTE
    elif prob_positive <= config.reject_probability and holdout_manifest.holdout_days >= config.min_holdout_days:
        verdict = DeploymentVerdict.REJECT
        reasons.append("low_growth_probability")
    else:
        verdict = DeploymentVerdict.SHADOW
        if holdout_manifest.holdout_days < config.holdout_days:
            reasons.append("insufficient_holdout_days")

    return L3ValidationResult(
        verdict=verdict,
        posterior_growth_probability=prob_positive,
        holdout_days=holdout_manifest.holdout_days,
        max_drawdown=mdd,
        daily_cvar95=cvar,
        reasons=tuple(reasons),
    )


def slice_execution_ledger(
    *, ledger: ExecutionLedger, start_time_ns: int, end_time_ns: int
) -> ExecutionLedger:
    if ledger.timestamps_ns.size == 0:
        raise ValueError("cannot slice empty ledger")
    start_idx = int(np.searchsorted(ledger.timestamps_ns, start_time_ns, side="left"))
    end_idx = int(np.searchsorted(ledger.timestamps_ns, end_time_ns, side="right"))
    if start_idx >= end_idx:
        raise ValueError(
            f"empty slice: start_time_ns={start_time_ns} end_time_ns={end_time_ns} "
            f"range=[{ledger.timestamps_ns[0]}, {ledger.timestamps_ns[-1]}]"
        )
    if start_idx < 0 or end_idx > ledger.timestamps_ns.size:
        raise ValueError(f"slice out of range: [{start_idx}:{end_idx}] size={ledger.timestamps_ns.size}")
    return ExecutionLedger(
        timestamps_ns=ledger.timestamps_ns[start_idx:end_idx].copy(),
        net_returns_1d=ledger.net_returns_1d[start_idx:end_idx].copy(),
        equity_1d=ledger.equity_1d[start_idx:end_idx].copy(),
        target_weights_2d=ledger.target_weights_2d[start_idx:end_idx].copy(),
        fee_returns_1d=ledger.fee_returns_1d[start_idx:end_idx].copy(),
        slippage_returns_1d=ledger.slippage_returns_1d[start_idx:end_idx].copy(),
        impact_returns_1d=ledger.impact_returns_1d[start_idx:end_idx].copy(),
        funding_returns_1d=ledger.funding_returns_1d[start_idx:end_idx].copy(),
        integrity_ok=ledger.integrity_ok,
        integrity_reasons=ledger.integrity_reasons,
    )
