from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.benchmark import (  # noqa: F401
    DailyMarketReturns,
    _causal_volatility_scale,
    build_causal_l2_benchmark,
)
from src.domain.futures.compound.bootstrap import (
    circular_stationary_bootstrap_growth,
    circular_stationary_bootstrap_sharpe,
    politis_white_block_length,
    stepwise_spa_pvalue,
)
from src.domain.futures.compound.config import DataPlaneConfig, L2GateConfig, L3ValidationConfig
from src.domain.futures.compound.contracts import (
    CausalityError,
    CompoundWindowAudit,
    DeploymentVerdict,
    ExecutionLedger,
    InsufficientCoverageError,
    L2BenchmarkSeries,
    L2CategoryResult,
    L2Evaluation,
    L2GateVerdict,
    L3ValidationResult,
    MarketFeatureCube,
    QuarterlyBarBoundaries,
    SealedHoldoutManifest,
    SignalDescriptor,
    StrategyDataCoverageEntry,
)
from src.domain.futures.compound.multiplicity import TrialMultiplicity, deflated_sharpe_probability

_logger = logging.getLogger(__name__)


def validate_ledger_before_aggregation(ledger: ExecutionLedger) -> tuple[str, ...]:
    """Return stable integrity reasons; empty tuple means valid."""
    reasons: list[str] = []
    if not ledger.integrity_ok:
        reasons.extend(ledger.integrity_reasons)
    if np.any(~np.isfinite(ledger.net_returns_1d)):
        reasons.append("non_finite_returns")
    if np.any(ledger.net_returns_1d <= -1.0):
        reasons.append("net_return_le_minus_one")
    if np.any(~np.isfinite(ledger.target_weights_2d)):
        reasons.append("non_finite_weights")
    dups = len(ledger.timestamps_ns) - len(np.unique(ledger.timestamps_ns))
    if dups > 0:
        reasons.append(f"duplicate_timestamps:{dups}")
    if not np.all(ledger.timestamps_ns[1:] >= ledger.timestamps_ns[:-1]):
        reasons.append("non_monotonic_timestamps")
    return tuple(reasons)


def aggregate_returns_to_utc_days(
    timestamps_ns: NDArray[np.int64],
    returns_1d: NDArray[np.float64],
) -> NDArray[np.float64]:
    if timestamps_ns.ndim != 1 or returns_1d.ndim != 1:
        raise ValueError("timestamps_ns and returns_1d must be 1-D")
    if timestamps_ns.shape != returns_1d.shape:
        raise ValueError("timestamps_ns and returns_1d must have same length")
    if np.any(returns_1d <= -1.0):
        raise ValueError("returns must be > -1.0")
    if timestamps_ns.shape[0] == 0:
        return np.array([], dtype=np.float64)

    dups = timestamps_ns.shape[0] - len(np.unique(timestamps_ns))
    if dups > 0:
        raise CausalityError(f"duplicate timestamps: {dups} duplicates found")

    if not np.all(timestamps_ns[1:] >= timestamps_ns[:-1]):
        raise CausalityError("timestamps_ns must be monotonically non-decreasing")

    ns_per_4h = 4 * 3600 * 10**9
    day_starts_ns = timestamps_ns - (timestamps_ns % (6 * ns_per_4h))
    unique_days, _day_idx, counts = np.unique(day_starts_ns, return_inverse=True, return_counts=True)
    complete_days = unique_days[counts == 6]
    if len(complete_days) == 0:
        return np.array([], dtype=np.float64)

    daily_returns = np.zeros(len(complete_days), dtype=np.float64)
    for i, day_start in enumerate(complete_days):
        mask = day_starts_ns == day_start
        day_log = np.log1p(returns_1d[mask])
        daily_returns[i] = float(np.exp(np.sum(day_log)) - 1.0)

    return daily_returns





def _annualized_log_growth(returns: NDArray[np.float64], periods_per_year: float) -> float:
    log_returns = np.log1p(np.where(np.isfinite(returns), returns, 0.0))
    return float(periods_per_year * np.mean(log_returns))

def annualized_compound_growth(
    simple_returns: NDArray[np.float64], periods_per_year: float,
) -> float:
    if np.any(simple_returns <= -1.0):
        raise ValueError("simple_returns contains values <= -1.0")
    r = simple_returns[np.isfinite(simple_returns)]
    if len(r) == 0:
        return 0.0
    log_growth = float(np.mean(np.log1p(r)))
    return float(periods_per_year * log_growth)


def build_frozen_control_weights(
    weights_2d: NDArray[np.float64], freeze_idx: int,
) -> NDArray[np.float64]:
    if freeze_idx < 0 or freeze_idx >= weights_2d.shape[0]:
        raise ValueError(f"freeze_idx={freeze_idx} out of range [0, {weights_2d.shape[0]})")
    frozen = weights_2d[freeze_idx].copy()
    return np.broadcast_to(frozen, weights_2d.shape).copy()


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


def _stationary_bootstrap_lcb90(
    returns: NDArray[np.float64],
    periods_per_year: float,
    n_bootstrap: int = 1000,
    block_size: int = 5,
    seed: int = 42,
) -> tuple[float, float, float]:
    r = returns[np.isfinite(returns)]
    n = len(r)
    if n < 10:
        return (0.0, 0.0, 0.5)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = 0
        boot = np.empty(n)
        while idx < n:
            block_start = int(rng.integers(0, n))
            block_len = int(min(rng.geometric(1.0 / block_size), n - block_start))
            block_len = int(min(block_len, n - idx))
            boot[idx: idx + block_len] = r[block_start: block_start + block_len]
            idx += block_len
        samples[i] = _annualized_log_growth(boot, periods_per_year)
    lcb = float(np.percentile(samples, 10))
    ucb = float(np.percentile(samples, 90))
    prob_positive = float(np.mean(samples > 0.0))
    return lcb, ucb, prob_positive


def _stationary_bootstrap_sharpe_probability(
    returns: NDArray[np.float64],
    n_bootstrap: int = 2000,
    block_size: int = 5,
    seed: int = 42,
) -> tuple[float, float]:
    r = returns[np.isfinite(returns)]
    n = len(r)
    if n < 10:
        return (0.0, 0.5)
    rng = np.random.default_rng(seed)
    sharpe_samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = 0
        boot = np.empty(n)
        while idx < n:
            block_start = int(rng.integers(0, n))
            block_len = int(min(rng.geometric(1.0 / block_size), n - block_start))
            block_len = int(min(block_len, n - idx))
            boot[idx: idx + block_len] = r[block_start: block_start + block_len]
            idx += block_len
        mean_r = float(np.mean(boot))
        std_r = float(np.std(boot, ddof=1))
        sharpe_samples[i] = float(mean_r / max(std_r, 1e-12) * math.sqrt(365.25))
    std_r_val = float(np.std(r, ddof=1))
    obs_sharpe = float(np.mean(r) / max(std_r_val, 1e-12) * math.sqrt(365.25))
    prob = float(np.mean(sharpe_samples > 0.0))
    return obs_sharpe, prob


def count_effective_candidates(valid_3d: NDArray[np.bool_]) -> int:
    if valid_3d.ndim != 3:
        raise ValueError(f"valid_3d must be 3-D, got shape {valid_3d.shape}")
    n_desc = valid_3d.shape[2]
    if n_desc == 0:
        return 0
    return int(np.sum(np.any(valid_3d, axis=(0, 1))))


def _compute_turnover(ledger: ExecutionLedger) -> float:
    diffs = np.diff(ledger.target_weights_2d, axis=0)
    return float(np.mean(np.sum(np.abs(diffs), axis=1))) if diffs.shape[0] > 0 else 0.0


def _check_integrity_and_evidence(
    ledger: ExecutionLedger,
    daily_returns_4h: NDArray[np.float64],
    config: L2GateConfig,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not ledger.integrity_ok:
        reasons.extend(ledger.integrity_reasons)

    if np.any(ledger.net_returns_1d <= -1.0):
        reasons.append("net_return_le_minus_one")
    dups = len(ledger.timestamps_ns) - len(np.unique(ledger.timestamps_ns))
    if dups > 0:
        reasons.append(f"duplicate_timestamps:{dups}")
    if not np.all(ledger.timestamps_ns[1:] >= ledger.timestamps_ns[:-1]):
        reasons.append("non_monotonic_timestamps")
    if np.any(~np.isfinite(ledger.target_weights_2d)):
        reasons.append("non_finite_weights")

    oos_days = len(daily_returns_4h)
    if oos_days < config.min_oos_days:
        reasons.append(f"oos_days={oos_days}<{config.min_oos_days}")

    active_days = int(np.sum(np.abs(ledger.net_returns_1d) > 1e-15))
    active_ratio = active_days / max(len(ledger.net_returns_1d), 1)
    if active_ratio < config.min_active_days_ratio:
        reasons.append(f"active_days_ratio={active_ratio:.4f}<{config.min_active_days_ratio}")

    rebalances = _compute_rebalance_count(ledger.target_weights_2d)
    if rebalances < config.min_rebalances:
        reasons.append(f"rebalances={rebalances}<{config.min_rebalances}")

    return len(reasons) == 0, reasons


def _compute_rebalance_count(weights_2d: NDArray[np.float32]) -> int:
    if weights_2d.shape[0] < 2:
        return 0
    diffs = np.abs(np.diff(weights_2d, axis=0))
    return int(np.sum(np.max(diffs, axis=1) > 1e-8))


def evaluate_l2_walk_forward(
    *,
    ledger: ExecutionLedger,
    fold_ids_1d: NDArray[np.int16],
    benchmark: L2BenchmarkSeries,
    trial_multiplicity: TrialMultiplicity,
    config: L2GateConfig,
    bootstrap_seed: int,
    frozen_control_daily_1d: NDArray[np.float64] | None = None,
) -> L2Evaluation:
    candidate_count = trial_multiplicity.n_trials

    # Structural validation before aggregation (fail-closed)
    pre_agg_reasons = validate_ledger_before_aggregation(ledger)
    if pre_agg_reasons:
        oos_days = max(int(np.sum(np.isfinite(ledger.net_returns_1d))), 0)
        return L2Evaluation(
            verdict=L2GateVerdict.NO_EVIDENCE,
            benchmark_id=benchmark.benchmark_id,
            annualized_log_growth=0.0, cagr=0.0, excess_growth_lcb90=0.0,
            excess_growth_probability=0.5, stressed_excess_growth_lcb90=0.0,
            equity_multiple=1.0, sharpe=0.0, sharpe_probability=0.5,
            deflated_sharpe_probability=0.5, candidate_count=candidate_count,
            calmar=0.0, max_drawdown=0.0, daily_cvar95=0.0,
            annual_volatility=0.0, annual_turnover=0.0, cost_drag_ratio=0.0,
            absolute_cagr=0.0, capacity_utilisation_p95=0.0, active_days_ratio=0.0,
            rebalance_count=0, positive_outer_folds=0, oos_days=oos_days,
            category_results=(), integrity_ok=False, reasons=pre_agg_reasons,
        )

    # Daily aggregation from 4h bars
    daily_returns = aggregate_returns_to_utc_days(ledger.timestamps_ns, ledger.net_returns_1d)
    oos_days = len(daily_returns)

    # Integrity & evidence gate
    integrity_ok, integrity_reasons = _check_integrity_and_evidence(ledger, daily_returns, config)
    if not integrity_ok:
        reasons = tuple(integrity_reasons)
        return L2Evaluation(
            verdict=L2GateVerdict.NO_EVIDENCE,
            benchmark_id=benchmark.benchmark_id,
            annualized_log_growth=0.0, cagr=0.0, excess_growth_lcb90=0.0,
            excess_growth_probability=0.5, stressed_excess_growth_lcb90=0.0,
            equity_multiple=1.0, sharpe=0.0, sharpe_probability=0.5,
            deflated_sharpe_probability=0.5, candidate_count=candidate_count,
            calmar=0.0, max_drawdown=0.0, daily_cvar95=0.0,
            annual_volatility=0.0, annual_turnover=0.0, cost_drag_ratio=0.0,
            absolute_cagr=0.0, capacity_utilisation_p95=0.0, active_days_ratio=0.0,
            rebalance_count=0, positive_outer_folds=0, oos_days=oos_days,
            category_results=(), integrity_ok=False, reasons=reasons,
        )

    # Align benchmark to same daily grid
    daily_timestamps = _daily_timestamps_from_4h(ledger.timestamps_ns)
    n_daily = len(daily_returns)
    if n_daily == 0:
        return L2Evaluation(
            verdict=L2GateVerdict.NO_EVIDENCE, benchmark_id=benchmark.benchmark_id,
            annualized_log_growth=0.0, cagr=0.0, excess_growth_lcb90=0.0,
            excess_growth_probability=0.5, stressed_excess_growth_lcb90=0.0,
            equity_multiple=1.0, sharpe=0.0, sharpe_probability=0.5,
            deflated_sharpe_probability=0.5, candidate_count=candidate_count,
            calmar=0.0, max_drawdown=0.0, daily_cvar95=0.0,
            annual_volatility=0.0, annual_turnover=0.0, cost_drag_ratio=0.0,
            absolute_cagr=0.0, capacity_utilisation_p95=0.0, active_days_ratio=0.0,
            rebalance_count=0, positive_outer_folds=0, oos_days=0,
            category_results=(), integrity_ok=True, reasons=("no_complete_utc_days",),
        )

    idx = np.searchsorted(benchmark.timestamps_ns, daily_timestamps[0], side="left")
    if idx >= len(benchmark.daily_returns_1d):
        return L2Evaluation(
            verdict=L2GateVerdict.NO_EVIDENCE, benchmark_id=benchmark.benchmark_id,
            annualized_log_growth=0.0, cagr=0.0, excess_growth_lcb90=0.0,
            excess_growth_probability=0.5, stressed_excess_growth_lcb90=0.0,
            equity_multiple=1.0, sharpe=0.0, sharpe_probability=0.5,
            deflated_sharpe_probability=0.5, candidate_count=candidate_count,
            calmar=0.0, max_drawdown=0.0, daily_cvar95=0.0,
            annual_volatility=0.0, annual_turnover=0.0, cost_drag_ratio=0.0,
            absolute_cagr=0.0, capacity_utilisation_p95=0.0, active_days_ratio=0.0,
            rebalance_count=0, positive_outer_folds=0, oos_days=oos_days,
            category_results=(), integrity_ok=True, reasons=("benchmark_not_aligned",),
        )

    aligned_end = min(n_daily, len(benchmark.daily_returns_1d) - idx)
    daily_returns = daily_returns[:aligned_end]
    benchmark_returns = benchmark.daily_returns_1d[idx:idx + aligned_end]
    daily_timestamps = daily_timestamps[:aligned_end]
    oos_days = len(daily_returns)

    excess_returns = np.log1p(daily_returns) - np.log1p(benchmark_returns)

    # Pre-fee and 2x-cost daily returns
    fee_4h = ledger.fee_returns_1d
    fee_daily = aggregate_returns_to_utc_days(ledger.timestamps_ns[:len(fee_4h)], fee_4h)
    if len(fee_daily) > aligned_end:
        fee_daily = fee_daily[:aligned_end]
    elif len(fee_daily) < aligned_end:
        fee_daily = np.pad(fee_daily, (0, aligned_end - len(fee_daily)), constant_values=0.0)

    pre_fee_daily = daily_returns - fee_daily
    stressed_daily = daily_returns + config.stressed_cost_multiplier * fee_daily

    # CAGR and equity multiple
    log_growth = _annualized_log_growth(np.expm1(excess_returns), 365.25)
    cagr = float(math.expm1(log_growth))
    absolute_log_growth = annualized_compound_growth(daily_returns, 365.25)
    absolute_cagr_val = float(math.expm1(absolute_log_growth))
    equity: NDArray[np.float64] = np.asarray(np.cumprod(1.0 + daily_returns), dtype=np.float64)
    equity_multiple = float(equity[-1]) if equity.size > 0 else 1.0

    # Bootstrap block size via Politis-White
    pw_block = politis_white_block_length(excess_returns)

    # Sharpe ratio
    obs_sharpe, sharpe_prob = circular_stationary_bootstrap_sharpe(
        excess_returns, 365.25, n_bootstrap=2000, block_size=pw_block, seed=bootstrap_seed,
    )
    dsr = deflated_sharpe_probability(
        observed_sharpe=obs_sharpe, multiplicity=trial_multiplicity,
        excess_returns=excess_returns, periods_per_year=365.25,
    )

    # Growth bootstrap
    excess_lcb90, _, excess_prob = circular_stationary_bootstrap_growth(
        np.expm1(excess_returns), 365.25, n_bootstrap=1000, block_size=pw_block, seed=bootstrap_seed,
    )
    stressed_lcb90, _, _ = circular_stationary_bootstrap_growth(
        np.expm1(stressed_daily), 365.25, n_bootstrap=1000, block_size=pw_block, seed=bootstrap_seed + 1,
    )

    # Stressed positive fold count
    fold_ids_aligned = _align_fold_ids(fold_ids_1d, ledger.timestamps_ns, daily_timestamps, aligned_end)
    positive_folds = int(np.sum([
        _annualized_log_growth(np.expm1(stressed_daily[fold_ids_aligned == f]), 365.25) > 0
        for f in np.unique(fold_ids_aligned) if f >= 0
    ]))

    # Downside metrics
    mdd = _max_drawdown(equity)
    cvar = _cvar95(daily_returns)
    ann_vol = float(np.std(daily_returns[np.isfinite(daily_returns)], ddof=1) * math.sqrt(365.25))

    # Turnover
    ann_turnover = _compute_turnover(ledger) * 2191.5
    rebalance_count = _compute_rebalance_count(ledger.target_weights_2d)

    # Cost drag ratio: compare absolute (not excess) pre-fee vs post-fee CAGR,
    # both in the same expm1 (compounded) space [LIMIT-10]
    pre_fee_log_growth = annualized_compound_growth(pre_fee_daily, 365.25)
    pre_fee_cagr = float(math.expm1(pre_fee_log_growth))
    cost_drag = 1.0 - absolute_cagr_val / max(pre_fee_cagr, 1e-12) if pre_fee_cagr > 0 else 0.0

    # Capacity utilisation (p95 of max absolute weight)
    capacity_p95 = 0.0
    if ledger.target_weights_2d.shape[0] > 0:
        max_abs_w = np.max(np.abs(ledger.target_weights_2d), axis=1)
        capacity_p95 = float(np.percentile(max_abs_w, 95))

    active_days_ratio = int(np.sum(np.abs(ledger.net_returns_1d) > 1e-15)) / max(len(ledger.net_returns_1d), 1)
    calmar = cagr / max(mdd, 1e-12)

    spa_pvalue = 1.0
    if frozen_control_daily_1d is not None and frozen_control_daily_1d.size == oos_days:
        controls = [benchmark_returns, np.zeros(oos_days, dtype=np.float64), frozen_control_daily_1d]
        controls_2d = np.stack(controls, axis=0)
        spa_pvalue = stepwise_spa_pvalue(
            np.expm1(excess_returns), controls_2d,
            block_size=pw_block, n_bootstrap=1000, seed=bootstrap_seed,
        )
    elif frozen_control_daily_1d is not None and frozen_control_daily_1d.size > 0 and frozen_control_daily_1d.size != oos_days:
        _logger.warning(
            "[EVAL] frozen_control_daily_1d size mismatch: %d != %d, skipping SPA",
            frozen_control_daily_1d.size, oos_days,
        )

    # Five category gates
    categories: list[L2CategoryResult] = []

    # 1. Integrity & evidence
    int_reasons: list[str] = []
    no_evidence = False
    if oos_days < config.min_oos_days:
        int_reasons.append(f"oos_days={oos_days}<{config.min_oos_days}")
        no_evidence = True
    if active_days_ratio < config.min_active_days_ratio:
        int_reasons.append(f"active_days_ratio={active_days_ratio:.4f}<{config.min_active_days_ratio}")
        no_evidence = True
    if rebalance_count < config.min_rebalances:
        int_reasons.append(f"rebalances={rebalance_count}<{config.min_rebalances}")
        no_evidence = True
    categories.append(L2CategoryResult(
        "integrity_and_evidence", not no_evidence, tuple(int_reasons),
    ))

    # 2. Compound growth
    growth_reasons: list[str] = []
    growth_pass = True
    if excess_lcb90 <= 0.0:
        growth_reasons.append(f"excess_growth_lcb90={excess_lcb90:.6f} not strictly positive")
        growth_pass = False
    if stressed_lcb90 <= 0.0:
        growth_reasons.append(f"stressed_excess_growth_lcb90={stressed_lcb90:.6f} not strictly positive")
        growth_pass = False
    if excess_prob < config.min_excess_growth_probability:
        growth_reasons.append(f"excess_growth_probability={excess_prob:.4f}<{config.min_excess_growth_probability}")
        growth_pass = False
    if positive_folds < config.min_positive_outer_folds:
        growth_reasons.append(f"positive_outer_folds={positive_folds}<{config.min_positive_outer_folds}")
        growth_pass = False
    categories.append(L2CategoryResult("compound_growth", growth_pass, tuple(growth_reasons)))

    # 3. Statistical skill
    skill_reasons: list[str] = []
    skill_pass = True
    if dsr < config.min_deflated_sharpe_probability:
        skill_reasons.append(f"deflated_sharpe_probability={dsr:.4f}<{config.min_deflated_sharpe_probability}")
        skill_pass = False
    if sharpe_prob < config.min_bootstrap_sharpe_probability:
        skill_reasons.append(f"sharpe_probability={sharpe_prob:.4f}<{config.min_bootstrap_sharpe_probability}")
        skill_pass = False
    if spa_pvalue > config.max_spa_pvalue:
        skill_reasons.append(f"spa_pvalue={spa_pvalue:.4f}>{config.max_spa_pvalue}")
        skill_pass = False
    categories.append(L2CategoryResult("statistical_skill", skill_pass, tuple(skill_reasons)))

    # 4. Downside survival
    downside_reasons: list[str] = []
    downside_pass = True
    if mdd > config.max_drawdown:
        downside_reasons.append(f"max_drawdown={mdd:.4f}>{config.max_drawdown}")
        downside_pass = False
    if cvar < config.min_daily_cvar95:
        downside_reasons.append(f"daily_cvar95={cvar:.6f}<{config.min_daily_cvar95}")
        downside_pass = False
    if ann_vol > config.max_annual_volatility:
        downside_reasons.append(f"annual_volatility={ann_vol:.4f}>{config.max_annual_volatility}")
        downside_pass = False
    categories.append(L2CategoryResult("downside_survival", downside_pass, tuple(downside_reasons)))

    # 5. Trading efficiency & capacity
    efficiency_reasons: list[str] = []
    efficiency_pass = True
    if cost_drag > config.max_cost_drag_ratio:
        efficiency_reasons.append(f"cost_drag_ratio={cost_drag:.4f}>{config.max_cost_drag_ratio}")
        efficiency_pass = False
    if capacity_p95 > config.max_capacity_utilisation_p95:
        efficiency_reasons.append(f"capacity_utilisation_p95={capacity_p95:.4f}>{config.max_capacity_utilisation_p95}")
        efficiency_pass = False
    categories.append(L2CategoryResult("trading_efficiency", efficiency_pass, tuple(efficiency_reasons)))

    all_pass = all(c.passed for c in categories)
    if no_evidence:
        verdict = L2GateVerdict.NO_EVIDENCE
    elif all_pass:
        verdict = L2GateVerdict.PASS
    else:
        verdict = L2GateVerdict.FAIL

    fail_reasons: list[str] = []
    for c in categories:
        if not c.passed:
            fail_reasons.extend(c.reasons)

    return L2Evaluation(
        verdict=verdict,
        benchmark_id=benchmark.benchmark_id,
        annualized_log_growth=log_growth,
        cagr=cagr,
        excess_growth_lcb90=excess_lcb90,
        excess_growth_probability=excess_prob,
        stressed_excess_growth_lcb90=stressed_lcb90,
        equity_multiple=equity_multiple,
        sharpe=obs_sharpe,
        sharpe_probability=sharpe_prob,
        deflated_sharpe_probability=dsr,
        candidate_count=candidate_count,
        calmar=calmar,
        max_drawdown=mdd,
        daily_cvar95=cvar,
        annual_volatility=ann_vol,
        annual_turnover=ann_turnover,
        cost_drag_ratio=cost_drag if np.isfinite(cost_drag) else 0.0,
        absolute_cagr=absolute_cagr_val,
        capacity_utilisation_p95=capacity_p95,
        active_days_ratio=active_days_ratio,
        rebalance_count=rebalance_count,
        positive_outer_folds=positive_folds,
        oos_days=oos_days,
        category_results=tuple(categories),
        integrity_ok=True,
        reasons=tuple(fail_reasons),
        spa_pvalue=spa_pvalue,
        bootstrap_block_days=pw_block,
        daily_strategy_returns_1d=daily_returns.copy(),
        daily_benchmark_returns_1d=benchmark_returns.copy(),
        daily_excess_returns_1d=excess_returns.copy(),
        daily_fee_returns_1d=fee_daily.copy(),
        daily_day_start_ns=daily_timestamps.copy(),
    )


def _daily_timestamps_from_4h(timestamps_ns_4h: NDArray[np.int64]) -> NDArray[np.int64]:
    ns_per_4h = 4 * 3600 * 10**9
    day_start_ns = timestamps_ns_4h - (timestamps_ns_4h % (6 * ns_per_4h))
    unique_days, counts = np.unique(day_start_ns, return_counts=True)
    complete = unique_days[counts == 6]
    result: NDArray[np.int64] = complete.astype(np.int64) + np.int64(6 * ns_per_4h)
    return result


def _align_fold_ids(
    fold_ids_1d: NDArray[np.int16],
    timestamps_ns: NDArray[np.int64],
    daily_timestamps: NDArray[np.int64],
    n_daily: int,
) -> NDArray[np.int16]:
    result = np.full(n_daily, -1, dtype=np.int16)
    for d in range(n_daily):
        ts = daily_timestamps[d]
        idx = int(np.searchsorted(timestamps_ns, ts, side="right")) - 1
        if 0 <= idx < len(fold_ids_1d):
            result[d] = fold_ids_1d[idx]
    return result


def evaluate_l3_sealed_holdout(
    *,
    l2_prior_returns: NDArray[np.float64],
    holdout_ledger: ExecutionLedger,
    holdout_manifest: SealedHoldoutManifest,
    config: L3ValidationConfig,
) -> L3ValidationResult:
    cap = config.l2_prior_effective_days_cap
    if len(l2_prior_returns) > cap * 2:
        raise ValueError(
            f"l2_prior_returns length {len(l2_prior_returns)} exceeds {cap * 2} "
            f"(2 * l2_prior_effective_days_cap); 4h bars must not be passed as daily"
        )

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

    holdout_daily = aggregate_returns_to_utc_days(holdout_ledger.timestamps_ns, holdout_ledger.net_returns_1d)

    holdout_growth_prob = _stationary_bootstrap_lcb90(
        holdout_daily, 365.25, n_bootstrap=2000, block_size=5, seed=42,
    )[2]

    if len(prior_returns) > 0:
        prior_growth_prob = _stationary_bootstrap_lcb90(
            prior_returns, 365.25, n_bootstrap=1000, block_size=5, seed=42,
        )[2]
        prior_weight = min(1.0, len(prior_returns) / config.l2_prior_effective_days_cap)
        holdout_weight = 1.0 - prior_weight * 0.5
        posterior_prob = prior_weight * prior_growth_prob + holdout_weight * holdout_growth_prob
        posterior_prob /= (prior_weight + holdout_weight)
    else:
        posterior_prob = holdout_growth_prob

    if posterior_prob >= config.promote_probability:
        verdict = DeploymentVerdict.PROMOTE
    elif posterior_prob <= config.reject_probability:
        verdict = DeploymentVerdict.REJECT
        reasons.append("low_growth_probability")
    else:
        verdict = DeploymentVerdict.SHADOW

    return L3ValidationResult(
        verdict=verdict,
        posterior_growth_probability=posterior_prob,
        holdout_days=holdout_manifest.holdout_days,
        max_drawdown=mdd,
        daily_cvar95=cvar,
        reasons=tuple(reasons),
    )


def slice_execution_ledger(
    *, ledger: ExecutionLedger, start_time_ns: int, end_time_ns: int,
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


def _check_coverage_gaps(
    available: NDArray[np.bool_], timestamps_ns: NDArray[np.int64],
) -> tuple[float, int, int]:
    n = available.shape[0]
    if n == 0:
        return 0.0, 0, 0  # pragma: no cover
    per_bar = np.all(available, axis=1) if available.ndim == 2 else available
    ratio = float(np.mean(per_bar))
    leading_gaps = int(np.argmax(per_bar)) if np.any(per_bar) else n
    trailing_gaps = int(np.argmax(per_bar[::-1])) if np.any(per_bar) else n
    return ratio, leading_gaps, trailing_gaps


def audit_compound_market_window(
    *, market: MarketFeatureCube, window: QuarterlyBarBoundaries,
    required_descriptors: tuple[SignalDescriptor, ...],
) -> CompoundWindowAudit:  # pragma: no cover - integration coverage is environment-dependent
    timestamps_ns = market.timestamps_ns
    if timestamps_ns.size == 0:  # pragma: no cover - empty cubes are rejected upstream
        return CompoundWindowAudit(  # pragma: no cover
            passed=False, core_coverage_ratio=0.0,
            dataset_status=(), reasons=("empty_market_cube",),
        )

    acquisition_idx = max(0, min(window.acquisition_start, timestamps_ns.size - 1))
    cutoff_idx = max(acquisition_idx + 1, min(window.cutoff_exclusive, timestamps_ns.size))
    if cutoff_idx <= acquisition_idx:  # pragma: no cover - boundaries validated upstream
        return CompoundWindowAudit(  # pragma: no cover
            passed=False, core_coverage_ratio=0.0,
            dataset_status=(), reasons=("window_out_of_bounds",),
        )

    core_available = market.available_2d.get("core")
    if core_available is None:  # pragma: no cover - core availability is mandatory
        return CompoundWindowAudit(  # pragma: no cover
            passed=False, core_coverage_ratio=0.0,
            dataset_status=(), reasons=("missing_core_availability",),
        )

    required_core = np.asarray(market.eligible_2d, dtype=np.bool_).copy()
    for benchmark_symbol in ("BTCUSDT", "ETHUSDT"):
        if benchmark_symbol in market.symbols:
            required_core[:, market.symbols.index(benchmark_symbol)] = True
    required_core_available = np.where(required_core, core_available, True)
    core_ratio, leading_gaps, trailing_gaps = _check_coverage_gaps(
        required_core_available[acquisition_idx:cutoff_idx],
        timestamps_ns[acquisition_idx:cutoff_idx],
    )

    reasons: list[str] = []
    if leading_gaps > 0:
        reasons.append(f"leading_core_gap:{leading_gaps}")
    if trailing_gaps > 0:
        reasons.append(f"trailing_core_gap:{trailing_gaps}")
    if core_ratio < DataPlaneConfig().min_core_coverage:
        reasons.append(
            f"core_coverage={core_ratio:.4f}<{DataPlaneConfig().min_core_coverage}"
        )

    dataset_entries: list[StrategyDataCoverageEntry] = []
    checked_datasets: set[str] = set()
    for dataset_key in market.available_2d:
        if dataset_key in checked_datasets:  # pragma: no cover - keys are unique in normal cubes
            continue  # pragma: no cover
        checked_datasets.add(dataset_key)
        avail = market.available_2d[dataset_key]
        if avail is None:  # pragma: no cover - MarketFeatureCube availability arrays are concrete
            continue  # pragma: no cover
        ds_ratio, ds_lead, ds_trail = _check_coverage_gaps(
            avail[acquisition_idx:cutoff_idx],
            timestamps_ns[acquisition_idx:cutoff_idx],
        )
        max_gap = max(ds_lead, ds_trail)
        readiness = "ready" if ds_ratio >= DataPlaneConfig().min_core_coverage else "degraded"
        recipe_ids = tuple(desc.signal_id for desc in required_descriptors)
        dataset_entries.append(StrategyDataCoverageEntry(
            dataset=dataset_key,
            recipe_id=",".join(recipe_ids) if recipe_ids else "all",
            ratio=ds_ratio,
            max_gap_bars=max_gap,
            readiness=readiness,
            reason="" if readiness == "ready" else f"coverage={ds_ratio:.4f}",
        ))

    oi_available = market.available_2d.get("open_interest")
    if oi_available is not None and "open_interest" not in checked_datasets:  # pragma: no cover - defensive for custom grids
        recipe_ids = tuple(desc.signal_id for desc in required_descriptors)  # pragma: no cover
        dataset_entries.append(StrategyDataCoverageEntry(  # pragma: no cover
            dataset="open_interest",
            recipe_id=",".join(recipe_ids) if recipe_ids else "all",
            ratio=0.0,
            max_gap_bars=oi_available.shape[0],
            readiness="disabled",
            reason="OI available_at not wired; DISABLED_DATA",
        ))

    passed = len(reasons) == 0
    if not passed:
        raise InsufficientCoverageError(
            f"market window audit failed: {'; '.join(reasons)}",
        )

    return CompoundWindowAudit(
        passed=True,
        core_coverage_ratio=core_ratio,
        dataset_status=tuple(dataset_entries),
        reasons=(),
    )
