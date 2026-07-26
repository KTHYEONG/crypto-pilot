from __future__ import annotations

import logging
import math
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import L2BenchmarkConfig, L2GateConfig, L3ValidationConfig
from src.domain.futures.compound.contracts import (
    CausalityError,
    DeploymentVerdict,
    ExecutionLedger,
    L2BenchmarkSeries,
    L2CategoryResult,
    L2Evaluation,
    L2GateVerdict,
    L3ValidationResult,
    SealedHoldoutManifest,
)

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


def _causal_volatility_scale(
    basket_returns: NDArray[np.float64],
    lookback: int,
    target_ann_vol: float,
) -> NDArray[np.float64]:
    n = len(basket_returns)
    scale = np.ones(n, dtype=np.float64)
    for d in range(n):
        if d < lookback:
            continue
        window = basket_returns[max(0, d - lookback):d]
        valid = window[np.isfinite(window)]
        if len(valid) < 10:
            continue
        realized_vol = float(np.std(valid, ddof=1)) * math.sqrt(365.25)
        if realized_vol > 1e-12:
            scale[d] = min(target_ann_vol / realized_vol, 3.0)
    return scale


def build_causal_l2_benchmark(
    *,
    daily_market_returns: Mapping[str, NDArray[np.float64]],
    timestamps_ns: NDArray[np.int64],
    config: L2BenchmarkConfig,
) -> L2BenchmarkSeries:
    if config.mode == "cash_collateral":
        raise NotImplementedError("cash_collateral benchmark mode not yet implemented")

    for sym in config.crypto_symbols:
        if sym not in daily_market_returns:
            raise ValueError(f"missing benchmark symbol: {sym}")

    n = len(timestamps_ns)
    basket_returns = np.zeros(n, dtype=np.float64)
    for sym, w in zip(config.crypto_symbols, config.crypto_weights, strict=True):
        sym_ret = np.asarray(daily_market_returns[sym], dtype=np.float64)
        if sym_ret.shape != (n,):
            raise ValueError(f"benchmark symbol {sym} shape mismatch: expected ({n},), got {sym_ret.shape}")
        basket_returns += w * sym_ret

    causal_scale = _causal_volatility_scale(basket_returns, config.volatility_lookback_days, config.target_ann_vol)
    scaled_returns = basket_returns * causal_scale

    benchmark_id = f"{config.mode}_{'_'.join(config.crypto_symbols)}_{config.volatility_lookback_days}d"
    return L2BenchmarkSeries(
        benchmark_id=benchmark_id,
        timestamps_ns=timestamps_ns,
        daily_returns_1d=scaled_returns,
        causal_scale_1d=causal_scale,
    )


def _annualized_log_growth(returns: NDArray[np.float64], periods_per_year: float) -> float:
    log_returns = np.log1p(np.where(np.isfinite(returns), returns, 0.0))
    return float(periods_per_year * np.mean(log_returns))


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


def _deflated_sharpe_probability(
    observed_sharpe: float,
    candidate_count: int,
    n_obs: int,
    periods_per_year: float = 365.25,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> float:
    if n_obs < 30:
        return 0.5
    if candidate_count <= 1 or observed_sharpe <= 0:
        return 0.5
    sigma = math.sqrt(periods_per_year / n_obs)
    rng = np.random.default_rng(seed)
    e_max_samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sharpe_null = rng.standard_t(df=10, size=candidate_count) * sigma
        e_max_samples[i] = float(np.max(sharpe_null))
    prob = float(np.mean(e_max_samples < float(observed_sharpe)))
    return prob


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
    candidate_count: int,
    config: L2GateConfig,
    bootstrap_seed: int,
) -> L2Evaluation:
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
    absolute_log_growth = _annualized_log_growth(np.expm1(daily_returns), 365.25)
    absolute_cagr_val = float(math.expm1(absolute_log_growth))
    equity: NDArray[np.float64] = np.asarray(np.cumprod(1.0 + daily_returns), dtype=np.float64)
    equity_multiple = float(equity[-1]) if equity.size > 0 else 1.0

    # Sharpe ratio
    obs_sharpe, sharpe_prob = _stationary_bootstrap_sharpe_probability(
        excess_returns, n_bootstrap=2000, block_size=5, seed=bootstrap_seed,
    )
    dsr = _deflated_sharpe_probability(obs_sharpe, candidate_count, n_obs=oos_days, periods_per_year=365.25, seed=bootstrap_seed)

    # Growth bootstrap
    excess_lcb90, _, excess_prob = _stationary_bootstrap_lcb90(
        np.expm1(excess_returns), 365.25, n_bootstrap=1000, block_size=5, seed=bootstrap_seed,
    )
    stressed_lcb90, _, _ = _stationary_bootstrap_lcb90(
        np.expm1(stressed_daily), 365.25, n_bootstrap=1000, block_size=5, seed=bootstrap_seed + 1,
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
    pre_fee_log_growth = _annualized_log_growth(np.expm1(pre_fee_daily), 365.25)
    pre_fee_cagr = float(math.expm1(pre_fee_log_growth))
    cost_drag = 1.0 - absolute_cagr_val / max(pre_fee_cagr, 1e-12) if pre_fee_cagr > 0 else 0.0

    # Capacity utilisation (p95 of max absolute weight)
    capacity_p95 = 0.0
    if ledger.target_weights_2d.shape[0] > 0:
        max_abs_w = np.max(np.abs(ledger.target_weights_2d), axis=1)
        capacity_p95 = float(np.percentile(max_abs_w, 95))

    active_days_ratio = int(np.sum(np.abs(ledger.net_returns_1d) > 1e-15)) / max(len(ledger.net_returns_1d), 1)
    calmar = cagr / max(mdd, 1e-12)

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
    )


def _daily_timestamps_from_4h(timestamps_ns_4h: NDArray[np.int64]) -> NDArray[np.int64]:
    ns_per_4h = 4 * 3600 * 10**9
    day_start_ns = timestamps_ns_4h - (timestamps_ns_4h % (6 * ns_per_4h))
    unique_days, counts = np.unique(day_start_ns, return_counts=True)
    complete = unique_days[counts == 6]
    day_end_ns: NDArray[np.int64] = complete.astype(np.int64) + np.int64(6 * ns_per_4h - 1)
    return day_end_ns


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
