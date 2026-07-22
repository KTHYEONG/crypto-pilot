from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.portfolio.allocation_policy import AllocationPolicy
from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.strategy.tiered_workflow.l2_gate import (
    DeploymentMode,
)

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class OnlineAllocatorConfig:
    policies: tuple[AllocationPolicy, ...] = (
        "equal_weight",
        "inverse_vol",
        "kelly",
        "l1_confidence_shrinkage",
    )
    min_evidence_blocks: int = 3
    max_history_blocks: int = 12
    growth_lcb_z: float = 1.0
    prior_strength: float = 2.0
    wealth_score_weight: float = 0.10
    max_expert_weight: float = 0.60
    leverage_step: float = 0.05


@dataclass(slots=True, frozen=True)
class OnlinePolicyState:
    policies: tuple[AllocationPolicy, ...]
    block_growth_by_policy: tuple[tuple[float, ...], ...]
    previous_policy_weights_2d: NDArray[np.float64]
    previous_target_weights: NDArray[np.float64]
    last_completed_return_idx: int
    last_decision_idx: int
    config_fingerprint: str


@dataclass(slots=True, frozen=True)
class OnlineAllocationDecision:
    mode: DeploymentMode
    target_weights: NDArray[np.float64]
    posterior_by_policy: tuple[float, ...]
    cash_weight: float
    growth_lcb_by_policy: tuple[float, ...]
    risk_scale: float
    reason: str


def initialize_online_policy_state(
    *,
    policies: tuple[AllocationPolicy, ...],
    n_symbols: int,
    config_fingerprint: str,
) -> OnlinePolicyState:
    n_policies = len(policies)
    pol_weights = np.ones((n_policies, n_symbols), dtype=np.float64) / n_symbols
    return OnlinePolicyState(
        policies=policies,
        block_growth_by_policy=tuple(() for _ in range(n_policies)),
        previous_policy_weights_2d=pol_weights,
        previous_target_weights=np.zeros(n_symbols, dtype=np.float64),
        last_completed_return_idx=-1,
        last_decision_idx=0,
        config_fingerprint=config_fingerprint,
    )


def update_online_policy_state(
    *,
    state: OnlinePolicyState,
    completed_block_returns_by_policy: NDArray[np.float64],
    completed_return_end_idx: int,
    next_decision_idx: int,
    config: OnlineAllocatorConfig,
) -> OnlinePolicyState:
    if completed_return_end_idx >= next_decision_idx:
        raise ValueError(
            "completed return must precede decision: "
            f"completed_return_end_idx={completed_return_end_idx} >= next_decision_idx={next_decision_idx}"
        )
    if completed_return_end_idx <= state.last_completed_return_idx:
        raise ValueError(
            "completed return index must advance: "
            f"completed_return_end_idx={completed_return_end_idx} <= last_completed_return_idx={state.last_completed_return_idx}"
        )
    n_policies = len(state.policies)
    growth_arrays: list[list[float]] = [list(h) for h in state.block_growth_by_policy]
    for p in range(n_policies):
        gr = math.log1p(max(completed_block_returns_by_policy[p], -0.999999))
        growth_arrays[p].append(gr)
        if len(growth_arrays[p]) > config.max_history_blocks:
            growth_arrays[p] = growth_arrays[p][-config.max_history_blocks:]
    return OnlinePolicyState(
        policies=state.policies,
        block_growth_by_policy=tuple(tuple(g) for g in growth_arrays),
        previous_policy_weights_2d=state.previous_policy_weights_2d.copy(),
        previous_target_weights=state.previous_target_weights.copy(),
        last_completed_return_idx=completed_return_end_idx,
        last_decision_idx=next_decision_idx,
        config_fingerprint=state.config_fingerprint,
    )


def allocate_online_policy_mix(
    *,
    state: OnlinePolicyState,
    policy_weights_2d: NDArray[np.float64],
    trailing_ensemble_returns: NDArray[np.float64],
    config: OnlineAllocatorConfig,
    caps: PortfolioCaps,
    btc_beta: NDArray[np.float64] | None,
    max_mdd: float,
    max_cvar_95: float,
    min_equity_multiplier: float,
    exchange_leverage_cap: float,
    decision_idx: int,
) -> OnlineAllocationDecision:
    n_symbols = policy_weights_2d.shape[1]
    n_policies = len(state.policies)

    growth_arrays = state.block_growth_by_policy

    lcb_vals: list[float] = []
    score_vals: list[float] = []
    for p in range(n_policies):
        hist = growth_arrays[p]
        n = len(hist)
        if n < config.min_evidence_blocks:
            lcb_vals.append(-1.0)
            score_vals.append(-1.0)
            continue
        mean_g = float(np.mean(hist))
        mean_shrunk = n / (n + config.prior_strength) * mean_g
        std_g = float(np.std(hist, ddof=1)) if n >= 2 else 0.0
        lcb = mean_shrunk - config.growth_lcb_z * std_g / math.sqrt(n)
        lcb_vals.append(lcb)
        wealth_score = float(np.sum(hist)) / math.sqrt(n)
        score_vals.append(lcb + config.wealth_score_weight * wealth_score)

    growth_lcb = tuple(lcb_vals)
    has_positive_lcb = any(lcb > 0.0 for lcb in lcb_vals)

    if not has_positive_lcb:
        zero_weights = np.zeros(n_symbols, dtype=np.float64)
        return OnlineAllocationDecision(
            mode="abstain_cash",
            target_weights=zero_weights,
            posterior_by_policy=tuple(0.0 for _ in range(n_policies)),
            cash_weight=1.0,
            growth_lcb_by_policy=growth_lcb,
            risk_scale=0.0,
            reason="all expert LCB <= 0",
        )

    positive_scores: list[float] = []
    positive_indices: list[int] = []
    for p in range(n_policies):
        if lcb_vals[p] > 0.0:
            positive_scores.append(score_vals[p])
            positive_indices.append(p)

    score_arr = np.array(positive_scores, dtype=np.float64)
    score_arr = score_arr - float(np.max(score_arr))
    exp_scores = np.exp(score_arr)
    raw_mass = exp_scores / float(np.sum(exp_scores))

    posterior = [0.0] * n_policies
    cash_mass = 0.0
    for i, p in enumerate(positive_indices):
        w = float(raw_mass[i])
        if w > config.max_expert_weight:
            excess = w - config.max_expert_weight
            w = config.max_expert_weight
            cash_mass += excess
        posterior[p] = w

    total = float(np.sum(posterior)) + cash_mass
    if total > 0.0 and not math.isclose(total, 1.0, rel_tol=1e-12):
        posteriors_np = np.array(posterior, dtype=np.float64) / total
        posterior = list(posteriors_np)
        cash_mass = cash_mass / total
    posterior_tuple = tuple(posterior)
    cash_weight = cash_mass + max(0.0, 1.0 - float(np.sum(posterior)) - cash_mass)

    if cash_weight >= 1.0 - 1e-12:
        zero_weights = np.zeros(n_symbols, dtype=np.float64)
        return OnlineAllocationDecision(
            mode="abstain_cash",
            target_weights=zero_weights,
            posterior_by_policy=posterior_tuple,
            cash_weight=1.0,
            growth_lcb_by_policy=growth_lcb,
            risk_scale=0.0,
            reason="all expert LCB <= 0 after posterior calculation",
        )

    risk_scale = _select_risk_scale(
        trailing_ensemble_returns=trailing_ensemble_returns,
        max_mdd=max_mdd,
        max_cvar_95=max_cvar_95,
        min_equity_multiplier=min_equity_multiplier,
        exchange_leverage_cap=exchange_leverage_cap,
        caps_gross=caps.gross,
    )

    if risk_scale == 0.0:
        zero_weights = np.zeros(n_symbols, dtype=np.float64)
        return OnlineAllocationDecision(
            mode="risk_off_cash",
            target_weights=zero_weights,
            posterior_by_policy=posterior_tuple,
            cash_weight=1.0,
            growth_lcb_by_policy=growth_lcb,
            risk_scale=0.0,
            reason="no safe positive risk scale found",
        )

    target = np.zeros(n_symbols, dtype=np.float64)
    for p in range(n_policies):
        if posterior[p] > 0.0:
            target = target + posterior[p] * policy_weights_2d[p]
    target = target * risk_scale

    return OnlineAllocationDecision(
        mode="risk_on",
        target_weights=target,
        posterior_by_policy=posterior_tuple,
        cash_weight=cash_weight,
        growth_lcb_by_policy=growth_lcb,
        risk_scale=risk_scale,
        reason=f"risk_on with scale={risk_scale:.2f}",
    )


def _select_risk_scale(
    *,
    trailing_ensemble_returns: NDArray[np.float64],
    max_mdd: float,
    max_cvar_95: float,
    min_equity_multiplier: float,
    exchange_leverage_cap: float,
    caps_gross: float,
) -> float:
    max_scale = min(exchange_leverage_cap, caps_gross)
    n_steps = round(max_scale / 0.05)
    scales = np.linspace(0.0, max_scale, n_steps + 1)

    feasible = 0.0
    for scale in scales:
        if scale == 0.0:
            feasible = 0.0
            continue
        scaled_returns = trailing_ensemble_returns * scale
        log_growth = np.sum(np.log1p(np.maximum(scaled_returns, -0.999999)))
        n = len(scaled_returns)
        if n < 2:
            feasible = scale
            continue
        mean_g = log_growth / n
        mean_shrunk = n / (n + 2.0) * mean_g
        std_g = float(np.std(scaled_returns, ddof=1))
        lcb = mean_shrunk - 1.0 * std_g / math.sqrt(n) if std_g > 0.0 else mean_shrunk
        if lcb <= 0.0:
            continue
        eq = _compute_equity_path(scaled_returns)
        eq_mult = float(eq[-1])
        if eq_mult < min_equity_multiplier:
            continue
        dd = _compute_max_drawdown(eq)
        if dd > max_mdd:
            continue
        cvar95 = _compute_cvar_95(scaled_returns)
        if cvar95 > max_cvar_95:
            continue
        feasible = float(scale)

    return feasible


def _compute_equity_path(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    eq = np.empty(len(returns) + 1, dtype=np.float64)
    eq[0] = 1.0
    for t in range(len(returns)):
        eq[t + 1] = eq[t] * (1.0 + returns[t])
    return eq


def _compute_max_drawdown(equity: NDArray[np.float64]) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd))


def _compute_cvar_95(returns: NDArray[np.float64]) -> float:
    sorted_rets = np.sort(returns)
    n = len(sorted_rets)
    var_idx = max(1, int(0.05 * n))
    tail = sorted_rets[:var_idx]
    return float(np.abs(np.mean(tail)))