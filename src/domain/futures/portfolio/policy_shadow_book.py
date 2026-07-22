from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.portfolio.allocation_policy import (
    AllocationPolicy,
    AllocationPolicyScore,
    build_policy_weights,
)
from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps


@dataclass(slots=True, frozen=True)
class FoldAllocationPlan:
    """Fit-local allocation policy plan. [ADR_20260722_L2_CAUSAL_POLICY_SHADOW_GROWTH_FIRST]"""

    fold_idx: int
    fit_start: int
    fit_end_exclusive: int
    oos_start: int
    oos_end_exclusive: int
    selected_policy: AllocationPolicy | None
    selected_leverage: float
    baseline_leverage: float
    scores: tuple[AllocationPolicyScore, ...]
    fallback_used: bool
    failure_reason: str


def build_policy_weight_matrix(
    *,
    policies: tuple[AllocationPolicy, ...],
    mu_bps: NDArray[np.float64],
    sigma: NDArray[np.float64],
    l1_edge_margin_bps_per_bar: NDArray[np.float64],
    quality_weight: NDArray[np.float64],
    caps: PortfolioCaps,
    previous_weights_2d: NDArray[np.float64],
    no_trade_band: float,
    vol_target: float | None,
    btc_beta: NDArray[np.float64] | None,
    bars_per_year: float,
    support_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Build one deployed weight row per policy. [ADR_20260722_L2_CAUSAL_POLICY_SHADOW_GROWTH_FIRST]"""

    p = len(policies)
    n = int(mu_bps.size) if mu_bps.ndim == 1 else mu_bps.shape[1]
    if previous_weights_2d.shape != (p, n):
        raise ValueError(
            f"previous_weights_2d shape {previous_weights_2d.shape} != ({p}, {n})"
        )
    matrix = np.empty((p, n), dtype=np.float64)
    for i, policy in enumerate(policies):
        matrix[i] = build_policy_weights(
            policy=policy,
            mu_bps=mu_bps,
            sigma=sigma,
            l1_edge_margin_bps_per_bar=l1_edge_margin_bps_per_bar,
            quality_weight=quality_weight,
            caps=caps,
            prev_w=previous_weights_2d[i],
            no_trade_band=no_trade_band,
            vol_target=vol_target,
            btc_beta=btc_beta,
            bars_per_year=bars_per_year,
            support_mask=support_mask,
        )
    return np.asarray(matrix, dtype=np.float64)


def compute_shadow_rebalance_costs(
    *,
    previous_weights_2d: NDArray[np.float64],
    target_weights_2d: NDArray[np.float64],
    round_trip_cost_bps: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute policy-specific turnover costs. [ADR_20260722_L2_CAUSAL_POLICY_SHADOW_GROWTH_FIRST]"""

    if previous_weights_2d.shape != target_weights_2d.shape:
        raise ValueError(
            f"shape mismatch: prev={previous_weights_2d.shape} target={target_weights_2d.shape}"
        )
    delta = np.abs(target_weights_2d - previous_weights_2d)
    costs = np.sum(delta * round_trip_cost_bps[np.newaxis, :] * 0.5 * 1e-4, axis=1)
    return np.asarray(costs, dtype=np.float64)


def compute_shadow_bar_returns(
    *,
    deployed_weights_2d: NDArray[np.float64],
    price_returns: NDArray[np.float64],
    funding_rates: NDArray[np.float64],
    rebalance_costs: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute net bar returns for all shadow policies. [ADR_20260722_L2_CAUSAL_POLICY_SHADOW_GROWTH_FIRST]"""

    _p, _n = deployed_weights_2d.shape
    pnl = np.sum(deployed_weights_2d * price_returns[np.newaxis, :], axis=1)
    funding = np.sum(deployed_weights_2d * funding_rates[np.newaxis, :], axis=1)
    net = pnl - funding
    if rebalance_costs is not None:
        net = net - rebalance_costs
    return np.asarray(net, dtype=np.float64)
