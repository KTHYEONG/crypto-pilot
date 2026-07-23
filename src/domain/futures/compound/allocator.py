from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import AllocatorConfig
from src.domain.futures.compound.contracts import (
    ActiveForecastState,
    AllocationConstraints,
    AlphaForecastTape,
    CombinedForecast,
    PortfolioDecision,
)

_logger = logging.getLogger(__name__)

_VARIANCE_FLOOR: float = 1e-12


def combine_alpha_forecasts(
    tape: AlphaForecastTape,
    time_idx: int,
    *,
    uncertainty_z: float,
) -> CombinedForecast:
    n_syms = len(tape.symbols)
    n_recipes = len(tape.recipe_ids)

    mu_robust = np.zeros(n_syms, dtype=np.float64)
    variance = np.full(n_syms, 1e-4, dtype=np.float64)
    support = np.zeros(n_syms, dtype=np.bool_)

    for n in range(n_syms):
        precisions: list[float] = []
        mu_vals: list[float] = []
        for k in range(n_recipes):
            if not tape.valid_3d[time_idx, n, k] or not tape.estimated_3d[time_idx, n, k]:
                continue
            reliability_k = float(tape.reliability_3d[time_idx, n, k])
            edge_var = float(tape.mean_edge_var_3d[time_idx, n, k])
            mu_k = float(tape.gross_mu_3d[time_idx, n, k])
            if not np.isfinite(mu_k) or not np.isfinite(edge_var):
                continue
            precision = reliability_k / max(edge_var, _VARIANCE_FLOOR)
            precisions.append(precision)
            mu_vals.append(mu_k)

        if not precisions:
            continue

        total_precision = sum(precisions)
        if total_precision <= 0:
            continue
        mu_symbol = sum(p * m for p, m in zip(precisions, mu_vals, strict=False)) / total_precision
        var_symbol = 1.0 / total_precision

        mu_robust_n = np.sign(mu_symbol) * max(abs(mu_symbol) - uncertainty_z * np.sqrt(var_symbol), 0.0)
        mu_robust[n] = mu_robust_n
        variance[n] = var_symbol
        support[n] = abs(mu_robust_n) > 0

    return CombinedForecast(mu_robust_1d=mu_robust, variance_1d=variance, support_1d=support)


def _project_weights(
    w: NDArray[np.float64],
    gross_cap: float,
    net_cap: float,
    per_sym_cap: NDArray[np.float64],
    beta_1d: NDArray[np.float64],
    beta_cap: float,
) -> tuple[NDArray[np.float64], list[str]]:
    constraints: list[str] = []

    abs_w = np.abs(w)
    if np.sum(abs_w) > gross_cap:
        scale = gross_cap / max(np.sum(abs_w), 1e-15)
        w = w * scale
        constraints.append("gross_cap")

    for i in range(len(w)):
        if abs_w[i] > per_sym_cap[i]:
            w[i] = np.sign(w[i]) * per_sym_cap[i]
            constraints.append(f"symbol_cap_{i}")

    net = np.sum(w)
    if abs(net) > net_cap:
        if abs(net) > 1e-15:
            w = w * (net_cap / abs(net))
        constraints.append("net_cap")

    beta = np.dot(beta_1d, w)
    if abs(beta) > beta_cap and abs(beta) > 1e-15:
        w = w * (beta_cap / abs(beta))
        constraints.append("beta_cap")

    return w, constraints


def _full_objective(
    w: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    prev_w: NDArray[np.float64],
    cost_bps: NDArray[np.float64],
    turnover_l2: float,
    fractional_kelly: float,
) -> float:
    turnover = w - prev_w
    cost_linear = float(np.sum(cost_bps * np.abs(turnover)) * 1e-4)
    cost_l2 = turnover_l2 * float(np.dot(turnover, turnover))
    risk = 0.5 / fractional_kelly * float(np.dot(w, sigma))
    return float(np.dot(mu, w)) - risk - cost_linear - cost_l2


def _solve_growth_weights_iterative(
    mu: NDArray[np.float64],
    covariance: NDArray[np.float64],
    previous_weights: NDArray[np.float64],
    constraints: AllocationConstraints,
    config: AllocatorConfig,
    support_mask: NDArray[np.bool_],
    per_sym_cap: NDArray[np.float64],
) -> PortfolioDecision:
    n_syms = len(mu)
    w = np.zeros(n_syms, dtype=np.float64)

    for i in range(n_syms):
        if support_mask[i] and per_sym_cap[i] > 0:
            w[i] = np.sign(mu[i]) * min(config.gross_cap / max(n_syms, 1), per_sym_cap[i])

    w, _ = _project_weights(
        w, constraints.gross_cap, constraints.net_cap, per_sym_cap,
        constraints.beta_1d, constraints.beta_cap,
    )

    prev_obj = -np.inf
    for iteration in range(config.max_iterations):
        sigma_vec = covariance @ w
        obj = _full_objective(w, mu, sigma_vec, previous_weights,
                              constraints.cost_bps_1d, config.turnover_l2, config.fractional_kelly)
        grad = mu - (1.0 / config.fractional_kelly) * sigma_vec
        grad -= constraints.cost_bps_1d * np.sign(w - previous_weights) * 1e-4
        grad -= 2.0 * config.turnover_l2 * (w - previous_weights)
        step = 1.0 / max(1.0 / config.fractional_kelly + 2.0 * config.turnover_l2, 1e-15)
        w_new = w + step * grad
        w_new *= support_mask.astype(np.float64)
        w_new, _ = _project_weights(
            w_new, constraints.gross_cap, constraints.net_cap, per_sym_cap,
            constraints.beta_1d, constraints.beta_cap,
        )
        new_obj = _full_objective(w_new, mu, covariance @ w_new, previous_weights,
                                  constraints.cost_bps_1d, config.turnover_l2, config.fractional_kelly)
        if new_obj > obj:
            w = w_new
        if iteration > 0 and abs(obj - prev_obj) < config.objective_tolerance:
            break
        prev_obj = obj

    gross_exp = float(np.sum(np.abs(w)))
    net_exp = float(np.sum(w))
    forecast_vol = float(np.sqrt(float(np.dot(w, covariance @ w))) * np.sqrt(8766.0))

    binding: list[str] = []
    if gross_exp >= constraints.gross_cap * 0.999:
        binding.append("gross_cap")
    if abs(net_exp) >= constraints.net_cap * 0.999:
        binding.append("net_cap")
    capped_syms = [f"symbol_cap_{i}" for i in range(n_syms) if abs(w[i]) >= per_sym_cap[i] * 0.999]
    binding.extend(capped_syms)
    beta_total = float(np.dot(constraints.beta_1d, w))
    if abs(beta_total) >= constraints.beta_cap * 0.999:
        binding.append("beta_cap")

    return PortfolioDecision(
        decision_idx=0,
        decision_time_ns=0,
        target_weights_1d=w,
        gross_exposure=gross_exp,
        net_exposure=net_exp,
        forecast_ann_vol=forecast_vol,
        risk_scale=1.0,
        binding_constraints=tuple(binding),
    )


def solve_event_growth_weights(
    *,
    state: ActiveForecastState,
    covariance_per_hour: NDArray[np.float64],
    previous_weights: NDArray[np.float64],
    constraints: AllocationConstraints,
    config: AllocatorConfig,
) -> PortfolioDecision:
    n_syms = len(state.symbols)

    if not np.all(np.isfinite(covariance_per_hour)):
        msg = "non-finite covariance matrix"
        raise ValueError(msg)

    per_sym_cap = constraints.per_symbol_cap.copy()
    for i in range(n_syms):
        capacity_usdt = constraints.capacity_weight_1d[i]
        nav_cap = capacity_usdt / max(config.portfolio_nav_usdt, 1.0) * config.risk_scale
        per_sym_cap[i] = min(per_sym_cap[i], nav_cap)
        if constraints.entry_block_1d[i]:
            per_sym_cap[i] = min(per_sym_cap[i], abs(previous_weights[i]))

    mu = state.alpha_rate_1d * 24.0

    mu_robust = np.sign(mu) * np.maximum(np.abs(mu) - config.uncertainty_z * np.sqrt(state.epistemic_variance_1d), 0.0)
    support = np.abs(mu_robust) > 0

    return _solve_growth_weights_iterative(
        mu=mu_robust,
        covariance=covariance_per_hour * 24.0,
        previous_weights=previous_weights,
        constraints=constraints,
        config=config,
        support_mask=support,
        per_sym_cap=per_sym_cap,
    )


def solve_growth_optimal_weights(
    *,
    combined: CombinedForecast,
    covariance: NDArray[np.float64],
    previous_weights: NDArray[np.float64],
    constraints: AllocationConstraints,
    decision_idx: int,
    decision_time_ns: int,
    config: AllocatorConfig,
) -> PortfolioDecision:
    n_syms = len(combined.mu_robust_1d)
    w = np.zeros(n_syms, dtype=np.float64)

    if not np.all(np.isfinite(covariance)):
        raise ValueError("non-finite covariance matrix")
    if config.portfolio_nav_usdt <= 0:
        raise ValueError(f"NAV must be positive, got {config.portfolio_nav_usdt}")

    per_sym_cap = constraints.per_symbol_cap.copy()
    for i in range(n_syms):
        capacity_usdt = constraints.capacity_weight_1d[i]
        nav_cap = capacity_usdt / max(config.portfolio_nav_usdt, 1.0) * config.risk_scale
        per_sym_cap[i] = min(per_sym_cap[i], nav_cap)
        if constraints.entry_block_1d[i]:
            per_sym_cap[i] = min(per_sym_cap[i], abs(previous_weights[i]))

    mu = combined.mu_robust_1d
    support_mask = combined.support_1d

    has_negative = np.any(mu < 0)
    if has_negative:
        short_blocked = np.zeros(n_syms, dtype=np.bool_)
        for i in range(n_syms):
            if mu[i] < 0 and short_blocked[i]:
                raise ValueError(
                    f"negative robust forecast for symbol {i} but short is structurally blocked"
                )

    prev_obj = -np.inf
    for i in range(n_syms):
        if support_mask[i] and per_sym_cap[i] > 0:
            w[i] = np.sign(mu[i]) * min(config.gross_cap / max(n_syms, 1), per_sym_cap[i])

    w, _ = _project_weights(
        w, constraints.gross_cap, constraints.net_cap, per_sym_cap,
        constraints.beta_1d, constraints.beta_cap,
    )

    sigma_vec = covariance @ w

    for iteration in range(config.max_iterations):
        sigma_vec = covariance @ w
        turnover = w - previous_weights

        obj = _full_objective(w, mu, sigma_vec, previous_weights,
                              constraints.cost_bps_1d, config.turnover_l2, config.fractional_kelly)

        grad = mu - (1.0 / config.fractional_kelly) * sigma_vec
        grad -= constraints.cost_bps_1d * np.sign(turnover) * 1e-4
        grad -= 2.0 * config.turnover_l2 * turnover

        step = 1.0 / max(1.0 / config.fractional_kelly + 2.0 * config.turnover_l2, 1e-15)
        w_new = w + step * grad
        w_new *= support_mask.astype(np.float64)
        w_new, _ = _project_weights(
            w_new, constraints.gross_cap, constraints.net_cap, per_sym_cap,
            constraints.beta_1d, constraints.beta_cap,
        )

        new_obj = _full_objective(w_new, mu, covariance @ w_new, previous_weights,
                                  constraints.cost_bps_1d, config.turnover_l2, config.fractional_kelly)
        if new_obj > obj:
            w = w_new

        if iteration > 0 and abs(obj - prev_obj) < config.objective_tolerance:
            break
        prev_obj = obj

    gross_exp = float(np.sum(np.abs(w)))
    net_exp = float(np.sum(w))
    forecast_vol = float(np.sqrt(float(np.dot(w, covariance @ w))) * np.sqrt(8766 / config.rebalance_bars))

    binding = []
    if gross_exp >= constraints.gross_cap * 0.999:
        binding.append("gross_cap")
    if abs(net_exp) >= constraints.net_cap * 0.999:
        binding.append("net_cap")
    capped_syms = [f"symbol_cap_{i}" for i in range(n_syms) if abs(w[i]) >= per_sym_cap[i] * 0.999]
    binding.extend(capped_syms)
    beta_total = float(np.dot(constraints.beta_1d, w))
    if abs(beta_total) >= constraints.beta_cap * 0.999:
        binding.append("beta_cap")

    return PortfolioDecision(
        decision_idx=decision_idx,
        decision_time_ns=decision_time_ns,
        target_weights_1d=w,
        gross_exposure=gross_exp,
        net_exposure=net_exp,
        forecast_ann_vol=forecast_vol,
        risk_scale=1.0,
        binding_constraints=tuple(binding),
    )
