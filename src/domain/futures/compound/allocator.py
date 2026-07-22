from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import AllocatorConfig
from src.domain.futures.compound.contracts import AlphaForecastTape, CombinedForecast, PortfolioDecision

_logger = logging.getLogger(__name__)

_VARIANCE_FLOOR: float = 1e-12


def combine_alpha_forecasts(
    *,
    tape: AlphaForecastTape,
    decision_idx: int,
    config: AllocatorConfig,
) -> CombinedForecast:
    n_syms = len(tape.symbols)
    n_recipes = len(tape.recipe_ids)

    mu_robust = np.zeros(n_syms, dtype=np.float64)
    variance = np.zeros(n_syms, dtype=np.float64)
    support = np.zeros(n_syms, dtype=np.bool_)

    for n in range(n_syms):
        precisions: list[float] = []
        mu_vals: list[float] = []
        for k in range(n_recipes):
            if not tape.valid_3d[decision_idx, n, k]:
                continue
            reliability_k = float(tape.reliability_3d[decision_idx, n, k])
            var_k = float(tape.forecast_var_3d[decision_idx, n, k])
            horizon_k = int(tape.horizon_bars_1d[k])
            mu_k = float(tape.gross_mu_3d[decision_idx, n, k])
            if not np.isfinite(mu_k) or not np.isfinite(var_k) or var_k <= 0:
                continue
            mu_per_bar = mu_k / max(horizon_k, 1)
            precision = reliability_k / max(var_k, _VARIANCE_FLOOR)
            precisions.append(precision)
            mu_vals.append(mu_per_bar)

        if not precisions:
            continue

        total_precision = sum(precisions)
        if total_precision <= 0:
            continue
        mu_symbol = sum(p * m for p, m in zip(precisions, mu_vals, strict=False)) / total_precision
        var_symbol = 1.0 / total_precision

        mu_robust_n = np.sign(mu_symbol) * max(abs(mu_symbol) - config.uncertainty_z * np.sqrt(var_symbol), 0.0)
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


def solve_growth_optimal_weights(
    *,
    forecast: CombinedForecast,
    covariance_2d: NDArray[np.float64],
    previous_weights_1d: NDArray[np.float64],
    cost_bps_1d: NDArray[np.float64],
    capacity_weight_1d: NDArray[np.float64],
    beta_1d: NDArray[np.float64],
    config: AllocatorConfig,
) -> PortfolioDecision:
    n_syms = len(forecast.mu_robust_1d)
    w = np.zeros(n_syms, dtype=np.float64)
    prev_obj = -np.inf

    per_sym_cap = np.full(n_syms, config.per_symbol_cap, dtype=np.float64)
    for i in range(n_syms):
        per_sym_cap[i] = min(per_sym_cap[i], capacity_weight_1d[i])
        if forecast.support_1d[i] and per_sym_cap[i] > 0:
            w[i] = min(config.gross_cap / max(n_syms, 1), per_sym_cap[i])

    w, _ = _project_weights(w, config.gross_cap, config.net_cap, per_sym_cap, beta_1d, config.beta_cap)

    for iteration in range(config.max_iterations):
        mu = forecast.mu_robust_1d
        sigma_w = covariance_2d @ w
        turnover = w - previous_weights_1d
        cost_linear = np.sum(cost_bps_1d * np.abs(turnover)) * 1e-4
        cost_l2 = config.turnover_l2 * np.dot(turnover, turnover)

        obj = np.dot(mu, w) - 0.5 / config.fractional_kelly * np.dot(w, sigma_w) - cost_linear - cost_l2

        grad = mu - (1.0 / config.fractional_kelly) * sigma_w
        grad -= cost_bps_1d * np.sign(turnover) * 1e-4
        grad -= 2.0 * config.turnover_l2 * turnover

        step = 1.0 / (1.0 / config.fractional_kelly + 2.0 * config.turnover_l2 + 1e-15)
        w_new = w + step * grad
        w_new = np.maximum(0, w_new)
        w_new *= forecast.support_1d.astype(np.float64)
        w_new, _constraints = _project_weights(
            w_new, config.gross_cap, config.net_cap, per_sym_cap, beta_1d, config.beta_cap
        )

        w_new_obj = np.dot(mu, w_new) - 0.5 / config.fractional_kelly * np.dot(w_new, covariance_2d @ w_new)
        if w_new_obj > obj:
            w = w_new

        if iteration > 0 and abs(obj - prev_obj) < config.objective_tolerance:
            break
        prev_obj = obj

    gross_exp = float(np.sum(np.abs(w)))
    net_exp = float(np.sum(w))
    forecast_vol = float(np.sqrt(np.dot(w, covariance_2d @ w))) * np.sqrt(8766 / config.rebalance_bars)

    binding = []
    if gross_exp >= config.gross_cap * 0.999:
        binding.append("gross_cap")
    if abs(net_exp) >= config.net_cap * 0.999:
        binding.append("net_cap")
    capped_syms = [f"symbol_cap_{i}" for i in range(n_syms) if abs(w[i]) >= per_sym_cap[i] * 0.999]
    binding.extend(capped_syms)
    beta_total = float(np.dot(beta_1d, w))
    if abs(beta_total) >= config.beta_cap * 0.999:
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
