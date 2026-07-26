from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import AllocatorConfig, DynamicCompoundingConfig
from src.domain.futures.compound.contracts import (
    ActiveForecastState,
    AllocationConstraints,
    AlphaForecastTape,
    CalibratedForecastPanel,
    CombinedForecast,
    PortfolioDecision,
)

_logger = logging.getLogger(__name__)

_VARIANCE_FLOOR: float = 1e-12

_COST_EDGE_THETA: float = 0.0006


def apply_cost_aware_net_edge(
    target_weights: NDArray[np.float64],
    previous_weights: NDArray[np.float64],
    mu: NDArray[np.float64],
    theta_cost: float = _COST_EDGE_THETA,
) -> NDArray[np.float64]:
    delta_w = target_weights - previous_weights
    edge = np.abs(delta_w * mu)
    above_threshold = edge > theta_cost
    return np.where(above_threshold, target_weights, previous_weights)


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


def _apply_portfolio_level_caps(
    w: NDArray[np.float64],
    max_long: float,
    max_short: float,
    max_gross: float,
) -> NDArray[np.float64]:
    long_sum = np.sum(np.maximum(w, 0.0))
    if long_sum > max_long:
        w = np.where(w > 0, w * (max_long / max(long_sum, 1e-15)), w)

    short_sum = np.sum(np.maximum(-w, 0.0))
    if short_sum > max_short:
        w = np.where(w < 0, w * (max_short / max(short_sum, 1e-15)), w)

    gross = np.sum(np.abs(w))
    if gross > max_gross:
        w = w * (max_gross / max(gross, 1e-15))

    return w


def _rankdata_abs_avg(values: NDArray[np.float64]) -> NDArray[np.float64]:
    n = values.shape[0]
    sorter = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and values[sorter[j]] == values[sorter[i]]:
            j += 1
        avg = float(i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[sorter[k]] = avg
        i = j
    return ranks


def build_rank_conviction_targets(
    mu_1d: NDArray[np.float64],
    eligible_1d: NDArray[np.bool_],
    *,
    min_breadth: int = 10,
) -> NDArray[np.float64]:
    if mu_1d.shape != eligible_1d.shape:
        raise ValueError(f"mu_1d shape {mu_1d.shape} != eligible_1d shape {eligible_1d.shape}")
    n = mu_1d.shape[0]
    eligible_count = int(np.sum(eligible_1d))
    if eligible_count < min_breadth:
        return np.zeros(n, dtype=np.float64)
    eligible_mu = mu_1d[eligible_1d]
    abs_ranks = _rankdata_abs_avg(np.abs(eligible_mu))
    raw = np.sign(eligible_mu) * abs_ranks
    raw = raw - np.mean(raw)
    abs_sum = float(np.sum(np.abs(raw)))
    if abs_sum > 0.0:
        raw = raw / abs_sum
    result = np.zeros(n, dtype=np.float64)
    result[eligible_1d] = raw
    return result


def compute_dynamic_compounding_path(
    forecast: CalibratedForecastPanel,
    sigma_2d: NDArray[np.float32],
    funding_rates_1h_2d: NDArray[np.float32],
    config: DynamicCompoundingConfig,
    *,
    close_2d: NDArray[np.float32],
    cost_bps: float,
) -> NDArray[np.float64]:
    n_bars = forecast.decision_timestamps_ns.size
    n_syms = len(forecast.symbols)
    weights = np.zeros((n_bars, n_syms), dtype=np.float64)

    equity = 1.0
    peak_equity = 1.0
    return_history: list[float] = []
    state = np.zeros(n_syms, dtype=np.float64)
    cooldown_counter = 0

    for t in range(n_bars):
        mu = np.nan_to_num(forecast.mu_2d[t], nan=0.0, posinf=0.0, neginf=0.0)
        sigma_t = np.nan_to_num(sigma_2d[t], nan=1e-4, posinf=1e-4, neginf=1e-4)

        eligible = np.abs(mu) > 0

        if t == 0:
            fr = np.zeros(n_syms, dtype=np.float64)
        else:
            idx = (t - 1) * 4
            fr_raw = funding_rates_1h_2d[idx].astype(np.float64) if idx < funding_rates_1h_2d.shape[0] else np.zeros(n_syms, dtype=np.float64)
            fr = np.nan_to_num(fr_raw, nan=0.0, posinf=0.0, neginf=0.0)

        mu_f64 = mu.astype(np.float64)

        if config.funding_carry_enabled:
            carry = np.where(mu_f64 > 0, 1.0, np.where(mu_f64 < 0, -1.0, 0.0)) * fr
            mu_f64 = mu_f64 + carry

        if config.use_rank_conviction:
            desired = build_rank_conviction_targets(mu_f64, eligible)
            if float(np.sum(np.abs(desired))) == 0.0:
                sigma_safe = np.maximum(sigma_t.astype(np.float64), config.sigma_floor)
                desired = np.where(eligible, config.kelly_fraction * mu_f64 / sigma_safe, 0.0)
        else:
            sigma_safe = np.maximum(sigma_t.astype(np.float64), config.sigma_floor)
            desired = np.where(eligible, config.kelly_fraction * mu_f64 / sigma_safe, 0.0)

        if len(return_history) >= config.min_vol_samples:
            rv_ann = float(np.std(return_history[-config.vol_lookback_bars:], ddof=1)) * np.sqrt(2190.0)
            leverage = min(config.target_ann_vol / max(rv_ann, 1e-15), config.max_gross_leverage)
        else:
            leverage = min(0.5, config.max_gross_leverage)

        desired = desired * leverage

        smoothed = config.alpha_smooth * desired + (1.0 - config.alpha_smooth) * state

        if config.band_frac > 0:
            band = config.band_frac * float(np.mean(np.abs(smoothed)))
            delta = np.abs(smoothed - state)
            state = np.where(delta > band, smoothed, state)
        else:
            state = smoothed

        if not np.any(eligible):
            state = np.zeros(n_syms, dtype=np.float64)

        if cooldown_counter > 0:
            cooldown_counter -= 1
            dd_scale = config.dd_scale_floor
        else:
            dd = 1.0 - equity / max(peak_equity, 1e-15)
            if dd >= config.hard_drawdown_limit:
                cooldown_counter = config.dd_cooldown_bars
                dd_scale = config.dd_scale_floor
            elif dd >= config.soft_drawdown_limit:
                frac = (dd - config.soft_drawdown_limit) / max(
                    config.hard_drawdown_limit - config.soft_drawdown_limit, 1e-15
                )
                dd_scale = max(config.dd_scale_floor, 1.0 - frac)
            else:
                dd_scale = 1.0

        w_exec = state * dd_scale
        w_exec = _apply_portfolio_level_caps(w_exec, config.max_long_leverage, config.max_short_leverage, config.max_gross_leverage)
        weights[t] = w_exec

        if t < n_bars - 1:
            prev_w = weights[t - 1] if t > 0 else np.zeros(n_syms, dtype=np.float64)
            close_t = close_2d[t].astype(np.float64)
            close_next = close_2d[t + 1].astype(np.float64)
            valid = (close_t > 0) & np.isfinite(close_t) & (close_next > 0) & np.isfinite(close_next)
            ret = np.where(valid, close_next / close_t - 1.0, 0.0)
            portfolio_ret = np.dot(w_exec, ret) - cost_bps * 1e-4 * np.sum(np.abs(w_exec - prev_w))
            equity = equity * (1.0 + portfolio_ret)
            peak_equity = max(peak_equity, equity)
            return_history.append(portfolio_ret)

    return weights


def compute_dynamic_compounding_weights(
    forecast: CombinedForecast,
    sigma_1d: NDArray[np.float64],
    funding_rates_1d: NDArray[np.float64],
    previous_weights_1d: NDArray[np.float64],
    config: DynamicCompoundingConfig,
    vol_scale: float = 1.0,
) -> NDArray[np.float64]:
    if not np.all(np.isfinite(forecast.mu_robust_1d)):
        raise ValueError("non-finite mu_robust_1d in forecast")
    if not np.all(np.isfinite(sigma_1d)):
        raise ValueError("non-finite sigma_1d")
    if not np.all(np.isfinite(funding_rates_1d)):
        raise ValueError("non-finite funding_rates_1d")
    if not np.all(np.isfinite(previous_weights_1d)):
        raise ValueError("non-finite previous_weights_1d")

    mu = forecast.mu_robust_1d.copy()
    support = forecast.support_1d

    sigma_safe = np.maximum(sigma_1d, config.sigma_floor)

    if config.funding_carry_enabled:
        carry = np.where(mu > 0, 1.0, np.where(mu < 0, -1.0, 0.0)) * funding_rates_1d
        mu = mu + carry

    mu_support = np.abs(mu) > 0
    support = support & mu_support

    raw_weights = np.where(support, config.kelly_fraction * mu / sigma_safe, 0.0)

    raw_weights = raw_weights * vol_scale

    smoothed = config.alpha_smooth * raw_weights + (1.0 - config.alpha_smooth) * previous_weights_1d

    if config.band_frac > 0:
        band = config.band_frac * float(np.mean(np.abs(smoothed)))
        delta = np.abs(smoothed - previous_weights_1d)
        hysteresis_mask = delta > band
        w = np.where(hysteresis_mask, smoothed, previous_weights_1d)
    else:
        w = smoothed

    w = np.where(support, w, 0.0)

    w = _apply_portfolio_level_caps(w, config.max_long_leverage, config.max_short_leverage, config.max_gross_leverage)

    return w


def compute_top_n_compounding_weights(
    forecast: CombinedForecast,
    sigma_2d: NDArray[np.float32],
    funding_rates_1h_2d: NDArray[np.float32],
    config: DynamicCompoundingConfig,
    top_n: int = 20,
) -> NDArray[np.float64]:
    if forecast.mu_robust_1d.ndim != 1:
        raise ValueError("forecast.mu_robust_1d must be 1-D")
    n_syms = forecast.mu_robust_1d.shape[0]
    if sigma_2d.ndim != 2 or sigma_2d.shape[1] != n_syms:
        raise ValueError("sigma_2d must be 2-D with n_syms columns")
    t_total = sigma_2d.shape[0]

    mu = forecast.mu_robust_1d
    support = forecast.support_1d

    ranked_idx = np.argsort(-np.abs(mu))
    top_mask = np.zeros(n_syms, dtype=bool)
    top_mask[ranked_idx[:top_n]] = True
    active = support & top_mask & (mu > 0)

    weights = np.zeros((t_total, n_syms), dtype=np.float64)
    for t in range(t_total):
        sigma_t = sigma_2d[t].astype(np.float64)

        if t == 0:
            fr = np.zeros(n_syms, dtype=np.float64)
        else:
            idx = (t - 1) * 4
            fr_raw = funding_rates_1h_2d[idx].astype(np.float64) if idx < funding_rates_1h_2d.shape[0] else np.zeros(n_syms, dtype=np.float64)
            fr = np.nan_to_num(fr_raw, nan=0.0, posinf=0.0, neginf=0.0)

        prev_w = weights[t - 1] if t > 0 else np.zeros(n_syms, dtype=np.float64)

        per_bar_forecast = CombinedForecast(mu_robust_1d=mu, variance_1d=np.ones(n_syms, dtype=np.float64), support_1d=active)
        raw_w = compute_dynamic_compounding_weights(
            forecast=per_bar_forecast,
            sigma_1d=sigma_t,
            funding_rates_1d=fr,
            previous_weights_1d=prev_w,
            config=config,
            vol_scale=1.0,
        )

        weights[t] = raw_w

    return weights
