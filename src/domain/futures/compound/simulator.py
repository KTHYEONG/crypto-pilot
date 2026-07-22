from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.allocator import combine_alpha_forecasts, solve_growth_optimal_weights
from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    AlphaForecastTape,
    ExecutionLedger,
    MarketFeatureCube,
)
from src.domain.futures.compound.factor_risk import estimate_causal_factor_covariance
from src.domain.futures.compound.risk_overlay import apply_risk_overlay

_logger = logging.getLogger(__name__)


def simulate_compound_portfolio(
    *,
    cube: MarketFeatureCube,
    tape: AlphaForecastTape,
    config: CompoundEngineConfig,
) -> ExecutionLedger:
    n_bars = cube.timestamps_ns.size
    n_syms = len(cube.symbols)
    if n_bars < 2:
        raise ValueError("at least two bars are required for simulation")
    if tape.timestamps_ns.shape != cube.timestamps_ns.shape or tape.symbols != cube.symbols:
        raise ValueError("cube and forecast tape axes do not match")

    rebalance_bars = config.allocator.rebalance_bars
    target_weights = np.zeros((n_bars, n_syms), dtype=np.float32)
    net_returns = np.zeros(n_bars, dtype=np.float64)
    equity = np.ones(n_bars, dtype=np.float64)
    fee_returns = np.zeros(n_bars, dtype=np.float64)
    slippage_returns = np.zeros(n_bars, dtype=np.float64)
    impact_returns = np.zeros(n_bars, dtype=np.float64)
    funding_returns = np.zeros(n_bars, dtype=np.float64)

    prev_w = np.zeros(n_syms, dtype=np.float64)
    beta_1d = np.zeros(n_syms, dtype=np.float64)
    integrity_failures: list[str] = []

    stale_count = 0
    pending: list[tuple[int, NDArray[np.float64]]] = []

    for decision_t in range(0, n_bars - 1, rebalance_bars):
        close_at_decision = cube.fields_2d["close"][decision_t]
        is_stale = bool(np.any(~np.isfinite(close_at_decision)))
        stale_count = stale_count + 1 if is_stale else 0
        if stale_count >= 2:
            integrity_failures.append(f"stale_data_at_bar_{decision_t}")
            pending.append((decision_t + 1, np.zeros(n_syms, dtype=np.float64)))
            prev_w = np.zeros(n_syms, dtype=np.float64)
            continue

        forecast = combine_alpha_forecasts(
            tape=tape, decision_idx=decision_t, config=config.allocator
        )

        daily_ret = _compute_daily_returns(cube, decision_t)
        cluster_ids = np.zeros(n_syms, dtype=np.int16)
        if "cluster_id" in cube.fields_2d:
            cluster_ids = cube.fields_2d["cluster_id"][decision_t].astype(np.int16)

        cov = estimate_causal_factor_covariance(
            daily_returns_2d=daily_ret,
            end_exclusive=daily_ret.shape[0],
            cluster_ids_1d=cluster_ids,
            config=config.factor_risk,
        )

        capacity_w = cube.capacity_usdt_2d[decision_t]
        cost_bps = cube.execution_cost_bps_2d[decision_t].astype(np.float64)

        decision = solve_growth_optimal_weights(
            forecast=forecast,
            covariance_2d=cov,
            previous_weights_1d=prev_w,
            cost_bps_1d=cost_bps,
            capacity_weight_1d=capacity_w,
            beta_1d=beta_1d,
            config=config.allocator,
        )

        risk_result = apply_risk_overlay(
            decision=decision,
            equity_1d=equity[: max(decision_t, 1)],
            cooldown_remaining=0,
            config=config.risk,
        )

        w = risk_result.target_weights_1d.astype(np.float64)
        pending.append((decision_t + 1, w))
        prev_w = w

    active_w = np.zeros(n_syms, dtype=np.float64)
    pending_idx = 0
    for t in range(n_bars):
        if pending_idx < len(pending) and pending[pending_idx][0] == t:
            new_w = pending[pending_idx][1]
            turnover = float(np.sum(np.abs(new_w - active_w)))
            fee = turnover * 0.0006
            slippage = turnover * 0.0002
            impact = float(np.sqrt(np.sum((new_w - active_w) ** 2)) * 0.0001)
            fee_returns[t] = -fee
            slippage_returns[t] = -slippage
            impact_returns[t] = -impact
            active_w = new_w
            pending_idx += 1
        target_weights[t] = active_w.astype(np.float32)
        if t == 0:
            net_returns[t] = fee_returns[t] + slippage_returns[t] + impact_returns[t]
        else:
            bar_return = _simulate_bar_return(cube, t, active_w, active_w)
            funding = cube.fields_2d.get("funding")
            funding_ret = 0.0 if funding is None else float(-np.nansum(active_w * funding[t]))
            funding_returns[t] = funding_ret
            net_returns[t] = bar_return + fee_returns[t] + slippage_returns[t] + impact_returns[t] + funding_ret
        equity[t] = (equity[t - 1] if t else 1.0) * max(1.0 + net_returns[t], 1e-12)

    if integrity_failures:
        integrity_ok = False
        integrity_reasons = tuple(integrity_failures)
    else:
        integrity_ok = True
        integrity_reasons = ()

    return ExecutionLedger(
        timestamps_ns=cube.timestamps_ns,
        net_returns_1d=net_returns,
        equity_1d=equity,
        target_weights_2d=target_weights,
        fee_returns_1d=fee_returns,
        slippage_returns_1d=slippage_returns,
        impact_returns_1d=impact_returns,
        funding_returns_1d=funding_returns,
        integrity_ok=integrity_ok,
        integrity_reasons=integrity_reasons,
    )


def _compute_daily_returns(cube: MarketFeatureCube, end_exclusive: int) -> NDArray[np.float64]:
    close = cube.fields_2d.get("close", None)
    n_syms = len(cube.symbols)
    if close is None or end_exclusive <= 0:
        return np.zeros((max(end_exclusive, 1), n_syms), dtype=np.float64)
    n = min(end_exclusive, close.shape[0])
    ret = np.full((n, n_syms), np.nan, dtype=np.float64)
    for i in range(1, n):
        mask = close[i - 1] > 0
        ret[i, mask] = np.log(close[i, mask] / close[i - 1, mask]).astype(np.float64)
    ret[0] = 0.0
    return ret


def _simulate_bar_return(
    cube: MarketFeatureCube, t: int, prev_w: NDArray[np.float64], curr_w: NDArray[np.float64]
) -> float:
    close = cube.fields_2d.get("close", None)
    if close is None or t >= close.shape[0] - 1:
        return 0.0
    price_return = 0.0
    for i in range(len(prev_w)):
        if abs(prev_w[i]) > 1e-12 and close[t][i] > 0 and close[t + 1][i] > 0:
            sym_ret = np.log(close[t + 1][i] / close[t][i])
            price_return += prev_w[i] * sym_ret
    return price_return
