from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.allocator import combine_alpha_forecasts, solve_growth_optimal_weights
from src.domain.futures.compound.alpha_events import build_active_forecast_state
from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    AllocationConstraints,
    AlphaEventTape,
    AlphaForecastTape,
    ExecutionLedger,
    MarketFeatureCube,
)
from src.domain.futures.compound.factor_risk import estimate_causal_factor_covariance
from src.domain.futures.compound.risk_overlay import apply_risk_overlay

_logger = logging.getLogger(__name__)

_PARTICIPATION_RATE: float = 0.05
_MAX_SLICE_MINUTES: int = 15


class PortfolioIntegrityError(RuntimeError):
    ...


def simulate_compound_portfolio(
    *,
    cube: MarketFeatureCube,
    alpha_tape: AlphaForecastTape,
    config: CompoundEngineConfig,
) -> ExecutionLedger:
    n_bars = cube.timestamps_ns.size
    n_syms = len(cube.symbols)
    if n_bars < 2:
        raise ValueError("at least two bars are required for simulation")
    if alpha_tape.timestamps_ns.shape != cube.timestamps_ns.shape or alpha_tape.symbols != cube.symbols:
        raise ValueError("cube and forecast tape axes do not match")

    target_weights = np.zeros((n_bars, n_syms), dtype=np.float32)
    net_returns = np.zeros(n_bars, dtype=np.float64)
    equity = np.ones(n_bars, dtype=np.float64)
    fee_returns = np.zeros(n_bars, dtype=np.float64)
    slippage_returns = np.zeros(n_bars, dtype=np.float64)
    impact_returns = np.zeros(n_bars, dtype=np.float64)
    funding_returns = np.zeros(n_bars, dtype=np.float64)

    curr_w = np.zeros(n_syms, dtype=np.float64)
    integrity_failures: list[str] = []

    for t in range(n_bars):
        curr_w = target_weights[t - 1].astype(np.float64) if t > 0 else np.zeros(n_syms, dtype=np.float64)

        if t == 0:
            net_returns[t] = 0.0
            equity[t] = 1.0
            target_weights[t] = curr_w.astype(np.float32)
            continue

        fee_returns[t] = 0.0
        slippage_returns[t] = 0.0
        impact_returns[t] = 0.0

        bar_return = _simulate_bar_return(cube, t - 1, curr_w)
        funding_field = cube.fields_2d.get("funding")
        funding_ret = 0.0
        if funding_field is not None and t - 1 < funding_field.shape[0]:
            for i in range(n_syms):
                if abs(curr_w[i]) > 1e-12:
                    funding_ret += -curr_w[i] * float(funding_field[t - 1, i])
        funding_returns[t] = funding_ret

        net_returns[t] = bar_return + funding_ret
        equity[t] = equity[t - 1] * max(1.0 + net_returns[t], 1e-12)

        exit_required_1d = cube.exit_required_2d[t] if t < cube.exit_required_2d.shape[0] else np.zeros(n_syms, dtype=np.bool_)
        if np.any(exit_required_1d):
            for i in range(n_syms):
                if exit_required_1d[i] and abs(curr_w[i]) > 0:
                    pass

        if alpha_tape.valid_3d[t].any():
            forecast = combine_alpha_forecasts(
                alpha_tape, t, uncertainty_z=config.allocator.uncertainty_z,
            )

            daily_ret = _compute_daily_returns(cube, t)
            cluster_ids = (
                cube.fields_2d["cluster_id"][t].astype(np.int16)
                if "cluster_id" in cube.fields_2d
                else np.zeros(n_syms, dtype=np.int16)
            )
            cov = estimate_causal_factor_covariance(
                daily_returns_2d=daily_ret,
                end_exclusive=daily_ret.shape[0],
                cluster_ids_1d=cluster_ids,
                config=config.factor_risk,
            )

            beta_1d = np.zeros(n_syms, dtype=np.float64)
            entry_b = cube.entry_block_2d[t] if t < cube.entry_block_2d.shape[0] else np.zeros(n_syms, dtype=np.bool_)
            exit_req = exit_required_1d
            capacity_w = cube.capacity_usdt_2d[t] if t < cube.capacity_usdt_2d.shape[0] else np.zeros(n_syms, dtype=np.float64)
            cost_bps = cube.execution_cost_bps_2d[t] if t < cube.execution_cost_bps_2d.shape[0] else np.full(n_syms, 12.0, dtype=np.float32)

            constraints = AllocationConstraints(
                gross_cap=config.allocator.gross_cap,
                net_cap=config.allocator.net_cap,
                per_symbol_cap=np.full(n_syms, config.allocator.per_symbol_cap, dtype=np.float64),
                beta_1d=beta_1d,
                beta_cap=config.allocator.beta_cap,
                capacity_weight_1d=capacity_w,
                cost_bps_1d=cost_bps.astype(np.float64),
                entry_block_1d=entry_b,
                exit_required_1d=exit_req,
            )

            decision = solve_growth_optimal_weights(
                combined=forecast,
                covariance=cov,
                previous_weights=curr_w,
                constraints=constraints,
                decision_idx=t,
                decision_time_ns=int(cube.timestamps_ns[t]),
                config=config.allocator,
            )

            risk_result = apply_risk_overlay(
                decision=decision,
                equity_1d=equity[:max(t, 1)],
                cooldown_remaining=0,
                config=config.risk,
            )

            new_w = risk_result.target_weights_1d.astype(np.float64)

            turnover = float(np.sum(np.abs(new_w - curr_w)))
            fee_returns[t] = -turnover * 0.0006
            slippage_returns[t] = -turnover * 0.0002
            impact_returns[t] = -float(np.sqrt(float(np.sum((new_w - curr_w) ** 2))) * 0.0001)
            net_returns[t] += fee_returns[t] + slippage_returns[t] + impact_returns[t]
            equity[t] = equity[t - 1] * max(1.0 + net_returns[t], 1e-12)

            curr_w = new_w
            target_weights[t] = curr_w.astype(np.float32)
        else:
            target_weights[t] = curr_w.astype(np.float32)

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
        if mask.any():
            ret[i, mask] = np.where(
                mask,
                np.log(close[i][mask] / close[i - 1][mask]),
                np.nan,
            ).astype(np.float64)[mask]
    ret[0] = 0.0
    return ret


def _simulate_bar_return(
    cube: MarketFeatureCube, t: int, prev_w: NDArray[np.float64],
) -> float:
    close = cube.fields_2d.get("close", None)
    if close is None or t >= close.shape[0] - 1:
        return 0.0
    price_return = 0.0
    for i in range(len(prev_w)):
        if abs(prev_w[i]) > 1e-12 and close[t][i] > 0 and close[t + 1][i] > 0:
            sym_ret = float(np.log(close[t + 1][i] / close[t][i]))
            price_return += prev_w[i] * sym_ret
    return price_return


def simulate_multiscale_portfolio(
    *,
    market: MarketFeatureCube,
    universe: object,
    handoff: AlphaEventTape,
    config: CompoundEngineConfig,
) -> ExecutionLedger:
    n_bars = market.timestamps_ns.size
    n_syms = len(market.symbols)
    if n_bars < 2:
        msg = "at least two bars required"
        raise ValueError(msg)

    target_weights = np.zeros((n_bars, n_syms), dtype=np.float32)
    net_returns = np.zeros(n_bars, dtype=np.float64)
    equity = np.ones(n_bars, dtype=np.float64)
    fee_returns = np.zeros(n_bars, dtype=np.float64)
    slippage_returns = np.zeros(n_bars, dtype=np.float64)
    impact_returns = np.zeros(n_bars, dtype=np.float64)
    funding_returns = np.zeros(n_bars, dtype=np.float64)

    curr_w = np.zeros(n_syms, dtype=np.float64)
    integrity_failures: list[str] = []
    rebalance_interval = config.allocator.rebalance_bars
    last_rebalance_bar = -rebalance_interval

    for t in range(n_bars):
        if t == 0:
            target_weights[t] = curr_w.astype(np.float32)
            continue

        bar_return = _simulate_bar_return(market, t - 1, curr_w)
        funding_field = market.fields_2d.get("funding")
        funding_ret = 0.0
        if funding_field is not None and t - 1 < funding_field.shape[0]:
            for i in range(n_syms):
                if abs(curr_w[i]) > 1e-12:
                    funding_ret += -curr_w[i] * float(funding_field[t - 1, i])
        funding_returns[t] = funding_ret

        net_returns[t] = bar_return + funding_ret
        equity[t] = equity[t - 1] * max(1.0 + net_returns[t], 1e-12)

        exit_req = market.exit_required_2d[t] if t < market.exit_required_2d.shape[0] else np.zeros(n_syms, dtype=np.bool_)
        if np.any(exit_req):
            for i in range(n_syms):
                if exit_req[i] and abs(curr_w[i]) > 0:
                    turnover_cost = abs(curr_w[i]) * 0.0006
                    fee_returns[t] -= turnover_cost
                    net_returns[t] -= turnover_cost
                    curr_w[i] = 0.0

        is_rebalance = (t - last_rebalance_bar) >= rebalance_interval

        state = build_active_forecast_state(
            tape=handoff, decision_time_ns=int(market.timestamps_ns[t]), symbols=market.symbols,
        )
        has_active = np.any(np.abs(state.alpha_rate_1d) > 0)

        if is_rebalance and has_active:
            daily_ret = _compute_daily_returns(market, t)
            cluster_ids = (
                market.fields_2d["cluster_id"][t].astype(np.int16)
                if "cluster_id" in market.fields_2d
                else np.zeros(n_syms, dtype=np.int16)
            )
            cov = estimate_causal_factor_covariance(
                daily_returns_2d=daily_ret,
                end_exclusive=daily_ret.shape[0],
                cluster_ids_1d=cluster_ids,
                config=config.factor_risk,
            )

            entry_b = market.entry_block_2d[t] if t < market.entry_block_2d.shape[0] else np.zeros(n_syms, dtype=np.bool_)
            capacity_w = market.capacity_usdt_2d[t] if t < market.capacity_usdt_2d.shape[0] else np.zeros(n_syms, dtype=np.float64)
            cost_bps = market.execution_cost_bps_2d[t] if t < market.execution_cost_bps_2d.shape[0] else np.full(n_syms, 12.0, dtype=np.float32)

            constraints = AllocationConstraints(
                gross_cap=config.allocator.gross_cap,
                net_cap=config.allocator.net_cap,
                per_symbol_cap=np.full(n_syms, config.allocator.per_symbol_cap, dtype=np.float64),
                beta_1d=np.zeros(n_syms, dtype=np.float64),
                beta_cap=config.allocator.beta_cap,
                capacity_weight_1d=capacity_w,
                cost_bps_1d=cost_bps.astype(np.float64),
                entry_block_1d=entry_b,
                exit_required_1d=exit_req,
            )

            from src.domain.futures.compound.allocator import solve_event_growth_weights

            decision = solve_event_growth_weights(
                state=state,
                covariance_per_hour=cov,
                previous_weights=curr_w,
                constraints=constraints,
                config=config.allocator,
            )

            new_w = decision.target_weights_1d.astype(np.float64)

            turnover = float(np.sum(np.abs(new_w - curr_w)))
            fee_returns[t] -= turnover * 0.0006 * 0.5
            slippage_returns[t] = -turnover * 0.0002
            impact_returns[t] = -float(np.sqrt(float(np.sum((new_w - curr_w) ** 2))) * 0.0001)
            net_returns[t] += fee_returns[t] + slippage_returns[t] + impact_returns[t]
            equity[t] = equity[t - 1] * max(1.0 + net_returns[t], 1e-12)

            curr_w = new_w
            target_weights[t] = curr_w.astype(np.float32)
            last_rebalance_bar = t
        else:
            target_weights[t] = curr_w.astype(np.float32)

        if not np.all(np.isfinite(curr_w)):
            integrity_failures.append(f"bar {t}: non-finite target weights")

    if integrity_failures:
        integrity_ok = False
        integrity_reasons = tuple(integrity_failures)
    else:
        integrity_ok = True
        integrity_reasons = ()

    return ExecutionLedger(
        timestamps_ns=market.timestamps_ns,
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
