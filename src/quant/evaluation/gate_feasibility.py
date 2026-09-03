"""Closed-form pre-backtest feasibility screen for the reliability gate, accurate to about +-3pp against the block bootstrap, and never a substitute for the real gate in a promotion decision."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

from src.research.evaluation.reliability import ReliabilityGateConfig


@dataclass(frozen=True, slots=True)
class GateFeasibility:
    sharpe: float
    vol: float
    years: float
    sharpe_floor_t_stat: float
    sharpe_floor_lcb_positive: float
    sharpe_floor_hurdle: float
    leverage_mdd_cap: float
    leverage_lcb_optimal: float
    leverage_recommended: float
    max_lcb90_achievable: float
    feasible: bool
    binding_constraint: str


@dataclass(frozen=True, slots=True)
class BreadthRequirement:
    n_legs: int
    mean_leg_sharpe: float
    mean_pairwise_correlation: float
    breadth_multiplier: float
    achievable_portfolio_sharpe: float
    required_portfolio_sharpe: float
    required_mean_leg_sharpe: float
    sufficient: bool


def _minimum_sharpe_floors(
    years: float, config: ReliabilityGateConfig
) -> tuple[float, float, float]:
    """Return the three sizing-invariant Sharpe floors (t_stat, LCB90-positive, LCB90-hurdle)."""
    z = float(NormalDist().inv_cdf(config.lcb_confidence))
    root_years = math.sqrt(years)
    return (
        config.t_stat_floor / root_years,
        z / root_years,
        z / root_years + math.sqrt(2.0 * config.hurdle_rate),
    )


def compute_gate_feasibility(
    sharpe: float,
    vol: float,
    mdd_at_unit_leverage: float,
    years: float,
    config: ReliabilityGateConfig = ReliabilityGateConfig(),  # noqa: B008
) -> GateFeasibility:
    """State whether a single leg can clear the reliability gate at any leverage.

    Models ``LCB90(f) = f*vol*(sharpe - z/sqrt(years)) - f*f*vol*vol/2`` with
    ``MDD(f) = (1 + mdd_at_unit_leverage)**f - 1`` and returns the compound-growth-
    maximizing sizing choice subject to the gate, plus which constraint binds.
    """
    if vol <= 0.0:
        raise ValueError(f"vol must be > 0, got {vol}")
    if years <= 0.0:
        raise ValueError(f"years must be > 0, got {years}")
    if mdd_at_unit_leverage >= 0.0 or mdd_at_unit_leverage <= -1.0:
        raise ValueError(
            f"mdd_at_unit_leverage must be in (-1.0, 0.0), got {mdd_at_unit_leverage}"
        )

    sharpe_floor_t_stat, sharpe_floor_lcb_positive, sharpe_floor_hurdle = (
        _minimum_sharpe_floors(years, config)
    )

    z = float(NormalDist().inv_cdf(config.lcb_confidence))
    root_years = math.sqrt(years)

    mdd_cap_raw = math.log(1.0 + config.mdd_floor) / math.log(1.0 + mdd_at_unit_leverage)
    leverage_mdd_cap = mdd_cap_raw if math.isfinite(mdd_cap_raw) and mdd_cap_raw > 0 else 0.0

    leverage_lcb_optimal = max((sharpe - z / root_years) / vol, 0.0)
    leverage_recommended = min(leverage_lcb_optimal, leverage_mdd_cap)

    edge = sharpe - z / root_years
    max_lcb90_achievable = (
        leverage_recommended * vol * edge
        - leverage_recommended * leverage_recommended * vol * vol / 2.0
    )

    feasible = (sharpe > sharpe_floor_t_stat) and (max_lcb90_achievable > config.hurdle_rate)

    if sharpe <= sharpe_floor_t_stat:
        binding_constraint = "t_stat"
    elif leverage_mdd_cap < leverage_lcb_optimal and max_lcb90_achievable <= config.hurdle_rate:
        binding_constraint = "mdd_floor"
    elif max_lcb90_achievable <= config.hurdle_rate:
        binding_constraint = "hurdle_rate"
    else:
        binding_constraint = "none"

    return GateFeasibility(
        sharpe=sharpe,
        vol=vol,
        years=years,
        sharpe_floor_t_stat=sharpe_floor_t_stat,
        sharpe_floor_lcb_positive=sharpe_floor_lcb_positive,
        sharpe_floor_hurdle=sharpe_floor_hurdle,
        leverage_mdd_cap=leverage_mdd_cap,
        leverage_lcb_optimal=leverage_lcb_optimal,
        leverage_recommended=leverage_recommended,
        max_lcb90_achievable=max_lcb90_achievable,
        feasible=feasible,
        binding_constraint=binding_constraint,
    )


def compute_breadth_requirement(
    leg_sharpes: Sequence[float],
    mean_pairwise_correlation: float,
    years: float,
    config: ReliabilityGateConfig = ReliabilityGateConfig(),  # noqa: B008
) -> BreadthRequirement:
    """State the mean per-leg Sharpe a portfolio needs to clear the gate by breadth alone."""
    if len(leg_sharpes) == 0:
        raise ValueError("leg_sharpes must not be empty")
    n = len(leg_sharpes)
    rho = float(mean_pairwise_correlation)
    equicorr_denom = 1.0 + (n - 1) * rho
    if equicorr_denom <= 0.0:
        raise ValueError(
            "mean_pairwise_correlation too negative to be a valid equicorrelation "
            f"matrix: 1 + ({n} - 1) * {rho} = {equicorr_denom}"
        )
    if years <= 0.0:
        raise ValueError(f"years must be > 0, got {years}")

    breadth_multiplier = math.sqrt(n / equicorr_denom)
    mean_leg_sharpe = float(sum(leg_sharpes)) / n
    _, _, required_portfolio_sharpe = _minimum_sharpe_floors(years, config)

    achievable_portfolio_sharpe = mean_leg_sharpe * breadth_multiplier
    required_mean_leg_sharpe = required_portfolio_sharpe / breadth_multiplier
    sufficient = achievable_portfolio_sharpe >= required_portfolio_sharpe

    return BreadthRequirement(
        n_legs=n,
        mean_leg_sharpe=mean_leg_sharpe,
        mean_pairwise_correlation=rho,
        breadth_multiplier=breadth_multiplier,
        achievable_portfolio_sharpe=achievable_portfolio_sharpe,
        required_portfolio_sharpe=required_portfolio_sharpe,
        required_mean_leg_sharpe=required_mean_leg_sharpe,
        sufficient=sufficient,
    )
