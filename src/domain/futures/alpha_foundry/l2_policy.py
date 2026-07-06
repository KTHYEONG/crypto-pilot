"""Alpha Foundry L2 posterior policy bridge. [ADR_20260706_ALPHA_FOUNDRY_SYNC]"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, cast

from src.domain.futures.alpha_foundry.contracts import (
    L1PosteriorEvidence,
    L2PosteriorPolicyConfig,
    L2PosteriorSleeve,
    StagedSearchBudget,
)
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def convert_posterior_to_l2_sleeves(
    *,
    posterior: Sequence[L1PosteriorEvidence],
    cost_model: ExecutionCostModel,
    config: L2PosteriorPolicyConfig,
) -> tuple[L2PosteriorSleeve, ...]:
    stress_cost = cost_model.stress_round_trip_bps()
    results: list[L2PosteriorSleeve] = []
    for ev in posterior:
        mu_eff = (
            ev.posterior_mu_bps
            - config.posterior_z * ev.posterior_sigma_bps
            - config.cost_safety_mult * stress_cost
        )
        if ev.quality_weight <= 0.0 or ev.lcb_net_bps <= 0.0:
            results.append(L2PosteriorSleeve(
                symbol=ev.symbol,
                recipe_id=ev.recipe_id,
                family=ev.family,
                timeframe=ev.timeframe,
                activation_context=ev.activation_context,
                mu_eff_bps=mu_eff,
                sigma_bps=ev.posterior_sigma_bps,
                quality_weight=ev.quality_weight,
                side=0,
                disabled_reason="non_positive_lcb",
            ))
        else:
            side = 1 if mu_eff > 0 else -1 if mu_eff < 0 else 0
            results.append(L2PosteriorSleeve(
                symbol=ev.symbol,
                recipe_id=ev.recipe_id,
                family=ev.family,
                timeframe=ev.timeframe,
                activation_context=ev.activation_context,
                mu_eff_bps=mu_eff,
                sigma_bps=ev.posterior_sigma_bps,
                quality_weight=ev.quality_weight,
                side=cast(Literal[-1, 0, 1], side),
                disabled_reason="",
            ))
    return tuple(results)


def build_staged_l2_search_spaces() -> Mapping[str, Mapping[str, Mapping[str, object]]]:
    return {
        "signal": {
            "quality_weight": {"type": "float", "low": 0.1, "high": 1.0},
            "rank_k": {"type": "int", "low": 1, "high": 10},
            "activation_contract": {"type": "categorical", "choices": ["hard", "soft"]},
        },
        "risk": {
            "kelly_fraction": {"type": "float", "low": 0.05, "high": 0.5},
            "posterior_z": {"type": "float", "low": 0.0, "high": 2.0},
            "risk_budget_target": {"type": "float", "low": 0.1, "high": 1.0},
        },
        "regime": {
            "gross_cap_bull": {"type": "float", "low": 0.5, "high": 1.0},
            "gross_cap_bear": {"type": "float", "low": 0.1, "high": 0.5},
            "gross_cap_crisis": {"type": "float", "low": 0.05, "high": 0.3},
        },
        "deployment": {
            "leverage_target": {"type": "float", "low": 0.5, "high": 3.0},
            "rebalance_bars": {"type": "int", "low": 1, "high": 12},
            "turnover_penalty": {"type": "float", "low": 0.0, "high": 5.0},
        },
    }


def resolve_staged_search_budget(
    *,
    n_dimensions: Mapping[str, int],
    requested_trials: int,
    seed_count: int,
) -> tuple[StagedSearchBudget, ...]:
    stages = ("signal", "risk", "regime", "deployment")
    budgets: list[StagedSearchBudget] = []
    for stage in stages:
        d = n_dimensions.get(stage, 1)
        n_trials_val = max(requested_trials // len(stages), 20 * d)
        budgets.append(StagedSearchBudget(
            stage=cast(Literal["signal", "risk", "regime", "deployment"], stage),
            n_trials=n_trials_val,
            min_feasible_eff=0.05,
            patience=max(5, d * 2),
            seed_count=seed_count,
        ))
    return tuple(budgets)
