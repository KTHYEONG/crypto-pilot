"""Alpha Foundry L1 budget allocation helpers. [ADR_20260706_ALPHA_FOUNDRY_SYNC]"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from src.domain.futures.alpha_foundry.contracts import (
    AlphaRecipe,
    CheapGateEvidence,
    L1PosteriorEvidence,
    L1VerificationUnit,
    PosteriorGateConfig,
)


def build_l1_verification_units(
    *,
    evidences: Sequence[CheapGateEvidence],
    recipes: Mapping[str, AlphaRecipe],
    symbols: tuple[str, ...],
    top_k_per_family_tf: int,
    initial_fold_budget: int,
) -> tuple[L1VerificationUnit, ...]:
    survivors = [e for e in evidences if e.gate_passed]

    units: list[L1VerificationUnit] = []
    for ev in survivors:
        recipe = recipes.get(ev.recipe_id)
        if recipe is None:
            continue
        unit = L1VerificationUnit(
            unit_id=f"{ev.recipe_id}:{ev.timeframe}",
            recipe_id=ev.recipe_id,
            timeframe=ev.timeframe,
            scope_symbols=symbols,
            prior_mu_bps=ev.mean_net_bps,
            prior_sigma_bps=max(ev.block_lcb_bps / max(ev.nw_tstat, 1e-10), 1.0),
            allocated_fold_budget=initial_fold_budget,
            early_stop_state="pending",
        )
        units.append(unit)
    return tuple(units)


def update_successive_halving_state(
    *,
    units: Sequence[L1VerificationUnit],
    posterior: Sequence[L1PosteriorEvidence],
    eta: int,
    max_fold_budget: int,
    config: PosteriorGateConfig,
) -> tuple[L1VerificationUnit, ...]:
    post_map: dict[str, L1PosteriorEvidence] = {}
    for p in posterior:
        key = f"{p.symbol}:{p.recipe_id}"
        post_map[key] = p

    updated: list[L1VerificationUnit] = []
    for unit in units:
        key = f"{unit.scope_symbols[0] if unit.scope_symbols else ''}:{unit.recipe_id}"
        post = post_map.get(key)
        if post is None:
            updated.append(unit)
            continue
        if post.prob_mu_gt_cost <= config.drop_prob_max:
            new_state: Literal["pending", "drop", "promote", "continue"] = "drop"
        elif post.prob_mu_gt_cost >= config.promote_prob_min and post.lcb_net_bps > config.min_lcb_net_bps:
            new_state = "promote"
            unit = L1VerificationUnit(
                unit_id=unit.unit_id,
                recipe_id=unit.recipe_id,
                timeframe=unit.timeframe,
                scope_symbols=unit.scope_symbols,
                prior_mu_bps=unit.prior_mu_bps,
                prior_sigma_bps=unit.prior_sigma_bps,
                allocated_fold_budget=min(unit.allocated_fold_budget * eta, max_fold_budget),
                early_stop_state=new_state,
            )
        else:
            new_state = "continue"
        if new_state in ("drop", "continue"):
            unit = L1VerificationUnit(
                unit_id=unit.unit_id,
                recipe_id=unit.recipe_id,
                timeframe=unit.timeframe,
                scope_symbols=unit.scope_symbols,
                prior_mu_bps=unit.prior_mu_bps,
                prior_sigma_bps=unit.prior_sigma_bps,
                allocated_fold_budget=unit.allocated_fold_budget,
                early_stop_state=new_state,
            )
        updated.append(unit)
    return tuple(updated)
