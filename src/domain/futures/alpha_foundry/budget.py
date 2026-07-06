"""Alpha Foundry L1 budget allocation helpers.

[ADR_20260706_ALPHA_FOUNDRY_SYNC][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY]
[ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from src.domain.futures.alpha_foundry.contracts import (
    AlphaRecipe,
    BucketKey,
    CheapGateEvidence,
    DiversitySelectionResult,
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

    bucket_counts: dict[tuple[str, str], int] = {}
    for ev in survivors:
        recipe = recipes.get(ev.recipe_id)
        if recipe is None:
            continue
        key = (recipe.family, recipe.timeframe)
        bucket_counts[key] = bucket_counts.get(key, 0) + 1
    for key, count in bucket_counts.items():
        if count > top_k_per_family_tf:
            raise ValueError(
                f"L0 diversity budget violated for bucket {key[0]}:{key[1]}: "
                f"{count} > {top_k_per_family_tf}"
            )

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


def allocate_global_l1_budget(
    *,
    bucket_results: Sequence[DiversitySelectionResult],
    evidence_by_recipe_id: Mapping[str, CheapGateEvidence],
    total_l1_verification_budget: int,
    top_k_max: int,
) -> dict[BucketKey, int]:
    if total_l1_verification_budget <= 0:
        raise ValueError(
            f"total_l1_verification_budget must be positive, got {total_l1_verification_budget}"
        )
    quality: dict[BucketKey, float] = {}
    for br in bucket_results:
        best = 0.0
        for rid in br.selected_recipe_ids:
            ev = evidence_by_recipe_id.get(rid)
            if ev is not None:
                best = max(best, ev.block_lcb_bps)
        quality[br.bucket_key] = best

    positive = {b: q for b, q in quality.items() if q > 0.0}
    if not positive:
        return dict.fromkeys(quality, 0)

    total_quality = sum(positive.values())
    raw = {b: total_l1_verification_budget * q / total_quality for b, q in positive.items()}
    floor_alloc = {b: min(top_k_max, int(v)) for b, v in raw.items()}

    remainder = total_l1_verification_budget - sum(floor_alloc.values())
    fractional = {b: raw[b] - floor_alloc[b] for b in positive}

    sorted_buckets = sorted(fractional.keys(), key=lambda b: (-fractional[b], b))
    for b in sorted_buckets:
        if remainder <= 0:
            break
        if floor_alloc[b] < top_k_max:
            floor_alloc[b] += 1
            remainder -= 1

    result: dict[BucketKey, int] = {}
    for b in quality:
        result[b] = floor_alloc.get(b, 0)
    return result
