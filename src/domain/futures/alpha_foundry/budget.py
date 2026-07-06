"""Alpha Foundry L1 budget allocation helpers.

[ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]
[ADR_20260706_ALPHA_FOUNDRY_SYNC][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY]
[ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from src.domain.futures.alpha_foundry.contracts import (
    AlphaArchetype,
    AlphaRecipe,
    BucketKey,
    CheapGateEvidence,
    DiversitySelectionResult,
    L0BucketBudget,
    L0SignalCandidate,
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
                f"L0 diversity budget violated for bucket {key[0]}:{key[1]}: {count} > {top_k_per_family_tf}"
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
    candidate_by_recipe_id: Mapping[str, L0SignalCandidate],
    total_l1_verification_budget: int,
    top_k_max: int,
    min_seed_slots_per_archetype: int = 1,
    min_seed_slots_per_timeframe: int = 1,
) -> tuple[dict[BucketKey, int], tuple[L0BucketBudget, ...]]:
    if total_l1_verification_budget <= 0:
        raise ValueError(f"total_l1_verification_budget must be positive, got {total_l1_verification_budget}")

    # Compute bucket quality as max l1_priority_score in bucket
    quality: dict[BucketKey, float] = {}
    archetype_by_bucket: dict[BucketKey, AlphaArchetype] = {}
    cand_count_by_bucket: dict[BucketKey, int] = {}
    for br in bucket_results:
        best = 0.0
        archetype: AlphaArchetype = "trend"
        for rid in br.ranked_recipe_ids:
            cand = candidate_by_recipe_id.get(rid)
            if cand is not None:
                best = max(best, cand.l1_priority_score)
                archetype = cand.archetype
        quality[br.bucket_key] = best
        archetype_by_bucket[br.bucket_key] = archetype
        cand_count_by_bucket[br.bucket_key] = len(br.selected_recipe_ids)

    # Stage 1: Archetype seed slots - ensure each active archetype gets at least one slot
    buckets_by_archetype: dict[AlphaArchetype, list[BucketKey]] = {}
    for bk, arch in archetype_by_bucket.items():
        if arch not in buckets_by_archetype:
            buckets_by_archetype[arch] = []
        buckets_by_archetype[arch].append(bk)

    allocated: dict[BucketKey, int] = {}
    for bk in quality:
        allocated[bk] = 0

    remaining_budget = total_l1_verification_budget

    for buckets in buckets_by_archetype.values():
        if remaining_budget <= 0:
            break
        # Assign min_seed to the highest-quality bucket per archetype
        best_bucket = max(buckets, key=lambda b: quality.get(b, 0.0))
        allocated[best_bucket] = min(min_seed_slots_per_archetype, remaining_budget)
        remaining_budget -= allocated[best_bucket]

    # Stage 2: Timeframe seed slots
    buckets_by_tf: dict[str, list[BucketKey]] = {}
    for bk in archetype_by_bucket:
        tf = bk[1]  # timeframe
        if tf not in buckets_by_tf:
            buckets_by_tf[tf] = []
        buckets_by_tf[tf].append(bk)

    for buckets in buckets_by_tf.values():
        if remaining_budget <= 0:
            break
        already_has_slot = any(allocated.get(b, 0) > 0 for b in buckets)
        if already_has_slot:
            continue
        best_bucket = max(buckets, key=lambda b: quality.get(b, 0.0))
        allocated[best_bucket] = allocated.get(best_bucket, 0) + min(min_seed_slots_per_timeframe, remaining_budget)
        remaining_budget -= min(min_seed_slots_per_timeframe, remaining_budget)

    # Stage 3: Residual proportional allocation (largest remainder)
    positive = {b: q for b, q in quality.items() if q > 0.0}
    if positive:
        total_quality = sum(positive.values())
        raw = {b: remaining_budget * q / total_quality for b, q in positive.items()}
        floor_alloc = {b: allocated.get(b, 0) + min(top_k_max - allocated.get(b, 0), int(v)) for b, v in raw.items()}

        remainder = total_l1_verification_budget - sum(floor_alloc.values())
        fractional = {b: raw[b] - (floor_alloc[b] - allocated.get(b, 0)) for b in positive}

        sorted_buckets = sorted(fractional.keys(), key=lambda b: (-fractional[b], b))
        for b in sorted_buckets:
            if remainder <= 0:
                break
            if floor_alloc[b] < top_k_max:
                floor_alloc[b] += 1
                remainder -= 1

        for b in positive:
            allocated[b] = floor_alloc.get(b, allocated.get(b, 0))

    # Clamp to top_k_max
    for bk in allocated:
        allocated[bk] = min(allocated[bk], top_k_max)

    # Build L0BucketBudget records
    budgets: list[L0BucketBudget] = []
    for br in bucket_results:
        bk = br.bucket_key
        budgets.append(
            L0BucketBudget(
                bucket_key=bk,
                archetype=archetype_by_bucket.get(bk, "trend"),
                candidate_count=cand_count_by_bucket.get(bk, 0),
                selected_count=len(br.selected_recipe_ids),
                min_seed_slots=min_seed_slots_per_archetype,
                max_slots=top_k_max,
                allocated_slots=allocated.get(bk, 0),
                bucket_quality=quality.get(bk, 0.0),
                effective_test_count=br.bucket_eff_test_count,
            )
        )

    return allocated, tuple(budgets)
