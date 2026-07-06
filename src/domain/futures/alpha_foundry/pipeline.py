"""Alpha Foundry L0-to-L2 orchestration bridge.

[ADR_20260706_ALPHA_FOUNDRY_SYNC][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY]
[ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
"""

from __future__ import annotations

import time as _time_module
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass

import pandas as pd

from src.domain.futures.alpha_foundry.budget import (
    allocate_global_l1_budget,
    build_l1_verification_units,
)
from src.domain.futures.alpha_foundry.cheap_gate import (
    evaluate_alpha_cheap_gate_batch,
)
from src.domain.futures.alpha_foundry.contracts import (
    AlphaFoundryEvidenceRow,
    AlphaRecipe,
    BucketKey,
    CheapGateConfig,
    CheapGateEvidence,
    CrossBucketDiversityResult,
    DiversitySelectionResult,
    L1PosteriorEvidence,
    L1VerificationUnit,
    L2PosteriorPolicyConfig,
    L2PosteriorSleeve,
    PosteriorGateConfig,
)
from src.domain.futures.alpha_foundry.diversity import (
    resolve_cross_bucket_diversity,
    select_bucket_diverse_recipes,
)
from src.domain.futures.alpha_foundry.l2_policy import (
    convert_posterior_to_l2_sleeves,
)
from src.domain.futures.alpha_foundry.posterior import shrink_l1_evidence_hierarchical
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


@dataclass(slots=True, frozen=True)
class AlphaFoundryL0Artifacts:
    evidences: tuple[CheapGateEvidence, ...]
    passed_recipe_ids: tuple[str, ...]
    reject_reason_counts: dict[str, int]
    evidence_rows: tuple[AlphaFoundryEvidenceRow, ...] = ()
    bucket_results: tuple[DiversitySelectionResult, ...] = ()
    cross_bucket_result: CrossBucketDiversityResult | None = None


def run_alpha_foundry_l0_pipeline(
    *,
    panels: Sequence[CandidateSignalPanel],
    recipes: Mapping[str, AlphaRecipe],
    aligned: AlignedMarketData,
    cost_model: ExecutionCostModel,
    cheap_gate_config: CheapGateConfig,
    run_id: str = "",
    top_k_per_family_tf: int = 5,
    min_conviction_lcb_bps: float = 5.0,
    total_l1_verification_budget: int = 30,
) -> AlphaFoundryL0Artifacts:
    cheap_evidences = evaluate_alpha_cheap_gate_batch(
        panels=panels,
        recipes=recipes,
        aligned=aligned,
        cost_model=cost_model,
        config=cheap_gate_config,
    )

    reject_reason_counts: dict[str, int] = {}
    for ev in cheap_evidences:
        for reason in ev.reject_reasons:
            reject_reason_counts[reason] = reject_reason_counts.get(reason, 0) + 1

    # Stage 2: Bucket diversity
    survivors = [ev for ev in cheap_evidences if ev.gate_passed]
    panel_by_rid: dict[str, CandidateSignalPanel] = {}
    for panel in panels:
        rid = panel.metadata.get("recipe_id", "")
        if rid:
            panel_by_rid[rid] = panel

    buckets: MutableMapping[BucketKey, list[CheapGateEvidence]] = {}
    for ev in survivors:
        recipe = recipes.get(ev.recipe_id)
        if recipe is None:
            continue
        key = (recipe.family, recipe.timeframe)
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(ev)

    active = aligned.active_mask & aligned.warm_mask & ~aligned.entry_block_mask & ~aligned.kill_mask

    bucket_results: list[DiversitySelectionResult] = []
    for bucket_key, cands in buckets.items():
        br = select_bucket_diverse_recipes(
            bucket_key=bucket_key,
            candidates=cands,
            panel_by_recipe_id=panel_by_rid,
            fwd_ret_by_recipe_id={},
            active_mask=active,
            top_k_per_family_tf=top_k_per_family_tf,
            max_novelty_corr=cheap_gate_config.max_novelty_corr,
            fdr_alpha=cheap_gate_config.fdr_alpha,
            min_conviction_lcb_bps=min_conviction_lcb_bps,
        )
        bucket_results.append(br)

    # Stage 3: Cross-bucket diversity
    cross_result = resolve_cross_bucket_diversity(
        bucket_results=bucket_results,
        panel_by_recipe_id=panel_by_rid,
        evidence_by_recipe_id={ev.recipe_id: ev for ev in cheap_evidences},
        active_mask=active,
        max_novelty_corr=cheap_gate_config.max_novelty_corr,
    )

    # Stage 3.5: Global L1 budget allocation
    evidence_by_rid = {ev.recipe_id: ev for ev in cheap_evidences}
    allocated = allocate_global_l1_budget(
        bucket_results=bucket_results,
        evidence_by_recipe_id=evidence_by_rid,
        total_l1_verification_budget=total_l1_verification_budget,
        top_k_max=top_k_per_family_tf,
    )

    final_selected = set(cross_result.final_selected_recipe_ids)
    budget_redundant: dict[str, str] = {}
    for br in bucket_results:
        bk = br.bucket_key
        slot_limit = allocated.get(bk, 0)
        bucket_selected = [rid for rid in br.selected_recipe_ids if rid in final_selected]
        bucket_selected.sort(
            key=lambda rid: evidence_by_rid[rid].block_lcb_bps if rid in evidence_by_rid else 0.0,
            reverse=True,
        )
        keep = set(bucket_selected[:slot_limit])
        for rid in bucket_selected:
            if rid not in keep:
                final_selected.discard(rid)
                budget_redundant[rid] = "budget_exhausted"

    # Build evidence_rows
    selected_in_bucket: dict[str, int] = {
        rid: rank
        for br in bucket_results
        for rank, rid in enumerate(br.selected_recipe_ids)
    }

    redundant_map: dict[str, str] = {}
    for br in bucket_results:
        redundant_map.update(br.redundant_reason_by_id)
    redundant_map.update(cross_result.demoted_reason_by_id)
    redundant_map.update(budget_redundant)

    created_at_ms = int(_time_module.time() * 1000)
    evidence_rows: list[AlphaFoundryEvidenceRow] = []
    for ev in cheap_evidences:
        recipe = recipes.get(ev.recipe_id)
        family = recipe.family if recipe else ""
        variant = recipe.variant if recipe else ""
        archetype = recipe.archetype if recipe else ""

        bucket_rank = selected_in_bucket.get(ev.recipe_id, -1)
        sel_for_l1 = ev.recipe_id in final_selected
        redundant_with = redundant_map.get(ev.recipe_id, "")

        bucket_eff = 0.0
        global_eff = 0.0
        for br in bucket_results:
            if ev.recipe_id in br.selected_recipe_ids or ev.recipe_id in br.redundant_recipe_ids:
                bucket_eff = br.bucket_eff_test_count
        global_eff = cross_result.global_eff_test_count

        evidence_rows.append(AlphaFoundryEvidenceRow(
            run_id=run_id,
            timeframe=ev.timeframe,
            family=family,
            variant=variant,
            recipe_id=ev.recipe_id,
            archetype=archetype,
            n_events=ev.n_events,
            effective_n=ev.effective_n,
            mean_net_bps=ev.mean_net_bps,
            nw_tstat=ev.nw_tstat,
            block_lcb_bps=ev.block_lcb_bps,
            rank_ic=ev.rank_ic,
            incremental_rank_ic=ev.incremental_rank_ic,
            cost_drag_ratio=ev.cost_drag_ratio,
            turnover_per_year=ev.turnover_per_year,
            compute_cost_score=ev.compute_cost_score,
            bootstrap_lcb_bps=ev.bootstrap_lcb_bps,
            bootstrap_agree=ev.bootstrap_agree,
            gate_passed=ev.gate_passed,
            reject_reasons="|".join(ev.reject_reasons),
            bucket_key=f"{family}:{ev.timeframe}",
            bucket_rank=bucket_rank,
            selected_for_l1=sel_for_l1,
            redundant_with=redundant_with,
            bucket_eff_test_count=bucket_eff,
            global_eff_test_count=global_eff,
            created_at_ms=created_at_ms,
        ))

    return AlphaFoundryL0Artifacts(
        evidences=cheap_evidences,
        passed_recipe_ids=tuple(final_selected),
        reject_reason_counts=reject_reason_counts,
        evidence_rows=tuple(evidence_rows),
        bucket_results=tuple(bucket_results),
        cross_bucket_result=cross_result,
    )


def build_posterior_from_l1_fold_rows(
    *,
    raw_rows: pd.DataFrame,
    cost_model: ExecutionCostModel,
    config: PosteriorGateConfig,
) -> tuple[L1PosteriorEvidence, ...]:
    if not isinstance(raw_rows, pd.DataFrame):
        raise ValueError("raw_rows must be a DataFrame")
    required = {"net_bps", "symbol"}
    missing = required - set(raw_rows.columns)
    if missing:
        raise ValueError(f"raw_rows missing required columns: {missing}")
    if raw_rows.empty:
        return ()
    return shrink_l1_evidence_hierarchical(
        raw_rows=raw_rows,
        cost_model=cost_model,
        config=config,
    )


def build_l2_sleeves_from_posterior(
    *,
    posterior: Sequence[L1PosteriorEvidence],
    cost_model: ExecutionCostModel,
    config: L2PosteriorPolicyConfig,
) -> tuple[L2PosteriorSleeve, ...]:
    return convert_posterior_to_l2_sleeves(
        posterior=posterior,
        cost_model=cost_model,
        config=config,
    )


def run_alpha_foundry_pipeline(
    *,
    panels: Sequence[CandidateSignalPanel],
    recipes: Mapping[str, AlphaRecipe],
    aligned: AlignedMarketData,
    cost_model: ExecutionCostModel,
    cheap_gate_config: CheapGateConfig,
    posterior_gate_config: PosteriorGateConfig,
    l2_config: L2PosteriorPolicyConfig,
    symbols: tuple[str, ...],
    top_k_per_family_tf: int = 5,
    initial_fold_budget: int = 3,
) -> tuple[
    tuple[CheapGateEvidence, ...],
    tuple[L1VerificationUnit, ...],
    tuple[L1PosteriorEvidence, ...],
    tuple[L2PosteriorSleeve, ...],
]:
    """Legacy wrapper — prefer the split API (L0 / posterior / L2)."""
    l0 = run_alpha_foundry_l0_pipeline(
        panels=panels,
        recipes=recipes,
        aligned=aligned,
        cost_model=cost_model,
        cheap_gate_config=cheap_gate_config,
    )

    l1_units = build_l1_verification_units(
        evidences=l0.evidences,
        recipes=recipes,
        symbols=symbols,
        top_k_per_family_tf=top_k_per_family_tf,
        initial_fold_budget=initial_fold_budget,
    )

    l1_posterior: tuple[L1PosteriorEvidence, ...] = ()
    l2_sleeves: tuple[L2PosteriorSleeve, ...] = ()

    return l0.evidences, l1_units, l1_posterior, l2_sleeves
