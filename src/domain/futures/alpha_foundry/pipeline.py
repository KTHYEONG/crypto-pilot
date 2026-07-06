"""Alpha Foundry L0-to-L2 orchestration bridge.

[ADR_20260706_ALPHA_FOUNDRY_SYNC][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY]
[ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
"""

from __future__ import annotations

import time as _time_module
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import pandas as pd

from src.domain.futures.alpha_foundry.budget import (
    allocate_global_l1_budget,
    build_l1_verification_units,
)
from src.domain.futures.alpha_foundry.cheap_gate import (
    build_l0_signal_candidate,
    evaluate_alpha_cheap_gate_batch,
    resolve_family_timeframe_gate_policy,
)
from src.domain.futures.alpha_foundry.contracts import (
    AlphaFoundryEvidenceRow,
    AlphaRecipe,
    BucketKey,
    CheapGateConfig,
    CheapGateEvidence,
    CrossBucketDiversityResult,
    DiversitySelectionResult,
    L0BucketBudget,
    L0ReportStageCounts,
    L0SignalCandidate,
    L1PosteriorEvidence,
    L1VerificationUnit,
    L2PosteriorPolicyConfig,
    L2PosteriorSleeve,
    MultiTimeframeEvidence,
    PosteriorGateConfig,
)
from src.domain.futures.alpha_foundry.diversity import (
    resolve_cross_bucket_diversity,
    select_bucket_diverse_candidates,
)
from src.domain.futures.alpha_foundry.l2_policy import (
    convert_posterior_to_l2_sleeves,
)
from src.domain.futures.alpha_foundry.multi_tf_fusion import (
    fuse_multi_timeframe_evidence,
    index_multi_timeframe_evidence,
)
from src.domain.futures.alpha_foundry.posterior import shrink_l1_evidence_hierarchical
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


@dataclass(slots=True, frozen=True)
class AlphaFoundryL0Artifacts:
    evidences: tuple[CheapGateEvidence, ...]
    candidates: tuple[L0SignalCandidate, ...] = ()
    passed_recipe_ids: tuple[str, ...] = ()
    reject_reason_counts: dict[str, int] = field(default_factory=dict)
    stage_counts: L0ReportStageCounts = field(default_factory=lambda: L0ReportStageCounts(0, 0, 0, 0, 0, 0))
    evidence_rows: tuple[AlphaFoundryEvidenceRow, ...] = ()
    bucket_results: tuple[DiversitySelectionResult, ...] = ()
    bucket_budgets: tuple[L0BucketBudget, ...] = ()
    cross_bucket_result: CrossBucketDiversityResult | None = None
    tf_fusion: tuple[MultiTimeframeEvidence, ...] = ()


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
    evidence_by_tf: Mapping[str, pd.DataFrame] | None = None,
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

    # Build panel_by_rid
    panel_by_rid: dict[str, CandidateSignalPanel] = {}
    for panel in panels:
        rid = panel.metadata.get("recipe_id", "")
        if rid:
            panel_by_rid[rid] = panel

    active = aligned.active_mask & aligned.warm_mask & ~aligned.entry_block_mask & ~aligned.kill_mask

    # TF fusion
    tf_fusion: tuple[MultiTimeframeEvidence, ...]
    if evidence_by_tf:
        tf_fusion = fuse_multi_timeframe_evidence(evidence_by_tf=evidence_by_tf)
        tf_fusion_index = index_multi_timeframe_evidence(tf_fusion)
    else:
        tf_fusion = ()
        tf_fusion_index = {}

    buckets: MutableMapping[BucketKey, list[CheapGateEvidence]] = {}
    for ev in cheap_evidences:
        recipe = recipes.get(ev.recipe_id)
        if recipe is None:
            continue
        key = (recipe.family, recipe.timeframe)
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(ev)

    # Convert evidences to L0SignalCandidate
    all_candidates: list[L0SignalCandidate] = []
    for ev in cheap_evidences:
        recipe = recipes.get(ev.recipe_id)
        if recipe is None:
            continue
        ev_panel: CandidateSignalPanel | None = panel_by_rid.get(ev.recipe_id)
        source: Literal["catalog_exact", "catalog_family_variant", "synthetic_recipe"] = "synthetic_recipe"

        recipe_source = getattr(ev_panel, 'metadata', {}).get('source', '') if ev_panel else ''
        if recipe_source in ("catalog_exact", "catalog_family_variant"):
            source = recipe_source  # type: ignore[assignment]

        policy = resolve_family_timeframe_gate_policy(
            recipe=recipe,
            config=cheap_gate_config,
        )

        tf_key = (recipe.family, recipe.variant, recipe.timeframe)
        tf_ev = tf_fusion_index.get(tf_key)

        candidate = build_l0_signal_candidate(
            run_id=run_id,
            evidence=ev,
            recipe=recipe,
            source=source,
            policy=policy,
            stress_cost_bps=cost_model.stress_round_trip_bps(),
            tf_fusion=tf_ev,
        )
        all_candidates.append(candidate)

    candidate_by_rid: dict[str, L0SignalCandidate] = {c.recipe_id: c for c in all_candidates}

    # Bucket diversity by l1_priority_score
    bucket_results: list[DiversitySelectionResult] = []
    for bucket_key, cands in buckets.items():
        bucket_candidates = [candidate_by_rid[ev.recipe_id] for ev in cands if ev.recipe_id in candidate_by_rid]
        br = select_bucket_diverse_candidates(
            bucket_key=bucket_key,
            candidates=bucket_candidates,
            panel_by_recipe_id=panel_by_rid,
            active_mask=active,
            top_k_per_family_tf=top_k_per_family_tf,
            max_novelty_corr=cheap_gate_config.max_novelty_corr,
        )
        bucket_results.append(br)

    # Cross-bucket diversity
    cross_result = resolve_cross_bucket_diversity(
        bucket_results=bucket_results,
        panel_by_recipe_id=panel_by_rid,
        candidate_by_recipe_id=candidate_by_rid,
        active_mask=active,
        max_novelty_corr=cheap_gate_config.max_novelty_corr,
    )

    # Global L1 budget allocation (seed + residual)
    allocated, bucket_budgets = allocate_global_l1_budget(
        bucket_results=bucket_results,
        candidate_by_recipe_id=candidate_by_rid,
        total_l1_verification_budget=total_l1_verification_budget,
        top_k_max=top_k_per_family_tf,
        min_seed_slots_per_archetype=cheap_gate_config.min_seed_slots_per_archetype,
        min_seed_slots_per_timeframe=cheap_gate_config.min_seed_slots_per_timeframe,
    )

    final_selected = set(cross_result.final_selected_recipe_ids)

    # Assign budget units to candidates
    for rid, _cand in candidate_by_rid.items():
        recipe = recipes.get(rid)
        if recipe is None:
            continue
        bk = (recipe.family, recipe.timeframe)
        slot_limit = allocated.get(bk, 0)
        updated = replace(_cand, l1_budget_units=1 if (rid in final_selected and slot_limit > 0) else 0)
        candidate_by_rid[rid] = updated

    # Build evidence_rows with enriched L0 metadata
    selected_in_bucket: dict[str, int] = {
        rid: rank
        for br in bucket_results
        for rank, rid in enumerate(br.selected_recipe_ids)
    }

    redundant_map: dict[str, str] = {}
    for br in bucket_results:
        redundant_map.update(br.redundant_reason_by_id)
    redundant_map.update(cross_result.demoted_reason_by_id)

    # Stage counts
    hard_reject_count = 0
    soft_reject_count = 0
    seeded_count = 0
    budget_exhausted_count = 0
    tf_contradicted_count = 0
    l1_queued_count = 0

    for c in all_candidates:
        if c.discovery_tier == "blocked":
            hard_reject_count += 1
            if "tf_contradicted" in c.hard_reject_reasons:
                tf_contradicted_count += 1
        elif c.discovery_tier == "seed":
            soft_reject_count += 1
            seeded_count += 1
        elif c.discovery_tier == "candidate":
            seeded_count += 1
        if c.l1_budget_units > 0:
            l1_queued_count += 1

    stage_counts = L0ReportStageCounts(
        hard_reject=hard_reject_count,
        soft_reject=soft_reject_count,
        seeded=seeded_count,
        budget_exhausted=budget_exhausted_count,
        tf_contradicted=tf_contradicted_count,
        l1_queued=l1_queued_count,
    )

    created_at_ms = int(_time_module.time() * 1000)
    evidence_rows: list[AlphaFoundryEvidenceRow] = []
    for ev in cheap_evidences:
        recipe = recipes.get(ev.recipe_id)
        family = recipe.family if recipe else ""
        variant = recipe.variant if recipe else ""
        archetype = recipe.archetype if recipe else ""

        ev_cand: L0SignalCandidate | None = candidate_by_rid.get(ev.recipe_id)
        bucket_rank = selected_in_bucket.get(ev.recipe_id, -1)
        sel_for_l1 = ev.recipe_id in final_selected and ev_cand is not None and ev_cand.l1_budget_units > 0
        redundant_with = redundant_map.get(ev.recipe_id, "")

        stage_label = ev_cand.discovery_tier if ev_cand else ""
        source_str = ev_cand.source if ev_cand else ""

        bucket_eff = 0.0
        for br in bucket_results:
            if ev.recipe_id in br.selected_recipe_ids or ev.recipe_id in br.redundant_recipe_ids:
                bucket_eff = br.bucket_eff_test_count

        tf_cc = ev_cand.tf_coverage_count if ev_cand else 0
        sign_ar = ev_cand.sign_agreement_ratio if ev_cand else 0.0
        corr_tier = ev_cand.corroboration_tier if ev_cand else ""
        l1_ps = ev_cand.l1_priority_score if ev_cand else 0.0
        l1_bu = ev_cand.l1_budget_units if ev_cand else 0
        hr_reasons = "|".join(ev_cand.hard_reject_reasons) if ev_cand else ""
        sf_flags = "|".join(ev_cand.soft_flags) if ev_cand else ""

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
            global_eff_test_count=cross_result.global_eff_test_count if cross_result else 0.0,
            created_at_ms=created_at_ms,
            source=source_str,
            discovery_tier=stage_label,
            hard_reject_reasons=hr_reasons,
            soft_flags=sf_flags,
            l1_priority_score=l1_ps,
            l1_budget_units=l1_bu,
            tf_coverage_count=tf_cc,
            sign_agreement_ratio=sign_ar,
            corroboration_tier=corr_tier,
            stage_label=stage_label,
        ))

    return AlphaFoundryL0Artifacts(
        evidences=cheap_evidences,
        candidates=tuple(all_candidates),
        passed_recipe_ids=tuple(final_selected),
        reject_reason_counts=reject_reason_counts,
        stage_counts=stage_counts,
        evidence_rows=tuple(evidence_rows),
        bucket_results=tuple(bucket_results),
        bucket_budgets=bucket_budgets,
        cross_bucket_result=cross_result,
        tf_fusion=tf_fusion,
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
