"""Alpha Foundry L0-to-L2 orchestration bridge.

[ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]
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
    L0HandoffDecision,
    L0HandoffExclusionReason,
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
    handoff_decisions: tuple[L0HandoffDecision, ...] = ()


def build_l0_handoff_decisions(
    *,
    candidates: Sequence[L0SignalCandidate],
    recipes: Mapping[str, AlphaRecipe],
    bucket_results: Sequence[DiversitySelectionResult],
    cross_result: CrossBucketDiversityResult | None,
    allocated_slots_by_bucket: Mapping[BucketKey, int],
    panel_by_recipe_id: Mapping[str, CandidateSignalPanel],
) -> tuple[L0HandoffDecision, ...]:
    final_selected: set[str] = set()
    if cross_result is not None:
        final_selected = set(cross_result.final_selected_recipe_ids)

    bucket_ranked: dict[BucketKey, tuple[str, ...]] = {}
    for br in bucket_results:
        bucket_ranked[br.bucket_key] = br.ranked_recipe_ids

    redundant_ids: set[str] = set()
    for br in bucket_results:
        redundant_ids.update(br.redundant_recipe_ids)
    if cross_result is not None:
        redundant_ids.update(cross_result.demoted_recipe_ids)

    decisions: list[L0HandoffDecision] = []

    for c in candidates:
        recipe = recipes.get(c.recipe_id)
        bk: BucketKey = (recipe.family, recipe.timeframe) if recipe else ("", "")
        tier = c.discovery_tier
        eligible_for_diversity = (
            tier in {"seed", "candidate", "verified"}
            and not c.hard_reject_reasons
            and c.corroboration_tier != "contradicted"
        )
        has_panel = c.recipe_id in panel_by_recipe_id

        if not eligible_for_diversity:
            decisions.append(
                L0HandoffDecision(
                    recipe_id=c.recipe_id,
                    bucket_key=bk,
                    candidate_tier=tier,
                    eligible_for_diversity=False,
                    eligible_for_budget=False,
                    selected_for_l1=False,
                    budget_units=0,
                    exclusion_reason="hard_reject",
                )
            )
            continue

        if not has_panel:
            decisions.append(
                L0HandoffDecision(
                    recipe_id=c.recipe_id,
                    bucket_key=bk,
                    candidate_tier=tier,
                    eligible_for_diversity=True,
                    eligible_for_budget=False,
                    selected_for_l1=False,
                    budget_units=0,
                    exclusion_reason="missing_panel",
                )
            )
            continue

        if c.recipe_id in redundant_ids:
            reason: L0HandoffExclusionReason = (
                "cross_bucket_redundant"
                if cross_result is not None and c.recipe_id in cross_result.demoted_recipe_ids
                else "bucket_redundant"
            )
            decisions.append(
                L0HandoffDecision(
                    recipe_id=c.recipe_id,
                    bucket_key=bk,
                    candidate_tier=tier,
                    eligible_for_diversity=True,
                    eligible_for_budget=False,
                    selected_for_l1=False,
                    budget_units=0,
                    exclusion_reason=reason,
                )
            )
            continue

        allocated = allocated_slots_by_bucket.get(bk, 0)
        ranked_in_bucket = bucket_ranked.get(bk, ())

        if allocated <= 0 or c.l1_priority_score <= 0.0:
            decisions.append(
                L0HandoffDecision(
                    recipe_id=c.recipe_id,
                    bucket_key=bk,
                    candidate_tier=tier,
                    eligible_for_diversity=True,
                    eligible_for_budget=True,
                    selected_for_l1=False,
                    budget_units=0,
                    exclusion_reason="budget_exhausted" if allocated <= 0 else "non_positive_priority",
                )
            )
            continue

        top_n_allocated = list(ranked_in_bucket[:allocated])
        if c.recipe_id in top_n_allocated and c.recipe_id in final_selected:
            decisions.append(
                L0HandoffDecision(
                    recipe_id=c.recipe_id,
                    bucket_key=bk,
                    candidate_tier=tier,
                    eligible_for_diversity=True,
                    eligible_for_budget=True,
                    selected_for_l1=True,
                    budget_units=1,
                    exclusion_reason="",
                )
            )
        else:
            decisions.append(
                L0HandoffDecision(
                    recipe_id=c.recipe_id,
                    bucket_key=bk,
                    candidate_tier=tier,
                    eligible_for_diversity=True,
                    eligible_for_budget=True,
                    selected_for_l1=False,
                    budget_units=0,
                    exclusion_reason="budget_exhausted",
                )
            )

    return tuple(decisions)


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

    # Convert evidences to L0SignalCandidate
    all_candidates: list[L0SignalCandidate] = []
    for ev in cheap_evidences:
        recipe = recipes.get(ev.recipe_id)
        if recipe is None:
            continue
        ev_panel = panel_by_rid.get(ev.recipe_id)
        source: Literal["catalog_exact", "catalog_family_variant", "synthetic_recipe"] = "synthetic_recipe"

        recipe_source = getattr(ev_panel, "metadata", {}).get("source", "") if ev_panel else ""
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
            min_conviction_lcb_bps=min_conviction_lcb_bps,
        )
        all_candidates.append(candidate)

    candidate_by_rid: dict[str, L0SignalCandidate] = {c.recipe_id: c for c in all_candidates}

    # Build viable candidates: non-blocked, non-contradicted, with bound panel
    viable_candidates = [
        c
        for c in all_candidates
        if c.discovery_tier in {"seed", "candidate", "verified"}
        and not c.hard_reject_reasons
        and c.corroboration_tier != "contradicted"
        and c.recipe_id in panel_by_rid
    ]
    viable_rids = {c.recipe_id for c in viable_candidates}

    # Bucket diversity — run only on viable candidates
    buckets_viable: MutableMapping[BucketKey, list[CheapGateEvidence]] = {}
    for ev in cheap_evidences:
        if ev.recipe_id not in viable_rids:
            continue
        recipe = recipes.get(ev.recipe_id)
        if recipe is None:
            continue
        key = (recipe.family, recipe.timeframe)
        if key not in buckets_viable:
            buckets_viable[key] = []
        buckets_viable[key].append(ev)

    bucket_results: list[DiversitySelectionResult] = []
    for bucket_key, cands in buckets_viable.items():
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

    # Cross-bucket diversity — only on viable bucket survivors
    cross_result = resolve_cross_bucket_diversity(
        bucket_results=bucket_results,
        panel_by_recipe_id=panel_by_rid,
        candidate_by_recipe_id=candidate_by_rid,
        active_mask=active,
        max_novelty_corr=cheap_gate_config.max_novelty_corr,
    )

    # Global L1 budget allocation — only on viable post-diversity buckets
    allocated, bucket_budgets = allocate_global_l1_budget(
        bucket_results=bucket_results,
        candidate_by_recipe_id=candidate_by_rid,
        total_l1_verification_budget=total_l1_verification_budget,
        top_k_max=top_k_per_family_tf,
        min_seed_slots_per_archetype=cheap_gate_config.min_seed_slots_per_archetype,
        min_seed_slots_per_timeframe=cheap_gate_config.min_seed_slots_per_timeframe,
    )

    # Build handoff decisions
    handoff_decisions = build_l0_handoff_decisions(
        candidates=tuple(viable_candidates),
        recipes=recipes,
        bucket_results=bucket_results,
        cross_result=cross_result,
        allocated_slots_by_bucket=allocated,
        panel_by_recipe_id=panel_by_rid,
    )
    decision_map: dict[str, L0HandoffDecision] = {d.recipe_id: d for d in handoff_decisions}

    # Update candidates with budget units from decisions
    for rid in candidate_by_rid:
        cand = candidate_by_rid[rid]
        decision = decision_map.get(rid)
        if decision is not None:
            candidate_by_rid[rid] = replace(cand, l1_budget_units=decision.budget_units)

    passed_recipe_ids = tuple(d.recipe_id for d in handoff_decisions if d.selected_for_l1)

    # Build evidence_rows
    selected_in_bucket: dict[str, int] = {
        rid: rank for br in bucket_results for rank, rid in enumerate(br.selected_recipe_ids)
    }

    redundant_map: dict[str, str] = {}
    for br in bucket_results:
        redundant_map.update(br.redundant_reason_by_id)
    if cross_result is not None:
        redundant_map.update(cross_result.demoted_reason_by_id)

    # Stage counts
    hard_reject_count = 0
    soft_reject_count = 0
    seeded_count = 0
    budget_exhausted_count = 0
    tf_contradicted_count = 0
    l1_queued_count = 0
    viable_count = len(viable_candidates)

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

    for d in handoff_decisions:
        if d.selected_for_l1:
            l1_queued_count += 1
        if d.exclusion_reason == "budget_exhausted":
            budget_exhausted_count += 1

    stage_counts = L0ReportStageCounts(
        hard_reject=hard_reject_count,
        soft_reject=soft_reject_count,
        seeded=seeded_count,
        budget_exhausted=budget_exhausted_count,
        tf_contradicted=tf_contradicted_count,
        l1_queued=l1_queued_count,
        viable_candidates=viable_count,
    )

    created_at_ms = int(_time_module.time() * 1000)
    evidence_rows: list[AlphaFoundryEvidenceRow] = []
    for ev in cheap_evidences:
        recipe = recipes.get(ev.recipe_id)
        family = recipe.family if recipe else ""
        variant = recipe.variant if recipe else ""
        archetype = recipe.archetype if recipe else ""

        ev_cand = candidate_by_rid.get(ev.recipe_id)
        decision = decision_map.get(ev.recipe_id)
        bucket_rank = selected_in_bucket.get(ev.recipe_id, -1)
        sel_for_l1 = decision.selected_for_l1 if decision is not None else False
        l1_bu = decision.budget_units if decision is not None else (ev_cand.l1_budget_units if ev_cand else 0)
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
        hr_reasons = "|".join(ev_cand.hard_reject_reasons) if ev_cand else ""
        sf_flags = "|".join(ev_cand.soft_flags) if ev_cand else ""

        evidence_rows.append(
            AlphaFoundryEvidenceRow(
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
            )
        )

    return AlphaFoundryL0Artifacts(
        evidences=cheap_evidences,
        candidates=tuple(all_candidates),
        passed_recipe_ids=passed_recipe_ids,
        reject_reason_counts=reject_reason_counts,
        stage_counts=stage_counts,
        evidence_rows=tuple(evidence_rows),
        bucket_results=tuple(bucket_results),
        bucket_budgets=bucket_budgets,
        cross_bucket_result=cross_result,
        tf_fusion=tf_fusion,
        handoff_decisions=handoff_decisions,
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
