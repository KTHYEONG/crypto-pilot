"""Alpha Foundry diversity and effective test count helpers.

[ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]
[ADR_20260706_ALPHA_FOUNDRY_SYNC][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY]
[ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from src.domain.futures.alpha_foundry.contracts import (
    BucketKey,
    CheapGateEvidence,
    CrossBucketDiversityResult,
    DiversitySelectionResult,
    L0SignalCandidate,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel


def _paired_score_corr(
    panel_a: CandidateSignalPanel,
    panel_b: CandidateSignalPanel,
    active_mask: NDArray[np.bool_],
) -> float:
    mask = panel_a.valid_mask_2d & panel_b.valid_mask_2d & active_mask
    flat_a = panel_a.signed_score_2d[mask]
    flat_b = panel_b.signed_score_2d[mask]
    if len(flat_a) < 2:
        return 0.0
    c = np.corrcoef(flat_a, flat_b)[0, 1]
    return c if np.isfinite(c) else 0.0


def compute_panel_correlation_matrix(
    panels: Sequence[CandidateSignalPanel],
    active_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    if not panels:
        raise ValueError("panels must not be empty")
    for p in panels:
        if p.signed_score_2d.shape != active_mask.shape:
            raise ValueError(f"panel shape {p.signed_score_2d.shape} != active_mask shape {active_mask.shape}")
    m = len(panels)
    corr = np.full((m, m), np.nan, dtype=np.float64)
    for i in range(m):
        for j in range(m):
            c = _paired_score_corr(panels[i], panels[j], active_mask)
            corr[i, j] = c if np.isfinite(c) else 0.0
        corr[i, i] = 1.0
    return corr


def cluster_correlated_recipes(
    *,
    evidences: Sequence[Any],
    corr: NDArray[np.float64],
    max_corr: float,
) -> tuple[tuple[str, ...], ...]:
    n = len(evidences)
    if corr.shape != (n, n):
        raise ValueError(f"corr shape {corr.shape} != (n_evidences={n}, {n})")
    recipe_ids = [e.recipe_id for e in evidences]
    assigned: set[int] = set()
    clusters: list[tuple[str, ...]] = []
    for i in range(n):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        for j in range(i + 1, n):
            if j in assigned:
                continue
            if corr[i, j] > max_corr:
                cluster.append(j)
                assigned.add(j)
        clusters.append(tuple(recipe_ids[idx] for idx in cluster))
    return tuple(clusters)


def estimate_effective_test_count(
    corr: NDArray[np.float64],
) -> float:
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"corr must be square, got shape {corr.shape}")
    n = corr.shape[0]
    if n <= 1:
        return float(n)
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = np.sum(eigenvalues)
    if total <= 0.0:
        return 1.0
    probs = eigenvalues / total
    entropy = -np.sum(probs * np.log(probs + 1e-15))
    m_eff = np.exp(entropy)
    return float(max(1.0, min(m_eff, float(n))))


def apply_bucket_bh_correction(
    candidates: Sequence[CheapGateEvidence],
    fdr_alpha: float,
) -> frozenset[str]:
    if not candidates:
        return frozenset()
    sorted_candidates = sorted(candidates, key=lambda c: abs(c.nw_tstat), reverse=True)
    m = len(sorted_candidates)
    selected: set[str] = set()
    max_i = -1
    for i, c in enumerate(sorted_candidates):
        p = 2.0 * (1.0 - scipy_stats.norm.cdf(abs(c.nw_tstat)))
        if p <= ((i + 1) / m) * fdr_alpha:
            max_i = i
    for c in sorted_candidates[: max_i + 1]:
        selected.add(c.recipe_id)
    return frozenset(selected)


def select_bucket_diverse_candidates(
    *,
    bucket_key: BucketKey,
    candidates: Sequence[L0SignalCandidate],
    panel_by_recipe_id: Mapping[str, CandidateSignalPanel],
    active_mask: NDArray[np.bool_],
    top_k_per_family_tf: int,
    max_novelty_corr: float,
) -> DiversitySelectionResult:
    redundant_reason_map: dict[str, str] = {}

    ranked = sorted(candidates, key=lambda e: (-e.l1_priority_score, e.recipe_id))
    ranked_ids = tuple(e.recipe_id for e in ranked)

    if not ranked:
        return DiversitySelectionResult(
            bucket_key=bucket_key,
            ranked_recipe_ids=(),
            selected_recipe_ids=(),
            redundant_recipe_ids=tuple(redundant_reason_map.keys()),
            redundant_reason_by_id=redundant_reason_map,
            bucket_corr=np.empty((0, 0), dtype=np.float64),
            bucket_eff_test_count=0.0,
        )

    if len(ranked) == 1:
        return DiversitySelectionResult(
            bucket_key=bucket_key,
            ranked_recipe_ids=ranked_ids,
            selected_recipe_ids=ranked_ids,
            redundant_recipe_ids=tuple(redundant_reason_map.keys()),
            redundant_reason_by_id=redundant_reason_map,
            bucket_corr=np.array([[1.0]], dtype=np.float64),
            bucket_eff_test_count=1.0,
        )

    selected: list[L0SignalCandidate] = [ranked[0]]
    redundant: list[L0SignalCandidate] = []

    for candidate in ranked[1:]:
        if len(selected) >= top_k_per_family_tf:
            redundant.append(candidate)
            continue
        max_corr = 0.0
        max_corr_id = ""
        for sel in selected:
            pa = panel_by_recipe_id.get(candidate.recipe_id)
            pb = panel_by_recipe_id.get(sel.recipe_id)
            if pa is None or pb is None:
                continue
            corr_val = _paired_score_corr(pa, pb, active_mask)
            if corr_val > max_corr:
                max_corr = corr_val
                max_corr_id = sel.recipe_id
        if max_corr > max_novelty_corr:
            redundant.append(candidate)
            redundant_reason_map[candidate.recipe_id] = max_corr_id
        else:
            selected.append(candidate)

    selected_ids = tuple(e.recipe_id for e in selected)
    redundant_ids = tuple(e.recipe_id for e in redundant)

    n_sel = len(selected)
    if n_sel > 1:
        selected_panels = [panel_by_recipe_id[rid] for rid in selected_ids if rid in panel_by_recipe_id]
        if len(selected_panels) == n_sel:
            bucket_corr = compute_panel_correlation_matrix(selected_panels, active_mask)
            bucket_eff = estimate_effective_test_count(bucket_corr)
        else:
            bucket_corr = np.eye(n_sel, dtype=np.float64)
            bucket_eff = float(n_sel)
    else:
        bucket_corr = np.array([[1.0]], dtype=np.float64)
        bucket_eff = 1.0

    return DiversitySelectionResult(
        bucket_key=bucket_key,
        ranked_recipe_ids=ranked_ids,
        selected_recipe_ids=selected_ids,
        redundant_recipe_ids=tuple(set(redundant_ids) | set(redundant_reason_map.keys())),
        redundant_reason_by_id=redundant_reason_map,
        bucket_corr=bucket_corr,
        bucket_eff_test_count=bucket_eff,
    )


def resolve_cross_bucket_diversity(
    *,
    bucket_results: Sequence[DiversitySelectionResult],
    panel_by_recipe_id: Mapping[str, CandidateSignalPanel],
    candidate_by_recipe_id: Mapping[str, L0SignalCandidate],
    active_mask: NDArray[np.bool_],
    max_novelty_corr: float,
) -> CrossBucketDiversityResult:
    all_selected: list[str] = []
    for br in bucket_results:
        all_selected.extend(br.selected_recipe_ids)

    if not all_selected:
        return CrossBucketDiversityResult(
            final_selected_recipe_ids=(),
            demoted_recipe_ids=(),
            demoted_reason_by_id={},
            cross_bucket_corr=np.empty((0, 0), dtype=np.float64),
            global_eff_test_count=0.0,
        )

    if len(all_selected) == 1:
        return CrossBucketDiversityResult(
            final_selected_recipe_ids=tuple(all_selected),
            demoted_recipe_ids=(),
            demoted_reason_by_id={},
            cross_bucket_corr=np.array([[1.0]], dtype=np.float64),
            global_eff_test_count=1.0,
        )

    panels = [panel_by_recipe_id[rid] for rid in all_selected if rid in panel_by_recipe_id]
    if len(panels) < 2:
        return CrossBucketDiversityResult(
            final_selected_recipe_ids=tuple(all_selected),
            demoted_recipe_ids=(),
            demoted_reason_by_id={},
            cross_bucket_corr=np.eye(len(all_selected), dtype=np.float64),
            global_eff_test_count=float(len(all_selected)),
        )

    cross_corr = compute_panel_correlation_matrix(panels, active_mask)
    all_candidates = [candidate_by_recipe_id[rid] for rid in all_selected if rid in candidate_by_recipe_id]
    if len(all_candidates) != len(all_selected):
        all_candidates = [
            L0SignalCandidate(
                run_id="",
                timeframe="",
                family="",
                variant="",
                recipe_id=rid,
                archetype="trend",
                source="synthetic_recipe",
                n_events=0,
                effective_n=0.0,
                mean_net_bps=0.0,
                block_lcb_bps=0.0,
                nw_tstat=0.0,
                bootstrap_lcb_bps=0.0,
                bootstrap_agree=True,
                cost_drag_ratio=0.0,
                turnover_per_year=0.0,
                max_abs_corr_in_bucket=0.0,
                tf_coverage_count=0,
                sign_agreement_ratio=0.0,
                corroboration_tier="insufficient_coverage",
                discovery_tier="candidate",
                l1_priority_score=0.0,
                l1_budget_units=0,
                hard_reject_reasons=(),
                soft_flags=(),
            )
            for rid in all_selected
        ]

    clusters = cluster_correlated_recipes(
        evidences=all_candidates,
        corr=cross_corr,
        max_corr=max_novelty_corr,
    )

    final_selected: list[str] = []
    demoted: list[str] = []
    demoted_reason_map: dict[str, str] = {}

    for cluster in clusters:
        if len(cluster) == 1:
            final_selected.append(cluster[0])
        else:

            def _best_key(rid: str) -> float:
                cand = candidate_by_recipe_id.get(rid)
                if cand is None:
                    idx = all_selected.index(rid)
                    cand = all_candidates[idx]
                return cand.l1_priority_score

            best_rid = max(cluster, key=_best_key)
            for rid in cluster:
                if rid == best_rid:
                    final_selected.append(rid)
                else:
                    demoted.append(rid)
                    demoted_reason_map[rid] = best_rid

    global_eff = estimate_effective_test_count(cross_corr)

    return CrossBucketDiversityResult(
        final_selected_recipe_ids=tuple(final_selected),
        demoted_recipe_ids=tuple(demoted),
        demoted_reason_by_id=demoted_reason_map,
        cross_bucket_corr=cross_corr,
        global_eff_test_count=global_eff,
    )


def audit_full_family_correlation(
    *,
    panels: Sequence[CandidateSignalPanel],
    active_mask: NDArray[np.bool_],
    run_id: str,
    timeframe: str,
    max_corr: float = 0.85,
) -> pd.DataFrame:
    """Pre-gate correlation audit across the FULL un-gated panel set (opt-in diagnostic).

    [ADR_20260707_L0_ALPHA_EFFECTIVENESS_REDESIGN]
    """
    if not panels:
        raise ValueError("panels must not be empty")

    m = len(panels)
    corr = compute_panel_correlation_matrix(panels, active_mask)

    assigned: set[int] = set()
    cluster_ids: list[int] = [-1] * m
    next_id = 0
    for i in range(m):
        if i in assigned:
            continue
        cluster_ids[i] = next_id
        assigned.add(i)
        for j in range(i + 1, m):
            if j in assigned:
                continue
            if corr[i, j] > max_corr:
                cluster_ids[j] = next_id
                assigned.add(j)
        next_id += 1

    eff_test_count = estimate_effective_test_count(corr)

    rows = [
        {
            "family_a": panels[i].family,
            "variant_a": panels[i].variant,
            "family_b": panels[j].family,
            "variant_b": panels[j].variant,
            "timeframe": timeframe,
            "pairwise_corr": corr[i, j],
            "cluster_id": cluster_ids[i],
            "run_id": run_id,
        }
        for i in range(m)
        for j in range(m)
    ]

    df = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "family_a": "__SUMMARY__",
        "variant_a": "",
        "family_b": "",
        "variant_b": "",
        "timeframe": timeframe,
        "pairwise_corr": eff_test_count,
        "cluster_id": -1,
        "run_id": run_id,
    }])

    return pd.concat([df, summary], ignore_index=True)


# Backward-compat alias
select_bucket_diverse_recipes = select_bucket_diverse_candidates
