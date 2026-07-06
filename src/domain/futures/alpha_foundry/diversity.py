"""Alpha Foundry diversity and effective test count helpers. [ADR_20260706_ALPHA_FOUNDRY_SYNC][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY]"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.alpha_foundry.contracts import (
    BucketKey,
    CheapGateEvidence,
    CrossBucketDiversityResult,
    DiversitySelectionResult,
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
            raise ValueError(
                f"panel shape {p.signed_score_2d.shape} != active_mask shape {active_mask.shape}"
            )
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
    evidences: Sequence[CheapGateEvidence],
    corr: NDArray[np.float64],
    max_corr: float,
) -> tuple[tuple[str, ...], ...]:
    n = len(evidences)
    if corr.shape != (n, n):
        raise ValueError(
            f"corr shape {corr.shape} != (n_evidences={n}, {n})"
        )
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


def select_bucket_diverse_recipes(
    *,
    bucket_key: BucketKey,
    candidates: Sequence[CheapGateEvidence],
    panel_by_recipe_id: Mapping[str, CandidateSignalPanel],
    fwd_ret_by_recipe_id: Mapping[str, NDArray[np.float64]],
    active_mask: NDArray[np.bool_],
    top_k_per_family_tf: int,
    max_novelty_corr: float,
) -> DiversitySelectionResult:
    ranked = sorted(candidates, key=lambda e: (-e.block_lcb_bps, e.recipe_id))
    ranked_ids = tuple(e.recipe_id for e in ranked)

    if not ranked:
        return DiversitySelectionResult(
            bucket_key=bucket_key,
            ranked_recipe_ids=(),
            selected_recipe_ids=(),
            redundant_recipe_ids=(),
            redundant_reason_by_id={},
            bucket_corr=np.empty((0, 0), dtype=np.float64),
            bucket_eff_test_count=0.0,
        )

    if len(ranked) == 1:
        return DiversitySelectionResult(
            bucket_key=bucket_key,
            ranked_recipe_ids=ranked_ids,
            selected_recipe_ids=ranked_ids,
            redundant_recipe_ids=(),
            redundant_reason_by_id={},
            bucket_corr=np.array([[1.0]], dtype=np.float64),
            bucket_eff_test_count=1.0,
        )

    selected: list[CheapGateEvidence] = [ranked[0]]
    redundant: list[CheapGateEvidence] = []
    redundant_reason_map: dict[str, str] = {}

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
            c = _paired_score_corr(pa, pb, active_mask)
            if c > max_corr:
                max_corr = c
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
        selected_panels = [
            panel_by_recipe_id[rid] for rid in selected_ids if rid in panel_by_recipe_id
        ]
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
        redundant_recipe_ids=redundant_ids,
        redundant_reason_by_id=redundant_reason_map,
        bucket_corr=bucket_corr,
        bucket_eff_test_count=bucket_eff,
    )


def resolve_cross_bucket_diversity(
    *,
    bucket_results: Sequence[DiversitySelectionResult],
    panel_by_recipe_id: Mapping[str, CandidateSignalPanel],
    evidence_by_recipe_id: Mapping[str, CheapGateEvidence],
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
    all_evidences = [evidence_by_recipe_id[rid] for rid in all_selected if rid in evidence_by_recipe_id]
    if len(all_evidences) != len(all_selected):
        all_evidences = [
            CheapGateEvidence(
                recipe_id=rid, timeframe="", symbol_scope="symbol",
                n_events=0, effective_n=0.0, mean_net_bps=0.0, nw_tstat=0.0,
                block_lcb_bps=0.0, rank_ic=0.0, cost_drag_ratio=0.0,
                turnover_per_year=0.0, novelty_corr_max=0.0,
                incremental_rank_ic=0.0, compute_cost_score=0.0,
                gate_passed=True, reject_reasons=(),
            )
            for rid in all_selected
        ]

    clusters = cluster_correlated_recipes(
        evidences=all_evidences,
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
                ev = evidence_by_recipe_id.get(rid)
                if ev is None:
                    idx = all_selected.index(rid)
                    ev = all_evidences[idx]
                return ev.block_lcb_bps
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
