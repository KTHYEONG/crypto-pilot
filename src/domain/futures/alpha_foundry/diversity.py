"""Alpha Foundry diversity and effective test count helpers.

[ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]
[ADR_20260706_ALPHA_FOUNDRY_SYNC][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY]
[ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
[ADR_20260711_L0_STRATEGY_DELIVERY_HARDENING]
[ADR_20260711_L0_CROSS_TF_PRUNING_ADMISSION]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from src.domain.futures.alpha_foundry.contracts import (
    BucketKey,
    CheapGateEvidence,
    CrossBucketDiversityResult,
    CrossTFCanonicalContext,
    CrossTFPairEvidence,
    CrossTFSharedContext,
    DiversitySelectionResult,
    L0IndependenceAudit,
    L0SignalCandidate,
)
from src.domain.futures.alpha_foundry.memory import admit_memory_stage, resolve_effective_memory_budget
from src.domain.futures.alpha_foundry.multi_tf_fusion import project_signal_to_canonical_grid
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData

_TF_TO_MINUTES_MAP: dict[str, int] = {
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "3h": 180,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
}


def _tf_to_minutes(tf: str) -> int:
    return _TF_TO_MINUTES_MAP.get(tf, 9999)


def _find_priority(
    recipe_id: str,
    candidates: Sequence[L0SignalCandidate],
    all_recipe_ids: Sequence[str],
) -> float:
    for c in candidates:
        if c.recipe_id == recipe_id:
            return c.l1_priority_score
    idx = all_recipe_ids.index(recipe_id) if recipe_id in all_recipe_ids else -1
    return float(idx)


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


# ── Economic Thesis Grouping (Fix 3 — additive/observability only) ────────

FAMILY_THESIS_GROUP: Mapping[str, str] = {
    "trend_ma": "trend_ma_cross", "ema_trend": "trend_ma_cross", "ichimoku_trend": "trend_ma_cross",
    "volume_participation_breakout": "breakout_retest_liquidity",
    "liquidity_participation_breakout": "breakout_retest_liquidity",
    "vol_contraction_breakout": "breakout_retest_liquidity",
    "carry_net_of_funding": "funding_carry", "funding_slope_carry": "funding_carry",
    "oi_lsr_unwind": "oi_unwind",
    "xs_momentum": "xs_momentum_continuation", "residual_momentum_xs": "xs_momentum_continuation",
    "xs_residual_rebalance": "xs_momentum_continuation",
    "btc_neutral_residual_reversal": "xs_residual_reversal",
    "price_band_reversion": "price_band_reversion",
}  # [ADR_20260710_L0_SIGNAL_BREADTH_DIVERSITY_REDESIGN] economic-thesis grouping for diversity reporting


def resolve_economic_thesis_id(family: str) -> str:
    """Map a family name to its shared economic-thesis group. [LIMIT-09]

    Unmapped families default to their own family name (singleton group) so a
    newly added family is never silently merged into an existing cluster.
    """
    return FAMILY_THESIS_GROUP.get(family, family)


def estimate_distinct_thesis_count(
    evidence_families: Sequence[str],
) -> int:
    """Count distinct economic-thesis groups among the given family names.

    Pure/additive: never consumed by gate_passed/handoff_tier/selected_for_l1
    decisions. [LIMIT-10]
    """
    if not evidence_families:
        return 0
    return len({resolve_economic_thesis_id(f) for f in evidence_families})


def resolve_cross_tf_canonical_context(
    *,
    selected_by_tf: Mapping[str, Sequence[L0SignalCandidate]],
    panel_by_recipe_id: Mapping[str, CandidateSignalPanel],
    aligned_by_tf: Mapping[str, AlignedMarketData],
    min_common_active_bars: int,
) -> CrossTFCanonicalContext:
    """Resolve a common canonical context from all selected candidates.

    [ADR_20260712_L0_EVIDENCE_CONDITIONED_CROSS_TF_ADMISSION]

    Raises:
        ValueError: if no common history exists or min_common_active_bars unmet.
    """
    if not selected_by_tf:
        raise ValueError("selected_by_tf must not be empty")

    selected_tfs = list(selected_by_tf)
    canonical_tf = min(selected_tfs, key=lambda tf: _tf_to_minutes(tf))

    aligned = aligned_by_tf[canonical_tf]
    canonical_datetimes_ns = aligned.datetimes.astype("datetime64[ns]").astype(np.int64)

    active_mask_2d = (
        aligned.active_mask
        & aligned.warm_mask
        & ~aligned.entry_block_mask
        & ~aligned.kill_mask
    )

    # Compute common start/end from all bound panels
    common_start_ns: int = 0
    common_end_ns: int = 0
    first = True
    all_recipe_ids: list[str] = []
    for candidates in selected_by_tf.values():
        all_recipe_ids.extend(c.recipe_id for c in candidates)
    for rid in all_recipe_ids:
        panel = panel_by_recipe_id.get(rid)
        if panel is None:
            continue
        panel_dt = np.asarray(panel.datetimes, dtype="datetime64[ns]").astype(np.int64)
        if first:
            common_start_ns = int(np.min(panel_dt))
            common_end_ns = int(np.max(panel_dt))
            first = False
        else:
            common_start_ns = max(common_start_ns, int(np.min(panel_dt)))
            common_end_ns = min(common_end_ns, int(np.max(panel_dt)))

    if first:
        raise ValueError("no panel with datetimes found in selected candidates")

    # Clip canonical grid to common interval
    grid_mask = (canonical_datetimes_ns >= common_start_ns) & (canonical_datetimes_ns <= common_end_ns)
    clipped_active = active_mask_2d[grid_mask]
    n_common_active_bars = int(np.sum(np.any(clipped_active, axis=1)))

    if n_common_active_bars < min_common_active_bars:
        raise ValueError(
            f"n_common_active_bars={n_common_active_bars} < min_common_active_bars={min_common_active_bars}"
        )

    return CrossTFCanonicalContext(
        canonical_tf=canonical_tf,
        canonical_datetimes_ns=canonical_datetimes_ns,
        active_mask_2d=active_mask_2d,
        common_start_ns=common_start_ns,
        common_end_ns=common_end_ns,
        n_common_active_bars=n_common_active_bars,
    )


def resolve_cross_tf_shared_context(
    *,
    selected_by_tf: Mapping[str, Sequence[L0SignalCandidate]],
    panel_by_recipe_id: Mapping[str, CandidateSignalPanel],
    aligned_by_tf: Mapping[str, AlignedMarketData],
    min_common_active_bars: int,
) -> CrossTFSharedContext:
    """Pure. Builds canonical context + proj_cache + side_entry_cache + upper-
    triangle-mirrored corr matrix ONCE. Raises ValueError (propagated, unchanged
    semantics) if resolve_cross_tf_canonical_context()'s min_common_active_bars
    guard fails or if memory budget is exceeded.

    [ADR_20260712_L0_CROSS_TF_BATCH_ACCELERATION] Also precomputes
    entry_pos_flat/entry_neg_flat/n_entries for batch jaccard (OPT-2) and
    valid_stack for corr-loop mask broadcast (OPT-1-a).

    [ADR_20260712_L0_CROSS_TF_BATCH_CORRELATION] Uses _batch_pairwise_corr
    (4 BLAS matmuls) instead of per-pair np.corrcoef (OPT-4)."""
    budget = resolve_effective_memory_budget()
    all_recipe_ids_list: list[str] = []
    for candidates in selected_by_tf.values():
        all_recipe_ids_list.extend(c.recipe_id for c in candidates)
    n_selected = len(all_recipe_ids_list)

    context = resolve_cross_tf_canonical_context(
        selected_by_tf=selected_by_tf,
        panel_by_recipe_id=panel_by_recipe_id,
        aligned_by_tf=aligned_by_tf,
        min_common_active_bars=min_common_active_bars,
    )

    n_canonical_bars = len(context.canonical_datetimes_ns)
    n_symbols_sample = 0
    for rid in all_recipe_ids_list:
        p = panel_by_recipe_id.get(rid)
        if p is not None:
            n_symbols_sample = p.signed_score_2d.shape[1]
            break

    estimated_increment_mb = (
        n_selected * n_canonical_bars * n_symbols_sample * 4 // 1_000_000
        + n_selected * n_canonical_bars * n_symbols_sample * 1 // 1_000_000
    )
    if not admit_memory_stage(
        budget=budget,
        stage="l0_cross_tf_shared_context",
        estimated_increment_mb=max(estimated_increment_mb, 1),
    ):
        raise ValueError(
            f"memory_budget_exceeded: estimated_increment_mb={estimated_increment_mb}"
            f" > budget_mb={budget.limit_mb}"
        )

    proj_cache: dict[str, tuple[NDArray[np.float32], NDArray[np.bool_]]] = {}
    side_entry_cache: dict[str, tuple[NDArray[np.int8], NDArray[np.bool_]]] = {}
    for rid in all_recipe_ids_list:
        p = panel_by_recipe_id.get(rid)
        if p is not None:
            proj = project_signal_to_canonical_grid(
                panel=p,
                canonical_datetimes=context.canonical_datetimes_ns,
                causal_lag_bars=1,
            )
            proj_cache[rid] = proj
            side_entry_cache[rid] = _causal_projected_side_and_entry(
                proj[0], proj[1], context.active_mask_2d,
            )

    entry_pos_flat: dict[str, "NDArray[np.int8]"] = {}  # noqa: UP037
    entry_neg_flat: dict[str, "NDArray[np.int8]"] = {}  # noqa: UP037
    n_entries: dict[str, int] = {}
    for rid in all_recipe_ids_list:
        if rid in side_entry_cache:
            side, entry = side_entry_cache[rid]
            pos = (entry & (side == 1)).ravel().astype(np.int8)
            neg = (entry & (side == -1)).ravel().astype(np.int8)
            entry_pos_flat[rid] = pos
            entry_neg_flat[rid] = neg
            n_entries[rid] = int(entry.sum())
        else:
            entry_pos_flat[rid] = np.array([], dtype=np.int8)
            entry_neg_flat[rid] = np.array([], dtype=np.int8)
            n_entries[rid] = 0

    n = len(all_recipe_ids_list)
    n_canonical_bars_v = len(context.canonical_datetimes_ns)
    n_symbols_v = (
        proj_cache[all_recipe_ids_list[0]][0].shape[1]
        if n > 0 and all_recipe_ids_list[0] in proj_cache
        else 0
    )
    valid_stack = np.zeros((n, n_canonical_bars_v, n_symbols_v), dtype=np.bool_)
    for i_idx, rid in enumerate(all_recipe_ids_list):
        if rid in proj_cache:
            valid_stack[i_idx] = proj_cache[rid][1]

    if n == 0:
        corr = np.zeros((0, 0), dtype=np.float64)
    else:
        projected_stack = np.empty((n, n_canonical_bars_v, n_symbols_v), dtype=np.float32)
        for idx, rid in enumerate(all_recipe_ids_list):
            if rid in proj_cache:
                projected_stack[idx] = proj_cache[rid][0]
            else:
                projected_stack[idx] = 0.0

        corr = _batch_pairwise_corr(
            projected_stack=projected_stack,
            mask_stack=valid_stack,
            active_mask=context.active_mask_2d,
        )
        missing_mask = np.array([rid not in proj_cache for rid in all_recipe_ids_list])
        if missing_mask.any():
            corr[missing_mask] = 0.0
            corr[:, missing_mask] = 0.0
            np.fill_diagonal(corr, 1.0)

    return CrossTFSharedContext(
        canonical_context=context,
        proj_cache=proj_cache,
        side_entry_cache=side_entry_cache,
        corr=corr,
        recipe_order=tuple(all_recipe_ids_list),
        entry_pos_flat=entry_pos_flat,
        entry_neg_flat=entry_neg_flat,
        n_entries=n_entries,
    )


def _batch_pairwise_corr(
    *,
    projected_stack: NDArray[np.float32],
    mask_stack: NDArray[np.bool_],
    active_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Pure. Batch Pearson via 4 BLAS matmuls.

    [ADR_20260712_L0_CROSS_TF_BATCH_CORRELATION] Replaces O(N²) per-pair
    np.corrcoef Python loop with batch matmul. NaN-safe via np.where."""
    ndim = projected_stack.ndim
    if ndim < 2:
        raise ValueError("projected_stack must be at least 2D")
    if projected_stack.shape != mask_stack.shape:
        raise ValueError(
            f"shape mismatch projected={projected_stack.shape} mask={mask_stack.shape}"
        )

    n = projected_stack.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    flat_len = np.prod(projected_stack.shape[1:], dtype=np.intp)
    proj_f64 = projected_stack.reshape(n, flat_len).astype(np.float64, copy=False)
    comb_bool = mask_stack.reshape(n, flat_len) & active_mask.ravel()
    comb_f64 = comb_bool.astype(np.float64)

    x_vm = np.where(comb_f64 > 0, proj_f64, 0.0)
    x2_vm = x_vm * x_vm
    mask_f64 = comb_f64

    cross_sum = x_vm @ x_vm.T
    cross_count = mask_f64 @ mask_f64.T
    row_sum = x_vm @ mask_f64.T
    row_sq = x2_vm @ mask_f64.T

    n_obs = cross_count
    num = n_obs * cross_sum - row_sum * row_sum.T
    d1 = n_obs * row_sq - row_sum * row_sum
    d2 = n_obs * row_sq.T - row_sum.T * row_sum.T
    denom = np.sqrt(np.maximum(d1 * d2, 0.0)) + 1e-12
    corr: NDArray[np.float64] = np.clip(num / denom, -1.0, 1.0)
    corr[n_obs < 2] = 0.0
    np.fill_diagonal(corr, 1.0)
    return (corr + corr.T) * 0.5


def _causal_projected_side_and_entry(
    score: NDArray[np.float32 | np.float64],
    valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
) -> tuple[NDArray[np.int8], NDArray[np.bool_]]:
    side = np.zeros(score.shape, dtype=np.int8)
    side[valid] = np.sign(score[valid]).astype(np.int8)
    side_prev = np.vstack([np.zeros_like(side[:1]), side[:-1, :]])
    entry: NDArray[np.bool_] = (side != 0) & (side != side_prev) & valid & active
    return side, entry


def compute_cross_tf_pair_evidence(
    *,
    recipe_id_a: str,
    recipe_id_b: str,
    panel_a: CandidateSignalPanel,
    panel_b: CandidateSignalPanel,
    context: CrossTFCanonicalContext,
    min_score_corr: float,
    min_directional_entry_jaccard: float,
    min_shared_directional_entries: int,
    # [NEW, additive, keyword-only, all default None => existing self-compute behavior]
    precomputed_proj_a: tuple[NDArray[np.float32], NDArray[np.bool_]] | None = None,
    precomputed_proj_b: tuple[NDArray[np.float32], NDArray[np.bool_]] | None = None,
    precomputed_side_entry_a: tuple[NDArray[np.int8], NDArray[np.bool_]] | None = None,
    precomputed_side_entry_b: tuple[NDArray[np.int8], NDArray[np.bool_]] | None = None,
    precomputed_score_corr: float | None = None,
) -> CrossTFPairEvidence:
    """[LIMIT-01][LIMIT-02][LIMIT-03] When precomputed_* is provided, skips the
    corresponding self-computation entirely. Signature/return type otherwise
    unchanged; all existing call sites (tests, single-pair callers) continue to
    work identically with default None."""
    if precomputed_score_corr is not None:
        score_corr = precomputed_score_corr
        shared_count: int = 0
        j_dir: float = 0.0
        if precomputed_side_entry_a is not None and precomputed_side_entry_b is not None:
            side_a, entry_a = precomputed_side_entry_a
            side_b, entry_b = precomputed_side_entry_b
            shared = entry_a & entry_b & (side_a == side_b)
            union = entry_a | entry_b
            shared_count = int(np.sum(shared))
            union_count = int(np.sum(union))
            j_dir = shared_count / max(union_count, 1)
        return CrossTFPairEvidence(
            recipe_id_a=recipe_id_a, recipe_id_b=recipe_id_b,
            score_corr=score_corr,
            shared_directional_entries=shared_count,
            directional_entry_jaccard=j_dir,
            is_redundant=(
                score_corr >= min_score_corr
                and shared_count >= min_shared_directional_entries
                and j_dir >= min_directional_entry_jaccard
            ),
        )

    canonical_dt = context.canonical_datetimes_ns
    if precomputed_proj_a is not None:
        proj_a_s, proj_a_v = precomputed_proj_a
    else:
        proj_a_s, proj_a_v = project_signal_to_canonical_grid(
            panel=panel_a, canonical_datetimes=canonical_dt, causal_lag_bars=1,
        )
    if precomputed_proj_b is not None:
        proj_b_s, proj_b_v = precomputed_proj_b
    else:
        proj_b_s, proj_b_v = project_signal_to_canonical_grid(
            panel=panel_b, canonical_datetimes=canonical_dt, causal_lag_bars=1,
        )

    active = context.active_mask_2d
    valid_ab = proj_a_v & proj_b_v & active
    flat_a = proj_a_s[valid_ab]
    flat_b = proj_b_s[valid_ab]
    if len(flat_a) < 2:
        return CrossTFPairEvidence(
            recipe_id_a=recipe_id_a, recipe_id_b=recipe_id_b,
            score_corr=0.0, shared_directional_entries=0,
            directional_entry_jaccard=0.0, is_redundant=False,
        )

    c = float(np.corrcoef(flat_a, flat_b)[0, 1])
    score_corr = c if np.isfinite(c) else 0.0

    if precomputed_side_entry_a is not None:
        side_a, entry_a = precomputed_side_entry_a
    else:
        side_a, entry_a = _causal_projected_side_and_entry(proj_a_s, proj_a_v, active)

    if precomputed_side_entry_b is not None:
        side_b, entry_b = precomputed_side_entry_b
    else:
        side_b, entry_b = _causal_projected_side_and_entry(proj_b_s, proj_b_v, active)

    shared = entry_a & entry_b & (side_a == side_b)
    union = entry_a | entry_b
    shared_count = int(np.sum(shared))
    union_count = int(np.sum(union))
    j_dir = shared_count / max(union_count, 1)

    is_redundant = (
        score_corr >= min_score_corr
        and shared_count >= min_shared_directional_entries
        and j_dir >= min_directional_entry_jaccard
    )

    return CrossTFPairEvidence(
        recipe_id_a=recipe_id_a, recipe_id_b=recipe_id_b,
        score_corr=score_corr,
        shared_directional_entries=shared_count,
        directional_entry_jaccard=float(j_dir),
        is_redundant=is_redundant,
    )


def compute_cross_tf_redundancy(
    *,
    selected_by_tf: Mapping[str, Sequence[L0SignalCandidate]],
    panel_by_recipe_id: Mapping[str, CandidateSignalPanel],
    aligned_by_tf: Mapping[str, AlignedMarketData],
    min_common_active_bars: int,
    max_novelty_corr: float,
    min_directional_entry_jaccard: float,
    min_shared_directional_entries: int,
    precomputed_shared_context: CrossTFSharedContext | None = None,
) -> CrossBucketDiversityResult:
    """Leader-based greedy demotion of cross-TF redundant candidates.

    [ADR_20260712_L0_CROSS_TF_BATCH_ACCELERATION] When precomputed_shared_context
    has entry_pos_flat populated, uses batch matmul jaccard (OPT-2) and dict-
    lookup leader greedy (OPT-3) instead of per-pair scan."""
    if not selected_by_tf:
        raise ValueError("selected_by_tf must not be empty")

    if precomputed_shared_context is not None:
        context = precomputed_shared_context.canonical_context
        proj_cache = precomputed_shared_context.proj_cache
        corr = precomputed_shared_context.corr
        all_recipe_ids = list(precomputed_shared_context.recipe_order)
        side_entry_cache = precomputed_shared_context.side_entry_cache
    else:
        context = resolve_cross_tf_canonical_context(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=min_common_active_bars,
        )
        all_selected_local: list[L0SignalCandidate] = []
        for candidates in selected_by_tf.values():
            all_selected_local.extend(candidates)
        all_recipe_ids = [c.recipe_id for c in all_selected_local]

        if len(all_selected_local) <= 1:
            return CrossBucketDiversityResult(
                final_selected_recipe_ids=tuple(all_recipe_ids),
                demoted_recipe_ids=(),
                demoted_reason_by_id={},
                cross_bucket_corr=(
                    np.eye(len(all_selected_local), dtype=np.float64)
                    if all_selected_local else np.empty((0, 0), dtype=np.float64)
                ),
                global_eff_test_count=float(len(all_selected_local)),
                canonical_tf=context.canonical_tf,
                common_start_ns=context.common_start_ns,
                common_end_ns=context.common_end_ns,
                n_common_active_bars=context.n_common_active_bars,
            )

        panels = [panel_by_recipe_id[rid] for rid in all_recipe_ids if rid in panel_by_recipe_id]
        if len(panels) < 2:
            return CrossBucketDiversityResult(
                final_selected_recipe_ids=tuple(all_recipe_ids),
                demoted_recipe_ids=(),
                demoted_reason_by_id={},
                cross_bucket_corr=np.eye(len(all_recipe_ids), dtype=np.float64),
                global_eff_test_count=float(len(all_recipe_ids)),
                canonical_tf=context.canonical_tf,
                common_start_ns=context.common_start_ns,
                common_end_ns=context.common_end_ns,
                n_common_active_bars=context.n_common_active_bars,
            )

        # Cache projections by recipe_id (reuse across pair evaluation)
        proj_cache = {}
        side_entry_cache = {}
        for rid in all_recipe_ids:
            p = panel_by_recipe_id.get(rid)
            if p is not None:
                proj = project_signal_to_canonical_grid(
                    panel=p,
                    canonical_datetimes=context.canonical_datetimes_ns,
                    causal_lag_bars=1,
                )
                proj_cache[rid] = proj
                side_entry_cache[rid] = _causal_projected_side_and_entry(
                    proj[0], proj[1], context.active_mask_2d,
                )

        # Score correlation matrix for report consumers — i<j only, mirror to j,i
        n = len(all_recipe_ids)
        corr = np.full((n, n), np.nan, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                ri, rj = all_recipe_ids[i], all_recipe_ids[j]
                pi = proj_cache.get(ri)
                pj = proj_cache.get(rj)
                if pi is None or pj is None:
                    corr[i, j] = 0.0
                    corr[j, i] = 0.0
                    continue
                mask = pi[1] & pj[1] & context.active_mask_2d
                a = pi[0][mask]
                b = pj[0][mask]
                c_val = 0.0 if len(a) < 2 else float(np.corrcoef(a, b)[0, 1])
                cv = c_val if np.isfinite(c_val) else 0.0
                corr[i, j] = cv
                corr[j, i] = cv
            corr[i, i] = 1.0

    all_selected: list[L0SignalCandidate] = []
    for candidates in selected_by_tf.values():
        all_selected.extend(candidates)

    # Build pair evidence for i<j only
    n = len(all_recipe_ids)
    pair_evidence_list: list[CrossTFPairEvidence] = []

    if precomputed_shared_context is not None and precomputed_shared_context.entry_pos_flat:
        # [OPT-2] Batch jaccard via matmul
        pos_flat_stack = np.array(
            [precomputed_shared_context.entry_pos_flat.get(rid, np.array([], dtype=np.int8))
             for rid in all_recipe_ids],
            dtype=np.int16,
        )
        neg_flat_stack = np.array(
            [precomputed_shared_context.entry_neg_flat.get(rid, np.array([], dtype=np.int8))
             for rid in all_recipe_ids],
            dtype=np.int16,
        )
        entry_flat = pos_flat_stack | neg_flat_stack
        dir_shared = pos_flat_stack @ pos_flat_stack.T + neg_flat_stack @ neg_flat_stack.T
        non_dir_shared = entry_flat @ entry_flat.T
        ent_vec = np.array(
            [precomputed_shared_context.n_entries.get(rid, 0) for rid in all_recipe_ids],
            dtype=np.int64,
        )
        for i in range(n):
            panel_i = panel_by_recipe_id.get(all_recipe_ids[i])
            if panel_i is None:
                continue
            for j in range(i + 1, n):
                panel_j = panel_by_recipe_id.get(all_recipe_ids[j])
                if panel_j is None:
                    continue
                shared_count = int(dir_shared[i, j])
                total_shared_non_dir = int(non_dir_shared[i, j])
                union_count = int(ent_vec[i]) + int(ent_vec[j]) - total_shared_non_dir
                j_dir = shared_count / max(union_count, 1)
                score_corr_val = float(corr[i, j]) if np.isfinite(corr[i, j]) else 0.0
                is_red = (
                    score_corr_val >= max_novelty_corr
                    and shared_count >= min_shared_directional_entries
                    and j_dir >= min_directional_entry_jaccard
                )
                pair_ev = CrossTFPairEvidence(
                    recipe_id_a=all_recipe_ids[i], recipe_id_b=all_recipe_ids[j],
                    score_corr=score_corr_val,
                    shared_directional_entries=shared_count,
                    directional_entry_jaccard=float(j_dir),
                    is_redundant=is_red,
                )
                pair_evidence_list.append(pair_ev)
    else:
        # Fallback per-pair path (precomputed_shared_context unavailable)
        for i in range(n):
            for j in range(i + 1, n):
                ri, rj = all_recipe_ids[i], all_recipe_ids[j]
                panel_i = panel_by_recipe_id.get(ri)
                panel_j = panel_by_recipe_id.get(rj)
                if panel_i is None or panel_j is None:
                    continue
                pair_ev = compute_cross_tf_pair_evidence(
                    recipe_id_a=ri, recipe_id_b=rj,
                    panel_a=panel_i, panel_b=panel_j,
                    context=context,
                    min_score_corr=max_novelty_corr,
                    min_directional_entry_jaccard=min_directional_entry_jaccard,
                    min_shared_directional_entries=min_shared_directional_entries,
                    precomputed_proj_a=proj_cache.get(ri),
                    precomputed_proj_b=proj_cache.get(rj),
                    precomputed_side_entry_a=side_entry_cache.get(ri),
                    precomputed_side_entry_b=side_entry_cache.get(rj),
                    precomputed_score_corr=float(corr[i, j]) if np.isfinite(corr[i, j]) else None,
                )
                pair_evidence_list.append(pair_ev)

    # [OPT-3] Leader-based demotion via dict O(1) lookup
    pair_map: dict[tuple[str, str], CrossTFPairEvidence] = {}
    for p_ev in pair_evidence_list:
        pair_map[(p_ev.recipe_id_a, p_ev.recipe_id_b)] = p_ev
        pair_map[(p_ev.recipe_id_b, p_ev.recipe_id_a)] = p_ev

    ranked = sorted(
        enumerate(all_selected),
        key=lambda x: (-x[1].l1_priority_score, x[1].recipe_id),
    )
    retained: list[str] = []
    demoted: list[str] = []
    demoted_reason_map: dict[str, str] = {}

    for _, candidate in ranked:
        rid = candidate.recipe_id
        is_demoted = False
        for leader_rid in retained:
            found_ev = pair_map.get((leader_rid, rid))
            if found_ev is not None and found_ev.is_redundant:
                demoted.append(rid)
                demoted_reason_map[rid] = leader_rid
                is_demoted = True
                break
        if not is_demoted:
            retained.append(rid)

    final_selected = tuple(retained)
    global_eff = estimate_effective_test_count(corr)

    return CrossBucketDiversityResult(
        final_selected_recipe_ids=final_selected,
        demoted_recipe_ids=tuple(demoted),
        demoted_reason_by_id=demoted_reason_map,
        cross_bucket_corr=corr,
        global_eff_test_count=global_eff,
        pair_evidence=tuple(pair_evidence_list),
        canonical_tf=context.canonical_tf,
        common_start_ns=context.common_start_ns,
        common_end_ns=context.common_end_ns,
        n_common_active_bars=context.n_common_active_bars,
    )


def audit_l0_selected_recipe_independence(
    *,
    selected_by_tf: Mapping[str, Sequence[L0SignalCandidate]],
    panel_by_recipe_id: Mapping[str, CandidateSignalPanel],
    aligned_by_tf: Mapping[str, AlignedMarketData],
    min_common_active_bars: int,
    max_corr: float = 0.70,
    precomputed_shared_context: CrossTFSharedContext | None = None,  # [NEW, additive]
) -> L0IndependenceAudit:
    all_selected: list[L0SignalCandidate] = []
    for candidates in selected_by_tf.values():
        all_selected.extend(candidates)

    n_selected = len(all_selected)
    families = [c.family for c in all_selected]
    n_thesis = len({resolve_economic_thesis_id(f) for f in families})

    if n_selected <= 1:
        return L0IndependenceAudit(
            n_selected_total=n_selected,
            n_distinct_thesis_ids=n_thesis,
            n_independent_clusters=n_selected,
            cluster_members={i: (c.recipe_id,) for i, c in enumerate(all_selected)} if all_selected else {},
            demoted_recipe_ids=(),
            demoted_reason_by_id={},
            canonical_tf="",
            max_corr_threshold=max_corr,
        )

    all_recipe_ids = [c.recipe_id for c in all_selected]

    if precomputed_shared_context is not None:
        context = precomputed_shared_context.canonical_context
        corr = precomputed_shared_context.corr
        panels = [panel_by_recipe_id[rid] for rid in all_recipe_ids if rid in panel_by_recipe_id]
    else:
        context = resolve_cross_tf_canonical_context(
            selected_by_tf=selected_by_tf,
            panel_by_recipe_id=panel_by_recipe_id,
            aligned_by_tf=aligned_by_tf,
            min_common_active_bars=min_common_active_bars,
        )

        panels = [panel_by_recipe_id[rid] for rid in all_recipe_ids if rid in panel_by_recipe_id]
        if len(panels) < 2:
            return L0IndependenceAudit(
                n_selected_total=n_selected,
                n_distinct_thesis_ids=n_thesis,
                n_independent_clusters=n_selected,
                cluster_members={i: (c.recipe_id,) for i, c in enumerate(all_selected)},
                demoted_recipe_ids=(),
                demoted_reason_by_id={},
                canonical_tf=context.canonical_tf,
                max_corr_threshold=max_corr,
            )

        proj_scores: list[NDArray[np.float32]] = []
        proj_valid: list[NDArray[np.bool_]] = []
        for p in panels:
            ps, pv = project_signal_to_canonical_grid(
                panel=p,
                canonical_datetimes=context.canonical_datetimes_ns,
                causal_lag_bars=1,
            )
            proj_scores.append(ps)
            proj_valid.append(pv)

        n = len(panels)
        corr = np.full((n, n), np.nan, dtype=np.float64)
        active = context.active_mask_2d
        for i in range(n):
            for j in range(i + 1, n):
                mask = proj_valid[i] & proj_valid[j] & active
                a = proj_scores[i][mask]
                b = proj_scores[j][mask]
                c_val = 0.0 if len(a) < 2 else float(np.corrcoef(a, b)[0, 1])
                cv = c_val if np.isfinite(c_val) else 0.0
                corr[i, j] = cv
                corr[j, i] = cv
            corr[i, i] = 1.0

    clusters = cluster_correlated_recipes(
        evidences=all_selected,
        corr=corr,
        max_corr=max_corr,
    )

    cluster_members: dict[int, tuple[str, ...]] = {}
    demoted: list[str] = []
    demoted_reason: dict[str, str] = {}
    for cluster in clusters:
        if len(cluster) == 1:
            cluster_members[len(cluster_members)] = cluster
        else:
            best_rid = max(cluster, key=lambda rid: _find_priority(rid, all_selected, all_recipe_ids))
            cluster_members[len(cluster_members)] = cluster
            for rid in cluster:
                if rid != best_rid:
                    demoted.append(rid)
                    demoted_reason[rid] = best_rid

    return L0IndependenceAudit(
        n_selected_total=n_selected,
        n_distinct_thesis_ids=n_thesis,
        n_independent_clusters=len(clusters),
        cluster_members=cluster_members,
        demoted_recipe_ids=tuple(demoted),
        demoted_reason_by_id=demoted_reason,
        canonical_tf=context.canonical_tf,
        max_corr_threshold=max_corr,
    )


def apply_cross_tf_survival_floor(
    *,
    cross_tf_result: CrossBucketDiversityResult,
    candidate_by_recipe_id: Mapping[str, L0SignalCandidate],
    min_survivors_per_archetype: int = 1,
    min_survivors_per_tf: int = 1,
) -> CrossBucketDiversityResult:
    """Post-process compute_cross_tf_redundancy's output so no archetype or
    TF is reduced to zero survivors purely by cross-TF demotion. [LIMIT-03]

    For each archetype (via L0SignalCandidate.archetype) / timeframe (via
    L0SignalCandidate.timeframe) with zero representatives in
    final_selected_recipe_ids, re-admits the single highest
    l1_priority_score candidate from demoted_recipe_ids belonging to that
    archetype/timeframe, removing it from demoted_recipe_ids and
    demoted_reason_by_id. Re-admissions from both rules are unioned and
    applied once (idempotent — a candidate is re-admitted at most once
    even if it satisfies both an archetype and a TF floor). [LIMIT-04]
    candidates absent from candidate_by_recipe_id are skipped (never
    re-admitted, never counted toward a floor) rather than raising, since
    this is a defensive floor, not a primary correctness gate.
    """
    final_set = set(cross_tf_result.final_selected_recipe_ids)
    demoted_set = set(cross_tf_result.demoted_recipe_ids)

    if not demoted_set:
        return cross_tf_result

    from collections import Counter

    selected_archetype_counts: Counter[str] = Counter()
    selected_tf_counts: Counter[str] = Counter()
    for rid in final_set:
        c = candidate_by_recipe_id.get(rid)
        if c is not None:
            selected_archetype_counts[c.archetype] += 1
            selected_tf_counts[c.timeframe] += 1

    demoted_by_archetype: dict[str, list[tuple[str, float]]] = {}
    demoted_by_tf: dict[str, list[tuple[str, float]]] = {}
    for rid in demoted_set:
        c = candidate_by_recipe_id.get(rid)
        if c is None:
            continue
        demoted_by_archetype.setdefault(c.archetype, []).append((rid, c.l1_priority_score))
        demoted_by_tf.setdefault(c.timeframe, []).append((rid, c.l1_priority_score))

    to_readmit: set[str] = set()

    for archetype, candidates in demoted_by_archetype.items():
        survivors = selected_archetype_counts.get(archetype, 0)
        if survivors < min_survivors_per_archetype and candidates:
            best = max(candidates, key=lambda x: x[1])
            to_readmit.add(best[0])

    for tf, candidates in demoted_by_tf.items():
        survivors = selected_tf_counts.get(tf, 0)
        if survivors < min_survivors_per_tf and candidates:
            best = max(candidates, key=lambda x: x[1])
            to_readmit.add(best[0])

    if not to_readmit:
        return cross_tf_result

    new_selected = list(cross_tf_result.final_selected_recipe_ids)
    new_demoted: list[str] = []
    new_demoted_reason: dict[str, str] = {}
    for rid in cross_tf_result.demoted_recipe_ids:
        if rid in to_readmit:
            new_selected.append(rid)
        else:
            new_demoted.append(rid)
            if rid in cross_tf_result.demoted_reason_by_id:
                new_demoted_reason[rid] = cross_tf_result.demoted_reason_by_id[rid]

    return replace(
        cross_tf_result,
        final_selected_recipe_ids=tuple(new_selected),
        demoted_recipe_ids=tuple(new_demoted),
        demoted_reason_by_id=new_demoted_reason,
    )


# Backward-compat alias
select_bucket_diverse_recipes = select_bucket_diverse_candidates
