"""Alpha Foundry L0 cheap gate and survivor filtering.

[ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]
[ADR_20260706_ALPHA_FOUNDRY_SYNC][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY]
[ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from src.domain.futures.alpha_foundry.contracts import (
    AlphaRecipe,
    CheapGateConfig,
    CheapGateEvidence,
    CheapGateRejectReason,
    DiscoveryTier,
    FamilyTimeframeGatePolicy,
    L0HardRejectReason,
    L0PriorityWeights,
    L0SignalCandidate,
    L0SoftFlag,
    MultiTimeframeEvidence,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def _validate_shape(panel: CandidateSignalPanel, aligned: AlignedMarketData) -> None:
    t, n = aligned.close_2d.shape
    checks: list[tuple[str, NDArray[np.float64 | np.bool_ | np.int8]]] = [
        ("signed_score_2d", panel.signed_score_2d),
        ("side_hint_2d", panel.side_hint_2d),
        ("valid_mask_2d", panel.valid_mask_2d),
        ("turnover_proxy_2d", panel.turnover_proxy_2d),
    ]
    for name, arr in checks:
        if arr.shape != (t, n):
            raise ValueError(f"panel.{name} shape {arr.shape} != aligned close_2d shape ({t}, {n})")


def _compute_block_means(values: NDArray[np.float64], block_bars: int) -> NDArray[np.float64]:
    n = len(values)
    if n < 2 or block_bars < 1:
        return np.array([])
    n_blocks = max(1, n // block_bars)
    return np.array([values[i * block_bars : (i + 1) * block_bars].mean() for i in range(n_blocks)])


def _block_moments(block_means: NDArray[np.float64]) -> tuple[float, float]:
    if len(block_means) < 2:
        return (0.0, 0.0)
    mu = float(np.nanmean(block_means))
    n_valid = float(np.sum(~np.isnan(block_means)))
    se = float(np.nanstd(block_means, ddof=1)) / max(np.sqrt(max(n_valid, 1.0)), 1.0)
    return (mu, se)


def _bootstrap_block_ci(
    block_means: NDArray[np.float64],
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if len(block_means) < 2:
        return (0.0, 0.5)
    indices = rng.integers(0, len(block_means), size=(n_resamples, len(block_means)))
    resampled_means = block_means[indices].mean(axis=1)
    lcb = float(np.percentile(resampled_means, 5))
    p_positive = float(np.mean(resampled_means > 0))
    return (lcb, p_positive)


def _sparse_entry_mask(
    side: NDArray[np.int8],
    valid: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Independent-trade entry mask: True where a bar starts a new position —
    either from flat (side==0) or a direct reversal (sign flip without going
    flat). A continuation bar (same side as t-1) is never an entry.
    """
    side_prev = np.vstack([np.zeros((1, side.shape[1]), dtype=side.dtype), side[:-1, :]])
    entry: NDArray[np.bool_] = (side != 0) & (side != side_prev) & valid
    return entry


def _compute_turnover_per_year(side: NDArray[np.int8], valid_mask: NDArray[np.bool_], bars_per_year: float) -> float:
    diff = np.abs(np.diff(side.astype(np.float64), axis=0))
    valid_slice = valid_mask[1:, :] & valid_mask[:-1, :]
    denom = max(np.sum(valid_slice), 1)
    return float(np.sum(diff * valid_slice) / denom * bars_per_year / 2.0)


def _compute_rank_ic(fwd_ret: NDArray[np.float64], score: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
    flat_ret = fwd_ret[mask]
    flat_score = score[mask]
    if len(flat_ret) < 3:
        return 0.0
    if np.std(flat_ret) < 1e-12 or np.std(flat_score) < 1e-12:
        return 0.0
    return float(scipy_stats.spearmanr(flat_ret, flat_score)[0])


def evaluate_panel_cheap_gate(
    *,
    panel: CandidateSignalPanel,
    aligned: AlignedMarketData,
    recipe: AlphaRecipe,
    cost_model: ExecutionCostModel,
    config: CheapGateConfig,
    bars_per_year: float,
) -> CheapGateEvidence:
    if bars_per_year <= 0.0:
        raise ValueError("bars_per_year must be positive")
    _validate_shape(panel, aligned)

    t, n = aligned.close_2d.shape
    close = aligned.close_2d
    funding = aligned.funding_2d
    active = aligned.active_mask & aligned.warm_mask & ~aligned.entry_block_mask & ~aligned.kill_mask

    side = panel.side_hint_2d
    valid = panel.valid_mask_2d & active & np.isfinite(close)

    causal_lag = recipe.causal_lag_bars
    holding_bars = panel.expected_holding_bars

    if causal_lag >= t or holding_bars >= t:
        return CheapGateEvidence(
            recipe_id=recipe.recipe_id,
            timeframe=recipe.timeframe,
            symbol_scope="symbol",
            n_events=0,
            effective_n=0.0,
            mean_net_bps=0.0,
            nw_tstat=0.0,
            block_lcb_bps=0.0,
            rank_ic=0.0,
            cost_drag_ratio=0.0,
            turnover_per_year=0.0,
            novelty_corr_max=0.0,
            incremental_rank_ic=0.0,
            compute_cost_score=0.0,
            bootstrap_lcb_bps=0.0,
            bootstrap_agree=True,
            gate_passed=False,
            reject_reasons=("insufficient_events",),
        )

    idx_start = causal_lag
    idx_end = t - holding_bars

    entry_full = _sparse_entry_mask(side, valid)
    event_mask = np.zeros_like(entry_full)
    if idx_start < idx_end:
        event_mask[idx_start:idx_end, :] = entry_full[idx_start:idx_end, :]

    n_events = int(np.sum(event_mask))
    if n_events < config.min_events:
        return CheapGateEvidence(
            recipe_id=recipe.recipe_id,
            timeframe=recipe.timeframe,
            symbol_scope="symbol",
            n_events=n_events,
            effective_n=0.0,
            mean_net_bps=0.0,
            nw_tstat=0.0,
            block_lcb_bps=0.0,
            rank_ic=0.0,
            cost_drag_ratio=0.0,
            turnover_per_year=0.0,
            novelty_corr_max=0.0,
            incremental_rank_ic=0.0,
            compute_cost_score=0.0,
            bootstrap_lcb_bps=0.0,
            bootstrap_agree=True,
            gate_passed=False,
            reject_reasons=("insufficient_events",),
        )

    fwd_ret_bps = np.full((t, n), np.nan, dtype=np.float64)
    for i in range(t - holding_bars):
        fwd_ret_bps[i, :] = side[i, :].astype(np.float64) * np.log(close[i + holding_bars, :] / close[i, :]) * 10000.0

    stress_cost = cost_model.stress_round_trip_bps()
    funding_cost = np.where(event_mask, funding * 10000.0 * holding_bars, 0.0)
    net_bps = fwd_ret_bps - stress_cost - funding_cost

    net_vals = net_bps[event_mask]
    gross_vals = fwd_ret_bps[event_mask]

    reject_reasons_list: list[CheapGateRejectReason] = []

    effective_n = float(n_events)
    if effective_n < config.min_effective_n:
        reject_reasons_list.append("insufficient_effective_n")

    mean_net_bps = float(np.nanmean(net_vals)) if n_events > 0 else 0.0

    block_bars_eff = max(config.block_bars, 2 * holding_bars)
    block_means = _compute_block_means(net_vals, block_bars_eff)
    mu_block, se_block = _block_moments(block_means)
    nw_tstat = mu_block / max(se_block, 1e-10)
    block_lcb_bps = mu_block - 1.0 * se_block

    if block_lcb_bps <= config.min_lcb_net_bps:
        reject_reasons_list.append("non_positive_lcb")
    if abs(nw_tstat) < config.min_nw_tstat:
        reject_reasons_list.append("weak_tstat")

    rank_ic = _compute_rank_ic(fwd_ret_bps, panel.signed_score_2d, event_mask)

    # cost drag
    total_gross = float(np.nansum(gross_vals)) if n_events > 0 else 0.0
    total_net = float(np.nansum(net_vals)) if n_events > 0 else 0.0
    total_cost = total_gross - total_net
    eps = 1e-10
    cost_drag_ratio = total_cost / max(abs(total_gross), eps)
    if cost_drag_ratio > config.max_cost_drag_ratio:
        reject_reasons_list.append("excess_cost_drag")

    # turnover
    turnover = _compute_turnover_per_year(side, valid, bars_per_year)
    max_turn = min(config.max_turnover_per_year, recipe.max_turnover_per_year)
    if turnover > max_turn:
        reject_reasons_list.append("excess_turnover")

    # bootstrap
    rng = np.random.default_rng(config.bootstrap_seed)
    bootstrap_lcb_bps, _ = _bootstrap_block_ci(block_means, config.bootstrap_samples, rng)
    bootstrap_agree = (bootstrap_lcb_bps > 0) == (block_lcb_bps > 0)

    gate_passed = len(reject_reasons_list) == 0

    return CheapGateEvidence(
        recipe_id=recipe.recipe_id,
        timeframe=recipe.timeframe,
        symbol_scope="symbol",
        n_events=n_events,
        effective_n=effective_n,
        mean_net_bps=mean_net_bps,
        nw_tstat=nw_tstat,
        block_lcb_bps=block_lcb_bps,
        rank_ic=rank_ic,
        cost_drag_ratio=cost_drag_ratio,
        turnover_per_year=turnover,
        novelty_corr_max=0.0,
        incremental_rank_ic=0.0,
        compute_cost_score=0.0,
        bootstrap_lcb_bps=bootstrap_lcb_bps,
        bootstrap_agree=bootstrap_agree,
        gate_passed=gate_passed,
        reject_reasons=tuple(reject_reasons_list),
    )


def evaluate_alpha_cheap_gate_batch(
    *,
    panels: Sequence[CandidateSignalPanel],
    recipes: Mapping[str, AlphaRecipe],
    aligned: AlignedMarketData,
    cost_model: ExecutionCostModel,
    config: CheapGateConfig,
) -> tuple[CheapGateEvidence, ...]:
    from src.domain.futures.optimization.metrics import _bars_per_year_for_tf

    bpy_cache: dict[str, float] = {}
    results: list[CheapGateEvidence] = []
    for panel in panels:
        recipe_id = panel.metadata.get("recipe_id", "")
        recipe = recipes.get(recipe_id)
        if recipe is None:
            continue
        tf = recipe.timeframe
        if tf not in bpy_cache:
            bpy_cache[tf] = _bars_per_year_for_tf(tf)
        evidence = evaluate_panel_cheap_gate(
            panel=panel,
            aligned=aligned,
            recipe=recipe,
            cost_model=cost_model,
            config=config,
            bars_per_year=bpy_cache[tf],
        )
        results.append(evidence)
    return tuple(results)


def resolve_family_timeframe_gate_policy(
    *,
    recipe: AlphaRecipe,
    config: CheapGateConfig,
) -> FamilyTimeframeGatePolicy:
    archetype = recipe.archetype
    family = recipe.family

    min_events = config.archetype_event_floors.get(archetype, config.min_events)
    if family in config.family_event_floors:
        min_events = config.family_event_floors[family]

    min_effective_n = config.min_effective_n
    target_effective_n = float(min_events)
    max_cost_drag_ratio = config.max_cost_drag_ratio
    max_turnover_per_year = min(config.max_turnover_per_year, recipe.max_turnover_per_year)

    deep_negative_lcb_bps = config.min_lcb_net_bps

    return FamilyTimeframeGatePolicy(
        archetype=archetype,
        min_events=min_events,
        min_effective_n=min_effective_n,
        target_effective_n=target_effective_n,
        max_cost_drag_ratio=max_cost_drag_ratio,
        max_turnover_per_year=max_turnover_per_year,
        deep_negative_lcb_bps=deep_negative_lcb_bps,
    )


def _compute_l1_priority_score(
    *,
    evidence: CheapGateEvidence,
    tf_fusion: MultiTimeframeEvidence | None,
    max_abs_corr_in_bucket: float,
    priority_weights: L0PriorityWeights,
) -> float:
    pw = priority_weights

    base = evidence.mean_net_bps * pw.edge_mean_weight + evidence.block_lcb_bps * (1.0 - pw.edge_mean_weight)

    if tf_fusion is not None:
        tier = tf_fusion.corroboration_tier
        if tier == "corroborated":
            mult = pw.corroborated_multiplier
        elif tier == "contradicted":
            mult = pw.contradicted_multiplier
        elif tier == "single_tf_strict":
            mult = pw.single_tf_multiplier
        else:
            mult = pw.insufficient_coverage_multiplier
    else:
        mult = 1.0

    priority = base * mult

    if max_abs_corr_in_bucket > pw.corr_soft_floor:
        priority *= pw.insufficient_coverage_multiplier

    return priority


def build_l0_signal_candidate(
    *,
    run_id: str,
    evidence: CheapGateEvidence,
    recipe: AlphaRecipe,
    source: Literal["catalog_exact", "catalog_family_variant", "synthetic_recipe"],
    policy: FamilyTimeframeGatePolicy,
    stress_cost_bps: float,
    tf_fusion: MultiTimeframeEvidence | None,
    min_conviction_lcb_bps: float = 5.0,
    max_abs_corr_in_bucket: float = 0.0,
) -> L0SignalCandidate:
    hard_reject_reasons: list[L0HardRejectReason] = []
    soft_flags: list[L0SoftFlag] = []

    if evidence.n_events < policy.min_events:
        hard_reject_reasons.append("insufficient_events")
    if evidence.effective_n < policy.min_effective_n:
        hard_reject_reasons.append("insufficient_effective_n")
    if evidence.cost_drag_ratio > policy.max_cost_drag_ratio:
        hard_reject_reasons.append("excess_cost_drag")
    if evidence.turnover_per_year > policy.max_turnover_per_year:
        hard_reject_reasons.append("excess_turnover")

    if "invalid_shape" in evidence.reject_reasons:
        hard_reject_reasons.append("invalid_shape")
    if "lookahead_risk" in evidence.reject_reasons:
        hard_reject_reasons.append("lookahead_risk")
    if "missing_required_field" in evidence.reject_reasons:
        hard_reject_reasons.append("missing_required_field")

    if evidence.block_lcb_bps < policy.deep_negative_lcb_bps:
        hard_reject_reasons.append("deep_negative_lcb")

    if tf_fusion is not None and tf_fusion.corroboration_tier == "contradicted":
        hard_reject_reasons.append("tf_contradicted")

    if "weak_tstat" in evidence.reject_reasons:
        soft_flags.append("weak_tstat")
    if not evidence.bootstrap_agree:
        soft_flags.append("bootstrap_disagree")

    if 0.0 <= evidence.block_lcb_bps < min_conviction_lcb_bps:
        soft_flags.append("below_conviction_floor")

    discovery_tier: DiscoveryTier
    if hard_reject_reasons:
        discovery_tier = "blocked"
    elif soft_flags:
        discovery_tier = "seed"
    else:
        discovery_tier = "candidate"

    l1_priority_score = _compute_l1_priority_score(
        evidence=evidence,
        tf_fusion=tf_fusion,
        max_abs_corr_in_bucket=max_abs_corr_in_bucket,
        priority_weights=L0PriorityWeights(),
    )

    corroboration_tier: Literal["corroborated", "single_tf_strict", "contradicted", "insufficient_coverage"]
    tf_coverage_count = 0
    sign_agreement_ratio = 0.0
    if tf_fusion is not None:
        corroboration_tier = tf_fusion.corroboration_tier
        tf_coverage_count = tf_fusion.tf_coverage_count
        sign_agreement_ratio = tf_fusion.sign_agreement_ratio
    else:
        corroboration_tier = "insufficient_coverage"

    return L0SignalCandidate(
        run_id=run_id,
        timeframe=recipe.timeframe,
        family=recipe.family,
        variant=recipe.variant,
        recipe_id=recipe.recipe_id,
        archetype=recipe.archetype,
        source=source,
        n_events=evidence.n_events,
        effective_n=evidence.effective_n,
        mean_net_bps=evidence.mean_net_bps,
        block_lcb_bps=evidence.block_lcb_bps,
        nw_tstat=evidence.nw_tstat,
        bootstrap_lcb_bps=evidence.bootstrap_lcb_bps,
        bootstrap_agree=evidence.bootstrap_agree,
        cost_drag_ratio=evidence.cost_drag_ratio,
        turnover_per_year=evidence.turnover_per_year,
        max_abs_corr_in_bucket=max_abs_corr_in_bucket,
        tf_coverage_count=tf_coverage_count,
        sign_agreement_ratio=sign_agreement_ratio,
        corroboration_tier=corroboration_tier,
        discovery_tier=discovery_tier,
        l1_priority_score=l1_priority_score,
        l1_budget_units=0,
        hard_reject_reasons=tuple(hard_reject_reasons),
        soft_flags=tuple(soft_flags),
    )
