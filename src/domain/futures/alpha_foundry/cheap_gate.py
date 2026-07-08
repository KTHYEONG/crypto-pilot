"""Alpha Foundry L0 cheap gate and survivor filtering.

[ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]
[ADR_20260706_ALPHA_FOUNDRY_SYNC][ADR_20260706_ALPHA_FOUNDRY_L0_DIVERSITY]
[ADR_20260707_ALPHA_FOUNDRY_RESULT_SYNC]
[ADR_20260707_ALPHA_FOUNDRY_CANONICAL_GATE_WIRING]
[ADR_20260707_L0_MULTI_TF_GATE_REDESIGN]
[ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from src.domain.futures.alpha_foundry.contracts import (
    AlphaGateConfig,
    AlphaGateEvidence,
    AlphaGateHandoffTier,
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
    values = values[np.isfinite(values)]
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
    """[ADR_20260708_L0_SIGNAL_YIELD_IMPROVEMENT] n_events floor resolved via
    resolve_family_timeframe_gate_policy (family/archetype-aware), not flat config.min_events."""
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
            mean_gross_bps=0.0,
            mean_cost_bps=0.0,
        )

    idx_start = causal_lag
    idx_end = t - holding_bars

    entry_full = _sparse_entry_mask(side, valid)
    event_mask = np.zeros_like(entry_full)
    if idx_start < idx_end:
        event_mask[idx_start:idx_end, :] = entry_full[idx_start:idx_end, :]

    n_events = int(np.sum(event_mask))
    resolved_min_events = resolve_family_timeframe_gate_policy(recipe=recipe, config=config).min_events
    if n_events < resolved_min_events:
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
            mean_gross_bps=0.0,
            mean_cost_bps=0.0,
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

    mean_gross_bps = total_gross / n_events if n_events > 0 else 0.0
    mean_cost_bps = total_cost / n_events if n_events > 0 else 0.0
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
        mean_gross_bps=mean_gross_bps,
        mean_cost_bps=mean_cost_bps,
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

def _rank_ic_soft_floor(n_events: int) -> float:
    """Approx. standard error of a Spearman rank correlation (Fisher-z style).

    [ADR_20260707_L0_ALPHA_EFFECTIVENESS_REDESIGN]
    """
    return 1.0 / math.sqrt(max(int(n_events) - 3, 1))


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
    weak_rank_ic: bool,
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
    if weak_rank_ic:
        priority *= pw.weak_rank_ic_multiplier

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

    if abs(evidence.rank_ic) < _rank_ic_soft_floor(evidence.n_events):
        soft_flags.append("weak_rank_ic")

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
        weak_rank_ic=("weak_rank_ic" in soft_flags),
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



# ── Cost-aware / Regime / TF Corroboration helpers ─────────────────────


def compute_capacity_score(
    *,
    aligned: AlignedMarketData,
    event_mask: NDArray[np.bool_],
    liquidity_cost_stress_bps: float,
) -> float:
    if aligned.execution_cost_bps_2d is None or aligned.adv_usdt_2d is None:
        return min(liquidity_cost_stress_bps / max(25.0, 1e-10), 0.25)
    cost_at_events = aligned.execution_cost_bps_2d[event_mask]
    adv_at_events = aligned.adv_usdt_2d[event_mask]
    if (
        len(cost_at_events) == 0
        or len(adv_at_events) == 0
        or not np.any(np.isfinite(cost_at_events))
        or not np.any(np.isfinite(adv_at_events))
    ):
        return 0.0
    mean_cost = float(np.nanmean(cost_at_events))
    mean_adv = float(np.nanmean(adv_at_events))
    capacity = 1.0 - (mean_cost / max(mean_adv * 1e-6 + 1e-10, 1e-10))
    return float(np.clip(capacity, 0.0, 1.0))


def compute_regime_stability(
    *,
    panel: CandidateSignalPanel,
    net_bps: NDArray[np.float64],
    event_mask: NDArray[np.bool_],
) -> float:
    net_at_events = net_bps[event_mask]
    net_at_events = net_at_events[np.isfinite(net_at_events)]
    if len(net_at_events) < 4:
        return 0.0
    splits = np.array_split(net_at_events, max(4, len(net_at_events) // 4))
    split_means = np.array([float(np.mean(s)) for s in splits if len(s) > 0])
    if len(split_means) < 2:
        return 0.0
    cv = float(np.std(split_means)) / max(abs(float(np.mean(split_means))), 1e-10)
    stability = 1.0 / (1.0 + cv)
    return float(np.clip(stability, 0.0, 1.0))


def compute_tf_corroboration(
    *,
    recipe: AlphaRecipe,
    tf_fusion: MultiTimeframeEvidence | None,
) -> float:
    if tf_fusion is None:
        return 0.0
    if tf_fusion.corroboration_tier == "contradicted":
        return 0.0
    base = 0.5 if tf_fusion.corroboration_tier == "single_tf_strict" else 1.0
    return float(np.clip(base * tf_fusion.sign_agreement_ratio, 0.0, 1.0))


# ── V2 Gate Metrics ────────────────────────────────────────────────────


def compute_cost_drag_ratio_v2(*, mean_cost_bps: float, mean_gross_bps: float, eps: float = 1e-10) -> float:
    return mean_cost_bps / max(abs(mean_gross_bps), eps)


def compute_rank_ic_with_tstat(
    *,
    fwd_ret_bps: NDArray[np.float64],
    score: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> tuple[float, float]:
    flat_ret = fwd_ret_bps[mask]
    flat_score = score[mask]
    if len(flat_ret) < 3:
        return (0.0, 0.0)
    if np.std(flat_ret) < 1e-12 or np.std(flat_score) < 1e-12:
        return (0.0, 0.0)
    ic = float(scipy_stats.spearmanr(flat_ret, flat_score)[0])
    n = len(flat_ret)
    z = float(np.arctanh(ic))
    se = 1.0 / np.sqrt(max(n - 3, 1))
    tstat = z / max(se, 1e-10)
    return (ic, tstat)


def compute_payoff_stats(values_bps: NDArray[np.float64]) -> tuple[float, float]:
    finite = values_bps[np.isfinite(values_bps)]
    if len(finite) == 0:
        return (0.0, 0.0)
    hit_rate = float(np.mean(finite > 0.0))
    pos = finite[finite > 0.0]
    neg = finite[finite < 0.0]
    mean_pos = float(np.mean(pos)) if len(pos) > 0 else 0.0
    mean_neg = float(np.mean(neg)) if len(neg) > 0 else 0.0
    payoff_skew = mean_pos / max(abs(mean_neg), 1e-10)
    return (hit_rate, payoff_skew)


def compute_xs_spread_lcb_bps(
    *,
    net_bps: NDArray[np.float64],
    score: NDArray[np.float64],
    event_mask: NDArray[np.bool_],
    min_symbols_per_bar: int,
    quantile: float = 0.20,
) -> float | None:
    t = min(net_bps.shape[0], score.shape[0])
    if t == 0:
        return None
    spreads: list[float] = []
    for bar in range(t):
        bar_mask = event_mask[bar, :].astype(bool)
        n_active = int(bar_mask.sum())
        if n_active < min_symbols_per_bar:
            continue
        bar_net = net_bps[bar, bar_mask]
        bar_score = score[bar, bar_mask]
        order = np.argsort(bar_score)
        top_idx = order[-max(1, int(n_active * quantile)):]
        bot_idx = order[: max(1, int(n_active * quantile))]
        top_mean = float(np.mean(bar_net[top_idx])) if len(top_idx) > 0 else 0.0
        bot_mean = float(np.mean(bar_net[bot_idx])) if len(bot_idx) > 0 else 0.0
        spreads.append(top_mean - bot_mean)
    if len(spreads) < 2:
        return None
    arr = np.array(spreads, dtype=np.float64)
    return float(np.percentile(arr, 5))


def compute_liquidity_cost_stress_bps(
    *,
    aligned: AlignedMarketData,
    event_mask: NDArray[np.bool_],
    stress_mult: float,
) -> float:
    if aligned.execution_cost_bps_2d is None:
        return 0.0
    cost_at_events = aligned.execution_cost_bps_2d[event_mask]
    if len(cost_at_events) == 0 or not np.any(np.isfinite(cost_at_events)):
        return 0.0
    return float(np.nanmean(cost_at_events)) * stress_mult


def evaluate_panel_gate_v2(
    *,
    panel: CandidateSignalPanel,
    aligned: AlignedMarketData,
    recipe: AlphaRecipe,
    cost_model: ExecutionCostModel,
    config: CheapGateConfig,
    bars_per_year: float,
) -> AlphaGateEvidence:
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

    reject_reasons_list: list[str] = []
    soft_flags_list: list[str] = []

    if causal_lag >= t or holding_bars >= t:
        n_events = 0
        effective_n = 0.0
        return AlphaGateEvidence(
            schema_version="unified",
            run_id="",
            timeframe=recipe.timeframe,
            family=recipe.family,
            variant=recipe.variant,
            recipe_id=recipe.recipe_id,
            archetype=recipe.archetype,
            symbol_scope="symbol",
            n_events=0,
            effective_n=0.0,
            mean_gross_bps=0.0,
            mean_cost_bps=0.0,
            mean_net_bps=0.0,
            gross_lcb_bps=0.0,
            net_lcb_bps=0.0,
            nw_tstat=0.0,
            rank_ic=0.0,
            rank_ic_tstat=0.0,
            cost_drag_ratio=0.0,
            turnover_per_year=0.0,
            novelty_corr_max=0.0,
            incremental_rank_ic=0.0,
            compute_cost_score=0.0,
            event_hit_rate=0.0,
            payoff_skew=0.0,
            xs_spread_lcb_bps=None,
            liquidity_cost_stress_bps=0.0,
            bootstrap_lcb_bps=0.0,
            bootstrap_agree=True,
            gate_passed=False,
            handoff_tier="blocked",
            selected_for_l1=False,
            reject_reasons=("insufficient_events",),
            soft_flags=(),
        )

    idx_start = causal_lag
    idx_end = t - holding_bars

    entry_full = _sparse_entry_mask(side, valid)
    event_mask = np.zeros_like(entry_full)
    if idx_start < idx_end:
        event_mask[idx_start:idx_end, :] = entry_full[idx_start:idx_end, :]

    n_events = int(np.sum(event_mask))
    min_events = config.archetype_event_floors.get(recipe.archetype, config.min_events)
    if n_events < min_events:
        return AlphaGateEvidence(
            schema_version="unified",
            run_id="",
            timeframe=recipe.timeframe,
            family=recipe.family,
            variant=recipe.variant,
            recipe_id=recipe.recipe_id,
            archetype=recipe.archetype,
            symbol_scope="symbol",
            n_events=n_events,
            effective_n=0.0,
            mean_gross_bps=0.0,
            mean_cost_bps=0.0,
            mean_net_bps=0.0,
            gross_lcb_bps=0.0,
            net_lcb_bps=0.0,
            nw_tstat=0.0,
            rank_ic=0.0,
            rank_ic_tstat=0.0,
            cost_drag_ratio=0.0,
            turnover_per_year=0.0,
            novelty_corr_max=0.0,
            incremental_rank_ic=0.0,
            compute_cost_score=0.0,
            event_hit_rate=0.0,
            payoff_skew=0.0,
            xs_spread_lcb_bps=None,
            liquidity_cost_stress_bps=0.0,
            bootstrap_lcb_bps=0.0,
            bootstrap_agree=True,
            gate_passed=False,
            handoff_tier="blocked",
            selected_for_l1=False,
            reject_reasons=("insufficient_events",),
            soft_flags=(),
        )

    # Forward gross return: close[t+holding]/close[t+lag] * side
    fwd_ret_bps = np.full((t, n), np.nan, dtype=np.float64)
    idx_end_fwd = t - holding_bars
    for i in range(idx_end_fwd):
        fwd_ret_bps[i, :] = (
            side[i, :].astype(np.float64)
            * (close[i + holding_bars, :] / close[i + causal_lag, :] - 1.0)
            * 10000.0
        )

    # Stress cost
    stress_cost = cost_model.stress_round_trip_bps()

    # Funding over holding period
    funding_cost = np.where(event_mask, funding * 10000.0 * holding_bars, 0.0)

    # Liquidity stress
    liquidity_stress_bps = compute_liquidity_cost_stress_bps(
        aligned=aligned,
        event_mask=event_mask,
        stress_mult=config.liquidity_cost_stress_mult,
    )

    total_cost_bps = stress_cost + liquidity_stress_bps
    total_cost_2d = total_cost_bps + funding_cost

    net_bps = fwd_ret_bps - total_cost_2d

    net_vals = net_bps[event_mask]
    gross_vals = fwd_ret_bps[event_mask]

    effective_n = float(n_events)
    if effective_n < config.min_effective_n:
        reject_reasons_list.append("insufficient_effective_n")

    mean_gross_bps = float(np.nanmean(gross_vals)) if n_events > 0 else 0.0
    mean_cost_bps = float(np.nanmean(total_cost_2d[event_mask])) if n_events > 0 else 0.0
    mean_net_bps = float(np.nanmean(net_vals)) if n_events > 0 else 0.0

    # Block-based LCB
    block_bars_eff = max(config.block_bars, 2 * holding_bars)
    block_means = _compute_block_means(net_vals, block_bars_eff)
    mu_block, se_block = _block_moments(block_means)
    nw_tstat = mu_block / max(se_block, 1e-10)
    net_lcb_bps = mu_block - 1.0 * se_block

    # Gross block LCB
    gross_block_means = _compute_block_means(gross_vals, block_bars_eff)
    mu_gross, se_gross = _block_moments(gross_block_means)
    gross_lcb_bps = mu_gross - 1.0 * se_gross

    if mean_gross_bps <= 0.0:
        reject_reasons_list.append("non_positive_gross")
    if net_lcb_bps <= config.min_lcb_net_bps:
        reject_reasons_list.append("non_positive_lcb")
    if abs(nw_tstat) < config.min_nw_tstat:
        reject_reasons_list.append("weak_tstat")

    # Cost drag ratio (V2: event means)
    cost_drag = compute_cost_drag_ratio_v2(mean_cost_bps=mean_cost_bps, mean_gross_bps=mean_gross_bps)
    if cost_drag > config.max_cost_drag_ratio:
        reject_reasons_list.append("excess_cost_drag")

    # Turnover
    turnover = _compute_turnover_per_year(side, valid, bars_per_year)
    max_turn = min(config.max_turnover_per_year, recipe.max_turnover_per_year)
    if turnover > max_turn:
        reject_reasons_list.append("excess_turnover")

    # High turnover: gross_lcb_below_cost check
    if turnover >= config.high_turnover_per_year and gross_lcb_bps <= mean_cost_bps + liquidity_stress_bps:
        reject_reasons_list.append("gross_lcb_below_cost")

    # Rank IC with t-stat
    rank_ic, rank_ic_tstat = compute_rank_ic_with_tstat(
        fwd_ret_bps=fwd_ret_bps,
        score=panel.signed_score_2d,
        mask=event_mask,
    )
    rank_ic_floor = 1.0 / math.sqrt(max(n_events - 3, 1))
    if abs(rank_ic) < rank_ic_floor:
        soft_flags_list.append("weak_rank_ic")
    if abs(rank_ic_tstat) < config.min_candidate_rank_ic_tstat:
        soft_flags_list.append("weak_rank_ic_tstat")

    # Bootstrap
    rng = np.random.default_rng(config.bootstrap_seed)
    bootstrap_lcb_bps, _ = _bootstrap_block_ci(block_means, config.bootstrap_samples, rng)
    bootstrap_agree = (bootstrap_lcb_bps > 0) == (net_lcb_bps > 0)
    if not bootstrap_agree:
        soft_flags_list.append("bootstrap_disagree")

    # Payoff stats
    hit_rate, payoff_skew = compute_payoff_stats(net_vals)

    # Cross-sectional spread LCB
    xs_spread_lcb: float | None = None
    if recipe.archetype == "cross_sectional":
        xs_spread_lcb = compute_xs_spread_lcb_bps(
            net_bps=net_bps,
            score=panel.signed_score_2d,
            event_mask=event_mask,
            min_symbols_per_bar=config.min_xs_symbols_per_bar,
        )
        if xs_spread_lcb is None or xs_spread_lcb <= 0.0:
            reject_reasons_list.append("xs_spread_fail")

    gate_passed = len(reject_reasons_list) == 0

    return AlphaGateEvidence(
        schema_version="unified",
        run_id="",
        timeframe=recipe.timeframe,
        family=recipe.family,
        variant=recipe.variant,
        recipe_id=recipe.recipe_id,
        archetype=recipe.archetype,
        symbol_scope="symbol",
        n_events=n_events,
        effective_n=effective_n,
        mean_gross_bps=mean_gross_bps,
        mean_cost_bps=mean_cost_bps,
        mean_net_bps=mean_net_bps,
        gross_lcb_bps=gross_lcb_bps,
        net_lcb_bps=net_lcb_bps,
        nw_tstat=nw_tstat,
        rank_ic=rank_ic,
        rank_ic_tstat=rank_ic_tstat,
        cost_drag_ratio=cost_drag,
        turnover_per_year=turnover,
        novelty_corr_max=0.0,
        incremental_rank_ic=0.0,
        compute_cost_score=0.0,
        event_hit_rate=hit_rate,
        payoff_skew=payoff_skew,
        xs_spread_lcb_bps=xs_spread_lcb,
        liquidity_cost_stress_bps=liquidity_stress_bps,
        bootstrap_lcb_bps=bootstrap_lcb_bps,
        bootstrap_agree=bootstrap_agree,
        gate_passed=gate_passed,
        handoff_tier="candidate" if gate_passed else "blocked",
        selected_for_l1=False,
        reject_reasons=tuple(reject_reasons_list),
        soft_flags=tuple(soft_flags_list),
    )


def downgrade_gate_v2_to_cheap_evidence(evidence: AlphaGateEvidence) -> CheapGateEvidence:
    _valid_reject: set[str] = {
        "insufficient_events",
        "insufficient_effective_n",
        "non_positive_lcb",
        "weak_tstat",
        "excess_cost_drag",
        "excess_turnover",
        "invalid_shape",
        "lookahead_risk",
        "missing_required_field",
    }
    v1_reasons = cast(
        "tuple[CheapGateRejectReason, ...]",
        tuple(r for r in evidence.reject_reasons if r in _valid_reject),
    )
    return CheapGateEvidence(
        recipe_id=evidence.recipe_id,
        timeframe=evidence.timeframe,
        symbol_scope=evidence.symbol_scope,
        n_events=evidence.n_events,
        effective_n=evidence.effective_n,
        mean_net_bps=evidence.mean_net_bps,
        nw_tstat=evidence.nw_tstat,
        block_lcb_bps=evidence.net_lcb_bps,
        rank_ic=evidence.rank_ic,
        cost_drag_ratio=evidence.cost_drag_ratio,
        turnover_per_year=evidence.turnover_per_year,
        novelty_corr_max=0.0,
        incremental_rank_ic=0.0,
        compute_cost_score=0.0,
        bootstrap_lcb_bps=evidence.bootstrap_lcb_bps,
        bootstrap_agree=evidence.bootstrap_agree,
        gate_passed=evidence.gate_passed,
        reject_reasons=v1_reasons,
        mean_gross_bps=evidence.mean_gross_bps,
        mean_cost_bps=evidence.mean_cost_bps,
    )


# ── Unified gate evaluator ─────────────────────────────────────────────


def _empty_gate_evidence(
    *,
    run_id: str,
    recipe: AlphaRecipe,
    reject_reasons: tuple[str, ...] = ("insufficient_events",),
) -> AlphaGateEvidence:
    return AlphaGateEvidence(
        schema_version="unified",
        run_id=run_id,
        timeframe=recipe.timeframe,
        family=recipe.family,
        variant=recipe.variant,
        recipe_id=recipe.recipe_id,
        archetype=recipe.archetype,
        symbol_scope="symbol",
        n_events=0,
        effective_n=0.0,
        mean_gross_bps=0.0,
        mean_cost_bps=0.0,
        mean_net_bps=0.0,
        gross_lcb_bps=0.0,
        net_lcb_bps=0.0,
        nw_tstat=0.0,
        rank_ic=0.0,
        rank_ic_tstat=0.0,
        cost_drag_ratio=0.0,
        turnover_per_year=0.0,
        novelty_corr_max=0.0,
        incremental_rank_ic=0.0,
        compute_cost_score=0.0,
        event_hit_rate=0.0,
        payoff_skew=0.0,
        xs_spread_lcb_bps=None,
        liquidity_cost_stress_bps=0.0,
        bootstrap_lcb_bps=0.0,
        bootstrap_agree=True,
        gate_passed=False,
        handoff_tier="blocked",
        selected_for_l1=False,
        reject_reasons=reject_reasons,
        soft_flags=(),
    )


def evaluate_panel_gate(
    *,
    panel: CandidateSignalPanel,
    aligned: AlignedMarketData,
    recipe: AlphaRecipe,
    cost_model: ExecutionCostModel,
    config: AlphaGateConfig,
    bars_per_year: float,
    run_id: str,
    tf_fusion: MultiTimeframeEvidence | None = None,
) -> AlphaGateEvidence:
    """[ADR_20260708_L0_SIGNAL_YIELD_IMPROVEMENT] n_events floor resolved via
    resolve_family_timeframe_gate_policy (family/archetype-aware), not flat config.min_events."""
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

    reject_reasons: list[str] = []
    soft_flags: list[str] = []

    if causal_lag >= t or holding_bars >= t:
        return _empty_gate_evidence(
            run_id=run_id, recipe=recipe, reject_reasons=("insufficient_events",)
        )

    idx_start = causal_lag
    idx_end = t - holding_bars

    entry_full = _sparse_entry_mask(side, valid)
    event_mask = np.zeros_like(entry_full)
    if idx_start < idx_end:
        event_mask[idx_start:idx_end, :] = entry_full[idx_start:idx_end, :]

    n_events = int(np.sum(event_mask))
    resolved_min_events = resolve_family_timeframe_gate_policy(recipe=recipe, config=config).min_events
    if n_events < resolved_min_events:
        return _empty_gate_evidence(
            run_id=run_id, recipe=recipe, reject_reasons=("insufficient_events",)
        )

    fwd_ret_bps = np.full((t, n), np.nan, dtype=np.float64)
    idx_end_fwd = t - holding_bars
    for i in range(idx_end_fwd):
        fwd_ret_bps[i, :] = (
            side[i, :].astype(np.float64)
            * (close[i + holding_bars, :] / close[i + causal_lag, :] - 1.0)
            * 10000.0
        )

    stress_cost = cost_model.stress_round_trip_bps()
    funding_cost = np.where(event_mask, funding * 10000.0 * holding_bars, 0.0)

    liquidity_stress_bps = compute_liquidity_cost_stress_bps(
        aligned=aligned,
        event_mask=event_mask,
        stress_mult=config.liquidity_cost_stress_mult,
    )

    total_cost_bps = stress_cost + liquidity_stress_bps
    total_cost_2d = total_cost_bps + funding_cost

    net_bps = fwd_ret_bps - total_cost_2d

    net_vals = net_bps[event_mask]
    gross_vals = fwd_ret_bps[event_mask]

    # Capacity score (liquidity-aware)
    capacity_score = compute_capacity_score(
        aligned=aligned,
        event_mask=event_mask,
        liquidity_cost_stress_bps=liquidity_stress_bps,
    )

    # Regime stability
    regime_stability = compute_regime_stability(
        panel=panel,
        net_bps=net_bps,
        event_mask=event_mask,
    )

    # TF corroboration
    tf_corroboration = compute_tf_corroboration(
        recipe=recipe,
        tf_fusion=tf_fusion,
    )

    # TF contradicted -> hard reject
    if tf_fusion is not None and tf_fusion.corroboration_tier == "contradicted":
        reject_reasons.append("tf_contradicted")

    # capacity_score <= 0.25 when no liquidity data
    if aligned.execution_cost_bps_2d is None or aligned.adv_usdt_2d is None:
        capacity_score = min(capacity_score, 0.25)

    # regime_stability < 0.5 -> max seed, no candidate
    if regime_stability < 0.5:
        soft_flags.append("low_regime_stability")

    effective_n = float(n_events)
    if effective_n < config.min_effective_n:
        reject_reasons.append("insufficient_effective_n")

    mean_gross_bps = (
        float(np.nanmean(gross_vals)) if n_events > 0 and np.any(np.isfinite(gross_vals)) else 0.0
    )
    cost_at_events = total_cost_2d[event_mask]
    mean_cost_bps = (
        float(np.nanmean(cost_at_events))
        if n_events > 0 and np.any(np.isfinite(cost_at_events))
        else 0.0
    )
    mean_net_bps = (
        float(np.nanmean(net_vals)) if n_events > 0 and np.any(np.isfinite(net_vals)) else 0.0
    )

    block_bars_eff = max(config.block_bars, 2 * holding_bars)
    block_means = _compute_block_means(net_vals, block_bars_eff)
    mu_block, se_block = _block_moments(block_means)
    nw_tstat = mu_block / max(se_block, 1e-10)
    net_lcb_bps = mu_block - 1.0 * se_block

    gross_block_means = _compute_block_means(gross_vals, block_bars_eff)
    mu_gross, se_gross = _block_moments(gross_block_means)
    gross_lcb_bps = mu_gross - 1.0 * se_gross

    if mean_gross_bps <= 0.0:
        reject_reasons.append("non_positive_gross")
    if net_lcb_bps <= config.min_lcb_net_bps:
        reject_reasons.append("non_positive_lcb")
    if abs(nw_tstat) < config.min_nw_tstat:
        reject_reasons.append("weak_tstat")

    cost_drag = compute_cost_drag_ratio_v2(mean_cost_bps=mean_cost_bps, mean_gross_bps=mean_gross_bps)

    # Fast TF stricter cost threshold (30m, 1h, 2h)
    fast_tf = recipe.timeframe in ("30m", "1h", "2h")
    effective_max_cost_drag = config.max_cost_drag_ratio * 0.75 if fast_tf else config.max_cost_drag_ratio
    if cost_drag > effective_max_cost_drag:
        reject_reasons.append("excess_cost_drag")

    turnover = _compute_turnover_per_year(side, valid, bars_per_year)
    max_turn = min(config.max_turnover_per_year, recipe.max_turnover_per_year)

    entry_mode = panel.metadata.get("entry_mode", "sparse")
    if turnover > max_turn:
        reject_reasons.append("excess_turnover")

    if turnover >= config.high_turnover_per_year and gross_lcb_bps <= mean_cost_bps + liquidity_stress_bps:
        reject_reasons.append("gross_lcb_below_cost")

    rank_ic, rank_ic_tstat = compute_rank_ic_with_tstat(
        fwd_ret_bps=fwd_ret_bps,
        score=panel.signed_score_2d,
        mask=event_mask,
    )
    rank_ic_floor = 1.0 / math.sqrt(max(n_events - 3, 1))
    weak_rank_ic = abs(rank_ic) < rank_ic_floor
    if weak_rank_ic:
        soft_flags.append("weak_rank_ic")
    if abs(rank_ic_tstat) < config.min_candidate_rank_ic_tstat:
        soft_flags.append("weak_rank_ic_tstat")

    rng = np.random.default_rng(config.bootstrap_seed)
    bootstrap_lcb_bps, _ = _bootstrap_block_ci(block_means, config.bootstrap_samples, rng)
    bootstrap_agree = (bootstrap_lcb_bps > 0) == (net_lcb_bps > 0)
    if not bootstrap_agree:
        soft_flags.append("bootstrap_disagree")

    hit_rate, payoff_skew = compute_payoff_stats(net_vals)

    xs_spread_lcb: float | None = None
    if recipe.archetype == "cross_sectional":
        xs_spread_lcb = compute_xs_spread_lcb_bps(
            net_bps=net_bps,
            score=panel.signed_score_2d,
            event_mask=event_mask,
            min_symbols_per_bar=config.min_xs_symbols_per_bar,
        )
        if xs_spread_lcb is None or xs_spread_lcb <= 0.0:
            reject_reasons.append("xs_spread_fail")

    gate_passed = len(reject_reasons) == 0

    # handoff_tier: weak_rank_ic alone does NOT downgrade to blocked (only priority affected)
    handoff_tier: AlphaGateHandoffTier
    if reject_reasons:
        handoff_tier = "blocked"
    elif regime_stability < 0.5 or tf_corroboration < 0.5 or soft_flags:
        handoff_tier = "seed"
    else:
        handoff_tier = "candidate"

    return AlphaGateEvidence(
        schema_version="unified",
        run_id=run_id,
        timeframe=recipe.timeframe,
        family=recipe.family,
        variant=recipe.variant,
        recipe_id=recipe.recipe_id,
        archetype=recipe.archetype,
        symbol_scope="symbol",
        n_events=n_events,
        effective_n=effective_n,
        mean_gross_bps=mean_gross_bps,
        mean_cost_bps=mean_cost_bps,
        mean_net_bps=mean_net_bps,
        gross_lcb_bps=gross_lcb_bps,
        net_lcb_bps=net_lcb_bps,
        nw_tstat=nw_tstat,
        rank_ic=rank_ic,
        rank_ic_tstat=rank_ic_tstat,
        cost_drag_ratio=cost_drag,
        turnover_per_year=turnover,
        novelty_corr_max=0.0,
        incremental_rank_ic=0.0,
        compute_cost_score=0.0,
        event_hit_rate=hit_rate,
        payoff_skew=payoff_skew,
        xs_spread_lcb_bps=xs_spread_lcb,
        liquidity_cost_stress_bps=liquidity_stress_bps,
        bootstrap_lcb_bps=bootstrap_lcb_bps,
        bootstrap_agree=bootstrap_agree,
        gate_passed=gate_passed,
        handoff_tier=handoff_tier,
        selected_for_l1=False,
        reject_reasons=tuple(reject_reasons),
        soft_flags=tuple(soft_flags),
        capacity_score=capacity_score,
        regime_stability=regime_stability,
        tf_corroboration=tf_corroboration,
        entry_mode=entry_mode,
    )


def evaluate_alpha_gate_batch(
    *,
    panels: Sequence[CandidateSignalPanel],
    recipes: Mapping[str, AlphaRecipe],
    aligned: AlignedMarketData,
    cost_model: ExecutionCostModel,
    config: AlphaGateConfig,
    run_id: str,
    tf_fusion_index: Mapping[tuple[str, str, str], MultiTimeframeEvidence] | None = None,
) -> tuple[AlphaGateEvidence, ...]:
    from src.domain.futures.optimization.metrics import _bars_per_year_for_tf

    bpy_cache: dict[str, float] = {}
    results: list[AlphaGateEvidence] = []
    for panel in panels:
        recipe_id = panel.metadata.get("recipe_id", "")
        recipe = recipes.get(recipe_id)
        if recipe is None:
            continue
        tf = recipe.timeframe
        if tf not in bpy_cache:
            bpy_cache[tf] = _bars_per_year_for_tf(tf)

        from src.domain.futures.alpha_foundry.multi_tf_fusion import _strip_tf_suffix

        normalized_variant = _strip_tf_suffix(recipe.variant, recipe.timeframe)
        tf_key = (recipe.family, normalized_variant, recipe.timeframe)
        tf_ev = tf_fusion_index.get(tf_key) if tf_fusion_index is not None else None

        evidence = evaluate_panel_gate(
            panel=panel,
            aligned=aligned,
            recipe=recipe,
            cost_model=cost_model,
            config=config,
            bars_per_year=bpy_cache[tf],
            run_id=run_id,
            tf_fusion=tf_ev,
        )
        results.append(evidence)
    return tuple(results)


def emit_alpha_generation_debug_summary(
    *,
    run_id: str,
    timeframe: str,
    evidences: Sequence[AlphaGateEvidence],
    debug_top_k_rows: int,
    debug_reject_bucket_rows: int,
) -> None:
    import logging

    logger = logging.getLogger(__name__)
    passed = [e for e in evidences if e.gate_passed]
    rejected = [e for e in evidences if not e.gate_passed]

    logger.debug(
        "[EVAL] stage=af_generation run_id=%s tf=%s total=%d passed=%d rejected=%d",
        run_id, timeframe, len(evidences), len(passed), len(rejected),
    )

    # Top candidates by mean_net_bps
    sorted_ev = sorted(evidences, key=lambda e: e.mean_net_bps, reverse=True)
    top_k = sorted_ev[:debug_top_k_rows]
    for e in top_k:
        logger.debug(
            "[ALGO] TOP recipe=%s net=%.2f gross=%.2f cost=%.2f handoff=%s cap=%.2f regime=%.2f tf_corr=%.2f",
            e.recipe_id, e.mean_net_bps, e.mean_gross_bps,
            e.mean_cost_bps, getattr(e, "handoff_tier", "n/a"), getattr(e, "capacity_score", 0.0),
            getattr(e, "regime_stability", 0.0), getattr(e, "tf_corroboration", 0.0),
        )

    # Reject bucket rows
    top_reject: dict[str, int] = {}
    for e in rejected:
        for r in e.reject_reasons:
            top_reject[r] = top_reject.get(r, 0) + 1
    if top_reject:
        reasons_str = " | ".join(f"{k}={v}" for k, v in sorted(top_reject.items(), key=lambda x: -x[1]))
        logger.debug("[DATA] reject_reasons: %s", reasons_str)

    # Cost drag bucket (rejected by excess_cost_drag or high cost_drag_ratio)
    cost_drag_bucket = [e for e in rejected if "excess_cost_drag" in e.reject_reasons]
    for e in cost_drag_bucket[:debug_reject_bucket_rows]:
        logger.debug(
            "[DATA] COST_DRAG recipe=%s cost_drag=%.2f cap=%.2f regime=%.2f",
            e.recipe_id, e.cost_drag_ratio, getattr(e, "capacity_score", 0.0), getattr(e, "regime_stability", 0.0),
        )
