"""Alpha Foundry L0 cheap gate and survivor filtering. [ADR_20260706_ALPHA_FOUNDRY_SYNC]"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats

from src.domain.futures.alpha_foundry.contracts import (
    AlphaRecipe,
    CheapGateConfig,
    CheapGateEvidence,
    CheapGateRejectReason,
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
            raise ValueError(
                f"panel.{name} shape {arr.shape} != aligned close_2d shape ({t}, {n})"
            )


def _block_lcb(values: NDArray[np.float64], block_bars: int) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    n_blocks = max(1, n // block_bars)
    block_means = np.array([
        values[i * block_bars : (i + 1) * block_bars].mean()
        for i in range(n_blocks)
    ])
    mu = float(np.nanmean(block_means))
    n_valid = float(np.sum(~np.isnan(block_means)))
    se_val = float(np.nanstd(block_means, ddof=1)) / max(np.sqrt(max(n_valid, 1.0)), 1.0)
    return float(mu - 1.0 * se_val)


def _compute_turnover_per_year(
    side: NDArray[np.int8], valid_mask: NDArray[np.bool_], bars_per_year: float
) -> float:
    diff = np.abs(np.diff(side.astype(np.float64), axis=0))
    valid_slice = valid_mask[1:, :] & valid_mask[:-1, :]
    denom = max(np.sum(valid_slice), 1)
    return float(np.sum(diff * valid_slice) / denom * bars_per_year / 2.0)


def _compute_rank_ic(
    fwd_ret: NDArray[np.float64], score: NDArray[np.float64], mask: NDArray[np.bool_]
) -> float:
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
    prior_panels: Sequence[CandidateSignalPanel] = (),
    regime_code_1d: NDArray[np.int8] | None = None,
) -> CheapGateEvidence:
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
            monotonic_bucket_score=0.0,
            regime_edges_bps={},
            cost_drag_ratio=0.0,
            turnover_per_year=0.0,
            novelty_corr_max=0.0,
            incremental_rank_ic=0.0,
            compute_cost_score=0.0,
            gate_passed=False,
            reject_reasons=("insufficient_events",),
        )

    idx_start = causal_lag
    idx_end = t - holding_bars

    event_mask = np.zeros((t, n), dtype=np.bool_)
    if idx_start < idx_end:
        event_mask[idx_start:idx_end, :] = (
            (side[idx_start:idx_end, :] != 0)
            & valid[idx_start:idx_end, :]
        )

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
            monotonic_bucket_score=0.0,
            regime_edges_bps={},
            cost_drag_ratio=0.0,
            turnover_per_year=0.0,
            novelty_corr_max=0.0,
            incremental_rank_ic=0.0,
            compute_cost_score=0.0,
            gate_passed=False,
            reject_reasons=("insufficient_events",),
        )

    fwd_ret_bps = np.full((t, n), np.nan, dtype=np.float64)
    for i in range(t - holding_bars):
        fwd_ret_bps[i, :] = (
            side[i, :].astype(np.float64)
            * np.log(close[i + holding_bars, :] / close[i, :])
            * 10000.0
        )

    stress_cost = cost_model.stress_round_trip_bps()
    funding_cost = np.where(event_mask, funding * 10000.0 * holding_bars, 0.0)
    net_bps = fwd_ret_bps - stress_cost - funding_cost

    net_vals = net_bps[event_mask]
    gross_vals = fwd_ret_bps[event_mask]

    reject_reasons_list: list[CheapGateRejectReason] = []

    # effective_n
    w = np.ones(n_events, dtype=np.float64)
    effective_n = float(np.sum(w) ** 2 / np.sum(w ** 2)) if np.any(w) else 0.0
    if effective_n < config.min_effective_n:
        reject_reasons_list.append("insufficient_effective_n")

    mean_net_bps = float(np.nanmean(net_vals)) if n_events > 0 else 0.0
    std_net = float(np.nanstd(net_vals, ddof=1)) if n_events > 1 else 0.0
    nw_tstat = mean_net_bps / max(std_net / np.sqrt(max(n_events, 1)), 1e-10)

    # block LCB
    block_lcb_bps = _block_lcb(net_vals, config.block_bars)

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
    bars_per_year = 365.0 * 24.0 / 4.0
    turnover = _compute_turnover_per_year(side, valid, bars_per_year)
    max_turn = min(config.max_turnover_per_year, recipe.max_turnover_per_year)
    if turnover > max_turn:
        reject_reasons_list.append("excess_turnover")

    # novelty
    novelty_corr_max = 0.0
    incremental_rank_ic = 0.0
    for prior in prior_panels:
        prior_flat = prior.signed_score_2d[event_mask]
        curr_flat = panel.signed_score_2d[event_mask]
        if len(prior_flat) > 2:
            corr = float(np.corrcoef(prior_flat, curr_flat)[0, 1]) if np.std(prior_flat) > 0 else 0.0
            if np.isfinite(corr):
                novelty_corr_max = max(novelty_corr_max, abs(corr))
    if novelty_corr_max > config.max_novelty_corr:
        reject_reasons_list.append("duplicate_signal")

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
        monotonic_bucket_score=0.0,
        regime_edges_bps={},
        cost_drag_ratio=cost_drag_ratio,
        turnover_per_year=turnover,
        novelty_corr_max=novelty_corr_max,
        incremental_rank_ic=incremental_rank_ic,
        compute_cost_score=0.0,
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
    regime_code_1d: NDArray[np.int8] | None = None,
) -> tuple[CheapGateEvidence, ...]:
    results: list[CheapGateEvidence] = []
    processed_panels: list[CandidateSignalPanel] = []
    for panel in panels:
        recipe_id = panel.metadata.get("recipe_id", "")
        recipe = recipes.get(recipe_id)
        if recipe is None:
            continue
        evidence = evaluate_panel_cheap_gate(
            panel=panel,
            aligned=aligned,
            recipe=recipe,
            cost_model=cost_model,
            config=config,
            prior_panels=tuple(processed_panels),
            regime_code_1d=regime_code_1d,
        )
        results.append(evidence)
        processed_panels.append(panel)
    return tuple(results)
