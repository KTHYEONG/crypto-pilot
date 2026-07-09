"""L0 point-in-time conditional cell slicing/gating.
[ADR_20260708_L0_EDGE_FAILURE_ATTRIBUTION][ADR_20260709_L0_CONDITIONAL_DIAGNOSTIC_WIRING]"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.alpha_foundry.contracts import (
    AlphaGateConfig,
    AlphaGateEvidence,
    AlphaRecipe,
    ConditionalAxis,
    ConditionalCellGateConfig,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel

_KNOWN_AXES: frozenset[str] = frozenset({
    "symbol_liquidity",
    "symbol_cluster",
    "market_regime",
    "volatility_regime",
    "funding_polarity",
    "score_quantile",
    "event_hour_utc",
    "source_tf",
})


@dataclass(slots=True, frozen=True)
class ConditionalCellSpec:
    cell_id: str
    axes: tuple[ConditionalAxis, ...]
    values: Mapping[str, str]
    min_events: int
    min_effective_n: float


@dataclass(slots=True, frozen=True)
class ConditionalCellEvidence:
    cell_id: str
    axes: tuple[ConditionalAxis, ...]
    values: Mapping[str, str]
    event_mask_2d: NDArray[np.bool_]
    gate_evidence: AlphaGateEvidence
    tested_horizons: tuple[int, ...] = ()
    selected_horizon: int = 0
    execution_style: str = "taker_now"
    fill_probability: float = 1.0
    adverse_selection_bps: float = 0.0
    failure_axis: str = ""


def _validate_shape(panel: CandidateSignalPanel, aligned: AlignedMarketData) -> None:
    t, n = aligned.close_2d.shape
    if panel.signed_score_2d.shape != (t, n):
        raise ValueError(
            f"shape mismatch: panel.signed_score_2d shape {panel.signed_score_2d.shape} != ({t}, {n})"
        )
    if panel.side_hint_2d.shape != (t, n):
        raise ValueError(
            f"shape mismatch: panel.side_hint_2d shape {panel.side_hint_2d.shape} != ({t}, {n})"
        )
    if panel.valid_mask_2d.shape != (t, n):
        raise ValueError(
            f"shape mismatch: panel.valid_mask_2d shape {panel.valid_mask_2d.shape} != ({t}, {n})"
        )


def _sparse_entry_mask(
    side: NDArray[np.int8],
    valid: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    side_prev = np.vstack([np.zeros((1, side.shape[1]), dtype=side.dtype), side[:-1, :]])
    entry: NDArray[np.bool_] = (side != 0) & (side != side_prev) & valid
    return entry


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


def _compute_turnover_per_year(side: NDArray[np.int8], valid_mask: NDArray[np.bool_], bars_per_year: float) -> float:
    diff = np.abs(np.diff(side.astype(np.float64), axis=0))
    valid_slice = valid_mask[1:, :] & valid_mask[:-1, :]
    denom = max(np.sum(valid_slice), 1)
    return float(np.sum(diff * valid_slice) / denom * bars_per_year / 2.0)


def evaluate_event_mask_gate(
    *,
    event_mask: NDArray[np.bool_],
    panel: CandidateSignalPanel,
    aligned: AlignedMarketData,
    recipe: AlphaRecipe,
    round_trip_cost_bps: float,
    gate_config: AlphaGateConfig,
    bars_per_year: float,
    run_id: str,
) -> AlphaGateEvidence:
    """Evaluates the economic gate over an arbitrary boolean event mask.

    Generalizes the former `_evaluate_cell_gate` (cell-slicing only) so execution-arm
    re-costing can reuse identical statistical machinery (block moments, NW t-stat,
    cost-drag, turnover) instead of duplicating it. `round_trip_cost_bps` replaces the
    prior `cost_model.stress_round_trip_bps()` call site — callers resolve their own
    cost figure (stress cost for cells, arm cost for execution-arm re-evaluation).
    """
    if bars_per_year <= 0.0:
        raise ValueError("bars_per_year must be positive")

    t, n = aligned.close_2d.shape
    close = aligned.close_2d
    side = panel.side_hint_2d
    holding_bars = panel.expected_holding_bars

    n_events = int(np.sum(event_mask))

    if n_events < gate_config.min_events:
        return AlphaGateEvidence(
            schema_version="unified",
            run_id=run_id, timeframe=recipe.timeframe,
            family=recipe.family, variant=recipe.variant,
            recipe_id=recipe.recipe_id, archetype=recipe.archetype,
            symbol_scope="symbol", n_events=n_events, effective_n=0.0,
            mean_gross_bps=0.0, mean_cost_bps=0.0, mean_net_bps=0.0,
            gross_lcb_bps=0.0, net_lcb_bps=0.0, nw_tstat=0.0,
            rank_ic=0.0, rank_ic_tstat=0.0, cost_drag_ratio=0.0,
            turnover_per_year=0.0, novelty_corr_max=0.0,
            incremental_rank_ic=0.0, compute_cost_score=0.0,
            event_hit_rate=0.0, payoff_skew=0.0, xs_spread_lcb_bps=None,
            liquidity_cost_stress_bps=0.0, bootstrap_lcb_bps=0.0,
            bootstrap_agree=True, gate_passed=False,
            handoff_tier="blocked", selected_for_l1=False,
            reject_reasons=("insufficient_events",), soft_flags=(),
        )

    idx_end_fwd = t - holding_bars
    fwd_ret_bps = np.full((t, n), np.nan, dtype=np.float64)
    for i in range(idx_end_fwd):
        close_entry = close[i, :]
        close_exit = close[i + holding_bars, :]
        fwd_ret_bps[i, :] = (
            side[i, :].astype(np.float64)
            * (close_exit / np.maximum(close_entry, 1e-10) - 1.0)
            * 10000.0
        )

    stress_cost = round_trip_cost_bps
    funding = aligned.funding_2d
    funding_cost = np.where(event_mask, funding * 10000.0 * holding_bars, 0.0)
    total_cost_2d = stress_cost + funding_cost
    net_bps = fwd_ret_bps - total_cost_2d

    net_vals = net_bps[event_mask]
    gross_vals = fwd_ret_bps[event_mask]

    reject_reasons: list[str] = []

    effective_n = float(n_events)
    if effective_n < gate_config.min_effective_n:
        reject_reasons.append("insufficient_effective_n")

    mean_gross_bps = float(np.nanmean(gross_vals)) if n_events > 0 else 0.0
    mean_cost_bps = float(np.nanmean(total_cost_2d[event_mask])) if n_events > 0 else 0.0
    mean_net_bps = float(np.nanmean(net_vals)) if n_events > 0 else 0.0

    block_bars_eff = max(gate_config.block_bars, 2 * holding_bars)
    block_means = _compute_block_means(net_vals, block_bars_eff)
    mu_block, se_block = _block_moments(block_means)
    nw_tstat = mu_block / max(se_block, 1e-10)
    net_lcb_bps = mu_block - 1.0 * se_block

    if net_lcb_bps <= gate_config.min_lcb_net_bps:
        reject_reasons.append("non_positive_lcb")
    if abs(nw_tstat) < gate_config.min_nw_tstat:
        reject_reasons.append("weak_tstat")

    total_gross = float(np.nansum(gross_vals)) if n_events > 0 else 0.0
    total_net = float(np.nansum(net_vals)) if n_events > 0 else 0.0
    total_cost = total_gross - total_net
    eps = 1e-10
    cost_drag_ratio = total_cost / max(abs(total_gross), eps)
    if cost_drag_ratio > gate_config.max_cost_drag_ratio:
        reject_reasons.append("excess_cost_drag")

    turnover = _compute_turnover_per_year(side, event_mask, bars_per_year)
    max_turn = min(gate_config.max_turnover_per_year, recipe.max_turnover_per_year)
    if turnover > max_turn:
        reject_reasons.append("excess_turnover")

    gate_passed = len(reject_reasons) == 0

    return AlphaGateEvidence(
        schema_version="unified",
        run_id=run_id, timeframe=recipe.timeframe,
        family=recipe.family, variant=recipe.variant,
        recipe_id=recipe.recipe_id, archetype=recipe.archetype,
        symbol_scope="symbol", n_events=n_events,
        effective_n=effective_n,
        mean_gross_bps=mean_gross_bps, mean_cost_bps=mean_cost_bps,
        mean_net_bps=mean_net_bps,
        gross_lcb_bps=0.0, net_lcb_bps=net_lcb_bps,
        nw_tstat=nw_tstat, rank_ic=0.0, rank_ic_tstat=0.0,
        cost_drag_ratio=cost_drag_ratio,
        turnover_per_year=turnover,
        novelty_corr_max=0.0, incremental_rank_ic=0.0,
        compute_cost_score=0.0, event_hit_rate=0.0, payoff_skew=0.0,
        xs_spread_lcb_bps=None, liquidity_cost_stress_bps=0.0,
        bootstrap_lcb_bps=0.0, bootstrap_agree=True,
        gate_passed=gate_passed,
        handoff_tier="candidate" if gate_passed else "blocked",
        selected_for_l1=False,
        reject_reasons=tuple(reject_reasons), soft_flags=(),
    )


def build_parent_event_mask(*, panel: CandidateSignalPanel, aligned: AlignedMarketData) -> NDArray[np.bool_]:
    """Sparse entry mask over the full recipe timeline, respecting warm-up (idx_start=1)
    and forward-return lookback (idx_end = t - holding_bars). Shared by cell slicing
    and execution-arm re-costing."""
    t, _ = aligned.close_2d.shape
    side = panel.side_hint_2d
    valid = panel.valid_mask_2d
    holding_bars = panel.expected_holding_bars
    idx_start = 1
    idx_end = t - holding_bars
    entry_full = _sparse_entry_mask(side, valid)
    event_mask = np.zeros_like(entry_full)
    if idx_start < idx_end:
        event_mask[idx_start:idx_end, :] = entry_full[idx_start:idx_end, :]
    return event_mask


def build_calibrated_cell_masks(
    *,
    panel: CandidateSignalPanel,
    aligned: AlignedMarketData,
    specs: Sequence[ConditionalCellSpec],
    calibration_fraction: float,
) -> Mapping[str, tuple[NDArray[np.bool_], int, int]]:
    """Like build_conditional_cell_masks, but score_quantile/symbol_liquidity thresholds
    are computed only from the chronologically-first `calibration_fraction` of parent
    events, then applied to the full timeline. Returns {cell_id: (mask, calibration_n, evaluation_n)}.
    [LIMIT-01]
    """
    t, _ = aligned.close_2d.shape
    side = panel.side_hint_2d
    valid = panel.valid_mask_2d
    holding_bars = panel.expected_holding_bars

    idx_start = 1
    idx_end = t - holding_bars
    entry_full = _sparse_entry_mask(side, valid)
    parent_mask = np.zeros_like(entry_full)
    if idx_start < idx_end:
        parent_mask[idx_start:idx_end, :] = entry_full[idx_start:idx_end, :]

    event_indices = np.where(parent_mask)
    if len(event_indices[0]) == 0:
        return {}

    # Chronological sort is guaranteed by np.where order (row-major)
    calib_count = max(1, int(len(event_indices[0]) * calibration_fraction))
    calib_rows = event_indices[0][:calib_count]
    calib_cols = event_indices[1][:calib_count]

    calib_mask = np.zeros_like(parent_mask)
    calib_mask[calib_rows, calib_cols] = True
    calib_n = int(np.sum(calib_mask))
    evaluation_n = int(np.sum(parent_mask)) - calib_n

    results: dict[str, tuple[NDArray[np.bool_], int, int]] = {}
    for spec in specs:
        mask = parent_mask.copy()
        for axis in spec.axes:
            val = spec.values.get(axis, "")
            if axis == "score_quantile":
                abs_score = np.abs(panel.signed_score_2d)
                calib_scores = abs_score[calib_mask]
                if len(calib_scores) < 5:
                    mask = np.zeros_like(mask)
                    break
                if val == "high":
                    threshold = float(np.quantile(calib_scores, 0.85))
                elif val == "very_high":
                    threshold = float(np.quantile(calib_scores, 0.95))
                else:
                    threshold = float(np.quantile(calib_scores, 0.70))
                mask = (abs_score >= threshold) & mask
            elif axis == "symbol_liquidity":
                if aligned.adv_usdt_2d is None:
                    continue
                adv = aligned.adv_usdt_2d
                calib_adv = adv[calib_mask]
                if len(calib_adv) < 2:
                    continue
                median_adv = float(np.nanmedian(calib_adv))
                mask = (adv >= median_adv) & mask if val == "high" else (adv < median_adv) & mask
            elif axis in _KNOWN_AXES:
                pass
            else:
                raise ValueError(f"unsupported conditional axis: {axis!r}")
        results[spec.cell_id] = (mask, calib_n, evaluation_n)
    return results


def _build_score_quantile_mask(
    panel: CandidateSignalPanel,
    aligned: AlignedMarketData,
    event_mask: NDArray[np.bool_],
    quantile: float,
) -> NDArray[np.bool_]:
    abs_score = np.abs(panel.signed_score_2d)
    event_scores = abs_score[event_mask]
    if len(event_scores) < 5:
        return np.zeros_like(event_mask)
    threshold = float(np.quantile(event_scores, quantile))
    return (abs_score >= threshold) & event_mask


def _build_liquidity_mask(
    panel: CandidateSignalPanel,
    aligned: AlignedMarketData,
    event_mask: NDArray[np.bool_],
    high_liquidity: bool,
) -> NDArray[np.bool_]:
    if aligned.adv_usdt_2d is None:
        return event_mask
    adv = aligned.adv_usdt_2d
    event_adv = adv[event_mask]
    if len(event_adv) < 2:
        return event_mask
    median_adv = float(np.nanmedian(event_adv))
    if high_liquidity:
        return (adv >= median_adv) & event_mask
    return (adv < median_adv) & event_mask


def build_conditional_cell_masks(
    *,
    panel: CandidateSignalPanel,
    aligned: AlignedMarketData,
    specs: Sequence[ConditionalCellSpec],
) -> Mapping[str, NDArray[np.bool_]]:
    t, _ = aligned.close_2d.shape
    side = panel.side_hint_2d
    valid = panel.valid_mask_2d
    holding_bars = panel.expected_holding_bars

    idx_start = 1
    idx_end = t - holding_bars
    entry_full = _sparse_entry_mask(side, valid)
    event_mask = np.zeros_like(entry_full)
    if idx_start < idx_end:
        event_mask[idx_start:idx_end, :] = entry_full[idx_start:idx_end, :]

    results: dict[str, NDArray[np.bool_]] = {}
    for spec in specs:
        mask = event_mask.copy()
        for axis in spec.axes:
            val = spec.values.get(axis, "")
            if axis == "score_quantile":
                if val == "high":
                    mask = _build_score_quantile_mask(panel, aligned, mask, 0.85)
                elif val == "very_high":
                    mask = _build_score_quantile_mask(panel, aligned, mask, 0.95)
            elif axis == "symbol_liquidity":
                mask = _build_liquidity_mask(panel, aligned, mask, val == "high")
            elif axis in _KNOWN_AXES:
                pass
            else:
                raise ValueError(f"unsupported conditional axis: {axis!r}")
        results[spec.cell_id] = mask
    return results


def generate_default_cell_specs(
    *,
    panel: CandidateSignalPanel,
    aligned: AlignedMarketData,
    config: ConditionalCellGateConfig,
) -> tuple[ConditionalCellSpec, ...]:
    if not config.enabled:
        return ()

    for axis in config.axes:
        if axis not in _KNOWN_AXES:
            raise ValueError(f"unsupported conditional axis: {axis!r}")

    t, _ = aligned.close_2d.shape
    side = panel.side_hint_2d
    valid = panel.valid_mask_2d
    holding_bars = panel.expected_holding_bars

    idx_start = 1
    idx_end = t - holding_bars
    entry_full = _sparse_entry_mask(side, valid)
    event_mask = np.zeros_like(entry_full)
    if idx_start < idx_end:
        event_mask[idx_start:idx_end, :] = entry_full[idx_start:idx_end, :]

    axis_buckets: dict[str, list[tuple[str, Mapping[str, str]]]] = {}

    for axis in config.axes:
        buckets: list[tuple[str, Mapping[str, str]]] = []
        if axis == "score_quantile":
            for q in config.quantile_bins:
                label = f"sq_{int(q * 100)}"
                buckets.append((label, {axis: "high" if q >= 0.85 else "medium"}))
        elif axis == "symbol_liquidity":
            if aligned.adv_usdt_2d is not None:
                buckets.append(("liq_high", {axis: "high"}))
                buckets.append(("liq_low", {axis: "low"}))
            else:
                buckets.append(("liq_all", {axis: "high"}))
        elif axis == "volatility_regime":
            buckets.append(("vol_low", {axis: "low"}))
            buckets.append(("vol_high", {axis: "high"}))
        elif axis == "funding_polarity":
            if aligned.funding_2d is not None:
                buckets.append(("fund_pos", {axis: "positive"}))
                buckets.append(("fund_neg", {axis: "negative"}))
        else:
            buckets.append((f"{axis}_all", {axis: "present"}))
        axis_buckets[axis] = buckets

    axes_list = [a for a in config.axes if a in axis_buckets]
    if not axes_list:
        return ()

    max_combine = min(config.max_axes_per_cell, len(axes_list))

    specs: list[ConditionalCellSpec] = []
    cell_count = 0

    for combine_n in range(1, max_combine + 1):
        if cell_count >= config.max_cells_per_recipe:
            break
        if combine_n == 1:
            for axis in axes_list:
                for bucket_label, bucket_values in axis_buckets[axis]:
                    if cell_count >= config.max_cells_per_recipe:
                        break
                    cell_id = f"{axis}:{bucket_label}"
                    specs.append(ConditionalCellSpec(
                        cell_id=cell_id,
                        axes=(axis,),
                        values=bucket_values,
                        min_events=config.min_cell_events,
                        min_effective_n=config.min_cell_effective_n,
                    ))
                    cell_count += 1
        elif combine_n == 2:
            for i, a1 in enumerate(axes_list):
                for a2 in axes_list[i + 1:]:
                    for b1_label, b1_values in axis_buckets[a1]:
                        for b2_label, b2_values in axis_buckets[a2]:
                            if cell_count >= config.max_cells_per_recipe:
                                break
                            cell_id = f"{a1}:{b1_label}|{a2}:{b2_label}"
                            merged = {**b1_values, **b2_values}
                            specs.append(ConditionalCellSpec(
                                cell_id=cell_id,
                                axes=(a1, a2),
                                values=merged,
                                min_events=config.min_cell_events,
                                min_effective_n=config.min_cell_effective_n,
                            ))
                            cell_count += 1
                        if cell_count >= config.max_cells_per_recipe:
                            break

    return tuple(specs)


def evaluate_conditional_l0_cells(
    *,
    panel: CandidateSignalPanel,
    aligned: AlignedMarketData,
    recipe: AlphaRecipe,
    cost_model: ExecutionCostModel,
    gate_config: AlphaGateConfig,
    cell_config: ConditionalCellGateConfig,
    bars_per_year: float,
    run_id: str,
) -> tuple[ConditionalCellEvidence, ...]:
    if bars_per_year <= 0.0:
        raise ValueError("bars_per_year must be positive")

    _validate_shape(panel, aligned)

    if not cell_config.enabled:
        return ()

    for axis in cell_config.axes:
        if axis not in _KNOWN_AXES:
            raise ValueError(f"unsupported conditional axis: {axis!r}")

    specs = generate_default_cell_specs(
        panel=panel,
        aligned=aligned,
        config=cell_config,
    )
    if not specs:
        return ()

    masks = build_conditional_cell_masks(
        panel=panel,
        aligned=aligned,
        specs=specs,
    )

    t, _ = aligned.close_2d.shape
    side = panel.side_hint_2d
    valid = panel.valid_mask_2d
    holding_bars = panel.expected_holding_bars

    idx_start = 1
    idx_end = t - holding_bars
    entry_full = _sparse_entry_mask(side, valid)
    event_mask = np.zeros_like(entry_full)
    if idx_start < idx_end:
        event_mask[idx_start:idx_end, :] = entry_full[idx_start:idx_end, :]

    cells: list[ConditionalCellEvidence] = []
    for spec in specs:
        cell_mask = masks.get(spec.cell_id, np.zeros_like(event_mask))
        cell_n = int(np.sum(cell_mask))

        if cell_n == 0:
            continue

        if cell_n < spec.min_events:
            gate_ev = evaluate_event_mask_gate(
                event_mask=cell_mask, panel=panel, aligned=aligned,
                recipe=recipe,
                round_trip_cost_bps=cost_model.stress_round_trip_bps(),
                gate_config=gate_config, bars_per_year=bars_per_year,
                run_id=run_id,
            )
            cells.append(ConditionalCellEvidence(
                cell_id=spec.cell_id,
                axes=spec.axes,
                values=spec.values,
                event_mask_2d=cell_mask,
                gate_evidence=gate_ev,
                failure_axis="insufficient_sample",
            ))
            continue

        n_symbols = int(np.sum(np.any(cell_mask, axis=0)))
        if n_symbols < cell_config.min_symbols_per_cell and not cell_config.allow_single_symbol_cells:
            gate_ev = evaluate_event_mask_gate(
                event_mask=cell_mask, panel=panel, aligned=aligned,
                recipe=recipe,
                round_trip_cost_bps=cost_model.stress_round_trip_bps(),
                gate_config=gate_config, bars_per_year=bars_per_year,
                run_id=run_id,
            )
            cells.append(ConditionalCellEvidence(
                cell_id=spec.cell_id,
                axes=spec.axes,
                values=spec.values,
                event_mask_2d=cell_mask,
                gate_evidence=gate_ev,
                failure_axis="insufficient_sample",
            ))
            continue

        gate_ev = evaluate_event_mask_gate(
            event_mask=cell_mask, panel=panel, aligned=aligned,
            recipe=recipe,
            round_trip_cost_bps=cost_model.stress_round_trip_bps(),
            gate_config=gate_config, bars_per_year=bars_per_year,
            run_id=run_id,
        )

        failure_axis = ""
        if not gate_ev.gate_passed:
            for reason in gate_ev.reject_reasons:
                if reason == "non_positive_lcb":
                    failure_axis = "cost_dominated"
                    break
                if reason == "weak_tstat":
                    failure_axis = "statistically_unstable"
                    break
                if reason == "excess_cost_drag":
                    failure_axis = "cost_dominated"
                    break
                if reason == "excess_turnover":
                    failure_axis = "turnover_dominated"
                    break
                if reason == "insufficient_effective_n":
                    failure_axis = "insufficient_sample"
                    break
            if not failure_axis:
                failure_axis = "unknown"

        cells.append(ConditionalCellEvidence(
            cell_id=spec.cell_id,
            axes=spec.axes,
            values=spec.values,
            event_mask_2d=cell_mask,
            gate_evidence=gate_ev,
            failure_axis=failure_axis,
        ))

    return tuple(cells)
