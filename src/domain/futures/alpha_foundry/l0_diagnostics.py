"""L0 diagnostic-only orchestration: wires edge_failure/conditional_cells/execution_arms
without touching L1/L2 handoff. [ADR_20260709_L0_CONDITIONAL_DIAGNOSTIC_WIRING]"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any

import numpy as np
import pandas as pd

from src.domain.futures.alpha_foundry.conditional_cells import (
    build_calibrated_cell_masks,
    evaluate_event_mask_gate,
    generate_default_cell_specs,
)
from src.domain.futures.alpha_foundry.contracts import (
    AlphaFoundryEvidenceRow,
    AlphaFoundryRuntimeConfig,
    AlphaGateConfig,
    AlphaGateEvidence,
    AlphaRecipe,
)
from src.domain.futures.alpha_foundry.edge_failure import classify_edge_failure_rows
from src.domain.futures.alpha_foundry.execution_arms import (
    evaluate_recipe_under_arm,
    resolve_execution_cost_arms,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel

_REQUIRED_CANONICAL_COLUMNS: frozenset[str] = frozenset(
    {
        "cost_drag_ratio",
        "gross_lcb_bps",
        "nw_tstat",
        "turnover_per_year",
        "effective_n",
    }
)


def run_l0_diagnostic_pass(
    *,
    canonical_evidences: Sequence[AlphaGateEvidence],
    panel_by_rid: Mapping[str, CandidateSignalPanel],
    aligned: AlignedMarketData,
    recipes: Mapping[str, AlphaRecipe],
    cost_model: ExecutionCostModel,
    gate_config: AlphaGateConfig,
    runtime_config: AlphaFoundryRuntimeConfig,
    run_id: str,
) -> tuple[AlphaFoundryEvidenceRow, ...]:
    """Returns EXTRA evidence rows only. Never mutates canonical_evidences, never
    referenced by passed_recipe_ids/handoff_decisions (those are already finalized by
    the caller before this runs). Returns () if enable_failure_attribution is False.
    """
    if not runtime_config.enable_failure_attribution:
        return ()

    # [LIMIT-04] Fail-fast on missing columns
    ev_dicts = [{f.name: getattr(ev, f.name) for f in fields(AlphaGateEvidence)} for ev in canonical_evidences]
    df = pd.DataFrame(ev_dicts)
    missing = _REQUIRED_CANONICAL_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"canonical evidence frame missing columns: {missing}")

    # Step 1: Classify failure axes
    failure_df = classify_edge_failure_rows(df)
    failure_map: dict[str, str] = {}
    for _, row in failure_df.iterrows():
        failure_map[str(row["recipe_id"])] = str(row["failure_axis"])

    from src.domain.futures.optimization.metrics import _bars_per_year_for_tf

    diagnostic_rows: list[AlphaFoundryEvidenceRow] = []

    # Step 2: Conditional cell search (if enabled)
    if runtime_config.enable_conditional_l0_cells:
        cell_config = runtime_config.conditional_cell
        diag_config = runtime_config.diagnostic

        qualifying: list[tuple[str, AlphaGateEvidence, float]] = []
        for ev in canonical_evidences:
            if ev.handoff_tier != "blocked":
                continue
            failure_axis = failure_map.get(ev.recipe_id, "unknown")
            if failure_axis not in diag_config.failure_axes_for_cell_search:
                continue
            qualifying.append((ev.recipe_id, ev, abs(ev.net_lcb_bps)))

        qualifying.sort(key=lambda x: x[2])  # ascending abs(net_lcb_bps)
        qualifying = qualifying[: diag_config.max_diagnostic_recipes]

        for recipe_id, canon_ev, _ in qualifying:
            panel = panel_by_rid.get(recipe_id)
            recipe = recipes.get(recipe_id)
            if panel is None or recipe is None:
                continue
            bars_per_year = _bars_per_year_for_tf(recipe.timeframe)

            specs = generate_default_cell_specs(
                panel=panel,
                aligned=aligned,
                config=cell_config,
            )
            if not specs:
                continue

            calibrated_masks = build_calibrated_cell_masks(
                panel=panel,
                aligned=aligned,
                specs=specs,
                calibration_fraction=diag_config.calibration_fraction,
            )

            cell_evidences: list[dict[str, Any]] = []
            for spec in specs:
                result = calibrated_masks.get(spec.cell_id)
                if result is None:
                    continue
                cell_mask, calib_n, eval_n = result
                cell_n = int(np.sum(cell_mask))
                if cell_n == 0:
                    continue
                if calib_n < cell_config.min_cell_events or cell_n < spec.min_events:
                    continue

                gate_ev = evaluate_event_mask_gate(
                    event_mask=cell_mask,
                    panel=panel,
                    aligned=aligned,
                    recipe=recipe,
                    round_trip_cost_bps=cost_model.stress_round_trip_bps(),
                    gate_config=gate_config,
                    bars_per_year=bars_per_year,
                    run_id=run_id,
                )

                cell_evidences.append(
                    {
                        "gate_ev": gate_ev,
                        "spec": spec,
                        "calib_n": calib_n,
                        "eval_n": eval_n,
                    }
                )

            # [LIMIT-02] BH-FDR correction across all cells in this run
            if cell_evidences:
                all_pvals = []
                for ce in cell_evidences:
                    tstat = ce["gate_ev"].nw_tstat
                    pval = 2.0 * (1.0 - _normal_cdf(abs(tstat)))
                    all_pvals.append(pval)

                sorted_idx = np.argsort(all_pvals)
                sorted_pvals = np.array(all_pvals)[sorted_idx]
                n_tests = len(all_pvals)
                max_k = 0
                for k in range(1, n_tests + 1):
                    bh_thresh = gate_config.fdr_alpha * k / max(n_tests, 1)
                    if sorted_pvals[k - 1] <= bh_thresh:
                        max_k = k
                survived = np.zeros(n_tests, dtype=bool)
                for i in range(max_k):
                    survived[sorted_idx[i]] = True

                for i, ce in enumerate(cell_evidences):
                    if not survived[i]:
                        continue
                    gate_ev = ce["gate_ev"]
                    spec = ce["spec"]
                    cell_id = spec.cell_id
                    axes_str = "|".join(spec.axes)
                    values_str = f"calib_n={ce['calib_n']},eval_n={ce['eval_n']}," + ",".join(
                        f"{k}={v}" for k, v in spec.values.items()
                    )
                    diagnostic_rows.append(
                        _build_diagnostic_row(
                            recipe_id=f"{recipe_id}::cell={cell_id}",
                            gate_ev=gate_ev,
                            run_id=run_id,
                            canon_ev=canon_ev,
                            cell_id=cell_id,
                            cell_axes=axes_str,
                            cell_values=values_str,
                            execution_style="",
                            selected_for_l1=False,
                            failure_axis=failure_map.get(recipe_id, "unknown"),
                        )
                    )

    # Step 3: Execution-arm re-costing (if enabled)
    if runtime_config.enable_execution_arms:
        diag_config = runtime_config.diagnostic
        for ev in canonical_evidences:
            failure_axis = failure_map.get(ev.recipe_id, "unknown")
            if failure_axis not in diag_config.failure_axes_for_arm_search:
                continue
            if ev.handoff_tier != "blocked":
                continue

            recipe = recipes.get(ev.recipe_id)
            panel = panel_by_rid.get(ev.recipe_id)
            if recipe is None or panel is None:
                continue
            bars_per_year = _bars_per_year_for_tf(recipe.timeframe)

            arms = resolve_execution_cost_arms(
                panel=panel,
                aligned=aligned,
                recipe=recipe,
                cost_model=cost_model,
                config=runtime_config.execution_arm,
            )

            for arm in arms:
                if arm.style == "taker_now":
                    continue
                arm_gate_ev = evaluate_recipe_under_arm(
                    panel=panel,
                    aligned=aligned,
                    recipe=recipe,
                    arm=arm,
                    gate_config=gate_config,
                    bars_per_year=bars_per_year,
                    run_id=run_id,
                )
                diagnostic_rows.append(
                    _build_diagnostic_row(
                        recipe_id=f"{ev.recipe_id}::arm={arm.style}",
                        gate_ev=arm_gate_ev,
                        run_id=run_id,
                        canon_ev=ev,
                        cell_id="",
                        cell_axes="",
                        cell_values="",
                        execution_style=arm.style,
                        selected_for_l1=False,
                        failure_axis=failure_axis,
                    )
                )

    return tuple(diagnostic_rows)


def _build_diagnostic_row(
    *,
    recipe_id: str,
    gate_ev: AlphaGateEvidence,
    run_id: str,
    canon_ev: AlphaGateEvidence,
    cell_id: str,
    cell_axes: str,
    cell_values: str,
    execution_style: str,
    selected_for_l1: bool,
    failure_axis: str,
) -> AlphaFoundryEvidenceRow:
    return AlphaFoundryEvidenceRow(
        run_id=run_id,
        timeframe=gate_ev.timeframe or canon_ev.timeframe,
        family=gate_ev.family or canon_ev.family,
        variant=gate_ev.variant or canon_ev.variant,
        recipe_id=recipe_id,
        archetype=gate_ev.archetype or canon_ev.archetype,
        n_events=gate_ev.n_events,
        effective_n=gate_ev.effective_n,
        mean_gross_bps=gate_ev.mean_gross_bps,
        mean_cost_bps=gate_ev.mean_cost_bps,
        mean_net_bps=gate_ev.mean_net_bps,
        gross_lcb_bps=gate_ev.gross_lcb_bps,
        net_lcb_bps=gate_ev.net_lcb_bps,
        nw_tstat=gate_ev.nw_tstat,
        rank_ic=gate_ev.rank_ic,
        rank_ic_tstat=gate_ev.rank_ic_tstat,
        cost_drag_ratio=gate_ev.cost_drag_ratio,
        turnover_per_year=gate_ev.turnover_per_year,
        novelty_corr_max=gate_ev.novelty_corr_max,
        incremental_rank_ic=gate_ev.incremental_rank_ic,
        compute_cost_score=gate_ev.compute_cost_score,
        event_hit_rate=gate_ev.event_hit_rate,
        payoff_skew=gate_ev.payoff_skew,
        xs_spread_lcb_bps=gate_ev.xs_spread_lcb_bps,
        liquidity_cost_stress_bps=gate_ev.liquidity_cost_stress_bps,
        bootstrap_lcb_bps=gate_ev.bootstrap_lcb_bps,
        bootstrap_agree=gate_ev.bootstrap_agree,
        gate_passed=gate_ev.gate_passed,
        handoff_tier=gate_ev.handoff_tier,
        selected_for_l1=selected_for_l1,
        reject_reasons="|".join(gate_ev.reject_reasons),
        soft_flags="|".join(gate_ev.soft_flags),
        bucket_key=f"{gate_ev.family}:{gate_ev.timeframe}",
        bucket_rank=-1,
        redundant_with="",
        bucket_eff_test_count=0.0,
        global_eff_test_count=0.0,
        l1_priority_score=0.0,
        l1_budget_units=0,
        tf_coverage_count=0,
        sign_agreement_ratio=0.0,
        corroboration_tier="",
        stage_label="diagnostic",
        created_at_ms=0,
        cell_id=cell_id,
        cell_axes=cell_axes,
        cell_values=cell_values,
        execution_style=execution_style,
        fill_probability=1.0,
        adverse_selection_bps=0.0,
        failure_axis=failure_axis,
        failure_axes="",
    )


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def _erf(x: float) -> float:
    """Approximation of the error function (Horner's method)."""
    # Abramowitz and Stegun 7.1.26
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    inner = ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592
    y = 1.0 - inner * t * np.exp(-x * x)
    return float(np.sign(x) * y if x != 0.0 else 0.0)
