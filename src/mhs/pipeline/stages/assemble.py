"""B5: Final report assembly.

Extracted verbatim from ``evaluation.py`` lines 4174-4231 (the
``synthetic_stress`` / ``mark_source`` / ``fill_source`` computation plus the
final ``MhsHorizonDiagnosticReport(...)`` construction). No branching is added
in this block; it is a pure field mapping over the already-populated ``ctx``.

Byte-identity (I-IDENTITY-v2): ``telemetry.record("final_return")`` maps to
``ctx.recorder.record`` and ``resource_measurements=telemetry.records`` maps to
``ctx.recorder.records`` (the original ``_StageRecorder``), so the
``resource_measurements`` field is now fully compared (not excluded).
"""

from __future__ import annotations

import time

from src.application.research.mhs.evaluation import (
    FEATURE_NAME,
    phase_1_anchored_purged_folds,
    required_cost_tiers,
    synthetic_stress_scenarios,
)
from src.mhs.pipeline.context import PipelineContext
from src.mhs.report.schema import MhsHorizonDiagnosticReport
from src.mhs.telemetry import StageTelemetry


def assemble_report(ctx: PipelineContext, telemetry: StageTelemetry) -> MhsHorizonDiagnosticReport:
    """Assemble the final ``MhsHorizonDiagnosticReport`` from the populated ctx."""
    synthetic_stress = {s.name: {"description": s.description} for s in synthetic_stress_scenarios()}

    mark_source = "NOT_RUN_NO_EXECUTION_DATA"
    fill_source = "NOT_RUN_NO_EXECUTION_DATA"
    if ctx.blend_report is not None and ctx.blend_report.primary is not None:
        mark_source = ctx.blend_report.primary.ledger.mark_source
        fill_source = "OHLCV_IMMEDIATE_TAKER"

    run_elapsed_seconds = time.perf_counter() - ctx.run_start
    ctx.recorder.record("final_return")

    return MhsHorizonDiagnosticReport(
        feature=FEATURE_NAME,
        status="COMPLETE",
        start=str(ctx.start),
        end=str(ctx.end),
        resolved_end=str(ctx.resolved_end),
        partition="dev",
        execution_tiers_bps=required_cost_tiers(),
        books=ctx.books,
        blend=ctx.blend_report,
        blend_target_gross=ctx.blend_gross,
        blend_cash_fraction=ctx.blend_cash_fraction,
        eligible_symbols=len(ctx.funded),
        trials_attempted=ctx.trials_attempted,
        deflated_sharpe_ratio=ctx.deflated_sharpe_ratio,
        dsr_decomposition=ctx.dsr_decomposition,
        fold_sharpe_dispersion=ctx.fold_sharpe_dispersion,
        xs_rank_ic=ctx.xs_ic,
        date_clustered_regression=ctx.regression,
        horizon_diagnostics=ctx.horizon_diagnostics,
        bootstrap_ci=ctx.bootstrap_ci,
        placebo_sharpe_percentile=ctx.placebo_percentile,
        deployment_readiness=ctx.deployment,
        synthetic_stress=synthetic_stress,
        participation_warnings=ctx.participation,
        termination_counts=ctx.termination_counts,
        unsupported_assumptions=ctx.unsupported,
        anchored_folds=phase_1_anchored_purged_folds(),
        folds=ctx.folds,
        research_go=ctx.research_go,
        fill_source=fill_source,
        mark_source=mark_source,
        execution_timeframe=ctx.config.execution_timeframe,
        execution_universe_size=ctx.config.execution_universe_size,
        execution_symbols=tuple(ctx.execution_symbols),
        run_elapsed_seconds=run_elapsed_seconds,
        resource_measurements=ctx.recorder.records,
        worker_plan=ctx.recorder.worker_plan if ctx.recorder is not None else {},
        discovery_qualification=ctx.discovery_qualification,
        realized_execution_roster_size=ctx.realized_execution_roster_size,
        full_history_yearly_net_t=ctx.full_history_yearly_net_t,
        funding_carry_worst_year_corr=ctx.funding_carry_worst_year_corr,
        trend_sleeve_diagnostic=ctx.trend_sleeve_diagnostic,
        multi_feature_diagnostic=ctx.multi_feature_diagnostic,
        committee_diagnostic=ctx.committee_diagnostic,
        funding_dropped_symbols=ctx.funding_dropped or None,
        fold_blend_parity=ctx.fold_blend_parity,
        fold_growth_concentration=ctx.fold_growth_concentration,
        fold_realized_risk_parity=ctx.fold_realized_risk_parity,
        evidence_calibration=ctx.evidence_calibration,
        fill_mark_parity=ctx._fill_mark_parity_census,
        growth_envelope=ctx._growth_envelope_payload,
        committee_member_attribution=ctx.committee_member_attribution,
        selection_overlap_fraction=(
            float(ctx.selection_overlap_fraction)
            if ctx.selection_overlap_fraction is not None
            else None
        ),
        trials_attempted_source=ctx.trials_attempted_source,
    )
