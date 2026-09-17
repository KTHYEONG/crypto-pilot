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
from pathlib import Path

from src.live.execution_quality import load_execution_quality_records
from src.mhs.data_provenance import (
    resolve_required_mhs_input_paths,
    validate_forward_execution_observations,
    validate_mhs_input_manifest,
)
from src.mhs.evaluation import (
    FEATURE_NAME,
    HOLDOUT_CUTOFF,
    required_cost_tiers,
    synthetic_stress_scenarios,
)
from src.mhs.evaluation.integrity import replay_ledger_certified
from src.mhs.evidence import holdout_tail_evidence, parameter_oos_split_evidence, resolved_anchored_folds
from src.mhs.params import COMMITTEE_OOS_START
from src.mhs.pipeline.context import PipelineContext
from src.mhs.reliability import (
    build_validation_track_disclosure,
    evaluate_backtest_reliability,
    gate_deployment_readiness,
)
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

    # 봉인 경계 넘은 실행만 hold-out 꼬리 성과를 채운다(미통과는 None = 정직한 신호).
    holdout_tail = (
        holdout_tail_evidence(ctx.blend_report.primary.ledger.equity, HOLDOUT_CUTOFF)
        if ctx.blend_report is not None and ctx.blend_report.primary is not None
        else None
    )

    # 관측 전용 분할: parameter-fit 경계 기준 in-sample/OOS 대비(게이트로 절대 사용되지 않음).
    parameter_oos_split = (
        parameter_oos_split_evidence(
            ctx.blend_report.primary.ledger.equity, COMMITTEE_OOS_START
        )
        if ctx.blend_report is not None and ctx.blend_report.primary is not None
        else None
    )

    run_elapsed_seconds = time.perf_counter() - ctx.run_start
    ctx.recorder.record("final_return")

    committee_member_weights = (dict(ctx._committee_weights_by_boundary["top_level"]) if ctx._committee_weights_by_boundary.get("top_level") else None)

    # Input provenance (INV-INPUT-SEAL): sealed manifest validated
    # metadata-only against the required panel/execution roster.
    data_root = Path(ctx.root or ".")
    required_paths = resolve_required_mhs_input_paths(
        data_root=data_root,
        panel_symbols=list(ctx.symbols),
        execution_symbols=list(ctx.execution_symbols),
        execution_timeframe=ctx.config.execution_timeframe,
    )
    manifest_path = Path(ctx.config.input_manifest_path) if ctx.config.input_manifest_path else None
    ctx.input_provenance = validate_mhs_input_manifest(
        manifest_path, data_root=data_root, required_paths=required_paths,
    )
    ctx.forward_provenance = None
    if ctx.config.forward_execution_quality_dir is not None and ctx.config.forward_strategy_digest is not None:
        ctx.forward_provenance = validate_forward_execution_observations(
            load_execution_quality_records(ctx.config.forward_execution_quality_dir),
            frozen_strategy_digest=ctx.config.forward_strategy_digest,
        )
    # 단일 인증 헬퍼의 판정을 그대로 전달한다(원시 플래그 직접 사용 금지).
    blend_primary = getattr(ctx.blend_report, "primary", None) if ctx.blend_report is not None else None
    blend_primary_ledger = getattr(blend_primary, "ledger", None)
    primary_valid = replay_ledger_certified(blend_primary) if blend_primary is not None else False
    primary_invalid_reasons = () if primary_valid else tuple(getattr(blend_primary_ledger, "invalid_reasons", None) or ())
    overlap_fraction = float(ctx.selection_overlap_fraction) if ctx.selection_overlap_fraction is not None else 0.0
    ctx.backtest_reliability = evaluate_backtest_reliability(
        primary_valid=primary_valid,
        primary_invalid_reasons=primary_invalid_reasons,
        selection_overlap_fraction=overlap_fraction,
        fold_committee_weight_leak=ctx.fold_committee_weight_leak,
        input_provenance=ctx.input_provenance,
        data_limitations=ctx.unsupported,
        forward_provenance=ctx.forward_provenance,
    )
    ctx.validation_tracks = build_validation_track_disclosure(
        selection_overlap_fraction=overlap_fraction,
        fold_committee_weight_leak=ctx.fold_committee_weight_leak,
        top_level_boundary=COMMITTEE_OOS_START,
        fold_boundaries=[fold.train_end for fold in resolved_anchored_folds(ctx.config)],
    )
    gated_deployment = ctx.deployment
    if ctx.deployment is not None:
        gated_deployment = gate_deployment_readiness(ctx.deployment, ctx.backtest_reliability)

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
        deflated_sharpe_ratio_fold_proxy=ctx.deflated_sharpe_ratio_fold_proxy,
        fold_committee_weight_leak=ctx.fold_committee_weight_leak,
        regime_conditional_sharpe=ctx.regime_conditional_sharpe,
        xs_rank_ic=ctx.xs_ic,
        date_clustered_regression=ctx.regression,
        horizon_diagnostics=ctx.horizon_diagnostics,
        bootstrap_ci=ctx.bootstrap_ci,
        placebo_sharpe_percentile=ctx.placebo_percentile,
        deployment_readiness=gated_deployment,
        synthetic_stress=synthetic_stress,
        participation_warnings=ctx.participation,
        termination_counts=ctx.termination_counts,
        unsupported_assumptions=ctx.unsupported,
        anchored_folds=resolved_anchored_folds(ctx.config),
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
        committee_member_weights=committee_member_weights,
        selection_overlap_fraction=(
            float(ctx.selection_overlap_fraction)
            if ctx.selection_overlap_fraction is not None
            else None
        ),
        trials_attempted_source=ctx.trials_attempted_source,
        holdout_tail=holdout_tail,
        parameter_oos_split=parameter_oos_split,
        trial_pool=ctx.trial_pool,
        backtest_reliability=ctx.backtest_reliability,
        validation_tracks=ctx.validation_tracks,
    )
