from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

TfOutcomeClass: TypeAlias = Literal[
    "invalid_measurement",
    "data_or_contract_failure",
    "l0_no_economic_edge",
    "l1_insufficient_evidence",
    "l1_no_net_edge",
    "l1_temporal_instability",
    "unresolved_coupling",
    "deployable",
]


@dataclass(frozen=True, slots=True)
class TfEvidenceMetrics:
    timeframe: str
    runtime_valid: bool
    grid_valid: bool
    l0_candidate_count: int
    l0_survivor_count: int
    l0_reject_counts: tuple[tuple[str, int], ...]
    total_fold_count: int
    passed_fold_count: int
    matched_event_count: int
    decision_points_per_calendar_year: float
    effective_symbol_count: float
    gross_probe_lcb_bps: float | None
    mean_execution_cost_bps: float | None
    mean_funding_bps: float | None
    net_probe_lcb_bps: float | None
    breakeven_bps: float
    cross_tf_first_divergence_stage: str | None = None


@dataclass(frozen=True, slots=True)
class TfOutcomeReport:
    timeframe: str
    outcome: TfOutcomeClass
    primary_reason: str
    blockers: tuple[str, ...]
    deploy_allowed: bool
    sentinel_reason: str | None


@dataclass(frozen=True, slots=True)
class L0L1OutcomeReport:
    by_tf: tuple[TfOutcomeReport, ...]
    deployable_tfs: tuple[str, ...]
    invalid_measurement: bool


def classify_tf_outcome(metrics: TfEvidenceMetrics) -> TfOutcomeReport:
    if not metrics.runtime_valid or not metrics.grid_valid:
        return TfOutcomeReport(
            timeframe=metrics.timeframe,
            outcome="invalid_measurement",
            primary_reason="runtime_or_grid_audit_failed",
            blockers=(),
            deploy_allowed=False,
            sentinel_reason=None,
        )

    if metrics.l0_survivor_count == 0:
        l0_reject_reasons = dict(metrics.l0_reject_counts)
        data_contract_rejects = {"invalid_shape", "lookahead_risk", "missing_required_field"}
        if l0_reject_reasons.keys() & data_contract_rejects:
            return TfOutcomeReport(
                timeframe=metrics.timeframe,
                outcome="data_or_contract_failure",
                primary_reason="l0_contract_reject",
                blockers=tuple(l0_reject_reasons.keys()),
                deploy_allowed=False,
                sentinel_reason=None,
            )
        return TfOutcomeReport(
            timeframe=metrics.timeframe,
            outcome="l0_no_economic_edge",
            primary_reason="l0_no_survivors",
            blockers=tuple(l0_reject_reasons.keys()),
            deploy_allowed=False,
            sentinel_reason=None,
        )

    if metrics.passed_fold_count == 0:
        return TfOutcomeReport(
            timeframe=metrics.timeframe,
            outcome="l1_insufficient_evidence",
            primary_reason="no_passed_fold",
            blockers=(),
            deploy_allowed=False,
            sentinel_reason="no_passed_fold",
        )

    if metrics.decision_points_per_calendar_year < 1.0 or metrics.effective_symbol_count < 1.0:
        return TfOutcomeReport(
            timeframe=metrics.timeframe,
            outcome="l1_insufficient_evidence",
            primary_reason="low_decision_density",
            blockers=(),
            deploy_allowed=False,
            sentinel_reason=None,
        )

    if metrics.net_probe_lcb_bps is not None and metrics.net_probe_lcb_bps <= metrics.breakeven_bps:
        return TfOutcomeReport(
            timeframe=metrics.timeframe,
            outcome="l1_no_net_edge",
            primary_reason=f"net_lcb_{metrics.net_probe_lcb_bps:.3f}_bps_le_breakeven_{metrics.breakeven_bps}",
            blockers=(),
            deploy_allowed=False,
            sentinel_reason=None,
        )

    if metrics.passed_fold_count < metrics.total_fold_count:
        return TfOutcomeReport(
            timeframe=metrics.timeframe,
            outcome="l1_temporal_instability",
            primary_reason=f"{metrics.passed_fold_count}/{metrics.total_fold_count}_folds_passed",
            blockers=(),
            deploy_allowed=False,
            sentinel_reason=None,
        )

    return TfOutcomeReport(
        timeframe=metrics.timeframe,
        outcome="deployable",
        primary_reason=f"net_lcb_{metrics.net_probe_lcb_bps:.3f}_bps",
        blockers=(),
        deploy_allowed=True,
        sentinel_reason=None,
    )


def build_l0_l1_outcome_report(
    metrics_by_tf: tuple[TfEvidenceMetrics, ...],
) -> L0L1OutcomeReport:
    """[ADR_20260715_L0_L1_NATIVE_CONTRACT] Classify measured TF outcomes."""
    reports = tuple(classify_tf_outcome(m) for m in metrics_by_tf)
    deployable_tfs = tuple(r.timeframe for r in reports if r.deploy_allowed)
    invalid = any(r.outcome == "invalid_measurement" for r in reports)
    return L0L1OutcomeReport(
        by_tf=reports,
        deployable_tfs=deployable_tfs,
        invalid_measurement=invalid,
    )
