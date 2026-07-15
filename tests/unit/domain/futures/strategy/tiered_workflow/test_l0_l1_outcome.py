from __future__ import annotations

from src.domain.futures.strategy.tiered_workflow.l0_l1_outcome import (
    L0L1OutcomeReport,
    TfEvidenceMetrics,
    build_l0_l1_outcome_report,
    classify_tf_outcome,
)


def _metrics(
    *,
    timeframe: str = "12h",
    runtime_valid: bool = True,
    grid_valid: bool = True,
    l0_candidate_count: int = 69,
    l0_survivor_count: int = 12,
    l0_reject_counts: tuple[tuple[str, int], ...] = (),
    total_fold_count: int = 4,
    passed_fold_count: int = 4,
    matched_event_count: int = 84_747,
    decision_points_per_calendar_year: float = 120.0,
    effective_symbol_count: float = 3.0,
    gross_probe_lcb_bps: float | None = 12.0,
    mean_execution_cost_bps: float | None = 2.0,
    mean_funding_bps: float | None = 1.0,
    net_probe_lcb_bps: float | None = 9.0,
    breakeven_bps: float = 7.5,
) -> TfEvidenceMetrics:
    return TfEvidenceMetrics(
        timeframe=timeframe,
        runtime_valid=runtime_valid,
        grid_valid=grid_valid,
        l0_candidate_count=l0_candidate_count,
        l0_survivor_count=l0_survivor_count,
        l0_reject_counts=l0_reject_counts,
        total_fold_count=total_fold_count,
        passed_fold_count=passed_fold_count,
        matched_event_count=matched_event_count,
        decision_points_per_calendar_year=decision_points_per_calendar_year,
        effective_symbol_count=effective_symbol_count,
        gross_probe_lcb_bps=gross_probe_lcb_bps,
        mean_execution_cost_bps=mean_execution_cost_bps,
        mean_funding_bps=mean_funding_bps,
        net_probe_lcb_bps=net_probe_lcb_bps,
        breakeven_bps=breakeven_bps,
    )


def test_no_passed_fold_is_sentinel_not_negative_infinity() -> None:
    report = classify_tf_outcome(
        _metrics(
            passed_fold_count=0,
            gross_probe_lcb_bps=None,
            mean_execution_cost_bps=None,
            mean_funding_bps=None,
            net_probe_lcb_bps=None,
        )
    )
    assert report.outcome == "l1_insufficient_evidence"
    assert report.sentinel_reason == "no_passed_fold"
    assert report.deploy_allowed is False


def test_deployable_tf() -> None:
    report = classify_tf_outcome(_metrics())
    assert report.outcome == "deployable"
    assert report.deploy_allowed is True
    assert report.sentinel_reason is None


def test_l0_no_economic_edge() -> None:
    rejects = (("weak_tstat", 10), ("fdr_rejected", 5))
    report = classify_tf_outcome(_metrics(l0_survivor_count=0, l0_reject_counts=rejects))
    assert report.outcome == "l0_no_economic_edge"
    assert report.deploy_allowed is False


def test_data_or_contract_failure() -> None:
    report = classify_tf_outcome(
        _metrics(l0_survivor_count=0, l0_reject_counts=(("invalid_shape", 3), ("lookahead_risk", 1)))
    )
    assert report.outcome == "data_or_contract_failure"
    assert report.deploy_allowed is False


def test_no_net_edge() -> None:
    report = classify_tf_outcome(_metrics(net_probe_lcb_bps=4.0, breakeven_bps=7.5))
    assert report.outcome == "l1_no_net_edge"
    assert report.deploy_allowed is False


def test_temporal_instability() -> None:
    report = classify_tf_outcome(_metrics(passed_fold_count=2, total_fold_count=4))
    assert report.outcome == "l1_temporal_instability"
    assert report.deploy_allowed is False


def test_insufficient_evidence_low_density() -> None:
    report = classify_tf_outcome(_metrics(decision_points_per_calendar_year=0.5))
    assert report.outcome == "l1_insufficient_evidence"
    assert report.deploy_allowed is False


def test_invalid_measurement() -> None:
    report = classify_tf_outcome(_metrics(runtime_valid=False))
    assert report.outcome == "invalid_measurement"
    assert report.deploy_allowed is False


def test_build_report_aggregates() -> None:
    m1 = _metrics(timeframe="2h")
    m2 = _metrics(timeframe="4h", passed_fold_count=0, net_probe_lcb_bps=None, gross_probe_lcb_bps=None,
                  mean_execution_cost_bps=None, mean_funding_bps=None)
    report = build_l0_l1_outcome_report((m1, m2))
    assert isinstance(report, L0L1OutcomeReport)
    assert report.deployable_tfs == ("2h",)
    assert report.invalid_measurement is False
    assert len(report.by_tf) == 2


def test_build_report_invalid_measurement_flag() -> None:
    m1 = _metrics(timeframe="2h", runtime_valid=False)
    m2 = _metrics(timeframe="4h")
    report = build_l0_l1_outcome_report((m1, m2))
    assert report.invalid_measurement is True


def test_build_report_no_deployable() -> None:
    m1 = _metrics(timeframe="2h", net_probe_lcb_bps=3.0, breakeven_bps=7.5)
    m2 = _metrics(timeframe="4h", passed_fold_count=1, total_fold_count=4)
    report = build_l0_l1_outcome_report((m1, m2))
    assert len(report.deployable_tfs) == 0


def test_classification_priority_blocks_lower() -> None:
    report = classify_tf_outcome(
        _metrics(runtime_valid=False, l0_survivor_count=0, passed_fold_count=0)
    )
    assert report.outcome == "invalid_measurement"


def test_6h_unresolved_placeholder() -> None:
    report = classify_tf_outcome(_metrics(timeframe="6h"))
    assert report.outcome == "deployable"
