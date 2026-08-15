"""SCENARIO_MHS_RESULT_LOG_04: build_mhs_run_history_record contract.

The curated record must round-trip through json.dumps/json.loads, set
flags=None when request is None, and tolerate absent optional report fields
(blend, discovery_qualification, full_history_yearly_net_t, placebo percentile).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from src.application.research.mhs import evaluation as ev
from src.mhs.discovery import DiscoveryQualificationResult


def _book(name: str = "slow_momentum") -> ev.MhsBookReport:
    return ev.MhsBookReport(
        name=name,
        band="SLOW",
        horizon_hours=168,
        step_hours=6,
        tranche_count=4,
        n_symbols=20,
        phase=ev.PhaseDiagnosticResult(4, 0.05, 0.5, 0.05, 0.03, 0.07, 0.04, False),
        prescreen={},
        tail=ev.TailSensitivityResult(0.0, 0.0, {}, 1, 0, 0.0, 0.0, 0.0, 0.0),
        primary=None,
        stress=None,
        primary_autocorr_sharpe=0.525674,
        primary_naive_sharpe=0.6,
        primary_net_ann=0.1,
        primary_geometric_cagr=0.075,
        primary_max_drawdown=-0.2,
        primary_annualized_turnover=2.0,
        stress_naive_sharpe=0.3,
    )


def _representative_report() -> ev.MhsHorizonDiagnosticReport:
    slow = _book()
    blend = dataclasses.replace(slow, name="blend")
    fold = ev.MhsFoldReport(
        fold_index=0,
        validation_start="2021-01-01",
        validation_end="2021-06-30",
        strict=None,
        stress=None,
        primary_valid=True,
        primary_autocorr_sharpe=0.804643,
        primary_naive_sharpe=0.7,
        primary_net_ann=0.12,
        primary_geometric_cagr=0.08,
        primary_max_drawdown=-0.15,
        stress_naive_sharpe=0.25,
        decision_intents=50,
        termination_counts={},
        failures=(),
        strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
    )
    discovery = {
        "momentum": DiscoveryQualificationResult(
            selected_horizon=None,
            admitted=False,
            discovery_scores=((168, -0.14),),
            discovery_aggregate_net_t=-0.14,
            qualification_net_t=None,
            qualification_sign_consistent=None,
        )
    }
    return ev.MhsHorizonDiagnosticReport(
        feature="multi_horizon_market_state",
        status="COMPLETE",
        start="2021-01-01",
        end="2025-12-31",
        resolved_end="2025-12-31",
        partition="dev",
        execution_tiers_bps=(2.5, 5.0),
        books={"slow_momentum": slow},
        blend=blend,
        blend_target_gross=0.75,
        blend_cash_fraction=0.25,
        eligible_symbols=446,
        trials_attempted=70,
        deflated_sharpe_ratio=0.426499,
        xs_rank_ic={"mean_ic": -0.040876, "t_stat": -46.073751, "n_dates": 43727},
        date_clustered_regression={"past_beta": -0.018679, "past_t": -1.390656},
        horizon_diagnostics={"realized_vol_48h_mean": 0.091262},
        bootstrap_ci=(-0.0000067, 0.0000277),
        placebo_sharpe_percentile=0.5,
        deployment_readiness=ev.DeploymentReadinessResult(
            0.075649,
            -0.201514,
            0.375412,
            -0.01,
            -0.01,
            -0.01,
            -0.01,
            0,
            None,
            0.1645,
            0.0,
            0.0,
            {},
            {},
            {},
            False,
            False,
            False,
            False,
        ),
        synthetic_stress={},
        participation_warnings={},
        termination_counts={"MISSING_DATA": 62},
        unsupported_assumptions=(),
        anchored_folds=(),
        folds=(fold,),
        research_go=ev.MhsResearchGoResult(
            eligible=False,
            reason_codes=("PRIMARY_AUTOCORR_SHARPE_BELOW_0_6",),
            evaluated_folds=1,
            folds_passed=0,
        ),
        fill_source="OHLCV_STRICT_PROXY",
        mark_source="MARK_PRICE",
        execution_timeframe="5m",
        execution_universe_size=30,
        execution_symbols=("A",),
        run_elapsed_seconds=394.923,
        resource_measurements=(
            ev.MhsResourceMeasurement(stage="run", elapsed_ms=100, rss_bytes=9_000_000_000),
            ev.MhsResourceMeasurement(stage="folds", elapsed_ms=200, rss_bytes=9_483_913_421),
        ),
        discovery_qualification=discovery,
        realized_execution_roster_size=41.928,
        full_history_yearly_net_t={"slow_momentum": {2021: -0.144821, 2022: 0.05}},
        funding_carry_worst_year_corr=-0.26574,
    )


def test_record_round_trips_and_curates_representative_report() -> None:
    report = _representative_report()
    request = ev.MhsDiagnosticRequest(start="2021-01-01", end="2025-12-31")
    record = ev.build_mhs_run_history_record(report, request, ev.MhsOutputTier.FULL, Path("docs/results/x.json"))

    assert json.loads(json.dumps(record)) == record
    assert record["output_tier"] == "full"
    assert record["flags"]["start"] == "2021-01-01"
    assert record["perf"]["peak_rss_bytes"] == 9_483_913_421
    assert record["perf"]["realized_execution_roster_size"] == pytest.approx(41.928)
    assert record["books"]["slow_momentum"]["primary_autocorr_sharpe"] == pytest.approx(0.525674)
    assert record["blend"]["name"] == "blend"
    assert record["blend_target_gross"] == pytest.approx(0.75)
    assert record["folds"][0]["primary_autocorr_sharpe"] == pytest.approx(0.804643)
    assert record["research_go"]["eligible"] is False
    assert record["discovery_qualification"]["momentum"]["admitted"] is False
    assert record["full_history_yearly_net_t"]["slow_momentum"]["2021"] == pytest.approx(-0.144821)
    assert record["bootstrap_ci"] == pytest.approx([-0.0000067, 0.0000277], abs=1e-6)
    assert record["deployment_readiness"]["calmar"] == pytest.approx(0.375412)
    assert record["deployment_readiness"]["execution_go_eligible"] is False
    assert record["termination_counts"] == {"MISSING_DATA": 62}
    assert record["report_path"] == "docs/results/x.json"


def test_record_omits_optional_fields_without_raising() -> None:
    report = dataclasses.replace(
        _representative_report(),
        blend=None,
        discovery_qualification=None,
        full_history_yearly_net_t=None,
        placebo_sharpe_percentile=None,
        bootstrap_ci=None,
        resource_measurements=(),
    )
    record = ev.build_mhs_run_history_record(report, None, ev.MhsOutputTier.COMPACT, None)

    assert json.loads(json.dumps(record)) == record
    assert record["flags"] is None
    assert record["blend"] is None
    assert record["discovery_qualification"] is None
    assert record["full_history_yearly_net_t"] is None
    assert record["placebo_sharpe_percentile"] is None
    assert record["bootstrap_ci"] is None
    assert record["perf"]["peak_rss_bytes"] is None
    assert record["report_path"] is None


def test_record_includes_committee_diagnostic_when_committee_book() -> None:
    # SCENARIO_MHS_COMMITTEE_DIAGNOSTIC_IN_RUN_HISTORY: the durable run-history
    # record carries the committee_diagnostic dict verbatim when present and
    # null when absent -- additive, JSON-round-trippable, no key-set change.
    with_committee = dataclasses.replace(
        _representative_report(),
        committee_diagnostic={
            "admitted": ["x"], "excluded": [], "walk_forward": {"per_tier": {}},
        },
    )
    without_committee = dataclasses.replace(
        _representative_report(), committee_diagnostic=None,
    )
    request = ev.MhsDiagnosticRequest(start="2021-01-01", end="2025-12-31")
    record_with = ev.build_mhs_run_history_record(
        with_committee, request, ev.MhsOutputTier.COMPACT, None,
    )
    record_without = ev.build_mhs_run_history_record(
        without_committee, request, ev.MhsOutputTier.COMPACT, None,
    )
    assert json.loads(json.dumps(record_with)) == record_with
    assert record_with["committee_diagnostic"] == with_committee.committee_diagnostic
    assert json.loads(json.dumps(record_without)) == record_without
    assert record_without["committee_diagnostic"] is None
