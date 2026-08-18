from __future__ import annotations

import json

import pandas as pd
import pytest

from src.application.research.mhs.evaluation import (
    MHS_GO_REASON_PATH_DIVERGENCE,
    MhsFoldReport,
    MhsHorizonDiagnosticReport,
    MhsOutputTier,
    MhsResearchGoResult,
    _fold_blend_parity,
    _incomplete_fold_report,
    _mhs_research_go,
    build_mhs_run_history_record,
)
from src.mhs.evaluation import AnchoredPurgedFold, DeploymentReadinessResult

_FOLD = AnchoredPurgedFold(
    pd.Timestamp("2021-01-01", tz="UTC"),
    pd.Timestamp("2021-01-31", tz="UTC"),
    pd.Timestamp("2021-02-10", tz="UTC"),
    pd.Timestamp("2021-04-19 08:00", tz="UTC"),
    168,
    168,
)


def _fold_report(
    fold_index: int = 0,
    book_structure: dict[str, float] | None = None,
) -> MhsFoldReport:
    return MhsFoldReport(
        fold_index=fold_index,
        validation_start="2021-02-10",
        validation_end="2021-04-19",
        strict=None,
        stress=None,
        primary_valid=False,
        primary_autocorr_sharpe=0.0,
        primary_naive_sharpe=0.0,
        primary_net_ann=0.0,
        primary_geometric_cagr=0.0,
        primary_max_drawdown=0.0,
        stress_naive_sharpe=0.0,
        decision_intents=0,
        termination_counts={},
        failures=(),
        strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
        book_structure=book_structure,
    )


def _trace(holdings_mean: float, gross_mean: float = 0.53) -> dict[str, float]:
    return {
        "n_rows": 100.0,
        "gross_mean": gross_mean,
        "holdings_mean": holdings_mean,
        "holdings_max": holdings_mean,
        "holdings_growth_slope": 0.0,
    }


def test_fold_report_fast_horizon_fields_default() -> None:
    # SCENARIO_MHS_FOLD_REPORT_FAST_HORIZON_FIELDS_DEFAULT: MhsFoldReport
    # constructed without explicit fast_horizon_hours/fast_horizon_source
    # defaults to (48, "frozen_default"), mirroring the slow_horizon_* fields'
    # existing defaults so every pre-existing call site stays valid.
    report = MhsFoldReport(
        fold_index=0, validation_start="2021-02-10", validation_end="2021-04-19",
        strict=None, stress=None, primary_valid=False, primary_autocorr_sharpe=0.0,
        primary_naive_sharpe=0.0, primary_net_ann=0.0, primary_geometric_cagr=0.0,
        primary_max_drawdown=0.0, stress_naive_sharpe=0.0, decision_intents=0,
        termination_counts={}, failures=(), strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
    )
    assert report.fast_horizon_hours == 48
    assert report.fast_horizon_source == "frozen_default"
    assert report.slow_horizon_hours == 168
    assert report.slow_horizon_source == "frozen_default"


def test_incomplete_fold_report_keeps_fast_horizon_default() -> None:
    # _incomplete_fold_report (fail-closed fold) must keep the same defaults so
    # a fold that cannot be replayed never fabricates a discovery source.
    report = _incomplete_fold_report(_FOLD, 0, ())
    assert report.fast_horizon_hours == 48
    assert report.fast_horizon_source == "frozen_default"


def test_fold_report_records_fast_discovery_source() -> None:
    # A fold run resolved with a fast fold-scoped override records the selected
    # horizon and source, mirroring the slow_horizon_* recording path.
    report = MhsFoldReport(
        fold_index=0, validation_start="2021-02-10", validation_end="2021-04-19",
        strict=None, stress=None, primary_valid=False, primary_autocorr_sharpe=0.0,
        primary_naive_sharpe=0.0, primary_net_ann=0.0, primary_geometric_cagr=0.0,
        primary_max_drawdown=0.0, stress_naive_sharpe=0.0, decision_intents=0,
        termination_counts={}, failures=(), strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
        fast_horizon_hours=96, fast_horizon_source="fold_train_only_discovery",
    )
    assert report.fast_horizon_hours == 96
    assert report.fast_horizon_source == "fold_train_only_discovery"
    assert report.slow_horizon_hours == 168
    assert report.slow_horizon_source == "frozen_default"


def test_fold_blend_parity_matching_traces_passes() -> None:
    # SCENARIO_MHS_EVAL_INTEGRITY_PARITY_GATE: matching fold/blend traces yield
    # no reason code and a zero log ratio.
    fold = _fold_report(0, _trace(42.0))
    payload, reasons = _fold_blend_parity({0: _trace(42.0)}, (fold,))
    assert reasons == ()
    assert payload["max_abs_log_holdings_ratio"] == pytest.approx(0.0)
    assert payload["max_abs_log_gross_ratio"] == pytest.approx(0.0)
    assert payload["tolerance"] == pytest.approx(0.25)


def test_fold_blend_parity_production_divergence_blocks_go() -> None:
    # SCENARIO_MHS_EVAL_INTEGRITY_PARITY_GATE: the measured production divergence
    # (fold 67.2 vs blend 149.9, log ratio -0.802) emits the path-divergence code.
    fold = _fold_report(0, _trace(67.2, 0.5))
    payload, reasons = _fold_blend_parity({0: _trace(149.9, 0.5)}, (fold,))
    assert reasons == (MHS_GO_REASON_PATH_DIVERGENCE,)
    assert payload["max_abs_log_holdings_ratio"] == pytest.approx(0.802, abs=1e-3)
    assert payload["folds"][0]["holdings_log_ratio"] == pytest.approx(-0.802, abs=1e-3)


def test_fold_blend_parity_unmeasured_fold_no_code() -> None:
    # SCENARIO_MHS_EVAL_INTEGRITY_PARITY_GATE: a fold with book_structure None
    # is recorded under 'unmeasured' and does not by itself emit the code.
    fold = _fold_report(0, None)
    payload, reasons = _fold_blend_parity({}, (fold,))
    assert reasons == ()
    assert payload["unmeasured"] == [0]


def test_research_go_path_divergence_is_data_integrity_blocker() -> None:
    # SCENARIO_MHS_EVAL_INTEGRITY_PARITY_GATE: the path-divergence code must
    # surface in BOTH reason_codes and data_integrity_reason_codes.
    result = _mhs_research_go((), (), (MHS_GO_REASON_PATH_DIVERGENCE,))
    assert result.eligible is False
    assert MHS_GO_REASON_PATH_DIVERGENCE in result.reason_codes
    assert MHS_GO_REASON_PATH_DIVERGENCE in result.data_integrity_reason_codes


def _minimal_report(fold_blend_parity_value=None) -> MhsHorizonDiagnosticReport:
    deployment = DeploymentReadinessResult(
        geometric_cagr=0.0,
        max_drawdown=0.0,
        calmar=0.0,
        expected_shortfall=0.0,
        worst_1d=0.0,
        worst_7d=0.0,
        worst_event=0.0,
        time_under_water_bars=0,
        recovery_bars=None,
        probability_final_wealth_below_initial=0.0,
        probability_mdd_over_20pct=0.0,
        probability_mdd_over_30pct=0.0,
        leverage_ruin_probabilities={},
        concentration={},
        participation_warnings={},
        research_go_eligible=False,
        execution_go_eligible=False,
        pilot_go_eligible=False,
        scale_go_eligible=False,
    )
    return MhsHorizonDiagnosticReport(
        feature="mhs",
        status="COMPLETE",
        start="2021-01-01",
        end="2025-12-31",
        resolved_end="2025-12-31",
        partition="dev",
        execution_tiers_bps=(4.18,),
        books={},
        blend=None,
        blend_target_gross=0.0,
        blend_cash_fraction=0.0,
        eligible_symbols=0,
        trials_attempted=0,
        deflated_sharpe_ratio=None,
        xs_rank_ic={},
        date_clustered_regression={},
        horizon_diagnostics={},
        bootstrap_ci=None,
        placebo_sharpe_percentile=None,
        deployment_readiness=deployment,
        synthetic_stress={},
        participation_warnings={},
        termination_counts={},
        unsupported_assumptions=(),
        anchored_folds=(),
        folds=(),
        research_go=MhsResearchGoResult(
            eligible=True, reason_codes=(), evaluated_folds=0, folds_passed=0,
        ),
        fill_source="OHLCV_IMMEDIATE_TAKER",
        mark_source="cache_required",
        execution_timeframe="1m",
        execution_universe_size=30,
        execution_symbols=(),
        run_elapsed_seconds=0.0,
        fold_blend_parity=fold_blend_parity_value,
    )


def test_run_history_record_includes_fold_blend_parity() -> None:
    # SCENARIO_MHS_EVAL_INTEGRITY_HISTORY_PAYLOAD: the parity payload reaches
    # the run-history record unchanged and round-trips through json.dumps.
    parity = {
        "max_abs_log_holdings_ratio": 0.802,
        "max_abs_log_gross_ratio": 0.0,
        "tolerance": 0.25,
        "folds": {"0": {"holdings_log_ratio": -0.802, "gross_log_ratio": 0.0}},
        "unmeasured": [],
    }
    report = _minimal_report(parity)
    record = build_mhs_run_history_record(report, None, MhsOutputTier.COMPACT, None)
    assert "fold_blend_parity" in record
    assert record["fold_blend_parity"] == report.fold_blend_parity
    json.dumps(record)


def test_run_history_record_fold_blend_parity_default_none() -> None:
    # SCENARIO_MHS_EVAL_INTEGRITY_HISTORY_PAYLOAD: a report constructed without
    # the field yields 'fold_blend_parity': None, never a missing key.
    report = _minimal_report(None)
    record = build_mhs_run_history_record(report, None, MhsOutputTier.COMPACT, None)
    assert "fold_blend_parity" in record
    assert record["fold_blend_parity"] is None
    json.dumps(record)