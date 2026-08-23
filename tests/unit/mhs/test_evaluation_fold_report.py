from __future__ import annotations

import dataclasses
import json

import pandas as pd
import pytest

from src.application.research.mhs.evaluation import (
    GO_REASON_FOLD_GROWTH_CONCENTRATION,
    GO_REASON_PATH_DIVERGENCE,
    MhsFoldReport,
    MhsHorizonDiagnosticReport,
    MhsOutputTier,
    MhsResearchGoResult,
    _fold_blend_parity,
    _fold_growth_concentration,
    _fold_realized_risk_parity,
    _incomplete_fold_report,
    build_mhs_run_history_record,
)
from src.application.research.mhs.research_go import _mhs_research_go
from src.mhs.evidence import AnchoredPurgedFold, DeploymentReadinessResult

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
    assert reasons == (GO_REASON_PATH_DIVERGENCE,)
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
    result = _mhs_research_go((), (), (GO_REASON_PATH_DIVERGENCE,))
    assert result.eligible is False
    assert GO_REASON_PATH_DIVERGENCE in result.reason_codes
    assert GO_REASON_PATH_DIVERGENCE in result.data_integrity_reason_codes


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


def test_fold_report_regime_characterization_defaults_none() -> None:
    # SCENARIO_MHS_FOLD_REPORT_REGIME_CHARACTERIZATION_DEFAULT: MhsFoldReport
    # constructed without explicit regime_characterization kwarg has it as None;
    # _incomplete_fold_report also yields None.
    report = MhsFoldReport(
        fold_index=0, validation_start="2021-02-10", validation_end="2021-04-19",
        strict=None, stress=None, primary_valid=False, primary_autocorr_sharpe=0.0,
        primary_naive_sharpe=0.0, primary_net_ann=0.0, primary_geometric_cagr=0.0,
        primary_max_drawdown=0.0, stress_naive_sharpe=0.0, decision_intents=0,
        termination_counts={}, failures=(), strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
    )
    assert report.regime_characterization is None
    incomplete = _incomplete_fold_report(_FOLD, 0, ())
    assert incomplete.regime_characterization is None


# ---------------------------------------------------------------------------
# _fold_growth_concentration tests
# ---------------------------------------------------------------------------


def _concentration_fold(
    fold_index: int,
    cagr: float,
    primary_valid: bool = True,
) -> MhsFoldReport:
    return MhsFoldReport(
        fold_index=fold_index,
        validation_start="2021-02-10",
        validation_end="2021-04-19",
        strict=None,
        stress=None,
        primary_valid=primary_valid,
        primary_autocorr_sharpe=0.0,
        primary_naive_sharpe=0.0,
        primary_net_ann=0.0,
        primary_geometric_cagr=cagr,
        primary_max_drawdown=0.0,
        stress_naive_sharpe=0.0,
        decision_intents=0,
        termination_counts={},
        failures=(),
        strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
    )


def test_fold_growth_concentration_balanced_no_code() -> None:
    # SCENARIO_MHS_FOLD_GROWTH_CONCENTRATION_BALANCED_NO_CODE
    folds = (
        _concentration_fold(0, 0.10),
        _concentration_fold(1, 0.12),
        _concentration_fold(2, 0.11),
    )
    payload, reasons = _fold_growth_concentration(folds)
    assert reasons == ()
    assert payload["max_fold_share"] <= 0.5
    assert len(payload["folds"]) == 3
    assert payload["unmeasured"] == []


def test_fold_growth_concentration_single_fold_dominance_blocks_go() -> None:
    # SCENARIO_MHS_FOLD_GROWTH_CONCENTRATION_SINGLE_FOLD_DOMINANCE_BLOCKS_GO
    folds = (
        _concentration_fold(0, 0.118316),
        _concentration_fold(1, 0.08147),
        _concentration_fold(2, 1.050152),
    )
    payload, reasons = _fold_growth_concentration(folds)
    assert reasons == (GO_REASON_FOLD_GROWTH_CONCENTRATION,)
    assert payload["max_fold_share"] == pytest.approx(0.790, abs=1e-2)

    result = _mhs_research_go((), (), reasons)
    assert result.eligible is False
    assert GO_REASON_FOLD_GROWTH_CONCENTRATION in result.reason_codes
    assert GO_REASON_FOLD_GROWTH_CONCENTRATION not in result.data_integrity_reason_codes


def test_fold_growth_concentration_invalid_fold_unmeasured() -> None:
    # SCENARIO_MHS_FOLD_GROWTH_CONCENTRATION_INVALID_FOLD_UNMEASURED
    folds = (
        _concentration_fold(0, 0.10, primary_valid=False),
        _concentration_fold(1, 0.12),
    )
    payload, reasons = _fold_growth_concentration(folds)
    assert reasons == ()
    assert 0 in payload["unmeasured"]
    assert 1 not in payload["unmeasured"]


def test_run_history_record_includes_fold_growth_concentration() -> None:
    # SCENARIO_MHS_RUN_HISTORY_INCLUDES_FOLD_GROWTH_CONCENTRATION
    concentration = {
        "max_fold_share": 0.790,
        "max_share": 0.5,
        "folds": {"2": {"logret": 0.718, "share": 0.790}},
        "unmeasured": [],
    }
    report = _minimal_report(fold_blend_parity_value=None)
    report = dataclasses.replace(report, fold_growth_concentration=concentration)
    record = build_mhs_run_history_record(report, None, MhsOutputTier.COMPACT, None)
    assert "fold_growth_concentration" in record
    assert record["fold_growth_concentration"] == concentration
    json.dumps(record)


def test_run_history_record_fold_growth_concentration_default_none() -> None:
    # SCENARIO_MHS_RUN_HISTORY_INCLUDES_FOLD_GROWTH_CONCENTRATION: default None
    report = _minimal_report(None)
    record = build_mhs_run_history_record(report, None, MhsOutputTier.COMPACT, None)
    assert "fold_growth_concentration" in record
    assert record["fold_growth_concentration"] is None
    json.dumps(record)


# ---------------------------------------------------------------------------
# _fold_realized_risk_parity tests (observation-only diagnostic)
# ---------------------------------------------------------------------------


def _parity_fold(
    fold_index: int,
    realized_annualized_vol: float | None,
    primary_valid: bool = True,
) -> MhsFoldReport:
    return MhsFoldReport(
        fold_index=fold_index,
        validation_start="2021-02-10",
        validation_end="2021-04-19",
        strict=None,
        stress=None,
        primary_valid=primary_valid,
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
        realized_annualized_vol=realized_annualized_vol,
    )


def test_fold_realized_risk_parity_observes_divergence() -> None:
    # SCENARIO_MHS_FOLD_REALIZED_RISK_PARITY_OBSERVES_DIVERGENCE: the measured
    # baseline vols breach the tolerance but never emit a reason code.
    folds = tuple(
        _parity_fold(i, v) for i, v in enumerate([0.294, 0.304, 0.278, 0.570])
    )
    payload, reasons = _fold_realized_risk_parity(folds)
    assert reasons == ()
    max_ratio = payload["max_abs_log_risk_ratio"]
    assert 0.60 <= max_ratio <= 0.70
    assert max_ratio > payload["tolerance"]
    assert payload["tolerance"] == pytest.approx(0.35)
    assert payload["unmeasured"] == []
    assert set(payload["folds"]) == {0, 1, 2, 3}

    balanced = tuple(
        _parity_fold(i, v) for i, v in enumerate([0.49, 0.47, 0.49, 0.50])
    )
    balanced_payload, balanced_reasons = _fold_realized_risk_parity(balanced)
    assert balanced_reasons == ()
    assert balanced_payload["max_abs_log_risk_ratio"] <= 0.10


def test_fold_realized_risk_parity_unmeasured_folds_excluded() -> None:
    # SCENARIO_MHS_FOLD_REALIZED_RISK_PARITY_OBSERVES_DIVERGENCE: None-vol and
    # invalid-primary folds go to unmeasured and never into the denominator.
    folds = (
        _parity_fold(0, 0.294),
        _parity_fold(1, None),
        _parity_fold(2, 0.304, primary_valid=False),
        _parity_fold(3, 0.570),
    )
    payload, reasons = _fold_realized_risk_parity(folds)
    assert reasons == ()
    assert sorted(payload["unmeasured"]) == [1, 2]
    assert set(payload["folds"]) == {0, 3}


def test_fold_realized_risk_parity_degenerate_fail_open() -> None:
    # SCENARIO_MHS_FOLD_REALIZED_RISK_PARITY_DEGENERATE_FAIL_OPEN: fewer than
    # two measurable folds yield zero divergence, no reasons, no exception.
    single_payload, single_reasons = _fold_realized_risk_parity(
        (_parity_fold(0, 0.294),),
    )
    assert single_reasons == ()
    assert single_payload["max_abs_log_risk_ratio"] == 0.0
    empty_payload, empty_reasons = _fold_realized_risk_parity(())
    assert empty_reasons == ()
    assert empty_payload["max_abs_log_risk_ratio"] == 0.0


def test_run_history_record_includes_fold_realized_risk_parity() -> None:
    # SCENARIO_MHS_FOLD_REALIZED_RISK_PARITY_OBSERVES_DIVERGENCE: the parity
    # payload reaches the run-history record unchanged and round-trips JSON.
    parity = {
        "folds": {"3": {"realized_annualized_vol": 0.570, "log_ratio": 0.645}},
        "unmeasured": [],
        "max_abs_log_risk_ratio": 0.645,
        "tolerance": 0.35,
    }
    report = dataclasses.replace(_minimal_report(None), fold_realized_risk_parity=parity)
    record = build_mhs_run_history_record(report, None, MhsOutputTier.COMPACT, None)
    assert "fold_realized_risk_parity" in record
    assert record["fold_realized_risk_parity"] == parity
    json.dumps(record)


def test_fold_report_realized_vol_defaults_none() -> None:
    # A fold constructed without realized_annualized_vol keeps None; an
    # incomplete fold never fabricates evidence either.
    report = MhsFoldReport(
        fold_index=0, validation_start="2021-02-10", validation_end="2021-04-19",
        strict=None, stress=None, primary_valid=False, primary_autocorr_sharpe=0.0,
        primary_naive_sharpe=0.0, primary_net_ann=0.0, primary_geometric_cagr=0.0,
        primary_max_drawdown=0.0, stress_naive_sharpe=0.0, decision_intents=0,
        termination_counts={}, failures=(), strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
    )
    assert report.realized_annualized_vol is None
    assert _incomplete_fold_report(_FOLD, 0, ()).realized_annualized_vol is None