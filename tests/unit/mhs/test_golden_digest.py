"""SCENARIO_MHS_PERF_P0_02_DIGEST_STRENGTH: compact golden digest gate.

- Perturbing any single ledger value by one ULP changes the series sha256.
- ``assert_report_digest_identical`` raises AssertionError naming that exact
  (replay_id, series, index) with golden/actual reprs from the landmarks.
- Total digest+summary bytes for a 24-replay x 161941-bar report stay far
  below 400 KB (the full payload measured 1085.61 MB -- >= 2700x reduction).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from src.application.research.mhs.contracts import (
    MhsBookReport,
    MhsFoldReport,
    MhsHorizonDiagnosticReport,
    MhsResearchGoResult,
)
from src.mhs.evidence import (
    DeploymentReadinessResult,
    PhaseDiagnosticResult,
    TailSensitivityResult,
)
from src.mhs.execution import (
    SimulatedInventoryLedgerResult,
    StrategyExecutionReplayResult,
)

N_BARS = 161_941


def _synthetic_replay(n: int = N_BARS, seed: int = 20260807) -> StrategyExecutionReplayResult:
    idx = pd.date_range("2021-01-01", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(seed)
    equity = pd.Series(1.0 + np.cumsum(rng.normal(0.0, 1e-6, n)), index=idx)
    ledger = SimulatedInventoryLedgerResult(
        equity=equity,
        net_returns=equity.pct_change().fillna(0.0),
        simulated_units=None,
        mark_to_market_pnl=pd.Series(rng.normal(0.0, 1e-7, n), index=idx),
        funding_charge=pd.Series(np.full(n, 1e-9), index=idx),
        fee_charge=pd.Series(np.full(n, 2e-9), index=idx),
        fill_turnover=pd.Series(np.zeros(n), index=idx),
        fill_source="OHLCV_IMMEDIATE_TAKER",
        mark_source="MARK_PRICE",
        primary_valid=True,
        invalid_reasons=(),
    )
    return StrategyExecutionReplayResult(
        simulated_fills=pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]")}),
        ledger=ledger,
        simulated_units=pd.DataFrame(index=idx[:0]),
        simulated_notional_weights=pd.DataFrame(index=idx[:0]),
        fill_source="OHLCV_IMMEDIATE_TAKER",
        mark_source="MARK_PRICE",
        submit_times=pd.Series(dtype="datetime64[ns, UTC]"),
        fill_times=pd.Series(dtype="datetime64[ns, UTC]"),
        fill_count=0,
        unfilled_count=0,
        fallback_count=0,
        all_intent_shortfall_bps=0.0,
        forced_exit_count=0,
        forced_exit_notional=0.0,
        termination_counts={"MISSING_DATA": 0, "UNKNOWN_TERMINATION": 0},
        unsupported_assumptions=(),
        elapsed_seconds=0.0,
    )


def _synthetic_book(replay: StrategyExecutionReplayResult) -> MhsBookReport:
    return MhsBookReport(
        name="fast_reversal", band="FAST", horizon_hours=24, step_hours=6,
        tranche_count=1, n_symbols=1,
        phase=PhaseDiagnosticResult(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False),
        prescreen={}, tail=TailSensitivityResult(0.0, 0.0, {}, 1, 0, 0.0, 0.0, 0.0, 0.0),
        primary=replay, stress=replay,
        primary_autocorr_sharpe=0.1, primary_naive_sharpe=0.1, primary_net_ann=0.01,
        primary_geometric_cagr=0.01, primary_max_drawdown=-0.01,
        primary_annualized_turnover=1.0, stress_naive_sharpe=0.1,
        touch=replay, touch_naive_sharpe=0.1,
        ladder=replay, ladder_naive_sharpe=0.1,
        patient_reference=replay, patient_reference_naive_sharpe=-0.7,
        pre_vol_target_reference=replay, pre_vol_target_reference_naive_sharpe=0.2,
    )


def _synthetic_fold(index: int, replay: StrategyExecutionReplayResult) -> MhsFoldReport:
    return MhsFoldReport(
        fold_index=index,
        validation_start=f"2021-0{index + 1}-01",
        validation_end=f"2021-0{index + 2}-28",
        strict=replay,
        stress=replay,
        primary_valid=True,
        primary_autocorr_sharpe=0.5,
        primary_naive_sharpe=0.6,
        primary_net_ann=0.02,
        primary_geometric_cagr=0.02,
        primary_max_drawdown=-0.05,
        stress_naive_sharpe=0.4,
        decision_intents=10,
        termination_counts={"MISSING_DATA": 0},
        failures=(),
        strict_elapsed_seconds=1.0,
        stress_elapsed_seconds=1.0,
    )


@pytest.fixture(scope="module")
def big_report():
    """24 reachable replay slots x 161941 ledger bars (the captured shape)."""
    replay = _synthetic_replay()
    book_a = dataclasses.replace(_synthetic_book(replay), name="fast_reversal")
    book_b = dataclasses.replace(_synthetic_book(replay), name="slow_momentum")
    blend = dataclasses.replace(_synthetic_book(replay), name="blend")
    # 2 books x 6 fields + blend x 6 fields + 3 folds x 2 fields = 24 replays.
    folds = tuple(_synthetic_fold(i, replay) for i in range(3))
    # 2 books x 6 fields + blend x 6 fields + 3 folds x 2 fields = 24 replays.
    return MhsHorizonDiagnosticReport(
        feature="multi_horizon_market_state", status="COMPLETE", start="2021-01-01",
        end="2021-06-30", resolved_end="2021-06-30", partition="dev",
        execution_tiers_bps=(2.5, 5.0), books={"fast_reversal": book_a, "slow_momentum": book_b},
        blend=blend,
        blend_target_gross=0.0, blend_cash_fraction=0.0, eligible_symbols=1,
        trials_attempted=1, deflated_sharpe_ratio=None, xs_rank_ic={},
        date_clustered_regression={}, horizon_diagnostics={}, bootstrap_ci=None,
        placebo_sharpe_percentile=None,
        deployment_readiness=DeploymentReadinessResult(
            0.01, -0.01, 1.0, -0.01, -0.01, -0.01, -0.01, 0, None, 0.5, 0.0, 0.0, {}, {},
            {}, False, False, False, False,
        ),
        synthetic_stress={}, participation_warnings={}, termination_counts={},
        unsupported_assumptions=(), anchored_folds=(), folds=folds,
        research_go=MhsResearchGoResult(False, (), 0, 0),
        fill_source="OHLCV_IMMEDIATE_TAKER", mark_source="MARK_PRICE",
        execution_timeframe="1m", execution_universe_size=1,
        execution_symbols=("A",), run_elapsed_seconds=0.1,
    )


def _perturb_one_ulp(report: MhsHorizonDiagnosticReport, index: int) -> MhsHorizonDiagnosticReport:
    """Bump one equity value by exactly one ULP at a landmark index."""
    replay = report.books["fast_reversal"].primary
    assert replay is not None
    equity = replay.ledger.equity
    values = equity.to_numpy(dtype="float64").copy()
    values[index] = float(np.nextafter(values[index], np.inf))
    perturbed_equity = pd.Series(values, index=equity.index)
    perturbed_ledger = dataclasses.replace(replay.ledger, equity=perturbed_equity)
    perturbed_replay = dataclasses.replace(replay, ledger=perturbed_ledger)
    perturbed_book = dataclasses.replace(report.books["fast_reversal"], primary=perturbed_replay)
    return dataclasses.replace(report, books={"fast_reversal": perturbed_book, **{
        k: v for k, v in report.books.items() if k != "fast_reversal"
    }})


def _landmark_index(n: int, k: int) -> int:
    """Mirror the digest's stratified landmark selection."""
    return int(np.unique(np.linspace(0, n - 1, min(64, n)).astype(int))[k])


def test_ulp_perturbation_changes_series_sha256(big_report) -> None:
    """One-ULP changes flip the sha256 of exactly the touched series."""
    from tests.fixtures.golden.digest import build_report_digest

    landmark_index = _landmark_index(N_BARS, 10)
    baseline = build_report_digest(big_report)["fast_reversal_primary"]
    perturbed = build_report_digest(
        _perturb_one_ulp(big_report, landmark_index)
    )["fast_reversal_primary"]

    assert perturbed["equity"]["sha256"] != baseline["equity"]["sha256"]
    assert perturbed["net_returns"]["sha256"] == baseline["net_returns"]["sha256"]
    assert perturbed["funding_charge"]["sha256"] == baseline["funding_charge"]["sha256"]


def test_assert_digest_names_exact_divergence(big_report) -> None:
    """A mismatch raises naming (replay_id, series, index, golden, actual)."""
    from tests.fixtures.golden.compare import assert_report_digest_identical
    from tests.fixtures.golden.digest import build_report_digest

    golden = build_report_digest(big_report)
    landmark_index = _landmark_index(N_BARS, 12)
    actual_report = _perturb_one_ulp(big_report, landmark_index)

    with pytest.raises(AssertionError) as excinfo:
        assert_report_digest_identical(golden, actual_report)

    message = str(excinfo.value)
    assert "fast_reversal_primary" in message
    assert "equity" in message
    assert f"[{landmark_index}]" in message
    assert "!=" in message


def test_missing_replay_in_actual_raises(big_report) -> None:
    """A golden replay absent from the actual report fails closed."""
    from tests.fixtures.golden.compare import assert_report_digest_identical
    from tests.fixtures.golden.digest import build_report_digest

    golden = build_report_digest(big_report)
    with pytest.raises(AssertionError, match="missing"):
        assert_report_digest_identical(golden, _no_blend(big_report))


def _no_blend(report: MhsHorizonDiagnosticReport) -> MhsHorizonDiagnosticReport:
    stripped = dataclasses.replace(
        report, blend=None,
        books={name: dataclasses.replace(b, primary=None) for name, b in report.books.items()},
    )
    return stripped


def test_digest_and_summary_size_budget(big_report, tmp_path) -> None:
    """digest+summary bytes < 400 KB for a 24-replay x 161941-bar report."""
    from tests.fixtures.golden.digest import build_report_digest, build_report_summary

    digest = build_report_digest(big_report)
    summary = build_report_summary(big_report)
    digest_bytes = len(json.dumps(digest).encode("utf-8"))
    summary_bytes = len(json.dumps(summary).encode("utf-8"))
    assert digest_bytes + summary_bytes < 400 * 1024
    assert summary_bytes < 100 * 1024
    # And every one of the 24 replay slots is covered by the digest.
    assert len(digest) >= 24
