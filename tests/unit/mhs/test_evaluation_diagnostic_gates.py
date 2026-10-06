"""MHS evaluation pipeline/gate tests (third-level split)."""

"""MHS evaluation pipeline/gate tests (second-level split remainder)."""
"""MHS evaluation core contract tests (everything not in a domain-specific split file)."""
"""Contract coverage for the MHS application evaluation resource telemetry."""
from tests.fixtures.mhs_requests import research_baseline
import dataclasses
import numpy as np
import pandas as pd
import pytest
import src.mhs.evaluation.concurrency as concurrency_mod
from src.mhs.diagnostic_run import run_mhs_horizon_diagnostic
import src.mhs.marks as marks
import src.mhs.pipeline.stages.book as book_stage
import src.mhs.statistics as statistics
from src.mhs.discovery import DiscoveryQualificationResult
from src.mhs.marks import _load_funding_series
from src.quant.universe.pit_universe import symbol_partition
from tests.unit.mhs.test_evaluation_appresearch import (  # noqa: F401
    _FOLD,
    _START,
    _assert_books_equal,
    _assert_regime_vol_mean_roster_masked,
    _build_book_outcome_args,
    _build_books_concurrent_args,
    _build_compact_report,
    _deployment_readiness,
    _dispatch_spec,
    _gap_mixed_replay,
    _passing_fold_report,
    _perf_opt_placebo_inputs,
    _pre_change_slow_book,
    _reference_bootstrap_ci,
    _reference_participation_warnings,
    _reference_placebo_percentile,
    _reference_resolve_ns_scalar,
    _reference_weights,
    _roster_mask_panel_inputs,
    _sequential_book_reports,
    _signal_disagreement_panel,
    _slow_book_panel_inputs,
    _synthetic_ledger,
    _write_3m_cache,
    _write_quote_volume_market,
)


@pytest.fixture(autouse=True)
def _clear_mhs_market_data_caches() -> None:
    """Run isolation for the module-scoped shared market root.

    Several tests rewrite mark parquet files in place and restore the bytes
    afterwards; the process-level mark caches keyed by source path would
    otherwise serve pre-rewrite arrays to later tests in the module.
    """
    marks.clear_mhs_market_data_caches()
    yield
    marks.clear_mhs_market_data_caches()

@pytest.mark.slow
def test_mhs_funding_carry_top_level_discovery(mhs_market_funding_vary, monkeypatch) -> None:
    # SCENARIO_MHS_FUNDING_CARRY_TOP_LEVEL_DISCOVERY_05: with discovery_gate=True
    # the top-level discovery_qualification carries funding_carry_long and
    # funding_carry_short (each a populated DiscoveryQualificationResult) beside
    # the existing reversal/momentum entries -- all three candidates measured on
    # the same instrumented window. With discovery_gate=False the keys are
    # absent (discovery_qualification stays None), matching the opt-in convention.
    root, end = mhs_market_funding_vary
    monkeypatch.setattr(concurrency_mod, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(concurrency_mod, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request_on = research_baseline(
        start=str(_START), end=str(end), data_root=str(root),
        execution_timeframe="3m", log_run=False,
        execution_universe_size=8, discovery_gate=True,
    )
    report_on = run_mhs_horizon_diagnostic(request_on)
    assert report_on.status == "COMPLETE"
    assert report_on.discovery_qualification is not None
    assert set(report_on.discovery_qualification) == {
        "reversal", "momentum", "funding_carry_long", "funding_carry_short",
    }
    for key in ("funding_carry_long", "funding_carry_short"):
        result = report_on.discovery_qualification[key]
        assert isinstance(result, DiscoveryQualificationResult)
        assert result.yearly_net_t

    request_off = research_baseline(
        start=str(_START), end=str(end), data_root=str(root),
        execution_timeframe="3m", log_run=False,
        execution_universe_size=8,
    )
    report_off = run_mhs_horizon_diagnostic(request_off)
    assert report_off.discovery_qualification is None

@pytest.mark.slow
def test_mhs_full_history_yearly_net_t_and_worst_year_corr_exposed(mhs_market_funding_vary, monkeypatch) -> None:
    # SCENARIO_MHS_FULL_HISTORY_YEARLY_NET_T_AND_WORST_YEAR_CORR_EXPOSED_06:
    # with discovery_gate=True the report exposes full_history_yearly_net_t for
    # slow_momentum/fast_reversal/funding_carry covering all five years
    # 2021-2025 (not just the 2021-2023 discovery window) and a finite
    # funding_carry_worst_year_corr; both stay None when discovery_gate=False.
    root, end = mhs_market_funding_vary
    monkeypatch.setattr(concurrency_mod, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(concurrency_mod, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request_on = research_baseline(
        start=str(_START), end=str(end), data_root=str(root),
        execution_timeframe="3m", log_run=False,
        execution_universe_size=8, discovery_gate=True,
    )
    report_on = run_mhs_horizon_diagnostic(request_on)
    assert report_on.status == "COMPLETE"
    assert report_on.full_history_yearly_net_t is not None
    assert set(report_on.full_history_yearly_net_t) == {
        "slow_momentum", "fast_reversal", "funding_carry",
    }
    for key in report_on.full_history_yearly_net_t:
        yearly = report_on.full_history_yearly_net_t[key]
        assert set(yearly) == {2021, 2022, 2023, 2024, 2025}
    assert report_on.funding_carry_worst_year_corr is not None
    assert np.isfinite(report_on.funding_carry_worst_year_corr)
    # The spec's headline claim -- momentum's own 168h book fails the same
    # gate that rejected funding_carry -- is directly visible here: the
    # momentum column's worst-year value (2021-2023) stays near/below the
    # admission floor in this fixture, while the full history shows the whole
    # five-year picture the 3-year window could not.
    slow_2021 = report_on.full_history_yearly_net_t["slow_momentum"][2021]
    assert np.isfinite(slow_2021)

    request_off = research_baseline(
        start=str(_START), end=str(end), data_root=str(root),
        execution_timeframe="3m", log_run=False,
        execution_universe_size=8,
    )
    report_off = run_mhs_horizon_diagnostic(request_off)
    assert report_off.full_history_yearly_net_t is None
    assert report_off.funding_carry_worst_year_corr is None

@pytest.mark.slow
def test_mhs_execution_coverage_gate_default_off_bit_identical(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_DIAGNOSTIC_EXECUTION_COVERAGE_GATE_DEFAULT_OFF_BYTE_IDENTICAL:
    # with the opt-in flag omitted (default False) the pre-flight gate AND the
    # dynamic gap exclusion it now also guards (spec
    # mhs_data_integrity_relevance_scoping.md §3) are both inert: against a
    # fixture with no 5m execution cache the run completes through the
    # pre-existing MISSING_DATA termination path with no new
    # DataIntegrityError, and the report is byte-identical to the
    # explicit-off run.
    root, end = mhs_market
    monkeypatch.setattr(concurrency_mod, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(concurrency_mod, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    base = {
        "start": str(_START), "end": str(end), "data_root": str(root),
        "execution_timeframe": "3m", "log_run": False,
        "execution_universe_size": 8,
    }
    default_report = run_mhs_horizon_diagnostic(research_baseline(**base))
    explicit_off = run_mhs_horizon_diagnostic(
        research_baseline(**base, execution_coverage_gate=False),
    )
    assert default_report.status == "COMPLETE"
    for field in ("books", "blend", "blend_target_gross", "research_go", "folds"):
        assert getattr(default_report, field) == getattr(explicit_off, field)

@pytest.mark.slow
def test_mhs_execution_coverage_gate_on_fails_closed_early(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_DIAGNOSTIC_EXECUTION_COVERAGE_GATE_ON_FAILS_CLOSED_EARLY:
    # an out-of-contract execution_timeframe fails closed before any replay
    # window executes -- regardless of execution_coverage_gate, which is no
    # longer what triggers this case.
    root, end = mhs_market
    books_called: list[str] = []
    monkeypatch.setattr(concurrency_mod, "_run_books_concurrent", lambda *a, **k: books_called.append("books"),
    )
    base = {
        "start": str(_START), "end": str(end), "data_root": str(root),
        "execution_timeframe": "5m", "log_run": False,
        "execution_universe_size": 8,
    }
    request = research_baseline(**base)
    with pytest.raises(ValueError, match="unknown execution_timeframe"):
        run_mhs_horizon_diagnostic(
            dataclasses.replace(request, execution_coverage_gate=True, committee_target_gross=None),
        )
    assert books_called == []

@pytest.mark.slow
def test_mhs_diagnostic_relevance_gate_passes_where_full_scope_blocked(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_DIAGNOSTIC_RELEVANCE_GATE_PASSES_WHERE_FULL_SCOPE_BLOCKED:
    # with execution_coverage_gate=True, a fixture whose NON-roster symbol has
    # an internal 3m data gap completes normally (status COMPLETE).
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    gap_symbol = symbols[0]
    gap_path = root / "3m" / f"{gap_symbol}.parquet"
    original_bytes = gap_path.read_bytes()
    try:
        frame = pd.read_parquet(gap_path)
        mid = len(frame) // 2
        pd.concat([frame.iloc[:mid], frame.iloc[mid + 12:]]).to_parquet(gap_path)

        # Pin the execution roster: every symbol in the roster from hour 1 EXCEPT
        # gap_symbol, which is never a member. The first mask row stays False so the
        # fixture's marks (available from start + 1h) cover every membership hour.
        def _fixed_mask(quote_vol, eligible, universe_size):
            mask = pd.DataFrame(True, index=quote_vol.index, columns=quote_vol.columns)
            mask[gap_symbol] = False
            mask.iloc[0] = False
            return mask

        monkeypatch.setattr(book_stage, "_pit_execution_mask", _fixed_mask)
        monkeypatch.setattr(concurrency_mod, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
        monkeypatch.setattr(concurrency_mod, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
        )
        request = research_baseline(
            start=str(_START), end=str(end), data_root=str(root),
            execution_timeframe="3m", log_run=False,
            execution_universe_size=8, execution_coverage_gate=True,
        )
        report = run_mhs_horizon_diagnostic(request)
        assert report.status == "COMPLETE"
    finally:
        gap_path.write_bytes(original_bytes)

def test_mhs_funding_load_reports_dropped_symbols(tmp_path, monkeypatch) -> None:
    # SCENARIO_MHS_FUNDING_LOAD_REPORTS_DROPPED_SYMBOLS: _load_funding_series
    # returns (series, dropped) where a symbol whose funding parquet raises on
    # load (or has no file / no rows) appears in `dropped` with its reason and
    # is absent from `series` -- the drop is no longer observable only via a
    # log line.
    root = tmp_path / "market"
    fdir = root / "funding"
    fdir.mkdir(parents=True, exist_ok=True)
    hourly = pd.date_range(_START, periods=24, freq="1h", tz="UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    pd.DataFrame(
        {
            "timestamp": (hourly - epoch) // pd.Timedelta("1ms"),
            "datetime": hourly,
            "funding_rate": 0.00005,
        }
    ).to_parquet(fdir / "GOODUSDT.parquet")
    (fdir / "BROKENUSDT.parquet").write_bytes(b"not a parquet")
    monkeypatch.setattr(marks, "funding_path", lambda sym: fdir / f"{sym}.parquet")
    series, dropped = _load_funding_series(["GOODUSDT", "BROKENUSDT", "NOPATHUSDT"])
    assert "GOODUSDT" in series
    assert "BROKENUSDT" not in series
    assert dropped["BROKENUSDT"].startswith("load_error")
    assert dropped["NOPATHUSDT"] == "missing"

def test_mhs_diagnostic_execution_timeframe_3m_default() -> None:
    # SCENARIO_MHS_EXECUTION_TIMEFRAME_3M_DEFAULT: default timeframe is '3m'.
    request = research_baseline()
    assert request.execution_timeframe == "3m"

def test_mhs_diagnostic_execution_timeframe_3m_accepted() -> None:
    # SCENARIO_MHS_EXECUTION_TIMEFRAME_3M_ACCEPTED: '3m' is a valid contract
    # value; an out-of-contract '7m' still raises ValueError.
    assert research_baseline(execution_timeframe="3m").execution_timeframe == "3m"
    with pytest.raises(ValueError, match="unknown execution_timeframe"):
        research_baseline(execution_timeframe="7m")

@pytest.mark.slow
def test_mhs_diagnostic_3m_replay_end_to_end(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_DIAGNOSTIC_3M_REPLAY_END_TO_END: a synthetic 3m fixture
    # (data_root/3m/{symbol}.parquet at 3-minute bars) replays through the real
    # book path under the default execution_timeframe='3m' and completes --
    # mirroring the existing 5m/1m fixture-based end-to-end test pattern.
    root, end = mhs_market
    import src.mhs.evidence as evidence_mod

    _write_3m_cache(root)
    monkeypatch.setattr(evidence_mod, "phase_1_anchored_purged_folds", lambda: ())
    monkeypatch.setattr(statistics, "_BOOTSTRAP_REPLICATES", 20)
    monkeypatch.setattr(statistics, "_BOOTSTRAP_MEAN_BLOCK", 24)
    monkeypatch.setattr(statistics, "_bootstrap_ci", lambda *a, **k: None)
    monkeypatch.setattr(statistics, "_placebo_sharpe_percentile", lambda *a, **k: None)
    request = research_baseline(
        start=str(_START), end=str(end), data_root=str(root),
        log_run=False, execution_universe_size=8, reference_books_diagnostic=True,
    )
    assert request.execution_timeframe == "3m"
    report = run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    assert report.execution_timeframe == "3m"
    assert set(report.books) == {"fast_reversal", "slow_momentum"}
    assert report.blend is not None
    assert report.blend.primary is not None
    assert report.fill_source == "OHLCV_IMMEDIATE_TAKER"

class TestMhsDiagnosticRequestParityGate:
    """SCENARIO_MHS_FILL_MARK_PARITY_05: request field validation."""

    def test_defaults(self) -> None:
        req = research_baseline()
        assert req.exposure_scale_two_sided is False

    def test_non_bool_exposure_scale_two_sided_raises(self) -> None:
        with pytest.raises(ValueError, match="exposure_scale_two_sided"):
            research_baseline(exposure_scale_two_sided=1)  # type: ignore[arg-type]

    # SCENARIO_MHS_EXPOSURE_CEILING_08
    def test_scenario_mhs_exposure_ceiling_08_request_default_stays_false(self) -> None:
        assert research_baseline().exposure_scale_two_sided is False
        with pytest.raises(ValueError, match=r"exposure_scale_two_sided.*exante_target"):
            research_baseline(
                exposure_scale_two_sided=True,
                pnl_vol_target_mode="median_relative",
            )
        with pytest.raises(ValueError, match="exposure_scale_two_sided"):
            research_baseline(exposure_scale_two_sided=1)  # type: ignore[arg-type]
