"""MHS evaluation pipeline/gate tests (third-level split)."""

"""MHS evaluation pipeline/gate tests (second-level split remainder)."""
"""MHS evaluation core contract tests (everything not in a domain-specific split file)."""
"""Contract coverage for the MHS application evaluation resource telemetry."""
import dataclasses
import numpy as np
import pandas as pd
import pytest
from src.application.data import mhs_execution_collection as mec
from src.application.research.mhs import evaluation as ev
import src.application.research.mhs.marks as marks
import src.application.research.mhs.statistics as statistics
from src.application.research.mhs.evaluation import (
    MhsDiagnosticRequest,
)
from src.common.errors import DataIntegrityError
from src.research.universe.pit_universe import symbol_partition
from tests.unit.application.research.mhs.test_evaluation import (  # noqa: F401
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

@pytest.mark.slow
def test_mhs_funding_carry_top_level_discovery(mhs_market_funding_vary, monkeypatch) -> None:
    # SCENARIO_MHS_FUNDING_CARRY_TOP_LEVEL_DISCOVERY_05: with discovery_gate=True
    # the top-level discovery_qualification carries funding_carry_long and
    # funding_carry_short (each a populated DiscoveryQualificationResult) beside
    # the existing reversal/momentum entries -- all three candidates measured on
    # the same instrumented window. With discovery_gate=False the keys are
    # absent (discovery_qualification stays None), matching the opt-in convention.
    root, end = mhs_market_funding_vary
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request_on = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, discovery_gate=True,
    )
    report_on = ev.run_mhs_horizon_diagnostic(request_on)
    assert report_on.status == "COMPLETE"
    assert report_on.discovery_qualification is not None
    assert set(report_on.discovery_qualification) == {
        "reversal", "momentum", "funding_carry_long", "funding_carry_short",
    }
    for key in ("funding_carry_long", "funding_carry_short"):
        result = report_on.discovery_qualification[key]
        assert isinstance(result, ev.DiscoveryQualificationResult)
        assert result.yearly_net_t

    request_off = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    report_off = ev.run_mhs_horizon_diagnostic(request_off)
    assert report_off.discovery_qualification is None

@pytest.mark.slow
def test_mhs_full_history_yearly_net_t_and_worst_year_corr_exposed(mhs_market_funding_vary, monkeypatch) -> None:
    # SCENARIO_MHS_FULL_HISTORY_YEARLY_NET_T_AND_WORST_YEAR_CORR_EXPOSED_06:
    # with discovery_gate=True the report exposes full_history_yearly_net_t for
    # slow_momentum/fast_reversal/funding_carry covering all five years
    # 2021-2025 (not just the 2021-2023 discovery window) and a finite
    # funding_carry_worst_year_corr; both stay None when discovery_gate=False.
    root, end = mhs_market_funding_vary
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request_on = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, discovery_gate=True,
    )
    report_on = ev.run_mhs_horizon_diagnostic(request_on)
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

    request_off = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    report_off = ev.run_mhs_horizon_diagnostic(request_off)
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
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    base = {
        "start": str(_START), "end": str(end), "data_root": str(root),
        "mark_mode": "cache_required", "execution_timeframe": "5m", "log_run": False,
        "execution_universe_size": 8,
    }
    default_report = ev.run_mhs_horizon_diagnostic(MhsDiagnosticRequest(**base))
    explicit_off = ev.run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(**base, execution_coverage_gate=False),
    )
    assert default_report.status == "COMPLETE"
    for field in ("books", "blend", "blend_target_gross", "research_go", "folds"):
        assert getattr(default_report, field) == getattr(explicit_off, field)

@pytest.mark.slow
def test_mhs_execution_coverage_gate_on_fails_closed_early(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_DIAGNOSTIC_EXECUTION_COVERAGE_GATE_ON_FAILS_CLOSED_EARLY:
    # a fixture whose execution_timeframe (5m) has no parquet files at all for
    # ANY funded symbol dynamically excludes every roster member (spec
    # mhs_data_integrity_relevance_scoping.md §3) and, since that empties the
    # entire roster rather than trimming a few noisy symbols, the always-on
    # total-exclusion safety net raises DataIntegrityError naming the
    # timeframe/data_root before any replay window executes -- regardless of
    # execution_coverage_gate, which is no longer what triggers this case.
    root, end = mhs_market
    books_called: list[str] = []
    monkeypatch.setattr(
        ev, "_run_books_concurrent", lambda *a, **k: books_called.append("books"),
    )
    base = {
        "start": str(_START), "end": str(end), "data_root": str(root),
        "mark_mode": "cache_required", "execution_timeframe": "5m", "log_run": False,
        "execution_universe_size": 8,
    }
    request = MhsDiagnosticRequest(**base)
    with pytest.raises(DataIntegrityError, match="removed every roster member"):
        ev.run_mhs_horizon_diagnostic(
            dataclasses.replace(request, execution_coverage_gate=True, committee_target_gross=None),
        )
    assert books_called == []

@pytest.mark.slow
def test_mhs_diagnostic_relevance_gate_passes_where_full_scope_blocked(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_DIAGNOSTIC_RELEVANCE_GATE_PASSES_WHERE_FULL_SCOPE_BLOCKED:
    # with execution_coverage_gate=True, a fixture whose NON-roster symbol has
    # an internal 3m data gap completes normally (status COMPLETE), whereas the
    # same fixture blocks under the old full-universe gate -- reproducing the
    # measured 36/36 false-positive the relevance scope removes.
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

        monkeypatch.setattr(ev, "_pit_execution_mask", _fixed_mask)
        monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
        monkeypatch.setattr(
            ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
        )
        request = MhsDiagnosticRequest(
            start=str(_START), end=str(end), data_root=str(root),
            mark_mode="cache_required", execution_timeframe="3m", log_run=False,
            execution_universe_size=8, execution_coverage_gate=True,
        )
        report = ev.run_mhs_horizon_diagnostic(request)
        assert report.status == "COMPLETE"

        # The same fixture blocks under the old full-universe scope (the gapped
        # symbol is funded, so it was part of the Cartesian product gate).
        with pytest.raises(DataIntegrityError, match=gap_symbol):
            mec.assert_execution_data_coverage(
                symbols, "3m", str(_START), str(end), root=str(root),
            )
    finally:
        gap_path.write_bytes(original_bytes)

@pytest.mark.slow
def test_mhs_diagnostic_mark_gate_fails_before_replay(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_DIAGNOSTIC_MARK_GATE_FAILS_BEFORE_REPLAY: with
    # execution_coverage_gate=True, a fixture where a roster symbol's mark data
    # starts after its first roster hour raises DataIntegrityError naming that
    # symbol, and raises before any execution replay window is materialized.
    # The missing span is kept well under DYNAMIC_GAP_EXCLUSION_HOURS (720h)
    # so the default dynamic gap exclusion (spec
    # mhs_data_integrity_relevance_scoping.md §3) leaves this symbol in the
    # mask and the strict opt-in gate is the one that catches it -- see
    # test_mhs_diagnostic_large_gap_auto_excluded_not_raised for the >=720h case.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    late_symbol = symbols[0]
    hourly = pd.date_range(_START, end, freq="1h", tz="UTC")
    late_idx = pd.date_range(hourly[100], end, freq="1h", tz="UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    mdir = root / "markPriceKlines" / "1h"
    mdir.mkdir(parents=True, exist_ok=True)
    mark_path = mdir / f"{late_symbol}.parquet"
    # ``mhs_market`` is a module-scoped shared root: overwriting the mark file
    # in place would permanently truncate this symbol's schema for every later
    # test in the module, so the original bytes (or absence) are restored.
    original_mark_bytes = mark_path.read_bytes() if mark_path.exists() else None
    try:
        pd.DataFrame(
            {
                "timestamp": (late_idx - epoch) // pd.Timedelta("1ms"),
                "datetime": late_idx,
                "close": 100.0,
            }
        ).to_parquet(mark_path)

        def _all_roster(quote_vol, eligible, universe_size):
            mask = pd.DataFrame(True, index=quote_vol.index, columns=quote_vol.columns)
            mask.iloc[0] = False
            return mask

        monkeypatch.setattr(ev, "_pit_execution_mask", _all_roster)
        window_calls = {"n": 0}
        original_windows = ev._iter_mhs_execution_windows

        def counting(*args, **kwargs):
            window_calls["n"] += 1
            return original_windows(*args, **kwargs)

        monkeypatch.setattr(ev, "_iter_mhs_execution_windows", counting)
        request = MhsDiagnosticRequest(
            start=str(_START), end=str(end), data_root=str(root),
            mark_mode="cache_required", execution_timeframe="1m", log_run=False,
            execution_universe_size=8, execution_coverage_gate=True,
        )
        with pytest.raises(DataIntegrityError) as exc_info:
            ev.run_mhs_horizon_diagnostic(request)
        assert late_symbol in str(exc_info.value)
        assert window_calls["n"] == 0
    finally:
        if original_mark_bytes is None:
            mark_path.unlink(missing_ok=True)
        else:
            mark_path.write_bytes(original_mark_bytes)

@pytest.mark.slow
def test_mhs_diagnostic_large_gap_auto_excluded_not_raised(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_DYNAMIC_GAP_EXCLUSION_LARGE_GAP_NO_RAISE: a roster symbol
    # whose mark data is missing for >= DYNAMIC_GAP_EXCLUSION_HOURS (720h)
    # is silently excluded from the execution mask by the default (always-on)
    # apply_dynamic_mark_gap_exclusion instead of raising -- even with
    # execution_coverage_gate=True, since that gate runs AFTER dynamic
    # exclusion and only ever sees what remains in the mask. Companion to
    # test_mhs_diagnostic_mark_gate_fails_before_replay (sub-threshold case).
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    late_symbol = symbols[0]
    hourly = pd.date_range(_START, end, freq="1h", tz="UTC")
    late_idx = pd.date_range(hourly[len(hourly) // 2], end, freq="1h", tz="UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    mdir = root / "markPriceKlines" / "1h"
    mdir.mkdir(parents=True, exist_ok=True)
    mark_path = mdir / f"{late_symbol}.parquet"
    # ``mhs_market`` is a module-scoped shared root: overwriting the mark file
    # in place would permanently truncate this symbol's data for every later
    # test in the module, so the original bytes (or absence) are restored.
    original_mark_bytes = mark_path.read_bytes() if mark_path.exists() else None
    try:
        pd.DataFrame(
            {
                "timestamp": (late_idx - epoch) // pd.Timedelta("1ms"),
                "datetime": late_idx,
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
            }
        ).to_parquet(mark_path)

        def _all_roster(quote_vol, eligible, universe_size):
            mask = pd.DataFrame(True, index=quote_vol.index, columns=quote_vol.columns)
            mask.iloc[0] = False
            return mask

        monkeypatch.setattr(ev, "_pit_execution_mask", _all_roster)
        request = MhsDiagnosticRequest(
            start=str(_START), end=str(end), data_root=str(root),
            mark_mode="cache_required", execution_timeframe="1m", log_run=False,
            execution_universe_size=8, execution_coverage_gate=True,
        )
        report = ev.run_mhs_horizon_diagnostic(request)
        assert report.status == "COMPLETE"
    finally:
        if original_mark_bytes is None:
            mark_path.unlink(missing_ok=True)
        else:
            mark_path.write_bytes(original_mark_bytes)

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
    series, dropped = ev._load_funding_series(["GOODUSDT", "BROKENUSDT", "NOPATHUSDT"])
    assert "GOODUSDT" in series
    assert "BROKENUSDT" not in series
    assert dropped["BROKENUSDT"].startswith("load_error")
    assert dropped["NOPATHUSDT"] == "missing"

def test_mhs_diagnostic_execution_timeframe_3m_default() -> None:
    # SCENARIO_MHS_EXECUTION_TIMEFRAME_3M_DEFAULT: default timeframe is '3m'.
    request = MhsDiagnosticRequest()
    assert request.execution_timeframe == "3m"

def test_mhs_diagnostic_execution_timeframe_3m_accepted() -> None:
    # SCENARIO_MHS_EXECUTION_TIMEFRAME_3M_ACCEPTED: '3m' is a valid contract
    # value; an out-of-contract '7m' still raises ValueError.
    assert MhsDiagnosticRequest(execution_timeframe="3m").execution_timeframe == "3m"
    with pytest.raises(ValueError, match="unknown execution_timeframe"):
        MhsDiagnosticRequest(execution_timeframe="7m")

@pytest.mark.slow
def test_mhs_diagnostic_3m_replay_end_to_end(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_DIAGNOSTIC_3M_REPLAY_END_TO_END: a synthetic 3m fixture
    # (data_root/3m/{symbol}.parquet at 3-minute bars) replays through the real
    # book path under the default execution_timeframe='3m' and completes --
    # mirroring the existing 5m/1m fixture-based end-to-end test pattern.
    root, end = mhs_market
    _write_3m_cache(root)
    monkeypatch.setattr(ev, "phase_1_anchored_purged_folds", lambda: ())
    monkeypatch.setattr(statistics, "_BOOTSTRAP_REPLICATES", 20)
    monkeypatch.setattr(statistics, "_BOOTSTRAP_MEAN_BLOCK", 24)
    monkeypatch.setattr(statistics, "_bootstrap_ci", lambda *a, **k: None)
    monkeypatch.setattr(statistics, "_placebo_sharpe_percentile", lambda *a, **k: None)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", log_run=False, execution_universe_size=8,
    )
    assert request.execution_timeframe == "3m"
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    assert report.execution_timeframe == "3m"
    assert set(report.books) == {"fast_reversal", "slow_momentum"}
    assert report.blend is not None
    assert report.blend.primary is not None
    assert report.fill_source == "OHLCV_IMMEDIATE_TAKER"

class TestFillMarkParityEligibility:
    """SCENARIO_MHS_FILL_MARK_PARITY_04: _fill_mark_parity_eligibility ALPACA regression."""

    def test_alpaca_shape_divergence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.application.research.mhs.evaluation import _fill_mark_parity_eligibility

        idx = pd.date_range("2025-04-01", periods=10, freq="1h", tz="UTC")
        symbols = ["GOOD", "FROZEN", "ALSO_GOOD"]
        close = pd.DataFrame(
            {
                "GOOD": [100.0] * 10,
                "FROZEN": [1.19] * 10,
                "ALSO_GOOD": [50.0] * 10,
            },
            index=idx, columns=symbols,
        )
        # FROZEN's mark decays from 0.575 -> 0.059 over the last 6 bars
        mark_values = [1.19, 1.19, 1.19, 0.575, 0.4, 0.3, 0.2, 0.1, 0.07, 0.059]
        mark = pd.DataFrame(
            {
                "GOOD": [100.0] * 10,
                "FROZEN": mark_values,
                "ALSO_GOOD": [50.0] * 10,
            },
            index=idx, columns=symbols,
        )
        eligible = pd.DataFrame(True, index=idx, columns=symbols)
        result, census = _fill_mark_parity_eligibility(close, eligible, True, mark_close=mark)
        # FROZEN rows 3-9 have |log(1.19/mark)| > log1p(0.05)
        for i in range(3, 10):
            assert result.loc[idx[i], "FROZEN"] == False  # noqa: E712
        # GOOD and ALSO_GOOD unaffected
        assert result["GOOD"].all()
        assert result["ALSO_GOOD"].all()
        assert census is not None
        assert census["cells_over_band"] == 7
        assert census["eligible_cells_removed"] == 7
        assert "FROZEN" in census["symbols"]

    def test_enabled_false_returns_unchanged(self) -> None:
        from src.application.research.mhs.evaluation import _fill_mark_parity_eligibility

        idx = pd.date_range("2025-04-01", periods=5, freq="1h", tz="UTC")
        close = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        eligible = pd.DataFrame({"A": [True, True, True, True, True]}, index=idx)
        result, census = _fill_mark_parity_eligibility(close, eligible, False)
        pd.testing.assert_frame_equal(result, eligible)
        assert census is None

class TestMhsDiagnosticRequestParityGate:
    """SCENARIO_MHS_FILL_MARK_PARITY_05: request field validation."""

    def test_defaults(self) -> None:
        req = MhsDiagnosticRequest()
        assert req.fill_mark_parity_gate is True
        assert req.exposure_scale_two_sided is False

    def test_two_sided_requires_exante(self) -> None:
        with pytest.raises(ValueError, match=r"exposure_scale_two_sided.*exante_target"):
            MhsDiagnosticRequest(
                exposure_scale_two_sided=True,
                pnl_vol_target_mode="median_relative",
            )

    def test_two_sided_exante_ok(self) -> None:
        req = MhsDiagnosticRequest(
            exposure_scale_two_sided=True,
            pnl_vol_target_mode="exante_target",
        )
        assert req.exposure_scale_two_sided is True

    def test_non_bool_fill_mark_parity_gate_raises(self) -> None:
        with pytest.raises(ValueError, match="fill_mark_parity_gate"):
            MhsDiagnosticRequest(fill_mark_parity_gate="yes")  # type: ignore[arg-type]

    def test_non_bool_exposure_scale_two_sided_raises(self) -> None:
        with pytest.raises(ValueError, match="exposure_scale_two_sided"):
            MhsDiagnosticRequest(exposure_scale_two_sided=1)  # type: ignore[arg-type]
