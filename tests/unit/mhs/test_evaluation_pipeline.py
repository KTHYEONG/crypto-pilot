"""MHS evaluation pipeline/gate tests (second-level split remainder)."""

"""MHS evaluation core contract tests (everything not in a domain-specific split file)."""
"""Contract coverage for the MHS application evaluation resource telemetry."""
import dataclasses
import numpy as np
import pandas as pd
import pytest
from src.mhs import evaluation as ev
from src.mhs.diagnostic_run import run_mhs_horizon_diagnostic
import src.mhs.resources as resources
import src.mhs.scaling as scaling
import src.mhs.research_go as _research_go
from src.mhs.evaluation import (
    MhsDiagnosticRequest,
    _assert_cache_required_marks,
    _iter_mhs_execution_windows,
    _truncate_replayable_decisions,
)
from src.common.errors import DataIntegrityError
from src.mhs.types import ExecutionSpec
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

def test_truncate_replayable_decisions_censors_terminal_window() -> None:
    grid = pd.date_range("2021-01-01 00:00", periods=752, freq="1min", tz="UTC")
    decision_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2021-01-01 11:00", tz="UTC"),
            pd.Timestamp("2021-01-01 11:30", tz="UTC"),
        ]
    )
    weights = pd.DataFrame({"A": [1.0, 1.0]}, index=decision_times)
    signals = decision_times + pd.Timedelta(hours=1)
    retained, retained_signals, censored = _truncate_replayable_decisions(
        weights, signals, grid, ExecutionSpec(),
    )
    # The 11:30 decision's submit bar (12:31) lies past the grid end; only the
    # 11:00 decision is replayable (submit 12:01, timeout 12:31 on grid).
    assert censored == 1
    assert list(retained.index) == [decision_times[0]]
    assert retained_signals[-1] < grid[-1]

    retained2, _, censored2 = _truncate_replayable_decisions(
        weights.iloc[0:1], signals[0:1], grid, ExecutionSpec(),
    )
    assert censored2 == 0
    assert retained2.equals(weights.iloc[0:1])

def test_truncate_replayable_decisions_requires_exact_timeout_bar() -> None:
    grid = pd.date_range("2021-01-01 00:00", periods=61, freq="1min", tz="UTC")
    decision_times = pd.DatetimeIndex([pd.Timestamp("2021-01-01 00:00", tz="UTC")])
    weights = pd.DataFrame({"A": [1.0]}, index=decision_times)
    signals = decision_times + pd.Timedelta(hours=1)
    retained, _, censored = _truncate_replayable_decisions(
        weights, signals, grid, ExecutionSpec(),
    )
    # The 60-minute grid ends at 01:00 and carries no 30-minute timeout bar for
    # the 01:01 submit, so the decision is censored as a terminal event.
    assert censored == 1
    assert retained.empty

def test_cache_required_marks_raise_structured_provenance() -> None:
    # MHS-STRICT-FAIL-CLOSED
    grid = pd.date_range("2021-01-01", periods=31, freq="1min", tz="UTC")
    weights = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 00:00", tz="UTC")])
    signals = pd.DatetimeIndex([pd.Timestamp("2021-01-01 01:00", tz="UTC")])
    marks = pd.DataFrame({"A": np.nan}, index=grid)
    with pytest.raises(DataIntegrityError, match="MISSING_DECISION_MARK") as excinfo:
        _assert_cache_required_marks("fold", weights, signals, marks)
    assert "symbol=A" in str(excinfo.value)
    assert "decision=2021-01-01 00:00:00+00:00" in str(excinfo.value)

def test_iter_mhs_execution_windows_preserves_columns_and_active_roster(tmp_path) -> None:
    start = pd.Timestamp("2021-01-01", tz="UTC")
    end = pd.Timestamp("2021-03-01", tz="UTC")
    grid = pd.date_range(start, end, freq="1min", tz="UTC")
    symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    (tmp_path / "1m").mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        frame = pd.DataFrame(
            {
                "timestamp": (grid - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms"),
                "high": 100.0,
                "low": 99.0,
                "close": 100.0,
            },
            index=grid,
        )
        frame["timestamp"] = frame["timestamp"].astype("int64")
        frame.reset_index(drop=True).to_parquet(tmp_path / "1m" / f"{sym}.parquet")

    decision_grid = pd.date_range(start, end, freq="6h", tz="UTC")
    weights = pd.DataFrame(0.0, index=decision_grid, columns=symbols)
    weights.loc[:, "AAAUSDT"] = 0.5
    signals = decision_grid + pd.Timedelta(hours=1)
    funding = {s: pd.Series([1e-5], index=[start]) for s in symbols}

    windows = list(
        _iter_mhs_execution_windows(
            weights, signals, str(tmp_path), "1m", start, end, funding,
            "ohlcv_close_fallback", ExecutionSpec(),
        )
    )
    assert windows
    for w in windows:
        assert w.columns == tuple(symbols)
        assert "AAAUSDT" in w.symbols
        assert w.target_weights.columns.tolist() == list(w.symbols)
        assert w.minute_grid.tz is not None
        assert len(w.minute_grid) > 1
    assert windows[-1].minute_grid[-1] == end

def test_request_validation_adjusted_without_gate() -> None:
    """SCENARIO_REQUEST_VALIDATION_ADJUSTED_WITHOUT_GATE: requesting the
    Bartlett/HAC-adjusted diagnostic while the discovery gate itself is off
    raises ValueError from ``__post_init__`` (fail-closed -- no silent no-op),
    and the flag defaults to False / composes with ``discovery_gate=True``."""
    with pytest.raises(ValueError, match="discovery_gate_adjusted_net_t"):
        MhsDiagnosticRequest(discovery_gate=False, discovery_gate_adjusted_net_t=True)
    assert MhsDiagnosticRequest().discovery_gate_adjusted_net_t is False
    assert MhsDiagnosticRequest(
        discovery_gate=True, discovery_gate_adjusted_net_t=True,
    ).discovery_gate_adjusted_net_t is True

def test_request_validation_regime_without_gate() -> None:
    """SCENARIO_REQUEST_VALIDATION_REGIME_WITHOUT_GATE: requesting the
    vol-regime cash-scale-adjusted diagnostic while the discovery gate itself is
    off raises ValueError from ``__post_init__`` (fail-closed, no silent
    no-op), independent of and in addition to the existing adjusted-net-t
    validation, and the flag defaults to False / composes with
    ``discovery_gate=True``."""
    with pytest.raises(ValueError, match="discovery_gate_regime_scaled_net_t"):
        MhsDiagnosticRequest(discovery_gate=False, discovery_gate_regime_scaled_net_t=True)
    assert MhsDiagnosticRequest().discovery_gate_regime_scaled_net_t is False
    assert MhsDiagnosticRequest(
        discovery_gate=True, discovery_gate_regime_scaled_net_t=True,
    ).discovery_gate_regime_scaled_net_t is True

class TestGrowthBudgetTargetVol:
    """SCENARIO_MHS_COMPOUNDING_ALPHA_AXES_04: _growth_budget_target_vol leak-free."""

    def test_never_reads_post_oos_rows(self) -> None:
        # Given a daily series whose pre-2023 slice is calm (std 0.005) and
        # whose post-2023 slice is violent (std 0.05), the returned target vol
        # is identical to the value returned when the post-2023 rows are
        # replaced by NaN.
        rng = np.random.default_rng(42)
        idx = pd.date_range("2021-01-01", periods=800, freq="D", tz="UTC")
        calm = rng.normal(0.0001, 0.005, 730)
        violent = rng.normal(0.0, 0.05, 70)
        returns = np.concatenate([calm, violent])
        r = pd.Series(returns, index=idx)
        result = scaling._growth_budget_target_vol(r)
        # NaN out post-2023 rows
        r_nan = r.copy()
        r_nan.loc[r_nan.index >= pd.Timestamp("2023-01-01", tz="UTC")] = np.nan
        result_nan = scaling._growth_budget_target_vol(r_nan)
        assert result == pytest.approx(result_nan)

    def test_fallback_when_fewer_than_burn_in_rows(self) -> None:
        # A series with fewer than PNL_VOL_TARGET_BURN_IN_DAYS pre-OOS
        # rows returns PNL_TARGET_ANNUAL_VOL.
        from src.mhs.params import PNL_TARGET_ANNUAL_VOL
        idx = pd.date_range("2022-06-01", periods=10, freq="D", tz="UTC")
        r = pd.Series(0.001, index=idx)
        assert scaling._growth_budget_target_vol(r) == PNL_TARGET_ANNUAL_VOL

class TestReplayExposureScaleGrowthBudget:
    """SCENARIO_MHS_COMPOUNDING_ALPHA_AXES_04: _replay_exposure_scale with growth_budget."""

    def test_growth_budget_returns_finite_bounded(self) -> None:
        rng = np.random.default_rng(42)
        idx = pd.date_range("2021-01-01", periods=500, freq="D", tz="UTC")
        r = pd.Series(rng.normal(0.0001, 0.01, 500), index=idx)
        request = MhsDiagnosticRequest(
            pnl_vol_target_mode="growth_budget",
            committee_capital=True,
        )
        result = scaling._replay_exposure_scale(r, request)
        assert result.index.equals(r.index)
        assert np.isfinite(result.to_numpy()).all()
        assert (result >= ev.PNL_VOL_TARGET_SCALE_FLOOR).all()
        assert (result <= 1.0).all()

class TestCommitteeMemberSetValidation:
    """SCENARIO_MHS_COMPOUNDING_ALPHA_AXES_06: committee_member_set validation."""

    def test_valid_member_set_accepted(self) -> None:
        req = MhsDiagnosticRequest(committee_capital=True, committee_member_set="risk_premia")
        assert req.committee_member_set == "risk_premia"

    def test_invalid_member_set_raises(self) -> None:
        with pytest.raises(ValueError, match="committee_member_set"):
            MhsDiagnosticRequest(committee_capital=True, committee_member_set="unregistered")

    def test_growth_budget_mode_in_pnl_vol_target_mode(self) -> None:
        req = MhsDiagnosticRequest(pnl_vol_target_mode="growth_budget")
        assert req.pnl_vol_target_mode == "growth_budget"

@pytest.mark.slow
def test_toplevel_vol_mean_masked_to_execution_roster(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_VOL_MEAN_ROSTER_MASK_TOPLEVEL_01: the top-level diagnostic
    # path applies the same execution_mask-filtered vol_mean to its blend regime
    # cash scale, matching the fold builder's fix.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    captured: dict[str, pd.Series] = {}
    real_scale = scaling._regime_cash_scale

    def spy(vol_mean, *args, **kwargs):
        captured["vol_mean"] = vol_mean.copy()
        return real_scale(vol_mean, *args, **kwargs)

    monkeypatch.setattr(scaling, "_regime_cash_scale", spy)
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(ev, "phase_1_anchored_purged_folds", lambda: ())
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    report = run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    assert "vol_mean" in captured, "top-level diagnostic must feed _regime_cash_scale its vol_mean"

    log_close, execution_mask, _grid = _roster_mask_panel_inputs(
        root, _START, end, funding_by_symbol, request.execution_universe_size,
    )
    _assert_regime_vol_mean_roster_masked(
        captured, log_close, execution_mask, captured["vol_mean"].index,
    )

class TestBookOutcomePaired:
    """SCENARIO_MHS_STREAM_BOOK_NO_MATERIALIZATION: the top-level book
    orchestrator streams the execution-window generator twice (reference pass +
    interleaved rescaled batch) instead of bulk materializing, routes the
    rescaled bounds through ``replay_execution_window_batch_isolated``, and preserves
    the typed book failure conversion."""

    def test_book_builds_window_iterator_twice_streaming(self, mhs_market, monkeypatch) -> None:
        # Pins the two-pass generator contract; disable coupled streaming.
        monkeypatch.setattr(ev._scaling, "is_streaming_scale_mode", lambda _request: False)
        args = _build_book_outcome_args(mhs_market)
        calls = {"n": 0}
        original = ev._iter_mhs_execution_windows

        def counting(*_args, **_kwargs):
            calls["n"] += 1
            return original(*_args, **_kwargs)

        monkeypatch.setattr(ev, "_iter_mhs_execution_windows", counting)
        batch_calls = {"n": 0}
        original_batch = ev.replay_execution_window_batch_isolated

        def counting_batch(*_args, **_kwargs):
            batch_calls["n"] += 1
            return original_batch(*_args, **_kwargs)

        monkeypatch.setattr(ev, "replay_execution_window_batch_isolated", counting_batch)
        report, _ = ev._book_outcome(**args)
        assert report.primary is not None
        assert report.stress is not None
        assert report.failure is None
        # Reference pass + one interleaved rescaled batch (bounded memory).
        assert calls["n"] == 2
        assert batch_calls["n"] == 1

    def test_book_strict_resource_breach_is_typed_failure(self, mhs_market, monkeypatch) -> None:
        args = _build_book_outcome_args(mhs_market)
        args["request"] = MhsDiagnosticRequest(
            start=str(_START), end=str(args["end"]), data_root=args["root"],
            mark_mode="cache_required", execution_timeframe="1m", log_run=False,
            max_rss_bytes=1_000,
        )
        monkeypatch.setattr(resources, "_current_rss_bytes", lambda: 100_000_000_000)
        report, _ = ev._book_outcome(**args)
        assert report.primary is None
        assert report.stress is None
        assert report.failure is not None
        assert report.failure.stage == "replay_fast_reversal"
        assert report.failure.reason == ev.GO_REASON_RESOURCE_BREACH

@pytest.mark.slow
def test_realized_execution_roster_size_exposed(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_REALIZED_ROSTER_SIZE_EXPOSED_06: the diagnostic report
    # exposes the realized mean execution-roster size (mean per-row True count
    # of the execution mask), and on a fixture where hysteresis retains members
    # it is strictly greater than the requested execution_universe_size.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    universe_size = 8
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(ev, "phase_1_anchored_purged_folds", lambda: ())
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=universe_size,
    )
    report = run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    assert report.realized_execution_roster_size is not None
    assert np.isfinite(report.realized_execution_roster_size)

    _log_close, execution_mask, _grid = _roster_mask_panel_inputs(
        root, _START, end, funding_by_symbol, universe_size,
    )
    assert report.realized_execution_roster_size == pytest.approx(
        float(execution_mask.sum(axis=1).mean())
    )

    # Hysteresis retention: engineer a volume panel over the same market
    # columns where the leading roster swaps early; the once-entered members
    # are kept past the entry rank (Schmitt-trigger), so the realized mean
    # roster strictly exceeds the requested universe size. The panel is longer
    # than the 1h grid so the 720-bar trailing warm-up does not bury the
    # retained members; downstream consumers realign by reindexing.
    idx = execution_mask.index
    cols = list(execution_mask.columns)
    engine_idx = pd.date_range(idx[0], periods=6000, freq="1h", tz="UTC")
    engineered_vol = pd.DataFrame(1.0, index=engine_idx, columns=cols)
    engineered_vol.loc[engine_idx[:720], cols[:universe_size]] = [
        100.0 - 10.0 * i for i in range(universe_size)
    ]
    engineered_vol.loc[engine_idx[720:], cols[universe_size:]] = [
        1000.0 - 10.0 * i for i in range(len(cols) - universe_size)
    ]
    eligible_all = pd.DataFrame(True, index=engine_idx, columns=cols)
    retention_mask = ev._pit_execution_mask(
        engineered_vol, eligible_all, universe_size,
    )
    retention_mean = float(retention_mask.sum(axis=1).mean())
    assert retention_mean > universe_size
    monkeypatch.setattr(
        ev, "_pit_execution_mask", lambda qv, el, usz: retention_mask,
    )
    retention_report = run_mhs_horizon_diagnostic(request)
    assert retention_report.realized_execution_roster_size == pytest.approx(retention_mean)
    assert retention_report.realized_execution_roster_size > universe_size

def test_mhs_perf_opt_004_window_frames_read_window_only(mhs_market, monkeypatch) -> None:
    # PERF_OPT_004_WINDOW_FRAMES: the production window loader
    # ``_load_window_minute_frames`` opens each symbol's Parquet exactly once
    # per window with a timestamp filter (never a full-period read), and a
    # missing symbol is skipped rather than raising.
    root, end = mhs_market
    syms = ["MHSAUSDT", "MHSBUSDT"]
    calls = {"n": 0}
    original = ev.pq.read_table

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ev.pq, "read_table", counting)

    ws = _START + pd.Timedelta(hours=6)
    we = _START + pd.Timedelta(hours=30)
    a = ev._load_window_minute_frames(str(root), syms, ws, we, "1m")
    assert a
    assert calls["n"] == len(syms)
    for k in a:
        assert a[k].index.min() >= ws
        assert a[k].index.max() <= we
    b = ev._load_window_minute_frames(str(root), syms, ws, we, "1m")
    assert set(b) == set(a)

def test_mhs_phase2_o6_window_frames_parity(mhs_market) -> None:
    # SCENARIO_O6_FRAME_PARITY: ``_load_window_minute_frames`` + ``_build_window_frames``
    # (the production window path, post fork-COW refactor) produce highs/lows/closes
    # on the window minute grid identical to the pre-refactor full-period slice path.
    root, end = mhs_market
    syms = ["MHSAUSDT", "MHSBUSDT", "MHSCUSDT"]
    ws = pd.Timestamp("2021-01-01 06:00", tz="UTC")
    we = pd.Timestamp("2021-01-02 06:00", tz="UTC")
    grid = pd.date_range(ws, we, freq="1min", tz="UTC")

    window_frames = ev._load_window_minute_frames(str(root), syms, ws, we, "1m")
    window_aligned = ev._build_window_frames(window_frames, syms, ws, we, grid, "1m")
    assert window_aligned is not None

    full_frames = {
        s: ev._load_window_minute_frames(str(root), [s], _START, end, "1m").get(s)
        for s in syms
    }
    slice_aligned = ev._build_window_frames(full_frames, syms, ws, we, grid, "1m")
    assert slice_aligned is not None

    window_highs, window_lows, window_closes = window_aligned
    slice_highs, slice_lows, slice_closes = slice_aligned
    for windowed, sliced, name in (
        (window_highs, slice_highs, "highs"),
        (window_lows, slice_lows, "lows"),
        (window_closes, slice_closes, "closes"),
    ):
        assert list(windowed.columns) == list(sliced.columns), name
        assert windowed.index.equals(sliced.index), name
        assert np.isclose(
            windowed.to_numpy(), sliced.to_numpy(), rtol=0, atol=0, equal_nan=True,
        ).all(), name

def test_mhs_phase2_o6_missing_symbol_skipped(mhs_market) -> None:
    # O6: the window loader silently skips a missing parquet (no full-period
    # cache exists to fail on after the fork-COW refactor).
    root, _end = mhs_market
    frames = ev._load_window_minute_frames(
        str(root), ["MHSAUSDT", "NOSUCHUSDT"], _START, _START + pd.Timedelta(hours=24), "1m",
    )
    assert "MHSAUSDT" in frames
    assert "NOSUCHUSDT" not in frames

def test_p11_resolve_ns_bit_identical() -> None:
    # SCENARIO_P11_RESOLVE_NS: the vectorized resolve_ns is bit-identical to the
    # scalar per-decision loop for grids with on-grid timeouts, off-grid
    # timeouts, and out-of-range submit positions.
    rng = np.random.default_rng(11)
    n_grid = 4096
    grid_ns = np.arange(n_grid, dtype="int64") * 60_000_000_000
    timeout_delta = 5 * 60_000_000_000
    for _ in range(5):
        spos_all = rng.integers(-10, n_grid + 10, size=300)
        expected = _reference_resolve_ns_scalar(spos_all, grid_ns, n_grid, timeout_delta)
        actual = ev._resolve_ns_vectorized(spos_all, grid_ns, n_grid, timeout_delta)
        assert actual.dtype == np.int64
        assert len(actual) == len(spos_all)
        assert np.array_equal(actual, expected)
    # A non-divisor timeout delta (never lands exactly on a grid bar) must be
    # all -1 exactly like the scalar path.
    odd_delta = 37 * 60_000_000_000
    spos_all = rng.integers(0, n_grid - 1, size=200)
    assert np.array_equal(
        ev._resolve_ns_vectorized(spos_all, grid_ns, n_grid, odd_delta),
        _reference_resolve_ns_scalar(spos_all, grid_ns, n_grid, odd_delta),
    )

def test_p10_concurrent_books_parity(mhs_market) -> None:
    # SCENARIO_P10_CONCURRENT / SCENARIO_MHS_REGIME_SCALE_OMITTED_BYTE_IDENTICAL_02:
    # three books executed concurrently in fork workers produce bit-identical
    # reports to the sequential path. This call omits regime_scale (defaults
    # to None), so it also proves every existing _run_books_concurrent caller
    # stays byte-identical after the regime_scale parameter was added.
    args = _build_books_concurrent_args(mhs_market)
    sequential = _sequential_book_reports(args)
    concurrent_fast, concurrent_slow, concurrent_blend, _, _ = ev._run_books_concurrent(**args)
    concurrent = (concurrent_fast, concurrent_slow, concurrent_blend)
    assert len(concurrent) == 3
    for seq, con, name in zip(sequential, concurrent, ("fast_reversal", "slow_momentum", "blend"), strict=True):
        _assert_books_equal(seq, con, name)

def test_p10_mark_cache_warmable_per_symbol(mhs_market) -> None:
    # Mark frame cache warms one symbol's mark parquet per call for COW inheritance.
    root, end = mhs_market
    syms = ["MHSAUSDT", "MHSBUSDT", "MHSCUSDT"]
    ev._get_symbol_mark_frame.cache_clear()
    for s in syms:
        assert ev._get_symbol_mark_frame(s, "1h") is not None
    assert ev._get_symbol_mark_frame.cache_info().currsize >= len(syms)

def test_p10_book_error_isolation(mhs_market, monkeypatch) -> None:
    # SCENARIO_P10_ISOLATION: a book whose outcome is a typed failure (primary
    # dropped, failure set) is delivered through the process pool without
    # blocking the other two books.
    args = _build_books_concurrent_args(mhs_market)
    real = ev._book_outcome
    import src.mhs.evaluation.windows as windows_mod
    real_windows = windows_mod._book_outcome

    def _failing(name, *a, **k):
        report, traces = real(name, *a, **k)
        if name == "slow_momentum":
            return dataclasses.replace(
                report,
                primary=None, stress=None,
                primary_autocorr_sharpe=None,
                primary_naive_sharpe=None,
                primary_net_ann=None,
                primary_geometric_cagr=None,
                primary_max_drawdown=None,
                primary_annualized_turnover=None,
                stress_naive_sharpe=None,
                failure=ev.MhsBookFailure(
                    stage="replay_slow_momentum",
                    error_class="DataIntegrityError",
                    reason=ev.GO_REASON_EXECUTION_GAP,
                    message="forced isolation failure",
                ),
            ), traces
        return report, traces

    def _failing_windows(name, *a, **k):
        report, traces = real_windows(name, *a, **k)
        if name == "slow_momentum":
            return dataclasses.replace(
                report,
                primary=None, stress=None,
                primary_autocorr_sharpe=None,
                primary_naive_sharpe=None,
                primary_net_ann=None,
                primary_geometric_cagr=None,
                primary_max_drawdown=None,
                primary_annualized_turnover=None,
                stress_naive_sharpe=None,
                failure=ev.MhsBookFailure(
                    stage="replay_slow_momentum",
                    error_class="DataIntegrityError",
                    reason=ev.GO_REASON_EXECUTION_GAP,
                    message="forced isolation failure",
                ),
            ), traces
        return report, traces

    monkeypatch.setattr(ev, "_book_outcome", _failing)
    monkeypatch.setattr(windows_mod, "_book_outcome", _failing_windows)
    fast, slow, blend, _, _ = ev._run_books_concurrent(**args)
    assert fast.primary is not None
    assert fast.failure is None
    assert slow.primary is None
    assert slow.failure is not None
    assert slow.failure.reason == ev.GO_REASON_EXECUTION_GAP
    assert blend.primary is not None
    assert blend.failure is None

def test_regime_scale_reaches_blend_replay_not_only_prescreen(mhs_market) -> None:
    # SCENARIO_MHS_REGIME_SCALE_REACHES_BLEND_REPLAY_01: blend replay reflects regime scale.
    args = _build_books_concurrent_args(mhs_market)
    active_grid = ev._active_blend_book_and_grid(
        args["fast"], args["slow"], args["fast_grid"], args["slow_grid"],
    )[1]
    half = len(active_grid) // 2
    scale = pd.Series(1.0, index=active_grid)
    scale.iloc[:half] = 0.5

    fast_base, slow_base, blend_base, _, _ = ev._run_books_concurrent(**args)
    fast_scaled, slow_scaled, blend_scaled, _, _ = ev._run_books_concurrent(**args, regime_scale=scale)

    assert blend_base.failure is None
    assert blend_scaled.failure is None
    assert blend_base.primary is not None
    assert blend_scaled.primary is not None
    # retain_event_snapshots=False throughout _book_outcome, so per-fill
    # notional weights are never materialized here -- the turnover/equity
    # series (always populated) are the observable proxy for "the replay
    # actually traded a smaller book," not a coincidence of unrelated noise.
    assert not blend_scaled.primary.ledger.equity.equals(blend_base.primary.ledger.equity)
    assert blend_scaled.primary.ledger.fill_turnover.sum() < blend_base.primary.ledger.fill_turnover.sum()

    # fast_reversal/slow_momentum's own standalone books are untouched by the
    # blend-only scale.
    pd.testing.assert_series_equal(
        fast_scaled.primary.ledger.equity, fast_base.primary.ledger.equity,
        check_exact=True, rtol=0.0, atol=0.0,
    )
    pd.testing.assert_series_equal(
        slow_scaled.primary.ledger.equity, slow_base.primary.ledger.equity,
        check_exact=True, rtol=0.0, atol=0.0,
    )





@pytest.mark.slow
def test_committee_streaming_regression(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_MHS_COMMITTEE_STREAMING_REGRESSION: the streaming committee
    # rewrite is behavior-transparent -- the pre-existing walk-forward wealth
    # scenario (block edges anchored at OOS_START, purge 720, empty skipped
    # blocks, finite per-tier fields) still holds after the per-member book
    # streaming + multi-tier ledger.
    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None, {}, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_book=True,
    )
    report = run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    diag = report.committee_diagnostic
    assert diag["walk_forward"]["block_edges"][0] == ev.COMMITTEE_OOS_START.isoformat()
    assert diag["walk_forward"]["purge_hours"] == 720
    assert diag["walk_forward"]["skipped_blocks"] == []
    per_tier = diag["walk_forward"]["per_tier"]
    assert set(per_tier) == set(ev.MEASURED_EXECUTION_COST_TIERS_BPS)
    for fields in per_tier.values():
        assert isinstance(fields["bars"], int)
        assert fields["bars"] >= 0
        for key in ("net_sharpe", "cagr", "mdd", "logret"):
            value = fields[key]
            assert value is None or np.isfinite(value)

def test_fold_primary_annual_return_floor_enforcement(mhs_market, monkeypatch) -> None:
    """SCENARIO_MHS_RESEARCH_GO_ELIGIBLE_WITH_REGISTERED_POLICY: the registered
    REGISTERED_POLICY_THRESHOLDS['primary_annual_return'] floor is enforced on
    the POOLED level evidence by research_go's pooled lower-bound gate, never
    as a per-fold failure code (I-FAMILY); an unregistered (None) threshold
    adds no code, matching the pre-registration conservative fail-closed
    default."""
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ][:8]
    funding_by_symbol, _ = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )

    # I-FAMILY: fold replay는 level 코드를 만들지 않는다(무결성 코드만).
    completed_fold = ev._run_anchored_fold(
        str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
    )
    assert ev.GO_REASON_PRIMARY_RETURN_BELOW_FLOOR not in completed_fold.failures

    from src.mhs import research_go as _research_go_module

    low_return_evidence = {
        "n_measured_folds": 4,
        "pooled_sharpe_lcb": 1.0,
        "pooled_stress_sharpe_lcb": 0.5,
        "pooled_annual_log_return": 0.01,
    }
    monkeypatch.setattr(
        _research_go_module, "REGISTERED_POLICY_THRESHOLDS",
        {"cap_60_roster": 60.0, "primary_annual_return": 10.0},
    )
    assert _research_go_module._pooled_level_gate_reasons(low_return_evidence) == (
        ev.GO_REASON_PRIMARY_RETURN_BELOW_FLOOR,
    )

    monkeypatch.setattr(
        _research_go_module, "REGISTERED_POLICY_THRESHOLDS",
        {"cap_60_roster": 60.0, "primary_annual_return": None},
    )
    assert _research_go_module._pooled_level_gate_reasons(low_return_evidence) == ()



















def test_target_gross_request_validation() -> None:
    # SCENARIO_MHS_TARGET_GROSS_REQUEST_VALIDATION
    default = MhsDiagnosticRequest()
    # Registered default exposure applies to a bare request without forcing
    # committee_capital=True. The unresolved sentinel is never mutated into
    # the frozen field (that would break dataclasses.replace()); resolution
    # happens lazily via _resolved_committee_target_gross.
    assert _research_go._resolved_committee_target_gross(default) == ev.COMMITTEE_TARGET_GROSS

    valid = MhsDiagnosticRequest(committee_target_gross=0.795, committee_capital=True)
    assert valid.committee_target_gross == 0.795

    with pytest.raises(ValueError, match="committee_capital"):
        MhsDiagnosticRequest(committee_target_gross=0.795, committee_capital=False)

    with pytest.raises(ValueError, match="committee_target_gross"):
        MhsDiagnosticRequest(committee_target_gross=0.0, committee_capital=True)
    with pytest.raises(ValueError, match="committee_target_gross"):
        MhsDiagnosticRequest(committee_target_gross=-1.0, committee_capital=True)
    with pytest.raises(ValueError, match="committee_target_gross"):
        MhsDiagnosticRequest(committee_target_gross=2.5, committee_capital=True)

def test_reference_bound_degraded_preserves_primary(mhs_market, monkeypatch) -> None:
    """SCENARIO_MHS_REFERENCE_BOUND_DEGRADED: when the isolated batch returns
    a None result for the strict-proxy slot plus an IsolatedBoundFailure,
    _book_outcome returns failure=None, primary not None, primary_geometric_cagr
    finite, patient_reference None, and reference_bound_failures has exactly
    one entry."""
    # Pins two-pass degraded-reference semantics; disable coupled streaming.
    monkeypatch.setattr(ev._scaling, "is_streaming_scale_mode", lambda _request: False)
    from src.mhs.execution import IsolatedBoundFailure, BatchReplayOutcome
    args = _build_book_outcome_args(mhs_market)
    baseline, _ = ev._book_outcome(**args)
    # Build a mock result for the strict slot
    strict_fallback = baseline.primary
    assert strict_fallback is not None
    mock_outcome = BatchReplayOutcome(
        results=(baseline.primary, baseline.stress, None),
        isolated_failures=(
            IsolatedBoundFailure(
                bound_index=2,
                execution_bound="OHLCV_STRICT_PROXY",
                error_class="DataIntegrityError",
                message="pre-trade equity must be positive and finite (ts=fail test)",
                windows_consumed=3,
            ),
        ),
    )
    real_isolated = ev.replay_execution_window_batch_isolated
    monkeypatch.setattr(ev, "replay_execution_window_batch_isolated", lambda *a, **k: mock_outcome)
    report, _ = ev._book_outcome(**args)
    monkeypatch.setattr(ev, "replay_execution_window_batch_isolated", real_isolated)
    assert report.failure is None
    assert report.primary is not None
    assert np.isfinite(report.primary_geometric_cagr)
    assert report.patient_reference is None
    assert report.patient_reference_naive_sharpe is None
    assert len(report.reference_bound_failures) == 1
    rbf = report.reference_bound_failures[0]
    assert "OHLCV_STRICT_PROXY" in rbf.stage
    assert rbf.reason == "CAPITAL_INVARIANT_BREACH"

def test_drawdown_budget_gate_reasons() -> None:
    """SCENARIO_MHS_DRAWDOWN_BUDGET_GATE: _drawdown_budget_reasons returns
    PRIMARY_MAX_DRAWDOWN_OVER_BUDGET only when the drawdown strictly exceeds
    the registered budget, and _mhs_research_go with that reason code yields
    eligible=False absent from data_integrity_reason_codes."""
    assert _research_go._drawdown_budget_reasons(-0.26) == ("PRIMARY_MAX_DRAWDOWN_OVER_BUDGET",)
    assert _research_go._drawdown_budget_reasons(-0.25) == ()
    assert _research_go._drawdown_budget_reasons(-0.1269) == ()
    assert _research_go._drawdown_budget_reasons(None) == ()
    assert _research_go._drawdown_budget_reasons(float("nan")) == ()
    with pytest.raises(ValueError, match="max_drawdown"):
        _research_go._drawdown_budget_reasons(-0.26, max_drawdown=0.0)
    # _mhs_research_go: extra reason gates eligible to False
    go = _research_go._mhs_research_go((), extra_reasons=("PRIMARY_MAX_DRAWDOWN_OVER_BUDGET",))
    assert go.eligible is False
    assert "PRIMARY_MAX_DRAWDOWN_OVER_BUDGET" in go.reason_codes
    assert "PRIMARY_MAX_DRAWDOWN_OVER_BUDGET" not in go.data_integrity_reason_codes



class TestCommitteeMemberAttribution:
    """Tests for _committee_member_attribution proxy_vs_ledger_rank_spearman."""

    def test_perfect_correlation(self) -> None:
        from src.mhs.evaluation import _committee_member_attribution
        ledger_sharpes = {"a": 3.0, "b": 2.0, "c": 1.0}
        proxy_sharpes = {"a": 3.0, "b": 2.0, "c": 1.0}
        result = _committee_member_attribution({}, proxy_sharpes)
        # With empty member_reports, ledger_sharpes is empty, so spearman is None
        assert result["proxy_vs_ledger_rank_spearman"] is None

    def test_empty_reports_yields_none_spearman(self) -> None:
        from src.mhs.evaluation import _committee_member_attribution
        result = _committee_member_attribution({}, {"a": 1.0})
        assert result["proxy_vs_ledger_rank_spearman"] is None
        assert result["members"] == {}
        assert result["daily_return_correlation"] == {}

    def test_fewer_than_three_shared_yields_none(self) -> None:
        from src.mhs.evaluation import _committee_member_attribution
        # Only 2 shared members < 3 threshold
        result = _committee_member_attribution({}, {"a": 1.0, "b": 2.0})
        assert result["proxy_vs_ledger_rank_spearman"] is None

@pytest.mark.slow
def test_committee_member_attribution_observational_only(mhs_market_with_taker_buy_quote) -> None:
    # SCENARIO_MEMBER_ATTRIBUTION_IS_OBSERVATIONAL: enabling per-member
    # attribution replays must not perturb the reported blend/research-go/
    # fold_growth_concentration evidence -- the member books are report-only
    # (I5) and never enter blend_1h, committee_execution_book, regime_scale,
    # any exposure scale, any fold report, or any Research-GO reason code.
    # committee_capital=True (member books are only built on this path) needs
    # the taker_buy_quote column, hence mhs_market_with_taker_buy_quote rather
    # than the taker_buy_quote-less mhs_market_long.
    root, end = mhs_market_with_taker_buy_quote
    base = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_capital=True,
    )
    off = run_mhs_horizon_diagnostic(base)
    on = run_mhs_horizon_diagnostic(
        dataclasses.replace(base, committee_member_attribution=True),
    )
    assert off.status == "COMPLETE"
    assert on.status == "COMPLETE"
    assert off.committee_member_attribution is None

    assert on.blend.primary_geometric_cagr == off.blend.primary_geometric_cagr
    assert on.blend.primary_naive_sharpe == off.blend.primary_naive_sharpe
    assert on.blend.primary_max_drawdown == off.blend.primary_max_drawdown
    assert on.research_go.reason_codes == off.research_go.reason_codes
    assert on.research_go.eligible == off.research_go.eligible
    assert on.fold_growth_concentration == off.fold_growth_concentration

    assert on.committee_member_attribution is not None
    from src.mhs.research_go import _resolved_committee_members
    expected_members = set(_resolved_committee_members(base))
    assert set(on.committee_member_attribution["members"]) == expected_members
