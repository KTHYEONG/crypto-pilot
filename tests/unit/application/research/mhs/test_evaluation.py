"""Contract coverage for the MHS application evaluation resource telemetry."""

import json
import time
import types
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.market_data.services.futures_collection as fc
from src.application.research.mhs import evaluation as ev
from src.application.research.mhs.evaluation import (
    MhsDiagnosticRequest,
    _StageRecorder,
    _assert_cache_required_ledger_valid,
    _assert_cache_required_marks,
    _assert_execution_rss_budget,
    _iter_mhs_execution_windows,
    _truncate_replayable_decisions,
)
from src.common.errors import DataIntegrityError
from src.mhs.contracts import BookSpec, ExecutionSpec, HorizonBand
from src.mhs.horizons import vol_normalized_horizon_signal
from src.mhs.execution import ExecutionReplayWindow, replay_execution_windows
from src.mhs.execution import strategy_aware_execution_replay
from src.mhs.evaluation import AnchoredPurgedFold
from src.research.universe.pit_universe import symbol_partition

_START = pd.Timestamp("2021-01-01", tz="UTC")


def _write_mhs_market(root: Path, n_hours: int = 2700, include_btc: bool = False) -> pd.Timestamp:
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    if include_btc:
        symbols = ["BTCUSDT", *symbols]
    hourly = pd.date_range(_START, periods=n_hours, freq="1h", tz="UTC")
    end = hourly[-1]
    rng = np.random.default_rng(20260807)
    epoch = (hourly - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    hdir = root / "1h"
    mdir = root / "1m"
    fdir = root / "funding"
    mkdir = root / "markPriceKlines" / "1h"
    for d in (hdir, mdir, fdir, mkdir):
        d.mkdir(parents=True, exist_ok=True)
    minute_idx = pd.date_range(_START, end, freq="1min", tz="UTC")
    minute_epoch = (minute_idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    for i, sym in enumerate(symbols):
        drift = 1e-5 * (i - len(symbols) / 2.0)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.002, n_hours)))
        pd.DataFrame(
            {"timestamp": epoch, "open": prices, "high": prices * 1.001,
             "low": prices * 0.999, "close": prices, "quote_vol": [1000.0] * n_hours},
        ).to_parquet(hdir / f"{sym}.parquet")
        mp = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.002, len(minute_idx))))
        pd.DataFrame(
            {"timestamp": minute_epoch, "open": mp, "high": mp * 1.0005,
             "low": mp * 0.9995, "close": mp, "quote_vol": [1000.0] * len(minute_idx)},
        ).to_parquet(mdir / f"{sym}.parquet")
        pd.DataFrame(
            {"timestamp": epoch, "funding_rate": [0.00005] * n_hours, "datetime": hourly},
        ).to_parquet(fdir / f"{sym}.parquet")
        mark = pd.Series(mp, index=minute_idx).resample("1h").last().reindex(hourly).to_numpy()
        pd.DataFrame(
            {"timestamp": epoch, "open": mark, "high": mark, "low": mark, "close": mark, "datetime": hourly},
        ).to_parquet(mkdir / f"{sym}.parquet")
    return end


@pytest.fixture
def mhs_market(tmp_path, monkeypatch):
    root = tmp_path / "market"
    end = _write_mhs_market(root)
    monkeypatch.setattr(ev, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    return root, end


@pytest.fixture
def mhs_market_with_btc(tmp_path, monkeypatch):
    """Same synthetic market plus a continuous BTCUSDT column, so the opt-in
    crash-regime tilt (reference basket = MHS_CRASH_REGIME_REFERENCE_SYMBOLS)
    has a real reference series to read."""
    root = tmp_path / "market"
    end = _write_mhs_market(root, include_btc=True)
    monkeypatch.setattr(ev, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    return root, end


def _signal_disagreement_panel():
    """Momentum signal disagreement fixture (horizon=2).

    Cross-sectionally the raw ``horizon_log_return`` ranks NOISY above QUIET
    while the vol-normalized signal ranks QUIET above NOISY, so ``_book_weights``
    / ``_phase_diagnostics`` built from the two signals must differ -- the
    fixture that proves a sign=+1 dispatch to ``vol_normalized_horizon_signal``
    actually took effect.
    """
    idx = pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC")
    log_close = pd.DataFrame(
        {
            "NOISY": [0.0, 0.5, -0.2, 0.4, 0.8, 0.7, 1.1, 0.9],
            "QUIET": [0.0, 0.01, 0.04, 0.06, 0.07, 0.10, 0.11, 0.14],
        },
        index=idx,
    )
    eligible = pd.DataFrame(True, index=idx, columns=log_close.columns)
    rng = np.random.default_rng(7)
    o2o = pd.DataFrame(rng.normal(0.0, 1e-3, (len(idx), 2)), index=idx, columns=log_close.columns)
    opens = pd.DataFrame(
        100.0 * np.exp(np.cumsum(o2o.to_numpy(), axis=0)), index=idx, columns=log_close.columns,
    )
    bar_funding = pd.DataFrame(0.0, index=idx, columns=log_close.columns)
    return log_close, eligible, opens, bar_funding, idx


def _dispatch_spec(sign: int) -> BookSpec:
    band = HorizonBand(name="test_dispatch", horizons_hours=(2,), sign=sign)
    return BookSpec(band=band, horizon_hours=2, step_hours=1, min_symbols=2)


def _reference_weights(log_close, eligible, step_grid, spec) -> pd.DataFrame:
    sig = ev.horizon_log_return(log_close, spec.horizon_hours).reindex(step_grid)
    return ev.phase_tranche_book(
        ev.rank_weight_book(sig, eligible.reindex(step_grid), spec.band.sign, spec.min_symbols),
        spec.tranche_count(),
    )


def test_mhs_resource_measurement_records_ordered_stage_data() -> None:
    recorder = _StageRecorder(log_run=False)
    recorder.record("unit_stage", grid_bars=3, n_symbols=2, fill_count=1)

    records = recorder.records
    assert len(records) == 1
    record = records[0]
    assert record.stage == "unit_stage"
    assert record.elapsed_ms >= 0
    assert record.rss_bytes > 0
    assert record.peak_rss_bytes == record.rss_bytes
    assert record.window_start is None
    assert record.window_end is None
    assert record.active_symbols is None
    assert record.grid_bars == 3
    assert record.n_symbols == 2
    assert record.fill_count == 1


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


def _pnl_vol_spike_returns() -> pd.Series:
    """Calm-then-high-vol daily returns with non-zero vol in each regime."""
    rng = np.random.default_rng(20260807)
    idx = pd.date_range("2024-01-01", periods=200, freq="D", tz="UTC")
    returns = np.concatenate([
        rng.normal(0.001, 0.002, 100),
        rng.normal(0.05, 0.05, 100),
    ])
    return pd.Series(returns, index=idx)


def test_pnl_vol_target_scale_no_lookahead() -> None:
    # SCENARIO_PNL_VOL_TARGET_SCALE_NO_LOOKAHEAD: scale_t must depend only on
    # reference_daily_returns strictly before t -- truncating the input to end
    # exactly at t leaves scale[:t+1] unchanged.
    r = _pnl_vol_spike_returns()
    full = ev._pnl_vol_target_scale(r)
    for t in (40, 90, 110, 150, 198):
        truncated = ev._pnl_vol_target_scale(r.iloc[: t + 1])
        pd.testing.assert_series_equal(
            full.iloc[: t + 1], truncated, check_names=False,
        )


def test_pnl_vol_target_scale_reduces_on_vol_spike() -> None:
    # SCENARIO_PNL_VOL_TARGET_SCALE_REDUCES_ON_VOL_SPIKE: a calm regime keeps
    # full exposure while a high-vol regime drives the scale toward the floor.
    r = _pnl_vol_spike_returns()
    out = ev._pnl_vol_target_scale(r)
    assert out.iloc[50] == pytest.approx(1.0)
    assert out.iloc[150] <= ev.MHS_PNL_VOL_TARGET_SCALE_FLOOR + 1e-9
    assert out.iloc[150] < out.iloc[50]


def test_pnl_vol_target_scale_never_exceeds_one() -> None:
    # SCENARIO_PNL_VOL_TARGET_SCALE_NEVER_EXCEEDS_ONE: no leverage-up ever and
    # the floor is respected across ultra-calm and ultra-volatile regimes; a
    # zero-std constant-return input is safe-divided to 1.0 (no inf).
    rng = np.random.default_rng(7)
    calm = pd.Series(rng.normal(1e-4, 1e-5, 150), index=pd.date_range("2024-01-01", periods=150, freq="D", tz="UTC"))
    wild = pd.Series(rng.normal(0.0, 0.5, 150), index=pd.date_range("2024-06-01", periods=150, freq="D", tz="UTC"))
    combo = pd.concat([calm, wild])
    out = ev._pnl_vol_target_scale(combo)
    assert (out >= ev.MHS_PNL_VOL_TARGET_SCALE_FLOOR).all()
    assert (out <= 1.0).all()
    constant = pd.Series(
        np.concatenate([np.full(100, 0.001), np.full(100, 0.05)]),
        index=pd.date_range("2024-01-01", periods=200, freq="D", tz="UTC"),
    )
    zero_out = ev._pnl_vol_target_scale(constant)
    assert np.isfinite(zero_out.to_numpy()).all()
    assert (zero_out >= 0.2).all()
    assert (zero_out <= 1.0).all()


def test_pnl_vol_target_scale_burn_in_is_unscaled() -> None:
    # SCENARIO_PNL_VOL_TARGET_SCALE_BURN_IN_IS_UNSCALED: before
    # MHS_PNL_VOL_TARGET_BURN_IN_DAYS samples exist, scale is exactly 1.0 no
    # matter how volatile the input is -- never an under-sampled estimate.
    r = _pnl_vol_spike_returns()
    out = ev._pnl_vol_target_scale(r)
    assert (out.iloc[: ev.MHS_PNL_VOL_TARGET_BURN_IN_DAYS - 1] == 1.0).all()


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


def test_mhs_mem_03_rss_budget_fails_closed(monkeypatch) -> None:
    """MHS-MEM-03: a configured RSS budget produces deterministic
    DataIntegrityError provenance rather than a process-level OOM or a valid
    partial report."""
    assert MhsDiagnosticRequest().max_rss_bytes is None
    with pytest.raises(ValueError, match="max_rss_bytes"):
        MhsDiagnosticRequest(max_rss_bytes=0)
    with pytest.raises(ValueError, match="max_rss_bytes"):
        MhsDiagnosticRequest(max_rss_bytes=-1)
    assert MhsDiagnosticRequest(max_rss_bytes=1_000_000_000).max_rss_bytes == 1_000_000_000

    monkeypatch.setattr("src.application.research.mhs.evaluation._current_rss_bytes", lambda: 5_000_000_000)
    with pytest.raises(DataIntegrityError, match="execution RSS budget exceeded") as excinfo:
        _assert_execution_rss_budget("execution_window", 1_000_000_000, 7)
    message = str(excinfo.value)
    assert "stage=execution_window" in message
    assert "observed_rss=5000000000" in message
    assert "budget=1000000000" in message
    assert "completed_windows=7" in message
    _assert_execution_rss_budget("execution_window", None, 7)


def test_crash_tilt_request_validation() -> None:
    # SCENARIO_MHS_CRASH_TILT_REQUEST_VALIDATION_05: the request-level opt-in
    # narrows the pure function's [0.0, 1.0] to (0.0, 1.0] -- an explicitly
    # set-but-no-op 0.0 is a footgun, and >1.0 breaks the unit-gross budget.
    with pytest.raises(ValueError, match="crash_regime_tilt_alpha"):
        MhsDiagnosticRequest(crash_regime_tilt_alpha=0.0)
    with pytest.raises(ValueError, match="crash_regime_tilt_alpha"):
        MhsDiagnosticRequest(crash_regime_tilt_alpha=1.5)
    assert MhsDiagnosticRequest().crash_regime_tilt_alpha is None
    assert MhsDiagnosticRequest(crash_regime_tilt_alpha=0.2).crash_regime_tilt_alpha == 0.2


def test_mhs_mem_04_strict_gap_preserved() -> None:
    """MHS-MEM-04: cache_required continues to fail closed on MISSING_HELD_MARK
    for a held-mark fixture; stale carry remains explicit diagnostic mode."""
    grid = pd.date_range("2021-01-01", periods=48, freq="5min", tz="UTC")
    px = pd.DataFrame({"A": [100.0] * len(grid)}, index=grid)
    marks = px.copy()
    marks.loc[grid[20]:grid[25], "A"] = np.nan
    target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 00:00", tz="UTC")])
    signals = pd.DatetimeIndex([pd.Timestamp("2021-01-01 01:00", tz="UTC")])
    window = ExecutionReplayWindow(
        window_start=grid[0],
        window_end=grid[-1],
        columns=("A",),
        symbols=("A",),
        minute_grid=grid,
        highs=px,
        lows=px,
        closes=px,
        marks=marks,
        bar_funding=pd.DataFrame(0.0, index=grid, columns=["A"]),
        target_weights=target,
        signal_available_at=signals,
    )
    replay = replay_execution_windows(
        [window], 1.0, "OHLCV_STRICT_PROXY", ExecutionSpec(),
    )
    assert replay.event_snapshots_retained is False
    gap_codes = {g.code for g in replay.data_gaps}
    assert "MISSING_HELD_MARK" in gap_codes
    assert replay.ledger.primary_valid is False
    with pytest.raises(DataIntegrityError, match="cache_required strict primary ledger invalid"):
        _assert_cache_required_ledger_valid("held_mark_book", replay)
    assert replay.ledger.primary_valid is False


_FOLD = AnchoredPurgedFold(
    pd.Timestamp("2021-01-01", tz="UTC"),
    pd.Timestamp("2021-01-31", tz="UTC"),
    pd.Timestamp("2021-02-10", tz="UTC"),
    pd.Timestamp("2021-04-19 08:00", tz="UTC"),
    168,
    168,
)


class TestAnchoredFoldBounded:
    """MHS-MEM-03-ANCHORED-FOLD-BOUNDED: each anchored fold uses bounded
    windowed replay (no dense fold-wide minute panel) and enforces the
    configured RSS budget with stable provenance."""

    def _run_fold(self, mhs_market, max_rss_bytes=None):
        root, end = mhs_market
        symbols = [
            s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                        "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
            if symbol_partition(s) == "dev"
        ][:8]
        funding_by_symbol = ev._load_funding_series(symbols)
        request = MhsDiagnosticRequest(
            start=str(_START), end=str(end), data_root=str(root),
            mark_mode="cache_required", execution_timeframe="1m", log_run=False,
            max_rss_bytes=max_rss_bytes,
        )
        return ev._run_anchored_fold(
            str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
        )

    def test_fold_uses_windowed_replay_dense_snapshots_disabled(self, mhs_market) -> None:
        report = self._run_fold(mhs_market)
        assert report.strict is not None
        assert report.strict.event_snapshots_retained is False
        assert report.stress is not None
        assert report.stress.event_snapshots_retained is False

    def test_fold_records_ordered_window_telemetry(self, mhs_market) -> None:
        root, end = mhs_market
        symbols = [
            s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                        "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
            if symbol_partition(s) == "dev"
        ][:8]
        funding_by_symbol = ev._load_funding_series(symbols)
        request = MhsDiagnosticRequest(
            start=str(_START), end=str(end), data_root=str(root),
            mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        )
        recorder = _StageRecorder(log_run=False)
        ev._run_anchored_fold(str(root), _FOLD, request, funding_by_symbol, 1.0, 0, recorder)
        window_stages = [m.stage for m in recorder.records if m.stage.startswith("anchored_fold_0_window_")]
        assert window_stages, "fold paired window telemetry must be recorded"
        # The two-pass primary records each window bound twice (Pass 1
        # reference, then the rescaled Pass 2), so the primary stages are two
        # identical internally-ordered passes; the cost-stress bound follows.
        primary_stages = [
            s for s in window_stages
            if not s.startswith("anchored_fold_0_window_cost_stress")
        ]
        assert len(primary_stages) % 2 == 0
        half = len(primary_stages) // 2
        assert primary_stages[:half] == primary_stages[half:]
        assert primary_stages[:half] == sorted(primary_stages[:half])
        stress_stages = [
            s for s in window_stages
            if s.startswith("anchored_fold_0_window_cost_stress")
        ]
        assert stress_stages == sorted(stress_stages)
        # The paired fan-out records one physical window per stage: the stress
        # bound consumes the same iterator, so no separate stress re-iteration
        # telemetry exists.
        assert not [
            m.stage for m in recorder.records
            if m.stage.startswith("anchored_fold_0_stress_window_")
        ]

    def test_fold_builds_window_iterator_once_per_bound(self, mhs_market, monkeypatch) -> None:
        """MHS-MEM-PAIR-02: the fold orchestrator constructs one execution-window
        iterator per replay bound (immediate-taker primary + cost-stressed
        stress) and consumes each independently without re-materializing."""
        root, end = mhs_market
        symbols = [
            s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                        "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
            if symbol_partition(s) == "dev"
        ][:8]
        funding_by_symbol = ev._load_funding_series(symbols)
        request = MhsDiagnosticRequest(
            start=str(_START), end=str(end), data_root=str(root),
            mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        )
        calls = {"n": 0}
        original = ev._iter_mhs_execution_windows

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(ev, "_iter_mhs_execution_windows", counting)
        report = ev._run_anchored_fold(
            str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
        )
        assert report.strict is not None
        assert report.stress is not None
        # Two-pass primary (reference + rescaled) plus one stress bound.
        assert calls["n"] == 3

    def test_rss_budget_enforced_inside_fold_fails_closed(self, mhs_market, monkeypatch) -> None:
        monkeypatch.setattr(ev, "_current_rss_bytes", lambda: 100_000_000_000)
        report = self._run_fold(mhs_market, max_rss_bytes=1_000)
        # The budget DataIntegrityError becomes a typed fold failure (not an
        # uncaught process error) under the fold contract's fail-closed code
        # set. An RSS breach is classified as RESOURCE_BUDGET_BREACH (spec
        # §3.3 ``fold_integrity``), never as an invalid primary ledger.
        assert report.strict is None
        assert report.stress is None
        assert report.failures == (ev.MHS_GO_REASON_RESOURCE_BREACH,)

    def test_no_rss_budget_returns_complete_fold(self, mhs_market) -> None:
        report = self._run_fold(mhs_market, max_rss_bytes=None)
        assert report.strict is not None or report.failures == (
            ev.MHS_GO_REASON_PRIMARY_SHARPE,
            ev.MHS_GO_REASON_STRESS_SHARPE,
        )


def test_anchored_fold_is_two_pass(mhs_market, monkeypatch) -> None:
    # SCENARIO_ANCHORED_FOLD_IS_TWO_PASS: the fold's reported primary
    # (strict/autocorr-sharpe/max-drawdown) reflects the P&L-vol-target
    # rescaled Pass-2 replay, not the unscaled reference. An engineered
    # non-trivial scale must move the fold's reported metrics away from the
    # all-ones (reference-equivalent) run.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ][:8]
    funding_by_symbol = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
    )

    def _all_ones_scale(reference_daily_returns: pd.Series) -> pd.Series:
        return pd.Series(1.0, index=reference_daily_returns.index)

    def _forced_step_scale(reference_daily_returns: pd.Series) -> pd.Series:
        idx = reference_daily_returns.index
        mid = idx[0] + (idx[-1] - idx[0]) / 2
        return pd.Series(np.where(idx < mid, 1.0, 0.2), index=idx)

    monkeypatch.setattr(ev, "_pnl_vol_target_scale", _all_ones_scale)
    reference = ev._run_anchored_fold(
        str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
    )
    monkeypatch.setattr(ev, "_pnl_vol_target_scale", _forced_step_scale)
    rescaled = ev._run_anchored_fold(
        str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
    )
    assert reference.strict is not None
    assert reference.stress is not None
    assert rescaled.strict is not None
    assert rescaled.stress is not None
    assert rescaled.primary_autocorr_sharpe != reference.primary_autocorr_sharpe
    assert rescaled.primary_max_drawdown != reference.primary_max_drawdown


def _build_book_outcome_args(mhs_market) -> dict[str, object]:
    """Replicate the top-level diagnostic setup needed to invoke ``_book_outcome``."""
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ][:8]
    funding_by_symbol = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
    )
    panel = ev.load_base_panel(
        root, "1h", ("close", "open", "quote_vol"), _START, end,
        partition="dev", min_bars=2000,
    )
    close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    grid_1h = close.index
    funded = [s for s in close.columns if s in funding_by_symbol]
    close = close[funded]
    opens = opens[funded]
    quote_vol = quote_vol[funded]
    bar_period = grid_1h[1] - grid_1h[0]
    funding_window = {
        s: funding_by_symbol[s].loc[
            (funding_by_symbol[s].index >= grid_1h[0])
            & (funding_by_symbol[s].index < grid_1h[-1] + bar_period)
        ]
        for s in funded
    }
    bar_funding = ev.bar_funding_panel(funding_window, grid_1h)
    aligned = list(bar_funding.columns)
    close = close[aligned]
    opens = opens[aligned]
    quote_vol = quote_vol[aligned]
    bar_funding = bar_funding[aligned]
    funding_by_symbol = {s: funding_by_symbol[s] for s in aligned}
    eligible = ev.liquid_half_eligibility(quote_vol, lookback_bars=720, min_history_bars=720)
    log_close = np.log(close)
    fast = ev.PHASE_1_BOOK_SPECS["fast_reversal"]
    fast_grid = pd.date_range(_START, end, freq="6h", tz="UTC")
    w_fast = ev._book_weights(log_close, eligible, fast, fast_grid)
    phase = ev._phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, fast)
    execution_mask = ev._pit_execution_mask(quote_vol, eligible, request.execution_universe_size)
    w_fast_execution = ev.renormalize_within_mask(
        w_fast, execution_mask.reindex(w_fast.index).fillna(False), fast.min_symbols,
    )
    return {
        "name": "fast_reversal",
        "spec": fast,
        "n_symbols": len(aligned),
        "step_grid": fast_grid,
        "weights_step": w_fast,
        "grid_1h": grid_1h,
        "opens": opens,
        "bar_funding": bar_funding,
        "phase": phase,
        "root": str(root),
        "request": request,
        "funding_by_symbol": funding_by_symbol,
        "start": _START,
        "end": end,
        "event_window_bars": fast.horizon_hours,
        "initial_equity": 1.0,
        "replay_weights_step": w_fast_execution,
    }


def test_fold_execution_weights_are_renormalized(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_FOLD_EXECUTION_WEIGHTS_ARE_RENORMALIZED: the fold builder
    # re-normalizes its execution weights onto the roster instead of collapsing
    # them to a partial-gross subset of the full-universe book.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    real = ev.renormalize_within_mask
    captured: list[tuple[pd.DataFrame, pd.DataFrame, int]] = []

    def spy(weights, mask, min_symbols):
        out = real(weights, mask, min_symbols)
        captured.append((out, mask, min_symbols))
        return out

    monkeypatch.setattr(ev, "renormalize_within_mask", spy)
    target_weights, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )
    assert captured, "fold builder must route execution weights through renormalize_within_mask"
    assert not target_weights.empty
    for out, mask, min_symbols in captured:
        live = mask.sum(axis=1) >= min_symbols
        assert live.any(), "fold decision rows must have a live roster"
        # unit-gross and dollar-neutral within the surviving roster cells
        assert out.abs().sum(axis=1).where(live).sub(1.0).abs().max() < 1e-9
        assert out.sum(axis=1).where(live).abs().max() < 1e-9
        # masked-out columns are exactly zero, never the unnormalized input
        assert float(out[~mask].abs().max().max()) == 0.0


def test_fold_weights_are_vol_tilted_before_renormalization(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_FOLD_WEIGHTS_ARE_VOL_TILTED_BEFORE_RENORMALIZATION: the fold
    # builder tilts each book by its own-horizon inverse realized vol before the
    # unchanged renormalize_within_mask, so a higher-vol roster symbol receives
    # a smaller post-tilt, pre-renormalization magnitude than an equal-rank
    # lower-vol symbol.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )

    tilt_calls: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    renorm_inputs: list[pd.DataFrame] = []
    real_tilt = ev.inverse_realized_vol_tilt
    real_renorm = ev.renormalize_within_mask

    def tilt_spy(weights, vol):
        tilt_calls.append((weights, vol))
        return real_tilt(weights, vol)

    def renorm_spy(weights, mask, min_symbols):
        renorm_inputs.append(weights)
        return real_renorm(weights, mask, min_symbols)

    monkeypatch.setattr(ev, "inverse_realized_vol_tilt", tilt_spy)
    monkeypatch.setattr(ev, "renormalize_within_mask", renorm_spy)
    ev._build_fold_target_weights(str(root), _FOLD, request, funding_by_symbol)

    assert len(tilt_calls) == 2, "fold builder must tilt both the fast and slow books"
    assert len(renorm_inputs) == 2, "fold builder must renormalize both tilted books"
    for (raw, vol), renorm_in in zip(tilt_calls, renorm_inputs, strict=True):
        # renormalize receives the tilt output -- the raw rank book scaled by
        # 1/vol -- never the untilted book.
        assert renorm_in.equals(real_tilt(raw, vol))
        valid = np.isfinite(vol.to_numpy(dtype="float64")) & (vol.to_numpy(dtype="float64") > 0.0)
        assert valid.any(), "tilt must be a real scaling, not a no-op"

    # The tilt is applied on each book's own horizon and reindexed onto its grid.
    fast = ev.PHASE_1_BOOK_SPECS["fast_reversal"]
    slow = ev.PHASE_1_BOOK_SPECS["slow_momentum"]
    panel_start = max(
        _FOLD.train_start,
        _FOLD.validation_start - pd.Timedelta(hours=ev.MHS_FOLD_PANEL_WARMUP_HOURS),
    )
    fast_grid = pd.date_range(panel_start, _FOLD.validation_end, freq="6h", tz="UTC")
    slow_grid = pd.date_range(panel_start, _FOLD.validation_end, freq="24h", tz="UTC")
    fast_raw, fast_vol = tilt_calls[0]
    slow_raw, slow_vol = tilt_calls[1]
    assert fast_raw.index.equals(fast_grid)
    assert fast_vol.index.equals(fast_grid)
    assert slow_raw.index.equals(slow_grid)
    assert slow_vol.index.equals(slow_grid)

    # Semantic ordering: among roster symbols sharing an equal raw rank-slot
    # magnitude (the book's symmetric extremes), the higher-realized-vol symbol
    # has the strictly smaller pre-renormalization magnitude.
    fast_tilted = real_tilt(fast_raw, fast_vol)
    pairs: list[tuple[int, int, int, float, float]] = []
    for row in range(len(fast_tilted)):
        mags = fast_raw.iloc[row].abs().to_numpy(dtype="float64")
        vols = fast_vol.iloc[row].to_numpy(dtype="float64")
        valid = np.isfinite(vols) & (vols > 0.0) & (mags > 1e-6)
        pairs.extend(
            (row, i, j, float(vols[i]), float(vols[j]))
            for i in range(len(mags))
            for j in range(i + 1, len(mags))
            if valid[i] and valid[j] and np.isclose(mags[i], mags[j]) and vols[i] != vols[j]
        )
    assert pairs, "fixture must contain equal-|rank-weight| pairs with differing realized vol"
    for row, i, j, vi, vj in pairs:
        hi, lo = (i, j) if vi > vj else (j, i)
        assert abs(fast_tilted.iloc[row, hi]) < abs(fast_tilted.iloc[row, lo])


def _roster_mask_panel_inputs(
    root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    funding_by_symbol: dict[str, pd.Series],
    universe_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Replicate the diagnostic panel prep to independently recompute the
    execution_mask-filtered vol_mean that production must feed _regime_cash_scale."""
    panel = ev.load_base_panel(
        root, "1h", ("close", "open", "quote_vol"), start, end,
        partition="dev", min_bars=2000,
    )
    close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    grid_1h = close.index
    funded = [s for s in close.columns if s in funding_by_symbol]
    close = close[funded]
    opens = opens[funded]
    quote_vol = quote_vol[funded]
    bar_period = grid_1h[1] - grid_1h[0]
    funding_window = {
        s: funding_by_symbol[s].loc[
            (funding_by_symbol[s].index >= grid_1h[0])
            & (funding_by_symbol[s].index < grid_1h[-1] + bar_period)
        ]
        for s in funded
    }
    bar_funding = ev.bar_funding_panel(funding_window, grid_1h)
    aligned = list(bar_funding.columns)
    close = close[aligned]
    quote_vol = quote_vol[aligned]
    eligible = ev.liquid_half_eligibility(quote_vol, lookback_bars=720, min_history_bars=720)
    log_close = np.log(close)
    execution_mask = ev._pit_execution_mask(quote_vol, eligible, universe_size)
    return log_close, execution_mask, grid_1h


def _assert_regime_vol_mean_roster_masked(
    captured: dict[str, pd.Series],
    log_close: pd.DataFrame,
    execution_mask: pd.DataFrame,
    grid: pd.DatetimeIndex,
) -> None:
    """Assert the production vol_mean equals the execution_mask-filtered mean and
    genuinely excludes non-roster symbols (masked mean != full-universe mean)."""
    expected = ev.realized_vol(log_close, 48).where(execution_mask).reindex(grid).mean(axis=1)
    all_universe = ev.realized_vol(log_close, 48).reindex(grid).mean(axis=1)
    pd.testing.assert_series_equal(captured["vol_mean"], expected)
    assert int(execution_mask.sum(axis=1).max()) < execution_mask.shape[1]
    assert not expected.equals(all_universe)


def test_fold_vol_mean_masked_to_execution_roster(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_VOL_MEAN_ROSTER_MASK_01: the fold builder's regime-cash-scale
    # vol_mean is computed from execution_mask-filtered realized vol -- a
    # high-vol symbol outside the traded roster must not pull the regime scale.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    captured: dict[str, pd.Series] = {}
    real_scale = ev._regime_cash_scale

    def spy(vol_mean, *args, **kwargs):
        captured["vol_mean"] = vol_mean.copy()
        return real_scale(vol_mean, *args, **kwargs)

    monkeypatch.setattr(ev, "_regime_cash_scale", spy)
    ev._build_fold_target_weights(str(root), _FOLD, request, funding_by_symbol)
    assert "vol_mean" in captured, "fold builder must feed _regime_cash_scale its vol_mean"

    panel_start = max(
        _FOLD.train_start,
        _FOLD.validation_start - pd.Timedelta(hours=ev.MHS_FOLD_PANEL_WARMUP_HOURS),
    )
    log_close, execution_mask, _grid = _roster_mask_panel_inputs(
        root, panel_start, _FOLD.validation_end, funding_by_symbol,
        request.execution_universe_size,
    )
    _assert_regime_vol_mean_roster_masked(
        captured, log_close, execution_mask, captured["vol_mean"].index,
    )


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
    funding_by_symbol = ev._load_funding_series(symbols)
    captured: dict[str, pd.Series] = {}
    real_scale = ev._regime_cash_scale

    def spy(vol_mean, *args, **kwargs):
        captured["vol_mean"] = vol_mean.copy()
        return real_scale(vol_mean, *args, **kwargs)

    monkeypatch.setattr(ev, "_regime_cash_scale", spy)
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(ev, "phase_1_anchored_purged_folds", lambda: ())
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    assert "vol_mean" in captured, "top-level diagnostic must feed _regime_cash_scale its vol_mean"

    log_close, execution_mask, _grid = _roster_mask_panel_inputs(
        root, _START, end, funding_by_symbol, request.execution_universe_size,
    )
    _assert_regime_vol_mean_roster_masked(
        captured, log_close, execution_mask, captured["vol_mean"].index,
    )


class TestBookOutcomePaired:
    """MHS-MEM-PAIR-02-BOOK: the top-level book orchestrator builds the
    execution-window iterator once per strict/stress pair and preserves the
    typed book failure conversion."""

    def test_book_builds_window_iterator_once_per_bound(self, mhs_market, monkeypatch) -> None:
        args = _build_book_outcome_args(mhs_market)
        calls = {"n": 0}
        original = ev._iter_mhs_execution_windows

        def counting(*_args, **_kwargs):
            calls["n"] += 1
            return original(*_args, **_kwargs)

        monkeypatch.setattr(ev, "_iter_mhs_execution_windows", counting)
        report = ev._book_outcome(**args)
        assert report.primary is not None
        assert report.stress is not None
        assert report.failure is None
        # Two-pass primary (reference + rescaled) plus stress and the
        # informational patient-reference bound.
        assert calls["n"] == 4

    def test_book_strict_resource_breach_is_typed_failure(self, mhs_market, monkeypatch) -> None:
        args = _build_book_outcome_args(mhs_market)
        args["request"] = MhsDiagnosticRequest(
            start=str(_START), end=str(args["end"]), data_root=args["root"],
            mark_mode="cache_required", execution_timeframe="1m", log_run=False,
            max_rss_bytes=1_000,
        )
        monkeypatch.setattr(ev, "_current_rss_bytes", lambda: 100_000_000_000)
        report = ev._book_outcome(**args)
        assert report.primary is None
        assert report.stress is None
        assert report.failure is not None
        assert report.failure.stage == "replay_fast_reversal"
        assert report.failure.reason == ev.MHS_GO_REASON_RESOURCE_BREACH


def test_book_outcome_is_two_pass(mhs_market, monkeypatch) -> None:
    # SCENARIO_BOOK_OUTCOME_IS_TWO_PASS: the reported primary is the
    # P&L-vol-target-rescaled Pass-2 replay, with Pass 1 kept as the
    # pre_vol_target_reference diagnostic. The fields are populated on the
    # natural fixture, and an engineered non-trivial scale proves Pass 2
    # genuinely re-ran with different weights (the 8th-iteration no-op class).
    args = _build_book_outcome_args(mhs_market)
    report = ev._book_outcome(**args)
    assert report.pre_vol_target_reference is not None
    assert report.pre_vol_target_reference_naive_sharpe is not None
    assert report.primary is not None
    assert report.pre_vol_target_reference.fill_source == report.primary.fill_source

    def _forced_step_scale(reference_daily_returns: pd.Series) -> pd.Series:
        idx = reference_daily_returns.index
        mid = idx[0] + (idx[-1] - idx[0]) / 2
        return pd.Series(np.where(idx < mid, 1.0, 0.2), index=idx)

    monkeypatch.setattr(ev, "_pnl_vol_target_scale", _forced_step_scale)
    forced = ev._book_outcome(**args)
    assert forced.pre_vol_target_reference is not None
    assert forced.pre_vol_target_reference_naive_sharpe is not None
    assert forced.primary_naive_sharpe != forced.pre_vol_target_reference_naive_sharpe


def test_xs_rank_ic_causal_forward_window_ignores_invalid_cells() -> None:
    # SCENARIO_MHS_ALPHA_ENGINE_06: the forward return is built internally as
    # opens.pct_change(forward_bars).shift(-(forward_bars + 1)); the measured
    # window starts at open_{t+1} and never overlaps the signal's own lookback.
    # With forward_bars=1, fwd[t] = (open[t+2] - open[t+1]) / open[t+1].
    index = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
    opens = pd.DataFrame(
        [[100.0, 100.0, 100.0, 100.0, 100.0],
         [100.0, 100.0, 100.0, 100.0, 100.0],
         [110.0, 105.0, 100.0, 95.0, 90.0],
         [110.0, 105.0, np.nan, 95.0, 90.0]],
        index=index,
        columns=list("ABCDE"),
    )
    signal = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0],
         [1.0, 2.0, np.nan, 4.0, 5.0],
         [1.0, 1.0, 1.0, 1.0, 1.0],
         [5.0, 4.0, 3.0, 2.0, 1.0]],
        index=index,
        columns=list("ABCDE"),
    )
    result = ev._xs_rank_ic(signal, opens, forward_bars=1)
    # Row 0 is the only valid cross section (>= 5 finite cells): ascending
    # signal ranks against the descending forward returns score IC exactly -1.
    # Row 1 has a NaN signal cell (< 5 valid cells, excluded), rows 2-3 have no
    # forward window (NaN, excluded).
    assert result["n_dates"] == 1
    assert result["mean_ic"] == pytest.approx(-1.0)
    assert result["forward_bars"] == 1


def test_xs_rank_ic_causal_window_scores_near_zero_on_unpredictable_returns() -> None:
    # SCENARIO_MHS_ALPHA_ENGINE_06: on IID returns, a signal equal to the
    # TRAILING return scores ~0 under the tradable (non-overlapping) window,
    # instead of the spuriously high overlap IC the old trailing convention
    # reported (+0.0957 vs the tradable -0.0278 in the spec).
    rng = np.random.default_rng(7)
    n_hours, n_syms = 60, 10
    index = pd.date_range("2021-01-01", periods=n_hours, freq="1h", tz="UTC")
    cols = [f"S{i}" for i in range(n_syms)]
    opens = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, (n_hours, n_syms)), axis=0)),
        index=index,
        columns=cols,
    )
    signal = opens.pct_change()
    result = ev._xs_rank_ic(signal, opens, forward_bars=1)
    assert result["n_dates"] > 20
    assert abs(result["mean_ic"]) < 0.3
    assert result["forward_bars"] == 1
    with pytest.raises(ValueError, match="forward_bars"):
        ev._xs_rank_ic(signal, opens, forward_bars=0)


def test_date_clustered_ols_causal_forward_window() -> None:
    # SCENARIO_MHS_ALPHA_ENGINE_06: the pooled panel regression builds its
    # dependent variable internally with shift(-(forward_bars + 1)); with
    # forward_bars=1 the forward window is fwd[t] = r[t + 2], and step returns
    # r[t] = 1.5 * past[t - 2] + 0.25 recover the known 1.5 slope exactly.
    index = pd.date_range("2021-01-01", periods=48, freq="1h", tz="UTC")
    past = pd.DataFrame(
        {"A": np.arange(48, dtype=float), "B": np.arange(48, dtype=float) + 2.0},
        index=index,
    )
    step_ret = np.vstack(
        [np.zeros((2, past.shape[1])), (0.25 + 1.5 * past.iloc[:-2]).to_numpy()],
    )
    opens = pd.DataFrame(100.0 * np.cumprod(1.0 + step_ret, axis=0), index=index, columns=past.columns)
    opens.iloc[3, 0] = np.nan

    result = ev._date_clustered_ols(opens, past, forward_bars=1)
    # The NaN at opens[3, "A"] poisons the pct_change for two forward cells
    # (fwd[1] reads r[3], fwd[2] divides by open[3]); the last two bars have no
    # forward window, so 96 - 4 (terminal) - 2 (poisoned) = 90 finite pairs.
    assert result["n"] == 90
    assert result["n_dates"] == 2
    assert result["past_beta"] == pytest.approx(1.5, rel=1e-3)
    assert result["forward_bars"] == 1
    with pytest.raises(ValueError, match="forward_bars"):
        ev._date_clustered_ols(opens, past, forward_bars=0)


def _perf_opt_placebo_inputs(seed: int) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, BookSpec,
]:
    """Synthetic but structurally faithful placebo inputs.

    Signals are continuous log-price levels, eligibility is a per-symbol
    monotone listing lifecycle (the same shape ``liquid_half_eligibility``
    produces), and opens are NaN before listing so the active-cell ledger guard
    is exercised exactly as in production.
    """
    rng = np.random.default_rng(seed)
    n_hours, n_syms = 600, 10
    grid = pd.date_range("2023-01-01", periods=n_hours, freq="1h", tz="UTC")
    cols = [f"SYM{i}" for i in range(n_syms)]
    base = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0003, (n_hours, n_syms)), axis=0))
    signal = pd.DataFrame(base, index=grid, columns=cols)
    listing = rng.integers(0, n_hours // 4, size=n_syms)
    elig_raw = np.arange(n_hours)[:, None] >= listing[None, :]
    eligible = pd.DataFrame(elig_raw, index=grid, columns=cols)
    opens = pd.DataFrame(
        base * (1.0 + rng.normal(0.0, 0.0001, (n_hours, n_syms))), index=grid, columns=cols,
    )
    opens = opens.mask(~elig_raw)
    bar_funding = pd.DataFrame(
        rng.normal(0.00005, 0.00001, (n_hours, n_syms)), index=grid, columns=cols,
    )
    return signal, eligible, opens, bar_funding, grid, ev.PHASE_1_BOOK_SPECS["fast_reversal"]


def _reference_placebo_percentile(
    signal, eligible, opens, bar_funding, grid_1h, spec, observed_sharpe, n_placebos, seed,
):
    """Original pandas DataFrame-per-iteration placebo loop (baseline)."""
    from src.mhs.books import phase_tranche_book, rank_weight_book
    from src.mhs.execution import mhs_ledger_pnl

    rng = np.random.default_rng(seed)
    ranks = []
    cols = list(signal.columns)
    sig_step = signal.reindex(grid_1h)
    el_step = eligible.reindex(grid_1h)
    for _p in range(n_placebos):
        perm = rng.permutation(len(cols))
        shuffled = sig_step.copy()
        permuted_cols = [cols[i] for i in perm]
        shuffled.columns = permuted_cols
        el_shuffled = el_step.copy()
        el_shuffled.columns = permuted_cols
        weights_p = rank_weight_book(shuffled, el_shuffled, spec.band.sign, spec.min_symbols)
        weights_p = phase_tranche_book(weights_p, spec.tranche_count())
        weights_1h = weights_p.reindex(grid_1h).ffill().fillna(0.0)
        try:
            net, _t = mhs_ledger_pnl(
                weights_1h, opens[permuted_cols], bar_funding[permuted_cols], 8.0,
            )
        except DataIntegrityError:
            continue
        sd = float(net.std(ddof=1)) if len(net) > 1 else 0.0
        if sd > 0:
            ranks.append(float(net.mean() / sd * np.sqrt(ev._PERIODS_PER_YEAR_1H)))
    if not ranks:
        return None
    return float(np.mean([1.0 if observed_sharpe >= r else 0.0 for r in ranks]))


def test_mhs_perf_opt_001_placebo_vectorized_exact_and_fast() -> None:
    # MHS_PERF_OPT_001_PLACEBO_VECTORIZED: the vectorized NumPy placebo must
    # reproduce the baseline percentile exactly and run >= 5x faster.
    signal, eligible, opens, bar_funding, grid, spec = _perf_opt_placebo_inputs(20260807)
    n_placebos = 300
    for observed in (0.7, -1.5, 0.0):
        expected = _reference_placebo_percentile(
            signal, eligible, opens, bar_funding, grid, spec, observed, n_placebos, 7,
        )
        actual = ev._placebo_sharpe_percentile(
            signal, eligible, opens, bar_funding, grid, spec, observed, n_placebos, 7,
        )
        assert (expected is None and actual is None) or (expected == actual)

    t0 = time.perf_counter()
    _reference_placebo_percentile(
        signal, eligible, opens, bar_funding, grid, spec, 0.7, n_placebos, 7,
    )
    reference_elapsed = time.perf_counter() - t0
    t1 = time.perf_counter()
    ev._placebo_sharpe_percentile(
        signal, eligible, opens, bar_funding, grid, spec, 0.7, n_placebos, 7,
    )
    vectorized_elapsed = time.perf_counter() - t1
    assert vectorized_elapsed < reference_elapsed / 5.0


def _write_quote_volume_market(root: Path, symbols: list[str]) -> tuple[pd.DatetimeIndex, int]:
    """Write 1-minute ``quote_vol`` parquet files and return the minute grid."""
    start = pd.Timestamp("2023-02-01", tz="UTC")
    grid = pd.date_range(start, periods=6 * 24, freq="1min", tz="UTC")
    epoch = (grid - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    base_vol = np.linspace(500.0, 900.0, len(grid))
    (root / "1m").mkdir(parents=True, exist_ok=True)
    for i, sym in enumerate(symbols):
        vol = base_vol * (1.0 + 0.1 * (i + 1)) + np.sin(np.arange(len(grid)) / 12.0) * 50.0
        pd.DataFrame({"timestamp": epoch, "quote_vol": vol}).to_parquet(
            root / "1m" / f"{sym}.parquet", index=False,
        )
    return grid, len(symbols)


def _reference_participation_warnings(replay, root, timeframe, symbols, minute_grid):
    """Original iterrows()/``.loc[t:window_end]`` participation loop (baseline)."""
    if replay.simulated_fills.empty:
        return {}
    fills = replay.simulated_fills
    notional = float((fills["quantity_delta"].abs() * fills["fill_price"]).sum())
    fills_by_symbol = {}
    for _sym, group in fills.groupby("symbol"):
        fills_by_symbol[str(_sym)] = group
    daily_volume = 0.0
    window_totals = {"1m": 0.0, "30m": 0.0}
    window_minutes = (("1m", 1), ("30m", 30))
    for sym in symbols:
        series = ev._load_symbol_quote_volume(
            root, sym, timeframe, minute_grid[0], minute_grid[-1],
        )
        if series is None:
            continue
        daily_volume += float(series.sum())
        group = fills_by_symbol.get(sym)
        if group is None:
            continue
        for _i, row in group.iterrows():
            t = row["timestamp"]
            if t not in series.index:
                continue
            for window_label, minutes in window_minutes:
                window_end = t + pd.Timedelta(minutes=minutes)
                window_totals[window_label] += float(series.loc[t:window_end].sum())
    warnings = {}
    for window_label, _minutes in window_minutes:
        total_volume = window_totals[window_label]
        warnings[f"fill_notional_to_{window_label}_quote_volume"] = (
            notional / total_volume if total_volume > 0 else float("nan")
        )
    warnings["daily_trade_notional_to_daily_quote_volume"] = (
        notional / daily_volume if daily_volume > 0 else float("nan")
    )
    return warnings


def test_mhs_perf_opt_002_participation_cumsum_exact(tmp_path) -> None:
    # MHS_PERF_OPT_002_PARTICIPATION_CUMSUM: the cumsum/searchsorted rewrite
    # must return the exact same warnings dict as the iterrows() baseline.
    symbols = ["SYMA", "SYMB", "SYMC"]
    grid, _ = _write_quote_volume_market(tmp_path, symbols)
    rng = np.random.default_rng(42)
    rows = []
    for i, sym in enumerate(symbols):
        # Minute-aligned fills inside the quote-volume window, some off-grid.
        ts = grid[200 + i::(600 + i * 5)].to_list()
        for j, t in enumerate(ts[:40]):
            rows.append(
                {
                    "timestamp": t,
                    "symbol": sym,
                    "quantity_delta": 0.5 if j % 2 == 0 else -0.5,
                    "fill_price": 100.0 + rng.normal(0.0, 1.0),
                }
            )
    fills = pd.DataFrame(rows)
    replay = types.SimpleNamespace(simulated_fills=fills)

    expected = _reference_participation_warnings(
        replay, str(tmp_path), "1m", symbols, grid,
    )
    actual = ev._participation_warnings(replay, str(tmp_path), "1m", symbols, grid)
    assert set(actual) == set(expected)
    for key in expected:
        assert actual[key] == expected[key]


def _reference_bootstrap_ci(net, n_replicates, mean_block, seed):
    """Original scalar while-loop block bootstrap (baseline)."""
    rng = np.random.default_rng(seed)
    arr = net.to_numpy(dtype="float64")
    n = len(arr)
    means = []
    p_block = 1.0 / mean_block if mean_block > 0 else 0.0
    for _r in range(n_replicates):
        blocks = []
        while len(blocks) < n:
            start = int(rng.integers(0, n))
            length = 1
            while length < n and rng.random() > p_block:
                length += 1
            length = min(length, n - len(blocks))
            blocks.extend(arr[start : start + length].tolist())
        means.append(float(np.mean(blocks[:n])))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def test_mhs_perf_opt_003_bootstrap_vectorized_equivalent() -> None:
    # MHS_PERF_OPT_003_BOOTSTRAP_VECTORIZED: 2D block sampling must produce
    # statistically equivalent CI bounds (the RNG draw order differs by design,
    # so exact reproduction is neither required nor possible).
    rng = np.random.default_rng(5)
    net = pd.Series(np.cumsum(rng.normal(0.0, 0.01, 400)))
    for seed in (20260807, 3, 11):
        lo_ref, hi_ref = _reference_bootstrap_ci(net, 800, 24, seed)
        lo_new, hi_new = ev._bootstrap_ci(net, 800, 24, seed)
        assert lo_new < hi_new
        assert lo_ref < hi_ref
        assert abs(lo_new - lo_ref) < 0.05
        assert abs(hi_new - hi_ref) < 0.05


def test_mhs_perf_opt_004_minute_frame_cache_reuses_reads(mhs_market, monkeypatch) -> None:
    # MHS_PERF_OPT_004_MINUTE_FRAME_CACHE: repeated ``_load_minute_frames``
    # calls with identical windows must not re-open the same Parquet files, and
    # the returned dict must be a fresh copy so callers cannot poison the cache.
    root, end = mhs_market
    syms = ["MHSAUSDT", "MHSBUSDT"]
    calls = {"n": 0}
    original = ev.pq.read_table

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ev.pq, "read_table", counting)

    a = ev._load_minute_frames(str(root), syms, _START, end, "1m")
    assert a
    first_calls = calls["n"]
    b = ev._load_minute_frames(str(root), syms, _START, end, "1m")
    assert calls["n"] == first_calls
    assert set(a) == set(b)
    for k in a:
        assert a[k].equals(b[k])

    ev._load_minute_frames(str(root), syms, _START, _START + pd.Timedelta(hours=1), "1m")
    assert calls["n"] == first_calls + len(syms)

    cached = ev._load_minute_frames(str(root), syms, _START, end, "1m")
    cached.clear()
    fresh = ev._load_minute_frames(str(root), syms, _START, end, "1m")
    assert set(fresh) == set(a)


def test_mhs_phase2_o6_window_frames_parity(mhs_market) -> None:
    # SCENARIO_O6_FRAME_PARITY: ``_get_symbol_minute_frame`` + ``_build_window_frames``
    # (per-symbol full-period cache + slice) produce identical highs/lows/closes to
    # the old ``_load_minute_frames`` + ``_align_minute_frames`` per-window read path.
    root, end = mhs_market
    syms = ["MHSAUSDT", "MHSBUSDT", "MHSCUSDT"]
    ws = pd.Timestamp("2021-01-01 06:00", tz="UTC")
    we = pd.Timestamp("2021-01-02 06:00", tz="UTC")
    grid = pd.date_range(ws, we, freq="1min", tz="UTC")

    old_frames = ev._load_minute_frames(str(root), syms, ws, we, "1m")
    old_aligned = ev._align_minute_frames(old_frames, "1m", ws, we)
    assert old_aligned is not None

    new_frames = {s: ev._get_symbol_minute_frame(str(root), s, "1m") for s in syms}
    new_aligned = ev._build_window_frames(new_frames, syms, ws, we, grid, "1m")
    assert new_aligned is not None

    old_highs, old_lows, old_closes = old_aligned
    new_highs, new_lows, new_closes = new_aligned
    for old, new, name in ((old_highs, new_highs, "highs"), (old_lows, new_lows, "lows"), (old_closes, new_closes, "closes")):
        assert list(new.columns) == list(old.columns), name
        assert new.index.equals(old.index), name
        assert np.isclose(new.to_numpy(), old.to_numpy(), rtol=0, atol=0, equal_nan=True).all(), name


def test_mhs_phase2_o6_missing_symbol_raises(mhs_market) -> None:
    # O6: full-period cache fails closed on a missing parquet (old per-window
    # path silently skipped it).
    root, _end = mhs_market
    with pytest.raises(ev.DataIntegrityError):
        ev._get_symbol_minute_frame(str(root), "NOSUCHUSDT", "1m")


def test_mhs_phase2_o10_bootstrap_chunk_adaptive() -> None:
    # SCENARIO_O10_RSS_GATE: chunk is capped so a (chunk, n) sample matrix stays
    # <= 128MB; at production 5m scale (525,600 bars) that means a small chunk.
    from src.mhs.evaluation import _bootstrap_chunk_size

    assert _bootstrap_chunk_size(525_600) <= 63
    assert _bootstrap_chunk_size(43_830) >= 100
    assert _bootstrap_chunk_size(0) == 500


def _reference_resolve_ns_scalar(
    spos_all: np.ndarray,
    full_grid_ns: np.ndarray,
    n_grid: int,
    timeout_ns_delta: int,
) -> np.ndarray:
    """The Phase-2 scalar resolve_ns loop (the parity reference for P11)."""
    resolve_ns = np.full(len(spos_all), -1, dtype="int64")
    for i in range(len(spos_all)):
        s = int(spos_all[i])
        if s >= n_grid:
            continue
        tns = full_grid_ns[s] + timeout_ns_delta
        tpos = int(np.searchsorted(full_grid_ns, tns, side="left"))
        if tpos < n_grid and full_grid_ns[tpos] == tns:
            resolve_ns[i] = tns
    return resolve_ns


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


def _build_books_concurrent_args(
    mhs_market, universe_size: int | None = None,
) -> dict[str, object]:
    """Replicate the top-level diagnostic setup for all three books.

    ``universe_size`` narrows the execution roster (default 30 keeps every
    fixture symbol); a value between ``min_symbols`` and the eligible count
    exercises the renormalization that rescales surviving roster cells.
    """
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        **({"execution_universe_size": universe_size} if universe_size is not None else {}),
    )
    panel = ev.load_base_panel(
        root, "1h", ("close", "open", "quote_vol"), _START, end,
        partition="dev", min_bars=2000,
    )
    close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    grid_1h = close.index
    funded = [s for s in close.columns if s in funding_by_symbol]
    close = close[funded]
    opens = opens[funded]
    quote_vol = quote_vol[funded]
    bar_period = grid_1h[1] - grid_1h[0]
    funding_window = {
        s: funding_by_symbol[s].loc[
            (funding_by_symbol[s].index >= grid_1h[0])
            & (funding_by_symbol[s].index < grid_1h[-1] + bar_period)
        ]
        for s in funded
    }
    bar_funding = ev.bar_funding_panel(funding_window, grid_1h)
    aligned = list(bar_funding.columns)
    close = close[aligned]
    opens = opens[aligned]
    quote_vol = quote_vol[aligned]
    bar_funding = bar_funding[aligned]
    funding_by_symbol = {s: funding_by_symbol[s] for s in aligned}
    eligible = ev.liquid_half_eligibility(quote_vol, lookback_bars=720, min_history_bars=720)
    log_close = np.log(close)
    fast = ev.PHASE_1_BOOK_SPECS["fast_reversal"]
    slow = ev.PHASE_1_BOOK_SPECS["slow_momentum"]
    fast_grid = pd.date_range(_START, end, freq="6h", tz="UTC")
    slow_grid = pd.date_range(_START, end, freq="24h", tz="UTC")
    w_fast = ev._book_weights(log_close, eligible, fast, fast_grid)
    w_slow = ev._book_weights(log_close, eligible, slow, slow_grid)
    phase_fast = ev._phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, fast)
    phase_slow = ev._phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, slow)
    phase_blend = ev._phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, fast)
    execution_mask = ev._pit_execution_mask(quote_vol, eligible, request.execution_universe_size)
    w_fast_execution = ev.renormalize_within_mask(
        w_fast, execution_mask.reindex(w_fast.index).fillna(False), fast.min_symbols,
    )
    w_slow_execution = ev.renormalize_within_mask(
        w_slow, execution_mask.reindex(w_slow.index).fillna(False), slow.min_symbols,
    )
    w_fast_1h = w_fast.reindex(grid_1h).ffill().fillna(0.0)
    w_slow_1h = w_slow.reindex(grid_1h).ffill().fillna(0.0)
    blend_1h = (
        ev.PHASE_1_BOOK_BLEND_WEIGHTS["fast_reversal"] * w_fast_1h
        + ev.PHASE_1_BOOK_BLEND_WEIGHTS["slow_momentum"] * w_slow_1h
    )
    vol_mean = ev.realized_vol(log_close, 48).where(execution_mask).reindex(grid_1h).mean(axis=1)
    regime_scale = ev._regime_cash_scale(vol_mean)
    blend_1h = blend_1h.mul(regime_scale, axis=0)
    return {
        "root": str(root),
        "request": request,
        "n_symbols": len(aligned),
        "grid_1h": grid_1h,
        "fast": fast,
        "slow": slow,
        "fast_grid": fast_grid,
        "slow_grid": slow_grid,
        "w_fast": w_fast,
        "w_slow": w_slow,
        "w_fast_execution": w_fast_execution,
        "w_slow_execution": w_slow_execution,
        "opens": opens,
        "bar_funding": bar_funding,
        "phase_fast": phase_fast,
        "phase_slow": phase_slow,
        "phase_blend": phase_blend,
        "start": _START,
        "end": end,
        "funding_by_symbol": funding_by_symbol,
        "blend_1h": blend_1h,
        "execution_mask": execution_mask,
        "initial_equity": 1.0,
    }


def _sequential_book_reports(args: dict[str, object]) -> tuple[object, object, object]:
    fast, slow = args["fast"], args["slow"]
    fast_grid, slow_grid = args["fast_grid"], args["slow_grid"]
    grid_1h = args["grid_1h"]
    fast_rpt = ev._book_outcome(
        "fast_reversal", fast, args["n_symbols"], fast_grid, args["w_fast"], grid_1h,
        args["opens"], args["bar_funding"], args["phase_fast"], args["root"],
        args["request"], args["funding_by_symbol"], args["start"], args["end"],
        fast.horizon_hours, args["initial_equity"], args["w_fast_execution"],
    )
    slow_rpt = ev._book_outcome(
        "slow_momentum", slow, args["n_symbols"], slow_grid, args["w_slow"], grid_1h,
        args["opens"], args["bar_funding"], args["phase_slow"], args["root"],
        args["request"], args["funding_by_symbol"], args["start"], args["end"],
        slow.horizon_hours, args["initial_equity"], args["w_slow_execution"],
    )
    active_spec, active_grid = ev._active_blend_book_and_grid(fast, slow, fast_grid, slow_grid)
    blend_step = args["blend_1h"].reindex(active_grid)
    blend_replay = (
        ev.PHASE_1_BOOK_BLEND_WEIGHTS["fast_reversal"] * args["w_fast_execution"].reindex(grid_1h).ffill().fillna(0.0)
        + ev.PHASE_1_BOOK_BLEND_WEIGHTS["slow_momentum"] * args["w_slow_execution"].reindex(grid_1h).ffill().fillna(0.0)
    ).reindex(active_grid)
    blend_rpt = ev._book_outcome(
        "blend", active_spec, args["n_symbols"], active_grid, blend_step, grid_1h,
        args["opens"], args["bar_funding"], args["phase_blend"], args["root"],
        args["request"], args["funding_by_symbol"], args["start"], args["end"],
        168, args["initial_equity"], blend_replay,
    )
    return fast_rpt, slow_rpt, blend_rpt


def _assert_books_equal(seq, con, name: str) -> None:
    assert con.name == name
    assert seq.failure is None
    assert con.failure is None
    assert seq.primary is not None
    assert con.primary is not None
    assert seq.stress is not None
    assert con.stress is not None
    assert len(con.primary.simulated_fills) == len(seq.primary.simulated_fills)
    assert con.primary_naive_sharpe == seq.primary_naive_sharpe
    assert con.primary_net_ann == seq.primary_net_ann
    assert con.primary_geometric_cagr == seq.primary_geometric_cagr
    assert con.stress_naive_sharpe == seq.stress_naive_sharpe
    pd.testing.assert_series_equal(
        con.primary.ledger.equity, seq.primary.ledger.equity,
        check_exact=True, rtol=0.0, atol=0.0,
    )


def test_p10_concurrent_books_parity(mhs_market) -> None:
    # SCENARIO_P10_CONCURRENT: three books executed concurrently in fork workers
    # produce bit-identical reports to the sequential path.
    args = _build_books_concurrent_args(mhs_market)
    sequential = _sequential_book_reports(args)
    concurrent = ev._run_books_concurrent(**args)
    assert len(concurrent) == 3
    for seq, con, name in zip(sequential, concurrent, ("fast_reversal", "slow_momentum", "blend"), strict=True):
        _assert_books_equal(seq, con, name)


def test_toplevel_blend_replay_matches_renormalized_components(mhs_market) -> None:
    # SCENARIO_MHS_TOPLEVEL_BLEND_REPLAY_MATCHES_RENORMALIZED_COMPONENTS: the
    # blend replay target is the weighted sum of the renormalized execution
    # books (each ffilled onto the 1h grid then reindexed onto the blend's
    # active execution grid), no longer a collapse of the pre-mask theoretical
    # blend.
    args = _build_books_concurrent_args(mhs_market, universe_size=8)
    grid_1h = args["grid_1h"]
    active_spec, active_grid = ev._active_blend_book_and_grid(
        args["fast"], args["slow"], args["fast_grid"], args["slow_grid"],
    )
    expected = (
        ev.PHASE_1_BOOK_BLEND_WEIGHTS["fast_reversal"] * args["w_fast_execution"].reindex(grid_1h).ffill().fillna(0.0)
        + ev.PHASE_1_BOOK_BLEND_WEIGHTS["slow_momentum"] * args["w_slow_execution"].reindex(grid_1h).ffill().fillna(0.0)
    ).reindex(active_grid)
    collapsed = args["blend_1h"].where(args["execution_mask"], other=0.0).reindex(active_grid)
    assert not expected.equals(collapsed), "renormalized blend must differ from the collapsed pre-mask blend"
    # the concurrent production path replays exactly the renormalized composition
    _, _, blend_report = ev._run_books_concurrent(**args)
    expected_report = ev._book_outcome(
        "blend", active_spec, args["n_symbols"], active_grid,
        args["blend_1h"].reindex(active_grid), grid_1h,
        args["opens"], args["bar_funding"], args["phase_blend"], args["root"],
        args["request"], args["funding_by_symbol"], args["start"], args["end"],
        168, args["initial_equity"], expected,
    )
    _assert_books_equal(expected_report, blend_report, "blend")


def test_p10_eager_cache_preload(mhs_market) -> None:
    # SCENARIO_P10_EAGER_CACHE: preloading fills the O6 cache for the execution
    # roster so forked workers never re-read the same Parquet.
    root, end = mhs_market
    syms = ["MHSAUSDT", "MHSBUSDT", "MHSCUSDT"]
    ev._get_symbol_minute_frame.cache_clear()
    ev._preload_symbol_minute_frames(str(root), syms, "1m")
    info = ev._get_symbol_minute_frame.cache_info()
    assert info.currsize >= len(syms)
    for s in syms:
        assert ev._get_symbol_minute_frame(str(root), s, "1m") is not None


def test_p10_book_error_isolation(mhs_market, monkeypatch) -> None:
    # SCENARIO_P10_ISOLATION: a book whose outcome is a typed failure (primary
    # dropped, failure set) is delivered through the process pool without
    # blocking the other two books.
    args = _build_books_concurrent_args(mhs_market)
    real = ev._book_outcome

    def _failing(name, *a, **k):
        report = real(name, *a, **k)
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
                    reason=ev.MHS_GO_REASON_EXECUTION_GAP,
                    message="forced isolation failure",
                ),
            )
        return report

    monkeypatch.setattr(ev, "_book_outcome", _failing)
    fast, slow, blend = ev._run_books_concurrent(**args)
    assert fast.primary is not None
    assert fast.failure is None
    assert slow.primary is None
    assert slow.failure is not None
    assert slow.failure.reason == ev.MHS_GO_REASON_EXECUTION_GAP
    assert blend.primary is not None
    assert blend.failure is None

def test_active_blend_grid_slow_only() -> None:
    # SCENARIO_MHS_ACTIVE_BLEND_GRID_SLOW_ONLY_01: with the frozen
    # PHASE_1_BOOK_BLEND_WEIGHTS == {fast_reversal: 0.0, slow_momentum: 1.0},
    # the blend adopts slow's own BookSpec and 24h-native grid by identity (not
    # equality) -- never fast's 6h grid.
    fast = ev.PHASE_1_BOOK_SPECS["fast_reversal"]
    slow = ev.PHASE_1_BOOK_SPECS["slow_momentum"]
    fast_grid = pd.date_range(_START, periods=4, freq="6h", tz="UTC")
    slow_grid = pd.date_range(_START, periods=1, freq="24h", tz="UTC")
    spec, grid = ev._active_blend_book_and_grid(fast, slow, fast_grid, slow_grid)
    assert spec is slow
    assert grid is slow_grid


def test_active_blend_grid_fast_weighted(monkeypatch) -> None:
    # SCENARIO_MHS_ACTIVE_BLEND_GRID_FAST_WEIGHTED_02: with a nonzero fast
    # weight (historical 50/50), the helper returns fast/fast_grid by identity,
    # reproducing the pre-fix behavior byte-for-byte when fast is re-admitted.
    monkeypatch.setattr(
        ev, "PHASE_1_BOOK_BLEND_WEIGHTS",
        {"fast_reversal": 0.5, "slow_momentum": 0.5},
    )
    fast = ev.PHASE_1_BOOK_SPECS["fast_reversal"]
    slow = ev.PHASE_1_BOOK_SPECS["slow_momentum"]
    fast_grid = pd.date_range(_START, periods=4, freq="6h", tz="UTC")
    slow_grid = pd.date_range(_START, periods=1, freq="24h", tz="UTC")
    spec, grid = ev._active_blend_book_and_grid(fast, slow, fast_grid, slow_grid)
    assert spec is fast
    assert grid is fast_grid


def test_active_blend_grid_no_weight_fails_closed(monkeypatch) -> None:
    # SCENARIO_MHS_ACTIVE_BLEND_GRID_NO_WEIGHT_03: with zero weight on both
    # books the allocation invariant is violated and the helper must fail
    # closed (ValueError) rather than silently pick a default grid.
    monkeypatch.setattr(
        ev, "PHASE_1_BOOK_BLEND_WEIGHTS",
        {"fast_reversal": 0.0, "slow_momentum": 0.0},
    )
    fast = ev.PHASE_1_BOOK_SPECS["fast_reversal"]
    slow = ev.PHASE_1_BOOK_SPECS["slow_momentum"]
    fast_grid = pd.date_range(_START, periods=4, freq="6h", tz="UTC")
    slow_grid = pd.date_range(_START, periods=1, freq="24h", tz="UTC")
    with pytest.raises(ValueError, match="allocates no capital"):
        ev._active_blend_book_and_grid(fast, slow, fast_grid, slow_grid)


def test_blend_report_adopts_slow_cadence(mhs_market) -> None:
    # SCENARIO_MHS_BLEND_REPORT_ADOPTS_SLOW_CADENCE_04: under the fixture with
    # the current frozen weights, the blend MhsBookReport produced by
    # _run_books_concurrent has step_hours==24 and horizon_hours==168
    # (slow_momentum's values), not step_hours==6/horizon_hours==48
    # (fast_reversal's) -- proving the _run_books_concurrent call site was
    # rewired, not just the helper added in isolation.
    args = _build_books_concurrent_args(mhs_market)
    _, _, blend_report = ev._run_books_concurrent(**args)
    assert blend_report.failure is None
    assert blend_report.step_hours == 24
    assert blend_report.horizon_hours == 168


def test_fold_decision_grid_matches_slow_cadence(mhs_market) -> None:
    # SCENARIO_MHS_FOLD_DECISION_GRID_MATCHES_SLOW_CADENCE_05: under the
    # fixture with the current frozen weights, _build_fold_target_weights's
    # target_weights index has a row spacing consistent with the 24h slow_grid
    # (not the 1h native grid_1h) for the validation window -- the fold-level
    # Research-GO gate no longer decides at native-hourly cadence when only
    # slow_momentum is admitted.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
    )
    target_weights, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )
    assert not target_weights.empty
    spacing = target_weights.index.to_series().diff().dropna()
    assert not spacing.empty
    assert (spacing == pd.Timedelta(hours=24)).all()


def test_crash_tilt_disabled_fold_is_byte_identical(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_CRASH_TILT_FOLD_BYTE_IDENTICAL_06: with the opt-in disabled
    # (crash_regime_tilt_alpha=None) the fold target weights are byte-identical
    # to the pre-overlay path. Proved by running the enabled path with the tilt
    # replaced by an identity: the new wiring then reproduces exactly the
    # disabled output, so the extra branch is value-neutral when no tilt applies.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
    )
    target_off, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )

    tilt_calls: list[tuple[int, float]] = []

    def _identity_tilt(rank_neutral_weights, _log_price, _eligible, _refs, horizon, alpha, min_symbols=8):
        tilt_calls.append((horizon, alpha))
        return rank_neutral_weights

    monkeypatch.setattr(ev, "crash_regime_tilt_weights", _identity_tilt)
    request_on = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        crash_regime_tilt_alpha=0.3,
    )
    target_ident, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request_on, funding_by_symbol,
    )
    pd.testing.assert_frame_equal(target_off, target_ident)
    assert tilt_calls, "enabled path must route through crash_regime_tilt_weights"
    assert tilt_calls[0][0] == 168, "tilt lookback must reuse slow.horizon_hours (168), not a new literal"
    assert tilt_calls[0][1] == 0.3


def test_crash_tilt_active_fold_reaches_replay(mhs_market_with_btc) -> None:
    # SCENARIO_MHS_CRASH_TILT_FOLD_ACTIVE_07: with a real BTCUSDT reference
    # series in the panel and crash_regime_tilt_alpha=0.3, the fold target
    # weights (a) differ from the disabled baseline, (b) stay finite, and
    # (c) keep the blended gross budget bounded by unit (the tilt offsets
    # dollar-neutral shorts rather than amplifying them).
    root, end = mhs_market_with_btc
    symbols = [
        s for s in ("BTCUSDT", "MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT",
                    "MHSEUSDT", "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT",
                    "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol = ev._load_funding_series(symbols)
    assert "BTCUSDT" in funding_by_symbol
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
    )
    target_off, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request, funding_by_symbol,
    )
    request_on = dataclasses.replace(request, crash_regime_tilt_alpha=0.3)
    target_on, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request_on, funding_by_symbol,
    )
    assert not target_off.equals(target_on)
    assert np.isfinite(target_on.to_numpy(dtype="float64")).all()
    assert float(target_on.abs().max().max()) <= 1.0 + 1e-9


def test_p14_postbook_concurrent_parity() -> None:
    # SCENARIO_P14_POSTBOOK: the deployment tail computed with the placeholder
    # ``research_go_eligible=None`` and then patched with the fold-derived flag
    # is identical to computing it directly with that flag, so the concurrent
    # post-book path cannot change the readiness result.
    idx = pd.date_range("2021-01-01", periods=3000, freq="1h", tz="UTC")
    rng = np.random.default_rng(42)
    equity = pd.Series(np.cumprod(1.0 + rng.normal(0.0002, 0.004, len(idx))), index=idx)
    full = ev.compute_deployment_readiness(
        equity, 365 * 24, research_go_eligible=False, n_bootstrap=200, seed=7,
    )
    placeholder = ev.compute_deployment_readiness(
        equity, 365 * 24, research_go_eligible=None, primary_valid=True,
        n_bootstrap=200, seed=7,
    )
    patched = dataclasses.replace(placeholder, research_go_eligible=False)
    assert patched == full


def test_p14_postbook_no_deadlock(monkeypatch) -> None:
    # SCENARIO_P14_NO_DEADLOCK: with no anchored folds the concurrent
    # orchestration degrades to the sequential diagnostics tail through the
    # same entry point, proving the fold-pool/thread orchestration never
    # deadlocks or hangs.
    class _FakePrimary:
        ledger = None

    class _FakeBlend:
        primary = _FakePrimary()

    monkeypatch.setattr(ev, "phase_1_anchored_purged_folds", lambda: ())
    calls = {"n": 0}

    def _fast_diag(*_args, **_kwargs):
        calls["n"] += 1
        return (None, None, {}, {}, None)

    monkeypatch.setattr(ev, "_run_post_diag_deploy", _fast_diag)
    result = ev._run_post_book_concurrently(
        _FakeBlend(), "root", None, [], None, None, None, None, None, None, None, {}, 1.0, None,
    )
    assert calls["n"] == 1
    assert result[4] == ()
    assert result[5] is None


def _build_compact_report() -> ev.MhsHorizonDiagnosticReport:
    """Minimal one-book report with a real small replay for tier tests."""
    idx = pd.date_range("2021-01-01 12:01", periods=4000, freq="1min", tz="UTC")
    px = pd.DataFrame({"A": [100.0] * len(idx)}, index=idx)
    target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 11:00", tz="UTC")])
    signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
    replay = strategy_aware_execution_replay(
        target, signal_at, px, px, px, px,
        pd.DataFrame(0.0, index=idx, columns=["A"]), 1.0,
        "OHLCV_STRICT_PROXY", ExecutionSpec(),
    )
    book = ev.MhsBookReport(
        name="fast_reversal", band="FAST", horizon_hours=24, step_hours=6,
        tranche_count=1, n_symbols=1,
        phase=ev.PhaseDiagnosticResult(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False),
        prescreen={}, tail=ev.TailSensitivityResult(
            0.0, 0.0, {}, 1, 0, 0.0, 0.0, 0.0, 0.0,
        ),
        primary=replay, stress=None,
        primary_autocorr_sharpe=0.1, primary_naive_sharpe=0.1, primary_net_ann=0.01,
        primary_geometric_cagr=0.01, primary_max_drawdown=-0.01,
        primary_annualized_turnover=1.0, stress_naive_sharpe=None,
    )
    return ev.MhsHorizonDiagnosticReport(
        feature="multi_horizon_market_state", status="COMPLETE", start="2021-01-01",
        end="2021-01-04", resolved_end="2021-01-04", partition="dev",
        execution_tiers_bps=(2.5, 5.0), books={"fast_reversal": book}, blend=None,
        blend_target_gross=0.0, blend_cash_fraction=0.0, eligible_symbols=1,
        trials_attempted=1, deflated_sharpe_ratio=None, xs_rank_ic={},
        date_clustered_regression={}, horizon_diagnostics={}, bootstrap_ci=None,
        placebo_sharpe_percentile=None,
        deployment_readiness=ev.DeploymentReadinessResult(
            0.01, -0.01, 1.0, -0.01, -0.01, -0.01, -0.01, 0, None, 0.5, 0.0, 0.0, {}, {},
            {}, False, False, False, False,
        ),
        synthetic_stress={}, participation_warnings={}, termination_counts={},
        unsupported_assumptions=(), anchored_folds=(), folds=(),
        research_go=ev.MhsResearchGoResult(False, (), 0, 0),
        fill_source="OHLCV_STRICT_PROXY", mark_source="MARK_PRICE",
        execution_timeframe="1m", execution_universe_size=1,
        execution_symbols=("A",), run_elapsed_seconds=0.1,
    )


def test_mhs_output_tier_enum_values() -> None:
    assert ev.MhsOutputTier.COMPACT.value == "compact"
    assert ev.MhsOutputTier.FULL.value == "full"
    assert ev.MhsOutputTier("compact") is ev.MhsOutputTier.COMPACT
    assert ev.MhsOutputTier("full") is ev.MhsOutputTier.FULL


def test_regression_existing_report_fields_unchanged() -> None:
    # SCENARIO_REGRESSION_EXISTING_REPORT_FIELDS_UNCHANGED: the two-pass
    # change must not rename or drop any existing MhsBookReport/MhsFoldReport
    # field (which doubles as the JSON key via to_payload()) -- only
    # pre_vol_target_reference/pre_vol_target_reference_naive_sharpe are new,
    # following the exact patient_reference field-addition precedent.
    book_fields = {f.name for f in dataclasses.fields(ev.MhsBookReport)}
    for field_name in (
        "name", "band", "horizon_hours", "step_hours", "tranche_count",
        "n_symbols", "phase", "prescreen", "tail", "primary", "stress",
        "primary_autocorr_sharpe", "primary_naive_sharpe", "primary_net_ann",
        "primary_geometric_cagr", "primary_max_drawdown",
        "primary_annualized_turnover", "stress_naive_sharpe",
        "terminal_censored_decisions", "failure", "touch", "touch_naive_sharpe",
        "ladder", "ladder_naive_sharpe", "patient_reference",
        "patient_reference_naive_sharpe",
    ):
        assert field_name in book_fields
    assert "pre_vol_target_reference" in book_fields
    assert "pre_vol_target_reference_naive_sharpe" in book_fields

    fold_fields = {f.name for f in dataclasses.fields(ev.MhsFoldReport)}
    for field_name in (
        "fold_index", "validation_start", "validation_end", "strict", "stress",
        "primary_valid", "primary_autocorr_sharpe", "primary_naive_sharpe",
        "primary_net_ann", "primary_geometric_cagr", "primary_max_drawdown",
        "stress_naive_sharpe", "decision_intents", "termination_counts",
        "failures", "strict_elapsed_seconds", "stress_elapsed_seconds",
        "terminal_censored_decisions",
    ):
        assert field_name in fold_fields

    report = _build_compact_report()
    book = report.books["fast_reversal"]
    for key in ("primary", "patient_reference", "patient_reference_naive_sharpe"):
        assert key in dataclasses.asdict(book)
    assert "pre_vol_target_reference" in dataclasses.asdict(book)
    payload = report.to_payload()
    assert "pre_vol_target_reference" in payload["books"]["fast_reversal"]


def test_daily_resample_ledger_fidelity() -> None:
    # COMPACT_DAILY_LEDGER_FIDELITY: the daily rollup preserves the source
    # ledger's per-day first/max/min/last equity and the cross-day return.
    idx = pd.date_range("2021-01-01", periods=48 * 3, freq="30min", tz="UTC")
    equity = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "equity": equity.to_numpy(),
            "fill_turnover": 0.0,
        }
    )
    frame.loc[2, "fill_turnover"] = 0.5
    frame.loc[5, "fill_turnover"] = 0.25
    daily = ev._daily_resample_ledger(frame)
    assert len(daily) == 3
    assert list(daily.columns) == [
        "date", "equity_open", "equity_high", "equity_low", "equity_close",
        "daily_turnover", "daily_fill_count", "daily_return",
    ]
    d0 = daily.iloc[0]
    day0 = idx.normalize()[0]
    day0_mask = idx < day0 + pd.Timedelta("1D")
    day0_eq = frame.loc[day0_mask, "equity"]
    assert d0["equity_open"] == pytest.approx(day0_eq.iloc[0], rel=1e-6)
    assert d0["equity_high"] == pytest.approx(day0_eq.max(), rel=1e-6)
    assert d0["equity_low"] == pytest.approx(day0_eq.min(), rel=1e-6)
    assert d0["equity_close"] == pytest.approx(day0_eq.iloc[-1], rel=1e-6)
    assert d0["daily_turnover"] == pytest.approx(0.75, rel=1e-6)
    assert d0["daily_fill_count"] == 2
    assert np.isnan(d0["daily_return"])
    d1 = daily.iloc[1]
    day1_mask = (idx >= day0 + pd.Timedelta("1D")) & (idx < day0 + pd.Timedelta("2D"))
    day1_eq = frame.loc[day1_mask, "equity"]
    assert d1["equity_open"] == pytest.approx(day1_eq.iloc[0], rel=1e-6)
    assert d1["daily_return"] == pytest.approx(day1_eq.iloc[-1] / d0["equity_close"] - 1.0, rel=1e-6)


def test_daily_resample_ledger_fails_closed_on_bad_equity() -> None:
    idx = pd.date_range("2021-01-01", periods=48, freq="30min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "equity": [100.0] * 47 + [np.nan],
            "fill_turnover": 0.0,
        }
    )
    with pytest.raises(DataIntegrityError, match="equity"):
        ev._daily_resample_ledger(frame)


def test_compact_json_stripped_and_wired(tmp_path) -> None:
    # COMPACT_JSON_STRIPPED: compact persist drops per-replay SHA-256/schema
    # references while retaining only row counts and the scalar report fields.
    report = _build_compact_report()
    out = tmp_path / "mhs_report.json"
    persisted = ev.persist_mhs_horizon_diagnostic_report(
        report, out, tier=ev.MhsOutputTier.COMPACT,
    )
    assert persisted == out
    payload = json.loads(out.read_text())
    raw = json.dumps(payload)
    assert "checksum_sha256" not in raw
    assert "schema_version" not in raw
    assert "time_bounds" not in raw
    ref = payload["books"]["fast_reversal"]["primary"]
    assert set(ref) == {"fills", "units", "notional_weights", "ledger", "times"}
    assert all(set(v) == {"row_count"} for v in ref.values())
    assert ref["ledger"]["row_count"] == len(report.books["fast_reversal"].primary.ledger.equity)
    assert ref["fills"]["row_count"] == len(report.books["fast_reversal"].primary.simulated_fills)
    assert payload["status"] == "COMPLETE"
    assert "daily_ledger" in payload["artifacts"]
    assert set(payload["artifacts"]["fills"]) == {"file", "row_count"}
    assert "fast_reversal_primary" in payload["replay_ids"]


def test_compact_size_budget(tmp_path) -> None:
    # COMPACT_SIZE_BUDGET: compact artifacts stay far below the git-friendly
    # budgets (daily ledger < 500KB, JSON < 20KB) for a small replay workload.
    report = _build_compact_report()
    out = tmp_path / "mhs_report.json"
    ev.persist_mhs_horizon_diagnostic_report(report, out, tier=ev.MhsOutputTier.COMPACT)
    artifact_dir = out.parent / "mhs_report_artifacts"
    daily_path = artifact_dir / "daily_ledger.parquet"
    assert daily_path.exists()
    assert daily_path.stat().st_size < 500 * 1024
    assert out.stat().st_size < 20 * 1024
    daily = pd.read_parquet(daily_path)
    assert "replay_id" in daily.columns
    assert daily["replay_id"].eq("fast_reversal_primary").all()
    assert len(daily) == 4
    assert daily["equity_close"].gt(0).all()


def test_compact_failure_escalates_past_artifacts(tmp_path, monkeypatch) -> None:
    # A non-DataIntegrityError resample failure logs and returns None without
    # writing compact artifacts (fail-closed escalation).
    report = _build_compact_report()

    def _boom(_table):
        raise RuntimeError("boom")

    monkeypatch.setattr(ev, "_daily_resample_ledger", _boom)
    out = tmp_path / "mhs_report.json"
    persisted = ev.persist_mhs_horizon_diagnostic_report(
        report, out, tier=ev.MhsOutputTier.COMPACT,
    )
    assert persisted is None
    assert not out.exists()


def test_gitignore_full_subdir_only() -> None:
    # GITIGNORE_FULL_SUBDIR: only the _full/ audit subdirectory is gitignored;
    # the compact daily ledger path and summary JSON stay trackable.
    gitignore = Path(".gitignore").read_text(encoding="utf-8").splitlines()
    assert "docs/results/mhs_horizon_diagnostic_artifacts/_full/" in gitignore
    assert "docs/results/mhs_horizon_diagnostic_artifacts/" not in gitignore
    assert "docs/results/mhs_horizon_diagnostic.json" not in gitignore
    assert "docs/results/mhs_horizon_diagnostic_artifacts/daily_ledger.parquet" not in gitignore


def test_book_weights_momentum_keeps_raw_signal() -> None:
    """SCENARIO_BOOK_WEIGHTS_MOMENTUM_STAYS_RAW: ``_book_weights`` for a
    sign=+1 spec stays on raw ``horizon_log_return``. ``vol_normalized_
    horizon_signal`` was tried in the live book and reverted -- it improved
    the discovery-gate prescreen but broke ``CAPITAL_INVARIANT_BREACH`` on
    the full 2021-2025 realistic-execution replay, so it stays wired into
    discovery.py's diagnostic-only scoring only
    (docs/specs/mhs_momentum_vol_normalization.md follow-up)."""
    log_close, eligible, _, _, idx = _signal_disagreement_panel()
    spec = _dispatch_spec(sign=1)
    weights = ev._book_weights(log_close, eligible, spec, idx)
    expected = _reference_weights(log_close, eligible, idx, spec)
    pd.testing.assert_frame_equal(weights, expected)
    vol_normalized = ev.phase_tranche_book(
        ev.rank_weight_book(
            vol_normalized_horizon_signal(log_close, spec.horizon_hours).reindex(idx),
            eligible.reindex(idx),
            spec.band.sign,
            spec.min_symbols,
        ),
        spec.tranche_count(),
    )
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(weights, vol_normalized)


def test_book_weights_reversal_keeps_raw_signal() -> None:
    """SCENARIO_BOOK_WEIGHTS_REVERSAL_UNCHANGED: ``_book_weights`` for a
    sign=-1 spec stays on raw ``horizon_log_return``."""
    log_close, eligible, _, _, idx = _signal_disagreement_panel()
    spec = _dispatch_spec(sign=-1)
    weights = ev._book_weights(log_close, eligible, spec, idx)
    expected = _reference_weights(log_close, eligible, idx, spec)
    pd.testing.assert_frame_equal(weights, expected)


def test_phase_diagnostics_momentum_keeps_raw_signal(monkeypatch) -> None:
    """SCENARIO_PHASE_DIAGNOSTICS_MOMENTUM_CONSISTENT_WITH_LIVE_SIGNAL:
    ``_phase_diagnostics`` for a sign=+1 spec ranks raw ``horizon_log_return``,
    consistent with ``_book_weights`` after the vol-normalized-signal revert."""
    log_close, eligible, opens, bar_funding, idx = _signal_disagreement_panel()
    spec = _dispatch_spec(sign=1)
    captured: list[pd.DataFrame] = []
    real_rank = ev.rank_weight_book

    def recording(signal, elig, sign, min_symbols):
        captured.append(signal)
        return real_rank(signal, elig, sign, min_symbols)

    monkeypatch.setattr(ev, "rank_weight_book", recording)
    ev._phase_diagnostics(log_close, eligible, opens, bar_funding, idx, spec)
    assert captured
    phase_grid = idx[0 :: spec.step_hours]
    expected = ev.horizon_log_return(log_close, spec.horizon_hours).reindex(phase_grid)
    vol_normalized = vol_normalized_horizon_signal(log_close, spec.horizon_hours).reindex(phase_grid)
    pd.testing.assert_frame_equal(captured[0], expected)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(captured[0], vol_normalized)


def test_phase_diagnostics_reversal_keeps_raw_signal(monkeypatch) -> None:
    """SCENARIO_PHASE_DIAGNOSTICS_REVERSAL_UNCHANGED: ``_phase_diagnostics``
    for a sign=-1 spec still ranks raw ``horizon_log_return``."""
    log_close, eligible, opens, bar_funding, idx = _signal_disagreement_panel()
    spec = _dispatch_spec(sign=-1)
    captured: list[pd.DataFrame] = []
    real_rank = ev.rank_weight_book

    def recording(signal, elig, sign, min_symbols):
        captured.append(signal)
        return real_rank(signal, elig, sign, min_symbols)

    monkeypatch.setattr(ev, "rank_weight_book", recording)
    ev._phase_diagnostics(log_close, eligible, opens, bar_funding, idx, spec)
    assert captured
    phase_grid = idx[0 :: spec.step_hours]
    expected = ev.horizon_log_return(log_close, spec.horizon_hours).reindex(phase_grid)
    pd.testing.assert_frame_equal(captured[0], expected)

def _admitted_selection(selected_horizon: int | None = 360) -> ev.DiscoveryQualificationResult:
    return ev.DiscoveryQualificationResult(
        selected_horizon=selected_horizon,
        admitted=selected_horizon is not None,
        discovery_scores=() if selected_horizon is None else ((selected_horizon, 2.5),),
        discovery_aggregate_net_t=2.5 if selected_horizon is not None else None,
        qualification_net_t=2.3 if selected_horizon is not None else None,
        qualification_sign_consistent=True if selected_horizon is not None else None,
    )


def test_fold_safe_slow_book_spec_admitted_vs_fallback() -> None:
    # SCENARIO_MHS_FOLD_SAFE_HORIZON_05_BOOK_SPEC_HELPER_ADMITTED_VS_FALLBACK:
    # the fold-spec resolver returns the unchanged frozen default (168,
    # "frozen_default") unless the fold-scoped gate both admitted AND selected
    # a candidate; only then does it build a BookSpec whose horizon is the
    # selected candidate with band/step_hours/min_symbols identical to the
    # default.
    default = ev.PHASE_1_BOOK_SPECS["slow_momentum"]
    fallback = ev.DiscoveryQualificationResult(
        selected_horizon=None, admitted=False, discovery_scores=(),
        discovery_aggregate_net_t=None, qualification_net_t=None,
        qualification_sign_consistent=None,
    )
    spec, horizon, source = ev._fold_safe_slow_book_spec(fallback, default)
    assert spec is default
    assert horizon == 168
    assert source == "frozen_default"

    admitted_none = _admitted_selection(selected_horizon=None)
    spec, horizon, source = ev._fold_safe_slow_book_spec(admitted_none, default)
    assert spec is default
    assert source == "frozen_default"

    admitted = _admitted_selection(360)
    spec, horizon, source = ev._fold_safe_slow_book_spec(admitted, default)
    assert source == "fold_train_only_discovery"
    assert horizon == 360
    assert spec is not default
    assert spec.horizon_hours == 360
    assert spec.band is default.band
    assert spec.step_hours == default.step_hours
    assert spec.min_symbols == default.min_symbols


def test_fold_safe_horizon_flag_off_is_byte_identical(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_FOLD_SAFE_HORIZON_06_FLAG_OFF_IS_BYTE_IDENTICAL: with
    # fold_safe_horizon_selection=False (the default) neither the fold worker
    # nor the parent diagnostic touches fold_train_only_discovery_qualification
    # (call-count 0) and the fold report records the frozen 168h default -- a
    # no-op regression guard matching the project's flag-gated ADR pattern.
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ][:8]
    funding_by_symbol = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    assert request.fold_safe_horizon_selection is False

    calls = {"n": 0}
    real_fn = ev.fold_train_only_discovery_qualification

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(ev, "fold_train_only_discovery_qualification", counting)
    report = ev._run_anchored_fold(str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None)
    assert calls["n"] == 0
    assert report.slow_horizon_hours == 168
    assert report.slow_horizon_source == "frozen_default"

    # Parent path: the default request passes an empty fold_slow_horizons dict
    # into the fold pool and never runs the fold-scoped selection.
    captured: dict = {}

    def _spy_books(*args, **kwargs):
        return (None, None, None)

    def _spy_post(*args, **kwargs):
        captured["fold_slow_horizons"] = args[14] if len(args) > 14 else None
        return (None, None, {}, {}, (), None)

    monkeypatch.setattr(ev, "_run_books_concurrent", _spy_books)
    monkeypatch.setattr(ev, "_run_post_book_concurrently", _spy_post)
    top_report = ev.run_mhs_horizon_diagnostic(request)
    assert top_report.status == "COMPLETE"
    assert calls["n"] == 0
    assert captured["fold_slow_horizons"] == {}


def test_fold_safe_horizon_records_source(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_FOLD_SAFE_HORIZON_07_FOLD_REPORT_RECORDS_SOURCE: MhsFoldReport
    # constructed without the new fields defaults to (168, "frozen_default"),
    # _incomplete_fold_report keeps that default, and a fold run resolved with a
    # 360h fold-scoped override records slow_horizon_hours==360 with source
    # "fold_train_only_discovery".
    default_report = ev.MhsFoldReport(
        fold_index=0, validation_start="2021-02-10", validation_end="2021-04-19",
        strict=None, stress=None, primary_valid=False, primary_autocorr_sharpe=0.0,
        primary_naive_sharpe=0.0, primary_net_ann=0.0, primary_geometric_cagr=0.0,
        primary_max_drawdown=0.0, stress_naive_sharpe=0.0, decision_intents=0,
        termination_counts={}, failures=(), strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
    )
    assert default_report.slow_horizon_hours == 168
    assert default_report.slow_horizon_source == "frozen_default"

    incomplete = ev._incomplete_fold_report(_FOLD, 0, ())
    assert incomplete.slow_horizon_source == "frozen_default"
    assert incomplete.slow_horizon_hours == 168

    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ][:8]
    funding_by_symbol = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    report = ev._run_anchored_fold(
        str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None, slow_horizon_override=360,
    )
    assert report.slow_horizon_hours == 360
    assert report.slow_horizon_source == "fold_train_only_discovery"

    # Parent wiring: with the flag on, the parent runs the fold-scoped selection
    # once per fold and threads only the resolved plain int down to the fold
    # pool; the top-level slow spec adopts fold 2's selection (360h here).
    captured: dict = {}
    monkeypatch.setattr(
        ev, "fold_train_only_discovery_qualification",
        lambda *args, **kwargs: _admitted_selection(360),
    )

    def _spy_books(*args, **kwargs):
        captured["top_level_slow"] = args[5]
        return (None, None, None)

    def _spy_post(*args, **kwargs):
        captured["fold_slow_horizons"] = args[14] if len(args) > 14 else None
        return (None, None, {}, {}, (), None)

    monkeypatch.setattr(ev, "_run_books_concurrent", _spy_books)
    monkeypatch.setattr(ev, "_run_post_book_concurrently", _spy_post)
    request_on = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, fold_safe_horizon_selection=True,
    )
    top_report = ev.run_mhs_horizon_diagnostic(request_on)
    assert top_report.status == "COMPLETE"
    assert captured["fold_slow_horizons"] == {0: 360, 1: 360, 2: 360}
    assert captured["top_level_slow"].horizon_hours == 360
    assert captured["top_level_slow"].band is ev.PHASE_1_BOOK_SPECS["slow_momentum"].band


def test_fold_safe_horizon_builds_candidate_weights_once_and_shares_across_folds(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_HORIZON_SEARCH_EFF_05_TOP_LEVEL_WIRING_SHARES_ONE_CACHE_ACROSS_FOLDS:
    # with fold_safe_horizon_selection=True the parent builds the discovery
    # weight book exactly once (call-count 1 wrapping the real implementation,
    # regardless of the 3 anchored folds) and passes the SAME cache object into
    # every fold_train_only_discovery_qualification(...) call -- the measured
    # 3x-redundant weight construction is eliminated without changing any value.
    root, end = mhs_market
    calls = {"n": 0}
    real_build = ev.build_candidate_weights

    def counting_build(*args, **kwargs):
        calls["n"] += 1
        return real_build(*args, **kwargs)

    monkeypatch.setattr(ev, "build_candidate_weights", counting_build)
    cache_objs: list = []

    def _spy_fold(*args, **kwargs):
        cache_objs.append(kwargs.get("precomputed_candidate_weights"))
        return _admitted_selection(360)

    monkeypatch.setattr(ev, "fold_train_only_discovery_qualification", _spy_fold)

    def _spy_books(*args, **kwargs):
        return (None, None, None)

    def _spy_post(*args, **kwargs):
        return (None, None, {}, {}, (), None)

    monkeypatch.setattr(ev, "_run_books_concurrent", _spy_books)
    monkeypatch.setattr(ev, "_run_post_book_concurrently", _spy_post)
    request_on = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, fold_safe_horizon_selection=True,
    )
    top_report = ev.run_mhs_horizon_diagnostic(request_on)
    assert top_report.status == "COMPLETE"
    assert calls["n"] == 1
    assert len(cache_objs) == 3
    assert all(c is cache_objs[0] for c in cache_objs)


def _slow_book_panel_inputs(mhs_market):
    """Panel inputs for the ``_slow_book_execution_weights`` and fold tests."""
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol = ev._load_funding_series(symbols)
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    panel = ev.load_base_panel(
        str(root), "1h", ("close", "open", "quote_vol"), _START, end,
        partition="dev", min_bars=2000,
    )
    close, _opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    grid_1h = close.index
    funded = [s for s in close.columns if s in funding_by_symbol]
    close = close[funded]
    quote_vol = quote_vol[funded]
    eligible = ev.liquid_half_eligibility(quote_vol, lookback_bars=720, min_history_bars=720)
    log_close = np.log(close)
    execution_mask = ev._pit_execution_mask(quote_vol, eligible, request.execution_universe_size)
    return log_close, eligible, execution_mask, request, grid_1h, end


def _pre_change_slow_book(
    log_close, eligible, execution_mask, spec, step_grid, ema_span,
):
    """The frozen production slow-book chain ``_slow_book_execution_weights``
    must reproduce byte-identically in single_horizon/raw mode."""
    w = ev._book_weights(log_close, eligible, spec, step_grid, ema_span=ema_span)
    w = ev.inverse_realized_vol_tilt(
        w, ev.realized_vol(log_close, spec.horizon_hours).reindex(step_grid),
    )
    return ev.renormalize_within_mask(
        w, execution_mask.reindex(step_grid).fillna(False), spec.min_symbols,
    )


def test_mhs_alpha_engine_slow_book_single_horizon_is_byte_identical(mhs_market) -> None:
    # SCENARIO_MHS_ALPHA_ENGINE_07: in single_horizon/raw mode
    # ``_slow_book_execution_weights`` reproduces the pre-change
    # ``_book_weights`` + tilt + renormalize sequence exactly.
    log_close, eligible, execution_mask, _request, _grid, end = _slow_book_panel_inputs(mhs_market)
    slow = ev.PHASE_1_BOOK_SPECS["slow_momentum"]
    slow_grid = pd.date_range(_START, end, freq="24h", tz="UTC")
    slow_ema = max(1, round(slow.horizon_hours / slow.step_hours * ev.MHS_SIGNAL_EMA_HORIZON_SPAN))
    expected = _pre_change_slow_book(
        log_close, eligible, execution_mask, slow, slow_grid, slow_ema,
    )
    actual = ev._slow_book_execution_weights(
        log_close, eligible, execution_mask, slow, slow_grid,
        "single_horizon", "raw", slow_ema,
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_mhs_alpha_engine_slow_book_ensemble_is_rowwise_mean_with_consensus_gross(mhs_market) -> None:
    # SCENARIO_MHS_ALPHA_ENGINE_07: in horizon_ensemble mode the output is the
    # row-wise mean of the per-horizon books on the same step grid, dollar-
    # neutral, with strictly smaller mean gross than any single horizon on a
    # panel where the horizons disagree (consensus-scaled exposure).
    log_close, eligible, execution_mask, _request, _grid, end = _slow_book_panel_inputs(mhs_market)
    slow = ev.PHASE_1_BOOK_SPECS["slow_momentum"]
    slow_grid = pd.date_range(_START, end, freq="24h", tz="UTC")
    slow_ema = max(1, round(slow.horizon_hours / slow.step_hours * ev.MHS_SIGNAL_EMA_HORIZON_SPAN))
    per_horizon: dict[int, pd.DataFrame] = {}
    for h in slow.band.horizons_hours:
        spec = dataclasses.replace(slow, horizon_hours=h)
        per_horizon[h] = _pre_change_slow_book(
            log_close, eligible, execution_mask, spec, slow_grid, slow_ema,
        )
    expected = sum(per_horizon.values()) / len(per_horizon)
    actual = ev._slow_book_execution_weights(
        log_close, eligible, execution_mask, slow, slow_grid,
        "horizon_ensemble", "raw", slow_ema,
    )
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.sum(axis=1).abs().max() < 1e-9
    per_horizon_gross = [float(w.abs().sum(axis=1).mean()) for w in per_horizon.values()]
    ensemble_gross = float(actual.abs().sum(axis=1).mean())
    assert ensemble_gross <= 1.0 + 1e-9
    assert ensemble_gross < max(per_horizon_gross) - 1e-6


def test_mhs_alpha_engine_slow_book_validates_mode_and_signal_kind(mhs_market) -> None:
    log_close, eligible, execution_mask, _request, _grid, end = _slow_book_panel_inputs(mhs_market)
    slow = ev.PHASE_1_BOOK_SPECS["slow_momentum"]
    slow_grid = pd.date_range(_START, end, freq="24h", tz="UTC")
    with pytest.raises(ValueError, match="mode"):
        ev._slow_book_execution_weights(
            log_close, eligible, execution_mask, slow, slow_grid,
            "bogus", "raw", None,
        )
    with pytest.raises(ValueError, match="signal_kind"):
        ev._slow_book_execution_weights(
            log_close, eligible, execution_mask, slow, slow_grid,
            "single_horizon", "bogus", None,
        )


def test_mhs_alpha_engine_fold_portfolio_trigger_preserves_invariants(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_ALPHA_ENGINE_08: with rebalance_filter='portfolio_trigger'
    # the fold target weights keep exact dollar neutrality and the realized
    # gross tracks regime_cash_scale (the trigger gates the UNSCALED book and
    # the gross scale multiplies afterwards), whereas the per-symbol deadband
    # branch leaks net exposure and decouples gross from the scale (RC-1).
    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]
    funding_by_symbol = ev._load_funding_series(symbols)
    forced_scale: dict[str, pd.Series] = {}

    def _forced_step_scale(vol_mean: pd.Series) -> pd.Series:
        out = pd.Series(
            np.where(np.arange(len(vol_mean)) < len(vol_mean) // 2, 0.5, 1.0),
            index=vol_mean.index,
        )
        forced_scale["series"] = out
        return out

    monkeypatch.setattr(ev, "_regime_cash_scale", _forced_step_scale)
    request_trig = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        rebalance_filter="portfolio_trigger",
    )
    target_trig, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request_trig, funding_by_symbol,
    )
    scale = forced_scale["series"].reindex(target_trig.index)
    assert target_trig.sum(axis=1).abs().max() < 1e-9
    assert (target_trig.abs().sum(axis=1) - scale).abs().max() < 1e-9
    assert target_trig.abs().sum(axis=1).max() <= 1.0 + 1e-9

    request_dead = dataclasses.replace(request_trig, rebalance_filter="per_symbol_deadband")
    target_dead, _signal, _roster, _grid = ev._build_fold_target_weights(
        str(root), _FOLD, request_dead, funding_by_symbol,
    )
    assert target_dead.sum(axis=1).abs().max() > 1e-3
    assert (target_dead.abs().sum(axis=1) - scale).abs().max() > 1e-3


def test_mhs_alpha_engine_request_field_validation() -> None:
    # SCENARIO_MHS_ALPHA_ENGINE_08 (second half): MhsDiagnosticRequest raises
    # ValueError on unknown slow_book_mode/rebalance_filter/ensemble_signal
    # values and on a non-bool beta_neutralize; the defaults stay frozen.
    req = MhsDiagnosticRequest()
    assert req.slow_book_mode == "single_horizon"
    assert req.rebalance_filter == "per_symbol_deadband"
    assert req.beta_neutralize is False
    assert req.ensemble_signal == "raw"
    with pytest.raises(ValueError, match="slow_book_mode"):
        MhsDiagnosticRequest(slow_book_mode="bogus")
    with pytest.raises(ValueError, match="rebalance_filter"):
        MhsDiagnosticRequest(rebalance_filter="bogus")
    with pytest.raises(ValueError, match="beta_neutralize"):
        MhsDiagnosticRequest(beta_neutralize=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ensemble_signal"):
        MhsDiagnosticRequest(ensemble_signal="bogus")
