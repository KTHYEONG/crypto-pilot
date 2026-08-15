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
from src.mhs.execution import SimulatedInventoryLedgerResult
from src.mhs.execution import strategy_aware_execution_replay
from src.mhs.evaluation import AnchoredPurgedFold
from src.research.universe.pit_universe import symbol_partition

_START = pd.Timestamp("2021-01-01", tz="UTC")


def _write_mhs_market(
    root: Path,
    n_hours: int = 2700,
    include_btc: bool = False,
    funding_cross_sectional: bool = False,
    with_minute: bool = True,
) -> pd.Timestamp:
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
        if with_minute:
            mp = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.002, len(minute_idx))))
            pd.DataFrame(
                {"timestamp": minute_epoch, "open": mp, "high": mp * 1.0005,
                 "low": mp * 0.9995, "close": mp, "quote_vol": [1000.0] * len(minute_idx)},
            ).to_parquet(mdir / f"{sym}.parquet")
            mark = pd.Series(mp, index=minute_idx).resample("1h").last().reindex(hourly).to_numpy()
            pd.DataFrame(
                {"timestamp": epoch, "open": mark, "high": mark, "low": mark, "close": mark, "datetime": hourly},
            ).to_parquet(mkdir / f"{sym}.parquet")
        funding_rate = 0.00005 * (1.0 + 0.2 * i) if funding_cross_sectional else 0.00005
        pd.DataFrame(
            {"timestamp": epoch, "funding_rate": [funding_rate] * n_hours, "datetime": hourly},
        ).to_parquet(fdir / f"{sym}.parquet")
    return end


@pytest.fixture
def mhs_market_long(tmp_path, monkeypatch):
    # B1 fixture: spans 2021-01-01 .. 2024-01-01 so the committee diagnostic's
    # OOS block grid anchored at MHS_COMMITTEE_OOS_START (2023-01-01) has real
    # test bars (docs/specs/mhs_committee_evaluation_integrity_fixes.md §1).
    # Minute frames are skipped: the committee diagnostic runs on 1h panels
    # only, and the execution/fold paths are monkeypatched in these tests.
    root = tmp_path / "market"
    end = _write_mhs_market(root, n_hours=26304, with_minute=False)
    monkeypatch.setattr(ev, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(fc, "_mark_price_path", lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet")
    return root, end


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


@pytest.fixture
def mhs_market_funding_vary(tmp_path, monkeypatch):
    """Synthetic market whose funding rate differs per symbol, so a trailing
    funding carry signal has genuine cross-sectional dispersion (the constant
    ``mhs_market`` funding collapses a funding-carry book to zero weights)."""
    root = tmp_path / "market_funding_vary"
    end = _write_mhs_market(root, funding_cross_sectional=True)
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


def test_trend_sleeve_request_validation() -> None:
    # SCENARIO_MHS_REQUEST_TREND_SLEEVE_VALIDATION: MhsDiagnosticRequest gains
    # trend_sleeve (bool, default False) and trend_sleeve_gross (float, default
    # 0.0). A positive gross without the opt-in, or a gross outside [0.0, 1.0],
    # raises ValueError (fail closed -- no silent no-op); the default
    # construction leaves both at their off values.
    default = MhsDiagnosticRequest()
    assert default.trend_sleeve is False
    assert default.trend_sleeve_gross == 0.0
    with pytest.raises(ValueError, match="trend_sleeve_gross"):
        MhsDiagnosticRequest(trend_sleeve_gross=0.3)
    with pytest.raises(ValueError, match="trend_sleeve_gross"):
        MhsDiagnosticRequest(trend_sleeve=True, trend_sleeve_gross=-0.1)
    with pytest.raises(ValueError, match="trend_sleeve_gross"):
        MhsDiagnosticRequest(trend_sleeve=True, trend_sleeve_gross=1.5)
    with pytest.raises(ValueError, match="trend_sleeve"):
        MhsDiagnosticRequest(trend_sleeve="yes")
    on = MhsDiagnosticRequest(trend_sleeve=True, trend_sleeve_gross=0.3)
    assert on.trend_sleeve is True
    assert on.trend_sleeve_gross == 0.3


def test_multi_feature_request_validation() -> None:
    # SCENARIO_MHS_REQUEST_MULTI_FEATURE_VALIDATION: MhsDiagnosticRequest gains
    # multi_feature_book (bool, default False). A non-bool value raises
    # ValueError (fail closed -- no silent no-op); the default construction
    # leaves it False and the report's multi_feature_diagnostic is None.
    assert MhsDiagnosticRequest().multi_feature_book is False
    with pytest.raises(ValueError, match="multi_feature_book"):
        MhsDiagnosticRequest(multi_feature_book="yes")
    on = MhsDiagnosticRequest(multi_feature_book=True)
    assert on.multi_feature_book is True


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
    off raises ValueError from ``__post_init__`` (fail-closed -- no silent
    no-op), independent of and in addition to the existing adjusted-net-t
    validation, and the flag defaults to False / composes with
    ``discovery_gate=True``."""
    with pytest.raises(ValueError, match="discovery_gate_regime_scaled_net_t"):
        MhsDiagnosticRequest(discovery_gate=False, discovery_gate_regime_scaled_net_t=True)
    assert MhsDiagnosticRequest().discovery_gate_regime_scaled_net_t is False
    assert MhsDiagnosticRequest(
        discovery_gate=True, discovery_gate_regime_scaled_net_t=True,
    ).discovery_gate_regime_scaled_net_t is True


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
    with pytest.raises(DataIntegrityError, match="invalid"):
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
        """MHS-MEM-PAIR-02: the fold orchestrator materializes the execution
        windows ONCE and reuses the same windows (with per-pass weight
        substitution) across every replay bound -- the strongest form of the
        no-re-materialization invariant."""
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
        # One materialization feeds the two-pass primary and the stress bound.
        assert calls["n"] == 1

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
        # One materialization feeds the two-pass primary, stress, and the
        # informational patient-reference bound.
        assert calls["n"] == 1

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


def test_book_outcome_realized_cost_reaches_report(mhs_market) -> None:
    # SCENARIO_MHS_REALIZED_EXECUTION_COST_REACHES_REPORT_04: _book_outcome
    # projects the already-computed per-fill shortfall aggregates onto
    # MhsBookReport (they were previously discarded); the stress spec triples
    # fees, so the realized stress shortfall must be strictly higher than the
    # primary's.
    args = _build_book_outcome_args(mhs_market)
    report = ev._book_outcome(**args)
    assert report.failure is None
    assert report.primary is not None
    assert report.stress is not None
    for field in (
        "primary_realized_shortfall_bps",
        "primary_notional_weighted_shortfall_bps",
        "stress_realized_shortfall_bps",
        "stress_notional_weighted_shortfall_bps",
        "primary_forced_exit_notional",
    ):
        value = getattr(report, field)
        assert value is not None
        assert np.isfinite(value)
    assert report.primary_fill_count is not None
    assert report.primary_unfilled_count is not None
    assert report.stress_realized_shortfall_bps > report.primary_realized_shortfall_bps
    assert report.stress_notional_weighted_shortfall_bps > report.primary_notional_weighted_shortfall_bps


def test_pnl_vol_target_flag_defaults_true_and_gates_only_pass_two(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_PNL_VOL_TARGET_FLAG_DEFAULTS_TRUE_AND_IS_IDENTITY_05: the
    # flag defaults True (a run at the default is byte-identical to today), a
    # non-bool value is rejected, and with pnl_vol_target=False ONLY the
    # vol-target multiplication is skipped -- Pass 1/pre_vol_target_reference
    # stay structurally unchanged.
    assert MhsDiagnosticRequest().pnl_vol_target is True
    with pytest.raises(ValueError, match="pnl_vol_target"):
        MhsDiagnosticRequest(pnl_vol_target="yes")

    args = _build_book_outcome_args(mhs_market)
    default_report = ev._book_outcome(**args)
    true_report = ev._book_outcome(
        **{**args, "request": dataclasses.replace(args["request"], pnl_vol_target=True)}
    )
    # The default reproduces the pre-change primary/stress metrics exactly.
    assert default_report.primary_naive_sharpe == pytest.approx(true_report.primary_naive_sharpe)
    assert default_report.stress_naive_sharpe == pytest.approx(true_report.stress_naive_sharpe)

    def _forced_step_scale(reference_daily_returns: pd.Series) -> pd.Series:
        idx = reference_daily_returns.index
        mid = idx[0] + (idx[-1] - idx[0]) / 2
        return pd.Series(np.where(idx < mid, 1.0, 0.2), index=idx)

    monkeypatch.setattr(ev, "_pnl_vol_target_scale", _forced_step_scale)
    on = ev._book_outcome(**args)
    off = ev._book_outcome(
        **{**args, "request": dataclasses.replace(args["request"], pnl_vol_target=False)}
    )
    # Pass-1 reference is identical across the two branches.
    assert on.pre_vol_target_reference_naive_sharpe == pytest.approx(
        off.pre_vol_target_reference_naive_sharpe
    )
    # Off branch: Pass 2 replays the same unscaled weights as Pass 1.
    assert off.primary_naive_sharpe == pytest.approx(off.pre_vol_target_reference_naive_sharpe)
    # On branch: the two passes differ by exactly the one multiplicative factor.
    assert on.primary_naive_sharpe != pytest.approx(on.pre_vol_target_reference_naive_sharpe)
    assert on.primary_naive_sharpe != pytest.approx(off.primary_naive_sharpe)


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
    funding_by_symbol = ev._load_funding_series(symbols)
    universe_size = 8
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(ev, "phase_1_anchored_purged_folds", lambda: ())
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=universe_size,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
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
    retention_report = ev.run_mhs_horizon_diagnostic(request)
    assert retention_report.realized_execution_roster_size == pytest.approx(retention_mean)
    assert retention_report.realized_execution_roster_size > universe_size


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


def test_mhs_perf_opt_004_window_frames_read_window_only(mhs_market, monkeypatch) -> None:
    # MHS_PERF_OPT_004_WINDOW_FRAMES: the production window loader
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
    # SCENARIO_P10_CONCURRENT / SCENARIO_MHS_REGIME_SCALE_OMITTED_BYTE_IDENTICAL_02:
    # three books executed concurrently in fork workers produce bit-identical
    # reports to the sequential path. This call omits regime_scale (defaults
    # to None), so it also proves every existing _run_books_concurrent caller
    # stays byte-identical after the regime_scale parameter was added.
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


def test_p10_mark_cache_warmable_per_symbol(mhs_market) -> None:
    # SCENARIO_P10_EAGER_CACHE (post fork-COW refactor): the parent-side mark
    # frame cache warms one symbol's mark parquet per call so fork children can
    # inherit the populated cache copy-on-write (the minute-frame preload was
    # removed; docs/specs/mhs_refactor.md §2.2 W6).
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


def test_regime_scale_reaches_blend_replay_not_only_prescreen(mhs_market) -> None:
    # SCENARIO_MHS_REGIME_SCALE_REACHES_BLEND_REPLAY_01: previously blend_replay
    # (the blend book's actual execution-replay weights inside
    # _run_books_concurrent) was reconstructed independently from
    # w_fast_execution/w_slow_execution and never multiplied by regime_scale,
    # so the top-level "blend" book's primary/stress replay never reflected the
    # R1 volatility-regime cash scale (or, transitively, the opt-in
    # trend_efficiency_overlay) even though blend_1h/blend_step (prescreen/tail
    # diagnostics) did -- contradicting the R1 comment's own stated intent
    # (docs/specs/mhs_capital_floor_and_overlay_validation.md §2). With a
    # regime_scale that is < 1.0 on part of the grid, the blend book's replay
    # must now show reduced gross exposure on those decisions relative to a
    # None (no-scale) baseline, while fast_reversal/slow_momentum's own
    # standalone books stay byte-identical (the scale is blend-only, matching
    # the anchored-fold path's design where it applies to the portfolio target,
    # not each book's own raw signal).
    args = _build_books_concurrent_args(mhs_market)
    active_grid = ev._active_blend_book_and_grid(
        args["fast"], args["slow"], args["fast_grid"], args["slow_grid"],
    )[1]
    half = len(active_grid) // 2
    scale = pd.Series(1.0, index=active_grid)
    scale.iloc[:half] = 0.5

    fast_base, slow_base, blend_base = ev._run_books_concurrent(**args)
    fast_scaled, slow_scaled, blend_scaled = ev._run_books_concurrent(**args, regime_scale=scale)

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
        equity, 365 * 24, research_go_eligible=False, n_bootstrap=20, seed=7,
    )
    placeholder = ev.compute_deployment_readiness(
        equity, 365 * 24, research_go_eligible=None, primary_valid=True,
        n_bootstrap=20, seed=7,
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

    def _admit_by_family(*args, **kwargs):
        # The funding-carry family's selected lookback must come from its own
        # measured grid; the slow/fast families keep the 360h selection.
        if kwargs.get("horizon_candidates") == ev.MHS_FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS:
            return _admitted_selection(72)
        return _admitted_selection(360)

    monkeypatch.setattr(ev, "fold_train_only_discovery_qualification", _admit_by_family)

    def _spy_books(*args, **kwargs):
        captured["top_level_slow"] = args[5]
        return (None, None, None)

    def _spy_post(*args, **kwargs):
        captured["fold_slow_horizons"] = args[14] if len(args) > 14 else None
        captured["fold_fast_horizons"] = args[15] if len(args) > 15 else None
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
    # The fast re-verification is diagnostic-only: the parent threads the
    # resolved (horizon, source) pairs to the fold pool but never alters the
    # top-level fast spec (still the frozen 48h default, and blend weights
    # stay 0.0).
    assert captured["fold_fast_horizons"] == {
        0: (360, "fold_train_only_discovery"),
        1: (360, "fold_train_only_discovery"),
        2: (360, "fold_train_only_discovery"),
    }
    assert captured["top_level_slow"].horizon_hours == 360
    assert captured["top_level_slow"].band is ev.PHASE_1_BOOK_SPECS["slow_momentum"].band


def test_fold_worker_records_fast_horizon_override(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_FOLD_REPORT_FAST_HORIZON_FIELDS_DEFAULT (fold worker path):
    # a fold run resolved with a fast fold-scoped override records the selected
    # (horizon, source) on the report while the slow fields stay on the frozen
    # default -- mirroring the slow_horizon_* recording path and keeping the
    # fast re-verification diagnostic-only (no BookSpec/weight construction).
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
        str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
        fast_horizon_override=(96, "fold_train_only_discovery"),
    )
    assert report.fast_horizon_hours == 96
    assert report.fast_horizon_source == "fold_train_only_discovery"
    assert report.slow_horizon_hours == 168
    assert report.slow_horizon_source == "frozen_default"


def test_fold_safe_horizon_builds_candidate_weights_once_and_shares_across_folds(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_HORIZON_SEARCH_EFF_05_TOP_LEVEL_WIRING_SHARES_ONE_CACHE_ACROSS_FOLDS:
    # with fold_safe_horizon_selection=True the parent precomputes every
    # discovery weight book exactly once (``_candidate_weight_books``) and every
    # fold's gate reuses that single precompute (fork-inherited copy-on-write in
    # the parallel fold-safe path) -- the measured 3x-redundant weight
    # construction is eliminated without changing any value. The
    # parallel/sequential value-equivalence is pinned by
    # ``test_mhs_perf_opt_fold_discovery_parallel_equivalence``.
    root, end = mhs_market
    calls = {"n": 0}
    real_builder = ev._candidate_weight_books

    def counting_builder(*args, **kwargs):
        calls["n"] += 1
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(ev, "_candidate_weight_books", counting_builder)

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


def test_horizon_diagnostics_exposes_effective_breadth(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_HORIZON_DIAGNOSTICS_EXPOSES_EFFECTIVE_BREADTH_04: with
    # discovery_gate=True run_mhs_horizon_diagnostic reports finite
    # slow_horizon_effective_breadth/fast_horizon_effective_breadth within
    # [1.0, nominal_candidate_count]; with discovery_gate=False (the default)
    # the two keys are absent -- opt-in, no default-path cost.
    root, end = mhs_market
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
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
    slow_n_eff = report_on.horizon_diagnostics.get("slow_horizon_effective_breadth")
    fast_n_eff = report_on.horizon_diagnostics.get("fast_horizon_effective_breadth")
    assert slow_n_eff is not None
    assert np.isfinite(slow_n_eff)
    assert fast_n_eff is not None
    assert np.isfinite(fast_n_eff)
    assert 1.0 <= slow_n_eff <= 19
    assert 1.0 <= fast_n_eff <= 7

    request_off = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    report_off = ev.run_mhs_horizon_diagnostic(request_off)
    assert report_off.status == "COMPLETE"
    assert "slow_horizon_effective_breadth" not in report_off.horizon_diagnostics
    assert "fast_horizon_effective_breadth" not in report_off.horizon_diagnostics


def test_trend_sleeve_default_off_bit_identical(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_TREND_SLEEVE_DEFAULT_OFF_BIT_IDENTICAL: with the flags
    # omitted (trend_sleeve=False, trend_sleeve_gross=0.0) the report's
    # trend_sleeve_diagnostic is None and every pre-existing field is
    # bit-identical to the explicit-off baseline -- the sleeve is inert unless
    # explicitly enabled, so a default run cannot change any existing output.
    root, end = mhs_market
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    base = {
        "start": str(_START), "end": str(end), "data_root": str(root),
        "mark_mode": "cache_required", "execution_timeframe": "1m", "log_run": False,
        "execution_universe_size": 8,
    }
    default_report = ev.run_mhs_horizon_diagnostic(MhsDiagnosticRequest(**base))
    explicit_off = ev.run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(**base, trend_sleeve=False, trend_sleeve_gross=0.0),
    )
    assert default_report.status == "COMPLETE"
    assert default_report.trend_sleeve_diagnostic is None
    assert explicit_off.trend_sleeve_diagnostic is None
    for field in ("books", "blend", "blend_target_gross", "research_go", "folds"):
        assert getattr(default_report, field) == getattr(explicit_off, field)


def test_trend_sleeve_diagnostic_populated(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_TREND_SLEEVE_DIAGNOSTIC_POPULATED: with trend_sleeve=True
    # and trend_sleeve_gross=0.3 the report's trend_sleeve_diagnostic is a dict
    # carrying the sleeve's standalone net Sharpe per measured cost tier, its
    # per-calendar-year net_t, its correlation to the slow_momentum book's pnl,
    # and the combined metrics; every value is finite or an explicit None,
    # never NaN silently coerced to 0.0.
    root, end = mhs_market
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, trend_sleeve=True, trend_sleeve_gross=0.3,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    diag = report.trend_sleeve_diagnostic
    assert isinstance(diag, dict)
    assert set(diag["net_sharpe_per_tier"]) == set(ev.MEASURED_EXECUTION_COST_TIERS_BPS)
    for value in diag["net_sharpe_per_tier"].values():
        assert value is None or np.isfinite(value)
    yearly = diag["yearly_net_t"]
    assert isinstance(yearly, dict)
    assert set(yearly) == {2021, 2022, 2023, 2024, 2025}
    for value in yearly.values():
        assert value is None or np.isfinite(value)
    corr = diag["slow_momentum_pnl_corr"]
    assert corr is None or np.isfinite(corr)
    combined = diag["combined"]
    assert set(combined["net_sharpe_per_tier"]) == set(ev.MEASURED_EXECUTION_COST_TIERS_BPS)
    for value in combined["net_sharpe_per_tier"].values():
        assert value is None or np.isfinite(value)
    worst = combined["worst_year_net_t"]
    assert worst is None or np.isfinite(worst)


def test_multi_feature_default_off_bit_identical(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_MULTI_FEATURE_DEFAULT_OFF_BIT_IDENTICAL: with the flag
    # omitted (multi_feature_book=False) the report's multi_feature_diagnostic
    # is None and every pre-existing field is bit-identical to the explicit-off
    # baseline -- the multi-feature axis is inert unless explicitly enabled.
    root, end = mhs_market
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    base = {
        "start": str(_START), "end": str(end), "data_root": str(root),
        "mark_mode": "cache_required", "execution_timeframe": "1m", "log_run": False,
        "execution_universe_size": 8,
    }
    default_report = ev.run_mhs_horizon_diagnostic(MhsDiagnosticRequest(**base))
    explicit_off = ev.run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(**base, multi_feature_book=False),
    )
    assert default_report.status == "COMPLETE"
    assert default_report.multi_feature_diagnostic is None
    assert explicit_off.multi_feature_diagnostic is None
    for field in ("books", "blend", "blend_target_gross", "research_go", "folds"):
        assert getattr(default_report, field) == getattr(explicit_off, field)


def test_multi_feature_diagnostic_reports_coverage_and_stability(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_MULTI_FEATURE_DIAGNOSTIC_REPORTS_COVERAGE_AND_STABILITY:
    # with multi_feature_book=True the report's multi_feature_diagnostic dict
    # carries, per admitted feature, its per-year coverage and its regime-split
    # stability fields, plus the combined book's net Sharpe per measured cost
    # tier and the effective breadth of the feature-book PnL panel; features
    # excluded by the coverage gate are listed under an explicit excluded key
    # with their failing year, never silently dropped.
    root, end = mhs_market
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, multi_feature_book=True,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    diag = report.multi_feature_diagnostic
    assert isinstance(diag, dict)
    admitted = diag["admitted"]
    assert isinstance(admitted, dict)
    assert admitted, "the fixture's feature columns must admit at least one feature"
    for feature_name, fields in admitted.items():
        assert isinstance(feature_name, str)
        coverage = fields["coverage"]
        assert isinstance(coverage, dict)
        for value in coverage.values():
            assert 0.0 <= value <= 1.0
        stability = fields["regime_split_stability"]
        assert isinstance(stability, dict)
        for label, sharpe in stability["window_sharpes"]:
            assert isinstance(label, str)
            assert sharpe is None or np.isfinite(sharpe)
        assert stability["min_window_sharpe"] is None or np.isfinite(
            stability["min_window_sharpe"]
        )
        assert isinstance(stability["sign_consistent"], bool)
        assert stability["decay"] is None or np.isfinite(stability["decay"])
    excluded = diag["excluded"]
    assert isinstance(excluded, dict)
    for fields in excluded.values():
        assert "failing_year" in fields
    combined = diag["combined"]
    assert set(combined["net_sharpe_per_tier"]) == set(ev.MEASURED_EXECUTION_COST_TIERS_BPS)
    for value in combined["net_sharpe_per_tier"].values():
        assert value is None or np.isfinite(value)
    breadth = diag["feature_book_effective_breadth"]
    assert isinstance(breadth, dict)
    assert "n_eff" in breadth
    assert "mean_corr" in breadth
    assert np.isfinite(breadth["n_eff"])
    assert np.isfinite(breadth["mean_corr"])


def test_committee_request_validation() -> None:
    # SCENARIO_MHS_REQUEST_COMMITTEE_VALIDATION: MhsDiagnosticRequest gains
    # committee_book (bool, default False). A non-bool value raises ValueError
    # (fail closed -- no silent no-op); the default construction leaves it False.
    assert MhsDiagnosticRequest().committee_book is False
    with pytest.raises(ValueError, match="committee_book"):
        MhsDiagnosticRequest(committee_book="yes")
    on = MhsDiagnosticRequest(committee_book=True)
    assert on.committee_book is True


def test_committee_default_off_bit_identical(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_COMMITTEE_DEFAULT_OFF_BIT_IDENTICAL: with the flag omitted
    # (committee_book=False) the report's committee_diagnostic is None and every
    # pre-existing field is bit-identical to the explicit-off baseline -- the
    # committee axis is inert unless explicitly enabled.
    root, end = mhs_market
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    base = {
        "start": str(_START), "end": str(end), "data_root": str(root),
        "mark_mode": "cache_required", "execution_timeframe": "1m", "log_run": False,
        "execution_universe_size": 8,
    }
    default_report = ev.run_mhs_horizon_diagnostic(MhsDiagnosticRequest(**base))
    explicit_off = ev.run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(**base, committee_book=False),
    )
    assert default_report.status == "COMPLETE"
    assert default_report.committee_diagnostic is None
    assert explicit_off.committee_diagnostic is None
    for field in ("books", "blend", "blend_target_gross", "research_go", "folds"):
        assert getattr(default_report, field) == getattr(explicit_off, field)


def test_committee_diagnostic_reports_walk_forward_wealth(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_MHS_COMMITTEE_DIAGNOSTIC_REPORTS_WALK_FORWARD_WEALTH: with
    # committee_book=True the report's committee_diagnostic dict carries the
    # declared member names, the admitted/excluded split against the coverage
    # gate (feature- and source-gated, each with a reason), per-required-column
    # source coverage audited before any fillna, and the purged walk-forward
    # wealth metrics (net Sharpe, CAGR, MDD, logret) per measured cost tier --
    # every reported value finite or an explicit None. The fixture spans past
    # MHS_COMMITTEE_OOS_START so the block grid has real test bars (B1).
    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_book=True,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    diag = report.committee_diagnostic
    assert isinstance(diag, dict)
    members = diag["members"]
    assert isinstance(members, list)
    assert len(members) == 6
    assert len(set(members)) == 6
    admitted = diag["admitted"]
    excluded = diag["excluded"]
    assert isinstance(admitted, list)
    assert isinstance(excluded, list)
    assert set(admitted) <= set(members)
    assert all(isinstance(e, dict) and e["name"] in members for e in excluded)
    for entry in excluded:
        assert entry["reason"] in ("feature_coverage", "source_coverage")
        if entry["reason"] == "source_coverage":
            assert entry["failing_source"] in (
                "close", "open", "high", "low", "quote_vol",
                "taker_buy_quote", "no_trades",
            )
            assert isinstance(entry["failing_year"], int)
    source_coverage = diag["source_coverage"]
    assert isinstance(source_coverage, dict)
    for per_source in source_coverage.values():
        assert isinstance(per_source, dict)
        for coverage in per_source.values():
            assert isinstance(coverage, dict)
            for value in coverage.values():
                assert 0.0 <= value <= 1.0
    wf = diag["walk_forward"]
    assert isinstance(wf["block_edges"], list)
    assert wf["block_edges"][0] == ev.MHS_COMMITTEE_OOS_START.isoformat()
    assert wf["purge_hours"] == 720
    assert wf["target_vol"] == pytest.approx(0.15)
    assert isinstance(wf["skipped_blocks"], list)
    per_tier = wf["per_tier"]
    assert set(per_tier) == set(ev.MEASURED_EXECUTION_COST_TIERS_BPS)
    for fields in per_tier.values():
        assert isinstance(fields["bars"], int)
        assert fields["bars"] >= 0
        for key in ("net_sharpe", "cagr", "mdd", "logret"):
            value = fields[key]
            assert value is None or np.isfinite(value)

def test_committee_diagnostic_uses_oos_start_not_raw_start(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_COMMITTEE_DIAGNOSTIC_USES_OOS_START_NOT_RAW_START (B1): on a
    # panel spanning 2021-2025 the committee diagnostic's walk-forward block
    # grid is anchored at MHS_COMMITTEE_OOS_START (2023-01-01), never the
    # diagnostic's own 2021 start; monkeypatching the constant to a different
    # date shifts the first edge, proving the constant is actually read.
    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_book=True,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    first_edge = report.committee_diagnostic["walk_forward"]["block_edges"][0]
    assert first_edge == ev.MHS_COMMITTEE_OOS_START.isoformat()
    assert first_edge == "2023-01-01T00:00:00+00:00"

    shifted = pd.Timestamp("2023-07-01", tz="UTC")
    monkeypatch.setattr(ev, "MHS_COMMITTEE_OOS_START", shifted)
    report2 = ev.run_mhs_horizon_diagnostic(request)
    assert report2.status == "COMPLETE"
    first_edge2 = report2.committee_diagnostic["walk_forward"]["block_edges"][0]
    assert first_edge2 == shifted.isoformat()
    assert first_edge2 == "2023-07-01T00:00:00+00:00"


def test_search_trials_attempted_raised_and_deflation_more_conservative() -> None:
    # SCENARIO_SEARCH_TRIALS_ATTEMPTED_RAISED_AND_DEFLATED_SHARPE_MORE_CONSERVATIVE
    # (B4): MHS_SEARCH_TRIALS_ATTEMPTED is raised to 70 (prior 20 + ~50 committee
    # configurations), and deflated_sharpe_ratio is strictly non-increasing in
    # the trial count, so the raised constant can only make the top-level
    # statistic more conservative, never more optimistic.
    from src.mhs.contracts import MHS_SEARCH_TRIALS_ATTEMPTED
    from src.mhs.evaluation import deflated_sharpe_ratio

    assert MHS_SEARCH_TRIALS_ATTEMPTED == 70
    kwargs = {"observed_sr": 0.12, "trial_sr_variance": 0.0025, "n_obs": 1200, "skew": 0.0, "kurtosis": 3.0}
    d70 = deflated_sharpe_ratio(n_trials=70, **kwargs)
    d20 = deflated_sharpe_ratio(n_trials=20, **kwargs)
    assert np.isfinite(d70)
    assert np.isfinite(d20)
    assert d70 <= d20


def test_committee_source_coverage_gates_admission(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_COMMITTEE_SOURCE_COVERAGE_GATES_ADMISSION (B3): a member whose
    # required RAW source column has coverage below MHS_FEATURE_MIN_COVERAGE in
    # any year is fail-closed excluded from admission -- the fixture's missing
    # taker_buy_quote column (mirroring the funding 45/452-symbol gap) gates
    # both flow_imb members BEFORE build_feature_books, and the excluded list
    # carries the failing source/year. With a full-coverage taker_buy_quote the
    # gate is a no-op and all 6 members are admitted (regression).
    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_book=True,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    diag = report.committee_diagnostic
    assert "flow_imb_720h" not in diag["admitted"]
    assert "flow_imb_168h" not in diag["admitted"]
    flow_excluded = {e["name"]: e for e in diag["excluded"] if e["name"] in ("flow_imb_720h", "flow_imb_168h")}
    assert set(flow_excluded) == {"flow_imb_720h", "flow_imb_168h"}
    for entry in flow_excluded.values():
        assert entry["reason"] == "source_coverage"
        assert entry["failing_source"] == "taker_buy_quote"
        assert isinstance(entry["failing_year"], int)

    # Regression: with a full-coverage taker_buy_quote the source gate admits
    # every member -- B3 is fail-closed-only and non-disruptive to the shipped
    # committee.
    real_load = ev._load_feature_panels
    def _full_coverage_panels(root_arg, start_arg, end_arg, grid_1h, aligned_symbols, columns=None):
        panels = real_load(root_arg, start_arg, end_arg, grid_1h, aligned_symbols, columns=columns)
        quote_vol = panels["quote_vol"]
        panels["taker_buy_quote"] = quote_vol * 0.5
        return panels
    monkeypatch.setattr(ev, "_load_feature_panels", _full_coverage_panels)
    report_full = ev.run_mhs_horizon_diagnostic(request)
    assert report_full.status == "COMPLETE"
    assert set(report_full.committee_diagnostic["admitted"]) == set(
        report_full.committee_diagnostic["members"]
    )


def test_committee_diagnostic_reports_trials_and_warning(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_COMMITTEE_DIAGNOSTIC_REPORTS_TRIALS_AND_WARNING (B4/B5): the
    # committee diagnostic reports trials_explored == 50 and a non-empty
    # selection_bias_warning naming the configuration count, and tags its
    # evaluation protocol as purged walk-forward OOS (B5).
    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_book=True,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    diag = report.committee_diagnostic
    assert diag["trials_explored"] == 50
    assert isinstance(diag["selection_bias_warning"], str)
    assert diag["selection_bias_warning"]
    assert "~50" in diag["selection_bias_warning"]
    assert diag["evaluation_protocol"] == "purged_walk_forward_oos"


def test_evaluation_protocol_field_distinguishes_in_sample_from_oos(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_EVALUATION_PROTOCOL_FIELD_DISTINGUISHES_IN_SAMPLE_FROM_OOS (B5):
    # the two opt-in diagnostics carry distinct protocol tags on every call, so
    # a reader can never mistake the in-sample full-period net Sharpe for the
    # purged walk-forward OOS numbers.
    from src.mhs.features import MHS_FEATURE_REGISTRY

    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_book=True, multi_feature_book=True,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    assert report.committee_diagnostic["evaluation_protocol"] == "purged_walk_forward_oos"
    assert report.multi_feature_diagnostic["evaluation_protocol"] == "in_sample_full_period"
    assert report.multi_feature_diagnostic["trials_explored"] == len(MHS_FEATURE_REGISTRY)
    assert "selection_bias_warning" not in report.multi_feature_diagnostic


def test_committee_diagnostic_reports_skipped_blocks(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_COMMITTEE_DIAGNOSTIC_REPORTS_SKIPPED_BLOCKS (B6): the committee
    # diagnostic reports skipped_blocks as a list of {block_start, reason}
    # entries computed independently of purged_walk_forward's internal skip
    # loop. On a 2021-2025 panel anchored at OOS_START 2023-01-01 every 6-month
    # block has both sufficient train and at least one test bar, so the list is
    # empty (report-only, never raises).
    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_book=True,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    skipped = report.committee_diagnostic["walk_forward"]["skipped_blocks"]
    assert isinstance(skipped, list)
    for entry in skipped:
        assert set(entry) == {"block_start", "reason"}
        assert entry["reason"] in ("insufficient_train", "no_test_bars")
    assert skipped == []


def test_committee_books_regression_unchanged_by_b1_b2(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_COMMITTEE_BOOKS_REGRESSION_UNCHANGED_BY_B1_B2: enabling
    # committee_book must not perturb any pre-existing non-committee report
    # field (books, blend, folds, research_go, trend_sleeve_diagnostic,
    # multi_feature_diagnostic) -- only committee_diagnostic's own walk-forward
    # numbers change by design (B1/B2).
    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    base = {
        "start": str(_START), "end": str(end), "data_root": str(root),
        "mark_mode": "cache_required", "execution_timeframe": "1m", "log_run": False,
        "execution_universe_size": 8,
    }
    off_report = ev.run_mhs_horizon_diagnostic(MhsDiagnosticRequest(**base))
    on_report = ev.run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(**base, committee_book=True),
    )
    assert off_report.status == "COMPLETE"
    assert on_report.status == "COMPLETE"
    for field in (
        "books", "blend", "blend_target_gross", "research_go", "folds",
        "trend_sleeve_diagnostic", "multi_feature_diagnostic",
    ):
        assert getattr(on_report, field) == getattr(off_report, field)

def test_ram_guard_resolve_budget(monkeypatch) -> None:
    # SCENARIO_MHS_RAM_GUARD_RESOLVE_BUDGET: _resolve_ram_budget maps the
    # request into (budget_bytes, reserve_bytes). ram_guard=False disables the
    # guard; ram_guard=True auto-derives 85% of total RAM and the reserve floor
    # max(5% of total, 256 MiB); an explicit max_rss_bytes overrides the budget
    # fraction; psutil failure / non-positive total yields (None, None).
    from src.mhs.contracts import (
        MHS_RAM_BUDGET_FRACTION,
        MHS_RAM_RESERVE_FLOOR_BYTES,
        MHS_RAM_RESERVE_FRACTION,
    )

    class _FakeMem:
        total: int
        available: int
        def __init__(self, total: int, available: int) -> None:
            self.total = total
            self.available = available

    assert ev._resolve_ram_budget(None, False) == (None, None)

    monkeypatch.setattr(ev.psutil, "virtual_memory", lambda: _FakeMem(8 * 2**30, 4 * 2**30))
    budget, reserve = ev._resolve_ram_budget(None, True)
    assert budget == int(8 * 2**30 * MHS_RAM_BUDGET_FRACTION)
    assert reserve == max(int(8 * 2**30 * MHS_RAM_RESERVE_FRACTION), MHS_RAM_RESERVE_FLOOR_BYTES)

    explicit, reserve2 = ev._resolve_ram_budget(123456789, True)
    assert explicit == 123456789
    assert reserve2 == reserve

    monkeypatch.setattr(ev.psutil, "virtual_memory", lambda: _FakeMem(0, 0))
    assert ev._resolve_ram_budget(None, True) == (None, None)

    def _boom() -> _FakeMem:
        raise RuntimeError("psutil unavailable")
    monkeypatch.setattr(ev.psutil, "virtual_memory", _boom)
    assert ev._resolve_ram_budget(None, True) == (None, None)


def test_ram_guard_stage_barrier_fails_closed(monkeypatch) -> None:
    # SCENARIO_MHS_RAM_GUARD_STAGE_BARRIER_FAIL_CLOSED: _assert_stage_rss_budget
    # fails closed deterministically -- process RSS above the budget raises a
    # DataIntegrityError naming the stage; system available memory below the
    # reserve raises; (None, None) is a no-op.
    with pytest.raises(ev.DataIntegrityError, match="RAM budget exceeded at stage 'test_stage'"):
        ev._assert_stage_rss_budget("test_stage", 1, None)

    class _FakeMem:
        total: int
        available: int
        def __init__(self, total: int, available: int) -> None:
            self.total = total
            self.available = available

    monkeypatch.setattr(ev.psutil, "virtual_memory", lambda: _FakeMem(8 * 2**30, 100))
    with pytest.raises(ev.DataIntegrityError, match="reserve breached at stage 'test_reserve'"):
        ev._assert_stage_rss_budget("test_reserve", None, 4096)

    ev._assert_stage_rss_budget("noop", None, None)


def test_ram_guard_request_field() -> None:
    # SCENARIO_MHS_RAM_GUARD_REQUEST_FIELD: ram_guard defaults True on the
    # request; a non-bool value fails closed; max_rss_bytes stays None (auto
    # resolution happens at run time).
    assert MhsDiagnosticRequest().ram_guard is True
    assert MhsDiagnosticRequest().max_rss_bytes is None
    with pytest.raises(ValueError, match="ram_guard"):
        MhsDiagnosticRequest(ram_guard="yes")
    assert MhsDiagnosticRequest(ram_guard=False).ram_guard is False


def test_pipeline_ram_guard_fails_closed_before_oom(mhs_market_long) -> None:
    # SCENARIO_MHS_PIPELINE_RAM_GUARD_FAILS_CLOSED_BEFORE_OOM: a tiny explicit
    # budget makes run_mhs_horizon_diagnostic fail closed with DataIntegrityError
    # at the base_1h_panel stage boundary instead of letting the OS OOM killer
    # terminate the process.
    root, end = mhs_market_long
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, max_rss_bytes=1,
    )
    with pytest.raises(ev.DataIntegrityError, match="RAM budget exceeded at stage 'base_1h_panel'"):
        ev.run_mhs_horizon_diagnostic(request)


def test_diagnostics_run_after_folds_and_evict_caches(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_MHS_DIAGNOSTICS_RUN_AFTER_FOLDS_AND_EVICT_CACHES: the opt-in
    # diagnostics run only after the fold pool returned, the minute/mark frame
    # caches are evicted by the time the run completes, and the committee
    # diagnostic is still populated (regression against the re-ordering).
    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    order: list[str] = []
    real_post = ev._run_post_book_concurrently
    real_committee = ev._committee_diagnostic

    def _spy_post(*args, **kwargs):
        order.append("post_folds")
        return real_post(*args, **kwargs)

    def _spy_committee(*args, **kwargs):
        order.append("committee")
        return real_committee(*args, **kwargs)

    monkeypatch.setattr(ev, "_run_post_book_concurrently", _spy_post)
    monkeypatch.setattr(ev, "_committee_diagnostic", _spy_committee)

    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_book=True,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    assert order == ["post_folds", "committee"]
    assert isinstance(report.committee_diagnostic, dict)
    assert report.committee_diagnostic["evaluation_protocol"] == "purged_walk_forward_oos"

    # The full-period mark frame cache was evicted during the run (the minute
    # frame caches were removed in the fork-COW refactor).
    assert ev._get_symbol_mark_frame.cache_info().currsize == 0


def test_multi_feature_streaming_combined_bit_identical() -> None:
    # SCENARIO_MHS_MULTI_FEATURE_STREAMING_BIT_IDENTICAL: the streaming
    # multi-feature diagnostic produces combined.book_mean_gross,
    # combined.net_sharpe_per_tier and feature_book_effective_breadth EXACTLY
    # equal to a batch reference built from the same panels with the existing
    # primitives (build_feature_books + mhs_ledger_pnl + equal_risk_combination).
    from src.mhs.execution import mhs_ledger_pnl
    from src.mhs.features import (
        MHS_FEATURE_REGISTRY,
        build_feature_books,
        equal_risk_combination,
        feature_coverage_audit,
    )

    grid = pd.date_range("2021-01-01", periods=2400, freq="1h", tz="UTC")
    symbols = [f"S{i:02d}" for i in range(10)]
    rng = np.random.default_rng(9)
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 1e-4, (len(grid), len(symbols))), axis=0)),
        index=grid, columns=symbols,
    )
    quote_vol = pd.DataFrame(rng.uniform(900.0, 1100.0, (len(grid), len(symbols))), index=grid, columns=symbols)
    taker_buy_quote = quote_vol * rng.uniform(0.4, 0.6, (len(grid), len(symbols)))
    panels = {
        "close": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "quote_vol": quote_vol,
        "taker_buy_quote": taker_buy_quote,
        "no_trades": pd.DataFrame(1000, index=grid, columns=symbols),
    }
    mask = pd.DataFrame(True, index=grid, columns=symbols)
    decision_grid = pd.date_range(grid[0], grid[-1], freq="24h", tz="UTC")

    diag = ev._multi_feature_diagnostic(
        "ignored", _START, grid[-1], grid, symbols, mask, close, quote_vol * 0.0,
        panels=panels,
    )

    # Batch reference using the existing primitives.
    books = build_feature_books(MHS_FEATURE_REGISTRY, panels, mask, decision_grid, min_symbols=8)
    ref_admitted: dict[str, dict] = {}
    ref_excluded: dict[str, dict] = {}
    for spec in MHS_FEATURE_REGISTRY:
        feature = spec.builder(panels)
        coverage = feature_coverage_audit(feature, mask)
        failing = [year for year, cov in coverage.items() if cov < spec.min_coverage]
        if failing:
            ref_excluded[spec.name] = {"failing_year": min(failing)}
            continue
        if spec.name not in books:
            continue
        base_net, _ = mhs_ledger_pnl(
            books[spec.name], close, quote_vol * 0.0,
            ev.MEASURED_EXECUTION_COST_TIERS_BPS["base"],
        )
        ref_admitted[spec.name] = {"_net": base_net}
    net_panel = {name: fields["_net"] for name, fields in ref_admitted.items()}
    combinable = {}
    for name, net in net_panel.items():
        cleaned = net.dropna()
        sd = float(cleaned.std(ddof=1)) if len(cleaned) > 1 else 0.0
        if np.isfinite(sd) and sd > 0:
            combinable[name] = net
    combined = None
    ref_per_tier: dict[str, float | None] = {}
    if combinable:
        combined = equal_risk_combination(
            {name: books[name] for name in combinable}, combinable,
        )
        for tier, cost_bps in ev.MEASURED_EXECUTION_COST_TIERS_BPS.items():
            per_feature = {
                name: mhs_ledger_pnl(books[name], close, quote_vol * 0.0, cost_bps)[0]
                for name in combinable
            }
            combined_net = (
                sum(per_feature[name] / combinable[name].std(ddof=1) for name in combinable)
                / len(combinable)
            )
            ref_per_tier[tier] = ev._annualized_1h_sharpe(combined_net)
    else:
        ref_per_tier = dict.fromkeys(ev.MEASURED_EXECUTION_COST_TIERS_BPS)

    ref_gross: float | None = None
    if combined is not None:
        ref_gross = float(
            (
                combined
                * len(combinable)
                / sum(1.0 / combinable[name].std(ddof=1) for name in combinable)
            ).abs().sum(axis=1).mean()
        )
    ref_breadth: dict[str, float] | None = None
    if len(net_panel) >= 2:
        n_eff, mean_corr = ev.effective_breadth(pd.DataFrame(net_panel).fillna(0.0))
        ref_breadth = {"n_eff": n_eff, "mean_corr": mean_corr}

    assert set(diag["admitted"]) == set(ref_admitted)
    assert set(diag["excluded"]) == set(ref_excluded)
    assert diag["combined"]["book_mean_gross"] == ref_gross
    for tier in ev.MEASURED_EXECUTION_COST_TIERS_BPS:
        got = diag["combined"]["net_sharpe_per_tier"][tier]
        want = ref_per_tier[tier]
        assert (got is None and want is None) or got == want
    if ref_breadth is not None:
        assert diag["feature_book_effective_breadth"] == ref_breadth
    else:
        assert diag["feature_book_effective_breadth"] is None


def test_committee_streaming_regression(mhs_market_long, monkeypatch) -> None:
    # SCENARIO_MHS_COMMITTEE_STREAMING_REGRESSION: the streaming committee
    # rewrite is behavior-transparent -- the pre-existing walk-forward wealth
    # scenario (block edges anchored at OOS_START, purge 720, empty skipped
    # blocks, finite per-tier fields) still holds after the per-member book
    # streaming + multi-tier ledger.
    root, end = mhs_market_long
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently", lambda *a, **k: (None, None, {}, {}, (), None),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8, committee_book=True,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    diag = report.committee_diagnostic
    assert diag["walk_forward"]["block_edges"][0] == ev.MHS_COMMITTEE_OOS_START.isoformat()
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


def test_fold_worker_records_funding_carry_override(mhs_market) -> None:
    # SCENARIO_MHS_FOLD_REPORT_CARRIES_FUNDING_CARRY_DISCOVERY_05 (fold worker
    # path): a fold run resolved with a funding-carry override records all four
    # fields on the report; without the override (flag off / no admission) they
    # fail closed to their dataclass defaults -- existing construction sites
    # are unaffected.
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
        str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
        funding_carry_override=(72, 1, "fold_train_only_discovery", 0.15),
    )
    assert report.funding_carry_lookback_hours == 72
    assert report.funding_carry_sign == 1
    assert report.funding_carry_source == "fold_train_only_discovery"
    assert report.funding_carry_vs_slow_momentum_daily_corr == 0.15

    default_report = ev._run_anchored_fold(
        str(root), _FOLD, request, funding_by_symbol, 1.0, 0, None,
    )
    assert default_report.funding_carry_lookback_hours is None
    assert default_report.funding_carry_sign is None
    assert default_report.funding_carry_source == "frozen_default"
    assert default_report.funding_carry_vs_slow_momentum_daily_corr is None

    incomplete = ev._incomplete_fold_report(_FOLD, 0, ())
    assert incomplete.funding_carry_lookback_hours is None
    assert incomplete.funding_carry_source == "frozen_default"


def test_fold_safe_funding_carry_parent_wiring(mhs_market_funding_vary, monkeypatch) -> None:
    # SCENARIO_MHS_FOLD_REPORT_CARRIES_FUNDING_CARRY_DISCOVERY_05 (parent path):
    # with fold_safe_horizon_selection=True and a funding-carry admission the
    # parent threads (lookback, sign, source, corr) per fold with a finite
    # train-window orthogonality correlation against slow_momentum; when no
    # candidate admits, all four fields fail closed to frozen_default/None.
    root, end = mhs_market_funding_vary
    captured: dict = {}

    def _run(captured):
        monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))

        def _spy_post(*args, **kwargs):
            captured["fold_funding_carry"] = args[16] if len(args) > 16 else None
            return (None, None, {}, {}, (), None)

        monkeypatch.setattr(ev, "_run_post_book_concurrently", _spy_post)
        request_on = MhsDiagnosticRequest(
            start=str(_START), end=str(end), data_root=str(root),
            mark_mode="cache_required", execution_timeframe="1m", log_run=False,
            execution_universe_size=8, fold_safe_horizon_selection=True,
        )
        top_report = ev.run_mhs_horizon_diagnostic(request_on)
        assert top_report.status == "COMPLETE"
        return captured["fold_funding_carry"]

    def _admit_funding_only(*args, **kwargs):
        if kwargs.get("horizon_candidates") == ev.MHS_FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS:
            return _admitted_selection(72)
        return _admitted_selection(None)

    monkeypatch.setattr(ev, "fold_train_only_discovery_qualification", _admit_funding_only)
    admitted = _run(captured)
    assert set(admitted) == {0, 1, 2}
    for lookback, sign, source, corr in admitted.values():
        assert lookback == 72
        assert sign == 1
        assert source == "fold_train_only_discovery"
        assert np.isfinite(corr)

    captured.clear()
    monkeypatch.setattr(
        ev, "fold_train_only_discovery_qualification",
        lambda *a, **k: _admitted_selection(None),
    )
    fail_closed = _run(captured)
    assert set(fail_closed) == {0, 1, 2}
    for lookback, sign, source, corr in fail_closed.values():
        assert lookback is None
        assert sign is None
        assert source == "frozen_default"
        assert corr is None


def _deployment_readiness() -> ev.DeploymentReadinessResult:
    """A placeholder deployment readiness result for monkeypatched
    ``_run_post_book_concurrently`` returns (the real folds stream is skipped)."""
    return ev.DeploymentReadinessResult(
        geometric_cagr=0.0, max_drawdown=0.0, calmar=0.0, expected_shortfall=0.0,
        worst_1d=0.0, worst_7d=0.0, worst_event=0.0, time_under_water_bars=0,
        recovery_bars=None, probability_final_wealth_below_initial=0.0,
        probability_mdd_over_20pct=0.0, probability_mdd_over_30pct=0.0,
        leverage_ruin_probabilities={}, concentration={},
        participation_warnings={}, research_go_eligible=False,
        execution_go_eligible=False, pilot_go_eligible=False, scale_go_eligible=False,
    )


def test_mhs_fast_book_mode_default_is_identity(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_FAST_BOOK_MODE_DEFAULT_IS_IDENTITY_03: the default
    # fast_book_mode is single_horizon, an invalid value raises ValueError, and
    # a full run at the default reproduces the pre-change single-horizon fast
    # book byte-identically -- the regression-invariant proof. The default-path
    # run keeps real books; the production w_fast_execution matrix is captured
    # by a spy and must equal the verbatim pre-change chain (vol tilt +
    # renormalize) built on the same panel.
    assert MhsDiagnosticRequest().fast_book_mode == "single_horizon"
    with pytest.raises(ValueError, match="unknown fast_book_mode"):
        MhsDiagnosticRequest(fast_book_mode="bogus")

    root, end = mhs_market
    captured: dict = {}
    real_books = ev._run_books_concurrent

    def _spy_books(*args, **kwargs):
        captured["w_fast_execution"] = args[10]
        return real_books(*args, **kwargs)

    monkeypatch.setattr(ev, "_run_books_concurrent", _spy_books)
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently",
        lambda *a, **k: (None, None, {}, {}, (), _deployment_readiness()),
    )
    request = MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="1m", log_run=False,
        execution_universe_size=8,
    )
    report = ev.run_mhs_horizon_diagnostic(request)
    assert report.status == "COMPLETE"
    book = report.books["fast_reversal"]
    assert book.primary_autocorr_sharpe is not None
    assert np.isfinite(book.primary_autocorr_sharpe)
    assert book.executed_prescreen_net_t is not None
    assert np.isfinite(book.executed_prescreen_net_t)

    log_close, eligible, execution_mask, _req, _grid, _end = _slow_book_panel_inputs(mhs_market)
    fast = ev.PHASE_1_BOOK_SPECS["fast_reversal"]
    fast_grid = pd.date_range(_START, end, freq="6h", tz="UTC")
    w_fast = ev._book_weights(log_close, eligible, fast, fast_grid)
    ref_execution = ev.renormalize_within_mask(
        ev.inverse_realized_vol_tilt(
            w_fast, ev.realized_vol(log_close, fast.horizon_hours).reindex(fast_grid),
        ),
        execution_mask.reindex(w_fast.index).fillna(False), fast.min_symbols,
    )
    pd.testing.assert_frame_equal(captured["w_fast_execution"], ref_execution)


def test_mhs_fast_book_mode_ensemble_produces_different_executed_book(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_FAST_BOOK_MODE_ENSEMBLE_PRODUCES_DIFFERENT_EXECUTED_BOOK_04:
    # with fast_book_mode='horizon_ensemble' the resulting fast_reversal report
    # carries a DIFFERENT primary_autocorr_sharpe (and different executed
    # prescreen net_t) than the single_horizon default on the same fixture --
    # proving the ensemble branch actually reaches the capital-book construction
    # and RC-1's dual-instrument (executed_prescreen) reflects it automatically.
    # The slow book is untouched by the fast flag. Fails against pre-change
    # code, which has no fast_book_mode branch at all.
    root, end = mhs_market
    monkeypatch.setattr(
        ev, "_run_post_book_concurrently",
        lambda *a, **k: (None, None, {}, {}, (), _deployment_readiness()),
    )
    base = {
        "start": str(_START), "end": str(end), "data_root": str(root),
        "mark_mode": "cache_required", "execution_timeframe": "1m", "log_run": False,
        "execution_universe_size": 8,
    }
    report_default = ev.run_mhs_horizon_diagnostic(MhsDiagnosticRequest(**base))
    report_ensemble = ev.run_mhs_horizon_diagnostic(
        MhsDiagnosticRequest(**base, fast_book_mode="horizon_ensemble"),
    )
    fast_default = report_default.books["fast_reversal"]
    fast_ensemble = report_ensemble.books["fast_reversal"]
    assert fast_default.primary_autocorr_sharpe is not None
    assert fast_ensemble.primary_autocorr_sharpe is not None
    assert fast_default.primary_autocorr_sharpe != fast_ensemble.primary_autocorr_sharpe
    assert fast_default.executed_prescreen_net_t != fast_ensemble.executed_prescreen_net_t
    slow_default = report_default.books["slow_momentum"]
    slow_ensemble = report_ensemble.books["slow_momentum"]
    assert slow_default.primary_autocorr_sharpe == slow_ensemble.primary_autocorr_sharpe
    assert slow_default.executed_prescreen_net_t == slow_ensemble.executed_prescreen_net_t


def test_mhs_funding_carry_top_level_discovery(mhs_market_funding_vary, monkeypatch) -> None:
    # SCENARIO_MHS_FUNDING_CARRY_TOP_LEVEL_DISCOVERY_05: with discovery_gate=True
    # the top-level discovery_qualification carries funding_carry_long and
    # funding_carry_short (each a populated DiscoveryQualificationResult) beside
    # the existing reversal/momentum entries -- all three candidates measured on
    # the same instrumented window. With discovery_gate=False the keys are
    # absent (discovery_qualification stays None), matching the opt-in convention.
    root, end = mhs_market_funding_vary
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
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


def test_mhs_full_history_yearly_net_t_and_worst_year_corr_exposed(mhs_market_funding_vary, monkeypatch) -> None:
    # SCENARIO_MHS_FULL_HISTORY_YEARLY_NET_T_AND_WORST_YEAR_CORR_EXPOSED_06:
    # with discovery_gate=True the report exposes full_history_yearly_net_t for
    # slow_momentum/fast_reversal/funding_carry covering all five years
    # 2021-2025 (not just the 2021-2023 discovery window) and a finite
    # funding_carry_worst_year_corr; both stay None when discovery_gate=False.
    root, end = mhs_market_funding_vary
    monkeypatch.setattr(ev, "_run_books_concurrent", lambda *a, **k: (None, None, None))
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


def _slow_book_panel_inputs(mhs_market):
    """Panel inputs for the ``_horizon_ensemble_execution_weights`` and fold tests."""
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
    """The frozen production slow-book chain ``_horizon_ensemble_execution_weights``
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
    # ``_horizon_ensemble_execution_weights`` reproduces the pre-change
    # ``_book_weights`` + tilt + renormalize sequence exactly.
    log_close, eligible, execution_mask, _request, _grid, end = _slow_book_panel_inputs(mhs_market)
    slow = ev.PHASE_1_BOOK_SPECS["slow_momentum"]
    slow_grid = pd.date_range(_START, end, freq="24h", tz="UTC")
    slow_ema = max(1, round(slow.horizon_hours / slow.step_hours * ev.MHS_SIGNAL_EMA_HORIZON_SPAN))
    expected = _pre_change_slow_book(
        log_close, eligible, execution_mask, slow, slow_grid, slow_ema,
    )
    actual = ev._horizon_ensemble_execution_weights(
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
    actual = ev._horizon_ensemble_execution_weights(
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
        ev._horizon_ensemble_execution_weights(
            log_close, eligible, execution_mask, slow, slow_grid,
            "bogus", "raw", None,
        )
    with pytest.raises(ValueError, match="signal_kind"):
        ev._horizon_ensemble_execution_weights(
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


def _synthetic_ledger(
    freq: str,
    n_bars: int,
    mean_ret: float,
    vol_ret: float,
    seed: int,
) -> SimulatedInventoryLedgerResult:
    idx = pd.date_range("2021-01-01", periods=n_bars, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    rets = rng.normal(mean_ret, vol_ret, n_bars)
    equity = pd.Series(np.cumprod(1.0 + rets), index=idx)
    turnover = pd.Series(np.full(n_bars, 0.01), index=idx)
    return SimulatedInventoryLedgerResult(
        equity=equity,
        net_returns=equity.pct_change().dropna(),
        simulated_units=None,
        mark_to_market_pnl=pd.Series(np.zeros(n_bars), index=idx),
        funding_charge=pd.Series(np.zeros(n_bars), index=idx),
        fee_charge=pd.Series(np.zeros(n_bars), index=idx),
        fill_turnover=turnover,
        fill_source="OHLCV_IMMEDIATE_TAKER",
        mark_source="MARK_PRICE",
        primary_valid=True,
        invalid_reasons=(),
    )


def test_naive_sharpe_uses_hourly_annualization() -> None:
    # SCENARIO_MHS_ANNUALIZATION_01: on a synthetic 5-minute ledger the
    # hourly-resampled naive Sharpe agrees with a calendar-correct manual
    # computation, while the pre-fix computation (sqrt(_PERIODS_PER_YEAR_1H)
    # applied to the raw 5-minute returns) understates it by ~sqrt(12).
    n_years = 3.0
    n_bars = round(365.25 * 288 * n_years)
    ledger = _synthetic_ledger("5min", n_bars, mean_ret=0.0004, vol_ret=0.001, seed=7)
    net_1h = ledger.equity.resample("1h").last().dropna().pct_change().dropna()
    ref = float(net_1h.mean() / net_1h.std(ddof=1) * np.sqrt(ev._PERIODS_PER_YEAR_1H))
    assert ev._naive_sharpe(ledger) == pytest.approx(ref)
    net_5m = ledger.net_returns
    pre_fix = float(net_5m.mean() / net_5m.std(ddof=1) * np.sqrt(ev._PERIODS_PER_YEAR_1H))
    assert ref / pre_fix == pytest.approx(np.sqrt(12.0), rel=0.05)


def test_hourly_ledger_series_hourly_input_is_identity() -> None:
    # SCENARIO_MHS_ANNUALIZATION_02: on an already-hourly ledger the helper is
    # byte-identical up to the leading NaN drop on pct_change, so the existing
    # hourly synthetic-fixture tests keep passing untouched.
    idx = pd.date_range("2021-01-01", periods=48, freq="1h", tz="UTC")
    equity = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
    turnover = pd.Series(np.linspace(0.01, 0.02, len(idx)), index=idx)
    eq_1h, net_1h, turn_1h = ev._hourly_ledger_series(equity, turnover)
    pd.testing.assert_series_equal(eq_1h, equity)
    pd.testing.assert_series_equal(net_1h, equity.pct_change().dropna())
    pd.testing.assert_series_equal(turn_1h, turnover.iloc[1:].rename(None))


def test_hourly_ledger_series_5m_returns_one_row_per_hour() -> None:
    # SCENARIO_MHS_ANNUALIZATION_02 (5-minute leg): one row per calendar hour,
    # turnover is summed (not last-sampled), and the return series drops the
    # leading NaN of the equity pct_change.
    idx = pd.date_range("2021-01-01", periods=72 * 12, freq="5min", tz="UTC")
    equity = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)
    turnover = pd.Series(np.full(len(idx), 0.01), index=idx)
    eq_1h, net_1h, turn_1h = ev._hourly_ledger_series(equity, turnover)
    assert len(eq_1h) == len(net_1h) + 1
    assert len(eq_1h) == 72
    assert (eq_1h.index.minute == 0).all()
    assert (eq_1h.index.second == 0).all()
    pd.testing.assert_series_equal(
        eq_1h, equity.resample("1h").last().dropna(),
    )
    assert turn_1h.index.equals(net_1h.index)
    np.testing.assert_allclose(turn_1h.to_numpy(), np.full(len(turn_1h), 12 * 0.01))


def test_hourly_ledger_series_empty_input_is_empty() -> None:
    # SCENARIO_MHS_ANNUALIZATION_02 (edge case): an empty equity series must
    # raise no exception and return three empty Series, mirroring the caller's
    # empty-input nan convention.
    eq = pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    turn = pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    eq_1h, net_1h, turn_1h = ev._hourly_ledger_series(eq, turn)
    assert eq_1h.empty
    assert net_1h.empty
    assert turn_1h.empty


def test_geometric_cagr_uses_hourly_annualization() -> None:
    # SCENARIO_MHS_ANNUALIZATION_03: on a synthetic 5-minute equity with known
    # total-return ratio R over T years, _geometric_cagr on the C2 hourly series
    # equals R**(1/T)-1, while the raw 5-minute call (the pre-fix path) is off by
    # exactly the 12x (5m) bar-count multiple in the exponent.
    n_years = 3.0
    n_bars = round(365.25 * 288 * n_years)
    idx = pd.date_range("2021-01-01", periods=n_bars, freq="5min", tz="UTC")
    rng = np.random.default_rng(11)
    rets = rng.normal(1.5e-6, 0.001, n_bars)
    equity = pd.Series(np.cumprod(1.0 + rets), index=idx)
    # Flatten the first hour so the hourly resample's opening close equals the
    # raw series' opening close, making the pre/post exponent ratio exactly 12.
    equity.iloc[1:12] = equity.iloc[0]
    turnover = pd.Series(np.zeros(n_bars), index=idx)
    eq_1h, _net_1h, _turn_1h = ev._hourly_ledger_series(equity, turnover)
    # True CAGR over the hourly span: ratio from the first to the last hourly
    # close, spanning (n_hours - 1) hourly intervals (the code annualizes with
    # n_hours bars, an O(1/n) approximation).
    ratio_h = float(eq_1h.iloc[-1] / eq_1h.iloc[0])
    span_years = (len(eq_1h) - 1) / ev._PERIODS_PER_YEAR_1H
    assert ev._geometric_cagr(eq_1h) == pytest.approx(
        ratio_h ** (1.0 / span_years) - 1.0, rel=1e-3,
    )
    post = ev._geometric_cagr(eq_1h)
    pre = ev._geometric_cagr(equity)
    assert np.log1p(post) / np.log1p(pre) == pytest.approx(12.0)


def test_book_outcome_executed_prescreen_reaches_report(mhs_market) -> None:
    """SCENARIO_MHS_EXECUTED_PRESCREEN_REACHES_REPORT_04: when ``_book_outcome``
    is handed an execution book (``replay_weights_step``) different from the
    reference book (``weights_step``), the report carries BOTH the reference
    prescreen and an executed_prescreen/executed_tail computed from the
    capital-carrying book, and their net_t values differ. With no execution
    book the executed fields mirror None. Fails against the pre-change code,
    which could only ever report the reference book."""
    args = _build_book_outcome_args(mhs_market)
    report = ev._book_outcome(**args)
    assert report.prescreen is not None
    assert report.executed_prescreen is not None
    assert report.executed_tail is not None
    base_bps = ev.MEASURED_EXECUTION_COST_TIERS_BPS["base"]
    assert report.executed_prescreen_net_t == report.executed_prescreen[base_bps].net_t
    assert report.prescreen[base_bps].net_t != report.executed_prescreen[base_bps].net_t

    reference_only = ev._book_outcome(**{**args, "replay_weights_step": None})
    assert reference_only.executed_prescreen is None
    assert reference_only.executed_tail is None
    assert reference_only.executed_prescreen_net_t is None


def test_book_outcome_existing_primary_metrics_unchanged(mhs_market) -> None:
    """SCENARIO_MHS_EXISTING_PRIMARY_METRICS_UNCHANGED_05: the executed-evidence
    addition is additive-only -- the primary/stress replay metrics stay present
    and finite, and the reference prescreen/tail remain bit-identical to the
    pre-change inline construction (the regression invariant)."""
    from src.mhs.evaluation import cost_response_curve, tail_sensitivity_curve

    args = _build_book_outcome_args(mhs_market)
    report = ev._book_outcome(**args)
    assert report.primary is not None
    assert report.stress is not None
    assert report.failure is None
    for field in (
        "primary_autocorr_sharpe", "primary_naive_sharpe", "primary_net_ann",
        "primary_geometric_cagr", "primary_max_drawdown",
        "primary_annualized_turnover", "stress_naive_sharpe",
    ):
        value = getattr(report, field)
        assert value is not None
        assert np.isfinite(value)

    weights_1h = args["weights_step"].reindex(args["grid_1h"]).ffill().fillna(0.0)
    cost_grid = tuple(dict.fromkeys((0.0, 2.0, 4.0, 8.0, *ev.required_cost_tiers())))
    expected_prescreen = cost_response_curve(
        weights_1h, args["opens"], args["bar_funding"], cost_grid, ev._PERIODS_PER_YEAR_1H,
    )
    _net, expected_turnover = ev.mhs_ledger_pnl(
        weights_1h, args["opens"], args["bar_funding"], 8.0,
    )
    expected_tail = tail_sensitivity_curve(
        weights_1h.shift(2).fillna(0.0), args["opens"].pct_change(),
        expected_turnover, 8.0, ev._PERIODS_PER_YEAR_1H, args["event_window_bars"],
    )
    assert report.prescreen == expected_prescreen
    assert report.tail == expected_tail


def _passing_fold_report(replay: object) -> ev.MhsFoldReport:
    return ev.MhsFoldReport(
        fold_index=0,
        validation_start="2023-01-08 00:00:00+00:00",
        validation_end="2023-12-31 00:00:00+00:00",
        strict=replay,
        stress=replay,
        primary_valid=True,
        primary_autocorr_sharpe=1.0,
        primary_naive_sharpe=1.0,
        primary_net_ann=0.1,
        primary_geometric_cagr=0.1,
        primary_max_drawdown=-0.02,
        stress_naive_sharpe=0.1,
        decision_intents=1,
        termination_counts={"MISSING_DATA": 0, "UNKNOWN_TERMINATION": 0},
        failures=(),
        strict_elapsed_seconds=0.01,
        stress_elapsed_seconds=0.01,
    )


def test_research_go_eligible_is_reachable(mhs_market, monkeypatch) -> None:
    """SCENARIO_MHS_RESEARCH_GO_ELIGIBLE_IS_REACHABLE_07: with every fold
    passing and every policy threshold registered in
    ``MHS_REGISTERED_POLICY_THRESHOLDS``, ``_mhs_research_go`` returns
    eligible=True with no reason codes -- a result the pre-change code (which
    unconditionally appended UNSPECIFIED_POLICY) could never produce. With a
    threshold missing it still fails closed to UNSPECIFIED_POLICY."""
    idx = pd.date_range("2021-01-01 12:01", periods=31, freq="1min", tz="UTC")
    target = pd.DataFrame({"A": [1.0]}, index=[pd.Timestamp("2021-01-01 11:00", tz="UTC")])
    signal_at = pd.DatetimeIndex([pd.Timestamp("2021-01-01 12:00", tz="UTC")])
    px = pd.DataFrame({"A": [100.0] * 31}, index=idx)
    replay = strategy_aware_execution_replay(
        target, signal_at, px, px, px, px,
        pd.DataFrame(0.0, index=idx, columns=["A"]), 1.0,
        "OHLCV_STRICT_PROXY", ExecutionSpec(),
    )
    passing = _passing_fold_report(replay)

    monkeypatch.setattr(
        ev, "MHS_REGISTERED_POLICY_THRESHOLDS",
        {"cap_30_roster": 30.0, "primary_annual_return": 0.05},
    )
    registered = ev._mhs_research_go((passing,))
    assert registered.eligible is True
    assert registered.reason_codes == ()
    assert registered.folds_passed == 1

    monkeypatch.setattr(
        ev, "MHS_REGISTERED_POLICY_THRESHOLDS",
        {"cap_30_roster": None, "primary_annual_return": 0.05},
    )
    missing = ev._mhs_research_go((passing,))
    assert missing.eligible is False
    assert ev.MHS_GO_REASON_UNSPECIFIED_POLICY in missing.reason_codes


def test_registered_policy_thresholds_contract() -> None:
    """P0-D contract: the two named policy gates exist in source contracts and
    default to the unregistered (conservative) state -- a deliberate policy act,
    never a performance-flattering literal at the call site."""
    from src.mhs.contracts import MHS_REGISTERED_POLICY_THRESHOLDS, MHS_SEARCH_TRIALS_ATTEMPTED

    assert set(MHS_REGISTERED_POLICY_THRESHOLDS) == {"cap_30_roster", "primary_annual_return"}
    assert all(v is None for v in MHS_REGISTERED_POLICY_THRESHOLDS.values())
    assert isinstance(MHS_SEARCH_TRIALS_ATTEMPTED, int)
    assert MHS_SEARCH_TRIALS_ATTEMPTED >= 1

def test_persist_wires_run_history_append_for_compact_and_full(tmp_path, monkeypatch) -> None:
    """SCENARIO_MHS_RESULT_LOG_05: ``persist_mhs_horizon_diagnostic_report``
    calls ``append_run_history_record`` exactly once per COMPACT/FULL tier."""
    report = _build_compact_report()
    calls: list[tuple[str, str]] = []

    def _spy_append(record, history_dir):
        calls.append((record["output_tier"], str(history_dir)))
        return Path(history_dir) / "active.jsonl"

    monkeypatch.setattr(ev, "append_run_history_record", _spy_append)
    out = tmp_path / "mhs_report.json"
    ev.persist_mhs_horizon_diagnostic_report(report, out, tier=ev.MhsOutputTier.COMPACT)
    ev.persist_mhs_horizon_diagnostic_report(report, out, tier=ev.MhsOutputTier.FULL)

    assert len(calls) == 2
    assert [tier for tier, _ in calls] == ["compact", "full"]
    assert all(history_dir.endswith("mhs_run_history") for _, history_dir in calls)


def test_persist_still_appends_when_compact_resample_fails(tmp_path, monkeypatch) -> None:
    """SCENARIO_MHS_RESULT_LOG_05 (COMPACT-None branch): the COMPACT path that
    returns ``None`` (resample failure escalated past artifacts) still appends
    a history record."""
    report = _build_compact_report()
    calls: list = []

    def _boom(_table):
        raise RuntimeError("boom")

    def _spy_append(record, history_dir):
        calls.append(record)
        return Path(history_dir) / "active.jsonl"

    monkeypatch.setattr(ev, "_daily_resample_ledger", _boom)
    monkeypatch.setattr(ev, "append_run_history_record", _spy_append)
    out = tmp_path / "mhs_report.json"
    persisted = ev.persist_mhs_horizon_diagnostic_report(
        report, out, tier=ev.MhsOutputTier.COMPACT,
    )

    assert persisted is None
    assert len(calls) == 1
    assert calls[0]["output_tier"] == "compact"


def test_persist_isolates_history_append_failure(tmp_path, monkeypatch) -> None:
    """SCENARIO_MHS_RESULT_LOG_06: an exception from ``append_run_history_record``
    never propagates and never changes the persist return value."""
    report = _build_compact_report()
    out = tmp_path / "mhs_report.json"

    baseline = ev.persist_mhs_horizon_diagnostic_report(
        report, out, tier=ev.MhsOutputTier.COMPACT,
    )

    def _boom(record, history_dir):
        raise RuntimeError("history boom")

    monkeypatch.setattr(ev, "append_run_history_record", _boom)
    isolated = ev.persist_mhs_horizon_diagnostic_report(
        report, out, tier=ev.MhsOutputTier.COMPACT,
    )

    assert isolated == baseline
