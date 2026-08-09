"""Contract coverage for the MHS application evaluation resource telemetry."""

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
from src.mhs.contracts import ExecutionSpec
from src.mhs.execution import ExecutionReplayWindow, replay_execution_windows
from src.mhs.evaluation import AnchoredPurgedFold
from src.research.universe.pit_universe import symbol_partition

_START = pd.Timestamp("2021-01-01", tz="UTC")


def _write_mhs_market(root: Path, n_hours: int = 2700) -> pd.Timestamp:
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ][:8]
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
        assert window_stages == sorted(window_stages)
        # The paired fan-out records one physical window per stage: the stress
        # bound consumes the same iterator, so no separate stress re-iteration
        # telemetry exists.
        assert not [
            m.stage for m in recorder.records
            if m.stage.startswith("anchored_fold_0_stress_window_")
        ]

    def test_fold_builds_window_iterator_once_per_pair(self, mhs_market, monkeypatch) -> None:
        """MHS-MEM-PAIR-02: the fold orchestrator constructs its execution-window
        iterator exactly once and feeds it to the paired strict/stress replay."""
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
    w_fast_execution = w_fast.where(
        execution_mask.reindex(w_fast.index).fillna(False), other=0.0,
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


class TestBookOutcomePaired:
    """MHS-MEM-PAIR-02-BOOK: the top-level book orchestrator builds the
    execution-window iterator once per strict/stress pair and preserves the
    typed book failure conversion."""

    def test_book_builds_window_iterator_once_per_pair(self, mhs_market, monkeypatch) -> None:
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


def test_xs_rank_ic_vectorized_contract_ignores_invalid_cross_section_cells() -> None:
    index = pd.date_range("2021-01-01", periods=3, freq="1h", tz="UTC")
    signal = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, np.nan, 4.0, 5.0], [1.0] * 5],
        index=index,
        columns=list("ABCDE"),
    )
    forward = pd.DataFrame(
        [[5.0, 4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0], [1.0] * 5],
        index=index,
        columns=list("ABCDE"),
    )

    result = ev._xs_rank_ic(signal, forward)

    assert result["n_dates"] == 1
    assert result["mean_ic"] == pytest.approx(-1.0)


def test_date_clustered_ols_vectorized_contract() -> None:
    index = pd.date_range("2021-01-01", periods=48, freq="1h", tz="UTC")
    past = pd.DataFrame(
        {"A": np.arange(48, dtype=float), "B": np.arange(48, dtype=float) + 2.0},
        index=index,
    )
    forward = 0.25 + 1.5 * past
    forward.loc[index[3], "A"] = np.nan

    result = ev._date_clustered_ols(forward, past)

    assert result["n"] == 95
    assert result["n_dates"] == 2
    assert result["past_beta"] == pytest.approx(1.5)
