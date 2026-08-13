"""Contract tests for MHS execution-performance optimizations (C1-C4).

Each ``SCENARIO_MHS_PERF_OPT_*`` test pins the bit-identical-equivalence
invariant of an optimization against the pre-optimization code path:

- ``MARK_PANEL_EQUIVALENCE``: ``_cached_mark_panel`` reproduces
  ``DataCollector().load_mark_price_panel`` element-for-element.
- ``WINDOW_SLICE_EQUIVALENCE``: ``_load_window_minute_frames`` reproduces the
  full-period-frame ``.loc`` slice byte-identically.
- ``WINDOW_REUSE_EQUIVALENCE``: a materialized window tuple + per-pass weight
  substitution reproduces per-pass generator regeneration.
- ``LAZY_FRAME_SCOPE``: the window generator reads only its roster symbols'
  window slices, never a full-period preload.
- ``FOLD_DISCOVERY_PARALLEL_EQUIVALENCE``: forked fold-safe discovery equals
  the sequential per-fold computation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.market_data.services.futures_collection as fc
from src.application.research.mhs import evaluation as ev
from src.application.research.mhs.evaluation import (
    _cached_mark_panel,
    _fold_safe_discovery_worker,
    _get_symbol_minute_frame,
    _iter_mhs_execution_windows,
    _load_window_minute_frames,
    _materialize_replay_windows,
    _precompute_fold_safe_candidate_weights,
    _rescaled_windows,
    _run_fold_safe_discovery_parallel,
)
from src.common.errors import DataIntegrityError
from src.market_data.services.futures_collection import DataCollector
from src.mhs.contracts import PHASE_1_BOOK_SPECS, ExecutionSpec
from src.mhs.evaluation import phase_1_anchored_purged_folds
from src.mhs.execution import replay_execution_windows

_START = pd.Timestamp("2021-01-01", tz="UTC")
_SYMBOLS = ["MHSAUSDT", "MHSBUSDT", "MHSCUSDT"]


@pytest.fixture(autouse=True)
def _clear_perf_caches() -> None:
    ev._get_symbol_mark_frame.cache_clear()
    ev._get_symbol_minute_frame.cache_clear()
    ev._load_minute_frames_cached.cache_clear()
    yield
    ev._get_symbol_mark_frame.cache_clear()


def _write_mark_market(
    root: Path,
    symbols: list[str],
    n_hours: int = 96,
) -> None:
    """1h mark + 5m OHLCV + 1h funding synthetic market (MHS convention)."""
    hourly = pd.date_range(_START, periods=n_hours, freq="1h", tz="UTC")
    minute = pd.date_range(_START, _START + pd.Timedelta(hours=n_hours - 1), freq="5min", tz="UTC")
    rng = np.random.default_rng(20260807)
    epoch_h = (hourly - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    epoch_m = (minute - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms")
    for d in (root / "5m", root / "1m", root / "1h", root / "funding", root / "markPriceKlines" / "1h"):
        d.mkdir(parents=True, exist_ok=True)
    for i, sym in enumerate(symbols):
        drift = 1e-5 * (i - len(symbols) / 2.0)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.002, n_hours)))
        mp = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.002, len(minute))))
        five = (
            pd.Series(mp, index=minute).resample("5min").last().dropna()
        )
        pd.DataFrame(
            {"timestamp": epoch_h, "open": prices, "high": prices * 1.001,
             "low": prices * 0.999, "close": prices, "quote_vol": [1000.0] * n_hours},
        ).to_parquet(root / "1h" / f"{sym}.parquet")
        pd.DataFrame(
            {"timestamp": epoch_m[: len(five)], "open": five.to_numpy(),
             "high": five.to_numpy() * 1.0005, "low": five.to_numpy() * 0.9995,
             "close": five.to_numpy(), "quote_vol": [1000.0] * len(five)},
        ).to_parquet(root / "5m" / f"{sym}.parquet")
        pd.DataFrame(
            {"timestamp": epoch_h, "funding_rate": [0.00005] * n_hours, "datetime": hourly},
        ).to_parquet(root / "funding" / f"{sym}.parquet")
        mark = (
            pd.Series(mp, index=minute).resample("1h").last().reindex(hourly).to_numpy()
        )
        pd.DataFrame(
            {"timestamp": epoch_h, "open": mark, "high": mark, "low": mark,
             "close": mark, "datetime": hourly},
        ).to_parquet(root / "markPriceKlines" / "1h" / f"{sym}.parquet")


@pytest.fixture
def mark_market(tmp_path, monkeypatch):
    root = tmp_path / "market"
    _write_mark_market(root, _SYMBOLS)
    monkeypatch.setattr(
        fc, "_mark_price_path",
        lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet",
    )
    monkeypatch.setattr(
        ev, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet",
    )
    return root


def _assert_panel_equal(a: pd.DataFrame, b: pd.DataFrame) -> None:
    assert a.index.equals(b.index)
    assert list(a.columns) == list(b.columns)
    assert a.dtypes.equals(b.dtypes)
    assert np.array_equal(a.to_numpy(dtype="float64"), b.to_numpy(dtype="float64"), equal_nan=True)


def test_mhs_perf_opt_mark_panel_equivalence(mark_market) -> None:
    """SCENARIO_MHS_PERF_OPT_MARK_PANEL_EQUIVALENCE: ``_cached_mark_panel`` is
    byte-identical to the DataCollector mark panel on 5m and 1m grids for both
    strict and stale-carry modes."""
    collector = DataCollector()
    grid_5m = pd.date_range(_START, _START + pd.Timedelta(hours=47), freq="5min", tz="UTC")
    grid_1m = pd.date_range(_START, _START + pd.Timedelta(hours=11), freq="1min", tz="UTC")
    for stale in (0, 24):
        for grid in (grid_5m, grid_1m):
            expected = collector.load_mark_price_panel(_SYMBOLS, "1h", grid, max_stale_hours=stale)
            actual = _cached_mark_panel(_SYMBOLS, "1h", grid, stale)
            _assert_panel_equal(actual, expected)


def test_mhs_perf_opt_mark_cache_read_once(mark_market) -> None:
    """The per-process mark cache reads each symbol's parquet exactly once."""
    ev._get_symbol_mark_frame.cache_clear()
    grid = pd.date_range(_START, _START + pd.Timedelta(hours=47), freq="5min", tz="UTC")
    _cached_mark_panel(_SYMBOLS, "1h", grid, 0)
    info = ev._get_symbol_mark_frame.cache_info()
    assert info.hits == 0
    assert info.misses == len(_SYMBOLS)
    _cached_mark_panel(_SYMBOLS, "1h", grid, 0)
    assert ev._get_symbol_mark_frame.cache_info().hits == len(_SYMBOLS)


def test_mhs_perf_opt_window_slice_equivalence(mark_market) -> None:
    """SCENARIO_MHS_PERF_OPT_WINDOW_SLICE_EQUIVALENCE: the window-filtered
    loader returns exactly the full-period-frame ``.loc`` slice."""
    ev._get_symbol_minute_frame.cache_clear()
    root = str(mark_market)
    ws = _START + pd.Timedelta(hours=24)
    we = _START + pd.Timedelta(hours=72)
    windowed = _load_window_minute_frames(root, _SYMBOLS, ws, we, "5m")
    assert set(windowed) == set(_SYMBOLS)
    for sym in _SYMBOLS:
        full = _get_symbol_minute_frame(root, sym, "5m")
        expected = full.loc[(full.index >= ws) & (full.index <= we)]
        pd.testing.assert_frame_equal(windowed[sym], expected)


def _build_small_funding(root: Path) -> dict[str, pd.Series]:
    funding: dict[str, pd.Series] = {}
    for sym in _SYMBOLS:
        df = pd.read_parquet(root / "funding" / f"{sym}.parquet")
        funding[sym] = pd.Series(
            df["funding_rate"].to_numpy(),
            index=pd.to_datetime(df["datetime"], utc=True),
            name="funding_rate",
        )
    return funding


def test_mhs_perf_opt_window_reuse_equivalence(mark_market) -> None:
    """SCENARIO_MHS_PERF_OPT_WINDOW_REUSE_EQUIVALENCE: materialized windows fed
    through ``_rescaled_windows`` reproduce per-pass generator regeneration."""
    root = str(mark_market)
    end = _START + pd.Timedelta(hours=48)
    decision_grid = pd.date_range(_START + pd.Timedelta(hours=1), end, freq="6h", tz="UTC")
    rng = np.random.default_rng(11)
    target = pd.DataFrame(0.0, index=decision_grid, columns=_SYMBOLS)
    for i, ts in enumerate(decision_grid):
        for sym in _SYMBOLS[: 1 + (i % 2)]:
            target.loc[ts, sym] = 0.05 + 0.01 * (i % 3)
    signals = decision_grid + pd.Timedelta(hours=1)
    funding = _build_small_funding(mark_market)

    def gen(t: pd.DataFrame):
        return _iter_mhs_execution_windows(
            t, signals, root, "5m", _START, end, funding, "cache_required", ExecutionSpec(),
        )

    spec = ExecutionSpec()

    def replay_pass(windows, scale):
        stream = _rescaled_windows(windows, scale)
        return replay_execution_windows(stream, 1.0, "OHLCV_IMMEDIATE_TAKER", spec)

    # Legacy: fresh generator per pass; pass B rescales the target DataFrame.
    legacy_a = replay_execution_windows(gen(target), 1.0, "OHLCV_IMMEDIATE_TAKER", spec)
    scale = pd.Series(
        0.5 + 0.5 * np.linspace(0.0, 1.0, len(target)), index=target.index,
    )
    scaled = target.mul(scale, axis=0)
    legacy_b = replay_execution_windows(gen(scaled), 1.0, "OHLCV_IMMEDIATE_TAKER", spec)

    # New: materialize once from the unscaled target; substitute per pass.
    windows = _materialize_replay_windows(lambda: gen(target))
    new_a = replay_execution_windows(iter(windows), 1.0, "OHLCV_IMMEDIATE_TAKER", spec)
    new_b = replay_pass(windows, scale)

    for legacy, new in ((legacy_a, new_a), (legacy_b, new_b)):
        assert len(legacy.simulated_fills) == len(new.simulated_fills)
        assert dict(legacy.termination_counts) == dict(new.termination_counts)
        assert np.allclose(
            legacy.ledger.equity.to_numpy(), new.ledger.equity.to_numpy(),
            rtol=1e-12, atol=1e-12,
        )


def test_mhs_perf_opt_lazy_frame_scope(mark_market, monkeypatch) -> None:
    """SCENARIO_MHS_PERF_OPT_LAZY_FRAME_SCOPE: ``_load_window_minute_frames``
    reads only the requested roster's window rows via parquet filters and never
    triggers a full-period frame load."""
    ev._get_symbol_minute_frame.cache_clear()
    root = str(mark_market)
    ws = _START + pd.Timedelta(hours=24)
    we = _START + pd.Timedelta(hours=48)
    calls: list[list[tuple[str, object]]] = []

    real_read_table = ev.pq.read_table

    def counting_read_table(*args, **kwargs):
        calls.append(list(kwargs.get("filters", []) or []))
        return real_read_table(*args, **kwargs)

    monkeypatch.setattr(ev.pq, "read_table", counting_read_table)

    def _forbidden(_root, _sym, _tf):
        raise AssertionError("full-period minute frame must not be loaded in the window path")

    monkeypatch.setattr(ev, "_get_symbol_minute_frame", _forbidden)
    frames = _load_window_minute_frames(root, _SYMBOLS, ws, we, "5m")
    assert set(frames) == set(_SYMBOLS)
    assert len(calls) == len(_SYMBOLS)
    for filt in calls:
        and_clause = filt[0]
        ops = [c[1] for c in and_clause]
        assert ">=" in ops
        assert "<=" in ops
        assert all(c[2] is not None for c in and_clause)
    start_ms = int(ws.value // 1_000_000)
    end_ms = int(we.value // 1_000_000)
    for frame in frames.values():
        assert frame.index[0] >= ws
        assert frame.index[-1] <= we
        assert frame.index.min().value // 1_000_000 >= start_ms
        assert frame.index.max().value // 1_000_000 <= end_ms


def _build_fold_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    symbols = ["MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT"]
    grid = pd.date_range("2021-01-01", "2023-06-30", freq="1h", tz="UTC")
    rng = np.random.default_rng(20260807)
    n = len(grid)
    rets = rng.normal(1e-5, 0.002, (n, len(symbols)))
    log_close = pd.DataFrame(np.cumsum(rets, axis=0), index=grid, columns=symbols)
    opens = pd.DataFrame(100.0 * np.exp(log_close.to_numpy()), index=grid, columns=symbols)
    bar_funding = pd.DataFrame(1.0e-5, index=grid, columns=symbols)
    eligible = pd.DataFrame(True, index=grid, columns=symbols)
    return log_close, eligible, opens, bar_funding, grid


def test_mhs_perf_opt_fold_discovery_parallel_equivalence() -> None:
    """SCENARIO_MHS_PERF_OPT_FOLD_DISCOVERY_PARALLEL_EQUIVALENCE: the forked
    fold-safe discovery returns exactly the sequential per-fold computation."""
    log_close, eligible, opens, bar_funding, grid = _build_fold_panel()
    specs = PHASE_1_BOOK_SPECS
    precomputed = _precompute_fold_safe_candidate_weights(specs, log_close, eligible, bar_funding)

    expected_slow: dict[int, int | None] = {}
    expected_fast: dict[int, tuple[int, str]] = {}
    expected_fc: dict[int, tuple[int | None, int | None, str, float | None]] = {}
    for idx, fold in enumerate(phase_1_anchored_purged_folds()):
        slow, fast, fc = _fold_safe_discovery_worker(
            fold, idx, specs, log_close, eligible, opens, bar_funding, grid, precomputed,
        )
        expected_slow[idx] = slow
        expected_fast[idx] = fast
        expected_fc[idx] = fc

    slow, fast, fc = _run_fold_safe_discovery_parallel(
        specs, log_close, eligible, opens, bar_funding, grid,
    )
    assert slow == expected_slow
    assert fast == expected_fast
    assert fc == expected_fc


def test_mhs_perf_opt_rescaled_windows_guards_zero_pattern(mark_market) -> None:
    """A scale that would erase a held position must fail closed (roster drift
    across passes is a correctness breach of window reuse)."""
    end = _START + pd.Timedelta(hours=48)
    decision_grid = pd.date_range(_START + pd.Timedelta(hours=1), end, freq="6h", tz="UTC")
    target = pd.DataFrame(0.0, index=decision_grid, columns=_SYMBOLS)
    target.loc[decision_grid[0], "MHSAUSDT"] = 0.05
    signals = decision_grid + pd.Timedelta(hours=1)
    funding = _build_small_funding(mark_market)
    windows = _materialize_replay_windows(
        lambda: _iter_mhs_execution_windows(
            target, signals, str(mark_market), "5m", _START, end,
            funding, "cache_required", ExecutionSpec(),
        ),
    )
    zero_scale = pd.Series(0.0, index=target.index)
    with pytest.raises(DataIntegrityError):
        next(_rescaled_windows(windows, zero_scale))


def test_mhs_perf_opt_mark_panel_invalid_grid(mark_market) -> None:
    """Invalid grid/symbol inputs fail closed exactly like the DataCollector."""
    grid = pd.date_range(_START, _START + pd.Timedelta(hours=47), freq="5min", tz="UTC")
    with pytest.raises(ValueError, match="unsupported timeframe"):
        _cached_mark_panel(_SYMBOLS, "2h", grid, 0)
    with pytest.raises(DataIntegrityError):
        _cached_mark_panel([], "1h", grid, 0)
    with pytest.raises(DataIntegrityError):
        _cached_mark_panel(["MHSAUSDT", "MHSAUSDT"], "1h", grid, 0)
    with pytest.raises(DataIntegrityError):
        _cached_mark_panel(_SYMBOLS, "1h", grid.tz_localize(None), 0)
    with pytest.raises(DataIntegrityError):
        _cached_mark_panel(_SYMBOLS, "1h", grid[::-1], 0)
