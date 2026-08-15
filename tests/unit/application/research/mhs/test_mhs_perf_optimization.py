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
    _candidate_weight_books,
    _fold_safe_discovery_worker,
    _iter_mhs_execution_windows,
    _load_window_minute_frames,
    _materialize_replay_windows,
    _rescaled_windows,
    _run_fold_safe_discovery_parallel,
)
from src.common.errors import DataIntegrityError
from src.market_data.services.futures_collection import DataCollector
from src.mhs.contracts import PHASE_1_BOOK_SPECS, ExecutionSpec
from src.mhs.evaluation import phase_1_anchored_purged_folds
from src.mhs.execution import replay_execution_windows
from src.mhs.parallel import fork_shared_payload

_START = pd.Timestamp("2021-01-01", tz="UTC")
_SYMBOLS = ["MHSAUSDT", "MHSBUSDT", "MHSCUSDT"]


@pytest.fixture(autouse=True)
def _clear_perf_caches() -> None:
    ev._get_symbol_mark_frame.cache_clear()
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
    root = str(mark_market)
    ws = _START + pd.Timedelta(hours=24)
    we = _START + pd.Timedelta(hours=72)
    windowed = _load_window_minute_frames(root, _SYMBOLS, ws, we, "5m")
    assert set(windowed) == set(_SYMBOLS)
    for sym in _SYMBOLS:
        table = ev.pq.read_table(
            f"{root}/5m/{sym}.parquet", columns=["timestamp", "high", "low", "close"],
        )
        idx = pd.to_datetime(table.column("timestamp").to_numpy(), unit="ms", utc=True)
        full = pd.DataFrame(
            {c: table.column(c).to_numpy().astype("float64") for c in ("high", "low", "close")},
            index=idx,
        )
        full = full[~full.index.duplicated(keep="last")].sort_index()
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
    root = str(mark_market)
    ws = _START + pd.Timedelta(hours=24)
    we = _START + pd.Timedelta(hours=48)
    calls: list[list[tuple[str, object]]] = []

    real_read_table = ev.pq.read_table

    def counting_read_table(*args, **kwargs):
        calls.append(list(kwargs.get("filters", []) or []))
        return real_read_table(*args, **kwargs)

    monkeypatch.setattr(ev.pq, "read_table", counting_read_table)
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
    precomputed = _candidate_weight_books(log_close, eligible, bar_funding, specs)

    expected_slow: dict[int, int | None] = {}
    expected_fast: dict[int, tuple[int, str]] = {}
    expected_fc: dict[int, tuple[int | None, int | None, str, float | None]] = {}
    with fork_shared_payload({
        "specs": specs, "log_close": log_close, "eligible": eligible,
        "opens": opens, "bar_funding": bar_funding, "grid_1h": grid,
        "precomputed": precomputed,
    }) as token:
        for idx, fold in enumerate(phase_1_anchored_purged_folds()):
            slow, fast, fc = _fold_safe_discovery_worker(fold, idx, token)
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


def test_scenario_04_candidate_weight_books_covers_union() -> None:
    """SCENARIO_MHS_REFACTOR_04: ``_candidate_weight_books`` returns horizon
    keys covering both the fold-safe BookSpec band horizons and the top-level
    MHS_DISCOVERY_* / funding-carry candidate sets, and every panel equals what
    ``build_candidate_weights`` would have produced for that key."""
    from src.application.research.mhs.evaluation import (
        MHS_DISCOVERY_MOMENTUM_CANDIDATES,
        MHS_DISCOVERY_REVERSAL_CANDIDATES,
        MHS_FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
        build_funding_carry_candidate_weights,
    )
    from src.mhs.discovery import build_candidate_weights as _bcw

    log_close, eligible, opens, bar_funding, grid = _build_fold_panel()
    specs = PHASE_1_BOOK_SPECS
    books = _candidate_weight_books(log_close, eligible, bar_funding, specs)
    assert set(books) == {"slow", "fast", "funding_long", "funding_short"}

    slow_keys = set(books["slow"])
    assert set(specs["slow_momentum"].band.horizons_hours) <= slow_keys
    assert set(MHS_DISCOVERY_MOMENTUM_CANDIDATES) <= slow_keys

    fast_keys = set(books["fast"])
    assert set(specs["fast_reversal"].band.horizons_hours) <= fast_keys
    assert set(MHS_DISCOVERY_REVERSAL_CANDIDATES) <= fast_keys

    funding_keys = set(books["funding_long"])
    assert set(MHS_FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS) <= funding_keys

    for h in slow_keys:
        expected = _bcw(log_close, eligible, 1, (h,), tranche_count=8)[h]
        pd.testing.assert_frame_equal(books["slow"][h], expected)
    for h in fast_keys:
        expected = _bcw(log_close, eligible, -1, (h,), tranche_count=8)[h]
        pd.testing.assert_frame_equal(books["fast"][h], expected)
    for h in funding_keys:
        expected = build_funding_carry_candidate_weights(
            bar_funding, eligible, 1, (h,), tranche_count=8,
        )[h]
        pd.testing.assert_frame_equal(books["funding_long"][h], expected)


def test_scenario_05_materialize_replay_windows_budget(mark_market) -> None:
    """SCENARIO_MHS_REFACTOR_05: unbounded materializes every window; a budget
    smaller than one window still returns at least one window; and the
    budgeted path replays to the same ledger as the unbounded path."""
    root = str(mark_market)
    end = _START + pd.Timedelta(hours=48)
    decision_grid = pd.date_range(_START + pd.Timedelta(hours=1), end, freq="6h", tz="UTC")
    target = pd.DataFrame(0.05, index=decision_grid, columns=_SYMBOLS)
    signals = decision_grid + pd.Timedelta(hours=1)
    funding = _build_small_funding(mark_market)
    spec = ExecutionSpec()

    def gen():
        return _iter_mhs_execution_windows(
            target, signals, root, "5m", _START, end, funding, "cache_required", spec,
        )

    all_windows = _materialize_replay_windows(gen)
    assert len(all_windows) >= 1
    bounded = _materialize_replay_windows(gen, budget_bytes=1)
    assert len(bounded) >= 1

    big_budget = sum(ev._window_footprint_bytes(w) for w in all_windows)
    bounded_full = _materialize_replay_windows(gen, budget_bytes=big_budget)
    assert len(bounded_full) == len(all_windows)

    full_ledger = replay_execution_windows(iter(all_windows), 1.0, "OHLCV_IMMEDIATE_TAKER", spec)
    budget_ledger = replay_execution_windows(iter(bounded_full), 1.0, "OHLCV_IMMEDIATE_TAKER", spec)
    assert len(full_ledger.simulated_fills) == len(budget_ledger.simulated_fills)
    assert np.allclose(
        full_ledger.ledger.equity.to_numpy(), budget_ledger.ledger.equity.to_numpy(),
        rtol=1e-12, atol=1e-12,
    )


def test_scenario_06_no_dataframe_in_submit_args(tmp_path, monkeypatch) -> None:
    """SCENARIO_MHS_REFACTOR_06: no ProcessPoolExecutor.submit call in
    evaluation.py passes a pd.DataFrame or pd.Series argument; large read-only
    panels travel through fork_shared_payload tokens."""
    import src.application.research.mhs.evaluation as ev_mod

    root = tmp_path / "market"
    _write_mark_market(root, _SYMBOLS, n_hours=2700)
    monkeypatch.setattr(
        fc, "_mark_price_path",
        lambda symbol, timeframe: root / "markPriceKlines" / timeframe / f"{symbol}.parquet",
    )
    monkeypatch.setattr(
        ev, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet",
    )

    recorded: list[list[object]] = []

    class _SynchronousFuture:
        def __init__(self, fn, args):
            self._result = fn(*args)

        def result(self, timeout=None):
            return self._result

    class _RecordingExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, *args, **kwargs):
            recorded.append(list(args))
            return _SynchronousFuture(fn, args)

    monkeypatch.setattr(ev_mod, "ProcessPoolExecutor", _RecordingExecutor)

    from src.application.research.mhs import evaluation as _ev
    args = _build_books_args_from_market(root, 2700)
    _ev._run_books_concurrent(**args)

    assert recorded, "the book pool must submit at least once"
    for submit_args in recorded:
        for arg in submit_args:
            assert not isinstance(arg, (pd.DataFrame, pd.Series))


def _build_books_args_from_market(root: Path, n_hours: int) -> dict[str, object]:
    """Minimal ``_run_books_concurrent`` arg set from a written market."""
    import src.application.research.mhs.evaluation as ev_mod

    end = _START + pd.Timedelta(hours=n_hours)
    symbols = _SYMBOLS
    funding_by_symbol = _build_small_funding(root)
    request = ev_mod.MhsDiagnosticRequest(
        start=str(_START), end=str(end), data_root=str(root),
        mark_mode="cache_required", execution_timeframe="5m", log_run=False,
        execution_universe_size=8,
    )
    panel = ev_mod.load_base_panel(
        str(root), "1h", ("close", "open", "quote_vol"), _START, end,
        partition="dev", min_bars=2000,
    )
    close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    grid_1h = close.index
    bar_period = grid_1h[1] - grid_1h[0]
    funding_window = {
        s: funding_by_symbol[s].loc[
            (funding_by_symbol[s].index >= grid_1h[0])
            & (funding_by_symbol[s].index < grid_1h[-1] + bar_period)
        ]
        for s in symbols
        if s in funding_by_symbol
    }
    bar_funding = ev_mod.bar_funding_panel(funding_window, grid_1h)
    aligned = list(bar_funding.columns)
    close = close[aligned]
    opens = opens[aligned]
    bar_funding = bar_funding[aligned]
    quote_vol = quote_vol[aligned]
    funding_by_symbol = {s: funding_by_symbol[s] for s in aligned}
    eligible = ev_mod.liquid_half_eligibility(
        quote_vol, lookback_bars=720, min_history_bars=720,
    )
    log_close = np.log(close)
    fast = ev_mod.PHASE_1_BOOK_SPECS["fast_reversal"]
    slow = ev_mod.PHASE_1_BOOK_SPECS["slow_momentum"]
    fast_grid = pd.date_range(_START, end, freq="6h", tz="UTC")
    slow_grid = pd.date_range(_START, end, freq="24h", tz="UTC")
    w_fast = ev_mod._book_weights(log_close, eligible, fast, fast_grid)
    w_slow = ev_mod._book_weights(log_close, eligible, slow, slow_grid)
    phase_fast = ev_mod._phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, fast)
    phase_slow = ev_mod._phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, slow)
    phase_blend = ev_mod._phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, fast)
    execution_mask = ev_mod._pit_execution_mask(quote_vol, eligible, 8)
    w_fast_execution = ev_mod.renormalize_within_mask(
        w_fast, execution_mask.reindex(w_fast.index).fillna(False), fast.min_symbols,
    )
    w_slow_execution = ev_mod.renormalize_within_mask(
        w_slow, execution_mask.reindex(w_slow.index).fillna(False), slow.min_symbols,
    )
    w_fast_1h = w_fast.reindex(grid_1h).ffill().fillna(0.0)
    w_slow_1h = w_slow.reindex(grid_1h).ffill().fillna(0.0)
    blend_1h = (
        ev_mod.PHASE_1_BOOK_BLEND_WEIGHTS["fast_reversal"] * w_fast_1h
        + ev_mod.PHASE_1_BOOK_BLEND_WEIGHTS["slow_momentum"] * w_slow_1h
    )
    vol_mean = ev_mod.realized_vol(log_close, 48).where(execution_mask).reindex(grid_1h).mean(axis=1)
    regime_scale = ev_mod._regime_cash_scale(vol_mean)
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
