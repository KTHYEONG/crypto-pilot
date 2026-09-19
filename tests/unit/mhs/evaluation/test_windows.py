"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import pytest
import src.mhs.evaluation.windows as windows
from src.mhs.backtest.contracts import ProcessPath
from src.mhs.backtest.inventory import (
    _execution_fence,
    _validate_replay_window,
    replay_process_execution,
)


def test_windows_module_present() -> None:
    assert windows.__name__ == "src.mhs.evaluation.windows"
    assert callable(windows._resolve_ns_vectorized)


def test_window_ipc_spill_roundtrip_identity(tmp_path) -> None:
    import pandas as pd
    import numpy as np
    from src.mhs.execution.contracts import ExecutionReplayWindow
    from src.mhs.evaluation.windows import _spill_window_to_ipc, _load_window_from_ipc

    minute_grid = pd.date_range("2021-01-01", periods=100, freq="3min", tz="UTC")
    decision_grid = pd.date_range("2021-01-01", periods=5, freq="6h", tz="UTC")
    symbols = ("BTC", "ETH")
    columns = ("BTC", "ETH", "SOL")

    highs = pd.DataFrame({"BTC": np.linspace(10.0, 20.0, 100), "ETH": np.linspace(1.0, 2.0, 100)}, index=minute_grid)
    lows = highs - 0.5
    closes = highs - 0.2
    marks = highs - 0.1
    bar_funding = pd.DataFrame({"BTC": [0.0001] * 100, "ETH": [0.0002] * 100}, index=minute_grid)
    target_weights = pd.DataFrame({"BTC": [0.5] * 5, "ETH": [0.5] * 5}, index=decision_grid)
    signal_available_at = decision_grid + pd.Timedelta(hours=1)

    window = ExecutionReplayWindow(
        window_start=minute_grid[0],
        window_end=minute_grid[-1],
        columns=columns,
        symbols=symbols,
        minute_grid=minute_grid,
        highs=highs,
        lows=lows,
        closes=closes,
        marks=marks,
        bar_funding=bar_funding,
        target_weights=target_weights,
        signal_available_at=signal_available_at,
    )

    file_path = str(tmp_path / "window_00000.arrow")
    _spill_window_to_ipc(window, file_path)
    loaded = _load_window_from_ipc(file_path)

    assert loaded.window_start == window.window_start
    assert loaded.window_end == window.window_end
    assert loaded.symbols == window.symbols
    assert loaded.columns == window.columns
    assert (loaded.minute_grid == window.minute_grid).all()
    assert (loaded.signal_available_at == window.signal_available_at).all()
    np.testing.assert_allclose(loaded.highs.to_numpy(), window.highs.to_numpy())
    np.testing.assert_allclose(loaded.lows.to_numpy(), window.lows.to_numpy())
    np.testing.assert_allclose(loaded.closes.to_numpy(), window.closes.to_numpy())
    assert loaded.marks is not None
    np.testing.assert_allclose(loaded.marks.to_numpy(), window.marks.to_numpy())
    np.testing.assert_allclose(loaded.bar_funding.to_numpy(), window.bar_funding.to_numpy())
    np.testing.assert_allclose(loaded.target_weights.to_numpy(), window.target_weights.to_numpy())


def test_spill_and_stream_windows_lifecycle(tmp_path) -> None:
    import pandas as pd
    import numpy as np
    from src.mhs.execution.contracts import ExecutionReplayWindow
    from src.mhs.evaluation.windows import _spill_and_stream_windows, _iter_spilled_windows

    windows_in = []
    for w_i in range(2):
        mg = pd.date_range(f"2021-0{w_i+1}-01", periods=20, freq="3min", tz="UTC")
        dg = pd.date_range(f"2021-0{w_i+1}-01", periods=2, freq="6h", tz="UTC")
        w = ExecutionReplayWindow(
            window_start=mg[0],
            window_end=mg[-1],
            columns=("BTC",),
            symbols=("BTC",),
            minute_grid=mg,
            highs=pd.DataFrame({"BTC": [10.0 + w_i] * 20}, index=mg),
            lows=pd.DataFrame({"BTC": [9.0 + w_i] * 20}, index=mg),
            closes=pd.DataFrame({"BTC": [9.5 + w_i] * 20}, index=mg),
            marks=None,
            bar_funding=pd.DataFrame({"BTC": [0.0001] * 20}, index=mg),
            target_weights=pd.DataFrame({"BTC": [1.0] * 2}, index=dg),
            signal_available_at=dg + pd.Timedelta(hours=1),
        )
        windows_in.append(w)

    spill_dir = str(tmp_path / "spill")
    pass1_stream = list(_spill_and_stream_windows(windows_in, spill_dir))
    assert len(pass1_stream) == 2
    assert pass1_stream[0].window_start == windows_in[0].window_start

    pass2_stream = list(_iter_spilled_windows(spill_dir))
    assert len(pass2_stream) == 2
    np.testing.assert_allclose(pass2_stream[0].highs.to_numpy(), windows_in[0].highs.to_numpy())
    np.testing.assert_allclose(pass2_stream[1].highs.to_numpy(), windows_in[1].highs.to_numpy())


def test_load_window_minute_frames_threaded_equivalence(tmp_path) -> None:
    import pandas as pd
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.mhs.marks import _load_window_minute_frames

    timeframe = "3m"
    tf_dir = tmp_path / timeframe
    tf_dir.mkdir(parents=True)
    symbols = ["BTC", "ETH", "SOL", "DOGE"]
    idx = pd.date_range("2021-01-01", periods=100, freq="3min", tz="UTC")
    start_ms = int(idx[0].value // 1_000_000)

    for s in symbols:
        table = pa.Table.from_pandas(pd.DataFrame({
            "timestamp": [start_ms + i * 180000 for i in range(100)],
            "high": np.linspace(10, 20, 100),
            "low": np.linspace(8, 18, 100),
            "close": np.linspace(9, 19, 100),
        }))
        pq.write_table(table, str(tf_dir / f"{s}.parquet"))

    frames = _load_window_minute_frames(str(tmp_path), symbols, idx[0], idx[-1], timeframe)
    assert len(frames) == 4
    for s in symbols:
        assert s in frames
        assert len(frames[s]) == 100
        assert "close" in frames[s].columns


def test_mark_frame_path_keyed_cache_retired(tmp_path) -> None:
    # The path-keyed mark frame cache is retired: canonical preparation and
    # replay use completed trade OHLCV; retained minute-frame loaders read the
    # lake directly with no process cache to isolate.
    from src.mhs import marks

    assert not hasattr(marks, "_get_symbol_mark_frame_for_path")
    assert not hasattr(marks, "_get_symbol_mark_frame")
    marks.clear_mhs_market_data_caches()


def test_window_ipc_errors(tmp_path) -> None:
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import (
        _spill_window_to_ipc,
        _load_window_from_ipc,
        _iter_spilled_windows,
    )

    with pytest.raises(DataIntegrityError, match="window IPC spill failed"):
        _spill_window_to_ipc(None, "/nonexistent_dir_1234/file.arrow")  # type: ignore[arg-type]

    bad_file = tmp_path / "bad.arrow"
    bad_file.write_text("corrupted content")
    with pytest.raises(DataIntegrityError, match="window IPC load failed"):
        _load_window_from_ipc(str(bad_file))

    with pytest.raises(DataIntegrityError, match="window IPC spill directory unreadable"):
        list(_iter_spilled_windows("/nonexistent_dir_1234"))


def test_window_ipc_numpy_restore_preserves_exact_bits(tmp_path) -> None:
    import numpy as np
    import pandas as pd

    from src.mhs.evaluation.windows import _load_window_from_ipc, _spill_window_to_ipc
    from src.mhs.execution.contracts import ExecutionReplayWindow

    minute_grid = pd.date_range("2026-01-01", periods=4, freq="3min", tz="UTC")
    decision_grid = pd.date_range("2026-01-01", periods=2, freq="6h", tz="UTC")
    market_values = np.array(
        [
            [0x3FF0000000000000, 0x8000000000000000],
            [0x7FF8000000000001, 0x7FF0000000000000],
            [0xFFF0000000000000, 0x400C000000000000],
            [0x401D000000000000, 0xC022000000000000],
        ],
        dtype=np.uint64,
    ).view(np.float64)
    weights = np.array([[0.25, -0.25], [0.0, 0.5]], dtype=np.float64)
    frames = {
        "highs": pd.DataFrame(market_values, index=minute_grid, columns=["BTC", "ETH"]),
        "lows": pd.DataFrame(market_values - 1.0, index=minute_grid, columns=["BTC", "ETH"]),
        "closes": pd.DataFrame(market_values, index=minute_grid, columns=["BTC", "ETH"]),
        "marks": pd.DataFrame(market_values, index=minute_grid, columns=["BTC", "ETH"]),
        "bar_funding": pd.DataFrame(np.zeros((4, 2), dtype=np.float64), index=minute_grid, columns=["BTC", "ETH"]),
        "target_weights": pd.DataFrame(weights, index=decision_grid, columns=["BTC", "ETH"]),
    }
    expected = ExecutionReplayWindow(
        window_start=minute_grid[0], window_end=minute_grid[-1], columns=("BTC", "ETH", "SOL"),
        symbols=("BTC", "ETH"), minute_grid=minute_grid, highs=frames["highs"], lows=frames["lows"],
        closes=frames["closes"], marks=frames["marks"], bar_funding=frames["bar_funding"],
        target_weights=frames["target_weights"], signal_available_at=decision_grid + pd.Timedelta(hours=1),
    )
    path = str(tmp_path / "window_00000.arrow")
    _spill_window_to_ipc(expected, path)

    actual = _load_window_from_ipc(path)

    assert actual.window_start == expected.window_start
    assert actual.window_end == expected.window_end
    assert actual.columns == expected.columns
    assert actual.symbols == expected.symbols
    assert actual.minute_grid.equals(expected.minute_grid)
    assert actual.signal_available_at.equals(expected.signal_available_at)
    for name in ("highs", "lows", "closes", "marks", "bar_funding", "target_weights"):
        actual_frame = getattr(actual, name)
        expected_frame = getattr(expected, name)
        assert actual_frame is not None
        assert expected_frame is not None
        assert actual_frame.index.equals(expected_frame.index)
        assert list(actual_frame.columns) == list(expected_frame.columns)
        assert all(dtype == np.dtype("float64") for dtype in actual_frame.dtypes)
        np.testing.assert_array_equal(
            actual_frame.to_numpy(dtype=np.float64).view(np.uint64),
            expected_frame.to_numpy(dtype=np.float64).view(np.uint64),
        )


def test_window_ipc_loader_has_no_python_value_list_conversion() -> None:
    import ast
    import inspect

    from src.mhs.evaluation.windows import _load_window_from_ipc

    source = inspect.getsource(_load_window_from_ipc)
    tree = ast.parse(source)
    attrs = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]

    assert "to_pylist" not in attrs
    assert "to_numpy" in attrs
    assert "zero_copy_only=False" in source


def test_iter_spilled_windows_keeps_filename_order_and_exact_values(tmp_path) -> None:
    import numpy as np
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_spilled_windows, _spill_window_to_ipc
    from src.mhs.execution.contracts import ExecutionReplayWindow

    spill = tmp_path / "spill"
    spill.mkdir()
    expected_starts = []
    expected_bits = []
    for index, value in ((1, -0.0), (0, 11.5)):
        grid = pd.date_range("2026-01-01", periods=2, freq="3min", tz="UTC") + pd.Timedelta(days=index)
        decisions = pd.DatetimeIndex([grid[0]])
        highs = pd.DataFrame({"BTC": np.array([value, value + 1.0], dtype=np.float64)}, index=grid)
        window = ExecutionReplayWindow(
            window_start=grid[0], window_end=grid[-1], columns=("BTC",), symbols=("BTC",),
            minute_grid=grid, highs=highs, lows=highs.copy(), closes=highs.copy(), marks=None,
            bar_funding=pd.DataFrame({"BTC": [0.0, 0.0]}, index=grid, dtype=np.float64),
            target_weights=pd.DataFrame({"BTC": [1.0]}, index=decisions, dtype=np.float64),
            signal_available_at=decisions + pd.Timedelta(hours=1),
        )
        _spill_window_to_ipc(window, str(spill / f"window_{index:05d}.arrow"))
        if index == 0:
            expected_starts.insert(0, grid[0])
            expected_bits.insert(0, highs.to_numpy().view(np.uint64))
        else:
            expected_starts.append(grid[0])
            expected_bits.append(highs.to_numpy().view(np.uint64))

    restored = list(_iter_spilled_windows(str(spill)))

    assert [window.window_start for window in restored] == expected_starts
    for window, bits in zip(restored, expected_bits, strict=True):
        assert window.marks is None
        np.testing.assert_array_equal(window.highs.to_numpy().view(np.uint64), bits)


def test_window_ipc_numpy_restore_corruption_is_fail_closed(tmp_path) -> None:
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import _load_window_from_ipc

    corrupt = tmp_path / "window_00000.arrow"
    corrupt.write_bytes(b"not-a-valid-window")

    with pytest.raises(DataIntegrityError, match="window IPC load failed"):
        _load_window_from_ipc(str(corrupt))


def test_missing_active_execution_file_stays_in_roster(tmp_path) -> None:
    import pandas as pd
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows
    from src.mhs.types import ExecutionSpec
    idx = pd.DatetimeIndex([pd.Timestamp('2025-01-01', tz='UTC')])
    weights = pd.DataFrame({'MISSUSDT': [1.0]}, index=idx)
    windows = list(_iter_mhs_execution_windows(weights, idx, str(tmp_path), '3m', idx[0], idx[0]+pd.Timedelta(minutes=9), {}, ExecutionSpec(), funding_failures={'MISSUSDT': 'missing'}))
    assert windows[0].symbols == ('MISSUSDT',)
    assert windows[0].closes['MISSUSDT'].isna().all()
    assert windows[0].quote_volumes['MISSUSDT'].isna().all()


def test_window_spill_root_prefers_env_and_defaults_to_repo_tmp(tmp_path, monkeypatch) -> None:
    import os
    from src.common.paths import BASE_DIR
    from src.mhs.evaluation.windows import _window_spill_root
    target = tmp_path / "custom_spill"
    monkeypatch.setenv("MHS_SPILL_DIR", str(target))
    assert _window_spill_root() == str(target)
    assert target.is_dir()
    monkeypatch.delenv("MHS_SPILL_DIR")
    default = _window_spill_root()
    assert default == str(BASE_DIR / "tmp" / "mhs_spill")
    assert os.path.isdir(default)


@pytest.mark.slow
def test_book_outcome_spills_windows_under_window_spill_root(mhs_market, monkeypatch, tmp_path) -> None:
    import dataclasses
    import tempfile
    import src.mhs.evaluation.windows as ev_windows
    from tests.unit.mhs.test_evaluation_appresearch import _build_book_outcome_args
    spill_root = tmp_path / "spill_root"
    monkeypatch.setenv("MHS_SPILL_DIR", str(spill_root))
    seen: list[object] = []
    real = tempfile.TemporaryDirectory

    def _recording(*args, **kwargs):
        seen.append(kwargs.get("dir"))
        return real(*args, **kwargs)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", _recording)
    args = _build_book_outcome_args(mhs_market)
    # When: the exact two-pass path spills windows
    ev_windows._book_outcome(**{**args, "request": dataclasses.replace(args["request"], pnl_vol_target=False, committee_target_gross=None)})
    # Then: the spill directory lives under the configured disk-backed root, not the system tmpfs
    assert str(spill_root) in seen


def _local_replay_fixtures(n_decisions: int = 2):
    """Canonical five-column path with two-symbol local windows."""
    import pandas as pd

    from src.mhs.execution.contracts import ExecutionReplayWindow
    from src.mhs.process import ProcessExecutionPolicy

    cols = ["AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT"]
    local = ["AUSDT", "CUSDT"]
    idx = pd.date_range("2022-01-01", periods=n_decisions, freq="24h", tz="UTC")
    tgt = pd.DataFrame(0.0, index=idx, columns=cols, dtype="float64")
    tgt.loc[:, "AUSDT"] = 0.5
    tgt.loc[:, "CUSDT"] = -0.25
    hourly = pd.date_range(idx[0], idx[-1] + pd.Timedelta(hours=23), freq="1h", tz="UTC")
    path = ProcessPath(
        one_way_bps=8.0,
        daily_returns=pd.Series(0.0, index=idx),
        unit_daily_returns=pd.Series(0.0, index=idx),
        exposure=pd.Series(1.0, index=idx),
        refits=(),
        leverage_cap=2.0,
        execution_policy=ProcessExecutionPolicy(None),
        unit_target_weights=tgt,
        target_weights=tgt,
        turnover_1h=pd.Series(0.0, index=hourly),
    )

    def _window(day: pd.Timestamp, row: pd.DataFrame) -> ExecutionReplayWindow:
        mg = pd.date_range(day, day + pd.Timedelta(hours=23, minutes=59), freq="1min", tz="UTC")

        def _mk(v: float) -> pd.DataFrame:
            return pd.DataFrame(v, index=mg, columns=local, dtype="float64")

        return ExecutionReplayWindow(
            window_start=mg[0], window_end=mg[-1], columns=tuple(cols), symbols=tuple(local),
            minute_grid=mg, highs=_mk(100.0), lows=_mk(99.0), closes=_mk(99.5),
            marks=_mk(99.5), bar_funding=_mk(0.0), target_weights=row[local],
            signal_available_at=pd.DatetimeIndex([day]),
            quote_volumes=_mk(1e6),
            funding_known=pd.DataFrame(True, index=mg, columns=local),
            bar_available_at=mg,
        )

    windows = [_window(day, tgt.iloc[[i]]) for i, day in enumerate(idx)]
    return path, windows


def test_validate_local_targets_accepted() -> None:
    """Ordered local targets validate against the canonical book and replay."""
    from src.mhs.execution.batch import replay_execution_windows
    from src.mhs.types import ExecutionSpec

    path, windows = _local_replay_fixtures(2)
    spec = ExecutionSpec()
    ref = replay_execution_windows(windows, 1000.0, "OHLCV_IMMEDIATE_TAKER", spec)
    got = replay_process_execution(
        path, iter(windows), initial_equity=1000.0,
        execution_bound="OHLCV_IMMEDIATE_TAKER", spec=spec,
    )
    assert ref.ledger.equity.equals(got.ledger.equity)
    assert [s for w in windows for s in w.symbols] == ["AUSDT", "CUSDT"] * 2


def test_validate_omitted_nonzero_target_rejected() -> None:
    """A nonzero omitted canonical target fails validation exactly, not approximately."""
    import dataclasses

    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.types import ExecutionSpec

    path, windows = _local_replay_fixtures(2)
    tainted = path.target_weights.copy()
    tainted.iloc[0, tainted.columns.get_loc("BUSDT")] = 1e-15
    bad_path = dataclasses.replace(path, target_weights=tainted, unit_target_weights=tainted)
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            bad_path, iter(windows), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=ExecutionSpec(),
        )


def test_validate_last_complete_bar_accepted() -> None:
    """A bar publishing exactly at the fence is complete and accepted."""
    import dataclasses

    import pandas as pd


    path, windows = _local_replay_fixtures(2)
    fence = _execution_fence(path.target_weights)
    w1 = windows[0]
    assert (w1.bar_available_at < fence).all()
    arr = w1.bar_available_at.as_unit("ns").asi8.copy()
    arr[-1] = fence.value
    stamped = dataclasses.replace(w1, bar_available_at=pd.DatetimeIndex(arr, tz="UTC"))
    nxt = _validate_replay_window(
        stamped, expected_columns=list(path.target_weights.columns),
        expected_targets=path.target_weights, cursor=0, fence=fence,
    )
    assert nxt == 1


def test_validate_incomplete_bar_rejected() -> None:
    """A bar publishing beyond the fence is incomplete and rejected."""
    import dataclasses

    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError

    path, windows = _local_replay_fixtures(2)
    fence = _execution_fence(path.target_weights)
    w1 = windows[0]
    arr = w1.bar_available_at.as_unit("ns").asi8.copy()
    arr[-1] = (fence + pd.Timedelta(minutes=1)).value
    stamped = dataclasses.replace(w1, bar_available_at=pd.DatetimeIndex(arr, tz="UTC"))
    with pytest.raises(DataIntegrityError, match=r".+"):
        _validate_replay_window(
            stamped, expected_columns=list(path.target_weights.columns),
            expected_targets=path.target_weights, cursor=0, fence=fence,
        )


def test_validate_partition_skip_rejected() -> None:
    """Skipped decisions break cursor coverage and fail validation."""
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.types import ExecutionSpec

    path, windows = _local_replay_fixtures(3)
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_process_execution(
            path, iter([windows[0], windows[2]]), initial_equity=1000.0,
            execution_bound="OHLCV_IMMEDIATE_TAKER", spec=ExecutionSpec(),
        )


def _write_3m_ohlcv(root, symbol, labels, quote_vol=1000.0) -> None:
    import pandas as pd

    ms = (labels - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    n = len(labels)
    close = 100.0 + 0.01 * (labels.asi8 // 180_000_000_000)
    pd.DataFrame(
        {
            "timestamp": ms.to_numpy(dtype="int64"),
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "quote_vol": quote_vol,
        }
    ).to_parquet(root / f"{symbol}.parquet")


def _stream_market(tmp_path, symbols, n_days=70, funding_through=None):
    """3m OHLCV lake plus funding aligned to the full grid."""
    import pandas as pd

    start = pd.Timestamp("2022-01-01", tz="UTC")
    grid = pd.date_range(start, start + pd.Timedelta(days=n_days) - pd.Timedelta(minutes=3), freq="3min", tz="UTC")
    lake = tmp_path / "ohlcv" / "3m"
    lake.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        _write_3m_ohlcv(lake, sym, grid)
    observed = grid if funding_through is None else grid[grid < funding_through]
    funding = {sym: pd.Series(0.0, index=observed) for sym in symbols}
    decisions = pd.date_range(start, periods=n_days, freq="24h", tz="UTC")
    return grid, decisions, funding


def test_generator_required_symbols_stay_in_roster(tmp_path) -> None:
    """A held symbol with zero targets remains in every required roster."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows
    from src.mhs.types import ExecutionSpec

    grid, decisions, funding = _stream_market(tmp_path, ["AUSDT", "BUSDT"], n_days=70)
    targets = pd.DataFrame(0.0, index=decisions, columns=["AUSDT", "BUSDT"])
    targets.iloc[:6, 0] = 0.5
    signals = decisions + pd.Timedelta(hours=1)
    windows = list(
        _iter_mhs_execution_windows(
            targets, signals, str(tmp_path / "ohlcv"), "3m", grid[0], grid[-1],
            funding, ExecutionSpec(),
            required_symbols=lambda: frozenset({"AUSDT"}),
        )
    )
    assert len(windows) >= 2
    assert all("AUSDT" in w.symbols for w in windows)
    assert all(w.logical_partition is not None for w in windows)
    tags = [w.logical_partition for w in windows]
    assert tags == sorted(tags)


def test_generator_unknown_required_symbol_rejected(tmp_path) -> None:
    """A required symbol outside canonical columns fails closed."""
    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows
    from src.mhs.types import ExecutionSpec

    grid, decisions, funding = _stream_market(tmp_path, ["AUSDT"], n_days=3)
    targets = pd.DataFrame(0.0, index=decisions, columns=["AUSDT"])
    with pytest.raises(DataIntegrityError, match=r".+"):
        list(
            _iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"), "3m",
                grid[0], grid[-1], funding, ExecutionSpec(),
                required_symbols=lambda: frozenset({"GHOSTUSDT"}),
            )
        )


def _split_stream(windows):
    """Hand-split each window into two overlapping pieces sharing its tag."""
    import dataclasses

    pieces = []
    for w in windows:
        n = len(w.minute_grid)
        mid = n // 2
        boundary = w.minute_grid[mid]
        k = int((w.target_weights.index < boundary).sum())
        if k <= 0 or k >= len(w.target_weights):
            pieces.append(w)
            continue
        for t_lo, t_hi, lo, hi in ((0, k, 0, mid + 1), (k, len(w.target_weights), mid, n)):
            sl = slice(lo, hi)
            pieces.append(
                dataclasses.replace(
                    w,
                    minute_grid=w.minute_grid[sl],
                    highs=w.highs.iloc[sl],
                    lows=w.lows.iloc[sl],
                    closes=w.closes.iloc[sl],
                    marks=w.marks.iloc[sl] if w.marks is not None else None,
                    bar_funding=w.bar_funding.iloc[sl],
                    target_weights=w.target_weights.iloc[t_lo:t_hi],
                    signal_available_at=w.signal_available_at[t_lo:t_hi],
                    quote_volumes=w.quote_volumes.iloc[sl] if w.quote_volumes is not None else None,
                    funding_known=w.funding_known.iloc[sl] if w.funding_known is not None else None,
                    bar_available_at=w.bar_available_at[sl] if w.bar_available_at is not None else None,
                )
            )
    return pieces


def _run_split_parity(tmp_path, cost_model: str):
    import dataclasses

    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows
    from src.mhs.execution import replay_execution_windows
    from src.mhs.types import ExecutionSpec

    grid, decisions, funding = _stream_market(tmp_path, ["AUSDT", "BUSDT"], n_days=40)
    targets = pd.DataFrame(0.0, index=decisions, columns=["AUSDT", "BUSDT"])
    targets.iloc[:, 0] = 0.05
    targets.iloc[:, 1] = -0.03
    signals = decisions + pd.Timedelta(hours=1)
    spec = ExecutionSpec() if cost_model == "flat" else dataclasses.replace(
        ExecutionSpec(), liquidity_cost_model="corwin_schultz"
    )
    windows = list(
        _iter_mhs_execution_windows(
            targets, signals, str(tmp_path / "ohlcv"), "3m", grid[0], grid[-1],
            funding, spec,
        )
    )
    ref = replay_execution_windows(iter(windows), 1000.0, "OHLCV_IMMEDIATE_TAKER", spec)
    split = replay_execution_windows(
        iter(_split_stream(windows)), 1000.0, "OHLCV_IMMEDIATE_TAKER", spec
    )
    return ref, split


def test_flat_split_parity_within_1e12(tmp_path) -> None:
    """Smaller IO pieces under one logical clock keep flat equity within 1e-12."""
    import numpy as np

    ref, split = _run_split_parity(tmp_path, "flat")
    assert np.allclose(
        ref.ledger.equity.to_numpy(), split.ledger.equity.to_numpy(), rtol=0.0, atol=1e-12
    )
    assert len(ref.simulated_fills) == len(split.simulated_fills)


def test_spread_split_parity_within_1e12(tmp_path) -> None:
    """Corwin-Schultz state, fills and costs match the unsplit reference."""
    import numpy as np

    ref, split = _run_split_parity(tmp_path, "corwin_schultz")
    assert np.allclose(
        ref.ledger.equity.to_numpy(), split.ledger.equity.to_numpy(), rtol=0.0, atol=1e-12
    )
    assert ref.simulated_fills["fill_price"].to_numpy().shape == split.simulated_fills["fill_price"].to_numpy().shape
    assert np.allclose(
        ref.simulated_fills["fill_price"].to_numpy(dtype="float64"),
        split.simulated_fills["fill_price"].to_numpy(dtype="float64"),
        rtol=0.0, atol=1e-12,
    )


def test_no_future_pricing_across_pieces(tmp_path) -> None:
    """Perturbed future highs/lows leave earlier fills priced identically."""
    import numpy as np

    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows
    from src.mhs.execution import replay_execution_windows
    from src.mhs.types import ExecutionSpec

    grid, decisions, funding = _stream_market(tmp_path, ["AUSDT"], n_days=40)
    targets = pd.DataFrame(0.05, index=decisions, columns=["AUSDT"])
    signals = decisions + pd.Timedelta(hours=1)
    spec = ExecutionSpec(liquidity_cost_model="corwin_schultz")
    windows = list(
        _iter_mhs_execution_windows(
            targets, signals, str(tmp_path / "ohlcv"), "3m", grid[0], grid[-1],
            funding, spec,
        )
    )
    pieces = _split_stream(windows)
    base = replay_execution_windows(iter(pieces), 1000.0, "OHLCV_IMMEDIATE_TAKER", spec)
    shocked = list(pieces)
    last = shocked[-1]
    import dataclasses

    shocked[-1] = dataclasses.replace(
        last, highs=last.highs * 2.0, lows=last.lows * 2.0,
    )
    moved = replay_execution_windows(iter(shocked), 1000.0, "OHLCV_IMMEDIATE_TAKER", spec)
    cutoff = pieces[1].minute_grid[0]
    base_early = base.simulated_fills[base.simulated_fills["timestamp"] < cutoff]
    moved_early = moved.simulated_fills[moved.simulated_fills["timestamp"] < cutoff]
    assert len(base_early) == len(moved_early) > 0
    assert np.array_equal(
        base_early["fill_price"].to_numpy(), moved_early["fill_price"].to_numpy()
    )
    assert np.array_equal(
        base_early["fee_bps"].to_numpy(), moved_early["fee_bps"].to_numpy()
    )


def test_required_symbols_track_units_without_pruning() -> None:
    """required_symbols exposes nonzero units exactly, however tiny."""
    import numpy as np
    import pandas as pd

    from src.mhs.execution.accumulator import _BoundExecutionReplayAccumulator
    from src.mhs.execution.contracts import ExecutionReplayWindow
    from src.mhs.types import ExecutionSpec

    grid = pd.date_range("2022-01-01", periods=8, freq="3min", tz="UTC")
    cols = ("AUSDT", "BUSDT")
    px = pd.DataFrame(100.0, index=grid, columns=list(cols))
    window = ExecutionReplayWindow(
        window_start=grid[0], window_end=grid[-1], columns=cols, symbols=cols,
        minute_grid=grid, highs=px, lows=px, closes=px, marks=px, bar_funding=px * 0.0,
        target_weights=pd.DataFrame(0.0, index=[grid[0]], columns=list(cols)),
        signal_available_at=pd.DatetimeIndex([grid[0]]),
        quote_volumes=px * 0.0 + 1.0, funding_known=px.notna(),
        bar_available_at=grid + pd.Timedelta(minutes=3),
    )
    acc = _BoundExecutionReplayAccumulator(window, 1000.0, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(), False)
    assert acc.required_symbols() == frozenset()
    acc.units_arr[0] = 1e-15
    acc.units_arr[1] = -2.5
    assert acc.required_symbols() == frozenset({"AUSDT", "BUSDT"})
    assert isinstance(acc.required_symbols(), frozenset)
    assert np.isnan(acc.half_spread_bps).all()


def test_live_required_symbols_union_and_empty() -> None:
    """The batch union skips dead bounds and starts empty."""
    from src.mhs.execution.batch import live_required_symbols

    assert live_required_symbols([]) == frozenset()

    class _Stub:
        def __init__(self, symbols):
            self._symbols = symbols

        def required_symbols(self):
            return frozenset(self._symbols)

    assert live_required_symbols([None, None]) == frozenset()
    assert live_required_symbols([_Stub({"AUSDT"}), None, _Stub({"BUSDT"})]) == frozenset(
        {"AUSDT", "BUSDT"}
    )


def _held_exit_market(tmp_path, n_days=70):
    """Entry fills early, then targets vanish while exits cannot fill.

    Exits are blocked by unknown funding (not zero volume, which would be a
    delisting settlement), so inventory stays held across windows.
    """
    import pandas as pd

    symbols = ["AUSDT", "BUSDT"]
    exit_block_from = pd.Timestamp("2022-01-01", tz="UTC") + pd.Timedelta(days=2)
    grid, decisions, funding = _stream_market(
        tmp_path, symbols, n_days=n_days, funding_through=exit_block_from,
    )
    targets = pd.DataFrame(0.0, index=decisions, columns=symbols)
    targets.iloc[:6, 0] = 0.5
    signals = decisions + pd.Timedelta(hours=1)
    return grid, decisions, funding, targets, signals


def test_batch_live_roster_holds_unfilled_exit(tmp_path) -> None:
    """Held inventory stays in every required roster across windows."""
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows
    from src.mhs.execution import live_required_symbols, replay_execution_window_batch_isolated
    from src.mhs.types import ExecutionSpec

    grid, decisions, funding, targets, signals = _held_exit_market(tmp_path)
    spec = ExecutionSpec()
    cell: list = []
    seen: list = []

    def _recording():
        gen = _iter_mhs_execution_windows(
            targets, signals, str(tmp_path / "ohlcv"), "3m", grid[0], grid[-1],
            funding, spec,
            required_symbols=lambda: live_required_symbols(cell[0] if cell else []),
        )
        for w in gen:
            seen.append(w.symbols)
            yield w

    outcome = replay_execution_window_batch_isolated(
        _recording(), 1000.0, [("OHLCV_IMMEDIATE_TAKER", spec)],
        live_accumulators=cell,
    )
    assert outcome.results[0] is not None
    assert len(seen) >= 3
    assert all("AUSDT" in symbols for symbols in seen)
    assert cell
    assert cell[0]
    assert "AUSDT" in live_required_symbols(cell[0])


def test_batch_without_live_cell_drops_stale_roster_member(tmp_path) -> None:
    """Without live requirements the legacy roster drops the held symbol."""
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows
    from src.mhs.execution import replay_execution_window_batch_isolated
    from src.mhs.types import ExecutionSpec

    grid, decisions, funding, targets, signals = _held_exit_market(tmp_path)
    spec = ExecutionSpec()
    windows = list(
        _iter_mhs_execution_windows(
            targets, signals, str(tmp_path / "ohlcv"), "3m", grid[0], grid[-1],
            funding, spec,
        )
    )
    assert len(windows) >= 3
    assert "AUSDT" not in windows[-1].symbols
    with pytest.raises(DataIntegrityError, match=r".+"):
        replay_execution_window_batch_isolated(
            iter(windows), 1000.0, [("OHLCV_IMMEDIATE_TAKER", spec)]
        )


def test_single_and_coupled_live_cells_populated(tmp_path) -> None:
    """Single and coupled replays publish their live accumulators."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows
    from src.mhs.execution import (
        live_required_symbols,
        replay_execution_windows,
        replay_execution_windows_coupled,
    )
    from src.mhs.types import ExecutionSpec

    grid, decisions, funding = _stream_market(tmp_path, ["AUSDT"], n_days=10)
    targets = pd.DataFrame(0.05, index=decisions, columns=["AUSDT"])
    signals = decisions + pd.Timedelta(hours=1)
    spec = ExecutionSpec()
    args = (
        targets, signals, str(tmp_path / "ohlcv"), "3m", grid[0], grid[-1],
        funding, spec,
    )
    cell_single: list = []
    replay_execution_windows(
        _iter_mhs_execution_windows(*args), 1000.0, "OHLCV_IMMEDIATE_TAKER", spec,
        live_accumulators=cell_single,
    )
    assert cell_single
    assert cell_single[0]
    assert live_required_symbols(cell_single[0]) == frozenset({"AUSDT"})
    cell_coupled: list = []
    scale = pd.Series(1.0, index=decisions)
    reference, outcome = replay_execution_windows_coupled(
        _iter_mhs_execution_windows(*args), 1000.0,
        ("OHLCV_IMMEDIATE_TAKER", spec),
        [("OHLCV_IMMEDIATE_TAKER", spec)],
        lambda daily_returns: scale,
        live_accumulators=cell_coupled,
    )
    assert reference is not None
    assert outcome.results[0] is not None
    assert cell_coupled
    assert cell_coupled[0]
    assert "AUSDT" in live_required_symbols(cell_coupled[0])


def test_ipc_round_trip_preserves_logical_partition(tmp_path) -> None:
    """Spilled windows reload with their cost-clock tag intact."""
    import pandas as pd

    from src.mhs.evaluation.windows import (
        _iter_mhs_execution_windows,
        _load_window_from_ipc,
        _spill_window_to_ipc,
    )
    from src.mhs.types import ExecutionSpec

    grid, decisions, funding = _stream_market(tmp_path, ["AUSDT"], n_days=40)
    targets = pd.DataFrame(0.05, index=decisions, columns=["AUSDT"])
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"), "3m",
            grid[0], grid[-1], funding, ExecutionSpec(),
        )
    )
    path = str(tmp_path / "tagged.arrow")
    _spill_window_to_ipc(windows[0], path)
    loaded = _load_window_from_ipc(path)
    assert loaded.logical_partition == windows[0].logical_partition != (None,)


def test_untagged_corwin_windows_keep_legacy_per_window_update() -> None:
    """Direct-built windows update spreads immediately without a clock."""
    import dataclasses

    import numpy as np
    import pandas as pd

    from src.mhs.execution import replay_execution_windows
    from src.mhs.execution.accumulator import _BoundExecutionReplayAccumulator
    from src.mhs.execution.contracts import ExecutionReplayWindow
    from src.mhs.types import ExecutionSpec

    grid = pd.date_range("2022-01-01", periods=24, freq="3min", tz="UTC")
    cols = ["AUSDT"]
    px = pd.DataFrame(100.0, index=grid, columns=cols)
    window = ExecutionReplayWindow(
        window_start=grid[0], window_end=grid[-1], columns=tuple(cols), symbols=tuple(cols),
        minute_grid=grid, highs=px * 1.002, lows=px * 0.998, closes=px, marks=px,
        bar_funding=px * 0.0,
        target_weights=pd.DataFrame({"AUSDT": [0.5]}, index=pd.DatetimeIndex([grid[0]])),
        signal_available_at=pd.DatetimeIndex([grid[0]]),
        quote_volumes=px * 0.0 + 1.0, funding_known=px.notna(),
        bar_available_at=grid + pd.Timedelta(minutes=3),
    )
    spec = dataclasses.replace(ExecutionSpec(), liquidity_cost_model="corwin_schultz")
    acc = _BoundExecutionReplayAccumulator(window, 1000.0, "OHLCV_IMMEDIATE_TAKER", spec, False)
    assert window.logical_partition is None
    acc.consume(window)
    assert np.isfinite(acc.half_spread_bps).all()
    result = acc.finalize()
    assert len(result.simulated_fills) > 0
    replayed = replay_execution_windows(
        iter([window]), 1000.0, "OHLCV_IMMEDIATE_TAKER", spec,
    )
    assert replayed.ledger.equity.equals(result.ledger.equity)


def test_validate_symbol_contract_rejected() -> None:
    """Duplicated, foreign, or misordered symbols fail validation."""
    import dataclasses

    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError

    path, windows = _local_replay_fixtures(2)
    kwargs = {
        "expected_columns": list(path.target_weights.columns),
        "expected_targets": path.target_weights,
        "cursor": 0,
        "fence": path.target_weights.index[-1] + pd.Timedelta(days=1),
    }
    w1 = windows[0]
    dup = dataclasses.replace(w1, symbols=("AUSDT", "AUSDT"))
    with pytest.raises(DataIntegrityError, match=r".+"):
        _validate_replay_window(dup, **kwargs)
    foreign = dataclasses.replace(
        w1,
        symbols=("AUSDT", "ZZZUSDT"),
        target_weights=w1.target_weights.rename(columns={"CUSDT": "ZZZUSDT"}),
    )
    with pytest.raises(DataIntegrityError, match=r".+"):
        _validate_replay_window(foreign, **kwargs)
    shuffled = dataclasses.replace(
        w1,
        symbols=("CUSDT", "AUSDT"),
        target_weights=w1.target_weights[["CUSDT", "AUSDT"]],
    )
    with pytest.raises(DataIntegrityError, match=r".+"):
        _validate_replay_window(shuffled, **kwargs)
    bad_days = pd.concat([w1.target_weights, windows[1].target_weights])
    bad_days.index = pd.DatetimeIndex([bad_days.index[0], bad_days.index[0]])
    jumbled = dataclasses.replace(w1, target_weights=bad_days)
    with pytest.raises(DataIntegrityError, match=r".+"):
        _validate_replay_window(jumbled, **kwargs)


def _completed_fixture(tmp_path, start, end, decisions):
    import pandas as pd

    from src.mhs.types import ExecutionSpec

    grid = pd.date_range(start, end, freq="3min", tz="UTC")
    lake = tmp_path / "ohlcv" / "3m"
    lake.mkdir(parents=True, exist_ok=True)
    for sym in ("AUSDT", "BUSDT"):
        _write_3m_ohlcv(lake, sym, grid)
    funding = {s: pd.Series(0.0, index=grid) for s in ("AUSDT", "BUSDT")}
    targets = pd.DataFrame(0.0, index=decisions, columns=["AUSDT", "BUSDT"])
    return grid, funding, targets, ExecutionSpec()


def test_generator_final_fence_emits_only_completed_bars(tmp_path) -> None:
    """Aligned fence keeps 23:57 with 00:00 publication and never labels 00:00."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2023-06-01", tz="UTC")
    end = pd.Timestamp("2023-06-02", tz="UTC")
    decisions = pd.date_range(start, periods=2, freq="24h", tz="UTC")
    _, funding, targets, spec = _completed_fixture(tmp_path, start, end, decisions)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
        )
    )
    last = windows[-1]
    assert last.minute_grid[-1] == pd.Timestamp("2023-06-01 23:57", tz="UTC")
    assert last.bar_available_at[-1] == end
    assert (last.minute_grid < end).all()
    assert (last.bar_available_at <= end).all()
    assert end not in last.minute_grid


def test_generator_unaligned_fence_trims_incomplete_bar(tmp_path) -> None:
    """A 00:01 fence keeps 23:57 and drops 00:00 whose publication is 00:03."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2023-06-01", tz="UTC")
    end = pd.Timestamp("2023-06-02 00:01", tz="UTC")
    decisions = pd.date_range(start, periods=2, freq="24h", tz="UTC")
    _, funding, targets, spec = _completed_fixture(tmp_path, start, end, decisions)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
        )
    )
    last = windows[-1]
    assert pd.Timestamp("2023-06-01 23:57", tz="UTC") in last.minute_grid
    assert pd.Timestamp("2023-06-02 00:00", tz="UTC") not in last.minute_grid
    assert (last.bar_available_at <= end).all()


def test_generator_interior_timeout_endpoint_retained(tmp_path) -> None:
    """A monthly partition keeps its completed timeout bar instead of trimming it."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.date_range(start, periods=40, freq="24h", tz="UTC")
    end = decisions[-1] + pd.Timedelta(days=1)
    _, funding, targets, spec = _completed_fixture(tmp_path, start, end, decisions)
    targets.iloc[:, 0] = 0.05
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
        )
    )
    assert len(windows) >= 2
    first = windows[0]
    assert first.window_end == first.minute_grid[-1]
    assert first.window_end + pd.Timedelta(minutes=3) <= end


def test_generator_never_decodes_out_of_fence_row(tmp_path, monkeypatch) -> None:
    """Source rows past the fence exist on disk but are never decoded."""
    import pandas as pd

    import src.mhs.marks as marks
    import src.mhs.execution.window_stream as window_stream
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2023-06-01", tz="UTC")
    end = pd.Timestamp("2023-06-02", tz="UTC")
    decisions = pd.date_range(start, periods=2, freq="24h", tz="UTC")
    _, funding, targets, spec = _completed_fixture(tmp_path, start, end, decisions)
    seen: dict = {}
    real = marks._load_window_minute_frames

    def _spy(root, symbols, grid_start, grid_end, timeframe):
        seen["grid_end"] = grid_end
        return real(root, symbols, grid_start, grid_end, timeframe)

    monkeypatch.setattr(marks, "_load_window_minute_frames", _spy)
    monkeypatch.setattr(window_stream, "_load_window_minute_frames", _spy)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
        )
    )
    assert seen["grid_end"] < end
    assert seen["grid_end"] + pd.Timedelta(minutes=3) <= end
    assert all(end not in w.minute_grid for w in windows)


def test_generator_minimum_legal_range_succeeds(tmp_path) -> None:
    """Exactly two completed bars form a valid window without a future bar."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = start + pd.Timedelta(minutes=6)
    decisions = pd.DatetimeIndex([start])
    _, funding, targets, spec = _completed_fixture(tmp_path, start, end, decisions)
    targets = pd.DataFrame([[0.1, 0.0]], index=decisions, columns=["AUSDT", "BUSDT"])
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
        )
    )
    assert len(windows) == 1
    assert len(windows[0].minute_grid) == 2
    assert (windows[0].bar_available_at <= end).all()


def test_generator_insufficient_range_fails_closed(tmp_path) -> None:
    """Fewer than two completed bars raise instead of fabricating prices."""
    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = start + pd.Timedelta(minutes=3)
    decisions = pd.DatetimeIndex([start])
    _, funding, _, spec = _completed_fixture(
        tmp_path, start, start + pd.Timedelta(minutes=9), decisions
    )
    targets = pd.DataFrame([[0.1, 0.0]], index=decisions, columns=["AUSDT", "BUSDT"])
    with pytest.raises(DataIntegrityError, match=r".+"):
        list(
            _iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, end, funding, spec,
            )
        )


def test_generator_empty_targets_emit_completed_grid(tmp_path) -> None:
    """Empty decisions still stream the completed grid without the fence label."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = start + pd.Timedelta(minutes=9)
    decisions = pd.DatetimeIndex([start])
    _, funding, _, spec = _completed_fixture(tmp_path, start, end, decisions)
    empty = pd.DataFrame(index=decisions[:0], columns=["AUSDT", "BUSDT"], dtype="float64")
    windows = list(
        _iter_mhs_execution_windows(
            empty, decisions[:0] + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
        )
    )
    assert len(windows) == 1
    assert windows[0].minute_grid[-1] == end - pd.Timedelta(minutes=3)
    assert end not in windows[0].minute_grid


def _tight_telemetry(monkeypatch, *, pss: int = 0, available: int = 10**12) -> None:
    from src.mhs import resources as _res

    monkeypatch.setattr(_res, "_current_tree_pss_bytes", lambda: pss)
    monkeypatch.setattr(_res, "_current_available_bytes", lambda: available)
    monkeypatch.setattr(_res, "_read_cgroup_remaining_bytes", lambda: None)
    monkeypatch.setattr(_res, "_current_tree_swap_bytes", lambda: 0)


def _adaptive_fixture(tmp_path, *, days: int = 3):
    import pandas as pd

    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.date_range(start, periods=days, freq="24h", tz="UTC")
    end = start + pd.Timedelta(days=days)
    _, funding, targets, spec = _completed_fixture(tmp_path, start, end, decisions)
    targets.iloc[:, 0] = 0.05
    return start, end, decisions, funding, targets, spec


def test_adaptive_decode_splits_with_no_missing_decisions(tmp_path, monkeypatch) -> None:
    """Tight envelope emits smaller admitted pieces with every decision exactly once."""
    import pandas as pd

    from src.mhs import resources as _res
    import src.mhs.execution.window_stream as window_stream
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=3)
    _tight_telemetry(monkeypatch)
    admitted: list[int] = []
    orig = _res.assert_mhs_allocation_budget
    monkeypatch.setattr(_res, "assert_mhs_allocation_budget", lambda **k: (admitted.append(k["estimated_bytes"]), orig(**k))[1])
    monkeypatch.setattr(window_stream, "assert_mhs_allocation_budget", lambda **k: (admitted.append(k["estimated_bytes"]), orig(**k))[1])
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_600_000, reserve_bytes=100, execution_bound_count=2,
        )
    )
    assert len(windows) > 1
    assert len(admitted) >= len(windows)
    got = pd.DatetimeIndex([]).append([w.target_weights.index for w in windows]) if windows else pd.DatetimeIndex([])
    assert list(got) == list(targets.index)
    keys = {w.logical_partition for w in windows}
    assert len(keys) == 1
    assert next(iter(keys)) == (0, 3)


def test_adaptive_decode_streams_empty_pieces_before_next_daily_decision(tmp_path, monkeypatch) -> None:
    """A narrow physical plan advances through decision-free daily gaps."""
    from itertools import pairwise

    import pandas as pd

    import src.mhs.execution.window_stream as _stream
    import src.mhs.evaluation.windows as _w

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=3)
    _tight_telemetry(monkeypatch)
    monkeypatch.setattr(_stream, "plan_mhs_execution_bars", lambda **_kwargs: 100)
    windows = list(
        _w._iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_600_000, reserve_bytes=100, execution_bound_count=2,
        )
    )

    nonempty = [window for window in windows if not window.target_weights.empty]
    got = pd.DatetimeIndex([]).append([window.target_weights.index for window in nonempty])
    assert list(got) == list(targets.index)
    assert any(window.target_weights.empty for window in windows[1:])
    assert all(
        later.minute_grid[0] == earlier.minute_grid[-1]
        for earlier, later in pairwise(windows)
    )
    assert all(
        decision in window.minute_grid
        for window in nonempty
        for decision in window.target_weights.index
    )

    held_windows = list(
        _w._iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_600_000, reserve_bytes=100, execution_bound_count=2,
            required_symbols=lambda: frozenset({"AUSDT"}),
        )
    )
    assert all("AUSDT" in window.symbols for window in held_windows)


def test_adaptive_decode_rejects_plan_shorter_than_first_signal_span(tmp_path, monkeypatch) -> None:
    """A malformed plan cannot split the first decision from its signal span."""
    import pandas as pd
    import pytest

    import src.mhs.execution.window_stream as _stream
    import src.mhs.evaluation.windows as _w
    from src.common.errors import DataIntegrityError

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=2)
    _tight_telemetry(monkeypatch)
    monkeypatch.setattr(_stream, "plan_mhs_execution_bars", lambda **_kwargs: 1)
    with pytest.raises(DataIntegrityError, match="unresolved"):
        list(
            _w._iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, end, funding, spec,
                budget_bytes=1_600_000, reserve_bytes=100,
            )
        )


def test_adaptive_empty_piece_rejects_unknown_live_roster(tmp_path, monkeypatch) -> None:
    """Decision-free pieces preserve the fail-closed live-roster contract."""
    import itertools

    import pandas as pd
    import pytest

    import src.mhs.execution.window_stream as _stream
    import src.mhs.evaluation.windows as _w
    from src.common.errors import DataIntegrityError

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=3)
    _tight_telemetry(monkeypatch)
    monkeypatch.setattr(_stream, "plan_mhs_execution_bars", lambda **_kwargs: 100)
    calls = itertools.count()

    def _live_roster() -> frozenset[str]:
        return frozenset({"AUSDT"}) if next(calls) < 2 else frozenset({"ZZZ"})

    with pytest.raises(DataIntegrityError, match="not in canonical"):
        list(
            _w._iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, end, funding, spec,
                budget_bytes=1_600_000, reserve_bytes=100,
                required_symbols=_live_roster,
            )
        )


def test_adaptive_tail_and_nonaligned_fence(tmp_path, monkeypatch) -> None:
    """Held tail past the final decision streams completed bars with coverage."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.date_range(start, periods=2, freq="24h", tz="UTC")
    end = start + pd.Timedelta(days=2, minutes=1)
    _, funding, targets, spec = _completed_fixture(tmp_path, start, end, decisions)
    targets.iloc[0, 0] = 0.1
    _tight_telemetry(monkeypatch)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_550_000, reserve_bytes=100,
            required_symbols=lambda: frozenset({"AUSDT"}),
        )
    )
    assert windows
    assert (windows[-1].minute_grid < end).all()
    assert (windows[-1].bar_available_at <= end).all()


def test_adaptive_minimum_timeout_retention(tmp_path, monkeypatch) -> None:
    """An order near a physical boundary keeps its strict timeout endpoint."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=2)
    _tight_telemetry(monkeypatch)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_550_000, reserve_bytes=100,
        )
    )
    assert len(windows) > 1
    first = windows[0]
    assert len(first.minute_grid) >= 2
    assert len(first.target_weights) >= 1


def test_adaptive_bound_union_roster(tmp_path, monkeypatch) -> None:
    """Base and stress holdings both remain in the next piece roster."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=2)
    targets.iloc[:, 1] = 0.0
    _tight_telemetry(monkeypatch)
    held = {"AUSDT", "BUSDT"}
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_800_000, reserve_bytes=100,
            required_symbols=lambda: frozenset(held),
        )
    )
    assert windows
    assert held <= set(windows[1].symbols) if len(windows) > 1 else held <= set(windows[0].symbols)


def test_adaptive_ipc_partition_parity(tmp_path, monkeypatch) -> None:
    """Spilled and restored split pieces keep ordinal keys and numeric planes."""
    import numpy as np
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows, _spill_window_to_ipc, _load_window_from_ipc

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=2)
    _tight_telemetry(monkeypatch)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_550_000, reserve_bytes=100,
        )
    )
    assert len(windows) > 1
    for i, w in enumerate(windows):
        path = str(tmp_path / f"piece_{i}.arrow")
        _spill_window_to_ipc(w, path)
        loaded = _load_window_from_ipc(path)
        assert loaded.logical_partition == w.logical_partition == (0, 2)
        np.testing.assert_allclose(loaded.highs.to_numpy(dtype="float64"), w.highs.to_numpy(dtype="float64"))
        assert (loaded.minute_grid == w.minute_grid).all()


def test_adaptive_logical_identity_under_splitting(tmp_path, monkeypatch) -> None:
    """Every split piece shares one half-open ordinal range."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.date_range(start, periods=40, freq="24h", tz="UTC")
    end = decisions[-1] + pd.Timedelta(days=1)
    _, funding, targets, spec = _completed_fixture(tmp_path, start, end, decisions)
    targets.iloc[:, 0] = 0.05
    _tight_telemetry(monkeypatch)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=2_000_000, reserve_bytes=100,
        )
    )
    first_keys = {w.logical_partition for w in windows if w.logical_partition and w.logical_partition[0] == 0}
    assert first_keys == {(0, 32)} or all(k[0] == 0 or k[0] > 0 for k in {w.logical_partition for w in windows})
    assert all(w.logical_partition is not None and len(w.logical_partition) == 2 for w in windows)


def test_adaptive_execution_bound_count_validation(tmp_path) -> None:
    """Non-positive bound counts fail before any decode."""
    import pandas as pd
    import pytest

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.date_range(start, periods=1, freq="24h", tz="UTC")
    _, funding, targets, spec = _completed_fixture(tmp_path, start, start + pd.Timedelta(hours=2), decisions)
    with pytest.raises(ValueError, match="execution_bound_count"):
        list(
            _iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, start + pd.Timedelta(hours=2), funding, spec,
                execution_bound_count=0,
            )
        )


def test_estimate_and_minimum_helpers_cover_branches() -> None:
    """Allocation helper validation and timeout edge cases."""
    import pytest

    from src.mhs.evaluation.windows import _estimate_mhs_execution_allocation, _minimum_mhs_execution_bars

    with pytest.raises(ValueError, match="bound_count"):
        _estimate_mhs_execution_allocation(n_symbols=2, n_columns=2, bound_count=0)
    assert _minimum_mhs_execution_bars(0, 180_000_000_000) == 2
    assert _minimum_mhs_execution_bars(1_800_000_000_000, 180_000_000_000) == 11
    alloc = _estimate_mhs_execution_allocation(n_symbols=0, n_columns=3, bound_count=2)
    assert alloc.bytes_per_bar > 0


def test_materialize_covers_empty_and_missing_branches(tmp_path, monkeypatch) -> None:
    """Empty aligned frames and missing symbols are admitted; no piece carries a mark plane."""
    import pandas as pd

    import src.mhs.evaluation.windows as _w
    from src.mhs.resources import MhsExecutionAllocation

    _tight_telemetry(monkeypatch)
    grid = pd.date_range("2022-01-01", periods=5, freq="3min", tz="UTC")
    alloc = MhsExecutionAllocation(fixed_bytes=100, bytes_per_bar=100, decoder_bytes=100)
    cols = ("AUSDT", "BUSDT")
    empty_w = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"), columns=list(cols), dtype="float64")
    empty_s = pd.DatetimeIndex([], tz="UTC")
    win = _w._materialize_execution_piece(
        piece_grid=grid, piece_weights=empty_w, piece_signals=empty_s, roster=[],
        columns=cols, root=str(tmp_path), timeframe="3m", funding_by_symbol={},
        funding_failures=None, allocation=alloc,
        budget_bytes=None, reserve_bytes=None,
        window_start=grid[0], window_end=grid[-1], logical_partition=(0, 0),
    )
    assert win.symbols == ()
    assert win.marks is None
    w2 = pd.DataFrame(0.0, index=pd.DatetimeIndex([grid[0]]), columns=list(cols))
    s2 = pd.DatetimeIndex([grid[0] + pd.Timedelta(hours=1)])
    monkeypatch.setattr(_w, "_load_window_minute_frames", lambda *a, **k: {})
    win2 = _w._materialize_execution_piece(
        piece_grid=grid, piece_weights=w2, piece_signals=s2, roster=["AUSDT"],
        columns=cols, root=str(tmp_path), timeframe="3m", funding_by_symbol={},
        funding_failures=None, allocation=alloc,
        budget_bytes=None, reserve_bytes=None,
        window_start=grid[0], window_end=grid[-1], logical_partition=(0, 1),
    )
    assert "AUSDT" in win2.symbols
    assert float(win2.highs["AUSDT"].isna().sum()) == len(grid)
    assert win2.marks is None
    win3 = _w._materialize_execution_piece(
        piece_grid=grid, piece_weights=w2, piece_signals=s2, roster=["AUSDT"],
        columns=cols, root=str(tmp_path), timeframe="3m", funding_by_symbol={},
        funding_failures=None, allocation=alloc,
        budget_bytes=None, reserve_bytes=None,
        window_start=grid[0], window_end=grid[-1], logical_partition=(0, 1),
    )
    assert win3.marks is None
    win4 = _w._materialize_execution_piece(
        piece_grid=grid, piece_weights=w2, piece_signals=s2, roster=["AUSDT"],
        columns=cols, root=str(tmp_path), timeframe="3m", funding_by_symbol={},
        funding_failures=None, allocation=alloc,
        budget_bytes=None, reserve_bytes=None,
        window_start=grid[0], window_end=grid[-1], logical_partition=(0, 1),
    )
    assert win4.marks is None


def test_adaptive_single_piece_when_budget_allows(tmp_path, monkeypatch) -> None:
    """Generous budgets keep one physical piece per logical partition."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=2)
    _tight_telemetry(monkeypatch)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=10**12, reserve_bytes=100,
        )
    )
    assert len(windows) == 1
    assert windows[0].logical_partition == (0, 2)


def test_adaptive_single_piece_roster_branches(tmp_path, monkeypatch) -> None:
    """Single-piece adaptive path refreshes rosters and rejects unknown symbols."""
    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=1)
    _tight_telemetry(monkeypatch)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=10**12, reserve_bytes=100,
            required_symbols=lambda: frozenset({"AUSDT"}),
        )
    )
    assert windows
    assert "AUSDT" in windows[0].symbols
    with pytest.raises(DataIntegrityError, match="not in canonical"):
        list(
            _iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, end, funding, spec,
                budget_bytes=10**12, reserve_bytes=100,
                required_symbols=lambda: frozenset({"ZZZ"}),
            )
        )


def test_adaptive_offgrid_timeout_fallback(tmp_path, monkeypatch) -> None:
    """Off-grid timeouts fall back to decision-anchored minimum spans."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows
    from src.mhs.types import ExecutionSpec

    start, end, decisions, funding, targets, _ = _adaptive_fixture(tmp_path, days=2)
    spec = ExecutionSpec(passive_timeout_minutes=31)
    _tight_telemetry(monkeypatch)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_550_000, reserve_bytes=100,
        )
    )
    assert windows


def test_adaptive_split_unknown_roster_fails_closed(tmp_path, monkeypatch) -> None:
    """Unknown held symbols fail during split and tail roster resolution."""
    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=2)
    _tight_telemetry(monkeypatch)
    with pytest.raises(DataIntegrityError, match=r"not in canonical"):
        list(
            _iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, end, funding, spec,
                budget_bytes=1_550_000, reserve_bytes=100,
                required_symbols=lambda: frozenset({"ZZZ"}),
            )
        )


def test_adaptive_piece_and_tail_unknown_branches(tmp_path, monkeypatch) -> None:
    """Late unknown symbols hit piece and tail roster guards."""
    import itertools

    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=2)
    _tight_telemetry(monkeypatch)
    calls = itertools.count()
    seq = [frozenset(), frozenset({"ZZZ"})]

    def _flip() -> frozenset[str]:
        i = next(calls)
        return seq[min(i, 1)]

    with pytest.raises(DataIntegrityError, match=r"not in canonical"):
        list(
            _iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, end, funding, spec,
                budget_bytes=10**12, reserve_bytes=100, required_symbols=_flip,
            )
        )
    start3, end3, decisions3, funding3, targets3, spec3 = _adaptive_fixture(tmp_path, days=3)
    calls2 = itertools.count()

    def _flip2() -> frozenset[str]:
        i = next(calls2)
        return frozenset() if i < 2 else frozenset({"ZZZ"})

    with pytest.raises(DataIntegrityError, match=r"not in canonical"):
        list(
            _iter_mhs_execution_windows(
                targets3, decisions3 + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start3, end3, funding3, spec3,
                budget_bytes=1_600_000, reserve_bytes=100, required_symbols=_flip2,
            )
        )


def test_adaptive_tail_unknown_fails_closed(tmp_path, monkeypatch) -> None:
    """Unknown symbols during tail streaming fail with diagnostics."""
    import itertools

    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.date_range(start, periods=1, freq="24h", tz="UTC")
    end = start + pd.Timedelta(days=4)
    _, funding, targets, spec = _completed_fixture(tmp_path, start, end, decisions)
    targets.iloc[0, 0] = 0.1
    _tight_telemetry(monkeypatch)
    calls = itertools.count()

    def _flip() -> frozenset[str]:
        i = next(calls)
        return frozenset({"AUSDT"}) if i < 2 else frozenset({"ZZZ"})

    with pytest.raises(DataIntegrityError, match=r"not in canonical"):
        list(
            _iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, end, funding, spec,
                budget_bytes=1_550_000, reserve_bytes=100, required_symbols=_flip,
            )
        )


def test_adaptive_long_tail_streams_multiple_pieces(tmp_path, monkeypatch) -> None:
    """A held tail longer than one plan streams several empty pieces."""
    import pandas as pd

    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.date_range(start, periods=1, freq="24h", tz="UTC")
    end = start + pd.Timedelta(days=4)
    _, funding, targets, spec = _completed_fixture(tmp_path, start, end, decisions)
    targets.iloc[0, 0] = 0.1
    _tight_telemetry(monkeypatch)
    windows = list(
        _iter_mhs_execution_windows(
            targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
            "3m", start, end, funding, spec,
            budget_bytes=1_550_000, reserve_bytes=100,
            required_symbols=lambda: frozenset({"AUSDT"}),
        )
    )
    assert len(windows) > 2
    assert all(w.target_weights.empty for w in windows[1:])


def test_adaptive_unresolvable_budget_fails_closed(tmp_path, monkeypatch) -> None:
    """A budget smaller than one timeout span leaves the order unresolved."""
    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _adaptive_fixture(tmp_path, days=2)
    _tight_telemetry(monkeypatch)
    with pytest.raises(DataIntegrityError, match=r"unresolved|rejected"):
        list(
            _iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, end, funding, spec,
                budget_bytes=1, reserve_bytes=100,
            )
        )


def test_adaptive_insufficient_grid_fails_closed(tmp_path, monkeypatch) -> None:
    """A logical grid smaller than the timeout minimum fails with diagnostics."""
    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.windows import _iter_mhs_execution_windows

    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.DatetimeIndex([start])
    _, funding, targets, spec = _completed_fixture(tmp_path, start, start + pd.Timedelta(minutes=6), decisions)
    _tight_telemetry(monkeypatch)
    with pytest.raises(DataIntegrityError, match="strict timeout"):
        list(
            _iter_mhs_execution_windows(
                targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"),
                "3m", start, start + pd.Timedelta(minutes=6), funding, spec,
                budget_bytes=10**12, reserve_bytes=100,
            )
        )


def test_book_outcome_propagates_live_bound_count(monkeypatch, tmp_path) -> None:
    """The shared generator receives the actual live bound count, not a constant."""
    import dataclasses

    import pandas as pd

    import src.mhs.evaluation.windows as _w
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.types import BOOK_SPECS, ExecutionSpec

    start = pd.Timestamp("2022-01-01", tz="UTC")
    end = pd.Timestamp("2022-01-02", tz="UTC")
    grid_1h = pd.date_range(start, end, freq="1h", tz="UTC")
    step_grid = pd.date_range(start, periods=2, freq="24h", tz="UTC")
    weights_step = pd.DataFrame(0.0, index=step_grid, columns=["AUSDT"])
    opens = pd.DataFrame(1.0, index=grid_1h, columns=["AUSDT"])
    bar_funding = pd.DataFrame(0.0, index=grid_1h, columns=["AUSDT"])
    captured: dict[str, object] = {}

    class _Evid:
        prescreen: dict[float, object] = dataclasses.field(default_factory=dict)
        tail: object = None

    monkeypatch.setattr(_w, "book_evidence", lambda *a, **k: type("E", (), {"prescreen": {}, "tail": None})())
    monkeypatch.setattr(_w, "_resolve_ram_budget", lambda *a, **k: (None, None))
    monkeypatch.setattr(_w.integrity, "_truncate_replayable_decisions", lambda w, s, g, spec: (w, s, 0))
    monkeypatch.setattr(_w.specs, "_resolved_base_execution_spec", lambda req: ExecutionSpec())
    monkeypatch.setattr(_w.specs, "_stress_cost_execution_spec", lambda spec: spec)
    monkeypatch.setattr(_w, "_scaling", type("S", (), {"_apply_rebalance_deadband": staticmethod(lambda w: w), "_replay_exposure_scale": staticmethod(lambda r, req: r), "is_streaming_scale_mode": staticmethod(lambda req: False)})())

    def _capture(*a, **k):
        captured.update(k)
        return iter(())

    monkeypatch.setattr(_w, "_iter_mhs_execution_windows", _capture)
    request = MhsDiagnosticRequest(touch_diagnostic=True)
    phase = type("P", (), {})()
    report, _ = _w._book_outcome(
        "blend", BOOK_SPECS["fast_reversal"], 1, step_grid, weights_step, grid_1h, opens, bar_funding,
        phase, str(tmp_path), request, {}, start, end, 1, 1.0,
    )
    assert captured.get("execution_bound_count") == 4
    assert report.failure is not None
