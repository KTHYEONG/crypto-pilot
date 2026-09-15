"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import src.mhs.evaluation.windows as windows


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


def test_mark_frame_path_keyed_cache(tmp_path) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.mhs.marks import _get_symbol_mark_frame_for_path

    root1 = tmp_path / "root1"
    root2 = tmp_path / "root2"
    root1.mkdir()
    root2.mkdir()

    f1 = root1 / "mark_BTC.parquet"
    f2 = root2 / "mark_BTC.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({"timestamp": [1000], "close": [100.0], "open": [100.0], "high": [100.0], "low": [100.0]})), str(f1))
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({"timestamp": [1000], "close": [200.0], "open": [200.0], "high": [200.0], "low": [200.0]})), str(f2))

    df1 = _get_symbol_mark_frame_for_path("BTC", "1h", str(f1))
    df2 = _get_symbol_mark_frame_for_path("BTC", "1h", str(f2))
    assert not df1.empty
    assert float(df1["close"].iloc[0]) == 100.0
    assert float(df2["close"].iloc[0]) == 200.0


def test_window_ipc_errors(tmp_path) -> None:
    import pytest
    from src.mhs.evaluation import DataIntegrityError
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
    windows = list(_iter_mhs_execution_windows(weights, idx, str(tmp_path), '3m', idx[0], idx[0]+pd.Timedelta(minutes=9), {}, 'ohlcv_close_fallback', ExecutionSpec(), funding_failures={'MISSUSDT': 'missing'}))
    assert windows[0].symbols == ('MISSUSDT',)
    assert windows[0].closes['MISSUSDT'].isna().all()
    assert windows[0].quote_volumes['MISSUSDT'].isna().all()
