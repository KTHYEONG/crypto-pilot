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

