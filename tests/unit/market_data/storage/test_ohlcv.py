from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.market_data.storage.ohlcv import merge_ohlcv_frames, write_ohlcv

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _ms(index: pd.DatetimeIndex) -> pd.Series:
    return (index - _EPOCH) // pd.Timedelta("1ms")


class TestWriteOhlcvTakerCanonicalColumns:
    def test_rest_only_1h_frame_gains_canonical_taker_columns(self, tmp_path: Path) -> None:
        # Given: a freshly listed symbol with REST rows only (no Vision month)
        idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": _ms(idx),
            "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0],
            "volume": [1.0, 1.0], "quote_vol": [1.0, 1.0],
            "taker_buy_base_volume": [5.0, 6.0], "taker_buy_quote_volume": [500.0, 600.0],
        })
        path = tmp_path / "1h" / "NEWUSDT.parquet"
        # When
        write_ohlcv(path, df, timeframe="1h")
        # Then: the columns the MHS panel loader requests are persisted
        out = pd.read_parquet(path)
        assert list(out["taker_buy_base"]) == [5.0, 6.0]
        assert list(out["taker_buy_quote"]) == [500.0, 600.0]

    def test_1m_frame_with_both_taker_forms_still_writes_canonical_layout(self, tmp_path: Path) -> None:
        # Given: a 1m frame carrying both forms
        idx = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        df = pd.DataFrame({
            "timestamp": _ms(idx),
            "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0],
            "volume": [1.0, 1.0], "quote_vol": [1.0, 1.0],
            "taker_buy_base_volume": [5.0, 6.0], "taker_buy_quote_volume": [500.0, 600.0],
        })
        path = tmp_path / "1m" / "X.parquet"
        # When
        write_ohlcv(path, df, timeframe="1m")
        # Then: 1m keeps its historical suffixed-only layout
        out = pd.read_parquet(path)
        assert "taker_buy_base" not in out.columns
        assert list(out["taker_buy_base_volume"]) == [5.0, 6.0]


class TestWriteOhlcv1m:
    def test_1m_layout_is_canonical_order_and_dtypes(self, tmp_path: Path) -> None:
        # SC-STORE-01: a 1m frame with the historical column order is persisted
        # byte-equivalent with the canonical futures 1m lake.
        idx = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
        df = pd.DataFrame({
            "timestamp": _ms(idx),
            "open": [100.0] * 3, "high": [101.0] * 3, "low": [99.0] * 3,
            "close": [100.5] * 3, "volume": [10.0] * 3,
            "quote_vol": [1000.0] * 3,
            "taker_buy_base_volume": [5.0] * 3,
            "taker_buy_quote_volume": [500.0] * 3,
        })
        path = tmp_path / "1m" / "BTCUSDT.parquet"
        write_ohlcv(path, df, timeframe="1m")

        out = pd.read_parquet(path)
        assert list(out.columns) == [
            "timestamp", "open", "high", "low", "close", "volume",
            "taker_buy_base_volume", "taker_buy_quote_volume", "quote_vol",
        ]
        assert str(out["open"].dtype) == "float32"
        assert str(out["timestamp"].dtype) == "int64"
        assert out["timestamp"].is_monotonic_increasing
        assert "datetime" not in out.columns

    def test_1m_renames_and_fills_missing_columns(self, tmp_path: Path) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="1min", tz="UTC")
        df = pd.DataFrame({
            "timestamp": _ms(idx),
            "open": [100.0] * 2, "high": [101.0] * 2, "low": [99.0] * 2,
            "close": [100.5] * 2, "volume": [10.0] * 2,
            "quote_volume": [1000.0] * 2,
            "taker_buy_base": [5.0] * 2,
            "taker_buy_quote": [500.0] * 2,
        })
        path = tmp_path / "m.parquet"
        write_ohlcv(path, df, timeframe="1m")
        out = pd.read_parquet(path)
        assert "quote_vol" in out.columns
        assert "taker_buy_base_volume" in out.columns
        assert "taker_buy_quote_volume" in out.columns
        assert len(out) == 2


class TestWriteOhlcvPreservesFuturesLayout:
    def test_non_1m_preserves_column_order_and_float32_ohlc(self, tmp_path: Path) -> None:
        # SC-STORE-02: the canonical futures 1h layout (both taker-buy naming
        # generations) survives the store migration unchanged.
        idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": _ms(idx),
            "open": [45505.0, 45539.0], "high": [45638.0, 45539.0],
            "low": [45338.0, 44861.0], "close": [45539.0, 44926.0],
            "volume": [10308.0, 24579.0],
            "quote_vol": [4.69e8, 1.11e9],
            "taker_buy_base": [5147.0, 10527.0],
            "taker_buy_quote": [2.34e8, 4.75e8],
            "taker_buy_base_volume": [0.0, 0.0],
            "taker_buy_quote_volume": [0.0, 0.0],
        })
        path = tmp_path / "BTCUSDT.parquet"
        write_ohlcv(path, df, timeframe="1h")
        out = pd.read_parquet(path)
        assert list(out.columns) == list(df.columns)
        assert str(out["open"].dtype) == "float32"
        assert str(out["volume"].dtype) == "float64"

    def test_write_is_atomic_and_deterministic(self, tmp_path: Path) -> None:
            idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
            df = pd.DataFrame({
                "timestamp": _ms(idx),
                "open": [100.0] * 4, "high": [101.0] * 4, "low": [99.0] * 4,
                "close": [100.5] * 4, "volume": [10.0] * 4,
                "quote_vol": [1000.0] * 4,
            })
            path = tmp_path / "BTCUSDT.parquet"
            write_ohlcv(path, df, timeframe="1h")
            first = path.read_bytes()
            write_ohlcv(path, df, timeframe="1h")
            assert path.read_bytes() == first
            assert sorted(p.name for p in tmp_path.iterdir()) == ["BTCUSDT.parquet"]

    def test_empty_frame_is_noop(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.parquet"
        write_ohlcv(path, pd.DataFrame(), timeframe="1h")
        assert not path.exists()


class TestMergeOhlcvFrames:
    def test_dedupes_by_timestamp_keeping_last_and_sorts(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        first = pd.DataFrame({
            "timestamp": _ms(idx),
            "open": [1.0, 2.0, 3.0], "close": [1.0, 2.0, 3.0],
        })
        second = pd.DataFrame({
            "timestamp": _ms(idx[[2, 0]]),
            "open": [30.0, 10.0], "close": [30.0, 10.0],
        })
        merged = merge_ohlcv_frames([first, second])
        assert list(merged["timestamp"]) == sorted(merged["timestamp"])
        assert len(merged) == 3
        last = merged[merged["timestamp"] == merged["timestamp"].iloc[-1]].iloc[0]
        assert last["open"] == 30.0


def test_write_ohlcv_temp_file_is_hidden_and_process_unique(tmp_path, monkeypatch) -> None:
    import os
    import pathlib
    import threading
    import pandas as pd
    from src.market_data.storage.ohlcv import write_ohlcv

    replaced: list[str] = []
    original_replace = pathlib.Path.replace

    def _spy(self, target):
        replaced.append(self.name)
        return original_replace(self, target)

    monkeypatch.setattr(pathlib.Path, "replace", _spy)
    idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": (idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1ms"),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })

    write_ohlcv(tmp_path / "BTCUSDT.parquet", df, timeframe="1h")

    assert replaced == [f".BTCUSDT.parquet.{os.getpid()}.{threading.get_ident()}.tmp"]
    assert not replaced[0].endswith(".parquet")


def test_is_temp_artifact_flags_legacy_tmp_parquet_names() -> None:
    from src.market_data.storage.ohlcv import is_temp_artifact

    assert is_temp_artifact("BTCUSDT.tmp.parquet") is True
    assert is_temp_artifact("BTCUSDT.prune.tmp.parquet") is True
    assert is_temp_artifact("BTCUSDT.parquet") is False
    assert is_temp_artifact("TMPUSDT.parquet") is False



def test_write_ohlcv_uses_31_day_row_groups_for_intraday_timeframes(tmp_path) -> None:
    import pandas as pd
    import pyarrow.parquet as pq
    from src.market_data.storage.ohlcv import write_ohlcv
    n = 480 * 31 + 5
    stamps = pd.date_range("2024-01-01", periods=n, freq="3min", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": (stamps.asi8 // 10**6 if stamps.unit == "ns" else stamps.as_unit("ms").asi8),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "quote_vol": 1.0,
    })
    three = tmp_path / "3m" / "X.parquet"
    write_ohlcv(three, frame, timeframe="3m")
    meta = pq.ParquetFile(three).metadata
    assert meta.num_row_groups == 2
    assert meta.row_group(0).num_rows == 480 * 31
    back = pd.read_parquet(three)
    assert back["timestamp"].tolist() == frame["timestamp"].tolist()
    # a timeframe without an intraday bars-per-day entry keeps the writer default (single group here)
    daily = tmp_path / "1d" / "X.parquet"
    write_ohlcv(daily, frame.head(10), timeframe="1d")
    assert pq.ParquetFile(daily).metadata.num_row_groups == 1


