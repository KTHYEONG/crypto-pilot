from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.ohlcv_store import merge_ohlcv_frames, write_ohlcv

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _ms(index: pd.DatetimeIndex) -> pd.Series:
    return (index - _EPOCH) // pd.Timedelta("1ms")


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
        assert not list(tmp_path.glob("*.tmp.parquet"))

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
