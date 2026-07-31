from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import DataIntegrityError, load_ohlcv_4h

BTC_PATH = Path("data/futures/ohlcv/1h/BTCUSDT.parquet")


class TestLoadOhlcv4h:
    def test_integrity_and_dtype(self) -> None:
        # BTC_PATH is a live cache that grows as scripts/collect_data.py runs, so this
        # asserts structural properties rather than an exact row count / end timestamp.
        df = load_ohlcv_4h(BTC_PATH)
        assert len(df) >= 9354, f"expected at least the 2022-04..2026-07 baseline, got {len(df)}"
        assert str(df.index[0]) == "2022-04-01 00:00:00+00:00"
        for c in ("open", "high", "low", "close"):
            assert df[c].dtype == "float64", f"{c} dtype is {df[c].dtype}"
        assert df.index.is_monotonic_increasing
        assert not df.index.has_duplicates

    def test_missing_bar_raises(self, tmp_path: Path) -> None:
        df = pd.read_parquet(BTC_PATH)
        df_bad = df.drop(1000).reset_index(drop=True)
        bad_path = tmp_path / "bad.parquet"
        df_bad.to_parquet(bad_path)
        with pytest.raises(DataIntegrityError, match="missing 1h bars"):
            load_ohlcv_4h(bad_path)

    def test_duplicate_timestamp_raises(self, tmp_path: Path) -> None:
        df = pd.read_parquet(BTC_PATH)
        dup = pd.concat([df, df.iloc[[100]]], ignore_index=True)
        dup_path = tmp_path / "dup.parquet"
        dup.to_parquet(dup_path)
        with pytest.raises(DataIntegrityError, match="duplicate"):
            load_ohlcv_4h(dup_path)

    def test_float32_input_becomes_float64(self, tmp_path: Path) -> None:
        df = pd.read_parquet(BTC_PATH)
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype("float32")
        float32_path = tmp_path / "f32.parquet"
        df.to_parquet(float32_path)
        result = load_ohlcv_4h(float32_path)
        for c in ("open", "high", "low", "close"):
            assert result[c].dtype == "float64"
