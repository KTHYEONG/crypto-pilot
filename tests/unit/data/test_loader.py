from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataIntegrityError, load_ohlcv_4h

BTC_PATH = Path("data/futures/ohlcv/1h/BTCUSDT.parquet")

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _ms_from_index(idx: pd.DatetimeIndex) -> pd.Series:
    """DatetimeIndex -> integer milliseconds since epoch (robust to ns/us dtype)."""
    return (idx - _EPOCH) // pd.Timedelta("1ms")


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
        assert "taker_buy_ratio" in df.columns

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

    def test_resamples_quote_weighted_taker_buy_ratio(self, tmp_path: Path) -> None:
        # SC-FLOW-01: one complete 4h bucket built from four 1h rows with
        # sum(taker_buy_quote)=52 and sum(quote_vol)=100 -> ratio 0.52.
        idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": _ms_from_index(idx),
            "open": [100.0] * 4, "high": [101.0] * 4,
            "low": [99.0] * 4, "close": [100.5] * 4,
            "volume": [10.0] * 4,
            "quote_vol": [25.0, 25.0, 25.0, 25.0],
            "taker_buy_quote": [13.0, 13.0, 13.0, 13.0],
        })
        flow_path = tmp_path / "flow.parquet"
        df.to_parquet(flow_path)
        out = load_ohlcv_4h(flow_path)
        assert len(out) == 1
        assert "taker_buy_ratio" in out.columns
        assert out["taker_buy_ratio"].iloc[0] == pytest.approx(0.52)
        assert out["quote_vol"].iloc[0] == pytest.approx(100.0)
        assert out["taker_buy_quote"].iloc[0] == pytest.approx(52.0)

    def test_zero_quote_volume_ratio_is_nan(self, tmp_path: Path) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": _ms_from_index(idx),
            "open": [100.0] * 4, "high": [101.0] * 4,
            "low": [99.0] * 4, "close": [100.5] * 4,
            "volume": [10.0] * 4,
            "quote_vol": [0.0] * 4,
            "taker_buy_quote": [10.0] * 4,
        })
        zero_path = tmp_path / "zero.parquet"
        df.to_parquet(zero_path)
        out = load_ohlcv_4h(zero_path)
        assert np.isnan(out["taker_buy_ratio"].iloc[0])

    def test_missing_flow_columns_ratio_is_nan(self, tmp_path: Path) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": _ms_from_index(idx),
            "open": [100.0] * 4, "high": [101.0] * 4,
            "low": [99.0] * 4, "close": [100.5] * 4,
            "volume": [10.0] * 4,
        })
        plain_path = tmp_path / "plain.parquet"
        df.to_parquet(plain_path)
        out = load_ohlcv_4h(plain_path)
        assert "taker_buy_ratio" in out.columns
        assert out["taker_buy_ratio"].isna().all()

    def test_fallback_taker_buy_quote_volume_column(self, tmp_path: Path) -> None:
        # Prefer taker_buy_quote; fall back to taker_buy_quote_volume only where
        # the former is null.
        idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        df = pd.DataFrame({
            "timestamp": _ms_from_index(idx),
            "open": [100.0] * 4, "high": [101.0] * 4,
            "low": [99.0] * 4, "close": [100.5] * 4,
            "volume": [10.0] * 4,
            "quote_vol": [100.0] * 4,
            "taker_buy_quote": [float("nan")] * 4,
            "taker_buy_quote_volume": [52.0] * 4,
        })
        fb_path = tmp_path / "fallback.parquet"
        df.to_parquet(fb_path)
        out = load_ohlcv_4h(fb_path)
        assert out["taker_buy_ratio"].iloc[0] == pytest.approx(0.52)
