from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.storage.loaders import (
    RESEARCH_TIMEFRAMES,
    load_funding_rates,
    load_ohlcv_1h_as,
    load_ohlcv_1h_as_4h,
    timeframe_period,
    timeframe_scale_factor,
    validate_timeframe,
)


def _write_1h_parquet(path: Path, n: int) -> None:
    """Write a gapless 1h UTC kline parquet with ``n`` bars."""
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    ms = (index - epoch) // pd.Timedelta(milliseconds=1)
    df = pd.DataFrame(
        {
            "timestamp": ms,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "quote_vol": 1000.0,
            "taker_buy_quote": 500.0,
        }
    )
    df.to_parquet(path)


def test_load_funding_rates_rejects_missing_rate_column(tmp_path) -> None:
    path = tmp_path / "funding.parquet"
    pd.DataFrame({"datetime": ["2024-01-01T00:00:00Z"]}).to_parquet(path)

    with pytest.raises(DataIntegrityError, match="funding_rate"):
        load_funding_rates(path)


def test_load_funding_rates_accepts_timestamp_column(tmp_path) -> None:
    path = tmp_path / "funding.parquet"
    pd.DataFrame({"timestamp": [1704067200000], "funding_rate": [0.0001]}).to_parquet(path)

    result = load_funding_rates(path)

    assert result.index[0] == pd.Timestamp("2024-01-01", tz="UTC")


def test_load_funding_rates_rejects_missing_timestamp_columns(tmp_path) -> None:
    path = tmp_path / "funding.parquet"
    pd.DataFrame({"funding_rate": [0.0001]}).to_parquet(path)

    with pytest.raises(DataIntegrityError, match="timestamp"):
        load_funding_rates(path)


def test_load_funding_rates_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(DataIntegrityError, match="does not exist"):
        load_funding_rates(tmp_path / "missing.parquet")


class TestTimeframeScaleFactor:
    def test_4h_reference_is_exactly_one(self) -> None:
        # TIS-03: the 4h reference factor is exactly 1.0, so every existing 4h
        # bar count is preserved under round(x * 1.0) == x for positive ints.
        assert timeframe_scale_factor("4h") == 1.0
        assert timeframe_scale_factor("4h", reference_timeframe="4h") == 1.0

    def test_factor_is_reference_period_over_target_period(self) -> None:
        # The factor is exactly the reference/target period ratio for every
        # research timeframe, both directions (up-scaling and down-scaling).
        for timeframe in RESEARCH_TIMEFRAMES:
            assert timeframe_scale_factor(timeframe) == (
                timeframe_period("4h") / timeframe_period(timeframe)
            )
            assert timeframe_scale_factor(
                timeframe, reference_timeframe="1h",
            ) == (timeframe_period("1h") / timeframe_period(timeframe))

    def test_unknown_timeframe_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="timeframe must be one of"):
            timeframe_scale_factor("90m")
        with pytest.raises(ValueError, match="timeframe must be one of"):
            timeframe_scale_factor("4h", reference_timeframe="1w")


class TestLoadOhlcv1hAs:
    def test_8h_resample_exact_grid(self, tmp_path) -> None:
        # 8h_resample_exact_grid: resampling a synthetic 1h parquet to 8h
        # yields an exact 8h grid and drops the trailing partial bucket.
        path = tmp_path / "btc.parquet"
        _write_1h_parquet(path, n=73)  # 73 hourly bars -> 9 full 8h buckets + 1 partial
        resampled = load_ohlcv_1h_as(path, "8h")
        assert set(resampled.index.to_series().diff().dropna().unique()) == {
            pd.Timedelta(hours=8)
        }
        assert len(resampled) == 9
        assert resampled.index[0] == pd.Timestamp("2024-01-01", tz="UTC")

    def test_every_research_timeframe_divides_whole_source_bars(self, tmp_path) -> None:
        # 8h_resample_exact_grid: every admitted bucket is exactly its nominal
        # period and every source bar lands in exactly one full bucket.
        path = tmp_path / "btc.parquet"
        _write_1h_parquet(path, n=96)
        for timeframe in RESEARCH_TIMEFRAMES:
            resampled = load_ohlcv_1h_as(path, timeframe)
            period = timeframe_period(timeframe)
            diffs = set(resampled.index.to_series().diff().dropna().unique())
            assert diffs == {period}, timeframe
            assert resampled.index[0] == pd.Timestamp("2024-01-01", tz="UTC")

    def test_invalid_timeframe_fails_closed(self, tmp_path) -> None:
        # invalid_timeframe_fails_closed: an unadmitted bucket raises ValueError
        # and never silently produces a rounded/partial resample.
        path = tmp_path / "btc.parquet"
        _write_1h_parquet(path, n=48)
        for invalid in ("3h", "45m", "5h", "90m", "1w"):
            with pytest.raises(ValueError, match="timeframe"):
                load_ohlcv_1h_as(path, invalid)
        with pytest.raises(ValueError, match="timeframe"):
            validate_timeframe("3h")

    def test_4h_wrapper_unchanged(self, tmp_path) -> None:
        # 4h_wrapper_unchanged: the legacy wrapper is byte-identical to the
        # direct generalized 4h call for the same input.
        path = tmp_path / "btc.parquet"
        _write_1h_parquet(path, n=48)
        legacy = load_ohlcv_1h_as_4h(path)
        direct = load_ohlcv_1h_as(path, "4h")
        pd.testing.assert_frame_equal(legacy, direct)

    def test_1h_passthrough_and_gap_detection_are_preserved(self, tmp_path) -> None:
        # The 1h passthrough retains every source bar and the same fail-closed
        # DataIntegrityError gap check applies on the 1h source.
        path = tmp_path / "btc.parquet"
        _write_1h_parquet(path, n=24)
        assert len(load_ohlcv_1h_as(path, "1h")) == 24

        gappy = tmp_path / "gappy.parquet"
        index = pd.DatetimeIndex(
            [pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=h) for h in range(8)]
            + [pd.Timestamp("2024-01-01 10:00", tz="UTC")]
        )
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        ms = (index - epoch) // pd.Timedelta(milliseconds=1)
        pd.DataFrame(
            {"timestamp": ms, "open": 100.0, "high": 101.0, "low": 99.0,
             "close": 100.5, "volume": 1.0},
        ).to_parquet(gappy)
        with pytest.raises(DataIntegrityError, match="missing 1h bars"):
            load_ohlcv_1h_as(gappy, "4h")
