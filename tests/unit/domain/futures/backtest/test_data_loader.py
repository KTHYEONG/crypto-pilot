from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.exchange.binance_client import BinanceKlinePermanentError
from src.domain.futures.backtest.data_loader import (
    DataCollector,
    _normalize_funding_frame,
    summarize_ohlcv_collection_integrity,
)


def test_ensure_ohlcv_data_when_permanent_api_error_records_negative_cache_and_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = DataCollector()
    monkeypatch.setattr(collector, "_load_cache", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(collector, "_save_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(collector, "_normalize_df", lambda df: df)

    def _raise_permanent(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise BinanceKlinePermanentError(
            symbol="GAIBUSDT",
            timeframe="1h",
            http_code=400,
            start_time_ms=1,
            end_time_ms=2,
            url="https://fapi.binance.com/fapi/v1/klines",
        )

    monkeypatch.setattr(collector.client, "fetch_ohlcv_with_taker", _raise_permanent)
    collector.ensure_ohlcv_data("GAIBUSDT", "1h", "2026-05-01", "2026-05-10")

    meta = collector._load_meta().get(collector._meta_key("GAIBUSDT", "1h"), {})
    failure = meta.get("last_permanent_failure")
    assert isinstance(failure, dict)
    assert int(failure["http_code"]) == 400


def test_ensure_ohlcv_data_persists_vision_partial_data_when_recent_gap_api_fails_permanently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = DataCollector()
    monkeypatch.setattr(collector, "_load_cache", lambda *_args, **_kwargs: pd.DataFrame())
    saved: dict[str, pd.DataFrame] = {}

    def _save_cache(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        saved[f"{symbol}:{timeframe}"] = df.copy()

    monkeypatch.setattr(collector, "_save_cache", _save_cache)

    def _vision_df(*_args: object, **_kwargs: object) -> pd.DataFrame:
        return pd.DataFrame(
            [
                [1711929600000, "1", "2", "0.5", "1.2", "10", 0, "10", 0, "4", "5", "0"],
            ]
        )

    monkeypatch.setattr(
        "src.core.exchange.binance_vision.BinanceVisionDownloader.fetch_klines_archive_monthly",
        _vision_df,
    )

    def _raise_permanent(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise BinanceKlinePermanentError(
            symbol="GAIBUSDT",
            timeframe="1h",
            http_code=400,
            start_time_ms=1,
            end_time_ms=2,
            url="https://fapi.binance.com/fapi/v1/klines",
        )

    monkeypatch.setattr(collector.client, "fetch_ohlcv_with_taker", _raise_permanent)

    collector.ensure_ohlcv_data("GAIBUSDT", "1h", "2024-01-01", "2026-05-10")
    key = "GAIBUSDT:1h"
    assert key in saved
    assert not saved[key].empty


def test_summarize_ohlcv_collection_integrity_counts_missing_bar_ratio_and_duplicates() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T03:00:00Z",
                ],
                utc=True,
            ),
            "open": [1.0, 1.1, 1.1, 1.2],
            "high": [1.2, 1.2, 1.2, 1.3],
            "low": [0.9, 1.0, 1.0, 1.1],
            "close": [1.1, 1.1, 1.1, 1.2],
            "volume": [10.0, 11.0, 11.0, 9.0],
        }
    )
    out = summarize_ohlcv_collection_integrity(
        df,
        timeframe="1h",
        expected_start=pd.Timestamp("2026-01-01T00:00:00Z"),
        expected_end=pd.Timestamp("2026-01-01T04:00:00Z"),
    )
    assert out["duplicate_dt"] > 0
    assert out["gap_count"] > 0
    assert out["missing_bar_ratio"] > 0


def test_summarize_ohlcv_collection_integrity_detects_price_invariant_breaks() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"], utc=True),
            "open": [3.0, 1.0],
            "high": [2.0, 2.0],
            "low": [2.5, 0.8],
            "close": [1.0, 0.7],
            "volume": [1.0, 1.0],
        }
    )
    out = summarize_ohlcv_collection_integrity(df, timeframe="1h")
    assert out["high_lt_low_count"] > 0
    assert out["open_outside_hl_count"] > 0
    assert out["close_outside_hl_count"] > 0


def test_summarize_ohlcv_collection_integrity_detects_negative_volume_and_boundaries() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-01T02:00:00Z", "2026-01-01T03:00:00Z"], utc=True),
            "open": [1.0, 1.0],
            "high": [1.1, 1.1],
            "low": [0.9, 0.9],
            "close": [1.0, 1.0],
            "volume": [-1.0, 2.0],
        }
    )
    out = summarize_ohlcv_collection_integrity(
        df,
        timeframe="1h",
        expected_start=pd.Timestamp("2026-01-01T00:00:00Z"),
        expected_end=pd.Timestamp("2026-01-01T04:00:00Z"),
    )
    assert out["negative_volume_count"] > 0
    assert out["coverage_start_miss"] == 1.0
    assert out["coverage_end_miss"] == 1.0


def test_normalize_funding_frame_drops_duplicate_columns_and_standardizes_schema() -> None:
    raw = pd.DataFrame(
        [[1711929600000, "BTCUSDT", 0.0001, "DUP"]],
        columns=["timestamp", "1", "funding_rate", "1"],
    )
    out = _normalize_funding_frame(raw)
    assert list(out.columns) == ["timestamp", "funding_rate", "datetime"]
    assert len(out) == 1


def test_ensure_funding_data_when_cache_read_fails_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector = DataCollector()
    monkeypatch.setattr("src.domain.futures.backtest.data_loader.FUTURES_DATA_DIR", tmp_path)
    path = tmp_path / "HOOKUSDT_funding.parquet"
    path.write_bytes(b"corrupt")

    monkeypatch.setattr(
        "src.domain.futures.backtest.data_loader.pd.read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("broken parquet")),
    )
    monkeypatch.setattr(
        collector.client,
        "fetch_funding_rate_history",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "src.core.exchange.binance_vision.BinanceVisionDownloader.fetch_funding_rate_monthly",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )

    collector.ensure_funding_data("HOOKUSDT", "2025-01-01", "2025-01-31")
