from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.exchange.binance_client import BinanceKlinePermanentError
from src.domain.futures.backtest.data_loader import (
    DataCollector,
    _coalesce_metrics_frames,
    _normalize_funding_frame,
    _normalize_metrics_frame,
    merge_metrics_into_ohlcv,
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


def test_ensure_ohlcv_data_backfills_past_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = DataCollector()
    
    # 2023-10-01 ~ 2026-03-31 캐시 데이터 시뮬레이션
    mock_cache = pd.DataFrame(
        {
            "timestamp": [1696118400000, 1774915200000],  # 2023-10-01, 2026-03-31
            "datetime": pd.to_datetime(["2023-10-01T00:00:00Z", "2026-03-31T00:00:00Z"], utc=True),
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.9, 1.9],
            "close": [1.1, 2.1],
            "volume": [10.0, 20.0],
        }
    )
    
    saved_dfs: list[pd.DataFrame] = []
    
    monkeypatch.setattr(collector, "_load_cache", lambda *_args, **_kwargs: mock_cache)
    monkeypatch.setattr(collector, "_save_cache", lambda symbol, tf, df: saved_dfs.append(df))
    monkeypatch.setattr(collector, "_load_meta", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(collector, "_save_meta", lambda *_args, **_kwargs: None)
    
    # fetch_ohlcv_with_taker 호출 인자 기록
    fetched_ranges = []
    
    def mock_fetch(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        fetched_ranges.append((start, end))
        # 2022-10-01 ~ 2023-10-01 사이의 데이터 반환 시뮬레이션
        return pd.DataFrame(
            {
                "timestamp": [1664582400000],  # 2022-10-01
                "datetime": pd.to_datetime(["2022-10-01T00:00:00Z"], utc=True),
                "open": [0.5],
                "high": [0.8],
                "low": [0.4],
                "close": [0.6],
                "volume": [5.0],
            }
        )
        
    monkeypatch.setattr(collector.client, "fetch_ohlcv_with_taker", mock_fetch)
    
    # 2022-10-01 ~ 2026-03-31 수집 요청
    collector.ensure_ohlcv_data("BTCUSDT", "1h", "2022-10-01", "2026-03-31")
    
    # 갭(2022-10-01 ~ 2023-10-01)에 대해 fetch가 호출되었는지 검증
    assert len(fetched_ranges) > 0
    # 첫 fetch 호출 범위가 갭의 시작점과 끝점인지 확인
    assert fetched_ranges[0][0] == "2022-10-01 00:00:00+00:00"
    assert fetched_ranges[0][1] == "2023-10-01 00:00:00+00:00"
    
    # 저장된 통합 데이터 검증
    assert len(saved_dfs) == 1
    combined_df = saved_dfs[0]
    # 최저점이 2022-10-01로 갱신되었는지 확인
    assert combined_df["datetime"].min() == pd.Timestamp("2022-10-01T00:00:00Z", tz="UTC")


def test_ensure_ohlcv_data_clips_to_onboard_date_and_prevents_infinite_backfill_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.domain.futures.universe.storage import SymbolSyncProfile
    mock_profile = SymbolSyncProfile(
        symbol="NEWCOINUSDT",
        onboard_date=pd.Timestamp("2023-07-24").date(),
        delivery_date=None,
        status="TRADING",
    )
    collector = DataCollector()
    monkeypatch.setattr(
        "src.domain.futures.universe.storage._load_symbol_sync_profiles",
        lambda: {"NEWCOINUSDT": mock_profile},
    )

    mock_cache = pd.DataFrame(
        {
            "timestamp": [1690185600000],
            "datetime": pd.to_datetime(["2023-07-24T08:00:00Z"], utc=True),
            "open": [1.0],
            "high": [1.5],
            "low": [0.9],
            "close": [1.1],
            "volume": [10.0],
        }
    )
    fetched_ranges: list[tuple[str, str]] = []

    monkeypatch.setattr(collector, "_load_cache", lambda *_args, **_kwargs: mock_cache)
    monkeypatch.setattr(collector, "_save_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(collector, "_load_meta", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(collector, "_save_meta", lambda *_args, **_kwargs: None)

    def mock_fetch(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        fetched_ranges.append((start, end))
        return pd.DataFrame()

    monkeypatch.setattr(collector.client, "fetch_ohlcv_with_taker", mock_fetch)

    collector.ensure_ohlcv_data("NEWCOINUSDT", "1h", "2022-10-01", "2026-03-31")

    assert fetched_ranges
    for start_str, _ in fetched_ranges:
        start_dt = pd.to_datetime(start_str, utc=True)
        assert start_dt >= pd.to_datetime("2023-07-24T08:00:00Z", utc=True)


def test_coalesce_metrics_frames_preserves_complementary_fields() -> None:
    oi = pd.DataFrame(
        {
            "timestamp": [1711929600000],
            "sum_open_interest": [100.0],
        }
    )
    lsr = pd.DataFrame(
        {
            "timestamp": [1711929600000],
            "long_short_ratio": [1.2],
        }
    )

    out = _coalesce_metrics_frames([oi, lsr], symbol="BTCUSDT")

    assert len(out) == 1
    assert out.loc[0, "sum_open_interest"] == pytest.approx(100.0)
    assert out.loc[0, "long_short_ratio"] == pytest.approx(1.2)


def test_ensure_metrics_data_when_cache_and_sources_empty_returns_canonical_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    collector = DataCollector()
    monkeypatch.setattr("src.domain.futures.backtest.data_loader.FUTURES_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        "src.core.exchange.binance_vision.BinanceVisionDownloader.fetch_range_metrics",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        collector.client,
        "fetch_open_interest_history",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )
    monkeypatch.setattr(
        collector.client,
        "fetch_long_short_ratio_history",
        lambda *_args, **_kwargs: pd.DataFrame(),
    )

    out = collector.ensure_metrics_data("BTCUSDT", "2025-01-01", "2025-01-31")

    assert out.empty
    assert list(out.columns) == list(_normalize_metrics_frame(pd.DataFrame(), symbol="BTCUSDT").columns)


def test_merge_metrics_into_ohlcv_uses_available_at_causally(tmp_path: Path) -> None:
    metrics = pd.DataFrame(
        {
            "timestamp": [1711929600000],
            "datetime": pd.to_datetime(["2024-04-01T00:00:00Z"], utc=True),
            "available_at": pd.to_datetime(["2024-04-01T00:05:00Z"], utc=True),
            "symbol": ["BTCUSDT"],
            "sum_open_interest": [100.0],
            "sum_open_interest_value": [200.0],
            "long_short_ratio": [1.1],
            "top_trader_long_short_ratio": [1.2],
            "sum_taker_long_short_vol_ratio": [0.9],
        }
    )
    metrics.to_parquet(tmp_path / "BTCUSDT_metrics.parquet", index=False)
    ohlcv = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-04-01T00:00:00Z", "2024-04-01T04:00:00Z"],
                utc=True,
            ),
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        }
    )

    out = merge_metrics_into_ohlcv("BTCUSDT", ohlcv, tmp_path)

    assert pd.isna(out.loc[0, "sum_open_interest"])
    assert out.loc[1, "sum_open_interest"] == pytest.approx(100.0)


# ── OPT-5: Baggage columns dropped in _load_cache ──────────────────────


def test_load_cache_drops_baggage_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPT-5: _load_cache drops close_time, no_trades, ignore columns."""
    import tempfile

    df = pd.DataFrame({
        "timestamp": [1704067200000, 1704153600000],
        "open": [1.0, 2.0],
        "high": [1.1, 2.1],
        "low": [0.9, 1.9],
        "close": [1.05, 2.05],
        "volume": [100.0, 200.0],
        "close_time": ["1710000000000", "1710000000001"],
        "no_trades": ["100", "200"],
        "ignore": ["0", "0"],
        "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
    })
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    df.to_parquet(tmp_path, index=False)

    collector = DataCollector()
    monkeypatch.setattr(collector, "_cache_path", lambda *a, **kw: Path(tmp_path))
    monkeypatch.setattr(collector, "_normalize_df", lambda x: x)

    result = collector._load_cache("TESTUSDT", "1h")
    assert "close_time" not in result.columns
    assert "no_trades" not in result.columns
    assert "ignore" not in result.columns
    assert "open" in result.columns
    assert "close" in result.columns
    assert "datetime" in result.columns
    Path(tmp_path).unlink(missing_ok=True)


def test_load_cache_no_baggage_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPT-5: _load_cache unchanged when no baggage columns present."""
    import tempfile

    df = pd.DataFrame({
        "timestamp": [1704067200000],
        "open": [1.0],
        "high": [1.1],
        "low": [0.9],
        "close": [1.05],
        "volume": [100.0],
        "datetime": pd.to_datetime(["2024-01-01"], utc=True),
    })
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    df.to_parquet(tmp_path, index=False)

    collector = DataCollector()
    monkeypatch.setattr(collector, "_cache_path", lambda *a, **kw: Path(tmp_path))
    monkeypatch.setattr(collector, "_normalize_df", lambda x: x)

    result = collector._load_cache("TESTUSDT", "1h")
    assert list(result.columns) == list(df.columns)
    Path(tmp_path).unlink(missing_ok=True)


# ── OPT-6: _normalize_df early exit for numeric cols ────────────────────


def test_normalize_df_returns_early_when_all_numeric() -> None:
    """OPT-6: _normalize_df skips string loop when all non-datetime cols are numeric."""
    df = pd.DataFrame({
        "open": pd.array([1.0, 2.0], dtype="Float64"),
        "close": pd.array([3.0, 4.0], dtype="Float64"),
        "volume": pd.array([100.0, 200.0], dtype="Float64"),
        "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
    })
    collector = DataCollector()
    result = collector._normalize_df(df)
    pd.testing.assert_frame_equal(result, df)


def test_normalize_df_still_converts_mixed_cols() -> None:
    """OPT-6: _normalize_df still runs string loop when non-numeric cols present."""
    df = pd.DataFrame({
        "open": ["1.0", "2.0"],
        "close": ["3.0", "4.0"],
        "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
    })
    collector = DataCollector()
    result = collector._normalize_df(df)
    assert result["open"].dtype != "object"
    assert float(result["open"].iloc[0]) == 1.0


# ── OPT-7: No redundant .copy() in collect_and_save ────────────────────


def test_collect_and_save_no_redundant_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPT-7: collect_and_save w/ fetch_network=False returns loc[mask] directly."""
    df = pd.DataFrame({
        "open": [1.0, 2.0, 3.0],
        "close": [1.1, 2.1, 3.1],
        "datetime": pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03"], utc=True
        ),
    })
    collector = DataCollector()
    monkeypatch.setattr(collector, "_load_cache", lambda *a, **kw: df)
    result = collector.collect_and_save(
        "TESTUSDT", "1h", "2024-01-01", "2024-01-02", fetch_network=False,
    )
    assert len(result) == 2
    assert result["open"].iloc[0] == 1.0
    assert result["open"].iloc[1] == 2.0
