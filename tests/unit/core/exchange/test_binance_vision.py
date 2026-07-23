from __future__ import annotations

import pandas as pd
import pytest

from src.core.exchange.binance_vision import BinanceVisionDownloader


def test_downloader_defaults_keep_archive_request_rate_below_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BINANCE_VISION_MAX_WEIGHT_PER_MIN", raising=False)
    monkeypatch.delenv("BINANCE_VISION_MIN_REQUEST_INTERVAL_SECONDS", raising=False)

    downloader = BinanceVisionDownloader()

    assert downloader.max_concurrency == 4
    assert downloader.max_weight_per_min == 600
    assert downloader.min_request_interval_seconds >= 0.1


def test_fetch_daily_metrics_encodes_unicode_symbol_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloader = BinanceVisionDownloader()
    requested_urls: list[str] = []

    def fetch(url: str) -> pd.DataFrame:
        requested_urls.append(url)
        return pd.DataFrame()

    monkeypatch.setattr(downloader, "_fetch_zip_csv", fetch)

    result = downloader.fetch_daily_metrics("币安人生USDT", pd.Timestamp("2026-07-01"))

    assert result.empty
    assert len(requested_urls) == 1
    assert "%25" not in requested_urls[0]
    assert "%E5%B8%81" in requested_urls[0]
