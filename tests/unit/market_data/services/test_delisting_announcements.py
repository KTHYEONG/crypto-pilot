from __future__ import annotations

from urllib.error import HTTPError

import pandas as pd

from src.market_data.services.delisting_announcements import collect_delisting_notices


def test_standard_http_429_retries_with_bounded_backoff() -> None:
    attempts = 0
    sleeps: list[float] = []

    def detail(code: str) -> dict:
        nonlocal attempts
        attempts += 1
        raise HTTPError("https://www.binance.com", 429, "rate limited", {}, None)

    rows, report = collect_delisting_notices(
        existing=(), fetch_page=lambda page: {"data": {"catalogs": [{
            "articles": [{"code": "a", "title": "Futures delist AAAUSDT", "releaseDate": 1000}],
            "total": 1,
        }]}}, fetch_detail=detail, sleep=sleeps.append, now=pd.Timestamp("2024-01-01T00:00:00Z"),
    )
    assert attempts == 5
    assert sleeps == [1.5, 1.5, 10.0, 1.5, 20.0, 1.5, 30.0, 1.5, 40.0, 1.5]
    assert rows == []
    assert report.unfetched == ("a",)
