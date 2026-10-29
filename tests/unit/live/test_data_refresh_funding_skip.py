"""A release-time refresh must not re-fetch funding that is already current on disk."""

from __future__ import annotations

import pandas as pd

from src.live import data_refresh


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 10**6)


def _write_klines(root, symbol: str, tail: pd.Timestamp) -> None:
    directory = root / "ohlcv" / "1h"
    directory.mkdir(parents=True, exist_ok=True)
    stamps = [_ms(tail - pd.Timedelta(hours=h)) for h in range(48)]
    pd.DataFrame({"timestamp": stamps, "close": [1.0] * 48}).to_parquet(directory / f"{symbol}.parquet", index=False)


def _write_funding(root, symbol: str, last: pd.Timestamp) -> None:
    directory = root / "funding"
    directory.mkdir(parents=True, exist_ok=True)
    stamps = [_ms(last - pd.Timedelta(hours=8 * k)) for k in (2, 1, 0)]
    pd.DataFrame({"timestamp": stamps, "funding_rate": [0.0001] * 3}).to_parquet(directory / f"{symbol}.parquet", index=False)


def test_stale_klines_with_fresh_funding_fetch_klines_only(tmp_path) -> None:
    now = pd.Timestamp("2026-09-01T23:00:15Z")
    symbols = ["AUSDT", "BUSDT"]
    funding_calls: list[str] = []
    ohlcv_calls: list[str] = []
    for sym in symbols:
        _write_klines(tmp_path, sym, now.floor("h") - pd.Timedelta(hours=2))
        _write_funding(tmp_path, sym, pd.Timestamp("2026-09-01T16:00:00Z"))

    class _Collector:
        def ensure_ohlcv_data(self, symbol, timeframe, start, end):
            ohlcv_calls.append(symbol)
            _write_klines(tmp_path, symbol, now.floor("h") - pd.Timedelta(hours=1))

        def ensure_funding_data(self, symbol, start, end):
            funding_calls.append(symbol)

    report = data_refresh.refresh_live_market_data(
        tmp_path, symbols=symbols, now=now, lookback_days=40, max_workers=2, deadline_s=60.0,
        min_symbols=1, max_fail_fraction=0.15, collector=_Collector(),
    )

    assert sorted(ohlcv_calls) == symbols
    assert funding_calls == []
    assert report.refreshed == 2
    assert report.ok is True
