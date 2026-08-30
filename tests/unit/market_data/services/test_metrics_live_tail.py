"""Contract coverage for the OI/LSR live REST tail top-up.

Covers _merge_metrics_frames precedence, ensure_metrics_live_tail canonical
schema + PIT lag, per-endpoint fail-soft, and a regression guard that the
authoritative merge path stays equivalent to the pre-refactor concat.
"""

from __future__ import annotations

import pandas as pd
import pytest

import src.market_data.services.futures_collection as fc
from src.market_data.services.futures_collection import DataCollector
from src.market_data.storage.schemas import METRICS_CANONICAL_COLUMNS


def _mk(ts: list[int], oi: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ts,
            "datetime": pd.to_datetime(ts, unit="ms", utc=True),
            "available_at": pd.to_datetime(ts, unit="ms", utc=True) + pd.Timedelta(minutes=5),
            "symbol": "BTCUSDT",
            "sum_open_interest": oi,
            "sum_open_interest_value": [x * 10 for x in oi],
            "long_short_ratio": [1.0] * len(ts),
            "top_trader_long_short_ratio": [1.1] * len(ts),
            "sum_taker_long_short_vol_ratio": [0.9] * len(ts),
        }
    )


def test_merge_metrics_frames_precedence() -> None:
    cache = _mk([0, 300_000], [10.0, 11.0])
    incoming = _mk([300_000, 600_000], [99.0, 12.0])  # overlaps at 300_000

    auth = DataCollector._merge_metrics_frames(cache, incoming, incoming_is_authoritative=True)
    row = auth.loc[auth["timestamp"] == 300_000, "sum_open_interest"].iloc[0]
    assert row == pytest.approx(99.0)

    non_auth = DataCollector._merge_metrics_frames(cache, incoming, incoming_is_authoritative=False)
    row2 = non_auth.loc[non_auth["timestamp"] == 300_000, "sum_open_interest"].iloc[0]
    assert row2 == pytest.approx(11.0)

    for frame in (auth, non_auth):
        assert frame["timestamp"].is_monotonic_increasing
        assert not frame["timestamp"].duplicated().any()
        assert list(frame.index) == list(range(len(frame)))


def test_ensure_metrics_data_still_byte_identical_after_merge_refactor() -> None:
    cache = _mk([0, 300_000], [10.0, 11.0])
    fetched = _mk([300_000, 600_000], [11.0, 12.0])

    expected = (
        pd.concat([cache, fetched], ignore_index=True)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    got = DataCollector._merge_metrics_frames(cache, fetched, incoming_is_authoritative=True)
    pd.testing.assert_frame_equal(got, expected)


_TS = [1_788_000_000_000, 1_788_000_300_000, 1_788_000_600_000]


def _canned(endpoint: str, symbol: str, *, period: str = "5m", limit: int = 500):
    if endpoint == "openInterestHist":
        return [{"timestamp": t, "sumOpenInterest": "100.0", "sumOpenInterestValue": "9000.0"} for t in _TS]
    if endpoint == "globalLongShortAccountRatio":
        return [{"timestamp": t, "longShortRatio": "1.5"} for t in _TS]
    if endpoint == "topLongShortPositionRatio":
        return [{"timestamp": t, "longShortRatio": "2.5"} for t in _TS]
    if endpoint == "takerlongshortRatio":
        return [{"timestamp": t, "buySellRatio": "0.8"} for t in _TS]
    raise AssertionError(endpoint)


def test_ensure_metrics_live_tail_writes_canonical_schema(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fc, "metrics_path", lambda sym: tmp_path / f"{sym}.parquet")
    collector = DataCollector()
    monkeypatch.setattr(collector.client, "fetch_futures_data_metric", _canned)

    collector.ensure_metrics_live_tail("BTCUSDT", lookback_days=30)

    df = pd.read_parquet(tmp_path / "BTCUSDT.parquet")
    assert list(df.columns) == list(METRICS_CANONICAL_COLUMNS)
    assert (df["available_at"] - df["datetime"] == pd.Timedelta(minutes=5)).all()
    assert df["long_short_ratio"].tolist() == pytest.approx([1.5] * len(df))
    assert df["top_trader_long_short_ratio"].tolist() == pytest.approx([2.5] * len(df))
    assert df["sum_taker_long_short_vol_ratio"].tolist() == pytest.approx([0.8] * len(df))
    assert df["sum_open_interest"].tolist() == pytest.approx([100.0] * len(df))


def test_ensure_metrics_live_tail_failsoft_on_partial_endpoint_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fc, "metrics_path", lambda sym: tmp_path / f"{sym}.parquet")
    collector = DataCollector()

    def _flaky(endpoint, symbol, *, period="5m", limit=500):
        if endpoint == "takerlongshortRatio":
            raise ConnectionError("boom")
        return _canned(endpoint, symbol, period=period, limit=limit)

    monkeypatch.setattr(collector.client, "fetch_futures_data_metric", _flaky)

    # The per-symbol loop in _refresh_live_universe wraps this in try/except; a
    # raised error must surface here (not be silently swallowed inside the method).
    with pytest.raises(ConnectionError):
        collector.ensure_metrics_live_tail("BTCUSDT")
