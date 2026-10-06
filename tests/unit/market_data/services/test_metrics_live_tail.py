"""Contract coverage for the metrics live merge helper.

Covers _merge_metrics_frames precedence and a regression guard that the
authoritative merge path stays equivalent to the pre-refactor concat.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.market_data.services.futures_collection import DataCollector


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
