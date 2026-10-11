from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.public_archive.derive import (
    derive_agg_trades,
    derive_book_depth,
    derive_metrics,
    slice_agg_trades,
)

_DAY = date(2024, 1, 15)
_DAY_START_MS = int(pd.Timestamp(_DAY, tz="UTC").value // 10**6)

_METRICS_COLS = [
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]


def _metrics_frame(times_ms: list[int], **overrides: object) -> pd.DataFrame:
    n = len(times_ms)
    data: dict[str, object] = {
        "create_time": times_ms,
        "symbol": ["BTCUSDT"] * n,
        "sum_open_interest": [1.0] * n,
        "sum_open_interest_value": [2.0] * n,
        "count_toptrader_long_short_ratio": [3.0] * n,
        "sum_toptrader_long_short_ratio": [4.0] * n,
        "count_long_short_ratio": [5.0] * n,
        "sum_taker_long_short_vol_ratio": [6.0] * n,
    }
    data.update(overrides)
    return pd.DataFrame(data, columns=_METRICS_COLS)


def test_metrics_rejects_off_grid_timestamp() -> None:
    frame = _metrics_frame([_DAY_START_MS + 3 * 60 * 1000])
    with pytest.raises(DataIntegrityError):
        derive_metrics(frame, symbol="BTCUSDT", day=_DAY)


def test_metrics_keeps_missing_points_missing() -> None:
    times = [_DAY_START_MS + i * 5 * 60 * 1000 for i in range(288) if i != 100]
    out = derive_metrics(_metrics_frame(times), symbol="BTCUSDT", day=_DAY)
    assert len(out) == 287
    assert out["timestamp"].is_unique


def test_metrics_empty_strings_become_nan() -> None:
    times = [_DAY_START_MS, _DAY_START_MS + 5 * 60 * 1000]
    frame = _metrics_frame(times, sum_toptrader_long_short_ratio=["", 4.0])
    out = derive_metrics(frame, symbol="BTCUSDT", day=_DAY)
    assert np.isnan(out["sum_toptrader_long_short_ratio"].iloc[0])
    assert out["sum_open_interest"].tolist() == [1.0, 1.0]
    assert str(out["sum_open_interest"].dtype) == "float64"


def test_book_depth_bin_means() -> None:
    base = _DAY_START_MS + 60_000
    frame = pd.DataFrame({
        "timestamp": [base, base, base, base],
        "percentage": [1.0, 1.0, -5.0, -5.0],
        "depth": [10.0, 20.0, 1.0, 1.0],
        "notional": [100.0, 200.0, 10.0, 10.0],
    })
    out = derive_book_depth(frame, symbol="BTCUSDT", day=_DAY)
    assert len(out) == 1
    assert out["depth_p1"].iloc[0] == pytest.approx(15.0)
    assert out["n_snapshots"].iloc[0] == 1


def test_book_depth_bin_means_two_snapshots() -> None:
    base = _DAY_START_MS + 60_000
    frame = pd.DataFrame({
        "timestamp": [base, base + 30_000, base, base + 30_000],
        "percentage": [1.0, 1.0, -5.0, -5.0],
        "depth": [10.0, 20.0, 1.0, 1.0],
        "notional": [100.0, 200.0, 10.0, 10.0],
    })
    out = derive_book_depth(frame, symbol="BTCUSDT", day=_DAY)
    assert out["depth_p1"].iloc[0] == pytest.approx(15.0)
    assert out["n_snapshots"].iloc[0] == 2


def test_book_depth_unknown_band_rejected() -> None:
    frame = pd.DataFrame({
        "timestamp": [_DAY_START_MS],
        "percentage": [7.0],
        "depth": [1.0],
        "notional": [1.0],
    })
    with pytest.raises(DataIntegrityError):
        derive_book_depth(frame, symbol="BTCUSDT", day=_DAY)


def _agg_frame(rows: list[tuple[int, float, float, object]]) -> pd.DataFrame:
    data = {
        "agg_trade_id": list(range(len(rows))),
        "price": [r[1] for r in rows],
        "quantity": [r[2] for r in rows],
        "first_trade_id": list(range(len(rows))),
        "last_trade_id": list(range(len(rows))),
        "transact_time": [r[0] for r in rows],
        "is_buyer_maker": [r[3] for r in rows],
    }
    return pd.DataFrame(data)


def test_agg_trades_taker_side_and_volume_conservation() -> None:
    base = _DAY_START_MS + 10_000
    frame = _agg_frame([
        (base, 100.0, 1.0, False),
        (base + 1_000, 101.0, 2.0, True),
        (base + 2_000, 102.0, 3.0, "false"),
    ])
    out = derive_agg_trades(frame, symbol="BTCUSDT", day=_DAY)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["taker_buy_qty"] == pytest.approx(4.0)
    assert row["taker_sell_qty"] == pytest.approx(2.0)
    assert row["taker_buy_qty"] + row["taker_sell_qty"] == pytest.approx(6.0)


def test_agg_trades_bar_alignment() -> None:
    day_start = pd.Timestamp("2024-01-15", tz="UTC")
    t0 = int((day_start + pd.Timedelta(minutes=2, seconds=59, milliseconds=999)).value // 10**6)
    t1 = int((day_start + pd.Timedelta(minutes=3)).value // 10**6)
    frame = _agg_frame([(t0, 100.0, 1.0, False), (t1, 100.0, 1.0, False)])
    out = derive_agg_trades(frame, symbol="BTCUSDT", day=_DAY)
    assert len(out) == 2
    assert out["timestamp"].iloc[0] == pd.Timestamp("2024-01-15 00:00", tz="UTC").as_unit("ms")
    assert out["timestamp"].iloc[1] == pd.Timestamp("2024-01-15 00:03", tz="UTC").as_unit("ms")


def test_agg_trades_rejects_nonpositive_price() -> None:
    frame = _agg_frame([(_DAY_START_MS + 1_000, 0.0, 1.0, False)])
    with pytest.raises(DataIntegrityError):
        derive_agg_trades(frame, symbol="BTCUSDT", day=_DAY)


def test_slice_window_bounds() -> None:
    day_start = pd.Timestamp("2024-01-15", tz="UTC")
    ms = lambda ts: int(ts.value // 10**6)  # noqa: E731
    rows = [
        (ms(day_start + pd.Timedelta(hours=22, minutes=59, seconds=59)), 100.0, 1.0, False),
        (ms(day_start + pd.Timedelta(hours=23)), 100.0, 1.0, False),
        (ms(day_start + pd.Timedelta(hours=23, minutes=59, seconds=59)), 100.0, 1.0, True),
        (ms(day_start + pd.Timedelta(days=1)), 100.0, 1.0, False),
    ]
    frame = _agg_frame(rows)
    out = slice_agg_trades(frame, day=_DAY, start_hour_utc=23, end_hour_utc=24)
    assert list(out["transact_time_ms"]) == [
        ms(day_start + pd.Timedelta(hours=23)),
        ms(day_start + pd.Timedelta(hours=23, minutes=59, seconds=59)),
    ]
