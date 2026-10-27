from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import pyarrow as pa

from src.common.errors import DataIntegrityError
from src.market_data.public_archive.derive import (
    StreamFallbackError,
    derive_agg_trades,
    derive_agg_trades_streamed,
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


def test_metrics_floors_sub_bucket_jitter_to_grid() -> None:
    times = [_DAY_START_MS + 5 * 60 * 1000 + 48_000, _DAY_START_MS + 10 * 60 * 1000 + 234_000]
    out = derive_metrics(_metrics_frame(times), symbol="BTCUSDT", day=_DAY)
    assert out["timestamp"].astype("int64").tolist() == [
        _DAY_START_MS + 5 * 60 * 1000,
        _DAY_START_MS + 10 * 60 * 1000,
    ]


def test_metrics_rejects_conflicting_rows_in_one_bucket() -> None:
    frame = _metrics_frame([_DAY_START_MS, _DAY_START_MS + 49_000])
    frame.loc[1, "sum_open_interest"] = 99.0
    with pytest.raises(DataIntegrityError):
        derive_metrics(frame, symbol="BTCUSDT", day=_DAY)


def test_metrics_shifted_window_drops_only_next_midnight_row() -> None:
    times = [_DAY_START_MS + i * 5 * 60 * 1000 + 30_000 for i in range(1, 289)]
    out = derive_metrics(_metrics_frame(times), symbol="BTCUSDT", day=_DAY)
    assert len(out) == 287
    assert out["timestamp"].iloc[0] == pd.Timestamp(_DAY_START_MS + 5 * 60 * 1000, unit="ms", tz="UTC")
    assert out["timestamp"].iloc[-1] == pd.Timestamp(_DAY_START_MS + 287 * 5 * 60 * 1000, unit="ms", tz="UTC")


def test_metrics_next_midnight_row_in_aligned_window_still_fails() -> None:
    times = [_DAY_START_MS + i * 5 * 60 * 1000 for i in range(0, 289)]
    with pytest.raises(DataIntegrityError):
        derive_metrics(_metrics_frame(times), symbol="BTCUSDT", day=_DAY)


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


_ARROW_TYPES = {
    "agg_trade_id": pa.int64(),
    "price": pa.float64(),
    "quantity": pa.float64(),
    "first_trade_id": pa.int64(),
    "last_trade_id": pa.int64(),
    "transact_time": pa.int64(),
    "is_buyer_maker": pa.bool_(),
}


def _random_trades(n: int, *, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = np.sort(rng.integers(_DAY_START_MS, _DAY_START_MS + 86_400_000, size=n))
    times[10:14] = times[10]
    first = np.cumsum(rng.integers(1, 4, size=n))
    return pd.DataFrame({
        "agg_trade_id": np.arange(n, dtype="int64"),
        "price": 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, size=n))),
        "quantity": rng.uniform(0.01, 5.0, size=n),
        "first_trade_id": first,
        "last_trade_id": first + rng.integers(0, 3, size=n),
        "transact_time": times,
        "is_buyer_maker": rng.random(n) < 0.5,
    })


def _batches(frame: pd.DataFrame, size: int) -> list[pa.RecordBatch]:
    schema = pa.schema(list(_ARROW_TYPES.items()))
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    return table.to_batches(max_chunksize=size)


@pytest.mark.parametrize("batch_rows", [1, 7, 333, 10_000])
def test_streamed_matches_full_frame_across_batch_boundaries(batch_rows: int) -> None:
    frame = _random_trades(4_000)
    bars, sliced = derive_agg_trades_streamed(_batches(frame, batch_rows), day=_DAY, slice_hours=(23, 24))
    expected = derive_agg_trades(frame, symbol="BTCUSDT", day=_DAY)
    pd.testing.assert_frame_equal(bars, expected, check_exact=False, rtol=1e-12, atol=1e-15)
    expected_slice = slice_agg_trades(frame, day=_DAY, start_hour_utc=23, end_hour_utc=24)
    assert sliced is not None
    pd.testing.assert_frame_equal(sliced, expected_slice, check_exact=True)


def test_streamed_rv_chains_within_bar_across_batches() -> None:
    frame = _agg_frame([
        (_DAY_START_MS + 1_000, 100.0, 1.0, False),
        (_DAY_START_MS + 2_000, 110.0, 1.0, False),
        (_DAY_START_MS + 3_000, 99.0, 1.0, True),
    ])
    bars, sliced = derive_agg_trades_streamed(_batches(frame, 1), day=_DAY, slice_hours=None)
    expected = (np.log(110.0) - np.log(100.0)) ** 2 + (np.log(99.0) - np.log(110.0)) ** 2
    assert bars["rv_trade"].iloc[0] == pytest.approx(expected, rel=1e-12)
    assert sliced is None


def test_streamed_slice_without_window_rows_is_typed_empty() -> None:
    frame = _agg_frame([(_DAY_START_MS + 1_000, 100.0, 1.0, False)])
    _, sliced = derive_agg_trades_streamed(_batches(frame, 10), day=_DAY, slice_hours=(23, 24))
    assert sliced is not None
    assert sliced.empty
    assert sliced.dtypes.equals(slice_agg_trades(frame, day=_DAY, start_hour_utc=0, end_hour_utc=1).iloc[:0].dtypes)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda f: f.assign(transact_time=f["transact_time"].iloc[::-1].to_numpy()),
        lambda f: f.assign(price=np.r_[f["price"].iloc[:-1].to_numpy(), 0.0]),
        lambda f: f.assign(transact_time=np.r_[f["transact_time"].iloc[:-1].to_numpy(), _DAY_START_MS + 86_400_000]),
        lambda f: f.assign(last_trade_id=f["first_trade_id"] - 1),
        lambda f: f.iloc[:0],
    ],
    ids=["unordered", "nonpositive_price", "outside_day", "trade_id_range", "empty"],
)
def test_streamed_falls_back_instead_of_deriving_invalid_input(mutate) -> None:
    frame = mutate(_random_trades(50))
    with pytest.raises(StreamFallbackError):
        derive_agg_trades_streamed(_batches(frame, 7), day=_DAY, slice_hours=(23, 24))


def test_streamed_unordered_across_batch_boundary_falls_back() -> None:
    frame = _random_trades(20)
    swapped = pd.concat([frame.iloc[10:], frame.iloc[:10]], ignore_index=True)
    with pytest.raises(StreamFallbackError):
        derive_agg_trades_streamed(_batches(swapped, 10), day=_DAY, slice_hours=None)


def test_streamed_null_field_falls_back() -> None:
    batch = _batches(_random_trades(20).iloc[:5], 5)[0]
    nulled = batch.set_column(1, "price", pa.array([1.0, None, 1.0, 1.0, 1.0], type=pa.float64()))
    with pytest.raises(StreamFallbackError):
        derive_agg_trades_streamed([nulled], day=_DAY, slice_hours=None)


def test_streamed_arrow_parse_error_falls_back() -> None:
    def broken():
        raise pa.ArrowInvalid("bad row")
        yield

    with pytest.raises(StreamFallbackError):
        derive_agg_trades_streamed(broken(), day=_DAY, slice_hours=None)


def test_streamed_rejects_invalid_slice_window() -> None:
    with pytest.raises(ValueError, match="invalid slice window"):
        derive_agg_trades_streamed([], day=_DAY, slice_hours=(5, 5))


def test_streamed_skips_empty_batches_and_legacy_empty_slice_is_typed() -> None:
    frame = _random_trades(30)
    batches = _batches(frame, 10)
    empty = batches[0].slice(0, 0)
    bars, _ = derive_agg_trades_streamed([empty, *batches, empty], day=_DAY, slice_hours=None)
    assert int(bars["n_agg_trades"].sum()) == 30
    legacy = slice_agg_trades(frame.iloc[:0], day=_DAY, start_hour_utc=23, end_hour_utc=24)
    assert legacy.empty
    assert [str(d) for d in legacy.dtypes] == ["int64", "int64", "float64", "float64", "bool"]
