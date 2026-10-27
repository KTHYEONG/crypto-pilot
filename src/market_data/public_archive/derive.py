from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa

from src.common.errors import DataIntegrityError

BOOK_DEPTH_BANDS: tuple[str, ...] = (
    "m5",
    "m4",
    "m3",
    "m2",
    "m1",
    "m02",
    "p02",
    "p1",
    "p2",
    "p3",
    "p4",
    "p5",
)

METRICS_NUMERIC: tuple[str, ...] = (
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)

_PERCENT_TO_TAG: dict[float, str] = {
    -5.0: "m5",
    -4.0: "m4",
    -3.0: "m3",
    -2.0: "m2",
    -1.0: "m1",
    -0.2: "m02",
    0.2: "p02",
    1.0: "p1",
    2.0: "p2",
    3.0: "p3",
    4.0: "p4",
    5.0: "p5",
}



def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise DataIntegrityError(f"missing columns: {missing}")


def _integers(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if not np.isfinite(values).all() or (values % 1 != 0).any() or (values < 0).any() or (values >= 2**63).any():
        raise DataIntegrityError("invalid integer archive field")
    return values.astype("int64")


def _to_utc_ms(series: pd.Series) -> pd.Series:
    try:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_datetime(series, unit="ms", utc=True)
        return pd.to_datetime(series, utc=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataIntegrityError("invalid timestamp") from exc


def _day_bounds(day: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(day, tz="UTC").as_unit("ms")
    end = start + pd.Timedelta(days=1)
    return start, end


def _check_inside_day(stamps: pd.Series, day: date) -> None:
    start, end = _day_bounds(day)
    if ((stamps < start) | (stamps >= end)).any():
        raise DataIntegrityError("timestamp outside day")


def derive_metrics(raw: pd.DataFrame, *, symbol: str, day: date) -> pd.DataFrame:
    """Derive canonical 5m metrics for one symbol-day.

    Sub-bucket publisher jitter is floored to the 5m grid. A window shifted by one bucket
    (first row after midnight) drops its trailing next-midnight row.
    """
    _require_columns(
        raw,
        (
            "create_time",
            "symbol",
            "sum_open_interest",
            "sum_open_interest_value",
            "count_toptrader_long_short_ratio",
            "sum_toptrader_long_short_ratio",
            "count_long_short_ratio",
            "sum_taker_long_short_vol_ratio",
        ),
    )
    if len(raw) == 0:
        out = pd.DataFrame(columns=["timestamp", *METRICS_NUMERIC])
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.as_unit("ms")
        for col in METRICS_NUMERIC:
            out[col] = out[col].astype("float64")
        return out
    if (raw["symbol"].astype(str) != symbol).any():
        raise DataIntegrityError("symbol column mismatch")
    stamps = _to_utc_ms(raw["create_time"])
    if stamps.isna().any():
        raise DataIntegrityError("null create_time")
    stamps = stamps.dt.floor("5min")
    start, end = _day_bounds(day)
    inside = (stamps < end).to_numpy()
    if stamps.min() > start and inside.any():
        keep_rows = inside
        raw, stamps = raw.loc[keep_rows], stamps.loc[keep_rows]
    _check_inside_day(stamps, day)
    dup_mask = stamps.duplicated(keep=False)
    if dup_mask.any():
        dup = raw[dup_mask].copy()
        dup["__ts"] = stamps[dup_mask].to_numpy()
        if dup.drop_duplicates().duplicated("__ts").any():
            raise DataIntegrityError("conflicting duplicate metrics rows")
        keep = ~stamps.duplicated(keep="first")
        raw = raw.loc[keep]
        stamps = stamps.loc[keep]
    out = pd.DataFrame({"timestamp": stamps.to_numpy()})
    for col in METRICS_NUMERIC:
        try:
            out[col] = pd.to_numeric(raw[col].replace("", np.nan), errors="raise").astype("float64").to_numpy()
        except (ValueError, TypeError) as exc:
            raise DataIntegrityError(f"invalid metrics value: {col}") from exc
    if np.isinf(out[list(METRICS_NUMERIC)].to_numpy()).any():
        raise DataIntegrityError("infinite metrics value")
    out = out.sort_values("timestamp", kind="mergesort").drop_duplicates(subset="timestamp", keep="first")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.as_unit("ms")
    return out[["timestamp", *METRICS_NUMERIC]].reset_index(drop=True)


def _band_tag(value: float) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DataIntegrityError(f"unknown bookDepth band: {value!r}") from exc
    for pct, tag in _PERCENT_TO_TAG.items():
        if abs(number - pct) < 1e-9:
            return tag
    raise DataIntegrityError(f"unknown bookDepth band: {value!r}")


def derive_book_depth(raw: pd.DataFrame, *, symbol: str, day: date) -> pd.DataFrame:
    """Derive 5m book-depth means for one symbol-day."""
    del symbol
    _require_columns(raw, ("timestamp", "percentage", "depth", "notional"))
    columns = (
        ["timestamp", "n_snapshots"]
        + [f"depth_{t}" for t in BOOK_DEPTH_BANDS]
        + [f"notional_{t}" for t in BOOK_DEPTH_BANDS]
    )
    if len(raw) == 0:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns}).astype(
            {"n_snapshots": "int64"}
        )
    depth = pd.to_numeric(raw["depth"], errors="coerce")
    notional = pd.to_numeric(raw["notional"], errors="coerce")
    if not np.isfinite(depth).all() or not np.isfinite(notional).all():
        raise DataIntegrityError("null bookDepth depth/notional")
    if ((depth < 0) | (notional < 0)).any():
        raise DataIntegrityError("negative bookDepth depth/notional")
    tags = raw["percentage"].map(_band_tag)
    stamps = _to_utc_ms(raw["timestamp"])
    if stamps.isna().any():
        raise DataIntegrityError("null bookDepth timestamp")
    _check_inside_day(stamps, day)
    bins = stamps.dt.floor("5min")
    work = pd.DataFrame({
        "bin": bins.to_numpy(),
        "snap": stamps.to_numpy(),
        "tag": tags.to_numpy(),
        "depth": depth.to_numpy(dtype="float64"),
        "notional": notional.to_numpy(dtype="float64"),
    })
    counts = work.groupby("bin")["snap"].nunique().rename("n_snapshots")
    depth_mean = work.pivot_table(index="bin", columns="tag", values="depth", aggfunc="mean")
    notional_mean = work.pivot_table(index="bin", columns="tag", values="notional", aggfunc="mean")
    out = pd.DataFrame({"timestamp": counts.index, "n_snapshots": counts.to_numpy(dtype="int64")})
    for tag in BOOK_DEPTH_BANDS:
        out[f"depth_{tag}"] = (
            depth_mean[tag].reindex(counts.index).to_numpy(dtype="float64")
            if tag in depth_mean.columns
            else np.nan
        )
        out[f"notional_{tag}"] = (
            notional_mean[tag].reindex(counts.index).to_numpy(dtype="float64")
            if tag in notional_mean.columns
            else np.nan
        )
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.as_unit("ms")
    out["n_snapshots"] = out["n_snapshots"].astype("int64")
    return out[columns].sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def _normalize_buyer_maker(series: pd.Series) -> pd.Series:
    def _one(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            if value in (0, 1):
                return bool(value)
            raise DataIntegrityError(f"invalid is_buyer_maker: {value!r}")
        text = str(value).strip().lower()
        if text == "true":
            return True
        if text == "false":
            return False
        raise DataIntegrityError(f"invalid is_buyer_maker: {value!r}")

    return series.map(_one).astype("bool")


def derive_agg_trades(raw: pd.DataFrame, *, symbol: str, day: date) -> pd.DataFrame:
    """Derive 3m aggregate-trade bars for one symbol-day."""
    del symbol
    _require_columns(
        raw,
        (
            "agg_trade_id",
            "price",
            "quantity",
            "first_trade_id",
            "last_trade_id",
            "transact_time",
            "is_buyer_maker",
        ),
    )
    columns = [
        "timestamp",
        "n_agg_trades",
        "n_trades",
        "taker_buy_qty",
        "taker_sell_qty",
        "taker_buy_notional",
        "taker_sell_notional",
        "max_agg_notional",
        "vwap",
        "rv_trade",
    ]
    if len(raw) == 0:
        return pd.DataFrame(columns=columns)
    price = pd.to_numeric(raw["price"], errors="coerce")
    quantity = pd.to_numeric(raw["quantity"], errors="coerce")
    if not np.isfinite(price).all() or not np.isfinite(quantity).all():
        raise DataIntegrityError("null agg trade price/quantity")
    if ((price <= 0) | (quantity <= 0)).any():
        raise DataIntegrityError("non-positive agg trade price/quantity")
    stamps = _to_utc_ms(_integers(raw["transact_time"]))
    _check_inside_day(stamps, day)
    maker = _normalize_buyer_maker(raw["is_buyer_maker"])
    with np.errstate(over="ignore"):
        notional = price.to_numpy(dtype="float64") * quantity.to_numpy(dtype="float64")
    if not np.isfinite(notional).all():
        raise DataIntegrityError("aggregate trade notional overflow")
    work = pd.DataFrame({
        "transact_time": stamps.to_numpy(),
        "agg_trade_id": _integers(raw["agg_trade_id"]).to_numpy(),
        "price": price.to_numpy(dtype="float64"),
        "quantity": quantity.to_numpy(dtype="float64"),
        "n_trades": (
            _integers(raw["last_trade_id"]).to_numpy()
            - _integers(raw["first_trade_id"]).to_numpy()
            + 1
        ),
        "maker": maker.to_numpy(dtype="bool"),
        "notional": notional,
    })
    if (work["n_trades"] < 1).any():
        raise DataIntegrityError("invalid trade id range")
    work = work.sort_values(["transact_time", "agg_trade_id"], kind="mergesort")
    work["bar"] = pd.to_datetime(work["transact_time"], utc=True).dt.floor("3min")
    buy = ~work["maker"]
    work["taker_buy_qty"] = work["quantity"].where(buy, 0.0)
    work["taker_sell_qty"] = work["quantity"].where(~buy, 0.0)
    work["taker_buy_notional"] = work["notional"].where(buy, 0.0)
    work["taker_sell_notional"] = work["notional"].where(~buy, 0.0)
    work["log_price"] = np.log(work["price"])
    work["rv_trade"] = work.groupby("bar")["log_price"].diff().fillna(0.0).pow(2)
    out = work.groupby("bar").agg(
        n_agg_trades=("agg_trade_id", "size"), n_trades=("n_trades", "sum"),
        taker_buy_qty=("taker_buy_qty", "sum"), taker_sell_qty=("taker_sell_qty", "sum"),
        taker_buy_notional=("taker_buy_notional", "sum"),
        taker_sell_notional=("taker_sell_notional", "sum"),
        max_agg_notional=("notional", "max"), total_qty=("quantity", "sum"),
        total_notional=("notional", "sum"), rv_trade=("rv_trade", "sum"),
    ).reset_index().rename(columns={"bar": "timestamp"})
    out["vwap"] = out["total_notional"] / out["total_qty"]
    out["n_trades"] = out["n_trades"].astype("int64")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.as_unit("ms")
    return out[columns].sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def _empty_slice() -> pd.DataFrame:
    return pd.DataFrame({
        "transact_time_ms": pd.Series(dtype="int64"),
        "agg_trade_id": pd.Series(dtype="int64"),
        "price": pd.Series(dtype="float64"),
        "quantity": pd.Series(dtype="float64"),
        "is_buyer_maker": pd.Series(dtype="bool"),
    })


def slice_agg_trades(
    raw: pd.DataFrame, *, day: date, start_hour_utc: int, end_hour_utc: int
) -> pd.DataFrame:
    """Slice raw aggregate trades to a UTC hour window."""
    if not (0 <= start_hour_utc < end_hour_utc <= 24):
        raise ValueError("invalid slice window")
    _require_columns(
        raw, ("agg_trade_id", "price", "quantity", "transact_time", "is_buyer_maker")
    )
    if len(raw) == 0:
        return _empty_slice()
    ms = _integers(raw["transact_time"])
    ms_values = ms.to_numpy(dtype="int64")
    day_start = pd.Timestamp(day, tz="UTC")
    lo = int((day_start + pd.Timedelta(hours=start_hour_utc)).value // 10**6)
    hi = int((day_start + pd.Timedelta(hours=end_hour_utc)).value // 10**6)
    keep = (ms_values >= lo) & (ms_values < hi)
    maker = _normalize_buyer_maker(raw["is_buyer_maker"])
    out = pd.DataFrame({
        "transact_time_ms": ms_values[keep],
        "agg_trade_id": _integers(raw["agg_trade_id"]).to_numpy()[keep],
        "price": pd.to_numeric(raw["price"], errors="coerce").to_numpy(dtype="float64")[keep],
        "quantity": pd.to_numeric(raw["quantity"], errors="coerce").to_numpy(dtype="float64")[keep],
        "is_buyer_maker": maker.to_numpy(dtype="bool")[keep],
    })
    if not np.isfinite(out[["price", "quantity"]]).all().all() or (out[["price", "quantity"]] <= 0).any().any():
        raise DataIntegrityError("invalid slice price/quantity")
    return out.sort_values(["transact_time_ms", "agg_trade_id"], kind="mergesort").reset_index(
        drop=True
    )


class StreamFallbackError(Exception):
    """Streamed aggTrades input breaks a fast-path assumption; derive from the full frame instead."""


_AGG_BAR_MS = 180_000
_DAY_MS = 86_400_000
_AGG_BARS_PER_DAY = _DAY_MS // _AGG_BAR_MS
_SLICE_SOURCE_COLUMNS = ("transact_time", "agg_trade_id", "price", "quantity", "is_buyer_maker")


class _AggTradeStream:
    """Exact per-bar reductions over (transact_time, agg_trade_id)-ordered batches.

    Order is required, not restored: it keeps each bar's trades contiguous so rv_trade chains log-price
    differences across batch boundaries, and keeps slice rows in the legacy sort order without a sort.
    """

    def __init__(self, day: date, slice_hours: tuple[int, int] | None) -> None:
        self.start = int(pd.Timestamp(day, tz="UTC").value // 10**6)
        self.slice_ms = (
            None
            if slice_hours is None
            else (self.start + slice_hours[0] * 3_600_000, self.start + slice_hours[1] * 3_600_000)
        )
        self.count = np.zeros(_AGG_BARS_PER_DAY, dtype="int64")
        self.trades = np.zeros(_AGG_BARS_PER_DAY, dtype="int64")
        # taker_buy_qty, taker_sell_qty, taker_buy_notional, taker_sell_notional, total_qty, total_notional
        self.sums = np.zeros((6, _AGG_BARS_PER_DAY), dtype="float64")
        self.max_notional = np.full(_AGG_BARS_PER_DAY, -np.inf)
        self.rv = np.zeros(_AGG_BARS_PER_DAY, dtype="float64")
        self.tail: tuple[int, int, int, float] | None = None
        self.slices: list[pa.RecordBatch] = []

    def _ordered(self, stamps: np.ndarray, ids: np.ndarray) -> bool:
        if self.tail is not None:
            last_t, last_id = self.tail[0], self.tail[1]
            if stamps[0] < last_t or (stamps[0] == last_t and ids[0] < last_id):
                return False
        step_t = stamps[1:] - stamps[:-1]
        return not bool(np.any((step_t < 0) | ((step_t == 0) & (ids[1:] < ids[:-1]))))

    def _validated(self, batch: pa.RecordBatch) -> tuple[np.ndarray, ...]:
        if any(column.null_count for column in batch.columns):
            raise StreamFallbackError("null field")
        stamps = batch.column("transact_time").to_numpy()
        ids = batch.column("agg_trade_id").to_numpy()
        price = batch.column("price").to_numpy()
        qty = batch.column("quantity").to_numpy()
        first = batch.column("first_trade_id").to_numpy()
        n_trades = batch.column("last_trade_id").to_numpy() - first + 1
        if stamps[0] < self.start or stamps[-1] >= self.start + _DAY_MS or not self._ordered(stamps, ids):
            raise StreamFallbackError("unordered or outside day")
        if ids.min() < 0 or first.min() < 0 or n_trades.min() < 1:
            raise StreamFallbackError("invalid trade ids")
        with np.errstate(over="ignore", invalid="ignore"):
            notional = price * qty
            valid = bool(np.isfinite(notional).all() and (price > 0).all() and (qty > 0).all())
        if not valid:
            raise StreamFallbackError("invalid price/quantity")
        maker = batch.column("is_buyer_maker").to_numpy(zero_copy_only=False)
        return stamps, ids, price, qty, notional, n_trades, maker

    def add(self, batch: pa.RecordBatch) -> None:
        if batch.num_rows == 0:
            return
        stamps, ids, price, qty, notional, n_trades, maker = self._validated(batch)
        bar = (stamps - self.start) // _AGG_BAR_MS
        starts = np.flatnonzero(np.r_[True, bar[1:] != bar[:-1]])
        bars = bar[starts]
        buy = ~maker
        self.count[bars] += np.diff(np.r_[starts, len(bar)])
        self.trades[bars] += np.add.reduceat(n_trades, starts)
        for row, values in enumerate((
            np.where(buy, qty, 0.0), np.where(buy, 0.0, qty),
            np.where(buy, notional, 0.0), np.where(buy, 0.0, notional), qty, notional,
        )):
            self.sums[row, bars] += np.add.reduceat(values, starts)
        self.max_notional[bars] = np.maximum(self.max_notional[bars], np.maximum.reduceat(notional, starts))
        log_price = np.log(price)
        step = np.empty_like(log_price)
        step[1:] = log_price[1:] - log_price[:-1]
        step[0] = log_price[0] - self.tail[3] if self.tail is not None and self.tail[2] == bar[0] else 0.0
        step[starts[1:]] = 0.0
        self.rv[bars] += np.add.reduceat(step * step, starts)
        self.tail = (int(stamps[-1]), int(ids[-1]), int(bar[-1]), float(log_price[-1]))
        if self.slice_ms is not None:
            keep = (stamps >= self.slice_ms[0]) & (stamps < self.slice_ms[1])
            if keep.any():
                self.slices.append(batch.select(list(_SLICE_SOURCE_COLUMNS)).filter(pa.array(keep)))

    def bars(self) -> pd.DataFrame:
        used = np.flatnonzero(self.count)
        sums = self.sums[:, used]
        stamps = pd.to_datetime(self.start + used * _AGG_BAR_MS, unit="ms", utc=True).as_unit("ms")
        return pd.DataFrame({
            "timestamp": stamps,
            "n_agg_trades": self.count[used],
            "n_trades": self.trades[used],
            "taker_buy_qty": sums[0],
            "taker_sell_qty": sums[1],
            "taker_buy_notional": sums[2],
            "taker_sell_notional": sums[3],
            "max_agg_notional": self.max_notional[used],
            "vwap": sums[5] / sums[4],
            "rv_trade": self.rv[used],
        })

    def sliced(self) -> pd.DataFrame | None:
        if self.slice_ms is None:
            return None
        if not self.slices:
            return _empty_slice()
        frame = pa.Table.from_batches(self.slices).to_pandas()
        return frame.rename(columns={"transact_time": "transact_time_ms"}).reset_index(drop=True)


def derive_agg_trades_streamed(
    batches: Iterable[pa.RecordBatch], *, day: date, slice_hours: tuple[int, int] | None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """3m bars and optional hour-window slice from ordered batches, matching derive/slice_agg_trades.

    Raises StreamFallbackError when input is empty, unparseable, unordered or invalid, so the caller can
    rerun the full-frame path and surface its exact error.
    """
    if slice_hours is not None and not (0 <= slice_hours[0] < slice_hours[1] <= 24):
        raise ValueError("invalid slice window")
    stream = _AggTradeStream(day, slice_hours)
    try:
        for batch in batches:
            stream.add(batch)
    except pa.ArrowException as exc:
        raise StreamFallbackError(f"unparseable csv: {type(exc).__name__}") from exc
    if not stream.count.any():
        raise StreamFallbackError("no trades")
    return stream.bars(), stream.sliced()
