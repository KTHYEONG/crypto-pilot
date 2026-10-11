from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Literal

from src.common.errors import DataIntegrityError

METRICS_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)

BOOK_DEPTH_TAGS: tuple[str, ...] = (
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

BOOK_DEPTH_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "n_snapshots",
    *[f"depth_{t}" for t in BOOK_DEPTH_TAGS],
    *[f"notional_{t}" for t in BOOK_DEPTH_TAGS],
)

AGG_TRADES_COLUMNS: tuple[str, ...] = (
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
)


@dataclass(frozen=True, slots=True)
class ArchiveDataset:
    name: Literal["metrics", "bookDepth", "aggTrades"]
    first_day: date
    derived_columns: tuple[str, ...]


DATASETS: Mapping[str, ArchiveDataset] = {
    "metrics": ArchiveDataset(
        name="metrics",
        first_day=date(2020, 9, 1),
        derived_columns=METRICS_COLUMNS,
    ),
    "bookDepth": ArchiveDataset(
        name="bookDepth",
        first_day=date(2023, 1, 1),
        derived_columns=BOOK_DEPTH_COLUMNS,
    ),
    "aggTrades": ArchiveDataset(
        name="aggTrades",
        first_day=date(2019, 9, 1),
        derived_columns=AGG_TRADES_COLUMNS,
    ),
}


def daily_key(dataset: str, symbol: str, day: date) -> str:
    return f"data/futures/um/daily/{dataset}/{symbol}/{symbol}-{dataset}-{day.isoformat()}.zip"


def key_day(key: str) -> date:
    try:
        match = re.search(r".+-(\d{4}-\d{2}-\d{2})\.zip$", key)
        if match is None:
            raise ValueError("missing archive date suffix")
        day = date.fromisoformat(match.group(1))
    except (ValueError, IndexError) as exc:
        raise DataIntegrityError(f"malformed archive key: {key!r}") from exc
    return day


def is_perpetual_usdt(symbol: str) -> bool:
    return symbol.endswith("USDT") and "_" not in symbol
