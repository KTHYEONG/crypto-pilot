from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

_MIRROR_BUCKET_PREFIX = "data/futures/um"
_KLINE_INFIX = "1m/"


@dataclass(frozen=True, slots=True)
class MirrorDataset:
    name: str
    cadence: Literal["daily", "monthly"]
    pack_by: Literal["symbol", "month"]
    bucket_root: str
    key_infix: str
    columns: tuple[str, ...]
    text_columns: frozenset[str]
    id_residual: bool = False


def _dataset(
    name: str,
    cadence: Literal["daily", "monthly"],
    pack_by: Literal["symbol", "month"],
    columns: tuple[str, ...],
    text_columns: frozenset[str] = frozenset(),
    *,
    key_infix: str = "",
    id_residual: bool = False,
) -> MirrorDataset:
    return MirrorDataset(
        name=name,
        cadence=cadence,
        pack_by=pack_by,
        bucket_root=f"{_MIRROR_BUCKET_PREFIX}/{cadence}/{name}/",
        key_infix=key_infix,
        columns=columns,
        text_columns=text_columns,
        id_residual=id_residual,
    )


_KLINE_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)

MIRROR_DATASET_SPECS: Mapping[str, MirrorDataset] = {
    spec.name: spec
    for spec in (
        _dataset(
            "metrics",
            "daily",
            "month",
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
            frozenset({"create_time", "symbol"}),
        ),
        _dataset(
            "fundingRate",
            "monthly",
            "month",
            ("calc_time", "funding_interval_hours", "last_funding_rate"),
        ),
        _dataset("klines", "monthly", "month", _KLINE_COLUMNS, key_infix=_KLINE_INFIX),
        _dataset("markPriceKlines", "monthly", "month", _KLINE_COLUMNS, key_infix=_KLINE_INFIX),
        _dataset("indexPriceKlines", "monthly", "month", _KLINE_COLUMNS, key_infix=_KLINE_INFIX),
        _dataset("premiumIndexKlines", "monthly", "month", _KLINE_COLUMNS, key_infix=_KLINE_INFIX),
        _dataset(
            "bookDepth",
            "daily",
            "symbol",
            ("timestamp", "percentage", "depth", "notional"),
            frozenset({"timestamp"}),
        ),
        _dataset(
            "aggTrades",
            "monthly",
            "symbol",
            (
                "agg_trade_id",
                "price",
                "quantity",
                "first_trade_id",
                "last_trade_id",
                "transact_time",
                "is_buyer_maker",
            ),
            frozenset({"is_buyer_maker"}),
            id_residual=True,
        ),
    )
}

MIRROR_DEFAULT_DATASETS: tuple[str, ...] = tuple(name for name in MIRROR_DATASET_SPECS if name != "aggTrades")
