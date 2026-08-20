"""Canonical Parquet schema column definitions for market-data storage.

Single source of truth for cross-module canonical column tuples, satisfying the
I2 single-source invariant.
"""

from __future__ import annotations

METRICS_CANONICAL_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "datetime",
    "available_at",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "long_short_ratio",
    "top_trader_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)
