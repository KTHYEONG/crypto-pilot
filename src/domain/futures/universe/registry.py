"""Append-only instrument lifecycle registry for PIT universe.

Time Complexity: O(S*R) where S = number of snapshots, R = rows per snapshot.
Space Complexity: O(S*R) for accumulated registry rows.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime

import pandas as pd

from .contracts import DataConfidence

__all__ = ["build_instrument_registry"]

_LOG = logging.getLogger(__name__)

_REGISTRY_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "symbol",
    "pair",
    "quote_asset",
    "margin_asset",
    "contract_type",
    "onboard_at",
    "status",
    "state_valid_from",
    "available_at",
    "confidence",
)

_REQUIRED_SNAPSHOT_COLUMNS: frozenset[str] = frozenset({"symbol", "status", "captured_at"})


def _build_instrument_id(symbol: str, onboard_at: datetime | None) -> str:
    """Construct deterministic instrument_id from symbol and optional onboard timestamp.

    Args:
        symbol: Exchange trading pair symbol (e.g. "BTCUSDT").
        onboard_at: UTC onboard datetime; None if unknown.

    Returns:
        Instrument ID string in canonical form.
    """
    if onboard_at is not None:
        ts_int = int(onboard_at.timestamp())
        return f"binance_usdt_perpetual:{symbol}:{ts_int}"
    return f"binance_usdt_perpetual:{symbol}"


def build_instrument_registry(
    raw_snapshots: Iterable[pd.DataFrame],
    *,
    first_observations: pd.DataFrame,
) -> pd.DataFrame:
    """Build append-only instrument registry from exchangeInfo snapshots.

    Args:
        raw_snapshots: Iterable of exchangeInfo DataFrames. Each must have:
            [symbol, status, captured_at]. Optional: [pair, quote_asset,
            margin_asset, contract_type, onboard_at].
            captured_at = when snapshot was collected (UTC datetime).
        first_observations: DataFrame with [instrument_id, symbol,
            first_observed_at, last_observed_at] for RECONSTRUCTED lifecycle.
            Used for instruments absent from all raw_snapshots.

    Returns:
        DataFrame with _REGISTRY_COLUMNS columns, sorted by
        (instrument_id, state_valid_from), no duplicate keys.
        OBSERVED rows come from raw_snapshots.
        RECONSTRUCTED rows come from first_observations fallback.

    Raises:
        ValueError: If a raw_snapshot is missing required columns
            [symbol, status, captured_at].
    """
    all_rows: list[dict[str, object]] = []
    observed_ids: set[str] = set()

    for snapshot_df in raw_snapshots:
        if snapshot_df.empty:
            continue

        missing = _REQUIRED_SNAPSHOT_COLUMNS - set(snapshot_df.columns)
        if missing:
            raise ValueError(f"raw_snapshot missing required columns: {sorted(missing)}")

        for _, row in snapshot_df.iterrows():
            symbol = str(row["symbol"])
            status = str(row["status"])
            captured_at = row["captured_at"]
            if hasattr(captured_at, "to_pydatetime"):
                captured_at = captured_at.to_pydatetime()
            elif isinstance(captured_at, str):
                captured_at = datetime.fromisoformat(captured_at)

            # Parse optional onboard_at
            onboard_raw = row.get("onboard_at") if "onboard_at" in row.index else None
            onboard_at: datetime | None = None
            if onboard_raw is not None and not (isinstance(onboard_raw, float) and pd.isna(onboard_raw)):
                if hasattr(onboard_raw, "to_pydatetime"):
                    onboard_at = onboard_raw.to_pydatetime()
                elif isinstance(onboard_raw, str):
                    onboard_at = datetime.fromisoformat(onboard_raw)
                elif isinstance(onboard_raw, datetime):
                    onboard_at = onboard_raw

            instrument_id = _build_instrument_id(symbol, onboard_at)
            observed_ids.add(instrument_id)

            all_rows.append(
                {
                    "instrument_id": instrument_id,
                    "symbol": symbol,
                    "pair": (
                        str(row.get("pair", symbol.replace("USDT", "")))
                        if "pair" in row.index
                        else symbol.replace("USDT", "")
                    ),
                    "quote_asset": (str(row.get("quote_asset", "USDT")) if "quote_asset" in row.index else "USDT"),
                    "margin_asset": (str(row.get("margin_asset", "USDT")) if "margin_asset" in row.index else "USDT"),
                    "contract_type": (
                        str(row.get("contract_type", "PERPETUAL")) if "contract_type" in row.index else "PERPETUAL"
                    ),
                    "onboard_at": onboard_at,
                    "status": status,
                    "state_valid_from": captured_at,
                    "available_at": captured_at,
                    "confidence": DataConfidence.OBSERVED.value,
                }
            )

    # RECONSTRUCTED fallback for instruments absent from all raw_snapshots
    if not first_observations.empty:
        for _, row in first_observations.iterrows():
            instrument_id = str(row["instrument_id"])
            if instrument_id in observed_ids:
                continue

            first_obs = row.get("first_observed_at")
            if hasattr(first_obs, "to_pydatetime"):
                first_obs = first_obs.to_pydatetime()
            elif isinstance(first_obs, str):
                first_obs = datetime.fromisoformat(first_obs)

            symbol = str(row.get("symbol", instrument_id.split(":")[-1] if ":" in instrument_id else instrument_id))

            all_rows.append(
                {
                    "instrument_id": instrument_id,
                    "symbol": symbol,
                    "pair": symbol.replace("USDT", ""),
                    "quote_asset": "USDT",
                    "margin_asset": "USDT",
                    "contract_type": "PERPETUAL",
                    "onboard_at": first_obs,
                    "status": "TRADING",
                    "state_valid_from": first_obs,
                    "available_at": first_obs,
                    "confidence": DataConfidence.RECONSTRUCTED.value,
                }
            )

    if not all_rows:
        _LOG.debug("build_instrument_registry: no rows — returning empty DataFrame")
        return pd.DataFrame(columns=list(_REGISTRY_COLUMNS))

    result = (
        pd.DataFrame(all_rows, columns=list(_REGISTRY_COLUMNS))
        .drop_duplicates(subset=["instrument_id", "state_valid_from"])
        .sort_values(["instrument_id", "state_valid_from"])
        .reset_index(drop=True)
    )
    _LOG.debug(
        "build_instrument_registry: %d rows, %d unique instruments",
        len(result),
        result["instrument_id"].nunique(),
    )
    return result
