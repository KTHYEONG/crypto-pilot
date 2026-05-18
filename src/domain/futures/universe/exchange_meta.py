"""Exchange metadata normalization for futures universe pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd


def _to_utc_date_str(value: Any) -> str | None:
    """Convert timestamp-like input into YYYY-MM-DD in UTC."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and value > 0:
        ts = datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
        return ts.date().isoformat()
    if isinstance(value, str) and value:
        return value[:10]
    return None


def normalize_exchange_info(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize exchangeInfo symbols into vectorizable metadata frame.

    Args:
        records: Raw symbol records from exchange info.

    Returns:
        DataFrame with normalized columns required by Stage 1/5.

    """
    if not records:
        return pd.DataFrame(
            columns=[
                "symbol",
                "pair",
                "contract_type",
                "status",
                "quote_asset",
                "margin_asset",
                "onboard_date",
                "delivery_date",
            ]
        )

    frame = pd.DataFrame.from_records(records)
    required = {
        "symbol": "",
        "pair": "",
        "contractType": "",
        "status": "",
        "quoteAsset": "",
        "marginAsset": "",
        "onboardDate": None,
        "deliveryDate": None,
    }
    for col, default in required.items():
        if col not in frame.columns:
            frame[col] = default

    out = pd.DataFrame(
        {
            "symbol": frame["symbol"].astype("string"),
            "pair": frame["pair"].astype("string"),
            "contract_type": frame["contractType"].astype("string"),
            "status": frame["status"].astype("string"),
            "quote_asset": frame["quoteAsset"].astype("string"),
            "margin_asset": frame["marginAsset"].astype("string"),
            "onboard_date": frame["onboardDate"].map(_to_utc_date_str).astype("string"),
            "delivery_date": frame["deliveryDate"].map(_to_utc_date_str).astype("string"),
            "is_listed": frame["onboardDate"].notna(),
            "is_trading": frame["status"].astype("string").eq("TRADING"),
        }
    )
    return out
