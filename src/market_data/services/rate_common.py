"""Shared rate-related constants for borrow/spot collection services.

Single source of truth for the cadence/liquidity constants that the borrow and
spot collectors both declare, satisfying the I2 single-source invariant.
"""

from __future__ import annotations

import pandas as pd

BORROW_CANONICAL_COLUMNS: tuple[str, ...] = ("timestamp", "borrow_rate", "accrual_seconds")

RATE_PERIOD_SECONDS: dict[str, int] = {
    "annual": 365 * 86400,
    "1y": 365 * 86400,
    "365d": 365 * 86400,
    "daily": 86400,
    "1d": 86400,
    "hourly": 3600,
    "1h": 3600,
    "3600s": 3600,
}

SECONDS_PER_DAY = 86400.0
INTEREST_HISTORY_BOUNDARY = pd.Timedelta(days=31)
