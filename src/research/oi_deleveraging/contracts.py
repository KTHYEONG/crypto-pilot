from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class OIDeleveragingMarketData:
    """Aligned causal inputs for the open-interest deleveraging screen.

    ``bars`` is the exact 4h futures OHLCV grid (tz-aware UTC, strictly
    monotonic) used for execution; ``joined`` carries one row per bar with the
    as-of matched daily metrics feature plus the completed 24h mark return, so
    each decision uses only data released before its ``decision_time``; and
    ``funding`` holds the published perp settlement rates that credit the short
    leg. A missing metric is a no-signal interval and is never imputed.
    """

    symbol: str
    bars: pd.DataFrame
    joined: pd.DataFrame
    funding: pd.Series
