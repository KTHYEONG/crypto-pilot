from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src.core.types import PortfolioSpec


def _valid_ohlcv_index(frame: pd.DataFrame) -> bool:
    """True only for a tz-aware, monotonic, duplicate-free DatetimeIndex."""
    idx = frame.index
    return (
        isinstance(idx, pd.DatetimeIndex)
        and idx.tz is not None
        and idx.is_unique
        and idx.is_monotonic_increasing
        and not idx.hasnans
    )


def select_liquid_universe(
    frames: Mapping[str, pd.DataFrame],
    as_of: pd.Timestamp,
    spec: PortfolioSpec,
) -> tuple[str, ...]:
    """Return up to ``spec.universe_size`` symbols ranked by trailing liquidity.

    Selection is strictly causal: for ``as_of``, only bars whose completed end
    is ``<= as_of`` are considered, and quote volume is summed over the trailing
    ``liquidity_lookback_days`` window declared by ``PortfolioSpec``. A symbol is
    excluded when its frame is malformed (missing/non-UTC/non-monotonic/gapped),
    when it has no completed bar inside the trailing window, or when any window
    ``quote_vol`` is invalid. Ties and the final ordering are resolved by
    descending trailing liquidity then symbol lexicographically, so the result
    is deterministic and never depends on realized returns or future volume.
    """
    if as_of.tzinfo is None:
        raise ValueError(f"as_of must be tz-aware UTC, got {as_of}")
    as_of = as_of.tz_convert("UTC")

    window_start = as_of - pd.Timedelta(days=spec.liquidity_lookback_days)
    liquidity: dict[str, float] = {}

    for symbol, frame in frames.items():
        if frame.empty or not _valid_ohlcv_index(frame):
            continue
        if "quote_vol" not in frame.columns:
            continue

        diffs = frame.index.to_series().diff().dropna()
        if diffs.empty:
            continue
        period = pd.Timedelta(diffs.median())
        if period <= pd.Timedelta(0) or (diffs != period).any():
            continue

        completed = frame[frame.index <= as_of - period]
        if completed.empty:
            continue
        window = completed[completed.index + period > window_start]
        if window.empty:
            continue

        quote_vol = pd.to_numeric(window["quote_vol"], errors="coerce")
        if quote_vol.isna().any() or (quote_vol < 0).any():
            continue
        liquidity[symbol] = float(quote_vol.sum())

    ranked = sorted(liquidity.items(), key=lambda item: (-item[1], item[0]))
    return tuple(symbol for symbol, _ in ranked[: spec.universe_size])
