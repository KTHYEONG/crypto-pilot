"""Common technical window resolver for the library admission pipeline.

Owns the deterministic resolver that validates continuous 1h OHLCV plus
funding availability for every requested symbol and returns the latest
compatible 4h-aligned start and the earliest fully completed common end. A
requested start earlier than that common boundary fails closed with every
blocking symbol and source; a window is never silently truncated, zero-filled,
or extended past its last settled bar.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from src.common.config import funding_path, ohlcv_path
from src.common.errors import DataIntegrityError
from src.market_data.storage.loaders import (
    load_funding_rates,
    load_ohlcv_1h_as,
    timeframe_period,
)
from src.research.technical_experts.catalog import TECHNICAL_CANDIDATES

# The longest frozen candidate requires min_history_bars completed bars of the
# research timeframe before its first decision; reported performance starts
# only after this warm-up. The count is timeframe-agnostic (bars), so the
# duration is derived per call from the resolved timeframe period.
_MAX_CANDIDATE_WARMUP_BARS = max(
    candidate.min_history_bars for candidate in TECHNICAL_CANDIDATES
)


@dataclass(frozen=True, slots=True)
class ResolvedTechnicalWindow:
    """Resolved common availability window for one technical universe.

    ``common_start`` is the latest required OHLCV or funding start aligned
    forward to the 4h grid, ``common_end`` is the earliest fully completed
    common 4h bar across every symbol and source (never extended past a settled
    funding boundary), ``effective_start`` is ``common_start`` advanced past the
    maximum candidate warm-up (no warm-up return is ever scored), and
    ``symbol_sources`` names every source start and end per symbol for
    fail-closed diagnostics.
    """

    requested_start: str | None
    common_start: pd.Timestamp
    common_end: pd.Timestamp
    effective_start: pd.Timestamp
    end: str | None
    symbol_sources: Mapping[str, Mapping[str, str]]


def _align_to_grid(ts: pd.Timestamp, period: pd.Timedelta) -> pd.Timestamp:
    minutes = ts.hour * 60 + ts.minute + ts.second / 60.0 + ts.microsecond / 60_000_000.0
    bucket_minutes = period.total_seconds() / 60.0
    slot = math.ceil(minutes / bucket_minutes)
    return ts.normalize() + pd.Timedelta(minutes=slot * bucket_minutes)


def _last_settled_bar(end: pd.Timestamp, period: pd.Timedelta) -> pd.Timestamp:
    """Return the latest completed bar start fully settled by a funding timestamp.

    A bar ``[t, t+period)`` is settled only when a funding event exists strictly
    before its close, so a funding timestamp exactly on a bucket boundary
    settles the preceding bar and is never extended.
    """
    bucket_minutes = int(period.total_seconds() / 60.0)
    minutes = end.hour * 60 + end.minute
    floored = end.normalize() + pd.Timedelta(minutes=(minutes // bucket_minutes) * bucket_minutes)
    if floored == end:
        return floored - period
    return floored


def _validate_funding_continuity(funding: pd.Series, symbol: str) -> None:
    if len(funding) == 0:
        raise DataIntegrityError(f"funding has no settled events for {symbol}")
    if not isinstance(funding.index, pd.DatetimeIndex) or funding.index.tz is None:
        raise DataIntegrityError(f"funding for {symbol} must have a tz-aware UTC index")
    if not funding.index.is_monotonic_increasing:
        raise DataIntegrityError(f"funding for {symbol} is not monotonic increasing")
    if funding.index.has_duplicates:
        raise DataIntegrityError(f"funding for {symbol} contains duplicate events")


def resolve_common_technical_window(
    symbols: tuple[str, ...],
    requested_start: str | None,
    end: str | pd.Timestamp | None,
    *,
    timeframe: str = "4h",
) -> ResolvedTechnicalWindow:
    """Resolve the latest common OHLCV/funding window for a technical universe.

    Every symbol's continuous 1h OHLCV and monotonic settled funding are
    validated through the generalized loader resampled to ``timeframe``, and the
    common start is aligned forward to that bucket grid while ``common_end`` is
    the earliest fully completed common bar, never extended past a settled
    funding boundary. A requested start earlier than the common boundary fails
    closed with every blocking symbol and source; a missing, empty,
    discontinuous, or non-overlapping source fails closed naming the source. A
    window is never silently truncated or zero-filled. The default ``"4h"``
    preserves existing behavior for legacy callers.
    """
    if not symbols:
        raise ValueError("symbols must not be empty")
    if len(symbols) != len(set(symbols)):
        raise ValueError(f"symbols must not contain duplicates, got {symbols}")

    period = timeframe_period(timeframe)
    symbol_sources: dict[str, dict[str, str]] = {}
    common_start: pd.Timestamp | None = None
    common_end: pd.Timestamp | None = None
    for symbol in symbols:
        ohlcv = load_ohlcv_1h_as(ohlcv_path(symbol, "1h"), timeframe)
        if len(ohlcv) == 0:
            raise DataIntegrityError(f"1h OHLCV has no bars for {symbol}")
        funding = load_funding_rates(funding_path(symbol))
        _validate_funding_continuity(funding, symbol)
        ohlcv_start = ohlcv.index[0]
        ohlcv_end = ohlcv.index[-1]
        funding_start = funding.index[0]
        funding_end = funding.index[-1]
        if funding_end < ohlcv_start or funding_start > ohlcv_end:
            raise DataIntegrityError(
                f"{symbol} OHLCV and funding sources do not overlap: ohlcv "
                f"[{ohlcv_start}, {ohlcv_end}] vs funding [{funding_start}, "
                f"{funding_end}]"
            )
        symbol_sources[symbol] = {
            "ohlcv": str(ohlcv_start),
            "ohlcv_end": str(ohlcv_end),
            "funding": str(funding_start),
            "funding_end": str(funding_end),
        }
        required = max(ohlcv_start, funding_start)
        if common_start is None or required > common_start:
            common_start = required
        usable_end = min(ohlcv_end, _last_settled_bar(funding_end, period))
        if common_end is None or usable_end < common_end:
            common_end = usable_end

    assert common_start is not None
    assert common_end is not None
    if end is not None:
        requested_end = pd.Timestamp(end, tz="UTC")
        common_end = min(common_end, _last_settled_bar(requested_end, period))
    if common_end < common_start:
        raise DataIntegrityError(
            "common availability window is empty after settling: common_end "
            f"{common_end} precedes common_start {common_start}"
        )
    common_start = _align_to_grid(common_start, period)
    if requested_start is not None:
        requested = pd.Timestamp(requested_start, tz="UTC")
        if requested < common_start:
            blockers = sorted(
                f"{symbol}:{source} ({start})"
                for symbol, starts in symbol_sources.items()
                for source, start in starts.items()
                if pd.Timestamp(start, tz="UTC") > requested
            )
            raise DataIntegrityError(
                f"requested start {requested_start} precedes the common available "
                f"start {common_start}; blocking source(s): {', '.join(blockers)}"
            )
        effective_start = _align_to_grid(requested, period)
    else:
        effective_start = common_start
    effective_start = effective_start + _MAX_CANDIDATE_WARMUP_BARS * period
    return ResolvedTechnicalWindow(
        requested_start=str(requested_start) if requested_start is not None else None,
        common_start=common_start,
        common_end=common_end,
        effective_start=effective_start,
        end=str(end) if end is not None else None,
        symbol_sources=symbol_sources,
    )
