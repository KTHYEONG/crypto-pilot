"""Regime-boundary reference prices from the local 1h OHLCV lake."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Collection
from decimal import Decimal
from pathlib import Path

import pandas as pd
from pyarrow import ArrowException

from src.live.tax_schema import (
    BOUNDARY_MARK_SOURCE_OHLCV_1H_CLOSE,
    BoundaryMark,
)

logger = logging.getLogger("TaxBoundaryMarks")


def derive_ohlcv_boundary_marks(
    symbols: Collection[str],
    boundary: pd.Timestamp,
    *,
    lake_path: Callable[[str], Path] | None = None,
) -> dict[str, BoundaryMark]:
    """Regime-boundary reference prices from the local 1h OHLCV lake, one per symbol.

    The reference is the close of the 1h bar that ends at ``boundary`` (lake rows are keyed by
    bar open time in epoch milliseconds, so the row with ``timestamp == boundary - 1h``). Only
    that exact row is read (column pruning and predicate pushdown); any other bar, an
    interpolation or a later price would be a fabricated value, so every failure yields
    ``price=None`` with a reason instead.

    Args:
        symbols: Symbols needing a mark.
        boundary: tz-aware UTC regime boundary (``regime_boundary_utc``).
        lake_path: Symbol -> 1h parquet path; None resolves ``src.common.paths.ohlcv_path(symbol,
            "1h")`` at call time.

    Returns:
        ``{symbol: BoundaryMark(source=BOUNDARY_MARK_SOURCE_OHLCV_1H_CLOSE, ...)}`` for every input
        symbol.

    Raises:
        ValueError: ``boundary`` is naive or not on an exact UTC hour.
    """
    if not isinstance(boundary, pd.Timestamp) or boundary.tzinfo is None:
        raise ValueError(f"boundary must be tz-aware: {boundary!r}")
    moment = boundary.tz_convert("UTC")
    if moment.minute != 0 or moment.second != 0 or moment.microsecond != 0 or moment.nanosecond != 0:
        raise ValueError(f"boundary must be on an exact UTC hour: {boundary!r}")
    open_ms = int((moment - pd.Timedelta(hours=1)).timestamp() * 1000)
    resolver: Callable[[str], Path]
    if lake_path is not None:
        resolver = lake_path
    else:
        from src.common.paths import ohlcv_path as _ohlcv_path

        def resolver(symbol: str) -> Path:
            return _ohlcv_path(symbol, "1h")
    out: dict[str, BoundaryMark] = {}
    for symbol in symbols:
        path = resolver(symbol)
        mark = _read_one(symbol, path, open_ms)
        out[symbol] = mark
    available = sum(1 for m in out.values() if m.price is not None)
    logger.info(
        "[DATA] tax_boundary_marks boundary=%s symbols=%d available=%d",
        moment.isoformat(),
        len(out),
        available,
    )
    for symbol, mark in out.items():
        if mark.price is None:
            logger.warning(
                "[DATA] tax_boundary_mark_unavailable symbol=%s reason=%s",
                symbol,
                mark.unavailable_reason,
            )
    return out


def _read_one(symbol: str, path: Path, open_ms: int) -> BoundaryMark:
    target = Path(path)
    if not target.exists():
        return BoundaryMark(
            price=None, source=BOUNDARY_MARK_SOURCE_OHLCV_1H_CLOSE, unavailable_reason="ohlcv_file_missing"
        )
    try:
        frame = pd.read_parquet(
            target, columns=["timestamp", "close"], filters=[("timestamp", "==", open_ms)]
        )
    except (OSError, ArrowException, KeyError, ValueError) as exc:
        return BoundaryMark(
            price=None,
            source=BOUNDARY_MARK_SOURCE_OHLCV_1H_CLOSE,
            unavailable_reason=f"ohlcv_unreadable:{type(exc).__name__}",
        )
    if "timestamp" not in frame.columns or "close" not in frame.columns:
        return BoundaryMark(None, BOUNDARY_MARK_SOURCE_OHLCV_1H_CLOSE, "ohlcv_unreadable:KeyError")
    if len(frame) == 0:
        return BoundaryMark(
            price=None, source=BOUNDARY_MARK_SOURCE_OHLCV_1H_CLOSE, unavailable_reason="ohlcv_bar_missing"
        )
    if len(frame) > 1:
        return BoundaryMark(
            price=None,
            source=BOUNDARY_MARK_SOURCE_OHLCV_1H_CLOSE,
            unavailable_reason="ohlcv_bar_duplicated",
        )
    close = frame["close"].iloc[0]
    try:
        numeric = float(close)
    except (TypeError, ValueError):
        return BoundaryMark(
            price=None, source=BOUNDARY_MARK_SOURCE_OHLCV_1H_CLOSE, unavailable_reason="ohlcv_close_invalid"
        )
    if not math.isfinite(numeric) or numeric <= 0:
        return BoundaryMark(
            price=None, source=BOUNDARY_MARK_SOURCE_OHLCV_1H_CLOSE, unavailable_reason="ohlcv_close_invalid"
        )
    return BoundaryMark(
        price=Decimal(str(close)), source=BOUNDARY_MARK_SOURCE_OHLCV_1H_CLOSE
    )
