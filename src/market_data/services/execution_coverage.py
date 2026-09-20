from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.paths import FUTURES_DATA_DIR
from src.market_data.storage.ohlcv import is_temp_artifact

_EXECUTION_STEP = timedelta(minutes=3)


@dataclass(frozen=True, slots=True)
class ExecutionCoverageDeficit:
    """One symbol whose execution-timeframe archive falls short of its signal archive.

    The roster and every feature are derived from the 1h plane while fills and marks are
    derived from the 3m plane, so a symbol the roster can still select but the execution
    engine cannot price is a silent trap: the position is entered, cannot be exited, and
    the ledger only fails much later at an unrelated bar. Measuring the shortfall up front
    turns that trap into a list.

    Attributes:
        symbol: Upper-case exchange symbol present in the signal plane.
        signal_end: Last observed 1h bar open, UTC.
        execution_end: Last observed 3m bar open, UTC; None when no 3m archive exists.
        horizon: Effective coverage target, the earlier of `signal_end` and the requested end.
        deficit_days: Whole days between `execution_end` and `horizon`; full span when absent.
    """

    symbol: str
    signal_end: pd.Timestamp
    execution_end: pd.Timestamp | None
    horizon: pd.Timestamp
    deficit_days: int


def _require_horizon_end(horizon_end: pd.Timestamp) -> None:
    if not isinstance(horizon_end, pd.Timestamp):
        raise DataIntegrityError("horizon_end must be a tz-aware UTC Timestamp")
    if horizon_end.tzinfo is None:
        raise DataIntegrityError("horizon_end must be tz-aware UTC")
    if horizon_end.utcoffset() != timedelta(0):
        raise DataIntegrityError("horizon_end must be UTC")


def _read_bar_times(path: Path) -> list[pd.Timestamp]:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise DataIntegrityError(f"execution coverage unreadable parquet: {path}") from exc
    if frame.empty:
        return []
    if "timestamp" in frame.columns:
        numeric = pd.to_numeric(frame["timestamp"], errors="coerce").dropna()
        if numeric.empty:
            return []
        moments = pd.to_datetime(numeric.astype("int64"), unit="ms", utc=True)
    elif "datetime" in frame.columns:
        moments = pd.to_datetime(frame["datetime"], utc=True, errors="coerce").dropna()
        if moments.empty:
            return []
    else:
        raise DataIntegrityError(f"execution coverage parquet missing time column: {path}")
    uniq = sorted(set(moments.tolist()))
    out: list[pd.Timestamp] = []
    for moment in uniq:
        stamp = pd.Timestamp(moment)
        out.append(stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC"))
    return out


def _lake_path(symbol: str, timeframe: str, root: Path) -> Path:
    return root / "ohlcv" / timeframe / f"{symbol}.parquet"


def _signal_symbols(root: Path) -> list[str]:
    directory = root / "ohlcv" / "1h"
    if not directory.is_dir():
        return []
    names = [p.name for p in directory.glob("*.parquet")]
    return sorted(Path(n).stem for n in names if not is_temp_artifact(n))


def measure_execution_coverage(
    *,
    horizon_end: pd.Timestamp,
    symbols: Sequence[str] | None = None,
    data_root: Path | None = None,
) -> tuple[ExecutionCoverageDeficit, ...]:
    """Measure where the 3m execution archive trails the 1h signal archive.

    Coverage is judged per symbol against the earlier of its own 1h end and the caller's
    horizon, because a symbol delisted in 2023 is complete at 2023 and must not be reported
    as deficient for the years after it stopped trading.

    Args:
        horizon_end: Exclusive UTC end of the interval execution data must cover.
        symbols: Restrict to these symbols; None measures every symbol in the 1h plane.
        data_root: OHLCV root override; defaults to the canonical futures lake.
    Returns:
        Deficits ordered by descending `deficit_days`; empty when every symbol is covered.
    Raises:
        DataIntegrityError: `horizon_end` is not tz-aware UTC or a parquet is unreadable.
    """
    _require_horizon_end(horizon_end)
    root = Path(data_root) if data_root is not None else FUTURES_DATA_DIR
    wanted = list(symbols) if symbols is not None else _signal_symbols(root)
    deficits: list[ExecutionCoverageDeficit] = []
    for symbol in wanted:
        signal_times = _read_bar_times(_lake_path(symbol, "1h", root)) if _lake_path(symbol, "1h", root).exists() else None
        if signal_times is None:
            continue
        if not signal_times:
            continue
        signal_start = signal_times[0]
        signal_end = signal_times[-1]
        horizon = signal_end if signal_end < horizon_end else horizon_end
        exec_path = _lake_path(symbol, "3m", root)
        if not exec_path.exists():
            span_days = int((horizon - signal_start).total_seconds() // 86400)
            deficits.append(
                ExecutionCoverageDeficit(
                    symbol=symbol,
                    signal_end=signal_end,
                    execution_end=None,
                    horizon=horizon,
                    deficit_days=max(0, span_days),
                )
            )
            continue
        exec_times = _read_bar_times(exec_path)
        if not exec_times:
            span_days = int((horizon - signal_start).total_seconds() // 86400)
            deficits.append(
                ExecutionCoverageDeficit(
                    symbol=symbol,
                    signal_end=signal_end,
                    execution_end=None,
                    horizon=horizon,
                    deficit_days=max(0, span_days),
                )
            )
            continue
        execution_end = exec_times[-1]
        if horizon - execution_end <= _EXECUTION_STEP:
            continue
        deficit_days = int((horizon - execution_end).total_seconds() // 86400)
        deficits.append(
            ExecutionCoverageDeficit(
                symbol=symbol,
                signal_end=signal_end,
                execution_end=execution_end,
                horizon=horizon,
                deficit_days=deficit_days,
            )
        )
    deficits.sort(key=lambda d: (-d.deficit_days, d.symbol))
    return tuple(deficits)


def plan_execution_coverage_backfill(
    deficits: Sequence[ExecutionCoverageDeficit],
    *,
    horizon_end: pd.Timestamp,
    lookback_days: int = 3,
) -> tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...]:
    """Turn measured deficits into exact per-symbol collection windows.

    Each window starts `lookback_days` before the last observed execution bar so the
    archive's own month boundary is re-read rather than stitched, which is how the earlier
    manual backfill left symbols truncated mid-holding.

    Args:
        deficits: Result of `measure_execution_coverage`.
        horizon_end: Exclusive UTC end shared by every planned window.
        lookback_days: Non-negative overlap re-read before each resume point.
    Returns:
        `(symbol, start, end)` windows in the deficits' order.
    Raises:
        DataIntegrityError: `lookback_days` is negative or a deficit is already covered.
    """
    _require_horizon_end(horizon_end)
    if lookback_days < 0:
        raise DataIntegrityError("lookback_days must be non-negative")
    windows: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for deficit in deficits:
        if deficit.execution_end is not None and horizon_end - deficit.execution_end <= _EXECUTION_STEP:
            raise DataIntegrityError(f"deficit already covered: {deficit.symbol}")
        if deficit.execution_end is None:
            start = deficit.horizon - timedelta(days=deficit.deficit_days)
        else:
            start = deficit.execution_end - timedelta(days=lookback_days)
        windows.append((deficit.symbol, start, horizon_end))
    return tuple(windows)
