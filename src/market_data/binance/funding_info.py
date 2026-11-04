"""Venue-attested funding cadence (``/fapi/v1/fundingInfo``).

The settlement interval of a perpetual changes without notice, and a stored series cannot tell an
interval change from a missed settlement. This module carries the venue's own statement of the
cadence as a point-in-time snapshot.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.durable_io import durable_replace
from src.common.errors import DataIntegrityError
from src.market_data.binance.futures import FUNDING_RATE_LIMITER

_logger = logging.getLogger(__name__)

FUNDING_INFO_URL: str = "https://fapi.binance.com/fapi/v1/fundingInfo"
FUNDING_INFO_TIMEOUT_S: float = 5.0
FUNDING_INTERVAL_HOURS_ALLOWED: frozenset[int] = frozenset({1, 2, 4, 8})
_FUNDING_HOUR_MS: int = 3_600_000
_SNAPSHOT_STEM_FORMAT: str = "%Y%m%dT%H%M%SZ"
_SNAPSHOT_STEM_LENGTH: int = 16


@dataclass(frozen=True, slots=True)
class FundingIntervalSnapshot:
    """Per-symbol funding interval as the venue stated it at ``captured_at``."""

    captured_at: pd.Timestamp
    interval_ms: Mapping[str, int]

    def interval_for(self, symbol: str) -> int | None:
        """Venue-stated interval in ms, or None when the venue did not list ``symbol``."""
        return self.interval_ms.get(symbol)


def _utc(stamp: pd.Timestamp, label: str) -> pd.Timestamp:
    ts = pd.Timestamp(stamp)
    if ts.tzinfo is None:
        raise DataIntegrityError(f"{label} must be timezone-aware UTC")
    return ts.tz_convert("UTC")


def _valid_hours(raw: object) -> int | None:
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return raw if raw in FUNDING_INTERVAL_HOURS_ALLOWED else None


def parse_funding_info(payload: object, *, captured_at: pd.Timestamp) -> FundingIntervalSnapshot:
    """Validate a ``fundingInfo`` payload into a snapshot.

    Raises:
        DataIntegrityError: naive ``captured_at``, a payload that is not a non-empty list, or no valid row.
    """
    captured = _utc(captured_at, "captured_at")
    if not isinstance(payload, list) or not payload:
        raise DataIntegrityError("fundingInfo payload must be a non-empty list")
    intervals: dict[str, int] = {}
    skipped = 0
    for row in payload:
        symbol = row.get("symbol") if isinstance(row, Mapping) else None
        hours = _valid_hours(row.get("fundingIntervalHours")) if isinstance(row, Mapping) else None
        if not isinstance(symbol, str) or not symbol or hours is None:
            skipped += 1
            continue
        intervals[symbol] = hours * _FUNDING_HOUR_MS
    if not intervals:
        raise DataIntegrityError("fundingInfo payload has no valid rows")
    if skipped:
        _logger.warning("[DATA] stage=funding_info skipped_rows=%d total=%d", skipped, len(payload))
    return FundingIntervalSnapshot(captured_at=captured, interval_ms=intervals)


def fetch_funding_info(
    *,
    timeout_s: float = FUNDING_INFO_TIMEOUT_S,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> FundingIntervalSnapshot:
    """One GET of ``fundingInfo`` (shares the fundingRate request budget); never retries.

    Raises:
        RuntimeError: any HTTP, transport, timeout or JSON failure.
        DataIntegrityError: the response parses but is not a valid payload.
    """
    FUNDING_RATE_LIMITER.acquire()
    try:
        with opener(FUNDING_INFO_URL, timeout=timeout_s) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(f"fundingInfo request failed: {exc!r}") from exc
    return parse_funding_info(payload, captured_at=pd.Timestamp.now(tz="UTC"))


def write_funding_interval_snapshot(snapshot: FundingIntervalSnapshot, root: Path) -> Path:
    """Persist ``snapshot`` as ``<root>/<UTC capture instant>.json``; older files are kept."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    captured = _utc(snapshot.captured_at, "captured_at")
    path = root_path / f"{captured.strftime(_SNAPSHOT_STEM_FORMAT)}.json"
    body = json.dumps(
        {
            "captured_at": captured.isoformat(),
            "interval_hours": {sym: ms // _FUNDING_HOUR_MS for sym, ms in snapshot.interval_ms.items()},
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    def _write(tmp: Path) -> None:
        tmp.write_text(body, encoding="utf-8")

    durable_replace(path, _write)
    return path


def _read_snapshot_file(path: Path) -> FundingIntervalSnapshot:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        hours = raw["interval_hours"]
        captured = pd.Timestamp(raw["captured_at"])
        intervals = {str(sym): int(h) * _FUNDING_HOUR_MS for sym, h in hours.items()}
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise DataIntegrityError(f"funding interval snapshot unreadable: {path.name}") from exc
    if not intervals or any(ms // _FUNDING_HOUR_MS not in FUNDING_INTERVAL_HOURS_ALLOWED for ms in intervals.values()):
        raise DataIntegrityError(f"funding interval snapshot malformed: {path.name}")
    return FundingIntervalSnapshot(captured_at=_utc(captured, "captured_at"), interval_ms=intervals)


def latest_funding_interval_snapshot_or_none(root: Path) -> FundingIntervalSnapshot | None:
    """Newest persisted snapshot, or None when none exists.

    Raises:
        DataIntegrityError: the newest file is unreadable or malformed.
    """
    root_path = Path(root)
    if not root_path.exists():
        return None
    files = sorted(p for p in root_path.glob("*.json") if len(p.stem) == _SNAPSHOT_STEM_LENGTH)
    if not files:
        return None
    return _read_snapshot_file(files[-1])
