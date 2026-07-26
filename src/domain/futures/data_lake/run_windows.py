from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pandas as pd

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class QuarterlyWindowConfig:
    warmup_days: int = 90
    l1_days: int = 365
    l2_days: int = 365
    l3_days: int = 90

    def __post_init__(self) -> None:
        if self.warmup_days <= 0:
            raise ValueError(f"warmup_days must be positive, got {self.warmup_days}")
        if self.l1_days <= 0:
            raise ValueError(f"l1_days must be positive, got {self.l1_days}")
        if self.l2_days <= 0:
            raise ValueError(f"l2_days must be positive, got {self.l2_days}")
        if self.l3_days <= 0:
            raise ValueError(f"l3_days must be positive, got {self.l3_days}")


@dataclass(slots=True, frozen=True)
class QuarterlyRunWindow:
    requested_date: date
    cutoff_date: date
    acquisition_start_ns: int
    l1_start_ns: int
    l2_start_ns: int
    l3_start_ns: int
    cutoff_exclusive_ns: int

    def __post_init__(self) -> None:
        if not (self.acquisition_start_ns < self.l1_start_ns < self.l2_start_ns < self.l3_start_ns < self.cutoff_exclusive_ns):
            raise ValueError(
                "window boundaries must be strictly increasing: "
                f"acq={self.acquisition_start_ns} < l1={self.l1_start_ns} "
                f"< l2={self.l2_start_ns} < l3={self.l3_start_ns} "
                f"< cutoff={self.cutoff_exclusive_ns}"
            )


def _quarter_end(requested: date) -> date:
    quarter_month = ((requested.month - 1) // 3) * 3 + 1
    quarter_start = date(requested.year, quarter_month, 1)
    if quarter_start >= requested:
        prior_quarter = quarter_start.replace(year=quarter_start.year - 1) if quarter_month == 1 else date(quarter_start.year, quarter_month - 3, 1)
        prior_quarter_end = prior_quarter.replace(day=28) + timedelta(days=4) - timedelta(days=1)
        return prior_quarter_end
    prior_quarter_end = quarter_start - timedelta(days=1)
    return prior_quarter_end


def resolve_completed_quarter_window(
    requested_date: date,
    config: QuarterlyWindowConfig,
) -> QuarterlyRunWindow:
    cutoff_date = _quarter_end(requested_date)
    cutoff_exclusive_dt = datetime(cutoff_date.year, cutoff_date.month, cutoff_date.day, tzinfo=UTC) + timedelta(days=1)
    cutoff_exclusive_ns = int(cutoff_exclusive_dt.timestamp() * 1_000_000_000)

    l3_start_dt = cutoff_exclusive_dt - timedelta(days=config.l3_days)
    l3_start_ns = int(l3_start_dt.timestamp() * 1_000_000_000)

    l2_start_dt = l3_start_dt - timedelta(days=config.l2_days)
    l2_start_ns = int(l2_start_dt.timestamp() * 1_000_000_000)

    l1_start_dt = l2_start_dt - timedelta(days=config.l1_days)
    l1_start_ns = int(l1_start_dt.timestamp() * 1_000_000_000)

    acquisition_start_dt = l1_start_dt - timedelta(days=config.warmup_days)
    acquisition_start_ns = int(acquisition_start_dt.timestamp() * 1_000_000_000)

    _logger.info(
        "resolved window: req=%s cutoff=%s acq=%s l1=%s l2=%s l3=%s cut_excl=%s",
        requested_date, cutoff_date,
        acquisition_start_dt.date(), l1_start_dt.date(),
        l2_start_dt.date(), l3_start_dt.date(),
        cutoff_exclusive_dt.date(),
    )

    return QuarterlyRunWindow(
        requested_date=requested_date,
        cutoff_date=cutoff_date,
        acquisition_start_ns=acquisition_start_ns,
        l1_start_ns=l1_start_ns,
        l2_start_ns=l2_start_ns,
        l3_start_ns=l3_start_ns,
        cutoff_exclusive_ns=cutoff_exclusive_ns,
    )


def build_quarterly_execution_calendar(window: QuarterlyRunWindow) -> pd.DatetimeIndex:
    cutoff_dt = datetime.fromtimestamp(window.cutoff_exclusive_ns / 1_000_000_000, tz=UTC)
    acq_dt = datetime.fromtimestamp(window.acquisition_start_ns / 1_000_000_000, tz=UTC)
    total_hours = int((cutoff_dt - acq_dt).total_seconds() // 3600)
    calendar = pd.date_range(start=acq_dt, periods=total_hours, freq="h", tz="UTC")
    return calendar


__all__ = [
    "QuarterlyRunWindow",
    "QuarterlyWindowConfig",
    "build_quarterly_execution_calendar",
    "resolve_completed_quarter_window",
]
