from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(slots=True, frozen=True)
class RunnerResult:
    exit_code: int
    reason: str


@dataclass(slots=True, frozen=True)
class RunWindow:
    fetch_start: str
    is_start: str
    oos_start: str
    end_date: str
    fetch_start_date: date
    is_start_date: date
    oos_start_date: date
    end_date_value: date


@dataclass(slots=True, frozen=True)
class MarketDataBundle:
    data_maps: dict[str, dict[str, Any]]
    oos_data_maps: dict[str, dict[str, Any]]
    valid_symbols: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class TradeableScope:
    admitted: tuple[str, ...]
    dropped_by_reason: Mapping[str, tuple[str, ...]]


@dataclass(slots=True, frozen=True)
class TimeframeProbeResult:
    manifest: Any
    winning_cells: tuple[Any, ...]
    selected_timeframes: frozenset[str]
