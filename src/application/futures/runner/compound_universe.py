from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.domain.futures.data_lake.contracts import DataSnapshot, LakeUniverse, UniverseStateRequest

_logger = logging.getLogger(__name__)


class UniverseCoverageError(RuntimeError):
    ...


class EmptyPITUniverseError(RuntimeError):
    ...


class UniverseAxisLimitError(RuntimeError):
    ...


def build_daily_pit_universe(
    *, snapshot: DataSnapshot, execution_calendar: pd.DatetimeIndex,
    config: CompoundRunConfig,
) -> LakeUniverse:
    calendar_values = execution_calendar
    if execution_calendar.tz is not None:
        calendar_values = execution_calendar.tz_convert("UTC").tz_localize(None)
    timestamps_ns = calendar_values.to_numpy(dtype="datetime64[ns]").astype(np.int64)

    from src.domain.futures.data_lake.query import load_pit_universe_state

    lake = load_pit_universe_state(
        snapshot=snapshot,
        request=UniverseStateRequest(
            execution_timestamps_ns=timestamps_ns,
            max_axis_symbols=config.max_axis_symbols,
        ),
    )
    if not lake.symbols:
        raise EmptyPITUniverseError("no eligible symbols in lake universe")

    _logger.info(
        "built daily PIT universe: %d symbols from snapshot %s",
        len(lake.symbols), snapshot.snapshot_id,
    )
    return lake


__all__ = [
    "EmptyPITUniverseError",
    "UniverseAxisLimitError",
    "UniverseCoverageError",
    "build_daily_pit_universe",
]
