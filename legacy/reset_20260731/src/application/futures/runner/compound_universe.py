from __future__ import annotations

import logging
from collections.abc import Collection

import numpy as np
import pandas as pd

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.domain.futures.data_lake.contracts import DataSnapshot, LakeUniverse, UniverseStateRequest
from src.domain.futures.universe.contracts import UniverseStateCube

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


def restrict_pit_universe_to_symbols(
    *, universe: LakeUniverse, allowed_symbols: Collection[str],
) -> LakeUniverse:
    """Return a PIT universe whose axis contains only CORE-complete symbols.

    The data lake's historical coverage plan and the market cube must share one
    symbol axis.  Retaining unavailable symbols as all-false rows leaks missing
    data handling into clustering and L1 admission.
    """
    allowed = frozenset(allowed_symbols)
    indices = np.asarray(
        [index for index, symbol in enumerate(universe.symbols) if symbol in allowed],
        dtype=np.intp,
    )
    if indices.size == 0:
        raise EmptyPITUniverseError("no CORE-complete symbols in PIT universe")

    symbols = tuple(universe.symbols[index] for index in indices)
    cube = universe.state_cube
    filtered_cube = UniverseStateCube(
        calendar=cube.calendar,
        instrument_ids=symbols,
        eligible=cube.eligible[:, indices].copy(),
        entry_block=cube.entry_block[:, indices].copy(),
        exit_required=cube.exit_required[:, indices].copy(),
        capacity_usdt=cube.capacity_usdt[:, indices].copy(),
        risk_scale=cube.risk_scale[:, indices].copy(),
        cost_bps=cube.cost_bps[:, indices].copy(),
    )
    return LakeUniverse(
        symbols=symbols,
        state_cube=filtered_cube,
        state_hash=universe.state_hash,
    )


__all__ = [
    "EmptyPITUniverseError",
    "UniverseAxisLimitError",
    "UniverseCoverageError",
    "build_daily_pit_universe",
    "restrict_pit_universe_to_symbols",
]
