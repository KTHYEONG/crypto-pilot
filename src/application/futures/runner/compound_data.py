from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.application.futures.runner.compound_universe import DailyPITUniverse
from src.domain.futures.compound.contracts import MarketFeatureCube
from src.domain.futures.data_lake.contracts import DataSnapshot, GridRequest
from src.domain.futures.data_lake.query import materialize_native_grid

_logger = logging.getLogger(__name__)
def check_data_readiness(
    data_maps: dict[str, dict[str, pd.DataFrame]],
    *,
    min_ready_pct: float = 0.80,
) -> bool:
    if not data_maps:
        return False
    ready = sum(1 for sym_data in data_maps.values() if "1h" in sym_data and not sym_data["1h"].empty)
    ratio = ready / max(len(data_maps), 1)
    if ratio < min_ready_pct:
        _logger.error(
            "data readiness %.1f%% < %.1f%% (%d/%d symbols ready)",
            ratio * 100, min_ready_pct * 100, ready, len(data_maps),
        )
        return False
    return True


def build_multiscale_market_cube(
    *, snapshot: DataSnapshot, universe: DailyPITUniverse, config: CompoundRunConfig
) -> MarketFeatureCube:
    _logger.info(
        "building multiscale market cube: %d symbols from snapshot %s",
        len(universe.symbols), snapshot.snapshot_id,
    )
    symbols = universe.symbols
    n_syms = len(symbols)

    from datetime import UTC, datetime

    ref_date_str = config.reference_date or datetime.now(UTC).strftime("%Y-%m-%d")
    ref_dt = pd.Timestamp(ref_date_str, tz="UTC")
    start_dt = ref_dt - pd.Timedelta(days=config.history_days)
    n_bars = config.history_days * 24
    execution_calendar = pd.date_range(start=start_dt, periods=n_bars, freq="h", tz="UTC")
    timestamps_ns = execution_calendar.to_numpy(dtype="datetime64[ns]").astype(np.int64)

    core_names = ("open", "high", "low", "close", "quote_volume")
    grid = materialize_native_grid(
        request=GridRequest(
            symbols=symbols,
            timeframe="1h",
            source_timeframe="1h",
            fields=core_names,
            start_time_ns=int(timestamps_ns[0]),
            end_time_ns=int(timestamps_ns[-1] + 3_600_000_000_000),
        ),
        snapshot=snapshot,
    )
    arrays: dict[str, NDArray[np.float64]] = {
        name: np.asarray(grid.fields[name], dtype=np.float64) for name in core_names
    }
    funding = np.zeros((n_bars, n_syms), dtype=np.float32)
    available_core = np.logical_and.reduce(
        [np.asarray(grid.available[name], dtype=np.bool_) for name in core_names]
    )
    eligible = np.ones((n_bars, n_syms), dtype=np.bool_)
    entry_block = np.zeros((n_bars, n_syms), dtype=np.bool_)
    exit_required = np.zeros((n_bars, n_syms), dtype=np.bool_)
    capacity = np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64)
    costs = np.full((n_bars, n_syms), 12.0, dtype=np.float32)

    fields: dict[str, NDArray[np.float32] | NDArray[np.float64]] = {
        **arrays,
        "quote_volume": arrays["quote_volume"].astype(np.float32),
        "funding": funding,
    }

    cube = MarketFeatureCube(
        timestamps_ns=timestamps_ns,
        symbols=symbols,
        fields_2d=fields,
        available_2d={"core": available_core},
        eligible_2d=eligible,
        entry_block_2d=entry_block,
        exit_required_2d=exit_required,
        capacity_usdt_2d=capacity,
        execution_cost_bps_2d=costs,
        data_manifest_hash=snapshot.manifest_hash,
    )

    _logger.info(
        "multiscale market cube built: %d bars x %d symbols",
        n_bars, n_syms,
    )
    return cube


__all__ = [
    "build_multiscale_market_cube",
    "check_data_readiness",
]
