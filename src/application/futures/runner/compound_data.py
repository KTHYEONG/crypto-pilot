from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.application.futures.runner.compound_config import CompoundRunConfig
from src.domain.futures.compound.contracts import MarketFeatureCube
from src.domain.futures.data_lake.contracts import (
    DatasetKind,
    DataSnapshot,
    GridRequest,
    LakeUniverse,
    NativeFeatureGrid,
)
from src.domain.futures.data_lake.query import materialize_causal_metrics_grid, materialize_feature_grid

_logger = logging.getLogger(__name__)


def build_multiscale_market_cube(
    *, snapshot: DataSnapshot, universe: LakeUniverse, config: CompoundRunConfig
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

    core_names = (
        "open", "high", "low", "close", "quote_volume", "taker_buy_quote",
    )
    grid = materialize_feature_grid(
        request=GridRequest(
            symbols=symbols,
            timeframe="1h",
            source_timeframe="1h",
            fields=core_names,
            start_time_ns=int(timestamps_ns[0]),
            end_time_ns=int(timestamps_ns[-1] + 3_600_000_000_000),
        ),
        snapshot=snapshot,
        dataset=DatasetKind.KLINES_1H,
    )
    arrays: dict[str, NDArray[np.float64]] = {
        name: np.asarray(grid.fields[name], dtype=np.float64) for name in core_names
    }
    funding_request = GridRequest(
            symbols=symbols,
            timeframe="1h",
            source_timeframe="1h",
            fields=("funding_rate",),
            start_time_ns=int(timestamps_ns[0]),
            end_time_ns=int(timestamps_ns[-1] + 3_600_000_000_000),
        )
    funding_grid = materialize_feature_grid(request=funding_request, snapshot=snapshot, dataset=DatasetKind.FUNDING_EVENT)
    funding = np.asarray(funding_grid.fields.get("funding_rate", np.full((n_bars, n_syms), np.nan, dtype=np.float64)), dtype=np.float32)
    funding_available = funding_grid.available.get("funding_rate", np.zeros((n_bars, n_syms), dtype=np.bool_))

    feature_request = GridRequest(
        symbols=symbols,
        timeframe="1h",
        source_timeframe="1h",
        fields=("close",),
        start_time_ns=int(timestamps_ns[0]),
        end_time_ns=int(timestamps_ns[-1] + 3_600_000_000_000),
    )
    premium_grid = materialize_feature_grid(
        request=feature_request, snapshot=snapshot, dataset=DatasetKind.PREMIUM_5M,
    )
    mark_grid = materialize_feature_grid(
        request=feature_request, snapshot=snapshot, dataset=DatasetKind.MARK_1M,
    )
    index_grid = materialize_feature_grid(
        request=feature_request, snapshot=snapshot, dataset=DatasetKind.INDEX_1M,
    )
    metrics_request = GridRequest(
        symbols=symbols,
        timeframe="1h",
        source_timeframe="1h",
        fields=("sum_open_interest_value",),
        start_time_ns=int(timestamps_ns[0]),
        end_time_ns=int(timestamps_ns[-1] + 3_600_000_000_000),
    )
    metrics_grid = materialize_feature_grid(
        request=metrics_request, snapshot=snapshot, dataset=DatasetKind.METRICS_5M,
    )

    causal_grids: dict[str, NativeFeatureGrid] = {}
    for metric_field in ("top_trader_long_short_ratio", "long_short_ratio"):
        cg = materialize_causal_metrics_grid(
            symbols=symbols,
            start_time_ns=int(timestamps_ns[0]),
            end_time_ns=int(timestamps_ns[-1] + 3_600_000_000_000),
            lake_root=config.data_lake.root,
            field=metric_field,
        )
        causal_grids[metric_field] = cg
        arrays[metric_field] = np.asarray(
            cg.fields.get(metric_field, np.full((n_bars, n_syms), np.nan, dtype=np.float64)),
            dtype=np.float64,
        )

    available_core = np.logical_and.reduce(
        [np.asarray(grid.available[name], dtype=np.bool_) for name in core_names]
    )

    state_cube = universe.state_cube
    eligible = state_cube.eligible
    entry_block = state_cube.entry_block
    exit_required = state_cube.exit_required
    capacity = state_cube.capacity_usdt
    costs = state_cube.cost_bps.astype(np.float32)

    fields: dict[str, NDArray[np.float32] | NDArray[np.float64]] = {
        **arrays,
        "quote_volume": arrays["quote_volume"].astype(np.float32),
        "funding": funding,
        "premium": np.asarray(premium_grid.fields["close"], dtype=np.float32),
        "mark": np.asarray(mark_grid.fields["close"], dtype=np.float32),
        "index": np.asarray(index_grid.fields["close"], dtype=np.float32),
        "taker_buy_quote": arrays["taker_buy_quote"].astype(np.float32),
        "open_interest": np.asarray(
            metrics_grid.fields["sum_open_interest_value"], dtype=np.float32,
        ),
    }

    available_all: dict[str, NDArray[np.bool_]] = {
        "core": available_core,
        "funding": funding_available,
        "premium": premium_grid.available["close"],
        "mark": mark_grid.available["close"],
        "index": index_grid.available["close"],
        "taker_buy_quote": grid.available["taker_buy_quote"],
        "open_interest": metrics_grid.available["sum_open_interest_value"],
        "top_trader_long_short_ratio": causal_grids["top_trader_long_short_ratio"].available.get("top_trader_long_short_ratio", np.zeros((n_bars, n_syms), dtype=np.bool_)),
        "long_short_ratio": causal_grids["long_short_ratio"].available.get("long_short_ratio", np.zeros((n_bars, n_syms), dtype=np.bool_)),
    }

    cube = MarketFeatureCube(
        timestamps_ns=timestamps_ns,
        symbols=symbols,
        fields_2d=fields,
        available_2d=available_all,
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
]
