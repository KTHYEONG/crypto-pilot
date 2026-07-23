from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.compound.config import DataPlaneConfig
from src.domain.futures.compound.contracts import MarketFeatureCube
from src.domain.futures.universe.contracts import UniverseStateCube

if TYPE_CHECKING:
    from src.domain.futures.strategy.common.alignment import AlignedMarketData

_logger = logging.getLogger(__name__)

_VALIDATION_FAILED_MSG: str = "market feature cube validation failed"


def build_market_feature_cube(
    *,
    aligned: AlignedMarketData,
    universe: UniverseStateCube,
    optional_fields: Mapping[str, NDArray[np.float32]],
    available_at_ns: Mapping[str, NDArray[np.int64]],
    config: DataPlaneConfig,
) -> MarketFeatureCube:
    timestamps_ns = aligned.datetimes.astype(np.int64)
    symbols = aligned.symbols
    fields_2d: dict[str, NDArray[np.float32] | NDArray[np.float64]] = {}
    fields_2d["open"] = aligned.open_2d
    fields_2d["high"] = aligned.high_2d
    fields_2d["low"] = aligned.low_2d
    fields_2d["close"] = aligned.close_2d
    fields_2d["quote_volume"] = aligned.volume_2d
    fields_2d["funding"] = aligned.funding_2d
    if aligned.basis_2d is not None:
        fields_2d["premium"] = aligned.basis_2d
    if aligned.taker_buy_2d is not None:
        fields_2d["taker_buy_quote"] = aligned.taker_buy_2d
    if aligned.trades_2d is not None:
        fields_2d["trades"] = aligned.trades_2d
    fields_2d.update(optional_fields)

    available_2d: dict[str, NDArray[np.bool_]] = {"core": aligned.active_mask.copy()}
    for field_name, avail_ns in available_at_ns.items():
        available_2d[field_name] = timestamps_ns[:, None] >= avail_ns

    n_bars = timestamps_ns.size
    n_syms = len(symbols)
    entry_block_2d = np.zeros((n_bars, n_syms), dtype=np.bool_)
    exit_required_2d = np.zeros((n_bars, n_syms), dtype=np.bool_)
    capacity_usdt_2d = np.full((n_bars, n_syms), np.nan, dtype=np.float64)
    exec_cost_2d = np.full((n_bars, n_syms), np.nan, dtype=np.float32)

    idx_map = {sym: i for i, sym in enumerate(symbols)}
    for i, sid in enumerate(universe.instrument_ids):
        if sid not in idx_map:
            continue
        col = idx_map[sid]
        entry_block_2d[:, col] = universe.entry_block[:, i]
        exit_required_2d[:, col] = universe.exit_required[:, i]
        capacity_usdt_2d[:, col] = universe.capacity_usdt[:, i]

    if aligned.execution_cost_bps_2d is not None:
        exec_cost_2d[:] = aligned.execution_cost_bps_2d

    fallback_cost_bps = np.float32(12.0)
    exec_cost_2d = np.where(np.isfinite(exec_cost_2d), exec_cost_2d, fallback_cost_bps)

    return MarketFeatureCube(
        timestamps_ns=timestamps_ns,
        symbols=symbols,
        fields_2d=fields_2d,
        available_2d=available_2d,
        eligible_2d=universe.eligible,
        entry_block_2d=entry_block_2d,
        exit_required_2d=exit_required_2d,
        capacity_usdt_2d=capacity_usdt_2d,
        execution_cost_bps_2d=exec_cost_2d,
        data_manifest_hash="",
    )


def build_compound_market_feature_cube(
    *,
    data_maps: dict[str, dict[str, object]],
    symbols: tuple[str, ...],
    state_cube: UniverseStateCube,
    timeframe: Literal["1h"],
    data_manifest_hash: str,
    config: DataPlaneConfig,
) -> MarketFeatureCube:
    """Build the PIT market cube directly from raw symbol frames.

    Every observation is selected with a backward as-of join against the
    authoritative state calendar.  A symbol missing from either the raw data
    or the universe is never entry eligible, while reductions remain possible
    through the simulator's previous-position handling.
    """
    if timeframe != "1h":
        raise ValueError(f"compound base timeframe must be 1h, got {timeframe!r}")
    if not symbols or len(symbols) > config.max_symbols:
        raise ValueError("symbols must be non-empty and within max_symbols")
    if state_cube.calendar.empty or state_cube.calendar.tz is None:
        raise ValueError("state calendar must be non-empty and timezone-aware")

    state_utc = state_cube.calendar.tz_convert(UTC)
    state_timestamps = np.asarray(
        state_utc.tz_localize(None).astype("datetime64[ns]").view("int64"),
        dtype=np.int64,
    )
    timestamps = state_timestamps
    n_bars = len(timestamps)
    n_syms = len(symbols)
    core_names = ("open", "high", "low", "close", "quote_volume")
    arrays: dict[str, NDArray[np.float64]] = {
        name: np.full((n_bars, n_syms), np.nan, dtype=np.float64)
        for name in core_names
    }
    funding = np.zeros((n_bars, n_syms), dtype=np.float32)
    available_core = np.zeros((n_bars, n_syms), dtype=np.bool_)
    entry_block = np.ones((n_bars, n_syms), dtype=np.bool_)
    exit_required = np.zeros((n_bars, n_syms), dtype=np.bool_)
    eligible = np.zeros((n_bars, n_syms), dtype=np.bool_)
    capacity = np.zeros((n_bars, n_syms), dtype=np.float64)
    costs = np.full((n_bars, n_syms), 12.0, dtype=np.float32)
    optional: dict[str, NDArray[np.float32]] = {}

    state_index = {sid: i for i, sid in enumerate(state_cube.instrument_ids)}
    for col, symbol in enumerate(symbols):
        frame_obj = data_maps.get(symbol, {}).get(timeframe)
        if not isinstance(frame_obj, pd.DataFrame) or frame_obj.empty:
            continue
        frame = frame_obj.copy()
        if "datetime" not in frame.columns:
            raise ValueError(f"missing datetime column: {symbol}")
        if "quote_volume" not in frame.columns and "volume" in frame.columns:
            frame["quote_volume"] = frame["volume"]
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["datetime"]).sort_values("datetime")
        if frame.empty or any(name not in frame.columns for name in core_names):
            raise ValueError(f"malformed core data: {symbol}")
        datetime_utc = pd.to_datetime(frame["datetime"], utc=True)
        source_ns = np.asarray(
            datetime_utc.dt.tz_localize(None).astype("datetime64[ns]").astype("int64"),
            dtype=np.int64,
        )
        positions = np.searchsorted(source_ns, timestamps, side="right") - 1
        valid_pos = positions >= 0
        for name in core_names:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
            arrays[name][valid_pos, col] = values[positions[valid_pos]]
        if "funding_rate_sum" in frame.columns:
            funding_values = frame["funding_rate_sum"]
        else:
            funding_values = frame.get("funding_rate", pd.Series(0.0, index=frame.index))
        funding[valid_pos, col] = pd.to_numeric(funding_values, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)[positions[valid_pos]]
        for source, target in (("basis", "premium"), ("basis_rate", "premium"), ("taker_buy_quote", "taker_buy_quote"), ("trades", "trades")):
            if source in frame.columns and target not in optional:
                optional[target] = np.full((n_bars, n_syms), np.nan, dtype=np.float32)
            if source in frame.columns:
                values = pd.to_numeric(frame[source], errors="coerce").to_numpy(dtype=np.float32)
                optional[target][valid_pos, col] = values[positions[valid_pos]]
        state_col = state_index.get(symbol)
        if state_col is not None:
            state_positions = np.searchsorted(
                state_timestamps,
                timestamps,
                side="right",
            ) - 1
            state_valid = state_positions >= 0
            eligible[state_valid, col] = state_cube.eligible[state_positions[state_valid], state_col]
            entry_block[state_valid, col] = state_cube.entry_block[state_positions[state_valid], state_col]
            exit_required[state_valid, col] = state_cube.exit_required[state_positions[state_valid], state_col]
            capacity[state_valid, col] = state_cube.capacity_usdt[state_positions[state_valid], state_col]
            costs[state_valid, col] = state_cube.cost_bps[state_positions[state_valid], state_col].astype(np.float32)
        available_core[:, col] = valid_pos & np.all(
            np.isfinite(np.column_stack([arrays[name][:, col] for name in core_names])), axis=1
        )
        eligible[:, col] &= available_core[:, col]
        entry_block[:, col] |= ~available_core[:, col]

    fields: dict[str, NDArray[np.float32] | NDArray[np.float64]] = {
        **arrays,
        "quote_volume": arrays["quote_volume"].astype(np.float32),
        "funding": funding,
    }
    fields.update(optional)
    cube = MarketFeatureCube(
        timestamps_ns=timestamps.astype(np.int64),
        symbols=symbols,
        fields_2d=fields,
        available_2d={"core": available_core},
        eligible_2d=eligible,
        entry_block_2d=entry_block,
        exit_required_2d=exit_required,
        capacity_usdt_2d=capacity,
        execution_cost_bps_2d=costs,
        data_manifest_hash=data_manifest_hash,
    )
    validate_market_feature_cube(cube)
    return cube


def validate_market_feature_cube(cube: MarketFeatureCube) -> None:
    n_bars = cube.timestamps_ns.size
    n_syms = len(cube.symbols)
    assert n_bars > 0
    assert n_syms > 0
    assert n_syms <= 120
    assert cube.timestamps_ns.ndim == 1
    diffs = np.diff(cube.timestamps_ns)
    assert np.all(diffs > 0)
    for fname, farr in cube.fields_2d.items():
        assert farr.shape == (n_bars, n_syms), f"{fname} shape {farr.shape} != ({n_bars}, {n_syms})"
        assert farr.dtype in (np.float32, np.float64)
    for aname, aarr in cube.available_2d.items():
        assert aarr.shape == (n_bars, n_syms), f"available_2d[{aname}] shape mismatch"
        assert aarr.dtype == np.bool_
    assert cube.eligible_2d.shape == (n_bars, n_syms)
    assert cube.entry_block_2d.shape == (n_bars, n_syms)
    assert cube.exit_required_2d.shape == (n_bars, n_syms)
    assert cube.capacity_usdt_2d.shape == (n_bars, n_syms)
    assert cube.capacity_usdt_2d.dtype == np.float64
    assert cube.execution_cost_bps_2d.shape == (n_bars, n_syms)
    assert cube.execution_cost_bps_2d.dtype == np.float32
    assert isinstance(cube.data_manifest_hash, str)


def materialize_hourly_execution_features(
    *, book_depth: pd.DataFrame, mark_price: pd.DataFrame, fallback_cost_bps: float
) -> pd.DataFrame:
    if book_depth.empty:
        result = pd.DataFrame(index=mark_price.index) if not mark_price.empty else pd.DataFrame()
        result["depth_spread_bps"] = fallback_cost_bps
        result["depth_notional_usdt"] = 0.0
        result["execution_cost_bps"] = fallback_cost_bps
        return result

    merged = book_depth.merge(mark_price, left_index=True, right_index=True, how="left", suffixes=("", "_mark"))
    if "mark" not in merged.columns and "close" in merged.columns:
        merged["mark"] = merged["close"]

    mid = (merged["bid"] + merged["ask"]) / 2.0
    spread_bps = ((merged["ask"] - merged["bid"]) / mid.replace(0, np.nan)) * 10_000
    depth_notional = merged["bid_depth"] + merged["ask_depth"]
    merged["depth_spread_bps"] = spread_bps.fillna(fallback_cost_bps).clip(0, 100)
    merged["depth_notional_usdt"] = depth_notional.fillna(0.0)
    merged["execution_cost_bps"] = merged["depth_spread_bps"]
    return merged[["depth_spread_bps", "depth_notional_usdt", "execution_cost_bps"]]
