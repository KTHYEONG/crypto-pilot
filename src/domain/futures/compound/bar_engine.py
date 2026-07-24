from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.contracts import (
    CausalityError,
    InsufficientCoverageError,
    MarketFeatureCube,
    MultiTimeframeBars,
    TimeframeBarCube,
)

_logger = logging.getLogger(__name__)

_TF_HOURS: Mapping[str, int] = {
    "1h": 1,
    "4h": 4,
    "1d": 24,
}

_OHLCV_FIELDS = ("open", "high", "low", "close", "quote_volume")

_AUX_FIELDS = ("funding", "premium", "mark", "index", "taker_buy_quote", "quote_volume")


def _validate_monotonic(timestamps_ns: NDArray[np.int64]) -> None:
    if timestamps_ns.size > 1 and not np.all(np.diff(timestamps_ns) > 0):
        raise CausalityError("timestamps_ns is not monotonically increasing")


def _get_field(market: MarketFeatureCube, name: str) -> NDArray[np.float32]:
    raw = market.fields_2d.get(name)
    if raw is None:
        return np.full((market.timestamps_ns.size, len(market.symbols)), np.nan, dtype=np.float32)
    return raw.astype(np.float32, copy=False)


def aggregate_timeframe_bars(
    market: MarketFeatureCube, timeframe: str,
) -> TimeframeBarCube:
    _validate_monotonic(market.timestamps_ns)

    if timeframe not in _TF_HOURS:
        raise ValueError(f"unknown timeframe: {timeframe}")

    n_total = market.timestamps_ns.size
    n_syms = len(market.symbols)
    group_size = _TF_HOURS[timeframe]
    n_full_groups = n_total // group_size
    usable = n_full_groups * group_size

    open_arr = _get_field(market, "open")
    high_arr = _get_field(market, "high")
    low_arr = _get_field(market, "low")
    close_arr = _get_field(market, "close")
    volume_arr = _get_field(market, "quote_volume")

    if timeframe == "1h":
        complete = np.ones((n_total, n_syms), dtype=np.bool_)
        for f in _OHLCV_FIELDS:
            complete &= np.isfinite(_get_field(market, f))
        return TimeframeBarCube(
            timeframe="1h",
            timestamps_ns=market.timestamps_ns.copy(),
            symbols=market.symbols,
            open_2d=open_arr.astype(np.float32),
            high_2d=high_arr.astype(np.float32),
            low_2d=low_arr.astype(np.float32),
            close_2d=close_arr.astype(np.float32),
            quote_volume_2d=volume_arr.astype(np.float32),
            complete_2d=complete,
        )

    o_r = open_arr[:usable].reshape(n_full_groups, group_size, n_syms)
    h_r = high_arr[:usable].reshape(n_full_groups, group_size, n_syms)
    l_r = low_arr[:usable].reshape(n_full_groups, group_size, n_syms)
    c_r = close_arr[:usable].reshape(n_full_groups, group_size, n_syms)
    v_r = volume_arr[:usable].reshape(n_full_groups, group_size, n_syms)

    all_finite = np.isfinite(o_r) & np.isfinite(h_r) & np.isfinite(l_r) & np.isfinite(c_r) & np.isfinite(v_r)
    complete_2d = np.all(all_finite, axis=1)

    return TimeframeBarCube(
        timeframe=timeframe,
        timestamps_ns=market.timestamps_ns[group_size - 1:usable:group_size].copy(),
        symbols=market.symbols,
        open_2d=o_r[:, 0, :].astype(np.float32, copy=False),
        high_2d=np.max(h_r, axis=1).astype(np.float32, copy=False),
        low_2d=np.min(l_r, axis=1).astype(np.float32, copy=False),
        close_2d=c_r[:, -1, :].astype(np.float32, copy=False),
        quote_volume_2d=np.sum(v_r, axis=1).astype(np.float32, copy=False),
        complete_2d=complete_2d,
    )


def build_multi_timeframe_bars(
    market: MarketFeatureCube,
    timeframes: tuple[str, ...] = ("1h", "4h", "1d"),
) -> MultiTimeframeBars:
    cubes: dict[str, TimeframeBarCube] = {}
    for tf in timeframes:
        cubes[tf] = aggregate_timeframe_bars(market, tf)

    decision_ts = cubes["4h"].timestamps_ns

    if decision_ts.size < 100:
        raise InsufficientCoverageError(
            f"4h decision grid has {decision_ts.size} bars, minimum 100 required",
        )

    aux_fields: dict[str, NDArray[np.float32]] = {}
    for field in _AUX_FIELDS:
        raw = market.fields_2d.get(field)
        if raw is not None:
            aux_fields[field] = raw.astype(np.float32, copy=False)

    _logger.info("built MultiTimeframeBars with %d decision bars, cubes=%s", decision_ts.size, list(cubes.keys()))

    return MultiTimeframeBars(
        decision_timestamps_ns=decision_ts,
        cubes=cubes,
        aux_1h_fields=aux_fields,
    )
