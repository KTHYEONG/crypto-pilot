"""Decision-label process feature grid with symbol-local work buffers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.features import FeatureSpec, _finite
from src.mhs.horizons import horizon_log_return
from src.mhs.params import PROCESS_FEATURE_CANDIDATES

PROCESS_FEATURE_COLUMN_BLOCK_SIZE: int = 32

_IDIO_HORIZON_BARS: int = 336
_IDIO_BETA_BARS: int = 336
_XS_FULL_UNIVERSE_FEATURES: frozenset[str] = frozenset({"xs_mom_336h", "xs_mom_720h"})


def build_process_feature_grid(
    spec: FeatureSpec,
    panels: Mapping[str, pd.DataFrame],
    decision_grid: pd.DatetimeIndex,
    *,
    column_block_size: int = PROCESS_FEATURE_COLUMN_BLOCK_SIZE,
) -> pd.DataFrame:
    """Build unchanged process features at decision labels with localized buffers.

    Full-history rolling state and the canonical cross-sectional market are
    preserved so memory localization cannot change member selection.

    Args:
        spec: Existing registered process feature and its original semantics.
        panels: Aligned float64 source planes in canonical symbol order.
        decision_grid: UTC labels consumed by the process decision clock.
        column_block_size: Positive maximum width of symbol-local work buffers.
    Returns:
        Original feature values reindexed to the requested decision labels,
        retaining source column order, float64 values and missingness.
    Raises:
        ValueError: Block size, source alignment or feature support is invalid.
        DataIntegrityError: Existing source or feature integrity checks fail.
    """
    if isinstance(column_block_size, bool) or not isinstance(column_block_size, int):
        raise ValueError(f"column_block_size must be a positive integer, got {column_block_size!r}")
    if column_block_size <= 0:
        raise ValueError(f"column_block_size must be positive, got {column_block_size}")
    if spec.name not in PROCESS_FEATURE_CANDIDATES:
        raise ValueError(f"unsupported process feature '{spec.name}'")
    missing = [c for c in spec.required_columns if c not in panels]
    if missing:
        raise ValueError(f"spec '{spec.name}' required_columns absent from panels: {missing}")
    if not isinstance(decision_grid, pd.DatetimeIndex):
        raise ValueError("decision_grid must be a DatetimeIndex")
    planes = {name: panels[name] for name in spec.required_columns}
    first = planes[spec.required_columns[0]]
    index = first.index
    columns = list(first.columns)
    for name, frame in panels.items():
        if not frame.index.equals(index):
            raise ValueError(f"source plane '{name}' must share the canonical source labels exactly")
        if list(frame.columns) != columns:
            raise ValueError(f"source plane '{name}' must share the canonical symbol order exactly")
    if index.has_duplicates or index.hasnans:
        raise DataIntegrityError("source labels must be unique without NaT")
    if decision_grid.hasnans:
        raise DataIntegrityError("decision labels must not contain NaT")
    if spec.name in _XS_FULL_UNIVERSE_FEATURES:
        grid = spec.builder(panels).reindex(decision_grid)
        return grid.astype("float64").reindex(columns=columns)
    if spec.name == "xs_idio_mom_336h":
        return _idio_grid(planes, columns, decision_grid, column_block_size)
    parts: list[pd.DataFrame] = []
    try:
        for left in range(0, len(columns), column_block_size):
            sliver = slice(left, left + column_block_size)
            local = {name: frame.iloc[:, sliver] for name, frame in panels.items()}
            signal = spec.builder(local)
            parts.append(signal.reindex(decision_grid))
            del signal, local
        grid = pd.concat(parts, axis=1)
    finally:
        del parts
    grid = grid.reindex(columns=columns)
    return grid.astype("float64")


def _idio_grid(
    planes: Mapping[str, pd.DataFrame],
    columns: list[str],
    decision_grid: pd.DatetimeIndex,
    column_block_size: int,
) -> pd.DataFrame:
    log_close = np.log(planes["close"])
    raw = horizon_log_return(log_close, _IDIO_HORIZON_BARS)
    del log_close
    market = raw.mean(axis=1)
    mean_m = market.rolling(_IDIO_BETA_BARS, min_periods=_IDIO_BETA_BARS).mean()
    var_m = market.pow(2).rolling(_IDIO_BETA_BARS, min_periods=_IDIO_BETA_BARS).mean() - mean_m.pow(2)
    parts: list[pd.DataFrame] = []
    try:
        for left in range(0, len(columns), column_block_size):
            sliver = slice(left, left + column_block_size)
            close_block = planes["close"].iloc[:, sliver]
            raw_block = horizon_log_return(np.log(close_block), _IDIO_HORIZON_BARS)
            del close_block
            mean_r = raw_block.rolling(_IDIO_BETA_BARS, min_periods=_IDIO_BETA_BARS).mean()
            mean_rm = (
                raw_block.mul(market, axis=0)
                .rolling(_IDIO_BETA_BARS, min_periods=_IDIO_BETA_BARS)
                .mean()
            )
            cov_rm = mean_rm - mean_r.mul(mean_m, axis=0)
            del mean_r, mean_rm
            beta = cov_rm.div(var_m.replace(0, np.nan), axis=0)
            del cov_rm
            residual = raw_block - beta.mul(market, axis=0)
            del beta, raw_block
            residual_vol = residual.rolling(_IDIO_HORIZON_BARS, min_periods=_IDIO_HORIZON_BARS).std(ddof=1) * np.sqrt(
                _IDIO_HORIZON_BARS
            )
            signal = _finite(residual.div(residual_vol.replace(0, np.nan)))
            del residual, residual_vol
            parts.append(signal.reindex(decision_grid))
            del signal
        grid = pd.concat(parts, axis=1)
    finally:
        del parts, raw, market, mean_m, var_m
    grid = grid.reindex(columns=columns)
    return grid.astype("float64")
