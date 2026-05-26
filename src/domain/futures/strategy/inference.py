from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.domain.futures.strategy.contracts import (
    ALPHA_FORECAST_CONTRACT_V3,
    ALPHA_FORECAST_V3_ATTR_KEY,
    FoldSpec,
    LongMatrixDataset,
)

_FORECAST_V3_NUMERIC_KEYS: tuple[str, ...] = (
    "q10_long",
    "q50_long",
    "q90_long",
    "q10_short",
    "q50_short",
    "q90_short",
    "confidence_long",
    "confidence_short",
)


@dataclass(slots=True, frozen=True)
class FoldAlpha:
    """Fold-level OOS alpha inference result."""

    fold_id: int
    ev_grid: np.ndarray


def infer_fold_alpha(
    *,
    fold: FoldSpec,
    test: LongMatrixDataset,
    ev_test: np.ndarray,
    t_size: int,
    n_size: int,
) -> FoldAlpha:
    """Restore fold test predictions to [T, N] grid."""
    if ev_test.shape[0] != test.index_map.shape[0]:
        raise ValueError("ev_test length mismatch")
    grid = np.zeros((t_size, n_size), dtype=np.float32)
    for row, (t_idx, s_idx) in enumerate(test.index_map):
        grid[int(t_idx), int(s_idx)] = np.float32(ev_test[row])
    return FoldAlpha(fold_id=fold.fold_id, ev_grid=grid)


def assemble_alpha_panel(
    datetimes: np.ndarray,
    symbols: tuple[str, ...],
    ev_grid: np.ndarray,
    clip_abs: float,
    eligible_mask: np.ndarray | None = None,
    forecast_metadata_v3: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Assemble alpha panel from signed EV grid."""
    if eligible_mask is not None and eligible_mask.shape != ev_grid.shape:
        raise ValueError("eligible_mask shape mismatch")
    ev_grid = np.clip(ev_grid, -clip_abs, clip_abs)
    if eligible_mask is not None:
        ev_grid = np.where(eligible_mask, ev_grid, 0.0)
    alpha_long = np.maximum(ev_grid, 0.0)
    alpha_short = np.maximum(-ev_grid, 0.0)
    idx = pd.MultiIndex.from_product([datetimes, symbols], names=["datetime", "symbol"])
    panel = pd.DataFrame(
        {"alpha_long": alpha_long.reshape(-1), "alpha_short": alpha_short.reshape(-1)},
        index=idx,
    ).sort_index()
    if list(panel.index.names) != ["datetime", "symbol"]:
        raise RuntimeError("alpha_panel index names mismatch")
    if list(panel.columns) != ["alpha_long", "alpha_short"]:
        raise RuntimeError("alpha_panel columns mismatch")
    if not panel.index.is_monotonic_increasing:
        raise RuntimeError("alpha_panel must be sorted")
    vals = panel[["alpha_long", "alpha_short"]].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        raise RuntimeError("alpha_panel contains non-finite values")
    if forecast_metadata_v3 is not None:
        panel.attrs["forecast_contract_version"] = ALPHA_FORECAST_CONTRACT_V3
        panel.attrs[ALPHA_FORECAST_V3_ATTR_KEY] = dict(forecast_metadata_v3)
        validate_alpha_forecast_metadata(panel)
    return panel


def validate_alpha_forecast_metadata(panel: pd.DataFrame) -> None:
    """Validate optional v3 forecast metadata when present."""
    attrs = panel.attrs
    payload = attrs.get(ALPHA_FORECAST_V3_ATTR_KEY)
    if payload is None:
        return
    if not isinstance(payload, dict):
        raise RuntimeError("alpha_panel v3 metadata must be a dict")
    row_count = len(panel.index)
    for key in _FORECAST_V3_NUMERIC_KEYS:
        if key not in payload:
            continue
        values = np.asarray(payload[key], dtype=np.float64).reshape(-1)
        if values.shape[0] != row_count:
            raise RuntimeError(f"alpha_panel v3 metadata length mismatch: {key}")
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"alpha_panel v3 metadata contains non-finite values: {key}")
