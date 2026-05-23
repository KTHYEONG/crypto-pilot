from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.domain.futures.strategy.contracts import FoldSpec, LongMatrixDataset


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
    return panel
