from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.config import MomentumConfig


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks for a 1D array (ties averaged)."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    n = sorted_values.size
    ranks_sorted = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks_sorted[i:j] = avg_rank
        i = j

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def compute_xs_momentum_alpha(
    close_2d: np.ndarray,
    cfg: MomentumConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cross-sectional momentum alpha without look-ahead."""
    if close_2d.ndim != 2:
        raise ValueError("close_2d must be 2D")
    t_len, n_syms = close_2d.shape
    lb = cfg.lookback_bars

    alpha_long = np.zeros((t_len, n_syms), dtype=np.float64)
    alpha_short = np.zeros((t_len, n_syms), dtype=np.float64)
    if t_len <= lb:
        return alpha_long, alpha_short

    prev = close_2d[:-lb]
    curr = close_2d[lb:]
    with np.errstate(divide="ignore", invalid="ignore"):
        mom = np.log(curr / np.maximum(prev, 1e-12))
    mom = np.where(np.isfinite(mom), mom, np.nan)

    for t_idx in range(mom.shape[0]):
        row = mom[t_idx]
        valid_mask = np.isfinite(row)
        n_valid = int(valid_mask.sum())
        if n_valid < cfg.min_symbols_for_xs:
            continue

        valid_vals = row[valid_mask]
        ranks = _rankdata_average(valid_vals) / float(n_valid)
        long_part = np.maximum(ranks - (1.0 - cfg.top_ratio), 0.0) / cfg.top_ratio
        short_part = np.maximum(cfg.bottom_ratio - ranks, 0.0) / cfg.bottom_ratio

        out_row = lb + t_idx
        alpha_long[out_row, valid_mask] = long_part * cfg.edge_scale_per_bar
        alpha_short[out_row, valid_mask] = short_part * cfg.edge_scale_per_bar

    return alpha_long, alpha_short
