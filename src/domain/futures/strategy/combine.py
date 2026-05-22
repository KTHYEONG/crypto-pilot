from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.normalize import winsorized_cs_zscore


def blend_sleeves(
    z_by_sleeve: dict[str, np.ndarray],
    ic_weights: dict[str, float],
    *,
    clip_z: float = 3.0,
    min_symbols: int = 5,
) -> np.ndarray:
    """Blends multiple standardized sleeve z-scores using lagged IC weights.

    Args:
        z_by_sleeve: Dictionary mapping sleeve name to its [T, N] z-score array.
        ic_weights: Dictionary mapping sleeve name to its scalar lagged IC weight.
        clip_z: Z-score threshold for clipping.
        min_symbols: Minimum required non-NaN symbols.

    Returns:
        [T, N] blended and re-standardized score array.
    """
    if not z_by_sleeve:
        raise ValueError("z_by_sleeve cannot be empty")

    # Verify shapes are consistent
    names = list(z_by_sleeve.keys())
    t_len, n_syms = z_by_sleeve[names[0]].shape

    for name in names:
        if z_by_sleeve[name].shape != (t_len, n_syms):
            raise ValueError(f"Shape of sleeve '{name}' does not match other sleeves")

    # Normalize weights: w_k = max(ic_weights[k], 0)
    w = np.zeros(len(names), dtype=np.float64)
    for idx, name in enumerate(names):
        w[idx] = max(ic_weights.get(name, 0.0), 0.0)

    w_sum = np.sum(w)
    if w_sum <= 1e-12:
        # All weights are zero or negative -> fallback to equal weight
        w = np.full(len(names), 1.0 / len(names), dtype=np.float64)
    else:
        w /= w_sum

    # Compute blended score: s[t, i] = sum_k w_k * z_k[t, i]
    blended = np.zeros((t_len, n_syms), dtype=np.float64)
    for idx, name in enumerate(names):
        blended += w[idx] * z_by_sleeve[name]

    # Re-standardize blended score
    return winsorized_cs_zscore(blended, clip_z=clip_z, min_symbols=min_symbols)
