"""Shared signal diagnostics utilities for the optimization layer."""

from __future__ import annotations

import numpy as np

__all__ = ["nonzero_ratio", "preservation_ratio"]


def nonzero_ratio(arr: np.ndarray, *, eps: float = 1e-12) -> float:
    """Return finite non-zero ratio for a numeric array.

    Args:
        arr: Input numeric array of any shape.
        eps: Absolute threshold below which a value is treated as zero.

    Returns:
        Fraction of finite elements whose absolute value exceeds *eps*.
        Returns ``0.0`` when the array is empty or contains no finite values.

    """
    finite = np.asarray(arr, dtype=np.float64)
    if finite.size == 0:
        return 0.0
    mask = np.isfinite(finite)
    if not np.any(mask):
        return 0.0
    vals = finite[mask]
    return float(np.count_nonzero(np.abs(vals) > eps) / vals.size)


def preservation_ratio(
    before: np.ndarray,
    after: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Return non-zero survival ratio after gating.

    Args:
        before: Signal array before the gating step.
        after: Signal array after the gating step.
        eps: Absolute threshold below which a value is treated as zero.

    Returns:
        Ratio of ``nonzero_ratio(after)`` to ``nonzero_ratio(before)``,
        clipped to ``[0.0, 1.0]``.  Returns ``0.0`` when *before* is all
        zero or contains no finite values.

    Raises:
        ValueError: If *before* and *after* have different shapes.

    """
    if before.shape != after.shape:
        raise ValueError("before and after must have the same shape")
    denom = nonzero_ratio(before, eps=eps)
    if denom <= 0.0:
        return 0.0
    ratio = float(nonzero_ratio(after, eps=eps) / denom)
    # Bounded contract: preservation ratio must stay within [0, 1].
    return float(np.clip(ratio, 0.0, 1.0))
