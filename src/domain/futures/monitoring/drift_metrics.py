"""P3.D-style diagnostics: KS drift, HMM transition stability, binary calibration ECE."""

from __future__ import annotations

import numpy as np


def ks_statistic_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov D = sup_t |F_a(t) - F_b(t)| (no SciPy)."""
    x = np.sort(np.asarray(a, dtype=np.float64)[np.isfinite(a)])
    y = np.sort(np.asarray(b, dtype=np.float64)[np.isfinite(b)])
    if x.size < 2 or y.size < 2:
        return 0.0
    grid = np.unique(np.concatenate([x, y]))
    d = 0.0
    nx, ny = float(x.size), float(y.size)
    for v in grid:
        fa = float(np.searchsorted(x, v, side="right")) / nx
        fb = float(np.searchsorted(y, v, side="right")) / ny
        d = max(d, abs(fa - fb))
    return float(d)


def frobenius_norm_delta(a: np.ndarray, b: np.ndarray) -> float:
    """||A - B||_F for square matrices (e.g. consecutive HMM transition matrices)."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape or aa.ndim != 2:
        return float("nan")
    return float(np.linalg.norm(aa - bb, ord="fro"))


def expected_calibration_error_binary(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Mean absolute gap between empirical positive rate and mean predicted prob per bin.
    probs, labels same length in [0,1] and {0,1}.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(labels, dtype=np.float64)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if p.size < n_bins * 5:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        w = float(mask.mean())
        ece += w * abs(acc - conf)
    return float(ece)
