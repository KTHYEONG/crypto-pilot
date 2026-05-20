from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _as_finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return np.asarray([0.0], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([0.0], dtype=np.float64)
    return arr


def lcb(values: Iterable[float], k: float = 1.0) -> float:
    arr = _as_finite_array(values)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))
    return mu - float(k) * sigma


def ucb(values: Iterable[float], k: float = 1.0) -> float:
    arr = _as_finite_array(values)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))
    return mu + float(k) * sigma


def summarize(values: Iterable[float]) -> tuple[list[float], float, float]:
    arr = _as_finite_array(values)
    return arr.tolist(), float(np.mean(arr)), float(np.std(arr))
