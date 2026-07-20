from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def causal_expanding_quantile(
    values: NDArray[np.float64],
    q: float,
    *,
    min_periods: int = 1,
    fill_value: float = np.nan,
) -> NDArray[np.float64]:
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    if min_periods < 1:
        raise ValueError(f"min_periods must be >= 1, got {min_periods}")
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got ndim={values.ndim}")
    if values.size == 0:
        return np.empty(0, dtype=np.float64)

    s = pd.Series(values, dtype=np.float64)
    expanding = s.expanding(min_periods=1)
    result: NDArray[np.float64] = cast(
        NDArray[np.float64],
        expanding.quantile(q, interpolation="linear").to_numpy(dtype=np.float64),
    )

    pre_min = np.arange(values.shape[0]) < min_periods
    if pre_min.any():
        valid_count = np.cumsum(np.isfinite(values).astype(np.intp))
        insufficient = valid_count < min_periods
        result = np.where(insufficient, fill_value, result)

    return result


def causal_expanding_robust_location_scale(
    values: NDArray[np.float64],
    *,
    min_periods: int = 1,
    eps: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if min_periods < 1:
        raise ValueError(f"min_periods must be >= 1, got {min_periods}")
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got ndim={values.ndim}")
    if values.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    s = pd.Series(values, dtype=np.float64)
    expanding = s.expanding(min_periods=1)
    q50: NDArray[np.float64] = cast(NDArray[np.float64], expanding.quantile(0.5, interpolation="linear").to_numpy(dtype=np.float64))
    q75: NDArray[np.float64] = cast(NDArray[np.float64], expanding.quantile(0.75, interpolation="linear").to_numpy(dtype=np.float64))
    q25: NDArray[np.float64] = cast(NDArray[np.float64], expanding.quantile(0.25, interpolation="linear").to_numpy(dtype=np.float64))

    iqr = q75 - q25
    scale = iqr / 1.3489795

    finite_scale = np.isfinite(scale)
    scale = np.where(finite_scale & (scale >= eps), scale, eps)

    valid_count = np.cumsum(np.isfinite(values).astype(np.intp))
    insufficient = valid_count < min_periods
    location = np.where(insufficient, np.nan, q50)
    scale = np.where(insufficient, np.nan, scale)

    return location, scale
