from __future__ import annotations

import numpy as np
import pytest

from src.domain.futures.strategy.causal_statistics import (
    causal_expanding_quantile,
    causal_expanding_robust_location_scale,
)


def _reference_expanding_quantile(values: np.ndarray, q: float, min_periods: int = 1) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for idx in range(values.shape[0]):
        finite = values[: idx + 1]
        finite = finite[np.isfinite(finite)]
        if finite.size >= min_periods:
            out[idx] = float(np.percentile(finite, q * 100.0))
    return out


def test_causal_expanding_quantile_matches_reference() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, 50).astype(np.float64)
    for q in [0.25, 0.5, 0.75, 0.9]:
        vec = causal_expanding_quantile(values, q)
        ref = _reference_expanding_quantile(values, q)
        np.testing.assert_allclose(vec, ref, atol=1e-12, err_msg=f"q={q} mismatch")


def test_causal_expanding_quantile_preserves_prefix_causality() -> None:
    rng = np.random.default_rng(42)
    prefix = rng.normal(0, 1, 20).astype(np.float64)
    suffix = np.concatenate([prefix, rng.normal(10, 1, 30).astype(np.float64)])
    q50_prefix = causal_expanding_quantile(prefix, 0.5)
    q50_suffix_prefix = causal_expanding_quantile(suffix, 0.5)[:20]
    np.testing.assert_allclose(q50_prefix, q50_suffix_prefix, atol=1e-12, err_msg="prefix causality violated")


def test_causal_expanding_quantile_handles_non_finite() -> None:
    values = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)
    result = causal_expanding_quantile(values, 0.5)
    ref = _reference_expanding_quantile(values, 0.5)
    np.testing.assert_allclose(result, ref, atol=1e-12)


def test_causal_expanding_quantile_respects_min_periods() -> None:
    sparse = np.full(10, np.nan, dtype=np.float64)
    sparse[[2, 4, 6]] = [1.0, 2.0, 3.0]
    result = causal_expanding_quantile(sparse, 0.5, min_periods=3)
    assert np.isfinite(result[6]), "min_periods=3 met at idx 6 (3rd finite)"
    assert np.isnan(result[:3]).all(), "pre-finite slots must be NaN"


def test_causal_expanding_quantile_empty_input() -> None:
    empty = np.empty(0, dtype=np.float64)
    result = causal_expanding_quantile(empty, 0.5)
    assert result.shape == (0,)


def test_causal_expanding_quantile_validates_q() -> None:
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="q must be in"):
        causal_expanding_quantile(values, -0.1)
    with pytest.raises(ValueError, match="q must be in"):
        causal_expanding_quantile(values, 1.5)


def test_causal_expanding_quantile_validates_ndim() -> None:
    with pytest.raises(ValueError, match="values must be 1-D"):
        causal_expanding_quantile(np.ones((5, 5), dtype=np.float64), 0.5)


def test_causal_expanding_quantile_validates_min_periods() -> None:
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="min_periods must be >= 1"):
        causal_expanding_quantile(values, 0.5, min_periods=0)


def test_causal_expanding_robust_location_scale_validates_min_periods() -> None:
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="min_periods must be >= 1"):
        causal_expanding_robust_location_scale(values, min_periods=0)


def test_causal_expanding_robust_location_scale_validates_ndim() -> None:
    with pytest.raises(ValueError, match="values must be 1-D"):
        causal_expanding_robust_location_scale(np.ones((5, 5), dtype=np.float64))


def test_causal_expanding_robust_location_scale_matches_reference() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(0, 1, 50).astype(np.float64)
    loc, scale = causal_expanding_robust_location_scale(values)
    loc_ref = _reference_expanding_quantile(values, 0.5)
    q75 = _reference_expanding_quantile(values, 0.75)
    q25 = _reference_expanding_quantile(values, 0.25)
    iqr = q75 - q25
    scale_ref = np.where(np.isfinite(iqr) & (iqr / 1.3489795 >= 1e-12), iqr / 1.3489795, 1e-12)
    np.testing.assert_allclose(loc, loc_ref, atol=1e-10)
    np.testing.assert_allclose(scale, scale_ref, atol=1e-10)


def test_causal_expanding_robust_location_scale_empty() -> None:
    loc, scale = causal_expanding_robust_location_scale(np.empty(0, dtype=np.float64))
    assert loc.shape == (0,)
    assert scale.shape == (0,)
