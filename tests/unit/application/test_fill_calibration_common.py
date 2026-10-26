"""Invariant scenarios for shared fill calibration statistics."""

from __future__ import annotations

import numpy as np
import pytest

import src.application.fill_calibration_common as fcc


def test_weighted_stats_ignore_non_finite_and_zero_weight_rows() -> None:
    assert np.isnan(fcc.weighted_mean(np.array([1.0]), np.array([0.0])))
    assert np.isnan(fcc.weighted_mean(np.array([np.nan]), np.array([1.0])))
    assert np.isnan(fcc.percentile(np.array([np.nan]), 50.0))
    assert fcc.percentile(np.array([1.0, 2.0]), 50.0) == pytest.approx(1.5)


def test_bootstrap_diff_ci_degenerate_inputs_are_nan() -> None:
    pt, lo, hi = fcc.bootstrap_diff_ci(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), samples=100, seed=7)
    assert np.isnan(pt)
    assert np.isnan(lo)
    assert np.isnan(hi)
    pt, lo, hi = fcc.bootstrap_diff_ci(
        np.array([1]), np.array([1, 1]), np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), samples=100, seed=7
    )
    assert np.isnan(lo)
    assert np.isnan(hi)
