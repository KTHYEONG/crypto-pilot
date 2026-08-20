"""Tests for the MHS application scaling module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.application.research.mhs.contracts import MhsDiagnosticRequest
from src.application.research.mhs import scaling
from src.mhs.params import MHS_PNL_TARGET_ANNUAL_VOL


def test_growth_budget_target_vol_fallback_on_short_series() -> None:
    """_growth_budget_target_vol returns fallback when train slice is too short."""
    idx = pd.date_range("2022-06-01", periods=10, freq="D", tz="UTC")
    r = pd.Series(0.001, index=idx)
    assert scaling._growth_budget_target_vol(r) == MHS_PNL_TARGET_ANNUAL_VOL


def test_replay_exposure_scale_growth_budget_mode() -> None:
    """_replay_exposure_scale with growth_budget returns finite bounded series."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2021-01-01", periods=500, freq="D", tz="UTC")
    r = pd.Series(rng.normal(0.0001, 0.01, 500), index=idx)
    request = MhsDiagnosticRequest(
        pnl_vol_target_mode="growth_budget",
        committee_capital=True,
    )
    result = scaling._replay_exposure_scale(r, request)
    assert result.index.equals(r.index)
    assert np.isfinite(result.to_numpy()).all()
    assert (result >= 0.2).all()
    assert (result <= 1.0).all()
