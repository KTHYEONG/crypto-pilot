"""Tests for the MHS application scaling module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application.research.mhs.contracts import MhsDiagnosticRequest
from src.application.research.mhs import scaling
from src.mhs.params import (
    COMMITTEE_TARGET_GROSS,
    GROWTH_RISK_ENVELOPES,
    PNL_TARGET_ANNUAL_VOL,
)


def test_growth_budget_target_vol_fallback_on_short_series() -> None:
    """_growth_budget_target_vol returns fallback when train slice is too short."""
    idx = pd.date_range("2022-06-01", periods=10, freq="D", tz="UTC")
    r = pd.Series(0.001, index=idx)
    assert scaling._growth_budget_target_vol(r) == PNL_TARGET_ANNUAL_VOL


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


_ENVELOPE_CAP_TEST_RETURNS = pd.Series(
    # mean/sd tuned to a ~Sharpe-2 daily series (matching the measured
    # production blend's realized Sharpe) over 750 rows -- required for the
    # "growth" envelope's bootstrap ruin-frontier check to find a plateau-
    # satisfying feasible grid point; a lower-drift series can be genuinely
    # infeasible for envelope="growth" on this solver, which is correct
    # fail-closed behavior, not a test bug.
    np.random.default_rng(20260821).normal(0.0021, 0.02, 750),
    index=pd.date_range("2021-01-01", periods=750, freq="D", tz="UTC"),
)


# SCENARIO_ENVELOPE_EXPOSURE_CAP_LIFTS_UNIT_GROSS
class TestEnvelopeExposureCap:
    def test_conservative_cap_with_target_gross(self) -> None:
        cap = scaling._envelope_exposure_cap(
            GROWTH_RISK_ENVELOPES["conservative"], COMMITTEE_TARGET_GROSS,
            _ENVELOPE_CAP_TEST_RETURNS,
        )
        assert cap == pytest.approx(1.0 / COMMITTEE_TARGET_GROSS)

    def test_growth_cap_with_target_gross(self) -> None:
        cap = scaling._envelope_exposure_cap(
            GROWTH_RISK_ENVELOPES["growth"], COMMITTEE_TARGET_GROSS,
            _ENVELOPE_CAP_TEST_RETURNS,
        )
        assert cap == pytest.approx(2.0 / COMMITTEE_TARGET_GROSS)

    def test_envelope_none_target_gross(self) -> None:
        cap = scaling._envelope_exposure_cap(
            GROWTH_RISK_ENVELOPES["conservative"], None,
            _ENVELOPE_CAP_TEST_RETURNS,
        )
        assert cap == GROWTH_RISK_ENVELOPES["conservative"].leverage_ceiling

    def test_growth_cap_raises_on_insufficient_history(self) -> None:
        short = pd.Series(
            [0.001], index=pd.date_range("2021-01-01", periods=1, freq="D", tz="UTC"),
        )
        with pytest.raises(ValueError, match="too little history"):
            scaling._envelope_exposure_cap(
                GROWTH_RISK_ENVELOPES["growth"], COMMITTEE_TARGET_GROSS, short,
            )
