"""Tests for the MHS application scaling module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application.research.mhs.contracts import MhsDiagnosticRequest
from src.application.research.mhs import scaling
from src.common.errors import DataIntegrityError
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


def test_growth_budget_boundary_resolves_each_train_end_slice() -> None:
    # SCENARIO_MHS_GROWTH_BUDGET_BOUNDARY_01: each boundary fits strictly on
    # reference rows before its own train_end -- fold_0's value equals a direct
    # growth_budget_annual_vol call on the pre-2022 slice (I3 leak-free), and
    # every resolved value stays finite within the registered [0.05, 1.0] band.
    from src.mhs.committee import growth_budget_annual_vol

    rng = np.random.default_rng(7)
    idx = pd.date_range("2021-01-01", periods=4 * 365, freq="D", tz="UTC")
    r = pd.Series(rng.normal(0.0002, 0.015, len(idx)), index=idx)
    envelope = GROWTH_RISK_ENVELOPES["balanced"]
    train_ends = {
        "top_level": pd.Timestamp("2023-01-01", tz="UTC"),
        "fold_0": pd.Timestamp("2022-01-01", tz="UTC"),
        "fold_1": pd.Timestamp("2023-01-01", tz="UTC"),
    }
    resolved = scaling._growth_budget_target_vol_by_boundary(r, envelope, train_ends)
    assert set(resolved) == set(train_ends)
    for value in resolved.values():
        assert np.isfinite(value)
        assert 0.05 <= value <= 1.0
    expected_fold_0 = growth_budget_annual_vol(
        r[r.index < pd.Timestamp("2022-01-01", tz="UTC")], envelope=envelope,
    )
    assert resolved["fold_0"] == pytest.approx(expected_fold_0, abs=1e-12)


def test_growth_budget_boundary_fail_closed_on_insufficient_train() -> None:
    # SCENARIO_MHS_GROWTH_BUDGET_BOUNDARY_02: a boundary whose train slice has
    # fewer than PNL_VOL_TARGET_BURN_IN_DAYS finite rows raises
    # DataIntegrityError naming the offending boundary key (I4 fail-closed);
    # the single-shot resolver keeps its silent fallback unless fail_closed.
    idx = pd.date_range("2023-06-01", periods=80, freq="D", tz="UTC")
    r = pd.Series(0.001, index=idx)
    envelope = GROWTH_RISK_ENVELOPES["balanced"]
    with pytest.raises(DataIntegrityError, match="fold_9"):
        scaling._growth_budget_target_vol_by_boundary(
            r, envelope, {"fold_9": pd.Timestamp("2023-01-01", tz="UTC")},
        )
    assert scaling._growth_budget_target_vol(r, envelope=envelope) == PNL_TARGET_ANNUAL_VOL
    with pytest.raises(DataIntegrityError):
        scaling._growth_budget_target_vol(r, envelope=envelope, fail_closed=True)


def test_replay_exposure_scale_override_equals_direct_composition() -> None:
    # SCENARIO_MHS_GROWTH_BUDGET_BOUNDARY_03: a boundary-resolved target vol is
    # used verbatim (no fold-local refit) and composes exactly like the direct
    # exante scale + committee-capital composition; passing None keeps the
    # conservative/exante default path byte-identical to the pre-change code.
    rng = np.random.default_rng(42)
    idx = pd.date_range("2021-01-01", periods=500, freq="D", tz="UTC")
    ref = pd.Series(rng.normal(0.0001, 0.01, 500), index=idx)
    request = MhsDiagnosticRequest(pnl_vol_target_mode="growth_budget")
    overridden = scaling._replay_exposure_scale(ref, request, 0.3509)
    composed = scaling._committee_capital_replay_scale(
        scaling._exante_vol_target_scale(ref, target_vol=0.3509, cap=1.0),
        ref, request.committee_capital, request.committee_kelly_sizing,
    )
    pd.testing.assert_series_equal(overridden, composed, check_exact=True)
    default_request = MhsDiagnosticRequest(pnl_vol_target_mode="exante_target")
    expected_default = scaling._committee_capital_replay_scale(
        scaling._exante_vol_target_scale(ref, cap=1.0),
        ref, default_request.committee_capital, default_request.committee_kelly_sizing,
    )
    pd.testing.assert_series_equal(
        scaling._replay_exposure_scale(ref, default_request, None),
        expected_default, check_exact=True,
    )


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
    # production blend's realized Sharpe) over 750 rows -- required so the
    # "growth" envelope's bootstrap ruin frontier stays feasible and its
    # frontier multiple (measured 3.0x reference risk) sits at or above the
    # registered leverage ceiling of 2.0.
    np.random.default_rng(20260821).normal(0.0021, 0.02, 750),
    index=pd.date_range("2021-01-01", periods=750, freq="D", tz="UTC"),
)

# SCENARIO_MHS_ENVELOPE_CAP_FRONTIER_04 fixture: a series whose conservative
# bootstrap frontier multiple is exactly 0.5x reference risk and whose growth
# envelope frontier (~1.75x) sits BELOW the registered leverage_ceiling of 2.0,
# so the ceiling can no longer be wired without failing closed.
_ENVELOPE_CAP_FRONTIER_RETURNS = pd.Series(
    np.random.default_rng(11).normal(0.0009, 0.02, 750),
    index=pd.date_range("2021-01-01", periods=750, freq="D", tz="UTC"),
)


def _frontier_multiple(r: pd.Series, envelope_name: str) -> float | None:
    """Direct conservative-style frontier readout used to pin fixture facts."""
    from src.mhs.params import (
        COMMITTEE_GROWTH_BARS_PER_YEAR,
        COMMITTEE_GROWTH_N_PATHS,
        COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS,
    )
    from src.research.risk.growth_sizing import GrowthSizingConfig, solve_growth_optimal_risk

    env = GROWTH_RISK_ENVELOPES[envelope_name]
    x = r.dropna()
    ref_risk = float(x.std(ddof=1))
    config = GrowthSizingConfig(
        risk_grid=tuple(sorted(ref_risk * m for m in COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS)),
        reference_risk=ref_risk,
        max_drawdown=env.max_drawdown,
        max_drawdown_prob=env.max_drawdown_prob,
        ruin_fraction=env.ruin_fraction,
        max_ruin_prob=env.max_ruin_prob,
        horizon_years=env.horizon_years,
        n_paths=COMMITTEE_GROWTH_N_PATHS,
        bars_per_year=COMMITTEE_GROWTH_BARS_PER_YEAR,
    )
    result = solve_growth_optimal_risk(x.to_numpy(), config, use_drawdown_overlay=False)
    return None if result.selected_risk is None else result.selected_risk / ref_risk


# SCENARIO_ENVELOPE_EXPOSURE_CAP_LIFTS_UNIT_GROSS /
# SCENARIO_MHS_ENVELOPE_CAP_FRONTIER_04
class TestEnvelopeExposureCap:
    def test_cap_raises_when_ceiling_exceeds_frontier(self) -> None:
        # growth envelope (leverage_ceiling=2.0) against a series whose
        # conservative frontier multiple is 0.5x: the growth-env frontier on
        # this series (~1.75x) is below the ceiling -> fail closed.
        conservative_multiple = _frontier_multiple(_ENVELOPE_CAP_FRONTIER_RETURNS, "conservative")
        assert conservative_multiple == pytest.approx(0.5)
        with pytest.raises(ValueError, match="must not exceed"):
            scaling._envelope_exposure_cap(
                GROWTH_RISK_ENVELOPES["growth"], COMMITTEE_TARGET_GROSS,
                _ENVELOPE_CAP_FRONTIER_RETURNS,
            )

    def test_cap_returns_budget_derived_float_at_or_below_frontier(self) -> None:
        cap = scaling._envelope_exposure_cap(
            GROWTH_RISK_ENVELOPES["growth"], COMMITTEE_TARGET_GROSS,
            _ENVELOPE_CAP_TEST_RETURNS,
        )
        assert isinstance(cap, float)
        assert cap >= 1.0

    def test_registered_conservative_ceiling_must_be_verifiable(self) -> None:
        # Even the registered conservative ceiling (1.0) fails closed when the
        # verified frontier sits below it -- a cap that outruns its own budget
        # evidence is never silently returned.
        with pytest.raises(ValueError, match="must not exceed"):
            scaling._envelope_exposure_cap(
                GROWTH_RISK_ENVELOPES["conservative"], COMMITTEE_TARGET_GROSS,
                _ENVELOPE_CAP_TEST_RETURNS,
            )

    def test_growth_cap_raises_on_insufficient_history(self) -> None:
        short = pd.Series(
            [0.001], index=pd.date_range("2021-01-01", periods=1, freq="D", tz="UTC"),
        )
        with pytest.raises(ValueError, match="too little history"):
            scaling._envelope_exposure_cap(
                GROWTH_RISK_ENVELOPES["growth"], COMMITTEE_TARGET_GROSS, short,
            )
