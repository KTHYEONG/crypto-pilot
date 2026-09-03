"""Tests for the MHS application scaling module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application.research.mhs.contracts import MhsDiagnosticRequest
from src.application.research.mhs import scaling
from src.common.errors import DataIntegrityError
from src.mhs.params import (
    COMMITTEE_OOS_START,
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


# SCENARIO_MHS_EXPOSURE_CEILING_03 fixture: same low-Sharpe generator as
# _ENVELOPE_CAP_FRONTIER_RETURNS, sized so the AUDITED pre-OOS slice itself
# carries a growth frontier (measured 1.5x) strictly below the registered
# growth ceiling of 2.0 -- the audit verifies train rows only (I3).
_ENVELOPE_CAP_AUDIT_RETURNS = pd.Series(
    np.random.default_rng(11).normal(0.0009, 0.02, 600),
    index=pd.date_range("2021-01-01", periods=600, freq="D", tz="UTC"),
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


# SCENARIO_MHS_EXPOSURE_CEILING_01
def test_scenario_mhs_exposure_ceiling_01_cap_returns_registered_policy_constant() -> None:
    cap_growth = scaling._envelope_exposure_cap(
        GROWTH_RISK_ENVELOPES["growth"], COMMITTEE_TARGET_GROSS,
        _ENVELOPE_CAP_TEST_RETURNS,
    )
    assert cap_growth == 2.0
    assert cap_growth == GROWTH_RISK_ENVELOPES["growth"].leverage_ceiling
    frontier_multiple = _frontier_multiple(_ENVELOPE_CAP_TEST_RETURNS, "growth")
    assert frontier_multiple is not None
    assert cap_growth != pytest.approx(frontier_multiple)
    cap_moderate = scaling._envelope_exposure_cap(
        GROWTH_RISK_ENVELOPES["growth_moderate"], COMMITTEE_TARGET_GROSS,
        _ENVELOPE_CAP_TEST_RETURNS,
    )
    assert cap_moderate == 1.5


# SCENARIO_MHS_EXPOSURE_CEILING_02
def test_scenario_mhs_exposure_ceiling_02_fail_closed_branches_preserved() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        scaling._envelope_exposure_cap(
            GROWTH_RISK_ENVELOPES["growth"], COMMITTEE_TARGET_GROSS,
            _ENVELOPE_CAP_FRONTIER_RETURNS,
        )
    short = pd.Series(
        [0.001], index=pd.date_range("2021-01-01", periods=1, freq="D", tz="UTC"),
    )
    with pytest.raises(ValueError, match="too little history"):
        scaling._envelope_exposure_cap(
            GROWTH_RISK_ENVELOPES["growth"], COMMITTEE_TARGET_GROSS, short,
        )


# SCENARIO_MHS_EXPOSURE_CEILING_03
def test_scenario_mhs_exposure_ceiling_03_audit_noop_on_fold_local_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _sentinel(*args: object, **kwargs: object) -> float:
        raise AssertionError("_envelope_exposure_cap must not be invoked")

    monkeypatch.setattr(scaling, "_envelope_exposure_cap", _sentinel)
    idx = pd.date_range("2023-06-01", periods=200, freq="D", tz="UTC")
    r = pd.Series(0.001, index=idx)  # every row >= COMMITTEE_OOS_START
    assert scaling._assert_envelope_leverage_ceiling_verified(
        GROWTH_RISK_ENVELOPES["growth"], r,
    ) is None


# SCENARIO_MHS_EXPOSURE_CEILING_03
def test_scenario_mhs_exposure_ceiling_03_audit_propagates_fail_closed_breach() -> None:
    from src.mhs.params import PNL_VOL_TARGET_BURN_IN_DAYS

    assert len(_ENVELOPE_CAP_AUDIT_RETURNS) >= 2 * PNL_VOL_TARGET_BURN_IN_DAYS
    with pytest.raises(ValueError, match="must not exceed"):
        scaling._assert_envelope_leverage_ceiling_verified(
            GROWTH_RISK_ENVELOPES["growth"], _ENVELOPE_CAP_AUDIT_RETURNS,
        )


# SCENARIO_MHS_EXPOSURE_CEILING_04
def test_scenario_mhs_exposure_ceiling_04_replay_two_sided_uses_policy_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _sentinel(*args: object, **kwargs: object) -> float:
        raise AssertionError("_envelope_exposure_cap must not be invoked from replay")

    monkeypatch.setattr(scaling, "_envelope_exposure_cap", _sentinel)
    rng = np.random.default_rng(42)
    idx = pd.date_range("2021-01-01", periods=4 * 365, freq="D", tz="UTC")
    ref = pd.Series(rng.normal(0.0002, 0.015, len(idx)), index=idx)
    two_sided = MhsDiagnosticRequest(
        pnl_vol_target_mode="growth_budget",
        growth_envelope="growth",
        exposure_scale_two_sided=True,
    )
    one_sided = MhsDiagnosticRequest(
        pnl_vol_target_mode="growth_budget",
        growth_envelope="growth",
        exposure_scale_two_sided=False,
    )
    scaled = scaling._replay_exposure_scale(ref, two_sided)
    unscaled = scaling._replay_exposure_scale(ref, one_sided)
    assert 1.0 < scaled.max() <= 2.0 + 1e-12
    assert unscaled.max() <= 1.0 + 1e-12
    assert scaled.mean() > unscaled.mean()


# SCENARIO_MHS_KELLY_TWO_SIDED_01
def test_scenario_mhs_kelly_two_sided_01_resolved_cap_policy_matrix() -> None:
    """resolved_exposure_cap is the single data-independent cap owner (I1/I2):
    two_sided=False or median_relative -> 1.0, conservative exante ->
    PNL_VOL_TARGET_MAX_SCALE, otherwise the envelope's leverage_ceiling."""
    from src.mhs.params import PNL_VOL_TARGET_MAX_SCALE

    cases: list[tuple[dict, float]] = [
        (
            {
                "exposure_scale_two_sided": False,
                "pnl_vol_target_mode": "growth_budget",
                "growth_envelope": "growth",
            },
            1.0,
        ),
        (
            {
                "exposure_scale_two_sided": True,
                "pnl_vol_target_mode": "exante_target",
                "growth_envelope": "conservative",
            },
            float(PNL_VOL_TARGET_MAX_SCALE),
        ),
        (
            {
                "exposure_scale_two_sided": True,
                "pnl_vol_target_mode": "growth_budget",
                "growth_envelope": "growth",
            },
            2.0,
        ),
        (
            {
                "exposure_scale_two_sided": True,
                "pnl_vol_target_mode": "growth_budget",
                "growth_envelope": "growth_extreme",
            },
            3.0,
        ),
        (
            {
                "exposure_scale_two_sided": True,
                "pnl_vol_target_mode": "growth_budget",
                "growth_envelope": "growth_moderate",
            },
            1.5,
        ),
    ]
    for kwargs, expected in cases:
        assert scaling.resolved_exposure_cap(MhsDiagnosticRequest(**kwargs)) == expected
    # The (two_sided=True, median_relative) combination is rejected by request
    # validation; resolved_exposure_cap still resolves it to 1.0 defensively.
    median_req = MhsDiagnosticRequest(pnl_vol_target_mode="median_relative")
    object.__setattr__(median_req, "exposure_scale_two_sided", True)
    assert scaling.resolved_exposure_cap(median_req) == 1.0


_KELLY_DRIFT_RETURNS = pd.Series(
    # Strong positive drift so the raw Kelly LCB ratio exceeds every tested cap.
    np.random.default_rng(20260823).normal(0.02, 0.005, 400),
    index=pd.date_range("2021-01-01", periods=400, freq="D", tz="UTC"),
)


def _raw_kelly_ratio(r: pd.Series) -> pd.Series:
    from src.mhs.params import PNL_VOL_TARGET_WINDOW_DAYS

    min_periods = max(5, PNL_VOL_TARGET_WINDOW_DAYS // 2)
    trailing_mean = r.rolling(PNL_VOL_TARGET_WINDOW_DAYS, min_periods=min_periods).mean().shift(1)
    trailing_std = r.rolling(PNL_VOL_TARGET_WINDOW_DAYS, min_periods=min_periods).std().shift(1)
    trailing_n = r.rolling(PNL_VOL_TARGET_WINDOW_DAYS, min_periods=min_periods).count().shift(1)
    se = trailing_std.div(np.sqrt(trailing_n))
    var = trailing_std.pow(2)
    return 0.25 * (trailing_mean - 1.0 * se).div(var.where(var > 0))


# SCENARIO_MHS_KELLY_TWO_SIDED_02
def test_scenario_mhs_kelly_two_sided_02_kelly_cap_param_and_guard() -> None:
    from src.mhs.params import PNL_VOL_TARGET_SCALE_FLOOR

    r = _KELLY_DRIFT_RETURNS
    raw = _raw_kelly_ratio(r)
    legacy = scaling._committee_kelly_scale(r)
    assert legacy.max() <= 1.0 + 1e-12
    pd.testing.assert_series_equal(
        legacy, raw.clip(lower=PNL_VOL_TARGET_SCALE_FLOOR, upper=1.0).fillna(1.0),
        check_exact=True,
    )
    widened = scaling._committee_kelly_scale(r, cap=3.0)
    assert widened.max() > 1.0
    assert widened.max() <= 3.0 + 1e-12
    assert widened.mean() > legacy.mean()
    with pytest.raises(ValueError, match=r"cap must be >= 1.0"):
        scaling._committee_kelly_scale(r, cap=0.5)


# SCENARIO_MHS_KELLY_TWO_SIDED_03
def test_scenario_mhs_kelly_two_sided_03_replay_threads_cap_through_blend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260824)
    idx = pd.date_range("2021-01-01", periods=500, freq="D", tz="UTC")
    ref = pd.Series(rng.normal(0.002, 0.01, len(idx)), index=idx)
    req = MhsDiagnosticRequest(
        pnl_vol_target_mode="growth_budget",
        growth_envelope="growth_extreme",
        exposure_scale_two_sided=True,
        committee_capital=True,
        committee_kelly_sizing=True,
    )
    threaded = scaling._replay_exposure_scale(ref, req)
    assert threaded.max() <= 3.0 + 1e-12

    real_kelly = scaling._committee_kelly_scale

    def _legacy_capped(*args: object, **kwargs: object) -> pd.Series:
        kwargs.pop("cap", None)
        return real_kelly(*args, cap=1.0)

    monkeypatch.setattr(scaling, "_committee_kelly_scale", _legacy_capped)
    de_threaded = scaling._replay_exposure_scale(ref, req)
    monkeypatch.undo()
    assert threaded.mean() > de_threaded.mean()

    no_kelly_req = MhsDiagnosticRequest(
        pnl_vol_target_mode="growth_budget",
        growth_envelope="growth_extreme",
        exposure_scale_two_sided=True,
        committee_capital=True,
        committee_kelly_sizing=False,
    )
    expected_no_kelly = scaling._exante_vol_target_scale(
        ref,
        target_vol=scaling._growth_budget_target_vol(
            ref, envelope=GROWTH_RISK_ENVELOPES["growth_extreme"],
        ),
        cap=3.0,
    )
    pd.testing.assert_series_equal(
        scaling._replay_exposure_scale(ref, no_kelly_req), expected_no_kelly,
        check_exact=True,
    )


# SCENARIO_MHS_KELLY_TWO_SIDED_04
@pytest.mark.parametrize("mode", ["median_relative", "exante_target", "growth_budget"])
@pytest.mark.parametrize("envelope", ["conservative", "growth"])
def test_scenario_mhs_kelly_two_sided_04_legacy_paths_byte_identical(
    mode: str, envelope: str,
) -> None:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2021-01-01", periods=500, freq="D", tz="UTC")
    ref = pd.Series(rng.normal(0.0001, 0.01, len(idx)), index=idx)
    request = MhsDiagnosticRequest(
        pnl_vol_target_mode=mode,
        growth_envelope=envelope,
        exposure_scale_two_sided=False,
        committee_capital=False,
        committee_kelly_sizing=False,
    )
    result = scaling._replay_exposure_scale(ref, request)
    assert result.max() <= 1.0 + 1e-12
    if mode == "median_relative":
        primitive = scaling._pnl_vol_target_scale(ref)
    elif mode == "exante_target" and envelope == "conservative":
        primitive = scaling._exante_vol_target_scale(ref, cap=1.0)
    else:
        primitive = scaling._exante_vol_target_scale(
            ref,
            target_vol=scaling._growth_budget_target_vol(
                ref, envelope=GROWTH_RISK_ENVELOPES[envelope],
            ),
            cap=1.0,
        )
    expected = scaling._committee_capital_replay_scale(primitive, ref, False, False, cap=1.0)
    pd.testing.assert_series_equal(result, expected, check_exact=True)


# SCENARIO_MHS_CONSTANT_RISK_SCALE_EQUALIZES_REALIZED_VOL
def test_constant_risk_scale_equalizes_realized_vol() -> None:
    # halflife=150d(CONSTANT_RISK_EWMA_HALFLIFE_DAYS)이 완전히 수렴하려면
    # 각 국면이 halflife의 수 배(>=600일)는 되어야 한다.
    rng = np.random.default_rng(20260823)
    idx = pd.date_range("2021-01-01", periods=1600, freq="D", tz="UTC")
    rets = np.concatenate([rng.normal(0.0, 0.01, 800), rng.normal(0.0, 0.03, 800)])
    r = pd.Series(rets, index=idx)
    scale = scaling._constant_risk_scale(r, target_vol=0.20, cap=8.0)
    assert scale.index.equals(r.index)
    scaled = scale * r
    v_low = float(scaled.iloc[200:800].std(ddof=1) * np.sqrt(365))
    v_high = float(scaled.iloc[1200:].std(ddof=1) * np.sqrt(365))
    ratio = v_high / v_low
    assert 0.8 <= ratio <= 1.25
    raw_low = float(r.iloc[:800].std(ddof=1) * np.sqrt(365))
    raw_high = float(r.iloc[800:].std(ddof=1) * np.sqrt(365))
    assert raw_high / raw_low >= 2.5


# SCENARIO_MHS_CONSTANT_RISK_SCALE_IS_STRICTLY_CAUSAL
def test_constant_risk_scale_causal() -> None:
    rng = np.random.default_rng(11)
    idx = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")
    r = pd.Series(rng.normal(0.0005, 0.02, 400), index=idx)
    k = 217
    perturbed = r.copy()
    perturbed.iloc[k] *= 100.0
    base = scaling._constant_risk_scale(r, target_vol=0.30, cap=5.0)
    changed = scaling._constant_risk_scale(perturbed, target_vol=0.30, cap=5.0)
    pd.testing.assert_series_equal(changed.iloc[: k + 1], base.iloc[: k + 1], check_exact=True)
    assert not changed.iloc[k + 1 :].equals(base.iloc[k + 1 :])


# SCENARIO_MHS_CONSTANT_RISK_WARMUP_REMOVES_DEAD_ZONE
def test_constant_risk_warmup_removes_dead_zone() -> None:
    from src.mhs.params import CONSTANT_RISK_MIN_PERIODS_DAYS

    rng = np.random.default_rng(5)
    idx = pd.date_range("2023-01-01", periods=200, freq="D", tz="UTC")
    r = pd.Series(rng.normal(0.001, 0.02, 200), index=idx)
    cold = scaling._constant_risk_scale(r, target_vol=0.25, cap=4.0)
    assert cold.index.equals(r.index)
    assert (cold.iloc[:CONSTANT_RISK_MIN_PERIODS_DAYS] == 1.0).all()
    warm_idx = pd.date_range(
        end=idx[0] - pd.Timedelta(days=1), periods=200, freq="D", tz="UTC",
    )
    warm = pd.Series(rng.normal(0.001, 0.02, 200), index=warm_idx)
    warmed = scaling._constant_risk_scale(r, target_vol=0.25, cap=4.0, warmup_returns=warm)
    assert warmed.index.equals(r.index)
    assert warmed.iloc[0] != 1.0
    assert float((warmed == 1.0).mean()) == 0.0


# SCENARIO_MHS_CONSTANT_RISK_WARMUP_OVERLAP_FAILS_CLOSED
def test_constant_risk_warmup_overlap_fails_closed() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
    r = pd.Series(0.01, index=idx)
    warm_idx = pd.date_range("2023-11-01", periods=70, freq="D", tz="UTC")
    overlapping_warmup = pd.Series(0.01, index=warm_idx)
    with pytest.raises(ValueError, match="warmup"):
        scaling._constant_risk_scale(r, 0.25, 4.0, warmup_returns=overlapping_warmup)


# SCENARIO_MHS_FEASIBLE_TARGET_CLAMPS_TO_LEVERAGE_CEILING
def test_feasible_constant_risk_target_clamps_to_leverage_ceiling() -> None:
    from src.mhs.params import (
        CONSTANT_RISK_CAP_BINDING_QUANTILE,
        CONSTANT_RISK_EWMA_HALFLIFE_DAYS,
        CONSTANT_RISK_MIN_PERIODS_DAYS,
    )

    envelope = GROWTH_RISK_ENVELOPES["growth_extreme"]
    rng = np.random.default_rng(17)
    idx = pd.date_range("2021-01-01", periods=1000, freq="D", tz="UTC")
    ref = pd.Series(rng.normal(0.0, 0.25 / np.sqrt(365), 1000), index=idx)
    clamped = scaling._feasible_constant_risk_target(ref, envelope, 0.862)
    sigma_book = ref.ewm(
        halflife=CONSTANT_RISK_EWMA_HALFLIFE_DAYS,
        min_periods=CONSTANT_RISK_MIN_PERIODS_DAYS,
    ).std().shift(1) * np.sqrt(365)
    expected_clamp = envelope.leverage_ceiling * float(sigma_book.quantile(CONSTANT_RISK_CAP_BINDING_QUANTILE))
    assert clamped < 0.862
    assert clamped == pytest.approx(expected_clamp, abs=1e-9)
    scale = scaling._constant_risk_scale(ref, target_vol=clamped, cap=envelope.leverage_ceiling)
    saturation = float((scale >= envelope.leverage_ceiling - 1e-12).mean())
    assert saturation <= CONSTANT_RISK_CAP_BINDING_QUANTILE + 0.05
    assert scaling._feasible_constant_risk_target(ref, envelope, 0.10) == 0.10


# SCENARIO_MHS_FEASIBLE_TARGET_INSUFFICIENT_HISTORY_FAILS_CLOSED
def test_feasible_constant_risk_target_insufficient_history_fails_closed() -> None:
    from src.mhs.params import CONSTANT_RISK_MIN_PERIODS_DAYS

    envelope = GROWTH_RISK_ENVELOPES["growth_extreme"]
    rng = np.random.default_rng(19)
    n_rows = int(CONSTANT_RISK_MIN_PERIODS_DAYS * 1.5)
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="D", tz="UTC")
    ref = pd.Series(rng.normal(0.0, 0.02, n_rows), index=idx)
    with pytest.raises(DataIntegrityError):
        scaling._feasible_constant_risk_target(ref, envelope, PNL_TARGET_ANNUAL_VOL)


# SCENARIO_MHS_CONSTANT_RISK_BOUNDARY_TARGETS_ARE_LEAK_FREE
def test_constant_risk_target_vol_by_boundary_leak_free() -> None:
    envelope = GROWTH_RISK_ENVELOPES["growth"]
    rng = np.random.default_rng(23)
    idx = pd.date_range("2021-01-01", periods=4 * 365, freq="D", tz="UTC")
    r = pd.Series(rng.normal(0.0003, 0.015, len(idx)), index=idx)
    train_ends = {
        "top_level": pd.Timestamp("2023-01-01", tz="UTC"),
        "fold_0": pd.Timestamp("2022-01-01", tz="UTC"),
        "fold_1": pd.Timestamp("2023-01-01", tz="UTC"),
    }
    resolved = scaling._constant_risk_target_vol_by_boundary(r, envelope, train_ends)
    assert set(resolved) == set(train_ends)
    perturbed = r.copy()
    # 모든 경계(최종 2023-01-01) 이후 구간만 100배로 변경한다.
    perturbed.loc[perturbed.index >= pd.Timestamp("2024-06-01", tz="UTC")] *= 100.0
    resolved_perturbed = scaling._constant_risk_target_vol_by_boundary(perturbed, envelope, train_ends)
    for label, value in resolved.items():
        assert resolved_perturbed[label] == pytest.approx(value, abs=1e-12)
    with pytest.raises(DataIntegrityError, match="fold_9"):
        scaling._constant_risk_target_vol_by_boundary(
            r, envelope, {"fold_9": pd.Timestamp("2021-03-01", tz="UTC")},
        )


# SCENARIO_MHS_CONSTANT_RISK_TOP_LEVEL_MATCHES_BOUNDARY
def test_constant_risk_top_level_target_vol_matches_boundary() -> None:
    """FOLD_BLEND_PATH_DIVERGENCE 회귀 방지: top-level dispatcher가 전체
    5년 참조로 sigma_book을 적합하면 fold의 oos_start 이전 전용 적합과
    갈라진다(실측: exposure_scale_mean 2.36~2.96 vs 1.77~3.0, 실제 파이프라인
    실행에서 FOLD_BLEND_PATH_DIVERGENCE 발화 확인). _replay_exposure_scale이
    growth_budget_target_vol=None일 때 계산하는 target_vol은
    _constant_risk_target_vol_by_boundary의 "top_level" 라벨 결과와
    정확히 일치해야 한다.
    """
    envelope = GROWTH_RISK_ENVELOPES["growth_extreme"]
    rng = np.random.default_rng(41)
    idx = pd.date_range("2021-01-01", periods=5 * 365, freq="D", tz="UTC")
    ref = pd.Series(rng.normal(0.0003, 0.02, len(idx)), index=idx)
    req = MhsDiagnosticRequest(
        pnl_vol_target_mode="constant_risk", growth_envelope="growth_extreme",
        exposure_scale_two_sided=True,
    )
    dispatched_scale = scaling._replay_exposure_scale(ref, req)
    boundary_target = scaling._constant_risk_target_vol_by_boundary(
        ref, envelope, {"top_level": COMMITTEE_OOS_START},
    )["top_level"]
    expected_scale = scaling._constant_risk_scale(
        ref, target_vol=boundary_target, cap=scaling.resolved_exposure_cap(req),
    )
    pd.testing.assert_series_equal(dispatched_scale, expected_scale)


# SCENARIO_MHS_CONSTANT_RISK_BYPASSES_KELLY_BLEND
def test_constant_risk_bypasses_kelly_blend() -> None:
    rng = np.random.default_rng(31)
    idx = pd.date_range("2021-01-01", periods=500, freq="D", tz="UTC")
    ref = pd.Series(rng.normal(0.0008, 0.02, 500), index=idx)
    kelly_req = MhsDiagnosticRequest(
        pnl_vol_target_mode="constant_risk",
        exposure_scale_two_sided=True,
        committee_capital=True,
        committee_kelly_sizing=True,
    )
    plain_req = MhsDiagnosticRequest(
        pnl_vol_target_mode="constant_risk",
        exposure_scale_two_sided=True,
        committee_capital=True,
        committee_kelly_sizing=False,
    )
    pd.testing.assert_series_equal(
        scaling._replay_exposure_scale(ref, kelly_req),
        scaling._replay_exposure_scale(ref, plain_req),
        check_exact=True,
    )
    gb_kelly = MhsDiagnosticRequest(
        pnl_vol_target_mode="growth_budget",
        exposure_scale_two_sided=True,
        committee_capital=True,
        committee_kelly_sizing=True,
    )
    gb_plain = MhsDiagnosticRequest(
        pnl_vol_target_mode="growth_budget",
        exposure_scale_two_sided=True,
        committee_capital=True,
        committee_kelly_sizing=False,
    )
    s_kelly = scaling._replay_exposure_scale(ref, gb_kelly)
    s_plain = scaling._replay_exposure_scale(ref, gb_plain)
    assert not s_kelly.equals(s_plain)


# SCENARIO_MHS_LEGACY_EXPOSURE_MODES_BYTE_IDENTICAL
def test_legacy_exposure_modes_byte_identical() -> None:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2021-01-01", periods=500, freq="D", tz="UTC")
    ref = pd.Series(rng.normal(0.0001, 0.01, 500), index=idx)

    median_req = MhsDiagnosticRequest(pnl_vol_target_mode="median_relative")
    pd.testing.assert_series_equal(
        scaling._replay_exposure_scale(ref, median_req),
        scaling._pnl_vol_target_scale(ref),
        check_exact=False, rtol=0,
    )
    assert scaling.resolved_exposure_cap(median_req) == 1.0

    exante_req = MhsDiagnosticRequest(pnl_vol_target_mode="exante_target")
    pd.testing.assert_series_equal(
        scaling._replay_exposure_scale(ref, exante_req),
        scaling._exante_vol_target_scale(ref, cap=1.0),
        check_exact=False, rtol=0,
    )
    assert scaling.resolved_exposure_cap(exante_req) == 1.0

    gb_req = MhsDiagnosticRequest(
        pnl_vol_target_mode="growth_budget",
        growth_envelope="growth",
        exposure_scale_two_sided=True,
        committee_capital=False,
    )
    expected_gb = scaling._exante_vol_target_scale(
        ref,
        target_vol=scaling._growth_budget_target_vol(
            ref, envelope=GROWTH_RISK_ENVELOPES["growth"],
        ),
        cap=2.0,
    )
    pd.testing.assert_series_equal(
        scaling._replay_exposure_scale(ref, gb_req), expected_gb,
        check_exact=False, rtol=0,
    )
    assert scaling.resolved_exposure_cap(gb_req) == 2.0

    constant_req = MhsDiagnosticRequest(pnl_vol_target_mode="constant_risk")
    assert scaling.is_streaming_scale_mode(constant_req) is False


# SCENARIO_MHS_DD_BRAKE_11_NO_REGRESSION: 기본값(False)에서 4개 pnl_vol_target_mode
# 전부 비트 동일 회귀는 아래 3파일 전체 스위트 실행으로 검증한다.
#   uv run pytest tests/unit/application/research/mhs/test_scaling.py \
#       tests/unit/mhs/test_compounding_growth.py tests/contract/test_request_cli_parity.py -q

# SCENARIO_MHS_DD_BRAKE_01_INERT_ON_MONOTONE_EQUITY
def test_scenario_mhs_dd_brake_01_drawdown_brake_inert_on_monotone_equity() -> None:
    idx = pd.date_range("2021-01-01", periods=30, freq="D", tz="UTC")
    r = pd.Series(0.01, index=idx)
    base = pd.Series(1.3, index=idx)
    result = scaling._equity_drawdown_brake_scale(r, base, cap=3.0)
    pd.testing.assert_series_equal(result, base.astype("float64"), check_names=False)


# SCENARIO_MHS_DD_BRAKE_02_ONLY_REDUCES
def test_scenario_mhs_dd_brake_02_drawdown_brake_only_reduces() -> None:
    rng = np.random.default_rng(20260824)
    idx = pd.date_range("2021-01-01", periods=400, freq="D", tz="UTC")
    r = pd.Series(rng.normal(0.0, 0.03, 400), index=idx)
    base = pd.Series(1.5, index=idx)
    result = scaling._equity_drawdown_brake_scale(r, base, cap=3.0)
    assert (result <= base.clip(0.0, 3.0) + 1e-12).all()
    assert (result >= 0.0).all()
    assert result.notna().all()
    assert result.lt(base - 1e-9).any()


# SCENARIO_MHS_DD_BRAKE_03_PREFIX_DETERMINISTIC
def test_scenario_mhs_dd_brake_03_drawdown_brake_prefix_deterministic() -> None:
    rng = np.random.default_rng(20260824)
    idx = pd.date_range("2021-01-01", periods=400, freq="D", tz="UTC")
    r = pd.Series(rng.normal(0.0, 0.03, 400), index=idx)
    base = pd.Series(1.5, index=r.index)
    full = scaling._equity_drawdown_brake_scale(r, base, cap=3.0)
    for m in (1, 7, 50, 199, 400):
        prefix = scaling._equity_drawdown_brake_scale(
            r.iloc[:m], base.iloc[:m], cap=3.0,
        )
        pd.testing.assert_series_equal(prefix, full.iloc[:m], check_exact=True)


# SCENARIO_MHS_DD_BRAKE_04_FLOOR_AND_K_BOUNDS
def test_scenario_mhs_dd_brake_04_drawdown_brake_floor_and_k_bounds() -> None:
    idx = pd.date_range("2022-01-01", periods=60, freq="D", tz="UTC")
    r = pd.Series(-0.05, index=idx)
    base = pd.Series(1.0, index=idx)
    floored = scaling._equity_drawdown_brake_scale(
        r, base, cap=3.0, k=50.0, floor=0.2,
    )
    assert floored.min() == pytest.approx(0.2)
    assert (floored >= 0.2 - 1e-12).all()
    identity = scaling._equity_drawdown_brake_scale(
        r, base, cap=3.0, k=0.0, floor=0.2,
    )
    pd.testing.assert_series_equal(
        identity, base.clip(0.0, 3.0).astype("float64"), check_names=False,
    )


# SCENARIO_MHS_DD_BRAKE_05_FAIL_CLOSED
def test_scenario_mhs_dd_brake_05_drawdown_brake_fail_closed() -> None:
    idx = pd.date_range("2023-01-01", periods=40, freq="D", tz="UTC")
    r = pd.Series(0.01, index=idx)
    base = pd.Series(1.0, index=idx)
    shifted_base = pd.Series(1.0, index=pd.date_range("2023-01-02", periods=40, freq="D", tz="UTC"))
    with pytest.raises(ValueError, match="indexes must be equal"):
        scaling._equity_drawdown_brake_scale(r, shifted_base, cap=3.0)
    with pytest.raises(ValueError, match="floor"):
        scaling._equity_drawdown_brake_scale(r, base, cap=3.0, floor=0.0)
    with pytest.raises(ValueError, match="floor"):
        scaling._equity_drawdown_brake_scale(r, base, cap=3.0, floor=1.5)
    with pytest.raises(ValueError, match=r"\bk\b"):
        scaling._equity_drawdown_brake_scale(r, base, cap=3.0, k=-1.0)
    with pytest.raises(ValueError, match="cap"):
        scaling._equity_drawdown_brake_scale(r, base, cap=-0.1)
    ruin = pd.Series(-1.5, index=idx)
    with pytest.raises(DataIntegrityError):
        scaling._equity_drawdown_brake_scale(ruin, base, cap=3.0)
    nan_lead_idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    nan_lead = pd.Series(np.nan, index=nan_lead_idx)
    nan_lead.iloc[1:] = [0.02, -0.01, 0.03, 0.01, -0.02, 0.04, 0.01, -0.03, 0.02]
    nan_result = scaling._equity_drawdown_brake_scale(
        nan_lead, pd.Series(1.0, index=nan_lead_idx), cap=3.0,
    )
    assert np.isfinite(nan_result.iloc[0])


# SCENARIO_MHS_DD_BRAKE_06_REPLAY_DEFAULT_BIT_IDENTICAL
def test_scenario_mhs_dd_brake_06_drawdown_brake_replay_default_bit_identical() -> None:
    from src.application.research.mhs.research_go import _resolved_growth_envelope

    rng = np.random.default_rng(37)
    idx = pd.date_range("2021-01-01", periods=500, freq="D", tz="UTC")
    ref = pd.Series(rng.normal(0.0005, 0.015, 500), index=idx)
    request = MhsDiagnosticRequest(pnl_vol_target_mode="constant_risk")
    expected = scaling._constant_risk_scale(
        ref,
        target_vol=scaling._constant_risk_target_vol(ref, _resolved_growth_envelope(request)),
        cap=scaling.resolved_exposure_cap(request),
    )
    pd.testing.assert_series_equal(
        scaling._replay_exposure_scale(ref, request), expected, check_exact=True,
    )
    braked_request = MhsDiagnosticRequest(
        pnl_vol_target_mode="constant_risk",
        exposure_drawdown_brake=True,
    )
    plain = scaling._replay_exposure_scale(ref, request)
    braked = scaling._replay_exposure_scale(ref, braked_request)
    assert (braked <= plain + 1e-12).all()
    assert braked.lt(plain - 1e-9).any()


# SCENARIO_MHS_DD_BRAKE_07_NON_FITTED_TARGET_VOL
def test_scenario_mhs_dd_brake_07_drawdown_brake_non_fitted_target_vol() -> None:
    from src.mhs.params import (
        CONSTANT_RISK_CAP_BINDING_QUANTILE,
        CONSTANT_RISK_EWMA_HALFLIFE_DAYS,
        CONSTANT_RISK_MIN_PERIODS_DAYS,
    )

    envelope = GROWTH_RISK_ENVELOPES["growth_extreme"]
    rng = np.random.default_rng(29)
    idx = pd.date_range("2021-08-01", periods=500, freq="D", tz="UTC")
    ref = pd.Series(rng.normal(0.0004, 0.018, 500), index=idx)
    braked = scaling._constant_risk_target_vol(ref, envelope, drawdown_brake=True)
    train = ref.loc[ref.index < COMMITTEE_OOS_START].dropna()
    sigma_book = (
        train.ewm(
            halflife=CONSTANT_RISK_EWMA_HALFLIFE_DAYS,
            min_periods=CONSTANT_RISK_MIN_PERIODS_DAYS,
        ).std().shift(1) * np.sqrt(365.0)
    )
    feasible = envelope.leverage_ceiling * float(sigma_book.quantile(CONSTANT_RISK_CAP_BINDING_QUANTILE))
    assert braked == pytest.approx(feasible)
    assert braked >= scaling._constant_risk_target_vol(ref, envelope, drawdown_brake=False)


# SCENARIO_MHS_DD_BRAKE_08_NEVER_STREAMING
@pytest.mark.parametrize("mode", ["median_relative", "exante_target"])
def test_scenario_mhs_dd_brake_08_drawdown_brake_never_streaming(mode: str) -> None:
    request = MhsDiagnosticRequest(
        pnl_vol_target_mode=mode,
        growth_envelope="conservative",
        committee_capital=False,
    )
    assert scaling.is_streaming_scale_mode(request) is True
    # 브레이크 ON + 비constant_risk는 validation이 거부하는 형태지만,
    # 스트리밍 가드의 방어깊이(I-BRAKE-NO-STREAM)는 모드 무관하게 단독 검증한다.
    braked = MhsDiagnosticRequest(pnl_vol_target_mode="constant_risk")
    object.__setattr__(braked, "pnl_vol_target_mode", mode)
    object.__setattr__(braked, "exposure_drawdown_brake", True)
    assert scaling.is_streaming_scale_mode(braked) is False


# SCENARIO_MHS_DD_BRAKE_09_REQUEST_VALIDATION
def test_scenario_mhs_dd_brake_09_drawdown_brake_request_validation() -> None:
    with pytest.raises(ValueError, match="constant_risk"):
        MhsDiagnosticRequest(
            pnl_vol_target_mode="growth_budget",
            exposure_drawdown_brake=True,
        )
    with pytest.raises(ValueError, match="pnl_vol_target"):
        MhsDiagnosticRequest(
            pnl_vol_target=False,
            pnl_vol_target_mode="constant_risk",
            exposure_drawdown_brake=True,
        )
    MhsDiagnosticRequest(
        pnl_vol_target=True,
        pnl_vol_target_mode="constant_risk",
        exposure_drawdown_brake=True,
    )
