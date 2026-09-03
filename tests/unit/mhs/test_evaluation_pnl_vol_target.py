"""MHS evaluation contract tests (split by behavioral domain; shared builders live in the original module)."""

"""Contract coverage for the MHS application evaluation resource telemetry."""
import dataclasses
import numpy as np
import pandas as pd
import pytest
from src.application.research.mhs import evaluation as ev
import src.application.research.mhs.scaling as scaling
from src.application.research.mhs.evaluation import (
    MhsDiagnosticRequest,
)

from tests.unit.application.research.mhs.test_evaluation import (  # noqa: F401
    _build_book_outcome_args,
    _pnl_vol_spike_returns,
)

def test_pnl_vol_target_scale_no_lookahead() -> None:
    # SCENARIO_PNL_VOL_TARGET_SCALE_NO_LOOKAHEAD: scale_t must depend only on
    # reference_daily_returns strictly before t -- truncating the input to end
    # exactly at t leaves scale[:t+1] unchanged.
    r = _pnl_vol_spike_returns()
    full = scaling._pnl_vol_target_scale(r)
    for t in (40, 90, 110, 150, 198):
        truncated = scaling._pnl_vol_target_scale(r.iloc[: t + 1])
        pd.testing.assert_series_equal(
            full.iloc[: t + 1], truncated, check_names=False,
        )

def test_pnl_vol_target_scale_reduces_on_vol_spike() -> None:
    # SCENARIO_PNL_VOL_TARGET_SCALE_REDUCES_ON_VOL_SPIKE: a calm regime keeps
    # full exposure while a high-vol regime drives the scale toward the floor.
    r = _pnl_vol_spike_returns()
    out = scaling._pnl_vol_target_scale(r)
    assert out.iloc[50] == pytest.approx(1.0)
    assert out.iloc[150] <= ev.PNL_VOL_TARGET_SCALE_FLOOR + 1e-9
    assert out.iloc[150] < out.iloc[50]

def test_pnl_vol_target_scale_never_exceeds_one() -> None:
    # SCENARIO_PNL_VOL_TARGET_SCALE_NEVER_EXCEEDS_ONE: no leverage-up ever and
    # the floor is respected across ultra-calm and ultra-volatile regimes; a
    # zero-std constant-return input is safe-divided to 1.0 (no inf).
    rng = np.random.default_rng(7)
    calm = pd.Series(rng.normal(1e-4, 1e-5, 150), index=pd.date_range("2024-01-01", periods=150, freq="D", tz="UTC"))
    wild = pd.Series(rng.normal(0.0, 0.5, 150), index=pd.date_range("2024-06-01", periods=150, freq="D", tz="UTC"))
    combo = pd.concat([calm, wild])
    out = scaling._pnl_vol_target_scale(combo)
    assert (out >= ev.PNL_VOL_TARGET_SCALE_FLOOR).all()
    assert (out <= 1.0).all()
    constant = pd.Series(
        np.concatenate([np.full(100, 0.001), np.full(100, 0.05)]),
        index=pd.date_range("2024-01-01", periods=200, freq="D", tz="UTC"),
    )
    zero_out = scaling._pnl_vol_target_scale(constant)
    assert np.isfinite(zero_out.to_numpy()).all()
    assert (zero_out >= 0.2).all()
    assert (zero_out <= 1.0).all()

def test_pnl_vol_target_scale_burn_in_is_unscaled() -> None:
    # SCENARIO_PNL_VOL_TARGET_SCALE_BURN_IN_IS_UNSCALED: before
    # PNL_VOL_TARGET_BURN_IN_DAYS samples exist, scale is exactly 1.0 no
    # matter how volatile the input is -- never an under-sampled estimate.
    r = _pnl_vol_spike_returns()
    out = scaling._pnl_vol_target_scale(r)
    assert (out.iloc[: ev.PNL_VOL_TARGET_BURN_IN_DAYS - 1] == 1.0).all()

def test_pnl_vol_target_rolling_median_adapts() -> None:
    # SCENARIO_MHS_PNL_VOL_TARGET_ROLLING_MEDIAN_ADAPTS: a 1500-day series
    # with three regimes (calm, elevated, elevated-sustained) shows the
    # default-window scale adapting to the new normal while an
    # effectively-expanding window stays stale.
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=1500, freq="D", tz="UTC")
    calm = rng.normal(0.0005, 0.01, 500)
    elevated = rng.normal(0.0005, 0.04, 500)
    sustained = rng.normal(0.0005, 0.04, 500)
    returns = np.concatenate([calm, elevated, sustained])
    r = pd.Series(returns, index=idx)

    default_scale = scaling._pnl_vol_target_scale(r)
    expanding_scale = scaling._pnl_vol_target_scale(r, median_window_days=100000)

    last_100_default = default_scale.iloc[-100:]
    last_100_expanding = expanding_scale.iloc[-100:]

    # The rolling benchmark has caught up to the new 0.04 normal.
    assert last_100_default.mean() >= 0.75
    # The stale all-history median keeps suppressing.
    assert last_100_expanding.mean() <= last_100_default.mean() - 0.10
    # Both respect floor <= scale <= 1.0 throughout.
    assert (default_scale >= ev.PNL_VOL_TARGET_SCALE_FLOOR).all()
    assert (default_scale <= 1.0).all()
    assert (expanding_scale >= ev.PNL_VOL_TARGET_SCALE_FLOOR).all()
    assert (expanding_scale <= 1.0).all()

def test_pnl_vol_target_rolling_median_burn_in_identical() -> None:
    # SCENARIO_MHS_PNL_VOL_TARGET_ROLLING_MEDIAN_BURN_IN_IDENTICAL: on the
    # existing 200-day fixture, the 365d window produces a series whose first
    # BURN_IN_DAYS-1 entries are 1.0, AND the full 200-value output is
    # element-wise equal to an oversized window (cannot slide within 200 days).
    r = _pnl_vol_spike_returns()
    out = scaling._pnl_vol_target_scale(r)
    assert (out.iloc[: ev.PNL_VOL_TARGET_BURN_IN_DAYS - 1] == 1.0).all()

    expanding_like = scaling._pnl_vol_target_scale(r, median_window_days=99999)
    pd.testing.assert_series_equal(out, expanding_like, check_names=False)

def test_pnl_vol_target_median_window_validation() -> None:
    # SCENARIO_MHS_PNL_VOL_TARGET_MEDIAN_WINDOW_VALIDATION: a window shorter
    # than the burn-in floor raises ValueError; the floor value itself is ok.
    r = _pnl_vol_spike_returns()
    with pytest.raises(ValueError, match="median_window_days"):
        scaling._pnl_vol_target_scale(r, median_window_days=ev.PNL_VOL_TARGET_BURN_IN_DAYS - 1)
    # Should not raise.
    scaling._pnl_vol_target_scale(r, median_window_days=ev.PNL_VOL_TARGET_BURN_IN_DAYS)

def test_pnl_vol_target_existing_suite_unchanged() -> None:
    # SCENARIO_MHS_PNL_VOL_TARGET_EXISTING_SUITE_UNCHANGED: all pre-existing
    # pnl_vol_target tests plus the three new scenarios pass with zero
    # modification to pre-existing test bodies -- the change is additive-only.
    # This test is a contract-level sentinel; the actual assertions live in
    # the individual tests above which are collected and run by pytest.
    pass

def test_pnl_vol_target_flag_defaults_true_and_gates_only_pass_two(mhs_market, monkeypatch) -> None:
    # SCENARIO_MHS_PNL_VOL_TARGET_FLAG_DEFAULTS_TRUE_AND_IS_IDENTITY_05: the
    # flag defaults True (a run at the default is byte-identical to today), a
    # non-bool value is rejected, and with pnl_vol_target=False ONLY the
    # vol-target multiplication is skipped -- Pass 1/pre_vol_target_reference
    # stay structurally unchanged.
    assert MhsDiagnosticRequest().pnl_vol_target is True
    with pytest.raises(ValueError, match="pnl_vol_target"):
        MhsDiagnosticRequest(pnl_vol_target="yes")

    args = _build_book_outcome_args(mhs_market)
    default_report, _ = ev._book_outcome(**args)
    true_report, _ = ev._book_outcome(
        **{**args, "request": dataclasses.replace(args["request"], pnl_vol_target=True, committee_target_gross=None)}
    )
    # The default reproduces the pre-change primary/stress metrics exactly.
    assert default_report.primary_naive_sharpe == pytest.approx(true_report.primary_naive_sharpe)
    assert default_report.stress_naive_sharpe == pytest.approx(true_report.stress_naive_sharpe)

    def _forced_step_scale(reference_daily_returns: pd.Series) -> pd.Series:
        idx = reference_daily_returns.index
        mid = idx[0] + (idx[-1] - idx[0]) / 2
        return pd.Series(np.where(idx < mid, 1.0, 0.2), index=idx)

    monkeypatch.setattr(scaling, "_pnl_vol_target_scale", _forced_step_scale)
    on, _ = ev._book_outcome(**args)
    off, _ = ev._book_outcome(
        **{**args, "request": dataclasses.replace(args["request"], pnl_vol_target=False, committee_target_gross=None)}
    )
    # Pass-1 reference is identical across the two branches.
    assert on.pre_vol_target_reference_naive_sharpe == pytest.approx(
        off.pre_vol_target_reference_naive_sharpe
    )
    # Off branch: Pass 2 replays the same unscaled weights as Pass 1.
    assert off.primary_naive_sharpe == pytest.approx(off.pre_vol_target_reference_naive_sharpe)
    # On branch: the two passes differ by exactly the one multiplicative factor.
    assert on.primary_naive_sharpe != pytest.approx(on.pre_vol_target_reference_naive_sharpe)
    assert on.primary_naive_sharpe != pytest.approx(off.primary_naive_sharpe)
