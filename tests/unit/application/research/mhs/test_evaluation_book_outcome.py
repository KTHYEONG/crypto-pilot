"""MHS evaluation contract tests (split by behavioral domain; shared builders live in the original module)."""

"""Contract coverage for the MHS application evaluation resource telemetry."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
from src.application.research.mhs import evaluation as ev
import src.application.research.mhs.scaling as scaling

from tests.unit.application.research.mhs.test_evaluation import (  # noqa: F401
    _START,
    _assert_books_equal,
    _build_book_outcome_args,
    _build_books_concurrent_args,
)

def test_book_outcome_is_two_pass(mhs_market, monkeypatch) -> None:
    # SCENARIO_BOOK_OUTCOME_IS_TWO_PASS: the reported primary is the
    # P&L-vol-target-rescaled Pass-2 replay, with Pass 1 kept as the
    # pre_vol_target_reference diagnostic. The fields are populated on the
    # natural fixture, and an engineered non-trivial scale proves Pass 2
    # genuinely re-ran with different weights (the 8th-iteration no-op class).
    args = _build_book_outcome_args(mhs_market)
    report, _ = ev._book_outcome(**args)
    assert report.pre_vol_target_reference is not None
    assert report.pre_vol_target_reference_naive_sharpe is not None
    assert report.primary is not None
    assert report.pre_vol_target_reference.fill_source == report.primary.fill_source

    def _forced_step_scale(reference_daily_returns: pd.Series) -> pd.Series:
        idx = reference_daily_returns.index
        mid = idx[0] + (idx[-1] - idx[0]) / 2
        return pd.Series(np.where(idx < mid, 1.0, 0.2), index=idx)

    monkeypatch.setattr(scaling, "_pnl_vol_target_scale", _forced_step_scale)
    forced, _ = ev._book_outcome(**args)
    assert forced.pre_vol_target_reference is not None
    assert forced.pre_vol_target_reference_naive_sharpe is not None
    assert forced.primary_naive_sharpe != forced.pre_vol_target_reference_naive_sharpe

def test_book_outcome_realized_cost_reaches_report(mhs_market) -> None:
    # SCENARIO_MHS_REALIZED_EXECUTION_COST_REACHES_REPORT_04: _book_outcome
    # projects the already-computed per-fill shortfall aggregates onto
    # MhsBookReport (they were previously discarded); the stress spec triples
    # fees, so the realized stress shortfall must be strictly higher than the
    # primary's.
    args = _build_book_outcome_args(mhs_market)
    report, _ = ev._book_outcome(**args)
    assert report.failure is None
    assert report.primary is not None
    assert report.stress is not None
    for field in (
        "primary_realized_shortfall_bps",
        "primary_notional_weighted_shortfall_bps",
        "stress_realized_shortfall_bps",
        "stress_notional_weighted_shortfall_bps",
        "primary_forced_exit_notional",
    ):
        value = getattr(report, field)
        assert value is not None
        assert np.isfinite(value)
    assert report.primary_fill_count is not None
    assert report.primary_unfilled_count is not None
    assert report.stress_realized_shortfall_bps > report.primary_realized_shortfall_bps
    assert report.stress_notional_weighted_shortfall_bps > report.primary_notional_weighted_shortfall_bps

def test_toplevel_blend_replay_matches_renormalized_components(mhs_market) -> None:
    # SCENARIO_MHS_TOPLEVEL_BLEND_REPLAY_MATCHES_RENORMALIZED_COMPONENTS: the
    # blend replay target is the weighted sum of the renormalized execution
    # books (each ffilled onto the 1h grid then reindexed onto the blend's
    # active execution grid), no longer a collapse of the pre-mask theoretical
    # blend.
    args = _build_books_concurrent_args(mhs_market, universe_size=8)
    grid_1h = args["grid_1h"]
    active_spec, active_grid = ev._active_blend_book_and_grid(
        args["fast"], args["slow"], args["fast_grid"], args["slow_grid"],
    )
    expected = (
        ev.BOOK_BLEND_WEIGHTS["fast_reversal"] * args["w_fast_execution"].reindex(grid_1h).ffill().fillna(0.0)
        + ev.BOOK_BLEND_WEIGHTS["slow_momentum"] * args["w_slow_execution"].reindex(grid_1h).ffill().fillna(0.0)
    ).reindex(active_grid)
    collapsed = args["blend_1h"].where(args["execution_mask"], other=0.0).reindex(active_grid)
    assert not expected.equals(collapsed), "renormalized blend must differ from the collapsed pre-mask blend"
    # the concurrent production path replays exactly the renormalized composition
    _, _, blend_report, _, _ = ev._run_books_concurrent(**args)
    expected_report, _ = ev._book_outcome(
        "blend", active_spec, args["n_symbols"], active_grid,
        args["blend_1h"].reindex(active_grid), grid_1h,
        args["opens"], args["bar_funding"], args["phase_blend"], args["root"],
        args["request"], args["funding_by_symbol"], args["start"], args["end"],
        168, args["initial_equity"], expected,
    )
    _assert_books_equal(expected_report, blend_report, "blend")

def test_active_blend_grid_slow_only() -> None:
    # SCENARIO_MHS_ACTIVE_BLEND_GRID_SLOW_ONLY_01: with the frozen
    # BOOK_BLEND_WEIGHTS == {fast_reversal: 0.0, slow_momentum: 1.0},
    # the blend adopts slow's own BookSpec and 24h-native grid by identity (not
    # equality) -- never fast's 6h grid.
    fast = ev.BOOK_SPECS["fast_reversal"]
    slow = ev.BOOK_SPECS["slow_momentum"]
    fast_grid = pd.date_range(_START, periods=4, freq="6h", tz="UTC")
    slow_grid = pd.date_range(_START, periods=1, freq="24h", tz="UTC")
    spec, grid = ev._active_blend_book_and_grid(fast, slow, fast_grid, slow_grid)
    assert spec is slow
    assert grid is slow_grid

def test_active_blend_grid_fast_weighted(monkeypatch) -> None:
    # SCENARIO_MHS_ACTIVE_BLEND_GRID_FAST_WEIGHTED_02: with a nonzero fast
    # weight (historical 50/50), the helper returns fast/fast_grid by identity,
    # reproducing the pre-fix behavior byte-for-byte when fast is re-admitted.
    monkeypatch.setattr(
        ev, "BOOK_BLEND_WEIGHTS",
        {"fast_reversal": 0.5, "slow_momentum": 0.5},
    )
    fast = ev.BOOK_SPECS["fast_reversal"]
    slow = ev.BOOK_SPECS["slow_momentum"]
    fast_grid = pd.date_range(_START, periods=4, freq="6h", tz="UTC")
    slow_grid = pd.date_range(_START, periods=1, freq="24h", tz="UTC")
    spec, grid = ev._active_blend_book_and_grid(fast, slow, fast_grid, slow_grid)
    assert spec is fast
    assert grid is fast_grid

def test_active_blend_grid_no_weight_fails_closed(monkeypatch) -> None:
    # SCENARIO_MHS_ACTIVE_BLEND_GRID_NO_WEIGHT_03: with zero weight on both
    # books the allocation invariant is violated and the helper must fail
    # closed (ValueError) rather than silently pick a default grid.
    monkeypatch.setattr(
        ev, "BOOK_BLEND_WEIGHTS",
        {"fast_reversal": 0.0, "slow_momentum": 0.0},
    )
    fast = ev.BOOK_SPECS["fast_reversal"]
    slow = ev.BOOK_SPECS["slow_momentum"]
    fast_grid = pd.date_range(_START, periods=4, freq="6h", tz="UTC")
    slow_grid = pd.date_range(_START, periods=1, freq="24h", tz="UTC")
    with pytest.raises(ValueError, match="allocates no capital"):
        ev._active_blend_book_and_grid(fast, slow, fast_grid, slow_grid)

def test_blend_report_adopts_slow_cadence(mhs_market) -> None:
    # SCENARIO_MHS_BLEND_REPORT_ADOPTS_SLOW_CADENCE_04: under the fixture with
    # the current frozen weights, the blend MhsBookReport produced by
    # _run_books_concurrent has step_hours==24 and horizon_hours==168
    # (slow_momentum's values), not step_hours==6/horizon_hours==48
    # (fast_reversal's) -- proving the _run_books_concurrent call site was
    # rewired, not just the helper added in isolation.
    args = _build_books_concurrent_args(mhs_market)
    _, _, blend_report, _, _ = ev._run_books_concurrent(**args)
    assert blend_report.failure is None
    assert blend_report.step_hours == 24
    assert blend_report.horizon_hours == 168

def test_book_outcome_executed_prescreen_reaches_report(mhs_market) -> None:
    """SCENARIO_MHS_EXECUTED_PRESCREEN_REACHES_REPORT_04: when ``_book_outcome``
    is handed an execution book (``replay_weights_step``) different from the
    reference book (``weights_step``), the report carries BOTH the reference
    prescreen and an executed_prescreen/executed_tail computed from the
    capital-carrying book, and their net_t values differ. With no execution
    book the executed fields mirror None. Fails against the pre-change code,
    which could only ever report the reference book."""
    args = _build_book_outcome_args(mhs_market)
    report, _ = ev._book_outcome(**args)
    assert report.prescreen is not None
    assert report.executed_prescreen is not None
    assert report.executed_tail is not None
    base_bps = ev.MEASURED_EXECUTION_COST_TIERS_BPS["base"]
    assert report.executed_prescreen_net_t == report.executed_prescreen[base_bps].net_t
    assert report.prescreen[base_bps].net_t != report.executed_prescreen[base_bps].net_t

    reference_only, _ = ev._book_outcome(**{**args, "replay_weights_step": None})
    assert reference_only.executed_prescreen is None
    assert reference_only.executed_tail is None
    assert reference_only.executed_prescreen_net_t is None

def test_book_outcome_existing_primary_metrics_unchanged(mhs_market) -> None:
    """SCENARIO_MHS_EXISTING_PRIMARY_METRICS_UNCHANGED_05: the executed-evidence
    addition is additive-only -- the primary/stress replay metrics stay present
    and finite, and the reference prescreen/tail remain bit-identical to the
    pre-change inline construction (the regression invariant)."""
    from src.mhs.evidence import cost_response_curve, tail_sensitivity_curve

    args = _build_book_outcome_args(mhs_market)
    report, _ = ev._book_outcome(**args)
    assert report.primary is not None
    assert report.stress is not None
    assert report.failure is None
    for field in (
        "primary_autocorr_sharpe", "primary_naive_sharpe", "primary_net_ann",
        "primary_geometric_cagr", "primary_max_drawdown",
        "primary_annualized_turnover", "stress_naive_sharpe",
    ):
        value = getattr(report, field)
        assert value is not None
        assert np.isfinite(value)

    weights_1h = args["weights_step"].reindex(args["grid_1h"]).ffill().fillna(0.0)
    cost_grid = tuple(dict.fromkeys((0.0, 2.0, 4.0, 8.0, *ev.required_cost_tiers())))
    expected_prescreen = cost_response_curve(
        weights_1h, args["opens"], args["bar_funding"], cost_grid, ev._PERIODS_PER_YEAR_1H,
    )
    _net, expected_turnover = ev.mhs_ledger_pnl(
        weights_1h, args["opens"], args["bar_funding"], 8.0,
    )
    expected_tail = tail_sensitivity_curve(
        weights_1h.shift(2).fillna(0.0), args["opens"].pct_change(),
        expected_turnover, 8.0, ev._PERIODS_PER_YEAR_1H, args["event_window_bars"],
    )
    assert report.prescreen == expected_prescreen
    assert report.tail == expected_tail


def test_book_outcome_blend_exposes_exposure_scale_series_constant_risk(mhs_market) -> None:
    # SCENARIO_MHS_BLEND_REPORT_EXPOSES_EXPOSURE_SCALE_SERIES: under
    # constant_risk the two-pass blend book carries the pnl_vol_target_scale it
    # already computed and applied on MhsBookReport.exposure_scale (read-only
    # reuse for folds, no new numeric logic), while a non-blend book leaves
    # the field at its None default.
    args = _build_book_outcome_args(mhs_market)
    request = dataclasses.replace(
        args["request"], pnl_vol_target_mode="constant_risk", committee_capital=True,
    )
    report, _ = ev._book_outcome(**{**args, "name": "blend", "request": request})
    assert report.failure is None
    assert report.pre_vol_target_reference is not None
    assert isinstance(report.exposure_scale, pd.Series)
    reference_daily_returns = (
        report.pre_vol_target_reference.ledger.equity.resample("1D").last().pct_change()
    )
    pd.testing.assert_index_equal(report.exposure_scale.index, reference_daily_returns.index)
    expected = scaling._replay_exposure_scale(reference_daily_returns, request)
    np.testing.assert_allclose(report.exposure_scale.to_numpy(), expected.to_numpy())

    non_blend, _ = ev._book_outcome(**{**args, "request": request})
    assert non_blend.failure is None
    assert non_blend.name != "blend"
    assert non_blend.exposure_scale is None
