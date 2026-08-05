"""Contract scenarios XABB-01..XABB-04 and XABJS-01..XABJS-02 (plus the frozen
python_assertions) for the XS alpha x baseline blend primitives
(``select_baseline_blend_weight`` / ``build_blended_ledger`` /
``discovery_reliability_score``).

Scenario coverage:
* XABB-01-DISCOVERY-ONLY-SELECTION
* XABB-02-ZERO-VARIANCE-SAFE-DIVISION
* XABB-03-FULL-HISTORY-APPLICATION
* XABB-04-FAIL-CLOSED-MISALIGNED-INDEX
* XABJS-01-OBJECTIVE-RESPONDS-TO-LEVERAGE
* XABJS-02-DISCOVERY-ONLY-WINDOW
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.research.technical_experts.xs_alpha_baseline_blend import (
    apply_fixed_gross_leverage,
    build_blended_ledger,
    discovery_reliability_score,
    select_baseline_blend_weight,
    select_robust_baseline_blend_weight,
)

_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _antiseries(n: int = 25, start: str = "2022-01-01") -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    """Perfectly anti-correlated pair whose 0.5 blend is a constant positive return."""
    idx = pd.date_range(start, periods=2 * n, freq="4h", tz="UTC")
    a = pd.Series([0.02, -0.01] * n, index=idx)
    b = pd.Series([-0.01, 0.02] * n, index=idx)
    return a, b, idx


def test_xabb_python_assertion_selects_zero_variance_blend() -> None:
    # Contract python_assertion: 50-bar anti-correlated pair, grid picks 0.5
    # (the exact blend where the pair collapses to a constant positive return).
    a, b, idx = _antiseries()
    assert select_baseline_blend_weight(a, b, idx[0], idx[-1], _GRID) == 0.5


def test_xabb_python_assertion_build_blended_ledger() -> None:
    # Contract python_assertion: 0.5 blend of the anti-correlated pair compounds
    # 1.005 over all 50 bars, and the weights frame columns are ordered.
    a, b, idx = _antiseries()
    aw = pd.DataFrame({"S": [1.0] * 50}, index=idx)
    bw = pd.Series([1.0] * 50, index=idx)
    equity, weights = build_blended_ledger(a, aw, b, bw, 0.5)
    assert abs(float(equity.iloc[-1]) - 1.005 ** 50) < 1e-9
    assert list(weights.columns) == ["xs_alpha", "baseline"]


def test_xabb_01_discovery_only_selection() -> None:
    # Appending qualification-window data with a different optimal weight to
    # either leg must not change the weight selected on the discovery window.
    disc_start = pd.Timestamp("2022-01-01", tz="UTC")
    disc_end = pd.Timestamp("2022-02-01", tz="UTC")
    idx = pd.date_range(disc_start, disc_end, freq="4h", tz="UTC")
    c = pd.Series(np.resize([0.02, -0.01], len(idx)), index=idx)

    # Identical discovery legs -> every grid blend has the same Sharpe -> the
    # deterministic lowest-weight tie-break selects 0.0.
    a_disc = c.copy()
    b_disc = c.copy()

    # If the appended qualification data were read, the blend would collapse to
    # a constant positive return at w=1.0 (0.0/1.0 anti-correlated constants),
    # so the unrestricted optimum would be 1.0 -- never 0.0.
    qual_idx = pd.date_range(
        disc_end + pd.Timedelta(hours=4), periods=40, freq="4h", tz="UTC",
    )
    a_qual = pd.Series(0.01, index=qual_idx)
    b_qual = pd.Series(-0.01, index=qual_idx)

    a_full = pd.concat([a_disc, a_qual])
    b_full = pd.concat([b_disc, b_qual])

    assert select_baseline_blend_weight(
        a_full, b_full, disc_start, disc_end, _GRID,
    ) == 0.0
    assert select_baseline_blend_weight(
        a_disc, b_disc, disc_start, disc_end, _GRID,
    ) == 0.0


def test_xabb_02_zero_variance_safe_division() -> None:
    # A grid point whose blended series has exactly zero std and a positive
    # mean is selected as +inf Sharpe without raising or emitting a warning.
    a, b, idx = _antiseries()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        weight = select_baseline_blend_weight(a, b, idx[0], idx[-1], _GRID)
    assert weight == 0.5


def test_xabb_03_full_history_application() -> None:
    # One fixed scalar weight is applied uniformly across the entire input
    # history, including bars outside the discovery window used to select it.
    a, b, idx = _antiseries()
    aw = pd.DataFrame({"S": [1.0] * 50, "T": [0.5] * 50}, index=idx)
    bw = pd.Series([1.0] * 50, index=idx)
    equity, weights = build_blended_ledger(a, aw, b, bw, 0.5)

    # 0.5*a + 0.5*b == 0.005 on every bar of the full 50-bar history.
    assert abs(float(equity.iloc[0]) - 1.005) < 1e-9
    assert abs(float(equity.iloc[-1]) - 1.005 ** 50) < 1e-9
    assert list(weights.columns) == ["xs_alpha", "baseline"]
    # xs_alpha = 0.5 * sum(abs([1.0, 0.5])) = 0.75; baseline = 0.5 * abs(1.0).
    assert np.allclose(weights["xs_alpha"].to_numpy(), 0.75)
    assert np.allclose(weights["baseline"].to_numpy(), 0.5)


def test_xabb_04_fail_closed_misaligned_index() -> None:
    # build_blended_ledger raises ValueError (never a silent reindex/fillna)
    # when the inputs do not share an identical DatetimeIndex.
    a, b, idx = _antiseries()
    aw = pd.DataFrame({"S": [1.0] * 50}, index=idx)
    bw = pd.Series([1.0] * 50, index=idx)
    shifted = b.copy()
    shifted.index = idx + pd.Timedelta(hours=4)
    with pytest.raises(ValueError, match="identical DatetimeIndex"):
        build_blended_ledger(a, aw, shifted, bw, 0.5)
    with pytest.raises(ValueError, match="identical DatetimeIndex"):
        build_blended_ledger(a, aw.iloc[:-1], b, bw, 0.5)


def test_xabb_fail_closed_selection_validation() -> None:
    a, b, idx = _antiseries()
    with pytest.raises(ValueError, match="must not be empty"):
        select_baseline_blend_weight(a, b, idx[0], idx[-1], ())
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        select_baseline_blend_weight(a, b, idx[0], idx[-1], (0.0, 1.5))
    bad = a.copy()
    bad.iloc[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        select_baseline_blend_weight(bad, b, idx[0], idx[-1], _GRID)
    with pytest.raises(ValueError, match="fewer than 2"):
        select_baseline_blend_weight(a.iloc[:1], b.iloc[:1], idx[0], idx[-1], _GRID)


def test_xabb_fail_closed_build_validation() -> None:
    a, b, idx = _antiseries()
    aw = pd.DataFrame({"S": [1.0] * 50}, index=idx)
    bw = pd.Series([1.0] * 50, index=idx)
    with pytest.raises(ValueError, match="xs_alpha_weight"):
        build_blended_ledger(a, aw, b, bw, 1.5)
    with pytest.raises(ValueError, match="xs_alpha_weight"):
        build_blended_ledger(a, aw, b, bw, -0.1)
    bad_net = a.copy()
    bad_net.iloc[3] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        build_blended_ledger(bad_net, aw, b, bw, 0.5)
    bad_aw = aw.copy()
    bad_aw.iloc[2, 0] = np.nan
    with pytest.raises(ValueError, match="realized-weight inputs must be finite"):
        build_blended_ledger(a, bad_aw, b, bw, 0.5)
    bad_bw = bw.copy()
    bad_bw.iloc[4] = np.inf
    with pytest.raises(ValueError, match="realized-weight inputs must be finite"):
        build_blended_ledger(a, aw, b, bad_bw, 0.5)

def test_xabrs_python_assertion_selects_worst_year_robust_weight() -> None:
    # Contract python_assertion: the baseline leg is strongly negative in 2022
    # and strongly positive in 2023, so the worst-year-robust selector must
    # refuse it and pick the all-xs_alpha grid point (1.0).
    idx_2022 = pd.date_range("2022-01-01", periods=20, freq="4h", tz="UTC")
    idx_2023 = pd.date_range("2023-01-01", periods=20, freq="4h", tz="UTC")
    idx = idx_2022.append(idx_2023)
    a = pd.Series([0.01, 0.008] * 20, index=idx)
    b = pd.Series(([-0.02, -0.018] * 10) + ([0.05, 0.048] * 10), index=idx)
    assert select_robust_baseline_blend_weight(
        a, b, idx[0], idx[-1], (0.0, 0.25, 0.5, 0.75, 1.0),
    ) == 1.0


def test_xabrs_python_assertion_apply_fixed_gross_leverage() -> None:
    # Contract python_assertion: every bar is exactly 2.0x the input, in both
    # the net returns and the realised weights.
    idx = pd.date_range("2022-01-01", periods=5, freq="4h", tz="UTC")
    net = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=idx)
    w = pd.DataFrame({"xs_alpha": [0.5] * 5, "baseline": [0.3] * 5}, index=idx)
    scaled_net, scaled_w = apply_fixed_gross_leverage(net, w, 2.0)
    assert list(scaled_net.round(6)) == [0.02, -0.04, 0.06, 0.0, 0.02]
    assert float(scaled_w["xs_alpha"].iloc[0]) == 1.0
    assert float(scaled_w["baseline"].iloc[0]) == 0.6


def test_xabrs_01_worst_year_robust_differs_from_aggregate() -> None:
    # XABRS-01: the baseline leg has one great year (2023) and one terrible
    # year (2022); the aggregate selector is fooled into over-weighting it,
    # while the worst-year-robust selector refuses it. The two selectors must
    # not coincide by construction.
    idx_2022 = pd.date_range("2022-01-01", periods=20, freq="4h", tz="UTC")
    idx_2023 = pd.date_range("2023-01-01", periods=20, freq="4h", tz="UTC")
    idx = idx_2022.append(idx_2023)
    a = pd.Series(np.resize([0.005, -0.004], 40), index=idx)
    b = pd.Series(
        np.concatenate([np.full(20, -0.05), np.full(20, 0.08)]), index=idx,
    )
    robust = select_robust_baseline_blend_weight(a, b, idx[0], idx[-1], _GRID)
    aggregate = select_baseline_blend_weight(a, b, idx[0], idx[-1], _GRID)
    assert robust == 1.0
    assert aggregate != robust


def test_xabrs_02_single_year_discovery_fails_closed() -> None:
    # XABRS-02: a discovery window spanning only one calendar year has no
    # meaningful per-year minimum and must raise, never silently return.
    a, b, idx = _antiseries()
    with pytest.raises(ValueError, match="distinct calendar years"):
        select_robust_baseline_blend_weight(a, b, idx[0], idx[-1], _GRID)


def test_xabrs_03_no_ladder_in_application() -> None:
    # XABRS-03: even immediately after a large drawdown bar, the overlay's
    # output is exactly scale times the input -- unlike apply_realised_risk_overlay
    # it never reduces the effective multiplier below scale on any bar.
    idx = pd.date_range("2022-01-01", periods=6, freq="4h", tz="UTC")
    net = pd.Series([0.01, -0.05, 0.02, 0.03, -0.01, 0.005], index=idx)
    w = pd.DataFrame({"xs_alpha": [0.5] * 6, "baseline": [0.3] * 6}, index=idx)
    scaled_net, scaled_w = apply_fixed_gross_leverage(net, w, 2.5)
    assert np.allclose(scaled_net.to_numpy(), 2.5 * net.to_numpy())
    assert np.allclose(scaled_w.to_numpy(), 2.5 * w.to_numpy())


def test_xabrs_fail_closed_robust_selection_validation() -> None:
    a, b, idx = _antiseries()
    with pytest.raises(ValueError, match="must not be empty"):
        select_robust_baseline_blend_weight(a, b, idx[0], idx[-1], ())
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        select_robust_baseline_blend_weight(a, b, idx[0], idx[-1], (0.0, 1.5))
    bad = a.copy()
    bad.iloc[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        select_robust_baseline_blend_weight(bad, b, idx[0], idx[-1], _GRID)


def test_xabrs_fail_closed_leverage_validation() -> None:
    idx = pd.date_range("2022-01-01", periods=5, freq="4h", tz="UTC")
    net = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01], index=idx)
    w = pd.DataFrame({"xs_alpha": [0.5] * 5, "baseline": [0.3] * 5}, index=idx)
    for bad_scale in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError, match="scale must be finite"):
            apply_fixed_gross_leverage(net, w, bad_scale)
    shifted = w.copy()
    shifted.index = idx + pd.Timedelta(hours=4)
    with pytest.raises(ValueError, match="identical index"):
        apply_fixed_gross_leverage(net, shifted, 2.0)
    non_dt = w.copy()
    non_dt.index = pd.RangeIndex(len(w))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        apply_fixed_gross_leverage(net, non_dt, 2.0)
    bad_net = net.copy()
    bad_net.iloc[2] = np.inf
    with pytest.raises(ValueError, match="only finite values"):
        apply_fixed_gross_leverage(bad_net, w, 2.0)
    bad_w = w.copy()
    bad_w.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="only finite values"):
        apply_fixed_gross_leverage(net, bad_w, 2.0)

def test_xabrs_fail_closed_leverage_non_monotonic_index() -> None:
    # A non-monotonic DatetimeIndex must be rejected, mirroring
    # apply_realised_risk_overlay's monotonicity contract.
    idx = pd.DatetimeIndex([
        pd.Timestamp("2022-01-01 08:00", tz="UTC"),
        pd.Timestamp("2022-01-01 00:00", tz="UTC"),
        pd.Timestamp("2022-01-01 04:00", tz="UTC"),
    ])
    net = pd.Series([0.01, 0.02, 0.03], index=idx)
    w = pd.DataFrame(
        {"xs_alpha": [0.5, 0.5, 0.5], "baseline": [0.3, 0.3, 0.3]}, index=idx,
    )
    with pytest.raises(ValueError, match="monotonic"):
        apply_fixed_gross_leverage(net, w, 2.0)

def _joint_discovery_fixture() -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.DatetimeIndex]:
    """Two-year daily fixture whose 0.5 blend has a positive drift.

    The window must span well over the 6-month equal-duration fold guard of
    ``evaluate_xs_reliability`` -- the contract's original 60-bar (10-day)
    window is too short for that composed gate, so the synthetic fixture uses a
    multi-year daily index instead.
    """
    idx = pd.date_range("2022-01-01", periods=730, freq="D", tz="UTC")
    a = pd.Series([0.0005, 0.001] * 365, index=idx)
    b = pd.Series([0.0008, -0.0002] * 365, index=idx)
    aw = pd.DataFrame({"S": [1.0] * 730}, index=idx)
    bw = pd.Series([1.0] * 730, index=idx)
    return a, aw, b, bw, idx


def test_xabjs_01_objective_responds_to_leverage() -> None:
    # XABJS-01: the joint objective must NOT be scale-invariant -- the same
    # xs_alpha_weight at two different leverage_scale values must yield two
    # different LCB90 scores (unlike a Sharpe/t_stat objective, which is blind
    # to pure linear leverage and is exactly why the sequential per-axis
    # searches needed two different objectives).
    a, aw, b, bw, idx = _joint_discovery_fixture()
    s1 = discovery_reliability_score(a, aw, b, bw, idx[0], idx[-1], 0.5, 1.0)
    s2 = discovery_reliability_score(a, aw, b, bw, idx[0], idx[-1], 0.5, 2.0)
    assert isinstance(s1, float)
    assert isinstance(s2, float)
    assert s1 != s2


def test_xabjs_02_discovery_only_window() -> None:
    # XABJS-02: appending qualification-window data with a very different return
    # profile to any of the four input series must not change the score -- the
    # window restriction happens before any of the three composed calls.
    a, aw, b, bw, idx = _joint_discovery_fixture()
    disc_start = idx[0]
    disc_end = idx[365]
    disc_only = discovery_reliability_score(
        a[:366], aw[:366], b[:366], bw[:366], disc_start, disc_end, 0.5, 1.5,
    )

    qual_idx = pd.date_range(
        disc_end + pd.Timedelta(days=1), periods=365, freq="D", tz="UTC",
    )
    a_qual = pd.Series(0.02, index=qual_idx)
    b_qual = pd.Series(-0.01, index=qual_idx)
    aw_qual = pd.DataFrame({"S": [1.0] * 365}, index=qual_idx)
    bw_qual = pd.Series([1.0] * 365, index=qual_idx)
    a_full = pd.concat([a[:366], a_qual])
    aw_full = pd.concat([aw[:366], aw_qual])
    b_full = pd.concat([b[:366], b_qual])
    bw_full = pd.concat([bw[:366], bw_qual])

    with_qualification = discovery_reliability_score(
        a_full, aw_full, b_full, bw_full, disc_start, disc_end, 0.5, 1.5,
    )
    assert disc_only == with_qualification

def test_xabjs_fail_closed_fewer_than_two_discovery_bars() -> None:
    # A discovery window admitting fewer than 2 common bars must fail closed
    # with ValueError, never return a score built on an empty/single-bar set.
    a, aw, b, bw, idx = _joint_discovery_fixture()
    with pytest.raises(ValueError, match="fewer than 2 common bars"):
        discovery_reliability_score(a, aw, b, bw, idx[-1], idx[-1], 0.5, 1.0)
