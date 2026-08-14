"""Unit tests for the wealth-committee primitives.

Covers the sign-safe cost accounting, long-only equal-risk weighting, wealth
metrics, volatility-target scaling, and the purged walk-forward harness that
benchmarks any future learned combiner against the curated committee
(docs/specs/mhs_committee_design_and_wealth_objective.md §0-§4).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import src.mhs.committee as committee
from src.mhs.committee import (
    decompose_cost,
    long_only_equal_risk_weights,
    purged_walk_forward,
    score_weighted_net,
    volatility_target_scale,
    wealth_metrics,
)
from src.mhs.contracts import MHS_COMMITTEE_MEMBERS
from src.mhs.features import MHS_FEATURE_REGISTRY

_PPY = 365.0 * 24.0


def _hourly_index(n: int = 500) -> pd.DatetimeIndex:
    return pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")


def test_decompose_cost_recovers_gross_and_cost() -> None:
    # SCENARIO_DECOMPOSE_COST_RECOVERS_GROSS_AND_COST: net_low = gross - tc*bps_low
    # and net_high = gross - tc*bps_high decompose back to the original gross and
    # tc within 1e-12. A negative recovered cost is clipped to 0.0. bps_high <=
    # bps_low, a negative bps, and mismatched panels each raise ValueError.
    idx = _hourly_index()
    rng = np.random.default_rng(0)
    gross = pd.DataFrame(rng.normal(0.0, 0.01, (len(idx), 2)), index=idx, columns=["A", "B"])
    tc = pd.DataFrame(rng.uniform(0.01, 0.05, (len(idx), 2)), index=idx, columns=["A", "B"])
    bps_low, bps_high = 2.0, 6.0
    net_low = gross - tc * bps_low
    net_high = gross - tc * bps_high
    recovered_gross, recovered_tc = decompose_cost(net_low, net_high, bps_low, bps_high)
    pd.testing.assert_frame_equal(recovered_gross, gross, check_dtype=False)
    pd.testing.assert_frame_equal(recovered_tc, tc, check_dtype=False)

    # A recovered cost that would be negative is clipped to 0.0 and gross is
    # then the low-tier panel itself (net_high > net_low everywhere implies
    # tc < 0, which a cost can never be).
    net_a = gross.copy()
    net_b = gross + 5.0
    g, t = decompose_cost(net_a, net_b, 2.0, 6.0)
    assert (t == 0.0).all().all()
    pd.testing.assert_frame_equal(g, net_a, check_dtype=False)

    with pytest.raises(ValueError, match="bps_high"):
        decompose_cost(net_low, net_high, 6.0, 2.0)
    with pytest.raises(ValueError, match="non-negative"):
        decompose_cost(net_low, net_high, -1.0, 6.0)
    with pytest.raises(ValueError, match="identically indexed"):
        decompose_cost(net_low.iloc[1:], net_high, bps_low, bps_high)
    with pytest.raises(ValueError, match="identically indexed"):
        decompose_cost(net_low.rename(columns={"A": "X"}), net_high, bps_low, bps_high)


def test_score_weighted_net_cost_drags_both_signs() -> None:
    # SCENARIO_SCORE_WEIGHTED_NET_COST_DRAGS_BOTH_SIGNS: with a positive
    # turnover_cost the cost term is SUBTRACTED for both weight +1 and weight -1
    # (the -1 case equals -gross - cost, never -gross + cost) -- the regression
    # guarding the v1 analysis bug. Column mismatch, mismatched gross/
    # turnover_cost shapes, and cost_bps < 0 raise ValueError.
    idx = _hourly_index()
    gross = pd.DataFrame({"A": np.full(len(idx), 0.01)}, index=idx)
    tc = pd.DataFrame({"A": np.full(len(idx), 0.02)}, index=idx)
    cost_bps = 5.0
    w_pos = pd.Series({"A": 1.0})
    w_neg = pd.Series({"A": -1.0})
    pos = score_weighted_net(w_pos, gross, tc, cost_bps)
    neg = score_weighted_net(w_neg, gross, tc, cost_bps)
    pd.testing.assert_series_equal(pos, gross["A"] - tc["A"] * cost_bps, check_names=False)
    pd.testing.assert_series_equal(neg, -gross["A"] - tc["A"] * cost_bps, check_names=False)

    with pytest.raises(ValueError, match="columns"):
        score_weighted_net(pd.Series({"X": 1.0}), gross, tc, cost_bps)
    with pytest.raises(ValueError, match="identically indexed"):
        score_weighted_net(w_pos, gross.iloc[1:], tc, cost_bps)
    with pytest.raises(ValueError, match="identically indexed"):
        score_weighted_net(w_pos, gross, tc.rename(columns={"A": "X"}), cost_bps)
    with pytest.raises(ValueError, match="cost_bps"):
        score_weighted_net(w_pos, gross, tc, -1.0)


def test_long_only_equal_risk_weights_are_nonnegative_and_normalized() -> None:
    # SCENARIO_LONG_ONLY_EQUAL_RISK_WEIGHTS_ARE_NONNEGATIVE_AND_NORMALIZED:
    # weights are all >= 0 and sum to 1; a double-volatility column receives
    # half the weight; a zero-variance or all-NaN column receives exactly 0.0
    # without raising, while an empty or all-degenerate panel raises ValueError.
    idx = _hourly_index()
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, len(idx))
    train = pd.DataFrame(
        {
            "low": 0.01 * x,
            "high": 0.02 * x,
        },
        index=idx,
    )
    w = long_only_equal_risk_weights(train)
    assert (w >= 0.0).all()
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert w["low"] == pytest.approx(2.0 * w["high"], rel=1e-9)

    train2 = train.copy()
    train2["flat"] = 1.0
    train2["nan_col"] = np.nan
    w2 = long_only_equal_risk_weights(train2)
    assert w2["flat"] == pytest.approx(0.0)
    assert w2["nan_col"] == pytest.approx(0.0)
    assert w2[["low", "high"]].sum() == pytest.approx(1.0, abs=1e-12)

    with pytest.raises(ValueError, match="must not be empty"):
        long_only_equal_risk_weights(pd.DataFrame())
    with pytest.raises(ValueError, match="positive finite volatility"):
        long_only_equal_risk_weights(pd.DataFrame({"x": np.ones(len(idx))}, index=idx))
    with pytest.raises(ValueError, match="positive finite volatility"):
        long_only_equal_risk_weights(pd.DataFrame({"x": np.nan}, index=idx[:1]))


def test_wealth_metrics_compound_correctly() -> None:
    # SCENARIO_WEALTH_METRICS_COMPOUND_CORRECTLY: a constant positive per-bar
    # return compounds to the analytically expected cagr and yields mdd == 0.0;
    # a series that rises then halves yields mdd near -0.5; a return of -1.0 is
    # floored so logret stays finite; an empty series yields nan metrics without
    # raising; periods_per_year <= 0 raises ValueError.
    n = 365 * 24
    r = pd.Series(np.full(n, 0.0001), index=_hourly_index(n))
    m = wealth_metrics(r, _PPY)
    expected_cagr = (1.0001 ** n) ** (_PPY / n) - 1.0
    assert m["cagr"] == pytest.approx(expected_cagr, rel=1e-9)
    assert m["mdd"] == pytest.approx(0.0)
    assert m["logret"] == pytest.approx(n * math.log1p(0.0001), rel=1e-9)

    r2 = pd.Series([0.0] * n + [1.0] + [0.0] * n + [-0.5] + [0.0] * n)
    m2 = wealth_metrics(r2, _PPY)
    assert m2["mdd"] == pytest.approx(-0.5, abs=1e-9)

    r3 = pd.Series([-1.0, 0.0, 0.5])
    m3 = wealth_metrics(r3, _PPY)
    assert np.isfinite(m3["logret"])
    assert m3["logret"] == pytest.approx(
        math.log1p(-0.99) + math.log1p(0.0) + math.log1p(0.5), rel=1e-9
    )
    # cagr uses the unfloored -1.0, so equity reaches zero
    assert m3["cagr"] == pytest.approx(-1.0)

    for empty in (pd.Series(dtype=float), pd.Series([np.nan, np.nan])):
        me = wealth_metrics(empty, _PPY)
        assert np.isnan(me["cagr"])
        assert np.isnan(me["mdd"])
        assert np.isnan(me["logret"])
        assert np.isnan(me["sharpe"])

    with pytest.raises(ValueError, match="periods_per_year"):
        wealth_metrics(pd.Series([0.0]), 0.0)


def test_volatility_target_scale_uses_train_only() -> None:
    # SCENARIO_VOLATILITY_TARGET_SCALE_USES_TRAIN_ONLY: scaling train returns by
    # the returned factor produces annualized volatility equal to target_vol
    # within 1e-9. A zero-variance, single-observation, or all-NaN train series
    # returns exactly 0.0 (fail closed, never inf). target_vol <= 0 and
    # periods_per_year <= 0 raise ValueError.
    rng = np.random.default_rng(0)
    train = pd.Series(rng.normal(0.0, 0.01, 1000))
    target = 0.15
    scale = volatility_target_scale(train, target, _PPY)
    ann_vol = float((train * scale).std(ddof=1) * math.sqrt(_PPY))
    assert ann_vol == pytest.approx(target, rel=1e-9)

    assert volatility_target_scale(pd.Series(np.ones(100)), target) == pytest.approx(0.0)
    assert volatility_target_scale(pd.Series([np.nan] * 100), target) == pytest.approx(0.0)
    assert volatility_target_scale(pd.Series([1.0]), target) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="target_vol"):
        volatility_target_scale(train, 0.0)
    with pytest.raises(ValueError, match="periods_per_year"):
        volatility_target_scale(train, target, 0.0)


def _walk_forward_panels(seed: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    idx = pd.date_range("2022-01-01", periods=4 * 24 * 90, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    gross = pd.DataFrame(
        {
            "A": rng.normal(0.0, 0.01, len(idx)),
            "B": rng.normal(0.0, 0.015, len(idx)),
        },
        index=idx,
    )
    tc = pd.DataFrame({"A": 0.01, "B": 0.02}, index=idx)
    return gross, tc, idx


def test_purged_walk_forward_excludes_purge_window(monkeypatch) -> None:
    # SCENARIO_PURGED_WALK_FORWARD_EXCLUDES_PURGE_WINDOW: for each test block
    # starting at t0 the fitted weights are derived only from bars strictly
    # before t0 - purge; blocks whose training window has fewer than
    # min_train_bars bars, and blocks with no test bars, are skipped rather than
    # raising. The returned series covers only test-block timestamps, is sorted,
    # and has no duplicate index entries.
    gross, tc, idx = _walk_forward_panels()
    purge = pd.Timedelta(hours=336)
    edges = [
        pd.Timestamp("2022-02-01", tz="UTC"),
        pd.Timestamp("2022-07-01", tz="UTC"),
        pd.Timestamp("2022-10-01", tz="UTC"),
    ]
    min_train_bars = 500

    captured: list[pd.DataFrame] = []
    real = committee.long_only_equal_risk_weights

    def spy(train_net: pd.DataFrame) -> pd.Series:
        captured.append(train_net)
        return real(train_net)

    monkeypatch.setattr(committee, "long_only_equal_risk_weights", spy)

    result = purged_walk_forward(
        gross, tc, 4.18, edges, purge, min_train_bars=min_train_bars,
    )
    # The 2022-02-01 block has ~17 days of train (< 500 bars) and is skipped.
    assert len(captured) == 2
    for train_net, edge in zip(captured, edges[1:], strict=True):
        assert bool((train_net.index < (edge - purge)).all())
    assert result.index.is_monotonic_increasing
    assert result.index.is_unique
    assert result.index[0] >= edges[1]

    # A block whose test window is empty is skipped rather than raising.
    beyond = pd.Timestamp("2023-02-01", tz="UTC")
    res2 = purged_walk_forward(
        gross, tc, 4.18, [edges[1], beyond], purge, min_train_bars=min_train_bars,
    )
    assert bool((res2.index < beyond).all())

    with pytest.raises(ValueError, match="block_edges"):
        purged_walk_forward(gross, tc, 4.18, [], purge, min_train_bars=min_train_bars)
    with pytest.raises(ValueError, match="strictly ascending"):
        purged_walk_forward(
            gross, tc, 4.18, [edges[1], edges[0]], purge, min_train_bars=min_train_bars,
        )
    with pytest.raises(ValueError, match="purge"):
        purged_walk_forward(gross, tc, 4.18, edges, pd.Timedelta(0), min_train_bars=min_train_bars)
    with pytest.raises(ValueError, match="cost_bps"):
        purged_walk_forward(gross, tc, -1.0, edges, purge, min_train_bars=min_train_bars)


def test_purged_walk_forward_scales_by_train_vol_only() -> None:
    # SCENARIO_PURGED_WALK_FORWARD_SCALES_BY_TRAIN_VOL_ONLY: scaling a test
    # block's gross and turnover by a constant does not change the fitted
    # weights or the train-derived scale factor -- the scaled output is exactly
    # the constant times the original (proving the scaling never reads
    # test-window data); and a block whose train combined series has zero
    # volatility contributes exactly zero exposure instead of inf. A single
    # block keeps the test segment disjoint from every training window (an
    # expanding train would otherwise feed block 1's test into block 2's train).
    gross, tc, idx = _walk_forward_panels(seed=2)
    purge = pd.Timedelta(hours=336)
    edges = [pd.Timestamp("2022-10-01", tz="UTC")]
    min_train_bars = 100
    result = purged_walk_forward(
        gross, tc, 4.18, edges, purge, min_train_bars=min_train_bars,
    )

    c = 2.5
    gross_scaled = gross.copy()
    tc_scaled = tc.copy()
    test_rows = idx >= edges[0]
    gross_scaled.loc[test_rows, ["A", "B"]] *= c
    tc_scaled.loc[test_rows, ["A", "B"]] *= c
    result_scaled = purged_walk_forward(
        gross_scaled, tc_scaled, 4.18, edges, purge, min_train_bars=min_train_bars,
    )
    pd.testing.assert_series_equal(result_scaled, result * c, check_dtype=False)

    # A train window whose combined series is exactly flat (zero turnover cost
    # plus equal-risk columns that are exact opposites) yields zero exposure --
    # never inf. Exactness matters: with any nonzero cost the per-column stds
    # differ in the last ulp, weights drift off 0.5, and the cancellation
    # leaves a ~1e-17 std that would blow the scale up instead of closing it.
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 0.01, len(idx))
    gross_zero = pd.DataFrame({"A": x, "B": -x}, index=idx)
    tc_zero = pd.DataFrame({"A": 0.0, "B": 0.0}, index=idx)
    res_zero = purged_walk_forward(
        gross_zero, tc_zero, 4.18, edges, purge, min_train_bars=min_train_bars,
    )
    assert len(res_zero) == len(result)
    assert (res_zero == 0.0).all()


def test_committee_members_resolve_in_feature_registry() -> None:
    # SCENARIO_COMMITTEE_MEMBERS_RESOLVE_IN_FEATURE_REGISTRY: every name in
    # MHS_COMMITTEE_MEMBERS resolves to exactly one FeatureSpec in
    # MHS_FEATURE_REGISTRY, the tuple has 6 unique entries, and it spans at
    # least 3 distinct economic families -- the composition invariant that stops
    # the committee from silently collapsing into one family.
    assert len(MHS_COMMITTEE_MEMBERS) == 6
    assert len(set(MHS_COMMITTEE_MEMBERS)) == 6
    registry_names = {spec.name for spec in MHS_FEATURE_REGISTRY}
    assert set(MHS_COMMITTEE_MEMBERS) <= registry_names
    for member in MHS_COMMITTEE_MEMBERS:
        matches = [spec for spec in MHS_FEATURE_REGISTRY if spec.name == member]
        assert len(matches) == 1

    families = set()
    for member in MHS_COMMITTEE_MEMBERS:
        if member.startswith("flow_"):
            families.add("order_flow")
        elif member.startswith(("xs_mom", "xs_idio_mom")):
            families.add("trend")
        elif member.startswith("mom3_"):
            families.add("higher_moment")
        else:
            families.add(member)
    assert len(families) >= 3
