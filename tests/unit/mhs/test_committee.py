"""Unit tests for the wealth-committee primitives."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

import src.mhs.committee as committee
from src.mhs.committee import (
    committee_block_edges_from,
    decompose_cost,
    kelly_lcb_scale,
    long_only_equal_risk_weights,
    purged_walk_forward,
    score_weighted_net,
    train_evidence_weights,
    volatility_target_scale,
    wealth_metrics,
)
from src.mhs.types import COMMITTEE_MEMBERS, COMMITTEE_OOS_START
from src.mhs.features import FEATURE_REGISTRY

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


def test_kelly_lcb_scale_fails_closed_on_zero_variance() -> None:
    # SCENARIO_KELLY_LCB_SCALE_FAIL_CLOSED_ZERO_VARIANCE: a constant train series
    # has zero variance, so the Kelly optimal leverage f*=mu/var is undefined --
    # kelly_lcb_scale returns exactly 0.0 (fail closed to no exposure), matching
    # volatility_target_scale's zero-variance discipline in the same module.
    assert kelly_lcb_scale(pd.Series([0.001] * 10, index=_hourly_index(10))) == pytest.approx(0.0)
    assert kelly_lcb_scale(pd.Series([np.nan] * 10, index=_hourly_index(10))) == pytest.approx(0.0)
    assert kelly_lcb_scale(pd.Series([0.001], index=_hourly_index(1))) == pytest.approx(0.0)


def test_kelly_lcb_scale_fails_closed_on_negative_lcb_mean() -> None:
    # SCENARIO_KELLY_LCB_SCALE_FAIL_CLOSED_NEGATIVE_LCB_MEAN: when the one-SE
    # lower confidence bound (sample mean - z*std/sqrt(n)) is <= 0 the estimate
    # cannot justify any exposure -- exactly 0.0, never a negative or tiny
    # positive bet on noise.
    # Symmetric +/-0.05 pairs plus a small positive offset: mean == +0.001 but
    # std/sqrt(n) ~ 0.0167, so the one-SE LCB is negative.
    vals = np.concatenate([np.full(5, 0.05), np.full(5, -0.05)]) + 0.001
    train = pd.Series(vals)
    assert train.mean() > 0
    assert train.mean() - train.std(ddof=1) / math.sqrt(len(train)) <= 0
    assert kelly_lcb_scale(train) == pytest.approx(0.0)


def test_kelly_lcb_scale_clips_at_cap() -> None:
    # SCENARIO_KELLY_LCB_SCALE_CLIPS_AT_CAP: a strongly positive mean with a
    # near-zero variance implies a huge Kelly leverage; the result is clipped at
    # cap (default 1.5), never exceeding it.
    base = np.full(200, 0.01)
    noise = np.full(200, 1e-6)
    noise[::2] = -1e-6
    train = pd.Series(base + noise)
    assert train.mean() > 0
    assert kelly_lcb_scale(train) == pytest.approx(1.5)
    assert kelly_lcb_scale(train, cap=2.0) == pytest.approx(2.0)


def test_kelly_lcb_scale_invalid_params_raise() -> None:
    # SCENARIO_KELLY_LCB_SCALE_INVALID_PARAMS_RAISE: fraction <= 0, cap <= 0,
    # and z < 0 each raise ValueError (fail closed on a misconfigured overlay).
    train = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 200))
    with pytest.raises(ValueError, match="fraction"):
        kelly_lcb_scale(train, fraction=0.0)
    with pytest.raises(ValueError, match="fraction"):
        kelly_lcb_scale(train, fraction=-0.25)
    with pytest.raises(ValueError, match="cap"):
        kelly_lcb_scale(train, cap=0.0)
    with pytest.raises(ValueError, match="cap"):
        kelly_lcb_scale(train, cap=-1.0)
    with pytest.raises(ValueError, match="z"):
        kelly_lcb_scale(train, z=-1.0)


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


def test_purged_walk_forward_default_sizing_mode_byte_identical() -> None:
    # SCENARIO_PURGED_WALK_FORWARD_DEFAULT_SIZING_MODE_BYTE_IDENTICAL: omitting
    # sizing_mode produces an identical Series (values and index) to the same
    # call made explicitly with sizing_mode='vol_target' -- the default keeps
    # every pre-existing call site byte-identical.
    gross, tc, idx = _walk_forward_panels()
    purge = pd.Timedelta(hours=336)
    edges = [pd.Timestamp("2022-10-01", tz="UTC")]
    default = purged_walk_forward(gross, tc, 4.18, edges, purge, min_train_bars=100)
    explicit = purged_walk_forward(
        gross, tc, 4.18, edges, purge, min_train_bars=100, sizing_mode="vol_target",
    )
    pd.testing.assert_series_equal(default, explicit)


def test_purged_walk_forward_kelly_blend_changes_output() -> None:
    # SCENARIO_PURGED_WALK_FORWARD_KELLY_BLEND_CHANGES_OUTPUT: on a fixture with
    # nonzero train variance sizing_mode='kelly_blend' produces a Series with
    # the same index as 'vol_target' but at least one differing value -- the
    # 50/50 blend scalar differs from the pure vol-target scalar.
    gross, tc, idx = _walk_forward_panels(seed=2)
    purge = pd.Timedelta(hours=336)
    edges = [pd.Timestamp("2022-10-01", tz="UTC")]
    vol = purged_walk_forward(gross, tc, 4.18, edges, purge, min_train_bars=100)
    kelly = purged_walk_forward(
        gross, tc, 4.18, edges, purge, min_train_bars=100, sizing_mode="kelly_blend",
    )
    assert kelly.index.equals(vol.index)
    assert not kelly.equals(vol)


def test_purged_walk_forward_invalid_sizing_mode_raises() -> None:
    # SCENARIO_PURGED_WALK_FORWARD_INVALID_SIZING_MODE_RAISES: an unknown
    # sizing_mode fails closed with a ValueError naming the parameter.
    gross, tc, idx = _walk_forward_panels()
    purge = pd.Timedelta(hours=336)
    edges = [pd.Timestamp("2022-10-01", tz="UTC")]
    with pytest.raises(ValueError, match="sizing_mode"):
        purged_walk_forward(
            gross, tc, 4.18, edges, purge, min_train_bars=100, sizing_mode="bogus",
        )


def test_committee_members_resolve_in_feature_registry() -> None:
    # SCENARIO_COMMITTEE_MEMBERS_RESOLVE_IN_FEATURE_REGISTRY and
    # SCENARIO_MHS_COMMITTEE_MEMBERS_SPAN_THREE_FAMILIES_AFTER_K5: every name in
    # COMMITTEE_MEMBERS resolves to exactly one FeatureSpec in
    # FEATURE_REGISTRY, the k=5 tuple has 5 unique entries (xs_mom_720h
    # removed as rank-invariant no-op), and it spans at least 3 distinct economic
    # families -- the composition invariant that stops the committee from
    # silently collapsing into one family.
    assert len(COMMITTEE_MEMBERS) == 5
    assert len(set(COMMITTEE_MEMBERS)) == 5
    registry_names = {spec.name for spec in FEATURE_REGISTRY}
    assert set(COMMITTEE_MEMBERS) <= registry_names
    for member in COMMITTEE_MEMBERS:
        matches = [spec for spec in FEATURE_REGISTRY if spec.name == member]
        assert len(matches) == 1

    families = set()
    for member in COMMITTEE_MEMBERS:
        if member.startswith("flow_"):
            families.add("order_flow")
        elif member.startswith(("xs_mom", "xs_idio_mom")):
            families.add("trend")
        elif member.startswith("mom3_"):
            families.add("higher_moment")
        elif member.startswith("lowvol_"):
            families.add("defensive")
        elif member.startswith("rev_"):
            families.add("reversal")
        else:
            families.add(member)
    assert len(families) >= 3

def test_committee_block_edges_from_anchored_at_oos_start() -> None:
    # SCENARIO_COMMITTEE_BLOCK_EDGES_ANCHORED_AT_OOS_START (B1): the walk-forward
    # block grid must be anchored at max(start, oos_start), never the raw
    # diagnostic start, so a purged walk-forward can no longer score pre-OOS
    # blocks as pseudo-OOS. Edges start at COMMITTEE_OOS_START (2023-01-01)
    # in 6-month steps and never earlier; a diagnostic whose own start is
    # already after oos_start is unaffected (max() semantics); end <=
    # max(start, oos_start) raises ValueError.
    edges = committee_block_edges_from(
        pd.Timestamp("2021-01-01", tz="UTC"),
        COMMITTEE_OOS_START,
        pd.Timestamp("2025-12-31", tz="UTC"),
    )
    assert edges[0] == COMMITTEE_OOS_START
    assert edges[0] == pd.Timestamp("2023-01-01", tz="UTC")
    for prev, nxt in itertools.pairwise(edges):
        assert nxt == prev + pd.DateOffset(months=6)
    assert all(e >= COMMITTEE_OOS_START for e in edges)

    late = committee_block_edges_from(
        pd.Timestamp("2024-06-01", tz="UTC"),
        COMMITTEE_OOS_START,
        pd.Timestamp("2025-12-31", tz="UTC"),
    )
    assert late[0] == pd.Timestamp("2024-06-01", tz="UTC")

    with pytest.raises(ValueError, match="max\\(start, oos_start\\)"):
        committee_block_edges_from(
            pd.Timestamp("2021-01-01", tz="UTC"),
            COMMITTEE_OOS_START,
            pd.Timestamp("2022-01-01", tz="UTC"),
        )


# ---------------------------------------------------------------------------
# train_evidence_weights tests
# ---------------------------------------------------------------------------


def test_evidence_weights_favor_stronger_member() -> None:
    rng = np.random.default_rng(42)
    n = 200
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    train_mask = pd.Series(True, index=idx)

    strong = pd.Series(rng.normal(0.01, 0.01, n), index=idx)
    weak = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
    noise = pd.Series(rng.normal(0.0, 0.01, n), index=idx)

    weights = train_evidence_weights(
        {"strong": strong, "weak": weak, "noise": noise}, train_mask,
    )
    assert weights["strong"] > weights["weak"] > weights["noise"]
    assert all(w >= 0.0 for w in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)


def test_evidence_weights_fail_closed_to_equal() -> None:
    rng = np.random.default_rng(7)
    n = 200
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    train_mask = pd.Series(True, index=idx)

    neg1 = pd.Series(rng.uniform(-0.01, -0.005, n), index=idx)
    neg2 = pd.Series(rng.uniform(-0.02, -0.01, n), index=idx)
    neg3 = pd.Series(rng.uniform(-0.03, -0.02, n), index=idx)
    w = train_evidence_weights({"a": neg1, "b": neg2, "c": neg3}, train_mask)
    assert w["a"] == pytest.approx(1 / 3, abs=1e-12)
    assert w["b"] == pytest.approx(1 / 3, abs=1e-12)
    assert w["c"] == pytest.approx(1 / 3, abs=1e-12)

    short_mask = pd.Series([True] * 5 + [False] * (n - 5), index=idx)
    w2 = train_evidence_weights(
        {"a": pd.Series(0.001, index=idx), "b": pd.Series(0.002, index=idx)},
        short_mask,
        min_train_rows=30,
    )
    assert w2["a"] == pytest.approx(0.5, abs=1e-12)
    assert w2["b"] == pytest.approx(0.5, abs=1e-12)


def test_evidence_weights_are_causal() -> None:
    rng = np.random.default_rng(99)
    n = 200
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    train_end = idx[150]
    train_mask = pd.Series(idx < train_end, index=idx)

    a = pd.Series(rng.normal(0.01, 0.01, n), index=idx)
    b = pd.Series(rng.normal(0.005, 0.01, n), index=idx)
    original = train_evidence_weights({"a": a, "b": b}, train_mask)

    a_polluted = a.copy()
    a_polluted.iloc[150:] = 1e6
    a_polluted.iloc[160] = float("nan")
    b_polluted = b.copy()
    b_polluted.iloc[150:] = -1e6
    polluted = train_evidence_weights({"a": a_polluted, "b": b_polluted}, train_mask)
    assert original == polluted


def test_evidence_weights_reject_invalid_input() -> None:
    idx = pd.date_range("2022-01-01", periods=100, freq="1h", tz="UTC")
    mask = pd.Series(True, index=idx)

    with pytest.raises(ValueError, match="must not be empty"):
        train_evidence_weights({}, mask)
    with pytest.raises(ValueError, match="min_train_rows"):
        train_evidence_weights({"a": pd.Series(1.0, index=idx)}, mask, min_train_rows=0)

    nan_series = pd.Series(float("nan"), index=idx)
    flat_series = pd.Series(1.0, index=idx)
    neg_series = pd.Series(-0.01, index=idx)
    w = train_evidence_weights(
        {"nan_col": nan_series, "flat": flat_series, "neg": neg_series}, mask, min_train_rows=30,
    )
    assert w["nan_col"] == pytest.approx(1 / 3, abs=1e-12)
    assert w["flat"] == pytest.approx(1 / 3, abs=1e-12)
    assert w["neg"] == pytest.approx(1 / 3, abs=1e-12)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# growth_budget_annual_vol tests
# ---------------------------------------------------------------------------


def test_growth_budget_annual_vol_returns_float_in_range() -> None:
    # SCENARIO_MHS_COMPOUNDING_ALPHA_AXES_03: growth_budget_annual_vol on a
    # 500-bar positive-drift series with std 0.01 returns a float in [0.05, 1.0].
    from src.mhs.committee import growth_budget_annual_vol

    rng = np.random.default_rng(42)
    idx = _hourly_index(500)
    returns = pd.Series(rng.normal(0.0001, 0.01, 500), index=idx)
    result = growth_budget_annual_vol(returns)
    assert isinstance(result, float)
    assert 0.05 <= result <= 1.0


def test_growth_budget_annual_vol_fallback_on_empty() -> None:
    # SCENARIO_MHS_COMPOUNDING_ALPHA_AXES_03: on an empty Series, returns
    # PNL_TARGET_ANNUAL_VOL.
    from src.mhs.committee import growth_budget_annual_vol
    from src.mhs.params import PNL_TARGET_ANNUAL_VOL

    assert growth_budget_annual_vol(pd.Series(dtype=float)) == PNL_TARGET_ANNUAL_VOL


def test_growth_budget_annual_vol_fallback_on_one_row() -> None:
    from src.mhs.committee import growth_budget_annual_vol
    from src.mhs.params import PNL_TARGET_ANNUAL_VOL

    idx = _hourly_index(1)
    assert growth_budget_annual_vol(pd.Series([0.001], index=idx)) == PNL_TARGET_ANNUAL_VOL


def test_growth_budget_annual_vol_fallback_on_zero_std() -> None:
    # SCENARIO_MHS_COMPOUNDING_ALPHA_AXES_03: an all-zero (std==0) Series
    # returns exactly PNL_TARGET_ANNUAL_VOL.
    from src.mhs.committee import growth_budget_annual_vol
    from src.mhs.params import PNL_TARGET_ANNUAL_VOL

    idx = _hourly_index(500)
    assert growth_budget_annual_vol(pd.Series(0.0, index=idx)) == PNL_TARGET_ANNUAL_VOL


def test_growth_budget_annual_vol_always_finite() -> None:
    from src.mhs.committee import growth_budget_annual_vol

    rng = np.random.default_rng(7)
    idx = _hourly_index(500)
    returns = pd.Series(rng.normal(0.0001, 0.01, 500), index=idx)
    result = growth_budget_annual_vol(returns)
    assert np.isfinite(result)


def test_growth_budget_annual_vol_high_sharpe_returns_larger() -> None:
    # SCENARIO_MHS_COMPOUNDING_ALPHA_AXES_03: A high-Sharpe series
    # (mean/std = 0.15) returns a strictly larger value than the same series
    # scaled to 4x its volatility, proving the budget binds on drawdown risk
    # rather than on raw magnitude.
    from src.mhs.committee import growth_budget_annual_vol

    rng = np.random.default_rng(12)
    idx = _hourly_index(500)
    high_sharpe = pd.Series(rng.normal(0.0015, 0.01, 500), index=idx)
    low_sharpe = pd.Series(rng.normal(0.0015, 0.04, 500), index=idx)
    result_high = growth_budget_annual_vol(high_sharpe)
    result_low = growth_budget_annual_vol(low_sharpe)
    assert result_high > result_low
