"""Invariant guards for nested process evidence over streamed markets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.backtest.labels import ProcessClockSpec, build_proxy_member_returns
from src.mhs.backtest.paths import build_inner_policy_evidence, run_process_paths
from src.mhs.backtest.selection import NestedSelectionSpec, TrainingWindowSpec, choose_refit_policy
from src.mhs.params import PROCESS_MIN_TRAIN_DAYS
from src.mhs.process import RefitPoint

POLICIES = (
    TrainingWindowSpec("expanding", "expanding", None),
    TrainingWindowSpec("rolling_12m", "rolling", 12),
    TrainingWindowSpec("rolling_24m", "rolling", 24),
    TrainingWindowSpec("equal_member", "equal_member", None),
)


def _spec(**over: object) -> NestedSelectionSpec:
    base: dict[str, object] = {
        "policies": POLICIES, "control_policy_id": "equal_member",
        "minimum_inner_labels": 5, "alpha": 0.05, "bootstrap_paths": 100,
        "seed": 7, "fit_latency": pd.Timedelta(0),
    }
    base.update(over)
    return NestedSelectionSpec(**base)  # type: ignore[arg-type]


def _market(n_days: int = 500, seed: int = 11):  # type: ignore[no-untyped-def]
    from src.mhs.backtest.contracts import ProcessMarketData

    rng = np.random.default_rng(seed)
    symbols = ["S00USDT", "S01USDT"]
    decision_grid = pd.date_range("2021-01-01", periods=n_days, freq="24h", tz="UTC")
    grid_1h = pd.date_range(decision_grid[0], decision_grid[-1] + pd.Timedelta(hours=23), freq="1h", tz="UTC")
    drift = np.zeros((len(grid_1h), 2))
    drift[:, 0] = 0.0001
    shocks = rng.normal(0, 0.001, (len(grid_1h), 2))
    log_close_1h = pd.DataFrame(np.cumsum(drift + shocks, axis=0), index=grid_1h, columns=symbols)
    books = {
        "a": pd.DataFrame(0.5, index=decision_grid, columns=symbols),
        "b": pd.DataFrame(-0.5, index=decision_grid, columns=symbols),
    }
    return ProcessMarketData(
        grid_1h=grid_1h, decision_grid=decision_grid, opens_1h=np.exp(log_close_1h),
        bar_funding_1h=pd.DataFrame(0.0, index=grid_1h, columns=symbols),
        log_close_step=log_close_1h.reindex(decision_grid),
        funding_step=pd.DataFrame(0.0, index=decision_grid, columns=symbols),
        member_books=books, execution_mask=pd.DataFrame(True, index=decision_grid, columns=symbols),
        funding_known_1h=pd.DataFrame(True, index=grid_1h, columns=symbols),
    )


def _prepared(n_days: int = 500):  # type: ignore[no-untyped-def]
    import hashlib

    data = _market(n_days)
    clock = ProcessClockSpec(pd.Timedelta(hours=24), pd.Timedelta(hours=1), pd.Timedelta(0))
    digest = hashlib.sha256(",".join(data.member_books.keys()).encode()).hexdigest()
    mev = build_proxy_member_returns(data, clock=clock, one_way_bps=8.0, procedure_digest=digest, input_manifest_digest=None)
    import dataclasses as _dc

    poisoned = mev.returns.copy()
    poisoned.iloc[400, 0] = float("nan")
    poisoned_known = mev.known.copy()
    poisoned_known.iloc[400, 0] = False
    mev = _dc.replace(mev, returns=poisoned, known=poisoned_known)
    return data, mev, clock


def test_build_inner_policy_evidence_reuses_common_source() -> None:
    """One streamed pass materializes the full pool without silently dropping history."""
    data, mev, clock = _prepared()
    spec = _spec()
    inner = build_inner_policy_evidence(
        data, mev, clock=clock, selection_spec=spec, decision_bps=8.0,
        leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
    )
    assert set(inner.keys()) == {p.policy_id for p in POLICIES}
    digests = {(e.procedure_digest, e.economic_source) for e in inner.values()}
    assert digests == {(mev.procedure_digest, "hourly_proxy")}
    lens = {len(e.daily_returns) for e in inner.values()}
    assert len(lens) == 1
    assert all(len(e.fit_audits) > 0 for e in inner.values())
    import dataclasses

    bad = mev.returns.rename(columns={"a": "zzz"})
    bad_mev = dataclasses.replace(mev, returns=bad)
    with pytest.raises(DataIntegrityError):
        build_inner_policy_evidence(
            data, bad_mev, clock=clock, selection_spec=spec, decision_bps=8.0,
            leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
        )


def test_nested_path_carries_state_across_policy_switch() -> None:
    """Neighboring months with different winners keep held state instead of resetting."""
    data, mev, clock = _prepared()
    inner = build_inner_policy_evidence(
        data, mev, clock=clock, selection_spec=_spec(), decision_bps=8.0,
        leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
    )
    idx = next(iter(inner.values())).daily_returns.index
    points = (
        RefitPoint(idx[10], idx[20], idx[10]),
        RefitPoint(idx[20], idx[30], idx[20]),
    )
    paths = run_process_paths(
        data, points, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0,
        clock=clock, member_evidence=mev,
        selection_spec=_spec(minimum_inner_labels=5), inner_evidence=inner,
    )
    assert len(paths[0].policy_choices) == 2
    assert paths[0].target_weights.index.is_monotonic_increasing
    assert bool((paths[0].target_weights.index == paths[0].unit_target_weights.index).all())
    assert not bool(paths[0].target_weights.isna().all(axis=None))


def test_inner_evidence_proxy_source_honesty() -> None:
    """Hourly inner evidence never claims execution-native selection authority."""
    data, mev, clock = _prepared()
    inner = build_inner_policy_evidence(
        data, mev, clock=clock, selection_spec=_spec(), decision_bps=8.0,
        leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
    )
    assert all(e.economic_source == "hourly_proxy" for e in inner.values())
    assert all(e.input_manifest_digest is None for e in inner.values())


def test_nested_choices_prefix_equivalence() -> None:
    """Extending the future cannot change prior nested choices or fitted rows."""
    data, mev, clock = _prepared()
    inner = build_inner_policy_evidence(
        data, mev, clock=clock, selection_spec=_spec(), decision_bps=8.0,
        leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
    )
    idx = next(iter(inner.values())).daily_returns.index
    point = RefitPoint(idx[30], idx[40], idx[30])
    spec = _spec()
    first = choose_refit_policy(inner, point, spec=spec)
    future_idx = idx.append(pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=5, freq="24h", tz="UTC"))
    extended = {}
    for pid, ev in inner.items():
        extra = pd.Series(0.05, index=future_idx[-5:])
        merged = pd.concat([ev.daily_returns, extra])
        merged_avail = ev.available_at.append(future_idx[-5:] + pd.Timedelta(days=30))
        import dataclasses

        from src.mhs.backtest.selection import InnerFitAudit
        from src.mhs.process import RefitPoint as _RefitPoint

        tail_start = max(future_idx[-5], max(a.point.effective_to for a in ev.fit_audits))
        tail_audits: tuple = ()
        if tail_start <= future_idx[-1]:
            tail_audits = (
                InnerFitAudit(
                    _RefitPoint(tail_start, future_idx[-1] + pd.Timedelta(days=1), idx[-1]),
                    None, 0, None, None,
                ),
            )
        extended[pid] = dataclasses.replace(ev, daily_returns=merged, available_at=merged_avail, turnover=pd.concat([ev.turnover, pd.Series(0.01, index=future_idx[-5:])]), valid=pd.concat([ev.valid, pd.Series(True, index=future_idx[-5:])]), fit_audits=ev.fit_audits + tail_audits)
    second = choose_refit_policy(extended, point, spec=spec)
    assert first.policy_id == second.policy_id
    assert first.n_inner_labels == second.n_inner_labels
    assert PROCESS_MIN_TRAIN_DAYS > 0


def _holding_market(n_days: int = 500, seed: int = 11):  # type: ignore[no-untyped-def]
    from src.mhs.backtest.contracts import ProcessMarketData

    rng = np.random.default_rng(seed)
    symbols = ["S00USDT", "S01USDT"]
    decision_grid = pd.date_range("2021-01-01", periods=n_days, freq="24h", tz="UTC")
    grid_1h = pd.date_range(decision_grid[0], decision_grid[-1] + pd.Timedelta(hours=23), freq="1h", tz="UTC")
    drift = np.zeros((len(grid_1h), 2))
    drift[:, 0] = 0.0001
    shocks = rng.normal(0, 0.001, (len(grid_1h), 2))
    log_close_1h = pd.DataFrame(np.cumsum(drift + shocks, axis=0), index=grid_1h, columns=symbols)
    books = {
        "a": pd.DataFrame({"S00USDT": 0.5, "S01USDT": 0.0}, index=decision_grid),
        "b": pd.DataFrame({"S00USDT": 0.0, "S01USDT": 0.5}, index=decision_grid),
    }
    return ProcessMarketData(
        grid_1h=grid_1h, decision_grid=decision_grid, opens_1h=np.exp(log_close_1h),
        bar_funding_1h=pd.DataFrame(0.0, index=grid_1h, columns=symbols),
        log_close_step=log_close_1h.reindex(decision_grid),
        funding_step=pd.DataFrame(0.0, index=decision_grid, columns=symbols),
        member_books=books, execution_mask=pd.DataFrame(True, index=decision_grid, columns=symbols),
        funding_known_1h=pd.DataFrame(True, index=grid_1h, columns=symbols),
    )


def _holding_prepared(n_days: int = 500):  # type: ignore[no-untyped-def]
    import hashlib

    data = _holding_market(n_days)
    clock = ProcessClockSpec(pd.Timedelta(hours=24), pd.Timedelta(hours=1), pd.Timedelta(0))
    digest = hashlib.sha256(",".join(data.member_books.keys()).encode()).hexdigest()
    mev = build_proxy_member_returns(data, clock=clock, one_way_bps=8.0, procedure_digest=digest, input_manifest_digest=None)
    return data, mev, clock


def _outer_single_policy_path(data, mev, clock, policy_id, **over):  # type: ignore[no-untyped-def]
    from src.mhs.process import matured_monthly_refit_schedule

    window = next(p for p in POLICIES if p.policy_id == policy_id)
    params: dict = {"decision_bps": 8.0, "leverage_cap": 2.0}
    params.update(over)
    schedule = matured_monthly_refit_schedule(
        mev, data.decision_grid[-1], min_train_days=PROCESS_MIN_TRAIN_DAYS,
        fit_latency=clock.fit_latency,
    )
    (path,) = run_process_paths(
        data, schedule, decision_bps=params["decision_bps"],
        evaluation_bps=(params["decision_bps"],), leverage_cap=params["leverage_cap"],
        execution_policy=params.get("execution_policy"), risk_sizing=params.get("risk_sizing"),
        memory_budget=None, clock=clock, member_evidence=mev, training_window=window,
    )
    return path


def test_inner_returns_follow_decision_cost() -> None:
    """Nonzero combined turnover makes inner economics sensitive to the decision cost."""
    data, mev, clock = _prepared()
    cheap = build_inner_policy_evidence(
        data, mev, clock=clock, selection_spec=_spec(), decision_bps=8.0,
        leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
    )
    rich = build_inner_policy_evidence(
        data, mev, clock=clock, selection_spec=_spec(), decision_bps=24.0,
        leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
    )
    ev = cheap["expanding"]
    assert bool((ev.turnover.to_numpy(dtype="float64") > 0.0).any())
    other = rich["expanding"]
    pd.testing.assert_index_equal(ev.daily_returns.index, other.daily_returns.index)
    both = ev.valid.to_numpy(dtype=bool) & other.valid.to_numpy(dtype=bool)
    assert bool(both.any())
    diff = (ev.daily_returns.to_numpy(dtype="float64") - other.daily_returns.to_numpy(dtype="float64"))[both]
    assert bool((np.abs(diff) > 1e-12).any())


def test_inner_and_outer_targets_share_availability() -> None:
    """A symbol masked after smoothing is zero in both the inner and outer targets."""
    import dataclasses

    data, mev, clock = _prepared()
    cutoff = data.decision_grid[300]
    masked = data.execution_mask.copy()
    masked.loc[masked.index >= cutoff, "S00USDT"] = False
    gated = dataclasses.replace(data, execution_mask=masked)
    inner = build_inner_policy_evidence(
        gated, mev, clock=clock, selection_spec=_spec(), decision_bps=8.0,
        leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
    )
    outer = _outer_single_policy_path(gated, mev, clock, "expanding")
    oos = outer.unit_target_weights.index >= cutoff
    assert bool((outer.unit_target_weights.loc[oos, "S00USDT"] == 0.0).all())
    shared = inner["expanding"].daily_returns.index
    pd.testing.assert_series_equal(
        inner["expanding"].daily_returns,
        outer.daily_returns.reindex(shared),
        check_names=False,
    )


def test_inner_return_is_combined_ledger_not_member_average() -> None:
    """Overlapping books net and scale in combination, never as a member average."""
    data, mev, clock = _holding_prepared()
    inner = build_inner_policy_evidence(
        data, mev, clock=clock, selection_spec=_spec(), decision_bps=8.0,
        leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
    )
    outer = _outer_single_policy_path(data, mev, clock, "equal_member")
    ev = inner["equal_member"]
    shared = ev.daily_returns.index
    pd.testing.assert_series_equal(
        ev.daily_returns, outer.daily_returns.reindex(shared), check_names=False
    )
    flags = ev.valid.to_numpy(dtype=bool)
    assert bool(flags.any())
    average = (
        mev.returns["a"].reindex(shared).to_numpy(dtype="float64")
        + mev.returns["b"].reindex(shared).to_numpy(dtype="float64")
    ) / 2.0
    diff = np.abs(ev.daily_returns.to_numpy(dtype="float64") - average)[flags]
    assert bool((diff > 1e-6).any())


def test_inner_evidence_retains_risk_contract() -> None:
    """A named causal sizing contract shapes inner evidence like the outer path."""
    from src.mhs.process import ProcessRiskSizingSpec

    data, mev, clock = _prepared()
    sizing = ProcessRiskSizingSpec(
        annual_volatility_target=0.25, ewma_halflife_days=60,
        minimum_observations=60, leverage_cap=2.0,
    )
    sized = build_inner_policy_evidence(
        data, mev, clock=clock, selection_spec=_spec(), decision_bps=8.0,
        leverage_cap=2.0, execution_policy=None, risk_sizing=sizing, memory_budget=None,
    )
    plain = build_inner_policy_evidence(
        data, mev, clock=clock, selection_spec=_spec(), decision_bps=8.0,
        leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
    )
    outer = _outer_single_policy_path(data, mev, clock, "expanding", risk_sizing=sizing)
    ev = sized["expanding"]
    shared = ev.daily_returns.index
    pd.testing.assert_series_equal(
        ev.daily_returns, outer.daily_returns.reindex(shared), check_names=False
    )
    base = plain["expanding"]
    both = ev.valid.to_numpy(dtype=bool) & base.valid.to_numpy(dtype=bool)
    diff = (ev.daily_returns.to_numpy(dtype="float64") - base.daily_returns.to_numpy(dtype="float64"))[both]
    assert bool((np.abs(diff) > 1e-12).any())


def test_inner_turnover_tracks_carried_holdings() -> None:
    """Refit boundaries carry smoothed holdings without synthetic liquidation."""
    data, mev, clock = _prepared()
    inner = build_inner_policy_evidence(
        data, mev, clock=clock, selection_spec=_spec(), decision_bps=8.0,
        leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
    )
    outer = _outer_single_policy_path(data, mev, clock, "expanding")
    ev = inner["expanding"]
    shared = ev.daily_returns.index
    held = outer.unit_target_weights.reindex(shared).to_numpy(dtype="float64")
    prev = np.vstack([np.zeros((1, held.shape[1])), held[:-1]])
    expected = np.abs(held - prev).sum(axis=1)
    np.testing.assert_allclose(ev.turnover.to_numpy(dtype="float64"), expected, rtol=1e-9, atol=1e-12)
    bounds = sorted({a.point.effective_from for a in ev.fit_audits} & set(shared))
    assert len(bounds) >= 1
    edge = ev.turnover.loc[bounds].to_numpy(dtype="float64")
    assert bool(np.isfinite(edge).all())
    assert bool((edge < 1.0).all())


def test_inner_unknown_dates_stay_indexed() -> None:
    """Unknown combined economics invalidate the date everywhere without dropping it."""
    import dataclasses

    data, mev, clock = _holding_prepared()
    day = data.decision_grid[450]
    gapped_known = data.funding_known_1h.copy()
    gapped_known.loc[
        (gapped_known.index > day) & (gapped_known.index <= day + pd.Timedelta(hours=24))
    ] = False
    gapped = dataclasses.replace(data, funding_known_1h=gapped_known)
    inner = build_inner_policy_evidence(
        gapped, mev, clock=clock, selection_spec=_spec(), decision_bps=8.0,
        leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
    )
    for ev in inner.values():
        assert day in ev.daily_returns.index
    for policy_id in ("expanding", "equal_member"):
        ev = inner[policy_id]
        assert not bool(ev.valid.loc[day])
        assert bool(np.isnan(ev.daily_returns.loc[day]))


def test_inner_evidence_rejects_misaligned_inputs() -> None:
    """Off-grid labels, misaligned economics and missing knowledge fail closed."""
    import dataclasses

    data, mev, clock = _holding_prepared()
    pushed = dataclasses.replace(
        mev,
        returns=mev.returns.set_axis(mev.returns.index + pd.Timedelta(hours=1)),
        known=mev.known.set_axis(mev.known.index + pd.Timedelta(hours=1)),
        label_start=mev.label_start + pd.Timedelta(hours=1),
        label_end=mev.label_end + pd.Timedelta(hours=1),
        available_at=mev.available_at + pd.Timedelta(hours=1),
    )
    with pytest.raises(DataIntegrityError, match="decision grid"):
        build_inner_policy_evidence(
            data, pushed, clock=clock, selection_spec=_spec(), decision_bps=8.0,
            leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
        )
    renamed = dataclasses.replace(
        data, log_close_step=data.log_close_step.rename(columns={"S00USDT": "ZZZ"})
    )
    with pytest.raises(DataIntegrityError, match="symbol-aligned"):
        build_inner_policy_evidence(
            renamed, mev, clock=clock, selection_spec=_spec(), decision_bps=8.0,
            leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
        )
    unknown = dataclasses.replace(data, funding_known_1h=None)
    with pytest.raises(DataIntegrityError, match="funding knowledge"):
        build_inner_policy_evidence(
            unknown, mev, clock=clock, selection_spec=_spec(), decision_bps=8.0,
            leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
        )
    dropped = dataclasses.replace(
        data, funding_known_1h=data.funding_known_1h.drop(columns=["S00USDT"])
    )
    with pytest.raises(DataIntegrityError, match="funding knowledge"):
        build_inner_policy_evidence(
            dropped, mev, clock=clock, selection_spec=_spec(), decision_bps=8.0,
            leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
        )


def test_inner_final_day_without_forward_stays_unknown() -> None:
    """A held final decision day without forward economics stays indexed and unknown."""
    from src.mhs.backtest.labels import MaturedMemberReturns

    data, _, clock = _holding_prepared()
    grid = data.decision_grid
    rng = np.random.default_rng(3)
    mev = MaturedMemberReturns(
        pd.DataFrame(
            {
                "a": np.linspace(0.001, 0.004, len(grid)) + rng.normal(0, 1e-4, len(grid)),
                "b": rng.normal(0.0005, 0.001, len(grid)),
            },
            index=grid,
        ),
        pd.DataFrame(True, index=grid, columns=["a", "b"]),
        grid, grid + pd.Timedelta(hours=24), grid + pd.Timedelta(hours=24),
        "daily_step_proxy", "p", None,
    )
    inner = build_inner_policy_evidence(
        data, mev, clock=clock, selection_spec=_spec(), decision_bps=8.0,
        leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
    )
    last = grid[-1]
    for ev in inner.values():
        assert last in ev.daily_returns.index
    ev = inner["equal_member"]
    assert not bool(ev.valid.loc[last])
    assert bool(np.isnan(ev.daily_returns.loc[last]))


def test_inner_daily_return_available_at_label_end() -> None:
    """Daily inner returns mature at the label end, not one bar lag after open."""
    data, mev, clock = _prepared()
    inner = build_inner_policy_evidence(
        data, mev, clock=clock, selection_spec=_spec(), decision_bps=8.0,
        leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
    )
    expected = inner["expanding"].daily_returns.index + clock.decision_period + clock.bar_completion_lag
    for ev in inner.values():
        pd.testing.assert_index_equal(ev.available_at, pd.DatetimeIndex(expected))
        assert bool((ev.available_at > ev.daily_returns.index).all())


def test_late_interval_returns_leave_prior_refit_choice_unchanged() -> None:
    """Mutating an unfinished daily interval cannot move an earlier refit choice."""
    import dataclasses

    from src.mhs.backtest.selection import NestedSelectionSpec

    data, mev, clock = _prepared()
    two = NestedSelectionSpec(
        policies=(POLICIES[0], POLICIES[3]), control_policy_id="equal_member",
        minimum_inner_labels=5, alpha=0.05, bootstrap_paths=100,
        seed=7, fit_latency=pd.Timedelta(0),
    )
    inner = build_inner_policy_evidence(
        data, mev, clock=clock, selection_spec=two, decision_bps=8.0,
        leverage_cap=2.0, execution_policy=None, risk_sizing=None, memory_budget=None,
    )
    idx = next(iter(inner.values())).daily_returns.index
    point = RefitPoint(idx[-70], idx[-60], idx[-70])
    first = choose_refit_policy(inner, point, spec=two)
    assert first.n_inner_labels >= two.minimum_inner_labels
    cutoff = point.train_end
    horizon = clock.decision_period + clock.bar_completion_lag
    late = [d for d in idx if d + clock.bar_completion_lag <= cutoff < d + horizon]
    assert len(late) >= 1
    boundary = late[-1]
    assert all(bool(ev.valid.loc[boundary]) for ev in inner.values())
    ready = max(ev.prerequisites_ready_at for ev in inner.values())
    expected = [
        d for d in idx
        if d >= ready and d + horizon <= cutoff
        and all(bool(ev.valid.loc[d]) for ev in inner.values())
    ]
    assert first.n_inner_labels == len(expected)
    assert first.inner_end == expected[-1]
    mutated: dict = {}
    for pid, ev in inner.items():
        rets = ev.daily_returns.copy()
        for stamp in late:
            if bool(ev.valid.loc[stamp]):
                rets.loc[stamp] = 0.5 if pid == "expanding" else -0.5
        mutated[pid] = dataclasses.replace(ev, daily_returns=rets)
    second = choose_refit_policy(mutated, point, spec=two)
    assert second.policy_id == first.policy_id
    assert second.n_inner_labels == first.n_inner_labels
    assert second.inner_start == first.inner_start
    assert second.inner_end == first.inner_end
