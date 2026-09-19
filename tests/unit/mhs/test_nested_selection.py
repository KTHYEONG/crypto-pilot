"""Invariant guards for chronological nested policy selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.backtest.selection import (
    CONTROL_PREREQUISITES_INVALID,
    INNER_EVIDENCE_INSUFFICIENT,
    NO_ESTIMATED_IMPROVEMENT,
    PAIRED_LCB_IMPROVEMENT,
    InnerFitAudit,
    InnerPolicyEvidence,
    NestedSelectionSpec,
    TrainingWindowSpec,
    choose_refit_policy,
)
from src.mhs.process import RefitPoint

POLICIES = (
    TrainingWindowSpec("expanding", "expanding", None),
    TrainingWindowSpec("rolling_12m", "rolling", 12),
    TrainingWindowSpec("rolling_24m", "rolling", 24),
    TrainingWindowSpec("equal_member", "equal_member", None),
)


def _spec(**over: object) -> NestedSelectionSpec:
    base: dict[str, object] = {
        "policies": POLICIES,
        "control_policy_id": "equal_member",
        "minimum_inner_labels": 5,
        "alpha": 0.05,
        "bootstrap_paths": 200,
        "seed": 11,
        "fit_latency": pd.Timedelta(0),
    }
    base.update(over)
    return NestedSelectionSpec(**base)  # type: ignore[arg-type]


def _evidence(
    dates: pd.DatetimeIndex,
    returns: dict[str, np.ndarray],
    *,
    turnover: dict[str, float] | None = None,
    valid: bool = True,
    digest: str = "p",
    ready: pd.Timestamp | None = None,
    cutoff_cover: bool = True,
) -> dict[str, InnerPolicyEvidence]:
    out: dict[str, InnerPolicyEvidence] = {}
    audit_point = RefitPoint(dates[0], dates[-1] + pd.Timedelta(days=1), dates[0])
    for pid, arr in returns.items():
        series = pd.Series(arr, index=dates)
        tval = (turnover or {}).get(pid, 0.01)
        out[pid] = InnerPolicyEvidence(
            pid, series, pd.DatetimeIndex(dates), pd.Series(tval, index=dates),
            pd.Series(valid, index=dates), digest, None, "hourly_proxy",
            pd.Timestamp(dates[0] if ready is None else ready),
            (InnerFitAudit(audit_point, None, len(dates), None, None),) if cutoff_cover else (),
        )
    return out


def _dates(n: int = 60) -> pd.DatetimeIndex:
    return pd.date_range("2021-01-01", periods=n, freq="24h", tz="UTC")


def _point(dates: pd.DatetimeIndex) -> RefitPoint:
    return RefitPoint(dates[-1] + pd.Timedelta(days=1), dates[-1] + pd.Timedelta(days=31), dates[-1])


def test_choose_refit_policy_outer_future_isolation() -> None:
    """Outer future profits cannot rewrite an earlier chronological choice."""
    dates = _dates()
    rng = np.random.default_rng(3)
    base = rng.normal(0.001, 0.005, len(dates))
    rets = {"expanding": base + 0.004, "rolling_12m": base, "rolling_24m": base, "equal_member": base}
    ev = _evidence(dates, rets)
    first = choose_refit_policy(ev, _point(dates), spec=_spec())
    later_point = RefitPoint(dates[-1] + pd.Timedelta(days=1), dates[-1] + pd.Timedelta(days=62), dates[-1])
    second = choose_refit_policy(ev, later_point, spec=_spec())
    assert (first.policy_id, first.train_start) == (second.policy_id, second.train_start)


def test_choose_refit_policy_rejects_future_fit() -> None:
    """An inner fit after its application start fails chronology."""
    dates = _dates()
    rets = {p.policy_id: np.zeros(len(dates)) for p in POLICIES}
    ev = _evidence(dates, rets)
    bad_audit = InnerFitAudit(RefitPoint(dates[0], dates[-1], dates[5]), None, 4, dates[9], None)
    bad = InnerPolicyEvidence(
        "expanding", pd.Series(np.zeros(len(dates)), index=dates), pd.DatetimeIndex(dates),
        pd.Series(0.01, index=dates), pd.Series(True, index=dates), "p", None,
        "hourly_proxy", dates[0], (bad_audit,),
    )
    ev["expanding"] = bad
    with pytest.raises(DataIntegrityError):
        choose_refit_policy(ev, _point(dates), spec=_spec())


def test_choose_refit_policy_rejects_unaudited_inner_trajectory() -> None:
    """Every inner application date requires an auditable preceding fit."""
    dates = _dates()
    rets = {p.policy_id: np.zeros(len(dates)) for p in POLICIES}
    ev = _evidence(dates, rets)
    first = ev["expanding"]
    ev["expanding"] = InnerPolicyEvidence(
        first.policy_id, first.daily_returns, first.available_at, first.turnover,
        first.valid, first.procedure_digest, first.input_manifest_digest,
        first.economic_source, first.prerequisites_ready_at, (),
    )
    with pytest.raises(DataIntegrityError, match="requires fit audits"):
        choose_refit_policy(ev, _point(dates), spec=_spec())


def test_choose_refit_policy_common_coverage() -> None:
    """All policies share the same admissible start and date set."""
    dates = _dates()
    short = dates[20:]
    rets_full = {p.policy_id: np.zeros(len(dates)) for p in POLICIES}
    ev = _evidence(dates, rets_full, ready=short[0])
    for pid in ev:
        old = ev[pid]
        mask = old.daily_returns.index.isin(short)
        ev[pid] = InnerPolicyEvidence(
            pid, old.daily_returns.loc[short], pd.DatetimeIndex(short),
            old.turnover.loc[short], old.valid.loc[short], "p", None,
            "hourly_proxy", short[0], old.fit_audits,
        )
        assert len(ev[pid].daily_returns) == len(short)
    choice = choose_refit_policy(ev, _point(dates), spec=_spec(minimum_inner_labels=3))
    assert choice.inner_start == short[0]
    assert choice.inner_end == short[-1]
    assert choice.n_inner_labels == len(short)


def test_choose_refit_policy_candidate_failure_retained() -> None:
    """A failed candidate trial invalidates the comparison instead of vanishing."""
    dates = _dates()
    rets = {p.policy_id: np.zeros(len(dates)) for p in POLICIES}
    rets["rolling_12m"] = np.full(len(dates), np.nan)
    ev = _evidence(dates, rets)
    with pytest.raises(DataIntegrityError):
        choose_refit_policy(ev, _point(dates), spec=_spec())


def test_choose_refit_policy_insufficient_shared_sample() -> None:
    """Without a common interval the fixed control or an inactive choice is kept."""
    dates = _dates(8)
    rets = {p.policy_id: np.zeros(len(dates)) for p in POLICIES}
    ev = _evidence(dates, rets)
    point = _point(dates)
    got = choose_refit_policy(ev, point, spec=_spec(minimum_inner_labels=50))
    assert got.policy_id == "equal_member"
    assert got.reason_codes == (INNER_EVIDENCE_INSUFFICIENT,)
    assert got.paired_growth_lcb is None
    late_ready = dates[-1] + pd.Timedelta(days=10)
    ev_late = _evidence(dates, rets, ready=late_ready)
    inactive = choose_refit_policy(ev_late, point, spec=_spec(minimum_inner_labels=50))
    assert inactive.policy_id is None
    assert CONTROL_PREREQUISITES_INVALID in inactive.reason_codes


def test_choose_refit_policy_no_improvement_selects_control() -> None:
    """Nonpositive paired LCBs keep the registered equal-member control."""
    dates = _dates()
    rng = np.random.default_rng(5)
    control = rng.normal(0.002, 0.004, len(dates))
    rets = {"expanding": control - 0.005, "rolling_12m": control - 0.004, "rolling_24m": control - 0.003, "equal_member": control}
    choice = choose_refit_policy(_evidence(dates, rets), _point(dates), spec=_spec())
    assert choice.policy_id == "equal_member"
    assert choice.reason_codes == (NO_ESTIMATED_IMPROVEMENT,)


def test_choose_refit_policy_tie_breaker() -> None:
    """Equal LCBs break by lower turnover, then declaration order, deterministically."""
    dates = _dates()
    rng = np.random.default_rng(9)
    control = rng.normal(0.0, 0.004, len(dates))
    bump = control + 0.004
    rets = {"expanding": bump, "rolling_12m": bump.copy(), "rolling_24m": control, "equal_member": control}
    ev = _evidence(dates, rets, turnover={"expanding": 0.09, "rolling_12m": 0.01, "rolling_24m": 0.01, "equal_member": 0.01})
    first = choose_refit_policy(ev, _point(dates), spec=_spec())
    second = choose_refit_policy(ev, _point(dates), spec=_spec())
    assert first.policy_id == "rolling_12m" == second.policy_id
    assert first.reason_codes == (PAIRED_LCB_IMPROVEMENT,)


def test_choose_refit_policy_pool_identity_mismatch() -> None:
    """Changed identities behind one policy ID fail rather than mix experiments."""
    dates = _dates()
    rets = {p.policy_id: np.zeros(len(dates)) for p in POLICIES}
    ev = _evidence(dates, rets)
    ev["rolling_12m"] = InnerPolicyEvidence(
        "rolling_12m", pd.Series(np.zeros(len(dates)), index=dates), pd.DatetimeIndex(dates),
        pd.Series(0.01, index=dates), pd.Series(True, index=dates), "other", None,
        "hourly_proxy", dates[0], (),
    )
    with pytest.raises(DataIntegrityError):
        choose_refit_policy(ev, _point(dates), spec=_spec())
    short_ev = {k: v for k, v in ev.items() if k != "rolling_24m"}
    with pytest.raises(DataIntegrityError):
        choose_refit_policy(short_ev, _point(dates), spec=_spec())


def test_choose_refit_policy_filters_invalid_future_and_unaudited_dates() -> None:
    """Losing/invalid days, future availability and unaudited dates share one date set."""
    dates = _dates(20)
    rets = {p.policy_id: np.zeros(len(dates)) for p in POLICIES}
    ev = _evidence(dates, rets)
    expanding = ev["expanding"]
    valid = expanding.valid.copy()
    valid.iloc[5] = False
    ev["expanding"] = InnerPolicyEvidence(
        "expanding", expanding.daily_returns, expanding.available_at, expanding.turnover,
        valid, "p", None, "hourly_proxy", dates[0], expanding.fit_audits,
    )
    roller = ev["rolling_12m"]
    future_avail = list(roller.available_at)
    future_avail[6] = dates[-1] + pd.Timedelta(days=30)
    avail = pd.DatetimeIndex(future_avail)
    ev["rolling_12m"] = InnerPolicyEvidence(
        "rolling_12m", roller.daily_returns, avail, roller.turnover,
        roller.valid, "p", None, "hourly_proxy", dates[0], roller.fit_audits,
    )
    narrow_audit = InnerFitAudit(RefitPoint(dates[0], dates[10], dates[0]), None, 10, None, None)
    old_24 = ev["rolling_24m"]
    ev["rolling_24m"] = InnerPolicyEvidence(
        "rolling_24m", old_24.daily_returns, old_24.available_at, old_24.turnover,
        old_24.valid, "p", None, "hourly_proxy", dates[0], (narrow_audit,),
    )
    choice = choose_refit_policy(ev, _point(dates), spec=_spec(minimum_inner_labels=3))
    assert choice.n_inner_labels == 8
    assert choice.inner_start == dates[0]
    assert choice.inner_end == dates[9]


def test_choose_refit_policy_rejects_non_datetime_index() -> None:
    """Inner trajectories without exact application dates fail validation."""
    dates = _dates()
    n = len(dates)
    bad_returns = pd.Series(np.zeros(n))
    ev = _evidence(dates, {p.policy_id: np.zeros(n) for p in POLICIES})
    first = ev["expanding"]
    ev["expanding"] = InnerPolicyEvidence(
        "expanding", bad_returns, first.available_at, pd.Series(0.01, index=range(n)),
        pd.Series(True, index=range(n)), "p", None, "hourly_proxy", dates[0], first.fit_audits,
    )
    with pytest.raises(DataIntegrityError):
        choose_refit_policy(ev, _point(dates), spec=_spec())


def test_run_process_paths_control_validation_boundaries() -> None:
    """Mismatched clocks, member columns and thin matured labels fail closed."""
    from src.mhs.backtest.labels import MaturedMemberReturns
    from src.mhs.backtest.labels import ProcessClockSpec
    from src.mhs.backtest.paths import run_process_paths

    data = _synthetic_market(n_days=30)
    dates = data.decision_grid[:4]
    members = pd.DataFrame({"a": [0.01, 0.02, 0.03, 0.04], "b": [0.0, 0.0, 0.0, 0.0]}, index=dates)
    known = pd.DataFrame(True, index=dates, columns=["a", "b"])
    mev = MaturedMemberReturns(
        members, known, dates - pd.Timedelta(days=1), dates, dates, "daily_step_proxy", "p", None,
    )
    clock = ProcessClockSpec(pd.Timedelta(hours=24), pd.Timedelta(hours=1), pd.Timedelta(0))
    points = (RefitPoint(dates[1], dates[2], dates[0]),)
    with pytest.raises(ValueError, match="must be supplied together"):
        run_process_paths(
            data, points, decision_bps=0.0, evaluation_bps=(0.0,), leverage_cap=1.0,
            clock=clock, member_evidence=None,
        )
    renamed = MaturedMemberReturns(
        members.rename(columns={"a": "zzz"}), known.rename(columns={"a": "zzz"}),
        dates - pd.Timedelta(days=1), dates, dates, "daily_step_proxy", "p", None,
    )
    with pytest.raises(DataIntegrityError):
        run_process_paths(
            data, points, decision_bps=0.0, evaluation_bps=(0.0,), leverage_cap=1.0,
            clock=clock, member_evidence=renamed,
        )
    thin = run_process_paths(
        data, points, decision_bps=0.0, evaluation_bps=(0.0,), leverage_cap=1.0,
        clock=clock, member_evidence=mev,
    )
    assert thin[0].refits[0].member_weights == {}
    assert thin[0].refits[0].n_train_labels == 1


def test_windowed_weights_insufficient_labels_go_flat() -> None:
    """A window without two matured labels holds flat instead of estimating."""
    from src.mhs.backtest.labels import MaturedMemberReturns
    from src.mhs.backtest.labels import ProcessClockSpec
    from src.mhs.backtest.paths import run_process_paths

    data = _synthetic_market(n_days=30)
    dates = data.decision_grid[:4]
    members = pd.DataFrame({"a": [0.01, 0.02, 0.03, 0.04], "b": [0.0, 0.0, 0.0, 0.0]}, index=dates)
    known = pd.DataFrame(True, index=dates, columns=["a", "b"])
    mev = MaturedMemberReturns(
        members, known, dates - pd.Timedelta(days=1), dates, dates, "daily_step_proxy", "p", None,
    )
    clock = ProcessClockSpec(pd.Timedelta(hours=24), pd.Timedelta(hours=1), pd.Timedelta(0))
    points = (RefitPoint(dates[1], dates[2], dates[0]),)
    paths = run_process_paths(
        data, points, decision_bps=0.0, evaluation_bps=(0.0,), leverage_cap=1.0,
        clock=clock, member_evidence=mev, training_window=POLICIES[0],
    )
    assert paths[0].refits[0].n_train_labels == 1
    assert paths[0].refits[0].member_weights == {}


def test_training_window_and_selection_specs_reject_invalid_controls() -> None:
    """Registered window and comparison controls reject invented values."""
    with pytest.raises(DataIntegrityError):
        TrainingWindowSpec("", "expanding", None)
    with pytest.raises(DataIntegrityError):
        TrainingWindowSpec("rolling_12m", "rolling", 0)
    with pytest.raises(DataIntegrityError):
        TrainingWindowSpec("expanding", "expanding", 12)
    with pytest.raises(DataIntegrityError):
        _spec(alpha=0.0)
    with pytest.raises(DataIntegrityError):
        InnerPolicyEvidence(
            "", pd.Series([0.0]), pd.DatetimeIndex(["2021-01-01"], tz="UTC"),
            pd.Series([0.0]), pd.Series([True]), "", None, "hourly_proxy",
            pd.Timestamp("2021-01-01", tz="UTC"), (),
        )


def _synthetic_market(n_days: int = 120, seed: int = 7):  # type: ignore[no-untyped-def]
    from src.mhs.backtest.contracts import ProcessMarketData

    rng = np.random.default_rng(seed)
    symbols = ["S00USDT", "S01USDT"]
    decision_grid = pd.date_range("2021-01-01", periods=n_days, freq="24h", tz="UTC")
    grid_1h = pd.date_range(decision_grid[0], decision_grid[-1] + pd.Timedelta(hours=23), freq="1h", tz="UTC")
    log_close_1h = pd.DataFrame(rng.normal(0, 0.001, (len(grid_1h), 2)).cumsum(axis=0), index=grid_1h, columns=symbols)
    opens_1h = np.exp(log_close_1h)
    bar_funding_1h = pd.DataFrame(0.0, index=grid_1h, columns=symbols)
    books = {
        "a": pd.DataFrame(0.5, index=decision_grid, columns=symbols),
        "b": pd.DataFrame(-0.5, index=decision_grid, columns=symbols),
    }
    return ProcessMarketData(
        grid_1h=grid_1h, decision_grid=decision_grid, opens_1h=opens_1h,
        bar_funding_1h=bar_funding_1h, log_close_step=log_close_1h.reindex(decision_grid),
        funding_step=pd.DataFrame(0.0, index=decision_grid, columns=symbols),
        member_books=books, execution_mask=pd.DataFrame(True, index=decision_grid, columns=symbols),
        funding_known_1h=pd.DataFrame(True, index=grid_1h, columns=symbols),
    )


def _member_evidence(data, n_days: int = 120):  # type: ignore[no-untyped-def]
    from src.mhs.backtest.labels import MaturedMemberReturns, ProcessClockSpec

    dates = data.decision_grid[:n_days]
    members = pd.DataFrame(
        {"a": np.linspace(-0.005, 0.005, len(dates)), "b": np.zeros(len(dates))}, index=dates,
    )
    known = pd.DataFrame(True, index=dates, columns=["a", "b"])
    mev = MaturedMemberReturns(members, known, dates - pd.Timedelta(days=1), dates, dates, "daily_step_proxy", "p", None)
    clock = ProcessClockSpec(pd.Timedelta(hours=24), pd.Timedelta(hours=1), pd.Timedelta(0))
    return mev, clock


def test_run_process_paths_records_auditable_cold_start_choice() -> None:
    """A refit before common evidence keeps the declared control choice and counts."""
    from src.mhs.backtest.paths import run_process_paths
    from src.mhs.process import RefitPoint

    data = _synthetic_market()
    mev, clock = _member_evidence(data)
    idx = pd.date_range("2021-01-05", periods=10, freq="24h", tz="UTC")
    inner = _evidence(idx, {p.policy_id: np.zeros(len(idx)) for p in POLICIES})
    points = (RefitPoint(idx[-1] + pd.Timedelta(days=1), idx[-1] + pd.Timedelta(days=5), idx[-1]),)
    paths = run_process_paths(
        data, points, decision_bps=0.0, evaluation_bps=(0.0,), leverage_cap=1.0,
        clock=clock, member_evidence=mev,
        selection_spec=_spec(minimum_inner_labels=50), inner_evidence=inner,
    )
    assert len(paths[0].policy_choices) == 1
    assert paths[0].policy_choices[0].reason_codes == (INNER_EVIDENCE_INSUFFICIENT,)
    assert paths[0].refits[0].policy_id == "equal_member"
    assert paths[0].refits[0].n_train_labels is not None


def test_run_process_paths_rejects_ambiguous_and_partial_membership() -> None:
    """Ambiguous dual controls and partial inner membership are rejected."""
    from src.mhs.backtest.paths import run_process_paths

    data = _synthetic_market()
    mev, clock = _member_evidence(data)
    idx = pd.date_range("2021-01-05", periods=10, freq="24h", tz="UTC")
    inner = _evidence(idx, {p.policy_id: np.zeros(len(idx)) for p in POLICIES})
    points = (RefitPoint(idx[-1] + pd.Timedelta(days=1), idx[-1] + pd.Timedelta(days=5), idx[-1]),)
    with pytest.raises(ValueError, match="must not be supplied together"):
        run_process_paths(
            data, points, decision_bps=0.0, evaluation_bps=(0.0,), leverage_cap=1.0,
            clock=clock, member_evidence=mev, training_window=POLICIES[0],
            selection_spec=_spec(), inner_evidence=inner,
        )
    partial = {k: v for k, v in inner.items() if k != "rolling_24m"}
    with pytest.raises(DataIntegrityError):
        run_process_paths(
            data, points, decision_bps=0.0, evaluation_bps=(0.0,), leverage_cap=1.0,
            clock=clock, member_evidence=mev,
            selection_spec=_spec(), inner_evidence=partial,
        )


def test_run_process_paths_single_window_and_inactive_choice() -> None:
    """Single-window trials and inactive choices flow through without selection."""
    from src.mhs.backtest.labels import MaturedMemberReturns
    from src.mhs.backtest.paths import run_process_paths

    data = _synthetic_market(n_days=30)
    dates = data.decision_grid[:4]
    members = pd.DataFrame({"a": [0.01, 0.02, 0.03, 0.04], "b": [0.0, 0.0, 0.0, 0.0]}, index=dates)
    known = pd.DataFrame(True, index=dates, columns=["a", "b"])
    mev = MaturedMemberReturns(members, known, dates - pd.Timedelta(days=1), dates, dates, "daily_step_proxy", "p", None)
    from src.mhs.backtest.labels import ProcessClockSpec

    clock = ProcessClockSpec(pd.Timedelta(hours=24), pd.Timedelta(hours=1), pd.Timedelta(0))
    points = (RefitPoint(dates[-1] + pd.Timedelta(days=1), dates[-1] + pd.Timedelta(days=2), dates[-1]),)
    rolling = run_process_paths(
        data, points, decision_bps=0.0, evaluation_bps=(0.0,), leverage_cap=1.0,
        clock=clock, member_evidence=mev, training_window=POLICIES[1],
    )
    assert rolling[0].refits[0].policy_id == "rolling_12m"
    idx = pd.date_range("2021-01-05", periods=6, freq="24h", tz="UTC")
    inner = _evidence(idx, {p.policy_id: np.zeros(len(idx)) for p in POLICIES}, ready=idx[-1] + pd.Timedelta(days=30))
    inactive = run_process_paths(
        data, points, decision_bps=0.0, evaluation_bps=(0.0,), leverage_cap=1.0,
        clock=clock, member_evidence=mev,
        selection_spec=_spec(minimum_inner_labels=50), inner_evidence=inner,
    )
    assert inactive[0].policy_choices[0].policy_id is None
    assert inactive[0].refits[0].policy_id is None
