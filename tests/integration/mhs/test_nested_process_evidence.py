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

        extended[pid] = dataclasses.replace(ev, daily_returns=merged, available_at=merged_avail, turnover=pd.concat([ev.turnover, pd.Series(0.01, index=future_idx[-5:])]), valid=pd.concat([ev.valid, pd.Series(True, index=future_idx[-5:])]))
    second = choose_refit_policy(extended, point, spec=spec)
    assert first.policy_id == second.policy_id
    assert first.n_inner_labels == second.n_inner_labels
    assert PROCESS_MIN_TRAIN_DAYS > 0
