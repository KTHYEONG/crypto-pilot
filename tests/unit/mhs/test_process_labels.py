"""Invariant guards for explicit label maturity and the refit clock."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.backtest.labels import (
    MaturedMemberReturns,
    ProcessClockSpec,
    build_proxy_member_returns,
    select_matured_training_returns,
)

SYMBOLS = ["S0USDT", "S1USDT", "S2USDT", "S3USDT"]
MEMBERS = ["mom", "carry"]


def _clock(
    period_hours: int = 24, lag_hours: int = 1, latency_hours: int = 0
) -> ProcessClockSpec:
    return ProcessClockSpec(
        decision_period=pd.Timedelta(hours=period_hours),
        bar_completion_lag=pd.Timedelta(hours=lag_hours),
        fit_latency=pd.Timedelta(hours=latency_hours),
    )


def _decisions(n: int, start: str = "2021-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="24h", tz="UTC")


def _hourly(decisions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.date_range(
        decisions[0], decisions[-1] + pd.Timedelta(hours=23), freq="1h", tz="UTC"
    )


def _tiny_data(n_days: int = 45, seed: int = 5, known_gap: tuple | None = None):  # type: ignore[no-untyped-def]
    from src.mhs.backtest.contracts import ProcessMarketData

    rng = np.random.default_rng(seed)
    decisions = _decisions(n_days)
    grid = _hourly(decisions)
    log_close = pd.DataFrame(
        np.cumsum(rng.normal(0, 0.002, (n_days, len(SYMBOLS))), axis=0),
        index=decisions,
        columns=SYMBOLS,
    )
    funding_step = pd.DataFrame(
        rng.normal(0, 1e-6, (n_days, len(SYMBOLS))), index=decisions, columns=SYMBOLS
    )
    tilt = pd.DataFrame(0.0, index=decisions, columns=SYMBOLS)
    tilt["S0USDT"] = 0.5
    tilt["S1USDT"] = -0.5
    carry = pd.DataFrame(0.0, index=decisions, columns=SYMBOLS)
    carry["S2USDT"] = 0.5
    carry["S3USDT"] = -0.5
    books = {"mom": tilt.copy(), "carry": carry.copy()}
    known = pd.DataFrame(True, index=grid, columns=SYMBOLS)
    if known_gap is not None:
        symbol, after, through = known_gap
        known.loc[(known.index > after) & (known.index <= through), symbol] = False
    opens = pd.DataFrame(
        100.0 + rng.normal(0, 0.5, (len(grid), len(SYMBOLS))), index=grid, columns=SYMBOLS
    )
    return ProcessMarketData(
        grid_1h=grid,
        decision_grid=decisions,
        opens_1h=opens,
        bar_funding_1h=pd.DataFrame(0.0, index=grid, columns=SYMBOLS),
        log_close_step=log_close,
        funding_step=funding_step,
        member_books=books,
        execution_mask=pd.DataFrame(True, index=decisions, columns=SYMBOLS),
        funding_known_1h=known,
    )


def _hand_labels(
    n: int = 3,
    start: str = "2021-01-01",
    members: tuple[str, ...] = ("A", "B"),
    last_avail_shift_hours: int = 1,
) -> MaturedMemberReturns:
    idx = _decisions(n, start)
    values = (
        np.arange(1, n * len(members) + 1, dtype="float64").reshape(n, len(members)) / 100.0
    )
    returns = pd.DataFrame(values, index=idx, columns=list(members))
    known = pd.DataFrame(True, index=idx, columns=list(members))
    label_start = idx
    label_end = idx + pd.Timedelta(hours=24)
    available_at = label_end.copy()
    available_at = available_at + pd.to_timedelta(
        [0] * (n - 1) + [last_avail_shift_hours], unit="h"
    )
    return MaturedMemberReturns(
        returns=returns,
        known=known,
        label_start=label_start,
        label_end=label_end,
        available_at=available_at,
        source="daily_step_proxy",
        procedure_digest="proc",
        input_manifest_digest=None,
    )


def _legacy_schedule(data):  # type: ignore[no-untyped-def]
    from src.mhs.process import monthly_refit_schedule

    return monthly_refit_schedule(
        data.decision_grid[0], data.decision_grid[-1], min_train_days=10
    )


def _evidence(n_days: int = 45, seed: int = 5, known_gap: tuple | None = None) -> MaturedMemberReturns:
    data = _tiny_data(n_days=n_days, seed=seed, known_gap=known_gap)
    return build_proxy_member_returns(
        data,
        clock=_clock(),
        one_way_bps=8.0,
        procedure_digest="proc",
        input_manifest_digest=None,
    )


def test_build_proxy_member_returns_marks_final_interval_omitted() -> None:
    """First label starts when its book is available; the final interval is omitted."""
    data = _tiny_data(n_days=10)
    evidence = build_proxy_member_returns(
        data, clock=_clock(), one_way_bps=8.0, procedure_digest="proc", input_manifest_digest=None
    )
    assert list(evidence.returns.index) == list(data.decision_grid[:-1])
    assert evidence.label_start[0] == data.decision_grid[0] + pd.Timedelta(hours=1)
    assert evidence.source == "daily_step_proxy"
    assert evidence.procedure_digest == "proc"
    assert evidence.input_manifest_digest is None
    assert bool((evidence.label_start < evidence.label_end).all())
    assert bool((evidence.label_end <= evidence.available_at).all())


def test_build_proxy_member_returns_aggregates_each_event_once() -> None:
    """Irregular settlement: each cash-flow event belongs to exactly one interval."""
    from src.mhs.backtest.contracts import ProcessMarketData

    decisions = _decisions(200)
    grid = _hourly(decisions)
    log_close = pd.DataFrame(
        np.arange(len(decisions), dtype="float64") * 0.001,
        index=decisions,
        columns=["S0USDT"],
    )
    funding_step = pd.DataFrame(0.0, index=decisions, columns=["S0USDT"])
    funding_step.iloc[150, 0] = 0.01
    books = {"solo": pd.DataFrame(1.0, index=decisions, columns=["S0USDT"])}
    known = pd.DataFrame(True, index=grid, columns=["S0USDT"])
    opens = pd.DataFrame(np.exp(log_close.to_numpy()), index=decisions, columns=["S0USDT"])
    shifted = funding_step.copy()
    shifted.iloc[150, 0] = 0.02

    def _build(funding: pd.DataFrame) -> MaturedMemberReturns:
        data = ProcessMarketData(
            grid_1h=grid,
            decision_grid=decisions,
            opens_1h=opens,
            bar_funding_1h=pd.DataFrame(0.0, index=grid, columns=["S0USDT"]),
            log_close_step=log_close,
            funding_step=funding,
            member_books=books,
            execution_mask=pd.DataFrame(True, index=decisions, columns=["S0USDT"]),
            funding_known_1h=known,
        )
        return build_proxy_member_returns(
            data, clock=_clock(), one_way_bps=0.0, procedure_digest="p", input_manifest_digest=None
        )

    before = _build(funding_step)
    after = _build(shifted)
    assert bool(before.known.to_numpy().all())
    changed = before.returns["solo"].ne(after.returns["solo"])
    assert int(changed.sum()) == 1
    assert changed.index[int(np.flatnonzero(changed.to_numpy())[0])] == decisions[150]


def test_build_proxy_member_returns_marks_unknown_financing() -> None:
    """Unknown financing: a held symbol without funding knowledge yields unknown labels."""
    data = _tiny_data(n_days=45, seed=5)
    gap_day = data.decision_grid[20]
    evidence = _evidence(
        n_days=45, seed=5, known_gap=("S2USDT", gap_day, gap_day + pd.Timedelta(hours=24))
    )
    row = evidence.returns.index.get_loc(gap_day)
    assert not bool(evidence.known.loc[gap_day, "carry"])
    assert bool(np.isnan(evidence.returns.loc[gap_day, "carry"]))
    assert bool(evidence.known.loc[gap_day, "mom"])
    assert bool(np.isfinite(evidence.returns.loc[gap_day, "mom"]))
    assert row == 20


def test_build_proxy_member_returns_rejects_invalid_clock() -> None:
    data = _tiny_data(n_days=10)
    for clock in (_clock(period_hours=0), _clock(lag_hours=0), _clock(latency_hours=-1)):
        with pytest.raises(DataIntegrityError, match="clock"):
            build_proxy_member_returns(
                data, clock=clock, one_way_bps=8.0, procedure_digest="p", input_manifest_digest=None
            )


def test_build_proxy_member_returns_rejects_bad_friction() -> None:
    data = _tiny_data(n_days=10)
    for bad in ("bad", -1.0, float("inf"), float("nan")):
        with pytest.raises(DataIntegrityError, match="one_way_bps"):
            build_proxy_member_returns(
                data, clock=_clock(), one_way_bps=bad, procedure_digest="p",  # type: ignore[arg-type]
                input_manifest_digest=None,
            )


def test_build_proxy_member_returns_rejects_empty_procedure_digest() -> None:
    data = _tiny_data(n_days=10)
    for bad in ("", None):
        with pytest.raises(DataIntegrityError, match="procedure_digest"):
            build_proxy_member_returns(
                data, clock=_clock(), one_way_bps=8.0, procedure_digest=bad,  # type: ignore[arg-type]
                input_manifest_digest=None,
            )


def test_build_proxy_member_returns_rejects_bad_manifest() -> None:
    data = _tiny_data(n_days=10)
    for bad in ("", 123):
        with pytest.raises(DataIntegrityError, match="input_manifest_digest"):
            build_proxy_member_returns(
                data, clock=_clock(), one_way_bps=8.0, procedure_digest="p",
                input_manifest_digest=bad,  # type: ignore[arg-type]
            )


def test_build_proxy_member_returns_rejects_empty_books() -> None:
    import dataclasses

    data = _tiny_data(n_days=10)
    empty = dataclasses.replace(data, member_books={})
    with pytest.raises(DataIntegrityError, match="candidate book"):
        build_proxy_member_returns(
            empty, clock=_clock(), one_way_bps=8.0, procedure_digest="p", input_manifest_digest=None
        )


def test_build_proxy_member_returns_rejects_misaligned_inputs() -> None:
    import dataclasses

    data = _tiny_data(n_days=10)
    shifted = dataclasses.replace(
        data, funding_step=data.funding_step.shift(1, freq="24h")
    )
    with pytest.raises(DataIntegrityError, match="decision labels"):
        build_proxy_member_returns(
            shifted, clock=_clock(), one_way_bps=8.0, procedure_digest="p", input_manifest_digest=None
        )
    renamed = {name: book.rename(columns={"S0USDT": "ZZZ"}) for name, book in data.member_books.items()}
    shuffled = dataclasses.replace(data, member_books=renamed)
    with pytest.raises(DataIntegrityError, match="candidate book"):
        build_proxy_member_returns(
            shuffled, clock=_clock(), one_way_bps=8.0, procedure_digest="p", input_manifest_digest=None
        )


def test_build_proxy_member_returns_rejects_missing_funding_knowledge() -> None:
    import dataclasses

    data = _tiny_data(n_days=10)
    for knowledge in (None, data.funding_known_1h.drop(columns=["S0USDT"])):
        missing = dataclasses.replace(data, funding_known_1h=knowledge)
        with pytest.raises(DataIntegrityError, match="funding knowledge"):
            build_proxy_member_returns(
                missing, clock=_clock(), one_way_bps=8.0, procedure_digest="p",
                input_manifest_digest=None,
            )


def test_select_matured_training_returns_excludes_unfinished_label() -> None:
    """Incomplete next-day label: final funding at T+25h is excluded at T+24h."""
    labels = _hand_labels(n=3, last_avail_shift_hours=1)
    cutoff = pd.Timestamp("2021-01-04", tz="UTC")
    picked = select_matured_training_returns(labels, fit_cutoff=cutoff)
    assert list(picked.index) == list(labels.returns.index[:2])


def test_select_matured_training_returns_admits_exact_cutoff_equality() -> None:
    """Label equality: end and availability equal to cutoff are admitted."""
    labels = _hand_labels(n=3, last_avail_shift_hours=1)
    cutoff = pd.Timestamp("2021-01-03", tz="UTC")
    picked = select_matured_training_returns(labels, fit_cutoff=cutoff)
    assert list(picked.index) == list(labels.returns.index[:2])


def test_select_matured_training_returns_excludes_straddling_label() -> None:
    """Trailing boundary: a complete straddling label is excluded, never truncated."""
    labels = _hand_labels(n=3, last_avail_shift_hours=0)
    picked = select_matured_training_returns(
        labels,
        fit_cutoff=pd.Timestamp("2021-01-04", tz="UTC"),
        train_start=pd.Timestamp("2021-01-02", tz="UTC"),
    )
    assert list(picked.index) == list(labels.returns.index[1:])


def test_select_matured_training_returns_keeps_member_set_on_common_rows() -> None:
    """Common candidate sample: one unknown member excludes the row, not the member."""
    labels = _hand_labels(n=3, last_avail_shift_hours=0)
    labels.known.iloc[1, 1] = False
    labels.returns.iloc[1, 1] = np.nan
    picked = select_matured_training_returns(
        labels, fit_cutoff=pd.Timestamp("2021-01-04", tz="UTC")
    )
    assert list(picked.columns) == ["A", "B"]
    assert list(picked.index) == [labels.returns.index[0], labels.returns.index[2]]


def test_select_matured_training_returns_ignores_future_perturbation() -> None:
    """Future return perturbation: later labels cannot move the selected matrix."""
    labels = _hand_labels(n=3, last_avail_shift_hours=24)
    import dataclasses

    later_returns = labels.returns.copy()
    later_returns.iloc[2] = later_returns.iloc[2] * 17.0
    later_avail = labels.available_at.copy()
    later_avail = later_avail + pd.to_timedelta([0, 0, 5], unit="h")
    perturbed = dataclasses.replace(
        labels, returns=later_returns, available_at=later_avail
    )
    cutoff = pd.Timestamp("2021-01-03", tz="UTC")
    pd.testing.assert_frame_equal(
        select_matured_training_returns(perturbed, fit_cutoff=cutoff),
        select_matured_training_returns(labels, fit_cutoff=cutoff),
    )


def test_select_matured_training_returns_rejects_mismatched_members() -> None:
    labels = _hand_labels(n=3)
    import dataclasses

    forged = dataclasses.replace(labels, known=labels.known.rename(columns={"B": "C"}))
    with pytest.raises(DataIntegrityError, match="member columns"):
        select_matured_training_returns(forged, fit_cutoff=pd.Timestamp("2021-01-04", tz="UTC"))


def test_select_matured_training_returns_rejects_misaligned_intervals() -> None:
    labels = _hand_labels(n=3)
    import dataclasses

    forged = dataclasses.replace(labels, label_end=labels.label_end[:-1])
    with pytest.raises(DataIntegrityError, match="align exactly"):
        select_matured_training_returns(forged, fit_cutoff=pd.Timestamp("2021-01-04", tz="UTC"))


def test_select_matured_training_returns_rejects_inverted_intervals() -> None:
    labels = _hand_labels(n=3)
    import dataclasses

    with pytest.raises(DataIntegrityError, match="label_start"):
        select_matured_training_returns(
            dataclasses.replace(labels, label_start=labels.label_end),
            fit_cutoff=pd.Timestamp("2021-01-04", tz="UTC"),
        )
    with pytest.raises(DataIntegrityError, match="label_start"):
        select_matured_training_returns(
            dataclasses.replace(labels, available_at=labels.label_start),
            fit_cutoff=pd.Timestamp("2021-01-04", tz="UTC"),
        )


def test_select_matured_training_returns_rejects_known_nonfinite_label() -> None:
    labels = _hand_labels(n=3)
    labels.returns.iloc[0, 0] = np.nan
    with pytest.raises(DataIntegrityError, match="minus one"):
        select_matured_training_returns(labels, fit_cutoff=pd.Timestamp("2021-01-04", tz="UTC"))
    labels.returns.iloc[0, 0] = -1.0
    with pytest.raises(DataIntegrityError, match="minus one"):
        select_matured_training_returns(labels, fit_cutoff=pd.Timestamp("2021-01-04", tz="UTC"))


def test_select_matured_training_returns_rejects_unknown_finite_label() -> None:
    labels = _hand_labels(n=3)
    labels.known.iloc[0, 0] = False
    with pytest.raises(DataIntegrityError, match="missing"):
        select_matured_training_returns(labels, fit_cutoff=pd.Timestamp("2021-01-04", tz="UTC"))


def test_select_matured_training_returns_rejects_naive_cutoff() -> None:
    labels = _hand_labels(n=3)
    with pytest.raises(DataIntegrityError, match="fit_cutoff"):
        select_matured_training_returns(labels, fit_cutoff=pd.Timestamp("2021-01-04"))


def test_select_matured_training_returns_rejects_naive_train_start() -> None:
    labels = _hand_labels(n=3)
    with pytest.raises(DataIntegrityError, match="train_start"):
        select_matured_training_returns(
            labels,
            fit_cutoff=pd.Timestamp("2021-01-04", tz="UTC"),
            train_start=pd.Timestamp("2021-01-02"),
        )


def test_select_matured_training_returns_rejects_inverted_window() -> None:
    labels = _hand_labels(n=3)
    with pytest.raises(DataIntegrityError, match="train_start"):
        select_matured_training_returns(
            labels,
            fit_cutoff=pd.Timestamp("2021-01-02", tz="UTC"),
            train_start=pd.Timestamp("2021-01-03", tz="UTC"),
        )


def _schedule_labels(n: int = 400, start: str = "2021-01-01") -> MaturedMemberReturns:
    idx = _decisions(n, start)
    returns = pd.DataFrame(0.001, index=idx, columns=["A"])
    known = pd.DataFrame(True, index=idx, columns=["A"])
    return MaturedMemberReturns(
        returns=returns,
        known=known,
        label_start=idx,
        label_end=idx + pd.Timedelta(hours=24),
        available_at=idx + pd.Timedelta(hours=24),
        source="daily_step_proxy",
        procedure_digest="p",
        input_manifest_digest=None,
    )


def test_matured_schedule_ignores_feature_lookback_embargo() -> None:
    """Warmup is not embargo: a 720-hour lookback adds no forward blackout."""
    from src.mhs.process import matured_monthly_refit_schedule

    labels = _schedule_labels(n=400)
    schedule = matured_monthly_refit_schedule(
        labels,
        pd.Timestamp("2022-02-05", tz="UTC"),
        min_train_days=365,
        fit_latency=pd.Timedelta(0),
    )
    assert len(schedule) >= 1
    assert schedule[0].effective_from <= pd.Timestamp("2022-04-01", tz="UTC")
    assert schedule[0].train_end == schedule[0].effective_from
    from itertools import pairwise

    for first, second in pairwise(schedule):
        assert first.effective_to == second.effective_from


def test_matured_schedule_derives_cutoff_from_fit_latency() -> None:
    """Nonzero fit latency: no training observation is later than start minus latency."""
    from src.mhs.process import matured_monthly_refit_schedule

    labels = _schedule_labels(n=60)
    latency = pd.Timedelta(hours=2)
    schedule = matured_monthly_refit_schedule(
        labels,
        labels.label_end[-1],
        min_train_days=10,
        fit_latency=latency,
    )
    assert len(schedule) >= 1
    for point in schedule:
        assert point.train_end == point.effective_from - latency
    first = schedule[0]
    train = select_matured_training_returns(labels, fit_cutoff=first.train_end)
    assert len(train) >= 10
    selected = labels.returns.index.isin(train.index)
    assert bool((labels.available_at[selected] <= first.train_end).all())


def test_matured_schedule_rejects_insufficient_evidence() -> None:
    """Sample shortage: insufficient evidence is explicit, never an arbitrary fallback."""
    from src.mhs.process import matured_monthly_refit_schedule

    labels = _schedule_labels(n=10)
    with pytest.raises(ValueError, match="sufficient"):
        matured_monthly_refit_schedule(
            labels,
            labels.label_end[-1],
            min_train_days=365,
            fit_latency=pd.Timedelta(0),
        )


def test_matured_schedule_rejects_invalid_controls() -> None:
    from src.mhs.process import matured_monthly_refit_schedule

    labels = _schedule_labels(n=60)
    with pytest.raises(ValueError, match="tz-aware"):
        matured_monthly_refit_schedule(
            labels, pd.Timestamp("2021-03-01"), min_train_days=10, fit_latency=pd.Timedelta(0)
        )
    for bad in (0, -3, True):
        with pytest.raises(ValueError, match="positive integer"):
            matured_monthly_refit_schedule(
                labels, labels.label_end[-1], min_train_days=bad, fit_latency=pd.Timedelta(0)  # type: ignore[arg-type]
            )
    for bad in (pd.Timedelta(hours=-1), "2h"):
        with pytest.raises(ValueError, match="nonnegative"):
            matured_monthly_refit_schedule(
                labels, labels.label_end[-1], min_train_days=10, fit_latency=bad  # type: ignore[arg-type]
            )
    empty = _schedule_labels(n=60).returns.iloc[:0]
    import dataclasses

    with pytest.raises(ValueError, match="sufficient"):
        matured_monthly_refit_schedule(
            dataclasses.replace(_schedule_labels(n=60), returns=empty, known=empty.iloc[:, :]),
            labels.label_end[-1],
            min_train_days=10,
            fit_latency=pd.Timedelta(0),
        )


def test_matured_schedule_rejects_misaligned_evidence() -> None:
    from src.mhs.process import matured_monthly_refit_schedule

    import dataclasses

    labels = _schedule_labels(n=60)
    forged = dataclasses.replace(labels, label_start=labels.label_start[:-1])
    with pytest.raises(DataIntegrityError, match="align exactly"):
        matured_monthly_refit_schedule(
            forged, labels.label_end[-1], min_train_days=10, fit_latency=pd.Timedelta(0)
        )


def test_run_process_paths_fits_only_matured_selection() -> None:
    """Causal wiring: estimators consume exactly the matured training matrix."""
    from src.mhs.backtest.paths import run_process_paths
    from src.mhs.process import (
        RefitPoint,
        estimation_adjusted_mean,
        ledoit_wolf_covariance,
        long_only_growth_weights,
    )

    data = _tiny_data(n_days=45, seed=5)
    evidence = _evidence(n_days=45, seed=5)
    cutoff = data.decision_grid[30]
    point = RefitPoint(
        effective_from=data.decision_grid[31],
        effective_to=data.decision_grid[40],
        train_end=cutoff,
    )
    (path,) = run_process_paths(
        data,
        (point,),
        decision_bps=8.0,
        evaluation_bps=(8.0,),
        leverage_cap=2.0,
        clock=_clock(),
        member_evidence=evidence,
    )
    assert path.clock_mode == "matured_labels"
    train = select_matured_training_returns(evidence, fit_cutoff=cutoff)
    expected = long_only_growth_weights(
        estimation_adjusted_mean(train), ledoit_wolf_covariance(train)
    )
    assert set(path.refits[0].member_weights) == {n for n in expected.index if float(expected[n]) != 0.0}
    for name, weight in path.refits[0].member_weights.items():
        assert weight == pytest.approx(float(expected[name]))


def test_run_process_paths_rejects_mixed_clock_controls() -> None:
    from src.mhs.backtest.paths import run_process_paths

    data = _tiny_data(n_days=90, seed=5)
    schedule = _legacy_schedule(data)
    with pytest.raises(ValueError, match="together"):
        run_process_paths(
            data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0,
            clock=_clock(),
        )
    with pytest.raises(ValueError, match="together"):
        run_process_paths(
            data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0,
            member_evidence=_evidence(n_days=45, seed=5),
        )


def test_run_process_paths_rejects_mismatched_evidence_members() -> None:
    from src.mhs.backtest.paths import run_process_paths

    import dataclasses

    data = _tiny_data(n_days=90, seed=5)
    evidence = _evidence(n_days=90, seed=5)
    forged = dataclasses.replace(evidence, returns=evidence.returns.rename(columns={"mom": "zzz"}))
    schedule = _legacy_schedule(data)
    with pytest.raises(DataIntegrityError, match="candidate books"):
        run_process_paths(
            data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0,
            clock=_clock(), member_evidence=forged,
        )


def test_run_process_paths_yields_no_trade_on_insufficient_sample() -> None:
    """Sample shortage at refit: explicit zero allocation, never a frozen profit."""
    from src.mhs.backtest.paths import run_process_paths
    from src.mhs.process import RefitPoint

    data = _tiny_data(n_days=6, seed=5)
    evidence = _evidence(n_days=6, seed=5)
    point = RefitPoint(
        effective_from=data.decision_grid[4],
        effective_to=data.decision_grid[5],
        train_end=evidence.label_end[0],
    )
    (path,) = run_process_paths(
        data,
        (point,),
        decision_bps=8.0,
        evaluation_bps=(8.0,),
        leverage_cap=2.0,
        clock=_clock(),
        member_evidence=evidence,
    )
    assert path.refits[0].member_weights == {}
    assert bool((path.target_weights == 0.0).all().all())


def test_run_process_paths_retains_legacy_path_without_controls() -> None:
    from src.mhs.backtest.paths import run_process_paths

    data = _tiny_data(n_days=90, seed=5)
    schedule = _legacy_schedule(data)
    first = run_process_paths(
        data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0
    )[0]
    assert first.clock_mode == "legacy_purge"
    assert first.signal_available_at is None
    second = run_process_paths(
        data, schedule, decision_bps=8.0, evaluation_bps=(8.0,), leverage_cap=2.0
    )[0]
    pd.testing.assert_frame_equal(first.target_weights, second.target_weights)


def test_run_process_paths_carries_signal_clock_on_causal_path() -> None:
    from src.mhs.backtest.paths import run_process_paths
    from src.mhs.process import RefitPoint

    data = _tiny_data(n_days=45, seed=5)
    evidence = _evidence(n_days=45, seed=5)
    point = RefitPoint(
        effective_from=data.decision_grid[30],
        effective_to=data.decision_grid[40],
        train_end=data.decision_grid[30],
    )
    (path,) = run_process_paths(
        data,
        (point,),
        decision_bps=8.0,
        evaluation_bps=(8.0,),
        leverage_cap=2.0,
        clock=_clock(),
        member_evidence=evidence,
    )
    assert path.signal_available_at is not None
    assert len(path.signal_available_at) == len(path.target_weights)
    assert path.signal_available_at.equals(
        pd.DatetimeIndex(path.target_weights.index + pd.Timedelta(hours=1))
    )
    assert bool((path.signal_available_at >= path.target_weights.index).all())


def test_evaluate_builds_causal_evidence_with_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.mhs.backtest.paths as bt_paths

    data = _tiny_data(n_days=60, seed=9)
    monkeypatch.setattr(bt_paths, "load_process_market_data", lambda *a, **k: data)
    monkeypatch.setattr(bt_paths, "PROCESS_MIN_TRAIN_DAYS", 30)
    seen: dict = {}
    real_run = bt_paths.run_process_paths

    def _spy(*args: object, **kwargs: object) -> object:  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(bt_paths, "run_process_paths", _spy)
    report = bt_paths.evaluate_process_backtest(data.decision_grid[0], data.decision_grid[-1])
    assert seen.get("clock") is not None
    assert seen.get("member_evidence") is not None
    assert report.base.clock_mode == "matured_labels"
    assert report.base.signal_available_at is not None
    assert len(report.base.signal_available_at) == len(report.base.target_weights)
    assert report.base.signal_available_at[0] == report.base.target_weights.index[0] + pd.Timedelta(hours=1)
    for refit in report.base.refits:
        assert refit.point.train_end == refit.point.effective_from
