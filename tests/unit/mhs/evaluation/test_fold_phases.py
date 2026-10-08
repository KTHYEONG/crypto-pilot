"""Invariant scenarios for phased fold orchestration (inline executor + stubs)."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event

import pandas as pd

import src.mhs.evaluation.folds as folds_mod
import src.core.parallel as parallel_mod
from src.common.errors import DataIntegrityError
from src.mhs.evaluation.folds import (
    _FoldValidationPlan,
    _complete_anchored_folds,
    _fold_failure_report,
    _submit_fold_validation_phase,
)
from src.mhs.evidence import AnchoredPurgedFold

_TRAIN_START = pd.Timestamp("2021-01-01", tz="UTC")


def _fold(train_end: str) -> AnchoredPurgedFold:
    te = pd.Timestamp(train_end, tz="UTC")
    return AnchoredPurgedFold(
        train_start=_TRAIN_START,
        train_end=te,
        validation_start=te + pd.Timedelta(days=8),
        validation_end=te + pd.Timedelta(days=20),
        forward_dependency_hours=168,
        purge_hours=168,
    )


def _plan() -> _FoldValidationPlan:
    return _FoldValidationPlan(
        target_weights=pd.DataFrame(),
        target_replay=pd.DataFrame(),
        signal_available_at=pd.DatetimeIndex([], tz="UTC"),
        terminal_censored=0,
        decision_intents=0,
    )


class _InlineExecutor:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> _InlineExecutor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future


class _ReverseExecutor(_InlineExecutor):
    def __init__(self) -> None:
        self.pending = []
        self.completion_order = []

    def submit(self, fn, /, *args, **kwargs):
        if fn is not folds_mod._fold_validation_task:
            return super().submit(fn, *args, **kwargs)
        future = Future()
        self.pending.append((future, fn, args, kwargs))
        return future

    def complete_reverse(self) -> None:
        for future, fn, args, kwargs in reversed(self.pending):
            future.set_result(fn(*args, **kwargs))
            self.completion_order.append(args[4])


def _report(fold: AnchoredPurgedFold, idx: int):
    return folds_mod._incomplete_fold_report(fold, idx, ())


def test_unusable_folds_never_reach_reference_phases(monkeypatch) -> None:
    folds = [_fold("2021-05-20"), _fold("2021-06-01"), _fold("2021-06-10"), _fold("2021-06-20")]
    bad_ends = {pd.Timestamp("2021-06-01", tz="UTC"), pd.Timestamp("2021-06-20", tz="UTC")}

    def _validation(root, fold, request, funding, slow, committee, *, committee_admission=None):
        if fold.train_end in bad_ends:
            raise ValueError("no symbol survived the panel filters")
        return _plan()

    shared_calls: list[tuple] = []
    run_calls: list[int] = []

    def _shared(root, group_folds, request, funding, equity, slow, committee, *, committee_admissions=None):
        shared_calls.append(tuple(idx for idx, _ in group_folds))
        return None

    def _run(root, fold, request, funding, equity, idx, telemetry=None, slow=None,
            fast=None, carry=None, committee=None, validation_plan=None, shared_reference=None,
            committee_admission=None):
        run_calls.append(idx)
        return _report(fold, idx)

    monkeypatch.setattr(folds_mod, "_build_fold_validation_plan", _validation)
    monkeypatch.setattr(folds_mod, "_build_shared_train_reference", _shared)
    monkeypatch.setattr(folds_mod, "_run_anchored_fold", _run)
    with _InlineExecutor() as pool, parallel_mod.fork_shared_payload({"fold_funding": {}}) as token:
        validation_futures = _submit_fold_validation_phase(pool, token, "root", folds, object(), None, None)
        ordered = _complete_anchored_folds(
            pool, token, validation_futures, "root", folds, object(), 1.0, None, None, None, None,
        )
    assert len(ordered) == 4
    assert ordered[1].failures == ("INCOMPLETE_ANCHORED_FOLD",)
    assert ordered[3].failures == ("INCOMPLETE_ANCHORED_FOLD",)
    assert shared_calls == [(0, 2)]
    assert sorted(run_calls) == [0, 2]


def test_group_horizon_ignores_unusable_latest_fold(monkeypatch) -> None:
    folds = [_fold("2021-05-20"), _fold("2021-06-01"), _fold("2021-06-10"), _fold("2021-06-20")]

    def _validation(root, fold, request, funding, slow, committee, *, committee_admission=None):
        if fold.train_end == pd.Timestamp("2021-06-20", tz="UTC"):
            raise ValueError("unusable")
        return _plan()

    shared_calls: list[tuple] = []
    monkeypatch.setattr(folds_mod, "_build_fold_validation_plan", _validation)
    monkeypatch.setattr(folds_mod, "_build_shared_train_reference",
                        lambda root, group_folds, request, funding, equity, slow, committee, **_k: (
                            shared_calls.append(tuple(idx for idx, _ in group_folds)), None)[1])
    monkeypatch.setattr(folds_mod, "_run_anchored_fold", lambda *a, **k: _report(a[1], a[5]))
    with _InlineExecutor() as pool, parallel_mod.fork_shared_payload({"fold_funding": {}}) as token:
        validation_futures = _submit_fold_validation_phase(pool, token, "root", folds, object(), None, None)
        _complete_anchored_folds(pool, token, validation_futures, "root", folds, object(), 1.0, None, None, None, None)
    assert shared_calls == [(0, 1, 2)]


def test_singletons_and_empty_windows_run_independently(monkeypatch) -> None:
    folds = [
        _fold("2021-06-20"),
        AnchoredPurgedFold(
            train_start=_TRAIN_START,
            train_end=pd.Timestamp("2021-01-10", tz="UTC"),
            validation_start=pd.Timestamp("2021-02-01", tz="UTC"),
            validation_end=pd.Timestamp("2021-03-01", tz="UTC"),
            forward_dependency_hours=168,
            purge_hours=168,
        ),
    ]
    slow = {0: 168, 1: None}
    monkeypatch.setattr(folds_mod, "_build_fold_validation_plan", lambda *a, **k: _plan())
    shared_calls: list = []
    monkeypatch.setattr(folds_mod, "_build_shared_train_reference",
                        lambda *a, **k: (shared_calls.append(a), None)[1])
    seen: list[tuple] = []
    monkeypatch.setattr(
        folds_mod, "_run_anchored_fold",
        lambda *a, **k: (seen.append((a[5], k.get("shared_reference"))), _report(a[1], a[5]))[1],
    )
    with _InlineExecutor() as pool, parallel_mod.fork_shared_payload({"fold_funding": {}}) as token:
        validation_futures = _submit_fold_validation_phase(pool, token, "root", folds, object(), slow, None)
        _complete_anchored_folds(pool, token, validation_futures, "root", folds, object(), 1.0, slow, None, None, None)
    assert shared_calls == []
    assert all(shared is None for _, shared in seen)
    assert sorted(idx for idx, _ in seen) == [0, 1]


def test_committee_weights_split_groups(monkeypatch) -> None:
    folds = [_fold("2021-05-20"), _fold("2021-06-10"), _fold("2021-06-20")]
    weights = {0: {"a": 0.5}, 1: {"a": 0.6}, 2: {"a": 0.7}}
    monkeypatch.setattr(folds_mod, "_build_fold_validation_plan", lambda *a, **k: _plan())
    shared_calls: list = []
    monkeypatch.setattr(folds_mod, "_build_shared_train_reference",
                        lambda *a, **k: (shared_calls.append(a), None)[1])
    monkeypatch.setattr(folds_mod, "_run_anchored_fold", lambda *a, **k: _report(a[1], a[5]))
    with _InlineExecutor() as pool, parallel_mod.fork_shared_payload({"fold_funding": {}}) as token:
        validation_futures = _submit_fold_validation_phase(pool, token, "root", folds, object(), None, weights)
        _complete_anchored_folds(pool, token, validation_futures, "root", folds, object(), 1.0, None, None, None, weights)
    assert shared_calls == []


def test_each_fold_receives_own_plan_and_group_reference(monkeypatch) -> None:
    folds = [_fold("2021-05-20"), _fold("2021-06-01"), _fold("2021-06-10"), _fold("2021-06-20")]
    slow = {0: 168, 1: 168, 2: None, 3: None}
    fast = {i: (24 + i, "frozen_default") for i in range(4)}
    carry = dict.fromkeys(range(4))
    sentinel_a, sentinel_b = object(), object()
    monkeypatch.setattr(folds_mod, "_build_fold_validation_plan", lambda *a, **k: _plan())
    monkeypatch.setattr(
        folds_mod, "_build_shared_train_reference",
        lambda root, group_folds, request, funding, equity, slow_o, committee, **_k: (
            sentinel_a if slow_o == 168 else sentinel_b),
    )
    seen: dict[int, tuple] = {}
    monkeypatch.setattr(
        folds_mod, "_run_anchored_fold",
        lambda *a, **k: (seen.__setitem__(a[5], (k.get("validation_plan"), k.get("shared_reference"),
                                                 a[7], a[8], a[9])), _report(a[1], a[5]))[1],
    )
    with _InlineExecutor() as pool, parallel_mod.fork_shared_payload({"fold_funding": {}}) as token:
        validation_futures = _submit_fold_validation_phase(pool, token, "root", folds, object(), slow, None)
        _complete_anchored_folds(pool, token, validation_futures, "root", folds, object(), 1.0, slow, fast, carry, None)
    assert seen[0][1] is sentinel_a
    assert seen[1][1] is sentinel_a
    assert seen[2][1] is sentinel_b
    assert seen[3][1] is sentinel_b
    assert seen[0][2] == 168
    assert seen[2][3] == (24 + 2, "frozen_default")


def test_reports_ordered_by_fold_index(monkeypatch) -> None:
    folds = [_fold("2021-05-20"), _fold("2021-06-10"), _fold("2021-06-20")]
    monkeypatch.setattr(folds_mod, "_build_fold_validation_plan", lambda *a, **k: _plan())
    monkeypatch.setattr(folds_mod, "_build_shared_train_reference", lambda *a, **k: None)
    monkeypatch.setattr(folds_mod, "_run_anchored_fold", lambda *a, **k: _report(a[1], a[5]))
    with _ReverseExecutor() as pool, parallel_mod.fork_shared_payload({"fold_funding": {}}) as token:
        validation_futures = _submit_fold_validation_phase(pool, token, "root", folds, object(), None, None)
        assert not any(future.done() for future in validation_futures)
        pool.complete_reverse()
        assert pool.completion_order == [2, 1, 0]
        ordered = _complete_anchored_folds(
            pool, token, validation_futures, "root", folds, object(), 1.0, None, None, None, None,
        )
    assert [r.fold_index for r in ordered] == [0, 1, 2]
    assert len(ordered) == 3


def test_validation_phase_failure_report_matches_in_fold_mapping(monkeypatch) -> None:
    fold = _fold("2021-05-20")
    exc = DataIntegrityError("x missing")
    monkeypatch.setattr(
        folds_mod, "_build_fold_validation_plan",
        lambda *a, **k: (_ for _ in ()).throw(DataIntegrityError("x missing")),
    )
    with parallel_mod.fork_shared_payload({"fold_funding": {}}) as token:
        report = folds_mod._fold_validation_task(token, "r", fold, object(), 0, None, None)
    assert report.failures == ("RELEVANT_EXECUTION_DATA_GAP",)
    assert report.failures == _fold_failure_report(fold, 0, exc).failures


def test_ready_group_executes_before_earlier_group_completes(monkeypatch) -> None:
    folds = [_fold("2021-05-20"), _fold("2021-06-01"), _fold("2021-06-10"), _fold("2021-06-20")]
    slow = {0: 168, 1: 168, 2: None, 3: None}
    later_group_executed = Event()
    monkeypatch.setattr(folds_mod, "_build_fold_validation_plan", lambda *a, **k: _plan())

    def _shared(root, group_folds, request, funding, equity, slow_override, committee, **_k):
        if slow_override == 168:
            assert later_group_executed.wait(timeout=5.0), "ready group was blocked by earlier group"
        return None

    def _run(*args, **kwargs):
        if args[5] == 2:
            later_group_executed.set()
        return _report(args[1], args[5])

    monkeypatch.setattr(folds_mod, "_build_shared_train_reference", _shared)
    monkeypatch.setattr(folds_mod, "_run_anchored_fold", _run)
    with parallel_mod.fork_shared_payload({"fold_funding": {}}) as token, ThreadPoolExecutor(max_workers=2) as pool:
        validation_futures = _submit_fold_validation_phase(pool, token, "root", folds, object(), slow, None)
        ordered = _complete_anchored_folds(
            pool, token, validation_futures, "root", folds, object(), 1.0, slow, None, None, None,
        )
    assert later_group_executed.is_set()
    assert [report.fold_index for report in ordered] == [0, 1, 2, 3]


def test_funding_travels_through_fork_shared_payload(monkeypatch) -> None:
    folds = [_fold("2021-05-20"), _fold("2021-06-20")]
    funding = {"A": pd.Series([1.0])}
    monkeypatch.setattr(folds_mod, "_build_fold_validation_plan",
                        lambda root, fold, request, f, slow, committee, **_k: (_plan(), f)[0] if f is funding else (_ for _ in ()).throw(AssertionError("funding not shared")))
    seen_ids: list[int] = []
    real_validation = folds_mod._fold_validation_task

    def _rec_validation(token, *a, **k):
        seen_ids.append(id(parallel_mod.resolve_fork_shared(token)["fold_funding"]))
        return real_validation(token, *a, **k)

    monkeypatch.setattr(folds_mod, "_fold_validation_task", _rec_validation)
    monkeypatch.setattr(folds_mod, "_build_shared_train_reference", lambda *a, **k: None)
    monkeypatch.setattr(folds_mod, "_run_anchored_fold", lambda *a, **k: _report(a[1], a[5]))
    submissions: list[tuple] = []
    pool = _InlineExecutor()
    orig_submit = pool.submit

    def _rec_submit(fn, /, *args, **kwargs):
        submissions.append((fn, args, kwargs))
        return orig_submit(fn, *args, **kwargs)

    pool.submit = _rec_submit  # type: ignore[method-assign]
    with parallel_mod.fork_shared_payload({"fold_funding": funding}) as token:
        validation_futures = _submit_fold_validation_phase(pool, token, "root", folds, object(), None, None)
        _complete_anchored_folds(pool, token, validation_futures, "root", folds, object(), 1.0, None, None, None, None)
    assert seen_ids
    assert all(i == id(funding) for i in seen_ids)
    for _, args, kwargs in submissions:
        for value in (*args, *kwargs.values()):
            assert value is not funding


def test_each_fold_receives_own_admission(monkeypatch) -> None:
    # I-FOLD-ADMISSION-PIT: fold i receives (fold_committee_admission or {}).get(i)
    # in phases V and E -- never another fold's admission.
    from src.strategy.features import FeatureAdmission

    folds = [_fold("2021-05-20"), _fold("2021-06-10"), _fold("2021-06-20")]
    admissions = {
        i: FeatureAdmission(cutoff=fold.train_end, admitted=(f"member_{i}",))
        for i, fold in enumerate(folds)
    }
    seen_validation: dict[int, object] = {}

    def _validation(root, fold, request, funding, slow, committee, *, committee_admission=None):
        idx = next(i for i, f in enumerate(folds) if f is fold)
        seen_validation[idx] = committee_admission
        return _plan()

    monkeypatch.setattr(folds_mod, "_build_fold_validation_plan", _validation)
    monkeypatch.setattr(folds_mod, "_build_shared_train_reference", lambda *a, **k: None)
    seen_execution: dict[int, object] = {}
    monkeypatch.setattr(
        folds_mod, "_run_anchored_fold",
        lambda *a, **k: (seen_execution.__setitem__(a[5], k.get("committee_admission")),
                         _report(a[1], a[5]))[1],
    )
    with _InlineExecutor() as pool, parallel_mod.fork_shared_payload({"fold_funding": {}}) as token:
        validation_futures = _submit_fold_validation_phase(
            pool, token, "root", folds, object(), None, None,
            fold_committee_admission=admissions,
        )
        _complete_anchored_folds(
            pool, token, validation_futures, "root", folds, object(), 1.0,
            None, None, None, None, fold_committee_admission=admissions,
        )
    assert set(seen_validation) == {0, 1, 2}
    assert set(seen_execution) == {0, 1, 2}
    for i in range(3):
        assert seen_validation[i] is admissions[i]
        assert seen_execution[i] is admissions[i]


def test_missing_committee_admission_yields_dedicated_fold_code(monkeypatch) -> None:
    # I-FOLD-ADMISSION-PIT: a committee request with committee_admission=None
    # fails closed before any panel I/O with the dedicated fold code, and no
    # train-reference replay is attempted.
    from tests.fixtures.mhs_requests import research_baseline

    fold = _fold("2021-05-20")
    request = research_baseline(committee_capital=True)

    def _no_replay(*a: object, **k: object):
        raise AssertionError("train-reference replay must not be attempted")

    monkeypatch.setattr(folds_mod, "_fold_train_reference_returns", _no_replay)
    monkeypatch.setattr(folds_mod, "_reference_from_shared", _no_replay)
    report = folds_mod._run_anchored_fold(
        "/nonexistent-root", fold, request, {}, 1.0, 0,
        None, None, None, None, None, committee_admission=None,
    )
    assert report.strict is None
    assert report.failures == ("COMMITTEE_ADMISSION_NOT_POINT_IN_TIME",)


def test_run_anchored_fold_threads_admission_into_shared_reference(monkeypatch) -> None:
    # I-FOLD-ADMISSION-PIT: _run_anchored_fold forwards the fold's own
    # admission into the shared-reference path (else branch).
    from src.strategy.features import FeatureAdmission
    from tests.fixtures.mhs_requests import research_baseline

    admission = FeatureAdmission(
        cutoff=pd.Timestamp("2021-05-20", tz="UTC"), admitted=("a",),
    )
    seen: dict[str, object] = {}

    def _raise(*a, **k):
        seen["admission"] = k.get("committee_admission")
        raise DataIntegrityError("shared boom")

    monkeypatch.setattr(folds_mod, "_reference_from_shared", _raise)
    request = research_baseline()
    report = folds_mod._run_anchored_fold(
        "root", _fold("2021-05-20"), request, {}, 1.0, 0,
        committee_admission=admission, validation_plan=_plan(),
        shared_reference=object(),
    )
    assert seen["admission"] is admission
    assert report.strict is None
