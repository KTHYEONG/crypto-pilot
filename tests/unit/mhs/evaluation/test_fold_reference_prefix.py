"""Invariant scenarios for train-reference prefix reuse (in-memory frames only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.evaluation import folds as folds_mod
from src.mhs.evaluation.folds import (
    _SharedTrainReference,
    _build_shared_train_reference,
    _fold_failure_report,
    _incomplete_fold_report,
    _reference_from_shared,
    _shared_train_reference_slice,
    _train_reference_group_key,
)
from src.mhs.evidence import AnchoredPurgedFold
from src.core.params import (
    PNL_VOL_TARGET_BURN_IN_DAYS,
    TRAIN_REFERENCE_PREFIX_RETURN_ATOL,
    TRAIN_REFERENCE_PREFIX_TARGET_ATOL,
)
from src.core.resources import MhsResourceAdmissionError
from src.core.types import ExecutionSpec

_TRAIN_START = pd.Timestamp("2021-01-01", tz="UTC")
_REF_START = _TRAIN_START + pd.Timedelta(hours=912)
_TE_K = pd.Timestamp("2021-05-20", tz="UTC")
_TE_H = pd.Timestamp("2021-06-20", tz="UTC")


def _fold(train_end: pd.Timestamp) -> AnchoredPurgedFold:
    return AnchoredPurgedFold(
        train_start=_TRAIN_START,
        train_end=train_end,
        validation_start=train_end + pd.Timedelta(days=8),
        validation_end=train_end + pd.Timedelta(days=20),
        forward_dependency_hours=168,
        purge_hours=168,
    )


def _decision_index(end: pd.Timestamp, freq: str = "24h") -> pd.DatetimeIndex:
    return pd.date_range(_REF_START, end, freq=freq, tz="UTC")


def _frame(index: pd.DatetimeIndex, columns: list[str], seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, (len(index), len(columns)))
    return pd.DataFrame(data, index=index, columns=columns, dtype="float64")


def _daily_series(end: pd.Timestamp, rows: int = 130) -> pd.Series:
    idx = pd.date_range(end=end - pd.Timedelta(days=1), periods=rows, freq="1D", tz="UTC")
    rng = np.random.default_rng(11)
    values = np.cumsum(rng.normal(0.0002, 0.004, rows))
    return pd.Series(values, index=idx, dtype="float64")


def _shared_fixture() -> tuple[AnchoredPurgedFold, pd.DataFrame, pd.DatetimeIndex, _SharedTrainReference]:
    fold_k = _fold(_TE_K)
    full_index = _decision_index(_TE_H)
    shared_targets = _frame(full_index, ["A", "B", "C"])
    shared_targets["C"] = 0.0
    after = shared_targets.index > _TE_K
    shared_targets.loc[after, "C"] = 0.05
    own_prefix = shared_targets.loc[shared_targets.index <= _TE_K, ["A", "B"]].copy()
    signals = pd.DatetimeIndex(full_index)
    own_signals = pd.DatetimeIndex(own_prefix.index)
    daily = _daily_series(_TE_H, rows=130)
    shared = _SharedTrainReference(
        group_key=_train_reference_group_key(_fold(_TE_H), None, None),
        reference_start=_REF_START,
        horizon_end=_TE_H,
        target_weights=shared_targets,
        signal_available_at=signals,
        daily_returns=daily,
    )
    return fold_k, own_prefix, own_signals, shared


def test_group_key_excludes_report_only_overrides() -> None:
    base = _fold(_TE_K)
    other_end = _fold(_TE_H)
    key = _train_reference_group_key(base, None, None)
    assert _train_reference_group_key(other_end, None, None) == key
    assert _train_reference_group_key(base, 168, None) != key
    assert _train_reference_group_key(base, None, {"m": 0.5}) != key
    assert _train_reference_group_key(base, None, {}) != _train_reference_group_key(base, None, None)
    assert _train_reference_group_key(base, None, {"b": 1.0, "a": 2.0}) == _train_reference_group_key(
        base, None, {"a": 2.0, "b": 1.0}
    )
    assert _train_reference_group_key(base, None, {"a": 0.1 + 0.2}) != _train_reference_group_key(
        base, None, {"a": 0.3}
    )
    moved = AnchoredPurgedFold(
        train_start=pd.Timestamp("2021-02-01", tz="UTC"),
        train_end=_TE_K,
        validation_start=_TE_K + pd.Timedelta(days=8),
        validation_end=_TE_K + pd.Timedelta(days=20),
        forward_dependency_hours=168,
        purge_hours=168,
    )
    assert _train_reference_group_key(moved, None, None) != key


def test_exact_prefix_with_extra_zero_columns_is_reused() -> None:
    fold_k, own, signals, shared = _shared_fixture()
    out = _shared_train_reference_slice(shared, fold_k, 0, own, signals, ExecutionSpec())
    assert out is not None
    expected = shared.daily_returns.loc[shared.daily_returns.index < _TE_K]
    pd.testing.assert_series_equal(out, expected.astype("float64"), check_exact=True)
    assert str(out.dtype) == "float64"
    assert str(out.index.tz) == "UTC"


def test_prefix_perturbation_beyond_tolerance_forces_fallback() -> None:
    fold_k, own, signals, shared = _shared_fixture()
    perturbed = shared.target_weights.copy()
    col = "A"
    row = perturbed.index[perturbed.index <= _TE_K][3]
    perturbed.loc[row, col] = perturbed.loc[row, col] + 10 * TRAIN_REFERENCE_PREFIX_TARGET_ATOL
    bad_shared = _SharedTrainReference(
        group_key=shared.group_key, reference_start=shared.reference_start,
        horizon_end=shared.horizon_end, target_weights=perturbed,
        signal_available_at=shared.signal_available_at, daily_returns=shared.daily_returns,
    )
    assert _shared_train_reference_slice(bad_shared, fold_k, 0, own, signals, ExecutionSpec()) is None
    after_only = shared.target_weights.copy()
    after_row = after_only.index[after_only.index > _TE_K][0]
    after_only.loc[after_row, "A"] = after_only.loc[after_row, "A"] + 10 * TRAIN_REFERENCE_PREFIX_TARGET_ATOL
    good_shared = _SharedTrainReference(
        group_key=shared.group_key, reference_start=shared.reference_start,
        horizon_end=shared.horizon_end, target_weights=after_only,
        signal_available_at=shared.signal_available_at, daily_returns=shared.daily_returns,
    )
    assert _shared_train_reference_slice(good_shared, fold_k, 0, own, signals, ExecutionSpec()) is not None


def test_sub_tolerance_association_drift_is_reused() -> None:
    fold_k, own, signals, shared = _shared_fixture()
    drifted = shared.target_weights.copy()
    row = drifted.index[drifted.index <= _TE_K][5]
    drifted.loc[row, "B"] = drifted.loc[row, "B"] + TRAIN_REFERENCE_PREFIX_TARGET_ATOL / 10
    drifted_shared = _SharedTrainReference(
        group_key=shared.group_key, reference_start=shared.reference_start,
        horizon_end=shared.horizon_end, target_weights=drifted,
        signal_available_at=shared.signal_available_at, daily_returns=shared.daily_returns,
    )
    assert _shared_train_reference_slice(drifted_shared, fold_k, 0, own, signals, ExecutionSpec()) is not None


def test_active_column_nan_index_and_signal_mismatches_force_fallback() -> None:
    fold_k, own, signals, shared = _shared_fixture()
    spec = ExecutionSpec()
    own_with_extra = own.copy()
    own_with_extra["D"] = 0.0
    own_with_extra.loc[own_with_extra.index[2], "D"] = 0.01
    assert _shared_train_reference_slice(shared, fold_k, 0, own_with_extra, signals, spec) is None
    own_nan = own.copy()
    own_nan.loc[own_nan.index[1], "A"] = np.nan
    assert _shared_train_reference_slice(shared, fold_k, 0, own_nan, signals, spec) is None
    own_short = own.iloc[:-1].copy()
    signals_short = signals[:-1].copy()
    assert _shared_train_reference_slice(shared, fold_k, 0, own_short, signals_short, spec) is None
    shifted_signals = signals + pd.Timedelta(hours=1)
    assert _shared_train_reference_slice(shared, fold_k, 0, own, shifted_signals, spec) is None


def test_censored_decision_executing_before_train_end_forces_fallback() -> None:
    index = _decision_index(_TE_K, freq="1h")
    own = _frame(index, ["A"])
    signals = pd.DatetimeIndex(index + pd.Timedelta(hours=1))
    daily_idx = pd.date_range(end=_TE_K - pd.Timedelta(days=1), periods=120, freq="1D", tz="UTC")
    daily = pd.Series(0.001, index=daily_idx, dtype="float64")
    shared = _SharedTrainReference(
        group_key=_train_reference_group_key(_fold(_TE_K), None, None),
        reference_start=_REF_START, horizon_end=_TE_K,
        target_weights=own.copy(), signal_available_at=signals, daily_returns=daily,
    )
    assert _shared_train_reference_slice(
        shared, _fold(_TE_K), 0, own, signals, ExecutionSpec(passive_timeout_minutes=240),
    ) is None
    index_24 = _decision_index(_TE_K)
    own_24 = _frame(index_24, ["A"])
    signals_24 = pd.DatetimeIndex(index_24)
    daily_24 = pd.Series(
        0.001,
        index=pd.date_range(end=_TE_K - pd.Timedelta(days=1), periods=120, freq="1D", tz="UTC"),
        dtype="float64",
    )
    shared_24 = _SharedTrainReference(
        group_key=_train_reference_group_key(_fold(_TE_K), None, None),
        reference_start=_REF_START, horizon_end=_TE_K,
        target_weights=own_24.copy(), signal_available_at=signals_24, daily_returns=daily_24,
    )
    assert _shared_train_reference_slice(shared_24, _fold(_TE_K), 0, own_24, signals_24, ExecutionSpec()) is not None


def test_mid_day_train_end_forces_fallback() -> None:
    fold_k, own, signals, shared = _shared_fixture()
    midday = _fold(pd.Timestamp("2021-05-20 12:00", tz="UTC"))
    assert _shared_train_reference_slice(shared, midday, 0, own, signals, ExecutionSpec()) is None


def test_too_short_slice_falls_back() -> None:
    fold_k, own, signals, shared = _shared_fixture()
    short_daily = shared.daily_returns.iloc[: PNL_VOL_TARGET_BURN_IN_DAYS - 1].copy()
    short_shared = _SharedTrainReference(
        group_key=shared.group_key, reference_start=shared.reference_start,
        horizon_end=shared.horizon_end, target_weights=shared.target_weights,
        signal_available_at=shared.signal_available_at, daily_returns=short_daily,
    )
    assert _shared_train_reference_slice(short_shared, fold_k, 0, own, signals, ExecutionSpec()) is None


def test_foreign_group_object_is_never_consumed(monkeypatch) -> None:
    fold_k, own, signals, shared = _shared_fixture()
    foreign = _SharedTrainReference(
        group_key=_train_reference_group_key(fold_k, 168, None),
        reference_start=shared.reference_start, horizon_end=shared.horizon_end,
        target_weights=shared.target_weights, signal_available_at=shared.signal_available_at,
        daily_returns=shared.daily_returns,
    )
    calls = {"targets": 0, "replay": 0, "slice": 0}
    monkeypatch.setattr(
        folds_mod, "_fold_reference_targets", lambda *a, **k: (calls.__setitem__("targets", calls["targets"] + 1), (own, signals, ["A", "B"]))[1],
    )
    sentinel = pd.Series(0.001, index=pd.date_range("2021-02-09", periods=100, freq="1D", tz="UTC"), dtype="float64")

    def _replay(*a, **k):
        calls["replay"] += 1
        assert a[6] is own
        assert a[7] is signals
        assert a[8] == ["A", "B"]
        return sentinel

    monkeypatch.setattr(folds_mod, "_replay_fold_train_reference", _replay)

    def _slice(*a, **k):
        calls["slice"] += 1
        raise AssertionError("slice must not be consulted")

    monkeypatch.setattr(folds_mod, "_shared_train_reference_slice", _slice)
    out = _reference_from_shared("root", fold_k, object(), {}, 1.0, 0, None, None, foreign)
    pd.testing.assert_series_equal(out, sentinel)
    assert calls == {"targets": 1, "replay": 1, "slice": 0}


def test_shared_builder_never_converts_failures_into_verdicts(monkeypatch) -> None:
    fold0 = _fold(_TE_K)
    fold1 = _fold(_TE_H)
    group = ((0, fold0), (1, fold1))
    for error in (
        DataIntegrityError("fold 1: train reference ledger not certified"),
        RuntimeError("boom"),
        ValueError("bad"),
    ):
        monkeypatch.setattr(
            folds_mod, "_fold_reference_targets",
            lambda *a, _error=error, **k: (_ for _ in ()).throw(_error),
        )
        assert _build_shared_train_reference("r", group, object(), {}, 1.0, None, None) is None
    monkeypatch.setattr(
        folds_mod, "_fold_reference_targets",
        lambda *a, **k: (_ for _ in ()).throw(KeyError("unexpected")),
    )
    with pytest.raises(KeyError):
        _build_shared_train_reference("r", group, object(), {}, 1.0, None, None)
    with pytest.raises(ValueError, match="at least two folds"):
        _build_shared_train_reference("r", ((0, fold0),), object(), {}, 1.0, None, None)
    moved = AnchoredPurgedFold(
        train_start=pd.Timestamp("2021-02-01", tz="UTC"), train_end=_TE_H,
        validation_start=_TE_H + pd.Timedelta(days=8), validation_end=_TE_H + pd.Timedelta(days=20),
        forward_dependency_hours=168, purge_hours=168,
    )
    real_targets = folds_mod._fold_reference_targets
    called = {"n": 0}

    def _counting(*a, **k):
        called["n"] += 1
        return real_targets(*a, **k)

    monkeypatch.setattr(folds_mod, "_fold_reference_targets", _counting)
    with pytest.raises(ValueError, match="common train_start"):
        _build_shared_train_reference("r", ((0, fold0), (1, moved)), object(), {}, 1.0, None, None)
    assert called["n"] == 0


def test_failure_mapping_is_unchanged() -> None:
    fold = _fold(_TE_K)
    cases = [
        (DataIntegrityError("x missing"), ("RELEVANT_EXECUTION_DATA_GAP",)),
        (MhsResourceAdmissionError(stage="s", error_code="MEMORY_BUDGET", message="rss budget exceeded"), ("RESOURCE_BUDGET_BREACH",)),
        (RuntimeError("boom"), ("INCOMPLETE_ANCHORED_FOLD",)),
        (ValueError("bad"), ("INCOMPLETE_ANCHORED_FOLD",)),
    ]
    for exc, codes in cases:
        assert _fold_failure_report(fold, 0, exc).failures == codes
        assert _fold_failure_report(fold, 0, exc).failures == _incomplete_fold_report(fold, 0, codes).failures


def test_prefix_constants_declared() -> None:
    assert TRAIN_REFERENCE_PREFIX_TARGET_ATOL == 1e-12
    assert TRAIN_REFERENCE_PREFIX_RETURN_ATOL == 1e-12


def test_horizon_guard_rejects_shorter_shared_horizon() -> None:
    fold_k, own, signals, shared = _shared_fixture()
    short_shared = _SharedTrainReference(
        group_key=shared.group_key, reference_start=shared.reference_start,
        horizon_end=pd.Timestamp("2021-05-01", tz="UTC"), target_weights=shared.target_weights,
        signal_available_at=shared.signal_available_at, daily_returns=shared.daily_returns,
    )
    assert _shared_train_reference_slice(short_shared, fold_k, 0, own, signals, ExecutionSpec()) is None
    moved = AnchoredPurgedFold(
        train_start=pd.Timestamp("2021-02-01", tz="UTC"), train_end=_TE_K,
        validation_start=_TE_K + pd.Timedelta(days=8), validation_end=_TE_K + pd.Timedelta(days=20),
        forward_dependency_hours=168, purge_hours=168,
    )
    assert _shared_train_reference_slice(shared, moved, 0, own, signals, ExecutionSpec()) is None


def test_misaligned_signal_length_forces_fallback() -> None:
    fold_k, own, signals, shared = _shared_fixture()
    broken = _SharedTrainReference(
        group_key=shared.group_key, reference_start=shared.reference_start,
        horizon_end=shared.horizon_end, target_weights=shared.target_weights,
        signal_available_at=shared.signal_available_at[:-1], daily_returns=shared.daily_returns,
    )
    assert _shared_train_reference_slice(broken, fold_k, 0, own, signals, ExecutionSpec()) is None


def test_group_key_separates_member_sets_not_cutoffs() -> None:
    # I-FOLD-ADMISSION-PIT: the admitted member tuple joins the train-reference
    # group key (never the admission cutoff, which differs per fold by
    # construction), so folds executing different member sets never share a replay.
    from src.strategy.features import FeatureAdmission

    fold_k = _fold(_TE_K)
    fold_h = _fold(_TE_H)
    assert fold_k.train_start == fold_h.train_start
    shared_a = FeatureAdmission(cutoff=_TE_K, admitted=("a", "b"))
    shared_b = FeatureAdmission(cutoff=_TE_H, admitted=("a", "b"))
    assert _train_reference_group_key(
        fold_k, None, None, committee_admission=shared_a
    ) == _train_reference_group_key(fold_h, None, None, committee_admission=shared_b)
    divergent = FeatureAdmission(cutoff=_TE_K, admitted=("a", "c"))
    assert _train_reference_group_key(
        fold_k, None, None, committee_admission=shared_a
    ) != _train_reference_group_key(fold_k, None, None, committee_admission=divergent)
    empty = FeatureAdmission(cutoff=_TE_K, admitted=())
    assert _train_reference_group_key(fold_k, None, None) != _train_reference_group_key(
        fold_k, None, None, committee_admission=empty
    )


def test_shared_reference_rejects_mixed_member_sets() -> None:
    # I-FOLD-ADMISSION-PIT: folds executing different member sets never share
    # a reference replay -- the group build fails closed before any replay.
    from src.strategy.features import FeatureAdmission

    fold0 = _fold(_TE_K)
    fold1 = _fold(_TE_H)
    group = ((0, fold0), (1, fold1))
    admissions = {
        0: FeatureAdmission(cutoff=_TE_K, admitted=("a", "b")),
        1: FeatureAdmission(cutoff=_TE_H, admitted=("a", "c")),
    }
    with pytest.raises(ValueError, match="common admitted tuple"):
        _build_shared_train_reference(
            "r", group, object(), {}, 1.0, None, None, committee_admissions=admissions,
        )


def test_reference_from_shared_threads_admission(monkeypatch) -> None:
    # Line 879: _reference_from_shared forwards the fold's own admission into
    # both the own-target build and the group-key comparison.
    from src.strategy.features import FeatureAdmission

    fold_k, own, signals, shared = _shared_fixture()
    admission = FeatureAdmission(cutoff=_TE_K, admitted=("a",))
    seen: dict[str, object] = {}

    def _targets(*a, **k):
        seen["admission"] = k.get("committee_admission")
        return own, signals, ["A", "B"]

    monkeypatch.setattr(folds_mod, "_fold_reference_targets", _targets)

    def _slice(*a, **k):
        return pd.Series(
            0.001,
            index=pd.date_range(end=_TE_K - pd.Timedelta(days=1), periods=100, freq="1D", tz="UTC"),
            dtype="float64",
        )

    monkeypatch.setattr(folds_mod, "_shared_train_reference_slice", _slice)
    monkeypatch.setattr(
        folds_mod.specs, "_resolved_base_execution_spec", lambda _r: ExecutionSpec(),
    )
    matching = _SharedTrainReference(
        group_key=folds_mod._train_reference_group_key(
            fold_k, None, None, committee_admission=admission,
        ),
        reference_start=shared.reference_start, horizon_end=shared.horizon_end,
        target_weights=shared.target_weights, signal_available_at=shared.signal_available_at,
        daily_returns=shared.daily_returns,
    )
    out = _reference_from_shared(
        "root", fold_k, object(), {}, 1.0, 0, None, None, matching,
        committee_admission=admission,
    )
    assert seen["admission"] is admission
    assert len(out) == 100
