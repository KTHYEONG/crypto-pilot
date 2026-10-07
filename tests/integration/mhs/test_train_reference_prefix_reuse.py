"""Train-reference prefix reuse on a real synthetic market (4400h)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

import src.mhs.marks as marks
from src.common.errors import DataIntegrityError
from src.mhs.evaluation import folds as folds_mod
from src.mhs.evaluation.folds import (
    _build_shared_train_reference,
    _fold_train_reference_returns,
    _reference_from_shared,
    _run_anchored_fold,
)
from src.mhs.evidence import AnchoredPurgedFold
from src.mhs.marks import _load_funding_series
from src.mhs.params import TRAIN_REFERENCE_PREFIX_RETURN_ATOL
from src.quant.universe.pit_universe import symbol_partition
from tests.fixtures.mhs_requests import research_baseline
from tests.unit.mhs.test_evaluation_appresearch import (
    _write_3m_cache,
    _write_mhs_market,
)

_TES = (
    pd.Timestamp("2021-05-20", tz="UTC"),
    pd.Timestamp("2021-06-10", tz="UTC"),
    pd.Timestamp("2021-06-20", tz="UTC"),
)


def _folds() -> tuple[AnchoredPurgedFold, ...]:
    return tuple(
        AnchoredPurgedFold(
            pd.Timestamp("2021-01-01", tz="UTC"), te,
            te + pd.Timedelta(days=8), te + pd.Timedelta(days=20), 168, 168,
        )
        for te in _TES
    )


def _dev_symbols() -> list[str]:
    return [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ]


@pytest.fixture(scope="module")
def prefix_market(tmp_path_factory):
    root = tmp_path_factory.mktemp("prefix_market")
    _write_mhs_market(root, n_hours=4400)
    _write_3m_cache(root)
    from src.mhs.marks import clear_mhs_market_data_caches

    with pytest.MonkeyPatch.context() as patch:
        _point_marks_at(root, patch)
        request = research_baseline(
            start="2021-01-01", end="2021-07-03", data_root=str(root),
            execution_timeframe="3m", log_run=False,
        )
        funding, _ = _load_funding_series(_dev_symbols())
        try:
            yield root, request, funding
        finally:
            clear_mhs_market_data_caches()


@pytest.fixture(scope="module")
def independent_refs(prefix_market):
    root, request, funding = prefix_market
    folds = _folds()
    refs = {}
    for idx, fold in enumerate(folds):
        refs[idx] = _fold_train_reference_returns(
            str(root), fold, request, funding, 1.0, idx, None, None,
        )
    assert len(refs[0]) == 100
    assert len(refs[1]) == 121
    assert len(refs[2]) == 131
    return refs


@pytest.fixture(scope="module")
def shared_ref(prefix_market):
    root, request, funding = prefix_market
    folds = _folds()
    group = tuple((idx, fold) for idx, fold in enumerate(folds))
    return _build_shared_train_reference(str(root), group, request, funding, 1.0, None, None)


def _copy_market(root: Path, tmp_path_factory, tag: str) -> Path:
    dest = tmp_path_factory.mktemp(f"prefix_{tag}")
    shutil.copytree(root, dest, dirs_exist_ok=True)
    return dest


def _point_marks_at(root: Path, monkeypatch) -> None:
    monkeypatch.setattr(marks, "funding_path", lambda sym: root / "funding" / f"{sym}.parquet")
    from src.mhs.marks import clear_mhs_market_data_caches

    clear_mhs_market_data_caches()


def test_shared_prefix_equals_independent_replay(prefix_market, independent_refs, shared_ref, monkeypatch) -> None:
    root, request, funding = prefix_market
    assert shared_ref is not None
    folds = _folds()
    real_replay = folds_mod._replay_fold_train_reference
    calls = {"n": 0}

    def _counting(*a, **k):
        calls["n"] += 1
        return real_replay(*a, **k)

    monkeypatch.setattr(folds_mod, "_replay_fold_train_reference", _counting)
    for idx, fold in enumerate(folds):
        used = _reference_from_shared(str(root), fold, request, funding, 1.0, idx, None, None, shared_ref)
        independent = independent_refs[idx]
        pd.testing.assert_index_equal(used.index, independent.index, exact=True)
        assert float((used - independent).abs().max()) <= TRAIN_REFERENCE_PREFIX_RETURN_ATOL
    pd.testing.assert_series_equal(
        _reference_from_shared(str(root), folds[2], request, funding, 1.0, 2, None, None, shared_ref),
        independent_refs[2], check_exact=True,
    )
    assert calls["n"] == 0


def test_staggered_listing_drift_stays_within_bound(prefix_market, tmp_path_factory, monkeypatch) -> None:
    root, request, funding = prefix_market
    dest = _copy_market(root, tmp_path_factory, "stag")
    cutoff = pd.Timestamp("2021-05-01", tz="UTC")
    sym = _dev_symbols()[0]
    for sub in ("1h", "3m", "funding"):
        path = dest / sub / f"{sym}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        ts_col = "timestamp"
        idx = pd.to_datetime(frame[ts_col], unit="ms", utc=True)
        frame = frame.loc[idx >= cutoff].reset_index(drop=True)
        frame.to_parquet(path)
    mark_path = dest / "markPriceKlines" / "1h" / f"{sym}.parquet"
    if mark_path.exists():
        frame = pd.read_parquet(mark_path)
        idx = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame.loc[idx >= cutoff].to_parquet(mark_path)
    _point_marks_at(dest, monkeypatch)
    stag_funding, _ = _load_funding_series(_dev_symbols())
    folds = _folds()
    group = tuple((idx, fold) for idx, fold in enumerate(folds))
    shared = _build_shared_train_reference(str(dest), group, request, stag_funding, 1.0, None, None)
    assert shared is not None
    independents = {
        idx: _fold_train_reference_returns(str(dest), folds[idx], request, stag_funding, 1.0, idx, None, None)
        for idx in (0, 1)
    }
    real_replay = folds_mod._replay_fold_train_reference
    calls = {"n": 0}

    def _counting(*a, **k):
        calls["n"] += 1
        return real_replay(*a, **k)

    monkeypatch.setattr(folds_mod, "_replay_fold_train_reference", _counting)
    for idx in (0, 1):
        used = _reference_from_shared(str(dest), folds[idx], request, stag_funding, 1.0, idx, None, None, shared)
        independent = independents[idx]
        pd.testing.assert_index_equal(used.index, independent.index, exact=True)
        assert float((used - independent).abs().max()) <= TRAIN_REFERENCE_PREFIX_RETURN_ATOL
    assert calls["n"] == 0


def _scale_prices_after(dest: Path, cutoff: pd.Timestamp) -> None:
    for sub in ("1h", "3m"):
        for path in sorted((dest / sub).glob("*.parquet")):
            frame = pd.read_parquet(path)
            mask = pd.to_datetime(frame["timestamp"], unit="ms", utc=True) > cutoff
            for col in ("open", "high", "low", "close"):
                if col in frame.columns:
                    frame.loc[mask, col] = frame.loc[mask, col] * 1.37
            frame.to_parquet(path)


def _truncate_symbol_after(dest: Path, symbol: str, cutoff: pd.Timestamp) -> None:
    for path in sorted(dest.glob(f"*/{symbol}.parquet")):
        frame = pd.read_parquet(path)
        frame.loc[pd.to_datetime(frame["timestamp"], unit="ms", utc=True) <= cutoff].to_parquet(path)


def test_price_corruption_after_train_end_leaves_shared_prefix_unchanged(
    prefix_market, independent_refs, tmp_path_factory, monkeypatch,
) -> None:
    """Certified shared replay over corrupted future prices still yields F0's uncorrupted reference via the slice path."""
    root, request, funding = prefix_market
    dest = _copy_market(root, tmp_path_factory, "corrupt_prices")
    _scale_prices_after(dest, _TES[0])
    _point_marks_at(dest, monkeypatch)
    folds = _folds()
    group = tuple((idx, fold) for idx, fold in enumerate(folds))
    shared = _build_shared_train_reference(str(dest), group, request, funding, 1.0, None, None)
    assert shared is not None
    used = _reference_from_shared(str(dest), folds[0], request, funding, 1.0, 0, None, None, shared)
    pd.testing.assert_index_equal(used.index, independent_refs[0].index, exact=True)
    assert float((used - independent_refs[0]).abs().max()) <= TRAIN_REFERENCE_PREFIX_RETURN_ATOL


def test_corrupting_data_after_train_end_leaves_used_reference_unchanged(
    prefix_market, independent_refs, tmp_path_factory, monkeypatch,
) -> None:
    root, request, funding = prefix_market
    dest = _copy_market(root, tmp_path_factory, "corrupt")
    te0 = _TES[0]
    _scale_prices_after(dest, te0)
    syms = _dev_symbols()
    frame = pd.read_parquet(dest / "funding" / f"{syms[0]}.parquet")
    idx = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame.loc[idx <= te0 + pd.Timedelta(days=2)].to_parquet(dest / "funding" / f"{syms[0]}.parquet")
    _truncate_symbol_after(dest, syms[1], te0 + pd.Timedelta(days=5))
    _point_marks_at(dest, monkeypatch)
    corrupt_funding, _ = _load_funding_series(syms)
    folds = _folds()
    group = tuple((idx, fold) for idx, fold in enumerate(folds))
    shared = _build_shared_train_reference(str(dest), group, request, corrupt_funding, 1.0, None, None)
    assert shared is None
    fallback = _fold_train_reference_returns(str(dest), folds[0], request, corrupt_funding, 1.0, 0, None, None)
    pd.testing.assert_series_equal(fallback, independent_refs[0], check_exact=True)


def test_shared_invalidity_after_train_end_never_fails_earlier_fold(
    prefix_market, independent_refs, tmp_path_factory, monkeypatch,
) -> None:
    root, request, funding = prefix_market
    dest = _copy_market(root, tmp_path_factory, "invalid_after")
    te1 = _TES[1]
    start = te1 + pd.Timedelta(days=1)
    end = te1 + pd.Timedelta(days=3)
    for path in sorted((dest / "funding").glob("*.parquet")):
        frame = pd.read_parquet(path)
        idx = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame.loc[(idx < start) | (idx > end)].to_parquet(path)
    _point_marks_at(dest, monkeypatch)
    bad_funding, _ = _load_funding_series(_dev_symbols())
    folds = _folds()
    from tests.unit.mhs.test_evaluation_folds import _InlineExecutor

    consumed = {}
    shared_results = []
    real_reference = folds_mod._fold_train_reference_returns
    real_shared = folds_mod._build_shared_train_reference

    def _capture_reference(*args, **kwargs):
        fold_index = args[5]
        try:
            result = real_reference(*args, **kwargs)
        except DataIntegrityError as exc:
            consumed[fold_index] = exc
            raise
        consumed[fold_index] = result
        return result

    def _capture_shared(*args, **kwargs):
        result = real_shared(*args, **kwargs)
        shared_results.append(result)
        return result

    monkeypatch.setattr(folds_mod, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(folds_mod, "resolved_anchored_folds", lambda _: folds)
    monkeypatch.setattr(folds_mod, "_build_shared_train_reference", _capture_shared)
    monkeypatch.setattr(folds_mod, "_fold_train_reference_returns", _capture_reference)
    reports = folds_mod._run_folds_parallel(str(dest), request, bad_funding, 1.0)
    assert shared_results == [None]
    assert set(consumed) == {0, 1, 2}
    for idx in (0, 1):
        pd.testing.assert_series_equal(consumed[idx], independent_refs[idx], check_exact=True)
        assert reports[idx].strict is not None
        assert "RELEVANT_EXECUTION_DATA_GAP" not in reports[idx].failures
    assert reports[2].failures == ("RELEVANT_EXECUTION_DATA_GAP",)
    assert reports[2].strict is None
    assert isinstance(consumed[2], DataIntegrityError)
    with pytest.raises(DataIntegrityError) as excinfo_ind:
        real_reference(str(dest), folds[2], request, bad_funding, 1.0, 2, None, None)
    assert str(consumed[2]) == str(excinfo_ind.value)


def test_shared_certification_implies_per_fold_certification(prefix_market, independent_refs, shared_ref) -> None:
    assert shared_ref is not None
    assert len(independent_refs) == 3


def test_invalidity_before_train_end_reported_per_fold(prefix_market, tmp_path_factory, monkeypatch) -> None:
    root, request, funding = prefix_market
    dest = _copy_market(root, tmp_path_factory, "invalid_before")
    te0 = _TES[0]
    start = te0 - pd.Timedelta(days=3)
    end = te0 - pd.Timedelta(days=1)
    for path in sorted((dest / "funding").glob("*.parquet")):
        frame = pd.read_parquet(path)
        idx = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame.loc[(idx < start) | (idx > end)].to_parquet(path)
    _point_marks_at(dest, monkeypatch)
    bad_funding, _ = _load_funding_series(_dev_symbols())
    folds = _folds()
    group = tuple((idx, fold) for idx, fold in enumerate(folds))
    shared = _build_shared_train_reference(str(dest), group, request, bad_funding, 1.0, None, None)
    report_shared = _run_anchored_fold(str(dest), folds[0], request, bad_funding, 1.0, 0, shared_reference=shared)
    report_ind = _run_anchored_fold(str(dest), folds[0], request, bad_funding, 1.0, 0)
    assert report_shared.failures == ("RELEVANT_EXECUTION_DATA_GAP",)
    assert report_shared.failures == report_ind.failures


def test_target_hazards_caught_by_prefix_guard(prefix_market, independent_refs, monkeypatch, caplog) -> None:
    import src.mhs.evaluation.fold_weights as fold_weights_mod

    root, request, funding = prefix_market
    folds = _folds()
    group = tuple((idx, fold) for idx, fold in enumerate(folds))
    real_load = fold_weights_mod.load_base_panel
    drop_sym = _dev_symbols()[0]

    def _wrapped(*args, **kwargs):
        end = kwargs.get("end", args[4] if len(args) > 4 else None)
        frame = real_load(*args, **kwargs)
        if end is not None and pd.Timestamp(end) == _TES[2] and drop_sym in frame.get("close", {}):
            frame = {k: v.drop(columns=[drop_sym]) if drop_sym in v.columns else v for k, v in frame.items()}
        return frame

    monkeypatch.setattr(fold_weights_mod, "load_base_panel", _wrapped)
    shared = _build_shared_train_reference(str(root), group, request, funding, 1.0, None, None)
    assert shared is not None
    real_replay = folds_mod._replay_fold_train_reference
    calls = {"n": 0}

    def _counting(*a, **k):
        calls["n"] += 1
        return real_replay(*a, **k)

    monkeypatch.setattr(folds_mod, "_replay_fold_train_reference", _counting)
    import logging

    with caplog.at_level(logging.DEBUG, logger="MhsHorizonDiagnostic"):
        used = _reference_from_shared(str(root), folds[0], request, funding, 1.0, 0, None, None, shared)
    pd.testing.assert_series_equal(used, independent_refs[0], check_exact=True)
    assert calls["n"] == 1
    assert any("target_prefix_mismatch" in r.message for r in caplog.records)


def test_orchestrated_folds_match_independent_reports(prefix_market, monkeypatch) -> None:
    from concurrent.futures import Future

    import src.mhs.evidence as evidence_mod
    import src.mhs.parallel as parallel_mod

    root, request, funding = prefix_market
    folds = _folds()
    monkeypatch.setattr(evidence_mod, "resolved_anchored_folds", lambda req: folds)
    monkeypatch.setattr(folds_mod, "resolved_anchored_folds", lambda req: folds)

    class _InlineExecutor:
        def __init__(self, *a, **k) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            return False

        def submit(self, fn, /, *args, **kwargs):
            future = Future()
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:  # noqa: BLE001
                future.set_exception(exc)
            return future

    monkeypatch.setattr(folds_mod, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(parallel_mod, "plan_worker_count", lambda *a, **k: 1)
    monkeypatch.setattr(folds_mod, "plan_worker_count", lambda *a, **k: 1)
    monkeypatch.setattr(parallel_mod, "assert_fork_admission", lambda *a, **k: None)
    monkeypatch.setattr(folds_mod, "assert_fork_admission", lambda *a, **k: None)
    ordered = folds_mod._run_folds_parallel(str(root), request, funding, 1.0)
    assert len(ordered) == 3
    for idx, fold in enumerate(folds):
        independent = _run_anchored_fold(str(root), fold, request, funding, 1.0, idx)
        assert ordered[idx].failures == independent.failures
        assert ordered[idx].decision_intents == independent.decision_intents
        assert ordered[idx].terminal_censored_decisions == independent.terminal_censored_decisions
        assert set(ordered[idx].book_structure or {}) == set(independent.book_structure or {})
        assert ordered[idx].strict is not None
        assert independent.strict is not None
        assert ordered[idx].stress is not None
        assert independent.stress is not None
        for actual, expected in ((ordered[idx].strict, independent.strict),
                                 (ordered[idx].stress, independent.stress)):
            pd.testing.assert_series_equal(
                actual.ledger.equity, expected.ledger.equity,
                check_exact=(idx == 2), rtol=1e-9, atol=0.0,
            )


@pytest.mark.slow
def test_corwin_schultz_keeps_bound(prefix_market, independent_refs, monkeypatch) -> None:
    import dataclasses

    root, request, funding = prefix_market
    folds = _folds()
    cs_request = dataclasses.replace(request, liquidity_cost_model="corwin_schultz")
    group = tuple((idx, fold) for idx, fold in enumerate(folds))
    shared = _build_shared_train_reference(str(root), group, cs_request, funding, 1.0, None, None)
    assert shared is not None
    for idx, fold in enumerate(folds):
        used = _reference_from_shared(str(root), fold, cs_request, funding, 1.0, idx, None, None, shared)
        independent = _fold_train_reference_returns(str(root), fold, cs_request, funding, 1.0, idx, None, None)
        pd.testing.assert_index_equal(used.index, independent.index, exact=True)
        assert float((used - independent).abs().max()) <= TRAIN_REFERENCE_PREFIX_RETURN_ATOL


@pytest.mark.slow
def test_committee_capital_keeps_bound_or_falls_back(tmp_path_factory, monkeypatch) -> None:
    root = tmp_path_factory.mktemp("prefix_tbq")
    _write_mhs_market(root, n_hours=4400, include_taker_buy_quote=True)
    _write_3m_cache(root)
    _point_marks_at(root, monkeypatch)
    request = research_baseline(
        start="2021-01-01", end="2021-07-03", data_root=str(root),
        execution_timeframe="3m", log_run=False, committee_capital=True,
    )
    funding, _ = _load_funding_series(_dev_symbols())
    folds = _folds()
    group = tuple((idx, fold) for idx, fold in enumerate(folds))
    shared = _build_shared_train_reference(str(root), group, request, funding, 1.0, None, None)
    assert shared is not None
    for idx, fold in enumerate(folds):
        used = _reference_from_shared(str(root), fold, request, funding, 1.0, idx, None, None, shared)
        independent = _fold_train_reference_returns(str(root), fold, request, funding, 1.0, idx, None, None)
        pd.testing.assert_index_equal(used.index, independent.index, exact=True)
        assert float((used - independent).abs().max()) <= TRAIN_REFERENCE_PREFIX_RETURN_ATOL
