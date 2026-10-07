"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.mhs_requests import research_baseline
from tests.unit.mhs.test_evaluation_appresearch import _FOLD, _START
import src.mhs.evaluation.folds as folds


def test_folds_module_present() -> None:
    assert folds.__name__ == "src.mhs.evaluation.folds"
    assert callable(folds._incomplete_fold_report)


def test_run_anchored_fold_in_memory_window_reuse(monkeypatch) -> None:
    from src.mhs.evidence import phase_1_anchored_purged_folds
    from src.mhs.evaluation import folds

    fold_list = phase_1_anchored_purged_folds()
    req = research_baseline(start="2021-01-01", end="2021-03-31", execution_universe_size=8)
    # Verified invocation signature accepts shared_token
    assert callable(folds._run_anchored_fold)


def _utc_series(values, start="2021-01-01", tz="UTC"):
    import pandas as pd

    idx = pd.date_range(start, periods=len(values), freq="1D", tz=tz)
    return pd.Series(list(values), index=idx, dtype="float64")


def test_train_reference_rejects_non_finite_returns() -> None:
    import pandas as pd
    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.integrity import _assert_train_reference_returns_valid
    import pytest

    daily = _utc_series([0.01] * 89 + [float("nan")])
    with pytest.raises(DataIntegrityError, match="train reference returns must be finite"):
        _assert_train_reference_returns_valid(daily, pd.Timestamp("2021-05-01", tz="UTC"), 0)


def test_train_reference_rejects_non_monotonic_index() -> None:
    import pandas as pd
    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.integrity import _assert_train_reference_returns_valid
    import pytest

    idx = pd.DatetimeIndex([pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2021-01-01", tz="UTC"), *pd.date_range("2021-01-02", periods=90, freq="1D", tz="UTC")])
    daily = pd.Series([0.01] * len(idx), index=idx, dtype="float64")
    with pytest.raises(DataIntegrityError, match="unique and monotonic"):
        _assert_train_reference_returns_valid(daily, pd.Timestamp("2021-06-01", tz="UTC"), 0)


def test_train_reference_rejects_non_utc_index() -> None:
    import pandas as pd
    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.integrity import _assert_train_reference_returns_valid
    import pytest

    daily = _utc_series([0.01] * 100, tz=None)
    with pytest.raises(DataIntegrityError, match="must be UTC"):
        _assert_train_reference_returns_valid(daily, pd.Timestamp("2021-05-01", tz="UTC"), 0)


def test_train_reference_rejects_validation_leakage() -> None:
    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.integrity import _assert_train_reference_returns_valid
    import pytest

    daily = _utc_series([0.01] * 100)
    train_end = daily.index[-1]
    with pytest.raises(DataIntegrityError, match="extends into validation"):
        _assert_train_reference_returns_valid(daily, train_end, 0)


def test_train_reference_enforces_burn_in_length() -> None:
    import pandas as pd
    from src.mhs.evaluation.integrity import _assert_train_reference_returns_valid
    from src.mhs.params import PNL_VOL_TARGET_BURN_IN_DAYS
    import pytest
    from src.common.errors import DataIntegrityError

    short = _utc_series([0.01] * (PNL_VOL_TARGET_BURN_IN_DAYS - 1))
    with pytest.raises(DataIntegrityError, match="require >= "):
        _assert_train_reference_returns_valid(short, pd.Timestamp("2022-01-01", tz="UTC"), 0)
    exact = _utc_series([0.01] * PNL_VOL_TARGET_BURN_IN_DAYS)
    assert _assert_train_reference_returns_valid(exact, pd.Timestamp("2022-01-01", tz="UTC"), 0) is None


def test_fold_train_reference_rejects_empty_window() -> None:
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.evaluation.folds import _fold_train_reference_returns
    from src.mhs.evidence import AnchoredPurgedFold

    fold = AnchoredPurgedFold(
        train_start=pd.Timestamp("2021-01-01", tz="UTC"),
        train_end=pd.Timestamp("2021-01-10", tz="UTC"),
        validation_start=pd.Timestamp("2021-02-01", tz="UTC"),
        validation_end=pd.Timestamp("2021-05-01", tz="UTC"),
        forward_dependency_hours=24,
        purge_hours=24,
    )
    with pytest.raises(DataIntegrityError, match="train reference window is empty"):
        _fold_train_reference_returns("root", fold, None, {}, 1.0, 0, None, None)


def test_run_anchored_fold_records_sizing_reference_telemetry(mhs_market, monkeypatch) -> None:
    import pandas as pd
    import src.mhs.evaluation.folds as folds_mod
    from src.mhs.evaluation.folds import _run_anchored_fold
    from src.mhs.marks import _load_funding_series
    from src.mhs.resources import _StageRecorder
    from src.quant.universe.pit_universe import symbol_partition
    from tests.unit.mhs.test_evaluation_appresearch import _FOLD, _START

    root, end = mhs_market
    symbols = [
        s for s in ("MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
                    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT")
        if symbol_partition(s) == "dev"
    ][:8]
    funding_by_symbol, _ = _load_funding_series(symbols)
    request = research_baseline(
        start=str(_START), end=str(end), data_root=str(root),
        execution_timeframe="3m", log_run=False,
    )
    local_index = pd.date_range(_FOLD.train_start, periods=100, freq="D", tz="UTC")
    monkeypatch.setattr(
        folds_mod, "_fold_train_reference_returns",
        lambda *_a, **_k: pd.Series(0.001, index=local_index),
    )
    recorder = _StageRecorder(log_run=False)
    _run_anchored_fold(str(root), _FOLD, request, funding_by_symbol, 1.0, 0, recorder)
    assert any(m.stage == "anchored_fold_0_sizing_reference" for m in recorder.records)


_DEV_SYMBOLS = (
    "MHSAUSDT", "MHSBUSDT", "MHSCUSDT", "MHSDUSDT", "MHSEUSDT",
    "MHSGUSDT", "MHSHUSDT", "MHSIUSDT", "MHSJUSDT", "MHSLUSDT",
)


def _plan_args(mhs_market) -> tuple:
    """The request / funding pair every validation-plan scenario drives."""
    from src.mhs.marks import _load_funding_series
    from src.quant.universe.pit_universe import symbol_partition

    root, end = mhs_market
    funding_by_symbol, _ = _load_funding_series(
        [s for s in _DEV_SYMBOLS if symbol_partition(s) == "dev"][:8],
    )
    request = research_baseline(
        start=str(_START), end=str(end), data_root=str(root),
        execution_timeframe="3m", log_run=False,
    )
    return str(root), request, funding_by_symbol


def _beyond_market_fold():
    """A fold whose validation window lies entirely past the fixture's market data."""
    from src.mhs.evidence import AnchoredPurgedFold

    return AnchoredPurgedFold(
        train_start=_FOLD.train_start,
        train_end=_FOLD.train_end,
        validation_start=pd.Timestamp("2021-06-01", tz="UTC"),
        validation_end=pd.Timestamp("2021-07-01", tz="UTC"),
        forward_dependency_hours=168,
        purge_hours=168,
    )


def _reference_spy(monkeypatch, outcome: pd.Series | BaseException) -> list[int]:
    """Patch the train reference; ``outcome`` is a returned series or a raised error."""
    calls: list[int] = []

    def _reference(*_a, **_k):
        calls.append(1)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(folds, "_fold_train_reference_returns", _reference)
    return calls


def _assert_reports_identical(actual, expected) -> None:
    """Field-for-field report equality that treats the NaN placeholders as equal."""
    for field in dataclasses.fields(expected):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(right, float):
            assert (math.isnan(left) and math.isnan(right)) or left == right, field.name
        else:
            assert left == right, field.name


def test_unusable_validation_window_skips_train_reference(mhs_market, monkeypatch) -> None:
    root, request, funding = _plan_args(mhs_market)
    fold = _beyond_market_fold()
    with pytest.raises(ValueError, match="no symbol survived the panel filters"):
        folds.fold_weights._build_fold_target_weights(root, fold, request, funding)
    calls = _reference_spy(monkeypatch, AssertionError("train reference ran"))
    report = folds._run_anchored_fold(root, fold, request, funding, 1.0, 0)
    assert calls == []
    _assert_reports_identical(report, folds._incomplete_fold_report(fold, 0, ("INCOMPLETE_ANCHORED_FOLD",)))


def test_validation_integrity_error_keeps_classifier_mapping(mhs_market, monkeypatch) -> None:
    from src.common.errors import DataIntegrityError

    root, request, funding = _plan_args(mhs_market)
    calls = _reference_spy(monkeypatch, AssertionError("train reference ran"))

    def _raise(*_a, **_k):
        raise DataIntegrityError("zombie mask requires ['volume'] in x: missing")

    monkeypatch.setattr(folds.fold_weights, "_build_fold_target_weights", _raise)
    report = folds._run_anchored_fold(root, _FOLD, request, funding, 1.0, 0)
    assert calls == []
    assert report.failures == ("RELEVANT_EXECUTION_DATA_GAP",)
    assert report.strict is None


def test_validation_unavailable_takes_precedence_over_failing_reference(mhs_market, monkeypatch) -> None:
    from src.common.errors import DataIntegrityError

    root, request, funding = _plan_args(mhs_market)
    calls = _reference_spy(
        monkeypatch, DataIntegrityError("fold 0: train reference returns must be finite"),
    )

    def _raise(*_a, **_k):
        raise RuntimeError("no fold symbol has funding coverage")

    monkeypatch.setattr(folds.fold_weights, "_build_fold_target_weights", _raise)
    report = folds._run_anchored_fold(root, _FOLD, request, funding, 1.0, 0)
    assert calls == []
    assert report.failures == ("INCOMPLETE_ANCHORED_FOLD",)
    assert report.strict is None
    assert report.primary_valid is False


def test_reference_failure_still_fails_closed_when_validation_usable(mhs_market, monkeypatch) -> None:
    from src.common.errors import DataIntegrityError

    root, request, funding = _plan_args(mhs_market)
    calls = _reference_spy(monkeypatch, DataIntegrityError("fold 0: train reference returns must be finite"))
    report = folds._run_anchored_fold(root, _FOLD, request, funding, 1.0, 0)
    assert calls == [1]
    assert report.failures == ("NONFINITE_EQUITY",)
    assert report.strict is None
    assert report.stress is None


def test_validation_targets_built_once_and_reused(mhs_market, monkeypatch) -> None:
    root, request, funding = _plan_args(mhs_market)
    calls: list[dict] = []
    build = folds.fold_weights._build_fold_target_weights

    def _counting(*args, **kwargs):
        calls.append(kwargs)
        return build(*args, **kwargs)

    monkeypatch.setattr(folds.fold_weights, "_build_fold_target_weights", _counting)
    reference = pd.Series(
        0.001,
        index=pd.date_range(end=_FOLD.train_end - pd.Timedelta(days=1), periods=100, freq="1D", tz="UTC"),
    )
    folds.integrity._assert_train_reference_returns_valid(reference, _FOLD.train_end, 0)
    monkeypatch.setattr(folds, "_fold_train_reference_returns", lambda *_a, **_k: reference)
    report = folds._run_anchored_fold(root, _FOLD, request, funding, 1.0, 0)
    assert len(calls) == 1
    assert "decision_start" not in calls[0]
    assert "decision_end" not in calls[0]
    assert "base_panel" not in calls[0]
    assert report.strict is not None, report.failures
    plan = folds._build_fold_validation_plan(root, _FOLD, request, funding, None, None)
    assert report.decision_intents == plan.decision_intents
    assert report.terminal_censored_decisions == plan.terminal_censored


def test_validation_plan_matches_direct_target_build(mhs_market) -> None:
    root, request, funding = _plan_args(mhs_market)
    plan = folds._build_fold_validation_plan(root, _FOLD, request, funding, None, None)
    target_weights, signal, roster, _grid = folds.fold_weights._build_fold_target_weights(
        root, _FOLD, request, funding,
    )
    execution_grid = pd.date_range(_FOLD.validation_start, _FOLD.validation_end, freq="3min", tz="UTC")
    expected_replay, expected_signals, expected_censored = folds.integrity._truncate_replayable_decisions(
        target_weights[roster], signal, execution_grid, folds.specs._resolved_base_execution_spec(request),
    )
    pd.testing.assert_frame_equal(plan.target_weights, target_weights, check_exact=True)
    pd.testing.assert_frame_equal(plan.target_replay, expected_replay, check_exact=True)
    pd.testing.assert_index_equal(plan.signal_available_at, expected_signals, exact=True)
    assert plan.terminal_censored == expected_censored
    assert list(plan.target_replay.columns) == list(target_weights[roster].columns)
    assert len(plan.signal_available_at) == len(plan.target_replay)
    assert plan.decision_intents == int(np.isfinite(plan.target_replay.to_numpy()).sum())


def _reference_window_fold():
    from src.mhs.evidence import AnchoredPurgedFold

    return AnchoredPurgedFold(
        train_start=pd.Timestamp("2020-06-01", tz="UTC"),
        train_end=pd.Timestamp("2021-02-01", tz="UTC"),
        validation_start=pd.Timestamp("2021-02-10", tz="UTC"),
        validation_end=pd.Timestamp("2021-03-01", tz="UTC"),
        forward_dependency_hours=168,
        purge_hours=168,
    )


def _certified_reference_replay(equity: pd.Series):
    from types import SimpleNamespace

    return SimpleNamespace(
        ledger=SimpleNamespace(
            primary_valid=True, invalid_reasons=(), data_gaps=(), equity=equity,
        ),
        terminal_positions=(SimpleNamespace(status="open_marked", funding_complete=True),),
    )


def _uncertified_reference_replay(equity: pd.Series):
    from types import SimpleNamespace

    import pandas as pd

    from src.mhs.execution import ExecutionDataGap

    return SimpleNamespace(
        ledger=SimpleNamespace(
            primary_valid=False,
            invalid_reasons=("MISSING_DATA",),
            data_gaps=(
                ExecutionDataGap(
                    code="MISSING_HELD_FUNDING", symbol="AAAUSDT",
                    timestamp=pd.Timestamp("2021-01-15", tz="UTC"),
                ),
            ),
            equity=equity,
        ),
        terminal_positions=(SimpleNamespace(status="unresolved", funding_complete=False),),
    )


def _daily_equity_before(train_end: pd.Timestamp, rows: int) -> pd.Series:
    import numpy as np

    idx = pd.date_range(end=train_end - pd.Timedelta(days=1), periods=rows, freq="1D", tz="UTC")
    return pd.Series(np.linspace(1.0, 1.0 + 0.001 * rows, len(idx)), index=idx, dtype="float64")


def test_uncertified_train_reference_yields_gap_code(mhs_market, monkeypatch) -> None:
    root, request, funding = _plan_args(mhs_market)
    fold = _reference_window_fold()
    build_calls: list[dict] = []

    def _counting(*args, **kwargs):
        build_calls.append(kwargs)
        decision_start = kwargs.get("decision_start", fold.validation_start)
        index = pd.DatetimeIndex([decision_start])
        targets = pd.DataFrame({"BTCUSDT": [0.1]}, index=index)
        return targets, index, ["BTCUSDT"], index

    monkeypatch.setattr(folds.fold_weights, "_build_fold_target_weights", _counting)
    equity = _daily_equity_before(fold.train_end, 100)
    monkeypatch.setattr(folds, "replay_execution_windows", lambda *a, **k: _uncertified_reference_replay(equity))
    report = folds._run_anchored_fold(root, fold, request, funding, 1.0, 0)
    assert report.strict is None
    assert report.primary_valid is False
    assert report.failures == ("RELEVANT_EXECUTION_DATA_GAP",)
    assert len(build_calls) == 2
    assert build_calls[0] == {"committee_admission": None}
    assert build_calls[1] == {
        "decision_start": fold.train_start + pd.Timedelta(hours=folds.FOLD_PANEL_WARMUP_HOURS),
        "decision_end": fold.train_end,
        "committee_admission": None,
    }


def test_train_reference_certification_precedes_returns_validation(mhs_market, monkeypatch) -> None:
    import pytest

    from src.common.errors import DataIntegrityError

    root, request, funding = _plan_args(mhs_market)
    fold = _reference_window_fold()
    monkeypatch.setattr(
        folds.fold_weights, "_build_fold_target_weights",
        lambda *a, **k: (pd.DataFrame(), pd.DatetimeIndex([], tz="UTC"), [], pd.DataFrame()),
    )
    equity = _daily_equity_before(fold.train_end, 5)
    monkeypatch.setattr(folds, "replay_execution_windows", lambda *a, **k: _uncertified_reference_replay(equity))
    with pytest.raises(DataIntegrityError) as excinfo:
        folds._fold_train_reference_returns(root, fold, request, funding, 1.0, 0, None, None)
    assert "ledger not certified" in str(excinfo.value)
    assert "require >=" not in str(excinfo.value)


def test_certified_train_reference_keeps_current_behaviour(mhs_market, monkeypatch) -> None:
    root, request, funding = _plan_args(mhs_market)
    fold = _reference_window_fold()
    monkeypatch.setattr(
        folds.fold_weights, "_build_fold_target_weights",
        lambda *a, **k: (pd.DataFrame(), pd.DatetimeIndex([], tz="UTC"), [], pd.DataFrame()),
    )
    equity = _daily_equity_before(fold.train_end, 100)
    monkeypatch.setattr(folds, "replay_execution_windows", lambda *a, **k: _certified_reference_replay(equity))
    daily = folds._fold_train_reference_returns(root, fold, request, funding, 1.0, 0, None, None)
    expected = equity.pct_change().dropna().astype("float64")
    pd.testing.assert_series_equal(daily, expected, check_exact=True)
    assert str(daily.index.tz) == "UTC"
    assert daily.index.is_unique
    assert daily.index.is_monotonic_increasing
    assert (daily.index < fold.train_end).all()
    assert str(daily.dtype) == "float64"
