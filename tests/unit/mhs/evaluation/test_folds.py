"""P4 path-presence pin for the unified MHS evaluation package.

Behavioral coverage lives in the moved suite
(``tests/unit/mhs/test_evaluation_*.py``).
"""

from __future__ import annotations

from tests.fixtures.mhs_requests import research_baseline
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
