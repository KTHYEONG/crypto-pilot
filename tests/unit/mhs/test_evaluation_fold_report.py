from __future__ import annotations

import pandas as pd

from src.application.research.mhs.evaluation import (
    MhsFoldReport,
    _incomplete_fold_report,
)
from src.mhs.evaluation import AnchoredPurgedFold

_FOLD = AnchoredPurgedFold(
    pd.Timestamp("2021-01-01", tz="UTC"),
    pd.Timestamp("2021-01-31", tz="UTC"),
    pd.Timestamp("2021-02-10", tz="UTC"),
    pd.Timestamp("2021-04-19 08:00", tz="UTC"),
    168,
    168,
)


def test_fold_report_fast_horizon_fields_default() -> None:
    # SCENARIO_MHS_FOLD_REPORT_FAST_HORIZON_FIELDS_DEFAULT: MhsFoldReport
    # constructed without explicit fast_horizon_hours/fast_horizon_source
    # defaults to (48, "frozen_default"), mirroring the slow_horizon_* fields'
    # existing defaults so every pre-existing call site stays valid.
    report = MhsFoldReport(
        fold_index=0, validation_start="2021-02-10", validation_end="2021-04-19",
        strict=None, stress=None, primary_valid=False, primary_autocorr_sharpe=0.0,
        primary_naive_sharpe=0.0, primary_net_ann=0.0, primary_geometric_cagr=0.0,
        primary_max_drawdown=0.0, stress_naive_sharpe=0.0, decision_intents=0,
        termination_counts={}, failures=(), strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
    )
    assert report.fast_horizon_hours == 48
    assert report.fast_horizon_source == "frozen_default"
    assert report.slow_horizon_hours == 168
    assert report.slow_horizon_source == "frozen_default"


def test_incomplete_fold_report_keeps_fast_horizon_default() -> None:
    # _incomplete_fold_report (fail-closed fold) must keep the same defaults so
    # a fold that cannot be replayed never fabricates a discovery source.
    report = _incomplete_fold_report(_FOLD, 0, ())
    assert report.fast_horizon_hours == 48
    assert report.fast_horizon_source == "frozen_default"


def test_fold_report_records_fast_discovery_source() -> None:
    # A fold run resolved with a fast fold-scoped override records the selected
    # horizon and source, mirroring the slow_horizon_* recording path.
    report = MhsFoldReport(
        fold_index=0, validation_start="2021-02-10", validation_end="2021-04-19",
        strict=None, stress=None, primary_valid=False, primary_autocorr_sharpe=0.0,
        primary_naive_sharpe=0.0, primary_net_ann=0.0, primary_geometric_cagr=0.0,
        primary_max_drawdown=0.0, stress_naive_sharpe=0.0, decision_intents=0,
        termination_counts={}, failures=(), strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
        fast_horizon_hours=96, fast_horizon_source="fold_train_only_discovery",
    )
    assert report.fast_horizon_hours == 96
    assert report.fast_horizon_source == "fold_train_only_discovery"
    assert report.slow_horizon_hours == 168
    assert report.slow_horizon_source == "frozen_default"
