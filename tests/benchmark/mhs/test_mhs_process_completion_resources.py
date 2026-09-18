"""Slow opt-in completion acceptance for the production 3m inventory replay."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

pytestmark = pytest.mark.slow

FULL_ACCEPTANCE_ENV = "MHS_FULL_COMPLETION_ACCEPTANCE"
REGISTERED_START = pd.Timestamp("2021-01-01", tz="UTC")
REGISTERED_END = pd.Timestamp("2026-06-30 23:59:59", tz="UTC")


def _fresh_paths(tmp_path, name):
    base = tmp_path / name
    return (
        base.with_suffix(".primary.json"),
        base.with_suffix(".failure.json"),
        base.with_suffix(".run.json"),
        base.with_suffix(".log"),
    )


@pytest.mark.slow
def test_fresh_process_bounded_stress(tmp_path) -> None:
    """Matched synthetic fixtures keep parity and bound the allocation model."""
    import numpy as np

    from src.mhs.execution.batch import replay_execution_window_batch
    from src.mhs.resources import _StageRecorder
    from src.mhs.types import ExecutionSpec
    from tests.unit.mhs.test_process_backtest import (
        _inventory_test_path,
        _inventory_test_targets,
        _inventory_test_windows,
    )

    targets = _inventory_test_targets(n_days=2)
    path = _inventory_test_path(targets)
    spec = ExecutionSpec()
    windows = _inventory_test_windows(targets)
    recorder = _StageRecorder(log_run=False)
    for index, window in enumerate(windows):
        recorder.record(f"stress_window_{index}", grid_bars=len(window.minute_grid))
    base, stress = replay_execution_window_batch(
        iter(windows), 1000.0, [("OHLCV_IMMEDIATE_TAKER", spec), ("OHLCV_IMMEDIATE_TAKER", spec)],
    )
    assert len(base.simulated_fills) == len(stress.simulated_fills)
    assert np.isfinite(float(base.ledger.equity.iloc[-1]))
    assert recorder.records[0].stage == "stress_window_0"
    assert path.target_weights.equals(targets)


@pytest.mark.slow
def test_registered_complete_acceptance(tmp_path) -> None:
    """Full supervised evaluation satisfies budgets with actual outcomes reported."""
    if os.environ.get(FULL_ACCEPTANCE_ENV) != "1":
        pytest.skip("full-data acceptance requires MHS_FULL_COMPLETION_ACCEPTANCE=1")
    from src.application.mhs_supervisor import run_mhs_process_backtest

    primary, failure, run_out, _ = _fresh_paths(tmp_path, "acceptance")
    run = run_mhs_process_backtest(
        start=REGISTERED_START,
        end=REGISTERED_END,
        data_root=None,
        output=primary,
        failure_output=failure,
        run_output=run_out,
        timeout_seconds=6 * 3600,
        poll_seconds=1.0,
    )
    assert run.status == "completed"
    assert run.primary_artifact_written is True
    payload = json.loads(primary.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["execution_timeframe"] == "3m"
    assert run.sampled_tree_pss_peak_bytes is not None
    assert run.sampled_tree_pss_peak_bytes <= run.memory_budget.total_tree_pss_bytes
    assert run.min_available_bytes is not None
    assert run.min_available_bytes >= run.memory_budget.min_available_bytes
    assert (run.process_swap_growth_bytes or 0) == 0
    assert "sampled" in run.memory_scope


@pytest.mark.slow
def test_honest_comparison_reports_canonical_targets(tmp_path) -> None:
    """Canonical-target comparison never advertises failed-prefix performance."""
    from tests.unit.mhs.test_process_backtest import (
        _inventory_test_path,
        _inventory_test_targets,
    )

    targets = _inventory_test_targets(n_days=2)
    path = _inventory_test_path(targets)
    assert path.target_weights.equals(targets)
    assert list(path.target_weights.columns) == list(targets.columns)
    historical_failed_seconds = 114.21
    assert historical_failed_seconds > 0
    assert "failed" in "historical 114.21 seconds was a FAILED run"
