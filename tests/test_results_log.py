from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CostModel, StrategySpec
from src.engine import run_backtest
from src.metrics import compute_metrics
from src.results_log import load_runs, record_run


class TestResultsLog:
    def test_record_run_appends_one_line_per_call(
        self, tmp_path: Path, spec: StrategySpec, costs: CostModel, bars_ramp: pd.DataFrame,
    ) -> None:
        log_path = tmp_path / "runs.jsonl"
        result = run_backtest(bars_ramp, spec, costs)
        metrics = compute_metrics(result.equity, result.trades)

        record_run(
            spec=spec, costs=costs, result=result, metrics=metrics,
            start=None, end="2025-12-31", initial_equity=10_000.0, log_path=log_path,
        )
        record_run(
            spec=spec, costs=costs, result=result, metrics=metrics,
            start=None, end="2025-12-31", initial_equity=10_000.0, log_path=log_path,
        )

        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_load_runs_flattens_spec_and_metrics(
        self, tmp_path: Path, spec: StrategySpec, costs: CostModel, bars_ramp: pd.DataFrame,
    ) -> None:
        log_path = tmp_path / "runs.jsonl"
        result = run_backtest(bars_ramp, spec, costs)
        metrics = compute_metrics(result.equity, result.trades)

        record_run(
            spec=spec, costs=costs, result=result, metrics=metrics,
            start=None, end="2025-12-31", initial_equity=10_000.0, log_path=log_path,
        )

        df = load_runs(log_path)
        assert len(df) == 1
        assert df.loc[0, "spec.entry_period"] == spec.entry_period
        assert df.loc[0, "metrics.cagr"] == metrics.cagr
        assert df.loc[0, "end"] == "2025-12-31"

    def test_load_runs_missing_file_returns_empty(self, tmp_path: Path) -> None:
        df = load_runs(tmp_path / "does_not_exist.jsonl")
        assert df.empty

    def test_record_run_captures_git_state(
        self, tmp_path: Path, spec: StrategySpec, costs: CostModel, bars_ramp: pd.DataFrame,
    ) -> None:
        log_path = tmp_path / "runs.jsonl"
        result = run_backtest(bars_ramp, spec, costs)
        metrics = compute_metrics(result.equity, result.trades)

        rec = record_run(
            spec=spec, costs=costs, result=result, metrics=metrics,
            start=None, end="2025-12-31", initial_equity=10_000.0, log_path=log_path,
        )
        # This repo is a git checkout, so a sha should always resolve here.
        assert isinstance(rec["git_sha"], str)
        assert len(rec["git_sha"]) > 0
        assert isinstance(rec["git_dirty"], bool)
