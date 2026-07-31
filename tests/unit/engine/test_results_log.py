from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.types import CostModel, StrategySpec
from src.engine.backtest import run_backtest
from src.validation.metrics import compute_metrics
from src.validation.reliability_gate import (
    FoldDistributionResult,
    ReliabilityGateResult,
)
from src.engine.results_log import load_runs, record_run


def _gate_fixture() -> tuple[ReliabilityGateResult, FoldDistributionResult, ReliabilityGateResult]:
    observation = ReliabilityGateResult(
        lcb90_cagr=0.05, lcb95_cagr=0.03, p_negative=0.0,
        point_cagr=0.10, t_stat=2.5, trade_count=50,
        block_size_used=1, verdict="PASS",
    )
    fold = FoldDistributionResult(
        n_folds=2, median_fold_cagr=0.05, worst_fold_cagr=0.02,
        median_fold_calmar=0.8, max_period_contribution=0.30, gate_pass=True,
    )
    stress = ReliabilityGateResult(
        lcb90_cagr=0.03, lcb95_cagr=0.01, p_negative=0.0,
        point_cagr=0.08, t_stat=2.0, trade_count=50,
        block_size_used=1, verdict="PASS",
    )
    return observation, fold, stress


class TestResultsLog:
    def test_record_run_appends_one_line_per_call(
        self, tmp_path: Path, spec: StrategySpec, costs: CostModel, bars_ramp: pd.DataFrame,
    ) -> None:
        log_path = tmp_path / "runs.jsonl"
        result = run_backtest(bars_ramp, spec, costs)
        metrics = compute_metrics(result.equity, result.trades)
        observation, fold, stress = _gate_fixture()

        record_run(
            spec=spec, costs=costs, result=result, metrics=metrics,
            start=None, end="2025-12-31", initial_equity=10_000.0,
            observation_gate=observation, fold_distribution=fold, stress_gate=stress,
            log_path=log_path,
        )
        record_run(
            spec=spec, costs=costs, result=result, metrics=metrics,
            start=None, end="2025-12-31", initial_equity=10_000.0,
            observation_gate=observation, fold_distribution=fold, stress_gate=stress,
            log_path=log_path,
        )

        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_load_runs_flattens_spec_and_metrics(
        self, tmp_path: Path, spec: StrategySpec, costs: CostModel, bars_ramp: pd.DataFrame,
    ) -> None:
        log_path = tmp_path / "runs.jsonl"
        result = run_backtest(bars_ramp, spec, costs)
        metrics = compute_metrics(result.equity, result.trades)
        observation, fold, stress = _gate_fixture()

        record_run(
            spec=spec, costs=costs, result=result, metrics=metrics,
            start=None, end="2025-12-31", initial_equity=10_000.0,
            observation_gate=observation, fold_distribution=fold, stress_gate=stress,
            log_path=log_path,
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
        observation, fold, stress = _gate_fixture()

        rec = record_run(
            spec=spec, costs=costs, result=result, metrics=metrics,
            start=None, end="2025-12-31", initial_equity=10_000.0,
            observation_gate=observation, fold_distribution=fold, stress_gate=stress,
            log_path=log_path,
        )
        # This repo is a git checkout, so a sha should always resolve here.
        assert isinstance(rec["git_sha"], str)
        assert len(rec["git_sha"]) > 0
        assert isinstance(rec["git_dirty"], bool)

    def test_record_run_reliability_schema_and_backward_compat(
        self, tmp_path: Path, spec: StrategySpec, costs: CostModel, bars_ramp: pd.DataFrame,
    ) -> None:
        log_path = tmp_path / "runs.jsonl"
        result = run_backtest(bars_ramp, spec, costs)
        metrics = compute_metrics(result.equity, result.trades)
        observation, fold, stress = _gate_fixture()

        # SC-LOG-01: sealed run (holdout_gate=None)
        sealed = record_run(
            spec=spec, costs=costs, result=result, metrics=metrics,
            start=None, end="2025-12-31", initial_equity=10_000.0,
            observation_gate=observation, fold_distribution=fold, stress_gate=stress,
            holdout_gate=None, log_path=log_path,
        )
        assert sealed["window"] == "observation"
        assert sealed["reliability"]["holdout"] is None
        assert sealed["reliability"]["observation"]["verdict"] == "PASS"
        assert sealed["reliability"]["fold_distribution"]["max_period_contribution"] == 0.30
        assert sealed["reliability"]["stress_test"]["verdict"] == "PASS"

        # SC-LOG-02: unsealed run (holdout_gate present)
        holdout = ReliabilityGateResult(
            lcb90_cagr=0.0, lcb95_cagr=0.0, p_negative=1.0,
            point_cagr=0.0, t_stat=0.0, trade_count=10,
            block_size_used=1, verdict="PENDING",
        )
        unsealed = record_run(
            spec=spec, costs=costs, result=result, metrics=metrics,
            start=None, end="2026-07-07", initial_equity=10_000.0,
            observation_gate=observation, fold_distribution=fold, stress_gate=stress,
            holdout_gate=holdout, log_path=log_path,
        )
        assert unsealed["window"] == "observation+holdout"
        assert unsealed["reliability"]["holdout"]["verdict"] == "PENDING"

        df = load_runs(log_path)
        assert df.loc[0, "reliability.observation.verdict"] == "PASS"
        assert pd.isna(df.loc[0, "reliability.holdout.verdict"])
        assert df.loc[1, "reliability.holdout.verdict"] == "PENDING"
        assert df.loc[0, "window"] == "observation"
        assert df.loc[1, "window"] == "observation+holdout"

        # Backward compat: a pre-revision record without reliability keys must load
        legacy = tmp_path / "legacy.jsonl"
        legacy.write_text(
            '{"ts":"2026-07-30T00:00:00+00:00","symbol":"BTCUSDT","metrics":{"cagr":0.05}}\n',
            encoding="utf-8",
        )
        legacy_df = load_runs(legacy)
        assert len(legacy_df) == 1
        assert legacy_df.loc[0, "metrics.cagr"] == 0.05
