from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from src.core.types import CashCarrySpec, CarryCostModel, CostModel, StrategySpec
from src.engine.backtest import run_backtest
from src.engine.results_log import load_runs, record_cash_carry_run, record_run
from src.validation.candidate_promotion import (
    CandidateIdentity,
    compose_promotion_verdict,
)
from src.validation.metrics import compute_metrics
from src.validation.reliability_gate import (
    FoldDistributionResult,
    ReliabilityGateResult,
)


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

    def test_record_run_persists_promotion_field(
        self, tmp_path: Path, spec: StrategySpec, costs: CostModel, bars_ramp: pd.DataFrame,
    ) -> None:
        # SC-LOG-03: one append-only JSONL row includes candidate identity, status,
        # and individual gate evidence in the promotion field.
        log_path = tmp_path / "runs.jsonl"
        result = run_backtest(bars_ramp, spec, costs)
        metrics = compute_metrics(result.equity, result.trades)
        observation, fold, stress = _gate_fixture()

        promotion = compose_promotion_verdict(observation, fold, stress, None)
        identity = CandidateIdentity(
            hypothesis_id="hyp-001", code_hash="sha-abc", parameters={"period": 20},
            data_start="2019-01-01", data_end="2025-12-31", return_source="breakout",
        )
        promotion = dataclasses.replace(promotion, candidate=identity)

        rec = record_run(
            spec=spec, costs=costs, result=result, metrics=metrics,
            start=None, end="2025-12-31", initial_equity=10_000.0,
            observation_gate=observation, fold_distribution=fold, stress_gate=stress,
            promotion=promotion, log_path=log_path,
        )

        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert rec["promotion"]["status"] == "OBSERVATION_PASS"
        assert rec["promotion"]["observation_verdict"] == "PASS"
        assert rec["promotion"]["fold_gate_pass"] is True
        assert rec["promotion"]["stress_verdict"] == "PASS"
        assert rec["promotion"]["holdout_verdict"] is None
        assert rec["promotion"]["candidate"]["hypothesis_id"] == "hyp-001"
        assert rec["promotion"]["candidate"]["return_source"] == "breakout"

        df = load_runs(log_path)
        assert df.loc[0, "promotion.status"] == "OBSERVATION_PASS"
        assert df.loc[0, "promotion.candidate.hypothesis_id"] == "hyp-001"

    def test_record_run_without_promotion_keeps_field_null_and_backward_compat(
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
        assert rec["promotion"] is None
        assert rec["reliability"]["observation"]["verdict"] == "PASS"
        assert rec["window"] == "observation"

    def test_record_cash_carry_run_persists_spec_costs_and_candidate(
        self, tmp_path: Path, make_carry_data,
    ) -> None:
        from src.engine.cash_carry_backtest import run_cash_carry_backtest

        log_path = tmp_path / "runs.jsonl"
        data = make_carry_data(
            n_bars=4,
            funding={
                "2024-01-01 00:00": 0.0,
                "2024-01-01 04:00": 0.001,
                "2024-01-01 08:00": 0.0,
                "2024-01-01 12:00": 0.0,
            },
            borrow=[0.0, 0.0, 0.0, 0.0],
        )
        spec = CashCarrySpec(symbol="BTCUSDT")
        costs = CarryCostModel()
        result = run_cash_carry_backtest(data, spec, costs)
        metrics = compute_metrics(result.equity, result.trades)
        observation, fold, stress = _gate_fixture()
        identity = CandidateIdentity(
            hypothesis_id="cash_and_carry_basis", code_hash="sha-carry",
            parameters={"margin_model": {"initial_margin_rate": 0.10}},
            data_start="2024-01-01", data_end="2024-01-01", return_source="funding_carry",
        )
        promotion = compose_promotion_verdict(observation, fold, stress, None)
        promotion = dataclasses.replace(promotion, candidate=identity)

        rec = record_cash_carry_run(
            symbol="BTCUSDT", cash_carry_spec=spec, costs=costs,
            result=result, metrics=metrics, start=None, end="2024-01-01",
            initial_equity=10_000.0, observation_gate=observation,
            fold_distribution=fold, stress_gate=stress, promotion=promotion,
            candidate=identity, log_path=log_path,
        )

        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert rec["kind"] == "cash_carry"
        assert rec["cash_carry_spec"]["initial_margin_rate"] == 0.10
        assert rec["costs"]["spot_fee_rate"] == 0.001
        assert rec["costs"]["perp_fee_rate"] == 0.0005
        assert rec["candidate"]["hypothesis_id"] == "cash_and_carry_basis"
        df = load_runs(log_path)
        assert df.loc[0, "cash_carry_spec.symbol"] == "BTCUSDT"
        assert df.loc[0, "candidate.return_source"] == "funding_carry"
