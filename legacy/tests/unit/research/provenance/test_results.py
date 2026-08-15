from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from src.research.cash_carry.contracts import CashCarrySpec, CarryCostModel
from src.research.contracts import CostModel, StrategySpec
from src.research.baseline.backtest import run_backtest
from src.research.provenance.results import (
    load_runs,
    record_cash_carry_run,
    record_expert_portfolio_run,
    record_run,
)

from src.research.evaluation.promotion import (
    CandidateIdentity,
    compose_promotion_verdict,
)
from src.research.evaluation.metrics import compute_metrics
from src.research.evaluation.reliability import (
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
        assert "candidate" not in rec["promotion"]
        assert rec["promotion"]["status"] == "OBSERVATION_PASS"

        df = load_runs(log_path)
        assert df.loc[0, "promotion.status"] == "OBSERVATION_PASS"

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

    def test_record_cash_carry_run_persists_comparison_summary(
        self, tmp_path: Path, make_carry_data,
    ) -> None:
        from src.research.cash_carry.backtest import run_cash_carry_backtest

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
        assert rec["cash_carry_spec"]["initial_margin_rate"] == 0.30
        assert rec["costs"]["spot_fee_rate"] == 0.001
        assert rec["costs"]["perp_fee_rate"] == 0.0005
        assert "candidate" not in rec
        assert rec["promotion"]["status"] == promotion.status
        assert rec["reliability"]["observation"]["verdict"] == observation.verdict
        df = load_runs(log_path)
        assert df.loc[0, "cash_carry_spec.symbol"] == "BTCUSDT"
        assert df.loc[0, "promotion.status"] == promotion.status

    def test_record_oi_deleveraging_run_persists_screen_summary(
        self, tmp_path: Path, make_oi_market_data,
    ) -> None:
        from src.research.oi_deleveraging.backtest import run_open_interest_deleveraging_screen
        from src.research.provenance.results import record_oi_deleveraging_run

        log_path = tmp_path / "runs.jsonl"
        data = make_oi_market_data(
            n_bars=8,
            mark_return_24h=[-0.01, -0.01, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            oi_change=[-1.0] * 8,
        )
        costs = CostModel()
        result = run_open_interest_deleveraging_screen(data, costs, signal_delay_bars=1)
        metrics = compute_metrics(result.equity, result.trades)
        observation, fold, stress = _gate_fixture()
        identity = CandidateIdentity(
            hypothesis_id="open_interest_deleveraging_v1", code_hash="sha-oi",
            parameters={"signal_delay_bars": 1}, data_start="2024-01-01",
            data_end="2024-01-01", return_source="open_interest_deleveraging_v1",
        )
        promotion = compose_promotion_verdict(observation, fold, stress, None)
        promotion = dataclasses.replace(promotion, candidate=identity)

        rec = record_oi_deleveraging_run(
            symbol="BTCUSDT", signal_delay_bars=1, costs=costs,
            result=result, metrics=metrics, start=None, end="2024-01-01",
            observation_gate=observation, fold_distribution=fold, stress_gate=stress,
            promotion=promotion, candidate=identity, log_path=log_path,
        )

        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert rec["kind"] == "oi_deleveraging"
        assert rec["signal_delay_bars"] == 1
        assert rec["initial_equity"] == 10_000.0
        assert rec["promotion"]["status"] == promotion.status
        df = load_runs(log_path)
        assert df.loc[0, "kind"] == "oi_deleveraging"
        assert df.loc[0, "promotion.status"] == promotion.status

    def test_record_expert_portfolio_run_is_append_only(
        self, tmp_path: Path, spec: StrategySpec, costs: CostModel, bars_ramp: pd.DataFrame,
    ) -> None:
        # EP-08: a valid expert-portfolio run is identifiable, persists the full
        # library fingerprint and realised allocation cost, and logging is
        # strictly append-only (never a rewrite of an existing run).
        log_path = tmp_path / "runs.jsonl"
        result = run_backtest(bars_ramp, spec, costs)
        metrics = compute_metrics(result.equity, result.trades)
        observation, fold, stress = _gate_fixture()
        fingerprint = {
            "experts": [{
                "expert_id": "pair_residual_v1",
                "return_source": "cointegration_residual",
                "family": "pair_residual",
                "symbols": ["A", "B"],
                "runner": "run_pair_residual",
                "code_hash": "abc",
            }],
            "gross_exposure": 1.0,
            "family_exposure_limit": 0.5,
            "symbol_exposure_limit": 0.5,
            "min_history_bars": 30,
            "confidence": 0.90,
        }

        first = record_expert_portfolio_run(
            library_fingerprint=fingerprint,
            allocation_cost_total=0.008,
            result=result,
            metrics=metrics,
            observation_gate=observation,
            fold_distribution=fold,
            stress_gate=stress,
            promotion=compose_promotion_verdict(observation, fold, stress, None),
            log_path=log_path,
        )
        second = record_expert_portfolio_run(
            library_fingerprint=fingerprint,
            allocation_cost_total=0.008,
            result=result,
            metrics=metrics,
            observation_gate=observation,
            fold_distribution=fold,
            stress_gate=stress,
            promotion=compose_promotion_verdict(observation, fold, stress, None),
            log_path=log_path,
        )

        assert first["kind"] == "expert_portfolio"
        assert second["kind"] == "expert_portfolio"
        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert first["library_fingerprint"]["experts"][0]["expert_id"] == "pair_residual_v1"
        assert first["allocation_cost_total"] == 0.008
        assert first["reliability"]["observation"]["verdict"] == "PASS"
        assert first["promotion"]["status"] == "OBSERVATION_PASS"
        df = load_runs(log_path)
        assert len(df) == 2
        assert df.loc[0, "kind"] == "expert_portfolio"
        assert df.loc[0, "allocation_cost_total"] == 0.008

    def test_record_expert_portfolio_run_rejects_incomplete_evidence(
        self, tmp_path: Path, spec: StrategySpec, costs: CostModel, bars_ramp: pd.DataFrame,
    ) -> None:
        # an incomplete fingerprint or non-finite realised cost appends no row.
        log_path = tmp_path / "runs.jsonl"
        result = run_backtest(bars_ramp, spec, costs)
        metrics = compute_metrics(result.equity, result.trades)
        observation, fold, stress = _gate_fixture()
        with pytest.raises(ValueError, match="experts"):
            record_expert_portfolio_run(
                library_fingerprint={"gross_exposure": 1.0},
                allocation_cost_total=0.001,
                result=result,
                metrics=metrics,
                observation_gate=observation,
                fold_distribution=fold,
                stress_gate=stress,
                log_path=log_path,
            )
        with pytest.raises(ValueError, match="finite"):
            record_expert_portfolio_run(
                library_fingerprint={"experts": []},
                allocation_cost_total=float("nan"),
                result=result,
                metrics=metrics,
                observation_gate=observation,
                fold_distribution=fold,
                stress_gate=stress,
                log_path=log_path,
            )
        assert not log_path.exists() or log_path.read_text(encoding="utf-8").strip() == ""
