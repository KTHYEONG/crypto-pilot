from __future__ import annotations

import sys

import pytest

from src.cli import run_expert_portfolio_backtest as cli
from src.research.expert_portfolio.contracts import ExpertPortfolioEvaluationRequest


def test_expert_portfolio_cli_parses_args_and_dispatches(monkeypatch) -> None:
    calls: list[ExpertPortfolioEvaluationRequest] = []
    monkeypatch.setattr(cli, "run_expert_portfolio_evaluation", calls.append)
    monkeypatch.setattr(sys, "argv", [
        "run_expert_portfolio_backtest",
        "--library-id", "pair_residual_v1",
        "--start", "2022-04-01",
        "--end", "2025-12-31",
        "--initial-equity", "5000",
        "--no-log-run",
    ])

    cli.main()

    assert calls == [ExpertPortfolioEvaluationRequest(
        library_id="pair_residual_v1",
        start="2022-04-01",
        end="2025-12-31",
        initial_equity=5000.0,
        unseal_holdout=False,
        log_run=False,
    )]


def test_expert_portfolio_cli_defaults_keep_holdout_sealed(monkeypatch) -> None:
    calls: list[ExpertPortfolioEvaluationRequest] = []
    monkeypatch.setattr(cli, "run_expert_portfolio_evaluation", calls.append)
    monkeypatch.setattr(sys, "argv", [
        "run_expert_portfolio_backtest", "--library-id", "pair_residual_v1", "--no-log-run",
    ])

    cli.main()

    assert calls == [ExpertPortfolioEvaluationRequest(
        library_id="pair_residual_v1",
        start=None,
        end=None,
        initial_equity=10_000.0,
        unseal_holdout=False,
        log_run=False,
    )]


def test_expert_portfolio_cli_requires_library_id(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_expert_portfolio_backtest", "--no-log-run"])
    with pytest.raises(SystemExit):
        cli.main()
