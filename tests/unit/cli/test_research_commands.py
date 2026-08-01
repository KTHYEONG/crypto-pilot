from __future__ import annotations

from src.cli.main import build_root_parser
from src.research.contracts import (
    BaselineEvaluationRequest,
    CashCarryEvaluationRequest,
    PortfolioEvaluationRequest,
    SleeveBlendEvaluationRequest,
)
from src.research.expert_portfolio.contracts import ExpertPortfolioEvaluationRequest


def test_expert_portfolio_cli_parses_args_and_dispatches(monkeypatch) -> None:
    calls: list[ExpertPortfolioEvaluationRequest] = []
    monkeypatch.setattr(
        "src.cli.commands.research.run_expert_portfolio_evaluation", calls.append,
    )
    args = build_root_parser().parse_args([
        "research", "run", "expert-portfolio",
        "--library-id", "pair_residual_v1",
        "--start", "2022-04-01",
        "--end", "2025-12-31",
        "--initial-equity", "5000",
        "--no-log-run",
    ])
    args.handler(args)
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
    monkeypatch.setattr(
        "src.cli.commands.research.run_expert_portfolio_evaluation", calls.append,
    )
    args = build_root_parser().parse_args([
        "research", "run", "expert-portfolio", "--library-id", "pair_residual_v1", "--no-log-run",
    ])
    args.handler(args)
    assert calls == [ExpertPortfolioEvaluationRequest(
        library_id="pair_residual_v1",
        start=None,
        end=None,
        initial_equity=10_000.0,
        unseal_holdout=False,
        log_run=False,
    )]


def test_expert_portfolio_cli_requires_library_id() -> None:
    import pytest

    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["research", "run", "expert-portfolio", "--no-log-run"])


def test_baseline_cli_parses_and_dispatches(monkeypatch) -> None:
    calls: list[BaselineEvaluationRequest] = []
    monkeypatch.setattr("src.cli.commands.research.run_baseline_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "baseline", "--symbol", "ETHUSDT", "--no-log-run",
    ])
    args.handler(args)
    assert calls == [BaselineEvaluationRequest(
        symbol="ETHUSDT", log_run=False, unseal_holdout=False,
    )]


def test_portfolio_cli_parses_symbols_and_dispatches(monkeypatch) -> None:
    calls: list[PortfolioEvaluationRequest] = []
    monkeypatch.setattr("src.cli.commands.research.run_portfolio_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "portfolio", "--symbols", "BTCUSDT", "ETHUSDT", "--no-log-run",
    ])
    args.handler(args)
    assert calls == [PortfolioEvaluationRequest(
        symbols=("BTCUSDT", "ETHUSDT"), log_run=False,
    )]


def test_cash_carry_cli_parses_and_dispatches(monkeypatch) -> None:
    calls: list[CashCarryEvaluationRequest] = []
    monkeypatch.setattr("src.cli.commands.research.run_cash_carry_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "cash-carry", "--symbol", "BTCUSDT", "--no-log-run",
    ])
    args.handler(args)
    assert calls == [CashCarryEvaluationRequest(symbol="BTCUSDT", log_run=False)]


def test_sleeve_blend_cli_parses_args_and_dispatches(monkeypatch) -> None:
    calls: list[SleeveBlendEvaluationRequest] = []
    monkeypatch.setattr("src.cli.commands.research.run_sleeve_blend_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "sleeve-blend",
        "--symbols", "BTCUSDT", "ETHUSDT",
        "--mdd-budget-fraction", "0.80",
        "--start", "2022-04-01",
        "--end", "2025-01-01",
        "--initial-equity", "5000",
        "--no-log-run",
    ])
    args.handler(args)
    assert calls == [SleeveBlendEvaluationRequest(
        symbols=("BTCUSDT", "ETHUSDT"),
        mdd_budget_fraction=0.80,
        candidate_kind="fixed_long_only_v1",
        start="2022-04-01",
        end="2025-01-01",
        initial_equity=5000.0,
        unseal_holdout=False,
        log_run=False,
    )]


def test_sleeve_blend_cli_directional_candidate_kind(monkeypatch) -> None:
    calls: list[SleeveBlendEvaluationRequest] = []
    monkeypatch.setattr("src.cli.commands.research.run_sleeve_blend_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "sleeve-blend",
        "--candidate-kind", "funding_signed_directional_v1", "--no-log-run",
    ])
    args.handler(args)
    assert calls == [SleeveBlendEvaluationRequest(
        symbols=("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT"),
        mdd_budget_fraction=0.85,
        candidate_kind="funding_signed_directional_v1",
        log_run=False,
    )]
