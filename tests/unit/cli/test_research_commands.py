from __future__ import annotations

import pytest

from src.cli.main import build_root_parser
from src.research.contracts import (
    BaselineEvaluationRequest,
    CashCarryEvaluationRequest,
    PortfolioEvaluationRequest,
    SleeveBlendEvaluationRequest,
    TechnicalExpertEvaluationRequest,
)
from src.research.expert_portfolio.models import ExpertPortfolioEvaluationRequest


def test_expert_portfolio_cli_parses_args_and_dispatches(monkeypatch) -> None:
    calls: list[ExpertPortfolioEvaluationRequest] = []
    monkeypatch.setattr(
        "src.application.research.expert_portfolio.evaluation.run_expert_portfolio_evaluation", calls.append,
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
        "src.application.research.expert_portfolio.evaluation.run_expert_portfolio_evaluation", calls.append,
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
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["research", "run", "expert-portfolio", "--no-log-run"])


def test_technical_expert_cli_parses_and_dispatches(monkeypatch) -> None:
    calls: list[TechnicalExpertEvaluationRequest] = []
    monkeypatch.setattr(
        "src.application.research.technical_experts.evaluation.run_technical_expert_evaluation", calls.append,
    )
    args = build_root_parser().parse_args([
        "research", "run", "technical-expert",
        "--candidate-id", "technical_macd_histogram_regime_long_v1",
        "--symbol", "BTCUSDT",
        "--start", "2022-04-01",
        "--end", "2025-12-31",
        "--initial-equity", "5000",
        "--no-log-run",
    ])
    args.handler(args)
    assert calls == [TechnicalExpertEvaluationRequest(
        candidate_id="technical_macd_histogram_regime_long_v1",
        symbol="BTCUSDT",
        start="2022-04-01",
        end="2025-12-31",
        initial_equity=5000.0,
        unseal_holdout=False,
        log_run=False,
    )]


def test_technical_expert_cli_accepts_candidate_and_symbol_only() -> None:
    # TE-CLI: the frozen screen takes only a candidate and symbol; any
    # indicator/threshold flag is rejected at the parser boundary.
    with pytest.raises(SystemExit):
        build_root_parser().parse_args([
            "research", "run", "technical-expert",
            "--candidate-id", "technical_macd_histogram_regime_long_v1",
            "--rsi-period", "14",
        ])


def test_technical_expert_cli_requires_candidate_id() -> None:
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["research", "run", "technical-expert", "--symbol", "BTCUSDT"])


def test_baseline_cli_parses_and_dispatches(monkeypatch) -> None:
    calls: list[BaselineEvaluationRequest] = []
    monkeypatch.setattr("src.application.research.baseline.evaluation.run_baseline_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "baseline", "--symbol", "ETHUSDT", "--no-log-run",
    ])
    args.handler(args)
    assert calls == [BaselineEvaluationRequest(
        symbol="ETHUSDT", log_run=False, unseal_holdout=False,
    )]


def test_portfolio_cli_parses_symbols_and_dispatches(monkeypatch) -> None:
    calls: list[PortfolioEvaluationRequest] = []
    monkeypatch.setattr("src.application.research.portfolio.evaluation.run_portfolio_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "portfolio", "--symbols", "BTCUSDT", "ETHUSDT", "--no-log-run",
    ])
    args.handler(args)
    assert calls == [PortfolioEvaluationRequest(
        symbols=("BTCUSDT", "ETHUSDT"), log_run=False,
    )]


def test_cash_carry_cli_parses_and_dispatches(monkeypatch) -> None:
    calls: list[CashCarryEvaluationRequest] = []
    monkeypatch.setattr("src.application.research.cash_carry.evaluation.run_cash_carry_evaluation", calls.append)
    args = build_root_parser().parse_args([
        "research", "run", "cash-carry", "--symbol", "BTCUSDT", "--no-log-run",
    ])
    args.handler(args)
    assert calls == [CashCarryEvaluationRequest(symbol="BTCUSDT", log_run=False)]


def test_sleeve_blend_cli_parses_args_and_dispatches(monkeypatch) -> None:
    calls: list[SleeveBlendEvaluationRequest] = []
    monkeypatch.setattr("src.application.research.sleeve_blend.evaluation.run_sleeve_blend_evaluation", calls.append)
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
    monkeypatch.setattr("src.application.research.sleeve_blend.evaluation.run_sleeve_blend_evaluation", calls.append)
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
