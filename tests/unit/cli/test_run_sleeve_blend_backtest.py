from __future__ import annotations

import sys

from src.cli import run_sleeve_blend_backtest as cli
from src.research.contracts import SleeveBlendEvaluationRequest


def test_sleeve_blend_cli_parses_args_and_dispatches(monkeypatch) -> None:
    calls: list[SleeveBlendEvaluationRequest] = []
    monkeypatch.setattr(cli, "run_sleeve_blend_evaluation", calls.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_sleeve_blend_backtest",
            "--symbols", "BTCUSDT", "ETHUSDT",
            "--mdd-budget-fraction", "0.80",
            "--start", "2022-04-01",
            "--end", "2025-01-01",
            "--initial-equity", "5000",
            "--no-log-run",
        ],
    )

    cli.main()

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


def test_sleeve_blend_cli_defaults_match_measured_sleeve_set(monkeypatch) -> None:
    calls: list[SleeveBlendEvaluationRequest] = []
    monkeypatch.setattr(cli, "run_sleeve_blend_evaluation", calls.append)
    monkeypatch.setattr(sys, "argv", ["run_sleeve_blend_backtest", "--no-log-run"])

    cli.main()

    assert calls == [SleeveBlendEvaluationRequest(
        symbols=("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT"),
        mdd_budget_fraction=0.85,
        candidate_kind="fixed_long_only_v1",
        start=None,
        end=None,
        initial_equity=10_000.0,
        unseal_holdout=False,
        log_run=False,
    )]


def test_sleeve_blend_cli_directional_candidate_kind(monkeypatch) -> None:
    calls: list[SleeveBlendEvaluationRequest] = []
    monkeypatch.setattr(cli, "run_sleeve_blend_evaluation", calls.append)
    monkeypatch.setattr(sys, "argv", [
        "run_sleeve_blend_backtest", "--candidate-kind", "funding_signed_directional_v1",
        "--no-log-run",
    ])

    cli.main()

    assert calls == [SleeveBlendEvaluationRequest(
        symbols=("BTCUSDT", "ETHUSDT", "AVAXUSDT", "BNBUSDT", "DOGEUSDT"),
        mdd_budget_fraction=0.85,
        candidate_kind="funding_signed_directional_v1",
        start=None,
        end=None,
        initial_equity=10_000.0,
        unseal_holdout=False,
        log_run=False,
    )]
