from __future__ import annotations

import argparse

from src.cli.commands.research import add_research_commands
from src.cli.main import build_root_parser
from src.research.contracts import OIDeleveragingEvaluationRequest


def test_oi_deleveraging_cli_dispatches_fixed_request(monkeypatch) -> None:
    calls: list[OIDeleveragingEvaluationRequest] = []
    monkeypatch.setattr(
        "src.application.research.oi.evaluation.run_oi_deleveraging_evaluation", calls.append,
    )
    args = build_root_parser().parse_args([
        "research", "run", "single", "oi",
        "--symbol", "ETHUSDT", "--start", "2022-04-01", "--no-log-run",
    ])
    args.handler(args)

    assert calls == [OIDeleveragingEvaluationRequest(
        symbol="ETHUSDT", start="2022-04-01", log_run=False,
    )]


def test_research_command_registry_loads_all_leaf_commands() -> None:
    parser = argparse.ArgumentParser()
    add_research_commands(parser.add_subparsers(dest="root", required=True).add_parser("research"))

    args = parser.parse_args([
        "research", "run", "portfolio", "mhs-horizon-diagnostic", "--no-log-run",
    ])
    assert args.portfolio_command == "mhs-horizon-diagnostic"
    assert args.execution_timeframe == "5m"
