from __future__ import annotations

import argparse

from src.cli.commands.research import add_research_commands


def test_research_command_registry_loads_mhs_leaf() -> None:
    parser = argparse.ArgumentParser()
    add_research_commands(parser.add_subparsers(dest="root", required=True).add_parser("research"))

    args = parser.parse_args([
        "research", "run", "portfolio", "mhs-horizon-diagnostic", "--no-log-run",
    ])
    assert args.portfolio_command == "mhs-horizon-diagnostic"
    from src.cli.dataclass_args import explicit_field_values
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import resolve_cli_request

    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    assert explicit == {"log_run": False}
    assert resolve_cli_request(explicit).execution_timeframe == "3m"


def test_research_command_registry_has_no_single_tier() -> None:
    import pytest

    parser = argparse.ArgumentParser()
    add_research_commands(parser.add_subparsers(dest="root", required=True).add_parser("research"))
    with pytest.raises(SystemExit):
        parser.parse_args(["research", "run", "single", "baseline"])


def test_research_command_registry_has_no_expert_tier() -> None:
    import pytest

    parser = argparse.ArgumentParser()
    add_research_commands(parser.add_subparsers(dest="root", required=True).add_parser("research"))
    with pytest.raises(SystemExit):
        parser.parse_args(["research", "run", "expert", "eval"])
