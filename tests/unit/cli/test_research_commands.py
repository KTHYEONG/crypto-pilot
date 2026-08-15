"""CLI research command dispatch for the reduced CLI surface.

SCENARIO_MHS_REFACTOR_09: ``build_root_parser`` exposes only the
``data``/``research`` groups and ``research run portfolio`` carries only the
MHS leaf.
"""

from __future__ import annotations

import pytest

from src.cli.main import build_root_parser


def test_research_portfolio_mhs_horizon_diagnostic_parses() -> None:
    args = build_root_parser().parse_args(
        ["research", "run", "portfolio", "mhs-horizon-diagnostic"],
    )
    assert args.group == "research"
    assert args.research_command == "run"
    assert args.run_command == "portfolio"
    assert args.portfolio_command == "mhs-horizon-diagnostic"
    assert callable(args.handler)


def test_research_run_requires_portfolio() -> None:
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["research", "run"])
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["research", "run", "single"])
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["research", "run", "expert"])


def test_removed_single_subcommands_raise_system_exit() -> None:
    for argv in (
        ["research", "run", "single", "baseline"],
        ["research", "run", "single", "technical"],
        ["research", "run", "single", "carry"],
        ["research", "run", "single", "oi"],
        ["research", "run", "single", "xs-screen"],
        ["research", "run", "single", "xs-growth-sizing"],
        ["research", "run", "single", "xs-baseline-blend"],
        ["research", "run", "single", "xs-baseline-blend-sized"],
        ["research", "run", "single", "xs-baseline-blend-joint"],
    ):
        with pytest.raises(SystemExit):
            build_root_parser().parse_args(argv)


def test_removed_portfolio_subcommands_raise_system_exit() -> None:
    for argv in (
        ["research", "run", "portfolio", "multi"],
        ["research", "run", "portfolio", "blend"],
        ["research", "run", "portfolio", "growth"],
    ):
        with pytest.raises(SystemExit):
            build_root_parser().parse_args(argv)


def test_removed_expert_subcommands_raise_system_exit() -> None:
    for argv in (
        ["research", "run", "expert", "eval"],
        ["research", "run", "expert", "backtest"],
        ["research", "run", "expert", "pipeline"],
        ["research", "run", "expert", "rolling"],
        ["research", "run", "expert", "exit-sweep"],
    ):
        with pytest.raises(SystemExit):
            build_root_parser().parse_args(argv)


def test_removed_provenance_group_raises_system_exit() -> None:
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["provenance"])
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["provenance", "compare-runs"])
