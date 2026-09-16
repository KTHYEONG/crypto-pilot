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


def test_cli_emit_deployment_refuses_live_parity_blockers(monkeypatch) -> None:
    import argparse

    import pytest

    import src.cli.commands.research.mhs as mhs_cli
    import src.mhs.pipeline.orchestrator as orchestrator

    # Given: 실제 파서로 인자 구성 (수동 Namespace 조립 금지)
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    mhs_cli.add_mhs_commands(sub)

    def _fail(*_a, **_k):
        raise AssertionError("diagnostic must not run when a parity blocker is set")

    # run_mhs_diagnostic 은 핸들러 안에서 지연 import 되므로 원본 모듈에 패치한다
    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", _fail)

    blocked = parser.parse_args(
        ["mhs-horizon-diagnostic", "--name-drift-trim", "--emit-deployment"]
    )

    # When / Then: 진단 실행 전에 거부
    with pytest.raises(SystemExit, match="name_drift_trim"):
        mhs_cli._run_mhs_horizon_diagnostic(blocked)

    # Given: emit-deployment 없이 trim 만이면 이 가드를 통과해 진단으로 진행한다
    allowed = parser.parse_args(["mhs-horizon-diagnostic", "--name-drift-trim"])
    with pytest.raises(AssertionError, match="diagnostic must not run"):
        mhs_cli._run_mhs_horizon_diagnostic(allowed)
