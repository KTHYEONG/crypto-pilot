"""I3 REQUEST/CLI DECLARE-ONCE contract.

The argparse flag set exposed by the mhs CLI must equal the flag set derived
from ``MhsDiagnosticRequest`` field ``cli`` metadata. Every CLI-exposed request
field carries that metadata, so adding one execution option requires editing
exactly one field.
"""

from __future__ import annotations

import argparse
import dataclasses

from src.application.research.mhs.evaluation import MhsDiagnosticRequest
from src.cli.commands.research.mhs import add_mhs_commands
from src.cli.dataclass_args import build_parser_from_dataclass


def _flag_set(parser: argparse.ArgumentParser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in ("-h", "--help")
    }


def _mhs_cli_flags() -> set[str]:
    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    flags = _flag_set(parser)
    # The output tier is a persistence switch on the parser, not a request field.
    flags.discard("--output-tier")
    # SCENARIO_MHS_LEVERAGE_SCAN_07: diagnostic-only short-circuit switches,
    # never construct MhsDiagnosticRequest.
    flags.discard("--leverage-frontier-scan")
    flags.discard("--leverage-frontier-multiples")
    # 연구-라이브 seam 스위치: 완료된 리포트를 사후 소비할 뿐 요청 필드가 아니다.
    flags.discard("--emit-target-weights")
    flags.discard("--emit-signal-state")
    return flags


def _metadata_flags() -> set[str]:
    parser = argparse.ArgumentParser()
    build_parser_from_dataclass(parser, MhsDiagnosticRequest)
    return _flag_set(parser)


def test_cli_flags_equal_metadata_derived_flags() -> None:
    cli_flags = _mhs_cli_flags()
    meta_flags = _metadata_flags()
    assert cli_flags == meta_flags, (
        f"CLI/metadata flag divergence: only_cli={sorted(cli_flags - meta_flags)} "
        f"only_metadata={sorted(meta_flags - cli_flags)}"
    )


def test_every_cli_exposed_field_carries_flag_metadata() -> None:
    for field in dataclasses.fields(MhsDiagnosticRequest):
        if "flag" in field.metadata:
            assert field.metadata["flag"], f"field {field.name} has empty flag metadata"


def test_pnl_vol_target_mode_choices_match_cli_and_metadata() -> None:
    """SCENARIO_MHS_CONSTANT_RISK_REQUEST_CLI_PARITY: the request contract's
    cli_param choices and the hand-written CLI argparse choices for
    --pnl-vol-target-mode stay exactly equal (4 registered values)."""
    sub = argparse.ArgumentParser().add_subparsers()
    add_mhs_commands(sub)
    parser = sub.choices["mhs-horizon-diagnostic"]
    cli_action = next(a for a in parser._actions if a.dest == "pnl_vol_target_mode")
    field = next(
        f for f in dataclasses.fields(MhsDiagnosticRequest)
        if f.name == "pnl_vol_target_mode"
    )
    meta_choices = field.metadata["choices"]
    # 선언 순서는 계약/CLI 간 다를 수 있으므로 등록 값집합의 정확한 일치를 단언한다.
    assert sorted(cli_action.choices) == sorted(meta_choices)
    assert len(meta_choices) == 4
    assert "constant_risk" in meta_choices


def test_scenario_mhs_dd_brake_10_cli_request_parity() -> None:
    """SCENARIO_MHS_DD_BRAKE_10_CLI_REQUEST_PARITY: --exposure-drawdown-brake is declared exactly
    once on the request (cli_param metadata) and mirrored by the hand-written
    CLI; the MhsRunConfig no-arg parity stays intact with brake=False."""
    cli_flags = _mhs_cli_flags()
    meta_flags = _metadata_flags()
    assert "--exposure-drawdown-brake" in cli_flags
    assert "--exposure-drawdown-brake" in meta_flags

    from src.cli.main import build_root_parser
    from src.mhs.pipeline.config import MhsRunConfig

    args = build_root_parser().parse_args(
        ["research", "run", "portfolio", "mhs-horizon-diagnostic"],
    )
    assert args.exposure_drawdown_brake is False
    assert (
        dataclasses.asdict(MhsRunConfig.from_namespace(args))
        == dataclasses.asdict(MhsRunConfig())
    )
    assert dataclasses.asdict(MhsRunConfig())["exposure_drawdown_brake"] is False
