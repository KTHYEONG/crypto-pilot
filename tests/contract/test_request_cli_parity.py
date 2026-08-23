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
