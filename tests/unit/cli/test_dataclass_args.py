"""Declare-once CLI contract for request dataclasses.

P4 path-presence pin: ``src.cli.dataclass_args`` moved with the layout
(behavioral coverage for the full CLI surface lives in
``tests/contract/test_request_cli_parity.py``).
"""

from __future__ import annotations

import argparse
import dataclasses

from src.cli.dataclass_args import build_parser_from_dataclass, request_from_namespace


@dataclasses.dataclass
class _SampleRequest:
    symbol: str = dataclasses.field(
        default="BTCUSDT", metadata={"flag": "--symbol", "help": "Trading symbol."}
    )
    verbose: bool = dataclasses.field(
        default=False, metadata={"flag": "--verbose", "help": "Verbose output."}
    )


def test_build_parser_registers_flagged_fields() -> None:
    parser = build_parser_from_dataclass(argparse.ArgumentParser(), _SampleRequest)
    args = parser.parse_args(["--symbol", "ETHUSDT", "--verbose"])
    assert args.symbol == "ETHUSDT"
    assert args.verbose is True


def test_request_round_trip_restores_dataclass() -> None:
    parser = build_parser_from_dataclass(argparse.ArgumentParser(), _SampleRequest)
    request = request_from_namespace(_SampleRequest, parser.parse_args([]))
    assert request == _SampleRequest()
