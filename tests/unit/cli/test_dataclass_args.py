"""Generic declare-once CLI generator contract."""

from __future__ import annotations

import argparse
import dataclasses

import pytest

from src.cli.dataclass_args import add_dataclass_arguments, explicit_field_values


@dataclasses.dataclass
class _SampleRequest:
    symbol: str = dataclasses.field(default="BTCUSDT", metadata={"flag": "--symbol", "help": "Trading symbol."})
    verbose: bool = dataclasses.field(default=False, metadata={"flag": "--verbose", "help": "Verbose."})
    color: bool = dataclasses.field(default=True, metadata={"flag": "--no-color", "help": "Disable color."})
    target: float | None = dataclasses.field(
        default=0.5,
        metadata={"flag": "--target", "help": "Target.", "negate_flag": "--no-target", "arg_type": float, "choices": (0.5, 0.7)},
    )
    context: str = "local"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_dataclass_arguments(parser, _SampleRequest)
    return parser


def test_suppressed_defaults() -> None:
    args = _parser().parse_args([])
    assert explicit_field_values(_SampleRequest, args) == {}
    assert not hasattr(args, "symbol")
    assert not hasattr(args, "verbose")
    assert not hasattr(args, "color")
    assert not hasattr(args, "target")


def test_explicit_values_only() -> None:
    args = _parser().parse_args(["--symbol", "ETHUSDT", "--no-color"])
    assert explicit_field_values(_SampleRequest, args) == {"symbol": "ETHUSDT", "color": False}


def test_negate_pair_exclusive_and_none() -> None:
    assert explicit_field_values(_SampleRequest, _parser().parse_args(["--target", "0.7"])) == {"target": 0.7}
    with pytest.raises(SystemExit):
        _parser().parse_args(["--target", "0.8"])
    with pytest.raises(SystemExit):
        _parser().parse_args(["--target", "0.7", "--no-target"])
    args = _parser().parse_args(["--no-target"])
    assert explicit_field_values(_SampleRequest, args) == {"target": None}


def test_polarity_violation_fails_fast() -> None:
    @dataclasses.dataclass
    class _Bad:
        verbose: bool = dataclasses.field(default=True, metadata={"flag": "--verbose", "help": "x"})

    with pytest.raises(ValueError, match="polarity"):
        add_dataclass_arguments(argparse.ArgumentParser(), _Bad)

    @dataclasses.dataclass
    class _BadNegate:
        verbose: bool = dataclasses.field(
            default=False, metadata={"flag": "--verbose", "help": "x", "negate_flag": "--no-verbose"}
        )

    with pytest.raises(ValueError, match="negate_flag"):
        add_dataclass_arguments(argparse.ArgumentParser(), _BadNegate)


def test_pure_extraction() -> None:
    args = _parser().parse_args(["--symbol", "ETHUSDT", "--no-color"])
    before = dict(vars(args))
    explicit_field_values(_SampleRequest, args)
    assert vars(args) == before


def test_extraction_includes_present_fields_without_metadata() -> None:
    args = argparse.Namespace(context="remote", unrelated="ignored")
    assert explicit_field_values(_SampleRequest, args) == {"context": "remote"}
    assert vars(args) == {"context": "remote", "unrelated": "ignored"}
