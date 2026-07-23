"""Compound-only futures runner CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from src.application.futures.runner.compound_config import (
    build_compound_run_config,
)
from src.application.futures.runner.compound_main import run_multiscale_compound_main

_logger = logging.getLogger(__name__)

_REMOVED_FLAGS: tuple[str, ...] = (
    "phase",
    "trials",
    "timeframe",
    "mode",
    "alpha_only",
    "skip_universe",
    "skip_data_sync",
    "bypass_champion_guard",
    "symbols",
    "quick_backtest",
    "tf",
    "alpha_foundry",
    "alpha_foundry_total_l1_budget",
    "alpha_foundry_min_conviction_lcb_bps",
    "alpha_foundry_enable_fast_tf",
    "sync_metrics",
    "reference_date",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compound-only Futures Runner")
    parser.add_argument("--date", type=str, default=None, help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--sync", type=str, default="auto", choices=["auto", "skip"],
                        help="Sync mode (auto, skip)")
    parser.add_argument("--refresh-universe", action="store_true", help="Force universe refresh")
    parser.add_argument("--allow-network-sync", action="store_true", help="Allow network sync if local snapshot incomplete")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    for flag in _REMOVED_FLAGS:
        parser.add_argument(f"--{flag.replace('_', '-')}", action="store_true", default=None,
                            help=argparse.SUPPRESS)
    return parser


def _to_mapping(parsed: object) -> Mapping[str, Any]:
    if isinstance(parsed, Mapping):
        return parsed
    if hasattr(parsed, "__dict__"):
        return vars(parsed)
    return {}


def check_removed_flags(parsed: object) -> None:
    parsed_dict = _to_mapping(parsed)
    for flag in _REMOVED_FLAGS:
        raw = parsed_dict.get(flag)
        if raw is not None and raw is not False:
            raise SystemExit(f"error: unrecognized arguments: --{flag.replace('_', '-')}")


def run_from_cli(argv: Sequence[str] | None = None) -> int:
    return run_multiscale_cli(argv)


def run_multiscale_cli(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        _logger.error("unrecognized arguments: %s", unknown)
        return 2
    try:
        check_removed_flags(args)
    except SystemExit:
        return 2
    try:
        config = build_compound_run_config(args)
    except ValueError as exc:
        _logger.error("config error: %s", exc)
        return 2
    result = run_multiscale_compound_main(config)
    return result.exit_code


def cli(argv: Sequence[str] | None = None) -> int:
    """Run the sole supported multiscale futures pipeline."""
    return run_multiscale_cli(argv)


def main() -> int:
    return cli(sys.argv[1:] if len(sys.argv) > 1 else None)
