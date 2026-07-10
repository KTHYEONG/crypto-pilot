"""Futures runner CLI with simplified phase-driven arguments."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from src.application.futures.runner.config import (
    build_run_config_from_args,
)
from src.application.futures.runner.pipeline import run_pipeline

_logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the futures runner CLI parser with --phase {{l0,l1,l2,l3}} and no --alpha-foundry* args."""
    parser = argparse.ArgumentParser(description="Futures Optimization Runner")
    parser.add_argument("--trials", type=int, default=42, help="Number of optimization trials")
    parser.add_argument("--timeframe", type=str, default="4h", help="Trading timeframe")
    parser.add_argument("--date", type=str, default=None, help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--phase", type=str, default="l3", help="Active phase (l0, l1, l2, l3)")
    parser.add_argument("--sync", type=str, default="auto", help="Sync mode (auto, skip)")
    parser.add_argument("--refresh-universe", action="store_true", help="Force universe refresh")
    parser.add_argument("--sync-metrics", action="store_true", help="Sync champion metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    from src.application.futures.runner.config import _REMOVED_ALPHA_FOUNDRY_ARG_KEYS, _REMOVED_ARG_KEYS

    for key in _REMOVED_ALPHA_FOUNDRY_ARG_KEYS:
        if key == "alpha_foundry":
            parser.add_argument(f"--{key.replace('_', '-')}", type=str, default=None, help=argparse.SUPPRESS)
        elif key == "alpha_foundry_total_l1_budget":
            parser.add_argument(f"--{key.replace('_', '-')}", type=int, default=None, help=argparse.SUPPRESS)
        elif key == "alpha_foundry_min_conviction_lcb_bps":
            parser.add_argument(f"--{key.replace('_', '-')}", type=float, default=None, help=argparse.SUPPRESS)
        elif key == "alpha_foundry_enable_fast_tf":
            parser.add_argument(f"--{key.replace('_', '-')}", action="store_true", help=argparse.SUPPRESS)
    for key in _REMOVED_ARG_KEYS:
        parser.add_argument(f"--{key.replace('_', '-')}", action="store_true", help=argparse.SUPPRESS)
    return parser


def run_from_cli(argv: Sequence[str] | None = None) -> int:
    """[ADR_20260705_MAJOR_SYMBOL_REGISTRY_REPLAY_SYNC] Parse CLI args and execute the runner."""
    parser = build_arg_parser()
    args, _ = parser.parse_known_args(argv)
    try:
        run_config = build_run_config_from_args(args)
    except (ValueError, SystemExit) as exc:
        _logger.error("Config error: %s", exc)
        return 2
    result = run_pipeline(run_config, seed=run_config.seed)
    return result.exit_code


def main() -> int:
    return run_from_cli(sys.argv[1:] if len(sys.argv) > 1 else None)
