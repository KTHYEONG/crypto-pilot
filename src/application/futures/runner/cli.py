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
    parser = argparse.ArgumentParser(description="Futures Optimization Runner")
    parser.add_argument("--trials", type=int, default=42, help="Number of optimization trials")
    parser.add_argument("--timeframe", type=str, default="4h", help="Trading timeframe")
    parser.add_argument("--date", type=str, default=None, help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--phase", type=str, default="l3", help="Active phase (l1, l2, l3)")
    parser.add_argument("--sync", type=str, default="skip", help="Sync mode (full, fast, skip)")
    parser.add_argument("--refresh-universe", action="store_true", help="Force universe refresh")
    parser.add_argument("--sync-metrics", action="store_true", help="Sync champion metrics")
    from src.application.futures.runner.config import _REMOVED_ARG_KEYS
    for key in _REMOVED_ARG_KEYS:
        parser.add_argument(f"--{key.replace('_', '-')}", action="store_true", help=argparse.SUPPRESS)
    return parser


def run_from_cli(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args, _ = parser.parse_known_args(argv)
    try:
        run_config = build_run_config_from_args(args)
    except (ValueError, SystemExit) as exc:
        _logger.error("Config error: %s", exc)
        return 2
    result = run_pipeline(run_config)
    return result.exit_code


def main() -> int:
    return run_from_cli(sys.argv[1:] if len(sys.argv) > 1 else None)
