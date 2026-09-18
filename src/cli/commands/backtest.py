"""Canonical three-minute inventory backtest command."""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import pandas as pd

from src.common.paths import RESULTS_DIR
from src.mhs.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING
from src.mhs.resources import MhsMemoryBudget

_logger = logging.getLogger("MhsBacktestCli")


def _utc_timestamp(raw: str | None, label: str, default: pd.Timestamp) -> pd.Timestamp:
    if raw is None:
        return default
    try:
        parsed = pd.Timestamp(raw)
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"invalid {label}: {exc}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def _resolve_budget(args: argparse.Namespace) -> MhsMemoryBudget:
    defaults = MhsMemoryBudget()
    total = args.total_tree_pss_bytes if args.total_tree_pss_bytes is not None else defaults.total_tree_pss_bytes
    replay = args.replay_tree_pss_bytes if args.replay_tree_pss_bytes is not None else defaults.replay_tree_pss_bytes
    reserve = args.min_available_bytes if args.min_available_bytes is not None else defaults.min_available_bytes
    try:
        return MhsMemoryBudget(
            total_tree_pss_bytes=total, replay_tree_pss_bytes=replay, min_available_bytes=reserve,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid resource budget: {exc}") from exc


def _resolve_destinations(args: argparse.Namespace) -> tuple[Path, Path, Path, Path | None]:
    if args.output is None:
        run_root = RESULTS_DIR / "mhs_backtest"
        run_root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=str(run_root)))
        output = run_dir / "primary.json"
        failure_output = run_dir / "failure.json"
        run_output = run_dir / "run.json"
    else:
        output = Path(args.output)
        if args.failure_output is not None:
            failure_output = Path(args.failure_output)
        else:
            failure_output = output.with_name(f"{output.stem}.failure.json")
        if args.run_output is not None:
            run_output = Path(args.run_output)
        else:
            run_output = output.with_name(f"{output.stem}.run.json")
    targets_output = Path(args.targets_output) if args.targets_output is not None else None
    return output, failure_output, run_output, targets_output


def add_backtest_commands(backtest_parser: argparse.ArgumentParser) -> None:
    """Register the default MHS inventory evaluation command.

    Args:
        backtest_parser: Root-owned parser for the backtest command group.
    Returns:
        None; registers the mhs leaf and its supervised handler.
    """
    sub = backtest_parser.add_subparsers(dest="command", required=True)
    mhs = sub.add_parser(
        "mhs",
        help="Supervised 3m inventory evaluation of the continuous MHS process.",
        description="Canonical three-minute inventory evidence with comparative hourly proxy state.",
    )
    mhs.add_argument("--start", default=None, help="UTC source start; date-only values are UTC.")
    mhs.add_argument("--end", default=None, help="UTC registered evaluation end.")
    mhs.add_argument("--data-root", default=None, help="Existing OHLCV root override.")
    mhs.add_argument(
        "--output", default=None,
        help="Fresh primary inventory JSON destination; omitted creates a unique run directory.",
    )
    mhs.add_argument(
        "--failure-output", default=None,
        help="Fresh domain-failure JSON destination; omitted uses the <output.stem>.failure.json sibling.",
    )
    mhs.add_argument(
        "--run-output", default=None,
        help="Fresh supervisor outcome JSON destination; omitted uses the <output.stem>.run.json sibling.",
    )
    mhs.add_argument("--targets-output", default=None, help="Optional fresh exact-target parquet destination.")
    mhs.add_argument(
        "--rebalance-tracking-error-threshold", type=float, default=None,
        help="Existing optional process adoption control; omitted preserves baseline.",
    )
    mhs.add_argument("--timeout-seconds", type=float, default=None, help="Optional positive finite wall timeout.")
    mhs.add_argument(
        "--poll-seconds", type=float, default=0.25, help="Positive finite resource observation interval.",
    )
    mhs.add_argument("--total-tree-pss-bytes", type=int, default=None, help="Total process-tree PSS ceiling in bytes.")
    mhs.add_argument("--replay-tree-pss-bytes", type=int, default=None, help="Replay process-tree PSS ceiling in bytes.")
    mhs.add_argument("--min-available-bytes", type=int, default=None, help="Minimum effective physical headroom in bytes.")
    mhs.add_argument(
        "--execution-timeframe", choices=["3m"], default="3m",
        help="Execution replay resolution; fixed to 3m and never changes strategy cadence.",
    )
    mhs.set_defaults(handler=run_mhs_backtest)


def run_mhs_backtest(args: argparse.Namespace) -> None:
    """Run the canonical three-minute process evaluation under source supervision.

    Args:
        args: Parsed dates, evidence paths, policy and resource controls.
    Returns:
        None after successful worker completion and outcome persistence.
    Raises:
        SystemExit: Arguments are invalid or observed execution is non-success.
        OSError: Launch or outcome persistence fails.
    """
    from src.application.mhs_supervisor import run_mhs_process_backtest

    if getattr(args, "execution_timeframe", "3m") != "3m":
        raise SystemExit(f"execution-timeframe must be 3m, got {getattr(args, 'execution_timeframe', None)!r}")
    start = _utc_timestamp(getattr(args, "start", None), "start", DISCOVERY_START)
    end = _utc_timestamp(getattr(args, "end", None), "end", PROCESS_EVALUATION_CEILING)
    budget = _resolve_budget(args)
    output, failure_output, run_output, targets_output = _resolve_destinations(args)
    _logger.info(
        "[EVAL] backtest mhs output=%s failure_output=%s run_output=%s targets_output=%s",
        output, failure_output, run_output, targets_output,
    )
    try:
        run = run_mhs_process_backtest(
            start=start, end=end, data_root=args.data_root, output=output,
            failure_output=failure_output, run_output=run_output,
            targets_output=targets_output,
            tracking_error_threshold=args.rebalance_tracking_error_threshold,
            timeout_seconds=args.timeout_seconds, poll_seconds=args.poll_seconds,
            memory_budget=budget,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid backtest controls: {exc}") from exc
    if run.status != "completed":
        raise SystemExit(1)
