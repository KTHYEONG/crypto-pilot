"""Canonical three-minute inventory backtest command."""

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path

import pandas as pd

from src.backtests.contracts import RetentionPolicy
from src.common.paths import BACKTESTS_DIR
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


def _resolve_destinations(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """Resolve the sole result document and optional exact-target export for one MHS run.

    Args:
        args: Parsed canonical MHS backtest arguments.
    Returns:
        A fresh `result.json` destination and an optional target parquet destination.
    Raises:
        SystemExit: An explicit destination is invalid or already occupied.
    """
    import os

    targets_output = Path(args.targets_output) if args.targets_output is not None else None
    if targets_output is not None:
        if targets_output.suffix != ".parquet":
            raise SystemExit(f"targets-output must be a parquet path, got {args.targets_output!r}")
        if os.path.lexists(targets_output):
            raise SystemExit(f"targets-output must be fresh: {targets_output} already exists")
    if args.output is None:
        run_root = BACKTESTS_DIR / "runs"
        run_root.mkdir(parents=True, exist_ok=True)
        run_dir = run_root / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / "result.json", targets_output
    output = Path(args.output)
    if output.suffix != ".json":
        raise SystemExit(f"output must be a JSON path, got {args.output!r}")
    if os.path.lexists(output):
        raise SystemExit(f"output must be fresh: {output} already exists")
    return output, targets_output


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
        help="Fresh complete result envelope JSON destination; omitted creates a unique run directory.",
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
    mhs.add_argument(
        "--max-detail-bytes", type=int, default=None,
        help="Optional destructive detail budget in bytes; omitted keeps every managed bundle.",
    )
    mhs.add_argument(
        "--max-detail-runs", type=int, default=None,
        help="Optional destructive detail budget in finalized runs; omitted keeps every managed bundle.",
    )
    mhs.add_argument(
        "--registry-path", default=None,
        help="Local execution registry; omitted uses the canonical backtests registry.",
    )
    mhs.add_argument(
        "--force", action="store_true", default=False,
        help="Execute an equivalent request again instead of reusing the finalized match.",
    )
    mhs.set_defaults(handler=run_mhs_backtest)


def _resolve_retention_policy(args: argparse.Namespace) -> RetentionPolicy | None:
    """Build explicit destructive detail budgets, failing before any workload launch."""
    from src.mhs.params import DEFAULT_DETAIL_RETENTION_MAX_BYTES, DEFAULT_DETAIL_RETENTION_MAX_RUNS

    max_bytes = getattr(args, "max_detail_bytes", None)
    max_runs = getattr(args, "max_detail_runs", None)
    if max_bytes is None:
        max_bytes = DEFAULT_DETAIL_RETENTION_MAX_BYTES
    if max_runs is None:
        max_runs = DEFAULT_DETAIL_RETENTION_MAX_RUNS
    if max_bytes is None and max_runs is None:
        return None
    try:
        return RetentionPolicy(max_detail_bytes=max_bytes, max_detail_runs=max_runs)
    except ValueError as exc:
        raise SystemExit(f"invalid detail retention budget: {exc}") from exc


def _resolve_fingerprint(args: argparse.Namespace, start: pd.Timestamp, end: pd.Timestamp, budget: MhsMemoryBudget) -> str:
    """Compute the immutable reuse fingerprint for one canonical request."""
    from src.application.mhs_supervisor import request_fingerprint

    return request_fingerprint(
        start=start, end=end, data_root=getattr(args, "data_root", None),
        tracking_error_threshold=getattr(args, "rebalance_tracking_error_threshold", None),
        memory_budget=budget,
        execution_timeframe=getattr(args, "execution_timeframe", "3m"),
    )


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
    from src.application.mhs_supervisor import find_reused_run, run_mhs_process_backtest

    if getattr(args, "execution_timeframe", "3m") != "3m":
        raise SystemExit(f"execution-timeframe must be 3m, got {getattr(args, 'execution_timeframe', None)!r}")
    start = _utc_timestamp(getattr(args, "start", None), "start", DISCOVERY_START)
    end = _utc_timestamp(getattr(args, "end", None), "end", PROCESS_EVALUATION_CEILING)
    budget = _resolve_budget(args)
    registry_path = Path(args.registry_path) if getattr(args, "registry_path", None) else BACKTESTS_DIR / "registry.sqlite3"
    retention_policy = _resolve_retention_policy(args)
    fingerprint = _resolve_fingerprint(args, start, end, budget)
    if not getattr(args, "force", False):
        reused = find_reused_run(registry_path, fingerprint)
        if reused is not None:
            existing_id, validity = reused
            _logger.info(
                "[EVAL] backtest mhs reuse run_id=%s validity=%s", existing_id, validity,
            )
            return
    result_output, targets_output = _resolve_destinations(args)
    run_id = result_output.parent.name if getattr(args, "output", None) is None else uuid.uuid4().hex
    _logger.info(
        "[EVAL] backtest mhs result_output=%s targets_output=%s",
        result_output, targets_output,
    )
    try:
        run = run_mhs_process_backtest(
            start=start, end=end, data_root=args.data_root, result_output=result_output,
            targets_output=targets_output,
            tracking_error_threshold=args.rebalance_tracking_error_threshold,
            timeout_seconds=args.timeout_seconds, poll_seconds=args.poll_seconds,
            memory_budget=budget,
            registry_path=registry_path,
            run_id=run_id,
            retention_policy=retention_policy,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid backtest controls: {exc}") from exc
    if run.status != "completed":
        raise SystemExit(1)
