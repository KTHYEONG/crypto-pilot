"""Canonical three-minute inventory backtest command."""

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.backtests.contracts import RetentionPolicy
from src.common.paths import BACKTESTS_DIR
from src.mhs.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING
from src.mhs.resources import MhsMemoryBudget

if TYPE_CHECKING:
    from src.mhs.frozen_research_candidate import FrozenMhsStrategySpec
    from src.mhs.types import ExecutionSpec

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
    frozen = sub.add_parser(
        "mhs-frozen",
        help="Research-only frozen MHS Top-20/variant 3m inventory evaluation.",
        description="Research-only frozen MHS Top-20/variant 3m inventory evaluation.",
    )
    frozen.add_argument("--source-start", default=None, help="UTC source history start; date-only values are UTC.")
    frozen.add_argument("--start", default=None, help="UTC evaluation start; date-only values are UTC.")
    frozen.add_argument("--end", default=None, help="UTC exclusive evaluation end.")
    frozen.add_argument("--breadth", type=int, default=20, help="Positive universe breadth; 20 is the primary Top-20.")
    frozen.add_argument("--output", default=None, help="Fresh complete research result envelope JSON destination.")
    frozen.add_argument("--data-root", default=None, help="Existing OHLCV root override.")
    frozen.add_argument("--total-tree-pss-bytes", type=int, default=None, help="Total process-tree PSS ceiling in bytes.")
    frozen.add_argument("--replay-tree-pss-bytes", type=int, default=None, help="Replay process-tree PSS ceiling in bytes.")
    frozen.add_argument("--min-available-bytes", type=int, default=None, help="Minimum effective physical headroom in bytes.")
    frozen.set_defaults(handler=run_frozen_mhs_backtest_command)


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

def _frozen_strategy(breadth: int) -> FrozenMhsStrategySpec:
    """Select the primary Top-20 policy or a breadth-labelled research control."""
    from src.mhs.frozen_research_candidate import (
        FROZEN_MHS_TOP20_V1,
        FROZEN_MHS_TOP40_CONTROL_V1,
    )

    if breadth == 20:
        return FROZEN_MHS_TOP20_V1
    if breadth == 40:
        return FROZEN_MHS_TOP40_CONTROL_V1
    import dataclasses

    return dataclasses.replace(
        FROZEN_MHS_TOP20_V1,
        strategy_id=f"frozen_mhs_b{breadth}_control_v1",
        breadth=breadth,
    )


def _frozen_specs() -> tuple[ExecutionSpec, ExecutionSpec]:
    """Build the registered six/eighteen-basis-point immediate-taker cost pair."""
    import dataclasses

    from src.mhs.types import ExecutionSpec

    base = dataclasses.replace(ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=1.0)
    stress = dataclasses.replace(base, taker_slippage_bps=13.0)
    return base, stress


def run_frozen_mhs_backtest_command(args: argparse.Namespace) -> None:
    """Run a research-only frozen MHS Top-20/variant 3m inventory evaluation.

    This command is a distinct strategy identity from ``backtest mhs``.  It
    accepts declared breadth and dates, builds no live artifact, and persists
    only completed research evidence.

    Args:
        args: Parsed frozen-MHS source/evaluation dates, breadth, paths, and
            existing memory-budget controls.
    Returns:
        None after a fresh completed research result is persisted.
    Raises:
        SystemExit: Arguments are invalid or the research replay fails.
    """
    import os

    from src.common.errors import DataIntegrityError
    from src.mhs.frozen_research_evidence import FrozenMhsReportPeriod
    from src.mhs.frozen_research_report import persist_frozen_mhs_backtest
    from src.mhs.frozen_research_run import FrozenMhsBacktestRequest, run_frozen_mhs_backtest

    breadth = getattr(args, "breadth", 20)
    if isinstance(breadth, bool) or not isinstance(breadth, int) or breadth <= 0:
        raise SystemExit(f"breadth must be a positive integer, got {breadth!r}")
    if getattr(args, "source_start", None) is None:
        raise SystemExit("source-start is required")
    if getattr(args, "start", None) is None:
        raise SystemExit("start is required")
    if getattr(args, "end", None) is None:
        raise SystemExit("end is required")
    source_start = _utc_timestamp(args.source_start, "source-start", DISCOVERY_START)
    start = _utc_timestamp(args.start, "start", DISCOVERY_START)
    end = _utc_timestamp(args.end, "end", PROCESS_EVALUATION_CEILING)
    if not source_start < start < end:
        raise SystemExit(f"require source-start < start < end, got {source_start} {start} {end}")
    raw_output = getattr(args, "output", None)
    if raw_output is None:
        raise SystemExit("output is required")
    output = Path(raw_output)
    if output.suffix != ".json":
        raise SystemExit(f"output must be a JSON path, got {raw_output!r}")
    if os.path.lexists(output):
        raise SystemExit(f"output must be fresh: {output} already exists")
    budget = _resolve_budget(args)
    strategy = _frozen_strategy(breadth)
    base_spec, stress_spec = _frozen_specs()
    try:
        report_periods = (
            FrozenMhsReportPeriod(
                label="evaluation",
                start=start.normalize(),
                end=(end - pd.Timedelta(days=1)).normalize(),
            ),
        )
        request = FrozenMhsBacktestRequest(
            source_start=source_start, evaluation_start=start, evaluation_end=end,
            strategy=strategy, initial_equity=100000.0,
            base_spec=base_spec, stress_spec=stress_spec,
            report_periods=report_periods,
            data_root=Path(args.data_root) if getattr(args, "data_root", None) else None,
            memory_budget=budget,
        )
    except (DataIntegrityError, ValueError) as exc:
        raise SystemExit(f"invalid frozen backtest request: {exc}") from exc
    try:
        run = run_frozen_mhs_backtest(request)
        persist_frozen_mhs_backtest(run, output)
        _logger.info(
            "[EVAL] backtest mhs-frozen source_gap_excluded=%s",
            list(getattr(run, "source_gap_excluded_symbols", ())),
        )
    except (DataIntegrityError, ValueError, OSError) as exc:
        raise SystemExit(f"frozen backtest failed: {exc}") from exc
    status = "primary" if breadth == 20 else "research control"
    _logger.info("[EVAL] backtest mhs-frozen strategy=%s status=%s", strategy.strategy_id, status)
    print(str(output))  # noqa: T201 -- frozen command prints only the finalized result path
