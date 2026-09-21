"""Canonical three-minute inventory backtest command."""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.backtests.contracts import RetentionPolicy
from src.common.paths import BACKTESTS_DIR, FROZEN_BACKTESTS_DIR, VENUE_RULES_DIR
from src.mhs.params import (
    ACCOUNT_DEFAULT_CAPITAL_USDT,
    ACCOUNT_EXPOSURE_MAX,
    ACCOUNT_EXPOSURE_STEP,
    ACCOUNT_IMPACT_Y,
    ACCOUNT_INITIAL_MARGIN_CAP,
    ACCOUNT_MARGIN_RESERVE,
    ACCOUNT_MEAN_HAIRCUT,
    ACCOUNT_SHOCK_PER_UNIT,
    ACCOUNT_TAKER_FEE_BPS,
    ACCOUNT_UNIT_DAILY_MEAN,
    ACCOUNT_UNIT_DAILY_SIGMA,
    DEFAULT_DETAIL_RETENTION_MAX_RUNS,
    DISCOVERY_START,
    FROZEN_GROWTH_NAME_CLIP,
    PROCESS_EVALUATION_CEILING,
)
from src.mhs.resources import MhsMemoryBudget

if TYPE_CHECKING:
    from src.mhs.frozen_research_candidate import FrozenMhsStrategySpec
    from src.mhs.frozen_research_evidence import FrozenExecutionBound
    from src.mhs.frozen_research_run import FrozenMhsBacktestRequest
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
    frozen.add_argument(
        "--variant", choices=("primary", "growth"), default="primary",
        help="Target policy: primary = unlevered consensus book; growth = per-name clip + registered exposure multiplier (Top-20 only).",
    )
    frozen.add_argument("--output", default=None, help="Fresh complete research result envelope JSON destination; omitted creates a unique frozen run directory under data/backtests/frozen/runs.")
    frozen.add_argument("--data-root", default=None, help="Existing OHLCV root override.")
    frozen.add_argument("--total-tree-pss-bytes", type=int, default=None, help="Total process-tree PSS ceiling in bytes.")
    frozen.add_argument("--replay-tree-pss-bytes", type=int, default=None, help="Replay process-tree PSS ceiling in bytes.")
    frozen.add_argument("--min-available-bytes", type=int, default=None, help="Minimum effective physical headroom in bytes.")
    frozen.add_argument(
        "--execution", choices=("taker", "maker"), default="taker",
        help="taker = immediate crossing; maker = resting limit for the passive timeout, remainder crosses as taker (research variant: candle trade-through fill model).",
    )
    frozen.set_defaults(handler=run_frozen_mhs_backtest_command)
    exposure = sub.add_parser(
        "mhs-frozen-exposure",
        help="Re-derive the growth exposure rung from a finished frozen run.",
        description="Re-derive the growth exposure rung from one finished frozen run's ledger evidence.",
    )
    exposure.add_argument("--run-dir", default=None, help="Existing frozen run directory holding result.json and daily.parquet.")
    exposure.add_argument("--data-root", default=None, help="Existing OHLCV root override.")
    exposure.add_argument("--total-tree-pss-bytes", type=int, default=None, help="Total process-tree PSS ceiling in bytes.")
    exposure.add_argument("--replay-tree-pss-bytes", type=int, default=None, help="Replay process-tree PSS ceiling in bytes.")
    exposure.add_argument("--min-available-bytes", type=int, default=None, help="Minimum effective physical headroom in bytes.")
    exposure.set_defaults(handler=run_frozen_exposure_command)
    account = sub.add_parser(
        "mhs-frozen-account",
        help="Replay the frozen growth book as one real account under venue rules.",
        description="Replay the frozen growth book as one real account under venue rules.",
    )
    account.add_argument("--source-start", default=None, help="UTC source history start; date-only values are UTC.")
    account.add_argument("--start", default=None, help="UTC evaluation start; date-only values are UTC.")
    account.add_argument("--end", default=None, help="UTC exclusive evaluation end.")
    account.add_argument("--capital", type=float, default=None, help="Account start capital in USDT.")
    account.add_argument("--policy", choices=("growth", "fixed"), default="growth", help="Daily exposure rule.")
    account.add_argument("--fixed-exposure", type=float, default=None, help="Exposure for --policy fixed (required).")
    account.add_argument("--impact-y", type=float, default=None, help="Square-root impact coefficient.")
    account.add_argument("--venue-rules", default=None, help="Venue snapshot path; omitted uses the latest collected snapshot.")
    account.add_argument("--no-order-filters", action="store_true", default=False, help="Trade continuous quantities without step/min-notional filters.")
    account.add_argument("--data-root", default=None, help="Existing OHLCV root override.")
    account.add_argument("--total-tree-pss-bytes", type=int, default=None, help="Total process-tree PSS ceiling in bytes.")
    account.add_argument("--replay-tree-pss-bytes", type=int, default=None, help="Replay process-tree PSS ceiling in bytes.")
    account.add_argument("--min-available-bytes", type=int, default=None, help="Minimum effective physical headroom in bytes.")
    account.set_defaults(handler=run_frozen_account_command)


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
    if result_output.is_file():
        payload = json.loads(result_output.read_text(encoding="utf-8"))
        base = payload.get("financial", {}).get("base", {})
        _append_backtest_index(
            index_path=BACKTESTS_DIR / "index.jsonl",
            kind="mhs", run_dir=result_output.parent, created_at=pd.Timestamp.now(tz="UTC"),
            evaluation_start=start, evaluation_end=end, strategy_id="process_inventory_3m",
            base_cagr=base.get("cagr"), base_max_drawdown=base.get("max_drawdown"),
        )

def _append_backtest_index(
    *, index_path: Path, kind: str, run_dir: Path, created_at: pd.Timestamp,
    evaluation_start: pd.Timestamp, evaluation_end: pd.Timestamp,
    strategy_id: str, base_cagr: float | None, base_max_drawdown: float | None,
    execution: str | None = None,
) -> None:
    """Append one headline row to the single cross-pipeline backtest catalog.

    This is the sole file a human or analysis script needs to read to see
    every backtest ever run (canonical or frozen-research), with headline
    metrics inline; `registry.sqlite3`/`evidence/` stay internal plumbing for
    fingerprint reuse and content-addressed detail dedup.

    Args:
        index_path: Destination `index.jsonl`, derived by the caller from
            whichever run-root it already owns (never hardcoded here).
        kind: Producing pipeline identity, `"mhs"` or `"mhs_frozen"`.
        run_dir: Directory holding that run's own result envelope.
        created_at: UTC timestamp of index-write time.
        evaluation_start: Registered evaluation start.
        evaluation_end: Registered evaluation end.
        strategy_id: Strategy identity string for the run.
        base_cagr: Headline base-cost CAGR, or `None` if not cheaply available.
        base_max_drawdown: Headline base-cost max drawdown, or `None` likewise.
    Returns:
        None after appending one JSON line to `index_path`.
    """
    try:
        rel_run_dir = str(run_dir.relative_to(index_path.parent))
    except ValueError:
        rel_run_dir = str(run_dir)
    record = {
        "kind": kind,
        "run_dir": rel_run_dir,
        "created_at": created_at.isoformat(),
        "evaluation_start": evaluation_start.isoformat(),
        "evaluation_end": evaluation_end.isoformat(),
        "strategy_id": strategy_id,
        "base_cagr": base_cagr,
        "base_max_drawdown": base_max_drawdown,
    }
    if execution is not None:
        record["execution"] = execution
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _frozen_strategy(breadth: int, variant: str = "primary") -> FrozenMhsStrategySpec:
    """Select the frozen target policy for one research run.

    ``primary`` keeps the unlevered consensus book at the requested breadth (20 is the primary
    Top-20, other breadths are labelled controls). ``growth`` is registered only for breadth 20,
    because its exposure rung was derived from that book's own drawdown distribution and does not
    transfer to other universes.

    Raises:
        SystemExit: ``growth`` is requested with a breadth other than 20, or ``variant`` is unknown.
    """
    from src.mhs.frozen_research_candidate import (
        FROZEN_MHS_TOP20_GROWTH_V2,
        FROZEN_MHS_TOP20_V2,
        FROZEN_MHS_TOP40_CONTROL_V2,
    )

    if variant == "growth":
        if breadth != 20:
            raise SystemExit(f"growth variant is registered only for breadth 20, got {breadth!r}")
        return FROZEN_MHS_TOP20_GROWTH_V2
    if variant != "primary":
        raise SystemExit(f"unknown frozen variant {variant!r}")
    if breadth == 20:
        return FROZEN_MHS_TOP20_V2
    if breadth == 40:
        return FROZEN_MHS_TOP40_CONTROL_V2
    import dataclasses

    return dataclasses.replace(
        FROZEN_MHS_TOP20_V2,
        strategy_id=f"frozen_mhs_b{breadth}_control_v2",
        breadth=breadth,
    )


def _frozen_specs() -> tuple[ExecutionSpec, ExecutionSpec]:
    """Build the registered six/eighteen-basis-point cost pair with the submit-bar anchor, so every order is sized and priced from the last mark published before it is sent."""
    import dataclasses

    from src.mhs.types import ExecutionSpec

    base = dataclasses.replace(ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=1.0, decision_anchor="submit_bar")
    stress = dataclasses.replace(base, taker_slippage_bps=13.0)
    return base, stress


def _frozen_run_name(start: pd.Timestamp, end: pd.Timestamp, breadth: int, created_at: pd.Timestamp, variant: str = "primary", execution: str = "taker") -> str:
    """Human-readable frozen run directory name: dates and breadth are legible without opening any file."""
    stem = f"{start:%Y%m%d}_{end:%Y%m%d}_top{breadth}"
    if variant == "growth":
        stem = f"{stem}_growth"
    if execution == "maker":
        stem = f"{stem}_maker"
    return f"{stem}_{created_at:%Y%m%dT%H%M%S}Z"


def _resolve_frozen_destination(args: argparse.Namespace, *, start: pd.Timestamp, end: pd.Timestamp, breadth: int, variant: str = "primary", execution: str = "taker") -> Path:
    """Resolve the frozen research result destination, defaulting to a fresh, human-readable run directory.

    Args:
        args: Parsed frozen-MHS backtest arguments.
        start: Registered evaluation start, used to name the default run directory.
        end: Registered evaluation end, used to name the default run directory.
        breadth: Declared universe breadth, used to name the default run directory.
    Returns:
        A fresh `result.json` destination under the frozen runs directory.
    Raises:
        SystemExit: An explicit destination is invalid or already occupied.
    """
    import os

    raw_output = getattr(args, "output", None)
    if raw_output is None:
        name = _frozen_run_name(start, end, breadth, pd.Timestamp.now(tz="UTC"), variant, execution)
        run_dir = FROZEN_BACKTESTS_DIR / name
        suffix = 1
        while os.path.lexists(run_dir):
            suffix += 1
            run_dir = FROZEN_BACKTESTS_DIR / f"{name}-{suffix}"
        run_dir.mkdir(parents=True)
        return run_dir / "result.json"
    output = Path(raw_output)
    if output.suffix != ".json":
        raise SystemExit(f"output must be a JSON path, got {raw_output!r}")
    if os.path.lexists(output):
        raise SystemExit(f"output must be fresh: {output} already exists")
    return output


def _write_frozen_manifest(output: Path, *, request: FrozenMhsBacktestRequest, breadth: int) -> None:
    """Persist the frozen research identity manifest beside the result envelope, and append it to the run index.

    Args:
        output: Finalized frozen result JSON destination.
        request: Executed frozen backtest request carrying identity timestamps.
        breadth: Declared universe breadth for the research variant.
    Returns:
        None after `manifest.json` is written beside `output` and appended to `index.jsonl`.
    """
    manifest = {
        "run_dir": output.parent.name,
        "source_start": request.source_start.isoformat(),
        "evaluation_start": request.evaluation_start.isoformat(),
        "evaluation_end": request.evaluation_end.isoformat(),
        "breadth": breadth,
        "strategy_id": request.strategy.strategy_id,
        "execution": "maker" if request.execution_bound == "OHLCV_STRICT_PROXY" else "taker",
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    (output.parent / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if output.parent.parent == FROZEN_BACKTESTS_DIR:
        payload = json.loads(output.read_text(encoding="utf-8"))
        evaluation = payload.get("report_periods", {}).get("evaluation", {})
        _append_backtest_index(
            index_path=FROZEN_BACKTESTS_DIR.parent.parent / "index.jsonl",
            kind="mhs_frozen", run_dir=output.parent, created_at=pd.Timestamp.now(tz="UTC"),
            evaluation_start=request.evaluation_start, evaluation_end=request.evaluation_end,
            strategy_id=request.strategy.strategy_id,
            base_cagr=evaluation.get("base_cagr"), base_max_drawdown=evaluation.get("base_max_drawdown"),
            execution="maker" if request.execution_bound == "OHLCV_STRICT_PROXY" else "taker",
        )


def _prune_frozen_runs(keep: int) -> None:
    """Reclaim result detail for finalized frozen runs beyond the most recent `keep`; the index line is never removed.

    Args:
        keep: Positive count of most-recently-created run directories to retain in full.
    Returns:
        None after removing older run directories' files from disk.
    """
    import shutil

    if not FROZEN_BACKTESTS_DIR.is_dir():
        return
    run_dirs = sorted(
        (d for d in FROZEN_BACKTESTS_DIR.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for stale in run_dirs[keep:]:
        shutil.rmtree(stale, ignore_errors=True)


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
    variant = getattr(args, "variant", "primary")
    strategy = _frozen_strategy(breadth, variant)
    execution = getattr(args, "execution", "taker")
    if execution not in ("taker", "maker"):
        raise SystemExit(f"execution must be 'taker' or 'maker', got {execution!r}")
    execution_bound: FrozenExecutionBound = "OHLCV_STRICT_PROXY" if execution == "maker" else "OHLCV_IMMEDIATE_TAKER"
    output = _resolve_frozen_destination(args, start=start, end=end, breadth=breadth, variant=variant, execution=execution)
    budget = _resolve_budget(args)
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
            execution_bound=execution_bound,
        )
    except (DataIntegrityError, ValueError) as exc:
        raise SystemExit(f"invalid frozen backtest request: {exc}") from exc
    try:
        run = run_frozen_mhs_backtest(request)
        persist_frozen_mhs_backtest(run, output)
        _write_frozen_manifest(output, request=request, breadth=breadth)
        _logger.info(
            "[EVAL] backtest mhs-frozen source_gap_excluded=%s",
            list(getattr(run, "source_gap_excluded_symbols", ())),
        )
    except (DataIntegrityError, ValueError, OSError) as exc:
        raise SystemExit(f"frozen backtest failed: {exc}") from exc
    if output.parent.parent == FROZEN_BACKTESTS_DIR and DEFAULT_DETAIL_RETENTION_MAX_RUNS is not None:
        _prune_frozen_runs(keep=DEFAULT_DETAIL_RETENTION_MAX_RUNS)
    status = "primary" if breadth == 20 else "research control"
    _logger.info(
        "[EVAL] backtest mhs-frozen strategy=%s status=%s variant=%s exposure=%s name_clip=%s",
        strategy.strategy_id, status, variant, strategy.exposure_multiplier, strategy.name_clip,
    )
    print(str(output))  # noqa: T201 -- frozen command prints only the finalized result path


def run_frozen_exposure_command(args: argparse.Namespace) -> None:
    """Re-derive the growth exposure rung from one finished frozen run's ledger evidence.

    Reads the run's ``result.json`` and ``daily.parquet``, unlevers base daily returns and the
    largest name weight by the run's ``exposure_multiplier``, samples single-name gaps from
    symbols the ledger has permanently excluded from trading (registry reason ``DELISTED``,
    e.g. LUNAUSDT) over the run's evaluation window, and solves the stressed log-growth curve
    with the registered parameters. Any other symbol's single-day moves, however large, already
    occurred inside the ledger's own realized returns and must not be re-added as a separate
    gap stress; only a structurally excluded symbol's price path reveals risk the ledger could
    never have realized on its own.

    Args:
        args: Parsed run directory, data root, and memory-budget controls.
    Returns:
        None after writing ``exposure.json`` beside the run's result and printing its path.
    Raises:
        SystemExit: Missing or invalid run artifacts, an existing ``exposure.json``, or a solver
            rejection.
    """
    import os

    import numpy as np

    from src.common.errors import DataIntegrityError
    from src.common.paths import FUTURES_DATA_DIR
    from src.mhs.frozen_research_universe import build_frozen_pit_roster
    from src.mhs.growth_exposure import (
        GapSample,
        roster_gap_sample,
        solve_log_growth_exposure,
        structurally_excluded_symbols,
    )
    from src.mhs.panel import load_base_panel
    from src.mhs.params import (
        COMMITTEE_GROWTH_HORIZON_YEARS,
        COMMITTEE_GROWTH_N_PATHS,
        FROZEN_EXPOSURE_GAP_THRESHOLD,
        FROZEN_EXPOSURE_GRID,
        FROZEN_EXPOSURE_MEAN_HAIRCUT,
        FROZEN_EXPOSURE_PLATEAU_TOLERANCE,
        FROZEN_EXPOSURE_SEED,
        NULL_BOOTSTRAP_MEAN_BLOCK_DAYS,
    )
    from src.mhs.resources import (
        _current_tree_swap_bytes,
        assert_mhs_stage_allocation,
        resolve_mhs_memory_budget,
    )

    raw_run_dir = getattr(args, "run_dir", None)
    if raw_run_dir is None:
        raise SystemExit("run-dir is required")
    run_dir = Path(raw_run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"run-dir must be an existing frozen run directory, got {raw_run_dir!r}")
    exposure_path = run_dir / "exposure.json"
    if os.path.lexists(exposure_path):
        raise SystemExit(f"exposure output must be fresh: {exposure_path} already exists")
    try:
        payload = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        daily = pd.read_parquet(run_dir / "daily.parquet")
        exposure_multiplier = float(payload["exposure_multiplier"])
        strategy_id = payload["strategy_id"]
        execution_bound = payload["execution_bound"]
        breadth = payload["breadth"]
        source_start = pd.Timestamp(payload["source_start"]).tz_convert("UTC")
        evaluation_start = pd.Timestamp(payload["evaluation_start"]).tz_convert("UTC")
        evaluation_end = pd.Timestamp(payload["evaluation_end"]).tz_convert("UTC")
        unit_returns = daily["base_return"] / exposure_multiplier
        unit_max_weight = float(daily["max_name_weight"].mean() / exposure_multiplier)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"invalid frozen run artifacts: {exc}") from exc
    budget = _resolve_budget(args)
    data_root = getattr(args, "data_root", None)
    root = str(Path(data_root)) if data_root is not None else str(FUTURES_DATA_DIR / "ohlcv")
    try:
        resolved = resolve_mhs_memory_budget(budget)
        initial_swap_bytes = _current_tree_swap_bytes()

        def _admit_panel(estimated_bytes: int) -> None:
            assert_mhs_stage_allocation(
                stage="frozen_source_panel", estimated_bytes=int(estimated_bytes),
                budget=resolved, replay=False, initial_swap_bytes=initial_swap_bytes,
            )

        assert_mhs_stage_allocation(
            stage="frozen_source_panel", estimated_bytes=0,
            budget=resolved, replay=False, initial_swap_bytes=initial_swap_bytes,
        )
        panel = load_base_panel(
            root, "1h", ("close", "quote_vol"), source_start, evaluation_end, partition="all",
            selection_mode="causal_history", allocation_admission=_admit_panel,
        )
        daily_close = panel["close"].resample("1D").last().astype("float64")
        daily_quote_volume = panel["quote_vol"].resample("1D").sum(min_count=1).astype("float64")
        census = tuple(panel["close"].columns)
        # 로스터는 원천 시작부터 만들어 30일 유동성·90일 시즌링 워밍업을 보존한 뒤 평가 구간으로 자른다.
        roster = build_frozen_pit_roster(
            daily_close, daily_quote_volume, census, breadth=breadth, blocked_decisions=None,
        )
        in_window = (daily_close.index >= evaluation_start) & (daily_close.index < evaluation_end)
        # 거래 제외(DELISTED) 종목만 갭 표본으로 쓴다: 그 외 종목의 급락은 원장이 이미 실제로
        # 겪어 실현수익률에 반영돼 있으므로, 여기서 다시 얹으면 같은 위험을 이중으로 계산하게 된다.
        gap_symbols = [s for s in census if s in structurally_excluded_symbols()]
        # 등록부에 배제 종목이 없으면 갭 표본은 비어 있고(solve_log_growth_exposure가 지원하는
        # 상태), roster_gap_sample을 0열 프레임으로 호출해 실패시키지 않는다.
        gaps = (
            roster_gap_sample(
                daily_close.loc[in_window, gap_symbols], roster.loc[in_window, gap_symbols],
                threshold=FROZEN_EXPOSURE_GAP_THRESHOLD,
            )
            if gap_symbols
            else GapSample(magnitudes=np.empty(0, dtype="float64"), events_per_year=0.0)
        )
        solution = solve_log_growth_exposure(
            unit_returns,
            max_name_weight=unit_max_weight,
            gaps=gaps,
            mean_haircut=FROZEN_EXPOSURE_MEAN_HAIRCUT,
            grid=FROZEN_EXPOSURE_GRID,
            plateau_tolerance=FROZEN_EXPOSURE_PLATEAU_TOLERANCE,
            n_paths=COMMITTEE_GROWTH_N_PATHS,
            horizon_years=COMMITTEE_GROWTH_HORIZON_YEARS,
            mean_block_days=NULL_BOOTSTRAP_MEAN_BLOCK_DAYS,
            seed=FROZEN_EXPOSURE_SEED,
        )
        exposure = {
            "run_dir": run_dir.name,
            "strategy_id": strategy_id,
            "execution_bound": execution_bound,
            "exposure_multiplier": exposure_multiplier,
            "grid": list(solution.grid),
            "growth": list(solution.growth),
            "ruin_probability": list(solution.ruin_probability),
            "argmax": solution.argmax,
            "chosen": solution.chosen,
            "gap_events_per_year": solution.gap_events_per_year,
            "gap_sample_size": solution.gap_sample_size,
            "gap_symbols": sorted(gap_symbols),
            "mean_haircut": FROZEN_EXPOSURE_MEAN_HAIRCUT,
            "plateau_tolerance": FROZEN_EXPOSURE_PLATEAU_TOLERANCE,
            "seed": FROZEN_EXPOSURE_SEED,
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "unlever_assumption": "unit returns and max name weight are linear rescalings of the levered ledger; residual drift/cost nonlinearity is accepted",
        }
        tmp_path = run_dir / "exposure.json.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(exposure, handle, sort_keys=True)
        os.replace(tmp_path, exposure_path)
    except (DataIntegrityError, ValueError, OSError) as exc:
        raise SystemExit(f"frozen exposure failed: {exc}") from exc
    _logger.info(
        "[EVAL] frozen exposure run=%s chosen=%.2f argmax=%.2f gaps_per_year=%.2f",
        run_dir.name, solution.chosen, solution.argmax, solution.gap_events_per_year,
    )
    print(str(exposure_path))  # noqa: T201 -- exposure command prints only the finalized exposure path


def _account_headlines(equity: pd.Series, capital: float) -> tuple[float, float, float]:
    """CAGR, daily max drawdown (negative or zero), and final equity for one account path."""
    import numpy as np

    values = equity.to_numpy(dtype="float64")
    final = float(values[-1])
    years = len(values) / 365.0
    cagr = float(final / capital) ** (1.0 / years) - 1.0 if final > 0 else -1.0
    running = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(running > 0, values / running, 1.0)
    return cagr, float((relative - 1.0).min()), final


def _latest_primary_reference(index_path: Path) -> dict[str, Any] | None:
    """Latest mhs_frozen primary row of the run catalog, or None when absent."""
    if not index_path.is_file():
        return None
    reference = None
    for line in index_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("kind") == "mhs_frozen" and record.get("strategy_id") == "frozen_mhs_top20_v2":
            reference = record
    return reference


def _resolve_account_destination(*, start: pd.Timestamp, end: pd.Timestamp, policy: str, capital: float) -> Path:
    """Resolve a fresh account run directory named by window, policy, and capital."""
    import os

    name = f"{start:%Y%m%d}_{end:%Y%m%d}_top20_account_{policy}_{capital:.0f}_{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%S}Z"
    run_dir = FROZEN_BACKTESTS_DIR / name
    suffix = 1
    while os.path.lexists(run_dir):
        suffix += 1
        run_dir = FROZEN_BACKTESTS_DIR / f"{name}-{suffix}"
    run_dir.mkdir(parents=True)
    return run_dir


def run_frozen_account_command(args: argparse.Namespace) -> None:
    """Replay the frozen growth book as one real account and persist account-scale evidence.

    Builds the unlevered (exposure 1.0) clip-0.05 candidate, assembles account inputs, replays
    with the requested policy and venue snapshot, and writes ``account.json`` + ``account_daily.parquet``
    in a fresh run directory ``<start>_<end>_top20_account_<policy>_<capital>_<ts>Z``.

    Raises:
        SystemExit: Invalid arguments, missing venue snapshot, or a data-integrity failure.
    """
    import dataclasses

    from src.common.errors import DataIntegrityError
    from src.market_data.binance.venue_rules import latest_venue_rule_snapshot, load_venue_rule_snapshot
    from src.mhs.account_ledger import replay_account
    from src.mhs.account_policy import ExposurePolicy
    from src.mhs.account_sources import assemble_account_inputs
    from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP20_V2
    from src.mhs.frozen_research_evidence import FrozenMhsReportPeriod
    from src.mhs.frozen_research_run import FrozenMhsBacktestRequest, build_frozen_request_candidate

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
    policy_name = getattr(args, "policy", "growth")
    if policy_name not in ("growth", "fixed"):
        raise SystemExit(f"policy must be 'growth' or 'fixed', got {policy_name!r}")
    raw_fixed = getattr(args, "fixed_exposure", None)
    if policy_name == "fixed" and raw_fixed is None:
        raise SystemExit("--fixed-exposure is required with --policy fixed")
    try:
        fixed_exposure = None if raw_fixed is None else float(raw_fixed)
        capital = ACCOUNT_DEFAULT_CAPITAL_USDT if getattr(args, "capital", None) is None else float(args.capital)
        impact_y = ACCOUNT_IMPACT_Y if getattr(args, "impact_y", None) is None else float(args.impact_y)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"invalid account controls: {exc}") from exc
    if fixed_exposure is not None and not 0 < fixed_exposure < float("inf"):
        raise SystemExit(f"--fixed-exposure must be a positive finite exposure, got {raw_fixed!r}")
    if not 0 < capital < float("inf"):
        raise SystemExit(f"--capital must be a positive finite capital, got {getattr(args, 'capital', None)!r}")
    raw_venue = getattr(args, "venue_rules", None)
    try:
        venue_path = Path(raw_venue) if raw_venue is not None else latest_venue_rule_snapshot(VENUE_RULES_DIR)
        rules = load_venue_rule_snapshot(venue_path)
    except (DataIntegrityError, FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        raise SystemExit(f"missing venue snapshot ({exc}); run data collect venue-rules first") from exc
    strategy = dataclasses.replace(
        FROZEN_MHS_TOP20_V2, exposure_multiplier=1.0, name_clip=FROZEN_GROWTH_NAME_CLIP,
    )
    budget = _resolve_budget(args)
    base_spec, stress_spec = _frozen_specs()
    request = FrozenMhsBacktestRequest(
        source_start=source_start, evaluation_start=start, evaluation_end=end,
        strategy=strategy, initial_equity=capital,
        base_spec=base_spec, stress_spec=stress_spec,
        report_periods=(
            FrozenMhsReportPeriod(
                label="evaluation",
                start=start.normalize(),
                end=(end - pd.Timedelta(days=1)).normalize(),
            ),
        ),
        data_root=Path(args.data_root) if getattr(args, "data_root", None) else None,
        memory_budget=budget,
        execution_bound="OHLCV_IMMEDIATE_TAKER",
    )
    try:
        candidate, context = build_frozen_request_candidate(request)
        unit_weights, marks, funding_cum, adv, daily_sigma = assemble_account_inputs(candidate, context)
    except (DataIntegrityError, ValueError, OSError) as exc:
        raise SystemExit(f"frozen account failed: {exc}") from exc
    policy = ExposurePolicy(
        kind="growth" if policy_name == "growth" else "fixed",
        exposure_max=float(fixed_exposure or ACCOUNT_EXPOSURE_MAX),
        exposure_step=ACCOUNT_EXPOSURE_STEP,
        unit_daily_mean=ACCOUNT_UNIT_DAILY_MEAN,
        unit_daily_sigma=ACCOUNT_UNIT_DAILY_SIGMA,
        mean_haircut=ACCOUNT_MEAN_HAIRCUT,
        shock_per_unit=ACCOUNT_SHOCK_PER_UNIT,
        margin_reserve=ACCOUNT_MARGIN_RESERVE,
        initial_margin_cap=ACCOUNT_INITIAL_MARGIN_CAP,
        impact_y=impact_y,
    )
    apply_filters = not getattr(args, "no_order_filters", False)
    try:
        result = replay_account(
            unit_weights, marks, funding_cum, adv, daily_sigma, rules, policy,
            capital=capital, taker_fee_bps=ACCOUNT_TAKER_FEE_BPS, apply_order_filters=apply_filters,
        )
    except (DataIntegrityError, ValueError) as exc:
        raise SystemExit(f"frozen account failed: {exc}") from exc
    index_path = FROZEN_BACKTESTS_DIR.parent.parent / "index.jsonl"
    try:
        recon_policy = dataclasses.replace(policy, kind="fixed", exposure_max=1.0, impact_y=0.0)
        recon = replay_account(
            unit_weights, marks, funding_cum, adv, daily_sigma, rules, recon_policy,
            capital=1e5, taker_fee_bps=ACCOUNT_TAKER_FEE_BPS, apply_order_filters=False,
        )
        recon_cagr, recon_mdd, _ = _account_headlines(recon.daily_equity, 1e5)
        reference = _latest_primary_reference(index_path)
        reconciliation: dict[str, Any] = {
            "status": "ok",
            "fixed_exposure": 1.0,
            "capital": 1e5,
            "order_filters": False,
            "impact_y": 0.0,
            "cagr": recon_cagr,
            "mdd": recon_mdd,
            "reference_canonical": None if reference is None else {
                "strategy_id": reference.get("strategy_id"),
                "run_dir": reference.get("run_dir"),
                "evaluation_start": reference.get("evaluation_start"),
                "evaluation_end": reference.get("evaluation_end"),
                "base_cagr": reference.get("base_cagr"),
                "base_max_drawdown": reference.get("base_max_drawdown"),
            },
            "cagr_gap": None if reference is None or reference.get("base_cagr") is None else recon_cagr - float(reference["base_cagr"]),
            "mdd_gap": None if reference is None or reference.get("base_max_drawdown") is None else recon_mdd - float(reference["base_max_drawdown"]),
        }
    except Exception as exc:  # noqa: BLE001 -- reconciliation is disclosed-only and never fails the run
        reconciliation = {"status": "failed", "error": str(exc)}
    cagr, mdd, final_equity = _account_headlines(result.daily_equity, capital)
    exposures = result.daily_exposure.to_numpy(dtype="float64")
    run_dir = _resolve_account_destination(start=start, end=end, policy=policy_name, capital=capital)
    payload = {
        "strategy_id": strategy.strategy_id,
        "capital": capital,
        "policy": {
            "kind": policy.kind,
            "exposure_max": policy.exposure_max,
            "exposure_step": policy.exposure_step,
            "unit_daily_mean": policy.unit_daily_mean,
            "unit_daily_sigma": policy.unit_daily_sigma,
            "mean_haircut": policy.mean_haircut,
            "shock_per_unit": policy.shock_per_unit,
            "margin_reserve": policy.margin_reserve,
            "initial_margin_cap": policy.initial_margin_cap,
            "impact_y": policy.impact_y,
        },
        "venue_captured_at": rules.captured_at.isoformat(),
        "venue_path": str(venue_path),
        "evaluation_start": start.isoformat(),
        "evaluation_end": end.isoformat(),
        "cagr": cagr,
        "mdd": mdd,
        "final_equity": final_equity,
        "liquidated_at": None if result.liquidated_at is None else result.liquidated_at.isoformat(),
        "mean_exposure": float(exposures.mean()),
        "min_exposure": float(exposures.min()),
        "last_exposure": float(exposures[-1]),
        "skipped_orders": result.skipped_orders,
        "untraded_fraction": result.untraded_fraction,
        "initial_margin_breaches": result.initial_margin_breaches,
        "fee_paid": result.fee_paid,
        "impact_paid": result.impact_paid,
        "funding_paid": result.funding_paid,
        "fallback_ladder_symbols": list(result.fallback_ladder_symbols),
        "missing_filter_symbols": list(result.missing_filter_symbols),
        "in_sample_moments": True,
        "venue_rules_applied_retroactively": True,
        "reconciliation": reconciliation,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    (run_dir / "account.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(
        {"equity": result.daily_equity.to_numpy(dtype="float64"), "exposure": exposures},
        index=result.daily_equity.index,
    ).to_parquet(run_dir / "account_daily.parquet")
    if run_dir.parent == FROZEN_BACKTESTS_DIR:
        _append_backtest_index(
            index_path=index_path,
            kind="mhs_frozen_account", run_dir=run_dir, created_at=pd.Timestamp.now(tz="UTC"),
            evaluation_start=start, evaluation_end=end,
            strategy_id=strategy.strategy_id,
            base_cagr=cagr, base_max_drawdown=mdd,
        )
    _logger.info(
        "[EVAL] mhs-frozen-account capital=%.0f policy=%s cagr=%.4f mdd=%.4f liquidated=%s",
        capital, policy_name, cagr, mdd, result.liquidated_at,
    )
