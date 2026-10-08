"""Canonical three-minute inventory backtest command."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import pandas as pd

from src.application.strategy_account import strategy_execution_specs
from src.backtests.catalog import append_backtest_index
from src.common.paths import STRATEGY_BACKTESTS_DIR, VENUE_RULES_DIR
from src.core.params import (
    ACCOUNT_DEFAULT_CAPITAL_USDT,
    ACCOUNT_IMPACT_Y,
    DEFAULT_DETAIL_RETENTION_MAX_RUNS,
    DISCOVERY_START,
    PROCESS_EVALUATION_CEILING,
)
from src.core.resources import MhsMemoryBudget

if TYPE_CHECKING:
    from src.engine.backtest_evidence import StrategyExecutionBound
    from src.engine.strategy_backtest import StrategyBacktestRequest
    from src.strategy.targets import StrategySpec

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


def add_backtest_commands(backtest_parser: argparse.ArgumentParser) -> None:
    """Register the deployable strategy backtest leaves.

    Args:
        backtest_parser: Root-owned parser for the backtest command group.
    Returns:
        None; registers the strategy, exposure and account leaves. The
        exploratory process backtest lives under ``lab process-backtest``.
    """
    sub = backtest_parser.add_subparsers(dest="command", required=True)
    strategy = sub.add_parser(
        "strategy",
        help="Deployed strategy Top-20/variant 3m inventory evaluation.",
        description="Deployed strategy Top-20/variant 3m inventory evaluation.",
    )
    strategy.add_argument("--source-start", default=None, help="UTC source history start; date-only values are UTC.")
    strategy.add_argument("--start", default=None, help="UTC evaluation start; date-only values are UTC.")
    strategy.add_argument("--end", default=None, help="UTC exclusive evaluation end.")
    strategy.add_argument(
        "--strategy", choices=("flow_mom_top20", "flow_mom_top40_control", "flow_mom_top20_growth"),
        default=None, help="Canonical registered strategy id; determines breadth and target policy.",
    )
    strategy.add_argument("--breadth", type=int, default=20, help="Positive universe breadth; 20 is the primary Top-20.")
    strategy.add_argument(
        "--variant", choices=("primary", "growth", "account_unit"), default="primary",
        help="Target policy: primary = unlevered consensus book; growth = per-name clip + registered exposure multiplier (Top-20 only); account_unit = unlevered clip book the account ledger replays (Top-20 only; reconciliation reference).",
    )
    strategy.add_argument("--output", default=None, help="Fresh complete result envelope JSON destination; omitted creates a unique strategy run directory under data/backtests/strategy/runs.")
    strategy.add_argument("--data-root", default=None, help="Existing OHLCV root override.")
    strategy.add_argument("--total-tree-pss-bytes", type=int, default=None, help="Total process-tree PSS ceiling in bytes.")
    strategy.add_argument("--replay-tree-pss-bytes", type=int, default=None, help="Replay process-tree PSS ceiling in bytes.")
    strategy.add_argument("--min-available-bytes", type=int, default=None, help="Minimum effective physical headroom in bytes.")
    strategy.add_argument(
        "--execution", choices=("taker", "maker"), default="taker",
        help="taker = immediate crossing; maker = resting limit for the passive timeout, remainder crosses as taker (strategy variant: candle trade-through fill model).",
    )
    strategy.set_defaults(handler=run_strategy_backtest_command)
    exposure = sub.add_parser(
        "exposure",
        help="Re-derive the growth exposure rung from a finished strategy run.",
        description="Re-derive the growth exposure rung from one finished strategy run's ledger evidence.",
    )
    exposure.add_argument("--run-dir", default=None, help="Existing strategy run directory holding result.json and daily.parquet.")
    exposure.add_argument("--data-root", default=None, help="Existing OHLCV root override.")
    exposure.add_argument("--total-tree-pss-bytes", type=int, default=None, help="Total process-tree PSS ceiling in bytes.")
    exposure.add_argument("--replay-tree-pss-bytes", type=int, default=None, help="Replay process-tree PSS ceiling in bytes.")
    exposure.add_argument("--min-available-bytes", type=int, default=None, help="Minimum effective physical headroom in bytes.")
    exposure.set_defaults(handler=run_exposure_scan_command)
    account = sub.add_parser(
        "account",
        help="Replay the strategy book as one real account under venue rules.",
        description="Replay the strategy book as one real account under venue rules.",
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
    account.add_argument("--execution", choices=("taker", "maker"), default="taker", help="Order execution model: immediate taker or canonical strict passive maker.")
    account.add_argument("--data-root", default=None, help="Existing OHLCV root override.")
    account.add_argument("--export-unit-returns", default=None, help="Write the unit reference ledger daily returns (sizing bootstrap for the live strategy step).")
    account.add_argument("--total-tree-pss-bytes", type=int, default=None, help="Total process-tree PSS ceiling in bytes.")
    account.add_argument("--replay-tree-pss-bytes", type=int, default=None, help="Replay process-tree PSS ceiling in bytes.")
    account.add_argument("--min-available-bytes", type=int, default=None, help="Minimum effective physical headroom in bytes.")
    account.set_defaults(handler=run_account_replay_command)


def _strategy_policy(breadth: int, variant: str = "primary") -> StrategySpec:
    """Select the strategy target policy for one backtest run.

    ``primary`` keeps the unlevered consensus book at the requested breadth (20 is the primary
    Top-20, other breadths are labelled controls). ``growth`` is registered only for breadth 20,
    because its exposure rung was derived from that book's own drawdown distribution and does not
    transfer to other universes. ``account_unit`` is the unlevered clip book the account ledger
    replays (Top-20 only; reconciliation reference).

    Raises:
        SystemExit: ``growth`` or ``account_unit`` is requested with a breadth other than 20,
            or ``variant`` is unknown.
    """
    from src.strategy.targets import (
        FLOW_MOM_TOP20,
        FLOW_MOM_TOP20_ACCOUNT_UNIT,
        FLOW_MOM_TOP20_GROWTH,
        FLOW_MOM_TOP40_CONTROL,
    )

    if variant == "growth":
        if breadth != 20:
            raise SystemExit(f"growth variant is registered only for breadth 20, got {breadth!r}")
        return FLOW_MOM_TOP20_GROWTH
    if variant == "account_unit":
        if breadth != 20:
            raise SystemExit(f"account_unit variant is registered only for breadth 20, got {breadth!r}")
        return FLOW_MOM_TOP20_ACCOUNT_UNIT
    if variant != "primary":
        raise SystemExit(f"unknown strategy variant {variant!r}")
    if breadth == 20:
        return FLOW_MOM_TOP20
    if breadth == 40:
        return FLOW_MOM_TOP40_CONTROL
    import dataclasses

    return dataclasses.replace(
        FLOW_MOM_TOP20,
        strategy_id=f"flow_mom_b{breadth}_control",
        breadth=breadth,
    )


def _strategy_run_name(start: pd.Timestamp, end: pd.Timestamp, breadth: int, created_at: pd.Timestamp, variant: str = "primary", execution: str = "taker") -> str:
    """Human-readable strategy run directory name: dates and breadth are legible without opening any file."""
    stem = f"{start:%Y%m%d}_{end:%Y%m%d}_top{breadth}"
    if variant == "growth":
        stem = f"{stem}_growth"
    if variant == "account_unit":
        stem = f"{stem}_account_unit"
    if execution == "maker":
        stem = f"{stem}_maker"
    return f"{stem}_{created_at:%Y%m%dT%H%M%S}Z"


def _resolve_strategy_destination(args: argparse.Namespace, *, start: pd.Timestamp, end: pd.Timestamp, breadth: int, variant: str = "primary", execution: str = "taker") -> Path:
    """Resolve the strategy result destination, defaulting to a fresh, human-readable run directory.

    Args:
        args: Parsed strategy backtest arguments.
        start: Registered evaluation start, used to name the default run directory.
        end: Registered evaluation end, used to name the default run directory.
        breadth: Declared universe breadth, used to name the default run directory.
    Returns:
        A fresh `result.json` destination under the strategy runs directory.
    Raises:
        SystemExit: An explicit destination is invalid or already occupied.
    """
    import os

    raw_output = getattr(args, "output", None)
    if raw_output is None:
        name = _strategy_run_name(start, end, breadth, pd.Timestamp.now(tz="UTC"), variant, execution)
        run_dir = STRATEGY_BACKTESTS_DIR / name
        suffix = 1
        while os.path.lexists(run_dir):
            suffix += 1
            run_dir = STRATEGY_BACKTESTS_DIR / f"{name}-{suffix}"
        run_dir.mkdir(parents=True)
        return run_dir / "result.json"
    output = Path(raw_output)
    if output.suffix != ".json":
        raise SystemExit(f"output must be a JSON path, got {raw_output!r}")
    if os.path.lexists(output):
        raise SystemExit(f"output must be fresh: {output} already exists")
    return output


def _write_strategy_manifest(output: Path, *, request: StrategyBacktestRequest, breadth: int) -> None:
    """Persist the strategy identity manifest beside the result envelope, and append it to the run index.

    Args:
        output: Finalized strategy result JSON destination.
        request: Executed strategy backtest request carrying identity timestamps.
        breadth: Declared universe breadth for the strategy variant.
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
    if output.parent.parent == STRATEGY_BACKTESTS_DIR:
        payload = json.loads(output.read_text(encoding="utf-8"))
        evaluation = payload.get("report_periods", {}).get("evaluation", {})
        append_backtest_index(
            index_path=STRATEGY_BACKTESTS_DIR.parent.parent / "index.jsonl",
            kind="mhs_frozen", run_dir=output.parent, created_at=pd.Timestamp.now(tz="UTC"),
            evaluation_start=request.evaluation_start, evaluation_end=request.evaluation_end,
            strategy_id=request.strategy.strategy_id,
            base_cagr=evaluation.get("base_cagr"), base_max_drawdown=evaluation.get("base_max_drawdown"),
            execution="maker" if request.execution_bound == "OHLCV_STRICT_PROXY" else "taker",
        )


def _prune_strategy_runs(keep: int) -> None:
    """Reclaim result detail for finalized strategy runs beyond the most recent `keep`; the index line is never removed.

    Args:
        keep: Positive count of most-recently-created run directories to retain in full.
    Returns:
        None after removing older run directories' files from disk.
    """
    import shutil

    if not STRATEGY_BACKTESTS_DIR.is_dir():
        return
    run_dirs = sorted(
        (d for d in STRATEGY_BACKTESTS_DIR.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for stale in run_dirs[keep:]:
        shutil.rmtree(stale, ignore_errors=True)


def run_strategy_backtest_command(args: argparse.Namespace) -> None:
    """Run a strategy Top-20/variant 3m inventory evaluation.

    This command is a distinct strategy identity from ``lab process-backtest``.  It
    accepts declared breadth and dates, builds no live artifact, and persists
    only completed strategy evidence.

    Args:
        args: Parsed strategy source/evaluation dates, breadth, paths, and
            existing memory-budget controls.
    Returns:
        None after a fresh completed research result is persisted.
    Raises:
        SystemExit: Arguments are invalid or the research replay fails.
    """
    from src.common.errors import DataIntegrityError
    from src.engine.backtest_evidence import StrategyReportPeriod
    from src.engine.backtest_persist import persist_strategy_backtest
    from src.engine.strategy_backtest import StrategyBacktestRequest, run_strategy_backtest

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
    selected_id = getattr(args, "strategy", None)
    if selected_id is not None:
        selected_breadth, selected_variant = {
            "flow_mom_top20": (20, "primary"),
            "flow_mom_top40_control": (40, "primary"),
            "flow_mom_top20_growth": (20, "growth"),
        }[selected_id]
        if (breadth != 20 and breadth != selected_breadth) or (variant != "primary" and variant != selected_variant):
            raise SystemExit("strategy conflicts with breadth or variant")
        breadth, variant = selected_breadth, selected_variant
    strategy = _strategy_policy(breadth, variant)
    execution = getattr(args, "execution", "taker")
    if execution not in ("taker", "maker"):
        raise SystemExit(f"execution must be 'taker' or 'maker', got {execution!r}")
    execution_bound: StrategyExecutionBound = "OHLCV_STRICT_PROXY" if execution == "maker" else "OHLCV_IMMEDIATE_TAKER"
    output = _resolve_strategy_destination(args, start=start, end=end, breadth=breadth, variant=variant, execution=execution)
    budget = _resolve_budget(args)
    base_spec, stress_spec = strategy_execution_specs()
    try:
        report_periods = (
            StrategyReportPeriod(
                label="evaluation",
                start=start.normalize(),
                end=(end - pd.Timedelta(days=1)).normalize(),
            ),
        )
        request = StrategyBacktestRequest(
            source_start=source_start, evaluation_start=start, evaluation_end=end,
            strategy=strategy, initial_equity=100000.0,
            base_spec=base_spec, stress_spec=stress_spec,
            report_periods=report_periods,
            data_root=Path(args.data_root) if getattr(args, "data_root", None) else None,
            memory_budget=budget,
            execution_bound=execution_bound,
        )
    except (DataIntegrityError, ValueError) as exc:
        raise SystemExit(f"invalid strategy backtest request: {exc}") from exc
    try:
        run = run_strategy_backtest(request)
        persist_strategy_backtest(run, output)
        _write_strategy_manifest(output, request=request, breadth=breadth)
        _logger.info(
            "[EVAL] backtest strategy source_gap_excluded=%s",
            list(getattr(run, "source_gap_excluded_symbols", ())),
        )
    except (DataIntegrityError, ValueError, OSError) as exc:
        raise SystemExit(f"strategy backtest failed: {exc}") from exc
    if output.parent.parent == STRATEGY_BACKTESTS_DIR and DEFAULT_DETAIL_RETENTION_MAX_RUNS is not None:
        _prune_strategy_runs(keep=DEFAULT_DETAIL_RETENTION_MAX_RUNS)
    status = "primary" if breadth == 20 else "research control"
    _logger.info(
        "[EVAL] backtest strategy strategy=%s status=%s variant=%s exposure=%s name_clip=%s",
        strategy.strategy_id, status, variant, strategy.exposure_multiplier, strategy.name_clip,
    )
    print(str(output))  # noqa: T201 -- strategy command prints only the finalized result path


def run_exposure_scan_command(args: argparse.Namespace) -> None:
    """Re-derive the growth exposure rung from one finished strategy run's ledger evidence.

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
    from src.application.strategy_account import AccountReplayError, ExposureScanRequest, run_exposure_scan

    raw_run_dir = getattr(args, "run_dir", None)
    if raw_run_dir is None:
        raise SystemExit("run-dir is required")
    data_root = getattr(args, "data_root", None)
    request = ExposureScanRequest(
        run_dir=Path(raw_run_dir),
        data_root=Path(data_root) if data_root is not None else None,
        memory_budget=_resolve_budget(args),
    )
    try:
        report = run_exposure_scan(request)
    except AccountReplayError as exc:
        raise SystemExit(str(exc)) from exc
    _logger.info(
        "[EVAL] exposure scan run=%s chosen=%.2f argmax=%.2f gaps_per_year=%.2f",
        request.run_dir.name, report.payload["chosen"], report.payload["argmax"], report.payload["gap_events_per_year"],
    )
    print(str(report.path))  # noqa: T201 -- exposure command prints only the finalized exposure path


def run_account_replay_command(args: argparse.Namespace) -> None:
    """Replay the strategy book as one real account and persist account-scale evidence.

    Builds the unlevered (exposure 1.0) clip-0.05 candidate and first replays it as the
    unit-exposure reference ledger (exposure 1, no order filters, no impact, reference
    capital). That ledger both supplies the causal posterior moments the growth policy
    sizes from and anchors the reconciliation against the canonical 3m ledger. The account
    is then replayed with the requested policy and venue snapshot, and ``account.json`` +
    ``account_daily.parquet`` are written in a fresh run directory
    ``<start>_<end>_top20_account_<policy>_<capital>_<ts>Z``. Under ``--execution maker``
    both the unit reference ledger and the account replay use the canonical strict passive
    rule, and reconciliation only references a canonical run with the same execution.

    Raises:
        SystemExit: Invalid arguments, missing venue snapshot, a data-integrity failure,
            or a failed unit reference replay.
    """
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
    execution = getattr(args, "execution", "taker")
    if execution not in ("taker", "maker"):
        raise SystemExit(f"execution must be 'taker' or 'maker', got {execution!r}")
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
    from src.application.strategy_account import AccountReplayError, AccountReplayRequest, run_account_replay

    raw_venue = getattr(args, "venue_rules", None)
    request = AccountReplayRequest(
        source_start=source_start, evaluation_start=start, evaluation_end=end,
        runs_root=STRATEGY_BACKTESTS_DIR, venue_rules_root=VENUE_RULES_DIR,
        policy=cast(Literal["growth", "fixed"], policy_name),
        execution=cast(Literal["taker", "maker"], execution),
        capital=capital, impact_y=impact_y,
        fixed_exposure=fixed_exposure,
        apply_order_filters=not getattr(args, "no_order_filters", False),
        venue_rules=Path(raw_venue) if raw_venue is not None else None,
        data_root=Path(args.data_root) if getattr(args, "data_root", None) else None,
        memory_budget=_resolve_budget(args),
        export_unit_returns=Path(export) if (export := getattr(args, "export_unit_returns", None)) is not None else None,
    )
    try:
        report = run_account_replay(request)
    except ValueError as exc:
        raise SystemExit(f"invalid account replay request: {exc}") from exc
    except AccountReplayError as exc:
        raise SystemExit(str(exc)) from exc
    _logger.info(
        "[EVAL] account-replay capital=%.0f policy=%s moment_source=%s cagr=%.4f mdd=%.4f liquidated=%s execution=%s maker_fill=%.3f",
        capital, policy_name, report.payload["moment_source"], report.payload["cagr"], report.payload["mdd"],
        report.result.liquidated_at, execution, report.result.maker_fill_fraction,
    )
