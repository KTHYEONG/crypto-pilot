"""``lab`` command group: exploratory research.

Results here never deploy by themselves; a strategy reaches live trading only
as a ``StrategyRelease`` accepted by ``evaluate strategy``.

Every ``src.lab`` import lives inside handler (or leaf-registration) function
bodies, so importing this module and building parsers for the other groups
never loads the research stack.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from src.backtests.contracts import RetentionPolicy
from src.cli.dataclass_args import add_dataclass_arguments
from src.core.params import LEVERAGE_FRONTIER_SCAN_MULTIPLES
from src.core.resources import MhsMemoryBudget

_logger = logging.getLogger("MhsHorizonDiagnosticCli")


def _parse_float_csv(raw: str) -> tuple[float, ...]:
    values: list[float] = []
    for token in raw.split(","):
        try:
            values.append(float(token.strip()))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"invalid float value in --leverage-frontier-multiples: {token!r}"
            ) from None
    return tuple(values)


def _run_horizon_diagnostic(args: argparse.Namespace) -> None:
    from src.cli.dataclass_args import explicit_field_values
    from src.core.params import CLI_GROWTH_ENVELOPE_DEFAULT
    from src.lab.mhs.contracts import MhsDiagnosticRequest

    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    if getattr(args, "leverage_frontier_scan", False):
        from src.lab.mhs.leverage_scan import run_leverage_frontier_scan

        run_leverage_frontier_scan(
            explicit.get("growth_envelope", CLI_GROWTH_ENVELOPE_DEFAULT),
            tuple(args.leverage_frontier_multiples),
        )
        return

    from src.common.paths import BACKTESTS_DIR, DATA_DIR
    from src.lab.mhs.contracts import MhsOutputTier
    from src.lab.mhs.pipeline.config import resolve_cli_request
    from src.lab.mhs.pipeline.orchestrator import run_mhs_diagnostic
    from src.lab.mhs.preregistration import PROCEDURE_REGISTRY_PATH
    from src.lab.mhs.report.persist import persist_mhs_horizon_diagnostic_report

    request = resolve_cli_request(explicit)
    if getattr(args, "register_procedure", False):
        from src.lab.mhs.preregistration import register_procedure

        registration = register_procedure(
            request,
            now=pd.Timestamp.now(tz="UTC"),
            registry_path=PROCEDURE_REGISTRY_PATH,
            history_dir=BACKTESTS_DIR,
        )
        _logger.info(
            "[EVAL] procedure_registered digest=%s effective_start=%s",
            registration.procedure_digest,
            registration.effective_start.isoformat(),
        )
        return
    from src.common.logging import LOG_DIR, setup_logger
    from src.lab.mhs.telemetry import TELEMETRY_LOGGER_NAME

    setup_logger(TELEMETRY_LOGGER_NAME, log_dir=LOG_DIR)
    report = run_mhs_diagnostic(request, procedure_registry=PROCEDURE_REGISTRY_PATH, history_dir=BACKTESTS_DIR)
    persist_start = time.perf_counter()
    from uuid import uuid4

    run_id = getattr(args, "run_id", None) or uuid4().hex
    report_path = DATA_DIR / "research" / "mhs" / run_id / "mhs_horizon_diagnostic.json"
    path = persist_mhs_horizon_diagnostic_report(
        report,
        report_path,
        tier=MhsOutputTier(args.output_tier),
        request=request,
        history_dir=BACKTESTS_DIR,
        procedure_registry=PROCEDURE_REGISTRY_PATH,
    )
    _logger.info(
        "[SYS] stage=persist_report elapsed_ms=%d",
        int((time.perf_counter() - persist_start) * 1000),
    )
    _logger.info(
        "[EVAL] horizon-diagnostic status=%s books=%s blend=%s path=%s",
        report.status,
        sorted(report.books),
        report.blend is not None,
        path,
    )


def _add_horizon_diagnostic_leaf(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    from src.lab.mhs.contracts import MhsDiagnosticRequest

    leaf = sub.add_parser(
        "horizon-diagnostic",
        help="Run the dev-only MHS Phase 1 two-band multi-horizon diagnostic",
    )
    add_dataclass_arguments(leaf, MhsDiagnosticRequest)
    leaf.add_argument(
        "--output-tier",
        choices=["compact", "full"],
        default="compact",
        help=(
            "Persistence tier: compact (default) writes a git-committable "
            "daily-resampled ledger + stripped summary JSON; full writes the "
            "lossless per-fill audit Parquet tables under _full/ (gitignored)"
        ),
    )
    leaf.add_argument(
        "--leverage-frontier-scan",
        action="store_true",
        default=False,
        help=(
            "Opt-in: skip the full diagnostic pipeline and instead scan a wide "
            "leverage-multiple grid against the registered growth envelope's "
            "bootstrap ruin/mdd frontier, using the already-persisted "
            "daily_ledger.parquet from a prior run. Diagnostic-only -- never "
            "mutates GROWTH_RISK_ENVELOPES or production state; adopting a "
            "candidate still requires registering a new envelope rung and "
            "re-running a real 3m replay under the registered adoption protocol"
        ),
    )
    leaf.add_argument(
        "--leverage-frontier-multiples",
        type=_parse_float_csv,
        default=LEVERAGE_FRONTIER_SCAN_MULTIPLES,
        help=(
            "Comma-separated candidate leverage multiples for "
            "--leverage-frontier-scan, e.g. 2.0,2.5,3.0. Defaults to "
            "LEVERAGE_FRONTIER_SCAN_MULTIPLES (0.25 through 5.0 in 0.25 steps)"
        ),
    )
    leaf.add_argument(
        "--register-procedure",
        action="store_true",
        default=False,
        help="Freeze this flag set as a pre-registered procedure in the procedure registry (data/backtests/procedure_registry.jsonl) and exit without running.",
    )
    leaf.add_argument(
        "--run-id",
        default=None,
        help="Explicit run identity for research output under data/research/mhs/<run_id>/; omitted generates one.",
    )
    leaf.set_defaults(handler=_run_horizon_diagnostic)


def _resolve_process_destinations(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """Resolve the sole result document and optional exact-target export for one lab run."""
    import os

    from src.common.paths import BACKTESTS_DIR

    targets_output = Path(args.targets_output) if args.targets_output is not None else None
    if targets_output is not None:
        if targets_output.suffix != ".parquet":
            raise SystemExit(f"targets-output must be a parquet path, got {args.targets_output!r}")
        if os.path.lexists(targets_output):
            raise SystemExit(f"targets-output must be fresh: {targets_output} already exists")
    if args.output is None:
        import uuid

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


def _resolve_process_retention_policy(args: argparse.Namespace) -> RetentionPolicy | None:
    """Build explicit destructive detail budgets, failing before any workload launch."""
    from src.core.params import DEFAULT_DETAIL_RETENTION_MAX_BYTES, DEFAULT_DETAIL_RETENTION_MAX_RUNS

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


def _resolve_process_fingerprint(
    args: argparse.Namespace,
    start: pd.Timestamp,
    end: pd.Timestamp,
    budget: MhsMemoryBudget | None,
) -> str:
    """Compute the immutable reuse fingerprint for one canonical request."""
    from src.lab.mhs.app.supervisor import request_fingerprint

    return request_fingerprint(
        start=start,
        end=end,
        data_root=getattr(args, "data_root", None),
        tracking_error_threshold=getattr(args, "rebalance_tracking_error_threshold", None),
        memory_budget=budget,
        execution_timeframe=getattr(args, "execution_timeframe", "3m"),
    )


def run_process_backtest(args: argparse.Namespace) -> None:
    """Run the canonical three-minute process evaluation, or reuse a verified equivalent run.

    Reuse is offered only for the default run-directory layout. An explicit
    destination is a demand for files at that path, and a reused run cannot
    satisfy it. A fingerprint match is reused only after the registry
    certifies that the run completed validly and its recorded artifacts still
    verify on disk. Any refused match falls through to a fresh supervised run,
    so stale or failed history can never end the command successfully.
    """
    import json
    import uuid

    from src.backtests.catalog import append_backtest_index
    from src.cli.commands.backtest import _resolve_budget, _utc_timestamp
    from src.common.errors import DataIntegrityError
    from src.common.paths import BACKTESTS_DIR
    from src.core.params import DISCOVERY_START, PROCESS_EVALUATION_CEILING
    from src.lab.mhs.app.supervisor import find_reused_run, run_mhs_process_backtest

    if getattr(args, "execution_timeframe", "3m") != "3m":
        raise SystemExit(f"execution-timeframe must be 3m, got {getattr(args, 'execution_timeframe', None)!r}")
    start = _utc_timestamp(getattr(args, "start", None), "start", DISCOVERY_START)
    end = _utc_timestamp(getattr(args, "end", None), "end", PROCESS_EVALUATION_CEILING)
    budget = _resolve_budget(args)
    registry_path = (
        Path(args.registry_path) if getattr(args, "registry_path", None) else BACKTESTS_DIR / "registry.sqlite3"
    )
    retention_policy = _resolve_process_retention_policy(args)
    fingerprint = _resolve_process_fingerprint(args, start, end, budget)
    explicit_destination = args.output is not None or args.targets_output is not None
    if not getattr(args, "force", False) and explicit_destination:
        _logger.info("[DATA] lab process-backtest reuse_skipped reason=explicit_destination")
    elif not getattr(args, "force", False):
        try:
            lookup = find_reused_run(registry_path, fingerprint)
        except DataIntegrityError as exc:
            _logger.error("[DATA] lab process-backtest reuse_lookup_failed registry=%s", registry_path, exc_info=True)
            raise SystemExit(f"registry integrity failure: {exc}") from exc
        if lookup.reused is not None:
            reused = lookup.reused
            _logger.info(
                "[DATA] lab process-backtest reuse run_id=%s result_path=%s targets_path=%s finalized_at=%s evidence_retained=%s",
                reused.run_id,
                reused.result_path,
                reused.targets_path,
                reused.finalized_at,
                reused.evidence_retained,
            )
            print(str(reused.result_path))  # noqa: T201 -- prints only the reused result path
            return
    result_output, targets_output = _resolve_process_destinations(args)
    run_id = result_output.parent.name if getattr(args, "output", None) is None else uuid.uuid4().hex
    _logger.info(
        "[EVAL] lab process-backtest result_output=%s targets_output=%s",
        result_output,
        targets_output,
    )
    try:
        run = run_mhs_process_backtest(
            start=start,
            end=end,
            data_root=args.data_root,
            result_output=result_output,
            targets_output=targets_output,
            tracking_error_threshold=args.rebalance_tracking_error_threshold,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
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
        append_backtest_index(
            index_path=BACKTESTS_DIR / "index.jsonl",
            kind="mhs",
            run_dir=result_output.parent,
            created_at=pd.Timestamp.now(tz="UTC"),
            evaluation_start=start,
            evaluation_end=end,
            strategy_id="process_inventory_3m",
            base_cagr=base.get("cagr"),
            base_max_drawdown=base.get("max_drawdown"),
        )
    print(str(result_output))  # noqa: T201 -- prints only the finalized result path


def _add_process_backtest_leaf(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    leaf = sub.add_parser(
        "process-backtest",
        help="Supervised 3m inventory evaluation of the continuous MHS process.",
        description="Canonical three-minute inventory evidence with comparative hourly proxy state.",
    )
    leaf.add_argument("--start", default=None, help="UTC source start; date-only values are UTC.")
    leaf.add_argument("--end", default=None, help="UTC registered evaluation end.")
    leaf.add_argument("--data-root", default=None, help="Existing OHLCV root override.")
    leaf.add_argument(
        "--output",
        default=None,
        help="Fresh complete result envelope JSON destination; omitted creates a unique run directory.",
    )
    leaf.add_argument("--targets-output", default=None, help="Optional fresh exact-target parquet destination.")
    leaf.add_argument(
        "--rebalance-tracking-error-threshold",
        type=float,
        default=None,
        help="Existing optional process adoption control; omitted preserves baseline.",
    )
    leaf.add_argument("--timeout-seconds", type=float, default=None, help="Optional positive finite wall timeout.")
    leaf.add_argument(
        "--poll-seconds",
        type=float,
        default=0.25,
        help="Positive finite resource observation interval.",
    )
    leaf.add_argument("--total-tree-pss-bytes", type=int, default=None, help="Total process-tree PSS ceiling in bytes.")
    leaf.add_argument(
        "--replay-tree-pss-bytes", type=int, default=None, help="Replay process-tree PSS ceiling in bytes."
    )
    leaf.add_argument(
        "--min-available-bytes", type=int, default=None, help="Minimum effective physical headroom in bytes."
    )
    leaf.add_argument(
        "--execution-timeframe",
        choices=["3m"],
        default="3m",
        help="Execution replay resolution; fixed to 3m and never changes strategy cadence.",
    )
    leaf.add_argument(
        "--max-detail-bytes",
        type=int,
        default=None,
        help="Optional destructive detail budget in bytes; omitted keeps every managed bundle.",
    )
    leaf.add_argument(
        "--max-detail-runs",
        type=int,
        default=None,
        help="Optional destructive detail budget in finalized runs; omitted keeps every managed bundle.",
    )
    leaf.add_argument(
        "--registry-path",
        default=None,
        help="Local execution registry; omitted uses the canonical backtests registry.",
    )
    leaf.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Execute an equivalent request again instead of reusing the finalized match.",
    )
    leaf.set_defaults(handler=run_process_backtest)


def _run_backtests_migrate(args: argparse.Namespace) -> None:
    from src.lab.mhs.app.backtests_migration import migrate_legacy_backtests

    registry_path = Path(args.registry_path)
    histories = tuple(Path(item) for item in args.history_directory)
    runs = tuple(Path(item) for item in args.run_directory)
    report = migrate_legacy_backtests(
        registry_path=registry_path,
        history_directories=histories,
        run_directories=runs,
        dry_run=not args.apply,
    )
    _logger.info(
        "[SYS] lab backtests-migrate imported=%s skipped=%s protected=%s rejected=%s",
        report["imported"],
        report["skipped"],
        report["protected"],
        report["rejected"],
    )


def _run_backtests_verify_history_migration(args: argparse.Namespace) -> None:
    from src.common.errors import DataIntegrityError
    from src.lab.mhs.app.backtests_migration import verify_legacy_history_migration

    try:
        report = verify_legacy_history_migration(
            registry_path=Path(args.registry_path), source=Path(args.history_directory)
        )
    except DataIntegrityError as exc:
        _logger.error("[SYS] lab backtests-verify-history-migration failed error=%s", exc)
        raise SystemExit(1) from exc
    _logger.info(
        "[SYS] lab backtests-verify-history-migration verified source=%s records=%s trials=%s",
        report["source"],
        report["source_records"],
        report["distinct_trial_identities"],
    )


def _run_procedure_registry_migrate(args: argparse.Namespace) -> None:
    from src.common.errors import DataIntegrityError
    from src.lab.mhs.preregistration import migrate_legacy_procedure_registry

    try:
        moved = migrate_legacy_procedure_registry(
            legacy_path=Path(args.legacy_path), target_path=Path(args.target_path)
        )
    except DataIntegrityError as exc:
        _logger.error("[DATA] lab procedure-registry-migrate failed error=%s", exc)
        raise SystemExit(1) from exc
    _logger.info(
        "[DATA] lab procedure-registry-migrate moved=%s legacy=%s target=%s",
        moved,
        args.legacy_path,
        args.target_path,
    )


def add_lab_commands(lab_parser: argparse.ArgumentParser) -> None:
    """Register the ``lab`` leaves on the lab group parser.

    The horizon-diagnostic leaf needs the research request dataclass, so this
    function loads ``src.lab``; it is called only when the ``lab`` group is
    selected, keeping every other command startup lab-free.
    """
    sub = lab_parser.add_subparsers(dest="lab_command", required=True)
    _add_horizon_diagnostic_leaf(sub)
    _add_process_backtest_leaf(sub)
    migrate = sub.add_parser("backtests-migrate", help="Import explicitly selected legacy backtest evidence")
    migrate.add_argument("--registry-path", type=str, required=True, help="Target registry SQLite path")
    migrate.add_argument(
        "--history-directory", type=str, action="append", default=[], help="Explicit legacy history directory"
    )
    migrate.add_argument("--run-directory", type=str, action="append", default=[], help="Explicit legacy run directory")
    migrate.add_argument("--apply", action="store_true", default=False, help="Write the registry instead of previewing")
    migrate.set_defaults(handler=_run_backtests_migrate)
    verify = sub.add_parser(
        "backtests-verify-history-migration",
        help="Verify a legacy history source is durably represented by the registry",
    )
    verify.add_argument("--registry-path", type=str, required=True, help="Target registry SQLite path")
    verify.add_argument("--history-directory", type=str, required=True, help="Explicit legacy history directory")
    verify.set_defaults(handler=_run_backtests_verify_history_migration)
    proc = sub.add_parser(
        "procedure-registry-migrate",
        help="Move the legacy docs-tree procedure registry to the backtests root exactly once",
    )
    proc.add_argument("--legacy-path", type=str, required=True, help="Legacy procedure registry JSONL")
    proc.add_argument("--target-path", type=str, required=True, help="Target procedure registry JSONL")
    proc.set_defaults(handler=_run_procedure_registry_migrate)
