"""MHS Phase 1 CLI: ``research run portfolio mhs-horizon-diagnostic``.

Dev-only: the command registers no ``--unseal-holdout`` flag -- final OOS needs
a later architecture-freeze command, not a Phase 1 convenience flag.
"""

from __future__ import annotations

import argparse
import logging
import time

from src.cli.dataclass_args import add_dataclass_arguments
from src.core.params import (
    CLI_GROWTH_ENVELOPE_DEFAULT,
    LEVERAGE_FRONTIER_SCAN_MULTIPLES,
)
from src.mhs.contracts import MhsDiagnosticRequest

# The application module imports numpy/pandas transitively; it is imported
# lazily inside the handler so that merely registering the parser never pulls
# numpy into a coverage or import-graph that must stay light.

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


def _run_mhs_horizon_diagnostic(args: argparse.Namespace) -> None:
    from src.cli.dataclass_args import explicit_field_values

    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    if getattr(args, "leverage_frontier_scan", False):
        # Diagnostic-only short-circuit: reads an already-persisted ledger and
        # returns before any heavy pipeline import; never builds a request.
        from src.mhs.leverage_scan import run_leverage_frontier_scan

        run_leverage_frontier_scan(
            explicit.get("growth_envelope", CLI_GROWTH_ENVELOPE_DEFAULT),
            tuple(args.leverage_frontier_multiples),
        )
        return

    from src.common.paths import BACKTESTS_DIR, DATA_DIR
    from src.mhs.contracts import MhsOutputTier
    from src.mhs.pipeline.config import resolve_cli_request
    from src.mhs.pipeline.orchestrator import run_mhs_diagnostic
    from src.mhs.preregistration import PROCEDURE_REGISTRY_PATH
    from src.mhs.report.persist import persist_mhs_horizon_diagnostic_report

    request = resolve_cli_request(explicit)
    if getattr(args, "register_procedure", False):
        import pandas as pd

        from src.mhs.preregistration import register_procedure
        registration = register_procedure(
            request, now=pd.Timestamp.now(tz="UTC"),
            registry_path=PROCEDURE_REGISTRY_PATH, history_dir=BACKTESTS_DIR,
        )
        _logger.info("[EVAL] procedure_registered digest=%s effective_start=%s", registration.procedure_digest, registration.effective_start.isoformat())
        return
    from src.common.logging import LOG_DIR, setup_logger
    from src.mhs.telemetry import TELEMETRY_LOGGER_NAME

    setup_logger(TELEMETRY_LOGGER_NAME, log_dir=LOG_DIR)
    report = run_mhs_diagnostic(request, procedure_registry=PROCEDURE_REGISTRY_PATH, history_dir=BACKTESTS_DIR)
    persist_start = time.perf_counter()
    from uuid import uuid4

    run_id = getattr(args, "run_id", None) or uuid4().hex
    report_path = DATA_DIR / "research" / "mhs" / run_id / "mhs_horizon_diagnostic.json"
    path = persist_mhs_horizon_diagnostic_report(
        report, report_path,
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
        "[EVAL] mhs-horizon-diagnostic status=%s books=%s blend=%s path=%s",
        report.status, sorted(report.books), report.blend is not None, path,
    )


def add_mhs_commands(portfolio_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the dev-only ``research run portfolio mhs-horizon-diagnostic`` subcommand."""
    mhs = portfolio_sub.add_parser(
        "mhs-horizon-diagnostic",
        help="Run the dev-only MHS Phase 1 two-band multi-horizon diagnostic",
    )
    add_dataclass_arguments(mhs, MhsDiagnosticRequest)
    mhs.add_argument(
        "--output-tier",
        choices=["compact", "full"],
        default="compact",
        help=(
            "Persistence tier: compact (default) writes a git-committable "
            "daily-resampled ledger + stripped summary JSON; full writes the "
            "lossless per-fill audit Parquet tables under _full/ (gitignored)"
        ),
    )
    mhs.add_argument(
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
    mhs.add_argument(
        "--leverage-frontier-multiples",
        type=_parse_float_csv,
        default=LEVERAGE_FRONTIER_SCAN_MULTIPLES,
        help=(
            "Comma-separated candidate leverage multiples for "
            "--leverage-frontier-scan, e.g. 2.0,2.5,3.0. Defaults to "
            "LEVERAGE_FRONTIER_SCAN_MULTIPLES (0.25 through 5.0 in 0.25 steps)"
        ),
    )
    mhs.add_argument("--register-procedure", action="store_true", default=False, help="Freeze this flag set as a pre-registered procedure in the procedure registry (data/backtests/procedure_registry.jsonl) and exit without running.")
    mhs.add_argument('--run-id', default=None, help='Explicit run identity for research output under data/research/mhs/<run_id>/; omitted generates one.')
    mhs.set_defaults(handler=_run_mhs_horizon_diagnostic)
