"""MHS Phase 1 CLI: ``research run portfolio mhs-horizon-diagnostic``.

Dev-only: the command registers no ``--unseal-holdout`` flag -- final OOS needs
a later architecture-freeze command, not a Phase 1 convenience flag.
"""

from __future__ import annotations

import argparse
import logging

# The application module imports numpy/pandas transitively; it is imported
# lazily inside the handler so that merely registering the parser never pulls
# numpy into a coverage or import-graph that must stay light.

_logger = logging.getLogger("MhsHorizonDiagnosticCli")


def _run_mhs_horizon_diagnostic(args: argparse.Namespace) -> None:
    from src.application.research.mhs.evaluation import MhsDiagnosticRequest, MhsOutputTier
    from src.application.research.mhs.evaluation import mhs_horizon_diagnostic_report_path, persist_mhs_horizon_diagnostic_report, run_mhs_horizon_diagnostic

    fold_safe_horizon = args.fold_safe_horizon
    request = MhsDiagnosticRequest(
        start=args.start,
        end=args.end,
        mark_mode=args.mark_mode,
        execution_timeframe=args.execution_timeframe,
        max_rss_bytes=args.max_rss_bytes,
        log_run=not args.no_log_run,
        touch_diagnostic=args.touch_diagnostic,
        ladder_diagnostic=args.ladder_diagnostic,
        discovery_gate=args.discovery_gate,
        fold_safe_horizon_selection=fold_safe_horizon,
        crash_regime_tilt_alpha=args.crash_regime_tilt_alpha,
    )
    report = run_mhs_horizon_diagnostic(request)
    path = persist_mhs_horizon_diagnostic_report(
        report, mhs_horizon_diagnostic_report_path(),
        tier=MhsOutputTier(args.output_tier),
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
    mhs.add_argument("--start", default=None)
    mhs.add_argument("--end", default=None)
    mhs.add_argument(
        "--mark-mode",
        choices=["cache_required", "cache_required_stale_carry", "ohlcv_close_fallback"],
        default="cache_required",
        help=(
            "Mark-price valuation source: cache_required builds the causal mark "
            "panel and fails closed; cache_required_stale_carry allows bounded "
            "diagnostic continuity; ohlcv_close_fallback is fixture-only"
        ),
    )
    mhs.add_argument(
        "--max-rss-bytes",
        type=int,
        default=None,
        help=(
            "Optional process RSS budget in bytes; exceeding it at an execution "
            "window boundary fails closed with DataIntegrityError instead of OOM"
        ),
    )
    mhs.add_argument("--no-log-run", action="store_true", default=False)
    mhs.add_argument(
        "--execution-timeframe",
        choices=["1m", "5m"],
        default="5m",
        help="OHLCV execution replay resolution; signal construction remains 1h",
    )
    mhs.add_argument(
        "--touch-diagnostic",
        action="store_true",
        default=False,
        help=(
            "Additionally replay slow_momentum/blend under OHLCV_TOUCH_PROXY "
            "alongside the strict/stress pair -- adds a second full window "
            "pass, opt-in only"
        ),
    )
    mhs.add_argument(
        "--ladder-diagnostic",
        action="store_true",
        default=False,
        help=(
            "Additionally replay slow_momentum/blend under OHLCV_LADDERED_PROXY "
            "alongside the strict/stress pair -- adds a third full window pass "
            "with the escalating limit ladder, opt-in only"
        ),
    )
    mhs.add_argument(
        "--discovery-gate",
        action="store_true",
        default=False,
        help=(
            "Run the discovery/qualification horizon-selection gate on the "
            "current panel for both sign families and record the outcome in "
            "the diagnostic report (opt-in; the result never changes contracts "
            "by itself)"
        ),
    )
    mhs.add_argument(
        "--fold-safe-horizon",
        action="store_true",
        default=False,
        help=(
            "Reselect the slow_momentum horizon per anchored fold from a "
            "leak-free discovery/qualification run confined to each fold's "
            "train data (opt-in; measured to be a safe no-op against the "
            "current admission floor)"
        ),
    )
    mhs.add_argument(
        "--crash-regime-tilt-alpha",
        type=float,
        default=None,
        help=(
            "Opt-in crash-regime directional tilt on slow_momentum: fraction "
            "(0.0, 1.0] of unit gross reallocated to a BTCUSDT-trend-scaled "
            "directional overlay (default None = disabled, byte-identical to "
            "the fully dollar-neutral book; no value is 'recommended' -- see "
            "docs/specs/mhs_crash_regime_tilt_overlay.md)"
        ),
    )
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
    mhs.set_defaults(handler=_run_mhs_horizon_diagnostic)
