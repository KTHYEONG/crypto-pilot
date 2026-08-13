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
        discovery_gate_adjusted_net_t=args.discovery_gate_adjusted_net_t,
        discovery_gate_regime_scaled_net_t=args.discovery_gate_regime_scaled_net_t,
        fold_safe_horizon_selection=fold_safe_horizon,
        crash_regime_tilt_alpha=args.crash_regime_tilt_alpha,
        slow_book_mode=args.slow_book_mode,
        fast_book_mode=args.fast_book_mode,
        rebalance_filter=args.rebalance_filter,
        beta_neutralize=args.beta_neutralize,
        ensemble_signal=args.ensemble_signal,
        trend_efficiency_overlay=args.trend_efficiency_overlay,
        pnl_vol_target=not args.no_pnl_vol_target,
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
        "--discovery-gate-adjusted-net-t",
        action="store_true",
        default=False,
        help=(
            "Opt-in: also compute a Bartlett/HAC-adjusted net_t diagnostic per "
            "discovery candidate (requires --discovery-gate; never changes "
            "admitted/selected_horizon)"
        ),
    )
    mhs.add_argument(
        "--discovery-gate-regime-scaled-net-t",
        action="store_true",
        default=False,
        help=(
            "Opt-in: also compute a vol-regime cash-scale-adjusted net_t "
            "diagnostic per discovery candidate (approximate market-vol proxy, "
            "requires --discovery-gate; never changes admitted/selected_horizon)"
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
        "--slow-book-mode",
        choices=["single_horizon", "horizon_ensemble"],
        default="single_horizon",
        help=(
            "Slow-book construction: single_horizon (frozen production chain) "
            "or horizon_ensemble (equal-weight average of every candidate "
            "horizon, no selection -- docs/specs/mhs_alpha_engine.md §2)"
        ),
    )
    mhs.add_argument(
        "--fast-book-mode",
        choices=["single_horizon", "horizon_ensemble"],
        default="single_horizon",
        help=(
            "Fast-book construction: single_horizon (frozen production chain) "
            "or horizon_ensemble (equal-weight average of every candidate "
            "horizon, no selection -- the same rescue momentum already received, "
            "docs/specs/mhs_carry_and_fast_fair_evaluation.md §2.3)"
        ),
    )
    mhs.add_argument(
        "--rebalance-filter",
        choices=["per_symbol_deadband", "portfolio_trigger"],
        default="per_symbol_deadband",
        help=(
            "Turnover gate on the decision targets: per_symbol_deadband "
            "(published baseline) or portfolio_trigger (invariant-preserving "
            "row hold gated before the gross scale -- "
            "docs/specs/mhs_alpha_engine.md §1)"
        ),
    )
    mhs.add_argument(
        "--beta-neutralize",
        action="store_true",
        default=False,
        help=(
            "Orthogonally project the slow book onto the causal rolling market "
            "beta (sum(w)==0 and sum(w*beta)==0 by construction; parameter-free "
            "replacement for the crash-regime tilt -- "
            "docs/specs/mhs_alpha_engine.md §4)"
        ),
    )
    mhs.add_argument(
        "--ensemble-signal",
        choices=["raw", "vol_normalized"],
        default="raw",
        help=(
            "Signal family for the slow book: raw horizon log return (frozen "
            "production) or vol-normalized (re-open for measurement after the "
            "RC-1 deadband fix -- docs/specs/mhs_alpha_engine.md §6.1)"
        ),
    )
    mhs.add_argument(
        "--trend-efficiency-overlay",
        action="store_true",
        default=False,
        help=(
            "Opt-in exposure timing overlay on slow_momentum: scales gross "
            "exposure down in low-efficiency-ratio (choppy, momentum-hostile) "
            "regimes using the fast band's own horizon, composed with the "
            "existing regime cash scale (default False = byte-identical -- "
            "docs/specs/mhs_fast_reversal_overlay_redesign.md §2.3)"
        ),
    )
    mhs.add_argument(
        "--no-pnl-vol-target",
        action="store_true",
        default=False,
        help=(
            "Opt-out of the P&L vol-target layer: skip the multiplicative "
            "P&L-vol-target rescale between Pass 1 and Pass 2 (default keeps "
            "the layer on -- default flip requires the preregistered "
            "fold-train-only criterion in "
            "docs/specs/mhs_execution_friction_and_exposure_layers.md §6.1)"
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
