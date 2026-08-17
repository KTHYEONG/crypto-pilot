"""MHS Phase 1 CLI: ``research run portfolio mhs-horizon-diagnostic``.

Dev-only: the command registers no ``--unseal-holdout`` flag -- final OOS needs
a later architecture-freeze command, not a Phase 1 convenience flag.
"""

from __future__ import annotations

import argparse
import logging
import time

# The application module imports numpy/pandas transitively; it is imported
# lazily inside the handler so that merely registering the parser never pulls
# numpy into a coverage or import-graph that must stay light.

_logger = logging.getLogger("MhsHorizonDiagnosticCli")


def _run_mhs_horizon_diagnostic(args: argparse.Namespace) -> None:
    from src.application.research.mhs.evaluation import MhsDiagnosticRequest, MhsOutputTier
    from src.application.research.mhs.evaluation import mhs_horizon_diagnostic_report_path, persist_mhs_horizon_diagnostic_report, run_mhs_horizon_diagnostic

    fold_safe_horizon = args.fold_safe_horizon
    # Main-logic default: committee_capital + regime-adaptive tranche is the
    # best-measured configuration (see ADR_20260817_MHS_COMMITTEE_REGIME_ADAPTIVE_TRANCHE),
    # so both default on and are opt-out (--no-committee-capital /
    # --no-committee-regime-adaptive-tranche) rather than opt-in. An explicit
    # --committee-tranche-smoothing request always wins over the adaptive
    # default so the two stay mutually exclusive without a hard CLI error.
    committee_capital = not args.no_committee_capital
    committee_regime_adaptive_tranche = (
        committee_capital
        and not args.no_committee_regime_adaptive_tranche
        and not args.committee_tranche_smoothing
    )
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
        trend_sleeve=args.trend_sleeve,
        trend_sleeve_gross=args.trend_sleeve_gross,
        multi_feature_book=args.multi_feature_book,
        committee_book=args.committee_book,
        committee_kelly_sizing=args.committee_kelly_sizing,
        committee_growth_diagnostic=args.committee_growth_diagnostic,
        committee_capital=committee_capital,
        committee_tranche_smoothing=args.committee_tranche_smoothing,
        committee_regime_adaptive_tranche=committee_regime_adaptive_tranche,
        execution_coverage_gate=args.execution_coverage_gate,
        ram_guard=not args.no_ram_guard,
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
    persist_start = time.perf_counter()
    path = persist_mhs_horizon_diagnostic_report(
        report, mhs_horizon_diagnostic_report_path(),
        tier=MhsOutputTier(args.output_tier),
        request=request,
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
            "Optional process RSS budget in bytes; when unset the RAM guard "
            "auto-derives 85%% of total RAM. Exceeding the "
            "budget at a stage/window boundary fails closed with "
            "DataIntegrityError instead of OOM"
        ),
    )
    mhs.add_argument(
        "--no-ram-guard",
        action="store_true",
        default=False,
        help=(
            "Disable the automatic RAM guard (85%% budget + system reserve checks); "
            "--max-rss-bytes still applies when set"
        ),
    )
    mhs.add_argument("--no-log-run", action="store_true", default=False)
    mhs.add_argument(
        "--execution-timeframe",
        choices=["1m", "3m", "5m"],
        default="3m",
        help=(
            "OHLCV execution replay resolution; signal construction remains 1h. "
            "3m (2026-08 default) gives ~+27%% fill precision vs 5m and is "
            "collected natively from Binance"
        ),
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
        "--trend-sleeve",
        action="store_true",
        default=False,
        help=(
            "Opt-in: measure an additive time-series trend sleeve on the "
            "eligible market basket (net directional exposure the dollar-neutral "
            "books cannot hold); diagnostic-only unless --trend-sleeve-gross is set. "
            "WARNING (see ADR_20260817_MHS_TREND_SLEEVE_NEGATIVE_RESULT): wiring the "
            "sleeve into the committee_capital execution replay at gross=0.15 raised "
            "CAGR/Calmar/stress-Sharpe on the anchored folds but triggered "
            "CAPITAL_INVARIANT_BREACH (negative equity, 2025-10-03) in the continuous "
            "full-history replay -- fold-level pass/fail does not certify compounding "
            "safety for this overlay. Do not set --trend-sleeve-gross > 0.0 for capital "
            "decisions without re-deriving a fix, not just re-testing the same gross grid"
        ),
    )
    mhs.add_argument(
        "--trend-sleeve-gross",
        type=float,
        default=0.0,
        help=(
            "Gross budget allocated to the directional trend sleeve, in "
            "[0.0, 1.0]; a risk-budget policy value, never a fitted parameter. "
            "Measured negative at 0.15/0.30 (CAPITAL_INVARIANT_BREACH) -- see "
            "--trend-sleeve's warning"
        ),
    )
    mhs.add_argument(
        "--multi-feature-book",
        action="store_true",
        default=False,
        help=(
            "Opt-in: build the feature-axis registry into dollar-neutral rank "
            "books on the 24h decision grid with a per-year coverage gate "
            "(fail-closed exclusion) and equal-risk combination; report each "
            "admitted feature's coverage and regime-split stability plus the "
            "combined net Sharpe per cost tier and feature-book breadth "
            "(diagnostic-only, never an admission input)"
        ),
    )
    mhs.add_argument(
        "--committee-book",
        action="store_true",
        default=False,
        help=(
            "Opt-in: measure the declared k=5 wealth committee -- build the "
            "committee members into dollar-neutral rank books, audit RAW source "
            "coverage before any fillna, recover sign-safe gross/turnover-cost "
            "panels, and run the purged expanding-train walk-forward reporting "
            "wealth metrics per cost tier (diagnostic-only, never a combiner or "
            "capital input)"
        ),
    )
    mhs.add_argument(
        "--committee-kelly-sizing",
        action="store_true",
        default=False,
        help=(
            "Opt-in (with --committee-book, or committee capital which is on by "
            "default -- see --no-committee-capital): blend the committee "
            "total-exposure scale 50/50 with a train-only quarter-Kelly LCB "
            "overlay (f=0.25, z=1.0 one-SE shrinkage, capped at 1.0x when "
            "applied to committee capital, at 1.5x when diagnostic-only via "
            "--committee-book) instead of the flat vol-target scale alone. "
            "Measured on the committee-capital execution-replay path: reduces "
            "MDD but also reduces CAGR and Calmar (net negative for compounded "
            "growth) -- enable only if drawdown control is prioritized over "
            "compounding, not as a default performance improvement; see run "
            "history for magnitudes"
        ),
    )
    mhs.add_argument(
        "--committee-growth-diagnostic",
        action="store_true",
        default=False,
        help=(
            "Opt-in (requires --committee-book): report whether the committee's "
            "current exposure sits near its Monte-Carlo constrained growth-optimal "
            "point (block-bootstrap search over a discovery-window-only risk grid, "
            "reusing src.research.risk.growth_sizing -- the same framework already "
            "used for xs_alpha); observational only, never feeds back into sizing "
            "or capital allocation"
        ),
    )
    mhs.add_argument(
        "--no-committee-capital",
        action="store_true",
        default=False,
        help=(
            "Main logic default is ON: the k=5 committee members build the FOLD "
            "decision targets and the TOP-LEVEL reported blend (equal-weight "
            "over admitted members, no leg-risk tilt), replacing the frozen "
            "momentum book in both places; measured to raise walk-forward blend "
            "Sharpe and reduce blend MDD relative to the momentum default (see "
            "the run history for magnitudes). Pass this flag to opt back out to "
            "the frozen momentum book (also disables "
            "--committee-regime-adaptive-tranche, which requires committee "
            "capital)."
        ),
    )
    mhs.add_argument(
        "--committee-tranche-smoothing",
        action="store_true",
        default=False,
        help=(
            "Opt-in (requires committee capital, on by default): smooth the committee capital "
            "book with a 3-decision staggered tranche mean (effective 72h signal "
            "life) instead of fully repositioning every 24h -- the committee's "
            "shortest member lookback is 168h, so the 24h cadence oversamples its "
            "own signals. Measured on the execution replay: CAGR 15.0%%->16.1%%, "
            "MDD -28.3%%->-26.5%%, Calmar +14.7%%, annualized turnover -18%%, "
            "cost-stress Sharpe +11%%; BUT anchored-fold pass count drops 2->1 "
            "(2023 recovers 0.151->1.537 while 2024 degrades 0.968->0.481), so "
            "top-level compounding improves while the Research GO fold gate worsens."
        ),
    )
    mhs.add_argument(
        "--no-committee-regime-adaptive-tranche",
        action="store_true",
        default=False,
        help=(
            "Main logic default is ON (requires committee capital, which is "
            "also on by default; auto-disabled if --committee-tranche-smoothing "
            "is explicitly passed instead, since the two are mutually "
            "exclusive): per-decision-row choice between the raw committee book "
            "and its 3-decision tranche smooth, gated by a "
            "causal trailing lag-1 autocorrelation of the raw book's own proxy "
            "return over the last 15 decision rows (negative/mean-reverting -> "
            "smooth, non-negative/trending -> raw) -- root cause: "
            "--committee-tranche-smoothing helps years where the raw book's own "
            "returns whipsaw (negative autocorrelation) and hurts years where "
            "they persist (positive autocorrelation), so a fixed choice always "
            "sacrifices one regime. Measured on the execution replay: CAGR "
            "15.0%%->20.7%%, MDD -28.3%%->-16.6%%, Calmar +135%%, annualized "
            "turnover -21%%, cost-stress Sharpe +38%%, AND anchored-fold pass "
            "count improves 2->3 (2023 0.151->1.16, 2024 0.968->0.85, 2025 "
            "3.08->3.39, all comfortably above the 0.6 floor) -- dominates the "
            "fixed-tranche choice on every fold simultaneously, not a tradeoff. "
            "The 15-decision-row window is frozen (not CLI-tunable): windows "
            "15-25 all pass every fold in real replay (plateau), but windows "
            "10 and 90 both trigger CAPITAL_INVARIANT_BREACH -- this is not a "
            "free parameter to fiddle with. Pass this flag to opt back out to "
            "the raw (tranche_count=1) committee book."
        ),
    )
    mhs.add_argument(
        "--execution-coverage-gate",
        action="store_true",
        default=False,
        help=(
            "Opt-in pre-flight check: verify every funded symbol has "
            "execution_timeframe OHLCV cache coverage for [start, end] before "
            "the replay runs; fails closed with the missing/gapped symbol list "
            "instead of a late opaque MISSING_DATA termination count"
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
            "the fully dollar-neutral book)"
        ),
    )
    mhs.add_argument(
        "--slow-book-mode",
        choices=["single_horizon", "horizon_ensemble"],
        default="single_horizon",
        help=(
            "Slow-book construction: single_horizon (frozen production chain) "
            "or horizon_ensemble (equal-weight average of every candidate "
            "horizon, no selection)"
        ),
    )
    mhs.add_argument(
        "--fast-book-mode",
        choices=["single_horizon", "horizon_ensemble"],
        default="single_horizon",
        help=(
            "Fast-book construction: single_horizon (frozen production chain) "
            "or horizon_ensemble (equal-weight average of every candidate "
            "horizon, no selection)"
        ),
    )
    mhs.add_argument(
        "--rebalance-filter",
        choices=["per_symbol_deadband", "portfolio_trigger"],
        default="per_symbol_deadband",
        help=(
            "Turnover gate on the decision targets: per_symbol_deadband "
            "(published baseline) or portfolio_trigger (invariant-preserving "
            "row hold gated before the gross scale)"
        ),
    )
    mhs.add_argument(
        "--beta-neutralize",
        action="store_true",
        default=False,
        help=(
            "Orthogonally project the slow book onto the causal rolling market "
            "beta (sum(w)==0 and sum(w*beta)==0 by construction; parameter-free "
            "replacement for the crash-regime tilt)"
        ),
    )
    mhs.add_argument(
        "--ensemble-signal",
        choices=["raw", "vol_normalized"],
        default="raw",
        help=(
            "Signal family for the slow book: raw horizon log return (frozen "
            "production) or vol-normalized"
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
            "existing regime cash scale (default False = byte-identical)"
        ),
    )
    mhs.add_argument(
        "--no-pnl-vol-target",
        action="store_true",
        default=False,
        help=(
            "Opt-out of the P&L vol-target layer: skip the multiplicative "
            "P&L-vol-target rescale between Pass 1 and Pass 2"
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
