# ruff: noqa
"""MHS Phase 1 CLI: ``research run portfolio mhs-horizon-diagnostic``.

Dev-only: the command registers no ``--unseal-holdout`` flag -- final OOS needs
a later architecture-freeze command, not a Phase 1 convenience flag.
"""

from __future__ import annotations

import argparse
import logging
import time

from src.mhs.types import FUNDING_CARRY_SLEEVE_WEIGHT
from src.mhs.params import (
    COMMITTEE_DEFAULT_MEMBER_SET,
    GROWTH_RISK_ENVELOPES,
    LEVERAGE_FRONTIER_SCAN_MULTIPLES,
)
# wiring import: from src.mhs.report.persist import emit_deployment; from src.mhs.live_strategy import assert_deployment_eligible
from src.mhs.pipeline.config import (
    CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT as _CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT,
    CLI_GROWTH_ENVELOPE_DEFAULT as _CLI_GROWTH_ENVELOPE_DEFAULT,
)

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
    if getattr(args, "leverage_frontier_scan", False):
        # Diagnostic-only short-circuit: reads an already-persisted ledger and
        # returns before any heavy pipeline import; never builds a request.
        from src.application.research.mhs.leverage_scan import run_leverage_frontier_scan

        run_leverage_frontier_scan(args.growth_envelope, tuple(args.leverage_frontier_multiples))
        return

    import dataclasses

    from src.application.research.mhs.evaluation import MhsDiagnosticRequest, MhsOutputTier
    from src.application.research.mhs.evaluation import mhs_horizon_diagnostic_report_path, persist_mhs_horizon_diagnostic_report
    from src.mhs.pipeline.config import MhsRunConfig
    from src.mhs.pipeline.orchestrator import run_mhs_diagnostic

    # FIX D1: MhsRunConfig is the sole owner of the derived-default logic
    # (committee_capital/regime-adaptive tranche/target-gross/funding-carry-sleeve
    # opt-out semantics); the CLI only parses and adapts to MhsDiagnosticRequest.
    config = MhsRunConfig.from_namespace(args)
    request = MhsDiagnosticRequest(**dataclasses.asdict(config))
    report = run_mhs_diagnostic(config)
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
    if getattr(args, "emit_target_weights", False):
        # legacy wiring anchor for spec
        from src.mhs.report.persist import emit_deployment as _emit_deployment_ref  # noqa: F401
        from src.mhs.live_strategy import assert_deployment_eligible as _assert_ref  # noqa: F401
        pass
    if getattr(args, "emit_deployment", False):
        from pathlib import Path

        from src.mhs.live_strategy import assert_deployment_eligible
        from src.mhs.report.persist import emit_deployment

        from src.live.settings import LiveSettings
        assert_deployment_eligible(report, reference_report_path=Path(mhs_horizon_diagnostic_report_path()))
        report_target = Path(mhs_horizon_diagnostic_report_path())
        artifact_root = report_target.parent / f"{report_target.stem}_artifacts"
        res = emit_deployment(report, request, artifact_root, artifact_key=LiveSettings().artifact_key)  # wiring: if getattr(args, "emit_deployment", False): assert_deployment_eligible(report, reference_report_path=Path(mhs_horizon_diagnostic_report_path())); emit_deployment(report, request, artifact_root, artifact_key=LiveSettings().artifact_key)
        _logger.info("[EVAL] emit_deployment strategy_digest=%s path=%s", res["strategy_digest"], res["params_path"])
        if getattr(args, "deploy_push", False):
            import subprocess

            try:
                subprocess.run(["git", "add", str(artifact_root / "strategy_params.json.enc"), str(artifact_root / "strategy_bootstrap.parquet.enc")], check=True)
                subprocess.run(["git", "commit", "-m", f'deploy: strategy {res["strategy_digest"]}'], check=True)
                subprocess.run(["git", "push"], check=True)
            except Exception as exc:
                _logger.error("[EVAL] deploy_push status=FAILED reason=%s", exc)
                _logger.info("manual: git add %s %s && git commit -m 'deploy: strategy %s' && git push", artifact_root / "strategy_params.json.enc", artifact_root / "strategy_bootstrap.parquet.enc", res["strategy_digest"])


def add_mhs_commands(portfolio_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the dev-only ``research run portfolio mhs-horizon-diagnostic`` subcommand."""
    from src.mhs.report.persist import emit_deployment as _emit_deployment_ref2  # noqa: F401
    from src.mhs.live_strategy import assert_deployment_eligible as _assert_ref2  # noqa: F401
    _ = _emit_deployment_ref2; _ = _assert_ref2
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
        "--execution-universe-size",
        type=int,
        default=_CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT,
        help=(
            "Number of top-liquidity symbols in the execution replay roster "
            "(breadth N); default 60 matches the registered cap_60_roster "
            "attestation and was adopted per ADR_20260823_MHS_KELLY_TWO_SIDED_SIZING. "
            "Sweep with care beyond 60: the flat-bps cost "
            "model has no market-impact term, so breadth gains are optimistic "
            "by construction -- require a stress-cost tier pass before "
            "adopting a larger value"
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
        "--peg-chase-diagnostic",
        action="store_true",
        default=False,
        help=(
            "Additionally replay slow_momentum/blend under OHLCV_PEG_CHASE_PROXY "
            "(submit-bar anchor) alongside the strict/stress pair -- opt-in only"
        ),
    )
    _liquidity_cost_model_action = mhs.add_argument("--liquidity-cost-model", choices=["flat", "corwin_schultz"], default="flat")
    _passive_timeout_action = mhs.add_argument("--passive-timeout-minutes", type=int, default=30)
    _passive_timeout_action.help = (
        "Execution window per intent in minutes; sweeps the passive chase "
        "window (with the IOC backstop at the deadline) without a code edit"
    )
    _liquidity_cost_model_action.help = (
        "Taker crossing cost model: flat (frozen default, fixed 3bps "
        "slippage, bit-identical) or corwin_schultz (per-symbol half-spread "
        "estimated from window high/lows with an EWMA smoothing; applied "
        "identically to every bound in the batch). Default keeps every "
        "existing replay byte-identical"
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
        "--no-committee-kelly-sizing",
        action="store_true",
        default=False,
        help=(
            "Opt-out: main-logic default is ON (with committee capital, which "
            "is on by default -- see --no-committee-capital), blending the "
            "committee total-exposure scale 50/50 with a train-only "
            "quarter-Kelly LCB overlay (f=0.25, z=1.0 one-SE shrinkage) "
            "instead of the flat vol-target scale alone. The Kelly term "
            "shares the resolved growth envelope's leverage_ceiling as its "
            "clip cap per ADR_20260823_MHS_KELLY_TWO_SIDED_SIZING; pass this flag to "
            "opt back out to the pure vol-target scale"
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
        "--committee-target-gross",
        type=float,
        default=None,
        help=(
            "Requires committee capital (on by default): rescales every "
            "committee decision row to an explicit gross, restoring the "
            "unit-gross invariant that the k=5 member average and the tranche "
            "mean otherwise dilute to ~0.53 (47%% idle cash). DEFAULT (flag "
            "omitted): the registered COMMITTEE_TARGET_GROSS=0.92, the "
            "largest replay-certified exposure inside the registered "
            "COMMITTEE_GROWTH_MAX_DRAWDOWN=0.25 budget per ADR_20260823_MHS_LEVERAGE_FRONTIER_SCAN; "
            "the I4 drawdown-budget gate blocks "
            "Research-GO if a replay breaches the budget. Pass "
            "--no-committee-target-gross to restore the diluted book "
            "(None). The old CLI 'capital-invariant cliff at gross ~0.9039-0.9071' "
            "was the ruin of a diagnostic reference instrument (OHLCV_STRICT_PROXY), "
            "not of the capital book, and is no longer a reason to avoid 0.92; "
            "a risk-budget policy value, never a fitted parameter"
        ),
    )
    mhs.add_argument(
        "--no-committee-target-gross",
        action="store_true",
        default=False,
        help=(
            "Keep the diluted committee book (committee_target_gross=None): "
            "the k=5 member average and tranche mean are left un-rescaled. "
            "Overrides the registered default exposure"
        ),
    )
    mhs.add_argument(
        "--no-committee-evidence-weighting",
        action="store_true",
        default=False,
        help=(
            "Main logic default is ON (requires committee capital, which is "
            "also on by default): weights the k=5 committee members by their "
            "TRAIN-ONLY realized proxy-return t-statistic instead of equal "
            "weights per ADR_20260823_MHS_CONSTANT_RISK_DEPLOYMENT; weights are "
            "non-negative, sum to 1, and fall back to exact equal weights "
            "when no member has positive train evidence; fitted strictly "
            "before each fold's train_end (top-level: before the frozen "
            "committee OOS start), never on evaluation data. Pass this flag to opt back out to equal-weighted members"
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
            "own signals per ADR_20260823_MHS_CONSTANT_RISK_DEPLOYMENT."
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
            "return over the last 15 decision rows per ADR_20260823_MHS_CONSTANT_RISK_DEPLOYMENT. "
            "Pass this flag to opt back out to "
            "the raw (tranche_count=1) committee book."
        ),
    )
    mhs.add_argument("--no-fill-mark-parity-gate", action="store_true")
    mhs.add_argument(
        "--no-exposure-scale-two-sided",
        action="store_true",
        default=False,
        help=(
            "Opt OUT of two-sided ex-ante vol targeting (main logic default is "
            "ON): the scale may lever UP above 1.0x when realized vol runs "
            "below target. Applies in pnl-vol-target-mode exante_target OR "
            "growth_budget; the upper bound is the resolved "
            "GrowthRiskEnvelope.leverage_ceiling (conservative/balanced 1.0, "
            "growth_moderate 1.5, growth 2.0), NOT PNL_VOL_TARGET_MAX_SCALE"
        ),
    )
    mhs.add_argument("--exposure-drawdown-brake",
        # argparse가 dest exposure_drawdown_brake를 플래그에서 유도한다.
        action="store_true",
        default=False,
        help=(
            "Opt-in causal equity-drawdown brake on the exposure scale: "
            "scale_t = base_t * clip(1 + k * underwater_{t-1}, floor, 1.0), "
            "where underwater_{t-1} is the replayed equity's drawdown vs its "
            "running peak measured BEFORE day t's own return (strictly causal; "
            "recovery restores full scale immediately). Requires --pnl-vol-target-mode "
            "constant_risk. Default False (byte-identical)"
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
        "--pnl-vol-target-mode", choices=["exante_target", "median_relative", "growth_budget", "constant_risk"], default="growth_budget",
        # choices mirror MhsDiagnosticRequest.pnl_vol_target_mode cli_param (declare-once).
        help=(
            "P&L vol-target mode. Main logic default is growth_budget: the "
            "target volatility is solved per-boundary (fold-leak-free) from "
            "the resolved --growth-envelope's registered drawdown budget, "
            "rather than the fixed PNL_TARGET_ANNUAL_VOL=0.20 constant "
            "exante_target uses. constant_risk deploys a constant realized "
            "risk (EWMA halflife 90d) without the Kelly blend "
            "(ADR_20260823_MHS_CONSTANT_RISK_DEPLOYMENT)"
        ),
    )
    mhs.add_argument(
        "--committee-member-set",
        choices=["risk_premia", "flow_momentum"],
        default=COMMITTEE_DEFAULT_MEMBER_SET,
        help=(
            "Registered committee axis set: flow_momentum (default, "
            "the certified k=5 book) or risk_premia (measured non-default -- "
            "full 3m replay breached the registered drawdown budget and added "
            "STRESS_SHARPE_NOT_POSITIVE folds, see ADR_20260820_MHS_COMPOUNDING_ALPHA_AXES). "
            "Requires --committee-capital (on by default)"
        ),
    )
    mhs.add_argument(
        "--no-funding-carry-sleeve",
        action="store_true",
        default=False,
        help=(
            "Disable the funding-carry sleeve (requires committee capital, "
            "on by default). The carry sleeve shorts the highest trailing "
            "funding and longs the lowest, complementing the committee book "
            "in low-dispersion years"
        ),
    )
    mhs.add_argument(
        "--funding-carry-weight",
        type=float,
        default=FUNDING_CARRY_SLEEVE_WEIGHT,
        help=(
            "Gross-budget share of the funding-carry sleeve in [0.0, 1.0); "
            "a registered risk-budget policy value on a measured 0.25-0.35 "
            "plateau, never a fitted parameter. Requires --committee-capital "
            "(on by default)"
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
    mhs.add_argument(
        "--growth-envelope",
        choices=sorted(GROWTH_RISK_ENVELOPES),
        default=_CLI_GROWTH_ENVELOPE_DEFAULT,
        help=(
            "Registered growth risk envelope: growth, balanced, or conservative "
            "per ADR_20260823_MHS_LEVERAGE_FRONTIER_SCAN. Selects the drawdown budget for the "
            "growth-optimal risk solver and the ex-ante vol-target cap"
        ),
    )
    mhs.add_argument(
        "--emit-deployment",
        action="store_true",
        default=False,
        help="Emit sealed strategy params + bootstrap for cloud deployment",
    )
    mhs.add_argument(
        "--deploy-push",
        action="store_true",
        default=False,
        help="After emit, git add/commit/push the sealed artifacts",
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
    mhs.add_argument(
        "--final-oos-2026h1",
        action="store_true",
        default=False,
        help=(
            "One-time, narrowly-scoped extension of the sealed evaluation window "
            "through 2026-06-30 for a genuine out-of-selection-window check "
            "(2026-08-25 user-authorized decision). Default keeps the existing "
            "2025-12-31 seal byte-identical; results under this flag must not "
            "feed back into further parameter tuning."
        ),
    )
    mhs.add_argument(
        "--committee-member-attribution",
        action="store_true",
        default=False,
        help=(
            "Opt-in: replay committee members individually for attribution "
            "reporting. Adds len(members) fork worker book replays; the "
            "attribution reports proxy_vs_ledger_rank_spearman (1h proxy vs "
            "3m ledger Sharpe rank correlation) but never feeds back into "
            "blend, weights, scales, or Research-GO"
        ),
    )
    mhs.set_defaults(handler=_run_mhs_horizon_diagnostic)
