"""Declarative CLI parameter metadata for MHS request fields (I3 declare-once).

``cli_param`` builds the field-metadata dict that drives both CLI argparse flag
generation by ``src.cli.dataclass_args.add_dataclass_arguments`` and request
validation of ``choices`` by ``src.mhs.validation``, so adding one MHS
execution option requires editing exactly one request field.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, cast

from src.core.data_policy import MHS_DATA_POLICY_DEFAULT
from src.core.params import (
    CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT,
    CLI_GROWTH_ENVELOPE_DEFAULT,
    COMMITTEE_DEFAULT_MEMBER_SET,
    COMMITTEE_MEMBER_SETS,
    COMMITTEE_TARGET_GROSS,
    COMMITTEE_TRANCHE_COUNT,
    FUNDING_CARRY_SLEEVE_WEIGHT,
    GROWTH_RISK_ENVELOPES,
)
from src.core.resources import MhsResourceMeasurement as MhsResourceMeasurement

if TYPE_CHECKING:
    import pandas as pd

    from src.engine.execution import StrategyExecutionReplayResult
    from src.mhs.evidence import (
        CostResponsePoint,
        PhaseDiagnosticResult,
        TailSensitivityResult,
    )


def cli_param(
    *,
    flag: str,
    help: str,
    choices: tuple[str, ...] | None = None,
    negate_flag: str | None = None,
    arg_type: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Declarative CLI presentation of one request field.

    ``flag`` is the single exposed option string: for a boolean field it is the
    switch that moves the field away from its default (``--x`` when the default
    is False, ``--no-x`` when it is True). ``negate_flag`` (value fields only)
    is an opt-out switch that sets the field to ``None``. ``choices`` is the
    closed value set used by both argparse and request validation.
    ``arg_type`` parses a value flag (``None`` = ``str``). Dependency rules are
    not metadata: they live in ``src.mhs.validation``.
    """
    return {
        "flag": flag,
        "help": help,
        "choices": choices,
        "negate_flag": negate_flag,
        "arg_type": arg_type,
    }


@dataclass(frozen=True, slots=True)
class MhsDiagnosticRequest:
    """One MHS research replay request; the single configuration type from CLI to pipeline.

    Defaults are the production configuration a no-argument CLI run executes,
    so programmatic callers and the CLI cannot drift. Dependent features are
    validated, never auto-normalized: turning committee capital off requires
    stating its dependents explicitly (or using the CLI opt-out, which
    resolves them). Three-minute OHLCV and settled funding are the only
    historical economic feeds; no flag can switch the accounting price.
    """

    start: str | pd.Timestamp | None = field(
        default=None,
        metadata=cli_param(flag="--start", help="Evaluation window start (ISO)."),
    )
    end: str | pd.Timestamp | None = field(
        default=None,
        metadata=cli_param(flag="--end", help="Evaluation window end (ISO)."),
    )
    partition: Literal["dev", "holdout", "all"] = "dev"
    data_root: str | None = None
    execution_timeframe: Literal["3m"] = field(
        default="3m",
        metadata=cli_param(
            flag="--execution-timeframe",
            help=(
                "OHLCV execution replay resolution; signal construction remains 1h. "
                "3m (2026-08 default) gives ~+27%% fill precision vs 5m and is "
                "collected natively from Binance"
            ),
            choices=("3m",),
        ),
    )
    execution_universe_size: int = field(
        default=CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT,
        metadata=cli_param(
            flag="--execution-universe-size",
            help=(
                "Number of top-liquidity symbols in the execution replay roster "
                "(breadth N); default 60 matches the registered cap_60_roster "
                "attestation and was adopted per ADR_20260823_MHS_KELLY_TWO_SIDED_SIZING. "
                "Sweep with care beyond 60: the flat-bps cost "
                "model has no market-impact term, so breadth gains are optimistic "
                "by construction -- require a stress-cost tier pass before "
                "adopting a larger value"
            ),
            arg_type=int,
        ),
    )
    max_rss_bytes: int | None = field(
        default=None,
        metadata=cli_param(
            flag="--max-rss-bytes",
            help=(
                "Optional process RSS budget in bytes; when unset the RAM guard "
                "auto-derives 85%% of total RAM. Exceeding the "
                "budget at a stage/window boundary fails closed with "
                "DataIntegrityError instead of OOM"
            ),
            arg_type=int,
        ),
    )
    log_run: bool = field(
        default=True,
        metadata=cli_param(
            flag="--no-log-run", help="Do not append a run-history record.",
        ),
    )
    touch_diagnostic: bool = field(
        default=False,
        metadata=cli_param(
            flag="--touch-diagnostic",
            help=(
                "Additionally replay every top-level book under OHLCV_TOUCH_PROXY "
                "alongside the strict/stress pair -- adds a second full window "
                "pass, opt-in only"
            ),
        ),
    )
    ladder_diagnostic: bool = field(
        default=False,
        metadata=cli_param(
            flag="--ladder-diagnostic",
            help=(
                "Additionally replay every top-level book under OHLCV_LADDERED_PROXY "
                "alongside the strict/stress pair -- adds a third full window pass "
                "with the escalating limit ladder, opt-in only"
            ),
        ),
    )
    peg_chase_diagnostic: bool = field(
        default=False,
        metadata=cli_param(
            flag="--peg-chase-diagnostic",
            help=(
                "Additionally replay every top-level book under OHLCV_PEG_CHASE_PROXY "
                "(submit-bar anchor) alongside the strict/stress pair -- opt-in only"
            ),
        ),
    )
    liquidity_cost_model: Literal["flat", "corwin_schultz"] = field(
        default="flat",
        metadata=cli_param(
            flag="--liquidity-cost-model",
            help=(
                "Taker crossing cost model: flat (frozen default, fixed 3bps "
                "slippage, bit-identical) or corwin_schultz (per-symbol half-spread "
                "estimated from window high/lows with an EWMA smoothing; applied "
                "identically to every bound in the batch). Default keeps every "
                "existing replay byte-identical"
            ),
            choices=("flat", "corwin_schultz"),
        ),
    )
    passive_timeout_minutes: int = field(
        default=30,
        metadata=cli_param(
            flag="--passive-timeout-minutes",
            help=(
                "Execution window per intent in minutes; sweeps the passive chase "
                "window (with the IOC backstop at the deadline) without a code edit"
            ),
            arg_type=int,
        ),
    )
    discovery_gate: bool = field(
        default=False,
        metadata=cli_param(
            flag="--discovery-gate",
            help=(
                "Run the discovery/qualification horizon-selection gate on the "
                "current panel for both sign families and record the outcome in "
                "the diagnostic report (opt-in; the result never changes contracts "
                "by itself)"
            ),
        ),
    )
    discovery_gate_adjusted_net_t: bool = field(
        default=False,
        metadata=cli_param(
            flag="--discovery-gate-adjusted-net-t",
            help=(
                "Opt-in: also compute a Bartlett/HAC-adjusted net_t diagnostic per "
                "discovery candidate (requires --discovery-gate; never changes "
                "admitted/selected_horizon)"
            ),
        ),
    )
    discovery_gate_regime_scaled_net_t: bool = field(
        default=False,
        metadata=cli_param(
            flag="--discovery-gate-regime-scaled-net-t",
            help=(
                "Opt-in: also compute a vol-regime cash-scale-adjusted net_t "
                "diagnostic per discovery candidate (approximate market-vol proxy, "
                "requires --discovery-gate; never changes admitted/selected_horizon)"
            ),
        ),
    )
    fold_safe_horizon_selection: bool = field(
        default=False,
        metadata=cli_param(
            flag="--fold-safe-horizon",
            help=(
                "Reselect the slow_momentum horizon per anchored fold from a "
                "leak-free discovery/qualification run confined to each fold's "
                "train data (opt-in; measured to be a safe no-op against the "
                "current admission floor)"
            ),
        ),
    )
    crash_regime_tilt_alpha: float | None = field(
        default=None,
        metadata=cli_param(
            flag="--crash-regime-tilt-alpha",
            help=(
                "Opt-in crash-regime directional tilt on slow_momentum: fraction "
                "(0.0, 1.0] of unit gross reallocated to a BTCUSDT-trend-scaled "
                "directional overlay (default None = disabled, byte-identical to "
                "the fully dollar-neutral book)"
            ),
            arg_type=float,
        ),
    )
    slow_book_mode: Literal["single_horizon", "horizon_ensemble"] = field(
        default="single_horizon",
        metadata=cli_param(
            flag="--slow-book-mode",
            help=(
                "Slow-book construction: single_horizon (frozen production chain) "
                "or horizon_ensemble (equal-weight average of every candidate "
                "horizon, no selection)"
            ),
            choices=("single_horizon", "horizon_ensemble"),
        ),
    )
    fast_book_mode: Literal["single_horizon", "horizon_ensemble"] = field(
        default="single_horizon",
        metadata=cli_param(
            flag="--fast-book-mode",
            help=(
                "Fast-book construction: single_horizon (frozen production chain) "
                "or horizon_ensemble (equal-weight average of every candidate "
                "horizon, no selection)"
            ),
            choices=("single_horizon", "horizon_ensemble"),
        ),
    )
    rebalance_filter: Literal["per_symbol_deadband", "portfolio_trigger"] = field(
        default="per_symbol_deadband",
        metadata=cli_param(
            flag="--rebalance-filter",
            help=(
                "Turnover gate on the decision targets: per_symbol_deadband "
                "(published baseline) or portfolio_trigger (invariant-preserving "
                "row hold gated before the gross scale)"
            ),
            choices=("per_symbol_deadband", "portfolio_trigger"),
        ),
    )
    beta_neutralize: bool = field(
        default=False,
        metadata=cli_param(
            flag="--beta-neutralize",
            help=(
                "Orthogonally project the slow book onto the causal rolling market "
                "beta (sum(w)==0 and sum(w*beta)==0 by construction; parameter-free "
                "replacement for the crash-regime tilt)"
            ),
        ),
    )
    ensemble_signal: Literal["raw", "vol_normalized"] = field(
        default="raw",
        metadata=cli_param(
            flag="--ensemble-signal",
            help=(
                "Signal family for the slow book: raw horizon log return (frozen "
                "production) or vol-normalized"
            ),
            choices=("raw", "vol_normalized"),
        ),
    )
    trend_efficiency_overlay: bool = field(
        default=False,
        metadata=cli_param(
            flag="--trend-efficiency-overlay",
            help=(
                "Opt-in exposure timing overlay on slow_momentum: scales gross "
                "exposure down in low-efficiency-ratio (choppy, momentum-hostile) "
                "regimes using the fast band's own horizon, composed with the "
                "existing regime cash scale (default False = byte-identical)"
            ),
        ),
    )
    pnl_vol_target: bool = field(
        default=True,
        metadata=cli_param(
            flag="--no-pnl-vol-target",
            help=(
                "Opt-out of the P&L vol-target layer: skip the multiplicative "
                "P&L-vol-target rescale between Pass 1 and Pass 2"
            ),
        ),
    )
    pnl_vol_target_mode: Literal["median_relative", "exante_target", "growth_budget", "constant_risk"] = field(
        default="growth_budget",
        metadata=cli_param(
            flag="--pnl-vol-target-mode",
            help=(
                "P&L vol-target mode. Main logic default is growth_budget: the "
                "target volatility is solved per-boundary (fold-leak-free) from "
                "the resolved --growth-envelope's registered drawdown budget, "
                "rather than the fixed PNL_TARGET_ANNUAL_VOL=0.20 constant "
                "exante_target uses. constant_risk deploys a constant realized "
                "risk (EWMA halflife 90d) without the Kelly blend "
                "(ADR_20260823_MHS_CONSTANT_RISK_DEPLOYMENT)"
            ),
            choices=("median_relative", "exante_target", "growth_budget", "constant_risk"),
        ),
    )
    trend_sleeve: bool = field(
        default=False,
        metadata=cli_param(
            flag="--trend-sleeve",
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
        ),
    )
    trend_sleeve_gross: float = field(
        default=0.0,
        metadata=cli_param(
            flag="--trend-sleeve-gross",
            help=(
                "Gross budget allocated to the directional trend sleeve, in "
                "[0.0, 1.0]; a risk-budget policy value, never a fitted parameter. "
                "Measured negative at 0.15/0.30 (CAPITAL_INVARIANT_BREACH) -- see "
                "--trend-sleeve's warning"
            ),
            arg_type=float,
        ),
    )
    multi_feature_book: bool = field(
        default=False,
        metadata=cli_param(
            flag="--multi-feature-book",
            help=(
                "Opt-in: build the feature-axis registry into dollar-neutral rank "
                "books on the 24h decision grid with a per-year coverage gate "
                "(fail-closed exclusion) and equal-risk combination; report each "
                "admitted feature's coverage and regime-split stability plus the "
                "combined net Sharpe per cost tier and feature-book breadth "
                "(diagnostic-only, never an admission input)"
            ),
        ),
    )
    committee_book: bool = field(
        default=False,
        metadata=cli_param(
            flag="--committee-book",
            help=(
                "Opt-in: measure the declared k=5 wealth committee -- build the "
                "committee members into dollar-neutral rank books, audit RAW source "
                "coverage before any fillna, recover sign-safe gross/turnover-cost "
                "panels, and run the purged expanding-train walk-forward reporting "
                "wealth metrics per cost tier (diagnostic-only, never a combiner or "
                "capital input)"
            ),
        ),
    )
    committee_kelly_sizing: bool = field(
        default=True,
        metadata=cli_param(
            flag="--no-committee-kelly-sizing",
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
        ),
    )
    committee_growth_diagnostic: bool = field(
        default=False,
        metadata=cli_param(
            flag="--committee-growth-diagnostic",
            help=(
                "Opt-in (requires --committee-book): report whether the committee's "
                "current exposure sits near its Monte-Carlo constrained growth-optimal "
                "point (block-bootstrap search over a discovery-window-only risk grid, "
                "reusing src.quant.risk.growth_sizing -- the same framework already "
                "used for xs_alpha); observational only, never feeds back into sizing "
                "or capital allocation"
            ),
        ),
    )
    committee_capital: bool = field(
        default=True,
        metadata=cli_param(
            flag="--no-committee-capital",
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
        ),
    )
    committee_member_set: Literal["risk_premia", "flow_momentum"] = field(
        default=cast(Literal["risk_premia", "flow_momentum"], COMMITTEE_DEFAULT_MEMBER_SET),
        metadata=cli_param(
            flag="--committee-member-set",
            help=(
                "Registered committee axis set: flow_momentum (default, "
                "the certified k=5 book) or risk_premia (measured non-default -- "
                "full 3m replay breached the registered drawdown budget and added "
                "STRESS_SHARPE_NOT_POSITIVE folds, see ADR_20260820_MHS_COMPOUNDING_ALPHA_AXES). "
                "Requires --committee-capital (on by default)"
            ),
            choices=tuple(sorted(COMMITTEE_MEMBER_SETS)),
        ),
    )
    committee_tranche_smoothing: bool = field(
        default=False,
        metadata=cli_param(
            flag="--committee-tranche-smoothing",
            help=(
                "Opt-in (requires committee capital, on by default): smooth the committee capital "
                "book with a 3-decision staggered tranche mean (effective 72h signal "
                "life) instead of fully repositioning every 24h -- the committee's "
                "shortest member lookback is 168h, so the 24h cadence oversamples its "
                "own signals per ADR_20260823_MHS_CONSTANT_RISK_DEPLOYMENT."
            ),
        ),
    )
    committee_regime_adaptive_tranche: bool = field(
        default=True,
        metadata=cli_param(
            flag="--no-committee-regime-adaptive-tranche",
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
        ),
    )
    committee_tranche_count: int = field(
        default=COMMITTEE_TRANCHE_COUNT,
        metadata=cli_param(
            flag="--committee-tranche-count",
            help=(
                "Committee tranche mean length in 24h decision rows (1..7). "
                "Non-default values require --committee-tranche-smoothing or the "
                "default regime-adaptive tranche."
            ),
            arg_type=int,
        ),
    )
    committee_target_gross: float | None = field(
        default=COMMITTEE_TARGET_GROSS,
        metadata=cli_param(
            flag="--committee-target-gross",
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
            negate_flag="--no-committee-target-gross",
            arg_type=float,
        ),
    )
    committee_evidence_weighting: bool = field(
        default=True,
        metadata=cli_param(
            flag="--no-committee-evidence-weighting",
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
        ),
    )
    funding_carry_sleeve: bool = field(
        default=True,
        metadata=cli_param(
            flag="--no-funding-carry-sleeve",
            help=(
                "Disable the funding-carry sleeve (requires committee capital, "
                "on by default). The carry sleeve shorts the highest trailing "
                "funding and longs the lowest, complementing the committee book "
                "in low-dispersion years"
            ),
        ),
    )
    funding_carry_weight: float = field(
        default=FUNDING_CARRY_SLEEVE_WEIGHT,
        metadata=cli_param(
            flag="--funding-carry-weight",
            help=(
                "Gross-budget share of the funding-carry sleeve in [0.0, 1.0); "
                "a registered risk-budget policy value on a measured 0.25-0.35 "
                "plateau, never a fitted parameter. Requires --committee-capital "
                "(on by default)"
            ),
            arg_type=float,
        ),
    )
    execution_coverage_gate: bool = field(
        default=False,
        metadata=cli_param(
            flag="--execution-coverage-gate",
            help=(
                "Opt-in pre-flight check: verify every funded symbol has "
                "execution_timeframe OHLCV cache coverage for [start, end] before "
                "the replay runs; fails closed with the missing/gapped symbol list "
                "instead of a late opaque MISSING_DATA termination count"
            ),
        ),
    )
    exposure_scale_two_sided: bool = field(
        default=True,
        metadata=cli_param(
            flag="--no-exposure-scale-two-sided",
            help=(
                "Opt OUT of two-sided ex-ante vol targeting (main logic default is "
                "ON): the scale may lever UP above 1.0x when realized vol runs "
                "below target. Applies in pnl-vol-target-mode exante_target OR "
                "growth_budget; the upper bound is the resolved "
                "GrowthRiskEnvelope.leverage_ceiling (conservative/balanced 1.0, "
                "growth_moderate 1.5, growth 2.0), NOT PNL_VOL_TARGET_MAX_SCALE"
            ),
        ),
    )
    exposure_drawdown_brake: bool = field(
        default=False,
        metadata=cli_param(
            flag="--exposure-drawdown-brake",
            help=(
                "Opt-in causal equity-drawdown brake on the exposure scale: "
                "scale_t = base_t * clip(1 + k * underwater_{t-1}, floor, 1.0), "
                "where underwater_{t-1} is the replayed equity's drawdown vs its "
                "running peak measured BEFORE day t's own return (strictly causal; "
                "recovery restores full scale immediately). Requires --pnl-vol-target-mode "
                "constant_risk. Default False (byte-identical)"
            ),
        ),
    )
    name_drift_trim: bool = field(
        default=False,
        metadata=cli_param(
            flag="--name-drift-trim",
            help=(
                "Opt-in execution-replay trim every NAME_DRIFT_TRIM_INTERVAL_HOURS "
                "after the first decision, cut back to NAME_DRIFT_TRIM_MAX_WEIGHT "
                "with an ordinary taker fill, research-only, default False (byte-identical)"
            ),
        ),
    )
    ram_guard: bool = field(
        default=True,
        metadata=cli_param(
            flag="--no-ram-guard",
            help=(
                "Disable the automatic RAM guard (85%% budget + system reserve checks); "
                "--max-rss-bytes still applies when set"
            ),
        ),
    )
    growth_envelope: str = field(
        default=CLI_GROWTH_ENVELOPE_DEFAULT,
        metadata=cli_param(
            flag="--growth-envelope",
            help=(
                "Registered growth risk envelope: growth, balanced, or conservative "
                "per ADR_20260823_MHS_LEVERAGE_FRONTIER_SCAN. Selects the drawdown budget for the "
                "growth-optimal risk solver and the ex-ante vol-target cap"
            ),
            choices=tuple(sorted(GROWTH_RISK_ENVELOPES)),
        ),
    )
    committee_member_attribution: bool = field(
        default=False,
        metadata=cli_param(
            flag="--committee-member-attribution",
            help=(
                "Opt-in: replay committee members individually for attribution "
                "reporting. Adds len(members) fork worker book replays; the "
                "attribution reports proxy_vs_ledger_rank_spearman (1h proxy vs "
                "3m ledger Sharpe rank correlation) but never feeds back into "
                "blend, weights, scales, or Research-GO"
            ),
        ),
    )
    final_oos_2026h1: bool = field(
        default=False,
        metadata=cli_param(
            flag="--final-oos-2026h1",
            help=(
                "One-time, narrowly-scoped extension of the sealed evaluation window "
                "through 2026-06-30 for a genuine out-of-selection-window check "
                "(2026-08-25 user-authorized decision). Default keeps the existing "
                "2025-12-31 seal byte-identical; results under this flag must not "
                "feed back into further parameter tuning."
            ),
        ),
    )
    forward_registration_digest: str | None = field(
        default=None,
        metadata=cli_param(
            flag="--forward-registration",
            help=(
                "Evaluate a pre-registered MHS procedure (digest) through a completed calendar quarter end; requires --end at that quarter end."
            ),
        ),
    )
    data_policy: Literal['legacy', 'zombie_mask_v1'] = field(
        default=MHS_DATA_POLICY_DEFAULT,
        metadata=cli_param(
            flag='--data-policy',
            help='Input-data contract for 1h panels (legacy keeps every bar; zombie_mask_v1 masks causally-detected delisted flat bars).',
            choices=('legacy', 'zombie_mask_v1'),
        ),
    )
    input_manifest_path: str | None = field(
        default=None, metadata=cli_param(flag='--input-manifest-path', help='Sealed MHS input manifest path.')
    )
    forward_execution_quality_dir: str | None = field(
        default=None,
        metadata=cli_param(flag='--forward-execution-quality-dir', help='Append-only live execution-quality directory.'),
    )
    forward_strategy_digest: str | None = field(
        default=None,
        metadata=cli_param(flag='--forward-strategy-digest', help='Frozen strategy digest expected in forward observations.'),
    )
    placebo_diagnostic: bool = field(
        default=False,
        metadata=cli_param(flag='--placebo-diagnostic', help='Opt-in report-only placebo: percentile of the blend naive Sharpe among 500 column-shuffled 48h fast-reversal books; not the committee null, never a Research-GO, DSR or deploy-gate input'),
    )
    phase_diagnostic: bool = field(
        default=False,
        metadata=cli_param(flag='--phase-diagnostic', help='Opt-in report-only phase robustness diagnostic (independently-run decision phases) on every top-level book; never a go/no-go input'),
    )
    signal_48h_diagnostic: bool = field(
        default=False,
        metadata=cli_param(flag='--signal-48h-diagnostic', help='Opt-in report-only 48h raw-return statistics: cross-sectional rank IC, date-clustered OLS slope, 48h realized-vol and efficiency-ratio means'),
    )
    bootstrap_ci_diagnostic: bool = field(
        default=False,
        metadata=cli_param(flag='--bootstrap-ci-diagnostic', help='Opt-in report-only stationary block-bootstrap 95%% CI of the blend mean hourly return; deployment-readiness and deploy-gate bootstraps stay mandatory'),
    )
    reference_books_diagnostic: bool = field(
        default=False,
        metadata=cli_param(flag='--reference-books-diagnostic', help='Opt-in report-only top-level replays of the standalone fast_reversal and slow_momentum books; their failures are reported but never block Research-GO (the blend and anchored folds are always replayed)'),
    )
    patient_reference_diagnostic: bool = field(
        default=False,
        metadata=cli_param(flag='--patient-reference-diagnostic', help='Opt-in report-only failure-isolated OHLCV_STRICT_PROXY patient-reference bound on every replayed top-level book; never a gate input'),
    )

    def __post_init__(self) -> None:
        from src.mhs.validation import validate_request
        validate_request(self)


@dataclass(frozen=True, slots=True)
class MhsBookFailure:
    """Typed, serializable book-level rejection of a strict replay error.

    ``stage`` names the failing replay stage, ``error_class`` is the exact
    exception class name, ``reason`` is a stable fail-closed code (one of the
    ``GO_REASON_*`` strings), and ``message`` carries the deterministic
    provenance. A failed book has no ledger/artifact reference and never
    fabricates metrics, deployment readiness, or Research-GO evidence.
    """

    stage: str
    error_class: str
    reason: str
    message: str


@dataclass(frozen=True, slots=True)
class MhsBookReport:
    """One top-level book replay report; ``phase`` is ``None`` unless the request set ``phase_diagnostic``."""

    name: str
    band: str
    horizon_hours: int
    step_hours: int
    tranche_count: int
    n_symbols: int
    phase: PhaseDiagnosticResult | None
    prescreen: dict[float, CostResponsePoint]
    tail: TailSensitivityResult
    primary: StrategyExecutionReplayResult | None
    stress: StrategyExecutionReplayResult | None
    primary_autocorr_sharpe: float | None
    primary_naive_sharpe: float | None
    primary_net_ann: float | None
    primary_geometric_cagr: float | None
    primary_max_drawdown: float | None
    primary_annualized_turnover: float | None
    stress_naive_sharpe: float | None
    terminal_censored_decisions: int = 0
    failure: MhsBookFailure | None = None
    reference_bound_failures: tuple[MhsBookFailure, ...] = ()
    touch: StrategyExecutionReplayResult | None = None
    touch_naive_sharpe: float | None = None
    ladder: StrategyExecutionReplayResult | None = None
    ladder_naive_sharpe: float | None = None
    peg_chase: StrategyExecutionReplayResult | None = None
    peg_chase_naive_sharpe: float | None = None
    peg_chase_fill_rate: float | None = None
    peg_chase_maker_share: float | None = None
    # Cost decomposition + min-notional diagnostics of the capital-carrying
    # primary replay (None when the book failed before a replay existed).
    notional_weighted_fee_bps: float | None = None
    notional_weighted_spread_bps: float | None = None
    notional_weighted_delay_bps: float | None = None
    min_notional_dropped_fraction: float | None = None
    patient_reference: StrategyExecutionReplayResult | None = None
    patient_reference_naive_sharpe: float | None = None
    pre_vol_target_reference: StrategyExecutionReplayResult | None = None
    pre_vol_target_reference_naive_sharpe: float | None = None
    executed_prescreen: dict[float, CostResponsePoint] | None = None
    executed_tail: TailSensitivityResult | None = None
    executed_prescreen_net_t: float | None = None
    primary_realized_shortfall_bps: float | None = None
    primary_notional_weighted_shortfall_bps: float | None = None
    stress_realized_shortfall_bps: float | None = None
    stress_notional_weighted_shortfall_bps: float | None = None
    primary_fill_count: int | None = None
    primary_unfilled_count: int | None = None
    primary_forced_exit_notional: float | None = None
    # I-SCALE-IS-DEPLOYED-OVERLAY: blend가 배치 확정한 노출 스케일 시계열.
    # constant_risk fold가 재적합하지 않고 이 값을 슬라이스해 재사용한다.
    exposure_scale: pd.Series | None = None
    # 연구-라이브 seam: 데드밴드 적용 후, exposure_scale 곱하기 전의 결정 격자
    # 목표비중. name=="blend"일 때만 채워진다.
    target_weights: pd.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class MhsResearchGoResult:
    """Machine-readable Research-GO gate decision built from the fold evidence.

    ``eligible`` is false unless every anchored fold passed and no policy gate
    is left unspecified; the exact blocking reasons are carried as stable codes.
    """

    eligible: bool
    reason_codes: tuple[str, ...]
    evaluated_folds: int
    folds_passed: int
    # Subset of ``reason_codes`` restricted to data-integrity failures; empty
    # when the only blocking reasons are alpha-quality or policy-registration.
    data_integrity_reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MhsFoldReport:
    """One independently flat anchored-fold replay over the blend book.

    ``strict``/``stress`` are ``None`` for an incomplete fold; ``failures``
    carries the stable reason codes that blocked this fold's evidence.
    """

    fold_index: int
    validation_start: str
    validation_end: str
    strict: StrategyExecutionReplayResult | None
    stress: StrategyExecutionReplayResult | None
    primary_valid: bool
    primary_autocorr_sharpe: float
    primary_naive_sharpe: float
    primary_net_ann: float
    primary_geometric_cagr: float
    primary_max_drawdown: float
    stress_naive_sharpe: float
    decision_intents: int
    termination_counts: dict[str, int]
    failures: tuple[str, ...]
    strict_elapsed_seconds: float
    stress_elapsed_seconds: float
    terminal_censored_decisions: int = 0
    slow_horizon_hours: int = 168
    slow_horizon_source: str = "frozen_default"
    fast_horizon_hours: int = 48
    fast_horizon_source: str = "frozen_default"
    funding_carry_lookback_hours: int | None = None
    funding_carry_sign: int | None = None
    funding_carry_source: str = "frozen_default"
    funding_carry_vs_slow_momentum_daily_corr: float | None = None
    book_structure: dict[str, float | str] | None = None
    regime_characterization: dict[str, float] | None = None
    # strict 원장 일간 수익률 std(ddof=1)*sqrt(365); 2행 미만이면 None.
    realized_annualized_vol: float | None = None


# Canonical definition moved to src.mhs.report.schema (P1); re-exported here
# so every existing `from ...contracts import MhsHorizonDiagnosticReport` keeps working.
from src.mhs.report.schema import (  # noqa: E402
    MhsHorizonDiagnosticReport as MhsHorizonDiagnosticReport,
)


class MhsOutputTier(StrEnum):
    """Persistence resolution for the MHS horizon diagnostic."""

    COMPACT = "compact"
    FULL = "full"
