"""Declarative CLI parameter metadata for MHS request fields (I3 declare-once).

``cli_param`` builds the field-metadata dict that drives both the CLI argparse
flag generation (``src.cli.dataclass_args.build_parser_from_dataclass``) and the
metadata-driven request validator (``src.application.research.mhs.validation``),
so adding one MHS execution option requires editing exactly one request field.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from src.mhs.params import _MHS_COMMITTEE_TARGET_GROSS_UNSET

if TYPE_CHECKING:
    import pandas as pd

    from src.mhs.discovery import DiscoveryQualificationResult
    from src.mhs.evaluation import (
        AnchoredPurgedFold,
        CostResponsePoint,
        DeploymentReadinessResult,
        PhaseDiagnosticResult,
        TailSensitivityResult,
    )
    from src.mhs.execution import StrategyExecutionReplayResult


def cli_param(
    *,
    flag: str,
    help: str,
    choices: tuple[str, ...] | None = None,
    bounds: tuple[float, float] | None = None,
    bounds_mode: str = "inclusive",
    requires: tuple[str, ...] = (),
    excludes: tuple[str, ...] = (),
    negate_flag: str | None = None,
    error_template: str | None = None,
) -> dict[str, Any]:
    """Build the declarative CLI metadata for one request field.

    ``flag`` is the argparse option string; ``help`` the help text; ``choices``
    the closed choice set; ``bounds`` an optional numeric range; ``requires``/
    ``excludes`` cross-field predicates; ``negate_flag`` an alternate flag that
    inverts a boolean; ``error_template`` the exact ``ValueError`` message (with
    ``{}`` placeholders) so validation messages are preserved verbatim.
    """
    return {
        "flag": flag,
        "help": help,
        "choices": choices,
        "bounds": bounds,
        "bounds_mode": bounds_mode,
        "requires": requires,
        "excludes": excludes,
        "negate_flag": negate_flag,
        "error_template": error_template,
    }


@dataclass(frozen=True, slots=True)
class MhsDiagnosticRequest:
    """Immutable request; the CLI carries ``--start``/``--end``/``--mark-mode``/``--no-log-run``.

    ``partition`` is forced to ``'dev'`` (a holdout request raises); ``data_root``
    allows tests to run against a synthetic market. ``mark_mode`` is a
    reproducibility parameter: ``cache_required`` builds the strict causal mark
    panel and fails closed, while ``cache_required_stale_carry`` permits a
    bounded 24-hour causal carry for diagnostic-only continuity. The latter is
    never a strict Research-GO source. ``ohlcv_close_fallback`` deliberately
    passes ``None`` for fixtures and explicit comparison runs only.

    ``execution_universe_size`` is the ENTRY rank threshold for the PIT
    top-volume execution roster, not the realized holdings count: hysteresis
    retains members past the entry rank, so realized holdings are approximately
    ``execution_universe_size * (1 + hysteresis effect)``, NOT
    ``execution_universe_size`` (see ``realized_execution_roster_size`` on
    ``MhsHorizonDiagnosticReport``).
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
    mark_mode: Literal["cache_required", "cache_required_stale_carry", "ohlcv_close_fallback"] = field(
        default="cache_required",
        metadata=cli_param(
            flag="--mark-mode",
            help="Mark-price valuation source.",
            choices=("cache_required", "cache_required_stale_carry", "ohlcv_close_fallback"),
        ),
    )
    execution_timeframe: Literal["1m", "3m", "5m"] = field(
        default="3m",
        metadata=cli_param(
            flag="--execution-timeframe",
            help="OHLCV execution replay resolution.",
            choices=("1m", "3m", "5m"),
        ),
    )
    execution_universe_size: int = 30
    max_rss_bytes: int | None = field(
        default=None,
        metadata=cli_param(flag="--max-rss-bytes", help="Optional process RSS budget in bytes."),
    )
    log_run: bool = field(
        default=True,
        metadata=cli_param(
            flag="--log-run", help="Append a run-history record.", negate_flag="--no-log-run",
        ),
    )
    touch_diagnostic: bool = field(
        default=False,
        metadata=cli_param(flag="--touch-diagnostic", help="Additionally replay under OHLCV_TOUCH_PROXY."),
    )
    ladder_diagnostic: bool = field(
        default=False,
        metadata=cli_param(flag="--ladder-diagnostic", help="Additionally replay under OHLCV_LADDERED_PROXY."),
    )
    discovery_gate: bool = field(
        default=False,
        metadata=cli_param(flag="--discovery-gate", help="Run the discovery/qualification horizon gate."),
    )
    discovery_gate_adjusted_net_t: bool = field(
        default=False,
        metadata=cli_param(flag="--discovery-gate-adjusted-net-t", help="Also compute adjusted net_t."),
    )
    discovery_gate_regime_scaled_net_t: bool = field(
        default=False,
        metadata=cli_param(flag="--discovery-gate-regime-scaled-net-t", help="Also compute regime-scaled net_t."),
    )
    fold_safe_horizon_selection: bool = field(
        default=False,
        metadata=cli_param(flag="--fold-safe-horizon", help="Reselect slow horizon per fold."),
    )
    crash_regime_tilt_alpha: float | None = field(
        default=None,
        metadata=cli_param(flag="--crash-regime-tilt-alpha", help="Crash-regime directional tilt fraction."),
    )
    slow_book_mode: Literal["single_horizon", "horizon_ensemble"] = field(
        default="single_horizon",
        metadata=cli_param(
            flag="--slow-book-mode",
            help="Slow-book construction mode.",
            choices=("single_horizon", "horizon_ensemble"),
        ),
    )
    fast_book_mode: Literal["single_horizon", "horizon_ensemble"] = field(
        default="single_horizon",
        metadata=cli_param(
            flag="--fast-book-mode",
            help="Fast-book construction mode.",
            choices=("single_horizon", "horizon_ensemble"),
        ),
    )
    rebalance_filter: Literal["per_symbol_deadband", "portfolio_trigger"] = field(
        default="per_symbol_deadband",
        metadata=cli_param(
            flag="--rebalance-filter",
            help="Turnover gate on decision targets.",
            choices=("per_symbol_deadband", "portfolio_trigger"),
        ),
    )
    beta_neutralize: bool = field(
        default=False,
        metadata=cli_param(flag="--beta-neutralize", help="Orthogonally project the slow book onto beta."),
    )
    ensemble_signal: Literal["raw", "vol_normalized"] = field(
        default="raw",
        metadata=cli_param(
            flag="--ensemble-signal",
            help="Signal family for the slow book.",
            choices=("raw", "vol_normalized"),
        ),
    )
    trend_efficiency_overlay: bool = field(
        default=False,
        metadata=cli_param(flag="--trend-efficiency-overlay", help="Scale gross by trend efficiency."),
    )
    pnl_vol_target: bool = field(
        default=True,
        metadata=cli_param(
            flag="--pnl-vol-target", help="Apply the P&L vol-target layer.", negate_flag="--no-pnl-vol-target",
        ),
    )
    pnl_vol_target_mode: Literal["median_relative", "exante_target"] = field(
        default="median_relative",
        metadata=cli_param(
            flag="--pnl-vol-target-mode",
            help="P&L vol-target mode.",
            choices=("median_relative", "exante_target"),
        ),
    )
    trend_sleeve: bool = field(
        default=False,
        metadata=cli_param(flag="--trend-sleeve", help="Measure an additive trend sleeve."),
    )
    trend_sleeve_gross: float = field(
        default=0.0,
        metadata=cli_param(flag="--trend-sleeve-gross", help="Gross budget for the trend sleeve."),
    )
    multi_feature_book: bool = field(
        default=False,
        metadata=cli_param(flag="--multi-feature-book", help="Build the feature-axis registry books."),
    )
    committee_book: bool = field(
        default=False,
        metadata=cli_param(flag="--committee-book", help="Measure the wealth committee."),
    )
    committee_kelly_sizing: bool = field(
        default=False,
        metadata=cli_param(flag="--committee-kelly-sizing", help="Blend kelly sizing with vol target."),
    )
    committee_growth_diagnostic: bool = field(
        default=False,
        metadata=cli_param(flag="--committee-growth-diagnostic", help="Report growth headroom."),
    )
    committee_capital: bool = field(
        default=False,
        metadata=cli_param(
            flag="--committee-capital", help="Build the committee capital book.", negate_flag="--no-committee-capital",
        ),
    )
    committee_tranche_smoothing: bool = field(
        default=False,
        metadata=cli_param(flag="--committee-tranche-smoothing", help="Smooth the committee book."),
    )
    committee_regime_adaptive_tranche: bool = field(
        default=False,
        metadata=cli_param(
            flag="--committee-regime-adaptive-tranche",
            help="Per-row adaptive tranche choice.",
            negate_flag="--no-committee-regime-adaptive-tranche",
        ),
    )
    committee_target_gross: float | None = field(
        default=_MHS_COMMITTEE_TARGET_GROSS_UNSET,  # type: ignore[assignment]
        metadata=cli_param(
            flag="--committee-target-gross",
            help="Committee book target gross exposure.",
            negate_flag="--no-committee-target-gross",
        ),
    )
    committee_evidence_weighting: bool = field(
        default=False,
        metadata=cli_param(flag="--committee-evidence-weighting", help="Weight members by train evidence."),
    )
    funding_carry_sleeve: bool = field(
        default=False,
        metadata=cli_param(
            flag="--funding-carry-sleeve", help="Short the highest funding.", negate_flag="--no-funding-carry-sleeve",
        ),
    )
    funding_carry_weight: float = field(
        default=0.0,
        metadata=cli_param(flag="--funding-carry-weight", help="Gross share of the carry sleeve."),
    )
    execution_coverage_gate: bool = field(
        default=False,
        metadata=cli_param(flag="--execution-coverage-gate", help="Pre-flight coverage check."),
    )
    fill_mark_parity_gate: bool = field(
        default=True,
        metadata=cli_param(
            flag="--fill-mark-parity-gate", help="Apply the fill/mark parity gate.", negate_flag="--no-fill-mark-parity-gate",
        ),
    )
    exposure_scale_two_sided: bool = field(
        default=False,
        metadata=cli_param(flag="--exposure-scale-two-sided", help="Allow lever-up above 1.0x."),
    )
    ram_guard: bool = field(
        default=True,
        metadata=cli_param(flag="--ram-guard", help="Enable the RAM guard.", negate_flag="--no-ram-guard"),
    )

    def __post_init__(self) -> None:
        from src.application.research.mhs.validation import validate_request
        validate_request(self, _MHS_COMMITTEE_TARGET_GROSS_UNSET)


@dataclass(frozen=True, slots=True)
class MhsBookFailure:
    """Typed, serializable book-level rejection of a strict replay error.

    ``stage`` names the failing replay stage, ``error_class`` is the exact
    exception class name, ``reason`` is a stable fail-closed code (one of the
    ``MHS_GO_REASON_*`` strings), and ``message`` carries the deterministic
    provenance. A failed book has no ledger/artifact reference and never
    fabricates metrics, deployment readiness, or Research-GO evidence.
    """

    stage: str
    error_class: str
    reason: str
    message: str


@dataclass(frozen=True, slots=True)
class MhsBookReport:
    name: str
    band: str
    horizon_hours: int
    step_hours: int
    tranche_count: int
    n_symbols: int
    phase: PhaseDiagnosticResult
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
    book_structure: dict[str, float] | None = None
    regime_characterization: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class MhsHorizonDiagnosticReport:
    feature: str
    status: str
    start: str
    end: str
    resolved_end: str
    partition: str
    execution_tiers_bps: tuple[float, ...]
    books: dict[str, MhsBookReport]
    blend: MhsBookReport | None
    blend_target_gross: float
    blend_cash_fraction: float
    eligible_symbols: int
    trials_attempted: int
    deflated_sharpe_ratio: float | None
    xs_rank_ic: dict[str, float]
    date_clustered_regression: dict[str, float]
    horizon_diagnostics: dict[str, float]
    bootstrap_ci: tuple[float, float] | None
    placebo_sharpe_percentile: float | None
    deployment_readiness: DeploymentReadinessResult
    synthetic_stress: dict[str, dict[str, Any]]
    participation_warnings: dict[str, float]
    termination_counts: dict[str, int]
    unsupported_assumptions: tuple[str, ...]
    anchored_folds: tuple[AnchoredPurgedFold, ...]
    folds: tuple[MhsFoldReport, ...]
    research_go: MhsResearchGoResult
    fill_source: str
    mark_source: str
    execution_timeframe: str
    execution_universe_size: int
    execution_symbols: tuple[str, ...]
    run_elapsed_seconds: float
    resource_measurements: tuple[MhsResourceMeasurement, ...] = ()
    discovery_qualification: dict[str, DiscoveryQualificationResult] | None = None
    realized_execution_roster_size: float | None = None
    full_history_yearly_net_t: dict[str, dict[int, float]] | None = None
    funding_carry_worst_year_corr: float | None = None
    trend_sleeve_diagnostic: dict[str, Any] | None = None
    multi_feature_diagnostic: dict[str, Any] | None = None
    committee_diagnostic: dict[str, Any] | None = None
    funding_dropped_symbols: dict[str, str] | None = None
    fold_blend_parity: dict[str, Any] | None = None
    fold_growth_concentration: dict[str, Any] | None = None
    fill_mark_parity: dict[str, Any] | None = None

    def to_payload(self) -> Any:
        from src.application.research.mhs.evaluation import _jsonable
        return _jsonable(dataclasses.asdict(self))


@dataclass(frozen=True, slots=True)
class MhsResourceMeasurement:
    """One ordered resource sample for a material diagnostic stage.

    ``elapsed_ms`` is the wall time since the previous recorded stage; ``rss_bytes``
    is the current process resident set size. Measurements are observational only
    and must never alter control flow, replay data, or the GO gate.
    """

    stage: str
    elapsed_ms: int
    rss_bytes: int
    grid_bars: int | None = None
    n_symbols: int | None = None
    fill_count: int | None = None
    window_start: str | None = None
    window_end: str | None = None
    active_symbols: int | None = None
    peak_rss_bytes: int | None = None


class MhsOutputTier(StrEnum):
    """Persistence resolution for the MHS horizon diagnostic."""

    COMPACT = "compact"
    FULL = "full"
