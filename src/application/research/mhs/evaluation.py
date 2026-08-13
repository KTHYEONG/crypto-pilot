"""MHS Phase 1 orchestrator: dev-only diagnostics and the Research-GO evidence path.

This module composes the frozen ``src.mhs`` primitives; no alpha, cost,
ranking, liquidity, funding, or inventory arithmetic is reimplemented here.
The target-weight ``cost_response_curve`` is pre-screen only, the
immediate-taker replay + simulated inventory ledger is the primary
Research-GO evidence (with a cost-stressed x3 bound and an informational
patient-passive reference), and every report separates the two.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import logging
import os
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from src.market_data.services.futures_collection import DataCollector
from src.mhs.evaluation import phase_diagnostic_metrics
from src.mhs.evaluation import AnchoredPurgedFold, DeploymentReadinessResult, autocorrelation_adjusted_sharpe, synthetic_stress_scenarios
from src.mhs.execution import ExecutionReplayWindow, simulated_inventory_ledger
from src.mhs.execution import replay_execution_windows
from src.mhs.panel import liquid_half_eligibility, load_base_panel
from src.research.evaluation.policy import resolve_evaluation_end

from src.common.config import FUTURES_DATA_DIR, funding_path
from src.common.errors import DataIntegrityError
from src.mhs.books import (
    equal_weight_book_ensemble,
    inverse_realized_vol_tilt,
    phase_tranche_book,
    portfolio_rebalance_trigger,
    rank_weight_book,
    renormalize_within_mask,
)
from src.mhs.contracts import MHS_DISCOVERY_START
from src.mhs.contracts import MHS_REGISTERED_POLICY_THRESHOLDS
from src.mhs.contracts import MHS_SEARCH_TRIALS_ATTEMPTED
from src.mhs.contracts import (
    MEASURED_EXECUTION_COST_TIERS_BPS,
    MHS_CRASH_REGIME_REFERENCE_SYMBOLS,
    PHASE_1_BOOK_BLEND_WEIGHTS,
    PHASE_1_BOOK_SPECS,
    BookSpec,
    ExecutionSpec,
)
from src.mhs.regime import (
    beta_neutralize_weights,
    causal_market_beta,
    crash_regime_tilt_weights,
)
from src.mhs.evaluation import book_evidence
from src.mhs.evaluation import deflated_sharpe_ratio
from src.mhs.evaluation import (
    CostResponsePoint,
    PhaseDiagnosticResult,
    TailSensitivityResult,
    compute_deployment_readiness,
    phase_1_anchored_purged_folds,
    required_cost_tiers,
)
from src.mhs.execution import SimulatedInventoryLedgerResult, StrategyExecutionReplayResult, bar_funding_panel, laddered_fill_schedule, mhs_ledger_pnl
from src.mhs.discovery import (
    DiscoveryQualificationResult,
    build_candidate_weights,
    fold_train_only_discovery_qualification,
    select_horizon_by_discovery_qualification,
)
from src.mhs.horizons import efficiency_ratio, horizon_log_return, realized_vol, vol_normalized_horizon_signal
from src.mhs.regime import trend_efficiency_scale
from src.research.evaluation.policy import HOLDOUT_CUTOFF
from src.research.technical_experts.trend_screen_catalog import DISCOVERY_END, QUALIFICATION_END
from src.market_data.storage.loaders import load_funding_rates

__all__ = ["simulated_inventory_ledger"]

MhsExecutionWindow = ExecutionReplayWindow

_logger = logging.getLogger("MhsHorizonDiagnostic")

MHS_DISCOVERY_GATE_TRANCHE_COUNT = 8
# SPREAD_AND_COST_X3 stress assumption: the realistic primary fill mechanic at
# 3x the default cost (same model, cost-shock robustness check).
MHS_STRESS_COST_MULTIPLIER = 3.0


def _stress_cost_execution_spec() -> ExecutionSpec:
    """SPREAD_AND_COST_X3: the same realistic fill mechanic at 3x cost."""
    base = ExecutionSpec()
    return ExecutionSpec(
        maker_fee_bps=base.maker_fee_bps * MHS_STRESS_COST_MULTIPLIER,
        taker_fee_bps=base.taker_fee_bps * MHS_STRESS_COST_MULTIPLIER,
        taker_slippage_bps=base.taker_slippage_bps * MHS_STRESS_COST_MULTIPLIER,
    )


MHS_DISCOVERY_REVERSAL_CANDIDATES: tuple[int, ...] = (24, 48, 72, 96, 120, 144, 168)
MHS_DISCOVERY_MOMENTUM_CANDIDATES: tuple[int, ...] = PHASE_1_BOOK_SPECS["slow_momentum"].band.horizons_hours
_MHS_FEATURE = "multi_horizon_market_state"
_PERIODS_PER_YEAR_1H = 365.0 * 24
_BOOTSTRAP_SEED = 20260807
_BOOTSTRAP_REPLICATES = 2000
_BOOTSTRAP_MEAN_BLOCK = 168

# Frozen strict-proxy Research-GO criterion (spec §3.3): the primary
# autocorrelation-adjusted Sharpe must be >= 0.6 for a candidate to pass.
MHS_GO_PRIMARY_SHARPE_FLOOR = 0.6

MHS_ARTIFACT_SCHEMA_VERSION = 1

# Unified artifact storage: every replay session is consolidated into exactly
# these category tables, partitioned by an explicit ``replay_id`` column.
MHS_ARTIFACT_CATEGORIES: tuple[str, ...] = (
    "fills",
    "units",
    "notional_weights",
    "ledger",
    "times",
)


class MhsOutputTier(StrEnum):
    """Persistence resolution for the MHS horizon diagnostic.

    ``COMPACT`` is the default: a git-committable daily-resampled ledger plus a
    stripped summary JSON. ``FULL`` persists the lossless per-fill audit tables
    under ``_full/`` (gitignored) for deep execution auditing.
    """

    COMPACT = "compact"
    FULL = "full"

# Signal-quality calibration (spec §3.2, ``signal_quality``).
# ``MHS_REBALANCE_MIN_NOTIONAL_DELTA`` is the turnover deadband cap: a per-symbol
# target-weight change smaller than 0.02 (2% of equity notional) is suppressed so
# the executor never churns the book on sub-threshold signal deltas.
MHS_REBALANCE_MIN_NOTIONAL_DELTA = 0.02
# The EMA smoothing span is one full horizon cycle in decision steps
# (``horizon_hours // step_hours``), the structural invariant that the smoothed
# signal cannot react faster than one horizon length.
MHS_SIGNAL_EMA_HORIZON_SPAN = 1.0


def _signal_ema_span(band_sign: int, horizon_hours: int, step_hours: int) -> int | None:
    """Whipsaw-suppressing EMA span, or None for a reversal band (sign=-1).

    ``_smooth_signal_ema`` exists to preserve trend polarity while removing the
    noise that drives negative return autocorrelation -- structurally backwards
    for a reversal signal, whose edge (if any) lives in that same short-term
    noise (docs/specs/mhs_fast_reversal_overlay_redesign.md §1.2).
    """
    if band_sign != 1:
        return None
    return max(1, round(horizon_hours / step_hours * MHS_SIGNAL_EMA_HORIZON_SPAN))
# Regime cash scaling: gross exposure is linearly reduced toward
# ``MHS_REGIME_CASH_SCALE_FLOOR`` when trailing realized vol exceeds its
# ``MHS_REGIME_CASH_MEDIAN_WINDOW_HOURS`` median, and never exceeds full exposure.
MHS_REGIME_CASH_SCALE_FLOOR = 0.5
MHS_REGIME_CASH_MEDIAN_WINDOW_HOURS = 720
# Reuses MHS_REGIME_CASH_SCALE_FLOOR's value: the standalone Pass-1 reference
# replay (_book_outcome, docs/specs/mhs_fast_reversal_overlay_redesign.md §2.1)
# forces flat once equity crosses this fraction of initial_equity, instead of
# hard-raising CAPITAL_INVARIANT_BREACH on a known negative-edge book.
MHS_REFERENCE_PASS_EQUITY_FLOOR = MHS_REGIME_CASH_SCALE_FLOOR
# Strategy-level P&L volatility targeting (momentum-crash mitigation): exposure
# is scaled by ``expanding_median(trailing_vol) / trailing_vol`` of the
# strategy's own daily P&L, clipped to ``[MHS_PNL_VOL_TARGET_SCALE_FLOOR, 1.0]``
# and never levered up. The 21-day window is deliberately shorter than the
# 30-day asset-vol regime window (fast-moving strategy shocks, not broad market
# regimes) and the 0.2 floor is deliberately lower than the 0.5 asset-vol floor
# (strategy-level crash protection must de-risk far more aggressively). Burn-in
# ``MHS_PNL_VOL_TARGET_BURN_IN_DAYS`` keeps the expanding-median target
# under-sampled (hence unscaled at exactly 1.0) until enough realized history.
MHS_PNL_VOL_TARGET_WINDOW_DAYS = 21
MHS_PNL_VOL_TARGET_SCALE_FLOOR = 0.2
MHS_PNL_VOL_TARGET_BURN_IN_DAYS = 90
# Anchored-fold panel warm-up bound (spec §3.1, ``memory_opt``): the fold panel
# is sliced to ``[validation_start - warmup, validation_end]`` instead of the
# full ``[train_start, validation_end]``. The warm-up covers the 720-bar
# liquidity-eligibility lookback plus the 168h slow horizon plus a one-day
# boundary buffer; features only ever see history at or before each decision.
MHS_FOLD_PANEL_WARMUP_HOURS = 720 + 168 + 24
# Execution-roster Schmitt-trigger band: a member entered at the top
# ``universe_size`` trailing-volume rank is kept until its rank exceeds
# ``universe_size * MHS_EXECUTION_ROSTER_EXIT_MULTIPLIER``. The factor is an
# engineering default (2x the entry threshold, the usual Schmitt-trigger
# convention), not a measured constant; the fold/full-period replay is the
# empirical check (docs/specs/mhs_roster_hysteresis_vol_tilt.md).
MHS_EXECUTION_ROSTER_EXIT_MULTIPLIER = 2.0
# Portfolio-level rebalance trigger (docs/specs/mhs_alpha_engine.md §1): the
# one-way tracking-error threshold, as a fraction of unit gross, at which the
# entire previously adopted target row is replaced instead of the invariant-
# breaking per-symbol deadband. Engineering default in the same class as
# MHS_EXECUTION_ROSTER_EXIT_MULTIPLIER -- NOT a measured constant and never
# selected from the full-period table in the alpha-engine spec (§6.1 forbids
# table-driven selection).
MHS_REBALANCE_TRACKING_ERROR_THRESHOLD = 0.20
# Causal market-beta window for beta neutralization (docs/specs/mhs_alpha_engine.md
# §4): a 720-bar trailing window with a 360-bar minimum sample matches the
# liquidity-eligibility lookback. Engineering defaults, not measured constants.
MHS_CAUSAL_BETA_LOOKBACK_BARS = 720
MHS_CAUSAL_BETA_MIN_PERIODS = 360

MHS_GO_REASON_INCOMPLETE_FOLD = "INCOMPLETE_ANCHORED_FOLD"
MHS_GO_REASON_INVALID_PRIMARY = "INVALID_PRIMARY_LEDGER"
MHS_GO_REASON_NONFINITE_EQUITY = "NONFINITE_EQUITY"
MHS_GO_REASON_EXECUTION_GAP = "RELEVANT_EXECUTION_DATA_GAP"
MHS_GO_REASON_PRIMARY_SHARPE = "PRIMARY_AUTOCORR_SHARPE_BELOW_0_6"
MHS_GO_REASON_STRESS_SHARPE = "STRESS_SHARPE_NOT_POSITIVE"
MHS_GO_REASON_CAPITAL_BREACH = "CAPITAL_INVARIANT_BREACH"
MHS_GO_REASON_UNSPECIFIED_POLICY = "UNSPECIFIED_POLICY"
MHS_GO_REASON_RESOURCE_BREACH = "RESOURCE_BUDGET_BREACH"


# Unrecoverable source gap exclusions (Binance REST API & Vision archives have >4h gaps):
# SLPUSDT, CTKUSDT, LITUSDT, AERGOUSDT, PUMPUSDT, CVXUSDT, CVCUSDT
MHS_SOURCE_GAP_EXCLUDED_SYMBOLS = frozenset({
    "SLPUSDT", "CTKUSDT", "LITUSDT", "AERGOUSDT", "PUMPUSDT", "CVXUSDT", "CVCUSDT",
})


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

    start: str | pd.Timestamp | None = None
    end: str | pd.Timestamp | None = None
    partition: Literal["dev", "holdout", "all"] = "dev"
    data_root: str | None = None
    mark_mode: Literal["cache_required", "cache_required_stale_carry", "ohlcv_close_fallback"] = "cache_required"
    execution_timeframe: Literal["1m", "5m"] = "5m"
    execution_universe_size: int = 30
    max_rss_bytes: int | None = None
    log_run: bool = True
    touch_diagnostic: bool = False
    ladder_diagnostic: bool = False
    discovery_gate: bool = False
    fold_safe_horizon_selection: bool = False
    crash_regime_tilt_alpha: float | None = None
    slow_book_mode: Literal["single_horizon", "horizon_ensemble"] = "single_horizon"
    rebalance_filter: Literal["per_symbol_deadband", "portfolio_trigger"] = "per_symbol_deadband"
    beta_neutralize: bool = False
    ensemble_signal: Literal["raw", "vol_normalized"] = "raw"
    trend_efficiency_overlay: bool = False
    pnl_vol_target: bool = True

    def __post_init__(self) -> None:
        if self.partition not in ("dev", "holdout", "all"):
            raise ValueError(f"unknown partition '{self.partition}'")
        if self.mark_mode not in ("cache_required", "cache_required_stale_carry", "ohlcv_close_fallback"):
            raise ValueError(f"unknown mark_mode '{self.mark_mode}'")
        if self.execution_timeframe not in ("1m", "5m"):
            raise ValueError(f"unknown execution_timeframe '{self.execution_timeframe}'")
        if self.execution_universe_size < 8:
            raise ValueError("execution_universe_size must be >= 8")
        if self.max_rss_bytes is not None and self.max_rss_bytes <= 0:
            raise ValueError("max_rss_bytes must be > 0")
        if self.crash_regime_tilt_alpha is not None and not (0.0 < self.crash_regime_tilt_alpha <= 1.0):
            raise ValueError(
                f"crash_regime_tilt_alpha must be in (0.0, 1.0] when set, got {self.crash_regime_tilt_alpha}"
            )
        if self.slow_book_mode not in ("single_horizon", "horizon_ensemble"):
            raise ValueError(f"unknown slow_book_mode '{self.slow_book_mode}'")
        if self.rebalance_filter not in ("per_symbol_deadband", "portfolio_trigger"):
            raise ValueError(f"unknown rebalance_filter '{self.rebalance_filter}'")
        if not isinstance(self.beta_neutralize, bool):
            raise ValueError("beta_neutralize must be a bool")
        if self.ensemble_signal not in ("raw", "vol_normalized"):
            raise ValueError(f"unknown ensemble_signal '{self.ensemble_signal}'")
        if not isinstance(self.trend_efficiency_overlay, bool):
            raise ValueError("trend_efficiency_overlay must be a bool")
        if not isinstance(self.pnl_vol_target, bool):
            raise ValueError("pnl_vol_target must be a bool")


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

    def to_payload(self) -> Any:
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


def _current_rss_bytes() -> int:
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001
        return -1


def _assert_cache_required_ledger_valid(
    name: str,
    primary: StrategyExecutionReplayResult,
) -> None:
    """Fail closed when a cache-required strict primary ledger is invalid.

    ``cache_required_stale_carry`` and ``ohlcv_close_fallback`` are explicit
    diagnostic modes and never call this gate.
    """
    if not primary.ledger.primary_valid:
        raise DataIntegrityError(
            f"cache_required strict primary ledger invalid for {name}: "
            f"{', '.join(primary.ledger.invalid_reasons)}"
        )


def _classify_execution_failure(exc: BaseException) -> str:
    """Stable fail-closed reason code for an expected strict replay error.

    The classifier is intentionally conservative: any unrecognized message maps
    to ``INVALID_PRIMARY_LEDGER`` so an unanticipated integrity error is never
    relabeled as a policy or Sharpe gate. Resource-budget breaches keep their
    own stable code so a fixed-RSS regression can be proven end to end.
    """
    message = str(exc).lower()
    if "pre-trade equity" in message or "capital" in message or "equity must be" in message:
        return MHS_GO_REASON_CAPITAL_BREACH
    if "rss budget" in message:
        return MHS_GO_REASON_RESOURCE_BREACH
    if "finite" in message:
        return MHS_GO_REASON_NONFINITE_EQUITY
    if "gap" in message or "mark" in message or "missing" in message:
        return MHS_GO_REASON_EXECUTION_GAP
    return MHS_GO_REASON_INVALID_PRIMARY


def _assert_execution_rss_budget(
    stage: str,
    budget: int | None,
    completed_windows: int,
) -> None:
    """Deterministic fail-closed provenance for a configured RSS budget.

    A positive ``budget`` exceeded at a window boundary raises
    ``DataIntegrityError`` carrying the stage, observed RSS, configured budget,
    and completed window count; the default ``None`` applies no artificial cap.
    """
    if budget is None:
        return
    observed = _current_rss_bytes()
    if observed > budget:
        raise DataIntegrityError(
            "execution RSS budget exceeded at window boundary: "
            f"stage={stage} observed_rss={observed} "
            f"budget={budget} completed_windows={completed_windows}"
        )


class _StageRecorder:
    """Collects ordered ``MhsResourceMeasurement`` records and emits ``[SYS]`` logs."""

    def __init__(self, log_run: bool) -> None:
        self._records: list[MhsResourceMeasurement] = []
        self._log_run = log_run
        self._last = time.perf_counter()
        self._peak_rss = -1

    @property
    def records(self) -> tuple[MhsResourceMeasurement, ...]:
        return tuple(self._records)

    def record(
        self,
        stage: str,
        grid_bars: int | None = None,
        n_symbols: int | None = None,
        fill_count: int | None = None,
        window_start: str | None = None,
        window_end: str | None = None,
        active_symbols: int | None = None,
    ) -> None:
        now = time.perf_counter()
        elapsed_ms = int((now - self._last) * 1000)
        self._last = now
        rss = _current_rss_bytes()
        self._peak_rss = max(self._peak_rss, rss)
        self._records.append(
            MhsResourceMeasurement(
                stage=stage,
                elapsed_ms=elapsed_ms,
                rss_bytes=rss,
                grid_bars=grid_bars,
                n_symbols=n_symbols,
                fill_count=fill_count,
                window_start=window_start,
                window_end=window_end,
                active_symbols=active_symbols,
                peak_rss_bytes=self._peak_rss,
            )
        )
        if self._log_run:
            _logger.info(
                "[SYS] stage=%s rss=%d elapsed_ms=%d",
                stage, rss, elapsed_ms,
            )

    def absorb(self, records: tuple[MhsResourceMeasurement, ...]) -> None:
        """Merge frozen records (e.g. from a book subprocess) into this recorder.

        Appends in arrival order, folds the peak-RSS tracking, and resets the
        elapsed baseline so the next ``record`` measures from the absorption
        point rather than from the last absorbed stage.
        """
        if not records:
            return
        self._records.extend(records)
        self._peak_rss = max(self._peak_rss, max(r.peak_rss_bytes or 0 for r in records))
        self._last = time.perf_counter()


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return _jsonable(value.to_dict())
    if isinstance(value, pd.DataFrame):
        if not value.index.is_unique:
            return _jsonable(value.to_dict(orient="records"))
        return {
            str(k): _jsonable(v)
            for k, v in value.astype(object).to_dict(orient="index").items()
        }
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _load_funding_series(symbols: list[str]) -> dict[str, pd.Series]:
    series: dict[str, pd.Series] = {}
    for sym in symbols:
        path = funding_path(sym)
        if not path.exists():
            continue
        try:
            rates = load_funding_rates(str(path))
        except Exception as exc:  # noqa: BLE001
            _logger.warning("[MHS] funding load failed symbol=%s error=%s", sym, exc)
            continue
        if len(rates):
            series[sym] = rates
    return series


def _pit_execution_mask(
    quote_volume: pd.DataFrame,
    eligible: pd.DataFrame,
    universe_size: int,
) -> pd.DataFrame:
    """Select the PIT top-volume execution roster with entry/exit hysteresis.

    ``universe_size`` is the ENTRY rank threshold only: a symbol enters by
    reaching the top ``universe_size`` trailing-volume rank, and once a member
    it is kept until its rank falls outside
    ``universe_size * MHS_EXECUTION_ROSTER_EXIT_MULTIPLIER`` (a Schmitt-trigger
    band). Because hysteresis retains members that have slipped past the entry
    threshold, the realized number of holdings is approximately
    ``universe_size * (1 + hysteresis effect)``, NOT ``universe_size`` (measured
    ~41.9 vs a declared 30) -- the true mean per-row True count is exposed as
    ``MhsHorizonDiagnosticReport.realized_execution_roster_size``. This prevents
    rank-boundary flicker from forcing a full entry/exit trade on every decision
    when the signal itself has not changed. The exit multiplier is an
    engineering default, not a measured constant; the fold/full-period replay is
    the empirical check (docs/specs/mhs_roster_hysteresis_vol_tilt.md).
    """
    exit_size = universe_size * MHS_EXECUTION_ROSTER_EXIT_MULTIPLIER
    trailing = quote_volume.rolling(720, min_periods=720).mean()
    ranked = trailing.where(eligible).rank(axis=1, ascending=False, method="first")
    enter = ranked.le(universe_size).fillna(False).to_numpy()
    keep = ranked.le(exit_size).fillna(False).to_numpy()
    held = np.zeros(enter.shape[1], dtype=bool)
    out = np.zeros_like(enter, dtype=bool)
    for i in range(len(enter)):
        held = enter[i] | (held & keep[i])
        out[i] = held
    return pd.DataFrame(out, index=quote_volume.index, columns=quote_volume.columns)


def _load_minute_frames(
    root: str, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp,
    timeframe: Literal["1m", "5m"],
) -> dict[str, pd.DataFrame]:
    """Load minute OHLCV frames for ``symbols`` over ``[start, end]``.

    Test-only convenience wrapper (MHS_PERF_OPT_004); the production replay
    path uses the per-symbol full-period cache ``_get_symbol_minute_frame``.
    """
    return dict(_load_minute_frames_cached(root, tuple(sorted(symbols)), start, end, timeframe))


_DATA_COLLECTOR: DataCollector | None = None


def _data_collector() -> DataCollector:
    """Lazily-instantiated shared mark-price collector.

    ``_iter_mhs_execution_windows`` previously constructed one ``DataCollector``
    per 31-day window; a module-level singleton pays the collector's
    construction cost once per diagnostic run instead of once per window
    (spec O5).  ``load_mark_price_panel`` resolves ``_mark_price_path``
    dynamically at call time, so test monkeypatches keep working.
    """
    global _DATA_COLLECTOR
    if _DATA_COLLECTOR is None:
        _DATA_COLLECTOR = DataCollector()
    return _DATA_COLLECTOR


@lru_cache(maxsize=8)
def _load_minute_frames_cached(
    root: str, symbols: tuple[str, ...], start: pd.Timestamp, end: pd.Timestamp,
    timeframe: Literal["1m", "5m"],
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    start_ms = int(start.value // 1_000_000)
    end_ms = int(end.value // 1_000_000)
    for sym in symbols:
        path = os.path.join(root, timeframe, f"{sym}.parquet")
        if not os.path.exists(path):
            continue
        table = pq.read_table(
            path,
            columns=["timestamp", "high", "low", "close"],
            filters=[[("timestamp", ">=", start_ms), ("timestamp", "<=", end_ms)]],
        )
        idx = pd.to_datetime(table.column("timestamp").to_numpy(), unit="ms", utc=True)
        frame = pd.DataFrame(
            {
                c: table.column(c).to_numpy().astype("float64")
                for c in ("high", "low", "close")
            },
            index=idx,
        )
        frame = frame[(frame.index >= start) & (frame.index <= end)]
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        if not frame.empty:
            frames[sym] = frame
    return frames


@lru_cache(maxsize=512)
def _get_symbol_minute_frame(
    root: str, symbol: str, timeframe: Literal["1m", "5m"],
) -> pd.DataFrame:
    """Full-period minute OHLCV frame for one symbol, cached for the process.

    Replaces the window-keyed LRU cache (0% hit rate in production because every
    31-day execution window has a distinct ``(start, end)`` key).  Reading each
    symbol's full Parquet once and slicing per window turns ~5,300 window reads
    into ~255 symbol reads.  The returned frame is immutable; callers must not
    mutate it.
    """
    path = os.path.join(root, timeframe, f"{symbol}.parquet")
    if not os.path.exists(path):
        raise DataIntegrityError(f"minute OHLCV parquet missing: {path}")
    table = pq.read_table(path, columns=["timestamp", "high", "low", "close"])
    idx = pd.to_datetime(table.column("timestamp").to_numpy(), unit="ms", utc=True)
    frame = pd.DataFrame(
        {
            c: table.column(c).to_numpy().astype("float64")
            for c in ("high", "low", "close")
        },
        index=idx,
    )
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def _build_window_frames(
    symbol_frames: dict[str, pd.DataFrame],
    roster: list[str],
    grid_start: pd.Timestamp,
    grid_end: pd.Timestamp,
    minute_grid: pd.DatetimeIndex,
    timeframe: Literal["1m", "5m"],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Slice per-symbol full-period frames onto a window minute grid.

    Identical output to the old ``_load_minute_frames`` + ``_align_minute_frames``
    path (same slicing, same reindex, same column order) but reads each symbol's
    frame from the in-memory full-period cache instead of re-reading Parquet.
    Returns ``None`` when no roster symbol has usable data.
    """
    if not symbol_frames:
        return None
    if grid_start >= grid_end:
        return None
    sliced: dict[str, pd.DataFrame] = {}
    for s in sorted(roster):
        full = symbol_frames.get(s)
        if full is None or full.empty:
            continue
        frame = full.loc[(full.index >= grid_start) & (full.index <= grid_end)]
        if not frame.empty:
            sliced[s] = frame
    if not sliced:
        return None
    highs = pd.DataFrame({s: f["high"] for s, f in sliced.items()}).reindex(minute_grid)
    lows = pd.DataFrame({s: f["low"] for s, f in sliced.items()}).reindex(minute_grid)
    closes = pd.DataFrame({s: f["close"] for s, f in sliced.items()}).reindex(minute_grid)
    return highs, lows, closes


def _align_minute_frames(
    frames: dict[str, pd.DataFrame], timeframe: Literal["1m", "5m"],
    start: pd.Timestamp, end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    if not frames:
        return None
    if start >= end:
        return None
    # The requested evaluation grid is the replay grid. A late listing is kept
    # as NaN on that grid and never trims the global start, so the replay
    # horizon is never shortened by the union of first-observed timestamps.
    grid = pd.date_range(
        start, end,
        freq={"1m": "1min", "5m": "5min"}[timeframe],
        tz="UTC",
    )
    highs = pd.DataFrame({s: f["high"] for s, f in frames.items()}).reindex(grid)
    lows = pd.DataFrame({s: f["low"] for s, f in frames.items()}).reindex(grid)
    closes = pd.DataFrame({s: f["close"] for s, f in frames.items()}).reindex(grid)
    return highs, lows, closes


def _smooth_signal_ema(signal: pd.DataFrame, span_steps: int) -> pd.DataFrame:
    """Apply an exponential moving average to a step-grid signal.

    The EMA is the spec's ``Autocorr Smoothing`` (§3.2): it removes the
    high-frequency noise that drives negative return autocorrelation (whipsaw)
    while preserving the trend polarity. ``span_steps`` is one full horizon
    cycle in decision steps; ``adjust=False`` so the span is the constant
    half-life ``span - 1`` and the filtered series is fully causal.
    """
    if span_steps < 1:
        raise ValueError(f"span_steps must be >= 1, got {span_steps}")
    return signal.ewm(span=span_steps, adjust=False).mean()


def _apply_rebalance_deadband(target: pd.DataFrame, min_delta: float) -> pd.DataFrame:
    """Suppress per-symbol rebalances smaller than the turnover deadband cap.

    A target-weight change ``|w_t - w_held|`` below ``min_delta`` (2% notional)
    carries the last decided (held) target forward instead of retrading, so the
    executor never churns on sub-threshold signal deltas; the hold is stateful,
    so a slow drift cannot creep through one small step at a time. The first
    observation is always a decision, NaN targets remain NaN (a delisting is
    never silently re-expressed), and a held NaN resets the deadband so a
    re-listed symbol trades from its own first finite target.
    """
    if min_delta < 0:
        raise ValueError(f"min_delta must be >= 0, got {min_delta}")
    if target.empty:
        return target.copy()
    values = target.to_numpy(dtype="float64")
    out = values.copy()
    held = out[0].copy()
    finite_values = np.isfinite(values)
    for i in range(1, len(values)):
        carry = (np.abs(values[i] - held) < min_delta) & finite_values[i] & np.isfinite(held)
        out[i] = np.where(carry, held, values[i])
        held = out[i]
    return pd.DataFrame(out, index=target.index, columns=target.columns).fillna(0.0)


def _trend_efficiency_overlay_scale(
    log_close: pd.DataFrame,
    execution_mask: pd.DataFrame,
    fast_horizon_hours: int,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Execution-roster mean efficiency_ratio at the fast band's horizon, timed
    into a slow_momentum exposure scale (docs/specs/mhs_fast_reversal_overlay_redesign.md §2.3).
    """
    mean_er = efficiency_ratio(log_close, fast_horizon_hours).where(execution_mask).reindex(target_index).mean(axis=1)
    return trend_efficiency_scale(mean_er)


def _regime_cash_scale(
    vol_mean: pd.Series,
    median_window_hours: int = MHS_REGIME_CASH_MEDIAN_WINDOW_HOURS,
    floor: float = MHS_REGIME_CASH_SCALE_FLOOR,
) -> pd.Series:
    """Per-decision gross-exposure scale that raises cash in high-vol regimes.

    Exposure is ``median(vol) / vol`` clipped to ``[floor, 1.0]``: a calm regime
    keeps full gross, a high-vol regime scales toward the cash floor, and a
    flat/insufficient-history window carries full exposure (never 0/0). This is
    the spec's ``Dynamic Band Weighting`` (§3.2) expressed as cash weighting.
    """
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"floor must be in (0, 1], got {floor}")
    if median_window_hours < 1:
        raise ValueError(f"median_window_hours must be >= 1, got {median_window_hours}")
    if vol_mean.empty:
        return pd.Series(1.0, index=vol_mean.index)
    median = vol_mean.rolling(
        median_window_hours, min_periods=min(48, median_window_hours),
    ).median()
    scale = median.div(vol_mean.clip(lower=1e-12))
    scale = scale.clip(lower=floor, upper=1.0)
    return scale.fillna(1.0)


def _pnl_vol_target_scale(
    reference_daily_returns: pd.Series,
    window_days: int = MHS_PNL_VOL_TARGET_WINDOW_DAYS,
    floor: float = MHS_PNL_VOL_TARGET_SCALE_FLOOR,
) -> pd.Series:
    """Strategy-own-P&L realized-vol targeting scale (Barroso & Santa-Clara).

    ``scale_t = clip(expanding_median(trailing_vol)_{t-1} / trailing_vol_{t-1},
    floor, 1.0)``: the strategy de-risks when its own daily P&L becomes more
    volatile than its historical median (momentum-crash protection), never
    levering up and never scaling on an under-sampled estimate. Causality is
    strict: both the trailing-vol window AND the expanding-median target are
    ``shift(1)`` before use (two independent shifts, not one combined), so
    ``scale_t`` depends only on realized returns strictly before ``t``.
    """
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"floor must be in (0, 1], got {floor}")
    if window_days < 1:
        raise ValueError(f"window_days must be >= 1, got {window_days}")
    if reference_daily_returns.empty:
        return pd.Series(1.0, index=reference_daily_returns.index)
    trailing_vol = reference_daily_returns.rolling(
        window_days, min_periods=max(5, window_days // 2),
    ).std().shift(1)
    expanding_target = trailing_vol.expanding(
        min_periods=MHS_PNL_VOL_TARGET_BURN_IN_DAYS,
    ).median().shift(1)
    scale = expanding_target.div(trailing_vol.where(trailing_vol > 0))
    return scale.clip(lower=floor, upper=1.0).fillna(1.0)


def _book_weights(
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    spec: BookSpec,
    step_grid: pd.DatetimeIndex,
    ema_span: int | None = None,
) -> pd.DataFrame:
    # NOTE: the live book intentionally stays on raw horizon_log_return for
    # both bands. vol_normalized_horizon_signal (sign==1) was measured to
    # improve the discovery-gate prescreen (+37% worst-year net_t) but broke
    # CAPITAL_INVARIANT_BREACH on the full 2021-2025 realistic-execution
    # replay (cumulative 2021-2024 drawdown deeper than raw momentum's,
    # exhausting equity before 2025's gains could recover it -- confirmed via
    # direct re-measurement, not merely the prescreen). It stays wired into
    # discovery.py's diagnostic-only sign=1 candidate scoring, matching the
    # existing OHLCV_STRICT_PROXY precedent (demoted to reference once the
    # full replay disproved it as primary) -- see
    # docs/specs/mhs_momentum_vol_normalization.md follow-up.
    sig = horizon_log_return(log_close, spec.horizon_hours)
    if ema_span is not None:
        sig = _smooth_signal_ema(sig, ema_span)
    sig_step = sig.reindex(step_grid)
    el_step = eligible.reindex(step_grid)
    weights = rank_weight_book(sig_step, el_step, spec.band.sign, spec.min_symbols)
    return phase_tranche_book(weights, spec.tranche_count())


def _slow_book_execution_weights(
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    execution_mask: pd.DataFrame,
    spec: BookSpec,
    step_grid: pd.DatetimeIndex,
    mode: Literal["single_horizon", "horizon_ensemble"],
    signal_kind: Literal["raw", "vol_normalized"],
    ema_span: int | None,
) -> pd.DataFrame:
    """Shared slow-book builder for the top-level diagnostic and fold replay.

    ``mode='single_horizon'`` reproduces the frozen production chain
    byte-identically (``horizon_log_return`` -> EMA -> ``rank_weight_book`` ->
    ``phase_tranche_book`` -> ``inverse_realized_vol_tilt`` ->
    ``renormalize_within_mask``). ``mode='horizon_ensemble'`` runs that same
    chain once per candidate horizon in ``spec.band.horizons_hours`` and
    combines the execution books with ``equal_weight_book_ensemble``, removing
    the discovery argmax from the capital path (RC-2). Each horizon's book is
    built on the same ``step_grid`` so the combination is a plain row-wise mean;
    each horizon's intermediates are released before the next is built (bounded
    RSS on the 43k-bar, 450-symbol panel).
    """
    if mode not in ("single_horizon", "horizon_ensemble"):
        raise ValueError(f"unknown mode '{mode}'")
    if signal_kind not in ("raw", "vol_normalized"):
        raise ValueError(f"unknown signal_kind '{signal_kind}'")
    mask = execution_mask.reindex(step_grid).fillna(False)
    horizons = (
        (spec.horizon_hours,) if mode == "single_horizon" else spec.band.horizons_hours
    )
    books: dict[int, pd.DataFrame] = {}
    for h in horizons:
        sig = (
            vol_normalized_horizon_signal(log_close, h)
            if signal_kind == "vol_normalized"
            else horizon_log_return(log_close, h)
        )
        if ema_span is not None:
            sig = _smooth_signal_ema(sig, ema_span)
        sig_step = sig.reindex(step_grid)
        weights = rank_weight_book(
            sig_step, eligible.reindex(step_grid), spec.band.sign, spec.min_symbols,
        )
        book = phase_tranche_book(weights, h // spec.step_hours)
        tilted = inverse_realized_vol_tilt(
            book, realized_vol(log_close, h).reindex(step_grid),
        )
        books[h] = renormalize_within_mask(tilted, mask, spec.min_symbols)
        del sig, sig_step, weights, book, tilted
        gc.collect()
    if mode == "single_horizon":
        return books[spec.horizon_hours]
    return equal_weight_book_ensemble(books)


def _phase_diagnostics(
    log_close: pd.DataFrame,
    eligible: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    spec: BookSpec,
) -> PhaseDiagnosticResult:
    phase_nets: dict[int, pd.Series] = {}
    signal = horizon_log_return(log_close, spec.horizon_hours)
    for offset in range(spec.step_hours):
        phase_grid = grid_1h[offset :: spec.step_hours]
        sig = signal.reindex(phase_grid)
        el = eligible.reindex(phase_grid)
        weights = rank_weight_book(sig, el, spec.band.sign, spec.min_symbols)
        weights_1h = weights.reindex(grid_1h, method="ffill").fillna(0.0)
        net, _turnover = mhs_ledger_pnl(weights_1h, opens, bar_funding, 8.0)
        phase_nets[offset] = net
        # Each phase is independent.  Explicitly dropping its full-grid
        # target/ledger intermediates prevents allocator high-water growth on
        # multi-year, hundreds-of-symbol diagnostics.
        del sig, el, weights, weights_1h, _turnover
        gc.collect()
    return phase_diagnostic_metrics(phase_nets, _PERIODS_PER_YEAR_1H)


def _xs_rank_ic(
    signal: pd.DataFrame, opens: pd.DataFrame, forward_bars: int,
) -> dict[str, float]:
    """Cross-sectional rank IC of ``signal`` on a tradable forward window.

    The forward return is built internally as
    ``opens.pct_change(forward_bars).shift(-(forward_bars + 1))`` so the
    measured window starts at ``open_{t+1}`` (the first bar a decision made on
    close_t can enter) and never overlaps the signal's own lookback
    (docs/specs/mhs_alpha_engine.md §3, RC-3). The trailing-return convention
    that previously reported a contaminated +0.0957 (t=113) where the tradable
    statistic is -0.0278 (t=-33) is removed: no caller may pass a pre-shifted
    return panel, the shift is owned by this function.
    """
    if forward_bars < 1:
        raise ValueError(f"forward_bars must be >= 1, got {forward_bars}")
    fwd = opens.pct_change(forward_bars).shift(-(forward_bars + 1))
    common_index = signal.index.intersection(fwd.index)
    common_columns = signal.columns.intersection(fwd.columns)
    if common_index.empty or common_columns.empty:
        return {}
    signal_common = signal.loc[common_index, common_columns]
    fwd_common = fwd.loc[common_index, common_columns]
    valid = signal_common.notna() & fwd_common.notna()
    signal_rank = signal_common.where(valid).rank(axis=1)
    fwd_rank = fwd_common.where(valid).rank(axis=1)
    signal_centered = signal_rank.sub(signal_rank.mean(axis=1), axis=0)
    fwd_centered = fwd_rank.sub(fwd_rank.mean(axis=1), axis=0)
    denominator = np.sqrt(
        signal_centered.pow(2).sum(axis=1) * fwd_centered.pow(2).sum(axis=1),
    )
    correlations = (
        (signal_centered * fwd_centered).sum(axis=1) / denominator
    ).where(valid.sum(axis=1).ge(5) & denominator.gt(0.0)).dropna()
    if correlations.empty:
        return {}
    series = correlations.astype("float64")
    n_dates = len(series)
    mean_ic = float(series.mean())
    sd = float(series.std(ddof=1)) if n_dates > 1 else 0.0
    t_stat = mean_ic / (sd / np.sqrt(n_dates)) if sd > 0 else float("nan")
    return {
        "n_dates": n_dates, "mean_ic": mean_ic, "t_stat": t_stat,
        "forward_bars": forward_bars,
    }


def _date_clustered_ols(
    opens: pd.DataFrame, past: pd.DataFrame, forward_bars: int,
) -> dict[str, float]:
    """Pooled panel regression of a tradable forward return on ``past``.

    Same causality fix as ``_xs_rank_ic`` (RC-3): the dependent variable is
    built internally from ``opens`` with the ``shift(-(forward_bars + 1))``
    convention, so the regression never regresses a return window that lies
    inside its own predictor's lookback. Standard errors are date-clustered.
    """
    if forward_bars < 1:
        raise ValueError(f"forward_bars must be >= 1, got {forward_bars}")
    fwd = opens.pct_change(forward_bars).shift(-(forward_bars + 1))
    common_index = past.index.intersection(fwd.index)
    common_columns = past.columns.intersection(fwd.columns)
    if common_index.empty or common_columns.empty:
        return {
            "n": 0, "n_dates": 0, "past_beta": float("nan"),
            "past_t": float("nan"), "forward_bars": forward_bars,
        }
    x = past.loc[common_index, common_columns].to_numpy(dtype="float64", copy=False)
    y = fwd.loc[common_index, common_columns].to_numpy(dtype="float64", copy=False)
    valid = np.isfinite(x) & np.isfinite(y)
    n = int(valid.sum())
    if n < 10:
        return {
            "n": n, "n_dates": 0, "past_beta": float("nan"),
            "past_t": float("nan"), "forward_bars": forward_bars,
        }
    x_valid = np.where(valid, x, 0.0)
    y_valid = np.where(valid, y, 0.0)
    sum_x = float(x_valid.sum())
    sum_y = float(y_valid.sum())
    xtx = np.array([[n, sum_x], [sum_x, float(np.square(x_valid).sum())]])
    xty = np.array([sum_y, float((x_valid * y_valid).sum())])
    inv_xtx = np.linalg.inv(xtx)
    beta = inv_xtx @ xty
    residual = np.where(valid, y - beta[0] - beta[1] * x, 0.0)
    daily_scores = pd.DataFrame(
        {"intercept": residual.sum(axis=1), "slope": (x_valid * residual).sum(axis=1)},
        index=common_index,
    ).resample("1D").sum()
    scores = daily_scores.to_numpy(dtype="float64", copy=False)
    meat = scores.T @ scores
    cov = inv_xtx @ meat @ inv_xtx
    se = np.sqrt(np.diag(cov))
    t_beta = beta[1] / se[1] if se[1] > 0 else float("nan")
    return {
        "n": n, "n_dates": len(daily_scores), "past_beta": float(beta[1]),
        "past_t": float(t_beta), "forward_bars": forward_bars,
    }


def _block_bootstrap_replicate_mean(
    arr: np.ndarray, n: int, p_block: float, rng: np.random.Generator,
) -> float:
    """Mean of one block-bootstrap replicate (scalar fallback path).

    Mirrors the original geometric block composition: block starts are uniform,
    block lengths grow while ``rng.random() > p_block``, blocks are truncated at
    the array end and again to the remaining sample length.  Only used for the
    degenerate ``mean_block <= 0`` configuration and for the astronomically
    rare vectorized shortfall, where a replicate's drawn blocks did not reach
    length ``n``.
    """
    blocks: list[float] = []
    while len(blocks) < n:
        start = int(rng.integers(0, n))
        length = 1
        while length < n and rng.random() > p_block:
            length += 1
        length = min(length, n - len(blocks))
        blocks.extend(arr[start : start + length].tolist())
    return float(np.mean(blocks[:n]))

def _bootstrap_ci(net: pd.Series, n_replicates: int, mean_block: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = net.to_numpy(dtype="float64")
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        m = float(arr[0])
        return m, m
    p_block = 1.0 / mean_block if mean_block > 0 else 0.0
    if p_block <= 0.0:
        means = np.empty(n_replicates, dtype=np.float64)
        for r in range(n_replicates):
            means[r] = _block_bootstrap_replicate_mean(arr, n, p_block, rng)
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    # Vectorized block bootstrap: block lengths are ``geometric(p_block)`` --
    # the same length law as the scalar ``while`` loop -- and block starts are
    # uniform.  A 6x block-count safety margin makes running short effectively
    # impossible; any shortfall still falls back to the scalar replicate path.
    max_blocks = min(n, int(np.ceil(n * 6.0 / mean_block)) + 16)
    means = np.empty(n_replicates, dtype=np.float64)
    chunk = 128
    for r0 in range(0, n_replicates, chunk):
        r1 = min(r0 + chunk, n_replicates)
        k = r1 - r0
        lengths = rng.geometric(p_block, size=(k, max_blocks))
        starts = rng.integers(0, n, size=(k, max_blocks))
        ends = np.cumsum(lengths, axis=1)
        short = ends[:, -1] < n
        for r in np.flatnonzero(short).tolist():
            means[r0 + r] = _block_bootstrap_replicate_mean(arr, n, p_block, rng)
        valid = ~short
        if valid.any():
            ends_trunc = np.minimum(ends, n)
            used = ends_trunc - np.concatenate(
                [np.zeros((k, 1), dtype=np.int64), ends_trunc[:, :-1]], axis=1,
            )
            u = used[valid].ravel()
            s = starts[valid].ravel()
            keep = u > 0
            u = u[keep]
            s = s[keep]
            block_start = np.cumsum(u) - u
            offsets = np.arange(int(u.sum()), dtype=np.int64) - np.repeat(block_start, u)
            arr_idx = (np.repeat(s, u) + offsets) % n
            sample = arr[arr_idx].reshape(int(valid.sum()), n)
            means[r0 + np.flatnonzero(valid)] = sample.mean(axis=1)

    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _placebo_sharpe_percentile(
    signal: pd.DataFrame,
    eligible: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    spec: BookSpec,
    observed_sharpe: float,
    n_placebos: int,
    seed: int,
) -> float | None:
    rng = np.random.default_rng(seed)
    ranks: list[float] = []
    cols = list(signal.columns)
    n_cols = len(cols)
    sig_step = signal.reindex(grid_1h)
    el_step = eligible.reindex(grid_1h)
    # The frozen ledger raises ``DataIntegrityError`` unless weights, opens, and
    # funding share an identical index and column set; preserve that contract
    # instead of silently aligning via ``reindex``.
    if not opens.index.equals(grid_1h) or not bar_funding.index.equals(grid_1h):
        raise DataIntegrityError("opens and bar_funding must share the placebo grid index")
    opens_arr = opens[cols].to_numpy(dtype="float64")
    funding_arr = bar_funding[cols].to_numpy(dtype="float64")

    # The placebo shuffle relabels the signal/eligible columns without moving
    # their values, so ``rank_weight_book`` on any shuffled copy returns the
    # identical weight matrix; only the price/funding columns are genuinely
    # permuted relative to those weights.  The whole weight pipeline is
    # therefore computed once as a 2D float64 matrix instead of re-materializing
    # pandas DataFrames inside the 500-step loop (spec §3, Optimization 1).
    weights = rank_weight_book(sig_step, el_step, spec.band.sign, spec.min_symbols)
    weights = phase_tranche_book(weights, spec.tranche_count())
    w_arr = weights.reindex(grid_1h).ffill().fillna(0.0).to_numpy(dtype="float64")

    n_rows = opens_arr.shape[0]
    lag = 1 + 1  # ``mhs_ledger_pnl`` uses ``execution_delay_bars=1``.
    lagged = np.zeros_like(w_arr)
    if lag < n_rows:
        lagged[lag:] = w_arr[: n_rows - lag]

    o2o = np.zeros_like(opens_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        o2o[1:] = opens_arr[1:] / opens_arr[:-1] - 1.0

    prev_lagged = np.zeros_like(lagged)
    prev_lagged[1:] = lagged[:-1]
    turnover = np.abs(lagged - prev_lagged).sum(axis=1)
    half = 8.0 / 2.0 * 1e-4
    cost_rate = half + half
    nonfinite = ~np.isfinite(o2o) | ~np.isfinite(funding_arr)
    safe_o2o = np.where(np.isfinite(o2o), o2o, 0.0)
    safe_funding = np.where(np.isfinite(funding_arr), funding_arr, 0.0)
    active = lagged != 0.0

    for _p in range(n_placebos):
        perm = rng.permutation(n_cols)
        # A shuffled placebo can pair a non-zero weight with a symbol outside
        # its lifecycle; such a placebo is invalid, not evidence that the
        # production ledger should relax its active-cell guard.
        if (active & nonfinite[:, perm]).any():
            continue
        book_return = (lagged * safe_o2o[:, perm]).sum(axis=1)
        funding_charge = (lagged * safe_funding[:, perm]).sum(axis=1)
        net_returns = book_return - turnover * cost_rate - funding_charge
        if np.any(net_returns <= -1.0):
            continue
        equity = 10000.0 * np.cumprod(1.0 + net_returns)
        net = equity[1:] / equity[:-1] - 1.0
        if len(net) <= 1:
            continue
        sd = float(np.std(net, ddof=1))
        if sd > 0:
            ranks.append(float(np.mean(net) / sd * np.sqrt(_PERIODS_PER_YEAR_1H)))
    if not ranks:
        return None
    return float(np.mean([1.0 if observed_sharpe >= r else 0.0 for r in ranks]))


def _load_symbol_quote_volume(
    root: str,
    symbol: str,
    timeframe: Literal["1m", "5m"],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series | None:
    """Load one symbol's ``quote_vol`` over ``[start, end]`` on demand.

    Reads only the ``timestamp``/``quote_vol`` columns so participation metrics
    never retain a wide quote-volume panel alongside the minute OHLCV frames.
    Returns ``None`` when the symbol has no data (the same absent-data behavior
    as a symbol missing from the historical wide panel).
    """
    path = os.path.join(root, timeframe, f"{symbol}.parquet")
    if not os.path.exists(path):
        return None
    start_ms = int(start.value // 1_000_000)
    end_ms = int(end.value // 1_000_000)
    table = pq.read_table(
        path,
        columns=["timestamp", "quote_vol"],
        filters=[[("timestamp", ">=", start_ms), ("timestamp", "<=", end_ms)]],
    )
    idx = pd.to_datetime(table.column("timestamp").to_numpy(), unit="ms", utc=True)
    series = pd.Series(
        table.column("quote_vol").to_numpy().astype("float64"), index=idx,
    )
    series = series[(series.index >= start) & (series.index <= end)]
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        return None
    return series


def _participation_warnings(
    replay: StrategyExecutionReplayResult,
    root: str,
    timeframe: Literal["1m", "5m"],
    symbols: list[str],
    minute_grid: pd.DatetimeIndex,
) -> dict[str, float]:
    if replay.simulated_fills.empty:
        return {}
    fills = replay.simulated_fills
    notional = float((fills["quantity_delta"].abs() * fills["fill_price"]).sum())
    fills_by_symbol: dict[str, pd.DataFrame] = {}
    for _sym, group in fills.groupby("symbol"):
        fills_by_symbol[str(_sym)] = group
    daily_volume = 0.0
    window_totals: dict[str, float] = {"1m": 0.0, "30m": 0.0}
    window_minutes = (("1m", 1), ("30m", 30))
    for sym in symbols:
        series = _load_symbol_quote_volume(
            root, sym, timeframe, minute_grid[0], minute_grid[-1],
        )
        if series is None:
            continue
        daily_volume += float(series.sum())
        group = fills_by_symbol.get(sym)
        if group is None:
            continue
        # Locate each fill's ``[t, t+window]`` inclusive span with two
        # ``searchsorted`` lookups (instead of an ``iterrows``/``.loc`` slice)
        # and sum the small bounded window with ``np.add.reduce``, which is
        # bit-identical to pandas ``Series.sum()`` over the same labels.
        vol = series.to_numpy(dtype="float64")
        idx = series.index.to_numpy(dtype="datetime64[ns]")
        t_arr = group["timestamp"].to_numpy(dtype="datetime64[ns]")
        start_pos = np.searchsorted(idx, t_arr, side="left")
        in_idx = (start_pos < len(idx)) & (idx[start_pos] == t_arr)
        valid_pos = start_pos[in_idx]
        for window_label, minutes in window_minutes:
            ends = t_arr[in_idx] + np.timedelta64(minutes, "m")
            end_pos = np.searchsorted(idx, ends, side="right") - 1
            # Accumulate directly into the running total in fill order -- the
            # baseline's flat ``window_totals += window_sum`` chain -- so the
            # float addition sequence is bit-identical to the iterrows baseline.
            for p, e in zip(valid_pos, end_pos, strict=True):
                window_totals[window_label] += float(np.add.reduce(vol[p : e + 1]))
    warnings: dict[str, float] = {}
    for window_label, _minutes in window_minutes:
        total_volume = window_totals[window_label]
        warnings[f"fill_notional_to_{window_label}_quote_volume"] = (
            notional / total_volume if total_volume > 0 else float("nan")
        )
    warnings["daily_trade_notional_to_daily_quote_volume"] = (
        notional / daily_volume if daily_volume > 0 else float("nan")
    )
    return warnings


def _daily_autocorr_sharpe(ledger: SimulatedInventoryLedgerResult) -> float:
    if ledger.equity.empty:
        return float("nan")
    daily = ledger.equity.resample("1D").last().dropna()
    if len(daily) < 9:
        return float("nan")
    return autocorrelation_adjusted_sharpe(daily.pct_change().dropna(), 365, 7)


def _hourly_ledger_series(
    equity: pd.Series, fill_turnover: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Resample a native execution-timeframe ledger to the annualization grid.

    The ``_PERIODS_PER_YEAR_1H`` annualization constant describes hourly bars,
    but the replay ledgers run on ``request.execution_timeframe`` (5m default),
    so every headline metric derived from them must first be resampled to 1h
    (the ``equity_1h`` pattern already present in ``_run_post_diag_deploy``).
    Turnover is a per-bar traded-notional fraction, so hourly aggregation is a
    sum (not a last-value sample, unlike equity). An already-hourly input passes
    through unchanged.
    """
    equity_1h = equity.resample("1h").last().dropna()
    net_returns_1h = equity_1h.pct_change().dropna()
    turnover_1h = (
        fill_turnover.resample("1h").sum()
        .reindex(net_returns_1h.index)
        .fillna(0.0)
    )
    return equity_1h, net_returns_1h, turnover_1h


def _naive_sharpe(ledger: SimulatedInventoryLedgerResult) -> float:
    net = ledger.equity.resample("1h").last().dropna().pct_change().dropna()
    if len(net) < 2:
        return float("nan")
    sd = float(net.std(ddof=1))
    if sd <= 0:
        return float("inf") if float(net.mean()) > 0 else float("-inf")
    return float(net.mean() / sd * np.sqrt(_PERIODS_PER_YEAR_1H))


def _validate_ladder_schedule_contract() -> None:
    """Runtime guard for the frozen ladder schedule contract (spec §1.4).

    Runs once per ``--ladder-diagnostic`` book pass, before the expensive
    windowed replay: a single tranche must reproduce the strict single-fill
    schedule and the ladder's ``qty_fraction`` values must conserve notional.
    """
    one = laddered_fill_schedule(
        100.0, 1, np.array([101.0]), np.array([101.0, 101.0]),
        1, ExecutionSpec(), True,
    )
    assert one == [(1, 101.0, ExecutionSpec().one_way_taker_bps(), 1.0)]
    ladder = laddered_fill_schedule(
        100.0, 1, np.array([101.0] * 4), np.array([101.0] * 5),
        4, ExecutionSpec(), True,
    )
    assert abs(sum(f[3] for f in ladder) - 1.0) < 1e-12


def _mean_ann(series: pd.Series, periods_per_year: float) -> float:
    return float(series.mean()) * periods_per_year if len(series) else float("nan")


def _geometric_cagr(equity: pd.Series) -> float:
    if equity.empty or float(equity.iloc[0]) <= 0 or float(equity.iloc[-1]) <= 0:
        return float("nan")
    n = len(equity)
    return float((equity.iloc[-1] / equity.iloc[0]) ** (_PERIODS_PER_YEAR_1H / n) - 1.0)


def _mdd(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    return float((equity / running_max - 1.0).min())


def _truncate_replayable_decisions(
    target_weights: pd.DataFrame,
    signal_available_at: pd.DatetimeIndex,
    execution_grid: pd.DatetimeIndex,
    spec: ExecutionSpec,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, int]:
    """Censor terminal-window decisions that can never be executed on the grid.

    A decision is retained only when its first post-signal submit bar exists and
    the exact strict ``passive_timeout_minutes`` bar exists on the execution
    grid. The strict boundary is applied to both strict and immediate-taker
    outputs so they cover the same decision population. Dropped rows are
    terminal telemetry, never ``MISSING_DATA``: the returned count records them
    so the report can distinguish censored terminal decisions from real source
    gaps. Retained targets are byte-for-byte unchanged.
    """
    if len(target_weights) != len(signal_available_at):
        raise DataIntegrityError("signal_available_at must align with target_weights")
    grid_ns = np.asarray(execution_grid, dtype="datetime64[ns]").astype("int64")
    n_grid = len(grid_ns)
    if n_grid == 0:
        return target_weights.iloc[0:0], signal_available_at[0:0], len(target_weights)
    timeout_ns_delta = int(spec.passive_timeout_minutes) * 60_000_000_000
    signal_ns = np.asarray(signal_available_at, dtype="datetime64[ns]").astype("int64")
    spos = np.searchsorted(grid_ns, signal_ns, side="right")
    spos_clipped = np.minimum(spos, n_grid - 1)
    timeout_pos = np.searchsorted(grid_ns, grid_ns[spos_clipped] + timeout_ns_delta, side="left")
    timeout_pos_clipped = np.minimum(timeout_pos, n_grid - 1)
    replayable = (
        (spos < n_grid)
        & (timeout_pos < n_grid)
        & (grid_ns[timeout_pos_clipped] == grid_ns[spos_clipped] + timeout_ns_delta)
    )
    censored = int((~replayable).sum())
    if censored == 0:
        return target_weights, signal_available_at, 0
    return target_weights.loc[replayable], signal_available_at[replayable], censored


def _assert_cache_required_marks(
    name: str,
    target_replay: pd.DataFrame,
    signal_available_at: pd.DatetimeIndex,
    minute_marks: pd.DataFrame,
) -> None:
    """Fail closed when a replay symbol lacks a finite positive mark at a decision point.

    A mark is required at every decision time where the target weight is
    non-zero. ``minute_marks`` is exactly aligned to the minute closes; a
    missing/non-positive mark at a required decision point raises
    ``DataIntegrityError`` carrying the stable provenance rather than silently
    falling back to OHLCV closes.
    """
    grid_set = set(minute_marks.index)
    for i, decision_time in enumerate(target_replay.index):
        signal_time = signal_available_at[i]
        for sym in target_replay.columns:
            weight = float(target_replay.loc[decision_time, sym])
            if not np.isfinite(weight) or weight == 0.0:
                continue
            mark = float("nan")
            if decision_time in grid_set:
                mark = float(minute_marks.loc[decision_time, sym])
            if not (np.isfinite(mark) and mark > 0):
                prior = minute_marks.index[(minute_marks.index <= signal_time)]
                if len(prior):
                    mark = float(minute_marks.loc[prior[-1], sym])
            if not (np.isfinite(mark) and mark > 0):
                after = minute_marks.index[minute_marks.index > signal_time]
                if len(after):
                    mark = float(minute_marks.loc[after[0], sym])
            if not (np.isfinite(mark) and mark > 0):
                raise DataIntegrityError(
                    "cache_required: no finite positive mark (MISSING_DECISION_MARK) "
                    f"symbol={sym} decision={decision_time} signal={signal_time} "
                    f"for {name}"
                )


def _resolve_ns_vectorized(
    spos_all: np.ndarray,
    full_grid_ns: np.ndarray,
    n_grid: int,
    timeout_ns_delta: int,
) -> np.ndarray:
    """Vectorized ``resolve_ns`` computation for the window generator.

    Bit-identical to the scalar per-decision loop: ``resolve_ns[i]`` is the
    exact timeout bar ``full_grid_ns[spos_all[i]] + timeout_ns_delta`` when it
    lies on the grid, else ``-1``.  ``searchsorted`` (``side="left"``) keeps the
    same semantics; the ``np.minimum`` guards keep out-of-range positions from
    raising instead of silently skipping (matching the scalar ``continue``).
    """
    resolve_ns = np.full(len(spos_all), -1, dtype="int64")
    s = np.minimum(spos_all, n_grid - 1)
    timeout_ns = full_grid_ns[s] + timeout_ns_delta
    tpos = np.searchsorted(full_grid_ns, timeout_ns, side="left")
    valid = (
        (spos_all < n_grid)
        & (tpos < n_grid)
        & (full_grid_ns[np.minimum(tpos, n_grid - 1)] == timeout_ns)
    )
    resolve_ns[valid] = timeout_ns[valid]
    return resolve_ns

def _iter_mhs_execution_windows(
    target_weights: pd.DataFrame,
    signal_available_at: pd.DatetimeIndex,
    root: str,
    timeframe: Literal["1m", "5m"],
    start: pd.Timestamp,
    end: pd.Timestamp,
    funding_by_symbol: dict[str, pd.Series],
    mark_mode: Literal["cache_required", "cache_required_stale_carry", "ohlcv_close_fallback"],
    spec: ExecutionSpec,
) -> Iterator[MhsExecutionWindow]:
    """Yield at-most-31-day execution windows with only the active roster read.

    Each window's minute grid starts at the previous window's last decision
    (the decision-time funding/MTM lead) and ends at the final order's strict
    timeout bar; the last window covers the full evaluation grid so a forced
    exit can always resolve. Only symbols with a non-zero target in the window
    or carried inventory from the previous window are read; the canonical
    column order is preserved on every window for artifact-shape equivalence.
    In ``cache_required`` mode each window's decision marks are asserted
    fail-closed before the window is yielded.
    """
    if len(target_weights) != len(signal_available_at):
        raise DataIntegrityError("signal_available_at must align with target_weights")
    if start >= end:
        raise DataIntegrityError("start must precede end")
    columns = tuple(target_weights.columns)
    freq = {"1m": "1min", "5m": "5min"}[timeframe]
    full_grid = pd.date_range(start, end, freq=freq, tz="UTC")
    full_grid_ns = np.asarray(full_grid, dtype="datetime64[ns]").astype("int64")
    n_grid = len(full_grid_ns)
    timeout_ns_delta = int(spec.passive_timeout_minutes) * 60_000_000_000
    signal_ns = np.asarray(signal_available_at, dtype="datetime64[ns]").astype("int64")
    spos_all = np.searchsorted(full_grid_ns, signal_ns, side="right")
    resolve_ns = _resolve_ns_vectorized(spos_all, full_grid_ns, n_grid, timeout_ns_delta)

    if target_weights.empty:
        empty_marks = (
            pd.DataFrame(index=full_grid) if mark_mode in ("cache_required", "cache_required_stale_carry") else None
        )
        yield ExecutionReplayWindow(
            window_start=start,
            window_end=end,
            columns=columns,
            symbols=(),
            minute_grid=full_grid,
            highs=pd.DataFrame(index=full_grid),
            lows=pd.DataFrame(index=full_grid),
            closes=pd.DataFrame(index=full_grid),
            marks=empty_marks,
            bar_funding=pd.DataFrame(index=full_grid),
            target_weights=target_weights,
            signal_available_at=signal_available_at,
        )
        return

    decision_times = pd.DatetimeIndex(target_weights.index)
    max_window = pd.Timedelta(days=31)
    bounds: list[tuple[int, int]] = []
    i0 = 0
    while i0 < len(decision_times):
        i1 = i0 + 1
        while i1 < len(decision_times) and decision_times[i1] - decision_times[i0] <= max_window:
            i1 += 1
        bounds.append((i0, i1))
        i0 = i1

    prev_active: set[str] = set()
    for wi, (i0, i1) in enumerate(bounds):
        w_weights = target_weights.iloc[i0:i1]
        w_signals = signal_available_at[i0:i1]
        is_last = wi == len(bounds) - 1
        grid_start = start if wi == 0 else decision_times[i0 - 1]
        if is_last:
            grid_end = end
        else:
            max_resolve = int(resolve_ns[i0:i1].max())
            if max_resolve < 0:
                max_resolve = int(
                    np.asarray(decision_times[i1 - 1] + pd.Timedelta(hours=2), dtype="datetime64[ns]").astype("int64")
                )
            grid_end = pd.Timestamp(max_resolve, unit="ns", tz="UTC")
        if grid_end > end:
            grid_end = end
        minute_grid = pd.date_range(grid_start, grid_end, freq=freq, tz="UTC")
        non_zero = w_weights.notna() & w_weights.ne(0.0)
        active = set(w_weights.columns[non_zero.any(axis=0)])
        roster_set = active | prev_active
        prev_active = active
        roster = [
            s for s in columns
            if s in roster_set and os.path.exists(os.path.join(root, timeframe, f"{s}.parquet"))
        ]

        symbol_frames = {
            s: _get_symbol_minute_frame(root, s, timeframe)
            for s in roster
            if os.path.exists(os.path.join(root, timeframe, f"{s}.parquet"))
        }
        aligned = _build_window_frames(
            symbol_frames, roster, grid_start, grid_end, minute_grid, timeframe,
        )
        if aligned is None:
            highs = pd.DataFrame(index=minute_grid)
            lows = pd.DataFrame(index=minute_grid)
            closes = pd.DataFrame(index=minute_grid)
        else:
            highs, lows, closes = aligned
        for s in roster:
            if s not in highs.columns:
                highs[s] = np.nan
                lows[s] = np.nan
                closes[s] = np.nan
        highs = highs.reindex(columns=roster)
        lows = lows.reindex(columns=roster)
        closes = closes.reindex(columns=roster)

        minute_marks: pd.DataFrame | None = None
        if mark_mode in ("cache_required", "cache_required_stale_carry"):
            if roster:
                stale_hours = 24 if mark_mode == "cache_required_stale_carry" else 0
                minute_marks = _data_collector().load_mark_price_panel(
                    roster, "1h", minute_grid, max_stale_hours=stale_hours,
                )
                if mark_mode == "cache_required":
                    _assert_cache_required_marks("window", w_weights[roster], w_signals, minute_marks)
            else:
                minute_marks = pd.DataFrame(index=minute_grid)

        minute_period = minute_grid[1] - minute_grid[0] if len(minute_grid) > 1 else pd.Timedelta(minutes=1)
        funding_window = {
            s: funding_by_symbol[s].loc[
                (funding_by_symbol[s].index >= grid_start)
                & (funding_by_symbol[s].index < grid_end + minute_period)
            ]
            for s in roster
            if s in funding_by_symbol
        }
        minute_funding = (
            bar_funding_panel(funding_window, minute_grid)
            .reindex(columns=roster)
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
            .fillna(0.0)
        )

        yield ExecutionReplayWindow(
            window_start=grid_start,
            window_end=grid_end,
            columns=columns,
            symbols=tuple(roster),
            minute_grid=minute_grid,
            highs=highs,
            lows=lows,
            closes=closes,
            marks=minute_marks,
            bar_funding=minute_funding,
            target_weights=w_weights[roster],
            signal_available_at=w_signals,
        )


def _book_outcome(
    name: str,
    spec: BookSpec,
    n_symbols: int,
    step_grid: pd.DatetimeIndex,
    weights_step: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    phase: PhaseDiagnosticResult,
    root: str,
    request: MhsDiagnosticRequest,
    funding_by_symbol: dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    event_window_bars: int,
    initial_equity: float,
    replay_weights_step: pd.DataFrame | None = None,
    telemetry: _StageRecorder | None = None,
) -> MhsBookReport:
    weights_1h = weights_step.reindex(grid_1h).ffill().fillna(0.0)
    cost_grid = tuple(dict.fromkeys((0.0, 2.0, 4.0, 8.0, *required_cost_tiers())))
    reference_evidence = book_evidence(
        weights_1h, opens, bar_funding, cost_grid, _PERIODS_PER_YEAR_1H, event_window_bars,
    )
    prescreen = reference_evidence.prescreen
    tail = reference_evidence.tail
    # The pre-screen matrices are consumed by ``book_evidence`` above and hold
    # no references from those results.  Releasing them before the minute
    # replay keeps three full multi-year price/weight matrices out of the replay
    # baseline (spec §3.1, ``memory_opt``).
    del weights_1h, reference_evidence
    gc.collect()

    # RC-1: the same significance instruments, pointed at the book that
    # actually carries capital (roster + ensemble + tilt + regime scale). The
    # reference (``weights_step``) and executed (``replay_weights_step``) books
    # are now measured side by side under distinct labels.
    executed_prescreen: dict[float, CostResponsePoint] | None = None
    executed_tail: TailSensitivityResult | None = None
    executed_prescreen_net_t: float | None = None
    if replay_weights_step is not None:
        replay_weights_1h = replay_weights_step.reindex(grid_1h).ffill().fillna(0.0)
        executed_evidence = book_evidence(
            replay_weights_1h, opens, bar_funding, cost_grid, _PERIODS_PER_YEAR_1H, event_window_bars,
        )
        executed_prescreen = executed_evidence.prescreen
        executed_tail = executed_evidence.tail
        executed_prescreen_net_t = executed_evidence.prescreen[
            MEASURED_EXECUTION_COST_TIERS_BPS["base"]
        ].net_t
        del replay_weights_1h, executed_evidence
        gc.collect()

    target_weights = (replay_weights_step if replay_weights_step is not None else weights_step).reindex(step_grid)
    if request.rebalance_filter == "portfolio_trigger":
        target_weights = portfolio_rebalance_trigger(
            target_weights, MHS_REBALANCE_TRACKING_ERROR_THRESHOLD,
        )
    else:
        target_weights = _apply_rebalance_deadband(target_weights, MHS_REBALANCE_MIN_NOTIONAL_DELTA)
    signal_available_at = step_grid + pd.Timedelta(hours=1)
    execution_grid = pd.date_range(
        start, end,
        freq={"1m": "1min", "5m": "5min"}[request.execution_timeframe],
        tz="UTC",
    )
    target_replay, signal_replay, censored = _truncate_replayable_decisions(
        target_weights, signal_available_at, execution_grid, ExecutionSpec(),
    )
    replay_symbols = list(target_replay.columns)

    def _windows() -> Iterator[MhsExecutionWindow]:
        return _iter_mhs_execution_windows(
            target_replay, signal_replay, root, request.execution_timeframe,
            start, end, funding_by_symbol, request.mark_mode, ExecutionSpec(),
        )

    def _window_telemetry(
        gen: Iterator[MhsExecutionWindow], prefix: str,
    ) -> Iterator[MhsExecutionWindow]:
        for idx, w in enumerate(gen):
            if telemetry is not None:
                telemetry.record(
                    f"{prefix}_{idx}",
                    grid_bars=len(w.minute_grid),
                    active_symbols=len(w.symbols),
                    window_start=str(w.window_start),
                    window_end=str(w.window_end),
                )
            yield w
            _assert_execution_rss_budget(prefix, request.max_rss_bytes, idx + 1)

    touch = None
    touch_naive_sharpe = None
    ladder = None
    ladder_naive_sharpe = None
    patient_reference = None
    patient_reference_naive_sharpe = None
    pre_vol_target_reference = None
    pre_vol_target_reference_naive_sharpe = None
    try:
        primary = replay_execution_windows(
            _window_telemetry(_windows(), "execution_window"),
            initial_equity, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
            retain_event_snapshots=False,
            min_equity_fraction=MHS_REFERENCE_PASS_EQUITY_FLOOR,
        )
        # Pass 1 (reference) -> P&L-vol-target scale -> Pass 2 (reported):
        # the strategy's own realized daily-return vol drives a causal
        # multiplicative exposure scalar, so the reported primary/stress/etc.
        # replay the rescaled weights while ``pre_vol_target_reference`` keeps
        # the unscaled Pass-1 result as a diagnostic field.
        pnl_vol_target_scale = _pnl_vol_target_scale(
            primary.ledger.equity.resample("1D").last().pct_change()
        )
        if request.pnl_vol_target:
            target_replay = target_replay.mul(
                pnl_vol_target_scale.reindex(target_replay.index, method="ffill").fillna(1.0),
                axis=0,
            )
        pre_vol_target_reference = primary
        pre_vol_target_reference_naive_sharpe = _naive_sharpe(primary.ledger)
        # Pass 2 (reported) replays the SAME scaled target_replay Pass 1's own
        # vol-target scale was derived from; a book whose Pass-1 reference is
        # forced flat for most of the window (a known negative-edge book) still
        # produces an artificially calm reference vol series, so Pass 2 must
        # carry its own floor rather than relying on the derived scale alone
        # (docs/specs/mhs_capital_floor_and_overlay_validation.md §1.1).
        primary = replay_execution_windows(
            _window_telemetry(_windows(), "execution_window"),
            initial_equity, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
            retain_event_snapshots=False,
            min_equity_fraction=MHS_REFERENCE_PASS_EQUITY_FLOOR,
        )
        stress = replay_execution_windows(
            _window_telemetry(_windows(), "execution_window_cost_stress"),
            initial_equity, "OHLCV_IMMEDIATE_TAKER", _stress_cost_execution_spec(),
            retain_event_snapshots=False,
            min_equity_fraction=MHS_REFERENCE_PASS_EQUITY_FLOOR,
        )
        patient_reference = replay_execution_windows(
            _window_telemetry(_windows(), "execution_window_patient_reference"),
            initial_equity, "OHLCV_STRICT_PROXY", ExecutionSpec(),
            retain_event_snapshots=False,
            min_equity_fraction=MHS_REFERENCE_PASS_EQUITY_FLOOR,
        )
        patient_reference_naive_sharpe = _naive_sharpe(patient_reference.ledger)
        if request.touch_diagnostic:
            touch = replay_execution_windows(
                _window_telemetry(_windows(), "execution_window_touch"),
                initial_equity, "OHLCV_TOUCH_PROXY", ExecutionSpec(),
                retain_event_snapshots=False,
                min_equity_fraction=MHS_REFERENCE_PASS_EQUITY_FLOOR,
            )
            touch_naive_sharpe = _naive_sharpe(touch.ledger)
        if request.ladder_diagnostic:
            _validate_ladder_schedule_contract()
            ladder = replay_execution_windows(
                _window_telemetry(_windows(), "execution_window_ladder"),
                initial_equity, "OHLCV_LADDERED_PROXY", ExecutionSpec(),
                retain_event_snapshots=False,
                min_equity_fraction=MHS_REFERENCE_PASS_EQUITY_FLOOR,
            )
            ladder_naive_sharpe = _naive_sharpe(ladder.ledger)
        if telemetry is not None:
            telemetry.record(
                f"replay_{name}_strict",
                n_symbols=len(replay_symbols),
                fill_count=len(primary.simulated_fills),
            )
            telemetry.record(
                f"replay_{name}_stress",
                n_symbols=len(replay_symbols),
                fill_count=len(stress.simulated_fills),
            )
        if request.mark_mode == "cache_required":
            _assert_cache_required_ledger_valid(name, primary)
    except DataIntegrityError as exc:
        failure = MhsBookFailure(
            stage=f"replay_{name}",
            error_class=type(exc).__name__,
            reason=_classify_execution_failure(exc),
            message=str(exc),
        )
        if telemetry is not None:
            telemetry.record(
                f"replay_{name}_failed",
                n_symbols=len(replay_symbols),
                fill_count=0,
            )
        return MhsBookReport(
            name=name,
            band=spec.band.name,
            horizon_hours=spec.horizon_hours,
            step_hours=spec.step_hours,
            tranche_count=spec.tranche_count(),
            n_symbols=n_symbols,
            phase=phase,
            prescreen=prescreen,
            tail=tail,
            primary=None,
            stress=None,
            primary_autocorr_sharpe=None,
            primary_naive_sharpe=None,
            primary_net_ann=None,
            primary_geometric_cagr=None,
            primary_max_drawdown=None,
            primary_annualized_turnover=None,
            stress_naive_sharpe=None,
            terminal_censored_decisions=censored,
            failure=failure,
            touch=touch,
            touch_naive_sharpe=touch_naive_sharpe,
            ladder=ladder,
            ladder_naive_sharpe=ladder_naive_sharpe,
            patient_reference=patient_reference,
            patient_reference_naive_sharpe=patient_reference_naive_sharpe,
            pre_vol_target_reference=pre_vol_target_reference,
            pre_vol_target_reference_naive_sharpe=pre_vol_target_reference_naive_sharpe,
            executed_prescreen=executed_prescreen,
            executed_tail=executed_tail,
            executed_prescreen_net_t=executed_prescreen_net_t,
        )
    equity_1h, net_returns_1h, turnover_1h = _hourly_ledger_series(
        primary.ledger.equity, primary.ledger.fill_turnover,
    )
    return MhsBookReport(
        name=name,
        band=spec.band.name,
        horizon_hours=spec.horizon_hours,
        step_hours=spec.step_hours,
        tranche_count=spec.tranche_count(),
        n_symbols=n_symbols,
        phase=phase,
        prescreen=prescreen,
        tail=tail,
        primary=primary,
        stress=stress,
        primary_autocorr_sharpe=_daily_autocorr_sharpe(primary.ledger),
        primary_naive_sharpe=_naive_sharpe(primary.ledger),
        primary_net_ann=_mean_ann(net_returns_1h, _PERIODS_PER_YEAR_1H),
        primary_geometric_cagr=_geometric_cagr(equity_1h),
        primary_max_drawdown=_mdd(primary.ledger.equity),
        primary_annualized_turnover=_mean_ann(turnover_1h, _PERIODS_PER_YEAR_1H),
        stress_naive_sharpe=_naive_sharpe(stress.ledger),
        terminal_censored_decisions=censored,
        touch=touch,
        touch_naive_sharpe=touch_naive_sharpe,
        ladder=ladder,
        ladder_naive_sharpe=ladder_naive_sharpe,
        patient_reference=patient_reference,
        patient_reference_naive_sharpe=patient_reference_naive_sharpe,
        pre_vol_target_reference=pre_vol_target_reference,
        pre_vol_target_reference_naive_sharpe=pre_vol_target_reference_naive_sharpe,
        executed_prescreen=executed_prescreen,
        executed_tail=executed_tail,
        executed_prescreen_net_t=executed_prescreen_net_t,
        primary_realized_shortfall_bps=primary.all_intent_shortfall_bps,
        primary_notional_weighted_shortfall_bps=primary.notional_weighted_shortfall_bps,
        stress_realized_shortfall_bps=stress.all_intent_shortfall_bps,
        stress_notional_weighted_shortfall_bps=stress.notional_weighted_shortfall_bps,
        primary_fill_count=primary.fill_count,
        primary_unfilled_count=primary.unfilled_count,
        primary_forced_exit_notional=primary.forced_exit_notional,
    )


def _book_outcome_worker(
    name: str,
    spec: BookSpec,
    n_symbols: int,
    step_grid: pd.DatetimeIndex,
    weights_step: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    phase: PhaseDiagnosticResult,
    root: str,
    request: MhsDiagnosticRequest,
    funding_by_symbol: dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    event_window_bars: int,
    initial_equity: float,
    replay_weights_step: pd.DataFrame | None,
) -> tuple[MhsBookReport, tuple[MhsResourceMeasurement, ...]]:
    """Run one ``_book_outcome`` in a fork child with its own telemetry recorder.

    The typed failure conversion inside ``_book_outcome`` is preserved; a book
    that fails its replay is still returned (with ``failure`` set) so the other
    two books' results are never lost.  The per-window telemetry is returned so
    the parent can merge it in declared order.
    """
    recorder = _StageRecorder(log_run=False)
    report = _book_outcome(
        name, spec, n_symbols, step_grid, weights_step, grid_1h,
        opens, bar_funding, phase, root, request, funding_by_symbol,
        start, end, event_window_bars, initial_equity, replay_weights_step,
        telemetry=recorder,
    )
    return report, recorder.records


def _preload_symbol_minute_frames(
    root: str, symbols: list[str], timeframe: Literal["1m", "5m"],
) -> None:
    """Populate the O6 per-symbol minute-frame cache before forking book workers.

    Forked children inherit the populated cache copy-on-write, so the three
    books share one set of full-period frames instead of independently re-reading
    Parquet (up to 3x redundant reads).  Missing Parquet files are skipped (the
    window generator applies the same existence guard).
    """
    for sym in symbols:
        if os.path.exists(os.path.join(root, timeframe, f"{sym}.parquet")):
            _get_symbol_minute_frame(root, sym, timeframe)


def _active_blend_book_and_grid(
    fast: BookSpec,
    slow: BookSpec,
    fast_grid: pd.DatetimeIndex,
    slow_grid: pd.DatetimeIndex,
) -> tuple[BookSpec, pd.DatetimeIndex]:
    """Select the blend's active book spec and execution grid from the capital contract.

    The blend's decision cadence must derive from the same contract that
    allocates capital (``PHASE_1_BOOK_BLEND_WEIGHTS``), never from a hardcoded
    book name: with only ``slow_momentum`` weighted the blend replays on slow's
    native 24h grid, while a nonzero ``fast_reversal`` weight (e.g. the
    historical 50/50) admits fast's 6h grid -- a superset of slow's from the
    same origin -- reproducing the pre-fix behavior byte-for-byte.  If no book
    carries capital the allocation invariant is violated and we fail closed.
    """
    if PHASE_1_BOOK_BLEND_WEIGHTS["fast_reversal"] != 0.0:
        return fast, fast_grid
    if PHASE_1_BOOK_BLEND_WEIGHTS["slow_momentum"] != 0.0:
        return slow, slow_grid
    raise ValueError(
        "PHASE_1_BOOK_BLEND_WEIGHTS allocates no capital to either book; "
        "blend has no active execution grid"
    )


def _run_books_concurrent(
    root: str,
    request: MhsDiagnosticRequest,
    n_symbols: int,
    grid_1h: pd.DatetimeIndex,
    fast: BookSpec,
    slow: BookSpec,
    fast_grid: pd.DatetimeIndex,
    slow_grid: pd.DatetimeIndex,
    w_fast: pd.DataFrame,
    w_slow: pd.DataFrame,
    w_fast_execution: pd.DataFrame,
    w_slow_execution: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    phase_fast: PhaseDiagnosticResult,
    phase_slow: PhaseDiagnosticResult,
    phase_blend: PhaseDiagnosticResult,
    start: pd.Timestamp,
    end: pd.Timestamp,
    funding_by_symbol: dict[str, pd.Series],
    blend_1h: pd.DataFrame,
    execution_mask: pd.DataFrame,
    initial_equity: float,
    telemetry: _StageRecorder | None = None,
    regime_scale: pd.Series | None = None,
) -> tuple[MhsBookReport, MhsBookReport, MhsBookReport]:
    """Run the three top-level books concurrently in fork children.

    The books share zero mutable state and only read the immutable 1h panels and
    the O6 minute-frame cache, so they are embarrassingly parallel.
    ``ProcessPoolExecutor`` (fork) is used instead of threads: the replay loops
    are a CPU-bound Python/numpy mix, so the GIL would serialize threads at
    ~1.6x rather than the ~3x fork workers achieve, and fork lets the workers
    share the read-only panels and preloaded cache via copy-on-write (no 3x RSS
    blow-up), matching the existing ``_run_folds_parallel`` pattern.  Per-book
    telemetry is merged into the parent recorder in declared book order.

    ``blend_replay`` (the blend book's actual execution-replay weights) is
    built independently from ``blend_1h``/``blend_step`` because it must stay
    restricted to the execution-roster (``w_*_execution``) symbols actually
    tradable at minute granularity -- ``blend_1h`` covers the full eligible
    universe and is prescreen/tail-diagnostic only (never itself replayed).
    ``regime_scale`` (the R1 volatility-regime cash scale, optionally composed
    with the opt-in trend-efficiency overlay) is applied to ``blend_1h`` by the
    caller already; it must also be applied here so the blend book's actual
    ``primary``/``stress`` replay reflects it, matching what the anchored-fold
    path already does to its own target weights (docs/specs/
    mhs_capital_floor_and_overlay_validation.md §2 -- previously this scale
    only reached the top-level prescreen/tail fields, never the replay).
    """
    active_spec, active_grid = _active_blend_book_and_grid(fast, slow, fast_grid, slow_grid)
    blend_step = blend_1h.reindex(active_grid)
    blend_replay = (
        PHASE_1_BOOK_BLEND_WEIGHTS["fast_reversal"] * w_fast_execution.reindex(grid_1h).ffill().fillna(0.0)
        + PHASE_1_BOOK_BLEND_WEIGHTS["slow_momentum"] * w_slow_execution.reindex(grid_1h).ffill().fillna(0.0)
    ).reindex(active_grid)
    if regime_scale is not None:
        blend_replay = blend_replay.mul(regime_scale.reindex(active_grid).fillna(1.0), axis=0)

    with ProcessPoolExecutor(max_workers=3) as pool:
        f_fast = pool.submit(
            _book_outcome_worker,
            "fast_reversal", fast, n_symbols, fast_grid, w_fast, grid_1h,
            opens, bar_funding, phase_fast, root, request, funding_by_symbol,
            start, end, fast.horizon_hours, initial_equity, w_fast_execution,
        )
        f_slow = pool.submit(
            _book_outcome_worker,
            "slow_momentum", slow, n_symbols, slow_grid, w_slow, grid_1h,
            opens, bar_funding, phase_slow, root, request, funding_by_symbol,
            start, end, slow.horizon_hours, initial_equity, w_slow_execution,
        )
        f_blend = pool.submit(
            _book_outcome_worker,
            "blend", active_spec, n_symbols, active_grid, blend_step, grid_1h,
            opens, bar_funding, phase_blend, root, request, funding_by_symbol,
            start, end, 168, initial_equity, blend_replay,
        )
        fast_report, fast_records = f_fast.result()
        slow_report, slow_records = f_slow.result()
        blend_report, blend_records = f_blend.result()

    if telemetry is not None:
        for records in (fast_records, slow_records, blend_records):
            telemetry.absorb(records)
    return fast_report, slow_report, blend_report


def _run_post_diag_deploy(
    blend_report: MhsBookReport,
    root: str,
    request: MhsDiagnosticRequest,
    execution_symbols: list[str],
    minute_grid: pd.DatetimeIndex,
    signal_48h: pd.DataFrame,
    eligible: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    fast: BookSpec,
) -> tuple[
    tuple[float, float] | None,
    float | None,
    dict[str, float],
    dict[str, int],
    DeploymentReadinessResult,
]:
    """Diagnostics + deployment readiness, one background-thread unit.

    ``compute_deployment_readiness`` is invoked with ``research_go_eligible=None``:
    the only value it needs from the anchored folds is the final Research-GO
    boolean flag, which the caller patches in after the folds resolve.  This is
    what lets the whole 77s post-book tail overlap the ~78s fold pool.
    """
    bootstrap_ci: tuple[float, float] | None = None
    if blend_report.primary is None:
        raise DataIntegrityError("post-book tail requires a blend primary replay")
    equity_1h = blend_report.primary.ledger.equity.resample("1h").last().dropna()
    net_1h = equity_1h.pct_change().dropna()
    if len(net_1h) >= 2:
        bootstrap_ci = _bootstrap_ci(
            net_1h, _BOOTSTRAP_REPLICATES, _BOOTSTRAP_MEAN_BLOCK, _BOOTSTRAP_SEED,
        )
    participation = _participation_warnings(
        blend_report.primary, root, request.execution_timeframe,
        execution_symbols, minute_grid,
    )
    termination_counts = dict(blend_report.primary.termination_counts)
    if blend_report.primary_naive_sharpe is None:
        raise DataIntegrityError("blend report requires a naive Sharpe for the placebo")
    placebo_percentile = _placebo_sharpe_percentile(
        signal_48h, eligible, opens, bar_funding, grid_1h,
        fast, blend_report.primary_naive_sharpe, 500, _BOOTSTRAP_SEED,
    )
    deployment = compute_deployment_readiness(
        equity_1h,
        _PERIODS_PER_YEAR_1H,
        participation_warnings=participation,
        primary_valid=blend_report.primary.ledger.primary_valid,
        research_go_eligible=None,
        n_bootstrap=_BOOTSTRAP_REPLICATES,
        mean_block_bars=_BOOTSTRAP_MEAN_BLOCK,
        seed=_BOOTSTRAP_SEED,
    )
    return bootstrap_ci, placebo_percentile, participation, termination_counts, deployment


def _run_post_book_concurrently(
    blend_report: MhsBookReport | None,
    root: str,
    request: MhsDiagnosticRequest,
    execution_symbols: list[str],
    minute_grid: pd.DatetimeIndex | None,
    signal_48h: pd.DataFrame,
    eligible: pd.DataFrame,
    opens: pd.DataFrame,
    bar_funding: pd.DataFrame,
    grid_1h: pd.DatetimeIndex,
    fast: BookSpec,
    fold_funding: dict[str, pd.Series],
    initial_equity: float,
    telemetry: _StageRecorder | None = None,
    fold_slow_horizons: dict[int, int | None] | None = None,
    fold_fast_horizons: dict[int, tuple[int, str]] | None = None,
) -> tuple[
    tuple[float, float] | None,
    float | None,
    dict[str, float],
    dict[str, int],
    tuple[MhsFoldReport, ...],
    DeploymentReadinessResult | None,
]:
    """Run anchored folds, diagnostics, and deployment readiness concurrently.

    The fold pool is forked while the main process is quiescent (the book
    workers have joined and no diagnostic thread exists yet), then a single
    background thread runs the diagnostics + deployment-readiness tail in
    parallel with the fold workers.  The fold result telemetry is recorded in
    fold order; ``blend_participation``/``statistical_diagnostics`` telemetry is
    left to the caller so the ordered-stage contract is preserved deterministically.
    """
    folds = phase_1_anchored_purged_folds()
    has_primary = blend_report is not None and blend_report.primary is not None

    bootstrap_ci: tuple[float, float] | None = None
    placebo_percentile: float | None = None
    participation: dict[str, float] = {}
    termination_counts: dict[str, int] = {}
    fold_reports: tuple[MhsFoldReport, ...] = ()
    deployment: DeploymentReadinessResult | None = None

    if not folds:
        if has_primary:
            (
                bootstrap_ci, placebo_percentile, participation,
                termination_counts, deployment,
            ) = _run_post_diag_deploy(
                blend_report, root, request, execution_symbols, minute_grid,
                signal_48h, eligible, opens, bar_funding, grid_1h, fast,
            )
        return (
            bootstrap_ci, placebo_percentile, participation,
            termination_counts, fold_reports, deployment,
        )

    reports: dict[int, MhsFoldReport] = {}
    with ProcessPoolExecutor(max_workers=min(3, len(folds))) as pool:
        futures = {
            pool.submit(
                _run_anchored_fold,
                root, fold, request, fold_funding, initial_equity, idx, None,
                (fold_slow_horizons or {}).get(idx),
                (fold_fast_horizons or {}).get(idx),
            ): idx
            for idx, fold in enumerate(folds)
        }
        # The fold pool is now forked; start the diagnostics/deployment thread.
        with ThreadPoolExecutor(max_workers=1) as tpool:
            post_future = None
            if has_primary:
                assert blend_report is not None
                post_future = tpool.submit(
                    _run_post_diag_deploy,
                    blend_report, root, request, execution_symbols, minute_grid,
                    signal_48h, eligible, opens, bar_funding, grid_1h, fast,
                )
            for future in as_completed(futures):
                idx = futures[future]
                reports[idx] = future.result()
            if post_future is not None:
                (
                    bootstrap_ci, placebo_percentile, participation,
                    termination_counts, deployment,
                ) = post_future.result()

    ordered = tuple(reports[i] for i in range(len(folds)))
    fold_reports = ordered
    if telemetry is not None:
        for fold_report in ordered:
            fill_count = (
                len(fold_report.strict.simulated_fills) + len(fold_report.stress.simulated_fills)
                if fold_report.strict is not None and fold_report.stress is not None
                else 0
            )
            telemetry.record(f"anchored_fold_{fold_report.fold_index}", fill_count=fill_count)
    return (
        bootstrap_ci, placebo_percentile, participation,
        termination_counts, fold_reports, deployment,
    )

def _incomplete_fold_report(
    fold: AnchoredPurgedFold, fold_index: int, failures: tuple[str, ...],
) -> MhsFoldReport:
    """A fold that could not be replayed, failed closed with its reason codes."""
    return MhsFoldReport(
        fold_index=fold_index,
        validation_start=str(fold.validation_start),
        validation_end=str(fold.validation_end),
        strict=None,
        stress=None,
        primary_valid=False,
        primary_autocorr_sharpe=float("nan"),
        primary_naive_sharpe=float("nan"),
        primary_net_ann=float("nan"),
        primary_geometric_cagr=float("nan"),
        primary_max_drawdown=float("nan"),
        stress_naive_sharpe=float("nan"),
        decision_intents=0,
        termination_counts={},
        failures=tuple(sorted(set(failures))),
        strict_elapsed_seconds=0.0,
        stress_elapsed_seconds=0.0,
    )


def _fold_safe_slow_book_spec(
    selection: DiscoveryQualificationResult,
    default: BookSpec,
) -> tuple[BookSpec, int, str]:
    """Resolve one fold's ``slow_momentum`` spec from its fold-scoped selection.

    Returns ``(spec, horizon_hours, source)``. ``source`` is
    ``"fold_train_only_discovery"`` only when the fold-scoped gate admitted a
    candidate (spec is ``default`` with ``horizon_hours`` replaced by the
    selected horizon, keeping band/step_hours/min_symbols identical to the
    frozen default); otherwise ``"frozen_default"`` with ``spec is default``
    unchanged.
    """
    if selection.admitted and selection.selected_horizon is not None:
        return (
            BookSpec(
                band=default.band,
                horizon_hours=selection.selected_horizon,
                step_hours=default.step_hours,
                min_symbols=default.min_symbols,
            ),
            selection.selected_horizon,
            "fold_train_only_discovery",
        )
    return default, default.horizon_hours, "frozen_default"


def _fold_safe_fast_horizon(
    selection: DiscoveryQualificationResult,
    default_horizon: int,
) -> tuple[int, str]:
    """Resolve one fold's ``fast_reversal`` horizon from its fold-scoped selection.

    Diagnostic-only: returns ``(horizon_hours, source)`` instead of a
    ``BookSpec`` because fast_reversal's book construction and
    ``PHASE_1_BOOK_BLEND_WEIGHTS`` stay frozen at 0.0 capital (the result is
    evidence for a separate governance decision, never a weight change).
    ``source`` is ``"fold_train_only_discovery"`` only when the fold-scoped
    gate admitted a candidate (``admitted`` and ``selected_horizon`` both
    truthy); otherwise ``"frozen_default"`` with ``default_horizon`` unchanged.
    """
    if selection.admitted and selection.selected_horizon is not None:
        return selection.selected_horizon, "fold_train_only_discovery"
    return default_horizon, "frozen_default"


def _build_fold_target_weights(
    root: str,
    fold: AnchoredPurgedFold,
    request: MhsDiagnosticRequest,
    funding_by_symbol: dict[str, pd.Series],
    slow_horizon_override: int | None = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str], pd.DatetimeIndex]:
    """Construct one fold's PIT decision targets with the quality calibration.

    Returns ``(target_weights, signal_available_at, minute_roster, grid_1h)``:
    the blend decision targets over the validation window, their ``+1h``
    signal-availability stamps, the minute-data roster, and the 1h feature
    grid. The 1h panel is sliced to ``[validation_start - warmup, validation_end]``
    (spec §3.1) so warm-up history feeds the 720-bar eligibility lookback and
    the 168h slow horizon without holding the full ``[train_start, validation_end]``
    panel. Signal quality (spec §3.2) applies EMA smoothing on each book, a
    volatility-regime cash scale, and the turnover deadband cap on the final
    blend targets. All objects are local to this builder and released when it
    returns, keeping per-fold peak memory bounded.
    """
    ts = fold.train_start
    vs = fold.validation_start
    ve = fold.validation_end
    panel_start = max(ts, vs - pd.Timedelta(hours=MHS_FOLD_PANEL_WARMUP_HOURS))
    panel = load_base_panel(
        root, "1h", ("close", "open", "quote_vol"), panel_start, ve,
        partition="dev", min_bars=2000,
    )
    close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    del panel
    grid_1h = close.index
    symbols = list(close.columns)
    funded = [
        s for s in symbols
        if s in funding_by_symbol and s not in MHS_SOURCE_GAP_EXCLUDED_SYMBOLS
    ]
    if not funded:
        raise RuntimeError("no fold symbol has funding coverage")
    close = close[funded]
    opens = opens[funded]
    quote_vol = quote_vol[funded]
    bar_period = grid_1h[1] - grid_1h[0]
    funding_window = {
        s: funding_by_symbol[s].loc[
            (funding_by_symbol[s].index >= grid_1h[0])
            & (funding_by_symbol[s].index < grid_1h[-1] + bar_period)
        ]
        for s in funded
    }
    bar_funding = bar_funding_panel(funding_window, grid_1h)
    del funding_window
    aligned_symbols = list(bar_funding.columns)
    if not aligned_symbols:
        raise RuntimeError("no fold symbol has causally aligned funding coverage")
    close = close[aligned_symbols]
    opens = opens[aligned_symbols]
    quote_vol = quote_vol[aligned_symbols]
    bar_funding = bar_funding[aligned_symbols]

    eligible = liquid_half_eligibility(quote_vol, lookback_bars=720, min_history_bars=720)
    log_close = np.log(close)
    del close
    fast = PHASE_1_BOOK_SPECS["fast_reversal"]
    slow = (
        dataclasses.replace(PHASE_1_BOOK_SPECS["slow_momentum"], horizon_hours=slow_horizon_override)
        if slow_horizon_override is not None
        else PHASE_1_BOOK_SPECS["slow_momentum"]
    )
    fast_grid = pd.date_range(panel_start, ve, freq="6h", tz="UTC")
    slow_grid = pd.date_range(panel_start, ve, freq="24h", tz="UTC")
    fast_ema = _signal_ema_span(fast.band.sign, fast.horizon_hours, fast.step_hours)
    slow_ema = _signal_ema_span(slow.band.sign, slow.horizon_hours, slow.step_hours)
    w_fast = _book_weights(log_close, eligible, fast, fast_grid, ema_span=fast_ema)
    w_fast_tilted = inverse_realized_vol_tilt(
        w_fast, realized_vol(log_close, fast.horizon_hours).reindex(fast_grid),
    )
    execution_mask = _pit_execution_mask(quote_vol, eligible, request.execution_universe_size)
    w_fast_execution = renormalize_within_mask(
        w_fast_tilted, execution_mask.reindex(w_fast.index).fillna(False), fast.min_symbols,
    )
    w_slow_execution = _slow_book_execution_weights(
        log_close, eligible, execution_mask, slow, slow_grid,
        request.slow_book_mode, request.ensemble_signal, slow_ema,
    )
    if request.beta_neutralize:
        w_slow_execution = beta_neutralize_weights(
            w_slow_execution,
            causal_market_beta(
                log_close, eligible,
                MHS_CAUSAL_BETA_LOOKBACK_BARS, MHS_CAUSAL_BETA_MIN_PERIODS,
            ).reindex(w_slow_execution.index),
            execution_mask.reindex(w_slow_execution.index).fillna(False),
            slow.min_symbols,
        )
    del quote_vol, eligible
    del w_fast, w_fast_tilted
    w_slow_execution_1h = w_slow_execution.reindex(grid_1h).ffill().fillna(0.0)
    if request.crash_regime_tilt_alpha is not None:
        w_slow_execution_1h = crash_regime_tilt_weights(
            w_slow_execution_1h, log_close,
            execution_mask.reindex(grid_1h).ffill().fillna(False),
            MHS_CRASH_REGIME_REFERENCE_SYMBOLS, slow.horizon_hours,
            request.crash_regime_tilt_alpha, min_symbols=slow.min_symbols,
        )
    blend_1h = (
        PHASE_1_BOOK_BLEND_WEIGHTS["fast_reversal"] * w_fast_execution.reindex(grid_1h).ffill().fillna(0.0)
        + PHASE_1_BOOK_BLEND_WEIGHTS["slow_momentum"] * w_slow_execution_1h
    )
    del w_fast_execution, w_slow_execution, w_slow_execution_1h
    _active_spec, active_grid = _active_blend_book_and_grid(fast, slow, fast_grid, slow_grid)
    del _active_spec
    decision_grid = active_grid[(active_grid >= vs) & (active_grid <= ve)]
    target_weights = blend_1h.reindex(decision_grid)
    del blend_1h

    # The regime cash scale must read the traded execution roster, not the
    # full eligible universe: only the execution_mask symbols carry capital, so
    # their realized vol is the quantity that decides high-vol cash scaling.
    vol_mean = realized_vol(log_close, 48).where(execution_mask).reindex(decision_grid).mean(axis=1)
    regime_scale = _regime_cash_scale(vol_mean)
    if request.trend_efficiency_overlay:
        regime_scale = regime_scale.mul(
            _trend_efficiency_overlay_scale(log_close, execution_mask, fast.horizon_hours, decision_grid),
        )
    del execution_mask
    del log_close
    if request.rebalance_filter == "portfolio_trigger":
        # Gate the UNSCALED book, then multiply the gross scale afterwards so
        # the trigger's invariant-preserving row holds cannot filter out the
        # regime de-risking (docs/specs/mhs_alpha_engine.md §1).
        target_weights = portfolio_rebalance_trigger(
            target_weights, MHS_REBALANCE_TRACKING_ERROR_THRESHOLD,
        ).mul(regime_scale, axis=0)
    else:
        target_weights = _apply_rebalance_deadband(
            target_weights.mul(regime_scale, axis=0), MHS_REBALANCE_MIN_NOTIONAL_DELTA,
        )

    if target_weights.empty:
        raise RuntimeError("fold decision grid is empty")
    execution_symbols = sorted(target_weights.columns[target_weights.ne(0.0).any(axis=0)])
    minute_roster = [
        s for s in execution_symbols
        if os.path.exists(os.path.join(root, request.execution_timeframe, f"{s}.parquet"))
    ]
    if not minute_roster:
        raise RuntimeError("no fold decision symbol has minute execution data")
    signal_available_at = target_weights.index + pd.Timedelta(hours=1)
    return target_weights, signal_available_at, minute_roster, grid_1h


def _run_anchored_fold(
    root: str,
    fold: AnchoredPurgedFold,
    request: MhsDiagnosticRequest,
    funding_by_symbol: dict[str, pd.Series],
    initial_equity: float,
    fold_index: int,
    telemetry: _StageRecorder | None = None,
    slow_horizon_override: int | None = None,
    fast_horizon_override: tuple[int, str] | None = None,
) -> MhsFoldReport:
    """One independently flat strict/immediate-taker blend replay per fold.

    The 1h panel spans ``[train_start, validation_end]`` so warm-up history
    feeds features only; the replay decisions and the fresh flat ledger cover
    only the validation window. The fold uses the same at-most-31-day windowed
    execution engine as the top-level books (``_iter_mhs_execution_windows`` +
    ``replay_execution_windows``, immediate-taker primary and cost-stressed
    stress) so dense event snapshots stay disabled
    and per-window resource telemetry/RSS budgets are applied inside the fold,
    not only at the top level. A fold that cannot be replayed is reported (not
    raised) with machine-readable failure codes.
    """
    try:
        vs = fold.validation_start
        ve = fold.validation_end
        target_weights, signal_available_at, minute_roster, _grid_1h = _build_fold_target_weights(
            root, fold, request, funding_by_symbol, slow_horizon_override,
        )
        target_replay = target_weights[minute_roster]
        execution_grid = pd.date_range(
            vs, ve,
            freq={"1m": "1min", "5m": "5min"}[request.execution_timeframe],
            tz="UTC",
        )
        target_replay, signal_available_at, terminal_censored = _truncate_replayable_decisions(
            target_replay, signal_available_at, execution_grid, ExecutionSpec(),
        )
        decision_intents = int(np.isfinite(target_replay.to_numpy()).sum())

        def _windows() -> Iterator[MhsExecutionWindow]:
            return _iter_mhs_execution_windows(
                target_replay, signal_available_at, root, request.execution_timeframe,
                vs, ve, funding_by_symbol, request.mark_mode, ExecutionSpec(),
            )

        def _window_telemetry(
            gen: Iterator[MhsExecutionWindow], prefix: str,
        ) -> Iterator[MhsExecutionWindow]:
            for idx, w in enumerate(gen):
                if telemetry is not None:
                    telemetry.record(
                        f"{prefix}_{idx}",
                        grid_bars=len(w.minute_grid),
                        active_symbols=len(w.symbols),
                        window_start=str(w.window_start),
                        window_end=str(w.window_end),
                    )
                yield w
                _assert_execution_rss_budget(prefix, request.max_rss_bytes, idx + 1)

        window_prefix = f"anchored_fold_{fold_index}_window"
        primary = replay_execution_windows(
            _window_telemetry(_windows(), window_prefix),
            initial_equity, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
            retain_event_snapshots=False,
        )
        # Two-pass primary (reference -> P&L-vol-target rescale -> reported):
        # same causal P&L-vol-target scale as the top-level books, computed from
        # the fold's own validation-window reference ledger.
        pnl_vol_target_scale = _pnl_vol_target_scale(
            primary.ledger.equity.resample("1D").last().pct_change()
        )
        target_replay = target_replay.mul(
            pnl_vol_target_scale.reindex(target_replay.index, method="ffill").fillna(1.0),
            axis=0,
        )
        primary = replay_execution_windows(
            _window_telemetry(_windows(), window_prefix),
            initial_equity, "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(),
            retain_event_snapshots=False,
        )
        stress = replay_execution_windows(
            _window_telemetry(_windows(), f"{window_prefix}_cost_stress"),
            initial_equity, "OHLCV_IMMEDIATE_TAKER", _stress_cost_execution_spec(),
            retain_event_snapshots=False,
        )

        failures: list[str] = []
        equity = primary.ledger.equity
        if not np.isfinite(equity.to_numpy()).all() or not (equity > 0).all():
            failures.append(MHS_GO_REASON_NONFINITE_EQUITY)
        if not primary.ledger.primary_valid:
            failures.append(MHS_GO_REASON_INVALID_PRIMARY)
        if (
            primary.termination_counts.get("MISSING_DATA", 0) > 0
            or primary.termination_counts.get("UNKNOWN_TERMINATION", 0) > 0
        ):
            failures.append(MHS_GO_REASON_EXECUTION_GAP)
        primary_autocorr = _daily_autocorr_sharpe(primary.ledger)
        if not np.isfinite(primary_autocorr) or primary_autocorr < MHS_GO_PRIMARY_SHARPE_FLOOR:
            failures.append(MHS_GO_REASON_PRIMARY_SHARPE)
        stress_sharpe = _naive_sharpe(stress.ledger)
        if not np.isfinite(stress_sharpe) or stress_sharpe <= 0.0:
            failures.append(MHS_GO_REASON_STRESS_SHARPE)

        equity_1h, net_returns_1h, _turnover_1h = _hourly_ledger_series(
            equity, primary.ledger.fill_turnover,
        )
        return MhsFoldReport(
            fold_index=fold_index,
            validation_start=str(vs),
            validation_end=str(ve),
            strict=primary,
            stress=stress,
            primary_valid=primary.ledger.primary_valid,
            primary_autocorr_sharpe=primary_autocorr,
            primary_naive_sharpe=_naive_sharpe(primary.ledger),
            primary_net_ann=_mean_ann(net_returns_1h, _PERIODS_PER_YEAR_1H),
            primary_geometric_cagr=_geometric_cagr(equity_1h),
            primary_max_drawdown=_mdd(equity),
            stress_naive_sharpe=stress_sharpe,
            decision_intents=decision_intents,
            termination_counts=dict(primary.termination_counts),
            failures=tuple(sorted(set(failures))),
            strict_elapsed_seconds=primary.elapsed_seconds,
            stress_elapsed_seconds=stress.elapsed_seconds,
            terminal_censored_decisions=terminal_censored,
            slow_horizon_hours=(
                slow_horizon_override
                if slow_horizon_override is not None
                else PHASE_1_BOOK_SPECS["slow_momentum"].horizon_hours
            ),
            slow_horizon_source=(
                "fold_train_only_discovery" if slow_horizon_override is not None else "frozen_default"
            ),
            fast_horizon_hours=(
                fast_horizon_override[0]
                if fast_horizon_override is not None
                else PHASE_1_BOOK_SPECS["fast_reversal"].horizon_hours
            ),
            fast_horizon_source=(
                fast_horizon_override[1] if fast_horizon_override is not None else "frozen_default"
            ),
        )
    except DataIntegrityError as exc:
        return _incomplete_fold_report(fold, fold_index, (_classify_execution_failure(exc),))
    except (RuntimeError, ValueError):
        return _incomplete_fold_report(fold, fold_index, (MHS_GO_REASON_INCOMPLETE_FOLD,))

def _run_folds_parallel(
    root: str,
    request: MhsDiagnosticRequest,
    fold_funding: dict[str, pd.Series],
    initial_equity: float,
    telemetry: _StageRecorder | None = None,
    fold_slow_horizons: dict[int, int | None] | None = None,
    fold_fast_horizons: dict[int, tuple[int, str]] | None = None,
) -> tuple[MhsFoldReport, ...]:
    """Run the three anchored folds concurrently, one process each.

    Each fold builds its own 1h panel and executes an independent strict/stress
    replay pair, so the folds are embarrassingly parallel.  ``ProcessPoolExecutor``
    (fork) keeps each worker's RSS independent and bounded: three workers at a
    measured peak of ~2.6GB each stay well inside the 8GB soft budget.  The
    ``MhsFoldReport`` returned by every worker is picklable (frozen+slots,
    holding only pd.Series/pd.DataFrame/numpy/native types), and per-worker
    telemetry is recorded by the parent after each fold completes.  A fold that
    cannot be replayed is reported (not raised) with machine-readable failure
    codes, matching the sequential path.

    ``fork`` (not ``spawn``) is required: spawn workers re-import the module and
    lose the caller's monkeypatched ``funding_path``/``_mark_price_path`` (used
    by the synthetic-market test suite and reproducible diagnostic fixtures),
    and the Phase-1 11.4GiB RSS regression was traced to the main process's own
    top-level matrices and minute-frame retention, not to fork-COW sharing, so
    spawn would not reduce it.
    """
    folds = phase_1_anchored_purged_folds()
    if not folds:
        return ()
    reports: dict[int, MhsFoldReport] = {}
    max_workers = min(3, len(folds))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _run_anchored_fold,
                root, fold, request, fold_funding, initial_equity, idx, None,
                (fold_slow_horizons or {}).get(idx),
                (fold_fast_horizons or {}).get(idx),
            ): idx
            for idx, fold in enumerate(folds)
        }
        for future in as_completed(futures):
            idx = futures[future]
            reports[idx] = future.result()
    ordered = tuple(reports[i] for i in range(len(folds)))
    if telemetry is not None:
        for fold_report in ordered:
            fill_count = (
                len(fold_report.strict.simulated_fills) + len(fold_report.stress.simulated_fills)
                if fold_report.strict is not None and fold_report.stress is not None
                else 0
            )
            telemetry.record(f"anchored_fold_{fold_report.fold_index}", fill_count=fill_count)
    return ordered


def _per_observation_sharpe(returns: pd.Series) -> float:
    """Per-observation (non-annualized) sample Sharpe of a return series.

    ``mean / std`` with no ``sqrt(periods_per_year)`` scaling, matching the
    per-observation input contract of ``probabilistic_sharpe_ratio``/
    ``deflated_sharpe_ratio``. Degenerate zero-variance returns NaN so a
    non-finite observed Sharpe never reaches the deflation statistic.
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return float("nan")
    sd = float(returns.std(ddof=1))
    if sd <= 0.0:
        return float("nan")
    return float(returns.mean() / sd)


def _deflated_sharpe_evidence(
    blend_report: MhsBookReport | None,
    folds: tuple[MhsFoldReport, ...],
    n_trials: int,
) -> float | None:
    """Per-observation deflated Sharpe of the blend primary against the
    anchored-fold trial dispersion (docs/specs/mhs_strategy_foundation_reset.md RC-6).

    The observed trial is the blend primary ledger's per-observation Sharpe; the
    trial dispersion is the sample variance of the per-observation Sharpe across
    the anchored folds (each fold is one independent validation replay), and the
    number of trials is the preregistered ``MHS_SEARCH_TRIALS_ATTEMPTED``
    constant. Returns None when there is no replayable blend primary or no fold
    trial; the pure ``deflated_sharpe_ratio`` statistic remains the single
    source of the Bailey-LdP arithmetic.
    """
    if blend_report is None or blend_report.primary is None or not folds:
        return None
    _equity_1h, net_returns_1h, _turnover = _hourly_ledger_series(
        blend_report.primary.ledger.equity,
        blend_report.primary.ledger.fill_turnover,
    )
    observed_sr = _per_observation_sharpe(net_returns_1h)
    if not np.isfinite(observed_sr):
        return None
    trial_sharpes: list[float] = []
    for fold in folds:
        if fold.strict is None or fold.failures:
            continue
        _fold_equity, fold_net, _fold_turnover = _hourly_ledger_series(
            fold.strict.ledger.equity, fold.strict.ledger.fill_turnover,
        )
        trial_sharpes.append(_per_observation_sharpe(fold_net))
    if not trial_sharpes:
        return None
    trial_variance = (
        float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) >= 2 else 0.0
    )
    returns = net_returns_1h.dropna()
    if len(returns) < 2:
        return None
    result = deflated_sharpe_ratio(
        observed_sr,
        trial_variance,
        n_trials,
        len(returns),
        float(returns.skew()),
        float(returns.kurt()) + 3.0,
    )
    # Fail closed: a degenerate skew/kurtosis can push the statistic to NaN,
    # which must never leak into the report payload as a real deflated value.
    return result if np.isfinite(result) else None


def _mhs_research_go(
    folds: tuple[MhsFoldReport, ...],
    book_reasons: tuple[str, ...] = (),
) -> MhsResearchGoResult:
    """Fail-closed top-level Research-GO decision from fold and book evidence.

    A fold that was not replayed, an invalid primary, non-finite equity, a
    relevant execution gap, a strict-Sharpe failure, or a non-positive stress
    Sharpe each block the decision with a stable reason code. A book-level
    strict replay rejection (capital invariant breach, execution gap, invalid
    primary, or resource-budget breach) is aggregated with the fold reasons.
    The cap-30 roster and primary annual-return gate thresholds live in
    ``MHS_REGISTERED_POLICY_THRESHOLDS``; while any is unregistered (``None``)
    the decision reports ``UNSPECIFIED_POLICY`` and stays conservative (false).
    Once every threshold is deliberately registered, ``eligible`` responds to
    the fold evidence alone (docs/specs/mhs_strategy_foundation_reset.md RC-5).
    """
    reasons: list[str] = list(book_reasons)
    passed = 0
    for fold_report in folds:
        if fold_report.strict is None:
            reasons.append(MHS_GO_REASON_INCOMPLETE_FOLD)
            continue
        if not fold_report.failures:
            passed += 1
        reasons.extend(fold_report.failures)
    # P0-D: UNSPECIFIED_POLICY only when a registered policy threshold is absent.
    # Registering every threshold makes the gate reachable, so ``eligible`` can
    # respond to actual evidence instead of being structurally always False.
    if any(v is None for v in MHS_REGISTERED_POLICY_THRESHOLDS.values()):
        reasons.append(MHS_GO_REASON_UNSPECIFIED_POLICY)
    reasons = sorted(set(reasons))
    return MhsResearchGoResult(
        eligible=not reasons,
        reason_codes=tuple(reasons),
        evaluated_folds=len(folds),
        folds_passed=passed,
    )


def run_mhs_horizon_diagnostic(request: MhsDiagnosticRequest) -> MhsHorizonDiagnosticReport:
    """Compose the dev-only Phase 1 diagnostic: pre-screen + strict-proxy evidence.

    Forces ``partition='dev'`` and resolves the sealed evaluation end; a holdout
    partition or an end past ``HOLDOUT_CUTOFF`` raises ``RuntimeError``. The
    diagnostic-ensemble and executable-tranche numbers are reported as separate
    fields; only strict simulated inventory is primary Research evidence.
    """
    # Each diagnostic run reads a static market snapshot; dropping the minute
    # frame caches here keeps re-runs against re-written Parquet sources fresh.
    _load_minute_frames_cached.cache_clear()
    _get_symbol_minute_frame.cache_clear()
    resolved_end = resolve_evaluation_end(request.end, unseal_holdout=False)
    _run_start = time.perf_counter()
    if request.partition != "dev":
        raise RuntimeError(
            "MHS Phase 1 is dev-only; the holdout partition requires an "
            "architecture-freeze final-OOS command"
        )
    if request.start is not None:
        start = pd.Timestamp(request.start)
        start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
    else:
        start = MHS_DISCOVERY_START

    if resolved_end is not None:
        end = pd.Timestamp(resolved_end)
        end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
    else:
        end = HOLDOUT_CUTOFF
    if end > HOLDOUT_CUTOFF:
        raise RuntimeError(f"Holdout sealed: requested end {end} past {HOLDOUT_CUTOFF}")

    telemetry = _StageRecorder(log_run=request.log_run)

    root = request.data_root or str(FUTURES_DATA_DIR / "ohlcv")
    panel = load_base_panel(
        root, "1h", ("close", "open", "quote_vol"), start, end, partition="dev", min_bars=2000,
    )
    close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    grid_1h = close.index
    symbols = list(close.columns)
    telemetry.record("base_1h_panel", grid_bars=len(grid_1h), n_symbols=len(symbols))

    funding_by_symbol = _load_funding_series(symbols)
    fold_funding = dict(funding_by_symbol)
    funded = [
        s for s in symbols
        if s in funding_by_symbol and s not in MHS_SOURCE_GAP_EXCLUDED_SYMBOLS
    ]
    if not funded:
        raise RuntimeError("no dev symbol has funding coverage; the MHS ledger requires funding")
    close = close[funded]
    opens = opens[funded]
    quote_vol = quote_vol[funded]
    bar_period = grid_1h[1] - grid_1h[0]
    funding_window = {
        s: funding_by_symbol[s].loc[
            (funding_by_symbol[s].index >= grid_1h[0])
            & (funding_by_symbol[s].index < grid_1h[-1] + bar_period)
        ]
        for s in funded
    }
    bar_funding = bar_funding_panel(funding_window, grid_1h)
    aligned_symbols = list(bar_funding.columns)
    if not aligned_symbols:
        raise RuntimeError("no dev symbol has causally aligned funding coverage")
    close = close[aligned_symbols]
    opens = opens[aligned_symbols]
    quote_vol = quote_vol[aligned_symbols]
    funding_by_symbol = {s: funding_by_symbol[s] for s in aligned_symbols}
    bar_funding = bar_funding[aligned_symbols]
    telemetry.record("funding_alignment", grid_bars=len(grid_1h), n_symbols=len(aligned_symbols))

    eligible = liquid_half_eligibility(quote_vol, lookback_bars=720, min_history_bars=720)
    log_close = np.log(close)
    # The raw close panel is not used after its log transform.  Releasing it
    # before phase/weight construction avoids retaining two full multi-year
    # price matrices at once.
    del close

    specs = PHASE_1_BOOK_SPECS
    fast = specs["fast_reversal"]
    slow = specs["slow_momentum"]
    # Fold-safe horizon selection (spec §1.5, ``wiring``): computed once in the
    # parent before either the top-level books or the fold pool are forked,
    # reusing the already-loaded full-period panel. Only the resolved plain
    # horizon ``int`` (or None) is passed down to fold workers, so no worker
    # ever reloads a wide ``[train_start, train_end]`` panel.
    fold_slow_horizons: dict[int, int | None] = {}
    fold_fast_horizons: dict[int, tuple[int, str]] = {}
    if request.fold_safe_horizon_selection:
        _precomputed_weights = build_candidate_weights(
            log_close, eligible, 1, specs["slow_momentum"].band.horizons_hours,
            tranche_count=MHS_DISCOVERY_GATE_TRANCHE_COUNT,
        )
        for _fold_idx, _fold in enumerate(phase_1_anchored_purged_folds()):
            _spec, _horizon, _source = _fold_safe_slow_book_spec(
                fold_train_only_discovery_qualification(
                    sign=1,
                    horizon_candidates=specs["slow_momentum"].band.horizons_hours,
                    log_close=log_close, eligible=eligible, opens=opens,
                    bar_funding=bar_funding, grid_1h=grid_1h, fold=_fold,
                    tranche_count=MHS_DISCOVERY_GATE_TRANCHE_COUNT,
                    precomputed_candidate_weights=_precomputed_weights,
                ),
                specs["slow_momentum"],
            )
            fold_slow_horizons[_fold_idx] = (
                _horizon if _source == "fold_train_only_discovery" else None
            )
        _fast_precomputed_weights = build_candidate_weights(
            log_close, eligible, -1, specs["fast_reversal"].band.horizons_hours,
            tranche_count=MHS_DISCOVERY_GATE_TRANCHE_COUNT,
        )
        for _fold_idx, _fold in enumerate(phase_1_anchored_purged_folds()):
            fold_fast_horizons[_fold_idx] = _fold_safe_fast_horizon(
                fold_train_only_discovery_qualification(
                    sign=-1,
                    horizon_candidates=specs["fast_reversal"].band.horizons_hours,
                    log_close=log_close, eligible=eligible, opens=opens,
                    bar_funding=bar_funding, grid_1h=grid_1h, fold=_fold,
                    tranche_count=MHS_DISCOVERY_GATE_TRANCHE_COUNT,
                    precomputed_candidate_weights=_fast_precomputed_weights,
                ),
                fast.horizon_hours,
            )
        # The top-level report uses fold index 2's selection (train=2021-2024,
        # the widest leak-free window that still excludes 2025), making the
        # full-period report's horizon choice walk-forward-safe relative to
        # 2025 without a second, redundant discovery scan.
        top_level_horizon = fold_slow_horizons.get(2)
        if top_level_horizon is not None:
            slow = dataclasses.replace(slow, horizon_hours=top_level_horizon)
    fast_grid = pd.date_range(start, end, freq="6h", tz="UTC")
    slow_grid = pd.date_range(start, end, freq="24h", tz="UTC")

    fast_ema = _signal_ema_span(fast.band.sign, fast.horizon_hours, fast.step_hours)
    slow_ema = _signal_ema_span(slow.band.sign, slow.horizon_hours, slow.step_hours)
    w_fast = _book_weights(log_close, eligible, fast, fast_grid, ema_span=fast_ema)
    w_slow = _book_weights(log_close, eligible, slow, slow_grid, ema_span=slow_ema)
    w_fast_1h = w_fast.reindex(grid_1h).ffill().fillna(0.0)
    w_slow_1h = w_slow.reindex(grid_1h).ffill().fillna(0.0)
    execution_mask = _pit_execution_mask(quote_vol, eligible, request.execution_universe_size)
    realized_execution_roster_size = float(execution_mask.sum(axis=1).mean())
    w_fast_tilted = inverse_realized_vol_tilt(
        w_fast, realized_vol(log_close, fast.horizon_hours).reindex(fast_grid),
    )
    w_fast_execution = renormalize_within_mask(
        w_fast_tilted, execution_mask.reindex(w_fast.index).fillna(False), fast.min_symbols,
    )
    w_slow_execution = _slow_book_execution_weights(
        log_close, eligible, execution_mask, slow, slow_grid,
        request.slow_book_mode, request.ensemble_signal, slow_ema,
    )
    if request.beta_neutralize:
        w_slow_execution = beta_neutralize_weights(
            w_slow_execution,
            causal_market_beta(
                log_close, eligible,
                MHS_CAUSAL_BETA_LOOKBACK_BARS, MHS_CAUSAL_BETA_MIN_PERIODS,
            ).reindex(w_slow_execution.index),
            execution_mask.reindex(w_slow_execution.index).fillna(False),
            slow.min_symbols,
        )
    del w_fast_tilted
    # Eligibility and the execution roster are now materialized.  The raw
    # volume matrix otherwise stays alive while phase diagnostics create their
    # temporary target-weight matrices.
    del quote_vol
    blend_1h = (
        PHASE_1_BOOK_BLEND_WEIGHTS["fast_reversal"] * w_fast_1h
        + PHASE_1_BOOK_BLEND_WEIGHTS["slow_momentum"] * w_slow_1h
    )
    blend_gross = float(blend_1h.abs().sum(axis=1).mean())
    blend_cash_fraction = float((1.0 - blend_1h.abs().sum(axis=1)).mean())
    # R1: apply the same volatility-regime cash scale the fold path applies to
    # its blended targets (_build_fold_target_weights) so top-level prescreen/
    # tail/execution diagnostics are comparable to fold primary evidence
    # (spec §3.2, ``regime_cash_scale``).
    vol_mean = realized_vol(log_close, 48).where(execution_mask).reindex(grid_1h).mean(axis=1)
    regime_scale = _regime_cash_scale(vol_mean)
    if request.trend_efficiency_overlay:
        regime_scale = regime_scale.mul(
            _trend_efficiency_overlay_scale(log_close, execution_mask, fast.horizon_hours, grid_1h),
        )
    blend_1h = blend_1h.mul(regime_scale, axis=0)
    del vol_mean
    # The 1h book views are only consumed by ``blend_1h`` above.  Releasing
    # them before phase diagnostics and the top-level replays keeps two full
    # multi-year weight matrices out of the replay baseline (spec §3.1).
    del w_fast_1h, w_slow_1h
    gc.collect()

    phase_fast = _phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, fast)
    phase_slow = _phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, slow)
    _blend_spec, _blend_grid = _active_blend_book_and_grid(fast, slow, fast_grid, slow_grid)
    del _blend_grid
    phase_blend = _phase_diagnostics(log_close, eligible, opens, bar_funding, grid_1h, _blend_spec)

    # R3: the 48h cross-sectional statistics depend only on the 1h panel, not
    # the book replays.  Computing them here -- and computing ``signal_48h``
    # once so the placebo reuses it -- lets ``log_close`` be released before the
    # three top-level replays instead of staying alive throughout them
    # (spec §3.1, ``memory_opt``).
    signal_48h = horizon_log_return(log_close, 48)
    xs_ic = _xs_rank_ic(signal_48h, opens, forward_bars=48)
    regression = _date_clustered_ols(opens, signal_48h, forward_bars=48)
    horizon_diagnostics = {
        "realized_vol_48h_mean": float(
            realized_vol(log_close, 48).mean().mean()
        ),
        "efficiency_ratio_48h_mean": float(
            efficiency_ratio(log_close, 48).mean().mean()
        ),
    }
    discovery_qualification = None
    if request.discovery_gate:
        discovery_qualification = {
            "reversal": select_horizon_by_discovery_qualification(
                sign=-1, horizon_candidates=MHS_DISCOVERY_REVERSAL_CANDIDATES,
                log_close=log_close, eligible=eligible, opens=opens,
                bar_funding=bar_funding, grid_1h=grid_1h,
                discovery_start=MHS_DISCOVERY_START, discovery_end=DISCOVERY_END,
                qualification_end=QUALIFICATION_END,
                tranche_count=MHS_DISCOVERY_GATE_TRANCHE_COUNT,
            ),
            "momentum": select_horizon_by_discovery_qualification(
                sign=1, horizon_candidates=MHS_DISCOVERY_MOMENTUM_CANDIDATES,
                log_close=log_close, eligible=eligible, opens=opens,
                bar_funding=bar_funding, grid_1h=grid_1h,
                discovery_start=MHS_DISCOVERY_START, discovery_end=DISCOVERY_END,
                qualification_end=QUALIFICATION_END,
                tranche_count=MHS_DISCOVERY_GATE_TRANCHE_COUNT,
            ),
        }
    del log_close
    gc.collect()

    execution_symbols = sorted(
        set(w_fast_execution.columns[w_fast_execution.ne(0.0).any(axis=0)])
        | set(w_slow_execution.columns[w_slow_execution.ne(0.0).any(axis=0)])
    )
    initial_equity = 1.0
    minute_grid = pd.date_range(
        start, end,
        freq={"1m": "1min", "5m": "5min"}[request.execution_timeframe],
        tz="UTC",
    )
    has_minute_data = any(
        os.path.exists(os.path.join(root, request.execution_timeframe, f"{s}.parquet"))
        for s in execution_symbols
    )
    if has_minute_data and execution_symbols:
        telemetry.record(
            "minute_market_mark_funding",
            grid_bars=len(minute_grid),
            n_symbols=len(execution_symbols),
        )
        # Populate the O6 minute-frame cache before forking so the three book
        # workers share one set of full-period frames via copy-on-write, then
        # run the three books concurrently in fork children (spec Phase 3, P10).
        _preload_symbol_minute_frames(root, execution_symbols, request.execution_timeframe)
        _data_collector()
        book_report_fast, book_report_slow, book_report_blend = _run_books_concurrent(
            root, request, len(funded), grid_1h, fast, slow, fast_grid, slow_grid,
            w_fast, w_slow, w_fast_execution, w_slow_execution, opens, bar_funding,
            phase_fast, phase_slow, phase_blend, start, end, funding_by_symbol,
            blend_1h, execution_mask, initial_equity, telemetry, regime_scale,
        )
        # All three books have completed; the single-use step-weight inputs are
        # released together (spec §3.1, ``memory_opt``).
        del w_fast, w_fast_execution, phase_fast
        del w_slow, w_slow_execution, phase_slow
        del blend_1h, execution_mask, phase_blend, regime_scale
        gc.collect()
        books = {"fast_reversal": book_report_fast, "slow_momentum": book_report_slow}
        blend_report = book_report_blend
    else:
        books = {}
        blend_report = None

    book_reasons = tuple(
        sorted(
            b.failure.reason
            for b in [*books.values(), blend_report]
            if b is not None and b.failure is not None
        )
    )

    trials_attempted = MHS_SEARCH_TRIALS_ATTEMPTED
    deflated_sharpe_ratio = None

    bootstrap_ci: tuple[float, float] | None = None
    placebo_percentile: float | None = None
    participation: dict[str, float] = {}
    termination_counts: dict[str, int] = {}
    unsupported = ("partial_fill", "queue_position", "post_only_rejection", "cancel_replace_latency", "order_size_impact")

    # Folds, statistical diagnostics, and deployment readiness are independent
    # post-book streams: the fold pool runs in fork workers while a background
    # thread computes the diagnostics + deployment tail (spec Phase 3, P14).
    # The top-level feature matrices stay alive through that thread and are
    # released after it joins so the wide multi-year panels never coexist with
    # the final assembly.
    (
        bootstrap_ci, placebo_percentile, participation, termination_counts,
        fold_reports, deployment,
    ) = _run_post_book_concurrently(
        blend_report, root, request, execution_symbols, minute_grid,
        signal_48h, eligible, opens, bar_funding, grid_1h, fast,
        fold_funding, initial_equity, telemetry, fold_slow_horizons, fold_fast_horizons,
    )
    folds = tuple(fold_reports)
    deflated_sharpe_ratio = _deflated_sharpe_evidence(
        blend_report, folds, trials_attempted,
    )
    research_go = _mhs_research_go(folds, book_reasons)

    if blend_report is not None and blend_report.primary is not None:
        if minute_grid is None:
            raise DataIntegrityError("blend report requires a minute replay grid")
        # The deployment tail was computed with ``research_go_eligible=None``;
        # patch in the fold-derived gate decision now that it is resolved.
        assert deployment is not None
        deployment = dataclasses.replace(
            deployment, research_go_eligible=research_go.eligible,
        )
        telemetry.record(
            "blend_participation",
            fill_count=len(blend_report.primary.simulated_fills),
        )
        telemetry.record("statistical_diagnostics")
    else:
        deployment = compute_deployment_readiness(
            pd.Series(
                [1.0, 1.0],
                index=pd.DatetimeIndex([start, start + pd.Timedelta(hours=1)]),
            ),
            _PERIODS_PER_YEAR_1H,
            research_go_eligible=research_go.eligible,
            n_bootstrap=_BOOTSTRAP_REPLICATES,
        )

    del eligible
    del opens, bar_funding
    del funding_window, minute_grid
    gc.collect()

    synthetic_stress = {s.name: {"description": s.description} for s in synthetic_stress_scenarios()}

    mark_source = "NOT_RUN_NO_EXECUTION_DATA"
    fill_source = "NOT_RUN_NO_EXECUTION_DATA"
    if blend_report is not None and blend_report.primary is not None:
        mark_source = blend_report.primary.ledger.mark_source
        fill_source = "OHLCV_IMMEDIATE_TAKER"

    run_elapsed_seconds = time.perf_counter() - _run_start
    telemetry.record("final_return")

    return MhsHorizonDiagnosticReport(
        feature=_MHS_FEATURE,
        status="COMPLETE",
        start=str(start),
        end=str(end),
        resolved_end=str(resolved_end),
        partition="dev",
        execution_tiers_bps=required_cost_tiers(),
        books=books,
        blend=blend_report,
        blend_target_gross=blend_gross,
        blend_cash_fraction=blend_cash_fraction,
        eligible_symbols=len(funded),
        trials_attempted=trials_attempted,
        deflated_sharpe_ratio=deflated_sharpe_ratio,
        xs_rank_ic=xs_ic,
        date_clustered_regression=regression,
        horizon_diagnostics=horizon_diagnostics,
        bootstrap_ci=bootstrap_ci,
        placebo_sharpe_percentile=placebo_percentile,
        deployment_readiness=deployment,
        synthetic_stress=synthetic_stress,
        participation_warnings=participation,
        termination_counts=termination_counts,
        unsupported_assumptions=unsupported,
        anchored_folds=phase_1_anchored_purged_folds(),
        folds=folds,
        research_go=research_go,
        fill_source=fill_source,
        mark_source=mark_source,
        execution_timeframe=request.execution_timeframe,
        execution_universe_size=request.execution_universe_size,
        execution_symbols=tuple(execution_symbols),
        run_elapsed_seconds=run_elapsed_seconds,
        resource_measurements=telemetry.records,
        discovery_qualification=discovery_qualification,
        realized_execution_roster_size=realized_execution_roster_size,
    )


def mhs_horizon_diagnostic_report_path() -> str:
    """Single source-controlled report path, sibling to the other ``*_report_path`` helpers."""
    return str(Path("docs/results") / "mhs_horizon_diagnostic.json")


def persist_mhs_horizon_diagnostic_report(
    report: MhsHorizonDiagnosticReport,
    path: str | Path,
    tier: MhsOutputTier = MhsOutputTier.COMPACT,
) -> Path | None:
    """Persist the MHS diagnostic in the requested output tier.

    COMPACT (default) writes a git-committable stripped summary JSON at ``path``
    plus a daily-resampled ``daily_ledger.parquet`` under the sibling
    ``*_artifacts`` directory; per-fill detail is intentionally dropped.
    FULL writes the lossless 5-category unified Parquet audit tables and a
    verbose checksummed JSON under ``*_artifacts/_full/`` (gitignored), keeping
    the pre-tiering behaviour byte-for-byte otherwise.

    Returns the persisted JSON path, or ``None`` when a COMPACT resample
    failure is escalated past the compact artifacts (fail-closed policy).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if tier == MhsOutputTier.FULL:
        return _persist_mhs_report_full(report, target)
    return _persist_mhs_report_compact(report, target)


def _collect_replay_entries(
    report: MhsHorizonDiagnosticReport,
) -> list[tuple[str, StrategyExecutionReplayResult]]:
    """Stable ordered replay sessions (books, blend, folds) for persistence."""
    replay_entries: list[tuple[str, StrategyExecutionReplayResult]] = []
    for book_name, book_report in report.books.items():
        if book_report.primary is not None:
            replay_entries.append((f"{book_name}_primary", book_report.primary))
        if book_report.stress is not None:
            replay_entries.append((f"{book_name}_stress", book_report.stress))
        if book_report.patient_reference is not None:
            replay_entries.append((f"{book_name}_patient_reference", book_report.patient_reference))
        if book_report.pre_vol_target_reference is not None:
            replay_entries.append(
                (f"{book_name}_pre_vol_target_reference", book_report.pre_vol_target_reference)
            )
    if report.blend is not None:
        if report.blend.primary is not None:
            replay_entries.append(("blend_primary", report.blend.primary))
        if report.blend.stress is not None:
            replay_entries.append(("blend_stress", report.blend.stress))
        if report.blend.patient_reference is not None:
            replay_entries.append(("blend_patient_reference", report.blend.patient_reference))
        if report.blend.pre_vol_target_reference is not None:
            replay_entries.append(("blend_pre_vol_target_reference", report.blend.pre_vol_target_reference))
    for fold_report in report.folds:
        if fold_report.strict is not None:
            replay_entries.append((f"fold{fold_report.fold_index}_strict", fold_report.strict))
        if fold_report.stress is not None:
            replay_entries.append((f"fold{fold_report.fold_index}_stress", fold_report.stress))
    return replay_entries


def _write_json_report(path: Path, payload: Any) -> None:
    """Serialize ``payload`` to ``path`` preferring orjson over stdlib json."""
    with path.open("w", encoding="utf-8") as fh:
        try:
            import orjson

            fh.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8"))
        except ImportError:
            json.dump(payload, fh, indent=2, ensure_ascii=False)


def _persist_mhs_report_full(
    report: MhsHorizonDiagnosticReport,
    target: Path,
) -> Path:
    """Lossless tier: the pre-tiering 5-category unified audit tables + JSON.

    Artifacts and the verbose report land under ``*_artifacts/_full/`` so the
    compact daily ledger at the artifact root stays git-trackable. The JSON
    carries the full per-replay SHA-256/checksum references exactly as before.
    """
    artifact_root = target.parent / f"{target.stem}_artifacts" / "_full"
    artifact_root.mkdir(parents=True, exist_ok=True)
    payload = report.to_payload()
    replay_entries = _collect_replay_entries(report)

    tables_by_replay: dict[str, dict[str, pd.DataFrame]] = {}
    for replay_id, replay in replay_entries:
        tables_by_replay[replay_id] = _build_replay_category_tables(replay)

    unified_tables = _write_unified_artifact_tables(tables_by_replay, artifact_root)

    # Keep the FULL artifact directory at exactly the 5 canonical unified tables:
    # superseded per-replay Parquet files from earlier persistence formats are
    # removed so re-persisting to the same directory never leaves orphans.
    canonical_names = {f"{category}.parquet" for category in MHS_ARTIFACT_CATEGORIES}
    for stale in artifact_root.glob("*.parquet"):
        if stale.name not in canonical_names:
            stale.unlink()

    # Fail-closed ledger integrity verification per replay_id partition.
    for replay_id, _replay in replay_entries:
        _verify_ledger_artifact(
            unified_tables["ledger"][0], replay_id, len(tables_by_replay[replay_id]["ledger"])
        )

    replay_references = {
        replay_id: _build_replay_artifact_reference(
            replay_id, replay, tables_by_replay[replay_id], artifact_root, unified_tables
        )
        for replay_id, replay in replay_entries
    }

    for book_name, book_report in report.books.items():
        book_payload = payload["books"][book_name]
        if book_report.primary is not None:
            book_payload["primary"] = replay_references[f"{book_name}_primary"]
        if book_report.stress is not None:
            book_payload["stress"] = replay_references[f"{book_name}_stress"]
    if report.blend is not None:
        if report.blend.primary is not None:
            payload["blend"]["primary"] = replay_references["blend_primary"]
        if report.blend.stress is not None:
            payload["blend"]["stress"] = replay_references["blend_stress"]
    for fold_report in report.folds:
        fold_payload = payload["folds"][fold_report.fold_index]
        if fold_report.strict is not None:
            fold_payload["strict"] = replay_references[f"fold{fold_report.fold_index}_strict"]
        if fold_report.stress is not None:
            fold_payload["stress"] = replay_references[f"fold{fold_report.fold_index}_stress"]

    payload["artifacts"] = {
        category: _artifact_reference(frame, path)
        for category, (path, frame) in unified_tables.items()
    }
    payload["replay_ids"] = [replay_id for replay_id, _ in replay_entries]

    report_path = artifact_root / "report.json"
    _write_json_report(report_path, payload)
    _logger.info("[MHS] full report persisted path=%s", report_path)
    return report_path


def _replay_category_row_counts(replay: StrategyExecutionReplayResult) -> dict[str, int]:
    """Cheap per-category row counts straight off the replay (no table build)."""
    return {
        "fills": len(replay.simulated_fills),
        "units": len(replay.simulated_units),
        "notional_weights": len(replay.simulated_notional_weights),
        "ledger": len(replay.ledger.equity),
        "times": len(replay.submit_times),
    }


def _ledger_table(replay: StrategyExecutionReplayResult) -> pd.DataFrame:
    """Minimal timestamped ledger table (timestamp, equity, fill_turnover)."""
    equity = replay.ledger.equity
    idx = equity.index
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(idx, utc=True),
            "equity": equity.to_numpy(dtype="float64"),
            "fill_turnover": replay.ledger.fill_turnover.reindex(idx).to_numpy(dtype="float64"),
        }
    )


def _compact_replay_ref(row_counts: dict[str, int]) -> dict[str, dict[str, int]]:
    """Stripped per-replay reference: category -> row_count only."""
    return {category: {"row_count": row_counts[category]} for category in MHS_ARTIFACT_CATEGORIES}


def _wire_compact_refs(
    payload: Any,
    report: MhsHorizonDiagnosticReport,
    replay_entries: list[tuple[str, StrategyExecutionReplayResult]],
    row_counts: dict[str, dict[str, int]],
) -> None:
    """Replace verbose per-replay artifact references with row-count stubs."""
    for book_name, book_report in report.books.items():
        book_payload = payload["books"][book_name]
        if book_report.primary is not None:
            book_payload["primary"] = _compact_replay_ref(row_counts[f"{book_name}_primary"])
        if book_report.stress is not None:
            book_payload["stress"] = _compact_replay_ref(row_counts[f"{book_name}_stress"])
        if book_report.patient_reference is not None:
            book_payload["patient_reference"] = _compact_replay_ref(
                row_counts[f"{book_name}_patient_reference"]
            )
        if book_report.pre_vol_target_reference is not None:
            book_payload["pre_vol_target_reference"] = _compact_replay_ref(
                row_counts[f"{book_name}_pre_vol_target_reference"]
            )
    if report.blend is not None:
        if report.blend.primary is not None:
            payload["blend"]["primary"] = _compact_replay_ref(row_counts["blend_primary"])
        if report.blend.stress is not None:
            payload["blend"]["stress"] = _compact_replay_ref(row_counts["blend_stress"])
        if report.blend.patient_reference is not None:
            payload["blend"]["patient_reference"] = _compact_replay_ref(
                row_counts["blend_patient_reference"]
            )
        if report.blend.pre_vol_target_reference is not None:
            payload["blend"]["pre_vol_target_reference"] = _compact_replay_ref(
                row_counts["blend_pre_vol_target_reference"]
            )
    for fold_report in report.folds:
        fold_payload = payload["folds"][fold_report.fold_index]
        if fold_report.strict is not None:
            fold_payload["strict"] = _compact_replay_ref(
                row_counts[f"fold{fold_report.fold_index}_strict"]
            )
        if fold_report.stress is not None:
            fold_payload["stress"] = _compact_replay_ref(
                row_counts[f"fold{fold_report.fold_index}_stress"]
            )


def _persist_mhs_report_compact(
    report: MhsHorizonDiagnosticReport,
    target: Path,
) -> Path | None:
    """Compact tier: daily-resampled ledger Parquet + stripped summary JSON.

    The daily rollup is written first; a fail-closed ``DataIntegrityError`` on
    non-finite equity propagates, while any other resample failure logs and
    escalates past compact persistence (returns ``None``).
    """
    artifact_root = target.parent / f"{target.stem}_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    replay_entries = _collect_replay_entries(report)

    row_counts: dict[str, dict[str, int]] = {}
    daily_frames: list[pd.DataFrame] = []
    for replay_id, replay in replay_entries:
        row_counts[replay_id] = _replay_category_row_counts(replay)
        try:
            daily = _daily_resample_ledger(_ledger_table(replay))
        except DataIntegrityError:
            raise
        except Exception:  # noqa: BLE001
            _logger.error(
                "[MHS] compact daily resample failed replay_id=%s", replay_id, exc_info=True
            )
            return None
        tagged = daily.copy()
        tagged.insert(0, "replay_id", replay_id)
        daily_frames.append(tagged)

    daily_table = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(
        {"replay_id": pd.Series(dtype="string")}
    )
    daily_path = artifact_root / "daily_ledger.parquet"
    daily_table.to_parquet(daily_path, index=False, compression="snappy")

    payload = report.to_payload()
    _wire_compact_refs(payload, report, replay_entries, row_counts)
    unified_row_counts = {
        category: sum(rc[category] for rc in row_counts.values())
        for category in MHS_ARTIFACT_CATEGORIES
    }
    payload["artifacts"] = {
        category: {"file": f"{category}.parquet", "row_count": unified_row_counts[category]}
        for category in MHS_ARTIFACT_CATEGORIES
    }
    payload["artifacts"]["daily_ledger"] = {
        "file": daily_path.name,
        "row_count": len(daily_table),
    }
    payload["replay_ids"] = [replay_id for replay_id, _ in replay_entries]

    _write_json_report(target, payload)
    size = target.stat().st_size
    if size > 50_000:
        _logger.warning(
            "[MHS] compact report exceeds 50KB size=%d path=%s", size, target
        )
    _logger.info("[MHS] compact report persisted path=%s", target)
    return target


def _to_timestamped_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Promote a DatetimeIndex into an explicit UTC timestamp column.

    The returned table carries a physical ``datetime64[ns, UTC]`` column named
    ``timestamp`` and a RangeIndex, so readers need no string-parsing guess.
    """
    out = frame.copy()
    if len(out):
        out.insert(0, "timestamp", out.index)
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    else:
        out["timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
    return out.reset_index(drop=True)


def _artifact_checksum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _artifact_reference(table: pd.DataFrame, path: Path) -> dict[str, Any]:
    """Row count, time bounds, schema version, and content checksum per table."""
    ts = (
        pd.to_datetime(table["timestamp"], utc=True)
        if "timestamp" in table.columns
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    return {
        "file": path.name,
        "schema_version": MHS_ARTIFACT_SCHEMA_VERSION,
        "row_count": len(table),
        "time_bounds": {
            "start": None if len(ts) == 0 else str(ts.iloc[0]),
            "end": None if len(ts) == 0 else str(ts.iloc[-1]),
        },
        "checksum_sha256": _artifact_checksum(path),
    }


def _verify_ledger_artifact(path: Path, replay_id: str, expected_rows: int) -> None:
    """Re-read a written unified ledger parquet and verify one replay partition fail-closed.

    Checksum-only provenance cannot detect a silently-written NULL or truncated
    equity column, so this pass re-reads the ``replay_id`` partition via PyArrow
    pushdown filtering and asserts the exact row count and a fully finite
    positive equity column (spec §3.3, ``fold_integrity``).
    """
    roundtrip = pd.read_parquet(path, filters=[("replay_id", "==", replay_id)])
    if len(roundtrip) != expected_rows:
        raise DataIntegrityError(
            f"ledger artifact row count mismatch path={path} replay_id={replay_id} "
            f"expected={expected_rows} got={len(roundtrip)}"
        )
    if expected_rows and "equity" in roundtrip.columns:
        equity = roundtrip["equity"].to_numpy(dtype="float64")
        if not np.isfinite(equity).all() or (equity <= 0).any():
            raise DataIntegrityError(
                f"ledger artifact equity must be finite and strictly positive "
                f"path={path} replay_id={replay_id}"
            )


def _daily_resample_ledger(ledger_table: pd.DataFrame) -> pd.DataFrame:
    """Resample one replay's minute ledger to a daily OHLCV rollup.

    ``ledger_table`` must carry at least ``timestamp``, ``equity`` and
    ``fill_turnover`` columns (the unified ledger schema). One row is emitted
    per UTC day with ``equity_open/high/low/close``, ``daily_return``
    (close/prev_close - 1), ``daily_turnover`` (sum of fill turnover) and
    ``daily_fill_count`` (count of fill-bearing grid rows). Non-finite or
    non-positive equity fails closed with ``DataIntegrityError``.
    """
    required = {"timestamp", "equity", "fill_turnover"}
    missing = required - set(ledger_table.columns)
    if missing:
        raise DataIntegrityError(f"daily resample requires columns {sorted(missing)}")
    frame = ledger_table.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    equity = frame["equity"].to_numpy(dtype="float64")
    if not np.isfinite(equity).all() or (equity <= 0).any():
        raise DataIntegrityError("ledger equity must be finite and strictly positive")
    grouped = frame.groupby(pd.Grouper(key="timestamp", freq="1D"))
    resampled = pd.DataFrame(
        {
            "equity_open": grouped["equity"].first(),
            "equity_high": grouped["equity"].max(),
            "equity_low": grouped["equity"].min(),
            "equity_close": grouped["equity"].last(),
            "daily_turnover": grouped["fill_turnover"].sum(),
            "daily_fill_count": grouped["fill_turnover"].agg(lambda s: int((s > 0).sum())),
        }
    )
    resampled = resampled.rename_axis("date").reset_index()
    resampled["daily_return"] = (
        resampled["equity_close"] / resampled["equity_close"].shift(1) - 1.0
    ).replace([np.inf, -np.inf], np.nan)
    # The daily rollup is a diagnostic aggregate, never a PnL source, so the
    # numeric columns are safely downcast to float32 (validated finite/positive
    # above) to keep the git-tracked compact artifact lean.
    for col in (
        "equity_open", "equity_high", "equity_low", "equity_close",
        "daily_return", "daily_turnover",
    ):
        resampled[col] = resampled[col].astype("float32")
    return resampled


def _build_replay_category_tables(
    replay: StrategyExecutionReplayResult,
) -> dict[str, pd.DataFrame]:
    """Build the five category tables for one replay, without the ``replay_id`` column."""
    fills = replay.simulated_fills.copy()
    if not fills.empty and "timestamp" in fills.columns:
        fills["timestamp"] = pd.to_datetime(fills["timestamp"], utc=True)

    units_table = _to_timestamped_table(replay.simulated_units)
    notional_table = _to_timestamped_table(replay.simulated_notional_weights)

    ledger = pd.concat(
        {
            "equity": replay.ledger.equity,
            "net_returns": replay.ledger.net_returns,
            "mark_to_market_pnl": replay.ledger.mark_to_market_pnl,
            "funding_charge": replay.ledger.funding_charge,
            "fee_charge": replay.ledger.fee_charge,
            "fill_turnover": replay.ledger.fill_turnover,
        },
        axis=1,
    )
    ledger_table = _to_timestamped_table(ledger)

    times = pd.DataFrame(
        {"submit_time": replay.submit_times, "fill_time": replay.fill_times}
    )
    times["submit_time"] = pd.to_datetime(times["submit_time"], utc=True)
    times["fill_time"] = pd.to_datetime(times["fill_time"], utc=True)

    return {
        "fills": fills,
        "units": units_table,
        "notional_weights": notional_table,
        "ledger": ledger_table,
        "times": times,
    }


def _write_unified_artifact_tables(
    tables_by_replay: dict[str, dict[str, pd.DataFrame]],
    artifact_root: Path,
) -> dict[str, tuple[Path, pd.DataFrame]]:
    """Concatenate per-replay category tables into exactly 5 unified Parquet files.

    Every unified table carries a leading ``replay_id`` column; the 5 files are
    written with snappy compression (much faster than zstd for these wide
    numeric tables) and returned as ``{category: (path, frame)}``.  Cross-replay
    schema promotion (timestamps at different precision, string vs large_string)
    is handled by ``pd.concat`` which promotes dtypes losslessly before the
    single snappy Parquet write (spec O8).
    """
    unified_frames: dict[str, list[pd.DataFrame]] = {
        category: [] for category in MHS_ARTIFACT_CATEGORIES
    }
    for replay_id, tables in tables_by_replay.items():
        for category in MHS_ARTIFACT_CATEGORIES:
            tagged = tables[category].copy()
            tagged.insert(0, "replay_id", replay_id)
            unified_frames[category].append(tagged)

    unified_tables: dict[str, tuple[Path, pd.DataFrame]] = {}
    for category in MHS_ARTIFACT_CATEGORIES:
        frames = unified_frames[category]
        if frames:
            frame = pd.concat(frames, ignore_index=True)
        else:
            frame = pd.DataFrame({"replay_id": pd.Series(dtype="string")})
        path = artifact_root / f"{category}.parquet"
        frame.to_parquet(path, index=False, compression="snappy")
        unified_tables[category] = (path, frame)
    return unified_tables


def _build_replay_artifact_reference(
    replay_id: str,
    replay: StrategyExecutionReplayResult,
    tables: dict[str, pd.DataFrame],
    artifact_root: Path,
    unified_tables: dict[str, tuple[Path, pd.DataFrame]],
) -> dict[str, Any]:
    """Unified-file reference for one replay: per-category row/time provenance and
    the shared unified file checksum, so readers load via ``load_mhs_replay_artifact``."""
    return {
        "artifact_format": "parquet",
        "artifact_dir": str(artifact_root),
        "replay_id": replay_id,
        "fills": _artifact_reference(tables["fills"], unified_tables["fills"][0]),
        "units": _artifact_reference(tables["units"], unified_tables["units"][0]),
        "notional_weights": _artifact_reference(
            tables["notional_weights"], unified_tables["notional_weights"][0]
        ),
        "ledger": _artifact_reference(tables["ledger"], unified_tables["ledger"][0]),
        "times": _artifact_reference(tables["times"], unified_tables["times"][0]),
        "fill_source": replay.fill_source,
        "mark_source": replay.mark_source,
        "event_snapshots_retained": replay.event_snapshots_retained,
        "fill_count": replay.fill_count,
        "unfilled_count": replay.unfilled_count,
        "fallback_count": replay.fallback_count,
        "all_intent_shortfall_bps": replay.all_intent_shortfall_bps,
        "forced_exit_count": replay.forced_exit_count,
        "forced_exit_notional": replay.forced_exit_notional,
        "termination_counts": dict(replay.termination_counts),
        "unsupported_assumptions": list(replay.unsupported_assumptions),
        "elapsed_seconds": replay.elapsed_seconds,
        "data_gaps": [
            {
                "code": gap.code,
                "symbol": gap.symbol,
                "timestamp": _jsonable(gap.timestamp),
                "decision_time": _jsonable(gap.decision_time),
                "signal_time": _jsonable(gap.signal_time),
                "execution_bound": gap.execution_bound,
            }
            for gap in replay.data_gaps
        ],
    }


def load_mhs_replay_artifact(
    artifact_root: str | Path,
    replay_id: str,
    category: Literal["fills", "units", "notional_weights", "ledger", "times"],
) -> pd.DataFrame:
    """Load a specific replay's artifact table using PyArrow pushdown filtering."""
    path = Path(artifact_root) / f"{category}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Unified artifact table missing: {path}")
    return pd.read_parquet(path, filters=[("replay_id", "==", replay_id)]).drop(
        columns=["replay_id"]
    )
