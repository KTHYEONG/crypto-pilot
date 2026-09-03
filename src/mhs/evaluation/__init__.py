# ruff: noqa
"""MHS Phase 1 orchestrator: dev-only diagnostics and the Research-GO evidence path.

This module composes the frozen ``src.mhs`` primitives; no alpha, cost,
ranking, liquidity, funding, or inventory arithmetic is reimplemented here.
The target-weight ``cost_response_curve`` is pre-screen only, the
immediate-taker replay + simulated inventory ledger is the primary
Research-GO evidence (with a cost-stressed x3 bound and an informational
patient-passive reference), and every report separates the two.
"""

from __future__ import annotations

import dataclasses  # noqa: F401
import gc
import glob
import logging
import math
import os
import time
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Facade re-exports consumed by the pipeline stage modules. evaluation used to
# import these for its own monolith; the stages now import them from here, so
# they are retained as a stable re-export surface (do not let ruff strip them).
from src.market_data.services.mhs_execution import (  # noqa: F401
    apply_dynamic_gap_exclusion,
    apply_dynamic_mark_gap_exclusion,
    assert_relevant_execution_data_coverage,
    assert_relevant_mark_price_coverage,
)
from src.mhs import research_go as _research_go
from src.mhs import scaling as _scaling
from src.mhs import statistics as _statistics
from src.mhs.contracts import (
    MhsBookFailure as MhsBookFailure,
)
from src.mhs.contracts import (
    MhsBookReport as MhsBookReport,
)
from src.mhs.contracts import (  # noqa: F401  (facade re-export; public API)
    MhsDiagnosticRequest as MhsDiagnosticRequest,
)
from src.mhs.contracts import (
    MhsFoldReport as MhsFoldReport,
)
from src.mhs.contracts import (
    MhsHorizonDiagnosticReport as MhsHorizonDiagnosticReport,
)
from src.mhs.contracts import (
    MhsOutputTier as MhsOutputTier,
)
from src.mhs.contracts import (
    MhsResearchGoResult as MhsResearchGoResult,
)
from src.mhs.contracts import (
    MhsResourceMeasurement as MhsResourceMeasurement,
)
from src.mhs.marks import (  # noqa: F401
    _build_window_frames,
    _cached_mark_panel,
    _fill_mark_parity_eligibility,
    _get_symbol_mark_frame,
    _load_funding_series,
    _load_window_minute_frames,
    _pit_execution_mask,
    _prewarm_mark_frames,
)

# Public GO reason-code constants are defined in research_go; re-exported here so
# the established ``ev.GO_REASON_*`` external API surface stays importable.
from src.mhs.research_go import (  # noqa: F401  (facade re-export of public GO reason-code constants)
    GO_REASON_CAPITAL_BREACH,  # noqa: F401
    GO_REASON_DATA_INTEGRITY_CODES,  # noqa: F401
    GO_REASON_DRAWDOWN_OVER_BUDGET,  # noqa: F401
    GO_REASON_EXECUTION_GAP,  # noqa: F401
    GO_REASON_FOLD_GROWTH_CONCENTRATION,  # noqa: F401
    GO_REASON_INCOMPLETE_FOLD,  # noqa: F401
    GO_REASON_INVALID_PRIMARY,  # noqa: F401
    GO_REASON_NONFINITE_EQUITY,  # noqa: F401
    GO_REASON_PATH_DIVERGENCE,  # noqa: F401
    GO_REASON_PRIMARY_RETURN_BELOW_FLOOR,  # noqa: F401
    GO_REASON_PRIMARY_SHARPE,  # noqa: F401
    GO_REASON_RESOURCE_BREACH,  # noqa: F401
    GO_REASON_STRESS_SHARPE,  # noqa: F401
    GO_REASON_UNSPECIFIED_POLICY,  # noqa: F401
)
from src.mhs.resources import (
    _assert_execution_rss_budget,
    _assert_stage_rss_budget,
    _resolve_ram_budget,
    _StageRecorder,
    _worker_plan_observer,
)
from src.common.paths import (
    FUTURES_DATA_DIR,  # noqa: F401
    funding_path,
)
from src.common.errors import DataIntegrityError
from src.market_data.services.futures_collection import DataCollector  # noqa: F401 - facade re-export
from src.mhs.books import (
    equal_weight_book_ensemble,
    inverse_realized_vol_tilt,
    phase_tranche_book,
    portfolio_rebalance_trigger,
    rank_weight_book,
    renormalize_within_mask,
    scale_book_to_target_gross,
)
from src.mhs.calibration import sharpe_lower_confidence_bound
from src.mhs.committee import (
    committee_block_edges_from,
    decompose_cost,
    long_only_equal_risk_weights,
    purged_walk_forward,
    score_weighted_net,
    train_evidence_weights,
    wealth_metrics,
)
from src.mhs.discovery import (
    DiscoveryQualificationResult,
    build_candidate_weights,
    fold_train_only_discovery_qualification,
    select_horizon_by_discovery_qualification,  # noqa: F401
    yearly_net_t_diagnostic,
)
from src.mhs.evidence import (
    AnchoredPurgedFold,
    CostResponsePoint,
    DeploymentReadinessResult,
    PhaseDiagnosticResult,
    TailSensitivityResult,
    book_evidence,
    compute_deployment_readiness,
    effective_breadth,
    phase_1_anchored_purged_folds,
    phase_diagnostic_metrics,
    required_cost_tiers,
    synthetic_stress_scenarios,  # noqa: F401
    year_restricted_correlation,
)
from src.mhs.execution import (
    ExecutionReplayWindow,
    StrategyExecutionReplayResult,
    bar_funding_panel,
    laddered_fill_schedule,
    mhs_ledger_pnl,
    mhs_ledger_pnl_multi_tier,
    replay_execution_window_batch,
    replay_execution_window_batch_isolated,
    replay_execution_windows,
    replay_execution_windows_coupled,
    simulated_inventory_ledger,
)
from src.mhs.features import (
    FEATURE_REGISTRY,
    FeatureSpec,
    build_feature_books,
    feature_coverage_audit,
    feature_registry_panel_columns,
    source_coverage_audit,
)
from src.mhs.funding import (
    build_funding_carry_candidate_weights,
    funding_carry_execution_book,  # noqa: F401 - contract wiring mandates the exact import line; the builder is invoked here
    funding_carry_signal,  # noqa: F401 - contract wiring mandates the exact import line; the builder is invoked here
)
from src.mhs.horizons import (
    efficiency_ratio,  # noqa: F401
    horizon_log_return,
    realized_vol,
    vol_normalized_horizon_signal,
)
from src.mhs.panel import liquid_half_eligibility, load_base_panel

# contract wiring: from src.mhs.parallel import FORK_CONTEXT, assert_fork_admission, fork_shared_payload, plan_worker_count
from src.mhs.parallel import (
    FORK_CONTEXT,
    assert_fork_admission,
    fork_shared_payload,
    plan_worker_count,
    resolve_fork_shared,
)
from src.mhs.params import (
    CAUSAL_BETA_LOOKBACK_BARS,
    CAUSAL_BETA_MIN_PERIODS,
    DISCOVERY_GATE_TRANCHE_COUNT,
    DISCOVERY_MOMENTUM_CANDIDATES,
    DISCOVERY_REVERSAL_CANDIDATES,
    EVIDENCE_GATE_ALPHA,
    FEATURE_NAME,
    FOLD_BLEND_PARITY_TOLERANCE,
    FOLD_PANEL_WARMUP_HOURS,
    FOLD_REALIZED_RISK_PARITY_TOLERANCE,
    GO_PRIMARY_SHARPE_FLOOR,  # noqa: F401  (facade re-export; public API)
    PNL_VOL_TARGET_BURN_IN_DAYS,  # noqa: F401  (facade re-export; public API)
    PNL_VOL_TARGET_SCALE_FLOOR,  # noqa: F401  (facade re-export; public API)
    REBALANCE_TRACKING_ERROR_THRESHOLD,
    REFERENCE_PASS_EQUITY_FLOOR,
    SEARCH_TRIALS_ATTEMPTED,  # noqa: F401
    SIGNAL_EMA_HORIZON_SPAN,
    STRESS_COST_MULTIPLIER,
    WALK_FORWARD_MIN_TRAIN_BARS,
)  # noqa: F401  (facade re-exports * tunables for public-API stability)
from src.mhs.params import (
    PERIODS_PER_YEAR_1H as _PERIODS_PER_YEAR_1H,
)
from src.mhs.regime import (
    beta_neutralize_weights,
    causal_market_beta,
    crash_regime_tilt_weights,
)

# Re-exports of the report persistence/artifact layer (moved to src/mhs/report).
# Kept importable from this module so legacy callers (CLI, tests, contracts) are
# unaffected by the extraction.
from src.mhs.report.artifacts import (  # noqa: F401
    _artifact_checksum,
    _artifact_reference,
    _build_replay_artifact_reference,
    _jsonable,
    _to_timestamped_table,
    _verify_ledger_artifact,
    load_mhs_replay_artifact,
)
from src.mhs.report.persist import (  # noqa: F401
    _book_summary,
    _build_replay_category_tables,
    _collect_replay_entries,
    _compact_replay_ref,
    _daily_resample_ledger,
    _fold_summary,
    _ledger_table,
    _persist_mhs_report_compact,
    _persist_mhs_report_full,
    _replay_category_row_counts,
    _round_6,
    _wire_compact_refs,
    _write_json_report,
    _write_unified_artifact_tables,
    build_mhs_run_history_record,
    mhs_horizon_diagnostic_report_path,
    persist_mhs_horizon_diagnostic_report,
    persist_mhs_report,
)
from src.mhs.stability import regime_split_stability
from src.mhs.telemetry import StageTelemetry
from src.mhs.trend_sleeve import (
    market_basket_log_price,
    time_series_trend_position,
    trend_sleeve_weights,
)
from src.mhs.types import (
    BOOK_BLEND_WEIGHTS,
    BOOK_SPECS,
    COMMITTEE_GROWTH_BARS_PER_YEAR,
    COMMITTEE_GROWTH_HORIZON_YEARS,
    COMMITTEE_GROWTH_MAX_DRAWDOWN,
    COMMITTEE_GROWTH_MAX_DRAWDOWN_PROB,
    COMMITTEE_GROWTH_MAX_RUIN_PROB,
    COMMITTEE_GROWTH_N_PATHS,
    COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS,
    COMMITTEE_GROWTH_RUIN_FRACTION,
    COMMITTEE_MEMBERS,
    COMMITTEE_OOS_START,
    COMMITTEE_PURGE_HOURS,
    COMMITTEE_REGIME_ADAPTIVE_WINDOW,
    COMMITTEE_TARGET_GROSS,  # noqa: F401  (facade re-export; public API)
    COMMITTEE_TARGET_VOL,
    COMMITTEE_TRANCHE_COUNT,
    CRASH_REGIME_REFERENCE_SYMBOLS,
    DISCOVERY_START,
    FEATURE_MIN_COVERAGE,
    FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS,
    FUNDING_CARRY_SLEEVE_LOOKBACK_HOURS,
    MEASURED_EXECUTION_COST_TIERS_BPS,
    REGISTERED_POLICY_THRESHOLDS,  # noqa: F401  (facade re-export; public API)
    TREND_SLEEVE_HORIZONS_HOURS,
    WORKER_PEAK_RSS_BYTES,
    BookSpec,
    ExecutionSpec,
)
from src.quant.evaluation.policy import HOLDOUT_CUTOFF, resolve_evaluation_end
from src.quant.risk.growth_sizing import GrowthSizingConfig, diagnose_growth_headroom, solve_growth_optimal_risk
from src.quant.technical_experts.trend_screen_catalog import (  # noqa: F401
    DISCOVERY_END,
    QUALIFICATION_END,
)

__all__ = ["funding_path", "simulated_inventory_ledger"]

MhsExecutionWindow = ExecutionReplayWindow

_logger = logging.getLogger("MhsHorizonDiagnostic")



from . import (
    books,
    committee,
    concurrency,
    diagnostics,
    evidence,
    fold_weights,
    folds,
    guards,
    integrity,
    participation,
    regime,
    specs,
    windows,
)
from .books import (
    _active_blend_book_and_grid,
    _book_structure_trace,
    _book_weights,
    _candidate_weight_books,
    _horizon_ensemble_execution_weights,
    _ordered_union,
)
from .committee import (
    _committee_diagnostic,
    _committee_evidence_weights_by_boundary,
    _committee_execution_book,
    _committee_growth_headroom,
    _committee_member_attribution,
    _committee_member_books,
)
from .concurrency import _run_books_concurrent, _run_post_book_concurrently, _run_post_diag_deploy
from .diagnostics import (
    _MULTI_FEATURE_PANEL_COLUMNS,
    _MULTI_FEATURE_REGIME_SPLIT,
    _available_panel_columns,
    _load_feature_panels,
    _multi_feature_diagnostic,
    _phase_diagnostics,
    _trend_sleeve_diagnostic,
)
from .evidence import (
    _fold_blend_parity,
    _fold_growth_concentration,
    _fold_realized_risk_parity,
    _log_growth,
    _pooled_fold_evidence,
)
from .fold_weights import _build_fold_target_weights
from .folds import (
    _apply_trend_sleeve,
    _fold_exposure_warmup,
    _fold_safe_discovery_worker,
    _fold_safe_fast_horizon,
    _fold_safe_slow_book_spec,
    _incomplete_fold_report,
    _prefer_funding_carry_selection,
    _run_anchored_fold,
    _run_fold_safe_discovery_parallel,
    _run_folds_parallel,
    _trend_sleeve_position,
)
from .guards import _guard_stage_or_breach, _terminal_resource_breach_report
from .integrity import (
    SOURCE_GAP_EXCLUDED_SYMBOLS,
    _assert_cache_required_ledger_valid,
    _assert_cache_required_marks,
    _classify_execution_failure,
    _truncate_replayable_decisions,
    _validate_ladder_schedule_contract,
)
from .participation import _load_symbol_quote_volume, _participation_warnings
from .regime import _fold_regime_characterization, _load_reference_close, _regime_reference_characterization
from .specs import (
    REFERENCE_ONLY_EXECUTION_BOUNDS,
    _peg_chase_fill_rate,
    _peg_chase_maker_share,
    _resolved_base_execution_spec,
    _signal_ema_span,
    _stress_cost_execution_spec,
)
from .windows import (
    _book_outcome,
    _book_outcome_worker,
    _iter_mhs_execution_windows,
    _rescaled_windows,
    _resolve_ns_vectorized,
)
