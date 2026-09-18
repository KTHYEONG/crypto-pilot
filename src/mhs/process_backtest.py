"""Single continuous causal replay of the algorithmic MHS process (1h ledger proxy).

Quarterly evidence is sliced from one out-of-sample path instead of replaying
each fold separately, so fold statistics and the deployed path can never
diverge. The hourly ledger is a proxy (``mhs_ledger_pnl`` contract): the report
is research evidence, never a deploy verdict, until Phase 2 routes the same
targets through the minute execution replay.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.paths import FUTURES_DATA_DIR
from src.market_data.services.mhs_execution import (
    apply_dynamic_gap_exclusion,
    apply_dynamic_mark_gap_exclusion,
)
from src.mhs.books import rank_weight_book, scale_book_to_target_gross
from src.mhs.contracts import MhsResourceMeasurement
from src.mhs.data_policy import MHS_DATA_POLICY_DEFAULT
from src.mhs.deploy_gate import DeployGateResult, evaluate_deploy_gate
from src.mhs.evaluation.integrity import SOURCE_GAP_EXCLUDED_SYMBOLS, replay_ledger_certified
from src.mhs.evaluation.specs import _stress_cost_execution_spec
from src.mhs.evaluation.windows import _iter_mhs_execution_windows
from src.mhs.execution import _ExecutionBound
from src.mhs.execution.batch import (
    _LiveAccumulatorSets,
    live_required_symbols,
    replay_execution_window_batch,
    replay_execution_windows,
)
from src.mhs.execution.contracts import (
    ExecutionDataGap,
    ExecutionReplayWindow,
    StrategyExecutionReplayResult,
    bar_funding_panel,
)
from src.mhs.execution.pnl import mhs_ledger_pnl
from src.mhs.features import FEATURE_REGISTRY
from src.mhs.funding import funding_carry_signal
from src.mhs.marks import _load_funding_series, _pit_execution_mask
from src.mhs.panel import liquid_half_eligibility, load_base_panel
from src.mhs.params import (
    CAUSAL_BETA_LOOKBACK_BARS,
    CAUSAL_BETA_MIN_PERIODS,
    DISCOVERY_START,
    GROWTH_RISK_ENVELOPES,
    PANEL_MIN_HISTORY_BARS,
    PROCESS_EVALUATION_CEILING,
    PROCESS_FEATURE_CANDIDATES,
    PROCESS_FUNDING_CARRY_CANDIDATES_HOURS,
    PROCESS_MIN_SYMBOLS,
    PROCESS_SMOOTHING_HALFLIFE_DAYS,
    STRESS_COST_MULTIPLIER,
)
from src.mhs.pipeline.config import (
    CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT,
    CLI_GROWTH_ENVELOPE_DEFAULT,
)
from src.mhs.process import (
    ProcessExecutionPolicy,
    RefitPoint,
    apply_process_execution_policy,
    ema_smoothing_rate,
    estimation_adjusted_mean,
    ledoit_wolf_covariance,
    long_only_growth_weights,
    monthly_refit_schedule,
    smoothed_book_path,
    step_proxy_net_returns,
    volatility_scaled_exposure,
)
from src.mhs.regime import beta_neutralize_weights, causal_market_beta
from src.mhs.resources import (
    MHS_REPLAY_BUDGET_BYTES,
    MhsMemoryBudget,
    MhsResourceAdmissionError,
    ProcessTreeMemoryStats,
    _assert_execution_rss_budget,
    _assert_stage_rss_budget,
    _current_tree_swap_bytes,
    _resolve_ram_budget,
    _StageRecorder,
    _TreeMemorySampler,
    assert_mhs_stage_allocation,
)
from src.mhs.types import ExecutionSpec

_logger = logging.getLogger(__name__)

PROCESS_REPORT_PATH: Path = Path("docs") / "results" / "mhs_process_backtest.json"
PROCESS_POLICY_REPORT_PATH: Path = Path("docs") / "results" / "mhs_process_execution_policy.json"
PROCESS_CERTIFICATION_LEVEL: str = "process_proxy_1h_ledger"
PROCESS_INVENTORY_REPORT_PATH: Path = Path("docs") / "results" / "mhs_process_3m_backtest.json"
PROCESS_INVENTORY_CERTIFICATION_LEVEL: str = "process_inventory_3m"
PROCESS_INVENTORY_INITIAL_EQUITY: float = 1.0
_INVENTORY_LEDGER_INVALID: str = "INVENTORY_LEDGER_INVALID"
_UNPRICED_TERMINAL_CODES: frozenset[str] = frozenset(
    {"MISSING_HELD_MARK", "MISSING_HELD_FUNDING", "MISSING_DECISION_MARK"}
)


@dataclass(frozen=True, slots=True)
class ProcessMarketData:
    """Causally aligned inputs shared by every cost tier of one process run."""

    grid_1h: pd.DatetimeIndex
    decision_grid: pd.DatetimeIndex
    opens_1h: pd.DataFrame
    bar_funding_1h: pd.DataFrame
    log_close_step: pd.DataFrame
    funding_step: pd.DataFrame
    member_books: dict[str, pd.DataFrame]
    execution_mask: pd.DataFrame


@dataclass(frozen=True, slots=True)
class RefitRecord:
    """Audit record of one refit's decisions."""

    point: RefitPoint
    member_weights: dict[str, float]
    smoothing_halflife_days: float


@dataclass(frozen=True, slots=True)
class ProcessPath:
    """One cost tier's continuous proxy path and its exact execution targets.

    Unit targets precede volatility sizing; target weights are the sized
    decision rows actually evaluated. Keeping them prevents a later
    inventory replay from rebuilding a different strategy from summary
    statistics. Policy and targets are identical across cost tiers.
    """

    one_way_bps: float
    daily_returns: pd.Series
    unit_daily_returns: pd.Series
    exposure: pd.Series
    refits: tuple[RefitRecord, ...]
    leverage_cap: float
    execution_policy: ProcessExecutionPolicy
    unit_target_weights: pd.DataFrame
    target_weights: pd.DataFrame
    turnover_1h: pd.Series


@dataclass(frozen=True, slots=True)
class ProcessBacktestReport:
    """Proxy evidence for the continuous process; ``gate`` is never a deploy verdict."""

    start: pd.Timestamp
    end: pd.Timestamp
    certification_level: str
    n_candidates: int
    base: ProcessPath
    stress: ProcessPath
    gate: DeployGateResult


def build_candidate_member_books(
    panels: Mapping[str, pd.DataFrame],
    bar_funding_1h: pd.DataFrame,
    eligible_1h: pd.DataFrame,
    execution_mask_1h: pd.DataFrame,
    decision_grid: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Beta-neutral unit-gross decision-grid books for every declared candidate.

    Books are not admission-filtered by coverage: a coverage audit over the full
    window would decide early membership with later data, and a member without
    evidence already receives zero weight from the growth weights.
    """
    registry = {spec.name: spec for spec in FEATURE_REGISTRY}
    beta_1h = causal_market_beta(
        np.log(panels["close"]), eligible_1h, CAUSAL_BETA_LOOKBACK_BARS, CAUSAL_BETA_MIN_PERIODS
    )
    beta_grid = beta_1h.reindex(decision_grid)
    del beta_1h
    mask_grid = execution_mask_1h.reindex(decision_grid).fillna(False)
    out: dict[str, pd.DataFrame] = {}
    for name in PROCESS_FEATURE_CANDIDATES:
        spec = registry[name]
        _logger.info("[DATA] stage=candidate_progress candidate=%s", name)
        feature = spec.builder(panels)
        book = rank_weight_book(feature, execution_mask_1h, 1, PROCESS_MIN_SYMBOLS)
        del feature
        grid_book = book.reindex(decision_grid).fillna(0.0)
        del book
        out[name] = beta_neutralize_weights(grid_book, beta_grid, mask_grid, PROCESS_MIN_SYMBOLS)
        del grid_book
    for lookback in PROCESS_FUNDING_CARRY_CANDIDATES_HOURS:
        key = f"funding_carry_{lookback}h"
        _logger.info("[DATA] stage=candidate_progress candidate=%s", key)
        signal = funding_carry_signal(bar_funding_1h, lookback)
        book = rank_weight_book(signal, execution_mask_1h, -1, PROCESS_MIN_SYMBOLS)
        del signal
        grid_book = book.reindex(decision_grid).fillna(0.0)
        del book
        out[key] = beta_neutralize_weights(grid_book, beta_grid, mask_grid, PROCESS_MIN_SYMBOLS)
        del grid_book
    return out


def _process_memory_budget(budget: MhsMemoryBudget | None) -> MhsMemoryBudget:
    return budget if budget is not None else MhsMemoryBudget()


def _admit_process_stage(
    *, stage: str, estimated_bytes: int, budget: MhsMemoryBudget,
    replay: bool, initial_swap_bytes: int | None,
) -> None:
    assert_mhs_stage_allocation(
        stage=stage, estimated_bytes=int(estimated_bytes),
        budget=budget, replay=replay, initial_swap_bytes=initial_swap_bytes,
    )


def _estimate_panel_bytes(n_bars: int, n_symbols: int, n_planes: int = 6) -> int:
    return max(int(n_bars) * max(int(n_symbols), 0) * max(int(n_planes), 0) * 8 * 2, 0)


def load_process_market_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    data_root: str | None = None,
    memory_budget: MhsMemoryBudget | None = None,
) -> ProcessMarketData:
    """Prepare causal hourly strategy inputs within a process-tree budget.

    Args:
        start: Timezone-aware source start.
        end: Timezone-aware source end within the registered ceiling.
        data_root: Existing OHLCV root override, not a mark/funding root override.
        memory_budget: Explicit limits or the validated default stage limits.

    Returns:
        Unchanged strategy inputs with corrected causal execution eligibility.

    Raises:
        RuntimeError: No development symbol has causally aligned funding.
        DataIntegrityError: Source provenance or allocation admission fails.
    """
    budget = _process_memory_budget(memory_budget)
    initial_swap_bytes = _current_tree_swap_bytes()
    span_hours = max(int((end - start).total_seconds() // 3600), 1)
    _admit_process_stage(
        stage="process_prepare_panel", estimated_bytes=_estimate_panel_bytes(span_hours + PANEL_MIN_HISTORY_BARS, CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT),
        budget=budget, replay=False, initial_swap_bytes=initial_swap_bytes,
    )
    root = data_root or str(FUTURES_DATA_DIR / "ohlcv")
    panel = load_base_panel(
        root, "1h",
        ("close", "open", "high", "low", "quote_vol", "taker_buy_quote"),
        start, end, partition="dev", min_bars=PANEL_MIN_HISTORY_BARS,
        data_policy=MHS_DATA_POLICY_DEFAULT,
    )
    close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    grid_1h = close.index
    _logger.info("[DATA] stage=base_1h_panel bars=%d symbols=%d", len(grid_1h), len(close.columns))
    funding_by_symbol, _dropped = _load_funding_series(list(close.columns))
    funded = [
        s for s in close.columns
        if s in funding_by_symbol and s not in SOURCE_GAP_EXCLUDED_SYMBOLS
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
    aligned = list(bar_funding.columns)
    if not aligned:
        raise RuntimeError("no dev symbol has causally aligned funding coverage")
    close = close[aligned]
    opens = opens[aligned]
    quote_vol = quote_vol[aligned]
    _logger.info("[DATA] stage=funding_alignment bars=%d symbols=%d", len(grid_1h), len(aligned))
    eligible = liquid_half_eligibility(quote_vol, PANEL_MIN_HISTORY_BARS, PANEL_MIN_HISTORY_BARS)
    mask = _pit_execution_mask(quote_vol, eligible, CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT)
    decision_grid = pd.date_range(start, end, freq="24h", tz="UTC")
    _admit_process_stage(
        stage="process_funding_prefix", estimated_bytes=_estimate_panel_bytes(len(grid_1h), len(aligned), 2),
        budget=budget, replay=False, initial_swap_bytes=initial_swap_bytes,
    )
    log_close_step = np.log(close).reindex(decision_grid)
    # (t, t+24h] 구간 합을 누적합 차분으로 계산한다(결정일마다 전체 스캔 금지).
    day = pd.Timedelta(hours=24)
    prefix = np.vstack([np.zeros((1, len(aligned))), np.cumsum(bar_funding.to_numpy(dtype="float64"), axis=0)])
    left = np.searchsorted(bar_funding.index.to_numpy(), decision_grid.to_numpy(), side="right")
    right = np.searchsorted(bar_funding.index.to_numpy(), (decision_grid + day).to_numpy(), side="right")
    funding_step = pd.DataFrame(prefix[right] - prefix[left], index=decision_grid, columns=aligned)
    _logger.info("[DATA] stage=decision_grid days=%d symbols=%d", len(decision_grid), len(aligned))
    panels: dict[str, pd.DataFrame] = {k: panel[k][aligned] for k in ("close", "open", "high", "low", "quote_vol", "taker_buy_quote")}
    del panel
    causal_mask, _ = apply_dynamic_gap_exclusion(mask, "1h", root=root)
    causal_mask, _ = apply_dynamic_mark_gap_exclusion(causal_mask)
    causal_mask, _ = apply_dynamic_gap_exclusion(causal_mask, "3m", root=root)
    n_candidates = len(PROCESS_FEATURE_CANDIDATES) + len(PROCESS_FUNDING_CARRY_CANDIDATES_HOURS)
    _admit_process_stage(
        stage="process_candidate_books", estimated_bytes=_estimate_panel_bytes(len(decision_grid), len(aligned), n_candidates),
        budget=budget, replay=False, initial_swap_bytes=initial_swap_bytes,
    )
    member_books = build_candidate_member_books(panels, bar_funding, eligible, causal_mask, decision_grid)
    del panels, prefix
    _logger.info("[DATA] stage=member_books candidates=%d", len(member_books))
    book_columns = list(next(iter(member_books.values())).columns)
    _admit_process_stage(
        stage="process_hourly_ledger", estimated_bytes=_estimate_panel_bytes(len(decision_grid), len(book_columns), 2),
        budget=budget, replay=False, initial_swap_bytes=initial_swap_bytes,
    )
    execution_mask = causal_mask.reindex(decision_grid, fill_value=False)[book_columns]
    return ProcessMarketData(
        grid_1h=grid_1h,
        decision_grid=decision_grid,
        opens_1h=opens,
        bar_funding_1h=bar_funding,
        log_close_step=log_close_step,
        funding_step=funding_step,
        member_books=member_books,
        execution_mask=execution_mask,
    )


def apply_process_execution_availability(target_weights: pd.DataFrame, execution_mask: pd.DataFrame) -> pd.DataFrame:
    """Prevent unavailable targets from being revived by smoothing or adoption.

    Args:
        target_weights: Smoothed and adopted canonical decision targets.
        execution_mask: Exactly aligned causal execution eligibility.

    Returns:
        Targets with unavailable cells exactly zero and available cells unchanged.

    Raises:
        DataIntegrityError: Labels, symbols or mask values are inconsistent.
    """
    if not target_weights.index.equals(execution_mask.index):
        raise DataIntegrityError("execution_mask must share target_weights decision labels exactly")
    if list(target_weights.columns) != list(execution_mask.columns):
        raise DataIntegrityError("execution_mask must share target_weights symbol columns exactly")
    if bool((execution_mask.dtypes.apply(lambda dt: dt.kind != "b")).any()):
        raise DataIntegrityError("execution_mask must be boolean")
    if bool(execution_mask.isna().to_numpy().any()):
        raise DataIntegrityError("execution_mask must not be missing")
    return target_weights.where(execution_mask, other=0.0)


def run_process_paths(
    data: ProcessMarketData, schedule: tuple[RefitPoint, ...], *,
    decision_bps: float, evaluation_bps: tuple[float, ...], leverage_cap: float,
    execution_policy: ProcessExecutionPolicy | None = None,
    memory_budget: MhsMemoryBudget | None = None,
) -> tuple[ProcessPath, ...]:
    """Construct identical process decisions with interval-local refit buffers.

    Args:
        data: Corrected causal features and registered candidate books.
        schedule: Chronological refits with purged training cutoffs.
        decision_bps: Finite nonnegative decision friction.
        evaluation_bps: Nonempty ordered evaluation tiers.
        leverage_cap: Positive finite exposure ceiling.
        execution_policy: Existing adoption policy or baseline.
        memory_budget: Explicit limits or validated defaults.

    Returns:
        Ordered paths preserving targets, proxy evidence and refit records.

    Raises:
        ValueError: Schedule or process controls are invalid.
        DataIntegrityError: Financial inputs or allocation admission fail.
    """
    budget = _process_memory_budget(memory_budget)
    initial_swap_bytes = _current_tree_swap_bytes()
    if not schedule:
        raise ValueError("schedule must not be empty")
    if not data.member_books:
        raise ValueError("no candidate books")
    if not evaluation_bps:
        raise ValueError("evaluation_bps must not be empty")
    try:
        decision_value = float(decision_bps)
    except (TypeError, ValueError):
        raise ValueError(f"decision_bps must be finite, got {decision_bps}") from None
    if not np.isfinite(decision_value) or decision_value < 0.0:
        raise ValueError(f"decision_bps must be finite and >= 0, got {decision_bps}")
    tier_values: list[float] = []
    for bps in evaluation_bps:
        try:
            value = float(bps)
        except (TypeError, ValueError):
            raise ValueError(f"evaluation_bps must be finite, got {bps}") from None
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"evaluation_bps must be finite and >= 0, got {bps}")
        tier_values.append(value)
    try:
        cap_value = float(leverage_cap)
    except (TypeError, ValueError):
        raise ValueError(f"leverage_cap must be finite, got {leverage_cap}") from None
    if not np.isfinite(cap_value) or cap_value <= 0.0:
        raise ValueError(f"leverage_cap must be finite and > 0, got {leverage_cap}")
    if execution_policy is not None and not isinstance(execution_policy, ProcessExecutionPolicy):
        raise ValueError("execution_policy must be a ProcessExecutionPolicy or None")
    policy = execution_policy if execution_policy is not None else ProcessExecutionPolicy()
    rate = ema_smoothing_rate(PROCESS_SMOOTHING_HALFLIFE_DAYS)
    names = list(data.member_books.keys())
    smoothed_members = {
        name: smoothed_book_path(
            data.member_books[name], pd.Series(rate, index=data.member_books[name].index)
        )
        for name in names
    }
    member_net = pd.DataFrame(
        {
            name: step_proxy_net_returns(
                smoothed_members[name], data.log_close_step, data.funding_step, decision_bps
            )
            for name in names
        }
    )
    del smoothed_members
    step = member_net.index[1] - member_net.index[0]
    oos_start = schedule[0].effective_from
    oos_end = schedule[-1].effective_to
    oos_days = data.decision_grid[(data.decision_grid >= oos_start) & (data.decision_grid < oos_end)]
    n_symbols = len(data.member_books[names[0]].columns)
    _admit_process_stage(
        stage="process_refit_combination", estimated_bytes=_estimate_panel_bytes(len(oos_days), n_symbols, len(names) + 1),
        budget=budget, replay=False, initial_swap_bytes=initial_swap_bytes,
    )
    refit_targets: list[pd.DataFrame] = []
    records: list[RefitRecord] = []
    for point in schedule:
        train_rows = member_net.index[member_net.index + step <= point.train_end]
        train = member_net.loc[train_rows]
        weights = long_only_growth_weights(
            estimation_adjusted_mean(train), ledoit_wolf_covariance(train)
        )
        effective_index = data.member_books[names[0]].index[
            (data.member_books[names[0]].index >= point.effective_from)
            & (data.member_books[names[0]].index < point.effective_to)
        ]
        combined = sum(
            (weights[name] * data.member_books[name].loc[effective_index] for name in names),
            start=data.member_books[names[0]].loc[effective_index] * 0.0,
        )
        target = scale_book_to_target_gross(combined, 1.0)
        del combined
        refit_targets.append(target)
        del target
        records.append(
            RefitRecord(
                point=point,
                member_weights={n: float(weights[n]) for n in names if float(weights[n]) != 0.0},
                smoothing_halflife_days=PROCESS_SMOOTHING_HALFLIFE_DAYS,
            )
        )
    targets_oos = pd.concat(refit_targets)
    del refit_targets
    smoothed_targets = smoothed_book_path(targets_oos, pd.Series(rate, index=oos_days))
    del targets_oos
    sized_targets = apply_process_execution_policy(smoothed_targets, policy)
    del smoothed_targets
    execution_mask = data.execution_mask
    sized_targets = apply_process_execution_availability(
        sized_targets, execution_mask.reindex(sized_targets.index)
    )
    _admit_process_stage(
        stage="process_hourly_ledger", estimated_bytes=_estimate_panel_bytes(len(data.grid_1h), n_symbols, 3),
        budget=budget, replay=False, initial_swap_bytes=initial_swap_bytes,
    )
    unit_1h = sized_targets.reindex(data.grid_1h, method="ffill").fillna(0.0)
    unit_net_1h, _ = mhs_ledger_pnl(unit_1h, data.opens_1h, data.bar_funding_1h, decision_value)
    _reject_invalid_ledger_returns(unit_net_1h)
    decision_unit_daily = (
        ((1.0 + unit_net_1h).resample("1D").prod() - 1.0).reindex(oos_days).fillna(0.0)
    )
    _reject_invalid_ledger_returns(decision_unit_daily)
    exposure = volatility_scaled_exposure(decision_unit_daily, cap=cap_value)
    sized = sized_targets.mul(exposure.reindex(sized_targets.index).fillna(0.0), axis=0)
    sized_1h = sized.reindex(data.grid_1h, method="ffill").fillna(0.0)
    paths: list[ProcessPath] = []
    for bps in tier_values:
        net_1h, turnover_1h = mhs_ledger_pnl(sized_1h, data.opens_1h, data.bar_funding_1h, bps)
        _reject_invalid_ledger_returns(net_1h)
        daily_returns = ((1.0 + net_1h).resample("1D").prod() - 1.0).reindex(oos_days).fillna(0.0)
        _reject_invalid_ledger_returns(daily_returns)
        if bps == decision_value:
            unit_daily_returns = decision_unit_daily
        else:
            alt_net_1h, _ = mhs_ledger_pnl(unit_1h, data.opens_1h, data.bar_funding_1h, bps)
            _reject_invalid_ledger_returns(alt_net_1h)
            unit_daily_returns = (
                ((1.0 + alt_net_1h).resample("1D").prod() - 1.0).reindex(oos_days).fillna(0.0)
            )
            _reject_invalid_ledger_returns(unit_daily_returns)
        paths.append(
            ProcessPath(
                one_way_bps=bps,
                daily_returns=daily_returns,
                unit_daily_returns=unit_daily_returns,
                exposure=exposure,
                refits=tuple(records),
                leverage_cap=cap_value,
                execution_policy=policy,
                unit_target_weights=sized_targets,
                target_weights=sized,
                turnover_1h=turnover_1h,
            )
        )
    del unit_1h, sized_1h
    return tuple(paths)


def _reject_invalid_ledger_returns(returns: pd.Series) -> None:
    values = returns.to_numpy(dtype="float64")
    if values.size and not bool(np.isfinite(values).all()):
        raise DataIntegrityError("ledger net return is non-finite")
    if values.size and bool((values <= -1.0).any()):
        raise DataIntegrityError("ledger net return does not exceed minus one")


def quarter_fold_returns(daily_returns: pd.Series) -> tuple[pd.Series, ...]:
    """Split a daily path into non-empty calendar quarters (``QE-DEC``), in order."""
    if daily_returns.empty:
        return ()
    periods = daily_returns.index.tz_convert("UTC").tz_localize(None).to_period("Q-DEC")
    folds: list[pd.Series] = []
    for period in sorted(set(periods)):
        fold = daily_returns.loc[periods == period]
        if not fold.empty:
            folds.append(fold)
    return tuple(folds)


def evaluate_process_backtest(
    start: pd.Timestamp = DISCOVERY_START,
    end: pd.Timestamp = PROCESS_EVALUATION_CEILING, *,
    data_root: str | None = None,
    execution_policy: ProcessExecutionPolicy | None = None,
    memory_budget: MhsMemoryBudget | None = None,
) -> ProcessBacktestReport:
    """Evaluate an explicitly configured process as hourly proxy evidence.

    The fixed evaluation ceiling preserves the unused forward interval.
    Passing a research control does not certify inventory execution,
    capacity, or eligibility for deployment.

    Args:
        start: Timezone-aware input start.
        end: Timezone-aware input end within the registered ceiling.
        data_root: Optional hourly OHLCV root using the existing dev loader.
        execution_policy: Explicit adoption control; None preserves baseline.
        memory_budget: Explicit process-tree limits or validated stage defaults.

    Returns:
        Same-decision base and stress evidence with the existing proxy gate.

    Raises:
        ValueError: Timestamps or process controls are invalid.
        DataIntegrityError: The evaluation exceeds the registered ceiling
            or market inputs violate a process invariant.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("timestamps must be tz-aware")
    if start >= end:
        raise ValueError("start must be before end")
    if end > PROCESS_EVALUATION_CEILING:
        raise DataIntegrityError(f"end {end} exceeds PROCESS_EVALUATION_CEILING")
    if execution_policy is not None and not isinstance(execution_policy, ProcessExecutionPolicy):
        raise ValueError("execution_policy must be a ProcessExecutionPolicy or None")
    envelope = GROWTH_RISK_ENVELOPES[CLI_GROWTH_ENVELOPE_DEFAULT]
    base_bps = ExecutionSpec().one_way_taker_bps()
    stress_bps = base_bps * STRESS_COST_MULTIPLIER
    data = load_process_market_data(start, end, data_root=data_root, memory_budget=memory_budget)
    schedule = monthly_refit_schedule(data.decision_grid[0], data.decision_grid[-1])
    if execution_policy is None:
        base, stress = run_process_paths(
            data, schedule, decision_bps=base_bps,
            evaluation_bps=(base_bps, stress_bps), leverage_cap=envelope.leverage_ceiling,
            execution_policy=execution_policy, memory_budget=memory_budget,
        )
    else:
        base, stress = run_process_paths(
            data, schedule, decision_bps=base_bps,
            evaluation_bps=(base_bps, stress_bps), leverage_cap=envelope.leverage_ceiling,
            execution_policy=execution_policy, memory_budget=memory_budget,
        )
    gate = evaluate_deploy_gate(
        fold_returns=quarter_fold_returns(base.daily_returns),
        fold_stress_returns=quarter_fold_returns(stress.daily_returns),
        integrity_reasons=(),
        envelope=envelope,
    )
    return ProcessBacktestReport(
        start=start,
        end=end,
        certification_level=PROCESS_CERTIFICATION_LEVEL,
        n_candidates=len(data.member_books),
        base=base,
        stress=stress,
        gate=gate,
    )


def _tier_payload(path: ProcessPath) -> dict[str, object]:
    daily = {ts.isoformat(): float(v) for ts, v in path.daily_returns.items()}
    exposure = {ts.isoformat(): float(v) for ts, v in path.exposure.items()}
    log_growth = float(np.log1p(path.daily_returns.to_numpy(dtype="float64")).mean() * 365.0) if len(path.daily_returns) else 0.0
    exposure_values = path.exposure.to_numpy(dtype="float64")
    zero_share = float((exposure_values == 0.0).mean()) if len(exposure_values) else 0.0
    cap_share = float((exposure_values >= path.leverage_cap).mean()) if len(exposure_values) else 0.0
    if len(path.daily_returns) and len(path.turnover_1h):
        start_label = path.daily_returns.index[0]
        turnover_slice = path.turnover_1h.loc[path.turnover_1h.index >= start_label]
        ann_turnover = float(turnover_slice.sum() * 365.0 / len(path.daily_returns))
    else:
        ann_turnover = 0.0
    mean_unit_gross = (
        float(path.unit_target_weights.abs().sum(axis=1).mean()) if len(path.unit_target_weights) else 0.0
    )
    mean_effective_gross = (
        float(path.target_weights.abs().sum(axis=1).mean()) if len(path.target_weights) else 0.0
    )
    return {
        "one_way_bps": path.one_way_bps,
        "leverage_cap": path.leverage_cap,
        "daily_returns": daily,
        "exposure": exposure,
        "refits": [
            {
                "effective_from": r.point.effective_from.isoformat(),
                "effective_to": r.point.effective_to.isoformat(),
                "train_end": r.point.train_end.isoformat(),
                "member_weights": dict(r.member_weights),
                "smoothing_halflife_days": r.smoothing_halflife_days,
            }
            for r in path.refits
        ],
        "ann_log_growth": log_growth,
        "exposure_zero_share": zero_share,
        "exposure_cap_share": cap_share,
        "execution_policy": {"tracking_error_threshold": path.execution_policy.tracking_error_threshold},
        "ann_turnover": ann_turnover,
        "mean_unit_gross": mean_unit_gross,
        "mean_effective_gross": mean_effective_gross,
    }


def _resolve_process_report_path(report: ProcessBacktestReport, path: Path | None) -> Path:
    if path is None:
        threshold = report.base.execution_policy.tracking_error_threshold
        if threshold is None:
            return PROCESS_REPORT_PATH
        return PROCESS_POLICY_REPORT_PATH
    out = Path(path)
    if out.suffix != ".json":
        raise ValueError(f"destination must be a JSON path, got {path}")
    threshold = report.base.execution_policy.tracking_error_threshold
    if threshold is not None and out.resolve() == PROCESS_REPORT_PATH.resolve():
        raise ValueError("enabled-policy report would overwrite the reserved baseline destination")
    return out


def persist_process_report(
    report: ProcessBacktestReport,
    path: Path | None = None,
) -> Path:
    """Persist proxy evidence without replacing the baseline with a trial.

    Args:
        report: The evaluated hourly proxy report.
        path: Optional JSON artifact path. None chooses the baseline path
            only when adoption is disabled, otherwise the research path.

    Returns:
        The path of the written JSON evidence.

    Raises:
        ValueError: The destination is not a JSON path or an enabled-policy
            report would overwrite the reserved baseline destination.
    """
    out = _resolve_process_report_path(report, path)
    payload = {
        "start": report.start.isoformat(),
        "end": report.end.isoformat(),
        "certification_level": report.certification_level,
        "n_candidates": report.n_candidates,
        "evidence_scope": "retrospective_discovery",
        "multiplicity_adjusted": False,
        "gate": {
            "go": report.gate.go,
            "reason_codes": list(report.gate.reason_codes),
            "metrics": dict(report.gate.metrics),
        },
        "base": _tier_payload(report.base),
        "stress": _tier_payload(report.stress),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return out


def persist_process_targets(path: ProcessPath, output: Path) -> Path:
    """Export exact sized targets without inventing signal availability.

    Args:
        path: Evaluated process path with its sized decision rows.
        output: Explicit parquet destination.

    Returns:
        The written parquet path, retaining float64 values and UTC labels.

    Raises:
        ValueError: The destination is not a parquet path.
    """
    out = Path(output)
    if out.suffix != ".parquet":
        raise ValueError(f"destination must be a parquet path, got {output}")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = path.target_weights.copy().astype("float64")
    frame.to_parquet(out)
    return out


def _execution_fence(target_weights: pd.DataFrame) -> pd.Timestamp:
    final_decision = target_weights.index[-1]
    ceiling_day = PROCESS_EVALUATION_CEILING.normalize()
    final_day = final_decision.normalize()
    earlier = final_day if final_day <= ceiling_day else ceiling_day
    return (earlier + pd.Timedelta(days=1)).tz_convert("UTC")


def _require_utc_index(values: pd.DatetimeIndex, name: str) -> pd.DatetimeIndex:
    if not isinstance(values, pd.DatetimeIndex):
        raise DataIntegrityError(f"{name} must be a DatetimeIndex")
    if values.hasnans:
        raise DataIntegrityError(f"{name} must not contain NaT")
    if values.tz is None:
        raise DataIntegrityError(f"{name} must be timezone-aware UTC")
    converted = values.tz_convert("UTC")
    if not converted.equals(values):
        raise DataIntegrityError(f"{name} must be UTC")
    return values


def _validate_replay_window(
    window: ExecutionReplayWindow,
    *,
    expected_columns: list[str],
    expected_targets: pd.DataFrame,
    cursor: int,
    fence: pd.Timestamp,
) -> int:
    """Validate local execution planes against canonical process decisions.

    Args:
        window: Explicit market and decision provenance for one partition.
        expected_columns: Canonical ordered symbols.
        expected_targets: Exact sized decision book.
        cursor: First unchecked decision row.
        fence: End of the evaluation interval.

    Returns:
        Next unchecked decision position.

    Raises:
        DataIntegrityError: Targets, publication times or coverage conflict.
    """
    if not isinstance(window, ExecutionReplayWindow):
        raise DataIntegrityError("windows must be ExecutionReplayWindow")
    if tuple(window.columns) != tuple(expected_columns):
        raise DataIntegrityError("window columns must equal the ordered path target columns")
    symbols = list(window.symbols)
    if len(set(symbols)) != len(symbols):
        raise DataIntegrityError("window symbols must be unique")
    if set(symbols) - set(expected_columns):
        raise DataIntegrityError("window symbols must be a subset of the canonical columns")
    if [c for c in expected_columns if c in set(symbols)] != symbols:
        raise DataIntegrityError("window symbols must follow canonical order")
    if list(window.target_weights.columns) != symbols:
        raise DataIntegrityError("window target columns must equal local symbols")
    decisions = window.target_weights.index
    if not isinstance(decisions, pd.DatetimeIndex):
        raise DataIntegrityError("window decisions must have a DatetimeIndex")
    n = len(decisions)
    if n == 0:
        raise DataIntegrityError("window must contain at least one decision")
    if decisions.has_duplicates or not decisions.is_monotonic_increasing:
        raise DataIntegrityError("window decisions must be chronological without duplication")
    expected_slice = expected_targets.index[cursor : cursor + n]
    if len(expected_slice) != n or not decisions.equals(expected_slice):
        raise DataIntegrityError("window decisions must be the next slice of path.target_weights")
    try:
        window_values = window.target_weights.to_numpy(dtype="float64")
    except (TypeError, ValueError):
        raise DataIntegrityError("window targets must be numeric") from None
    local_values = expected_targets.loc[decisions, symbols].to_numpy(dtype="float64")
    if window_values.shape != local_values.shape or not bool((window_values == local_values).all()):
        raise DataIntegrityError("window targets must exactly match path.target_weights")
    omitted = [c for c in expected_columns if c not in set(symbols)]
    if omitted:
        omitted_values = expected_targets.loc[decisions, omitted].to_numpy(dtype="float64")
        if not bool((omitted_values == 0.0).all()):
            raise DataIntegrityError("omitted canonical targets must be exactly zero")
    if window.marks is None or window.quote_volumes is None:
        raise DataIntegrityError("window requires explicit marks and quote_volumes")
    if window.funding_known is None or window.bar_available_at is None:
        raise DataIntegrityError("window requires explicit funding_known and bar_available_at")
    minute_grid = _require_utc_index(window.minute_grid, "minute_grid")
    if len(minute_grid) < 2 or minute_grid.has_duplicates or not minute_grid.is_monotonic_increasing:
        raise DataIntegrityError("minute_grid must have at least two unique increasing UTC labels")
    diffs = minute_grid.to_series().diff().dropna().unique()
    if len(diffs) != 1 or diffs[0] <= pd.Timedelta(0):
        raise DataIntegrityError("minute_grid must have a constant positive bar interval")
    frames = {
        "highs": window.highs,
        "lows": window.lows,
        "closes": window.closes,
        "marks": window.marks,
        "bar_funding": window.bar_funding,
        "quote_volumes": window.quote_volumes,
        "funding_known": window.funding_known,
    }
    for name, frame in frames.items():
        if not frame.index.equals(minute_grid) or list(frame.columns) != symbols:
            raise DataIntegrityError(f"{name} must share minute_grid and ordered symbol columns")
    if window.funding_known.isna().to_numpy().any():
        raise DataIntegrityError("funding_known must not be missing")
    if bool((window.funding_known.dtypes.apply(lambda dt: dt.kind != "b")).any()):
        raise DataIntegrityError("funding_known must be boolean")
    signal_at = _require_utc_index(window.signal_available_at, "signal_available_at")
    if len(signal_at) != n or not signal_at.is_monotonic_increasing:
        raise DataIntegrityError("signal_available_at must align one-to-one and be non-decreasing")
    for decision_label, signal_label in zip(decisions, signal_at, strict=True):
        if signal_label < decision_label:
            raise DataIntegrityError("signal_available_at must be no earlier than its decision label")
    bar_at = _require_utc_index(window.bar_available_at, "bar_available_at")
    if len(bar_at) != len(minute_grid) or not bar_at.equals(bar_at.sort_values()) or bar_at.has_duplicates:
        raise DataIntegrityError("bar_available_at must be increasing and aligned to minute_grid")
    for bar_label, avail_label in zip(minute_grid, bar_at, strict=True):
        if avail_label < bar_label:
            raise DataIntegrityError("bar_available_at must be no earlier than its bar label")
    for label in list(minute_grid) + list(signal_at):
        if label >= fence:
            raise DataIntegrityError("execution bar labels and signals must be before the fence")
    for avail_label in bar_at:
        if avail_label > fence:
            raise DataIntegrityError("completed bar availability must not be beyond the fence")
    return cursor + n


def replay_process_execution(
    path: ProcessPath,
    windows: Iterable[ExecutionReplayWindow],
    *,
    initial_equity: float,
    execution_bound: _ExecutionBound,
    spec: ExecutionSpec,
    retain_event_snapshots: bool = False,
    min_equity_fraction: float | None = None,
) -> StrategyExecutionReplayResult:
    """Verify the exact sized process targets through the inventory engine.

    The caller supplies point-in-time execution windows and signal release
    times. Targets are checked against the process path, never reconstructed
    from returns or shifted by a guessed availability delay. The existing
    replay engine remains the sole cash, units, fees, and funding authority.

    Args:
        path: Hourly proxy evidence containing the exact sized decision book.
        windows: Chronological execution windows partitioning those decisions
            once, with explicit marks, volume, funding knowledge, and bar
            availability. Window market data may overlap for order resolution.
        initial_equity: Positive finite starting capital.
        execution_bound: Existing supported OHLCV execution model.
        spec: Existing fill and cost contract; no cheaper implicit override.
        retain_event_snapshots: Opt in to dense diagnostic event tables.
        min_equity_fraction: Optional existing inventory-engine ruin guard.

    Returns:
        The unmodified inventory replay result, including validity and gaps.
        An invalid result is evidence of a failed execution verification.

    Raises:
        DataIntegrityError: Targets or execution provenance are missing,
            inconsistent, duplicated, out of order, or cross the forward
            evaluation fence; or the existing engine rejects the inputs.
    """
    if len(path.target_weights) == 0:
        raise DataIntegrityError("path.target_weights must not be empty")
    if list(path.target_weights.columns) == []:
        raise DataIntegrityError("path.target_weights must have symbol columns")
    fence = _execution_fence(path.target_weights)
    expected_columns = list(path.target_weights.columns)
    expected_targets = path.target_weights

    def _validated() -> Iterable[ExecutionReplayWindow]:
        cursor = 0
        for window in windows:
            cursor = _validate_replay_window(
                window,
                expected_columns=expected_columns,
                expected_targets=expected_targets,
                cursor=cursor,
                fence=fence,
            )
            yield window
        if cursor != len(expected_targets):
            raise DataIntegrityError("windows must cover every path decision exactly once")

    return replay_execution_windows(
        _validated(),
        initial_equity,
        execution_bound,
        spec,
        retain_event_snapshots=retain_event_snapshots,
        min_equity_fraction=min_equity_fraction,
    )


@dataclass(frozen=True, slots=True)
class ProcessInventoryReport:
    """Production 3m inventory evidence for identical process decisions."""

    proxy: ProcessBacktestReport
    base: StrategyExecutionReplayResult
    stress: StrategyExecutionReplayResult
    gate: DeployGateResult
    resource_measurements: tuple[MhsResourceMeasurement, ...]
    memory_stats: ProcessTreeMemoryStats


@dataclass(frozen=True, slots=True)
class ProcessInventoryFailureReport:
    """Failed production three-minute evaluation evidence.
Completed coverage means decisions consumed by every required bound, not merely
validated inputs. Unknown totals or resource observations remain null. Partial
coverage and observed source gaps are diagnostics, never completed performance,
deployment certification or proof of operating-system OOM."""

    status: Literal["failed"]
    start: pd.Timestamp
    end: pd.Timestamp
    data_root: str | None
    execution_policy: ProcessExecutionPolicy
    stage: str
    error_code: Literal[
        "MEMORY_BUDGET", "MEMORY_RESERVE", "SWAP_GROWTH",
        "RESOURCE_TELEMETRY", "DATA_INTEGRITY", "UNEXPECTED_ERROR",
    ]
    error_type: str
    error_message: str
    total_decisions: int | None
    validated_decisions: int
    completed_decisions: int
    completed_windows: int
    completed_decision_start: pd.Timestamp | None
    completed_decision_end: pd.Timestamp | None
    source_gaps: tuple[ExecutionDataGap, ...]
    source_gap_excluded_symbols: tuple[str, ...]
    resource_measurements: tuple[MhsResourceMeasurement, ...]
    memory_stats: ProcessTreeMemoryStats | None


class ProcessInventoryBacktestError(DataIntegrityError):
    """Carry typed failure evidence while preserving fail-closed callers."""

    report: ProcessInventoryFailureReport

    def __init__(self, report: ProcessInventoryFailureReport) -> None:
        """Carry typed failure evidence while preserving fail-closed callers.

        Args:
            report: Observed failed-run coverage, provenance and resources.

        Returns:
            None; the original evaluation exception remains the chained cause.
        """
        super().__init__(report.error_message)
        self.report = report


@dataclass(slots=True)
class _ProcessInventoryProgress:
    """Monotonic production replay coverage shared with the evaluator.
Validation and successful all-bound consumption are distinct observations."""

    validated_decisions: int = 0
    completed_decisions: int = 0
    completed_windows: int = 0
    completed_decision_start: pd.Timestamp | None = None
    completed_decision_end: pd.Timestamp | None = None


def _inventory_daily_returns(
    result: StrategyExecutionReplayResult,
    *,
    initial_equity: float = PROCESS_INVENTORY_INITIAL_EQUITY,
) -> pd.Series:
    """Daily net returns resampled from the 3m inventory equity ledger.

    The first daily bar is anchored to ``initial_equity`` so first-day
    profit and loss is never dropped by ``pct_change``.
    """
    if initial_equity <= 0:
        raise DataIntegrityError("initial_equity must be > 0")
    levels = result.ledger.equity.resample("1D").last().astype("float64")
    if len(levels) == 0:
        return levels
    first = pd.Series(
        [float(levels.iloc[0] / initial_equity - 1.0)],
        index=levels.index[:1],
        dtype="float64",
    )
    if len(levels) == 1:
        return first
    rest = levels.pct_change().iloc[1:].astype("float64")
    daily = pd.concat([first, rest]).astype("float64")
    return daily


def _inventory_ledger_summary(result: StrategyExecutionReplayResult) -> dict[str, float]:
    """Primary performance totals derived from the 3m ledger engine fields."""
    daily = _inventory_daily_returns(result)
    if len(daily):
        curve = (1.0 + daily).cumprod()
        years = len(daily) / 365.25
        cagr = float(curve.iloc[-1] ** (1.0 / years) - 1.0)
        anchored = np.concatenate([[1.0], curve.to_numpy(dtype="float64")])
        peak = np.maximum.accumulate(anchored)
        mdd = float((anchored / peak - 1.0).min())
        turnover = float(result.ledger.fill_turnover.sum() * 365.0 / len(daily))
    else:
        cagr = 0.0
        mdd = 0.0
        turnover = 0.0
    return {
        "cagr": cagr,
        "max_drawdown": mdd,
        "annualized_turnover": turnover,
        "total_fees": float(result.ledger.fee_charge.sum()),
        "total_funding": float(result.ledger.funding_charge.sum()),
    }


def _inventory_terminal_state(result: StrategyExecutionReplayResult) -> dict[str, object]:
    """Disclosed terminal inventory split by mark availability."""
    gaps = list(result.ledger.data_gaps)
    unpriced = sorted({g.symbol for g in gaps if g.code in _UNPRICED_TERMINAL_CODES})
    unpriced_set = set(unpriced)
    open_inventory: dict[str, float] = {}
    fills = result.simulated_fills
    if len(fills) and "symbol" in fills.columns and "quantity_delta" in fills.columns:
        totals = fills.groupby("symbol")["quantity_delta"].sum()
        for symbol, quantity in totals.items():
            if float(quantity) != 0.0 and str(symbol) not in unpriced_set:
                open_inventory[str(symbol)] = float(quantity)
    return {
        "primary_valid": bool(result.ledger.primary_valid),
        "invalid_reasons": list(result.ledger.invalid_reasons),
        "terminal_certified": bool(replay_ledger_certified(result)),
        "open_inventory": open_inventory,
        "unpriced_terminal_symbols": unpriced,
        "data_gaps": [
            {"code": g.code, "symbol": g.symbol, "timestamp": g.timestamp.isoformat()} for g in gaps
        ],
    }


def _inventory_result_payload(result: StrategyExecutionReplayResult) -> dict[str, object]:
    """Serializable 3m ledger evidence with engine-native fill accounting."""
    daily = _inventory_daily_returns(result)
    summary = _inventory_ledger_summary(result)
    fills = result.simulated_fills
    return {
        "daily_returns": {ts.isoformat(): float(v) for ts, v in daily.items()},
        "cagr": summary["cagr"],
        "max_drawdown": summary["max_drawdown"],
        "annualized_turnover": summary["annualized_turnover"],
        "total_fees": summary["total_fees"],
        "total_funding": summary["total_funding"],
        "total_fills": len(fills),
        "passive_fills": int(result.fill_count),
        "unfilled_count": int(result.unfilled_count),
        "fallback_count": int(result.fallback_count),
        "forced_exit_count": int(result.forced_exit_count),
        "forced_exit_notional": float(result.forced_exit_notional),
        "termination_counts": dict(result.termination_counts),
        "terminal": _inventory_terminal_state(result),
    }


def _inventory_gate(
    base: StrategyExecutionReplayResult,
    stress: StrategyExecutionReplayResult,
) -> DeployGateResult:
    """Deploy gate over valid inventory daily returns with integrity reasons."""
    base_daily = _inventory_daily_returns(base)
    stress_daily = _inventory_daily_returns(stress)
    if bool(base.ledger.primary_valid) and bool(stress.ledger.primary_valid):
        integrity: tuple[str, ...] = ()
    else:
        integrity = (_INVENTORY_LEDGER_INVALID,)
    envelope = GROWTH_RISK_ENVELOPES[CLI_GROWTH_ENVELOPE_DEFAULT]
    return evaluate_deploy_gate(
        fold_returns=quarter_fold_returns(base_daily),
        fold_stress_returns=quarter_fold_returns(stress_daily),
        integrity_reasons=integrity,
        envelope=envelope,
    )


def _inventory_window_stream(
    path: ProcessPath, signal_available_at: pd.DatetimeIndex, root: str,
    start: pd.Timestamp, end: pd.Timestamp,
    funding_by_symbol: dict[str, pd.Series], funding_failures: Mapping[str, str],
    spec: ExecutionSpec, budget_bytes: int | None, reserve_bytes: int | None,
    recorder: _StageRecorder,
    required_symbols: Callable[[], frozenset[str]] | None = None, *,
    progress: _ProcessInventoryProgress | None = None,
) -> Iterator[ExecutionReplayWindow]:
    """Validate canonical targets and record observed all-bound replay coverage.
Completed coverage advances only after the consumer returns from a yielded
window. Validation alone cannot certify a consumed decision or result."""
    targets = path.target_weights
    expected_columns = list(targets.columns)
    fence = _execution_fence(targets)
    cursor = 0
    for index, window in enumerate(
        _iter_mhs_execution_windows(
            targets,
            signal_available_at,
            root,
            "3m",
            start,
            end,
            funding_by_symbol,
            "cache_required",
            spec,
            funding_failures=funding_failures,
            budget_bytes=budget_bytes,
            reserve_bytes=reserve_bytes,
            execution_bound_count=2,
            required_symbols=required_symbols,
        )
    ):
        n = len(window.target_weights)
        if n == 0:
            recorder.record(
                f"process_3m_window_{index}",
                grid_bars=len(window.minute_grid),
                n_symbols=len(window.symbols),
                window_start=str(window.window_start),
                window_end=str(window.window_end),
                active_symbols=len(window.symbols),
            )
            if progress is not None:
                progress.completed_windows += 1
            yield window
            _assert_execution_rss_budget(
                f"process_3m_window_{index}", budget_bytes, index + 1, reserve_bytes=reserve_bytes
            )
            continue
        cursor = _validate_replay_window(
            window,
            expected_columns=expected_columns,
            expected_targets=targets,
            cursor=cursor,
            fence=fence,
        )
        if progress is not None:
            progress.validated_decisions += n
        recorder.record(
            f"process_3m_window_{index}",
            grid_bars=len(window.minute_grid),
            n_symbols=len(window.symbols),
            window_start=str(window.window_start),
            window_end=str(window.window_end),
            active_symbols=len(window.symbols),
        )
        yield window
        if progress is not None:
            progress.completed_decisions += n
            progress.completed_windows += 1
            if progress.completed_decision_start is None:
                progress.completed_decision_start = window.target_weights.index[0]
            progress.completed_decision_end = window.target_weights.index[-1]
        _assert_execution_rss_budget(
            f"process_3m_window_{index}", budget_bytes, index + 1, reserve_bytes=reserve_bytes
        )
    if cursor != len(targets):
        raise DataIntegrityError("windows must cover every path decision exactly once")


def evaluate_process_inventory_backtest(
    start: pd.Timestamp = DISCOVERY_START,
    end: pd.Timestamp = PROCESS_EVALUATION_CEILING, *,
    data_root: str | None = None,
    execution_policy: ProcessExecutionPolicy | None = None,
    memory_budget: MhsMemoryBudget | None = None,
) -> ProcessInventoryReport:
    """Evaluate identical process decisions through production 3m inventory accounting.

    Args:
        start: Timezone-aware source start.
        end: Timezone-aware end within the registered evaluation ceiling.
        data_root: Existing source root override, resolved per dataset contract.
        execution_policy: Existing adoption policy; None preserves baseline.
        memory_budget: Explicit process-tree limits or validated stage defaults.

    Returns:
        Inventory base/stress results, explicitly comparative hourly evidence,
        gate validity, gap provenance and measured resource evidence.

    Raises:
        ValueError: Dates or policy are invalid.
        DataIntegrityError: Required execution inputs or budget are invalid.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("timestamps must be tz-aware")
    if start >= end:
        raise ValueError("start must be before end")
    if end > PROCESS_EVALUATION_CEILING:
        raise DataIntegrityError(f"end {end} exceeds PROCESS_EVALUATION_CEILING")
    if execution_policy is not None and not isinstance(execution_policy, ProcessExecutionPolicy):
        raise ValueError("execution_policy must be a ProcessExecutionPolicy or None")
    budget = _process_memory_budget(memory_budget)
    run_swap_baseline = _current_tree_swap_bytes()
    sampler = _TreeMemorySampler()
    recorder = _StageRecorder(log_run=False)
    sampler.start()
    sampler.set_stage("preparation")
    recorder.record("process_inventory_start")
    progress = _ProcessInventoryProgress()
    proxy: ProcessBacktestReport | None = None
    targets: pd.DataFrame | None = None
    root: str | None = None
    live_sets: _LiveAccumulatorSets = []
    current_stage = "preparation"
    try:
        budget_bytes, reserve_bytes = _resolve_ram_budget(MHS_REPLAY_BUDGET_BYTES, True)
        proxy = evaluate_process_backtest(
            start, end, data_root=data_root, execution_policy=execution_policy,
            memory_budget=memory_budget,
        )
        recorder.record("process_inventory_proxy")
        path = proxy.base
        targets = path.target_weights
        if len(targets) == 0:
            raise DataIntegrityError("path.target_weights must not be empty")
        sampler.set_stage("replay")
        current_stage = "replay"
        signal_available_at = pd.DatetimeIndex(targets.index + pd.Timedelta(hours=1))
        columns = list(targets.columns)
        _admit_process_stage(
            stage="process_replay_entry", estimated_bytes=_estimate_panel_bytes(len(targets), len(columns), 4),
            budget=budget, replay=True, initial_swap_bytes=run_swap_baseline,
        )
        funding_by_symbol, funding_failures = _load_funding_series(columns)
        root = data_root or str(FUTURES_DATA_DIR / "ohlcv")
        window_start = targets.index[0]
        window_end = _execution_fence(targets)
        base_spec = ExecutionSpec()
        stress_spec = _stress_cost_execution_spec(base_spec)

        def _live_required() -> frozenset[str]:
            if not live_sets:
                return frozenset()
            return live_required_symbols(live_sets[-1])

        stream = _inventory_window_stream(
            path, signal_available_at, root, window_start, window_end,
            funding_by_symbol, funding_failures, base_spec, budget_bytes,
            reserve_bytes, recorder, _live_required, progress=progress,
        )
        bound: _ExecutionBound = "OHLCV_IMMEDIATE_TAKER"
        base, stress = replay_execution_window_batch(
            stream,
            PROCESS_INVENTORY_INITIAL_EQUITY,
            [(bound, base_spec), (bound, stress_spec)],
            live_accumulators=live_sets,
        )
        current_stage = "finalize"
        _assert_stage_rss_budget("process_3m_replay", budget_bytes, reserve_bytes)
        recorder.record("process_inventory_replay")
        gate = _inventory_gate(base, stress)
        recorder.record("process_inventory_total")
        memory_stats = sampler.stop()
    except Exception as exc:
        with suppress(Exception):
            recorder.record("process_inventory_failed")
        try:
            failure_stats = sampler.stop()
        except Exception:
            failure_stats = None
        stage: str
        error_code: Literal[
            "MEMORY_BUDGET", "MEMORY_RESERVE", "SWAP_GROWTH",
            "RESOURCE_TELEMETRY", "DATA_INTEGRITY", "UNEXPECTED_ERROR",
        ]
        if isinstance(exc, MhsResourceAdmissionError):
            stage = exc.stage
            error_code = exc.error_code
        elif isinstance(exc, DataIntegrityError):
            error_code = "DATA_INTEGRITY"
            stage = f"process_3m_window_{progress.completed_windows}" if current_stage == "replay" else current_stage
        else:
            error_code = "UNEXPECTED_ERROR"
            stage = f"process_3m_window_{progress.completed_windows}" if current_stage == "replay" else current_stage
        observed: list[ExecutionDataGap] = []
        seen_keys: set[tuple[object, ...]] = set()
        for acc_set in live_sets:
            for acc in acc_set:
                if acc is None:
                    continue
                for gap in acc.data_gaps:
                    key = (
                        gap.code, gap.symbol, gap.timestamp,
                        gap.decision_time, gap.signal_time, gap.execution_bound,
                    )
                    if key not in seen_keys:
                        seen_keys.add(key)
                        observed.append(gap)
        observed.sort(key=lambda g: (g.timestamp, g.code, g.symbol))
        if proxy is not None:
            effective_policy = proxy.base.execution_policy
        elif execution_policy is not None:
            effective_policy = execution_policy
        else:
            effective_policy = ProcessExecutionPolicy()
        failure_report = ProcessInventoryFailureReport(
            status="failed",
            start=start,
            end=end,
            data_root=root if root is not None else data_root,
            execution_policy=effective_policy,
            stage=stage,
            error_code=error_code,
            error_type=type(exc).__name__,
            error_message=str(exc),
            total_decisions=len(targets) if targets is not None else None,
            validated_decisions=progress.validated_decisions,
            completed_decisions=progress.completed_decisions,
            completed_windows=progress.completed_windows,
            completed_decision_start=progress.completed_decision_start,
            completed_decision_end=progress.completed_decision_end,
            source_gaps=tuple(observed),
            source_gap_excluded_symbols=tuple(sorted(SOURCE_GAP_EXCLUDED_SYMBOLS)),
            resource_measurements=recorder.records,
            memory_stats=failure_stats,
        )
        failure = ProcessInventoryBacktestError(failure_report)
        setattr(failure, "resource_measurements", recorder.records)  # noqa: B010
        setattr(failure, "memory_stats", failure_stats)  # noqa: B010
        raise failure from exc
    return ProcessInventoryReport(
        proxy=proxy,
        base=base,
        stress=stress,
        gate=gate,
        resource_measurements=recorder.records,
        memory_stats=memory_stats,
    )


def persist_process_inventory_report(report: ProcessInventoryReport, output: Path) -> Path:
    """Persist separately identified 3m execution evidence.

    Args:
        report: Inventory evaluation and explicit comparison evidence.
        output: JSON destination distinct from reserved hourly evidence.

    Returns:
        The persisted report destination.

    Raises:
        ValueError: Output format is unsupported.
        DataIntegrityError: Destination would overwrite reserved evidence.
    """
    out = Path(output)
    if out.suffix != ".json":
        raise ValueError(f"destination must be a JSON path, got {output}")
    reserved = {PROCESS_REPORT_PATH.resolve(), PROCESS_POLICY_REPORT_PATH.resolve()}
    if out.resolve() in reserved:
        raise DataIntegrityError(f"destination {out} would overwrite reserved hourly evidence")
    decisions = report.proxy.base.target_weights.index
    payload = {
        "status": "completed",
        "execution_timeframe": "3m",
        "start": report.proxy.start.isoformat(),
        "end": report.proxy.end.isoformat(),
        "certification_level": PROCESS_INVENTORY_CERTIFICATION_LEVEL,
        "source_policy": {"tracking_error_threshold": report.proxy.base.execution_policy.tracking_error_threshold},
        "periods": {
            "decision_start": decisions[0].isoformat() if len(decisions) else None,
            "decision_end": decisions[-1].isoformat() if len(decisions) else None,
            "n_decisions": len(decisions),
        },
        "gate": {
            "go": report.gate.go,
            "reason_codes": list(report.gate.reason_codes),
            "metrics": dict(report.gate.metrics),
        },
        "base": _inventory_result_payload(report.base),
        "stress": _inventory_result_payload(report.stress),
        "proxy": {
            "scope": "hourly_proxy_comparison",
            "certification_level": report.proxy.certification_level,
            "base": _tier_payload(report.proxy.base),
            "stress": _tier_payload(report.proxy.stress),
        },
        "resource_measurements": [dataclasses.asdict(m) for m in report.resource_measurements],
        "memory_stats": dataclasses.asdict(report.memory_stats),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return out


def _failure_gap_payload(gap: ExecutionDataGap) -> dict[str, object]:
    return {
        "code": gap.code,
        "symbol": gap.symbol,
        "timestamp": gap.timestamp.isoformat(),
        "decision_time": gap.decision_time.isoformat() if gap.decision_time is not None else None,
        "signal_time": gap.signal_time.isoformat() if gap.signal_time is not None else None,
        "execution_bound": gap.execution_bound,
    }


def persist_process_inventory_failure(
    report: ProcessInventoryFailureReport, output: Path,
) -> Path:
    """Persist failure diagnostics without overwriting certified evidence.

    Args:
        report: Typed failed evaluation evidence, never a performance report.
        output: Dedicated JSON failure destination.

    Returns:
        Persisted failure artifact path.

    Raises:
        ValueError: Destination format is unsupported.
        DataIntegrityError: Destination conflicts with reserved or success evidence.
        OSError: Persistence fails; the caller must also retain evaluation cause.
    """
    out = Path(output)
    if out.suffix != ".json":
        raise ValueError(f"destination must be a JSON path, got {output}")
    reserved = {
        PROCESS_REPORT_PATH.resolve(),
        PROCESS_POLICY_REPORT_PATH.resolve(),
        PROCESS_INVENTORY_REPORT_PATH.resolve(),
    }
    if out.resolve() in reserved:
        raise DataIntegrityError(f"destination {out} would overwrite reserved evidence")
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            existing = None
        if isinstance(existing, dict) and existing.get("status") == "completed":
            raise DataIntegrityError(f"destination {out} already holds completed evidence")
    payload: dict[str, object] = {
        "status": "failed",
        "execution_timeframe": "3m",
        "certification_level": PROCESS_INVENTORY_CERTIFICATION_LEVEL,
        "start": report.start.isoformat(),
        "end": report.end.isoformat(),
        "data_root": report.data_root,
        "execution_policy": {
            "tracking_error_threshold": report.execution_policy.tracking_error_threshold
        },
        "stage": report.stage,
        "error_code": report.error_code,
        "error_type": report.error_type,
        "error_message": report.error_message,
        "total_decisions": report.total_decisions,
        "validated_decisions": report.validated_decisions,
        "completed_decisions": report.completed_decisions,
        "completed_windows": report.completed_windows,
        "completed_decision_start": (
            report.completed_decision_start.isoformat()
            if report.completed_decision_start is not None else None
        ),
        "completed_decision_end": (
            report.completed_decision_end.isoformat()
            if report.completed_decision_end is not None else None
        ),
        "source_gaps": [_failure_gap_payload(g) for g in report.source_gaps],
        "source_gap_excluded_symbols": list(report.source_gap_excluded_symbols),
        "resource_measurements": [dataclasses.asdict(m) for m in report.resource_measurements],
        "memory_stats": (
            dataclasses.asdict(report.memory_stats) if report.memory_stats is not None else None
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=str(out.parent), suffix=".tmp", delete=False, encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name
            tmp.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, out)
    except Exception:
        if tmp_path is not None:
            with suppress(Exception):
                os.unlink(tmp_path)
        raise
    return out
