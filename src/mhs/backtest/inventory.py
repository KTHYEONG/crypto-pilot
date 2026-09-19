"""Replay original process targets through the primary three-minute engine."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.paths import FUTURES_DATA_DIR
from src.mhs.backtest.certification import (
    REQUIRED_CHECK_NAMES,
    EvaluationContext,
    EvidenceCheck,
    assess_process_validation,
)
from src.mhs.backtest.contracts import (
    ProcessBacktestReport,
    ProcessInventoryBacktestError,
    ProcessInventoryFailureReport,
    ProcessInventoryReport,
    ProcessPath,
)
from src.mhs.backtest.market_data import _admit_process_stage, _estimate_panel_bytes
from src.mhs.backtest.paths import evaluate_process_backtest, quarter_fold_returns
from src.mhs.data_policy import SOURCE_GAP_EXCLUDED_SYMBOLS
from src.mhs.deploy_gate import DeployGateResult, evaluate_deploy_gate
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
    FundingCoverageGap,
    StrategyExecutionReplayResult,
)
from src.mhs.execution.specs import _stress_cost_execution_spec
from src.mhs.execution.window_stream import _iter_mhs_execution_windows
from src.mhs.marks import _load_funding_series
from src.mhs.params import (
    CLI_GROWTH_ENVELOPE_DEFAULT,
    DISCOVERY_START,
    GROWTH_RISK_ENVELOPES,
    PROCESS_EVALUATION_CEILING,
)
from src.mhs.process import ProcessExecutionPolicy, ProcessRiskSizingSpec
from src.mhs.resources import (
    MhsMemoryBudget,
    MhsResourceAdmissionError,
    _assert_execution_rss_budget,
    _assert_stage_rss_budget,
    _current_tree_swap_bytes,
    _StageRecorder,
    _TreeMemorySampler,
    resolve_mhs_memory_budget,
)
from src.mhs.types import ExecutionSpec

if TYPE_CHECKING:
    from src.mhs.backtest.journal import ProcessProcedureDefinition
    from src.mhs.backtest.labels import MaturedMemberReturns

_logger = logging.getLogger(__name__)

PROCESS_INVENTORY_INITIAL_EQUITY: float = 1.0
_INVENTORY_LEDGER_INVALID: str = "INVENTORY_LEDGER_INVALID"
_UNPRICED_TERMINAL_CODES: frozenset[str] = frozenset(
    {"MISSING_HELD_MARK", "MISSING_HELD_FUNDING", "MISSING_DECISION_MARK"}
)
_CANONICAL_MARK_TO_OHLCV_GAP: dict[str, str] = {
    "MISSING_DECISION_MARK": "MISSING_ORDER_OHLCV",
    "MISSING_HELD_MARK": "MISSING_FORCED_EXIT_CLOSE",
}


def _to_canonical_ohlcv_gap(gap: ExecutionDataGap) -> ExecutionDataGap:
    """Map legacy mark-cache gap codes onto OHLCV source gaps for canonical replay."""
    mapped = _CANONICAL_MARK_TO_OHLCV_GAP.get(str(gap.code))
    if mapped is None:
        return gap
    return ExecutionDataGap(
        code=mapped,  # type: ignore[arg-type]
        symbol=gap.symbol,
        timestamp=gap.timestamp,
        decision_time=gap.decision_time,
        signal_time=gap.signal_time,
        execution_bound=gap.execution_bound,
    )


def _actual_excluded_symbols(target_columns: list[str] | None) -> tuple[str, ...]:
    """Report only static-registry symbols actually absent from this target path."""
    if not target_columns:
        return ()
    held = set(target_columns)
    return tuple(sorted(s for s in SOURCE_GAP_EXCLUDED_SYMBOLS if s not in held))


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
    if window.quote_volumes is None:
        raise DataIntegrityError("window requires explicit quote_volumes")
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
        "bar_funding": window.bar_funding,
        "quote_volumes": window.quote_volumes,
        "funding_known": window.funding_known,
    }
    if window.marks is not None:
        frames["marks"] = window.marks
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
    path: ProcessPath,
    signal_available_at: pd.DatetimeIndex,
    root: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    funding_by_symbol: dict[str, pd.Series],
    funding_failures: Mapping[str, str],
    spec: ExecutionSpec,
    budget_bytes: int | None,
    reserve_bytes: int | None,
    recorder: _StageRecorder,
    required_symbols: Callable[[], frozenset[str]] | None = None,
    *,
    progress: _ProcessInventoryProgress | None = None,
    initial_swap_bytes: int | None = None,
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
            spec,
            funding_failures=funding_failures,
            budget_bytes=budget_bytes,
            reserve_bytes=reserve_bytes,
            execution_bound_count=2,
            required_symbols=required_symbols,
            initial_swap_bytes=initial_swap_bytes,
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
            _logger.info(
                "[DATA] stage=process_3m_window_loaded window=%d decisions=0",
                index,
            )
            if progress is not None:
                progress.completed_windows += 1
            yield window
            if progress is not None:
                _logger.info(
                    "[DATA] stage=process_3m_window_consumed window=%d completed_windows=%d completed_decisions=%d",
                    index,
                    progress.completed_windows,
                    progress.completed_decisions,
                )
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
        _logger.info(
            "[DATA] stage=process_3m_window_loaded window=%d decisions=%d",
            index,
            n,
        )
        yield window
        if progress is not None:
            progress.completed_decisions += n
            progress.completed_windows += 1
            if progress.completed_decision_start is None:
                progress.completed_decision_start = window.target_weights.index[0]
            progress.completed_decision_end = window.target_weights.index[-1]
            _logger.info(
                "[DATA] stage=process_3m_window_consumed window=%d completed_windows=%d "
                "completed_decisions=%d completed_decision_start=%s completed_decision_end=%s",
                index,
                progress.completed_windows,
                progress.completed_decisions,
                progress.completed_decision_start,
                progress.completed_decision_end,
            )
        _assert_execution_rss_budget(f"process_3m_window_{index}", budget_bytes, index + 1, reserve_bytes=reserve_bytes)
    if cursor != len(targets):
        raise DataIntegrityError("windows must cover every path decision exactly once")


def _union_funding_coverage(
    *sources: tuple[FundingCoverageGap, ...],
) -> tuple[FundingCoverageGap, ...]:
    """Union coverage intervals without duplicate identical entries."""
    seen: dict[tuple[str, pd.Timestamp, pd.Timestamp, str], FundingCoverageGap] = {}
    for gaps in sources:
        for gap in gaps:
            seen[(gap.symbol, gap.start, gap.end, gap.reason)] = gap
    return tuple(sorted(seen.values(), key=lambda g: (g.start, g.end, g.symbol)))


def _live_funding_coverage(live_sets: _LiveAccumulatorSets) -> tuple[FundingCoverageGap, ...]:
    """Union coverage intervals observed by live accumulators."""
    seen: dict[tuple[str, pd.Timestamp, pd.Timestamp, str], FundingCoverageGap] = {}
    for acc_set in live_sets:
        for acc in acc_set:
            if acc is None:
                continue
            for gap in acc.funding_coverage_gaps.values():
                seen[(gap.symbol, gap.start, gap.end, gap.reason)] = gap
    return tuple(sorted(seen.values(), key=lambda g: (g.start, g.end, g.symbol)))


def build_process_evidence_checks(
    procedure: ProcessProcedureDefinition,
    context: EvaluationContext,
    proxy: ProcessBacktestReport,
    base: StrategyExecutionReplayResult,
    stress: StrategyExecutionReplayResult,
    *,
    memory_budget: MhsMemoryBudget,
) -> tuple[EvidenceCheck, ...]:
    """Produce requirement evidence from source-owned validated outputs, not caller flags.

    Args:
        procedure: Exact frozen decision and economic definition.
        context: Journal-reserved identities and assessed interval.
        proxy: Auditable label, policy-choice and target generation evidence.
        base: Primary execution accounting and terminal observations.
        stress: Exact-decision cost-stress execution result.
        memory_budget: Existing guards for input validation and evidence evaluation.
    Returns:
        One matching passed, failed or unverified record for every required check.
    Raises:
        DataIntegrityError: Producer identities, inputs or interval coverage conflict.
    """
    from src.mhs.backtest.journal import ProcessProcedureDefinition as _Procedure
    from src.mhs.backtest.journal import process_procedure_digest as _digest

    if not isinstance(procedure, _Procedure):
        raise DataIntegrityError("procedure must be a ProcessProcedureDefinition")
    if not isinstance(context, EvaluationContext):
        raise DataIntegrityError("context must be an EvaluationContext")
    if not isinstance(proxy, ProcessBacktestReport):
        raise DataIntegrityError("proxy must be a ProcessBacktestReport")
    if not isinstance(base, StrategyExecutionReplayResult):
        raise DataIntegrityError("base must be a StrategyExecutionReplayResult")
    if not isinstance(stress, StrategyExecutionReplayResult):
        raise DataIntegrityError("stress must be a StrategyExecutionReplayResult")
    if not isinstance(memory_budget, MhsMemoryBudget):
        raise DataIntegrityError("memory_budget must be a MhsMemoryBudget")
    if context.procedure_digest != _digest(procedure):
        raise DataIntegrityError("context procedure identity does not match the procedure")
    if context.code_digest != procedure.code_digest:
        raise DataIntegrityError("context code identity does not match the procedure")
    names = tuple(procedure.required_checks)
    if len(set(names)) != len(names):
        raise DataIntegrityError("procedure required checks must be unique")
    for name in names:
        if name not in REQUIRED_CHECK_NAMES:
            raise DataIntegrityError(f"unknown requirement '{name}'")
    return tuple(
        EvidenceCheck(
            requirement=name,
            status="unverified",
            procedure_digest=context.procedure_digest,
            input_manifest_digest=context.input_manifest_digest,
            code_digest=context.code_digest,
            interval_start=context.interval_start,
            interval_end=context.interval_end,
            artifact_digest=None,
            reason_codes=("PRODUCER_ABSENT",),
        )
        for name in names
    )


def evaluate_process_inventory_backtest(
    start: pd.Timestamp = DISCOVERY_START,
    end: pd.Timestamp = PROCESS_EVALUATION_CEILING,
    *,
    data_root: str | None = None,
    execution_policy: ProcessExecutionPolicy | None = None,
    risk_sizing: ProcessRiskSizingSpec | None = None,
    memory_budget: MhsMemoryBudget | None = None,
    procedure: ProcessProcedureDefinition | None = None,
    evaluation_context: EvaluationContext | None = None,
    member_evidence: MaturedMemberReturns | None = None,
) -> ProcessInventoryReport:
    """Validate the canonical target against the streamed trade bars, source publication times, funding knowledge and exact replay ledger. OHLCV-only valuation is explicit in the result; absent mark data has no bearing on historical strategy validity.

    Preparation, replay-window and finalization diagnostics are emitted before
    completion so a resource termination cannot erase the last observed phase.
    Resource observations never change financial decisions or certification.

    Args:
        start: Existing timezone-aware full source start.
        end: Existing timezone-aware source end within the registered ceiling.
        data_root: Existing OHLCV source override.
        execution_policy: Existing adoption policy or baseline.
        risk_sizing: Research-only causal volatility sizing; None preserves baseline.
        memory_budget: Explicit stage limits or validated defaults.
        procedure: Frozen decision definition; None preserves legacy research.
        evaluation_context: Reserved journal context; None yields historical context.
        member_evidence: Inventory-native labels admitted only on identity match.
    Returns:
        Original base/stress inventory evidence with independent validity flags.
    Raises:
        ValueError: Existing request contracts are invalid.
        ProcessInventoryBacktestError: Evaluation fails with preserved diagnostics.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("timestamps must be tz-aware")
    if start >= end:
        raise ValueError("start must be before end")
    if end > PROCESS_EVALUATION_CEILING:
        raise DataIntegrityError(f"end {end} exceeds PROCESS_EVALUATION_CEILING")
    if execution_policy is not None and not isinstance(execution_policy, ProcessExecutionPolicy):
        raise ValueError("execution_policy must be a ProcessExecutionPolicy or None")
    if risk_sizing is not None and not isinstance(risk_sizing, ProcessRiskSizingSpec):
        raise ValueError("risk_sizing must be a ProcessRiskSizingSpec or None")
    if procedure is not None:
        from src.mhs.backtest.journal import ProcessProcedureDefinition as _Procedure

        if not isinstance(procedure, _Procedure):
            raise DataIntegrityError("procedure must be a ProcessProcedureDefinition or None")
    if evaluation_context is not None and not isinstance(evaluation_context, EvaluationContext):
        raise DataIntegrityError("evaluation_context must be an EvaluationContext or None")
    if member_evidence is not None:
        from src.mhs.backtest.labels import MaturedMemberReturns as _Labels

        if not isinstance(member_evidence, _Labels):
            raise DataIntegrityError("member_evidence must be MaturedMemberReturns or None")
    budget = resolve_mhs_memory_budget(memory_budget)
    run_swap_baseline = _current_tree_swap_bytes()
    sampler = _TreeMemorySampler()
    recorder = _StageRecorder(log_run=True)
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
        budget_bytes, reserve_bytes = budget.replay_tree_pss_bytes, budget.min_available_bytes
        proxy = evaluate_process_backtest(
            start,
            end,
            data_root=data_root,
            execution_policy=execution_policy,
            risk_sizing=risk_sizing,
            memory_budget=budget,
            procedure=procedure,
            member_evidence=member_evidence,
        )
        recorder.record("process_inventory_proxy")
        path = proxy.base
        targets = path.target_weights
        if len(targets) == 0:
            raise DataIntegrityError("path.target_weights must not be empty")
        sampler.set_stage("replay")
        current_stage = "replay"
        signal_available_at = (
            path.signal_available_at
            if path.signal_available_at is not None
            else pd.DatetimeIndex(targets.index + pd.Timedelta(hours=1))
        )
        columns = list(targets.columns)
        _admit_process_stage(
            stage="process_replay_entry",
            estimated_bytes=_estimate_panel_bytes(len(targets), len(columns), 4),
            budget=budget,
            replay=True,
            initial_swap_bytes=run_swap_baseline,
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
            path,
            signal_available_at,
            root,
            window_start,
            window_end,
            funding_by_symbol,
            funding_failures,
            base_spec,
            budget_bytes,
            reserve_bytes,
            recorder,
            _live_required,
            progress=progress,
            initial_swap_bytes=run_swap_baseline,
        )
        bound: _ExecutionBound = "OHLCV_IMMEDIATE_TAKER"
        recorder.record("process_inventory_replay_start")
        _logger.info("[DATA] stage=replay_start windows=%d", len(targets))
        initial_equity = float(procedure.initial_equity) if procedure is not None else PROCESS_INVENTORY_INITIAL_EQUITY
        base, stress = replay_execution_window_batch(
            stream,
            initial_equity,
            [(bound, base_spec), (bound, stress_spec)],
            live_accumulators=live_sets,
        )
        current_stage = "finalize"
        recorder.record("process_inventory_finalize")
        _logger.info("[DATA] stage=finalize_start completed_windows=%d", progress.completed_windows)
        _assert_stage_rss_budget("process_3m_replay", budget_bytes, reserve_bytes)
        recorder.record("process_inventory_replay")
        if procedure is not None:
            from src.mhs.backtest.journal import process_procedure_digest as _digest

            envelope = procedure.envelope
            interval_start = start.tz_convert("UTC")
            interval_end = end.tz_convert("UTC")
            if evaluation_context is not None:
                resolved_context = evaluation_context
            else:
                resolved_context = EvaluationContext(
                    role="historical",
                    procedure_digest=_digest(procedure),
                    code_digest=procedure.code_digest,
                    input_manifest_digest=None,
                    interval_start=interval_start,
                    interval_end=interval_end,
                    registered_at=None,
                    consulted_through=None,
                    family_id=None,
                    look_ordinal=None,
                    inference_spec=procedure.inference,
                    journal_complete=False,
                    observed_through=interval_end,
                )
            checks = build_process_evidence_checks(
                procedure, resolved_context, proxy, base, stress, memory_budget=budget
            )
            validation = assess_process_validation(
                base, stress, context=resolved_context, checks=checks, envelope=envelope, memory_budget=budget
            )
        else:
            envelope = GROWTH_RISK_ENVELOPES[CLI_GROWTH_ENVELOPE_DEFAULT]
            interval_start = start.tz_convert("UTC")
            interval_end = end.tz_convert("UTC")
            legacy_context = evaluation_context
            if legacy_context is None:
                legacy_context = EvaluationContext(
                    role="historical",
                    procedure_digest="unregistered",
                    code_digest="unregistered",
                    input_manifest_digest=None,
                    interval_start=interval_start,
                    interval_end=interval_end,
                    registered_at=None,
                    consulted_through=None,
                    family_id=None,
                    look_ordinal=None,
                    inference_spec=None,
                    journal_complete=False,
                    observed_through=interval_end,
                )
            evidence_checks: tuple[EvidenceCheck, ...] = ()
            validation = assess_process_validation(
                base, stress, context=legacy_context, checks=evidence_checks, envelope=envelope, memory_budget=budget
            )
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
            "MEMORY_BUDGET",
            "MEMORY_RESERVE",
            "SWAP_GROWTH",
            "RESOURCE_TELEMETRY",
            "DATA_INTEGRITY",
            "UNEXPECTED_ERROR",
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
                    canonical = _to_canonical_ohlcv_gap(gap)
                    key = (
                        canonical.code,
                        canonical.symbol,
                        canonical.timestamp,
                        canonical.decision_time,
                        canonical.signal_time,
                        canonical.execution_bound,
                    )
                    if key not in seen_keys:
                        seen_keys.add(key)
                        observed.append(canonical)
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
            source_gap_excluded_symbols=_actual_excluded_symbols(
                list(targets.columns) if targets is not None else None
            ),
            resource_measurements=recorder.records,
            memory_stats=failure_stats,
            funding_coverage_gaps=_live_funding_coverage(live_sets),
        )
        failure = ProcessInventoryBacktestError(failure_report)
        setattr(failure, "resource_measurements", recorder.records)  # noqa: B010
        setattr(failure, "memory_stats", failure_stats)  # noqa: B010
        raise failure from exc
    return ProcessInventoryReport(
        proxy=proxy,
        base=base,
        stress=stress,
        gate=validation.gate,
        resource_measurements=recorder.records,
        memory_stats=memory_stats,
        funding_coverage_gaps=_union_funding_coverage(base.funding_coverage_gaps, stress.funding_coverage_gaps),
        validation=validation,
    )
