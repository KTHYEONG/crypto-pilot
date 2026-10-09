"""Historical strategy inventory runner on the shared 3m ledger."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.paths import FUTURES_DATA_DIR
from src.core.data_provenance import resolve_mhs_input_layout
from src.core.instrument_settlements import (
    InstrumentSettlementRegistry,
    settlement_registry_for_root,
    source_gap_superseded_by_settlement,
)
from src.core.marks import _load_funding_series
from src.core.panel import load_base_panel
from src.core.params import DELIST_ROSTER_BLOCK_LEAD
from src.core.resources import (
    MhsMemoryBudget,
    _current_tree_swap_bytes,
    assert_mhs_stage_allocation,
    resolve_mhs_memory_budget,
)
from src.core.settlement_evidence import assert_settlement_registry_complete
from src.core.source_gaps import active_intervals, load_source_gap_registry
from src.core.types import ExecutionSpec
from src.core.venue_halts import venue_halt_registry_for_root
from src.engine.backtest_evidence import (
    StrategyBacktestEvidence,
    StrategyExecutionBound,
    StrategyReportPeriod,
    evaluate_strategy_backtest,
)
from src.engine.execution import ExecutionReplayWindow, live_required_symbols
from src.engine.execution.batch import _LiveAccumulatorSets
from src.engine.execution.window_stream import _iter_mhs_execution_windows
from src.strategy.targets import (
    MemberSnapshotCache,
    StrategySpec,
    StrategyTargets,
    build_strategy_targets,
)
from src.strategy.universe import build_pit_roster

if TYPE_CHECKING:
    from src.core.source_gaps import SourceGapInterval


@dataclass(frozen=True, slots=True)
class StrategyBacktestRequest:
    """Describe one reproducible historical strategy inventory experiment.

    Source history is distinct from the scored interval so liquidity and feature
    warm-up are observable rather than manufactured.  The request identifies
    a target policy, exact execution cost bounds, and report periods without
    allowing the runner to select a better strategy from its results.

    ``execution_bound`` selects how both cost cases cross: immediate taker,
    or a resting maker limit for ``passive_timeout_minutes`` with the unfilled remainder crossing
    as taker. Both require the ``submit_bar`` decision anchor so no order is sized or priced off a
    mark published after its own submission.
    """

    source_start: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    strategy: StrategySpec
    initial_equity: float
    base_spec: ExecutionSpec
    stress_spec: ExecutionSpec
    report_periods: tuple[StrategyReportPeriod, ...]
    data_root: Path | None = None
    memory_budget: MhsMemoryBudget | None = None
    execution_bound: StrategyExecutionBound = "OHLCV_IMMEDIATE_TAKER"

    def __post_init__(self) -> None:
        for name in ("source_start", "evaluation_start", "evaluation_end"):
            value = getattr(self, name)
            if not isinstance(value, pd.Timestamp) or pd.isna(value):
                raise DataIntegrityError(f"{name} must be a valid timestamp")
            if value.tzinfo is None or value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
                raise DataIntegrityError(f"{name} must be timezone-aware UTC")
        if not self.source_start < self.evaluation_start < self.evaluation_end:
            raise DataIntegrityError("request must satisfy source_start < evaluation_start < evaluation_end")
        if not isinstance(self.strategy, StrategySpec):
            raise DataIntegrityError("strategy must be a StrategySpec")
        if (
            isinstance(self.initial_equity, bool)
            or not isinstance(self.initial_equity, (int, float))
            or not np.isfinite(float(self.initial_equity))
            or not float(self.initial_equity) > 0.0
        ):
            raise DataIntegrityError("initial_equity must be a positive finite capital")
        for label, spec in (("base_spec", self.base_spec), ("stress_spec", self.stress_spec)):
            if not isinstance(spec, ExecutionSpec):
                raise DataIntegrityError(f"{label} must be an ExecutionSpec")
        if self.base_spec.one_way_taker_bps() != 6.0 or self.stress_spec.one_way_taker_bps() != 18.0:
            raise DataIntegrityError("base cost must be 6 bps and stress cost 18 bps one-way")
        if self.execution_bound not in ("OHLCV_IMMEDIATE_TAKER", "OHLCV_STRICT_PROXY"):
            raise DataIntegrityError(f"execution_bound must be a registered crossing model, got {self.execution_bound!r}")
        if self.base_spec.decision_anchor != "submit_bar" or self.stress_spec.decision_anchor != "submit_bar":
            raise DataIntegrityError("base and stress specs must use decision_anchor='submit_bar'")
        if not isinstance(self.report_periods, tuple) or not self.report_periods:
            raise DataIntegrityError("report_periods must be a non-empty tuple of StrategyReportPeriod")
        if any(not isinstance(p, StrategyReportPeriod) for p in self.report_periods):
            raise DataIntegrityError("report_periods must be a non-empty tuple of StrategyReportPeriod")
        if self.data_root is not None and not isinstance(self.data_root, Path):
            raise DataIntegrityError("data_root must be a Path or None")
        if self.memory_budget is not None and not isinstance(self.memory_budget, MhsMemoryBudget):
            raise DataIntegrityError("memory_budget must be a MhsMemoryBudget or None")


@dataclass(frozen=True, slots=True)
class StrategySourceContext:
    """One loaded source bundle shared by the inventory replay and alternative ledgers."""

    census: tuple[str, ...]
    funding_by_symbol: dict[str, pd.Series]
    funding_failures: dict[str, str]
    root: str
    budget: MhsMemoryBudget
    daily_close: pd.DataFrame
    daily_quote_volume: pd.DataFrame


@dataclass(frozen=True, slots=True)
class LoadedStrategySource:
    """One loaded source bundle shared by the inventory replay and alternative ledgers."""

    fingerprint: tuple[object, ...]
    daily_close: pd.DataFrame
    daily_quote_volume: pd.DataFrame
    hourly_panels: dict[str, pd.DataFrame]
    hourly_available_at: pd.DataFrame
    census: tuple[str, ...]
    funding_by_symbol: dict[str, pd.Series]
    funding_failures: dict[str, str]
    root: str
    budget: MhsMemoryBudget


def _strategy_source_fingerprint(request: StrategyBacktestRequest, root: str) -> tuple[object, ...]:
    import hashlib

    settlement_digest = settlement_registry_for_root(root).digest
    halt_digest = venue_halt_registry_for_root(root).digest
    gap_digest = hashlib.sha256(repr(load_source_gap_registry()).encode("utf-8")).hexdigest()
    return (
        request.source_start.isoformat(),
        request.evaluation_end.isoformat(),
        str(root),
        settlement_digest,
        gap_digest,
        halt_digest,
    )


def load_strategy_source(request: StrategyBacktestRequest) -> LoadedStrategySource:
    """Read the complete historical 1h census and derive daily and funding planes."""
    budget = resolve_mhs_memory_budget(request.memory_budget)
    initial_swap_bytes = _current_tree_swap_bytes()
    _admit_source_stage(budget, initial_swap_bytes)
    daily_close, daily_quote_volume, hourly_panels, hourly_available_at, census, funding_by_symbol, funding_failures, root = _load_strategy_source(
        request, budget, initial_swap_bytes
    )
    fingerprint = _strategy_source_fingerprint(request, root)
    return LoadedStrategySource(
        fingerprint=fingerprint,
        daily_close=daily_close,
        daily_quote_volume=daily_quote_volume,
        hourly_panels=dict(hourly_panels),
        hourly_available_at=hourly_available_at,
        census=tuple(census),
        funding_by_symbol=dict(funding_by_symbol),
        funding_failures=dict(funding_failures),
        root=str(root),
        budget=budget,
    )


@dataclass(frozen=True, slots=True)
class StrategyBacktestRun:
    """Return exact target provenance and paired 3m evidence for one request."""

    request: StrategyBacktestRequest
    candidate: StrategyTargets
    evidence: StrategyBacktestEvidence
    execution_start: pd.Timestamp
    execution_end: pd.Timestamp
    source_symbols: tuple[str, ...]
    source_gap_excluded_symbols: tuple[str, ...] = ()
    source_gap_blocked_decisions: int = 0
    delisting_blocked_decisions: int = 0
    data_availability_withdrawals: tuple[Mapping[str, object], ...] = ()


class LakeCoverageError(DataIntegrityError):
    """Raised when the local 3m lake lacks bars for a selected instrument."""


def _strategy_delisting_block(
    decision_index: pd.DatetimeIndex,
    census: list[str],
    column_of: dict[str, int],
    *,
    snapshot_hour: int,
    settlement_registry: InstrumentSettlementRegistry,
) -> np.ndarray:
    """Cause-1 roster withdrawal mirroring the live delisting block, PIT by announcement."""
    values = np.zeros((len(decision_index), len(census)), dtype=bool)
    if not census or not settlement_registry.settlements:
        return values
    snapshot_instants = decision_index + pd.Timedelta(hours=snapshot_hour)
    horizon_ends = decision_index + pd.Timedelta(days=1, hours=snapshot_hour) + DELIST_ROSTER_BLOCK_LEAD
    for record in settlement_registry.settlements:
        column = column_of.get(record.symbol)
        if column is None:
            continue
        announced = np.asarray(snapshot_instants >= record.announced_at, dtype=bool)
        reaches = np.asarray(horizon_ends >= record.delivery_at, dtype=bool)
        values[:, column] |= announced & reaches
    return values


def _withdrawable_gap_intervals(
    settlement_registry: InstrumentSettlementRegistry,
) -> list[SourceGapInterval]:
    """Unresolved 3m gaps that withdraw a roster seat: INTERIOR and LISTING_EDGE only."""
    return [
        iv
        for iv in active_intervals(plane="ohlcv_3m")
        if iv.extent in ("INTERIOR", "LISTING_EDGE")
        and not source_gap_superseded_by_settlement(iv, settlement_registry)
    ]


def _unresolved_lake_gap_intervals(
    settlement_registry: InstrumentSettlementRegistry,
) -> list[SourceGapInterval]:
    """Unresolved 3m gaps that fail the run when a roster seat would cross them."""
    return [
        iv
        for iv in active_intervals(plane="ohlcv_3m")
        if iv.extent in ("OPEN_EDGE", "UNSCOPED")
        and not source_gap_superseded_by_settlement(iv, settlement_registry)
    ]


def _holding_window(
    decision_index: pd.DatetimeIndex, *, entry_hour: int, holding: pd.Timedelta,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    starts = decision_index + pd.Timedelta(days=1, hours=int(entry_hour))
    return starts, starts + holding


def assert_lake_coverage(
    decision_index: pd.DatetimeIndex,
    roster: pd.DataFrame,
    *,
    strategy: StrategySpec,
    base_spec: ExecutionSpec,
    settlement_registry: InstrumentSettlementRegistry,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> None:
    """Fail closed when a roster seat would cross an unrepaired 3m lake gap.

    The roster here carries no source-gap withdrawal, so every selected
    (symbol, decision day) is what the strategy would have traded on its own
    data completeness. A decision day counts when its entry instant lies in
    ``[evaluation_start, evaluation_end)`` and its execution/holding window
    overlaps an unresolved ``OPEN_EDGE`` or ``UNSCOPED`` 3m gap.
    """
    intervals = _unresolved_lake_gap_intervals(settlement_registry)
    if not intervals or roster.empty:
        return
    holding = pd.Timedelta(days=1) + pd.Timedelta(minutes=int(base_spec.passive_timeout_minutes))
    starts, ends = _holding_window(decision_index, entry_hour=int(strategy.entry_hour_utc), holding=holding)
    in_scope = (starts >= evaluation_start) & (starts < evaluation_end)
    if not bool(in_scope.any()):
        return
    by_symbol: dict[str, list[SourceGapInterval]] = {}
    for iv in intervals:
        by_symbol.setdefault(iv.symbol, []).append(iv)
    offending: dict[str, pd.Timestamp] = {}
    scope_positions = np.flatnonzero(np.asarray(in_scope, dtype=bool))
    for symbol, gaps in by_symbol.items():
        if symbol not in roster.columns:
            continue
        selected = roster[symbol].to_numpy(dtype=bool)
        for pos in scope_positions:
            if not bool(selected[int(pos)]):
                continue
            window_start = pd.Timestamp(starts[int(pos)])
            window_end = pd.Timestamp(ends[int(pos)])
            for iv in gaps:
                gap_start = pd.Timestamp(iv.start)
                gap_end = None if iv.end is None else pd.Timestamp(iv.end)
                overlaps = gap_start < window_end and (gap_end is None or window_start < gap_end)
                if overlaps:
                    day = pd.Timestamp(decision_index[int(pos)])
                    if symbol not in offending or day < offending[symbol]:
                        offending[symbol] = day
                    break
            else:
                continue
            break
    if offending:
        details = "; ".join(
            f"{sym} (first_day={day.date().isoformat()})" for sym, day in sorted(offending.items())
        )
        raise LakeCoverageError(
            f"strategy universe crosses unrepaired 3m lake gaps for {len(offending)} symbol(s): "
            f"{details}; run `data collect` (3m) + `data verify-source-gaps` to repair the lake"
        )


def strategy_interior_withdrawals(
    decision_index: pd.DatetimeIndex,
    census_symbols: tuple[str, ...],
    *,
    strategy: StrategySpec,
    base_spec: ExecutionSpec,
    settlement_registry: InstrumentSettlementRegistry,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Count INTERIOR-gap withdrawals per symbol over the full decision grid."""
    census = list(census_symbols)
    frame = pd.DataFrame(False, index=decision_index, columns=census, dtype=bool)
    if not census:
        return {}, frame
    column_of = {sym: pos for pos, sym in enumerate(census)}
    values = np.zeros((len(decision_index), len(census)), dtype=bool)
    holding = pd.Timedelta(days=1) + pd.Timedelta(minutes=int(base_spec.passive_timeout_minutes))
    starts, ends = _holding_window(
        decision_index, entry_hour=int(strategy.entry_hour_utc), holding=holding,
    )
    for iv in active_intervals(plane="ohlcv_3m"):
        if iv.extent != "INTERIOR":
            continue
        if source_gap_superseded_by_settlement(iv, settlement_registry):
            continue
        column = column_of.get(iv.symbol)
        if column is None:
            continue
        overlap = np.asarray(pd.Timestamp(iv.start) < ends, dtype=bool)
        if iv.end is not None:
            overlap &= np.asarray(pd.Timestamp(iv.end) > starts, dtype=bool)
        values[:, column] |= overlap
    frame = pd.DataFrame(values, index=decision_index, columns=census, dtype=bool)
    counts = {sym: int(frame[sym].sum()) for sym in census if bool(frame[sym].any())}
    return counts, frame


def strategy_blocked_decisions(
    decision_index: pd.DatetimeIndex,
    census_symbols: tuple[str, ...],
    *,
    strategy: StrategySpec,
    base_spec: ExecutionSpec,
    settlement_registry: InstrumentSettlementRegistry,
) -> pd.DataFrame:
    """Mask decisions crossing announced delistings or unsuperseded bounded source gaps."""
    census = list(census_symbols)
    entry_hour = int(strategy.entry_hour_utc)
    snapshot_hour = int(strategy.snapshot_hour_utc)
    holding = pd.Timedelta(days=1) + pd.Timedelta(minutes=int(base_spec.passive_timeout_minutes))
    frame = pd.DataFrame(False, index=decision_index, columns=census, dtype=bool)
    if not census:
        return frame
    column_of = {sym: pos for pos, sym in enumerate(census)}
    values = _strategy_delisting_block(
        decision_index, census, column_of,
        snapshot_hour=snapshot_hour, settlement_registry=settlement_registry,
    )
    intervals = _withdrawable_gap_intervals(settlement_registry)
    if intervals:
        starts = decision_index + pd.Timedelta(days=1, hours=entry_hour)
        ends = starts + holding
        for iv in intervals:
            column = column_of.get(iv.symbol)
            if column is None:
                continue
            overlap = np.asarray(pd.Timestamp(iv.start) < ends, dtype=bool)
            if iv.end is not None:
                overlap &= np.asarray(pd.Timestamp(iv.end) > starts, dtype=bool)
            values[:, column] |= overlap
    return pd.DataFrame(values, index=decision_index, columns=census, dtype=bool)


def _strategy_execution_available_end(path: Path) -> pd.Timestamp | str:
    """Last 3m bar open in one archive, or a reason string when no extent can be read.

    An absent archive and a corrupt one both block replay, but they demand different
    operator action, so the reason is carried instead of collapsing both to one state.
    """
    if not path.exists():
        return "MISSING"
    try:
        frame = pd.read_parquet(path, columns=["timestamp"])
    except Exception as exc:  # noqa: BLE001 - elevate unreadable cause to diagnostic
        return f"UNREADABLE({type(exc).__name__})"
    stamps = pd.to_numeric(frame["timestamp"], errors="coerce").dropna() if not frame.empty else frame
    if not len(stamps):
        return "EMPTY"
    return pd.Timestamp(int(stamps.max()), unit="ms", tz="UTC")


def assert_strategy_execution_coverage(
    roster: pd.DataFrame,
    *,
    execution_end: pd.Timestamp,
    settlement: pd.Timedelta,
    entry_hour_utc: int,
    data_root: Path | None = None,
) -> None:
    """Reject archives lacking closing-bar coverage for any selected roster seat."""
    root = resolve_mhs_input_layout(data_root).ohlcv_root
    deficient: list[str] = []
    for symbol in roster.columns:
        granted = roster[symbol].to_numpy(dtype=bool)
        if not bool(granted.any()):
            continue
        last_true = pd.Timestamp(roster.index[int(np.flatnonzero(granted)[-1])])
        # Entry at D+1 entry_hour, exit at D+2; requires prices through settlement buffer.
        required = last_true + pd.Timedelta(days=2, hours=int(entry_hour_utc)) + settlement
        if required > execution_end:
            required = execution_end
        available = _strategy_execution_available_end(root / "3m" / f"{symbol}.parquet")
        if isinstance(available, str):
            deficient.append(f"{symbol} (required={required.isoformat()}, available={available})")
        elif required - available > pd.Timedelta(minutes=3):
            deficient.append(
                f"{symbol} (required={required.isoformat()}, available={available.isoformat()})"
            )
    if deficient:
        raise DataIntegrityError(
            f"strategy execution coverage incomplete for {len(deficient)} symbol(s): "
            + "; ".join(deficient)
        )


def _admit_source_stage(budget: MhsMemoryBudget, initial_swap_bytes: int | None) -> None:
    """Admit the source panel stage before any wide allocation."""
    assert_mhs_stage_allocation(
        stage="strategy_source_panel", estimated_bytes=0,
        budget=budget, replay=False, initial_swap_bytes=initial_swap_bytes,
    )


def _load_strategy_source(
    request: StrategyBacktestRequest,
    budget: MhsMemoryBudget,
    initial_swap_bytes: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, tuple[str, ...], dict[str, pd.Series], dict[str, str], str]:
    """Read the complete historical 1h census and derive daily and funding planes."""
    root = str(request.data_root) if request.data_root is not None else str(FUTURES_DATA_DIR / "ohlcv")

    def _admit_panel(estimated_bytes: int) -> None:
        assert_mhs_stage_allocation(
            stage="strategy_source_panel", estimated_bytes=int(estimated_bytes),
            budget=budget, replay=False, initial_swap_bytes=initial_swap_bytes,
        )

    panel = load_base_panel(
        root, "1h", ("close", "quote_vol", "taker_buy_quote"),
        request.source_start, request.evaluation_end, partition="all",
        selection_mode="causal_history", allocation_admission=_admit_panel,
    )
    close_1h = panel["close"]
    quote_1h = panel["quote_vol"]
    census = tuple(close_1h.columns)
    assert_settlement_registry_complete(
        Path(root), census,
        audit_end=request.evaluation_end,
        registry=settlement_registry_for_root(root),
    )
    daily_close = close_1h.resample("1D").last().astype("float64")
    daily_quote_volume = quote_1h.resample("1D").sum(min_count=1).astype("float64")
    grid_1h = close_1h.index
    completed = (grid_1h + pd.Timedelta(hours=1)).to_numpy(dtype="datetime64[ns]")
    hourly_available_at = pd.DataFrame(
        np.tile(completed[:, None], (1, len(census))),
        index=grid_1h, columns=list(census),
    ).apply(lambda col: pd.to_datetime(col).dt.tz_localize("UTC"))
    funding_by_symbol, funding_failures = _load_funding_series(list(census))
    hourly_panels = {key: panel[key] for key in ("close", "quote_vol", "taker_buy_quote")}
    return daily_close, daily_quote_volume, hourly_panels, hourly_available_at, census, funding_by_symbol, funding_failures, root


def _strategy_execution_fence(candidate: StrategyTargets) -> pd.Timestamp:
    """Extend the 3m stream past the final entry so it can be marked."""
    last = candidate.target_weights.index[-1]
    return (last.normalize() + pd.Timedelta(days=1)).tz_convert("UTC")


def _strategy_window_stream(
    candidate: StrategyTargets,
    stream_start: pd.Timestamp,
    stream_end: pd.Timestamp,
    root: str,
    funding_by_symbol: dict[str, pd.Series],
    funding_failures: dict[str, str],
    base_spec: ExecutionSpec,
    budget: MhsMemoryBudget,
    live_accumulators: _LiveAccumulatorSets,
) -> Iterator[ExecutionReplayWindow]:
    """Yield one materialized 3m window at a time with carried holdings retained."""
    def _required() -> frozenset[str]:
        if not live_accumulators:
            return frozenset()
        return live_required_symbols(live_accumulators[-1])

    yield from _iter_mhs_execution_windows(
        candidate.target_weights, candidate.signal_available_at, root, "3m",
        stream_start, stream_end, funding_by_symbol, base_spec,
        funding_failures=funding_failures, required_symbols=_required,
        budget_bytes=budget.replay_tree_pss_bytes, reserve_bytes=budget.min_available_bytes,
    )


def build_request_targets(
    request: StrategyBacktestRequest,
    *,
    source: LoadedStrategySource | None = None,
    snapshot_cache: MemberSnapshotCache | None = None,
) -> tuple[StrategyTargets, StrategySourceContext]:
    """Build the scored candidate exactly as ``run_strategy_backtest`` does, without replay.

    Returns the candidate plus the loaded source context (census, funding series, OHLCV root,
    resolved memory budget) so an alternative ledger can reuse one source load."""
    if source is None:
        source = load_strategy_source(request)
    else:
        root_for_request = str(request.data_root) if request.data_root is not None else str(FUTURES_DATA_DIR / "ohlcv")
        expected = _strategy_source_fingerprint(request, source.root)
        if tuple(source.fingerprint) != tuple(expected) or str(source.root) != root_for_request:
            raise DataIntegrityError("loaded strategy source does not match request window, root, or registries")
    budget = resolve_mhs_memory_budget(request.memory_budget)
    daily_close = source.daily_close
    daily_quote_volume = source.daily_quote_volume
    hourly_panels = source.hourly_panels
    hourly_available_at = source.hourly_available_at
    census = source.census
    funding_by_symbol = source.funding_by_symbol
    funding_failures = source.funding_failures
    root = source.root
    blocked_decisions = strategy_blocked_decisions(
        pd.DatetimeIndex(daily_close.index), census, strategy=request.strategy, base_spec=request.base_spec,
        settlement_registry=settlement_registry_for_root(root),
    )
    roster = build_pit_roster(
        daily_close, daily_quote_volume, census,
        breadth=request.strategy.breadth, blocked_decisions=blocked_decisions,
    )
    registry = settlement_registry_for_root(root)
    delisting_values = _strategy_delisting_block(
        pd.DatetimeIndex(daily_close.index), list(census),
        {sym: pos for pos, sym in enumerate(census)},
        snapshot_hour=int(request.strategy.snapshot_hour_utc),
        settlement_registry=registry,
    )
    delisting_frame = pd.DataFrame(delisting_values, index=pd.DatetimeIndex(daily_close.index), columns=list(census), dtype=bool)
    roster_no_gap = build_pit_roster(
        daily_close, daily_quote_volume, census,
        breadth=request.strategy.breadth, blocked_decisions=delisting_frame,
    )
    assert_lake_coverage(
        pd.DatetimeIndex(daily_close.index), roster_no_gap,
        strategy=request.strategy, base_spec=request.base_spec,
        settlement_registry=registry,
        evaluation_start=request.evaluation_start, evaluation_end=request.evaluation_end,
    )
    ever_selected = [sym for sym in census if bool(roster[sym].any())]
    assert_strategy_execution_coverage(
        roster,
        execution_end=request.evaluation_end,
        settlement=pd.Timedelta(minutes=int(request.base_spec.passive_timeout_minutes)),
        entry_hour_utc=int(request.strategy.entry_hour_utc),
        data_root=request.data_root,
    )
    if not ever_selected:
        raise DataIntegrityError("request strategy selects no historical symbol")
    selected_panels = (
        {key: frame[ever_selected] for key, frame in hourly_panels.items()}
        if snapshot_cache is None else hourly_panels
    )
    selected_available = hourly_available_at[ever_selected] if snapshot_cache is None else hourly_available_at
    full_candidate = build_strategy_targets(
        selected_panels,
        selected_available,
        daily_close,
        daily_quote_volume,
        census,
        market_close=hourly_panels["close"],
        strategy=request.strategy,
        blocked_decisions=blocked_decisions,
        snapshot_cache=snapshot_cache,
    )
    labels = full_candidate.target_weights.index
    scored = (labels >= request.evaluation_start) & (labels < request.evaluation_end)
    if not bool(scored.any()):
        raise DataIntegrityError("evaluation interval contains no candidate entry row")
    candidate = StrategyTargets(
        target_weights=full_candidate.target_weights.loc[scored],
        signal_available_at=full_candidate.signal_available_at[scored],
        strategy=request.strategy,
    )
    context = StrategySourceContext(
        census=census, funding_by_symbol=funding_by_symbol, funding_failures=funding_failures,
        root=root, budget=budget, daily_close=daily_close, daily_quote_volume=daily_quote_volume,
    )
    return candidate, context


def run_strategy_backtest(
    request: StrategyBacktestRequest,
    *,
    source: LoadedStrategySource | None = None,
    snapshot_cache: MemberSnapshotCache | None = None,
) -> StrategyBacktestRun:
    """Build causal PIT targets and return paired base/stress 3m inventory evidence."""
    candidate, context = build_request_targets(request, source=source, snapshot_cache=snapshot_cache)
    budget = context.budget
    root = context.root
    funding_by_symbol = context.funding_by_symbol
    funding_failures = context.funding_failures
    census = context.census
    settlement_registry = settlement_registry_for_root(root)
    blocked_decisions = strategy_blocked_decisions(
        pd.DatetimeIndex(context.daily_close.index), census, strategy=request.strategy, base_spec=request.base_spec,
        settlement_registry=settlement_registry,
    )
    delisting_only = _strategy_delisting_block(
        pd.DatetimeIndex(context.daily_close.index), list(census),
        {sym: pos for pos, sym in enumerate(census)},
        snapshot_hour=int(request.strategy.snapshot_hour_utc),
        settlement_registry=settlement_registry,
    )
    execution_start = candidate.signal_available_at[0]
    execution_end = _strategy_execution_fence(candidate)
    live_accumulators: _LiveAccumulatorSets = []

    def _window_stream() -> Iterator[ExecutionReplayWindow]:
        yield from _strategy_window_stream(
            candidate, candidate.signal_available_at[0], execution_end, root,
            funding_by_symbol, funding_failures, request.base_spec, budget, live_accumulators,
        )

    window_stream = _window_stream()
    evidence = evaluate_strategy_backtest(
        candidate, window_stream, initial_equity=request.initial_equity,
        base_spec=request.base_spec, stress_spec=request.stress_spec,
        report_periods=request.report_periods, live_accumulators=live_accumulators,
        execution_bound=request.execution_bound,
    )
    for period in request.report_periods:
        row = evidence.period_metrics.loc[period.label]
        if float(row["base_coverage"]) != 1.0 or float(row["stress_coverage"]) != 1.0:
            raise DataIntegrityError(f"report period {period.label!r} is not fully covered by daily evidence")
    _, interior_frame = strategy_interior_withdrawals(
        pd.DatetimeIndex(context.daily_close.index), census, strategy=request.strategy,
        base_spec=request.base_spec, settlement_registry=settlement_registry,
    )
    decision_index = pd.DatetimeIndex(context.daily_close.index)
    no_gap_roster = build_pit_roster(
        context.daily_close, context.daily_quote_volume, census, breadth=request.strategy.breadth,
        blocked_decisions=pd.DataFrame(delisting_only, index=decision_index, columns=list(census)),
    )
    entries = decision_index + pd.Timedelta(days=1, hours=int(request.strategy.entry_hour_utc))
    scored = (entries >= request.evaluation_start) & (entries < request.evaluation_end)
    affected = (interior_frame & no_gap_roster).loc[scored].sum()
    interior_counts = {str(sym): int(days) for sym, days in affected.items() if days > 0}
    withdrawals: tuple[Mapping[str, object], ...] = tuple(
        {"symbol": sym, "extent": "INTERIOR", "days": int(days)}
        for sym, days in sorted(interior_counts.items())
    )
    return StrategyBacktestRun(
        request=request, candidate=candidate, evidence=evidence,
        execution_start=execution_start, execution_end=execution_end, source_symbols=census,
        source_gap_excluded_symbols=tuple(sorted(sym for sym in census if bool(blocked_decisions[sym].any()))),
        source_gap_blocked_decisions=int(blocked_decisions.to_numpy(dtype=bool).sum()),
        delisting_blocked_decisions=int(delisting_only.sum()),
        data_availability_withdrawals=withdrawals,
    )
