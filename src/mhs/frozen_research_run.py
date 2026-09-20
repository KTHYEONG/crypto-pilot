"""Historical frozen-MHS inventory runner on the shared 3m ledger."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.paths import FUTURES_DATA_DIR
from src.mhs.execution import ExecutionReplayWindow, live_required_symbols
from src.mhs.execution.batch import _LiveAccumulatorSets
from src.mhs.execution.window_stream import _iter_mhs_execution_windows
from src.mhs.frozen_research_candidate import (
    FrozenMhsCandidate,
    FrozenMhsStrategySpec,
    build_frozen_mhs_candidate,
)
from src.mhs.frozen_research_evidence import (
    FrozenMhsReportPeriod,
    FrozenMhsResearchEvidence,
    evaluate_frozen_mhs_research,
)
from src.mhs.frozen_research_universe import build_frozen_pit_roster
from src.mhs.marks import _load_funding_series
from src.mhs.panel import load_base_panel
from src.mhs.resources import (
    MhsMemoryBudget,
    _current_tree_swap_bytes,
    assert_mhs_stage_allocation,
    resolve_mhs_memory_budget,
)
from src.mhs.source_gaps import active_intervals
from src.mhs.types import ExecutionSpec


@dataclass(frozen=True, slots=True)
class FrozenMhsBacktestRequest:
    """Describe one reproducible historical frozen-MHS inventory experiment.

    Source history is distinct from the scored interval so liquidity and feature
    warm-up are observable rather than manufactured.  The request identifies
    a target policy, exact execution cost bounds, and report periods without
    allowing the runner to select a better strategy from its results.
    """

    source_start: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    strategy: FrozenMhsStrategySpec
    initial_equity: float
    base_spec: ExecutionSpec
    stress_spec: ExecutionSpec
    report_periods: tuple[FrozenMhsReportPeriod, ...]
    data_root: Path | None = None
    memory_budget: MhsMemoryBudget | None = None

    def __post_init__(self) -> None:
        for name in ("source_start", "evaluation_start", "evaluation_end"):
            value = getattr(self, name)
            if not isinstance(value, pd.Timestamp) or pd.isna(value):
                raise DataIntegrityError(f"{name} must be a valid timestamp")
            if value.tzinfo is None or value.utcoffset() is None or value.utcoffset().total_seconds() != 0:
                raise DataIntegrityError(f"{name} must be timezone-aware UTC")
        if not self.source_start < self.evaluation_start < self.evaluation_end:
            raise DataIntegrityError("request must satisfy source_start < evaluation_start < evaluation_end")
        if not isinstance(self.strategy, FrozenMhsStrategySpec):
            raise DataIntegrityError("strategy must be a FrozenMhsStrategySpec")
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
        if not isinstance(self.report_periods, tuple) or not self.report_periods:
            raise DataIntegrityError("report_periods must be a non-empty tuple of FrozenMhsReportPeriod")
        if any(not isinstance(p, FrozenMhsReportPeriod) for p in self.report_periods):
            raise DataIntegrityError("report_periods must be a non-empty tuple of FrozenMhsReportPeriod")
        if self.data_root is not None and not isinstance(self.data_root, Path):
            raise DataIntegrityError("data_root must be a Path or None")
        if self.memory_budget is not None and not isinstance(self.memory_budget, MhsMemoryBudget):
            raise DataIntegrityError("memory_budget must be a MhsMemoryBudget or None")


@dataclass(frozen=True, slots=True)
class FrozenMhsBacktestRun:
    """Return exact target provenance and paired 3m evidence for one request."""

    request: FrozenMhsBacktestRequest
    candidate: FrozenMhsCandidate
    evidence: FrozenMhsResearchEvidence
    execution_start: pd.Timestamp
    execution_end: pd.Timestamp
    source_symbols: tuple[str, ...]
    source_gap_excluded_symbols: tuple[str, ...] = ()
    source_gap_blocked_decisions: int = 0


def frozen_blocked_decisions(
    decision_index: pd.DatetimeIndex,
    census_symbols: tuple[str, ...],
    *,
    strategy: FrozenMhsStrategySpec,
    base_spec: ExecutionSpec,
) -> pd.DataFrame:
    """Map evidenced 3m source gaps onto the decision days they would have traded through.

    A decision is blocked when the interval it must execute and hold through overlaps an
    unrecoverable gap, which is the only condition under which the ledger could not have
    been produced in live trading. Gaps that fall entirely outside a decision's execution
    and holding window leave the decision untouched.

    Args:
        decision_index: Daily UTC decision grid shared with the roster inputs.
        census_symbols: Canonical column order for the returned frame.
        strategy: Supplies the entry hour that anchors each decision's execution window.
        base_spec: Supplies the passive timeout that extends each holding window.
    Returns:
        Boolean frame indexed by `decision_index` with `census_symbols` as columns.
    """
    census = list(census_symbols)
    entry_hour = int(strategy.entry_hour_utc)
    holding = pd.Timedelta(days=1) + pd.Timedelta(minutes=int(base_spec.passive_timeout_minutes))
    frame = pd.DataFrame(False, index=decision_index, columns=census, dtype=bool)
    if not census:
        return frame
    intervals = active_intervals(plane="ohlcv_3m")
    if not intervals:
        return frame
    # 결정일 수가 수천 개라 구간마다 벡터 비교 한 번으로 겹침을 판정한다(일별 루프 금지).
    column_of = {sym: pos for pos, sym in enumerate(census)}
    starts = decision_index + pd.Timedelta(days=1, hours=entry_hour)
    ends = starts + holding
    values = np.zeros((len(decision_index), len(census)), dtype=bool)
    for iv in intervals:
        column = column_of.get(iv.symbol)
        if column is None:
            continue
        overlap = np.asarray(pd.Timestamp(iv.start) < ends, dtype=bool)
        if iv.end is not None:
            overlap &= np.asarray(pd.Timestamp(iv.end) > starts, dtype=bool)
        values[:, column] |= overlap
    return pd.DataFrame(values, index=decision_index, columns=census, dtype=bool)


def _frozen_execution_available_end(path: Path) -> pd.Timestamp | str:
    """Last 3m bar open in one archive, or a reason string when no extent can be read.

    An absent archive and a corrupt one both block replay, but they demand different
    operator action, so the reason is carried instead of collapsing both to one state.
    """
    if not path.exists():
        return "MISSING"
    try:
        frame = pd.read_parquet(path, columns=["timestamp"])
    except Exception as exc:  # noqa: BLE001 - 판독 불가 사유를 진단 메시지로 승격한다.
        return f"UNREADABLE({type(exc).__name__})"
    stamps = pd.to_numeric(frame["timestamp"], errors="coerce").dropna() if not frame.empty else frame
    if not len(stamps):
        return "EMPTY"
    return pd.Timestamp(int(stamps.max()), unit="ms", tz="UTC")


def assert_frozen_execution_coverage(
    roster: pd.DataFrame,
    *,
    execution_end: pd.Timestamp,
    settlement: pd.Timedelta,
    entry_hour_utc: int,
    data_root: Path | None = None,
) -> None:
    """Fail closed before replay when a selected symbol lacks execution evidence.

    Every decision the roster grants must be executable and markable on the 3m plane
    through the bar that closes it, otherwise the engine enters a position it can never
    exit and the failure only surfaces much later at an unrelated symbol's bar. Checking
    the whole roster up front converts a multi-minute replay crash into one actionable list.

    The check consumes only archive extents, never future prices, so it states what the
    researcher's own lake contains and makes no claim about what was knowable at any
    decision time.

    Args:
        roster: Boolean decision-day roster in canonical column order.
        execution_end: Exclusive UTC fence the replay will stream to.
        settlement: Extra span past a decision's holding window during which its closing
            order may still cross; derived from the execution spec, never guessed.
        entry_hour_utc: Hour a decision's entry lands on, taken from the strategy rather
            than assumed, so a variant that enters off midnight is checked at its own clock.
        data_root: OHLCV root override; defaults to the canonical futures lake.
    Raises:
        DataIntegrityError: At least one selected symbol's 3m archive ends before the bar
            that closes its last granted decision. The message names every such symbol with
            its required and available coverage end.
    """
    root = Path(data_root) if data_root is not None else FUTURES_DATA_DIR
    deficient: list[str] = []
    for symbol in roster.columns:
        granted = roster[symbol].to_numpy(dtype=bool)
        if not bool(granted.any()):
            continue
        last_true = pd.Timestamp(roster.index[int(np.flatnonzero(granted)[-1])])
        # 결정일 D의 진입은 D+1 entry_hour, 청산은 다음 진입(D+2)이며 정산 여유까지 가격이 필요하다.
        required = last_true + pd.Timedelta(days=2, hours=int(entry_hour_utc)) + settlement
        if required > execution_end:
            required = execution_end
        available = _frozen_execution_available_end(root / "ohlcv" / "3m" / f"{symbol}.parquet")
        if isinstance(available, str):
            deficient.append(f"{symbol} (required={required.isoformat()}, available={available})")
        elif required - available > pd.Timedelta(minutes=3):
            deficient.append(
                f"{symbol} (required={required.isoformat()}, available={available.isoformat()})"
            )
    if deficient:
        raise DataIntegrityError(
            f"frozen execution coverage incomplete for {len(deficient)} symbol(s): "
            + "; ".join(deficient)
        )


def _admit_source_stage(budget: MhsMemoryBudget, initial_swap_bytes: int | None) -> None:
    """Admit the source panel stage before any wide allocation."""
    assert_mhs_stage_allocation(
        stage="frozen_source_panel", estimated_bytes=0,
        budget=budget, replay=False, initial_swap_bytes=initial_swap_bytes,
    )


def _load_frozen_source(
    request: FrozenMhsBacktestRequest,
    budget: MhsMemoryBudget,
    initial_swap_bytes: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, tuple[str, ...], dict[str, pd.Series], dict[str, str], str]:
    """Read the complete historical 1h census and derive daily and funding planes."""
    root = str(request.data_root) if request.data_root is not None else str(FUTURES_DATA_DIR / "ohlcv")

    def _admit_panel(estimated_bytes: int) -> None:
        assert_mhs_stage_allocation(
            stage="frozen_source_panel", estimated_bytes=int(estimated_bytes),
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


def _frozen_execution_fence(candidate: FrozenMhsCandidate) -> pd.Timestamp:
    """Extend the 3m stream past the final entry so it can be marked."""
    last = candidate.target_weights.index[-1]
    return (last.normalize() + pd.Timedelta(days=1)).tz_convert("UTC")


def _frozen_window_stream(
    candidate: FrozenMhsCandidate,
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


def run_frozen_mhs_backtest(request: FrozenMhsBacktestRequest) -> FrozenMhsBacktestRun:
    """Build and replay a frozen PIT strategy using the shared 3m inventory ledger.

    The runner first reconstructs the full historical universe from Binance
    archive sources, then materializes only historically selected hourly and
    active/held 3m symbols.  It produces research evidence for the exact
    request interval without changing or consulting live strategy state.

    Args:
        request: Complete historical source, target-policy, cost, and resource request.
    Returns:
        Exact candidate provenance and paired base/stress inventory evidence.
    Raises:
        DataIntegrityError: Input chronology, source coverage, timing, or
            execution evidence is incomplete.
        MhsResourceAdmissionError: A declared memory budget cannot admit work.
    """
    budget = resolve_mhs_memory_budget(request.memory_budget)
    initial_swap_bytes = _current_tree_swap_bytes()
    _admit_source_stage(budget, initial_swap_bytes)
    daily_close, daily_quote_volume, hourly_panels, hourly_available_at, census, funding_by_symbol, funding_failures, root = _load_frozen_source(
        request, budget, initial_swap_bytes
    )
    blocked_decisions = frozen_blocked_decisions(
        pd.DatetimeIndex(daily_close.index), census, strategy=request.strategy, base_spec=request.base_spec,
    )
    roster = build_frozen_pit_roster(
        daily_close, daily_quote_volume, census,
        breadth=request.strategy.breadth, blocked_decisions=blocked_decisions,
    )
    ever_selected = [sym for sym in census if bool(roster[sym].any())]
    assert_frozen_execution_coverage(
        roster,
        execution_end=request.evaluation_end,
        settlement=pd.Timedelta(minutes=int(request.base_spec.passive_timeout_minutes)),
        entry_hour_utc=int(request.strategy.entry_hour_utc),
        data_root=request.data_root,
    )
    if not ever_selected:
        raise DataIntegrityError("request strategy selects no historical symbol")
    selected_panels = {key: frame[ever_selected] for key, frame in hourly_panels.items()}
    selected_available = hourly_available_at[ever_selected]
    full_candidate = build_frozen_mhs_candidate(
        selected_panels,
        selected_available,
        daily_close,
        daily_quote_volume,
        census,
        strategy=request.strategy,
        blocked_decisions=blocked_decisions,
    )
    labels = full_candidate.target_weights.index
    scored = (labels >= request.evaluation_start) & (labels < request.evaluation_end)
    if not bool(scored.any()):
        raise DataIntegrityError("evaluation interval contains no candidate entry row")
    candidate = FrozenMhsCandidate(
        target_weights=full_candidate.target_weights.loc[scored],
        signal_available_at=full_candidate.signal_available_at[scored],
        strategy=request.strategy,
    )
    execution_start = candidate.signal_available_at[0]
    execution_end = _frozen_execution_fence(candidate)
    live_accumulators: _LiveAccumulatorSets = []

    def _window_stream() -> Iterator[ExecutionReplayWindow]:
        yield from _frozen_window_stream(
            candidate, candidate.signal_available_at[0], execution_end, root,
            funding_by_symbol, funding_failures, request.base_spec, budget, live_accumulators,
        )

    window_stream = _window_stream()
    evidence = evaluate_frozen_mhs_research(
        candidate, window_stream, initial_equity=request.initial_equity,
        base_spec=request.base_spec, stress_spec=request.stress_spec,
        report_periods=request.report_periods, live_accumulators=live_accumulators,
    )
    for period in request.report_periods:
        row = evidence.period_metrics.loc[period.label]
        if float(row["base_coverage"]) != 1.0 or float(row["stress_coverage"]) != 1.0:
            raise DataIntegrityError(f"report period {period.label!r} is not fully covered by daily evidence")
    return FrozenMhsBacktestRun(
        request=request, candidate=candidate, evidence=evidence,
        execution_start=execution_start, execution_end=execution_end, source_symbols=census,
        source_gap_excluded_symbols=tuple(sorted(sym for sym in census if bool(blocked_decisions[sym].any()))),
        source_gap_blocked_decisions=int(blocked_decisions.to_numpy(dtype=bool).sum()),
    )
