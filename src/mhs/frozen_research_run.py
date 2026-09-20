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
    from src.mhs.data_policy import frozen_research_source_gap_exclusions

    resolved = frozenset(s for s in frozen_research_source_gap_exclusions() if s in set(census))
    roster = build_frozen_pit_roster(
        daily_close, daily_quote_volume, census, breadth=request.strategy.breadth, excluded_symbols=resolved
    )
    ever_selected = [sym for sym in census if bool(roster[sym].any())]
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
        excluded_symbols=resolved,
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
        source_gap_excluded_symbols=tuple(sorted(resolved)),
    )
