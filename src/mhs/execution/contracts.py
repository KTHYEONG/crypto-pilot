"""Execution contracts — funding panel, result dataclasses, ruin guard."""

from __future__ import annotations

import datetime as _datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.quant.baseline.backtest import _align_funding_rates

from . import _ExecutionBound, _ExecutionGapCode, _MarkSource

_DEFAULT_MAX_OBSERVATION_GAP = pd.Timedelta(hours=8, minutes=5)


def _utc_epoch_ns(index: pd.Index) -> np.ndarray:
    """Fast UTC epoch-nanosecond conversion, bit-identical to the legacy path."""
    if isinstance(index, pd.DatetimeIndex):
        utc = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
        return np.asarray(utc.as_unit("ns").asi8, dtype="int64")
    return np.asarray(pd.DatetimeIndex(pd.to_datetime(index, utc=True)), dtype="datetime64[ns]").astype("int64")


@dataclass(frozen=True, slots=True)
class FundingKnowledgeObservation:
    """Record observed funding knowledge separately from a file's future coverage.

    An attested no-settlement observation is distinct from absent source data.
    """

    symbol: str
    event_time: pd.Timestamp
    available_at: pd.Timestamp
    status: Literal["settled", "no_settlement", "unknown"]
    source_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise DataIntegrityError("funding observation symbol must be a nonempty string")
        if self.status not in ("settled", "no_settlement", "unknown"):
            raise DataIntegrityError(f"unknown funding observation status {self.status!r}")
        if not isinstance(self.source_digest, str) or not self.source_digest:
            raise DataIntegrityError("funding observation source_digest must be a nonempty string")
        for name in ("event_time", "available_at"):
            value = getattr(self, name)
            if not isinstance(value, pd.Timestamp) or pd.isna(value):
                raise DataIntegrityError(f"funding observation {name} must be a valid timestamp")
            if value.tzinfo is None or value.utcoffset() != _datetime.timedelta(0):
                raise DataIntegrityError(f"funding observation {name} must be timezone-aware UTC")
        if self.available_at < self.event_time:
            raise DataIntegrityError("funding observation available_at must not precede event_time")


@dataclass(frozen=True, slots=True)
class FundingAlignment:
    """Funding rates split from funding knowledge (INV-FUNDING-KNOWLEDGE).

    ``rates`` carries the per-bar settlement rates (0.0 where unknown);
    ``known`` marks the bars whose funding state is actually observed.
    Unknown funding overlapping held inventory or an active order fails
    closed downstream; unknown funding over inactive stretches is a recorded
    limitation only. ``source_failures`` echoes the per-symbol load failures.
    """

    rates: pd.DataFrame
    known: pd.DataFrame
    source_failures: Mapping[str, str]
    knowledge_source: Literal["recorded", "archive_recency_proxy"] = "archive_recency_proxy"
    limitations: tuple[str, ...] = ()


def _validate_knowledge_observations(
    knowledge_observations: tuple[FundingKnowledgeObservation, ...],
) -> dict[str, list[FundingKnowledgeObservation]]:
    """Group validated funding knowledge observations by symbol (prefix-only)."""
    grouped: dict[str, list[FundingKnowledgeObservation]] = {}
    for obs in knowledge_observations:
        if not isinstance(obs, FundingKnowledgeObservation):
            raise DataIntegrityError("knowledge_observations must be FundingKnowledgeObservation records")
        grouped.setdefault(obs.symbol, []).append(obs)
    for observations in grouped.values():
        observations.sort(key=lambda o: (o.event_time.value, o.available_at.value))
    return grouped


def align_funding_with_knowledge(
    funding_by_symbol: Mapping[str, pd.Series],
    grid: pd.DatetimeIndex,
    *,
    symbols: Sequence[str],
    source_failures: Mapping[str, str] | None = None,
    max_observation_gap: pd.Timedelta = _DEFAULT_MAX_OBSERVATION_GAP,
    knowledge_observations: tuple[FundingKnowledgeObservation, ...] = (),
) -> FundingAlignment:
    """Align funding onto ``grid`` with an explicit known/unknown mask.

    ``funding_by_symbol`` carries full-history series: rates settle from the
    window slice, while ``known`` reflects the file's own coverage (only
    no-event bars inside a covered span may read as zero). A symbol absent
    from the mapping, listed in ``source_failures``, or carrying an empty
    series is unknown everywhere (rates 0.0). Bars outside the observed span
    or strictly inside an inter-observation gap longer than
    ``max_observation_gap`` are unknown. Financial results stay float64;
    knowledge stays bool (no downcast, per the performance budget).

    Knowledge consumes only observations published through each row. Archived
    event recency may be reported as a screening proxy, but future file bounds
    and future recovery events cannot certify earlier no-settlement intervals.
    """
    failed = dict(source_failures) if source_failures else {}
    grouped = _validate_knowledge_observations(tuple(knowledge_observations))
    recorded = bool(grouped)
    gap_ns = int(max_observation_gap.value)
    grid_ns = np.asarray(grid, dtype="datetime64[ns]").astype("int64")
    period = grid[1] - grid[0] if len(grid) > 1 else pd.Timedelta(minutes=1)
    rate_cols: dict[str, pd.Series] = {}
    known_cols: dict[str, pd.Series] = {}
    for sym in symbols:
        series = funding_by_symbol.get(sym)
        if sym in failed or series is None or len(series) == 0:
            rate_cols[sym] = pd.Series(np.zeros(len(grid), dtype="float64"), index=grid, dtype="float64")
            known_cols[sym] = pd.Series(np.zeros(len(grid), dtype=bool), index=grid, dtype=bool)
            continue
        full_ts = _utc_epoch_ns(series.index)
        full_ts = np.sort(full_ts)
        windowed = series.loc[(series.index >= grid[0]) & (series.index < grid[-1] + period)]
        aligned = np.asarray(_align_funding_rates(windowed, grid), dtype="float64")
        observations = grouped.get(sym, [])
        if observations:
            event_ns = np.asarray([o.event_time.value for o in observations], dtype="int64")
            avail_ns = np.asarray([o.available_at.value for o in observations], dtype="int64")
            attested = np.asarray([o.status in ("settled", "no_settlement") for o in observations], dtype=bool)
            known = np.zeros(len(grid), dtype=bool)
            for i in range(len(grid_ns)):
                eligible = (event_ns <= grid_ns[i]) & (avail_ns <= grid_ns[i])
                if not bool(eligible.any()):
                    continue
                latest = int(np.flatnonzero(eligible)[-1])
                known[i] = bool(attested[latest])
        else:
            prev = np.searchsorted(full_ts, grid_ns, side="right") - 1
            recency = np.zeros(len(grid), dtype=bool)
            valid_prev = prev >= 0
            recency[valid_prev] = (grid_ns[valid_prev] - full_ts[prev[valid_prev]]) <= gap_ns
            known = (grid_ns >= full_ts[0]) & valid_prev & recency
        rate_cols[sym] = pd.Series(aligned, index=grid, dtype="float64")
        known_cols[sym] = pd.Series(known, index=grid, dtype=bool)
    rates = pd.DataFrame(rate_cols, index=grid).astype("float64")
    known_frame = pd.DataFrame(known_cols, index=grid).astype(bool)
    if recorded:
        return FundingAlignment(
            rates=rates, known=known_frame, source_failures=failed,
            knowledge_source="recorded", limitations=(),
        )
    return FundingAlignment(
        rates=rates, known=known_frame, source_failures=failed,
        knowledge_source="archive_recency_proxy",
        limitations=("ARCHIVE_RECENCY_PROXY_NO_PUBLICATION_PROOF",),
    )


@dataclass(frozen=True, slots=True)
class FundingCoverageGap:
    """One uncovered funding interval that makes inventory economics uncertifiable.

    The interval is expressed on the requested execution grid. It describes
    unavailable source knowledge, never an assumed zero payment.

    Args:
        symbol: Canonical perpetual-futures symbol.
        start: First grid label whose funding state is unknown.
        end: Last grid label whose funding state is unknown.
        reason: Stable source-coverage reason.
    """

    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    reason: str


def _require_coverage_grid(grid: pd.DatetimeIndex, alignment: FundingAlignment) -> None:
    """Validate the coverage grid against the alignment frames."""
    if not isinstance(grid, pd.DatetimeIndex):
        raise DataIntegrityError("grid must be a DatetimeIndex")
    if len(grid) == 0:
        raise DataIntegrityError("grid must be non-empty")
    if grid.hasnans:
        raise DataIntegrityError("grid must not contain NaT")
    if grid.tz is None:
        raise DataIntegrityError("grid must be timezone-aware UTC")
    converted = grid.tz_convert("UTC")
    if not converted.equals(grid):
        raise DataIntegrityError("grid must be UTC")
    if grid.has_duplicates or not grid.is_monotonic_increasing:
        raise DataIntegrityError("grid must be unique and increasing")
    if not alignment.rates.index.equals(grid):
        raise DataIntegrityError("alignment rates index must equal grid")
    if not alignment.known.index.equals(grid):
        raise DataIntegrityError("alignment known index must equal grid")
    if list(alignment.rates.columns) != list(alignment.known.columns):
        raise DataIntegrityError("rate and known columns must match exactly")


def funding_coverage_gaps(
    alignment: FundingAlignment,
    grid: pd.DatetimeIndex,
) -> tuple[FundingCoverageGap, ...]:
    """Compress unknown funding knowledge into deterministic source intervals.

    Args:
        alignment: Existing rate and knowledge alignment for the same grid.
        grid: Strictly increasing UTC execution labels.

    Returns:
        Time-ordered, symbol-ordered maximal unknown intervals.

    Raises:
        DataIntegrityError: Alignment or grid shapes, labels, or columns differ.
    """
    _require_coverage_grid(grid, alignment)
    failures = alignment.source_failures
    out: list[FundingCoverageGap] = []
    for symbol in list(alignment.rates.columns):
        known = alignment.known[symbol].to_numpy(dtype=bool)
        if bool(known.all()):
            continue
        reason = "SOURCE_UNAVAILABLE" if symbol in failures or not bool(known.any()) else "OBSERVATION_GAP"
        idx = np.flatnonzero(~known)
        run_start = int(idx[0])
        prev = int(idx[0])
        for pos in idx[1:].tolist():
            pos = int(pos)
            if pos == prev + 1:
                prev = pos
                continue
            out.append(
                FundingCoverageGap(
                    symbol=symbol,
                    start=grid[run_start],
                    end=grid[prev],
                    reason=reason,
                )
            )
            run_start = pos
            prev = pos
        out.append(
            FundingCoverageGap(
                symbol=symbol,
                start=grid[run_start],
                end=grid[prev],
                reason=reason,
            )
        )
    out.sort(key=lambda g: (g.start, g.end, g.symbol))
    return tuple(out)


def bar_funding_panel(
    funding_by_symbol: Mapping[str, pd.Series],
    grid: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Column-wise ``_align_funding_rates`` reuse over the decision grid.

    A symbol whose funding series cannot be causally aligned is excluded from
    the output: missing funding is never silently zero-filled.
    """
    cols: dict[str, pd.Series] = {}
    for sym, series in funding_by_symbol.items():
        try:
            cols[sym] = pd.Series(_align_funding_rates(series, grid), index=grid, dtype="float64")
        except DataIntegrityError:
            continue
    df = pd.DataFrame(cols, index=grid)
    # Sanitize internal alignment NaNs/Infs by forward filling and zero-filling
    return df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


@dataclass(frozen=True, slots=True)
class ExecutionDataGap:
    """Deterministic provenance for one cache-required data gap.

    ``decision_time``/``signal_time`` carry the intent that would have traded
    through the gap (null when no intent applies, e.g. a held-position mark or
    funding gap). Terminal-window censoring is report telemetry and is never
    represented by this record.
    """

    code: _ExecutionGapCode
    symbol: str
    timestamp: pd.Timestamp
    decision_time: pd.Timestamp | None = None
    signal_time: pd.Timestamp | None = None
    execution_bound: str = "OHLCV_STRICT_PROXY"


@dataclass(frozen=True, slots=True)
class ExecutionReplayWindow:
    """One chronological execution window fed to ``replay_execution_windows``.

    ``columns`` is the canonical artifact column order (identical across every
    window); ``symbols`` is this window's active roster actually present in the
    ``highs``/``lows``/``closes``/``marks``/``bar_funding`` frames. The minute
    grid covers the strict timeout overlap of the window's final order plus the
    boundary bars the engine needs for decision-time funding and MTM.
    """

    window_start: pd.Timestamp
    window_end: pd.Timestamp
    columns: tuple[str, ...]
    symbols: tuple[str, ...]
    minute_grid: pd.DatetimeIndex
    highs: pd.DataFrame
    lows: pd.DataFrame
    closes: pd.DataFrame
    marks: pd.DataFrame | None
    bar_funding: pd.DataFrame
    target_weights: pd.DataFrame
    signal_available_at: pd.DatetimeIndex
    # Tradability and knowledge overlays (P2_DATA_AND_POLICY_PARITY). ``None``
    # preserves the legacy direct-construction semantics (all bars tradable,
    # all funding known, effective time equals the grid label); the window
    # generator always materializes explicit frames.
    quote_volumes: pd.DataFrame | None = None
    funding_known: pd.DataFrame | None = None
    bar_available_at: pd.DatetimeIndex | None = None
    # Half-open global decision-ordinal range `(first_ordinal, stop_ordinal)` of the
    # existing logical cost partition. Physical pieces sharing this key accumulate
    # one observation and settle costs once. None retains legacy direct-construction
    # semantics where each window is its own observation range.
    logical_partition: tuple[int, int] | None = None
    funding_coverage_gaps: tuple[FundingCoverageGap, ...] = ()
    funding_knowledge_source: Literal["recorded", "archive_recency_proxy", "legacy"] = "legacy"
    settlement_events: tuple[InstrumentSettlementEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class InstrumentSettlementEvent:
    """Describe an evidenced exchange settlement, not a price inferred from inactivity.

    The source binds the contractual settlement time, price and fee. Publication
    timing prevents a later announcement from generating an earlier fictional fill.
    """

    event_id: str
    symbol: str
    effective_at: pd.Timestamp
    available_at: pd.Timestamp
    settlement_price: float
    fee_bps: float
    source_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise DataIntegrityError("settlement event_id must be a nonempty string")
        if not isinstance(self.symbol, str) or not self.symbol:
            raise DataIntegrityError("settlement symbol must be a nonempty string")
        if not isinstance(self.source_digest, str) or not self.source_digest:
            raise DataIntegrityError("settlement source_digest must be a nonempty string")
        for name in ("effective_at", "available_at"):
            value = getattr(self, name)
            if not isinstance(value, pd.Timestamp) or pd.isna(value):
                raise DataIntegrityError(f"settlement {name} must be a valid timestamp")
            if value.tzinfo is None or value.utcoffset() != _datetime.timedelta(0):
                raise DataIntegrityError(f"settlement {name} must be timezone-aware UTC")
        price = float(self.settlement_price)
        if not np.isfinite(price) or price <= 0.0:
            raise DataIntegrityError("settlement_price must be a finite positive price")
        fee = float(self.fee_bps)
        if not np.isfinite(fee) or fee < 0.0:
            raise DataIntegrityError("fee_bps must be a finite nonnegative fee")


@dataclass(frozen=True, slots=True)
class TerminalPositionEvidence:
    """Distinguish priced open inventory at an observation cutoff from an instrument
    settlement and from unobservable financial state. Open valuation does not prove
    immediate liquidation capacity or future tradability.
    """

    symbol: str
    quantity: float
    cutoff: pd.Timestamp
    status: Literal["open_marked", "settled", "unresolved"]
    mark: float | None
    mark_available_at: pd.Timestamp | None
    funding_complete: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise DataIntegrityError("terminal symbol must be a nonempty string")
        if self.status not in ("open_marked", "settled", "unresolved"):
            raise DataIntegrityError(f"unknown terminal status {self.status!r}")
        if not isinstance(self.cutoff, pd.Timestamp) or pd.isna(self.cutoff):
            raise DataIntegrityError("terminal cutoff must be a valid timestamp")
        if self.cutoff.tzinfo is None or self.cutoff.utcoffset() != _datetime.timedelta(0):
            raise DataIntegrityError("terminal cutoff must be timezone-aware UTC")
        if not np.isfinite(float(self.quantity)):
            raise DataIntegrityError("terminal quantity must be finite")
        if self.mark is not None and not (np.isfinite(float(self.mark)) and float(self.mark) > 0.0):
            raise DataIntegrityError("terminal mark must be a finite positive price when present")
        if self.mark_available_at is not None:
            value = self.mark_available_at
            if not isinstance(value, pd.Timestamp) or pd.isna(value):
                raise DataIntegrityError("terminal mark_available_at must be a valid timestamp")
            if value.tzinfo is None or value.utcoffset() != _datetime.timedelta(0):
                raise DataIntegrityError("terminal mark_available_at must be timezone-aware UTC")


@dataclass(frozen=True, slots=True)
class ForwardExecutionObservation:
    """Phase 4B forward collection record for one signal intent.

    Every intent is recorded, including rejected, cancelled, unfilled, and
    partial-filled orders. This data calibrates proxy fill/cost bounds and
    gates Execution/Pilot/Scale; it must never alter an already frozen signal,
    stop, exit, or sizing architecture after final OOS.
    """

    symbol: str
    signal_time: pd.Timestamp
    intent_time: pd.Timestamp
    submit_time: pd.Timestamp | None
    fill_time: pd.Timestamp | None
    side: int
    requested_quantity: float
    filled_quantity: float
    limit_price: float | None
    fill_price: float | None
    best_bid: float | None
    best_ask: float | None
    top_n_depth_notional: float | None
    trade_print_notional: float | None
    reject_reason: str | None
    cancel_replace_count: int
    latency_ms: int | None


@dataclass(frozen=True, slots=True)
class SimulatedInventoryLedgerResult:
    """The only PnL source allowed for Research GO, OOS, and capital metrics."""

    equity: pd.Series
    net_returns: pd.Series
    simulated_units: pd.DataFrame | None
    mark_to_market_pnl: pd.Series
    funding_charge: pd.Series
    fee_charge: pd.Series
    fill_turnover: pd.Series
    fill_source: str
    mark_source: _MarkSource
    primary_valid: bool
    invalid_reasons: tuple[str, ...]
    equity_floor_breached_at: tuple[pd.Timestamp, ...] = ()
    data_gaps: tuple[ExecutionDataGap, ...] = ()


@dataclass(frozen=True, slots=True)
class StrategyExecutionReplayResult:
    """Outcome of one OHLCV proxy-execution replay over target weights."""

    simulated_fills: pd.DataFrame
    ledger: SimulatedInventoryLedgerResult
    simulated_units: pd.DataFrame
    simulated_notional_weights: pd.DataFrame
    fill_source: _ExecutionBound
    mark_source: _MarkSource
    submit_times: pd.Series
    fill_times: pd.Series
    fill_count: int
    unfilled_count: int
    fallback_count: int
    all_intent_shortfall_bps: float
    forced_exit_count: int
    forced_exit_notional: float
    termination_counts: Mapping[str, int]
    unsupported_assumptions: tuple[str, ...]
    elapsed_seconds: float
    data_gaps: tuple[ExecutionDataGap, ...] = ()
    event_snapshots_retained: bool = True
    notional_weighted_shortfall_bps: float = float("nan")
    residual_count: int = 0
    residual_notional: float = 0.0
    notional_weighted_fee_bps: float = float("nan")
    notional_weighted_spread_bps: float = float("nan")
    notional_weighted_delay_bps: float = float("nan")
    min_notional_dropped_fraction: float = float("nan")
    funding_coverage_gaps: tuple[FundingCoverageGap, ...] = ()
    terminal_positions: tuple[TerminalPositionEvidence, ...] = ()
    ledger_available_at: pd.DatetimeIndex | None = None


@dataclass(frozen=True, slots=True)
class IsolatedBoundFailure:
    bound_index: int
    execution_bound: str
    error_class: str
    message: str
    windows_consumed: int


@dataclass(frozen=True, slots=True)
class BatchReplayOutcome:
    results: tuple[StrategyExecutionReplayResult | None, ...]
    isolated_failures: tuple[IsolatedBoundFailure, ...]


def ruin_guard_equity(fill_track_equity: float, last_ledger_equity: float | None) -> float:
    if last_ledger_equity is None:
        return float(fill_track_equity)
    if not np.isfinite(last_ledger_equity):
        return float(fill_track_equity)
    return float(min(fill_track_equity, last_ledger_equity))
