"""Bound-specific streaming replay accumulator (cohesive stateful class)."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.core.types import ExecutionSpec
from src.engine.execution.accounting import QTY_EPS, CausalPortfolioState, reconcile_causal_state

from . import _ExecutionBound, _MarkSource
from . import contracts as _contracts
from . import fill_terms as _fill_terms
from . import microstructure as _microstructure
from .contracts import (
    ExecutionDataGap,
    ExecutionReplayWindow,
    ExitBlockDisclosure,
    FundingCoverageGap,
    InstrumentSettlementEvent,
    SimulatedInventoryLedgerResult,
    StrategyExecutionReplayResult,
    TerminalPositionEvidence,
)
from .exit_deferral import ExitBlockCause, ExitEpisodeClock, classify_exit_block_cause, first_viable_retry_bar
from .fill_flow import assemble_fill_flow
from .funding_attribution import FundingAttribution
from .settlement import _INT64_MAX
from .window_staging import WindowStaging

_BOOKED_FILL_REASONS: frozenset[str] = frozenset({"passive_fill", "timeout_taker", "delist_settlement"})


def _halted_rows(w: ExecutionReplayWindow, grid_ns: np.ndarray) -> tuple[np.ndarray, list[str | None]]:
    """Stage halt membership and identities once, preserving all-False membership when absent."""
    halted = np.zeros(len(grid_ns), dtype=bool)
    ids: list[str | None] = [None] * len(grid_ns)
    for halt in w.venue_halts:
        start = int(halt.start.value)
        stop = int(halt.end.value)
        lo = int(np.searchsorted(grid_ns, np.int64(start), side="left"))
        hi = int(np.searchsorted(grid_ns, np.int64(stop), side="left"))
        halted[lo:hi] = True
        for pos in range(lo, min(hi, len(ids))):
            ids[pos] = halt.halt_id
    return halted, ids


@dataclasses.dataclass(frozen=True, slots=True)
class _WindowFrame:
    """Read-only view of one execution window staged for the replay phases.

    Every field references an array staged once per window by
    ``stage_window_arrays`` and shared read-only across bound accumulators;
    the frame never owns copies, so passing it costs one pointer and the
    window's working set is unchanged. It exists so phase methods receive the
    window as one named bundle instead of 16-24 positional arrays whose order
    differed per method. Arrays are shared with the caller and must be treated
    as immutable for the frame's lifetime (one ``consume`` call).

    Attributes:
        local_cols: Window roster in canonical order (a list because pandas column selection
            treats a tuple as one key).
        gpos: Local-to-canonical column positions, ``intp``.
        grid: Window bar labels (UTC).
        grid_ns: ``grid`` as ``int64`` nanoseconds.
        bar_ns: Bar width in nanoseconds.
        marks_values: Valuation marks, shape ``(n_grid, n_local)``.
        highs_values: Bar highs, same shape.
        lows_values: Bar lows, same shape.
        closes_values: Bar closes, same shape.
        mark_valid: Finite strictly-positive mark mask, same shape.
        funding_matrix: Per-bar funding rates, same shape.
        last_close_idx: Last finite-close row index at or before each row, ``-1`` when none.
        decision_ns_all: Decision labels in nanoseconds, one per target row.
        spos_all: First grid position strictly after each signal availability.
        dpos_all: Grid position of each decision label (``searchsorted`` left).
        on_grid_all: Whether each decision label is an exact grid label.
        target_values: Target weights, shape ``(n_decisions, n_local)``.
        submit_anchored: ``spec.decision_anchor == "submit_bar"``.
        fill_start: Index of the first fill booked while consuming this window.
        tw_index: Decision index (boxed lazily, only on gap/breach paths).
        sig_index: Signal availability index aligned to ``tw_index``.
    """

    local_cols: list[str]
    gpos: np.ndarray
    grid: pd.DatetimeIndex
    grid_ns: np.ndarray
    bar_ns: int
    marks_values: np.ndarray
    highs_values: np.ndarray
    lows_values: np.ndarray
    closes_values: np.ndarray
    mark_valid: np.ndarray
    funding_matrix: np.ndarray
    last_close_idx: np.ndarray
    decision_ns_all: np.ndarray
    spos_all: np.ndarray
    dpos_all: np.ndarray
    on_grid_all: np.ndarray
    target_values: np.ndarray
    submit_anchored: bool
    fill_start: int
    tw_index: pd.DatetimeIndex
    sig_index: pd.DatetimeIndex

    @property
    def n_local(self) -> int:
        """Local symbol count."""
        return len(self.local_cols)

    @property
    def n_grid(self) -> int:
        """Window bar count."""
        return len(self.grid_ns)


def _forced_exit_nan_override(
    row: np.ndarray, frame: _WindowFrame, units_arr: np.ndarray,
    admitted_by_symbol: dict[str, InstrumentSettlementEvent], *, information_ns: int,
) -> np.ndarray:
    """Promote NaN-hold targets to zero weight where the lifecycle demands a forced exit.

    A NaN target normally means hold, but R3 (forced exit inside the lead) must
    still be processed; returning a copy with 0.0 in exactly those columns lets
    the unchanged active-column selection issue the exit intent.
    """
    from src.core.params import DELIST_FORCED_EXIT_LEAD

    lead_ns = int(DELIST_FORCED_EXIT_LEAD.value)
    nan_cols = np.flatnonzero(~np.isfinite(row))
    if nan_cols.size == 0:
        return row
    out: np.ndarray | None = None
    for col in nan_cols.tolist():
        col = int(col)
        if float(units_arr[int(frame.gpos[col])]) == 0.0:
            continue
        event = admitted_by_symbol.get(frame.local_cols[col])
        if event is None:
            continue
        announced = event.announced_at if event.announced_at is not None else event.effective_at
        if int(announced.value) > information_ns:
            continue
        if information_ns + lead_ns < int(event.effective_at.value):
            continue
        if out is None:
            out = np.array(row, dtype="float64", copy=True)
        out[col] = 0.0
    return out if out is not None else row


class _BoundExecutionReplayAccumulator:
    """Private streaming accumulator for one execution bound.

    ``replay_execution_windows`` and ``replay_execution_window_pair`` share
    this bound-specific state machine. Windows are consumed one at a time:
    cash, units, last prices, the last finite-close mark provenance, and the
    streamed ledger carry into the next window, and a completed window's
    frames are released before the next is read. Each window's grid covers the
    strict timeout overlap of its final order plus the boundary bars needed
    for decision-time funding/MTM, so an order never crosses a window boundary
    unresolved. The six ledger series are computed per window in chronological
    order and concatenated once in ``finalize``, matching the single-panel
    oracle at ``rtol=atol=1e-12`` where the inputs are equal.

    ``retain_event_snapshots`` defaults to ``False`` for bounded memory: the
    dense per-fill ``simulated_units``/``simulated_notional_weights`` event
    tables are then empty (correctly columned) and ``event_snapshots_retained``
    is ``False``, so empty tables cannot be mistaken for no fills. Diagnostic
    callers that compare event snapshots (the single-panel oracle and
    equivalence tests) must explicitly opt in with ``True``; the ledger, fills,
    gaps, termination data, and numerical results are identical either way.
    """

    def __init__(
        self,
        first: ExecutionReplayWindow,
        initial_equity: float,
        execution_bound: _ExecutionBound,
        spec: ExecutionSpec,
        retain_event_snapshots: bool,
        min_equity_fraction: float | None = None,
    ) -> None:
        if initial_equity <= 0:
            raise DataIntegrityError("initial_equity must be > 0")
        if min_equity_fraction is not None and not (0.0 < min_equity_fraction < 1.0):
            raise ValueError(f"min_equity_fraction must be in (0.0, 1.0) when set, got {min_equity_fraction}")
        self.min_equity_fraction = min_equity_fraction
        self.initial_equity = float(initial_equity)
        self.equity_floor_breaches: list[pd.Timestamp] = []
        if execution_bound not in (
            "OHLCV_STRICT_PROXY",
            "OHLCV_TOUCH_PROXY",
            "OHLCV_IMMEDIATE_TAKER",
            "OHLCV_LADDERED_PROXY",
            "OHLCV_PEG_CHASE_PROXY",
        ):
            raise ValueError(f"unknown execution_bound '{execution_bound}'")
        self.execution_bound = execution_bound
        self.require_strict = execution_bound == "OHLCV_STRICT_PROXY"
        self.spec = spec
        self.retain_event_snapshots = retain_event_snapshots
        self.timeout_ns_delta = int(spec.passive_timeout_minutes) * 60_000_000_000

        self.columns = tuple(first.columns)
        self.n_cols = len(self.columns)
        self.gpos_of = {sym: i for i, sym in enumerate(self.columns)}
        self.mark_source: _MarkSource = "MARK_PRICE" if first.marks is not None else "OHLCV_CLOSE_FALLBACK"
        self.first_grid = first.minute_grid

        self.units_arr = np.zeros(self.n_cols, dtype="float64")
        self.cash = float(initial_equity)
        self.last_prices_arr = np.full(self.n_cols, np.nan, dtype="float64")
        self.last_time_ns: int | None = None

        # Causal accounting mirror (P1): shadows the fill track event by event,
        # reconciled in ``finalize``; queued fills drain in bar order (funding first).
        self.accounting_state = CausalPortfolioState(
            cash=float(initial_equity),
            units=np.zeros(self.n_cols, dtype="float64"),
            last_marks=np.full(self.n_cols, np.nan, dtype="float64"),
            last_event_ns=None,
        )
        self._mirror_pending: list[tuple[int, int, float, float, float]] = []
        self._w_qv = np.zeros((0, 0), dtype="float64")
        self._w_volume_symbols: frozenset[str] = frozenset()
        self._w_fknown = np.zeros((0, 0), dtype=bool)
        self._w_avail_ns: np.ndarray = np.zeros(0, dtype="int64")
        self._w_avail_explicit = False
        self._w_mark_avail = np.zeros((0, 0), dtype="int64")
        self.last_liquid_ns = np.full(self.n_cols, -1, dtype="int64")
        self._w_last_liquid_idx = np.zeros((0, 0), dtype=np.intp)
        self._span_scan_from = 0
        self.fill_bar_ns: list[int] = []
        self.fill_gcol: list[int] = []

        self.ledger_cash = float(initial_equity)
        self.ledger_units = np.zeros(self.n_cols, dtype="float64")
        self.last_valid_mark = np.full(self.n_cols, np.nan, dtype="float64")
        self._last_valid_mark_avail_ns = np.full(self.n_cols, -1, dtype="int64")
        self.ledger_start_ns: int | None = None

        self.last_close_ts: dict[str, pd.Timestamp] = {}
        self.last_close_value: dict[str, float] = {}
        self.last_close_mark: dict[str, float] = {}
        self.fill_ts: list[pd.Timestamp] = []
        self.fill_symbol: list[str] = []
        self.fill_qty: list[float] = []
        self.fill_post_units: list[float] = []
        self.fill_price: list[float] = []
        self.fill_fee_bps: list[float] = []
        self.fill_reason: list[str] = []
        self.fill_pre_trade_equity: list[float] = []
        self.fill_bar_qv: list[float] = []
        self.submit_times: list[pd.Timestamp] = []
        self.fill_times: list[pd.Timestamp] = []
        self.shortfalls: list[float] = []
        self.shortfall_notionals: list[float] = []
        self.fill_count = 0
        self.unfilled_count = 0
        self.fallback_count = 0
        self.residual_count = 0
        self.residual_notional = 0.0
        # Liquidity-aware taker cost state per column, nan until bars are consumed.
        self.half_spread_bps = np.full(self.n_cols, np.nan, dtype="float64")
        # Logical cost-clock observations for the in-progress decision
        # partition: additive Corwin-Schultz sufficient statistics plus the
        # previous piece's closing bar per column for the cross-piece
        # boundary pair. O(window) scratch per piece and O(symbols) retained,
        # never full high/low histories.
        self._spread_pending_key: tuple[int, int] | None = None
        self._spread_pair_sums = np.zeros(self.n_cols, dtype="float64")
        self._spread_pair_counts = np.zeros(self.n_cols, dtype="float64")
        self._spread_bar_counts = np.zeros(self.n_cols, dtype="float64")
        self._spread_carry_high = np.full(self.n_cols, np.nan, dtype="float64")
        self._spread_carry_low = np.full(self.n_cols, np.nan, dtype="float64")
        self._spread_carry_ns = np.full(self.n_cols, -1, dtype="int64")
        # Cost decomposition terms paired 1:1 with ``self.shortfalls``.
        self.fee_terms: list[float] = []
        self.spread_terms: list[float] = []
        self.delay_terms: list[float] = []
        # Min-notional diagnostic accumulators (report-only; never the ledger).
        self.min_notional_probe_usdt = float(spec.min_notional_probe_usdt)
        self.min_notional_total_notional = 0.0
        self.min_notional_dropped_notional = 0.0
        self.termination_counts: dict[str, int] = {"MISSING_DATA": 0, "UNKNOWN_TERMINATION": 0}
        self.data_gaps: list[ExecutionDataGap] = []
        self.funding_coverage_gaps: dict[tuple[str, pd.Timestamp, pd.Timestamp, str], FundingCoverageGap] = {}
        self._settlement_seen: set[str] = set()
        self._pending_settlements: list[InstrumentSettlementEvent] = []
        self._settled_events: list[InstrumentSettlementEvent] = []
        self._booked_settlement_events: list[InstrumentSettlementEvent] = []
        self._admitted_events: dict[str, InstrumentSettlementEvent] = {}
        self._admitted_by_symbol: dict[str, InstrumentSettlementEvent] = {}
        self.exit_block_disclosures: list[ExitBlockDisclosure] = []
        from src.core.params import EXIT_DEFERRAL_MAX_AGE
        self._exit_clock = ExitEpisodeClock(int(EXIT_DEFERRAL_MAX_AGE.value))
        self._w_halted = np.zeros(0, dtype=bool)
        self._w_halt_id: list[str | None] = []
        self._last_trade_ns = np.full(self.n_cols, _INT64_MAX, dtype="int64")
        self._held_funding_first: dict[str, pd.Timestamp] = {}
        self._settlement_overrides: dict[tuple[int, int], float] = {}
        self._settlement_prices: dict[str, float] = {}
        self._final_mark = np.full(self.n_cols, np.nan, dtype="float64")
        self._final_mark_avail_ns = np.full(self.n_cols, -1, dtype="int64")
        self._ledger_avail_chunks: list[np.ndarray] = []
        self._ledger_avail_complete = True
        self.units_after_events: list[tuple[pd.Timestamp, np.ndarray]] = []
        self.notional_after_events: list[tuple[pd.Timestamp, np.ndarray]] = []

        self.equity_chunks: list[np.ndarray] = []
        self.equity_times: list[pd.DatetimeIndex] = []
        self.mtm_chunks: list[np.ndarray] = []
        self.funding_chunks: list[np.ndarray] = []
        self.fee_chunks: list[np.ndarray] = []
        self.turnover_chunks: list[np.ndarray] = []
        self._funding = FundingAttribution(self.columns)
        self.ledger_valid = True
        self.invalid_reasons: set[str] = set()
        self.first_held_mark: tuple[str, pd.Timestamp] | None = None
        self.first_held_funding: tuple[str, pd.Timestamp] | None = None
        self.full_grid_end: pd.Timestamp = first.minute_grid[-1]
        self._trim_anchor_ns: int | None = None
        self._t0 = time.perf_counter()

    def required_symbols(self) -> frozenset[str]:
        """Expose actual carried inventory and unresolved-order source requirements.

        Returns:
            Symbols with nonzero units or unresolved orders, without tiny-unit pruning.
        """
        held = {self.columns[int(i)] for i in np.flatnonzero(self.units_arr != 0.0).tolist()}
        pending = {self.columns[int(gcol)] for (_, gcol, _, _, _) in self._mirror_pending}
        return frozenset(held | pending)

    def _equity_at(self, gpos: np.ndarray | None = None) -> float:
        # NaN-only zeroing instead of nan_to_num: bit-identical on the
        # reachable domain (prices are NaN or finite positives), ~2x faster,
        # and a +/-inf -- which nan_to_num silently mapped to +-finfo.max --
        # now fails closed instead of corrupting the equity ledger.
        prices = self.last_prices_arr if gpos is None else self.last_prices_arr[gpos]
        if np.isinf(prices).any():
            raise DataIntegrityError(
                "last prices must never be infinite; a non-finite mark slipped "
                "past the strictly-positive finite-mark invariant"
            )
        units = self.units_arr if gpos is None else self.units_arr[gpos]
        return self.cash + float(
            np.sum(units * np.where(np.isnan(prices), 0.0, prices))
        )

    def _taker_cost_bps(self, gcol: int) -> float:
        """Liquidity-aware taker crossing cost for one column.

        Under ``corwin_schultz`` the column's EWMA half-spread replaces the
        flat slippage whenever it is finite; a degenerate estimate (nan)
        falls back to ``taker_slippage_bps``. Under ``flat`` this is exactly
        the fixed slippage, reproducing legacy behaviour bit-identically.
        """
        if self.spec.liquidity_cost_model == "corwin_schultz":
            est = float(self.half_spread_bps[gcol])
            if np.isfinite(est):
                return est
        return float(self.spec.taker_slippage_bps)

    def _record_terms(
        self,
        decision_price: float,
        fill_price: float,
        side: int,
        fee_component_bps: float,
        spread_component_bps: float,
    ) -> None:
        """Append the fee/spread/delay decomposition terms for one shortfall.

        ``delay`` is ``side * (fill_price / decision_price - 1) * 1e4`` -- the
        pure timing cost of filling away from the anchor -- so that
        ``fee + spread + delay`` reconstructs the recorded shortfall.
        """
        self.fee_terms.append(float(fee_component_bps))
        self.spread_terms.append(float(spread_component_bps))
        self.delay_terms.append(side * (fill_price / decision_price - 1.0) * 1e4)

    def _probe_intent_notional(self, net_units: float, decision_price: float) -> None:
        """Accumulate the min-notional diagnostic for one intent (ledger-neutral)."""
        if self.min_notional_probe_usdt <= 0:
            return
        dollar = (
            abs(net_units * decision_price)
            * self.spec.reference_equity_usdt
            / self.initial_equity
        )
        self.min_notional_total_notional += dollar
        if dollar < self.min_notional_probe_usdt:
            self.min_notional_dropped_notional += dollar

    def consume(self, w: ExecutionReplayWindow, staging: WindowStaging | None = None) -> None:
        """Consume one window through the ordered replay phases. Evidenced settlements are booked
        at their delivery bar before any later decision, drift check, funding span or ledger append, so no
        decision sizes on delivered inventory and no funding accrues after delivery. Physical IO
        boundaries do not reset inventory, liquidity or the logical spread clock; overlap observations
        and settlements are applied once.

        Args:
            w: Window to consume.
            staging: Shared staging built for exactly ``w`` by the batch driver;
                None stages privately (single-bound replay and direct callers).
        Raises:
            ValueError: ``staging`` was built for a different window object
                (``"staging was built for a different window"``); raised before
                any state mutation.
            DataIntegrityError: Window validation or replay integrity fails.
        """
        frame = self._consume_validate_window(w, staging)
        for gap in w.funding_coverage_gaps:
            self.funding_coverage_gaps[(gap.symbol, gap.start, gap.end, gap.reason)] = gap
        self._admit_settlement_events(w)
        self._advance_spread_clock(w.logical_partition)
        for i in range(len(frame.tw_index)):
            self._consume_drift_trims(frame, int(frame.decision_ns_all[i]))
            self._consume_single_intent(frame, i)
        self._consume_drift_trims(frame, None)
        self._settle_events_through(frame, int(frame.grid_ns[-1]))
        self._consume_append_ledger(frame)
        self._advance_liquidity_carry(frame)
        self._observe_spread_partition(frame, w.logical_partition)

    def _consume_validate_window(self, w: ExecutionReplayWindow, staging: WindowStaging | None = None) -> _WindowFrame:
        """Validate one window and stage its grids, marks, and funding."""
        columns = self.columns
        n_cols = self.n_cols
        gpos_of = self.gpos_of
        if w.columns != columns:
            raise DataIntegrityError("all execution windows must share an identical column order")
        local_cols = list(w.symbols)
        gpos = np.asarray([gpos_of[s] for s in local_cols], dtype=np.intp)
        in_window = np.zeros(n_cols, dtype=bool)
        in_window[gpos] = True
        outside = np.flatnonzero((np.abs(self.units_arr) >= QTY_EPS) & ~in_window)
        if outside.size:
            j = int(outside[0])
            raise DataIntegrityError(f"held position outside execution window roster (symbol={columns[j]!r} units={float(self.units_arr[j])!r})")
        if staging is None:
            staging = WindowStaging(w)
        if staging.window is not w:
            raise ValueError("staging was built for a different window")
        staged = staging.arrays()
        self.full_grid_end = staged.grid[-1]
        self._w_qv = staged.quote_volumes
        self._w_volume_symbols = frozenset(w.quote_volumes.columns) if w.quote_volumes is not None else frozenset()
        self._w_last_liquid_idx = staged.last_liquid_idx
        self._w_fknown = staged.funding_known
        self._w_avail_ns = staged.avail_ns
        self._w_avail_explicit = staged.avail_explicit
        self._w_mark_avail = staged.mark_avail
        self._w_halted, self._w_halt_id = _halted_rows(w, staged.grid_ns)
        submit_anchored = self.spec.decision_anchor == "submit_bar"
        fill_start = len(self.fill_ts)
        return _WindowFrame(
            local_cols=list(staged.local_cols), gpos=gpos, grid=staged.grid, grid_ns=staged.grid_ns, bar_ns=staged.bar_ns,
            marks_values=staged.marks_values, highs_values=staged.highs_values, lows_values=staged.lows_values,
            closes_values=staged.closes_values, mark_valid=staged.mark_valid, funding_matrix=staged.funding_matrix,
            last_close_idx=staged.last_close_idx, decision_ns_all=staged.decision_ns_all, spos_all=staged.spos_all,
            dpos_all=staged.dpos_all, on_grid_all=staged.on_grid_all, target_values=staged.target_values,
            submit_anchored=submit_anchored, fill_start=fill_start,
            tw_index=w.target_weights.index, sig_index=w.signal_available_at,
        )


    def _advance_window(self, frame: _WindowFrame, target_ns: int, dpos: int, on_grid: bool, *, pre_fill_funding: bool = False) -> None:
        """Advance fill-track MTM and funding state to a decision time.

        Single-MTM (INV-ACCOUNTING-SINGLE-MTM): marks update last prices but
        never touch cash; cash moves only through fills, fees, and funding.
        Funding settles per bar on pre-fill inventory (INV-EVENT-ORDER): fills
        recorded since the previous decision are backed out bar by bar so a
        post-entry decision never charges pre-entry funding (F3). A mark read
        whose availability postdates the decision is a FUTURE_DATA_REFERENCE
        gap, never a sizing input (INV-PIT-SOURCE-TIME).
        """
        if self.last_time_ns is not None and target_ns < self.last_time_ns:
            raise DataIntegrityError("decision times must be monotonically increasing")
        if on_grid:
            m = frame.marks_values[dpos]
            avail = self._w_mark_avail[dpos]
            usable = np.isfinite(m) & (m > 0.0) & (avail <= target_ns)
            prev = self.last_prices_arr[frame.gpos]
            self.last_prices_arr[frame.gpos] = np.where(usable, m, prev)
            future_read = np.isfinite(m) & (m > 0.0) & ~usable
            if bool(future_read.any()):
                self.ledger_valid = False
                self.invalid_reasons.add("MISSING_DATA")
                event_ts = pd.Timestamp(target_ns, unit="ns", tz="UTC")
                for j in np.flatnonzero(future_read).tolist():
                    self.data_gaps.append(
                        ExecutionDataGap(
                            code="FUTURE_DATA_REFERENCE", symbol=self.columns[int(frame.gpos[j])],
                            timestamp=event_ts, execution_bound=self.execution_bound,
                        )
                    )
        lo = np.searchsorted(frame.grid_ns, self.last_time_ns, side="right") if self.last_time_ns is not None else 0
        hi = int(np.searchsorted(frame.grid_ns, target_ns, side="right"))
        lo = int(lo)
        if lo < hi:
            span_rates = frame.funding_matrix[lo:hi, :]
            span_known = self._w_fknown[lo:hi, :]
            span_marks = frame.marks_values[lo:hi, :]
            span_units = np.repeat(self.units_arr[frame.gpos][None, :], hi - lo, axis=0)
            for fns, fj, fqty in self._fills_in_span(self.last_time_ns, target_ns, frame.grid_ns, lo, frame.gpos):
                span_units[:fns + int(pre_fill_funding), int(fj)] -= float(fqty)
            held_unknown = (np.abs(span_units) >= QTY_EPS) & ~span_known
            if bool(held_unknown.any()):
                self.ledger_valid = False
                self.invalid_reasons.add("MISSING_DATA")
                for j in np.flatnonzero(held_unknown.any(axis=0)).tolist():
                    rows = np.flatnonzero(held_unknown[:, int(j)])
                    first_row = int(rows[0])
                    sym = self.columns[int(frame.gpos[int(j)])]
                    stamp = pd.Timestamp(int(frame.grid_ns[lo + first_row]), unit="ns", tz="UTC")
                    self.data_gaps.append(
                        ExecutionDataGap(
                            code="MISSING_HELD_FUNDING", symbol=sym,
                            timestamp=stamp,
                            execution_bound=self.execution_bound,
                        )
                    )
                    if sym not in self._held_funding_first:
                        self._held_funding_first[sym] = stamp
                        if self.first_held_funding is None:
                            self.first_held_funding = (sym, stamp)
            priced = np.where(np.isfinite(span_marks), span_marks, 0.0)
            self.cash -= float(np.sum(np.where(span_known, span_rates, 0.0) * span_units * priced))
        self.last_time_ns = target_ns

    def _fills_in_span(
        self,
        last_ns: int | None,
        target_ns: int,
        grid_ns: np.ndarray,
        lo: int,
        gpos: np.ndarray,
    ) -> list[tuple[int, int, float]]:
        """Recorded fills inside ``(last_ns, target_ns]`` as span adjustments.

        Returns ``(relative_bar, local_col, quantity)`` triples used to back
        post-fill inventory out of pre-fill bars. ``relative_bar`` is clamped
        at zero so fills predating the span leave every span bar untouched.
        """
        out: list[tuple[int, int, float]] = []
        floor = -1 if last_ns is None else int(last_ns)
        target = int(target_ns)
        n = len(self.fill_bar_ns)
        start = self._span_scan_from
        # Monotonic last_time_ns cursor skips prefix fills at or below floor to avoid O(N) rescans.
        while start < n and int(self.fill_bar_ns[start]) <= floor:
            start += 1
        self._span_scan_from = start
        local_of = {int(g): j for j, g in enumerate(gpos.tolist())}
        for k in range(start, n):
            fns = int(self.fill_bar_ns[k])
            if fns <= floor or fns > target:
                continue
            j = local_of.get(int(self.fill_gcol[k]))
            if j is None:
                continue
            rel = int(np.searchsorted(grid_ns, fns, side="left")) - int(lo)
            out.append((max(rel, 0), j, float(self.fill_qty[k])))
        return out

    def _bar_viability_gap(
        self,
        *,
        fill_pos: int,
        col: int,
        sym: str,
        decision_ts: pd.Timestamp,
        signal_ts: pd.Timestamp,
        submit_ns: int,
        signal_ns: int,
        avail_submit_ns: int,
    ) -> ExecutionDataGap | None:
        """Fail-closed per-fill viability: volume, funding knowledge, timing.

        A fill bar with zero/unknown/non-positive quote volume is untradable
        (INV-EXECUTION-VIABILITY); unknown funding at the fill bar cannot
        price the order (INV-FUNDING-KNOWLEDGE); an order whose submit time
        falls outside ``(signal, availability]`` has no causal standing
        (INV-BAR-AVAILABILITY). Returns the blocking gap, else ``None``.
        """
        if not (int(signal_ns) < int(submit_ns) <= int(avail_submit_ns)):
            return ExecutionDataGap(
                code="CAUSAL_TIMING_VIOLATION", symbol=sym,
                timestamp=decision_ts, decision_time=decision_ts, signal_time=signal_ts,
                execution_bound=self.execution_bound,
            )
        qv = float(self._w_qv[fill_pos, col])
        if qv == 0.0 or bool(self._w_halted[fill_pos]):
            return ExecutionDataGap(
                code="KNOWN_ZERO_VOLUME", symbol=sym,
                timestamp=decision_ts, decision_time=decision_ts, signal_time=signal_ts,
                execution_bound=self.execution_bound,
            )
        if not (qv > 0.0):
            return ExecutionDataGap(
                code="ZERO_OR_UNKNOWN_VOLUME", symbol=sym,
                timestamp=decision_ts, decision_time=decision_ts, signal_time=signal_ts,
                execution_bound=self.execution_bound,
            )
        if not bool(self._w_fknown[fill_pos, col]):
            return ExecutionDataGap(
                code="MISSING_ACTIVE_FUNDING", symbol=sym,
                timestamp=decision_ts, decision_time=decision_ts, signal_time=signal_ts,
                execution_bound=self.execution_bound,
            )
        return None

    def _block_fill(
        self, gap: ExecutionDataGap, *, prior_units: float, net_units: float,
        frame: _WindowFrame | None = None, col: int | None = None,
        fill_pos: int | None = None, timeout_pos: int | None = None,
        submit_pos: int | None = None, side: int | None = None,
        decision_price: float | None = None, equity: float | None = None,
        weight: float | None = None,
    ) -> bool:
        """Record a viability block; a deferrable exit is re-attempted inside its timeout.

        Entries never defer. A blocked exit whose mark is valued and whose
        funding is known is disclosed and retried rather than invalidated,
        bounded by the episode age; anything else takes the legacy fail-closed path.
        """
        is_exit = abs(prior_units) >= QTY_EPS and abs(prior_units + net_units) < abs(prior_units)
        if (
            gap.code == "KNOWN_ZERO_VOLUME" and is_exit and col is not None
            and fill_pos is not None
            and not bool(self._w_fknown[fill_pos, col])
        ):
            gap = dataclasses.replace(gap, code="MISSING_ACTIVE_FUNDING")
        if (
            gap.code == "KNOWN_ZERO_VOLUME" and is_exit
            and frame is not None and col is not None and fill_pos is not None
            and timeout_pos is not None and 0 <= int(fill_pos) < len(self._w_halted)
        ):
            cause = classify_exit_block_cause(
                halted=bool(self._w_halted[int(fill_pos)]),
                quote_volume=float(self._w_qv[int(fill_pos), int(col)]),
                mark_valid=bool(frame.mark_valid[int(fill_pos), int(col)]),
                funding_known=bool(self._w_fknown[int(fill_pos), int(col)]),
            )
            if cause is not None:
                decision_ns = int(gap.decision_time.value) if gap.decision_time is not None else int(frame.grid_ns[int(fill_pos)])
                if self._exit_clock.admit(int(frame.gpos[int(col)]), float(prior_units), decision_ns):
                    return self._defer_blocked_exit(
                        gap, cause, frame=frame, col=int(col), fill_pos=int(fill_pos),
                        timeout_pos=int(timeout_pos), submit_pos=submit_pos, side=side,
                        decision_price=decision_price, equity=equity, weight=weight,
                        net_units=float(net_units),
                    )
                self.termination_counts["EXIT_BLOCK_BOUND_EXCEEDED"] = self.termination_counts.get("EXIT_BLOCK_BOUND_EXCEEDED", 0) + 1
        if gap.code == "KNOWN_ZERO_VOLUME" and not is_exit:
            # Known zero volume is unfillable venue state; retry next decision (mirrors live).
            self.unfilled_count += 1
            self.termination_counts["NO_VOLUME_UNFILLED"] = self.termination_counts.get("NO_VOLUME_UNFILLED", 0) + 1
            return True
        if gap.code == "MISSING_ACTIVE_FUNDING" and not is_exit:
            # Blocked new entries carry no inventory risk before fill; record unfilled and retry.
            self.unfilled_count += 1
            self.termination_counts["NO_FUNDING_UNFILLED"] = self.termination_counts.get("NO_FUNDING_UNFILLED", 0) + 1
            return True
        if gap.code == "MISSING_ACTIVE_FUNDING":
            recorded = dataclasses.replace(gap, code="BLOCKED_EXIT_UNKNOWN_FUNDING")
            self.data_gaps.append(recorded)
            self.ledger_valid = False
            self.invalid_reasons.add("MISSING_DATA")
            self.unfilled_count += 1
            self.termination_counts["BLOCKED_EXIT_UNKNOWN_FUNDING"] = self.termination_counts.get("BLOCKED_EXIT_UNKNOWN_FUNDING", 0) + 1
            return True
        self.data_gaps.append(gap)
        self.ledger_valid = False
        self.invalid_reasons.add("MISSING_DATA")
        self.unfilled_count += 1
        if gap.code == "KNOWN_ZERO_VOLUME":
            self.termination_counts["NO_VOLUME_UNFILLED"] = self.termination_counts.get("NO_VOLUME_UNFILLED", 0) + 1
        return True

    def _defer_blocked_exit(
        self, gap: ExecutionDataGap, cause: ExitBlockCause, *, frame: _WindowFrame, col: int,
        fill_pos: int, timeout_pos: int, submit_pos: int | None, side: int | None,
        decision_price: float | None, equity: float | None, weight: float | None,
        net_units: float,
    ) -> bool:
        """Book a blocked exit at the first viable bar or disclose a next-decision retry."""
        gcol = int(frame.gpos[col])
        sym = frame.local_cols[col]
        blocked_bar = frame.grid[fill_pos]
        decision_ts = gap.decision_time if gap.decision_time is not None else blocked_bar
        cause_str = cause.value
        halted_cause = cause_str == ExitBlockCause.VENUE_HALT.value
        halt_id = self._w_halt_id[fill_pos] if 0 <= fill_pos < len(self._w_halt_id) else None
        defer = first_viable_retry_bar(
            after_pos=int(fill_pos), deadline_pos=min(int(timeout_pos), int(frame.n_grid) - 1),
            halted=self._w_halted, quote_volume=self._w_qv[:, col],
            funding_known=self._w_fknown[:, col], close=frame.closes_values[:, col],
            grid_ns=frame.grid_ns, last_trade_ns=int(self._last_trade_ns[gcol]),
        )
        if defer < 0:
            self.unfilled_count += 1
            retry_key = "BLOCKED_EXIT_VENUE_HALT" if halted_cause else "BLOCKED_EXIT_SYMBOL_NO_TRADE"
            self.termination_counts[retry_key] = self.termination_counts.get(retry_key, 0) + 1
            self.exit_block_disclosures.append(
                ExitBlockDisclosure(
                    symbol=sym, decision_time=decision_ts, blocked_bar=blocked_bar,
                    cause=cause_str, halt_id=str(halt_id) if halted_cause and halt_id is not None else None,
                    outcome="retry_next_decision", filled_bar=None, quantity=float(net_units),
                )
            )
            return True
        fill_price = float(frame.closes_values[defer, col])
        fee_bps = self.spec.taker_fee_bps + self._taker_cost_bps(gcol)
        order_side = int(side) if side is not None else (1 if float(net_units) > 0 else -1)
        anchor = float(decision_price) if decision_price is not None else fill_price
        shortfall = order_side * (fill_price / anchor - 1.0) * 1e4 + fee_bps
        if self.spec.liquidity_cost_model == "corwin_schultz":
            self._record_terms(
                anchor, fill_price, order_side,
                self.spec.taker_fee_bps, fee_bps - self.spec.taker_fee_bps,
            )
        else:
            self._record_terms(anchor, fill_price, order_side, fee_bps, 0.0)
        self.shortfalls.append(shortfall)
        self.shortfall_notionals.append(abs(float(net_units)) * fill_price)
        self.unfilled_count += 1
        self.fallback_count += 1
        filled_key = "VENUE_HALT_DEFERRED_EXIT" if halted_cause else "SYMBOL_NO_TRADE_DEFERRED_EXIT"
        self.termination_counts[filled_key] = self.termination_counts.get(filled_key, 0) + 1
        self._book_fill(
            frame, gcol=gcol, symbol=sym, bar_pos=defer,
            submit_pos=int(submit_pos) if submit_pos is not None else int(fill_pos),
            quantity=float(net_units), fill_price=fill_price, fee_bps=fee_bps,
            reason="timeout_taker",
            valuation_mark=float(frame.marks_values[defer, col]),
            pre_trade_equity=float(equity) if equity is not None else self._equity_at(frame.gpos),
            target_weight=float(weight) if weight is not None else 0.0,
            decision_price=anchor,
        )
        self.exit_block_disclosures.append(
            ExitBlockDisclosure(
                symbol=sym, decision_time=decision_ts, blocked_bar=blocked_bar,
                cause=cause_str, halt_id=str(halt_id) if halted_cause and halt_id is not None else None,
                outcome="deferred_fill", filled_bar=frame.grid[defer],
                quantity=float(net_units),
            )
        )
        return True


    def _admit_settlement_events(self, w: ExecutionReplayWindow) -> None:
        """Admit evidenced settlement events once across physical overlap.

        Each event is validated on admission; an event without price, fee, or
        source identity is rejected rather than inferred from the last mark.
        Event IDs are idempotent across physical overlap and window boundaries.
        Absent actual source events never justify an inferred settlement.
        """
        for event in w.settlement_events:
            if not isinstance(event, InstrumentSettlementEvent):
                raise DataIntegrityError("settlement_events must be InstrumentSettlementEvent records")
            if event.symbol not in self.gpos_of:
                raise DataIntegrityError(f"settlement symbol {event.symbol!r} is not in canonical columns")
            if event.event_id in self._settlement_seen:
                if self._admitted_events[event.event_id] != event:
                    raise DataIntegrityError(f"conflicting settlement event {event.event_id}")
                continue
            due_ns = max(int(event.effective_at.value), int(event.available_at.value))
            if self.units_arr[self.gpos_of[event.symbol]] != 0.0 and self.last_time_ns is not None and due_ns <= self.last_time_ns:
                raise DataIntegrityError(f"settlement event {event.event_id} admitted after its due time")
            self._settlement_seen.add(event.event_id)
            self._pending_settlements.append(event)
            self._admitted_events[event.event_id] = event
            prev = self._admitted_by_symbol.get(event.symbol)
            if prev is None or event.effective_at < prev.effective_at:
                self._admitted_by_symbol[event.symbol] = event
            gcol = int(self.gpos_of[event.symbol])
            last_trade = event.last_trade_at if event.last_trade_at is not None else event.effective_at
            last_ns = int(last_trade.value)
            if last_ns < int(self._last_trade_ns[gcol]):
                self._last_trade_ns[gcol] = last_ns
        self._pending_settlements.sort(
            key=lambda e: (max(int(e.effective_at.value), int(e.available_at.value)), e.event_id)
        )

    def _settle_events_through(self, frame: _WindowFrame, through_ns: int) -> None:
        """Book admitted settlement events due at or before through_ns in causal order."""
        if not self._pending_settlements:
            return
        from src.engine.execution.settlement import settlement_fill_price as _settlement_fill_price

        local_of = {sym: (col, int(gcol)) for col, (sym, gcol) in enumerate(zip(frame.local_cols, frame.gpos.tolist(), strict=True))}
        remaining: list[InstrumentSettlementEvent] = []
        p0_ns = int(self.ledger_start_ns) if self.ledger_start_ns is not None else int(frame.grid_ns[0])
        for event in self._pending_settlements:
            due_ns = max(int(event.effective_at.value), int(event.available_at.value))
            if due_ns > min(int(through_ns), int(frame.grid_ns[-1])):
                remaining.append(event)
                continue
            entry = local_of.get(event.symbol)
            if entry is None:
                remaining.append(event)
                continue
            _, gcol = entry
            units = float(self.units_arr[gcol])
            if units == 0.0:
                self.termination_counts["DELIST_SETTLEMENT_NOOP"] = self.termination_counts.get("DELIST_SETTLEMENT_NOOP", 0) + 1
                continue
            if due_ns < p0_ns:
                raise DataIntegrityError(
                    f"settlement event {event.event_id!r} admitted after its due time "
                    f"(due={event.effective_at.isoformat()} last_time={pd.Timestamp(int(self.last_time_ns) if self.last_time_ns is not None else p0_ns, unit='ns', tz='UTC').isoformat()})"
                )
            due_pos = int(np.searchsorted(frame.grid_ns, np.int64(due_ns), side="left"))
            due_pos = max(due_pos, int(np.searchsorted(frame.grid_ns, np.int64(p0_ns), side="left")))
            due_pos = min(due_pos, int(frame.n_grid) - 1)
            price = float(_settlement_fill_price(event, units, self.spec))
            self._settlement_overrides[(due_pos, int(gcol))] = price
            # Funding on a shared delivery bar values every delivered instrument at its bound price.
            for peer in self._pending_settlements:
                if max(peer.effective_at.value, peer.available_at.value) == due_ns and peer.symbol in local_of:
                    peer_gcol = local_of[peer.symbol][1]
                    if self.units_arr[peer_gcol] != 0.0:
                        self._settlement_overrides[(due_pos, peer_gcol)] = _settlement_fill_price(peer, self.units_arr[peer_gcol], self.spec)
            if self.last_time_ns is None or int(frame.grid_ns[due_pos]) > self.last_time_ns:
                priced_frame = dataclasses.replace(frame, marks_values=self._settlement_patched_marks(frame))
                self._advance_window(priced_frame, int(frame.grid_ns[due_pos]), due_pos, True, pre_fill_funding=True)
            self._book_delist_settlement(
                frame, gcol, event.symbol, due_pos, units, price,
                fee_bps=float(event.fee_bps), event_id=event.event_id,
            )
            self._settled_events.append(event)
            self._booked_settlement_events.append(event)
            self._settlement_prices[event.event_id] = price
            self._settlement_overrides[(due_pos, int(gcol))] = price
        self._pending_settlements = remaining

    def _book_fill(
        self,
        frame: _WindowFrame,
        *,
        gcol: int,
        symbol: str,
        bar_pos: int,
        submit_pos: int,
        quantity: float,
        fill_price: float,
        fee_bps: float,
        reason: str,
        valuation_mark: float,
        pre_trade_equity: float,
        target_weight: float,
        decision_price: float,
    ) -> None:
        """Book one executed fill into the fill track, the mirror queue, and every fill column at once.

        Parallel fill lists append together after validation; rejected fills mutate nothing.

        Cash moves in two roundings (notional, then fee) because the fill-track cash feeds the next
        decision's sizing equity and therefore every downstream fill quantity; the ledger and the
        causal mirror settle the same fill in one rounding. Both conventions are pinned by the golden
        digests and must not be harmonized.

        The fill is only queued for the causal mirror; the mirror applies it after that bar's mark and
        funding, so the event order mark -> funding -> fill -> fee is decided in
        ``_settle_mirror_window``, never here.

        Args:
            gcol: Canonical column index of the instrument.
            symbol: Instrument symbol (``self.columns[gcol]``).
            bar_pos: Window grid position of the fill bar; its label is the fill's bar time and its
                ``bar_available_at`` stamp is the recorded fill timestamp.
            submit_pos: Window grid position whose availability stamp is the recorded submit time.
            quantity: Signed unit delta.
            fill_price: Execution price per unit.
            fee_bps: All-in fee rate in basis points of traded notional.
            reason: One of ``_BOOKED_FILL_REASONS``.
            valuation_mark: Price written to the last-price carry when finite (the published mark at
                the fill bar; the settlement price for a settlement). A non-finite value leaves the
                carry untouched.
            pre_trade_equity: Decision-time sizing equity: the account equity the engine used to size
                the intent that produced this fill (identical for every fill of one decision); for a
                settlement, the equity immediately before the settlement mutated any state. Mirrors
                the live ``FillEvent.pre_trade_equity`` definition.
            target_weight: Intent target weight, reported only in the failure message.
            decision_price: Intent anchor price, reported only in the failure message.

        Raises:
            DataIntegrityError: ``reason`` is not a booking reason; ``quantity`` or ``fill_price`` is
                non-finite (capital accounting invariant); ``pre_trade_equity`` is non-finite or
                not strictly positive. Raised before any state mutation.
        """
        if reason not in _BOOKED_FILL_REASONS:
            raise DataIntegrityError(f"unknown fill booking reason {reason!r} not in {sorted(_BOOKED_FILL_REASONS)}")
        fill_time = pd.Timestamp(int(self._w_avail_ns[bar_pos]), unit="ns", tz="UTC")
        if not (np.isfinite(quantity) and np.isfinite(fill_price)):
            raise DataIntegrityError(
                "non-finite fill sizing breaches the capital accounting invariant "
                f"(symbol={symbol!r} ts={fill_time!r} weight={target_weight!r} equity={pre_trade_equity!r} "
                f"decision_price={decision_price!r} qty={quantity!r} fill_price={fill_price!r})"
            )
        if not (np.isfinite(pre_trade_equity) and pre_trade_equity > 0):
            raise DataIntegrityError(
                "pre-trade equity must be positive and finite "
                f"(symbol={symbol!r} ts={fill_time!r} equity={pre_trade_equity!r})"
            )
        if np.isfinite(valuation_mark):
            self.last_prices_arr[gcol] = valuation_mark
        self.cash -= quantity * fill_price
        fee = fee_bps / 1e4 * abs(quantity) * fill_price
        self.cash -= fee
        prior_units = float(self.units_arr[gcol])
        self.units_arr[gcol] += quantity
        if self.units_arr[gcol] != prior_units:
            self._exit_clock.inventory_changed(gcol)
        self.fill_bar_ns.append(int(frame.grid_ns[bar_pos]))
        self.fill_gcol.append(int(gcol))
        self._mirror_pending.append(
            (int(frame.grid_ns[bar_pos]), int(gcol), float(quantity), float(fill_price), float(fee_bps))
        )
        submit_time = pd.Timestamp(int(self._w_avail_ns[submit_pos]), unit="ns", tz="UTC")
        self.fill_ts.append(fill_time)
        self.fill_symbol.append(symbol)
        self.fill_qty.append(quantity)
        self.fill_post_units.append(float(self.units_arr[gcol]))
        self.fill_price.append(fill_price)
        self.fill_fee_bps.append(fee_bps)
        self.fill_reason.append(reason)
        self.fill_pre_trade_equity.append(pre_trade_equity)
        _fill_terms.book_fill_quote_volume(
            self.fill_bar_qv, frame.local_cols, self._w_qv, bar_pos, symbol, self._w_volume_symbols)
        self.fill_times.append(fill_time)
        self.submit_times.append(submit_time)
        if self.retain_event_snapshots:
            marks_row = np.full(self.n_cols, np.nan, dtype="float64")
            marks_row[frame.gpos] = frame.marks_values[bar_pos]
            self.units_after_events.append((fill_time, self.units_arr.copy()))
            self.notional_after_events.append((fill_time, self.units_arr * marks_row))

    def _book_delist_settlement(self, frame: _WindowFrame, gcol: int, sym: str, spos: int, units: float, price: float, *, fee_bps: float = 0.0, event_id: str | None = None) -> None:
        """Book a causal delisting settlement.

        Books the validated event price, fee, and identity into the fill track,
        accounting mirror, and ledger identically. The recorded ``pre_trade_equity``
        is the equity observable immediately before the settlement.
        """
        pre_trade_equity = self._equity_at(frame.gpos)
        self._book_fill(
            frame, gcol=gcol, symbol=sym, bar_pos=spos, submit_pos=spos,
            quantity=-units, fill_price=price, fee_bps=float(fee_bps),
            reason="delist_settlement", valuation_mark=price, pre_trade_equity=pre_trade_equity,
            target_weight=0.0, decision_price=price,
        )
        self.termination_counts["DELIST_SETTLEMENT"] = self.termination_counts.get("DELIST_SETTLEMENT", 0) + 1

    def _advance_liquidity_carry(self, frame: _WindowFrame) -> None:
        """Carry last-liquid timestamps forward to the next window (causal)."""
        if len(frame.decision_ns_all) == 0 or len(frame.gpos) == 0:
            return
        rows = int(np.searchsorted(self._w_avail_ns, int(frame.decision_ns_all[-1]), side="right"))
        if rows <= 0:
            return
        idx = self._w_last_liquid_idx[rows - 1]
        cand = np.where(idx >= 0, frame.grid_ns[np.maximum(idx, 0)], -1).astype("int64")
        # Carry forward only bars available up to the final decision to prevent future leakage.
        self.last_liquid_ns[frame.gpos] = np.maximum(self.last_liquid_ns[frame.gpos], cand)

    def _consume_decision_price(self, frame: _WindowFrame, col: int, on_grid: bool, dpos: int, spos: int) -> float | None:
        """Resolve the anchor price for one intent, carried closes included."""
        if on_grid and frame.mark_valid[dpos, col]:
            return float(frame.marks_values[dpos, col])
        j = int(frame.last_close_idx[spos - 1, col]) if spos > 0 else -1
        if j >= 0 and frame.mark_valid[j, col]:
            return float(frame.marks_values[j, col])
        sym = frame.local_cols[col]
        carried_ts = self.last_close_ts.get(sym)
        if carried_ts is not None:
            carried_mark = self.last_close_mark[sym]
            if np.isfinite(carried_mark) and carried_mark > 0.0:
                return float(carried_mark)
        if spos < frame.n_grid and frame.mark_valid[spos, col]:
            return float(frame.marks_values[spos, col])
        return None


    def _consume_drift_trims(self, frame: _WindowFrame, until_ns: int | None) -> None:
        """Run due intraday single-name drift checks before a decision or the ledger."""
        cap = self.spec.name_drift_trim_max_weight
        if cap is None:
            return
        if self._trim_anchor_ns is None:
            if len(frame.decision_ns_all) == 0:
                return
            # Anchor to initial decision time without mutation.
            self._trim_anchor_ns = int(frame.decision_ns_all[0])
        anchor = int(self._trim_anchor_ns)
        step = int(self.spec.name_drift_trim_interval_hours) * 3_600_000_000_000
        # Enforce monotonic floor to prevent redundant drift checks in overlapping windows.
        floor = anchor if self.last_time_ns is None else max(anchor, int(self.last_time_ns))
        k = (floor - anchor) // step + 1
        bar_ns = int(frame.grid_ns[1] - frame.grid_ns[0])
        grid_end = int(frame.grid_ns[-1])
        while True:
            dpos = int(np.searchsorted(frame.grid_ns, anchor + k * step, side="left"))
            k += 1
            if dpos >= frame.n_grid:
                return
            # Snap off-grid checks to next bar without skipping.
            check_ns = int(frame.grid_ns[dpos])
            resolve_ns = check_ns + bar_ns + self.timeout_ns_delta
            # Defer checks that overlap with decision orders to decision rebalance.
            if resolve_ns > grid_end or (until_ns is not None and resolve_ns >= until_ns):
                return
            self._consume_drift_trim_at(frame, check_ns, dpos, cap)

    def _consume_drift_trim_at(self, frame: _WindowFrame, check_ns: int, dpos: int, cap: float) -> None:
        """Trim names breached above the cap at one intraday check via taker fills."""
        self._settle_events_through(frame, int(check_ns))
        self._advance_window(frame, check_ns, dpos, True)
        equity = self._equity_at(frame.gpos)
        prices = self.last_prices_arr[frame.gpos]
        units = self.units_arr[frame.gpos]
        weights = np.where(np.isfinite(prices), units * prices, 0.0) / equity
        trigger = cap * (1.0 + self.spec.one_way_taker_bps() / 1e4)
        # Buffer prevents immediate re-trim due to one-way taker fee drag.
        over = (np.abs(units) >= QTY_EPS) & (np.abs(weights) > trigger)
        if not bool(over.any()):
            return
        row = np.full(len(frame.local_cols), np.nan, dtype="float64")
        row[over] = np.sign(weights[over]) * cap
        stamp = pd.DatetimeIndex([pd.Timestamp(check_ns, unit="ns", tz="UTC")])
        spos = int(np.searchsorted(frame.grid_ns, check_ns, side="right"))
        for col in np.flatnonzero(over).tolist():
            n_before = len(self.fill_reason)
            self._consume_single_fill(frame, 0, col, row, True, dpos, spos, equity, stamp, stamp, apply_lifecycle=False)
            if len(self.fill_reason) > n_before:
                for j in range(n_before, len(self.fill_reason)):
                    self.fill_reason[j] = "drift_trim"
                self.termination_counts["DRIFT_TRIM"] = self.termination_counts.get("DRIFT_TRIM", 0) + 1

    def _consume_single_intent(self, frame: _WindowFrame, i: int) -> None:
        """Process one decision index across its active symbols."""
        dns = int(frame.decision_ns_all[i])
        dpos = int(frame.dpos_all[i])
        on_grid = bool(frame.on_grid_all[i])
        self._settle_events_through(frame, int(dns))
        self._advance_window(frame, dns, dpos, on_grid)
        equity = self._equity_at(frame.gpos)
        last_ledger_equity: float | None = None
        if self.equity_chunks:
            last_ledger_equity = float(self.equity_chunks[-1][-1])
        guard_equity = _contracts.ruin_guard_equity(equity, last_ledger_equity)
        row = frame.target_values[i]
        if self.min_equity_fraction is not None and guard_equity <= self.min_equity_fraction * self.initial_equity:
            if not self.equity_floor_breaches or self.equity_floor_breaches[-1] != frame.tw_index[i]:
                self.equity_floor_breaches.append(frame.tw_index[i])
            row = np.zeros_like(row)
        spos = int(frame.spos_all[i])
        row = _forced_exit_nan_override(
            row, frame, self.units_arr, self._admitted_by_symbol,
            information_ns=int(frame.sig_index[i].value),
        )
        active = np.where(np.isfinite(row) & ((row != 0.0) | (self.units_arr[frame.gpos] != 0.0)))[0]
        for col in active.tolist():
            if self._consume_single_fill(frame, i, col, row, on_grid, dpos, spos, equity, frame.tw_index, frame.sig_index):
                continue


    def _apply_lifecycle_policy(
        self, frame: _WindowFrame, i: int, gcol: int, sym: str,
        desired_units: float, sig_index: pd.DatetimeIndex,
    ) -> float:
        """Clamp one intent's desired units to the announced-delisting policy.

        Columns without admitted events are bit-identical to today; the policy
        reads the admitted event index with no registry access in the hot path.
        """
        from src.core.params import DELIST_FORCED_EXIT_LEAD
        from src.engine.execution.lifecycle import lifecycle_desired_units

        event = self._admitted_by_symbol.get(sym)
        if event is None:
            return float(desired_units)
        effective, action = lifecycle_desired_units(
            event, information_ns=int(sig_index[i].value),
            current_units=float(self.units_arr[gcol]),
            desired_units=float(desired_units),
            forced_exit_lead_ns=int(DELIST_FORCED_EXIT_LEAD.value),
        )
        if action != "unchanged":
            key = f"DELISTING_{action.upper()}"
            self.termination_counts[key] = self.termination_counts.get(key, 0) + 1
        return float(effective)

    def _delivery_submit_cancelled(self, frame: _WindowFrame, gcol: int, spos: int) -> bool:
        """Whether an order submitted at ``spos`` is dead on arrival at delivery."""
        last_trade = int(self._last_trade_ns[gcol])
        if last_trade == _INT64_MAX:
            return False
        if int(spos) >= int(frame.n_grid):
            return int(frame.grid_ns[-1]) >= last_trade
        return int(frame.grid_ns[int(spos)]) >= last_trade

    def _delivery_cap_pos(self, frame: _WindowFrame, gcol: int, timeout_pos: int) -> int:
        """Cap an order window so no fill books on a bar at/after ``last_trade_at``."""
        last_trade = int(self._last_trade_ns[gcol])
        if last_trade == _INT64_MAX:
            return int(timeout_pos)
        cutoff = int(np.searchsorted(frame.grid_ns, np.int64(last_trade), side="left"))
        return min(int(timeout_pos), int(cutoff))

    def _record_delivery_cancel(self) -> None:
        self.termination_counts["CANCELLED_AT_DELIVERY"] = self.termination_counts.get("CANCELLED_AT_DELIVERY", 0) + 1
        self.unfilled_count += 1

    def _consume_single_fill(self, frame: _WindowFrame, i: int, col: int, row: np.ndarray, on_grid: bool, dpos: int, spos: int, equity: float, tw_index: pd.DatetimeIndex, sig_index: pd.DatetimeIndex, *, apply_lifecycle: bool = True) -> bool:
        """Resolve and book one symbol intent; True advances to the next symbol."""
        gcol = int(frame.gpos[col])
        sym = frame.local_cols[col]
        weight = float(row[col])
        if self._delivery_submit_cancelled(frame, gcol, spos):
            self._record_delivery_cancel()
            return True
        decision_price = self._consume_decision_price(frame, col, on_grid and not frame.submit_anchored, dpos, spos)
        if decision_price is None:
            self.termination_counts["MISSING_DATA"] += 1
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_DECISION_MARK", symbol=sym, timestamp=tw_index[i],
                    decision_time=tw_index[i], signal_time=sig_index[i],
                    execution_bound=self.execution_bound,
                )
            )
            return True
        if not np.isfinite(self.last_prices_arr[gcol]):
            self.last_prices_arr[gcol] = decision_price
        desired_units = weight * equity / decision_price
        if apply_lifecycle:
            desired_units = self._apply_lifecycle_policy(
                frame, i, gcol, sym, desired_units, sig_index,
            )
        net_units = desired_units - self.units_arr[gcol]
        if abs(net_units) < 1e-12:
            return True
        side = 1 if net_units > 0 else -1
        self._probe_intent_notional(net_units, decision_price)
        if spos >= frame.n_grid:
            self.termination_counts["MISSING_DATA"] += 1
            return True
        submit_pos = spos
        timeout_ns = frame.grid_ns[spos] + self.timeout_ns_delta
        timeout_pos = int(np.searchsorted(frame.grid_ns, timeout_ns, side="left"))
        timeout_close = float("nan")
        adverse = np.array([], dtype="float64")
        if self.execution_bound == "OHLCV_IMMEDIATE_TAKER":
            fill_pos = submit_pos
            fill_price = float(frame.closes_values[fill_pos, col])
            if not np.isfinite(fill_price):
                self.termination_counts["MISSING_DATA"] += 1
                self.data_gaps.append(
                    ExecutionDataGap(
                        code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                        timestamp=frame.grid[fill_pos], decision_time=tw_index[i],
                        signal_time=sig_index[i], execution_bound=self.execution_bound,
                    )
                )
                return True
            taker_cost_bps = self._taker_cost_bps(gcol)
            fee_bps = self.spec.taker_fee_bps + taker_cost_bps
            reason = "timeout_taker"
        else:
            if self.execution_bound == "OHLCV_LADDERED_PROXY" and self._consume_fill_laddered(frame, i, col, gcol, sym, side, decision_price, net_units, weight, equity, spos, timeout_pos, timeout_ns, tw_index, sig_index):
                return True
            if self.execution_bound == "OHLCV_PEG_CHASE_PROXY" and self._consume_fill_peg_chase(frame, i, col, gcol, sym, side, decision_price, net_units, weight, equity, spos, timeout_pos, tw_index, sig_index):
                return True
            _proceed, fill_pos, fill_price, fee_bps, reason, timeout_close, adverse = self._consume_fill_strict_touch(frame, i, col, gcol, sym, side, decision_price, spos, timeout_pos, timeout_ns, tw_index, sig_index)
            if _proceed:
                return True
        block = self._bar_viability_gap(
            fill_pos=fill_pos, col=col, sym=sym,
            decision_ts=tw_index[i], signal_ts=sig_index[i],
            submit_ns=int(frame.grid_ns[submit_pos]), signal_ns=int(sig_index[i].value),
            avail_submit_ns=int(self._w_avail_ns[submit_pos]),
        )
        if block is not None:
            return self._block_fill(
                block, prior_units=float(self.units_arr[gcol]), net_units=float(net_units),
                frame=frame, col=col, fill_pos=int(fill_pos),
                timeout_pos=int(timeout_pos), submit_pos=int(submit_pos),
                side=int(side), decision_price=float(decision_price),
                equity=float(equity), weight=float(weight),
            )
        if reason == "passive_fill":
            self.fill_count += 1
        shortfall, term_price, fee_term, spread_term = _fill_terms.resolve_single_fill_terms(
            execution_bound=self.execution_bound,
            spec=self.spec,
            decision_price=decision_price,
            fill_price=fill_price,
            timeout_close=timeout_close,
            adverse=adverse,
            side=side,
            fee_bps=fee_bps,
            taker_cost_bps=self._taker_cost_bps(gcol),
            reason=reason,
        )
        self._record_terms(decision_price, term_price, side, fee_term, spread_term)
        self.shortfalls.append(shortfall)
        self.shortfall_notionals.append(abs(net_units) * fill_price)
        self._book_fill(
            frame, gcol=gcol, symbol=sym, bar_pos=fill_pos, submit_pos=submit_pos,
            quantity=net_units, fill_price=fill_price, fee_bps=fee_bps, reason=reason,
            valuation_mark=float(frame.marks_values[fill_pos, col]), pre_trade_equity=equity,
            target_weight=weight, decision_price=decision_price,
        )
        return False


    def _consume_fill_laddered(self, frame: _WindowFrame, i: int, col: int, gcol: int, sym: str, side: int, decision_price: float, net_units: float, weight: float, equity: float, spos: int, timeout_pos: int, timeout_ns: int, tw_index: pd.DatetimeIndex, sig_index: pd.DatetimeIndex) -> bool:
        """Book laddered tranches; always advances to the next symbol."""
        original_length = timeout_pos - spos
        capped_pos = self._delivery_cap_pos(frame, gcol, timeout_pos)
        taker_allowed = int(timeout_ns) < int(self._last_trade_ns[gcol])
        timeout_pos = capped_pos
        if timeout_pos <= spos:
            self.termination_counts["MISSING_DATA"] += 1
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                    timestamp=frame.grid[spos], decision_time=tw_index[i],
                    signal_time=sig_index[i], execution_bound=self.execution_bound,
                )
            )
            return True
        adverse = (
            frame.lows_values[spos:timeout_pos, col]
            if side == 1
            else frame.highs_values[spos:timeout_pos, col]
        )
        if not np.isfinite(adverse).all():
            self.termination_counts["MISSING_DATA"] += 1
            first_bad = spos + int(np.argmax(~np.isfinite(adverse)))
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                    timestamp=frame.grid[first_bad], decision_time=tw_index[i],
                    signal_time=sig_index[i], execution_bound=self.execution_bound,
                )
            )
            return True
        closes_window = frame.closes_values[spos:timeout_pos + int(taker_allowed), col]
        if not taker_allowed:
            closes_window = np.append(closes_window, closes_window[-1])
        if not np.isfinite(closes_window).all():
            self.termination_counts["MISSING_DATA"] += 1
            first_bad = spos + int(np.argmax(~np.isfinite(closes_window)))
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                    timestamp=frame.grid[first_bad], decision_time=tw_index[i],
                    signal_time=sig_index[i], execution_bound=self.execution_bound,
                )
            )
            return True
        ladder_entry_units = float(self.units_arr[gcol])
        if not taker_allowed:
            padding = original_length - len(adverse)
            adverse = np.pad(adverse, (0, padding), constant_values=np.finfo(float).max if side == 1 else 0.0)
            closes_window = np.pad(closes_window, (0, padding), mode="edge")
        for rel_pos, tranche_price, tranche_fee_bps, qty_fraction in _microstructure.laddered_fill_schedule(
            decision_price, side, adverse,
            closes_window,
            self.spec.ladder_tranches, self.spec, True,
        ):
            fill_pos = spos + rel_pos
            if rel_pos == len(adverse):
                if not taker_allowed:
                    self._record_delivery_cancel()
                    continue
                if timeout_pos >= frame.n_grid or frame.grid_ns[timeout_pos] != timeout_ns:
                    self.termination_counts["MISSING_DATA"] += 1
                    self.data_gaps.append(
                        ExecutionDataGap(
                            code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                            timestamp=frame.grid[spos], decision_time=tw_index[i],
                            signal_time=sig_index[i], execution_bound=self.execution_bound,
                        )
                    )
                    continue
                timeout_close = float(frame.closes_values[timeout_pos, col])
                if not np.isfinite(timeout_close):
                    self.termination_counts["MISSING_DATA"] += 1
                    self.data_gaps.append(
                        ExecutionDataGap(
                            code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                            timestamp=frame.grid[timeout_pos], decision_time=tw_index[i],
                            signal_time=sig_index[i], execution_bound=self.execution_bound,
                        )
                    )
                    continue
                self.unfilled_count += 1
                self.fallback_count += 1
                reason = "timeout_taker"
            else:
                self.fill_count += 1
                reason = "passive_fill"
            fill_price = float(tranche_price)
            fee_bps = float(tranche_fee_bps)
            qty = net_units * float(qty_fraction)
            block = self._bar_viability_gap(
                fill_pos=fill_pos, col=col, sym=sym,
                decision_ts=tw_index[i], signal_ts=sig_index[i],
                submit_ns=int(frame.grid_ns[spos]), signal_ns=int(sig_index[i].value),
                avail_submit_ns=int(self._w_avail_ns[spos]),
            )
            if block is not None:
                current_units = float(self.units_arr[gcol])
                return self._block_fill(
                    block, prior_units=current_units,
                    net_units=float(ladder_entry_units + net_units - current_units),
                    frame=frame, col=col, fill_pos=int(fill_pos),
                    timeout_pos=int(timeout_pos), submit_pos=int(spos),
                    side=int(side), decision_price=float(decision_price),
                    equity=float(equity), weight=float(weight),
                )
            if reason == "passive_fill":
                self._record_terms(
                    decision_price, fill_price, side, self.spec.maker_fee_bps, 0.0,
                )
            else:
                self._record_terms(
                    decision_price, fill_price, side,
                    self.spec.taker_fee_bps + self.spec.taker_slippage_bps, 0.0,
                )
            shortfall = (
                self.fee_terms[-1] + self.spread_terms[-1] + self.delay_terms[-1]
            )
            self.shortfalls.append(shortfall)
            self.shortfall_notionals.append(abs(qty) * fill_price)
            self._book_fill(
                frame, gcol=gcol, symbol=sym, bar_pos=fill_pos, submit_pos=spos,
                quantity=qty, fill_price=fill_price, fee_bps=fee_bps, reason=reason,
                valuation_mark=float(frame.marks_values[fill_pos, col]), pre_trade_equity=equity,
                target_weight=weight, decision_price=decision_price,
            )
        return True


    def _consume_fill_peg_chase(self, frame: _WindowFrame, i: int, col: int, gcol: int, sym: str, side: int, decision_price: float, net_units: float, weight: float, equity: float, spos: int, timeout_pos: int, tw_index: pd.DatetimeIndex, sig_index: pd.DatetimeIndex) -> bool:
        """Book the peg-chase schedule; always advances to the next symbol."""
        original_length = timeout_pos - spos
        capped_pos = self._delivery_cap_pos(frame, gcol, timeout_pos)
        delivery_truncated = capped_pos != timeout_pos
        timeout_pos = capped_pos
        if timeout_pos <= spos:
            self.termination_counts["MISSING_DATA"] += 1
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                    timestamp=frame.grid[spos], decision_time=tw_index[i],
                    signal_time=sig_index[i], execution_bound=self.execution_bound,
                )
            )
            return True
        adverse = (
            frame.lows_values[spos:timeout_pos, col]
            if side == 1
            else frame.highs_values[spos:timeout_pos, col]
        )
        if not np.isfinite(adverse).all():
            self.termination_counts["MISSING_DATA"] += 1
            first_bad = spos + int(np.argmax(~np.isfinite(adverse)))
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                    timestamp=frame.grid[first_bad], decision_time=tw_index[i],
                    signal_time=sig_index[i], execution_bound=self.execution_bound,
                )
            )
            return True
        closes_window = frame.closes_values[spos:timeout_pos, col]
        if not np.isfinite(closes_window).all():
            self.termination_counts["MISSING_DATA"] += 1
            first_bad = spos + int(np.argmax(~np.isfinite(closes_window)))
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                    timestamp=frame.grid[first_bad], decision_time=tw_index[i],
                    signal_time=sig_index[i], execution_bound=self.execution_bound,
                )
            )
            return True
        liquidity_cost_bps = self._taker_cost_bps(gcol)
        if delivery_truncated:
            # Keep the original tranche clock; future delivery cannot accelerate earlier pegs.
            padding = original_length - len(adverse)
            adverse = np.pad(adverse, (0, padding), constant_values=np.finfo(float).max if side == 1 else 0.0)
            closes_window = np.pad(closes_window, (0, padding), mode="edge")
        schedule = _microstructure.peg_chase_partial_schedule(
            decision_price, side, adverse, closes_window, self.spec,
            taker_cost_bps=self.spec.taker_fee_bps + liquidity_cost_bps,
        )
        if not schedule:
            # Residual: cash and units stay untouched (I3), so
            # the next decision recomputes net from the stale
            # position and carries the intent forward.
            self.residual_count += 1
            self.residual_notional += abs(net_units) * decision_price
            return True
        peg_entry_units = float(self.units_arr[gcol])
        for rel_pos, fill_price, fee_bps, qty_fraction, sched_reason in schedule:
            qty = net_units * qty_fraction
            fill_pos = spos + rel_pos
            reason = "passive_fill" if sched_reason == "maker_fill" else "timeout_taker"
            if delivery_truncated and fill_pos >= capped_pos:
                self._record_delivery_cancel()
                continue
            block = self._bar_viability_gap(
                fill_pos=fill_pos, col=col, sym=sym,
                decision_ts=tw_index[i], signal_ts=sig_index[i],
                submit_ns=int(frame.grid[spos].value), signal_ns=int(sig_index[i].value),
                avail_submit_ns=int(self._w_avail_ns[spos]),
            )
            if block is not None:
                current_units = float(self.units_arr[gcol])
                return self._block_fill(
                    block, prior_units=current_units,
                    net_units=float(peg_entry_units + net_units - current_units),
                    frame=frame, col=col, fill_pos=int(fill_pos),
                    timeout_pos=int(timeout_pos), submit_pos=int(spos),
                    side=int(side), decision_price=float(decision_price),
                    equity=float(equity), weight=float(weight),
                )
            if reason == "passive_fill":
                self.fill_count += 1
            else:
                # Backstop conversion mirrors the strict/touch
                # timeout convention: one unfilled intent that
                # completed via the taker fallback.
                self.unfilled_count += 1
                self.fallback_count += 1
            if reason == "passive_fill":
                self._record_terms(decision_price, fill_price, side, self.spec.maker_fee_bps, 0.0)
            elif self.spec.liquidity_cost_model == "corwin_schultz":
                self._record_terms(
                    decision_price, fill_price, side,
                    self.spec.taker_fee_bps, liquidity_cost_bps,
                )
            else:
                self._record_terms(
                    decision_price, fill_price, side,
                    self.spec.taker_fee_bps + liquidity_cost_bps, 0.0,
                )
            shortfall = (
                side * (fill_price / decision_price - 1.0) * 1e4
                + (
                    self.spec.maker_fee_bps
                    if sched_reason == "maker_fill"
                    else self.spec.taker_fee_bps + liquidity_cost_bps
                )
            )
            self.shortfalls.append(shortfall)
            self.shortfall_notionals.append(abs(qty) * fill_price)
            self._book_fill(
                frame, gcol=gcol, symbol=sym, bar_pos=fill_pos, submit_pos=spos,
                quantity=qty, fill_price=fill_price, fee_bps=fee_bps, reason=reason,
                valuation_mark=float(frame.marks_values[fill_pos, col]), pre_trade_equity=equity,
                target_weight=weight, decision_price=decision_price,
            )
        return True


    def _consume_fill_strict_touch(self, frame: _WindowFrame, i: int, col: int, gcol: int, sym: str, side: int, decision_price: float, spos: int, timeout_pos: int, timeout_ns: int, tw_index: pd.DatetimeIndex, sig_index: pd.DatetimeIndex) -> tuple[bool, int, float, float, str, float, np.ndarray]:
        """Resolve a strict/touch/timeout fill; False carries vars for booking."""
        timeout_close = float("nan")
        capped_pos = self._delivery_cap_pos(frame, gcol, timeout_pos)
        taker_allowed = int(timeout_ns) < int(self._last_trade_ns[gcol])
        timeout_pos = capped_pos
        if timeout_pos <= spos:
            self.termination_counts["MISSING_DATA"] += 1
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                    timestamp=frame.grid[spos], decision_time=tw_index[i],
                    signal_time=sig_index[i], execution_bound=self.execution_bound,
                )
            )
            return True, 0, float("nan"), 0.0, "", float("nan"), np.empty(0, dtype="float64")
        adverse = (
            frame.lows_values[spos:capped_pos, col]
            if side == 1
            else frame.highs_values[spos:capped_pos, col]
        )
        if not np.isfinite(adverse).all():
            self.termination_counts["MISSING_DATA"] += 1
            first_bad = spos + int(np.argmax(~np.isfinite(adverse)))
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                    timestamp=frame.grid[first_bad], decision_time=tw_index[i],
                    signal_time=sig_index[i], execution_bound=self.execution_bound,
                )
            )
            return True, 0, float("nan"), 0.0, "", float("nan"), np.empty(0, dtype="float64")
        if side == 1:
            crossed = (adverse < decision_price) if self.require_strict else (adverse <= decision_price)
        else:
            crossed = (adverse > decision_price) if self.require_strict else (adverse >= decision_price)
        if crossed.any():
            hit = int(np.argmax(crossed))
            fill_pos = spos + hit
            fill_price = decision_price
            fee_bps = self.spec.maker_fee_bps
            reason = "passive_fill"
        else:
            if not taker_allowed:
                self._record_delivery_cancel()
                return True, 0, float("nan"), 0.0, "", float("nan"), np.empty(0, dtype="float64")
            if timeout_pos >= frame.n_grid or frame.grid_ns[timeout_pos] != timeout_ns:
                self.termination_counts["MISSING_DATA"] += 1
                self.data_gaps.append(
                    ExecutionDataGap(
                        code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                        timestamp=frame.grid[spos], decision_time=tw_index[i],
                        signal_time=sig_index[i], execution_bound=self.execution_bound,
                    )
                )
                return True, 0, float("nan"), 0.0, "", float("nan"), np.empty(0, dtype="float64")
            timeout_close = float(frame.closes_values[timeout_pos, col])
            if not np.isfinite(timeout_close):
                self.termination_counts["MISSING_DATA"] += 1
                self.data_gaps.append(
                    ExecutionDataGap(
                        code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                        timestamp=frame.grid[timeout_pos], decision_time=tw_index[i],
                        signal_time=sig_index[i], execution_bound=self.execution_bound,
                    )
                )
                return True, 0, float("nan"), 0.0, "", float("nan"), np.empty(0, dtype="float64")
            self.unfilled_count += 1
            self.fallback_count += 1
            fill_pos = timeout_pos
            fill_price = timeout_close
            fee_bps = self.spec.taker_fee_bps + self._taker_cost_bps(gcol)
            reason = "timeout_taker"
        return False, fill_pos, fill_price, fee_bps, reason, timeout_close, adverse


    def _settlement_patched_marks(self, frame: _WindowFrame) -> np.ndarray:
        """Copy-on-write marks with booked settlement prices on their delivery bars."""
        marks = frame.marks_values
        if not self._settlement_overrides:
            return marks
        patched = np.array(marks, dtype="float64", copy=True)
        n_grid = int(frame.n_grid)
        gpos = frame.gpos
        for (opos, ogcol), oprice in self._settlement_overrides.items():
            hit = np.flatnonzero(gpos == int(ogcol))
            if len(hit) and 0 <= int(opos) < n_grid:
                patched[int(opos), int(hit[0])] = float(oprice)
        return patched

    def _record_ledger_funding_gap(self, symbol: str, stamp: pd.Timestamp) -> None:
        """Record one per-symbol held-funding gap, keeping the legacy scalar first."""
        if symbol not in self._held_funding_first:
            self._held_funding_first[symbol] = stamp
            if self.first_held_funding is None:
                self.first_held_funding = (symbol, stamp)

    def _consume_append_ledger(self, frame: _WindowFrame) -> None:
        """Append reconciled inventory accounting with bounded scratch storage.

        The first retained bar may use only the immediately preceding consumed,
        published valuation mark for continuity, because window splitting must
        not create fictitious missing data.
        Ledger-track inventory is the step function of intent-track post-fill levels anchored at the
        carried start level, so a position the intent track closed to exactly zero is exactly zero on
        the ledger track. Re-deriving levels by summing fill quantities changes float association and
        leaves sub-epsilon dust that the held-inventory checks would otherwise treat as a live position.

        Args:
            frame: Window staged by ``_consume_validate_window``.

        Returns:
            None; append chronological ledger series and preserve carried state.

        Raises:
            DataIntegrityError: Accounting or required provenance is invalid.
        """
        grid_ns = frame.grid_ns
        n_grid = frame.n_grid
        local_cols = frame.local_cols
        fill_start = frame.fill_start
        n_local = frame.n_local
        marks_values = self._settlement_patched_marks(frame)
        gpos = frame.gpos
        grid = frame.grid
        funding_matrix = frame.funding_matrix
        closes_values = frame.closes_values

        # ---- streamed ledger chunk over [ledger_start_ns, grid end] ----
        p0 = 0 if self.ledger_start_ns is None else int(np.searchsorted(grid_ns, self.ledger_start_ns, side="left"))
        if p0 >= n_grid:
            raise DataIntegrityError("execution windows must not leave an uncovered grid gap")
        chunk_len = n_grid - p0
        if chunk_len:
            n_fill = len(self.fill_ts) - fill_start
            fill = assemble_fill_flow(
                grid_ns, n_grid, local_cols, n_fill, fill_start, self.fill_bar_ns, self.fill_symbol,
                self.fill_qty, self.fill_price, self.fill_fee_bps, self.fill_post_units,
            )

            # Bounded per-symbol pass: each column's arithmetic matches the
            # legacy window-by-symbol plane, accumulated left to right in
            # canonical order. Scratch stays O(window bars).
            row_idx = np.arange(n_grid)
            mtm_arr = np.zeros(n_grid, dtype="float64")
            funding_arr = np.zeros(n_grid, dtype="float64")
            notional_arr = np.zeros(n_grid, dtype="float64")
            notional_before_arr = np.zeros(n_grid, dtype="float64")
            funding_chunk = self._funding.begin_chunk(grid, local_cols, p0)
            start_units = np.asarray(self.ledger_units[gpos], dtype="float64")
            start_valid = np.asarray(self.last_valid_mark[gpos], dtype="float64")
            end_units = np.empty(n_local, dtype="float64")
            end_valid = np.empty(n_local, dtype="float64")
            for j in range(n_local):
                sel = fill.columns == j
                if bool(sel.any()):
                    pos_j = fill.positions[sel]
                    post_j = fill.post_units[sel]
                    order = np.argsort(pos_j, kind="stable")
                    pos_s = pos_j[order]
                    post_s = post_j[order]
                    _, uniq_idx = np.unique(pos_s[::-1], return_index=True)
                    keep_rev = np.zeros(len(pos_s), dtype=bool)
                    keep_rev[uniq_idx] = True
                    keep = keep_rev[::-1]
                    pos_u = pos_s[keep]
                    post_u = post_s[keep]
                    tmp = np.full(n_grid, np.nan, dtype="float64")
                    tmp[pos_u] = post_u
                    last_pos = np.maximum.accumulate(np.where(np.isfinite(tmp), row_idx, -1))
                    units = np.where(last_pos >= 0, tmp[np.maximum(last_pos, 0)], float(start_units[j]))
                else:
                    units = np.full(n_grid, float(start_units[j]), dtype="float64")
                before = np.empty(n_grid, dtype="float64")
                before[0] = float(start_units[j])
                before[1:] = units[:-1]
                marks_col = marks_values[:, j].astype("float64", copy=True)
                finite = np.isfinite(marks_col)
                last_idx = np.maximum.accumulate(np.where(finite, row_idx, -1))
                ff_base = marks_col[np.maximum(last_idx, 0)]
                carry = float(start_valid[j])
                m_ff_col = np.where(finite, marks_col, np.where(last_idx >= 0, ff_base, carry))
                valuation_col = np.where(
                    finite | (units != 0.0),
                    np.where(finite, marks_col, m_ff_col),
                    0.0,
                )
                held = np.abs(before) >= QTY_EPS
                joint = np.zeros(n_grid, dtype=bool)
                joint[1:] = finite[1:] & finite[:-1]
                kept = row_idx >= p0
                boundary_continuous = False
                boundary_carry = float("nan")
                if chunk_len and p0 == 0 and self.ledger_start_ns is not None:
                    gcol = int(gpos[j])
                    prior_mark = float(start_valid[j])
                    prior_avail = int(self._last_valid_mark_avail_ns[gcol])
                    prior_bar_ns = int(self.ledger_start_ns) - int(frame.bar_ns)
                    curr_mark = float(marks_col[0])
                    curr_avail = int(self._w_mark_avail[0, j])
                    curr_ok = bool(
                        np.isfinite(curr_mark) and curr_mark > 0.0 and curr_avail <= int(grid_ns[0])
                    )
                    prior_ok = bool(
                        np.isfinite(prior_mark)
                        and prior_mark > 0.0
                        and prior_avail >= 0
                        and prior_avail <= prior_bar_ns
                    )
                    if bool(abs(float(before[0])) >= QTY_EPS) and curr_ok and prior_ok:
                        boundary_continuous = True
                        boundary_carry = prior_mark
                gap_mask = (held & ~joint) & kept
                if boundary_continuous:
                    gap_mask = gap_mask.copy()
                    gap_mask[0] = False
                if bool(gap_mask.any()):
                    self.ledger_valid = False
                    self.invalid_reasons.add("MISSING_DATA")
                    if self.first_held_mark is None:
                        first_pos = int(np.flatnonzero(gap_mask)[0])
                        self.first_held_mark = (local_cols[j], grid[first_pos])
                mtm_col = np.zeros(n_grid, dtype="float64")
                mtm_col[1:] = np.where(
                    joint[1:], before[1:] * (marks_col[1:] - marks_col[:-1]), 0.0,
                )
                if boundary_continuous:
                    mtm_col[0] = float(before[0]) * (float(marks_col[0]) - float(boundary_carry))
                mtm_arr += mtm_col
                fknown_col = self._w_fknown[:, j]
                funding_col = funding_matrix[:, j]
                usable = finite & fknown_col
                charged_col = np.where(usable, funding_col * before * marks_col, 0.0)
                funding_arr += charged_col
                funding_chunk.add_column(j, charged_col)
                unpriceable = ~finite & (funding_col != 0.0)
                funding_gap = (held & (~fknown_col | unpriceable)) & kept
                if bool(funding_gap.any()):
                    self.ledger_valid = False
                    self.invalid_reasons.add("MISSING_DATA")
                    bad = np.flatnonzero(funding_gap)
                    self._record_ledger_funding_gap(local_cols[j], grid[int(bad[0])])
                notional_arr += units * valuation_col
                notional_before_arr += before * valuation_col
                end_units[j] = float(units[-1])
                end_valid[j] = float(m_ff_col[-1])
            if n_local:
                self.ledger_units[gpos] = end_units
                self.last_valid_mark[gpos] = end_valid
                for j in range(n_local):
                    gcol = int(gpos[j])
                    marks_col_tail = marks_values[:, j].astype("float64", copy=False)
                    tail_finite = np.isfinite(marks_col_tail)
                    if bool(tail_finite[-1]):
                        self._last_valid_mark_avail_ns[gcol] = int(self._w_mark_avail[-1, j])
                    else:
                        hit = np.flatnonzero(tail_finite)
                        if len(hit):
                            self._last_valid_mark_avail_ns[gcol] = int(
                                self._w_mark_avail[int(hit[-1]), j]
                            )
                        elif not np.isfinite(float(end_valid[j])):
                            self._last_valid_mark_avail_ns[gcol] = -1

            # The cash cumsum starts at the chunk's first bar (p0): positions
            # [0, p0) belong to the previous chunk's ledger and must not be
            # re-accumulated from the carried cash.
            chunk_flow = fill.flow[p0:] - funding_arr[p0:]
            cash_after = self.ledger_cash + np.cumsum(chunk_flow)
            cash_pre_fill = np.empty(chunk_len, dtype="float64")
            cash_pre_fill[0] = self.ledger_cash - funding_arr[p0]
            cash_pre_fill[1:] = cash_after[:-1] - funding_arr[p0 + 1 :]
            equity_arr = cash_after + notional_arr[p0:]
            turnover_arr = np.zeros(chunk_len, dtype="float64")
            if len(fill.positions):
                pre_trade_equity = (
                    cash_pre_fill[fill.positions - p0] + notional_before_arr[fill.positions]
                )
                if not np.isfinite(pre_trade_equity).all() or (pre_trade_equity <= 0).any():
                    bad = np.where(~np.isfinite(pre_trade_equity) | (pre_trade_equity <= 0))[0]
                    bad_pos = fill.positions[bad[0]]
                    raise DataIntegrityError(
                        f"pre-trade equity must be positive and finite "
                        f"(ts={grid[bad_pos]!r} pre_trade_equity={pre_trade_equity[bad[0]]!r})"
                    )
                np.add.at(
                    turnover_arr, fill.positions - p0,
                    np.abs(fill.quantities * fill.prices) / pre_trade_equity,
                )
            if not np.isfinite(equity_arr).all() or (equity_arr <= 0).any():
                raise DataIntegrityError("simulated inventory equity must be finite and strictly positive")
            self.equity_chunks.append(equity_arr)
            self.equity_times.append(grid[p0:])
            self.mtm_chunks.append(mtm_arr[p0:])
            self.funding_chunks.append(funding_arr[p0:])
            self.fee_chunks.append(fill.fee_by_ts[p0:])
            self.turnover_chunks.append(turnover_arr)
            self._funding.commit_chunk(funding_chunk)
            self.ledger_cash = float(cash_after[-1])
            if self._w_avail_explicit:
                self._ledger_avail_chunks.append(np.asarray(self._w_avail_ns[p0:], dtype="int64"))
            else:
                self._ledger_avail_complete = False
        if len(gpos):
            self._final_mark[gpos] = np.asarray(marks_values[-1], dtype="float64")
            self._final_mark_avail_ns[gpos] = np.asarray(self._w_mark_avail[-1], dtype="int64")
        self._settle_mirror_window(frame, p0)
        self._settlement_overrides.clear()
        # Carried last-finite-close provenance advances only over bars this
        # window actually consumed: a decision can never see the window tail
        # (F2), only strictly earlier closes from its own or prior windows.
        close_finite_tail = np.isfinite(closes_values)
        last_idx = np.where(close_finite_tail, np.arange(n_grid)[:, None], -1).max(axis=0)
        for j in np.flatnonzero(last_idx >= 0).tolist():
            sym = local_cols[j]
            pos = int(last_idx[j])
            ts = grid[pos]
            prev_ts = self.last_close_ts.get(sym)
            if prev_ts is None or ts > prev_ts:
                self.last_close_ts[sym] = ts
                self.last_close_value[sym] = float(closes_values[pos, j])
                self.last_close_mark[sym] = float(marks_values[pos, j])
        self.ledger_start_ns = int(grid_ns[-1]) + frame.bar_ns


    def _settle_mirror_window(self, frame: _WindowFrame, p0: int) -> None:
        """Replay the kept bars through the causal mirror in timestamp order.

        Each bar settles mark then known funding on pre-fill inventory before
        that bar's queued fills are applied (INV-EVENT-ORDER), so the mirror
        reproduces the independent ledger's economics exactly. Fills queued
        before the kept region (window-overlap backlog) join inventory
        without cash flow, mirroring the ledger chunk handoff. Unknown
        funding over held inventory records MISSING_HELD_FUNDING and
        invalidates instead of settling an invented cost. Arithmetic lives in
        ``CausalPortfolioState.settle_window``.
        """
        pending = self._mirror_pending
        self._mirror_pending = []
        marks = self._settlement_patched_marks(frame)[p0:]
        settlement = self.accounting_state.settle_window(
            bar_ns=frame.grid_ns[p0:],
            columns=frame.gpos,
            marks=marks,
            funding_rates=frame.funding_matrix[p0:],
            funding_known=self._w_fknown[p0:],
            fill_ns=np.fromiter((e[0] for e in pending), dtype="int64", count=len(pending)),
            fill_columns=np.fromiter((e[1] for e in pending), dtype=np.intp, count=len(pending)),
            fill_quantities=np.fromiter((e[2] for e in pending), dtype="float64", count=len(pending)),
            fill_prices=np.fromiter((e[3] for e in pending), dtype="float64", count=len(pending)),
            fill_fee_bps=np.fromiter((e[4] for e in pending), dtype="float64", count=len(pending)),
        )
        if settlement.gap_bar_offsets.size:
            self.ledger_valid = False
            self.invalid_reasons.add("MISSING_DATA")
            stamps = frame.grid[p0 + settlement.gap_bar_offsets]
            self.data_gaps.extend(
                ExecutionDataGap(
                    code="MISSING_HELD_FUNDING", symbol=self.columns[int(c)], timestamp=ts,
                    execution_bound=self.execution_bound,
                )
                for ts, c in zip(stamps, settlement.gap_witness_columns.tolist(), strict=True)
            )

    def _consume_update_spreads(self, frame: _WindowFrame) -> None:
        """Roll the liquidity-aware spread estimate forward, causally."""

        # Liquidity-aware spread EWMA update -- strictly AFTER this window's
        # fills were priced, so a window's own bars can never price its own
        # costs (causality). A degenerate (nan) estimate carries the prior
        # value forward instead of poisoning it.
        if self.spec.liquidity_cost_model == "corwin_schultz":
            est = _microstructure.corwin_schultz_half_spread_bps(frame.highs_values, frame.lows_values)
            old = self.half_spread_bps[frame.gpos]
            alpha = self.spec.spread_ewma_alpha
            updated = alpha * est + (1.0 - alpha) * old
            merged = np.where(np.isnan(est), old, updated)
            self.half_spread_bps[frame.gpos] = np.where(np.isnan(old), est, merged)

    def _advance_spread_clock(self, logical_partition: tuple[int, int] | None) -> None:
        """Settle the previous logical partition before pricing this window. Physical IO boundaries do not reset inventory, liquidity or the logical spread clock. Overlap observations and settlements are applied once.

        Untagged windows keep the legacy per-window update. Tagged windows
        sharing one partition key accumulate observations without updating;
        the pending partition settles exactly once when the key advances, so
        fills are always priced with estimates from strictly earlier
        observations and an IO split alone never triggers an update.
        """
        if self.spec.liquidity_cost_model != "corwin_schultz":
            return
        if logical_partition is None:
            return
        if self._spread_pending_key is not None and self._spread_pending_key != logical_partition:
            self._finalize_spread_partition()

    def _observe_spread_partition(self, frame: _WindowFrame, logical_partition: tuple[int, int] | None) -> None:
        """Record this window's bars into the logical cost-clock observations.

        Untagged windows keep the legacy immediate per-window update.
        """
        if self.spec.liquidity_cost_model != "corwin_schultz":
            return
        if logical_partition is None:
            self._consume_update_spreads(frame)
            return
        gpos = frame.gpos
        carry_high = self._spread_carry_high[gpos]
        carry_low = self._spread_carry_low[gpos]
        carry_ns = self._spread_carry_ns[gpos]
        adjacent = (carry_ns + frame.bar_ns == frame.grid_ns[0]) & np.isfinite(carry_high) & np.isfinite(carry_low)
        ext_high = np.vstack([carry_high[None, :], frame.highs_values])
        ext_low = np.vstack([carry_low[None, :], frame.lows_values])
        sums, counts, bars = _microstructure._corwin_schultz_pair_sums_counts(
            ext_high, ext_low, first_pair_allowed=adjacent,
        )
        carry_valid = (
            np.isfinite(carry_high)
            & np.isfinite(carry_low)
            & (carry_high > 0.0)
            & (carry_low > 0.0)
            & (carry_high >= carry_low)
        )
        self._spread_pair_sums[gpos] += sums
        self._spread_pair_counts[gpos] += counts
        self._spread_bar_counts[gpos] += bars - carry_valid.astype("float64")
        self._spread_carry_high[gpos] = frame.highs_values[-1]
        self._spread_carry_low[gpos] = frame.lows_values[-1]
        self._spread_carry_ns[gpos] = frame.grid_ns[-1]
        self._spread_pending_key = logical_partition

    def _finalize_spread_partition(self) -> None:
        """Settle the pending logical partition into the EWMA, once per partition."""
        if self.spec.liquidity_cost_model != "corwin_schultz":
            self._spread_pending_key = None
            return
        if self._spread_pending_key is None:
            return
        self._spread_pending_key = None
        est = _microstructure._corwin_schultz_combine_half_spread_bps(
            self._spread_pair_sums, self._spread_pair_counts, self._spread_bar_counts,
        )
        old = self.half_spread_bps
        alpha = self.spec.spread_ewma_alpha
        updated = alpha * est + (1.0 - alpha) * old
        merged = np.where(np.isnan(est), old, updated)
        self.half_spread_bps = np.where(np.isnan(old), est, merged)
        self._spread_pair_sums[:] = 0.0
        self._spread_pair_counts[:] = 0.0
        self._spread_bar_counts[:] = 0.0


    def _terminal_positions(self, grid_end: pd.Timestamp) -> tuple[TerminalPositionEvidence, ...]:
        """Classify each held position without converting the cutoff into a fictional exit.

        An ordinary cutoff with a valid fresh published mark and complete held
        financing produces ``open_marked`` and preserves units and cash. A
        missing or stale held mark produces unresolved valuation and invalid
        accounting. Missing held funding remains invalid regardless of later
        fills, their absence, or any inferred end-of-life classification.
        """
        cutoff_ns = int(grid_end.value)
        funded_incomplete = {
            g.symbol for g in self.data_gaps if g.code == "MISSING_HELD_FUNDING"
        }
        funded_incomplete |= set(self._held_funding_first)
        if self.first_held_funding is not None:
            funded_incomplete.add(self.first_held_funding[0])
        booked_ids = {e.event_id for e in self._booked_settlement_events}
        records: list[TerminalPositionEvidence] = []
        for col in range(self.n_cols):
            quantity = float(self.units_arr[col])
            if abs(quantity) < 1e-12:
                continue
            sym = self.columns[col]
            funding_complete = sym not in funded_incomplete
            unsettled = any(
                e.symbol == sym
                and int((e.last_trade_at if e.last_trade_at is not None else e.effective_at).value) <= cutoff_ns
                and e.event_id not in booked_ids
                for e in self._admitted_events.values()
            )
            final_mark = float(self._final_mark[col])
            final_avail_ns = int(self._final_mark_avail_ns[col])
            fresh = (
                np.isfinite(final_mark)
                and final_mark > 0.0
                and final_avail_ns >= 0
                and final_avail_ns <= cutoff_ns
            )
            if fresh and funding_complete and not unsettled:
                mark: float | None = final_mark
                mark_available_at = pd.Timestamp(final_avail_ns, unit="ns", tz="UTC")
                records.append(
                    TerminalPositionEvidence(
                        symbol=sym, quantity=quantity, cutoff=grid_end,
                        status="open_marked", mark=mark,
                        mark_available_at=mark_available_at,
                        funding_complete=True, reason_codes=("FRESH_MARK", "FUNDING_COMPLETE"),
                    )
                )
                continue
            codes = list(("FRESH_MARK",) if fresh else ("STALE_MARK",))
            if not funding_complete:
                codes.append("MISSING_HELD_FUNDING")
            if unsettled:
                codes.append("UNSETTLED_DELIVERY")
            self.ledger_valid = False
            self.invalid_reasons.add("MISSING_DATA")
            if unsettled:
                self.data_gaps.append(
                    ExecutionDataGap(
                        code="UNSETTLED_DELIVERY", symbol=sym, timestamp=grid_end,
                        execution_bound=self.execution_bound,
                    )
                )
            if not fresh:
                self.data_gaps.append(
                    ExecutionDataGap(
                        code="MISSING_HELD_MARK", symbol=sym, timestamp=grid_end,
                        execution_bound=self.execution_bound,
                    )
                )
                codes.append("MISSING_HELD_MARK")
            stale_mark: float | None = final_mark if np.isfinite(final_mark) and final_mark > 0.0 else None
            stale_avail = (
                pd.Timestamp(final_avail_ns, unit="ns", tz="UTC")
                if stale_mark is not None and final_avail_ns >= 0
                else None
            )
            records.append(
                TerminalPositionEvidence(
                    symbol=sym, quantity=quantity, cutoff=grid_end,
                    status="unresolved", mark=stale_mark,
                    mark_available_at=stale_avail,
                    funding_complete=funding_complete, reason_codes=tuple(codes),
                )
            )
        for event in self._booked_settlement_events:
            settled_mark = self._settlement_prices[event.event_id]
            records.append(
                TerminalPositionEvidence(
                    symbol=event.symbol, quantity=0.0, cutoff=grid_end,
                    status="settled", mark=settled_mark,
                    mark_available_at=event.available_at,
                    funding_complete=event.symbol not in funded_incomplete,
                    reason_codes=("SETTLEMENT_EVENT",),
                )
            )
        records.sort(key=lambda r: r.symbol)
        return tuple(records)

    def finalize(self) -> StrategyExecutionReplayResult:
        """Finalize observed equity without converting the cutoff into a fictional exit.

        Returns:
            Reconciled replay evidence with priced or unresolved terminal positions.
            Open priced positions remain on the book; held financial gaps remain invalid.
        Raises:
            DataIntegrityError: Terminal observations or settlement state are inconsistent.
        """
        columns = self.columns
        self._finalize_spread_partition()

        forced_exit_count = 0
        forced_exit_notional = 0.0
        grid_end = self.full_grid_end
        elapsed_seconds = time.perf_counter() - self._t0

        simulated_fills = pd.DataFrame(
            {
                "timestamp": self.fill_ts,
                "symbol": self.fill_symbol,
                "quantity_delta": self.fill_qty,
                "fill_price": self.fill_price,
                "fee_bps": self.fill_fee_bps,
                "reason": self.fill_reason,
                "pre_trade_equity": self.fill_pre_trade_equity,
                "bar_quote_volume": self.fill_bar_qv,
            }
        )[
            [
                "timestamp", "symbol", "quantity_delta", "fill_price",
                "fee_bps", "reason", "pre_trade_equity", "bar_quote_volume",
            ]
        ]
        if simulated_fills.empty:
            simulated_fills = simulated_fills.astype(
                {"quantity_delta": "float64", "fill_price": "float64",
                 "fee_bps": "float64", "bar_quote_volume": "float64"}
            )

        ledger_available_at: pd.DatetimeIndex | None = None
        if self.equity_chunks:
            full_index = self.equity_times[0].append(self.equity_times[1:]) if len(self.equity_times) > 1 else self.equity_times[0]
            equity_values_arr = np.concatenate(self.equity_chunks)
            mtm_arr = np.concatenate(self.mtm_chunks)
            funding_arr = np.concatenate(self.funding_chunks)
            fee_arr = np.concatenate(self.fee_chunks)
            turnover_arr = np.concatenate(self.turnover_chunks)
            if self._ledger_avail_complete and self._ledger_avail_chunks:
                avail_values = np.concatenate(self._ledger_avail_chunks).astype("datetime64[ns]")
                ledger_available_at = pd.DatetimeIndex(avail_values, tz="UTC")
            del self.equity_chunks, self.equity_times, self.mtm_chunks, self.funding_chunks, self.fee_chunks, self.turnover_chunks
        else:
            full_index = self.first_grid
            equity_values_arr = np.array([], dtype="float64")
            mtm_arr = np.array([], dtype="float64")
            funding_arr = np.array([], dtype="float64")
            fee_arr = np.array([], dtype="float64")
            turnover_arr = np.array([], dtype="float64")
        equity = pd.Series(equity_values_arr, index=full_index, dtype="float64")
        if not np.isfinite(equity_values_arr).all() or (equity_values_arr <= 0).any():
            raise DataIntegrityError("simulated inventory equity must be finite and strictly positive")
        funding_daily = self._funding.daily_frame(list(columns))
        ledger = SimulatedInventoryLedgerResult(
            equity=equity,
            net_returns=equity.pct_change().dropna(),
            simulated_units=None,
            mark_to_market_pnl=pd.Series(mtm_arr, index=full_index, dtype="float64"),
            funding_charge=pd.Series(funding_arr, index=full_index, dtype="float64"),
            fee_charge=pd.Series(fee_arr, index=full_index, dtype="float64"),
            fill_turnover=pd.Series(turnover_arr, index=full_index, dtype="float64"),
            fill_source=self.execution_bound,
            mark_source=self.mark_source,
            primary_valid=self.ledger_valid,
            invalid_reasons=tuple(sorted(self.invalid_reasons)),
            equity_floor_breached_at=tuple(self.equity_floor_breaches),
            funding_by_symbol=self._funding.totals(),
            funding_by_symbol_daily=funding_daily,
        )
        if self.first_held_mark is not None:
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_HELD_MARK", symbol=self.first_held_mark[0],
                    timestamp=self.first_held_mark[1], execution_bound=self.execution_bound,
                )
            )
        existing_funding = {gap.symbol for gap in self.data_gaps if gap.code == "MISSING_HELD_FUNDING"}
        for sym, stamp in sorted(self._held_funding_first.items()):
            if sym not in existing_funding:
                self.data_gaps.append(ExecutionDataGap(code="MISSING_HELD_FUNDING", symbol=sym, timestamp=stamp, execution_bound=self.execution_bound))
        terminal_positions = self._terminal_positions(grid_end)
        self.data_gaps.sort(key=lambda g: (g.timestamp, g.code, g.symbol))
        ledger = dataclasses.replace(
            ledger,
            primary_valid=self.ledger_valid,
            invalid_reasons=tuple(sorted(self.invalid_reasons)),
            data_gaps=tuple(self.data_gaps),
            funding_by_symbol=self._funding.totals(),
            funding_by_symbol_daily=ledger.funding_by_symbol_daily,
        )

        if self.units_after_events:
            events_index = pd.DatetimeIndex([t for t, _ in self.units_after_events])
            simulated_units = pd.DataFrame(
                [row for _t, row in self.units_after_events], index=events_index, columns=list(columns),
            )
            notional_events_index = pd.DatetimeIndex([t for t, _ in self.notional_after_events])
            simulated_notional_weights = pd.DataFrame(
                [row for _t, row in self.notional_after_events],
                index=notional_events_index,
                columns=list(columns),
            )
        else:
            simulated_units = pd.DataFrame(columns=list(columns))
            simulated_notional_weights = pd.DataFrame(columns=list(columns))

        all_intent_shortfall_bps = (
            float(np.mean(self.shortfalls)) if self.shortfalls else float("nan")
        )
        weighted_shortfall_bps = _microstructure.notional_weighted_shortfall_bps(
            self.shortfalls, self.shortfall_notionals
        )
        weighted_fee_bps = _microstructure.notional_weighted_shortfall_bps(self.fee_terms, self.shortfall_notionals)
        weighted_spread_bps = _microstructure.notional_weighted_shortfall_bps(
            self.spread_terms, self.shortfall_notionals
        )
        weighted_delay_bps = _microstructure.notional_weighted_shortfall_bps(
            self.delay_terms, self.shortfall_notionals
        )
        probe_fraction = (
            self.min_notional_dropped_notional / self.min_notional_total_notional
            if self.min_notional_probe_usdt > 0 and self.min_notional_total_notional > 0
            else float("nan")
        )
        reconcile_causal_state(self.accounting_state, ledger)
        coverage = tuple(sorted(self.funding_coverage_gaps.values(), key=lambda g: (g.start, g.end, g.symbol)))
        return StrategyExecutionReplayResult(
            simulated_fills=simulated_fills,
            ledger=ledger,
            simulated_units=simulated_units,
            simulated_notional_weights=simulated_notional_weights,
            fill_source=self.execution_bound,
            mark_source=self.mark_source,
            submit_times=pd.Series(self.submit_times, dtype="datetime64[ns, UTC]"),
            fill_times=pd.Series(self.fill_times, dtype="datetime64[ns, UTC]"),
            fill_count=self.fill_count,
            unfilled_count=self.unfilled_count,
            fallback_count=self.fallback_count,
            all_intent_shortfall_bps=all_intent_shortfall_bps,
            forced_exit_count=forced_exit_count,
            forced_exit_notional=forced_exit_notional,
            termination_counts=self.termination_counts,
            unsupported_assumptions=(
                "partial_fill",
                "queue_position",
                "post_only_rejection",
                "cancel_replace_latency",
                "order_size_impact",
            ),
            elapsed_seconds=elapsed_seconds,
            data_gaps=tuple(self.data_gaps),
            event_snapshots_retained=self.retain_event_snapshots,
            notional_weighted_shortfall_bps=weighted_shortfall_bps,
            residual_count=self.residual_count,
            residual_notional=self.residual_notional,
            notional_weighted_fee_bps=weighted_fee_bps,
            notional_weighted_spread_bps=weighted_spread_bps,
            notional_weighted_delay_bps=weighted_delay_bps,
            min_notional_dropped_fraction=probe_fraction,
            funding_coverage_gaps=coverage,
            terminal_positions=terminal_positions,
            ledger_available_at=ledger_available_at,
            settlement_events=tuple(self._booked_settlement_events),
            exit_block_disclosures=tuple(self.exit_block_disclosures),
        )
