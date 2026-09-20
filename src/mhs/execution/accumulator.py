"""Bound-specific streaming replay accumulator (cohesive stateful class)."""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.mhs.execution.accounting import QTY_EPS, CausalPortfolioState, reconcile_causal_state
from src.mhs.types import ExecutionSpec

from . import _ExecutionBound, _MarkSource
from . import contracts as _contracts
from . import microstructure as _microstructure
from .contracts import (
    ExecutionDataGap,
    ExecutionReplayWindow,
    FundingCoverageGap,
    InstrumentSettlementEvent,
    SimulatedInventoryLedgerResult,
    StrategyExecutionReplayResult,
    TerminalPositionEvidence,
)

# 마지막 유동 봉 이후 24h(zombie_mask_v1 K=24 1h와 동일 기간) — 실측 거래소 전체 중단 최장 69분이라 중단을 상폐로 오판하지 않는다.
DELIST_SETTLEMENT_IDLE_NS: int = 24 * 3600 * 1_000_000_000


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

        # Causal accounting mirror (P1): shadows the fill track event by
        # event and is reconciled against the independent ledger in
        # ``finalize``. Queued fills drain in bar order inside the ledger
        # append so funding always settles on pre-fill inventory.
        self.accounting_state = CausalPortfolioState(
            cash=float(initial_equity),
            units=np.zeros(self.n_cols, dtype="float64"),
            last_marks=np.full(self.n_cols, np.nan, dtype="float64"),
            last_event_ns=None,
        )
        self._mirror_pending: list[tuple[int, int, float, float, float]] = []
        self._w_qv = np.zeros((0, 0), dtype="float64")
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
        self.fill_price: list[float] = []
        self.fill_fee_bps: list[float] = []
        self.fill_reason: list[str] = []
        self.fill_pre_trade_equity: list[float] = []
        self.submit_times: list[pd.Timestamp] = []
        self.fill_times: list[pd.Timestamp] = []
        self.shortfalls: list[float] = []
        self.shortfall_notionals: list[float] = []
        self.fill_count = 0
        self.unfilled_count = 0
        self.fallback_count = 0
        self.residual_count = 0
        self.residual_notional = 0.0
        # Liquidity-aware taker cost state: one half-spread estimate per
        # canonical column, nan until a window's bars have been consumed.
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
        the frozen slippage, reproducing legacy behaviour bit-identically.
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

    def consume(self, w: ExecutionReplayWindow) -> None:
        """Consume one window through the ordered replay phases. Physical IO boundaries do not reset inventory, liquidity or the logical spread clock. Overlap observations and settlements are applied once."""
        (n_cols, local_cols, n_local, gpos, grid, grid_ns, n_grid, bar_ns, marks_values, highs_values, lows_values, closes_values, close_finite, mark_valid, funding_matrix) = self._consume_validate_window(w)
        for gap in w.funding_coverage_gaps:
            self.funding_coverage_gaps[(gap.symbol, gap.start, gap.end, gap.reason)] = gap
        self._admit_settlement_events(w)
        self._advance_spread_clock(w.logical_partition)
        (last_close_idx, decision_ns_all, spos_all, dpos_all, on_grid_all, target_values, submit_anchored, fill_start, tw_index, sig_index) = self._consume_prepare_tables(w, grid_ns, n_grid, close_finite, local_cols)
        for i in range(len(tw_index)):
            self._consume_drift_trims(int(decision_ns_all[i]), decision_ns_all, gpos, local_cols, submit_anchored, n_grid, grid_ns, closes_values, grid, marks_values, n_cols, mark_valid, last_close_idx, lows_values, highs_values, funding_matrix)
            self._consume_single_intent(i, decision_ns_all, dpos_all, on_grid_all, gpos, target_values, spos_all, marks_values, grid_ns, funding_matrix, tw_index, sig_index, mark_valid, last_close_idx, local_cols, submit_anchored, n_grid, closes_values, grid, n_cols, lows_values, highs_values)
        self._consume_drift_trims(None, decision_ns_all, gpos, local_cols, submit_anchored, n_grid, grid_ns, closes_values, grid, marks_values, n_cols, mark_valid, last_close_idx, lows_values, highs_values, funding_matrix)
        self._settle_due_events(grid_ns, n_grid, local_cols, gpos, marks_values, n_cols)
        self._consume_append_ledger(grid_ns, n_grid, local_cols, fill_start, n_local, marks_values, gpos, grid, funding_matrix, bar_ns, closes_values)
        self._advance_liquidity_carry(grid_ns, gpos, decision_ns_all)
        self._observe_spread_partition(highs_values, lows_values, gpos, grid_ns, bar_ns, w.logical_partition)

    def _consume_validate_window(self, w: ExecutionReplayWindow) -> tuple[int, list[str], int, np.ndarray, pd.DatetimeIndex, np.ndarray, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Validate one window and stage its grids, marks, and funding."""
        columns = self.columns
        n_cols = self.n_cols
        gpos_of = self.gpos_of
        if w.columns != columns:
            raise DataIntegrityError("all execution windows must share an identical column order")
        local_cols = list(w.symbols)
        n_local = len(local_cols)
        gpos = np.asarray([gpos_of[s] for s in local_cols], dtype=np.intp)
        in_window = np.zeros(n_cols, dtype=bool)
        in_window[gpos] = True
        outside = np.flatnonzero((np.abs(self.units_arr) >= QTY_EPS) & ~in_window)
        if outside.size:
            j = int(outside[0])
            raise DataIntegrityError(f"held position outside execution window roster (symbol={columns[j]!r} units={float(self.units_arr[j])!r})")
        grid = w.minute_grid
        grid_ns = np.asarray(grid, dtype="datetime64[ns]").astype("int64")
        n_grid = len(grid_ns)
        if n_grid < 2:
            raise DataIntegrityError("an execution window must span at least two grid bars")
        if not w.bar_funding.index.equals(grid):
            raise DataIntegrityError("bar_funding must align exactly to the window minute grid")
        bar_ns = int(grid_ns[1] - grid_ns[0])
        self.full_grid_end = grid[-1]
        marks = w.marks if w.marks is not None else w.closes
        marks_values = marks[local_cols].to_numpy(dtype="float64")
        highs_values = w.highs[local_cols].to_numpy(dtype="float64")
        lows_values = w.lows[local_cols].to_numpy(dtype="float64")
        closes_values = w.closes[local_cols].to_numpy(dtype="float64")
        close_finite = np.isfinite(closes_values)
        sym_finite = np.isfinite(marks_values)
        mark_valid = sym_finite & (marks_values > 0.0)
        if n_local:
            funding_matrix = np.stack(
                [w.bar_funding[s].to_numpy(dtype="float64") for s in local_cols], axis=1,
            )
        else:
            funding_matrix = np.zeros((n_grid, 0), dtype="float64")
        if not np.isfinite(funding_matrix).all():
            raise DataIntegrityError("bar_funding must be finite")
        finite_marks = marks_values[sym_finite]
        if (finite_marks <= 0).any():
            raise DataIntegrityError("finite marks must be strictly positive")

        # Window context for the causal replay: quote volumes (ones when the
        # window carries none, preserving legacy direct construction),
        # funding knowledge (all-known when absent), and per-bar availability
        # (the grid itself when absent, so effective time equals the label).
        n_grid_int = int(n_grid)
        if w.quote_volumes is not None:
            qv = np.full((n_grid_int, n_local), 1.0, dtype="float64")
            for j, sym in enumerate(local_cols):
                if sym in w.quote_volumes.columns:
                    qv[:, j] = w.quote_volumes[sym].to_numpy(dtype="float64")
        else:
            qv = np.ones((n_grid_int, n_local), dtype="float64")
        self._w_qv = qv
        self._w_last_liquid_idx = np.maximum.accumulate(np.where(qv > 0.0, np.arange(n_grid_int)[:, None], -1), axis=0)
        if w.funding_known is not None:
            fknown = np.zeros((n_grid_int, n_local), dtype=bool)
            for j, sym in enumerate(local_cols):
                if sym in w.funding_known.columns:
                    fknown[:, j] = w.funding_known[sym].to_numpy(dtype=bool)
        else:
            fknown = np.ones((n_grid_int, n_local), dtype=bool)
        self._w_fknown = fknown
        if w.bar_available_at is not None and len(w.bar_available_at) == n_grid_int:
            avail_ns = np.asarray(w.bar_available_at, dtype="datetime64[ns]").astype("int64")
            self._w_avail_explicit = True
        else:
            avail_ns = grid_ns.copy()
            self._w_avail_explicit = False
        self._w_avail_ns = avail_ns
        # Mark availability trails bar availability by one bar: the mark read
        # at a decision is sourced from the previous bar's close (hourly-carry
        # semantics), so the decision bar's own close tolerance from F8 stays
        # usable while strictly later bars fail the PIT check downstream.
        bar_step = int(avail_ns[1] - avail_ns[0]) if len(avail_ns) > 1 else 0
        self._w_mark_avail = np.repeat((avail_ns - bar_step)[:, None], n_local, axis=1)
        return (n_cols, local_cols, n_local, gpos, grid, grid_ns, n_grid, bar_ns, marks_values, highs_values, lows_values, closes_values, close_finite, mark_valid, funding_matrix)


    def _advance_window(self, target_ns: int, dpos: int, on_grid: bool, marks_values: np.ndarray, mark_available_ns: np.ndarray, grid_ns: np.ndarray, funding_matrix: np.ndarray, funding_known: np.ndarray, gpos: np.ndarray) -> None:
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
            m = marks_values[dpos]
            avail = mark_available_ns[dpos]
            usable = np.isfinite(m) & (m > 0.0) & (avail <= target_ns)
            prev = self.last_prices_arr[gpos]
            self.last_prices_arr[gpos] = np.where(usable, m, prev)
            future_read = np.isfinite(m) & (m > 0.0) & ~usable
            if bool(future_read.any()):
                self.ledger_valid = False
                self.invalid_reasons.add("MISSING_DATA")
                event_ts = pd.Timestamp(target_ns, unit="ns", tz="UTC")
                for j in np.flatnonzero(future_read).tolist():
                    self.data_gaps.append(
                        ExecutionDataGap(
                            code="FUTURE_DATA_REFERENCE", symbol=self.columns[int(gpos[j])],
                            timestamp=event_ts, execution_bound=self.execution_bound,
                        )
                    )
        lo = np.searchsorted(grid_ns, self.last_time_ns, side="right") if self.last_time_ns is not None else 0
        hi = int(np.searchsorted(grid_ns, target_ns, side="right"))
        lo = int(lo)
        if lo < hi:
            span_rates = funding_matrix[lo:hi, :]
            span_known = funding_known[lo:hi, :]
            span_marks = marks_values[lo:hi, :]
            span_units = np.repeat(self.units_arr[gpos][None, :], hi - lo, axis=0)
            for fns, fj, fqty in self._fills_in_span(self.last_time_ns, target_ns, grid_ns, lo, gpos):
                span_units[:fns, int(fj)] -= float(fqty)
            held_unknown = (np.abs(span_units) >= QTY_EPS) & ~span_known
            if bool(held_unknown.any()):
                self.ledger_valid = False
                self.invalid_reasons.add("MISSING_DATA")
                hit = np.flatnonzero(held_unknown.any(axis=1))[0]
                witness = int(np.flatnonzero(held_unknown[int(hit)])[0])
                self.data_gaps.append(
                    ExecutionDataGap(
                        code="MISSING_HELD_FUNDING", symbol=self.columns[int(gpos[witness])],
                        timestamp=pd.Timestamp(int(grid_ns[lo + int(hit)]), unit="ns", tz="UTC"),
                        execution_bound=self.execution_bound,
                    )
                )
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
        # last_time_ns가 단조증가하므로 floor 이하 접두 체결은 이후 모든 호출에서도 제외 — 커서로 건너뛰어 O(전체 체결) 재주사를 없앤다.
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
        qv = float(self._w_qv[fill_pos, col])
        if qv == 0.0:
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
        if not (int(signal_ns) < int(submit_ns) <= int(avail_submit_ns)):
            return ExecutionDataGap(
                code="CAUSAL_TIMING_VIOLATION", symbol=sym,
                timestamp=decision_ts, decision_time=decision_ts, signal_time=signal_ts,
                execution_bound=self.execution_bound,
            )
        return None

    def _block_fill(
        self, gap: ExecutionDataGap, *, prior_units: float, net_units: float,
    ) -> bool:
        """Record a viability block, distinguishing a blocked entry from a blocked exit.

        A blocked entry leaves the book exactly as it was, so retrying at the next decision is
        both correct and riskless. A blocked exit does not: the position the strategy decided to
        shed stays on, and its valuation and financing risk continue to accrue while the engine
        reports nothing. Treating both as the same benign retry understates realized risk and is
        the reason an unexitable position only surfaces later, at an unrelated bar.

        Args:
            gap: The viability gap that blocked this fill.
            prior_units: Signed inventory held immediately before the blocked intent.
            net_units: Signed quantity the blocked intent would have transacted.
        Returns:
            True, so the caller advances to the next symbol exactly as before.
        """
        is_exit = abs(prior_units) >= QTY_EPS and abs(prior_units + net_units) < abs(prior_units)
        if gap.code == "KNOWN_ZERO_VOLUME" and not is_exit:
            # 알려진 0 거래량은 데이터 공백이 아니라 체결 불가(거래소 중단·상폐 꼬리) — 다음 결정에서 재시도(라이브와 동일).
            self.unfilled_count += 1
            self.termination_counts["NO_VOLUME_UNFILLED"] = self.termination_counts.get("NO_VOLUME_UNFILLED", 0) + 1
            return True
        if gap.code == "MISSING_ACTIVE_FUNDING" and not is_exit:
            # 펀딩 unknown으로 막힌 신규 체결 시도는 보유 리스크가 없다(체결 전이라
            # 자본 노출이 아직 없음) — KNOWN_ZERO_VOLUME과 동일하게 미체결·재시도.
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
                continue
            self._settlement_seen.add(event.event_id)
            self._pending_settlements.append(event)
        self._pending_settlements.sort(key=lambda e: (e.available_at.value, e.effective_at.value, e.event_id))

    def _settle_due_events(
        self,
        grid_ns: np.ndarray,
        n_grid: int,
        local_cols: list[str],
        gpos: np.ndarray,
        marks_values: np.ndarray,
        n_cols: int,
    ) -> None:
        """Execute evidenced settlements due within this window at their event price.

        A settlement executes no earlier than both its effective and available
        time. An observation endpoint or idle threshold cannot create this event.
        """
        if not self._pending_settlements:
            return
        grid_end_ns = int(grid_ns[-1])
        local_of = {sym: (col, int(gcol)) for col, (sym, gcol) in enumerate(zip(local_cols, gpos.tolist(), strict=True))}
        remaining: list[InstrumentSettlementEvent] = []
        p0_ns = int(self.ledger_start_ns) if self.ledger_start_ns is not None else int(grid_ns[0])
        for event in self._pending_settlements:
            due_ns = max(int(event.effective_at.value), int(event.available_at.value))
            if due_ns > grid_end_ns:
                remaining.append(event)
                continue
            entry = local_of.get(event.symbol)
            if entry is None:
                remaining.append(event)
                continue
            col, gcol = entry
            units = float(self.units_arr[gcol])
            if abs(units) < QTY_EPS:
                self._settled_events.append(event)
                continue
            due_pos = int(np.searchsorted(grid_ns, np.int64(due_ns), side="left"))
            due_pos = max(due_pos, int(np.searchsorted(grid_ns, np.int64(p0_ns), side="left")))
            self._book_delist_settlement(
                col, gcol, event.symbol, due_pos, units, float(event.settlement_price),
                grid_ns, marks_values, gpos, n_cols,
                fee_bps=float(event.fee_bps), event_id=event.event_id,
            )
            self._settled_events.append(event)
        self._pending_settlements = remaining

    def _settle_idle_holdings(self, dns: int, spos: int, gpos: np.ndarray, local_cols: list[str], grid_ns: np.ndarray, marks_values: np.ndarray, n_grid: int, n_cols: int) -> None:
        """Carry last-liquid provenance without inferring any settlement.

        Prolonged observed zero volume never creates a free settlement or
        quantity cancellation; only an evidenced exchange settlement event
        closes inventory.
        """
        return

    def _book_delist_settlement(self, col: int, gcol: int, sym: str, spos: int, units: float, price: float, grid_ns: np.ndarray, marks_values: np.ndarray, gpos: np.ndarray, n_cols: int, *, fee_bps: float = 0.0, event_id: str | None = None) -> None:
        """Book a causal delisting settlement.

        Books the validated event price, fee, and identity into the fill track,
        accounting mirror, and ledger identically.
        """
        qty = -units
        self.last_prices_arr[gcol] = price
        self.cash -= qty * price
        fee = float(fee_bps) / 1e4 * abs(qty) * price
        self.cash -= fee
        self.units_arr[gcol] = 0.0
        self.fill_bar_ns.append(int(grid_ns[spos]))
        self.fill_gcol.append(int(gcol))
        self._mirror_pending.append((int(grid_ns[spos]), int(gcol), float(qty), float(price), float(fee_bps)))
        fill_time = pd.Timestamp(int(self._w_avail_ns[spos]), unit="ns", tz="UTC")
        pre_trade_equity = self._equity_at(gpos)
        self.fill_ts.append(fill_time)
        self.fill_symbol.append(sym)
        self.fill_qty.append(qty)
        self.fill_price.append(price)
        self.fill_fee_bps.append(float(fee_bps))
        self.fill_reason.append("delist_settlement")
        self.fill_pre_trade_equity.append(pre_trade_equity)
        self.fill_times.append(fill_time)
        self.submit_times.append(fill_time)
        self.termination_counts["DELIST_SETTLEMENT"] = self.termination_counts.get("DELIST_SETTLEMENT", 0) + 1
        if self.retain_event_snapshots:
            marks_row = np.full(n_cols, np.nan, dtype="float64")
            marks_row[gpos] = marks_values[spos]
            self.units_after_events.append((fill_time, self.units_arr.copy()))
            self.notional_after_events.append((fill_time, self.units_arr * marks_row))

    def _advance_liquidity_carry(self, grid_ns: np.ndarray, gpos: np.ndarray, decision_ns_all: np.ndarray) -> None:
        """Carry last-liquid timestamps forward to the next window (causal)."""
        if len(decision_ns_all) == 0 or len(gpos) == 0:
            return
        rows = int(np.searchsorted(self._w_avail_ns, int(decision_ns_all[-1]), side="right"))
        if rows <= 0:
            return
        idx = self._w_last_liquid_idx[rows - 1]
        cand = np.where(idx >= 0, grid_ns[np.maximum(idx, 0)], -1).astype("int64")
        # 다음 윈도우는 이 윈도우의 마지막 결정 시각부터 시작하므로 그 시각까지 가용한 봉만 이월한다(미래 봉 누설 없음).
        self.last_liquid_ns[gpos] = np.maximum(self.last_liquid_ns[gpos], cand)

    def _consume_decision_price(self, col: int, on_grid: bool, dpos: int, spos: int, marks_values: np.ndarray, mark_valid: np.ndarray, last_close_idx: np.ndarray, local_cols: list[str], n_grid: int) -> float | None:
        """Resolve the anchor price for one intent, carried closes included."""
        if on_grid and mark_valid[dpos, col]:
            return float(marks_values[dpos, col])
        j = int(last_close_idx[spos - 1, col]) if spos > 0 else -1
        if j >= 0 and mark_valid[j, col]:
            return float(marks_values[j, col])
        sym = local_cols[col]
        carried_ts = self.last_close_ts.get(sym)
        if carried_ts is not None:
            carried_mark = self.last_close_mark[sym]
            if np.isfinite(carried_mark) and carried_mark > 0.0:
                return float(carried_mark)
        if spos < n_grid and mark_valid[spos, col]:
            return float(marks_values[spos, col])
        return None


    def _consume_prepare_tables(self, w: ExecutionReplayWindow, grid_ns: np.ndarray, n_grid: int, close_finite: np.ndarray, local_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, int, pd.DatetimeIndex, pd.DatetimeIndex]:
        """Build per-window close tables and vectorised decision anchors."""

        # Per-window last-finite-close index table: ``last_close_idx[i, col]`` is
        # the largest ``j <= i`` with ``close_finite[j, col]`` True, else -1.
        # This makes the scalar ``_decision_price`` backward scan a single
        # vectorised lookup (bit-identical: it returns the same last finite
        # close position the while-loop would stop at).
        close_row = np.where(close_finite, np.arange(n_grid)[:, None], -1)
        last_close_idx = np.maximum.accumulate(close_row, axis=0)
        decision_ns_all = np.asarray(w.target_weights.index, dtype="datetime64[ns]").astype("int64")
        signal_ns_all = np.asarray(w.signal_available_at, dtype="datetime64[ns]").astype("int64")
        spos_all = np.searchsorted(grid_ns, signal_ns_all, side="right")
        dpos_all = np.searchsorted(grid_ns, decision_ns_all, side="left")
        dpos_clipped = np.minimum(dpos_all, n_grid - 1)
        on_grid_all = np.where(dpos_all < n_grid, grid_ns[dpos_clipped] == decision_ns_all, False)
        target_values = w.target_weights[local_cols].to_numpy(dtype="float64")
        # submit_bar anchor: the reference is the mark at bar spos-1 -- the bar
        # that closes exactly at the submission bar's open, hence observable at
        # submit time (no look-ahead). decision_bar keeps the frozen default.
        submit_anchored = self.spec.decision_anchor == "submit_bar"

        fill_start = len(self.fill_ts)
        # Lazy pd.Timestamp boxing: the decision/fill hot path never touches
        # the index; gaps, fills, and equity-floor breaches memoise on demand.
        tw_index = w.target_weights.index
        sig_index = w.signal_available_at
        return (last_close_idx, decision_ns_all, spos_all, dpos_all, on_grid_all, target_values, submit_anchored, fill_start, tw_index, sig_index)


    def _consume_drift_trims(self, until_ns: int | None, decision_ns_all: np.ndarray, gpos: np.ndarray, local_cols: list[str], submit_anchored: bool, n_grid: int, grid_ns: np.ndarray, closes_values: np.ndarray, grid: pd.DatetimeIndex, marks_values: np.ndarray, n_cols: int, mark_valid: np.ndarray, last_close_idx: np.ndarray, lows_values: np.ndarray, highs_values: np.ndarray, funding_matrix: np.ndarray) -> None:
        """Run due intraday single-name drift checks before a decision or the ledger."""
        cap = self.spec.name_drift_trim_max_weight
        if cap is None:
            return
        if self._trim_anchor_ns is None:
            if len(decision_ns_all) == 0:
                return
            # 최초 결정 시각을 앵커로 고정하고 이후 절대 갱신하지 않는다.
            self._trim_anchor_ns = int(decision_ns_all[0])
        anchor = int(self._trim_anchor_ns)
        step = int(self.spec.name_drift_trim_interval_hours) * 3_600_000_000_000
        # 단조 last_time_ns를 하한으로 삼아 겹치는 윈도우에서 중복 검사를 막는다.
        floor = anchor if self.last_time_ns is None else max(anchor, int(self.last_time_ns))
        k = (floor - anchor) // step + 1
        bar_ns = int(grid_ns[1] - grid_ns[0])
        grid_end = int(grid_ns[-1])
        while True:
            dpos = int(np.searchsorted(grid_ns, anchor + k * step, side="left"))
            k += 1
            if dpos >= n_grid:
                return
            # 오프그리드 검사는 다음 바에 스냅하고 절대 건너뛰지 않는다.
            check_ns = int(grid_ns[dpos])
            resolve_ns = check_ns + bar_ns + self.timeout_ns_delta
            # 결정 주문과 겹치는 검사는 건너뛰고 결정 리밸런스에 맡긴다.
            if resolve_ns > grid_end or (until_ns is not None and resolve_ns >= until_ns):
                return
            self._consume_drift_trim_at(check_ns, dpos, cap, gpos, local_cols, submit_anchored, n_grid, grid_ns, closes_values, grid, marks_values, n_cols, mark_valid, last_close_idx, lows_values, highs_values, funding_matrix)

    def _consume_drift_trim_at(self, check_ns: int, dpos: int, cap: float, gpos: np.ndarray, local_cols: list[str], submit_anchored: bool, n_grid: int, grid_ns: np.ndarray, closes_values: np.ndarray, grid: pd.DatetimeIndex, marks_values: np.ndarray, n_cols: int, mark_valid: np.ndarray, last_close_idx: np.ndarray, lows_values: np.ndarray, highs_values: np.ndarray, funding_matrix: np.ndarray) -> None:
        """Trim names breached above the cap at one intraday check via taker fills."""
        self._advance_window(check_ns, dpos, True, marks_values, self._w_mark_avail, grid_ns, funding_matrix, self._w_fknown, gpos)
        equity = self._equity_at(gpos)
        prices = self.last_prices_arr[gpos]
        units = self.units_arr[gpos]
        weights = np.where(np.isfinite(prices), units * prices, 0.0) / equity
        trigger = cap * (1.0 + self.spec.one_way_taker_bps() / 1e4)
        # cap에 정확히 맞추면 편도 테이커 비용 비율만큼 초과가 남으므로 밴드가 매 검사 재트림을 막는다.
        over = (np.abs(units) >= QTY_EPS) & (np.abs(weights) > trigger)
        if not bool(over.any()):
            return
        row = np.full(len(local_cols), np.nan, dtype="float64")
        row[over] = np.sign(weights[over]) * cap
        stamp = pd.DatetimeIndex([pd.Timestamp(check_ns, unit="ns", tz="UTC")])
        spos = int(np.searchsorted(grid_ns, check_ns, side="right"))
        for col in np.flatnonzero(over).tolist():
            n_before = len(self.fill_reason)
            self._consume_single_fill(0, col, gpos, local_cols, row, True, submit_anchored, dpos, spos, equity, n_grid, grid_ns, closes_values, grid, marks_values, n_cols, stamp, stamp, mark_valid, last_close_idx, lows_values, highs_values)
            if len(self.fill_reason) > n_before:
                for j in range(n_before, len(self.fill_reason)):
                    self.fill_reason[j] = "drift_trim"
                self.termination_counts["DRIFT_TRIM"] = self.termination_counts.get("DRIFT_TRIM", 0) + 1

    def _consume_single_intent(self, i: int, decision_ns_all: np.ndarray, dpos_all: np.ndarray, on_grid_all: np.ndarray, gpos: np.ndarray, target_values: np.ndarray, spos_all: np.ndarray, marks_values: np.ndarray, grid_ns: np.ndarray, funding_matrix: np.ndarray, tw_index: pd.DatetimeIndex, sig_index: pd.DatetimeIndex, mark_valid: np.ndarray, last_close_idx: np.ndarray, local_cols: list[str], submit_anchored: bool, n_grid: int, closes_values: np.ndarray, grid: pd.DatetimeIndex, n_cols: int, lows_values: np.ndarray, highs_values: np.ndarray) -> None:
        """Process one decision index across its active symbols."""
        dns = int(decision_ns_all[i])
        dpos = int(dpos_all[i])
        on_grid = bool(on_grid_all[i])
        self._advance_window(dns, dpos, on_grid, marks_values, self._w_mark_avail, grid_ns, funding_matrix, self._w_fknown, gpos)
        self._settle_idle_holdings(dns, int(spos_all[i]), gpos, local_cols, grid_ns, marks_values, n_grid, n_cols)
        equity = self._equity_at(gpos)
        last_ledger_equity: float | None = None
        if self.equity_chunks:
            last_ledger_equity = float(self.equity_chunks[-1][-1])
        guard_equity = _contracts.ruin_guard_equity(equity, last_ledger_equity)
        row = target_values[i]
        if self.min_equity_fraction is not None and guard_equity <= self.min_equity_fraction * self.initial_equity:
            if not self.equity_floor_breaches or self.equity_floor_breaches[-1] != tw_index[i]:
                self.equity_floor_breaches.append(tw_index[i])
            row = np.zeros_like(row)
        spos = int(spos_all[i])
        active = np.where(np.isfinite(row) & ((row != 0.0) | (self.units_arr[gpos] != 0.0)))[0]
        for col in active.tolist():
            if self._consume_single_fill(i, col, gpos, local_cols, row, on_grid, submit_anchored, dpos, spos, equity, n_grid, grid_ns, closes_values, grid, marks_values, n_cols, tw_index, sig_index, mark_valid, last_close_idx, lows_values, highs_values):
                continue


    def _consume_single_fill(self, i: int, col: int, gpos: np.ndarray, local_cols: list[str], row: np.ndarray, on_grid: bool, submit_anchored: bool, dpos: int, spos: int, equity: float, n_grid: int, grid_ns: np.ndarray, closes_values: np.ndarray, grid: pd.DatetimeIndex, marks_values: np.ndarray, n_cols: int, tw_index: pd.DatetimeIndex, sig_index: pd.DatetimeIndex, mark_valid: np.ndarray, last_close_idx: np.ndarray, lows_values: np.ndarray, highs_values: np.ndarray) -> bool:
        """Resolve and book one symbol intent; True advances to the next symbol."""
        gcol = int(gpos[col])
        sym = local_cols[col]
        weight = float(row[col])
        decision_price = self._consume_decision_price(col, on_grid and not submit_anchored, dpos, spos, marks_values, mark_valid, last_close_idx, local_cols, n_grid)
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
        net_units = desired_units - self.units_arr[gcol]
        if abs(net_units) < 1e-12:
            return True
        side = 1 if net_units > 0 else -1
        self._probe_intent_notional(net_units, decision_price)
        if spos >= n_grid:
            self.termination_counts["MISSING_DATA"] += 1
            return True
        submit_pos = spos
        timeout_ns = grid_ns[spos] + self.timeout_ns_delta
        timeout_pos = int(np.searchsorted(grid_ns, timeout_ns, side="left"))
        timeout_close = float("nan")
        adverse = np.array([], dtype="float64")
        if self.execution_bound == "OHLCV_IMMEDIATE_TAKER":
            fill_pos = submit_pos
            fill_price = float(closes_values[fill_pos, col])
            if not np.isfinite(fill_price):
                self.termination_counts["MISSING_DATA"] += 1
                self.data_gaps.append(
                    ExecutionDataGap(
                        code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                        timestamp=grid[fill_pos], decision_time=tw_index[i],
                        signal_time=sig_index[i], execution_bound=self.execution_bound,
                    )
                )
                return True
            taker_cost_bps = self._taker_cost_bps(gcol)
            fee_bps = self.spec.taker_fee_bps + taker_cost_bps
            reason = "timeout_taker"
        else:
            if self.execution_bound == "OHLCV_LADDERED_PROXY" and self._consume_fill_laddered(timeout_pos, spos, sym, grid, side, lows_values, col, highs_values, closes_values, decision_price, n_grid, grid_ns, timeout_ns, net_units, submit_pos, marks_values, gcol, weight, equity, gpos, n_cols, tw_index, sig_index, i):
                return True
            if self.execution_bound == "OHLCV_PEG_CHASE_PROXY" and self._consume_fill_peg_chase(timeout_pos, spos, sym, grid, side, lows_values, col, highs_values, closes_values, gcol, decision_price, net_units, submit_pos, marks_values, weight, equity, gpos, n_cols, tw_index, sig_index, i):
                return True
            _proceed, fill_pos, fill_price, fee_bps, reason, timeout_close, adverse = self._consume_fill_strict_touch(timeout_pos, spos, sym, grid, side, lows_values, col, highs_values, decision_price, n_grid, grid_ns, timeout_ns, closes_values, gcol, tw_index, sig_index, i)
            if _proceed:
                return True
        block = self._bar_viability_gap(
            fill_pos=fill_pos, col=col, sym=sym,
            decision_ts=tw_index[i], signal_ts=sig_index[i],
            submit_ns=int(grid_ns[submit_pos]), signal_ns=int(sig_index[i].value),
            avail_submit_ns=int(self._w_avail_ns[submit_pos]),
        )
        if block is not None:
            return self._block_fill(
                block, prior_units=float(self.units_arr[gcol]), net_units=float(net_units)
            )
        if reason == "passive_fill":
            self.fill_count += 1
        if self.execution_bound == "OHLCV_IMMEDIATE_TAKER":
            shortfall = side * (fill_price / decision_price - 1.0) * 1e4 + fee_bps
            if self.spec.liquidity_cost_model == "corwin_schultz":
                self._record_terms(
                    decision_price, fill_price, side,
                    self.spec.taker_fee_bps,
                    fee_bps - self.spec.taker_fee_bps,
                )
            else:
                # Flat model: the fixed slippage folds into the fee
                # term and the spread term stays exactly zero.
                self._record_terms(decision_price, fill_price, side, fee_bps, 0.0)
        else:
            shortfall = _microstructure.passive_fill_shortfall_bps(
                decision_price, adverse, timeout_close, side, self.spec,
                taker_cost_bps=self._taker_cost_bps(gcol),
            )
            # The residual after timing is the all-in fee component;
            # deriving it keeps fee+spread+delay == shortfall exact
            # even on degenerate exact-touch fills.
            anchor = fill_price if reason == "passive_fill" else timeout_close
            self._record_terms(
                decision_price, anchor, side,
                shortfall - side * (anchor / decision_price - 1.0) * 1e4,
                0.0,
            )
        self.shortfalls.append(shortfall)
        self.shortfall_notionals.append(abs(net_units) * fill_price)
        fill_time = pd.Timestamp(int(self._w_avail_ns[fill_pos]), unit="ns", tz="UTC")
        submit_time = pd.Timestamp(int(self._w_avail_ns[submit_pos]), unit="ns", tz="UTC")
        mark_price = float(marks_values[fill_pos, col])
        if np.isfinite(mark_price):
            self.last_prices_arr[gcol] = mark_price
        if not (np.isfinite(net_units) and np.isfinite(fill_price)):
            raise DataIntegrityError(
                "non-finite fill sizing breaches the capital accounting invariant "
                f"(symbol={sym!r} ts={fill_time!r} weight={weight!r} equity={equity!r} "
                f"decision_price={decision_price!r} qty={net_units!r} fill_price={fill_price!r})"
            )
        self.cash -= net_units * fill_price
        fee = fee_bps / 1e4 * abs(net_units) * fill_price
        self.cash -= fee
        self.units_arr[gcol] += net_units
        self.fill_bar_ns.append(int(grid_ns[fill_pos]))
        self.fill_gcol.append(int(gcol))
        self._mirror_pending.append(
            (int(grid_ns[fill_pos]), int(gcol), float(net_units), float(fill_price), float(fee_bps))
        )
        if reason in ("passive_fill", "timeout_taker"):
            pre_trade_equity = self._equity_at(gpos)
            self.fill_ts.append(fill_time)
            self.fill_symbol.append(sym)
            self.fill_qty.append(net_units)
            self.fill_price.append(fill_price)
            self.fill_fee_bps.append(fee_bps)
            self.fill_reason.append(reason)
            self.fill_pre_trade_equity.append(pre_trade_equity)
            self.fill_times.append(fill_time)
            self.submit_times.append(submit_time)
            if self.retain_event_snapshots:
                marks_row = np.full(n_cols, np.nan, dtype="float64")
                marks_row[gpos] = marks_values[fill_pos]
                self.units_after_events.append((fill_time, self.units_arr.copy()))
                self.notional_after_events.append((fill_time, self.units_arr * marks_row))
        return False


    def _consume_fill_laddered(self, timeout_pos: int, spos: int, sym: str, grid: pd.DatetimeIndex, side: int, lows_values: np.ndarray, col: int, highs_values: np.ndarray, closes_values: np.ndarray, decision_price: float, n_grid: int, grid_ns: np.ndarray, timeout_ns: int, net_units: float, submit_pos: int, marks_values: np.ndarray, gcol: int, weight: float, equity: float, gpos: np.ndarray, n_cols: int, tw_index: pd.DatetimeIndex, sig_index: pd.DatetimeIndex, i: int) -> bool:
        """Book laddered tranches; always advances to the next symbol."""
        if timeout_pos <= spos:
            self.termination_counts["MISSING_DATA"] += 1
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                    timestamp=grid[spos], decision_time=tw_index[i],
                    signal_time=sig_index[i], execution_bound=self.execution_bound,
                )
            )
            return True
        adverse = (
            lows_values[spos:timeout_pos, col]
            if side == 1
            else highs_values[spos:timeout_pos, col]
        )
        if not np.isfinite(adverse).all():
            self.termination_counts["MISSING_DATA"] += 1
            first_bad = spos + int(np.argmax(~np.isfinite(adverse)))
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                    timestamp=grid[first_bad], decision_time=tw_index[i],
                    signal_time=sig_index[i], execution_bound=self.execution_bound,
                )
            )
            return True
        closes_window = closes_values[spos:timeout_pos + 1, col]
        if not np.isfinite(closes_window).all():
            self.termination_counts["MISSING_DATA"] += 1
            first_bad = spos + int(np.argmax(~np.isfinite(closes_window)))
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                    timestamp=grid[first_bad], decision_time=tw_index[i],
                    signal_time=sig_index[i], execution_bound=self.execution_bound,
                )
            )
            return True
        ladder_entry_units = float(self.units_arr[gcol])
        for rel_pos, tranche_price, tranche_fee_bps, qty_fraction in _microstructure.laddered_fill_schedule(
            decision_price, side, adverse,
            closes_window,
            self.spec.ladder_tranches, self.spec, True,
        ):
            fill_pos = spos + rel_pos
            if rel_pos == len(adverse):
                if timeout_pos >= n_grid or grid_ns[timeout_pos] != timeout_ns:
                    self.termination_counts["MISSING_DATA"] += 1
                    self.data_gaps.append(
                        ExecutionDataGap(
                            code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                            timestamp=grid[spos], decision_time=tw_index[i],
                            signal_time=sig_index[i], execution_bound=self.execution_bound,
                        )
                    )
                    continue
                timeout_close = float(closes_values[timeout_pos, col])
                if not np.isfinite(timeout_close):
                    self.termination_counts["MISSING_DATA"] += 1
                    self.data_gaps.append(
                        ExecutionDataGap(
                            code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                            timestamp=grid[timeout_pos], decision_time=tw_index[i],
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
                submit_ns=int(grid_ns[submit_pos]), signal_ns=int(sig_index[i].value),
                avail_submit_ns=int(self._w_avail_ns[submit_pos]),
            )
            if block is not None:
                current_units = float(self.units_arr[gcol])
                return self._block_fill(
                    block, prior_units=current_units,
                    net_units=float(ladder_entry_units + net_units - current_units),
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
            fill_time = pd.Timestamp(int(self._w_avail_ns[fill_pos]), unit="ns", tz="UTC")
            submit_time = pd.Timestamp(int(self._w_avail_ns[submit_pos]), unit="ns", tz="UTC")
            mark_price = float(marks_values[fill_pos, col])
            if np.isfinite(mark_price):
                self.last_prices_arr[gcol] = mark_price
            if not (np.isfinite(qty) and np.isfinite(fill_price)):
                raise DataIntegrityError(
                    "non-finite fill sizing breaches the capital accounting invariant "
                    f"(symbol={sym!r} ts={fill_time!r} weight={weight!r} equity={equity!r} "
                    f"decision_price={decision_price!r} qty={qty!r} fill_price={fill_price!r})"
                )
            self.cash -= qty * fill_price
            fee = fee_bps / 1e4 * abs(qty) * fill_price
            self.cash -= fee
            self.units_arr[gcol] += qty
            self.fill_bar_ns.append(int(grid_ns[fill_pos]))
            self.fill_gcol.append(int(gcol))
            self._mirror_pending.append(
                (int(grid_ns[fill_pos]), int(gcol), float(qty), float(fill_price), float(fee_bps))
            )
            pre_trade_equity = self._equity_at(gpos)
            self.fill_ts.append(fill_time)
            self.fill_symbol.append(sym)
            self.fill_qty.append(qty)
            self.fill_price.append(fill_price)
            self.fill_fee_bps.append(fee_bps)
            self.fill_reason.append(reason)
            self.fill_pre_trade_equity.append(pre_trade_equity)
            self.fill_times.append(fill_time)
            self.submit_times.append(submit_time)
            if self.retain_event_snapshots:
                marks_row = np.full(n_cols, np.nan, dtype="float64")
                marks_row[gpos] = marks_values[fill_pos]
                self.units_after_events.append((fill_time, self.units_arr.copy()))
                self.notional_after_events.append((fill_time, self.units_arr * marks_row))
        return True
        return True


    def _consume_fill_peg_chase(self, timeout_pos: int, spos: int, sym: str, grid: pd.DatetimeIndex, side: int, lows_values: np.ndarray, col: int, highs_values: np.ndarray, closes_values: np.ndarray, gcol: int, decision_price: float, net_units: float, submit_pos: int, marks_values: np.ndarray, weight: float, equity: float, gpos: np.ndarray, n_cols: int, tw_index: pd.DatetimeIndex, sig_index: pd.DatetimeIndex, i: int) -> bool:
        """Book the peg-chase schedule; always advances to the next symbol."""
        if timeout_pos <= spos:
            self.termination_counts["MISSING_DATA"] += 1
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                    timestamp=grid[spos], decision_time=tw_index[i],
                    signal_time=sig_index[i], execution_bound=self.execution_bound,
                )
            )
            return True
        adverse = (
            lows_values[spos:timeout_pos, col]
            if side == 1
            else highs_values[spos:timeout_pos, col]
        )
        if not np.isfinite(adverse).all():
            self.termination_counts["MISSING_DATA"] += 1
            first_bad = spos + int(np.argmax(~np.isfinite(adverse)))
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                    timestamp=grid[first_bad], decision_time=tw_index[i],
                    signal_time=sig_index[i], execution_bound=self.execution_bound,
                )
            )
            return True
        closes_window = closes_values[spos:timeout_pos, col]
        if not np.isfinite(closes_window).all():
            self.termination_counts["MISSING_DATA"] += 1
            first_bad = spos + int(np.argmax(~np.isfinite(closes_window)))
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                    timestamp=grid[first_bad], decision_time=tw_index[i],
                    signal_time=sig_index[i], execution_bound=self.execution_bound,
                )
            )
            return True
        liquidity_cost_bps = self._taker_cost_bps(gcol)
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
            block = self._bar_viability_gap(
                fill_pos=fill_pos, col=col, sym=sym,
                decision_ts=tw_index[i], signal_ts=sig_index[i],
                submit_ns=int(grid[submit_pos].value), signal_ns=int(sig_index[i].value),
                avail_submit_ns=int(self._w_avail_ns[submit_pos]),
            )
            if block is not None:
                current_units = float(self.units_arr[gcol])
                return self._block_fill(
                    block, prior_units=current_units,
                    net_units=float(peg_entry_units + net_units - current_units),
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
            fill_time = pd.Timestamp(int(self._w_avail_ns[fill_pos]), unit="ns", tz="UTC")
            submit_time = pd.Timestamp(int(self._w_avail_ns[submit_pos]), unit="ns", tz="UTC")
            mark_price = float(marks_values[fill_pos, col])
            if np.isfinite(mark_price):
                self.last_prices_arr[gcol] = mark_price
            if not (np.isfinite(qty) and np.isfinite(fill_price)):
                raise DataIntegrityError(
                    "non-finite fill sizing breaches the capital accounting invariant "
                    f"(symbol={sym!r} ts={fill_time!r} weight={weight!r} equity={equity!r} "
                    f"decision_price={decision_price!r} qty={qty!r} fill_price={fill_price!r})"
                )
            self.cash -= qty * fill_price
            fee = fee_bps / 1e4 * abs(qty) * fill_price
            self.cash -= fee
            self.units_arr[gcol] += qty
            self.fill_bar_ns.append(int(grid[fill_pos].value))
            self.fill_gcol.append(int(gcol))
            self._mirror_pending.append(
                (int(grid[fill_pos].value), int(gcol), float(qty), float(fill_price), float(fee_bps))
            )
            pre_trade_equity = self._equity_at(gpos)
            self.fill_ts.append(fill_time)
            self.fill_symbol.append(sym)
            self.fill_qty.append(qty)
            self.fill_price.append(fill_price)
            self.fill_fee_bps.append(fee_bps)
            self.fill_reason.append(reason)
            self.fill_pre_trade_equity.append(pre_trade_equity)
            self.fill_times.append(fill_time)
            self.submit_times.append(submit_time)
            if self.retain_event_snapshots:
                marks_row = np.full(n_cols, np.nan, dtype="float64")
                marks_row[gpos] = marks_values[fill_pos]
                self.units_after_events.append((fill_time, self.units_arr.copy()))
                self.notional_after_events.append((fill_time, self.units_arr * marks_row))
        return True
        return True


    def _consume_fill_strict_touch(self, timeout_pos: int, spos: int, sym: str, grid: pd.DatetimeIndex, side: int, lows_values: np.ndarray, col: int, highs_values: np.ndarray, decision_price: float, n_grid: int, grid_ns: np.ndarray, timeout_ns: int, closes_values: np.ndarray, gcol: int, tw_index: pd.DatetimeIndex, sig_index: pd.DatetimeIndex, i: int) -> tuple[bool, int, float, float, str, float, np.ndarray]:
        """Resolve a strict/touch/timeout fill; False carries vars for booking."""
        timeout_close = float("nan")
        if timeout_pos <= spos:
            self.termination_counts["MISSING_DATA"] += 1
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                    timestamp=grid[spos], decision_time=tw_index[i],
                    signal_time=sig_index[i], execution_bound=self.execution_bound,
                )
            )
            return True, 0, float("nan"), 0.0, "", float("nan"), np.empty(0, dtype="float64")
        adverse = (
            lows_values[spos:timeout_pos, col]
            if side == 1
            else highs_values[spos:timeout_pos, col]
        )
        if not np.isfinite(adverse).all():
            self.termination_counts["MISSING_DATA"] += 1
            first_bad = spos + int(np.argmax(~np.isfinite(adverse)))
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                    timestamp=grid[first_bad], decision_time=tw_index[i],
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
            if timeout_pos >= n_grid or grid_ns[timeout_pos] != timeout_ns:
                self.termination_counts["MISSING_DATA"] += 1
                self.data_gaps.append(
                    ExecutionDataGap(
                        code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                        timestamp=grid[spos], decision_time=tw_index[i],
                        signal_time=sig_index[i], execution_bound=self.execution_bound,
                    )
                )
                return True, 0, float("nan"), 0.0, "", float("nan"), np.empty(0, dtype="float64")
            timeout_close = float(closes_values[timeout_pos, col])
            if not np.isfinite(timeout_close):
                self.termination_counts["MISSING_DATA"] += 1
                self.data_gaps.append(
                    ExecutionDataGap(
                        code="MISSING_ACTIVE_ORDER_OHLCV", symbol=sym,
                        timestamp=grid[timeout_pos], decision_time=tw_index[i],
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


    def _consume_append_ledger(self, grid_ns: np.ndarray, n_grid: int, local_cols: list[str], fill_start: int, n_local: int, marks_values: np.ndarray, gpos: np.ndarray, grid: pd.DatetimeIndex, funding_matrix: np.ndarray, bar_ns: int, closes_values: np.ndarray) -> None:
        """Append reconciled inventory accounting with bounded scratch storage.

        The first retained bar may use only the immediately preceding consumed,
        published valuation mark for continuity, because window splitting must
        not create fictitious missing data.

        Args:
            grid_ns: Nanosecond execution labels.
            n_grid: Number of labels.
            local_cols: Ordered local symbols.
            fill_start: First fill belonging to this window.
            n_local: Local symbol count.
            marks_values: Published valuation prices.
            gpos: Local-to-canonical positions.
            grid: UTC execution labels.
            funding_matrix: Existing funding and knowledge representation.
            bar_ns: Execution interval in nanoseconds.
            closes_values: Observed closes.

        Returns:
            None; append chronological ledger series and preserve carried state.

        Raises:
            DataIntegrityError: Accounting or required provenance is invalid.
        """

        # ---- streamed ledger chunk over [ledger_start_ns, grid end] ----
        p0 = 0 if self.ledger_start_ns is None else int(np.searchsorted(grid_ns, self.ledger_start_ns, side="left"))
        if p0 >= n_grid:
            raise DataIntegrityError("execution windows must not leave an uncovered grid gap")
        chunk_len = n_grid - p0
        if chunk_len:
            sym_to_local = {s: j for j, s in enumerate(local_cols)}
            n_fill = len(self.fill_ts) - fill_start
            turnover_pos_arr: np.ndarray
            turnover_qty_arr: np.ndarray
            turnover_price_arr: np.ndarray
            if n_fill:
                wf_pos = np.searchsorted(
                    grid_ns, np.asarray(self.fill_bar_ns[fill_start:], dtype="int64"), side="left",
                )
                wf_j = np.asarray(
                    [sym_to_local[s] for s in self.fill_symbol[fill_start:]],
                    dtype=np.intp,
                )
                wf_qty = np.asarray(self.fill_qty[fill_start:], dtype="float64")
                wf_price = np.asarray(self.fill_price[fill_start:], dtype="float64")
                wf_fee = np.asarray(self.fill_fee_bps[fill_start:], dtype="float64")
                wf_fee_amt = wf_fee / 1e4 * np.abs(wf_qty) * wf_price
                fill_flow = np.zeros(n_grid, dtype="float64")
                fee_by_ts = np.zeros(n_grid, dtype="float64")
                np.add.at(fill_flow, wf_pos, -(wf_qty * wf_price + wf_fee_amt))
                np.add.at(fee_by_ts, wf_pos, wf_fee_amt)
                turnover_pos_arr = wf_pos
                turnover_qty_arr = wf_qty
                turnover_price_arr = wf_price
            else:
                wf_pos = np.empty(0, dtype=np.intp)
                wf_j = np.empty(0, dtype=np.intp)
                wf_qty = np.empty(0, dtype="float64")
                fill_flow = np.zeros(n_grid, dtype="float64")
                fee_by_ts = np.zeros(n_grid, dtype="float64")
                turnover_pos_arr = np.empty(0, dtype=np.intp)
                turnover_qty_arr = np.empty(0, dtype="float64")
                turnover_price_arr = np.empty(0, dtype="float64")

            # Bounded per-symbol pass: each column's arithmetic matches the
            # legacy window-by-symbol plane, accumulated left to right in
            # canonical order. Scratch stays O(window bars).
            row_idx = np.arange(n_grid)
            mtm_arr = np.zeros(n_grid, dtype="float64")
            funding_arr = np.zeros(n_grid, dtype="float64")
            notional_arr = np.zeros(n_grid, dtype="float64")
            notional_before_arr = np.zeros(n_grid, dtype="float64")
            start_units = np.asarray(self.ledger_units[gpos], dtype="float64")
            start_valid = np.asarray(self.last_valid_mark[gpos], dtype="float64")
            end_units = np.empty(n_local, dtype="float64")
            end_valid = np.empty(n_local, dtype="float64")
            for j in range(n_local):
                d_col = np.zeros(n_grid, dtype="float64")
                sel = wf_j == j
                if bool(sel.any()):
                    np.add.at(d_col, wf_pos[sel], wf_qty[sel])
                units = np.cumsum(d_col) + float(start_units[j])
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
                    prior_bar_ns = int(self.ledger_start_ns) - int(bar_ns)
                    curr_mark = float(marks_col[0])
                    curr_avail = int(self._w_mark_avail[0, j])
                    curr_ok = bool(
                        np.isfinite(curr_mark)
                        and curr_mark > 0.0
                        and curr_avail <= int(grid_ns[0])
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
                if bool(((~usable & held & (funding_col != 0.0)) & kept).any()):
                    self.ledger_valid = False
                    self.invalid_reasons.add("MISSING_DATA")
                    if self.first_held_funding is None:
                        first_pos = int(np.flatnonzero((~usable & held & (funding_col != 0.0)) & kept)[0])
                        self.first_held_funding = (local_cols[j], grid[first_pos])
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
            chunk_flow = fill_flow[p0:] - funding_arr[p0:]
            cash_after = self.ledger_cash + np.cumsum(chunk_flow)
            cash_pre_fill = np.empty(chunk_len, dtype="float64")
            cash_pre_fill[0] = self.ledger_cash - funding_arr[p0]
            cash_pre_fill[1:] = cash_after[:-1] - funding_arr[p0 + 1 :]
            equity_arr = cash_after + notional_arr[p0:]
            turnover_arr = np.zeros(chunk_len, dtype="float64")
            if len(turnover_pos_arr):
                pre_trade_equity = (
                    cash_pre_fill[turnover_pos_arr - p0] + notional_before_arr[turnover_pos_arr]
                )
                if not np.isfinite(pre_trade_equity).all() or (pre_trade_equity <= 0).any():
                    bad = np.where(~np.isfinite(pre_trade_equity) | (pre_trade_equity <= 0))[0]
                    bad_pos = turnover_pos_arr[bad[0]]
                    raise DataIntegrityError(
                        f"pre-trade equity must be positive and finite "
                        f"(ts={grid[bad_pos]!r} pre_trade_equity={pre_trade_equity[bad[0]]!r})"
                    )
                np.add.at(
                    turnover_arr, turnover_pos_arr - p0,
                    np.abs(turnover_qty_arr * turnover_price_arr) / pre_trade_equity,
                )
            if not np.isfinite(equity_arr).all() or (equity_arr <= 0).any():
                raise DataIntegrityError("simulated inventory equity must be finite and strictly positive")
            self.equity_chunks.append(equity_arr)
            self.equity_times.append(grid[p0:])
            self.mtm_chunks.append(mtm_arr[p0:])
            self.funding_chunks.append(funding_arr[p0:])
            self.fee_chunks.append(fee_by_ts[p0:])
            self.turnover_chunks.append(turnover_arr)
            self.ledger_cash = float(cash_after[-1])
            if self._w_avail_explicit:
                self._ledger_avail_chunks.append(np.asarray(self._w_avail_ns[p0:], dtype="int64"))
            else:
                self._ledger_avail_complete = False
        if len(gpos):
            self._final_mark[gpos] = np.asarray(marks_values[-1], dtype="float64")
            self._final_mark_avail_ns[gpos] = np.asarray(self._w_mark_avail[-1], dtype="int64")
        self._settle_mirror_window(grid_ns, n_grid, p0, marks_values, funding_matrix, self._w_fknown, gpos, grid)
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
        self.ledger_start_ns = int(grid_ns[-1]) + bar_ns


    def _settle_mirror_window(
        self,
        grid_ns: np.ndarray,
        n_grid: int,
        p0: int,
        marks_values: np.ndarray,
        funding_matrix: np.ndarray,
        funding_known: np.ndarray,
        gpos: np.ndarray,
        grid: pd.DatetimeIndex,
    ) -> None:
        """Replay the kept bars through the causal mirror in timestamp order.

        Each bar settles mark then known funding on pre-fill inventory before
        that bar's queued fills are applied (INV-EVENT-ORDER), so the mirror
        reproduces the independent ledger's economics exactly. Fills queued
        before the kept region (window-overlap backlog) join inventory
        without cash flow, mirroring the ledger chunk handoff. Unknown
        funding over held inventory records MISSING_HELD_FUNDING and
        invalidates instead of settling an invented cost.
        """
        state = self.accounting_state
        p0_ns = int(grid_ns[p0])
        queue = sorted(self._mirror_pending, key=lambda entry: entry[0])
        self._mirror_pending = []
        qi = 0
        while qi < len(queue) and queue[qi][0] < p0_ns:
            state.units[int(queue[qi][1])] += float(queue[qi][2])
            qi += 1
        gmarks = np.full(self.n_cols, np.nan, dtype="float64")
        grates = np.zeros(self.n_cols, dtype="float64")
        gknown = np.ones(self.n_cols, dtype=bool)
        all_known = np.ones(self.n_cols, dtype=bool)
        for b in range(int(p0), int(n_grid)):
            bns = int(grid_ns[b])
            gmarks[gpos] = marks_values[b]
            grates[gpos] = funding_matrix[b]
            gknown[gpos] = funding_known[b]
            held_unknown = (np.abs(state.units) >= QTY_EPS) & ~gknown
            if bool(held_unknown.any()):
                self.ledger_valid = False
                self.invalid_reasons.add("MISSING_DATA")
                witness = int(np.flatnonzero(held_unknown)[0])
                self.data_gaps.append(
                    ExecutionDataGap(
                        code="MISSING_HELD_FUNDING", symbol=self.columns[witness],
                        timestamp=grid[b], execution_bound=self.execution_bound,
                    )
                )
                state.advance_to(
                    event_ns=bns, marks=gmarks,
                    funding_rates=np.where(gknown, grates, 0.0), funding_known=all_known,
                )
            else:
                state.advance_to(event_ns=bns, marks=gmarks, funding_rates=grates, funding_known=gknown)
            while qi < len(queue) and queue[qi][0] == bns:
                state.apply_fill(
                    symbol_index=int(queue[qi][1]), quantity_delta=float(queue[qi][2]),
                    fill_price=float(queue[qi][3]), fee_bps=float(queue[qi][4]),
                )
                qi += 1

    def _consume_update_spreads(self, highs_values: np.ndarray, lows_values: np.ndarray, gpos: np.ndarray) -> None:
        """Roll the liquidity-aware spread estimate forward, causally."""

        # Liquidity-aware spread EWMA update -- strictly AFTER this window's
        # fills were priced, so a window's own bars can never price its own
        # costs (causality). A degenerate (nan) estimate carries the prior
        # value forward instead of poisoning it.
        if self.spec.liquidity_cost_model == "corwin_schultz":
            est = _microstructure.corwin_schultz_half_spread_bps(highs_values, lows_values)
            old = self.half_spread_bps[gpos]
            alpha = self.spec.spread_ewma_alpha
            updated = alpha * est + (1.0 - alpha) * old
            merged = np.where(np.isnan(est), old, updated)
            self.half_spread_bps[gpos] = np.where(np.isnan(old), est, merged)

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

    def _observe_spread_partition(
        self,
        highs_values: np.ndarray,
        lows_values: np.ndarray,
        gpos: np.ndarray,
        grid_ns: np.ndarray,
        bar_ns: int,
        logical_partition: tuple[int, int] | None,
    ) -> None:
        """Record this window's bars into the logical cost-clock observations.

        Untagged windows keep the legacy immediate per-window update.
        """
        if self.spec.liquidity_cost_model != "corwin_schultz":
            return
        if logical_partition is None:
            self._consume_update_spreads(highs_values, lows_values, gpos)
            return
        carry_high = self._spread_carry_high[gpos]
        carry_low = self._spread_carry_low[gpos]
        carry_ns = self._spread_carry_ns[gpos]
        adjacent = (carry_ns + bar_ns == grid_ns[0]) & np.isfinite(carry_high) & np.isfinite(carry_low)
        ext_high = np.vstack([carry_high[None, :], highs_values])
        ext_low = np.vstack([carry_low[None, :], lows_values])
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
        self._spread_carry_high[gpos] = highs_values[-1]
        self._spread_carry_low[gpos] = lows_values[-1]
        self._spread_carry_ns[gpos] = grid_ns[-1]
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
        if self.first_held_funding is not None:
            funded_incomplete.add(self.first_held_funding[0])
        records: list[TerminalPositionEvidence] = []
        for col in range(self.n_cols):
            quantity = float(self.units_arr[col])
            if abs(quantity) < 1e-12:
                continue
            sym = self.columns[col]
            funding_complete = sym not in funded_incomplete
            final_mark = float(self._final_mark[col])
            final_avail_ns = int(self._final_mark_avail_ns[col])
            fresh = (
                np.isfinite(final_mark)
                and final_mark > 0.0
                and final_avail_ns >= 0
                and final_avail_ns <= cutoff_ns
            )
            if fresh and funding_complete:
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
            self.ledger_valid = False
            self.invalid_reasons.add("MISSING_DATA")
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
        for event in self._settled_events:
            settled_mark = float(event.settlement_price)
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
            }
        )[
            [
                "timestamp", "symbol", "quantity_delta", "fill_price",
                "fee_bps", "reason", "pre_trade_equity",
            ]
        ]
        if simulated_fills.empty:
            simulated_fills = simulated_fills.astype(
                {"quantity_delta": "float64", "fill_price": "float64", "fee_bps": "float64"}
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
        )
        if self.first_held_mark is not None:
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_HELD_MARK", symbol=self.first_held_mark[0],
                    timestamp=self.first_held_mark[1], execution_bound=self.execution_bound,
                )
            )
        if self.first_held_funding is not None:
            self.data_gaps.append(
                ExecutionDataGap(
                    code="MISSING_HELD_FUNDING", symbol=self.first_held_funding[0],
                    timestamp=self.first_held_funding[1], execution_bound=self.execution_bound,
                )
            )
        terminal_positions = self._terminal_positions(grid_end)
        self.data_gaps.sort(key=lambda g: (g.timestamp, g.code, g.symbol))
        ledger = dataclasses.replace(
            ledger,
            primary_valid=self.ledger_valid,
            invalid_reasons=tuple(sorted(self.invalid_reasons)),
            data_gaps=tuple(self.data_gaps),
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
        )
