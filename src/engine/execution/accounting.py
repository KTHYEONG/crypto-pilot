"""Causal cash-and-inventory accounting state (P1_CAUSAL_EXECUTION_CORE).

Single-MTM invariant (INV-ACCOUNTING-SINGLE-MTM): cash moves only through
fill principal, fees, and funding settlement. Price moves never touch cash;
they surface exactly once in ``equity()`` as ``cash + sum(units * marks)``.

Event order inside one timestamp (INV-EVENT-ORDER) is mark, then known
funding settlement on the pre-fill inventory, then fill, then fee. Funding
for inventory held through an unknown-funding bar fails closed with
``DataIntegrityError``; the replay engine maps that primitive into
``MISSING_HELD_FUNDING`` gaps plus primary-invalid instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.common.errors import DataIntegrityError

from .contracts import SimulatedInventoryLedgerResult

# Dust quantity threshold matching finalize and net_units threshold.
QTY_EPS: float = 1e-12


@dataclass(frozen=True, slots=True)
class CausalWindowSettlement:
    """Outcome of settling one window's kept bars through the causal mirror.

    The replay engine turns ``gap_bar_offsets``/``gap_witness_columns`` into
    ``MISSING_HELD_FUNDING`` gaps plus primary-invalid; this record carries only
    positions so the accounting primitive never builds timestamps or symbols.

    Attributes:
        gap_bar_offsets: ``intp``, strictly ascending offsets into the settled bar range of every
            bar whose pre-fill held inventory (``|units| >= QTY_EPS``) met unknown funding.
        gap_witness_columns: ``intp``, same length; per gap bar the lowest canonical column among
            the held unknown-funding columns.
        funding_charged: ``float64``, shape ``(n_bars,)``; the amount subtracted from cash at each
            bar before that bar's fills (known funding only on gap bars).
        fills_applied: Fills settled on a bar of the range (cash and inventory moved).
        backlog_applied: Fills dated before the first bar that joined inventory without cash flow.
    """

    gap_bar_offsets: npt.NDArray[np.intp]
    gap_witness_columns: npt.NDArray[np.intp]
    funding_charged: npt.NDArray[np.float64]
    fills_applied: int
    backlog_applied: int


def _coerce_bar_ns(bar_ns: npt.NDArray[np.int64], last_event_ns: int | None) -> npt.NDArray[np.int64]:
    bars = np.asarray(bar_ns, dtype=np.int64)
    if bars.ndim != 1 or bars.size == 0:
        raise DataIntegrityError("settle_window requires a non-empty 1-D bar_ns")
    if bars.size > 1 and not bool(np.all(bars[1:] > bars[:-1])):
        raise DataIntegrityError("settle_window requires strictly increasing bar_ns")
    if last_event_ns is not None and int(bars[0]) < int(last_event_ns):
        raise DataIntegrityError(
            f"causal events must be monotonically increasing (last={last_event_ns} event={int(bars[0])})"
        )
    return bars


def _coerce_columns(columns: npt.NDArray[np.intp], n_cols: int) -> npt.NDArray[np.intp]:
    cols = np.asarray(columns, dtype=np.intp)
    if cols.ndim != 1:
        raise DataIntegrityError("settle_window requires a 1-D columns array")
    if cols.size and (bool((cols < 0).any()) or bool((cols >= n_cols).any())):
        raise DataIntegrityError("settle_window columns out of range")
    if cols.size != np.unique(cols).size:
        raise DataIntegrityError("settle_window columns must be unique")
    return cols


def _check_plane_shapes(
    nb: int,
    nl: int,
    marks: npt.NDArray[np.float64],
    funding_rates: npt.NDArray[np.float64],
    funding_known: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    m = np.asarray(marks, dtype=np.float64)
    r = np.asarray(funding_rates, dtype=np.float64)
    k = np.asarray(funding_known, dtype=bool)
    if m.shape != (nb, nl) or r.shape != (nb, nl) or k.shape != (nb, nl):
        raise DataIntegrityError("settle_window array shapes disagree")
    return m, r, k


def _stable_sort_fills(
    fill_ns: npt.NDArray[np.int64],
    fill_columns: npt.NDArray[np.intp],
    fill_quantities: npt.NDArray[np.float64],
    fill_prices: npt.NDArray[np.float64],
    fill_fee_bps: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.intp],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    ns = np.asarray(fill_ns, dtype=np.int64).reshape(-1)
    col = np.asarray(fill_columns, dtype=np.intp).reshape(-1)
    qty = np.asarray(fill_quantities, dtype=np.float64).reshape(-1)
    px = np.asarray(fill_prices, dtype=np.float64).reshape(-1)
    fee = np.asarray(fill_fee_bps, dtype=np.float64).reshape(-1)
    if not (col.size == qty.size == px.size == fee.size == ns.size):
        raise DataIntegrityError("settle_window array shapes disagree")
    order = np.argsort(ns, kind="stable")
    return ns[order], col[order], qty[order], px[order], fee[order]


def _validate_applied_fills(
    s_ns: npt.NDArray[np.int64],
    s_col: npt.NDArray[np.intp],
    backlog_count: int,
    bars: npt.NDArray[np.int64],
    cols: npt.NDArray[np.intp],
    n_cols: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    nb = int(bars.size)
    nf_applied = int(s_ns.size - backlog_count)
    if nf_applied == 0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    a_ns = s_ns[backlog_count:]
    a_col = s_col[backlog_count:]
    if bool((a_col < 0).any()) or bool((a_col >= n_cols).any()):
        raise DataIntegrityError("settle_window fill column out of range")
    pos = np.searchsorted(bars, a_ns, side="left")
    if bool((pos >= nb).any()) or not bool(np.all(bars[pos] == a_ns)):
        raise DataIntegrityError("settle_window fill is not an exact bar label")
    lut = np.full(n_cols, -1, dtype=np.intp)
    if cols.size:
        lut[cols] = np.arange(cols.size, dtype=np.intp)
    local = lut[a_col]
    if bool((local < 0).any()):
        raise DataIntegrityError("settle_window fill column not in columns")
    return pos.astype(np.intp, copy=False), local


def _pre_fill_units(
    nb: int,
    nl: int,
    start_local: npt.NDArray[np.float64],
    f_local: npt.NDArray[np.intp],
    fill_bar: npt.NDArray[np.intp],
    f_qty: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    rows = np.empty((nb, nl), dtype=np.float64)
    for j in range(nl):
        sel = np.flatnonzero(f_local == j)
        col = np.full(nb, float(start_local[j]), dtype=np.float64)
        if sel.size:
            seed = np.concatenate(([float(start_local[j])], f_qty[sel]))
            levels = np.add.accumulate(seed)[1:]
            bars_j = fill_bar[sel]
            for k in range(len(sel)):
                b = int(bars_j[k]) + 1
                if b < nb:
                    col[b:] = float(levels[k])
        rows[:, j] = col
    return rows


def _fold_charges(
    terms: npt.NDArray[np.float64],
    canon_order: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    nb = int(terms.shape[0])
    charges = np.zeros(nb, dtype=np.float64)
    for j in canon_order.tolist():
        charges += terms[:, int(j)]
    return charges


def _gap_outputs(
    held_unknown: npt.NDArray[np.bool_],
    cols: npt.NDArray[np.intp],
    canon_order: npt.NDArray[np.intp],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    gap_mask = held_unknown.any(axis=1)
    offsets = np.flatnonzero(gap_mask).astype(np.intp, copy=False)
    if offsets.size == 0:
        return offsets, np.empty(0, dtype=np.intp)
    ordered = held_unknown[offsets][:, canon_order]
    first = np.argmax(ordered, axis=1)
    witness = cols[canon_order[first]]
    return offsets, witness.astype(np.intp, copy=False)


def _fold_cash(
    cash: float,
    charges: npt.NDArray[np.float64],
    fill_bar: npt.NDArray[np.intp],
    f_qty: npt.NDArray[np.float64],
    f_px: npt.NDArray[np.float64],
    f_fee: npt.NDArray[np.float64],
) -> float:
    nb = int(charges.size)
    nf = int(fill_bar.size)
    if nf == 0:
        seq = np.concatenate(([cash], charges))
        return float(np.subtract.accumulate(seq)[-1])
    fee = f_fee / 1e4 * np.abs(f_qty) * f_px
    fill_delta = f_qty * f_px + fee
    bar_slots = np.arange(nb) + np.searchsorted(fill_bar, np.arange(nb), side="left")
    deltas = np.empty(nb + nf, dtype=np.float64)
    deltas[bar_slots] = charges
    deltas[fill_bar + 1 + np.arange(nf)] = fill_delta
    seq = np.concatenate(([cash], deltas))
    return float(np.subtract.accumulate(seq)[-1])


def _backlog_start(
    units: npt.NDArray[np.float64],
    cols: npt.NDArray[np.intp],
    s_col: npt.NDArray[np.intp],
    s_qty: npt.NDArray[np.float64],
    backlog_count: int,
) -> npt.NDArray[np.float64]:
    start = np.asarray(units[cols], dtype=np.float64)
    if backlog_count == 0 or cols.size == 0:
        return start
    lut = np.full(int(units.size), -1, dtype=np.intp)
    lut[cols] = np.arange(cols.size, dtype=np.intp)
    for i in range(backlog_count):
        li = int(lut[int(s_col[i])])
        if li >= 0:
            start[li] += float(s_qty[i])
    return start


def _last_finite_marks(
    marks: npt.NDArray[np.float64],
    current: npt.NDArray[np.float64],
    cols: npt.NDArray[np.intp],
) -> None:
    if cols.size == 0 or marks.shape[0] == 0:
        return
    finite = np.isfinite(marks)
    has = finite.any(axis=0)
    if not bool(has.any()):
        return
    rev = finite[::-1]
    last_idx = int(marks.shape[0]) - 1 - np.argmax(rev, axis=0).astype(np.intp)
    picked = marks[last_idx, np.arange(cols.size)]
    current[cols] = np.where(has, picked, current[cols])


@dataclass(slots=True)
class CausalPortfolioState:
    """Mutable causal portfolio state shadowing the replay fill track."""

    cash: float
    units: np.ndarray
    last_marks: np.ndarray
    last_event_ns: int | None

    def equity(self, marks: np.ndarray | None = None) -> float:
        """Single-MTM equity: cash plus inventory at the reference marks."""
        ref = self.last_marks if marks is None else marks
        priced = np.where(np.isfinite(ref), ref, 0.0)
        return float(self.cash) + float(np.sum(self.units * priced))

    def advance_to(
        self,
        *,
        event_ns: int,
        marks: np.ndarray,
        funding_rates: np.ndarray,
        funding_known: np.ndarray,
    ) -> float:
        """Settle one mark-then-funding event; returns the funding charged.

        The charge is the left-to-right fold over canonical column order of
        ``(rate * units) * mark`` (non-finite mark contributes 0.0) -- the same fold
        ``settle_window`` uses, so a per-bar loop over this method is the exact scalar
        reference for the window path. Unknown funding over held inventory raises.
        """
        if self.last_event_ns is not None and event_ns < self.last_event_ns:
            raise DataIntegrityError(
                f"causal events must be monotonically increasing (last={self.last_event_ns} event={event_ns})"
            )
        finite = np.isfinite(marks)
        self.last_marks = np.where(finite, marks, self.last_marks)
        held_unknown = (np.abs(self.units) >= QTY_EPS) & ~np.asarray(funding_known, dtype=bool)
        if bool(np.any(held_unknown)):
            raise DataIntegrityError(
                "unknown funding for a held position fails closed: cannot settle funding"
            )
        priced = np.where(finite, marks, 0.0)
        rates = np.asarray(funding_rates, dtype="float64")
        units = np.asarray(self.units, dtype="float64")
        charged = 0.0
        for j in range(int(units.size)):
            charged += float((float(rates[j]) * float(units[j])) * float(priced[j]))
        self.cash -= charged
        self.last_event_ns = int(event_ns)
        return charged

    def apply_fill(
        self,
        *,
        symbol_index: int,
        quantity_delta: float,
        fill_price: float,
        fee_bps: float,
    ) -> float:
        """Book one fill against cash and inventory; returns the fee charged."""
        qty = float(quantity_delta)
        price = float(fill_price)
        fee = float(fee_bps) / 1e4 * abs(qty) * price
        self.cash -= qty * price + fee
        self.units[int(symbol_index)] += qty
        return fee

    def settle_window(
        self,
        *,
        bar_ns: np.ndarray,
        columns: np.ndarray,
        marks: np.ndarray,
        funding_rates: np.ndarray,
        funding_known: np.ndarray,
        fill_ns: np.ndarray,
        fill_columns: np.ndarray,
        fill_quantities: np.ndarray,
        fill_prices: np.ndarray,
        fill_fee_bps: np.ndarray,
    ) -> CausalWindowSettlement:
        """Settle a contiguous run of bars and their queued fills in one vectorized pass.

        Bit-identical to calling ``advance_to`` once per bar and ``apply_fill`` once per fill in
        timestamp order (INV-EVENT-ORDER: mark, then known funding on pre-fill inventory, then fill
        principal plus fee), except that unknown funding over held inventory does not raise: the bar
        settles known funding only and is reported in ``gap_bar_offsets`` so the replay engine can
        record ``MISSING_HELD_FUNDING`` and invalidate instead of crashing or inventing a cost.
        Held inventory over an unpriceable mark with a nonzero rate is reported the same way:
        the charge prices at zero (nothing is booked) and the bar is a funding gap as well.

        Arithmetic is fixed so results never depend on numpy reduction internals or vector width:
        the per-bar charge is the left-to-right fold in ascending canonical column order of
        ``(rate * units) * mark`` (non-finite mark contributes 0.0), and cash is the sequential
        left fold of ``-charge`` per bar followed by ``-(quantity * price + fee)`` per fill in
        queue order, with ``fee = fee_bps / 1e4 * abs(quantity) * price``. Columns outside
        ``columns`` contribute exactly zero, so settling at roster width equals settling at
        canonical width. Duplicate entries in ``columns`` are rejected.

        Args:
            bar_ns: ``int64`` bar labels, strictly increasing, length ``n_bars >= 1``.
            columns: ``intp`` unique canonical positions of the roster (window order; production
                rosters are ascending, directly built windows need not be), length ``n_local``.
            marks: ``float64`` ``(n_bars, n_local)`` valuation marks; NaN means unobserved.
            funding_rates: ``float64`` ``(n_bars, n_local)`` finite per-bar funding rates.
            funding_known: ``bool`` ``(n_bars, n_local)`` funding knowledge.
            fill_ns: ``int64`` fill bar labels in booking order (stable-sorted here by time).
            fill_columns: ``intp`` canonical column of each fill; must be in ``columns`` unless the
                fill is dated before ``bar_ns[0]``.
            fill_quantities: ``float64`` signed unit deltas.
            fill_prices: ``float64`` execution prices.
            fill_fee_bps: ``float64`` all-in fee rates in basis points.

        Returns:
            ``CausalWindowSettlement`` for the range; ``cash``, ``units``, ``last_marks`` and
            ``last_event_ns`` are updated in place.

        Raises:
            DataIntegrityError: ``bar_ns`` is empty or not strictly increasing; ``bar_ns[0]`` precedes
                ``last_event_ns``; array shapes disagree; a fill dated at or after ``bar_ns[0]`` is
                not an exact bar label or lies after ``bar_ns[-1]``; a non-backlog fill column is not
                in ``columns``; ``columns`` has duplicates. Raised before any state mutation.
        """
        n_cols = int(self.units.size)
        bars = _coerce_bar_ns(bar_ns, self.last_event_ns)
        cols = _coerce_columns(columns, n_cols)
        nb = int(bars.size)
        nl = int(cols.size)
        m, r, k = _check_plane_shapes(nb, nl, marks, funding_rates, funding_known)
        s_ns, s_col, s_qty, s_px, s_fee = _stable_sort_fills(
            fill_ns, fill_columns, fill_quantities, fill_prices, fill_fee_bps
        )
        backlog_count = int(np.searchsorted(s_ns, int(bars[0]), side="left"))
        for i in range(backlog_count):
            c = int(s_col[i])
            if c < 0 or c >= n_cols:
                raise DataIntegrityError("settle_window fill column out of range")
        fill_bar, f_local = _validate_applied_fills(s_ns, s_col, backlog_count, bars, cols, n_cols)
        canon_order = np.argsort(cols, kind="stable").astype(np.intp, copy=False) if nl else np.empty(0, dtype=np.intp)
        if nl:
            start_local = _backlog_start(self.units, cols, s_col, s_qty, backlog_count)
            f_qty = s_qty[backlog_count:]
            units_rows = _pre_fill_units(nb, nl, start_local, f_local, fill_bar, f_qty)
            unpriceable = ~np.isfinite(m) & (r != 0.0)
            held_unknown = (np.abs(units_rows) >= QTY_EPS) & (~k | unpriceable)
            gap_bar: npt.NDArray[np.bool_] = np.asarray(held_unknown.any(axis=1))
            eff_rates = np.where(gap_bar[:, None] & ~k, 0.0, r)
            priced = np.where(np.isfinite(m), m, 0.0)
            terms = eff_rates * units_rows * priced
            charges = _fold_charges(terms, canon_order)
            gap_offsets, witness = _gap_outputs(held_unknown, cols, canon_order)
        else:
            charges = np.zeros(nb, dtype=np.float64)
            gap_offsets = np.empty(0, dtype=np.intp)
            witness = np.empty(0, dtype=np.intp)
        if s_ns.size:
            f_qty_all = s_qty[backlog_count:]
            f_px_all = s_px[backlog_count:]
            f_fee_all = s_fee[backlog_count:]
        else:
            f_qty_all = np.empty(0, dtype=np.float64)
            f_px_all = np.empty(0, dtype=np.float64)
            f_fee_all = np.empty(0, dtype=np.float64)
        new_cash = _fold_cash(float(self.cash), charges, fill_bar, f_qty_all, f_px_all, f_fee_all)
        for i in range(backlog_count):
            self.units[int(s_col[i])] += float(s_qty[i])
        for t in range(int(fill_bar.size)):
            canon = int(s_col[backlog_count + t])
            self.units[canon] += float(s_qty[backlog_count + t])
        self.cash = float(new_cash)
        _last_finite_marks(m, self.last_marks, cols)
        self.last_event_ns = int(bars[-1])
        return CausalWindowSettlement(
            gap_bar_offsets=np.asarray(gap_offsets, dtype=np.intp),
            gap_witness_columns=np.asarray(witness, dtype=np.intp),
            funding_charged=np.asarray(charges, dtype=np.float64),
            fills_applied=int(fill_bar.size),
            backlog_applied=int(backlog_count),
        )


def reconcile_causal_state(
    state: CausalPortfolioState,
    ledger: SimulatedInventoryLedgerResult,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> None:
    """Fail-closed tripwire: the online causal state must match the ledger.

    Raises ``DataIntegrityError`` when the causal equity diverges from the
    independently recomputed ledger equity beyond the given tolerances.
    """
    expected = float(ledger.equity.iloc[-1]) if len(ledger.equity) else float("nan")
    actual = state.equity()
    if not bool(np.isclose(actual, expected, atol=atol, rtol=rtol)):
        raise DataIntegrityError(
            f"causal accounting diverged from ledger (state={actual!r} ledger={expected!r})"
        )
