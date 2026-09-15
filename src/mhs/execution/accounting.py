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

from src.common.errors import DataIntegrityError

from .contracts import SimulatedInventoryLedgerResult

# 먼지 수량 임계, finalize의 기존 1e-12 및 net_units 임계와 동일
QTY_EPS: float = 1e-12


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
        """Settle one mark-then-funding event; returns the funding charged."""
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
        charged = float(np.sum(np.asarray(funding_rates, dtype="float64") * self.units * priced))
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
