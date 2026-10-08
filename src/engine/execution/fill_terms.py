"""Single-fill shortfall and fee/spread decomposition against the decision price."""

from __future__ import annotations

import numpy as np

from src.core.types import ExecutionSpec

from . import _ExecutionBound
from .microstructure import passive_fill_shortfall_bps


def resolve_single_fill_terms(
    *,
    execution_bound: _ExecutionBound,
    spec: ExecutionSpec,
    decision_price: float,
    fill_price: float,
    timeout_close: float,
    adverse: np.ndarray,
    side: int,
    fee_bps: float,
    taker_cost_bps: float,
    reason: str,
) -> tuple[float, float, float, float]:
    """Decompose one booked fill into shortfall and ledger term components.

    Returns ``(shortfall_bps, term_price, fee_term_bps, spread_term_bps)`` with
    fee+spread+delay kept exactly equal to shortfall on every path.
    """
    if execution_bound == "OHLCV_IMMEDIATE_TAKER":
        shortfall = side * (fill_price / decision_price - 1.0) * 1e4 + fee_bps
        if spec.liquidity_cost_model == "corwin_schultz":
            return shortfall, fill_price, spec.taker_fee_bps, fee_bps - spec.taker_fee_bps
        # Flat model: the fixed slippage folds into the fee term and the spread term stays zero.
        return shortfall, fill_price, fee_bps, 0.0
    shortfall = passive_fill_shortfall_bps(
        decision_price, adverse, timeout_close, side, spec,
        taker_cost_bps=taker_cost_bps,
    )
    # The residual after timing is the all-in fee component; deriving it keeps
    # fee+spread+delay exact even on degenerate exact-touch fills.
    anchor = fill_price if reason == "passive_fill" else timeout_close
    return shortfall, anchor, shortfall - side * (anchor / decision_price - 1.0) * 1e4, 0.0
