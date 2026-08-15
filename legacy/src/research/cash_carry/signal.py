from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.research.cash_carry.contracts import (
    CarryCostModel,
    CarryHysteresisConfig,
    CarryMarketData,
)

CarryTarget = Literal["OPEN", "HOLD", "CLOSE"]


def round_trip_cost_frac(costs: CarryCostModel) -> float:
    """Fraction of notional consumed by one full open-and-close cycle.

    Each leg pays its venue fee on both entry and exit plus two slippage legs
    per side (spot buy slips up, perp sell slips down). The spec's cost model
    therefore prices a round trip at ``2*(spot_fee + perp_fee + 2*slippage)``.
    """
    one_way = costs.spot_fee_rate + costs.perp_fee_rate + 2 * costs.slippage_rate
    return 2 * one_way


def _settlement_readings(data: CarryMarketData) -> np.ndarray:
    """Per-settlement net-carry rates (funding - borrow), event-aligned.

    Each funding settlement event j at bar ``b_j`` yields the reading
    ``funding_j - borrow(b_{j-1}+1 .. b_j)``: the borrow accrued over the whole
    elapsed interval since the prior settlement event (or from the window start
    for the first event). This is the vectorized equivalent of the original
    causal net-carry computation, evaluated at each event's own bar.
    """
    grid = data.spot.index
    events = pd.DatetimeIndex(data.funding.index)
    funding_vals = pd.to_numeric(data.funding, errors="coerce").to_numpy(dtype=np.float64)
    borrow_arr = pd.to_numeric(data.borrow, errors="coerce").to_numpy(dtype=np.float64)
    bar_idx = np.asarray(grid.searchsorted(events, side="right"), dtype=np.int64) - 1
    borrow_cum = np.concatenate(([0.0], np.cumsum(borrow_arr)))
    prev_bar = np.concatenate((np.asarray([-1], dtype=np.int64), bar_idx[:-1]))
    borrow_cost = borrow_cum[bar_idx + 1] - borrow_cum[prev_bar + 1]
    return np.asarray(funding_vals - borrow_cost, dtype=np.float64)


def generate_cash_carry_target(
    data: CarryMarketData,
    decision_time: pd.Timestamp,
    settlements_since_open: int | None,
    costs: CarryCostModel,
    hysteresis: CarryHysteresisConfig,
) -> CarryTarget:
    """Causal same-asset carry state target with a cost-derived hysteresis band.

    Only funding and borrow observations settled no later than ``decision_time``
    are used, and a decision is only evaluated on a bar that contains a fresh
    funding settlement: on bars without one the current state is preserved
    (``HOLD``). ``settlements_since_open=None`` means flat (replaces
    ``is_open=False``); a non-negative int means open with that many elapsed
    fresh-settlement bars since entry.

    OPEN (flat only): the trailing rolling mean net-carry rate over
    ``hysteresis.lookback_settlements`` settlements must clear the breakeven
    rate ``round_trip_cost_frac(costs) / lookback_settlements``, so a single
    noisy interval can never trigger a round trip whose cost dwarfs one funding
    capture.

    CLOSE (open only): ``settlements_since_open`` must reach
    ``hysteresis.min_hold_settlements`` and the trailing
    ``hysteresis.confirm_settlements`` readings must all be non-positive, so a
    regime break is confirmed before the position is round-tripped. The returned
    target is executable no earlier than the next bar, never for the same event.
    """
    grid = data.spot.index
    if not isinstance(decision_time, pd.Timestamp):
        raise ValueError(
            f"decision_time must be a Timestamp, got {type(decision_time).__name__}"
        )
    dt = decision_time
    if dt.tzinfo is None and grid.tz is not None:
        dt = dt.tz_localize(grid.tz)
    elif dt.tzinfo is not None and grid.tz is not None:
        dt = dt.tz_convert(grid.tz)
    if dt not in grid:
        raise ValueError(
            f"decision_time {dt} is not an aligned completed timestamp on the bar grid"
        )
    t = int(grid.get_loc(dt))

    funding = pd.to_numeric(data.funding, errors="coerce").astype("float64")
    fresh = (
        funding.index <= grid[t]
        if t == 0
        else (funding.index > grid[t - 1]) & (funding.index <= grid[t])
    )
    if not bool(fresh.any()):
        return "HOLD"

    readings = _settlement_readings(data)
    settled_events = int(funding.index.searchsorted(grid[t], side="right"))
    window = readings[max(0, settled_events - hysteresis.lookback_settlements): settled_events]
    rolling_mean = float(np.mean(window)) if window.size else 0.0
    breakeven_rate = round_trip_cost_frac(costs) / hysteresis.lookback_settlements

    if settlements_since_open is None:
        return "OPEN" if rolling_mean > breakeven_rate else "HOLD"

    if settlements_since_open < hysteresis.min_hold_settlements:
        return "HOLD"
    confirm = window[-hysteresis.confirm_settlements:]
    if confirm.size >= hysteresis.confirm_settlements and bool(np.all(confirm <= 0.0)):
        return "CLOSE"
    return "HOLD"


def _check_contract() -> None:
    """Executable assertions locking the frozen carry signal surface."""
    from inspect import signature  # noqa: PLC0415

    params = list(signature(generate_cash_carry_target).parameters)
    assert params == [
        "data", "decision_time", "settlements_since_open", "costs", "hysteresis",
    ]


_check_contract()
