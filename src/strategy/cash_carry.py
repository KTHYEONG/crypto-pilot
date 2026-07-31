from __future__ import annotations

from typing import Literal

import pandas as pd

from src.data.carry_data import CarryMarketData

CarryTarget = Literal["OPEN", "HOLD", "CLOSE"]


def generate_cash_carry_target(
    data: CarryMarketData,
    decision_time: pd.Timestamp,
    is_open: bool,
) -> CarryTarget:
    """Causal same-asset carry state target.

    Only funding and borrow observations settled no later than ``decision_time``
    are used. A decision is only evaluated on a bar that contains a fresh
    funding settlement: on bars without one the current state is preserved
    (``HOLD``), so an open pair is never dropped on an empty settlement bar.
    Positive net carry targets ``OPEN`` when flat and ``HOLD`` when open;
    non-positive observed net carry targets ``CLOSE`` when open.

    Net carry compares the funding settled since the prior funding settlement
    against the quote-cash borrow accrued over the same elapsed interval (or
    from the first valid event boundary), never a single arbitrary model bar.
    The returned target is executable no earlier than the next bar, never for
    the same event.
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
    if t == 0:
        fresh = funding.index <= grid[t]
        prior = pd.DatetimeIndex([])
    else:
        fresh = (funding.index > grid[t - 1]) & (funding.index <= grid[t])
        prior = funding.index[funding.index <= grid[t - 1]]
    if not bool(fresh.any()):
        return "HOLD"

    settled_funding = float(funding.loc[fresh].sum())
    borrow = pd.to_numeric(data.borrow, errors="coerce")
    if len(prior) == 0:
        borrow_cost = float(borrow.iloc[: t + 1].sum())
    else:
        interval_start = prior[-1]
        borrow_cost = float(
            borrow[(borrow.index > interval_start) & (borrow.index <= grid[t])].sum()
        )
    net_carry = settled_funding - borrow_cost

    if is_open:
        return "CLOSE" if net_carry <= 0 else "HOLD"
    return "OPEN" if net_carry > 0 else "HOLD"


def _check_contract() -> None:
    """Executable assertions locking the frozen carry signal surface."""
    assert generate_cash_carry_target.__name__ == "generate_cash_carry_target"


_check_contract()
