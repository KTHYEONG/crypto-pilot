"""Announcement-driven intent policy for delisted instruments (spec 34 part 3)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from src.common.errors import DataIntegrityError
from src.engine.execution.contracts import InstrumentSettlementEvent

LifecycleIntentAction = Literal["unchanged", "entry_blocked", "flip_blocked", "forced_exit"]


def lifecycle_desired_units(
    event: InstrumentSettlementEvent | None,
    *,
    information_ns: int,
    current_units: float,
    desired_units: float,
    forced_exit_lead_ns: int,
) -> tuple[float, LifecycleIntentAction]:
    """Apply the announced-delisting intent policy to one symbol's desired inventory.

    Before the announcement is known the target passes through untouched, so a record announced
    after the decision's information instant can never influence it. Once known, exposure may only
    shrink toward zero (no new entry, no add, no flip), and within the forced-exit lead before
    delivery the desired inventory is zero, mirroring the live roster withdrawal.

    Args:
        event: The symbol's admitted lifecycle event, or None.
        information_ns: Decision information instant (signal availability), epoch ns.
        current_units: Signed inventory before the intent.
        desired_units: Signed inventory implied by the target.
        forced_exit_lead_ns: ``DELIST_FORCED_EXIT_LEAD`` in ns.
    Returns:
        ``(effective desired units, action)``; ``unchanged`` returns ``desired_units`` bit-identically.
    Raises:
        DataIntegrityError: non-finite units.
    """
    if not np.isfinite(float(current_units)) or not np.isfinite(float(desired_units)):
        raise DataIntegrityError("lifecycle units must be finite")
    if event is None:
        return desired_units, "unchanged"
    announced = event.announced_at if event.announced_at is not None else event.effective_at
    if int(announced.value) > int(information_ns):
        return desired_units, "unchanged"
    delivery_ns = int(event.effective_at.value)
    if int(information_ns) + int(forced_exit_lead_ns) >= delivery_ns:
        if float(current_units) == 0.0 and float(desired_units) == 0.0:
            return desired_units, "unchanged"
        return 0.0, "forced_exit"
    cur = float(current_units)
    des = float(desired_units)
    if cur == 0.0:
        if des == 0.0:
            return desired_units, "unchanged"
        return 0.0, "entry_blocked"
    if des == 0.0 or (np.sign(des) == np.sign(cur) and abs(des) <= abs(cur)):
        return desired_units, "unchanged"
    if np.sign(des) != np.sign(cur):
        return 0.0, "flip_blocked"
    return float(cur), "entry_blocked"


__all__ = ["LifecycleIntentAction", "lifecycle_desired_units"]
