"""Part 3 lifecycle intent-policy invariants: pure announcement-driven desired units."""

from __future__ import annotations

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.execution.contracts import InstrumentSettlementEvent
from src.mhs.execution.lifecycle import lifecycle_desired_units
from src.mhs.params import DELIST_FORCED_EXIT_LEAD

LEAD_NS = int(DELIST_FORCED_EXIT_LEAD.value)


def _event(
    announced: str,
    delivery: str = "2022-02-15 01:00",
    last_trade: str = "2022-02-15 01:00",
) -> InstrumentSettlementEvent:
    dv = pd.Timestamp(delivery, tz="UTC")
    lt = pd.Timestamp(last_trade, tz="UTC")
    return InstrumentSettlementEvent(
        event_id=f"AUSDT:{int(dv.value // 1_000_000)}",
        symbol="AUSDT",
        effective_at=dv,
        available_at=dv,
        settlement_price=2.0,
        fee_bps=5.0,
        source_digest="sha256:test",
        announced_at=pd.Timestamp(announced, tz="UTC"),
        last_trade_at=lt,
        price_source="venue",  # type: ignore[arg-type]
    )


def test_unknown_announcement_passes_through_bit_identical() -> None:
    event = _event("2022-02-10 00:00")
    info_ns = int(pd.Timestamp("2022-02-09 23:59:59.999999999", tz="UTC").value)
    desired = 37.0
    effective, action = lifecycle_desired_units(
        event,
        information_ns=info_ns,
        current_units=10.0,
        desired_units=desired,
        forced_exit_lead_ns=LEAD_NS,
    )
    assert action == "unchanged"
    assert effective == desired


def test_known_announcement_blocks_entry_and_add() -> None:
    event = _event("2022-02-01 00:00")
    info_ns = int(pd.Timestamp("2022-02-05 00:00", tz="UTC").value)
    effective, action = lifecycle_desired_units(
        event,
        information_ns=info_ns,
        current_units=0.0,
        desired_units=50.0,
        forced_exit_lead_ns=LEAD_NS,
    )
    assert (effective, action) == (0.0, "entry_blocked")
    effective, action = lifecycle_desired_units(
        event,
        information_ns=info_ns,
        current_units=10.0,
        desired_units=60.0,
        forced_exit_lead_ns=LEAD_NS,
    )
    assert (effective, action) == (10.0, "entry_blocked")


def test_known_announcement_blocks_flip() -> None:
    event = _event("2022-02-01 00:00")
    info_ns = int(pd.Timestamp("2022-02-05 00:00", tz="UTC").value)
    effective, action = lifecycle_desired_units(
        event,
        information_ns=info_ns,
        current_units=10.0,
        desired_units=-5.0,
        forced_exit_lead_ns=LEAD_NS,
    )
    assert (effective, action) == (0.0, "flip_blocked")


def test_reductions_follow_target() -> None:
    event = _event("2022-02-01 00:00")
    info_ns = int(pd.Timestamp("2022-02-05 00:00", tz="UTC").value)
    desired = 4.0
    effective, action = lifecycle_desired_units(
        event,
        information_ns=info_ns,
        current_units=10.0,
        desired_units=desired,
        forced_exit_lead_ns=LEAD_NS,
    )
    assert action == "unchanged"
    assert effective == desired


def test_forced_exit_inside_lead() -> None:
    event = _event("2022-02-01 00:00")
    delivery_ns = int(pd.Timestamp("2022-02-15 01:00", tz="UTC").value)
    inside = delivery_ns - 71 * 3_600_000_000_000
    effective, action = lifecycle_desired_units(
        event,
        information_ns=inside,
        current_units=10.0,
        desired_units=4.0,
        forced_exit_lead_ns=LEAD_NS,
    )
    assert (effective, action) == (0.0, "forced_exit")
    outside = delivery_ns - 73 * 3_600_000_000_000
    effective, action = lifecycle_desired_units(
        event,
        information_ns=outside,
        current_units=10.0,
        desired_units=4.0,
        forced_exit_lead_ns=LEAD_NS,
    )
    assert action == "unchanged"
    assert effective == 4.0


def test_no_event_passes_through() -> None:
    desired = -12.5
    effective, action = lifecycle_desired_units(
        None,
        information_ns=0,
        current_units=10.0,
        desired_units=desired,
        forced_exit_lead_ns=LEAD_NS,
    )
    assert action == "unchanged"
    assert effective == desired


def test_non_finite_units_fail_closed() -> None:
    event = _event("2022-02-01 00:00")
    with pytest.raises(DataIntegrityError):
        lifecycle_desired_units(
            event,
            information_ns=0,
            current_units=float("nan"),
            desired_units=1.0,
            forced_exit_lead_ns=LEAD_NS,
        )


def test_flat_noop_inside_lead_is_unchanged() -> None:
    event = _event("2022-02-01 00:00")
    delivery_ns = int(pd.Timestamp("2022-02-15 01:00", tz="UTC").value)
    inside = delivery_ns - 71 * 3_600_000_000_000
    effective, action = lifecycle_desired_units(
        event,
        information_ns=inside,
        current_units=0.0,
        desired_units=0.0,
        forced_exit_lead_ns=LEAD_NS,
    )
    assert action == "unchanged"
    assert effective == 0.0


def test_flat_noop_outside_lead_is_unchanged() -> None:
    event = _event("2022-02-01 00:00")
    info_ns = int(pd.Timestamp("2022-02-05 00:00", tz="UTC").value)
    effective, action = lifecycle_desired_units(
        event,
        information_ns=info_ns,
        current_units=0.0,
        desired_units=0.0,
        forced_exit_lead_ns=LEAD_NS,
    )
    assert action == "unchanged"
    assert effective == 0.0
