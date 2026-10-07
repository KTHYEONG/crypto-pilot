"""Part 2 producer invariants: registry -> engine events per replay piece."""

from __future__ import annotations

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.execution.settlement import (
    settled_before_piece,
    settlement_event_from_json,
    settlement_event_from_record,
    settlement_event_to_json,
    settlement_events_for_piece,
    settlement_fill_price,
)
from src.mhs.instrument_settlements import (
    InstrumentSettlementRecord,
    InstrumentSettlementRegistry,
    assemble_instrument_settlement_registry,
)
from src.mhs.types import ExecutionSpec


def _record(
    symbol: str,
    announced: str,
    last_trade: str,
    delivery: str,
    price: float = 100.0,
    source: str = "twap30_proxy",
) -> InstrumentSettlementRecord:
    ann = pd.Timestamp(announced, tz="UTC")
    lt = pd.Timestamp(last_trade, tz="UTC")
    dv = pd.Timestamp(delivery, tz="UTC")
    return InstrumentSettlementRecord(
        symbol=symbol,
        event_id=f"{symbol}:{int(dv.value // 1_000_000)}",
        announced_at=ann,
        announcement_source="proxy_lead",
        announcement_evidence="",
        last_trade_at=lt,
        delivery_at=dv,
        settlement_price=price,
        price_source=source,  # type: ignore[arg-type]
        price_evidence="lake",
        fee_bps=5.0,
        evidence_digest="sha256:abc",
        verified_at=dv,
    )


def _registry(*records: InstrumentSettlementRecord) -> InstrumentSettlementRegistry:
    return assemble_instrument_settlement_registry(list(records), [])


def test_emits_announced_lifecycles_of_roster_symbols_only() -> None:
    grid = pd.date_range("2022-01-10", "2022-01-20", freq="3min", tz="UTC")
    a = _record("AUSDT", "2022-01-08", "2022-01-14 02:00", "2022-01-14 02:03")
    b = _record("BUSDT", "2022-01-25", "2022-01-26 02:00", "2022-01-26 02:03")
    reg = _registry(a, b)
    qv = pd.DataFrame(float("nan"), index=grid, columns=["AUSDT", "BUSDT", "CUSDT"])
    out = settlement_events_for_piece(
        reg, ["AUSDT", "BUSDT", "CUSDT"], grid, qv,
        replay_start=grid[0], replay_end=grid[-1] + pd.Timedelta(minutes=3),
    )
    assert [e.symbol for e in out] == ["AUSDT"]


def test_reemission_is_stable() -> None:
    grid1 = pd.date_range("2022-01-10", "2022-01-15", freq="3min", tz="UTC")
    grid2 = pd.date_range("2022-01-13", "2022-01-18", freq="3min", tz="UTC")
    a = _record("AUSDT", "2022-01-08", "2022-01-14 02:00", "2022-01-14 02:03")
    reg = _registry(a)
    qv1 = pd.DataFrame(float("nan"), index=grid1, columns=["AUSDT"])
    qv2 = pd.DataFrame(float("nan"), index=grid2, columns=["AUSDT"])
    e1 = settlement_events_for_piece(reg, ["AUSDT"], grid1, qv1, replay_start=grid1[0], replay_end=grid2[-1])
    e2 = settlement_events_for_piece(reg, ["AUSDT"], grid2, qv2, replay_start=grid1[0], replay_end=grid2[-1])
    assert e1 == e2


def test_liquid_bar_after_last_trade_rejected() -> None:
    grid = pd.date_range("2022-01-14 01:00", "2022-01-14 03:00", freq="3min", tz="UTC")
    a = _record("AUSDT", "2022-01-08", "2022-01-14 02:00", "2022-01-14 02:03")
    reg = _registry(a)
    qv = pd.DataFrame(float("nan"), index=grid, columns=["AUSDT"])
    qv.loc[pd.Timestamp("2022-01-14 02:00", tz="UTC"), "AUSDT"] = 500.0
    with pytest.raises(DataIntegrityError):
        settlement_events_for_piece(reg, ["AUSDT"], grid, qv, replay_start=grid[0], replay_end=grid[-1])


def test_relisting_inside_one_replay_rejected() -> None:
    grid = pd.date_range("2022-01-01", "2022-03-01", freq="3min", tz="UTC")
    r1 = _record("AUSDT", "2022-01-08", "2022-01-14 02:00", "2022-01-14 02:03")
    r2 = _record("AUSDT", "2022-02-08", "2022-02-14 02:00", "2022-02-14 02:03")
    reg = _registry(r1, r2)
    qv = pd.DataFrame(float("nan"), index=grid[:10], columns=["AUSDT"])
    small = grid[:10]
    with pytest.raises(DataIntegrityError):
        settlement_events_for_piece(
            reg, ["AUSDT"], small, qv, replay_start=grid[0], replay_end=grid[-1],
        )


def test_settled_roster_drop_rule() -> None:
    grid = pd.date_range("2022-02-01", "2022-02-02", freq="3min", tz="UTC")
    a = _record("AUSDT", "2022-01-08", "2022-01-14 02:00", "2022-01-14 02:03")
    d = _record("DUSDT", "2022-01-08", "2022-01-14 02:00", "2022-01-14 02:03")
    e = _record("EUSDT", "2022-01-08", "2022-01-14 02:00", "2022-01-14 02:03")
    reg = _registry(a, d, e)
    weights = pd.DataFrame(
        {"AUSDT": [0.0, 0.0], "DUSDT": [float("nan"), 0.0], "EUSDT": [0.5, 0.0]},
        index=pd.date_range("2022-02-01", periods=2, freq="24h", tz="UTC"),
    )
    dropped = settled_before_piece(reg, ["AUSDT", "DUSDT", "EUSDT"], grid, weights)
    assert dropped == frozenset({"AUSDT"})


def test_haircut_direction() -> None:
    grid = pd.Timestamp("2022-01-14 02:03", tz="UTC")
    proxy = _record("AUSDT", "2022-01-08", "2022-01-14 02:00", "2022-01-14 02:03", price=100.0)
    reg = _registry(proxy)
    event = settlement_event_from_record(reg.settlements[0])
    stress = ExecutionSpec(settlement_price_haircut_bps=450.0)
    assert settlement_fill_price(event, 10.0, stress) == pytest.approx(95.5)
    assert settlement_fill_price(event, -10.0, stress) == pytest.approx(104.5)
    curated = _record("CUSDT", "2022-01-08", "2022-01-14 02:00", "2022-01-14 02:03", price=100.0, source="curated")
    curated_event = settlement_event_from_record(_registry(curated).settlements[0])
    assert settlement_fill_price(curated_event, 10.0, stress) == pytest.approx(100.0)
    assert grid is not None


def test_codec_round_trip() -> None:
    a = _record("AUSDT", "2022-01-08", "2022-01-14 02:00", "2022-01-14 02:03", price=2.5)
    event = settlement_event_from_record(_registry(a).settlements[0])
    assert settlement_event_from_json(settlement_event_to_json(event)) == event


@pytest.mark.parametrize("field", ["event_id", "effective_at_ns", "announced_at_ns", "last_trade_at_ns"])
def test_codec_missing_fields_fail_closed(field) -> None:
    event = settlement_event_from_record(_record("AUSDT", "2022-01-08", "2022-01-14 02:00", "2022-01-14 02:03"))
    payload = settlement_event_to_json(event)
    payload.pop(field)
    with pytest.raises(DataIntegrityError, match="missing field"):
        settlement_event_from_json(payload)
