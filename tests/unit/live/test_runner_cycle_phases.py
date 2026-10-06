# ruff: noqa
"""Guard tests for run_shadow_cycle phase extraction (spec 21)."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import src.live.runner as runner_mod
from src.common.errors import DataIntegrityError
from src.live.account import AccountSnapshot
from src.live.audit import AuditLog
from src.live.delisting_settlement import DelistingSettlement
from src.live.ledger import LedgerState
from src.live.ledger import append_position_snapshot, load_ledger, save_ledger
from src.live.tax_ledger import load_tax_records
from src.live.order_journal import OrderJournal
from src.live.runner import (
    _collect_live_tax_fail_soft,
    _cycle_fallback_attempt,
    _flush_already_executed,
    _reconcile_pre_trade,
    _record_delisting_settlements,
    _settle_paper_funding_and_delistings,
    _write_portfolio_state_fail_soft,
)
from src.live.settings import LiveSettings

from tests.unit.live._runner_stubs import DECISION_TIME, NOW

DECISION = DECISION_TIME


def _tmp_state_kwargs(tmp_path: Path, stem: str) -> dict[str, str]:
    base = tmp_path / stem
    return {
        "order_journal_path": str(base / "order_journal.jsonl"),
        "fills_dir": str(base / "fills"),
        "tax_ledger_dir": str(base / "tax"),
        "execution_quality_dir": str(base / "execution_quality"),
        "portfolio_state_dir": str(base / "portfolio_state"),
    }


def _paper_settings(tmp_path: Path, stem: str = "s") -> LiveSettings:
    return LiveSettings(
        mode="paper",
        notional_equity_usdt=2000.0,
        ledger_path=str(tmp_path / f"{stem}.json"),
        **_tmp_state_kwargs(tmp_path, stem),
    )


def _flat_snapshot() -> AccountSnapshot:
    return AccountSnapshot(
        taken_at=NOW,
        wallet_balance=Decimal("2000"),
        available_balance=Decimal("1900"),
        total_maint_margin=Decimal("10"),
        unrealized_pnl=Decimal("0"),
        positions={},
        dual_side_position=False,
        multi_assets_margin=False,
    )


def test_fallback_attempt_is_sentinel(tmp_path) -> None:
    settings = _paper_settings(tmp_path, "fb")
    attempt = _cycle_fallback_attempt(settings, DECISION, NOW)
    assert attempt.attempt_seq == -1
    assert attempt.run_id == "20260824"
    assert attempt.pre_trade_equity == 0
    assert attempt.sizing_anchor == "decision_ohlcv_close"
    assert attempt.decision_marks == {}
    assert attempt.started_at == NOW
    assert attempt.mode == "paper"


def test_flush_skips_commit_when_current(monkeypatch, tmp_path) -> None:
    settings = _paper_settings(tmp_path, "flush_skip")
    ledger_path = Path(str(tmp_path / "flush_skip.json"))
    state = LedgerState(journal_applied_fill_seq=5, journal_recorded_fill_seq=5)

    class FakeJournal:
        def __init__(self, *a, **k) -> None:
            pass

    monkeypatch.setattr(runner_mod, "OrderJournal", FakeJournal)
    monkeypatch.setattr(runner_mod, "load_ledger", lambda p: state)
    calls: list[str] = []

    def _boom(*a, **k):
        calls.append("commit")
        raise AssertionError("must not commit")

    monkeypatch.setattr(runner_mod, "_commit_and_record", _boom)
    assert _flush_already_executed(settings, ledger_path, DECISION, NOW) is None
    assert calls == []


def test_flush_commits_and_swallows(monkeypatch, tmp_path, caplog) -> None:
    settings = _paper_settings(tmp_path, "flush_go")
    ledger_path = Path(str(tmp_path / "flush_go.json"))
    state = LedgerState(journal_applied_fill_seq=7, journal_recorded_fill_seq=3)
    calls: list[str] = []

    class FakeJournal:
        def __init__(self, *a, **k) -> None:
            pass

    monkeypatch.setattr(runner_mod, "OrderJournal", FakeJournal)
    monkeypatch.setattr(runner_mod, "load_ledger", lambda p: state)
    monkeypatch.setattr(runner_mod, "AuditLog", lambda *a, **k: AuditLog(tmp_path / "a.jsonl"))

    def _ok(*a, **k):
        calls.append("commit")
        return state

    monkeypatch.setattr(runner_mod, "_commit_and_record", _ok)
    _flush_already_executed(settings, ledger_path, DECISION, NOW)
    assert calls == ["commit"]

    def _fail(*a, **k):
        raise OSError("disk")

    monkeypatch.setattr(runner_mod, "_commit_and_record", _fail)
    _flush_already_executed(settings, ledger_path, DECISION, NOW)
    assert [record.getMessage() for record in caplog.records] == [
        "[SYS] already_executed flush failed error=disk"
    ]


def test_record_settlements_both_flags(tmp_path) -> None:
    settings = _paper_settings(tmp_path, "rec")
    audit = AuditLog(tmp_path / "audit.jsonl")
    settlements = (
        DelistingSettlement(
            symbol="AAAUSDT",
            quantity=Decimal("1"),
            price=Decimal("100"),
            fee=Decimal("0.01"),
            delivery_time=DECISION,
            evidence_source="flat_1h_klines",
        ),
        DelistingSettlement(
            symbol="BUSDT",
            quantity=Decimal("-2"),
            price=None,
            fee=Decimal("0"),
            delivery_time=DECISION,
            evidence_source="venue",
        ),
    )
    events: list[tuple[str, str]] = []

    def _fake_notify(s, *, event, detail, decision_time, now, discriminator=None):
        events.append((event, detail))

    import src.live.runner as rm

    orig = rm._notify_event
    rm._notify_event = _fake_notify  # type: ignore[method-assign]
    try:
        _record_delisting_settlements(
            settings, audit, settlements, decision_time=DECISION, now=NOW, alert_includes_price=False
        )
        _record_delisting_settlements(
            settings, audit, settlements, decision_time=DECISION, now=NOW, alert_includes_price=True
        )
    finally:
        rm._notify_event = orig
    assert len(events) == 4
    assert "price=" not in events[0][1]
    assert "price=" in events[2][1]


def _fake_journal(last: int) -> Any:
    class J:
        def last_fill_seq(self) -> int:
            return last

    return J()


def _patch_reconcile_common(monkeypatch, tmp_path, state: LedgerState):
    from src.live.executor import OrphanSweep
    from src.live.order_cancel import UnresolvedOrder
    from src.live.recovery import RecoveryReport, UnresolvedSettlement

    sweep = OrphanSweep(fills=(), foreign_symbols=())
    recovery = RecoveryReport(recovered=(), resolved_ids=(), unresolved_ids=(), unresolved=())
    settlement = UnresolvedSettlement(recovered=(), resolved_ids=(), still_open=())
    monkeypatch.setattr(runner_mod, "cancel_orphan_orders", lambda *a, **k: sweep)
    monkeypatch.setattr(runner_mod, "recover_unresolved_orders", lambda *a, **k: recovery)
    monkeypatch.setattr(runner_mod, "settle_unresolved_orders", lambda *a, **k: settlement)
    monkeypatch.setattr(
        runner_mod, "_commit_and_record", lambda *a, **k: state
    )
    monkeypatch.setattr(runner_mod, "_record_unresolved_orders", lambda *a, **k: ())
    monkeypatch.setattr(runner_mod, "_notify_event", lambda *a, **k: None)
    return sweep


def test_reconcile_regression_halts(monkeypatch, tmp_path) -> None:
    settings = _paper_settings(tmp_path, "reg")
    audit = AuditLog(tmp_path / "audit.jsonl")
    state = LedgerState(journal_applied_fill_seq=10)
    alerts: list[str] = []
    monkeypatch.setattr(
        runner_mod, "_notify_event", lambda *a, **k: alerts.append(k.get("event", ""))
    )

    def _boom(*a, **k):
        raise AssertionError("sweep must not run")

    monkeypatch.setattr(runner_mod, "cancel_orphan_orders", _boom)
    with pytest.raises(DataIntegrityError, match="order journal regressed"):
        _reconcile_pre_trade(
            settings,
            order_client=object(),
            journal=_fake_journal(-1),
            audit=audit,
            ledger_path=tmp_path / "l.json",
            ledger_state=state,
            snapshot=_flat_snapshot(),
            exchange_info={},
            run_id="20260824",
            decision_time=DECISION,
            now=NOW,
        )
    assert alerts == ["order_journal_regressed"]


def test_reconcile_suppressed_happy(monkeypatch, tmp_path) -> None:
    settings = _paper_settings(tmp_path, "sup")
    audit = AuditLog(tmp_path / "audit.jsonl")
    state = LedgerState(journal_applied_fill_seq=-1)
    _patch_reconcile_common(monkeypatch, tmp_path, state)
    out = _reconcile_pre_trade(
        settings,
        order_client=object(),
        journal=_fake_journal(0),
        audit=audit,
        ledger_path=tmp_path / "l.json",
        ledger_state=state,
        snapshot=_flat_snapshot(),
        exchange_info={},
        run_id="20260824",
        decision_time=DECISION,
        now=NOW,
    )
    assert out.derisk_reasons == ()
    assert out.sweep.foreign_symbols == ()
    assert out.ledger_state is state


def _live_settings(tmp_path: Path, stem: str, **over: Any) -> LiveSettings:
    base = dict(
        mode="live_testnet",
        notional_equity_usdt=2000.0,
        ledger_path=str(tmp_path / f"{stem}.json"),
        **_tmp_state_kwargs(tmp_path, stem),
    )
    base.update(over)
    return LiveSettings(**base)  # type: ignore[arg-type]


def test_reconcile_mutating_derisk_with_booked_and_adopt(monkeypatch, tmp_path) -> None:
    from src.live.account import PositionBreach
    from src.live.executor import OrphanSweep
    from src.live.recovery import RecoveryReport, UnresolvedSettlement

    settings = _live_settings(tmp_path, "mut", venue_force_close_auto_adopt=False)
    audit = AuditLog(tmp_path / "audit.jsonl")
    state = LedgerState(journal_applied_fill_seq=-1, positions={"AAAUSDT": Decimal("1")})
    snapshot = AccountSnapshot(
        taken_at=NOW,
        wallet_balance=Decimal("2000"),
        available_balance=Decimal("100"),
        total_maint_margin=Decimal("10"),
        unrealized_pnl=Decimal("0"),
        positions={},
        dual_side_position=False,
        multi_assets_margin=False,
    )
    sweep = OrphanSweep(fills=(), foreign_symbols=())
    recovery = RecoveryReport(recovered=(), resolved_ids=(), unresolved_ids=(), unresolved=())
    settlement = UnresolvedSettlement(recovered=(), resolved_ids=(), still_open=())
    monkeypatch.setattr(runner_mod, "cancel_orphan_orders", lambda *a, **k: sweep)
    monkeypatch.setattr(runner_mod, "recover_unresolved_orders", lambda *a, **k: recovery)
    monkeypatch.setattr(runner_mod, "settle_unresolved_orders", lambda *a, **k: settlement)
    monkeypatch.setattr(runner_mod, "_commit_and_record", lambda *a, **k: state)
    monkeypatch.setattr(runner_mod, "_record_unresolved_orders", lambda *a, **k: ())
    monkeypatch.setattr(runner_mod, "_notify_event", lambda *a, **k: None)
    monkeypatch.setattr(
        runner_mod, "settled_delisting_symbols", lambda *a, **k: ["AAAUSDT"]
    )
    booked = (
        DelistingSettlement(
            symbol="AAAUSDT",
            quantity=Decimal("1"),
            price=None,
            fee=Decimal("0"),
            delivery_time=DECISION,
            evidence_source="venue",
        ),
    )
    monkeypatch.setattr(
        runner_mod,
        "book_delisting_settlements",
        lambda s, **k: (s, booked),
    )
    monkeypatch.setattr(runner_mod, "save_ledger", lambda *a, **k: None)
    monkeypatch.setattr(
        runner_mod, "_record_delisting_settlements", lambda *a, **k: None
    )
    breach = PositionBreach(symbol="AAAUSDT", venue_qty=Decimal("0"), ledger_qty=Decimal("1"))
    monkeypatch.setattr(runner_mod, "find_position_breaches", lambda *a, **k: [breach])
    out = _reconcile_pre_trade(
        settings,
        order_client=object(),
        journal=_fake_journal(0),
        audit=audit,
        ledger_path=tmp_path / "l.json",
        ledger_state=state,
        snapshot=snapshot,
        exchange_info={},
        run_id="20260824",
        decision_time=DECISION,
        now=NOW,
    )
    assert out.derisk_reasons == ("reconciliation_breach",)


def test_reconcile_mutating_raise_when_derisk_disabled(monkeypatch, tmp_path) -> None:
    from src.live.account import PositionBreach
    from src.live.executor import ForeignOpenOrderError
    from src.live.executor import OrphanSweep
    from src.live.recovery import RecoveryReport, UnresolvedSettlement

    settings = _live_settings(
        tmp_path, "raise", venue_force_close_auto_adopt=False, derisk_mode_enabled=False
    )
    audit = AuditLog(tmp_path / "audit.jsonl")
    state = LedgerState(journal_applied_fill_seq=-1)
    sweep = OrphanSweep(fills=(), foreign_symbols=("ZZZUSDT",))
    recovery = RecoveryReport(recovered=(), resolved_ids=(), unresolved_ids=(), unresolved=())
    settlement = UnresolvedSettlement(recovered=(), resolved_ids=(), still_open=())
    monkeypatch.setattr(runner_mod, "cancel_orphan_orders", lambda *a, **k: sweep)
    monkeypatch.setattr(runner_mod, "recover_unresolved_orders", lambda *a, **k: recovery)
    monkeypatch.setattr(runner_mod, "settle_unresolved_orders", lambda *a, **k: settlement)
    monkeypatch.setattr(runner_mod, "_commit_and_record", lambda *a, **k: state)
    monkeypatch.setattr(runner_mod, "_record_unresolved_orders", lambda *a, **k: ())
    monkeypatch.setattr(runner_mod, "_notify_event", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "settled_delisting_symbols", lambda *a, **k: [])
    monkeypatch.setattr(runner_mod, "book_delisting_settlements", lambda s, **k: (s, ()))
    monkeypatch.setattr(
        runner_mod, "find_position_breaches", lambda *a, **k: []
    )
    with pytest.raises(ForeignOpenOrderError, match="foreign open order"):
        _reconcile_pre_trade(
            settings,
            order_client=object(),
            journal=_fake_journal(0),
            audit=audit,
            ledger_path=tmp_path / "l.json",
            ledger_state=state,
            snapshot=_flat_snapshot(),
            exchange_info={},
            run_id="20260824",
            decision_time=DECISION,
            now=NOW,
        )


def test_reconcile_mutating_carryover_and_adopt(monkeypatch, tmp_path) -> None:
    from src.live.executor import OrphanSweep
    from src.live.recovery import RecoveryReport, UnresolvedSettlement

    settings = _live_settings(tmp_path, "carry", venue_force_close_auto_adopt=True)
    audit = AuditLog(tmp_path / "audit.jsonl")
    state = LedgerState(
        journal_applied_fill_seq=-1, derisk_since=NOW, derisk_reasons=("reconciliation_breach",)
    )
    sweep = OrphanSweep(fills=(), foreign_symbols=())
    recovery = RecoveryReport(recovered=(), resolved_ids=(), unresolved_ids=(), unresolved=())
    settlement = UnresolvedSettlement(recovered=(), resolved_ids=(), still_open=())
    monkeypatch.setattr(runner_mod, "cancel_orphan_orders", lambda *a, **k: sweep)
    monkeypatch.setattr(runner_mod, "recover_unresolved_orders", lambda *a, **k: recovery)
    monkeypatch.setattr(runner_mod, "settle_unresolved_orders", lambda *a, **k: settlement)
    monkeypatch.setattr(runner_mod, "_commit_and_record", lambda *a, **k: state)
    monkeypatch.setattr(runner_mod, "_record_unresolved_orders", lambda *a, **k: ())
    monkeypatch.setattr(runner_mod, "_notify_event", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "settled_delisting_symbols", lambda *a, **k: [])
    monkeypatch.setattr(runner_mod, "book_delisting_settlements", lambda s, **k: (s, ()))
    from src.live.account import PositionBreach

    breach = PositionBreach(symbol="AAAUSDT", venue_qty=Decimal("0"), ledger_qty=Decimal("1"))
    monkeypatch.setattr(runner_mod, "find_position_breaches", lambda *a, **k: [breach])
    monkeypatch.setattr(runner_mod, "_adopt_force_closes", lambda *a, **k: state)
    out = _reconcile_pre_trade(
        settings,
        order_client=object(),
        journal=_fake_journal(0),
        audit=audit,
        ledger_path=tmp_path / "l.json",
        ledger_state=state,
        snapshot=_flat_snapshot(),
        exchange_info={},
        run_id="20260824",
        decision_time=DECISION,
        now=NOW,
    )
    assert "reconciliation_breach" in out.derisk_reasons


def test_reconcile_mutating_carryover_no_reasons(monkeypatch, tmp_path) -> None:
    from src.live.executor import OrphanSweep
    from src.live.recovery import RecoveryReport, UnresolvedSettlement

    settings = _live_settings(tmp_path, "carry2", venue_force_close_auto_adopt=False)
    audit = AuditLog(tmp_path / "audit.jsonl")
    state = LedgerState(
        journal_applied_fill_seq=-1, derisk_since=NOW, derisk_reasons=("old_reason",)
    )
    sweep = OrphanSweep(fills=(), foreign_symbols=())
    recovery = RecoveryReport(recovered=(), resolved_ids=(), unresolved_ids=(), unresolved=())
    settlement = UnresolvedSettlement(recovered=(), resolved_ids=(), still_open=())
    monkeypatch.setattr(runner_mod, "cancel_orphan_orders", lambda *a, **k: sweep)
    monkeypatch.setattr(runner_mod, "recover_unresolved_orders", lambda *a, **k: recovery)
    monkeypatch.setattr(runner_mod, "settle_unresolved_orders", lambda *a, **k: settlement)
    monkeypatch.setattr(runner_mod, "_commit_and_record", lambda *a, **k: state)
    monkeypatch.setattr(runner_mod, "_record_unresolved_orders", lambda *a, **k: ())
    monkeypatch.setattr(runner_mod, "_notify_event", lambda *a, **k: None)
    monkeypatch.setattr(runner_mod, "settled_delisting_symbols", lambda *a, **k: [])
    monkeypatch.setattr(runner_mod, "book_delisting_settlements", lambda s, **k: (s, ()))
    monkeypatch.setattr(runner_mod, "find_position_breaches", lambda *a, **k: [])
    out = _reconcile_pre_trade(
        settings,
        order_client=object(),
        journal=_fake_journal(0),
        audit=audit,
        ledger_path=tmp_path / "l.json",
        ledger_state=state,
        snapshot=_flat_snapshot(),
        exchange_info={},
        run_id="20260824",
        decision_time=DECISION,
        now=NOW,
    )
    assert out.derisk_reasons == ("old_reason",)


def test_paper_absent_halts(tmp_path) -> None:
    settings = _paper_settings(tmp_path, "abs")
    audit = AuditLog(tmp_path / "audit.jsonl")
    state = LedgerState(positions={"AAAUSDT": Decimal("1")}, cash_usdt=Decimal("1000"))
    with pytest.raises(DataIntegrityError, match="absent"):
        _settle_paper_funding_and_delistings(
            settings,
            audit,
            ledger_state=state,
            ledger_path=tmp_path / "l.json",
            exchange_info={"symbols": []},
            run_id="20260824",
            decision_time=DECISION,
            now=NOW,
        )


@pytest.mark.parametrize("evidence", ["missing", "traded"])
def test_paper_delisted_full_path(monkeypatch, tmp_path, evidence) -> None:
    settings = _paper_settings(tmp_path, "pd")
    audit = AuditLog(tmp_path / "audit.jsonl")
    state = LedgerState(positions={"AAAUSDT": Decimal("1")}, cash_usdt=Decimal("1000"))
    ledger_path = tmp_path / "l.json"
    save_ledger(ledger_path, state)
    _isolate_paper_inputs(monkeypatch, tmp_path)
    delivery = DECISION - pd.Timedelta(hours=4)
    if evidence == "traded":
        _write_settlement_bars(tmp_path, delivery, volume=1.0)
    alerts = []
    monkeypatch.setattr(runner_mod, "dispatch_alert", lambda *a, **k: alerts.append(k))
    with pytest.raises(DataIntegrityError, match="settlement evidence"):
        _settle_paper_funding_and_delistings(
            settings,
            audit,
            ledger_state=state,
            ledger_path=ledger_path,
            exchange_info=_paper_exchange_info(delivery),
            run_id="20260824",
            decision_time=DECISION,
            now=NOW,
        )
    persisted = load_ledger(ledger_path)
    assert persisted.cash_usdt == Decimal("1000")
    assert persisted.positions == {"AAAUSDT": Decimal("1")}
    assert load_tax_records(Path(settings.tax_ledger_dir)).empty
    records = [json.loads(line) for line in audit.path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["paper_delisted_unresolved"]
    assert [alert["event"] for alert in alerts] == ["paper_delisted_unresolved"]


def _paper_exchange_info(delivery=None):
    entry = {"symbol": "AAAUSDT", "status": "TRADING"}
    if delivery is not None:
        entry.update(status="CLOSE", deliveryDate=int(delivery.timestamp() * 1000))
    return {"symbols": [entry]}


def _isolate_paper_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_mod, "FUTURES_DATA_DIR", tmp_path)
    monkeypatch.setattr(runner_mod, "_load_paper_funding", lambda symbols: {})
    monkeypatch.setattr(runner_mod, "_load_paper_trade_closes", lambda symbols: {})


def _write_settlement_bars(tmp_path, delivery, *, volume=0.0):
    directory = tmp_path / "ohlcv" / "1h"
    directory.mkdir(parents=True, exist_ok=True)
    stamps = pd.date_range(delivery, periods=3, freq="h")
    pd.DataFrame({
        "timestamp": stamps.as_unit("ms").asi8,
        "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
        "volume": volume,
    }).to_parquet(directory / "AAAUSDT.parquet")


@pytest.mark.parametrize("quantity", [Decimal("1"), Decimal("-1")])
def test_paper_delisted_evidence_and_tax_records(monkeypatch, tmp_path, quantity) -> None:
    settings = _paper_settings(tmp_path, "pd2").model_copy(update={
        "delisting_settlement_fee_bps": 10.0,
        "delisting_settlement_min_flat_bars": 3,
    })
    audit = AuditLog(tmp_path / "audit.jsonl")
    state = LedgerState(positions={"AAAUSDT": quantity}, cash_usdt=Decimal("1000"))
    ledger_path = tmp_path / "l.json"
    save_ledger(ledger_path, state)
    _isolate_paper_inputs(monkeypatch, tmp_path)
    delivery = DECISION - pd.Timedelta(hours=4)
    _write_settlement_bars(tmp_path, delivery)
    monkeypatch.setattr(runner_mod, "dispatch_alert", lambda *a, **k: None)
    out = _settle_paper_funding_and_delistings(
        settings,
        audit,
        ledger_state=state,
        ledger_path=ledger_path,
        exchange_info=_paper_exchange_info(delivery),
        run_id="20260824",
        decision_time=DECISION,
        now=NOW,
    )
    expected_cash = Decimal("1000") + quantity * Decimal("100") - Decimal("0.1")
    assert out.cash_before == Decimal("1000")
    assert out.ledger_state.positions["AAAUSDT"] == 0
    assert out.ledger_state.cash_usdt == expected_cash
    persisted = load_ledger(ledger_path)
    assert persisted.positions == {}
    assert persisted.cash_usdt == expected_cash
    assert len(out.settlement_records) == 1
    record = out.settlement_records[0]
    assert (record.kind, record.side, record.price, record.quantity, record.fee) == (
        "TRADE", "SELL" if quantity > 0 else "BUY", 100.0, 1.0, 0.1
    )
    assert record.event_time == delivery
    tax_frame = load_tax_records(Path(settings.tax_ledger_dir))
    assert tax_frame["record_id"].tolist() == [record.record_id]
    assert tax_frame["fee"].tolist() == [0.1]
    assert tax_frame["side"].tolist() == [record.side]
    retry = _settle_paper_funding_and_delistings(
        settings,
        audit,
        ledger_state=persisted,
        ledger_path=ledger_path,
        exchange_info=_paper_exchange_info(delivery),
        run_id="20260824",
        decision_time=DECISION,
        now=NOW,
    )
    assert retry.ledger_state.cash_usdt == expected_cash
    assert retry.settlement_records == ()
    pd.testing.assert_frame_equal(load_tax_records(Path(settings.tax_ledger_dir)), tax_frame)


def test_paper_no_delisted_success(monkeypatch, tmp_path) -> None:
    settings = _paper_settings(tmp_path, "pn")
    audit = AuditLog(tmp_path / "audit.jsonl")
    start = DECISION - pd.Timedelta(hours=8)
    positions = {"AAAUSDT": Decimal("1")}
    state = LedgerState(
        positions=positions, cash_usdt=Decimal("1000"),
        funding_watermarks={"AAAUSDT": start},
        position_history=append_position_snapshot((), start, positions),
    )
    ledger_path = tmp_path / "l.json"
    save_ledger(ledger_path, state)
    _isolate_paper_inputs(monkeypatch, tmp_path)
    rates = pd.Series([0.001], index=pd.DatetimeIndex([DECISION]))
    closes = pd.Series([100.0], index=pd.DatetimeIndex([DECISION]))
    monkeypatch.setattr(runner_mod, "_load_paper_funding", lambda symbols: {"AAAUSDT": rates})
    monkeypatch.setattr(runner_mod, "_load_paper_trade_closes", lambda symbols: {"AAAUSDT": closes})
    out = _settle_paper_funding_and_delistings(
        settings,
        audit,
        ledger_state=state,
        ledger_path=ledger_path,
        exchange_info=_paper_exchange_info(),
        run_id="20260824",
        decision_time=DECISION,
        now=NOW,
    )
    assert out.cash_before == Decimal("1000")
    assert out.ledger_state.cash_usdt == Decimal("999.9")
    assert out.ledger_state.positions == positions
    assert out.ledger_state.funding_watermarks == {"AAAUSDT": DECISION}
    assert load_ledger(ledger_path).cash_usdt == Decimal("999.9")
    assert len(out.funding_records) == 1
    assert out.funding_records[0].realized_pnl == -0.1
    tax_frame = load_tax_records(Path(settings.tax_ledger_dir))
    assert tax_frame["record_id"].tolist() == [out.funding_records[0].record_id]
    assert tax_frame["realized_pnl"].tolist() == [-0.1]


def test_portfolio_fail_soft_success_and_failure(monkeypatch, tmp_path) -> None:
    settings = _paper_settings(tmp_path, "pf")
    audit = AuditLog(tmp_path / "audit.jsonl")
    state = LedgerState(positions={}, cash_usdt=Decimal("1000"))
    _write_portfolio_state_fail_soft(
        settings,
        audit,
        final_state=state,
        snapshot=_flat_snapshot(),
        equity=Decimal("2000"),
        marks={},
        intent_count=0,
        dropped_fraction=0.0,
        decision_time=DECISION,
    )
    monkeypatch.setattr(
        runner_mod, "append_portfolio_state", lambda *a, **k: (_ for _ in ()).throw(OSError("x"))
    )
    _write_portfolio_state_fail_soft(
        settings,
        audit,
        final_state=state,
        snapshot=_flat_snapshot(),
        equity=Decimal("2000"),
        marks={},
        intent_count=0,
        dropped_fraction=0.0,
        decision_time=DECISION,
    )
    records = [json.loads(line) for line in audit.path.read_text().splitlines()]
    assert [(record["event"], record["error"]) for record in records] == [
        ("portfolio_state_write_failed", "x")
    ]


def test_tax_fail_soft_paths(monkeypatch, tmp_path) -> None:
    settings = _paper_settings(tmp_path, "tx")
    audit = AuditLog(tmp_path / "audit.jsonl")
    _collect_live_tax_fail_soft(
        settings, audit, order_client=object(), symbols=[], decision_time=DECISION, now=NOW
    )
    live_settings = _live_settings(tmp_path, "txl")

    def _corrupt(*a, **k):
        raise DataIntegrityError("watermark")

    monkeypatch.setattr(runner_mod, "collect_and_persist_live_tax", _corrupt)
    monkeypatch.setattr(runner_mod, "dispatch_alert", lambda *a, **k: None)
    _collect_live_tax_fail_soft(
        live_settings, audit, order_client=object(), symbols=["AAAUSDT"], decision_time=DECISION, now=NOW
    )
    class Issue:
        stream = "s"
        stage = "retention_gap"
        detail = "gap"

    monkeypatch.setattr(
        runner_mod, "collect_and_persist_live_tax", lambda *a, **k: ((), [Issue()])
    )
    _collect_live_tax_fail_soft(
        live_settings, audit, order_client=object(), symbols=["AAAUSDT"], decision_time=DECISION, now=NOW
    )
    monkeypatch.setattr(
        runner_mod,
        "collect_and_persist_live_tax",
        lambda *a, **k: (_ for _ in ()).throw(OSError("io")),
    )
    _collect_live_tax_fail_soft(
        live_settings, audit, order_client=object(), symbols=["AAAUSDT"], decision_time=DECISION, now=NOW
    )
    records = [json.loads(line) for line in audit.path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "tax_watermark_invalid", "tax_collect_issue", "tax_ledger_write_failed"
    ]
    assert records[0]["error"] == "watermark"
    assert records[1]["stage"] == "retention_gap"
    assert records[2]["error"] == "io"
