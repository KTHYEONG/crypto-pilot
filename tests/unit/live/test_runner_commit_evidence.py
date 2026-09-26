"""Evidence emission guards of the journal-driven ledger commit (PAPER)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd

import src.live.runner as runner_mod
from src.live.audit import AuditLog
from src.live.ledger import LedgerState, load_ledger, save_ledger
from src.live.order_journal import JournalAttempt, OrderJournal
from src.live.settings import LiveSettings

_T = pd.Timestamp("2026-09-20T23:10:00Z")


def _settings(tmp_path: Path) -> LiveSettings:
    return LiveSettings(
        mode="paper",
        notional_equity_usdt=2000.0,
        ledger_path=str(tmp_path / "ledger.json"),
        order_journal_path=str(tmp_path / "journal.jsonl"),
        fills_dir=str(tmp_path / "fills"),
        tax_ledger_dir=str(tmp_path / "tax"),
    )


def _attempt() -> JournalAttempt:
    return JournalAttempt(
        attempt_seq=-1, decision_time=pd.Timestamp("2026-09-20T00:00:00Z"), run_id="20260920",
        mode="paper", pre_trade_equity=Decimal("2000"), sizing_anchor="decision_ohlcv_close",
        decision_marks={}, started_at=_T,
    )


def _record_buy(journal: OrderJournal) -> None:
    journal.record_fill(
        kind="execution", attempt_seq=None, symbol="AAAUSDT", side="BUY", quantity=Decimal("1"),
        price=Decimal("100"), fee_bps=5.0, liquidity="taker", reason="maker_fill", filled_at=_T,
        client_order_id="mhA", leg_index=0, cumulative_executed_qty=None, simulated=True,
    )


def _capture_alerts(monkeypatch) -> list[str]:
    events: list[str] = []
    monkeypatch.setattr(
        runner_mod, "dispatch_alert",
        lambda settings, *, event, detail, decision_time, dedupe_key, now: events.append(event) or True,
    )
    return events


def test_evidence_retry_of_already_applied_fills_does_not_raise_cash_mismatch(tmp_path, monkeypatch) -> None:
    """Fills applied by an earlier call but not yet recorded are re-emitted without a false cash alarm."""
    events = _capture_alerts(monkeypatch)
    settings = _settings(tmp_path)
    journal = OrderJournal(Path(settings.order_journal_path))
    _record_buy(journal)
    # 앞선 호출이 원장 반영까지 끝내고 증거 기록 전에 중단된 상태.
    state = LedgerState(
        positions={"AAAUSDT": Decimal("1")}, equity_high_water_mark=Decimal(0),
        cash_usdt=Decimal("1899.95"), journal_applied_fill_seq=0, journal_recorded_fill_seq=-1,
    )
    save_ledger(Path(settings.ledger_path), state)

    committed = runner_mod._commit_and_record(
        settings, Path(settings.ledger_path), state, journal, AuditLog(tmp_path / "audit.jsonl"),
        fallback_attempt=_attempt(), equity=None, executed_decision_time=None,
    )

    assert "ledger_reconcile_mismatch" not in events
    assert committed.journal_recorded_fill_seq == 0
    assert load_ledger(Path(settings.ledger_path)).cash_usdt == Decimal("1899.95")


def test_corrupt_tax_shard_on_trade_path_alerts(tmp_path, monkeypatch) -> None:
    """A mid-file corrupt tax shard on the PAPER TRADE path sends tax_ledger_corrupt, not just an audit line."""
    events = _capture_alerts(monkeypatch)
    settings = _settings(tmp_path)
    tax_dir = Path(settings.tax_ledger_dir)
    tax_dir.mkdir(parents=True)
    (tax_dir / "tax_ledger_202609.jsonl").write_text('not json\n{"record_id": "x"}\n', encoding="utf-8")
    journal = OrderJournal(Path(settings.order_journal_path))
    _record_buy(journal)
    state = LedgerState(positions={}, equity_high_water_mark=Decimal(0), cash_usdt=Decimal("2000"))
    save_ledger(Path(settings.ledger_path), state)

    committed = runner_mod._commit_and_record(
        settings, Path(settings.ledger_path), state, journal, AuditLog(tmp_path / "audit.jsonl"),
        fallback_attempt=_attempt(), equity=None, executed_decision_time=None,
    )

    assert events.count("tax_ledger_corrupt") == 1
    assert committed.journal_recorded_fill_seq == -1
