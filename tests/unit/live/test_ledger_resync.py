# ruff: noqa
"""Live ledger_resync tests - operator resync clears de-risk mode."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import src.live.runner as runner_mod
import src.live.ledger_resync as resync_mod
from src.live.errors import LiveTradingError
from src.live.ledger import LedgerState, enter_derisk, load_ledger, save_ledger
from src.live.settings import LiveSettings

from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient


class _ResyncOrderClient(StubOrderClient):
    """Venue AAAUSDT 0.5; no open orders."""

    def request(self, method, path, params=None, *, signed=False):
        if path == "/fapi/v2/positionRisk":
            return [{"symbol": "AAAUSDT", "positionAmt": "0.5"}]
        return super().request(method, path, params, signed=signed)

    def open_orders(self):
        return []

    def query_order(self, symbol, orig_client_order_id):
        return {"status": "CANCELED", "side": "BUY", "avgPrice": "100", "executedQty": "0"}


@pytest.fixture
def resync_env(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    client = _ResyncOrderClient()
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: client)
    monkeypatch.setattr(
        resync_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl"
    )
    monkeypatch.setattr(
        runner_mod, "dispatch_alert", lambda settings, **kwargs: True,
    )
    import src.live.orderbook as ob_mod

    monkeypatch.setattr(ob_mod, "capture_order_books", lambda *a, **k: [])
    monkeypatch.setattr(ob_mod, "append_order_book_snapshots", lambda *a, **k: [])
    return client


def _resync_settings(tmp_path: Path, name: str) -> LiveSettings:
    return LiveSettings(
        mode="live_testnet",
        notional_equity_usdt=2000.0,
        ledger_path=str(tmp_path / f"ledger_{name}.json"),
        order_journal_path=str(tmp_path / f"journal_{name}.jsonl"),
        heartbeat_path=str(tmp_path / f"hb_{name}.json"),
        ledger_resync_backup_dir=str(tmp_path / f"backups_{name}"),
        fills_dir=str(tmp_path / f"fills_{name}"),
        tax_ledger_dir=str(tmp_path / f"tax_{name}"),
        execution_quality_dir=str(tmp_path / f"eq_{name}"),
        portfolio_state_dir=str(tmp_path / f"port_{name}"),
        microstructure_dir=str(tmp_path / f"micro_{name}"),
    )


def test_ledger_resync_dry_run_writes_nothing(resync_env, tmp_path) -> None:
    """Dry run writes nothing."""
    settings = _resync_settings(tmp_path, "dry")
    ledger_path = Path(settings.ledger_path)
    save_ledger(ledger_path, LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    before = ledger_path.read_bytes()
    journal_path = Path(settings.order_journal_path)
    backup_dir = Path(settings.ledger_resync_backup_dir)

    plan = resync_mod.run_ledger_resync(settings, apply=False, now=NOW)

    assert [b.symbol for b in plan.adjustments] == ["AAAUSDT"]
    assert plan.applied is False
    assert ledger_path.read_bytes() == before
    assert not journal_path.exists()
    assert not backup_dir.exists()


def test_ledger_resync_apply_adopts_snapshot_and_clears_flag(resync_env, tmp_path) -> None:
    """Apply adopts the venue snapshot and clears the flag."""
    settings = _resync_settings(tmp_path, "apply")
    ledger_path = Path(settings.ledger_path)
    save_ledger(ledger_path, LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    state = load_ledger(ledger_path)
    enter_derisk(ledger_path, state, reasons=("reconciliation_breach",), now=NOW)
    before = ledger_path.read_bytes()
    journal_path = Path(settings.order_journal_path)

    plan = resync_mod.run_ledger_resync(settings, apply=True, now=NOW)

    assert plan.applied is True
    assert plan.backup_path is not None
    assert plan.backup_path.exists()
    assert plan.backup_path.read_bytes() == before
    from src.live.order_journal import OrderJournal

    fills = [f for f in OrderJournal(journal_path).fills_after(-1) if f.kind == "operator_resync"]
    assert len(fills) == 1
    assert (fills[0].symbol, fills[0].side) == ("AAAUSDT", "BUY")
    after = load_ledger(ledger_path)
    assert after.positions == {"AAAUSDT": Decimal("0.5")}
    assert after.derisk_since is None
    assert after.cash_usdt is None


def test_ledger_resync_busy_daemon_blocks_apply(resync_env, tmp_path) -> None:
    """Busy daemon blocks apply."""
    import json

    settings = _resync_settings(tmp_path, "busy")
    ledger_path = Path(settings.ledger_path)
    save_ledger(ledger_path, LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    Path(settings.heartbeat_path).write_text(
        json.dumps({"stage": "execute", "status": "RUNNING", "ts": NOW.isoformat()}),
        encoding="utf-8",
    )

    with pytest.raises(LiveTradingError, match="daemon busy"):
        resync_mod.run_ledger_resync(settings, apply=True, now=NOW)
    assert not Path(settings.order_journal_path).exists()
    assert not Path(settings.ledger_resync_backup_dir).exists()


def test_ledger_resync_paper_mode_is_rejected(resync_env, tmp_path) -> None:
    """PAPER mode is rejected."""
    paper = LiveSettings(
        mode="paper",
        notional_equity_usdt=2000.0,
        ledger_path=str(tmp_path / "ledger_paper_reject.json"),
        heartbeat_path=str(tmp_path / "hb_paper_reject.json"),
    )
    with pytest.raises(LiveTradingError):
        resync_mod.run_ledger_resync(paper, apply=False, now=NOW)


def test_ledger_resync_second_apply_is_noop(resync_env, tmp_path) -> None:
    """Second apply is a no-op."""
    from src.live.order_journal import OrderJournal

    settings = _resync_settings(tmp_path, "noop")
    ledger_path = Path(settings.ledger_path)
    save_ledger(ledger_path, LedgerState(positions={}, equity_high_water_mark=Decimal(0)))

    first = resync_mod.run_ledger_resync(settings, apply=True, now=NOW)
    assert first.applied is True
    journal_path = Path(settings.order_journal_path)
    count = len(OrderJournal(journal_path).fills_after(-1))
    second = resync_mod.run_ledger_resync(settings, apply=True, now=NOW + pd.Timedelta(minutes=1))

    assert second.adjustments == ()
    assert len(OrderJournal(journal_path).fills_after(-1)) == count


def test_ledger_resync_backup_survives_unfsyncable_directory(resync_env, tmp_path, monkeypatch) -> None:
    """A directory that cannot be fsynced still leaves a valid backup."""
    import os

    real_open = os.open

    def _boom_dir(path, *args, **kwargs):
        if os.path.isdir(path):
            raise OSError("read-only mount")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", _boom_dir)
    settings = _resync_settings(tmp_path, "rodir")
    save_ledger(Path(settings.ledger_path), LedgerState(positions={}, equity_high_water_mark=Decimal("0")))

    plan = resync_mod.run_ledger_resync(settings, apply=True, now=NOW)

    assert plan.applied is True
    assert plan.backup_path is not None
    assert plan.backup_path.exists()


def test_ledger_resync_rejects_naive_now(resync_env, tmp_path) -> None:
    """A naive wall clock fails closed."""
    import pandas as pd
    import pytest

    settings = _resync_settings(tmp_path, "naive")
    with pytest.raises(ValueError, match="tz-aware"):
        resync_mod.run_ledger_resync(settings, apply=False, now=pd.Timestamp("2026-09-14 00:00"))


def test_ledger_resync_uses_default_backup_dir(resync_env, tmp_path, monkeypatch) -> None:
    """Without an override the backup lands under the state dir."""
    settings = _resync_settings(tmp_path, "defaultdir")
    settings.ledger_resync_backup_dir = None
    ledger_path = Path(settings.ledger_path)
    save_ledger(ledger_path, LedgerState(positions={}, equity_high_water_mark=Decimal("0")))
    monkeypatch.setattr("src.common.paths.DATA_DIR", tmp_path / "data_root")

    plan = resync_mod.run_ledger_resync(settings, apply=True, now=NOW)

    assert plan.applied is True
    assert plan.backup_path is not None
    assert plan.backup_path.parent == tmp_path / "data_root" / "state" / "ledger_backups"


def test_ledger_resync_apply_rejects_foreign_orders(tmp_path, monkeypatch) -> None:
    """Apply refuses while foreign open orders exist."""
    import pytest

    class _ForeignClient(_ResyncOrderClient):
        def open_orders(self):
            return [{"symbol": "AAAUSDT", "clientOrderId": "web_manual_1"}]

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: _ForeignClient())
    monkeypatch.setattr(
        resync_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl"
    )
    settings = _resync_settings(tmp_path, "foreign")
    save_ledger(Path(settings.ledger_path), LedgerState(positions={}, equity_high_water_mark=Decimal(0)))

    with pytest.raises(LiveTradingError, match="foreign open orders"):
        resync_mod.run_ledger_resync(settings, apply=True, now=NOW)


def test_ledger_resync_apply_fails_when_verification_breaches(resync_env, tmp_path, monkeypatch) -> None:
    """A post-commit breach fails closed without clearing the flag."""
    import pytest

    settings = _resync_settings(tmp_path, "verify")
    ledger_path = Path(settings.ledger_path)
    save_ledger(ledger_path, LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    state = load_ledger(ledger_path)
    enter_derisk(ledger_path, state, reasons=("reconciliation_breach",), now=NOW)
    from src.live.errors import ReconciliationBreach

    monkeypatch.setattr(
        resync_mod, "reconcile_or_halt",
        lambda *a, **k: (_ for _ in ()).throw(ReconciliationBreach("position divergence gap")),
    )

    with pytest.raises(LiveTradingError, match="verification"):
        resync_mod.run_ledger_resync(settings, apply=True, now=NOW)
    assert load_ledger(ledger_path).derisk_since is not None



def test_ledger_resync_dry_run_reports_only_foreign_order_symbols(tmp_path, monkeypatch) -> None:
    """Dry run lists symbols with non-namespaced orders and ignores our own order ids."""
    from src.live.executor import CLIENT_ORDER_NAMESPACE

    class _MixedClient(_ResyncOrderClient):
        def open_orders(self):
            return [
                {"symbol": "AAAUSDT", "clientOrderId": "web_manual_1"},
                {"symbol": "BBBUSDT", "clientOrderId": f"{CLIENT_ORDER_NAMESPACE}20260925-1"},
            ]

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: _MixedClient())
    audit_path = tmp_path / "ledger_resync.jsonl"
    monkeypatch.setattr(resync_mod, "default_audit_log_path", lambda name, for_date=None: audit_path)
    settings = _resync_settings(tmp_path, "mixed")
    save_ledger(Path(settings.ledger_path), LedgerState(positions={}, equity_high_water_mark=Decimal(0)))

    plan = resync_mod.run_ledger_resync(settings, apply=False, now=NOW)

    assert plan.applied is False
    import json

    planned = [json.loads(line) for line in audit_path.read_text().splitlines() if "ledger_resync_planned" in line]
    assert planned[-1]["foreign_symbols"] == ["AAAUSDT"]


def test_ledger_resync_apply_without_valid_mark_writes_nothing(resync_env, tmp_path, monkeypatch) -> None:
    """A breach symbol without a positive mark fails closed; no resync fill and no ledger change."""
    settings = _resync_settings(tmp_path, "unpriced")
    ledger_path = Path(settings.ledger_path)
    save_ledger(ledger_path, LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    before = ledger_path.read_bytes()
    monkeypatch.setattr(runner_mod, "_marks_from_tickers", lambda client, symbols: {})

    with pytest.raises(LiveTradingError, match="no valid mark"):
        resync_mod.run_ledger_resync(settings, apply=True, now=NOW)

    from src.live.order_journal import OrderJournal

    assert ledger_path.read_bytes() == before
    journal_path = Path(settings.order_journal_path)
    if journal_path.exists():
        assert not [f for f in OrderJournal(journal_path).fills_after(-1) if f.kind == "operator_resync"]
