# ruff: noqa
"""Unresolved own-order guard: per-symbol freeze, DEGRADED report, episode alerts."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import src.live.runner as runner_mod
from src.live.errors import VenueError
from src.live.executor import UnresolvedOrder
from src.live.ledger import LedgerState, load_ledger, save_ledger
from src.live.order_journal import OrderJournal
from src.live.planner import build_client_order_id

from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient
from tests.unit.live.test_runner_reconcile import _live_settings, _tmp_state_kwargs

OLD_DT = DECISION_TIME - pd.Timedelta(days=1)
OLD_ID = build_client_order_id(OLD_DT.strftime("%Y%m%d"), "AAAUSDT", 0, 0, 0)

DAY2 = DECISION_TIME + pd.Timedelta(days=1)
DAY3 = DECISION_TIME + pd.Timedelta(days=2)


@pytest.fixture
def artifact(tmp_path):
    frame = pd.DataFrame(
        {"AAAUSDT": [0.02, 0.02, 0.02], "BUSDT": [-0.02, -0.02, -0.02]},
        index=pd.DatetimeIndex([DECISION_TIME, DAY2, DAY3]),
    )
    path = tmp_path / "deployed_target_weights.parquet"
    frame.to_parquet(path, index=True)
    closes = pd.DataFrame(
        {"AAAUSDT": [100.0, 100.0, 100.0], "BUSDT": [100.0, 100.0, 100.0]},
        index=pd.DatetimeIndex([DECISION_TIME, DAY2, DAY3]),
    )
    closes.to_parquet(tmp_path / "deployed_decision_ohlcv_close.parquet", index=True)
    return path


@pytest.fixture
def artifact_aaa(tmp_path):
    frame = pd.DataFrame(
        {"AAAUSDT": [0.02, 0.02, 0.02]},
        index=pd.DatetimeIndex([DECISION_TIME, DAY2, DAY3]),
    )
    path = tmp_path / "deployed_target_weights.parquet"
    frame.to_parquet(path, index=True)
    closes = pd.DataFrame(
        {"AAAUSDT": [100.0, 100.0, 100.0]},
        index=pd.DatetimeIndex([DECISION_TIME, DAY2, DAY3]),
    )
    closes.to_parquet(tmp_path / "deployed_decision_ohlcv_close.parquet", index=True)
    return path


def _seed_unresolved_submit(settings, client_order_id: str = OLD_ID, symbol: str = "AAAUSDT") -> None:
    journal = OrderJournal(Path(settings.order_journal_path))
    attempt = journal.begin_attempt(
        decision_time=OLD_DT, run_id=OLD_DT.strftime("%Y%m%d"), mode="live_testnet",
        pre_trade_equity=Decimal("2000"), sizing_anchor="decision_ohlcv_close",
        decision_marks={}, started_at=OLD_DT,
    )
    journal.record_submit(
        client_order_id, symbol, journal.next_submit_seq(), attempt_seq=attempt.attempt_seq,
        side="BUY", quantity=Decimal("0.4"), reduce_only=False, leg_index=0,
    )


class _VenueWithRestingOrder(StubOrderClient):
    def __init__(self, variant: str, position: str = "0") -> None:
        self.variant = variant
        self.position = position
        self.cancels: list[str] = []
        self.queries: list[str] = []
        self.query_answer: dict | None = None

    def request(self, method, path, params=None, *, signed=False):
        if path == "/fapi/v2/positionRisk":
            if self.position == "0":
                return []
            if self.position == "busdt_breach":
                return [{"symbol": "BUSDT", "positionAmt": "0.5"}]
            return [{"symbol": "AAAUSDT", "positionAmt": self.position}]
        return super().request(method, path, params, signed=signed)

    def open_orders(self):
        if self.variant == "cancel_2011":
            return [{"symbol": "AAAUSDT", "clientOrderId": OLD_ID, "side": "BUY"}]
        return []

    def cancel_order(self, symbol, orig_client_order_id):
        self.cancels.append(orig_client_order_id)
        raise VenueError("unknown order", code=-2011, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)

    def query_order(self, symbol, orig_client_order_id):
        self.queries.append(orig_client_order_id)
        if self.query_answer is not None:
            return dict(self.query_answer)
        if self.variant == "empty_status":
            return {"side": "BUY", "executedQty": "0"}
        return {"status": "NEW", "side": "BUY", "executedQty": "0", "avgPrice": "0", "updateTime": 1}


def _env(monkeypatch, tmp_path, client):
    import src.live.orderbook as ob_mod

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: client)
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")
    monkeypatch.setattr(ob_mod, "capture_order_books", lambda *a, **k: [])
    monkeypatch.setattr(ob_mod, "append_order_book_snapshots", lambda *a, **k: [])
    posted: list[Any] = []
    alerts: list[dict] = []

    def fake_execute_intents(c, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        posted.extend(intents)
        from src.live.executor import ExecutionOutcome

        outcomes = tuple(
            ExecutionOutcome(symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                             avg_fill_price=Decimal("100"), chases=0, status="FILLED")
            for i in intents
        )
        from tests.unit.live.test_runner_reconcile import _journal_fake_fills

        _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, outcomes)
        return outcomes

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)

    def fake_dispatch(settings, *, event, detail, decision_time, dedupe_key, now):
        alerts.append({"event": event, "detail": detail, "dedupe_key": dedupe_key})
        return True

    monkeypatch.setattr(runner_mod, "dispatch_alert", fake_dispatch)
    return posted, alerts


def _events(tmp_path: Path) -> list[dict]:
    path = tmp_path / "shadow_cycle.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("variant", ["not_listed", "cancel_2011"])
def test_unresolved_order_freezes_symbol_and_degrades_cycle(variant, artifact, monkeypatch, tmp_path) -> None:
    client = _VenueWithRestingOrder(variant)
    posted, alerts = _env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, f"guard_{variant}")
    save_ledger(Path(settings.ledger_path), LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    _seed_unresolved_submit(settings)

    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "DEGRADED"
    assert "unresolved_own_orders" in (report.reason or "")
    assert [i.symbol for i in posted if i.symbol == "AAAUSDT"] == []
    assert [i.symbol for i in posted if i.symbol == "BUSDT"]
    ev = _events(tmp_path)
    rec = [e for e in ev if e["event"] == "order_recovery_unresolved"]
    assert len(rec) == 1 and OLD_ID in rec[0]["client_order_ids"]
    assert [a for a in alerts if a["event"] == "order_recovery_unresolved"]


def test_unresolved_order_does_not_stamp_decision(artifact, monkeypatch, tmp_path) -> None:
    client = _VenueWithRestingOrder("not_listed")
    _env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "guard_nostamp")
    save_ledger(Path(settings.ledger_path), LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    _seed_unresolved_submit(settings)

    runner_mod.run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    state = load_ledger(Path(settings.ledger_path))
    assert state.last_executed_decision_time != DECISION_TIME
    assert state.derisk_since is None
    assert state.derisk_reasons == ()


def test_orphan_cancel_without_confirmation_writes_no_orphan_cancelled(artifact, monkeypatch, tmp_path) -> None:
    client = _VenueWithRestingOrder("cancel_2011")
    _env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "guard_orphan")
    save_ledger(Path(settings.ledger_path), LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    _seed_unresolved_submit(settings)

    runner_mod.run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    ev = _events(tmp_path)
    assert [e for e in ev if e["event"] == "orphan_cancel_unconfirmed" and e["client_order_id"] == OLD_ID]
    assert not [e for e in ev if e["event"] == "orphan_cancelled"]


def test_status_less_recovery_answer_halts_cycle(artifact, monkeypatch, tmp_path) -> None:
    client = _VenueWithRestingOrder("empty_status")
    posted, _alerts = _env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "guard_halt")
    save_ledger(Path(settings.ledger_path), LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    _seed_unresolved_submit(settings)

    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "HALT"
    assert posted == []


def test_confirmed_cancel_clears_freeze(artifact, monkeypatch, tmp_path) -> None:
    class _ConfirmingVenue(_VenueWithRestingOrder):
        def __init__(self) -> None:
            super().__init__("not_listed")
            self.seen_cancel = False

        def cancel_order(self, symbol, orig_client_order_id):
            self.cancels.append(orig_client_order_id)
            self.seen_cancel = True
            return {}

        def query_order(self, symbol, orig_client_order_id):
            self.queries.append(orig_client_order_id)
            if self.seen_cancel:
                return {"status": "CANCELED", "side": "BUY", "executedQty": "0", "avgPrice": "0", "updateTime": 1}
            return {"status": "NEW", "side": "BUY", "executedQty": "0", "avgPrice": "0", "updateTime": 1}

    client = _ConfirmingVenue()
    posted, _alerts = _env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "guard_clear")
    save_ledger(Path(settings.ledger_path), LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    _seed_unresolved_submit(settings)

    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "COMPLETE"
    journal = OrderJournal(Path(settings.order_journal_path))
    assert journal.observed_qty(OLD_ID) == Decimal("0")
    assert [i.symbol for i in posted if i.symbol == "AAAUSDT"]


def test_partial_fill_of_still_open_order_booked_before_reconciliation(artifact, monkeypatch, tmp_path) -> None:
    fill_ms = int((NOW - pd.Timedelta(minutes=5)).value // 1_000_000)

    class _PartialVenue(_VenueWithRestingOrder):
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/positionRisk":
                return [{"symbol": "AAAUSDT", "positionAmt": "0.1"}]
            return super().request(method, path, params, signed=signed)

        def query_order(self, symbol, orig_client_order_id):
            self.queries.append(orig_client_order_id)
            return {"status": "PARTIALLY_FILLED", "side": "BUY", "executedQty": "0.1", "avgPrice": "100", "updateTime": fill_ms}

    client = _PartialVenue("not_listed")
    posted, _alerts = _env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "guard_partial")
    save_ledger(Path(settings.ledger_path), LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    _seed_unresolved_submit(settings)

    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    journal = OrderJournal(Path(settings.order_journal_path))
    rec = [f for f in journal.fills_after(-1) if f.kind == "recovered" and f.client_order_id == OLD_ID]
    assert len(rec) == 1 and rec[0].quantity == Decimal("0.1")
    assert load_ledger(Path(settings.ledger_path)).positions.get("AAAUSDT") == Decimal("0.1")
    assert "reconciliation_breach" not in (report.reason or "")
    assert [i.symbol for i in posted if i.symbol == "AAAUSDT"] == []


def test_reduce_only_exit_on_frozen_symbol_is_blocked(artifact, monkeypatch, tmp_path) -> None:
    class _ReduceVenue(_VenueWithRestingOrder):
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/positionRisk":
                return [{"symbol": "AAAUSDT", "positionAmt": "0.8"}]
            return super().request(method, path, params, signed=signed)

    client = _ReduceVenue("not_listed")
    posted, _alerts = _env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "guard_reduce")
    save_ledger(Path(settings.ledger_path), LedgerState(positions={"AAAUSDT": Decimal("0.8")}, equity_high_water_mark=Decimal(0)))
    _seed_unresolved_submit(settings)

    runner_mod.run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert [i.symbol for i in posted if i.symbol == "AAAUSDT"] == []
    blocked = [e for e in _events(tmp_path) if e["event"] == "unresolved_order_blocked" and e["symbol"] == "AAAUSDT"]
    assert blocked and all(e["reduce_only"] is True and e["reason"] == "unresolved_order_symbol" for e in blocked)


def test_freeze_composes_with_derisk_without_persisting(artifact, monkeypatch, tmp_path) -> None:
    class _BreachFreezeVenue(_VenueWithRestingOrder):
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/positionRisk":
                return [{"symbol": "BUSDT", "positionAmt": "0.5"}]
            return StubOrderClient.request(self, method, path, params, signed=signed)

    client = _BreachFreezeVenue("not_listed")
    _env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "guard_compose")
    save_ledger(Path(settings.ledger_path), LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    _seed_unresolved_submit(settings)

    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.reason == "reconciliation_breach,unresolved_own_orders"
    assert load_ledger(Path(settings.ledger_path)).derisk_reasons == ("reconciliation_breach",)


def test_late_fill_after_freeze_converges_without_over_exposure(artifact_aaa, monkeypatch, tmp_path) -> None:
    client = _VenueWithRestingOrder("not_listed")
    posted, _alerts = _env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "guard_late")
    save_ledger(Path(settings.ledger_path), LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    _seed_unresolved_submit(settings)

    r1 = runner_mod.run_shadow_cycle(settings, DECISION_TIME, artifact_aaa, now=NOW)
    n1 = len(posted)
    assert [i.symbol for i in posted if i.symbol == "AAAUSDT"] == []

    client.position = "0.4"
    fill_ms = int((NOW + pd.Timedelta(hours=3)).value // 1_000_000)
    client.query_answer = {"status": "FILLED", "side": "BUY", "executedQty": "0.4", "avgPrice": "97", "updateTime": fill_ms}
    r2 = runner_mod.run_shadow_cycle(settings, DAY2, artifact_aaa, now=NOW + pd.Timedelta(days=1))

    journal = OrderJournal(Path(settings.order_journal_path))
    rec = [(f.kind, f.symbol, str(f.quantity)) for f in journal.fills_after(-1) if f.kind == "recovered"]
    assert ("recovered", "AAAUSDT", "0.4") in rec
    assert "unresolved_own_orders" not in (r2.reason or "")
    assert r1.status == "DEGRADED"


def test_ticker_marks_return_quotes_without_function_state(monkeypatch) -> None:
    from src.live.runner import _marks_from_tickers

    class _A(StubMarketClient):
        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "1", "askPrice": "2", "bidQty": "1", "askQty": "1"}}

    result = _marks_from_tickers(_A(), ["AAAUSDT"])
    assert set(result.quotes) == {"AAAUSDT"}
    assert result.marks["AAAUSDT"] == result.quotes["AAAUSDT"].mid
    assert not hasattr(_marks_from_tickers, "_last_quotes")


def test_failed_ticker_fetch_leaves_no_stale_quotes() -> None:
    from src.live.runner import _marks_from_tickers

    class _A(StubMarketClient):
        def book_tickers(self):
            return {"AAAUSDT": {"symbol": "AAAUSDT", "bidPrice": "1", "askPrice": "2", "bidQty": "1", "askQty": "1"}}

    class _B(StubMarketClient):
        def book_tickers(self):
            raise RuntimeError("feed down")

    _marks_from_tickers(_A(), ["AAAUSDT"])
    with pytest.raises(RuntimeError):
        _marks_from_tickers(_B(), ["AAAUSDT"])
    assert not hasattr(_marks_from_tickers, "_last_quotes")


def test_unresolved_episode_alerts_once_across_cycles(artifact_aaa, monkeypatch, tmp_path) -> None:
    import src.live.alerting as alerting_mod
    from src.live.alert_outbox import AlertOutbox, resolve_outbox_path

    client = _VenueWithRestingOrder("cancel_2011")
    posted_all, _ = _env(monkeypatch, tmp_path, client)
    # Use the durable outbox path for real dispatch; capture dispatch keys.
    keys: list[str] = []
    real_dispatch = alerting_mod.dispatch_alert
    monkeypatch.setattr(runner_mod, "dispatch_alert", real_dispatch)
    orig_enqueue = AlertOutbox.enqueue

    def _spy_enqueue(self, **kwargs):
        keys.append(kwargs.get("dedupe_key", ""))
        return orig_enqueue(self, **kwargs)

    monkeypatch.setattr(AlertOutbox, "enqueue", _spy_enqueue)
    monkeypatch.setattr(alerting_mod, "post_alert", lambda *a, **k: True)
    settings = _live_settings(
        tmp_path, "guard_episode",
        alert_outbox_path=str(tmp_path / "outbox.json"),
        alert_webhook_url="https://hooks.example.com/x",
    )
    save_ledger(Path(settings.ledger_path), LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    _seed_unresolved_submit(settings)

    for day, now in [(DECISION_TIME, NOW), (DAY2, NOW + pd.Timedelta(days=1)), (DAY3, NOW + pd.Timedelta(days=2))]:
        report = runner_mod.run_shadow_cycle(settings, day, artifact_aaa, now=now)
        assert report.status == "DEGRADED"
        assert [i.symbol for i in posted_all if i.symbol == "AAAUSDT"] == []

    audits = [e for e in _events(tmp_path) if e["event"] == "order_recovery_unresolved"]
    assert len(audits) == 3
    episode_keys = [k for k in keys if k.startswith("order_recovery_unresolved:")]
    assert episode_keys and all(k == f"order_recovery_unresolved:AAAUSDT:{OLD_ID}" for k in episode_keys)
    raw = json.loads(Path(resolve_outbox_path(settings)).read_text(encoding="utf-8"))
    stored = [r for r in raw["records"] if r["event"] == "order_recovery_unresolved"]
    assert len(stored) == 1 and stored[0]["dedupe_key"] == f"order_recovery_unresolved:AAAUSDT:{OLD_ID}"
    assert not [k for k in keys if k.startswith("cycle_degraded:")]


def test_freeze_clears_automatically_once_order_terminal(artifact_aaa, monkeypatch, tmp_path) -> None:
    client = _VenueWithRestingOrder("cancel_2011")
    posted, alerts = _env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "guard_selfclear")
    save_ledger(Path(settings.ledger_path), LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    _seed_unresolved_submit(settings)

    r1 = runner_mod.run_shadow_cycle(settings, DECISION_TIME, artifact_aaa, now=NOW)
    assert r1.status == "DEGRADED"
    n1_alerts = len([a for a in alerts if a["event"] == "order_recovery_unresolved"])
    assert n1_alerts == 1

    fill_ms = int((NOW + pd.Timedelta(hours=3)).value // 1_000_000)
    client.variant = "terminal"
    client.query_answer = {"status": "CANCELED", "side": "BUY", "executedQty": "0", "avgPrice": "0", "updateTime": fill_ms}
    client.open_orders = lambda: []  # type: ignore[method-assign]
    n1_posted = len(posted)
    audits_before = len(_events(tmp_path))

    r2 = runner_mod.run_shadow_cycle(settings, DAY2, artifact_aaa, now=NOW + pd.Timedelta(days=1))

    assert r2.status == "COMPLETE"
    assert len(posted) > n1_posted and [i.symbol for i in posted[n1_posted:] if i.symbol == "AAAUSDT"]
    cycle2_audits = _events(tmp_path)[audits_before:]
    assert not [e for e in cycle2_audits if e["event"] == "order_recovery_unresolved"]
    assert len([a for a in alerts if a["event"] == "order_recovery_unresolved"]) == n1_alerts
    state = load_ledger(Path(settings.ledger_path))
    assert state.derisk_reasons == ()
    assert state.last_executed_decision_time == DAY2


def test_new_episode_alerts_again(artifact_aaa, monkeypatch, tmp_path) -> None:
    import src.live.alerting as alerting_mod
    from src.live.alert_outbox import AlertOutbox, resolve_outbox_path

    client = _VenueWithRestingOrder("cancel_2011")
    _env(monkeypatch, tmp_path, client)
    real_dispatch = alerting_mod.dispatch_alert
    monkeypatch.setattr(runner_mod, "dispatch_alert", real_dispatch)
    monkeypatch.setattr(alerting_mod, "post_alert", lambda *a, **k: True)
    settings = _live_settings(
        tmp_path, "guard_newep",
        alert_outbox_path=str(tmp_path / "outbox_new.json"),
        alert_webhook_url="https://hooks.example.com/x",
    )
    save_ledger(Path(settings.ledger_path), LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    _seed_unresolved_submit(settings)

    r1 = runner_mod.run_shadow_cycle(settings, DECISION_TIME, artifact_aaa, now=NOW)
    assert r1.status == "DEGRADED"

    fill_ms = int((NOW + pd.Timedelta(hours=3)).value // 1_000_000)
    client.query_answer = {"status": "CANCELED", "side": "BUY", "executedQty": "0", "avgPrice": "0", "updateTime": fill_ms}
    client.open_orders = lambda: []  # type: ignore[method-assign]
    r2 = runner_mod.run_shadow_cycle(settings, DAY2, artifact_aaa, now=NOW + pd.Timedelta(days=1))
    assert r2.status == "COMPLETE"

    new_id = build_client_order_id(DAY2.strftime("%Y%m%d"), "AAAUSDT", 0, 0, 0)
    assert new_id != OLD_ID
    journal = OrderJournal(Path(settings.order_journal_path))
    attempt = journal.begin_attempt(
        decision_time=DAY2, run_id=DAY2.strftime("%Y%m%d"), mode="live_testnet",
        pre_trade_equity=Decimal("2000"), sizing_anchor="decision_ohlcv_close",
        decision_marks={}, started_at=DAY2,
    )
    journal.record_submit(new_id, "AAAUSDT", journal.next_submit_seq(), attempt_seq=attempt.attempt_seq,
                          side="BUY", quantity=Decimal("0.4"), reduce_only=False, leg_index=0)
    client.query_answer = None
    client.variant = "cancel_2011"
    client.open_orders = lambda: [{"symbol": "AAAUSDT", "clientOrderId": new_id, "side": "BUY"}]  # type: ignore[method-assign]
    orig_query = client.query_order

    def _query(symbol, oid):
        client.queries.append(oid)
        return {"status": "NEW", "side": "BUY", "executedQty": "0", "avgPrice": "0", "updateTime": 1}

    client.query_order = _query  # type: ignore[method-assign]
    r3 = runner_mod.run_shadow_cycle(settings, DAY3, artifact_aaa, now=NOW + pd.Timedelta(days=2))
    assert r3.status == "DEGRADED"

    raw = json.loads(Path(resolve_outbox_path(settings)).read_text(encoding="utf-8"))
    stored = [r for r in raw["records"] if r["event"] == "order_recovery_unresolved" and "AAAUSDT" in r["dedupe_key"]]
    assert sorted(r["dedupe_key"] for r in stored) == sorted([
        f"order_recovery_unresolved:AAAUSDT:{OLD_ID}",
        f"order_recovery_unresolved:AAAUSDT:{new_id}",
    ])


def test_one_alert_per_frozen_symbol(monkeypatch, tmp_path) -> None:
    from src.live.runner import _record_unresolved_orders

    settings = _live_settings(tmp_path, "guard_persym")
    import src.live.runner as rm

    calls: list[dict] = []
    monkeypatch.setattr(rm, "dispatch_alert", lambda s, *, event, detail, decision_time, dedupe_key, now: calls.append({"event": event, "detail": detail, "dedupe_key": dedupe_key}) or True)
    monkeypatch.setattr(rm, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")
    from src.live.audit import AuditLog

    audit = AuditLog(tmp_path / "audit_persym.jsonl")
    try:
        frozen = _record_unresolved_orders(
            settings, audit,
            [UnresolvedOrder(client_order_id="b", symbol="AAAUSDT"),
             UnresolvedOrder(client_order_id="a", symbol="AAAUSDT"),
             UnresolvedOrder(client_order_id="c", symbol="BUSDT")],
            decision_time=DECISION_TIME, now=NOW,
        )
    finally:
        audit.close()

    assert frozen == ("AAAUSDT", "BUSDT")
    assert sorted(c["dedupe_key"] for c in calls) == [
        "order_recovery_unresolved:AAAUSDT:a",
        "order_recovery_unresolved:BUSDT:c",
    ]
