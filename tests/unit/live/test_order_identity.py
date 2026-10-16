from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.live.audit import AuditLog
from src.live.executor import FeeSchedule, _IntentRuntime, _sync_venue_order
from src.live.filters import SymbolFilters
from src.live.order_identity import StaleOrderIdentity, assert_order_identity
from src.live.order_journal import OrderJournal
from src.live.planner import OrderIntent
from src.live.recovery import recover_unresolved_orders

_SUBMIT = pd.Timestamp("2026-10-14T00:00:00Z")
_TOL = 35_000


def _payload(time_ms: object, **overrides):
    base = {
        "status": "FILLED",
        "executedQty": "1.0",
        "avgPrice": "100",
        "side": "BUY",
        "time": time_ms,
        "updateTime": time_ms if isinstance(time_ms, int) else 1,
        "orderId": 4224402074,
    }
    base.update(overrides)
    return base


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)


def test_fresh_order_passes() -> None:
    assert_order_identity(
        _payload(_ms(_SUBMIT) + 50),
        client_order_id="oid",
        submitted_at=_SUBMIT,
        tolerance_ms=_TOL,
    )


def test_old_order_with_reused_id_rejected() -> None:
    with pytest.raises(StaleOrderIdentity):
        assert_order_identity(
            _payload(_ms(_SUBMIT) - 85_000),
            client_order_id="oid",
            submitted_at=_SUBMIT,
            tolerance_ms=_TOL,
        )


def test_clock_skew_within_tolerance_passes() -> None:
    assert_order_identity(
        _payload(_ms(_SUBMIT) - 1_000),
        client_order_id="oid",
        submitted_at=_SUBMIT,
        tolerance_ms=_TOL,
    )


@pytest.mark.parametrize("bad", [None, "x", 0, -1, float("nan"), float("inf"), True, "0.5", Decimal("NaN"), "secret-signature"])
def test_missing_or_invalid_time_fails_closed(bad) -> None:
    with pytest.raises(DataIntegrityError):
        assert_order_identity(
            _payload(bad),
            client_order_id="oid",
            submitted_at=_SUBMIT,
            tolerance_ms=_TOL,
        )


def _runtime(journal: OrderJournal | None, order_id: str) -> _IntentRuntime:
    intent = OrderIntent(
        symbol="AAAUSDT",
        side="BUY",
        quantity=Decimal("1"),
        reduce_only=False,
        target_qty=Decimal("1"),
        current_qty=Decimal("0"),
        client_order_prefix="run1",
        leg_index=0,
        decision_price=Decimal("100"),
    )
    filters = SymbolFilters(
        symbol="AAAUSDT",
        tick_size=Decimal("0.10"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
        max_qty=Decimal("1000000"),
        quantity_precision=3,
        price_precision=2,
    )
    rt = _IntentRuntime(intent=intent, filters=filters, fee_schedule=FeeSchedule(2.0, 5.0))
    rt.journal = journal
    rt.journal_enabled = journal is not None
    rt.active_id = order_id
    rt.active_price = Decimal("100")
    rt.active_post_qty = Decimal("1")
    rt.order_identity_tolerance_ms = _TOL
    return rt


def test_executor_books_no_phantom_fill(tmp_path) -> None:
    journal = OrderJournal(tmp_path / "j.jsonl")
    attempt = journal.begin_attempt(
        decision_time=_SUBMIT,
        run_id="run1",
        mode="LIVE",
        pre_trade_equity=Decimal("1000"),
        sizing_anchor="equity",
        decision_marks={},
        started_at=_SUBMIT,
    )
    order_id = "mh20261014-ABCDEFGHIJ-0-0-1"
    journal.record_submit(
        order_id, "AAAUSDT", journal.next_submit_seq(), attempt_seq=attempt.attempt_seq,
        side="BUY", quantity=Decimal("1"), reduce_only=False, leg_index=0,
    )
    submit = next(s for s in journal.submits() if s.client_order_id == order_id)
    stale = _payload(_ms(submit.recorded_at) - 85_000, orderId=4224402074)

    class _Client:
        def query_order(self, *_a):
            return dict(stale)

    rt = _runtime(journal, order_id)
    with pytest.raises(StaleOrderIdentity):
        _sync_venue_order(_Client(), rt, now=1.0, audit=AuditLog(tmp_path / "a.jsonl"), touch=None)
    assert rt.filled_total == Decimal("0")
    assert journal.fills_after(-1) == ()


def test_recovery_books_no_phantom_fill(tmp_path) -> None:
    journal = OrderJournal(tmp_path / "j.jsonl")
    attempt = journal.begin_attempt(
        decision_time=_SUBMIT,
        run_id="run1",
        mode="LIVE",
        pre_trade_equity=Decimal("1000"),
        sizing_anchor="equity",
        decision_marks={},
        started_at=_SUBMIT,
    )
    order_id = "mh20261014-ABCDEFGHIJ-0-0-1"
    journal.record_submit(
        order_id, "AAAUSDT", journal.next_submit_seq(), attempt_seq=attempt.attempt_seq,
        side="BUY", quantity=Decimal("1"), reduce_only=False, leg_index=0,
    )
    submit = next(s for s in journal.submits() if s.client_order_id == order_id)
    stale = _payload(_ms(submit.recorded_at) - 85_000, orderId=4224402074)

    class _Client:
        def query_order(self, *_a):
            return dict(stale)

    with pytest.raises(StaleOrderIdentity):
        recover_unresolved_orders(
            _Client(),
            journal,
            AuditLog(tmp_path / "a.jsonl"),
            now=submit.recorded_at + pd.Timedelta(hours=1),
            lookback=pd.Timedelta(hours=72),
            taker_fee_bps=4.5,
            order_identity_tolerance_ms=_TOL,
        )
    assert journal.fills_after(-1) == ()


def test_legacy_and_journal_less_runs_skip(tmp_path) -> None:
    rt = _runtime(None, "oid")
    _sync_venue_order(
        object(), rt, now=1.0, audit=AuditLog(tmp_path / "a.jsonl"), touch=None,
        payload={"status": "NEW", "executedQty": "0", "time": 1},
    )
    assert rt.filled_total == Decimal("0")


def test_executor_skips_guard_without_matching_submit(tmp_path) -> None:
    from types import SimpleNamespace

    journal = OrderJournal(tmp_path / "j.jsonl")
    rt = _runtime(journal, "unknown-id")
    status = _sync_venue_order(
        object(), rt, now=1.0, audit=AuditLog(tmp_path / "a.jsonl"), touch=None,
        payload={"status": "NEW", "executedQty": "0"},
    )
    assert status == "NEW"

    for recorded in (pd.Timestamp.min.tz_localize("UTC"),):
        stub = SimpleNamespace(
            submits=lambda recorded=recorded: (
                SimpleNamespace(client_order_id="oid", recorded_at=recorded),
            )
        )
        rt = _runtime(None, "oid")
        rt.journal = stub
        status = _sync_venue_order(
            object(), rt, now=1.0, audit=AuditLog(tmp_path / "b.jsonl"), touch=None,
            payload={"status": "NEW", "executedQty": "0"},
        )
        assert status == "NEW"


def test_recovery_skips_guard_without_matching_or_legacy_submit(tmp_path) -> None:
    from types import SimpleNamespace

    from src.live.recovery import _book_recovered_delta, _guard_order_identity

    journal = OrderJournal(tmp_path / "j.jsonl")
    audit = AuditLog(tmp_path / "a.jsonl")
    assert (
        _book_recovered_delta(
            journal, audit, order_id="orphan-id", symbol="AAAUSDT",
            attempt_seq=None, leg_index=0, payload={"executedQty": "0"},
            taker_fee_bps=4.5,
        )
        is None
    )
    stub = SimpleNamespace(
        submits=lambda: (
            SimpleNamespace(
                client_order_id="legacy-id",
                recorded_at=pd.Timestamp.min.tz_localize("UTC"),
            ),
        )
    )
    _guard_order_identity(stub, {"executedQty": "0"}, "legacy-id", None)


def test_settings_tolerance_default_and_bounds() -> None:
    import pytest
    from pydantic import ValidationError

    from src.live.settings import LiveSettings

    settings = LiveSettings()
    assert settings.order_identity_tolerance_ms == settings.recv_window_ms + 30_000
    assert LiveSettings(recv_window_ms=12_000).order_identity_tolerance_ms == 42_000
    assert LiveSettings(order_identity_tolerance_ms=100).order_identity_tolerance_ms == 100
    with pytest.raises(ValidationError):
        LiveSettings(order_identity_tolerance_ms=0)


def _submitted_runtime(tmp_path, monkeypatch):
    from src.live import order_journal

    monkeypatch.setattr(order_journal, "_utc_now", lambda: _SUBMIT)
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = journal.begin_attempt(
        decision_time=_SUBMIT, run_id="run1", mode="LIVE", pre_trade_equity=Decimal("1000"),
        sizing_anchor="equity", decision_marks={}, started_at=_SUBMIT,
    )
    journal.record_submit(
        "mh-oid", "AAAUSDT", 0, attempt_seq=attempt.attempt_seq,
        side="BUY", quantity=Decimal("1"), reduce_only=False, leg_index=0,
    )
    return _runtime(journal, "mh-oid")


@pytest.mark.parametrize("entry", ["query", "cancel", "recovery_open", "orphan"])
def test_all_venue_adoption_paths_reject_old_identity(tmp_path, monkeypatch, entry):
    from src.live.executor import _cancel_and_settle, cancel_orphan_orders

    rt = _submitted_runtime(tmp_path, monkeypatch)
    journal = rt.journal
    payload = _payload(_ms(_SUBMIT) - 85_000)

    class Client:
        def query_order(self, *_args):
            return payload

        def cancel_order(self, *_args):
            return {}

        def open_orders(self):
            return [{"symbol": "AAAUSDT", "clientOrderId": "mh-oid"}]

    client = Client()
    audit = AuditLog(tmp_path / "audit.jsonl")
    def exercise():
        if entry == "query":
            _sync_venue_order(client, rt, now=1.0, audit=audit, touch=None)
        elif entry == "cancel":
            _cancel_and_settle(client, rt, audit, "test", now=1.0)
        elif entry == "recovery_open":
            payload["status"] = "NEW"
            recover_unresolved_orders(client, journal, audit, now=_SUBMIT, lookback=pd.Timedelta(hours=1), taker_fee_bps=5)
        else:
            cancel_orphan_orders(client, "prefix", audit, journal=journal, now=_SUBMIT, taker_fee_bps=5)
    with pytest.raises(StaleOrderIdentity):
        exercise()
    assert journal.fills_after(-1) == ()
    assert journal.terminal("mh-oid") is None
    assert rt.reported_executed == 0


def test_clock_margin_is_applied_without_accumulating_on_polls(tmp_path, monkeypatch):
    rt = _submitted_runtime(tmp_path, monkeypatch)
    rt.order_identity_tolerance_ms = 1000
    payload = _payload(_ms(_SUBMIT) - 2000, executedQty="0", status="NEW")

    class Client:
        clock_offset_margin_ms = 1000

        def query_order(self, *_args):
            return payload

    client = Client()
    audit = AuditLog(tmp_path / "audit.jsonl")
    for _ in range(2):
        _sync_venue_order(client, rt, now=1.0, audit=audit, touch=None)
    payload["time"] -= 1
    with pytest.raises(StaleOrderIdentity):
        _sync_venue_order(client, rt, now=1.0, audit=audit, touch=None)
    assert rt.order_identity_tolerance_ms == 1000


def test_error_messages_exclude_arbitrary_payload_fields():
    marker = "signature=redaction-fixture"
    with pytest.raises(DataIntegrityError) as invalid:
        assert_order_identity({"time": marker}, client_order_id="oid", submitted_at=_SUBMIT, tolerance_ms=1)
    assert marker not in str(invalid.value)
    with pytest.raises(StaleOrderIdentity) as stale:
        assert_order_identity(_payload(1, orderId=marker), client_order_id="oid", submitted_at=_SUBMIT, tolerance_ms=1)
    assert marker not in str(stale.value)


@pytest.mark.parametrize(("stamp", "tolerance"), [(pd.NaT, 1), (pd.Timestamp("2026-10-14"), 1), (_SUBMIT, 0)])
def test_invalid_guard_context_fails_closed(stamp, tolerance):
    with pytest.raises(DataIntegrityError, match="context"):
        assert_order_identity(_payload(_ms(_SUBMIT)), client_order_id="oid", submitted_at=stamp, tolerance_ms=tolerance)
