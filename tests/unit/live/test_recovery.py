"""Spec 03 restart recovery: journal-submitted orders queried on restart (src/live/recovery.py)."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.live.audit import AuditLog
from src.live.errors import VenueError
from src.live.order_journal import OrderJournal
from src.live.recovery import RecoveryReport, recover_unresolved_orders

_NOW = pd.Timestamp("2026-09-14T00:00:00Z")
_LOOKBACK = pd.Timedelta(hours=72)
_TAKER_BPS = 4.5
_UPDATE_MS = int((_NOW - pd.Timedelta(minutes=7)).value // 1_000_000)


def _journal_with_submit(tmp_path, name="journal.jsonl", **overrides):
    """Fresh journal with one v2 BUY submit (qty 1.0, observed 0.4) and its attempt."""
    journal = OrderJournal(tmp_path / name)
    attempt = journal.begin_attempt(
        decision_time=_NOW,
        run_id="run1",
        mode="LIVE",
        pre_trade_equity=Decimal("1000"),
        sizing_anchor="equity",
        decision_marks={},
        started_at=_NOW,
    )
    order_id = overrides.get("order_id", "mh20260914-ABCDEFGHIJ-0-0-0")
    symbol = overrides.get("symbol", "AAAUSDT")
    journal.record_submit(
        order_id, symbol, journal.next_submit_seq(), attempt_seq=attempt.attempt_seq,
        side="BUY", quantity=Decimal("1.0"), reduce_only=False, leg_index=0,
    )
    journal.record_fill(
        kind="execution", attempt_seq=attempt.attempt_seq, symbol=symbol, side="BUY",
        quantity=Decimal("0.4"), price=Decimal("99"), fee_bps=2.0, liquidity="maker",
        reason="maker_fill", filled_at=_NOW, client_order_id=order_id, leg_index=0,
        cumulative_executed_qty=Decimal("0.4"), simulated=False,
    )
    return journal, attempt, order_id


class _VenueClient:
    """Query answers keyed by client order id; counts every call."""

    def __init__(self, answers):
        self.answers = dict(answers)
        self.calls: list = []

    def query_order(self, symbol, orig_client_order_id):
        self.calls.append((symbol, orig_client_order_id))
        answer = self.answers[orig_client_order_id]
        if isinstance(answer, Exception):
            raise answer
        return dict(answer)


def _venue_error(code):
    return VenueError("venue", code=code, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)


def test_recovered_fill_for_closed_order(tmp_path) -> None:
    """FILLED with executedQty 1.0 over observed 0.4 journals one recovered fill of 0.6 @ 100 and terminals."""
    import json

    journal, attempt, order_id = _journal_with_submit(tmp_path)
    audit_path = tmp_path / "audit.jsonl"
    audit = AuditLog(audit_path)
    client = _VenueClient({order_id: {"status": "FILLED", "side": "BUY", "executedQty": "1.0", "avgPrice": "100", "updateTime": _UPDATE_MS}})

    report = recover_unresolved_orders(client, journal, audit, now=_NOW, lookback=_LOOKBACK, taker_fee_bps=_TAKER_BPS)

    assert isinstance(report, RecoveryReport)
    assert len(report.recovered) == 1
    fill = report.recovered[0]
    assert fill.kind == "recovered"
    assert fill.quantity == Decimal("0.6")
    assert fill.price == Decimal("100")
    assert (fill.side, fill.liquidity, fill.reason) == ("BUY", "taker", "recovered_fill")
    assert fill.fee_bps == _TAKER_BPS
    assert fill.filled_at == pd.Timestamp(_UPDATE_MS, unit="ms", tz="UTC")
    assert fill.simulated is False
    assert fill.cumulative_executed_qty == Decimal("1.0")
    assert fill.attempt_seq == attempt.attempt_seq
    assert fill.leg_index == 0
    assert report.resolved_ids == (order_id,)
    assert report.unresolved_ids == ()
    assert journal.observed_qty(order_id) == Decimal("1.0")
    assert journal.unresolved_submits(since=_NOW - _LOOKBACK) == ()
    events = [json.loads(line)["event"] for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert "order_recovered" in events
    assert "order_recovery_resolved" in events


def test_missing_order_resolved_without_fill(tmp_path) -> None:
    """Venue -2013 journals terminal NOT_FOUND and no fill."""
    journal, _, order_id = _journal_with_submit(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")
    client = _VenueClient({order_id: _venue_error(-2013)})

    report = recover_unresolved_orders(client, journal, audit, now=_NOW, lookback=_LOOKBACK, taker_fee_bps=_TAKER_BPS)

    assert report.recovered == ()
    assert report.resolved_ids == (order_id,)
    assert report.unresolved_ids == ()
    assert journal.fills_after(0) == ()
    assert journal.unresolved_submits(since=_NOW - _LOOKBACK) == ()


def test_open_orders_left_for_sweep(tmp_path) -> None:
    """PARTIALLY_FILLED (and NEW) submits stay unresolved with no fill and no terminal."""
    journal, _, order_id = _journal_with_submit(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")
    client = _VenueClient({order_id: {"status": "PARTIALLY_FILLED", "side": "BUY", "executedQty": "0.4", "avgPrice": "99"}})

    report = recover_unresolved_orders(client, journal, audit, now=_NOW, lookback=_LOOKBACK, taker_fee_bps=_TAKER_BPS)

    assert report.recovered == ()
    assert report.resolved_ids == ()
    assert report.unresolved_ids == (order_id,)
    assert [s.client_order_id for s in journal.unresolved_submits(since=_NOW - _LOOKBACK)] == [order_id]


def test_submits_outside_lookback_not_queried(tmp_path) -> None:
    """A submit recorded 100 h ago is outside the 72 h window: zero venue calls."""
    journal, _, order_id = _journal_with_submit(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")
    client = _VenueClient({})
    # The submit was recorded at the real wall clock; look 100 h past it so the 72 h window excludes it.
    late_now = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=100)

    report = recover_unresolved_orders(client, journal, audit, now=late_now, lookback=_LOOKBACK, taker_fee_bps=_TAKER_BPS)

    assert client.calls == []
    assert report == RecoveryReport(recovered=(), resolved_ids=(), unresolved_ids=())


def test_suppressed_client_makes_no_calls(tmp_path) -> None:
    """NullOrderClient (and mode-suppressed clients) return an empty report with zero calls."""
    journal, _, order_id = _journal_with_submit(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")

    class NullOrderClient:
        def __init__(self):
            self.calls = 0

        def query_order(self, symbol, oid):
            self.calls += 1
            return {}

    null_client = NullOrderClient()
    report = recover_unresolved_orders(null_client, journal, audit, now=_NOW, lookback=_LOOKBACK, taker_fee_bps=_TAKER_BPS)
    assert report == RecoveryReport(recovered=(), resolved_ids=(), unresolved_ids=())
    assert null_client.calls == 0

    from src.live.settings import ExecutionMode

    class _SuppressedClient:
        mode = ExecutionMode.PAPER

        def query_order(self, symbol, oid):  # pragma: no cover - must never be called
            raise AssertionError("suppressed client must not be queried")

    suppressed_report = recover_unresolved_orders(
        _SuppressedClient(), journal, audit, now=_NOW, lookback=_LOOKBACK, taker_fee_bps=_TAKER_BPS
    )
    assert suppressed_report == RecoveryReport(recovered=(), resolved_ids=(), unresolved_ids=())


def test_recovery_non_benign_error_propagates_without_journaling(tmp_path) -> None:
    """A non-benign venue error propagates and journals nothing for the failing id."""
    journal, _, order_id = _journal_with_submit(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")
    client = _VenueClient({order_id: _venue_error(-1022)})

    with pytest.raises(VenueError):
        recover_unresolved_orders(client, journal, audit, now=_NOW, lookback=_LOOKBACK, taker_fee_bps=_TAKER_BPS)

    assert journal.fills_after(0) == ()
    assert [s.client_order_id for s in journal.unresolved_submits(since=_NOW - _LOOKBACK)] == [order_id]


def test_recovery_fill_without_price_fails_closed(tmp_path) -> None:
    """Executed quantity with a non-positive avgPrice raises DataIntegrityError and journals nothing."""
    journal, _, order_id = _journal_with_submit(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")
    client = _VenueClient({order_id: {"status": "FILLED", "side": "BUY", "executedQty": "1.0", "avgPrice": "0"}})

    with pytest.raises(DataIntegrityError):
        recover_unresolved_orders(client, journal, audit, now=_NOW, lookback=_LOOKBACK, taker_fee_bps=_TAKER_BPS)

    assert journal.fills_after(0) == ()
    assert [s.client_order_id for s in journal.unresolved_submits(since=_NOW - _LOOKBACK)] == [order_id]


@pytest.mark.parametrize(
    "answer",
    [
        {"status": "FILLED", "side": "BUY", "avgPrice": "100"},
        {"status": "FILLED", "side": "BUY", "executedQty": "abc", "avgPrice": "100"},
        {"status": "FILLED", "side": "BUY", "executedQty": "NaN", "avgPrice": "100"},
        {"status": "FILLED", "executedQty": "1.0", "avgPrice": "100"},
        {"status": "FILLED", "side": "BUY", "executedQty": "1.0"},
        {"status": "FILLED", "side": "BUY", "executedQty": "1.0", "avgPrice": "abc"},
    ],
    ids=["no_executed", "bad_executed", "nan_executed", "no_side", "no_price", "bad_price"],
)
def test_recovery_malformed_venue_answer_fails_closed(tmp_path, answer) -> None:
    """A closed order whose query lacks or garbles executedQty/side/avgPrice raises and journals nothing."""
    journal, _, order_id = _journal_with_submit(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")
    client = _VenueClient({order_id: answer})

    with pytest.raises(DataIntegrityError):
        recover_unresolved_orders(client, journal, audit, now=_NOW, lookback=_LOOKBACK, taker_fee_bps=_TAKER_BPS)

    assert journal.fills_after(0) == ()
    assert [s.client_order_id for s in journal.unresolved_submits(since=_NOW - _LOOKBACK)] == [order_id]


@pytest.mark.parametrize(
    "answer",
    [{"executedQty": "1.0"}, {"status": None, "executedQty": "1.0"}, {"status": "", "executedQty": "1.0"}],
    ids=["missing_status", "none_status", "empty_status"],
)
def test_recovery_status_less_answer_fails_closed(tmp_path, answer) -> None:
    """A query answer without a status fails closed (I-RECOVERY-STATUS): no fill, no terminal."""
    journal, _, order_id = _journal_with_submit(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")
    client = _VenueClient({order_id: dict(answer)})

    with pytest.raises(DataIntegrityError):
        recover_unresolved_orders(client, journal, audit, now=_NOW, lookback=_LOOKBACK, taker_fee_bps=_TAKER_BPS)

    assert journal.fills_after(0) == ()
    assert [s.client_order_id for s in journal.unresolved_submits(since=_NOW - _LOOKBACK)] == [order_id]


def test_recovery_closed_order_without_new_execution_terminals_only(tmp_path) -> None:
    """CANCELED with executedQty equal to the observed 0.4 is terminalled without a recovered fill."""
    journal, _, order_id = _journal_with_submit(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")
    client = _VenueClient({order_id: {"status": "CANCELED", "side": "BUY", "executedQty": "0.4", "avgPrice": "99"}})

    report = recover_unresolved_orders(client, journal, audit, now=_NOW, lookback=_LOOKBACK, taker_fee_bps=_TAKER_BPS)

    assert report.recovered == ()
    assert report.resolved_ids == (order_id,)
    assert journal.observed_qty(order_id) == Decimal("0.4")
    assert journal.unresolved_submits(since=_NOW - _LOOKBACK) == ()


@pytest.mark.parametrize("update_time", [None, "abc", 0])
def test_recovery_fill_without_venue_update_time_fails_closed(tmp_path, update_time) -> None:
    """A recovered fill is never stamped with the wall clock: a missing/invalid updateTime fails closed."""
    journal, _attempt, order_id = _journal_with_submit(tmp_path)
    answer = {"status": "FILLED", "side": "BUY", "executedQty": "1.0", "avgPrice": "100"}
    if update_time is not None:
        answer["updateTime"] = update_time
    client = _VenueClient({order_id: answer})
    with pytest.raises(DataIntegrityError, match="updateTime"):
        recover_unresolved_orders(
            client, journal, AuditLog(tmp_path / "a.jsonl"), now=_NOW, lookback=_LOOKBACK, taker_fee_bps=_TAKER_BPS
        )
    assert journal.observed_qty(order_id) == Decimal("0.4")


# ---- Spec 02a: RecoveryReport.unresolved and settle_unresolved_orders ----

def _settle_client(*, cancel_exc=None, answers=None):
    """Cancel/query client counting venue calls; answers map order id to payload or exception."""

    class _C:
        def __init__(self):
            self.cancels: list = []
            self.queries: list = []

        def cancel_order(self, symbol, oid):
            self.cancels.append((symbol, oid))
            if cancel_exc is not None:
                raise cancel_exc
            return {}

        def query_order(self, symbol, oid):
            self.queries.append((symbol, oid))
            answer = (answers or {})[oid]
            if isinstance(answer, Exception):
                raise answer
            return dict(answer)

    return _C()


def test_recovery_unresolved_carries_symbol_and_attempt(tmp_path) -> None:
    """A PARTIALLY_FILLED answer returns unresolved with symbol and attempt attribution."""
    from src.live.executor import UnresolvedOrder

    journal, attempt, order_id = _journal_with_submit(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")
    client = _VenueClient({order_id: {"status": "PARTIALLY_FILLED", "side": "BUY", "executedQty": "0.4", "avgPrice": "99"}})

    report = recover_unresolved_orders(client, journal, audit, now=_NOW, lookback=_LOOKBACK, taker_fee_bps=_TAKER_BPS)

    assert report.unresolved == (UnresolvedOrder(order_id, "AAAUSDT", attempt.attempt_seq, 0),)
    assert report.unresolved_ids == (order_id,)


def test_settle_confirmed_cancel_books_delta_and_terminals(tmp_path) -> None:
    """A confirmed CANCELED close books the executed delta as recovered and terminals the id."""
    from src.live.executor import UnresolvedOrder
    from src.live.recovery import settle_unresolved_orders

    journal, attempt, order_id = _journal_with_submit(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")
    order = UnresolvedOrder(order_id, "AAAUSDT", attempt.attempt_seq, 0)
    client = _settle_client(answers={order_id: {
        "status": "CANCELED", "side": "BUY", "executedQty": "0.7", "avgPrice": "100", "updateTime": _UPDATE_MS}})

    settlement = settle_unresolved_orders(client, journal, audit, (order,), taker_fee_bps=_TAKER_BPS)

    assert len(settlement.recovered) == 1
    fill = settlement.recovered[0]
    assert (fill.quantity, fill.price) == (Decimal("0.3"), Decimal("100"))
    assert fill.fee_bps == _TAKER_BPS
    assert fill.filled_at == pd.Timestamp(_UPDATE_MS, unit="ms", tz="UTC")
    assert fill.attempt_seq == attempt.attempt_seq
    assert fill.cumulative_executed_qty == Decimal("0.7")
    assert settlement.resolved_ids == (order_id,)
    assert settlement.still_open == ()
    assert journal.unresolved_submits(since=_NOW - _LOOKBACK) == ()


def test_settle_2011_still_open_books_partial_delta_without_terminal(tmp_path) -> None:
    """Cancel -2011 with a PARTIALLY_FILLED query books the delta but leaves the id non-terminal."""
    from src.live.executor import UnresolvedOrder
    from src.live.recovery import settle_unresolved_orders

    journal, attempt, order_id = _journal_with_submit(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")
    order = UnresolvedOrder(order_id, "AAAUSDT", attempt.attempt_seq, 0)
    client = _settle_client(
        cancel_exc=_venue_error(-2011),
        answers={order_id: {
            "status": "PARTIALLY_FILLED", "side": "BUY", "executedQty": "0.6", "avgPrice": "100", "updateTime": _UPDATE_MS}})

    settlement = settle_unresolved_orders(client, journal, audit, (order,), taker_fee_bps=_TAKER_BPS)

    assert [f.quantity for f in settlement.recovered] == [Decimal("0.2")]
    assert settlement.resolved_ids == ()
    assert settlement.still_open == (order,)
    assert [s.client_order_id for s in journal.unresolved_submits(since=_NOW - _LOOKBACK)] == [order_id]


def test_settle_is_idempotent_on_unchanged_venue_state(tmp_path) -> None:
    """Re-running settle on the same venue state books zero additional quantity."""
    from src.live.executor import UnresolvedOrder
    from src.live.recovery import settle_unresolved_orders

    journal, attempt, order_id = _journal_with_submit(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")
    order = UnresolvedOrder(order_id, "AAAUSDT", attempt.attempt_seq, 0)
    answer = {"status": "PARTIALLY_FILLED", "side": "BUY", "executedQty": "0.6", "avgPrice": "100", "updateTime": _UPDATE_MS}

    first = settle_unresolved_orders(_settle_client(answers={order_id: answer}),
                                     journal, audit, (order,), taker_fee_bps=_TAKER_BPS)
    second = settle_unresolved_orders(_settle_client(answers={order_id: answer}),
                                      journal, audit, (order,), taker_fee_bps=_TAKER_BPS)

    assert len(first.recovered) == 1
    assert second.recovered == ()
    assert second.still_open == (order,)


def test_settle_not_found_terminals_without_fill(tmp_path) -> None:
    """A -2013 confirming query terminals NOT_FOUND with no fill."""
    import json

    from src.live.executor import ORDER_NOT_FOUND_STATUS, UnresolvedOrder
    from src.live.recovery import settle_unresolved_orders

    journal, attempt, order_id = _journal_with_submit(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")
    order = UnresolvedOrder(order_id, "AAAUSDT", attempt.attempt_seq, 0)
    client = _settle_client(answers={order_id: _venue_error(-2013)})

    settlement = settle_unresolved_orders(client, journal, audit, (order,), taker_fee_bps=_TAKER_BPS)

    assert settlement.recovered == ()
    assert settlement.resolved_ids == (order_id,)
    assert settlement.still_open == ()
    assert journal.unresolved_submits(since=_NOW - _LOOKBACK) == ()
    assert journal.observed_qty(order_id) == Decimal("0.4")
    terminals = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [t for t in terminals if t.get("event") == "terminal"][-1]["status"] == ORDER_NOT_FOUND_STATUS


def test_settle_status_less_confirmation_fails_closed(tmp_path) -> None:
    """A confirming query without status raises DataIntegrityError and journals no terminal."""
    from src.live.recovery import settle_unresolved_orders

    journal, attempt, order_id = _journal_with_submit(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")
    from src.live.executor import UnresolvedOrder

    client = _settle_client(answers={order_id: {"executedQty": "0"}})

    with pytest.raises(DataIntegrityError):
        settle_unresolved_orders(client, journal, audit,
                                 (UnresolvedOrder(order_id, "AAAUSDT", attempt.attempt_seq, 0),),
                                 taker_fee_bps=_TAKER_BPS)

    assert [s.client_order_id for s in journal.unresolved_submits(since=_NOW - _LOOKBACK)] == [order_id]


def test_settle_suppressed_client_makes_no_calls(tmp_path) -> None:
    """A mutation-suppressed client returns an empty settlement with zero venue calls."""
    from src.live.executor import UnresolvedOrder
    from src.live.recovery import settle_unresolved_orders
    from src.live.settings import ExecutionMode

    journal, attempt, order_id = _journal_with_submit(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")

    class _SuppressedClient:
        mode = ExecutionMode.PAPER

        def cancel_order(self, symbol, oid):  # pragma: no cover - must never be called
            raise AssertionError("suppressed client must not be asked to cancel")

        def query_order(self, symbol, oid):  # pragma: no cover - must never be called
            raise AssertionError("suppressed client must not be queried")

    settlement = settle_unresolved_orders(_SuppressedClient(), journal, audit,
                                          (UnresolvedOrder(order_id, "AAAUSDT", attempt.attempt_seq, 0),),
                                          taker_fee_bps=_TAKER_BPS)

    assert settlement.recovered == ()
    assert settlement.resolved_ids == ()
    assert settlement.still_open == ()


def test_settle_empty_input_makes_no_calls(tmp_path) -> None:
    """Settling no orders performs zero venue calls."""
    from src.live.recovery import settle_unresolved_orders

    journal, _, _ = _journal_with_submit(tmp_path)
    client = _settle_client(answers={})

    settlement = settle_unresolved_orders(client, journal, AuditLog(tmp_path / "a.jsonl"), (), taker_fee_bps=_TAKER_BPS)

    assert settlement.recovered == ()
    assert settlement.resolved_ids == ()
    assert settlement.still_open == ()
    assert client.cancels == []
    assert client.queries == []
