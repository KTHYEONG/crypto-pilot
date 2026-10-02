from __future__ import annotations

from decimal import Decimal

import pytest

from src.live.audit import AuditLog
from src.common.errors import DataIntegrityError
from src.live.executor import (
    FeeSchedule,
    PassiveExecutionPolicy,
    _IntentRuntime,
    _apply_fill,
    _cancel_and_settle,
    _finalize,
    _parse_avg_price,
    _poll_active,
    _poll_or_post,
    _sync_venue_order,
)
from src.live.filters import SymbolFilters
from src.live.planner import OrderIntent
from src.live.rest import PaperResponse


def _runtime(*, quantity: str = "1", fee_schedule: FeeSchedule | None = None) -> _IntentRuntime:
    intent = OrderIntent(
        symbol="AAAUSDT",
        side="BUY",
        quantity=Decimal(quantity),
        reduce_only=False,
        target_qty=Decimal(quantity),
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
    return _IntentRuntime(
        intent=intent,
        filters=filters,
        fee_schedule=fee_schedule or FeeSchedule(2.0, 5.0),
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({}, None),
        ({"avgPrice": ""}, None),
        ({"avgPrice": "0"}, None),
        ({"avgPrice": "0.00"}, None),
        ({"avgPrice": "abc"}, None),
        ({"avgPrice": "NaN"}, None),
        ({"avgPrice": "99.50"}, Decimal("99.50")),
        ({"avg_price": "99.50"}, Decimal("99.50")),
    ],
)
def test_avg_price_parser(payload, expected) -> None:
    assert _parse_avg_price(payload) == expected


def test_cancel_settlement_uses_runtime_maker_fee(tmp_path) -> None:
    class Journal:
        def __init__(self):
            self.fills = []

        def record_fill(self, **kwargs):
            self.fills.append(kwargs)

        def record_terminal(self, *_args):
            pass

    class Client:
        def cancel_order(self, *_args):
            return {}

        def query_order(self, *_args):
            return {"status": "CANCELED", "executedQty": "0.4", "avgPrice": "99.50"}

    rt = _runtime(fee_schedule=FeeSchedule(0.9, 3.6))
    journal = Journal()
    rt.journal = journal
    rt.journal_enabled = True
    rt.journal_submitted.add("order-1")
    rt.active_id = "order-1"
    rt.active_price = Decimal("100")
    rt.active_post_qty = Decimal("1")
    rt.phase = "passive"

    _cancel_and_settle(Client(), rt, AuditLog(tmp_path / "cancel.jsonl"), "test", now=2.0)

    assert rt.fills[0][2] == 0.9
    assert journal.fills[0]["fee_bps"] == 0.9
    assert rt.filled_total == Decimal("0.4")
    assert rt.active_id is None


def test_window_end_settlement_uses_runtime_taker_fee(tmp_path) -> None:
    class Client:
        def cancel_order(self, *_args):
            return {}

        def query_order(self, *_args):
            return {"status": "CANCELED", "executedQty": "0.25", "avgPrice": "101"}

    rt = _runtime(fee_schedule=FeeSchedule(0.9, 3.6))
    rt.active_id = "order-2"
    rt.active_price = Decimal("100")
    rt.active_post_qty = Decimal("1")
    rt.phase = "ioc"
    _finalize(Client(), [rt], AuditLog(tmp_path / "finalize.jsonl"), lambda: 10.0, lambda _s: None)
    assert rt.fills[0][2:5] == (3.6, "timeout_taker", "taker")


def test_stale_venue_reads_do_not_double_count(tmp_path) -> None:
    class Client:
        quantities = iter(("3", "2", "3"))

        def query_order(self, *_args):
            return {"status": "PARTIALLY_FILLED", "executedQty": next(self.quantities), "avgPrice": "100"}

    rt = _runtime(fee_schedule=FeeSchedule(0.9, 3.6))
    rt.active_id = "order-3"
    rt.active_price = Decimal("100")
    rt.active_post_qty = Decimal("4")
    client = Client()
    for tick in range(3):
        _sync_venue_order(client, rt, now=float(tick), audit=AuditLog(tmp_path / f"stale-{tick}.jsonl"), touch=None)
    assert rt.filled_total == Decimal("3")
    assert rt.reported_executed == Decimal("3")
    assert len(rt.fills) == 1


def test_journal_failure_leaves_fill_runtime_untouched(tmp_path) -> None:
    class BrokenJournal:
        def record_fill(self, **_kwargs):
            raise OSError("disk full")

    rt = _runtime()
    rt.journal = BrokenJournal()
    rt.journal_enabled = True
    rt.active_id = "order-4"
    before = (rt.filled_total, rt.fill_notional, list(rt.fills), rt.reported_executed)
    with pytest.raises(OSError, match="disk full"):
        _apply_fill(
            rt,
            quantity=Decimal("0.2"),
            price=Decimal("100"),
            liquidity="maker",
            client_order_id="order-4",
            order_cumulative_qty=Decimal("0.2"),
            simulated=False,
            now=1.0,
            audit=AuditLog(tmp_path / "broken.jsonl"),
        )
    assert (rt.filled_total, rt.fill_notional, rt.fills, rt.reported_executed) == before


@pytest.mark.parametrize(
    ("quantity", "price"),
    [(Decimal("0"), Decimal("100")), (Decimal("1"), Decimal("0"))],
)
def test_fill_application_rejects_nonpositive_quantity_or_price(tmp_path, quantity, price) -> None:
    rt = _runtime()
    with pytest.raises(ValueError, match="must be positive"):
        _apply_fill(
            rt,
            quantity=quantity,
            price=price,
            liquidity="maker",
            client_order_id="bad-fill",
            order_cumulative_qty=quantity,
            simulated=False,
            now=1.0,
            audit=AuditLog(tmp_path / "invalid-fill.jsonl"),
        )


def test_venue_execution_cannot_exceed_submitted_order_quantity(tmp_path) -> None:
    class Client:
        def query_order(self, *_args):
            return {"status": "FILLED", "executedQty": "1.1", "avgPrice": "100"}

    rt = _runtime()
    rt.active_id = "overfilled-order"
    rt.active_price = Decimal("100")
    rt.active_post_qty = Decimal("1")
    with pytest.raises(DataIntegrityError, match="exceeds submitted quantity"):
        _sync_venue_order(Client(), rt, now=1.0, audit=AuditLog(tmp_path / "overfill.jsonl"), touch=None)


def test_paper_resting_fill_is_capped_to_active_order_quantity(tmp_path) -> None:
    rt = _runtime(quantity="20")
    rt.active_id = "paper-1"
    rt.active_price = Decimal("100")
    rt.active_post_qty = Decimal("5")
    rt.paper_active = True
    rt.phase = "passive"
    policy = PassiveExecutionPolicy(
        passive_deadline_s=10,
        window_deadline_s=20,
        passive_pricing="anchored",
    )
    _poll_active(
        object(), rt, (Decimal("99"), Decimal("99.9")), 1.0, policy, AuditLog(tmp_path / "paper.jsonl")
    )
    assert rt.fills[0][0] == Decimal("5")
    assert rt.filled_total == Decimal("5")
    assert rt.filled_total < rt.intent.quantity
    assert rt.active_id is None


def test_paper_slice_progression_conserves_intent_quantity(tmp_path, monkeypatch) -> None:
    import src.live.executor as executor

    rt = _runtime(quantity="20")
    policy = PassiveExecutionPolicy(
        passive_deadline_s=10,
        window_deadline_s=20,
        max_slices=4,
        passive_pricing="anchored",
    )
    submitted = []

    class Client:
        def new_order(self, params):
            submitted.append(Decimal(params["quantity"]))
            return PaperResponse.suppressed("POST", "/order", "0" * 12)

    monkeypatch.setattr(executor, "_simulate_paper_fill", lambda _rt, _touch, _tif, _price, qty: qty)
    client = Client()
    audit = AuditLog(tmp_path / "slices.jsonl")
    for tick in range(4):
        _poll_or_post(client, rt, (Decimal("99"), Decimal("99.8")), float(tick + 1), policy, audit)
    assert submitted == [Decimal("5.000")] * 4
    assert rt.filled_total == rt.intent.quantity
    assert all(fill[0] <= qty for fill, qty in zip((f for f in rt.fills), submitted, strict=True))
    assert rt.terminal_status == "FILLED"


def test_fully_filled_paper_order_is_released_before_next_tick(tmp_path, monkeypatch) -> None:
    import json

    import src.live.executor as executor
    import pandas as pd
    from src.live.order_journal import OrderJournal

    rt = _runtime(quantity="20")
    journal = OrderJournal(tmp_path / "paper-journal.jsonl")
    started = pd.Timestamp("2026-10-02T00:00:00Z")
    attempt = journal.begin_attempt(
        decision_time=started,
        run_id="test",
        mode="PAPER",
        pre_trade_equity=Decimal("1000"),
        sizing_anchor="equity",
        decision_marks={},
        started_at=started,
    )
    rt.journal = journal
    rt.attempt_seq = attempt.attempt_seq
    rt.journal_enabled = True
    policy = PassiveExecutionPolicy(
        passive_deadline_s=10,
        window_deadline_s=20,
        max_slices=4,
        passive_pricing="anchored",
    )

    class Client:
        def new_order(self, _params):
            return PaperResponse.suppressed("POST", "/order", "0" * 12)

    monkeypatch.setattr(executor, "_simulate_paper_fill", lambda _rt, _touch, _tif, _price, qty: qty)
    audit_path = tmp_path / "filled-order-audit.jsonl"
    _poll_or_post(Client(), rt, (Decimal("99"), Decimal("99.8")), 1.0, policy, AuditLog(audit_path))
    assert rt.filled_total == Decimal("5.000")
    assert rt.active_id is None
    assert rt.paper_active is False
    journal_rows = [json.loads(line) for line in (tmp_path / "paper-journal.jsonl").read_text().splitlines()]
    order_id = next(row["client_order_id"] for row in journal_rows if row["event"] == "submit")
    assert any(row.get("event") == "terminal" and row.get("client_order_id") == order_id and row.get("status") == "FILLED" for row in journal_rows)
    audit_rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert not any(row.get("event") == "order_cancelled" and row.get("client_order_id") == order_id for row in audit_rows)


def test_anchored_repeg_ignores_exhausted_post_only_chase_budget(tmp_path) -> None:
    class Client:
        def __init__(self):
            self.cancelled = []
            self.orders = []

        def cancel_order(self, symbol, order_id):
            self.cancelled.append(order_id)

        def query_order(self, *_args):
            return {"status": "CANCELED", "executedQty": "0"}

        def new_order(self, params):
            self.orders.append(params)
            return {"orderId": len(self.orders)}

    from src.live.executor import strict_passive_repeg_execution_policy

    rt = _runtime()
    rt.active_id = "repeg-old"
    rt.active_price = Decimal("98.10")
    rt.active_post_qty = Decimal("1")
    rt.passive_started_at = 1.0
    rt.last_repeg_at = 1.0
    policy = strict_passive_repeg_execution_policy(FeeSchedule(0.9, 3.6), 3.0, 10, repeg_interval_s=1.0)
    rt.chases = policy.max_chases
    client = Client()
    _poll_or_post(client, rt, (Decimal("98.00"), Decimal("98.20")), 3.0, policy, AuditLog(tmp_path / "repeg.jsonl"))
    assert client.cancelled == ["repeg-old"]
    assert client.orders
    assert Decimal(client.orders[-1]["price"]) == Decimal("98.00")


def test_touch_chase_budget_holds_active_order(tmp_path) -> None:
    class Client:
        def cancel_order(self, *_args):
            raise AssertionError("exhausted touch chase must hold")

        def query_order(self, *_args):
            return {"status": "NEW", "executedQty": "0"}

        def new_order(self, _params):
            raise AssertionError("held order must not be reposted")

    rt = _runtime()
    rt.active_id = "touch-held"
    rt.active_price = Decimal("99")
    rt.active_post_qty = Decimal("1")
    rt.chases = 3
    rt.passive_started_at = 1.0
    rt.last_repeg_at = 1.0
    policy = PassiveExecutionPolicy(
        passive_deadline_s=50,
        window_deadline_s=600,
        max_chases=3,
        passive_pricing="touch_chase",
    )
    _poll_or_post(
        Client(), rt, (Decimal("99.50"), Decimal("99.70")), 2.0, policy, AuditLog(tmp_path / "touch-hold.jsonl")
    )
    assert rt.active_id == "touch-held"


def test_anchored_repeg_next_slice_keeps_decision_anchor_before_first_repeg(tmp_path, monkeypatch) -> None:
    import src.live.executor as executor
    from src.live.executor import strict_passive_repeg_execution_policy

    rt = _runtime(quantity="20")
    policy = strict_passive_repeg_execution_policy(FeeSchedule(0.9, 3.6), 3.0, 10, repeg_interval_s=60.0)
    prices = []

    class Client:
        def new_order(self, params):
            prices.append(Decimal(params["price"]))
            return PaperResponse.suppressed("POST", "/order", "0" * 12)

    monkeypatch.setattr(executor, "_simulate_paper_fill", lambda _rt, _touch, _tif, _price, qty: qty)
    audit = AuditLog(tmp_path / "anchor-slices.jsonl")
    _poll_or_post(Client(), rt, (Decimal("99.0"), Decimal("99.8")), 1.0, policy, audit)
    _poll_or_post(Client(), rt, (Decimal("99.2"), Decimal("100.4")), 2.0, policy, audit)
    # Slice two is quoted at the decision anchor (100.0), not re-pegged to the moved bid (99.2).
    assert prices == [Decimal("99.7"), Decimal("100.0")]
    assert rt.passive_quote == Decimal("0")


def test_anchored_repeg_repost_uses_last_repeg_quote_not_current_touch(tmp_path) -> None:
    from src.live.executor import strict_passive_repeg_execution_policy

    class Client:
        def __init__(self):
            self.orders = []

        def new_order(self, params):
            self.orders.append(Decimal(params["price"]))
            return {"orderId": len(self.orders)}

    policy = strict_passive_repeg_execution_policy(FeeSchedule(0.9, 3.6), 3.0, 10, repeg_interval_s=60.0)
    rt = _runtime(quantity="20")
    rt.passive_started_at = 1.0
    rt.last_repeg_at = 1.0
    rt.passive_quote = Decimal("98.50")
    client = Client()
    _poll_or_post(client, rt, (Decimal("98.00"), Decimal("99.00")), 2.0, policy, AuditLog(tmp_path / "q1.jsonl"))
    rt.active_id = None
    _poll_or_post(client, rt, (Decimal("98.00"), Decimal("98.40")), 3.0, policy, AuditLog(tmp_path / "q2.jsonl"))
    # Off-cadence reposts keep the re-peg quote and only yield one tick inside the opposite touch.
    assert client.orders == [Decimal("98.50"), Decimal("98.30")]
    assert rt.passive_quote == Decimal("98.50")


def test_anchored_repeg_records_repeg_price_as_quote_level(tmp_path) -> None:
    from src.live.executor import strict_passive_repeg_execution_policy

    class Client:
        def cancel_order(self, *_args):
            return None

        def query_order(self, *_args):
            return {"status": "CANCELED", "executedQty": "0"}

        def new_order(self, params):
            return {"orderId": 1}

    policy = strict_passive_repeg_execution_policy(FeeSchedule(0.9, 3.6), 3.0, 10, repeg_interval_s=1.0)
    rt = _runtime()
    rt.active_id = "old"
    rt.active_price = Decimal("99.70")
    rt.active_post_qty = Decimal("1")
    rt.passive_started_at = 1.0
    rt.last_repeg_at = 1.0
    _poll_or_post(Client(), rt, (Decimal("99.10"), Decimal("99.30")), 3.0, policy, AuditLog(tmp_path / "rq.jsonl"))
    assert rt.passive_quote == Decimal("99.10")
    assert rt.active_price == Decimal("99.10")
