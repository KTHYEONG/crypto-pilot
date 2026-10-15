"""Production executor and recovery exercised against an HTTP-level venue."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from src.live.testnet_drill.execution import cancel_probe, make_intent, run_cycle, settle_journal, submit_probe
from src.live.testnet_drill.guard import DrillRefused


def test_production_ioc_wal_and_rejection_isolation(exchange_session):
    session, venue, _ = exchange_session()
    seed = make_intent(session, "BTCUSDT")
    bad = replace(seed, quantity=seed.quantity + session.filters[seed.symbol].step_size / 2, leg_index=1)
    trace = run_cycle(session, [seed, bad, replace(seed, leg_index=2)])
    assert trace.escaped is None
    assert [outcome.status for outcome in trace.outcomes] == ["FILLED", "REJECTED", "FILLED"]
    assert trace.outcomes[1].reject_code == -1111
    assert all(outcome.filled_qty + outcome.unfilled_qty == intent.quantity for outcome, intent in zip(trace.outcomes, trace.intents, strict=True))
    assert session.journal.submits()
    assert session.journal.fills_after(-1)
    assert trace.post_quotes
    assert venue.account("order-key-abc")["positions"]["BTCUSDT"] == seed.quantity * 2


def test_timeout_preserves_explicit_partial_outcomes(exchange_session):
    session, _, _ = exchange_session(config={"scenario_timeout_s": 3})
    trace = run_cycle(session, [make_intent(session, "BTCUSDT")])
    assert trace.escaped == "ExecutionInterrupted"
    assert trace.outcomes[0].unfilled_qty > 0
    assert not session.client.open_orders()


def test_direct_probes_are_durable_before_delivery(exchange_session):
    session, _, _ = exchange_session()
    params = {"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT", "timeInForce": "GTX", "price": "99.90", "quantity": "0.1", "newClientOrderId": session.next_client_order_id("BTCUSDT")}
    response = submit_probe(session, params)
    assert response["status"] == "NEW"
    assert session.journal.submits()[0].client_order_id == params["newClientOrderId"]
    confirmation = cancel_probe(session, "BTCUSDT", params["newClientOrderId"])
    assert confirmation.closed
    assert session.journal.terminal(params["newClientOrderId"]).status == "CANCELED"


def test_guard_refusal_is_terminal_without_delivery(exchange_session):
    session, _, _ = exchange_session()
    with pytest.raises(DrillRefused, match="max_notional"):
        submit_probe(session, {"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT", "price": "100", "quantity": "1"})
    assert session.journal.terminal(session.journal.submits()[0].client_order_id).status == "NOT_PLACED"


def test_reduce_only_sub_minimum_is_quantized_without_opening_minimum(exchange_session):
    session, _, _ = exchange_session()
    with pytest.raises(DrillRefused, match="below_venue_minimum"):
        make_intent(session, "BTCUSDT", quantity=Decimal("0.001"))
    intent = make_intent(session, "BTCUSDT", quantity=Decimal("0.001"), side="SELL", reduce_only=True)
    assert intent.quantity == Decimal("0.001")
    settle_journal(session)


def test_intent_above_probe_cap_is_refused(exchange_session):
    session, venue, _ = exchange_session()
    with pytest.raises(DrillRefused, match="intent_notional_cap"):
        make_intent(session, "BTCUSDT", quantity=Decimal("1"))
    assert not any(method == "POST" for _, method, _, _ in venue.calls)


def test_settlement_refuses_unconfirmed_cancels(exchange_session, monkeypatch):
    from types import SimpleNamespace

    import src.live.testnet_drill.execution as execution_module

    session, _, _ = exchange_session()
    params = {"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT", "timeInForce": "GTX", "price": "99.90", "quantity": "0.1"}
    submit_probe(session, params)
    monkeypatch.setattr(
        execution_module, "settle_unresolved_orders", lambda *a, **kw: SimpleNamespace(still_open=a[3])
    )
    with pytest.raises(DrillRefused, match="unsettled_orders"):
        settle_journal(session)


def test_reused_journal_id_is_sent_without_second_submit_record(exchange_session):
    session, _, _ = exchange_session()
    order_id = session.next_client_order_id("BTCUSDT")
    params = {"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT", "timeInForce": "GTX", "price": "99.90", "quantity": "0.1", "newClientOrderId": order_id}
    submit_probe(session, params)
    cancel_probe(session, "BTCUSDT", order_id)
    hooks = []
    response = submit_probe(session, params, before_send=lambda: hooks.append("sent"))
    assert response["status"] == "NEW"
    assert hooks == ["sent"]
    assert [s.client_order_id for s in session.journal.submits()] == [order_id]
