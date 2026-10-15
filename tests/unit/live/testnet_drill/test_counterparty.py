"""Separate-account routing, bounded cleanup, and primary journal isolation."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest
from pydantic import SecretStr

from src.live.testnet_drill.guard import DrillRefused
from src.live.testnet_drill.session import close_session_handles, open_counterparty_session, teardown_session


def test_second_credentials_are_used_only_on_separate_state(exchange_session):
    session, venue, _ = exchange_session()
    aux = open_counterparty_session(session)
    assert session.aux_client is aux.client
    assert open_counterparty_session(session) is aux
    assert aux.workspace == session.workspace / "counterparty"
    assert aux.journal.path != session.journal.path
    assert not aux.client_order_prefix.startswith("mh")
    assert all(key == "counterparty-key" for key, _, _, _ in venue.calls[2:])
    assert "counterparty-key" not in (aux.workspace / "wire.jsonl").read_text()
    assert "counterparty-secret" not in (aux.workspace / "wire.jsonl").read_text()


@pytest.mark.parametrize(("key", "secret", "message"), [
    (None, None, "credentials"), (SecretStr(""), SecretStr(""), "credentials"),
    (SecretStr("order-key-abc"), SecretStr("aux"), "separation"),
    (SecretStr("main-key-xyz"), SecretStr("aux"), "separation"),
])
def test_missing_or_reused_credentials_refused_before_aux_contact(exchange_session, key, secret, message):
    session, venue, _ = exchange_session(settings={"order_api_key2": key, "order_api_secret2": secret})
    before = len(venue.calls)
    with pytest.raises(DrillRefused, match=message):
        open_counterparty_session(session)
    assert len(venue.calls) == before
    assert session.aux_client is None


def test_dirty_counterparty_is_not_modified(exchange_session):
    session, venue, _ = exchange_session()
    venue.account("counterparty-key")["positions"]["BTCUSDT"] = Decimal("1")
    with pytest.raises(DrillRefused, match="not_flat"):
        open_counterparty_session(session)
    assert venue.account("counterparty-key")["positions"]["BTCUSDT"] == Decimal("1")
    assert not any(method != "GET" for key, method, _, _ in venue.calls if key == "counterparty-key")
    assert not session.counterparties


def test_aux_probe_never_changes_primary_journal_or_wallet(exchange_session):
    session, venue, _ = exchange_session()
    aux = open_counterparty_session(session)
    wallet = venue.account("order-key-abc")["wallet"]
    aux.guarded_new_order({"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET", "quantity": "0.1"})
    assert venue.account("order-key-abc")["wallet"] == wallet
    assert session.journal.fills_after(-1) == ()
    assert not session.journal.path.exists()
    assert venue.account("counterparty-key")["positions"]["BTCUSDT"] == Decimal("0.1")
    teardown_session(session, close_handles=False)
    assert not any(venue.account("counterparty-key")["positions"].values())
    assert venue.close_count == 0
    close_session_handles(session)
    assert venue.close_count == 2


def test_aux_notional_guard_precedes_wire_send(exchange_session):
    session, venue, _ = exchange_session()
    aux = open_counterparty_session(session)
    before = len(venue.calls)
    with pytest.raises(DrillRefused, match="max_notional"):
        aux.guarded_new_order({"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT", "timeInForce": "GTC", "price": "100", "quantity": "1"})
    assert len(venue.calls) == before


def test_aux_is_testnet_only_even_for_constructed_settings(exchange_session):
    session, _, _ = exchange_session()
    bad = replace(session, settings=session.settings.model_copy(update={"order_base_url": "https://fapi.binance.com"}))
    with pytest.raises(DrillRefused, match="order_base_url"):
        open_counterparty_session(bad)
