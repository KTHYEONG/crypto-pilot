"""Isolated drill session invariants."""

from __future__ import annotations

import dataclasses
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.common.errors import DataIntegrityError
from src.live.testnet_drill.guard import DrillRefused
from src.live.testnet_drill.session import (
    SlateNotClean,
    _flatten_positions,
    _margin_setup,
    _order_notional,
    _read_positions,
    open_drill_session,
    prepare_clean_slate,
    teardown_session,
)
from tests.unit.live.testnet_drill._stubs import StatefulVenue, make_config, make_settings, patch_drill_root


class OriginalError(Exception):
    """Scenario failure that teardown must never mask."""


@pytest.fixture
def drill_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return patch_drill_root(monkeypatch, tmp_path)


def _open(drill_root: Path, venue: StatefulVenue, **overrides: Any) -> Any:
    config = overrides.pop("config", make_config())
    run_id = overrides.pop("run_id", "20261016a")
    return open_drill_session(make_settings(), config, run_id=run_id, inner=venue, **overrides)


def _raise_and_teardown(session: Any) -> None:
    try:
        raise OriginalError("boom")
    finally:
        teardown_session(session)


def test_session_paths_are_isolated(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT"])
    session = _open(drill_root, venue)
    assert session.workspace.is_relative_to(drill_root)
    assert session.journal.path.is_relative_to(session.workspace)
    assert session.audit.path.is_relative_to(session.workspace)
    assert session.client_order_prefix.startswith("mhdrl")
    assert len(session.transport.records) == 2


def test_invalid_run_id_refused(drill_root: Path) -> None:
    with pytest.raises(DrillRefused, match="run_id"):
        _open(drill_root, StatefulVenue(["BTCUSDT"]), run_id="bad/id!")


def test_missing_symbol_aborts(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT"])
    with pytest.raises(DrillRefused, match="symbols"):
        _open(drill_root, venue, config=make_config(symbols=("MISSING",)))
    assert [call for call in venue.calls if call[0] == "POST"] == []


def test_notional_cap_enforced_centrally(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT"])
    session = _open(drill_root, venue)
    posts = len([call for call in venue.calls if call[0] == "POST"])
    with pytest.raises(DrillRefused, match="max_notional"):
        session.guarded_new_order({"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT", "quantity": "10", "price": "100"})
    assert len([call for call in venue.calls if call[0] == "POST"]) == posts


def test_raw_cap_is_enforced_before_transport(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT"])
    session = _open(drill_root, venue)
    before = len(venue.calls)
    with pytest.raises(DrillRefused, match="max_notional"):
        session.guarded_new_order({"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT", "quantity": "10.0001", "price": "100.0001"}, raw=True)
    assert len(venue.calls) == before


def test_probe_cannot_trade_exchange_symbol_outside_drill_scope(drill_root: Path) -> None:
    session = _open(drill_root, StatefulVenue(["BTCUSDT", "ETHUSDT"]))
    with pytest.raises(DrillRefused, match="symbol"):
        session.guarded_new_order({"symbol": "ETHUSDT", "quantity": "0.001", "price": "100"})


def test_clean_slate_refuses_foreign_symbols_before_cancel(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT", "ETHUSDT"], orders=[{"symbol": "ETHUSDT", "clientOrderId": "foreign"}])
    session = _open(drill_root, venue)
    with pytest.raises(DrillRefused, match="out_of_scope"):
        prepare_clean_slate(session)
    assert not any(method != "GET" for method, _, _ in venue.calls)


def test_guarded_new_order_assigns_id_and_sends(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT"])
    session = _open(drill_root, venue)
    response = session.guarded_new_order({"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT", "quantity": "0.001", "price": "10000"})
    assert response["status"] == "NEW"
    assert f"newClientOrderId={session.client_order_prefix}" in venue.calls[-1][1]


def test_guarded_new_order_rejects_foreign_id(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT"])
    session = _open(drill_root, venue)
    with pytest.raises(DrillRefused, match="client_order_id"):
        session.guarded_new_order({"symbol": "BTCUSDT", "newClientOrderId": "mhforeign-1", "quantity": "0.001", "price": "10000"})


def test_guarded_new_order_requires_symbol(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT"])
    session = _open(drill_root, venue)
    with pytest.raises(DrillRefused, match="symbol"):
        session.guarded_new_order({"quantity": "0.001", "price": "10000"})


def test_guarded_new_order_rejects_unevaluable_notional(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT"])
    session = _open(drill_root, venue)
    with pytest.raises(DrillRefused, match="notional"):
        session.guarded_new_order({"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT", "quantity": "0.001"})


def test_order_notional_variants() -> None:
    assert _order_notional({"quoteOrderQty": "12.5"}) is None
    assert _order_notional({"quantity": "0.5", "price": "100"}) == Decimal("50")
    assert _order_notional({"quantity": "0.5"}) is None
    assert _order_notional({"quantity": "nonsense", "price": "100"}) is None


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "0"])
def test_invalid_order_numbers_never_send(drill_root: Path, value: str) -> None:
    venue = StatefulVenue(["BTCUSDT"])
    session = _open(drill_root, venue)
    with pytest.raises(DrillRefused, match="notional"):
        session.guarded_new_order({"symbol": "BTCUSDT", "quantity": value, "price": "100", "quoteOrderQty": "1"})
    assert not any(method == "POST" for method, _, _ in venue.calls)


def test_market_probe_and_cleanup_share_cap(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT"], positions={"BTCUSDT": Decimal("100")})
    session = _open(drill_root, venue)
    with pytest.raises(DrillRefused, match="max_notional"):
        session.guarded_new_order({"symbol": "BTCUSDT", "type": "MARKET", "quantity": "1"})
    session.guarded_new_order({"symbol": "BTCUSDT", "type": "MARKET", "quantity": "0.1"})
    teardown_session(session)
    import urllib.parse

    for method, url, _ in venue.calls:
        if method == "POST":
            query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
            assert Decimal(query["quantity"]) * Decimal(100) <= session.config.max_probe_notional_usdt


def test_parent_kill_is_refused_without_io(drill_root: Path) -> None:
    from src.live.testnet_drill.faults import FaultAction, FaultRule

    venue = StatefulVenue(["BTCUSDT"])
    with pytest.raises(DrillRefused, match="parent_kill"):
        _open(drill_root, venue, rules=[FaultRule("POST", "/fapi/v1/order", action=FaultAction.KILL_AFTER_DELIVERY)])
    assert not venue.calls


def test_symlinked_audit_is_refused(drill_root: Path, tmp_path: Path) -> None:
    workspace = drill_root / "20261016a"
    workspace.mkdir(parents=True)
    outside = tmp_path / "outside.jsonl"
    (workspace / "audit.jsonl").symlink_to(outside)
    with pytest.raises(DrillRefused, match="session_paths"):
        _open(drill_root, StatefulVenue(["BTCUSDT"]))
    assert not outside.exists()


def test_hedged_positions_are_never_netted() -> None:
    client = SimpleNamespace(request=lambda *a, **k: [{"symbol": "BTCUSDT", "positionAmt": "1", "positionSide": "LONG"}])
    with pytest.raises(DrillRefused, match="position_mode"):
        _read_positions(client)


def test_cleanup_failure_is_audited(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT"], positions={"BTCUSDT": "1"}, auto_flatten=False)
    session = _open(drill_root, venue)
    teardown_session(session)
    assert "teardown_issue" in session.audit.path.read_text()


def test_too_small_cap_refuses_flatten(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT"], positions={"BTCUSDT": "1"})
    session = _open(drill_root, venue, config=make_config(probe_notional_usdt=Decimal("0.01"), max_probe_notional_usdt=Decimal("0.01")))
    with pytest.raises(DrillRefused, match="flatten_cap"):
        _flatten_positions(session, {"BTCUSDT": Decimal(1)})


def test_invalid_quote_cannot_bypass_cap(drill_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _open(drill_root, StatefulVenue(["BTCUSDT"]))
    monkeypatch.setattr(session.client, "book_ticker", lambda s: {"bidPrice": "NaN", "askPrice": "100"})
    with pytest.raises(DrillRefused, match="market_price"):
        session.guarded_new_order({"symbol": "BTCUSDT", "type": "MARKET", "quantity": "1"})


def test_income_explains_wallet_change(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT"])
    venue.income = [{"asset": "USDT", "income": "995"}]
    session = _open(drill_root, venue)
    checkpoint = session.workspace / "last_wallet.json"
    checkpoint.write_text(json.dumps({"balance": "5", "ended_at_ms": 1}))
    assert prepare_clean_slate(session).testnet_reset_suspected is False
    venue.income *= 1000
    assert prepare_clean_slate(session).testnet_reset_suspected is False


def test_next_client_order_id_length_guard(drill_root: Path) -> None:
    session = _open(drill_root, StatefulVenue(["BTCUSDT"]))
    oversized = dataclasses.replace(session, client_order_prefix="x" * 40)
    with pytest.raises(DrillRefused, match="client_order_id"):
        oversized.next_client_order_id("BTCUSDT")


def test_flatten_skips_zero(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT"])
    session = _open(drill_root, venue)
    posts = len(venue.calls)
    _flatten_positions(session, {"BTCUSDT": Decimal(0)})
    assert len(venue.calls) == posts


def test_clean_slate_postcondition(drill_root: Path) -> None:
    venue = StatefulVenue(
        ["BTCUSDT"],
        positions={"BTCUSDT": Decimal("0.5")},
        orders=[{"symbol": "BTCUSDT", "clientOrderId": "stale-1"}],
    )
    session = _open(drill_root, venue)
    report = prepare_clean_slate(session)
    assert report.starting_wallet_balance == Decimal("1000.0")
    assert report.testnet_reset_suspected is False
    assert venue.positions["BTCUSDT"] == 0
    assert venue.orders == []
    flatten_call = next(call for call in venue.calls if call[0] == "POST")
    assert "reduceOnly=true" in flatten_call[1]
    assert "type=MARKET" in flatten_call[1]


def test_clean_slate_raises_when_not_flat(drill_root: Path) -> None:
    venue = StatefulVenue(
        ["BTCUSDT"],
        positions={"BTCUSDT": Decimal("0.5")},
        orders=[{"symbol": "BTCUSDT", "clientOrderId": "stale-1"}],
        auto_flatten=False,
        auto_cancel=False,
    )
    session = _open(drill_root, venue)
    with pytest.raises(SlateNotClean) as excinfo:
        prepare_clean_slate(session)
    assert "BTCUSDT" in excinfo.value.residuals
    assert "open_orders" in excinfo.value.residuals
    assert len([call for call in venue.calls if "allOpenOrders" in call[1]]) == 3


def test_clean_slate_rejects_bad_account(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT"])
    venue.malformed_account = True
    session = _open(drill_root, venue)
    with pytest.raises(DataIntegrityError, match="account endpoint"):
        prepare_clean_slate(session)


def test_wallet_reset_suspected(drill_root: Path) -> None:
    session = _open(drill_root, StatefulVenue(["BTCUSDT"]))
    (session.workspace / "last_wallet.json").write_text(json.dumps({"balance": "5.0", "run_id": "old", "ended_at_ms": 1}), encoding="utf-8")
    assert prepare_clean_slate(session).testnet_reset_suspected is True


def test_wallet_write_failure_is_informational(drill_root: Path) -> None:
    session = _open(drill_root, StatefulVenue(["BTCUSDT"]))
    (session.workspace / "last_wallet.json").mkdir()
    assert prepare_clean_slate(session).testnet_reset_suspected is False


def test_margin_setup_variants() -> None:
    account = {"positions": [{"symbol": "BTCUSDT", "leverage": "n/a", "isolated": True}, "junk"]}
    leverage, margin_type = _margin_setup(account, ["BTCUSDT", "ETHUSDT"])
    assert dict(leverage) == {"BTCUSDT": 0, "ETHUSDT": 0}
    assert dict(margin_type) == {"BTCUSDT": "isolated", "ETHUSDT": "unknown"}


def test_read_positions_rejects_bad_schema() -> None:
    client = SimpleNamespace(request=lambda *a, **k: {"unexpected": True})
    with pytest.raises(DataIntegrityError, match="positionRisk"):
        _read_positions(client)
    broken = SimpleNamespace(request=lambda *a, **k: [{"symbol": "BTCUSDT"}])
    with pytest.raises(DataIntegrityError, match="positionAmt"):
        _read_positions(broken)


def test_teardown_never_masks_original_error(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT"])
    session = _open(drill_root, venue)
    venue.failing = True
    with pytest.raises(OriginalError, match="boom"):
        _raise_and_teardown(session)


def test_teardown_cancels_own_and_flattens(drill_root: Path) -> None:
    venue = StatefulVenue(["BTCUSDT"], positions={"BTCUSDT": Decimal("0.25")})
    session = _open(drill_root, venue)
    own = f"{session.client_order_prefix}-ABC-1"
    venue.orders = [
        {"symbol": "BTCUSDT", "clientOrderId": own},
        {"symbol": "BTCUSDT", "clientOrderId": "mhforeign-9"},
        "junk",
    ]
    teardown_session(session)
    assert [o["clientOrderId"] for o in venue.orders if isinstance(o, dict)] == ["mhforeign-9"]
    assert venue.positions["BTCUSDT"] == 0
    assert venue.closed is True
