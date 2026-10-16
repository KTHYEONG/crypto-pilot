"""Venue-bound id reuse and durable drill decision-window isolation."""

import json
from dataclasses import replace
from urllib.parse import urlsplit

import pytest

from src.live.testnet_drill.guard import DrillRefused
from src.live.testnet_drill.registry import SCENARIOS
from src.live.testnet_drill.report import Verdict
from src.live.testnet_drill.scenarios.cycle_e2e import _decision_days


def test_decision_windows_do_not_overlap_even_for_identical_seeds(exchange_session):
    exchange_session()
    first = _decision_days(3, seed="same-run")
    second = _decision_days(3, seed="same-run")
    assert set(first).isdisjoint(second)
    assert (first[-1] - first[0]).days == 2


def test_g4_real_reuse_refutes_h12(exchange_session):
    session, venue, _ = exchange_session(config={"symbols": ("BTCUSDT", "ETHUSDT")})
    spec = SCENARIOS["G4"]
    evidence = spec.run(session, session.config)
    result = spec.evaluate(evidence)
    assert result.verdict is Verdict.PASS
    assert result.outcome_for("H12") == "REFUTED"
    first = evidence.payload["first_order_id"]
    replacement = evidence.payload["replacement_order_id"]
    orders = {order["orderId"]: order for order in venue.orders}
    assert orders[first]["clientOrderId"] == orders[replacement]["clientOrderId"]
    assert orders[first]["status"] == "FILLED"
    assert orders[replacement]["status"] == "CANCELED"
    assert orders[replacement]["executedQty"] == "0"


def test_g4_fails_when_identity_guard_is_disabled(exchange_session, monkeypatch):
    from src.live import order_identity

    monkeypatch.setattr(order_identity, "assert_order_identity", lambda *args, **kwargs: None)
    session, _, _ = exchange_session(config={"symbols": ("BTCUSDT", "ETHUSDT")})
    spec = SCENARIOS["G4"]
    evidence = spec.run(session, session.config)
    result = spec.evaluate(evidence)
    assert result.verdict is Verdict.FAIL
    assert evidence.payload["no_phantom_fill"] is False
    assert result.outcome_for("H12") == "CONFIRMED"


def test_exhausted_decision_window_refuses_before_any_order(exchange_session):
    session, venue, _ = exchange_session()
    with pytest.raises(DrillRefused, match="window_exhausted"):
        _decision_days(100_000, seed=session.run_id)
    assert not venue.orders


@pytest.mark.parametrize("variant", ["initial_unfilled", "budget", "replacement_unrest", "same_order_id"])
def test_g4_refuses_unproven_venue_preconditions(exchange_session, monkeypatch, variant):
    session, venue, _ = exchange_session(
        config={"symbols": ("BTCUSDT", "ETHUSDT"), "scenario_timeout_s": 10 if variant == "budget" else 500},
    )
    original = venue.call
    posts = []

    def call(method, url, headers):
        response = original(method, url, headers)
        if method == "GET" and urlsplit(url).path == "/fapi/v1/order" and variant == "initial_unfilled" and posts:
            row = json.loads(response.body)
            if row.get("clientOrderId") == posts[0]["clientOrderId"]:
                row.update(status="EXPIRED", executedQty="0")
                return replace(response, body=json.dumps(row).encode())
        if method == "POST" and urlsplit(url).path == "/fapi/v1/order":
            row = json.loads(response.body)
            posts.append(row)
            # posts: 1 seed IOC, 2 unjournaled flatten, 3 replacement
            if len(posts) == 3 and variant == "replacement_unrest":
                row["status"] = "FILLED"
            elif len(posts) == 3 and variant == "same_order_id":
                row["orderId"] = posts[0]["orderId"]
            return replace(response, body=json.dumps(row).encode())
        return response

    monkeypatch.setattr(venue, "call", call)
    with pytest.raises(DrillRefused, match="g4_"):
        SCENARIOS["G4"].run(session, session.config)
