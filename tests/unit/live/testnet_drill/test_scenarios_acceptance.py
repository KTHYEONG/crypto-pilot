"""Acceptance group (B) pure-evaluate invariants."""

from __future__ import annotations

import pytest

from src.live.errors import BINANCE_ERROR_POLICY
from src.live.planner import build_client_order_id
from src.live.testnet_drill.report import Verdict
from src.live.testnet_drill.scenarios import acceptance
from src.live.testnet_drill.scenarios.base import ev

_SPECS = {spec.scenario_id: spec for spec in acceptance.scenario_specs()}


def test_unregistered_code_yields_proposal_not_pass() -> None:
    evidence = ev("B9", code=-4116, cycle_halted=True)
    result = _SPECS["B9"].evaluate(evidence)
    assert result.verdict is Verdict.OBSERVED
    assert result.observations[0].code == -4116
    assert result.observations[0].registry_action == "fail_closed"
    assert result.proposed_action == "fail_closed"
    assert -4116 not in BINANCE_ERROR_POLICY


def test_scoped_reject_isolation_evaluated() -> None:
    passing = ev(
        "B8",
        outcomes=["FILLED", "REJECTED", "FILLED"],
        escaped_exception=False,
        terminal_once=True,
        conservation_passed=True,
    )
    assert _SPECS["B8"].evaluate(passing).verdict is Verdict.PASS
    failing = ev("B8", outcomes=["FILLED", "REJECTED", "FILLED"], escaped_exception=True, terminal_once=True)
    assert _SPECS["B8"].evaluate(failing).verdict is Verdict.FAIL


def test_min_notional_agreement() -> None:
    evidence = ev("B2", level="exact_min", client_decision="skip", venue_decision="accept", agreed=False)
    result = _SPECS["B2"].evaluate(evidence)
    assert result.verdict is Verdict.FAIL


def test_raw_send_needs_guard() -> None:
    refused = ev("B1", refused_before_transport=True, conservation_passed=True)
    assert acceptance.evaluate_raw_notional_guard(refused).verdict is Verdict.PASS
    sent = ev("B1", refused_before_transport=False)
    assert acceptance.evaluate_raw_notional_guard(sent).verdict is Verdict.FAIL


def test_client_id_length_boundary() -> None:
    candidate = build_client_order_id("r" * 17, "BTCUSDT", 0, 0, 0)
    assert len(candidate) == 36
    with pytest.raises(ValueError, match="client order id"):
        build_client_order_id("r" * 18, "BTCUSDT", 0, 0, 0)
    passing = ev(
        "B11",
        ok_36=True,
        rejected_37=True,
        venue_accepted=True,
        non_ascii_accepted=True,
        legacy_accepted=True,
        conservation_passed=True,
    )
    assert _SPECS["B11"].evaluate(passing).verdict is Verdict.PASS
    failing = ev("B11", ok_36=True, rejected_37=False)
    assert _SPECS["B11"].evaluate(failing).verdict is Verdict.FAIL


def test_registry_covers_all_b_scenarios() -> None:
    assert set(_SPECS) == {f"B{i}" for i in range(1, 12)}
