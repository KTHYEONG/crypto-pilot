"""Scenario execution and conservation on a two-account HTTP exchange."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.live.testnet_drill.registry import SCENARIOS
from src.live.testnet_drill.report import Verdict


@pytest.mark.parametrize("scenario_id", ["C1", "C3", "C5", "C6", "C8", "C9", "C11", "C12", "C13", "B1", "B8", "B9", "B11", "A2", "A3", "A4", "A5", "A7", "A8"])
def test_recorded_evidence_from_actual_production_paths(exchange_session, scenario_id):
    session, venue, _ = exchange_session(config={"symbols": ("BTCUSDT", "ETHUSDT", "SOLUSDT"), "race_repetitions": 3, "slate_max_attempts": 10})
    spec = SCENARIOS[scenario_id]
    evidence = spec.run(session, session.config)
    assert evidence.scenario_id == scenario_id
    assert "note" not in evidence.payload
    assert evidence.payload["primary_clean"] is True
    assert evidence.payload["counterparty_clean"] is True
    result = spec.evaluate(evidence)
    assert result.verdict in (Verdict.PASS, Verdict.OBSERVED, Verdict.FAIL)
    assert any(method == "POST" for _, method, _, _ in venue.calls) or scenario_id == "A8"
    if scenario_id in ("C5", "C6", "C9"):
        assert any(key == "counterparty-key" and method == "POST" for key, method, _, _ in venue.calls)
        assert any(key == "order-key-abc" and method == "POST" for key, method, _, _ in venue.calls)
        assert all(check.passed for check in evidence.payload["reconciliation"] if check.name in ("position_parity", "order_watermark_parity", "no_duplicate_fills"))
        assert session.counterparties[0].journal.fills_after(-1) == ()
        assert not any(venue.account("counterparty-key")["positions"].values())
        assert not any(venue.account("order-key-abc")["positions"].values())


@pytest.mark.parametrize("scenario_id", ["C5", "C6", "C9"])
def test_two_accounts_pass_conservation_without_combining_fills(exchange_session, scenario_id):
    session, venue, _ = exchange_session(config={"race_repetitions": 3})
    spec = SCENARIOS[scenario_id]
    evidence = spec.run(session, session.config)
    assert evidence.payload["conservation_passed"] is True
    assert spec.evaluate(evidence).verdict is Verdict.PASS
    primary = venue.account("order-key-abc")
    aux = venue.account("counterparty-key")
    # C6 cancel-vs-fill ordering is thread-scheduled; either side may win every repetition.
    assert primary["trades"] or scenario_id == "C6"
    assert aux["trades"] or scenario_id == "C9"
    assert sum(Decimal(row["qty"]) for row in primary["trades"]) == sum(
        fill.quantity for fill in session.journal.fills_after(-1)
    )
    assert all(fill.client_order_id.startswith(session.client_order_prefix) for fill in session.journal.fills_after(-1))


def test_slicing_uses_real_unique_submissions(exchange_session):
    session, _, _ = exchange_session(config={"probe_notional_usdt": Decimal("600"), "max_probe_notional_usdt": Decimal("700")})
    evidence = SCENARIOS["C10"].run(session, session.config)
    assert evidence.payload["slice_ids_unique"] is True
    assert len(session.journal.submits()) >= 3
    assert SCENARIOS["C10"].evaluate(evidence).verdict is Verdict.PASS


def test_cleanup_budget_failure_cannot_pass(exchange_session):
    session, _, _ = exchange_session(config={"symbols": ("BTCUSDT", "ETHUSDT", "SOLUSDT"), "slate_max_attempts": 1})
    evidence = SCENARIOS["C13"].run(session, session.config)
    assert evidence.payload["primary_clean"] is False
    assert evidence.payload["conservation_passed"] is False
    assert SCENARIOS["C13"].evaluate(evidence).verdict is Verdict.FAIL


def test_min_notional_boundary_levels_stay_on_step_grid(exchange_session):
    session, _, _ = exchange_session()
    evidence = SCENARIOS["B2"].run(session, session.config)
    rows = evidence.payload["boundary_rows"]
    assert len(rows) == 6
    assert all(row["code"] != -1111 for row in rows)
    assert evidence.payload["client_decision"] == evidence.payload["venue_decision"] == "accept"
