# ruff: noqa
"""Live runner tests - ledger."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import src.live.runner as runner_mod
from src.live.account import (
    AccountSnapshot,
    assert_drawdown_within_limit,
    assert_suppressed_venue_flat,
    resolve_sizing_equity,
)
from src.live.errors import ReconciliationBreach, RiskGateBreach, VenueError
from src.live.executor import ExecutionOutcome
from src.live.ledger import LedgerState, load_ledger, save_ledger
from src.live.runner import check_risk_gates, run_shadow_cycle
from src.live.settings import LiveSettings

from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient

@pytest.fixture(autouse=True)
def _maybe_disable_orderbook_capture(monkeypatch, request):  # noqa: ARG001
    if "captures_orderbook" in request.node.name:
        return
    import src.live.orderbook as ob_mod

    monkeypatch.setattr(ob_mod, "capture_order_books", lambda *a, **k: [])
    monkeypatch.setattr(ob_mod, "append_order_book_snapshots", lambda *a, **k: [])

@pytest.fixture
def artifact(tmp_path):
    frame = pd.DataFrame(
        {"AAAUSDT": [0.02], "BUSDT": [-0.02]},
        index=pd.DatetimeIndex([DECISION_TIME]),
    )
    path = tmp_path / "deployed_target_weights.parquet"
    frame.to_parquet(path, index=True)
    return path

@pytest.fixture
def live_env(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient()
    )
    calls: list[Any] = []

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        calls.extend(intents)
        outcomes = tuple(
            ExecutionOutcome(
                symbol=intent.symbol,
                filled_qty=intent.quantity,
                unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"),
                chases=0,
                status="FILLED",
            )
            for intent in intents
        )
        audit.record("intents_executed", count=len(outcomes))
        return outcomes

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)
    monkeypatch.setattr(
        runner_mod,
        "default_audit_log_path",
        lambda name, for_date=None: tmp_path / f"{name}.jsonl",
    )
    import src.live.orderbook as ob_mod

    monkeypatch.setattr(ob_mod, "capture_order_books", lambda *a, **k: [])
    monkeypatch.setattr(ob_mod, "append_order_book_snapshots", lambda *a, **k: [])
    return calls

def test_SCENARIO_LIVE_DAEMON_08_audit_keyed_by_decision_date(
    monkeypatch, artifact, tmp_path
) -> None:
    # 실제 default_audit_log_path를 쓰되 루트만 tmp로 격리해 파일 경로를 검증한다.
    import src.live.audit as audit_mod

    monkeypatch.setattr(audit_mod, "AUDIT_LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient()
    )

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        outcomes = tuple(
            ExecutionOutcome(
                symbol=intent.symbol,
                filled_qty=intent.quantity,
                unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"),
                chases=0,
                status="FILLED",
            )
            for intent in intents
        )
        audit.record("intents_executed", count=len(outcomes))
        return outcomes

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)

    # 캐치업: wall-clock now가 decision_time보다 이틀 뒤다.
    # 스테일 게이트는 별도 시나리오(LIVE_19/21)가 담당하므로 여기서는 상한을
    # 넉넉히 열어 감사 로그의 decision_date 파티셔닝만 검증한다.
    settings = LiveSettings(
        notional_equity_usdt=2000.0,
        ledger_path=str(tmp_path / "ledger_keyed.json"),
        max_signal_staleness_hours=72.0,
    )
    late_now = DECISION_TIME + pd.Timedelta(days=2)
    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=late_now)
    assert report.status == "COMPLETE"

    decision_log = tmp_path / "logs" / "live" / "shadow_cycle" / "2026-08-24.jsonl"
    wall_clock_log = tmp_path / "logs" / "live" / "shadow_cycle" / "2026-08-26.jsonl"
    assert decision_log.exists()
    assert not wall_clock_log.exists()



def test_SCENARIO_LIVE_18_ledger_durability_on_execution_failure(
    artifact, monkeypatch, tmp_path
) -> None:
    """I-LEDGER-DURABLE/R6: 집행 중 예외여도 확인된 체결은 원장에 영속되고 HALT 를 반환한다."""
    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient()
    )
    monkeypatch.setattr(
        runner_mod,
        "default_audit_log_path",
        lambda name, for_date=None: tmp_path / f"{name}.jsonl",
    )

    raised_fill = Decimal("0.398")

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        first = intents[0]
        outcome = ExecutionOutcome(
            symbol=first.symbol,
            filled_qty=min(raised_fill, first.quantity),
            unfilled_qty=max(first.quantity - raised_fill, Decimal("0")),
            avg_fill_price=Decimal("100"),
            chases=0,
            status="RESIDUAL",
        )
        exc = VenueError(
            "venue connection lost",
            code=-1000,
            http_status=500,
            path="/fapi/v1/order",
            payload_digest="0" * 12,
        )
        exc.partial_outcomes = (outcome,)
        raise exc

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)

    ledger_path = tmp_path / "ledger_durability.json"
    settings = LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(ledger_path))
    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "HALT"
    state = load_ledger(Path(settings.ledger_path or ""))
    assert state.positions["AAAUSDT"] == raised_fill  # 첫 intent 의 부호 있는 체결량
    assert state.equity_high_water_mark == Decimal("2000")



def test_SCENARIO_LIVE_48_CASH_TRACKING_SURVIVES_PARTIAL_FILL_HALT(tmp_path, monkeypatch) -> None:
    """SCENARIO_LIVE_48: in PAPER mode, a HALT mid-execution still persists
    cash reflecting exactly the confirmed partial fill."""
    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient()
    )
    monkeypatch.setattr(
        runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl"
    )

    raised_fill = Decimal("0.398")

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        first = intents[0]
        outcome = ExecutionOutcome(
            symbol=first.symbol,
            filled_qty=min(raised_fill, first.quantity),
            unfilled_qty=max(first.quantity - raised_fill, Decimal("0")),
            avg_fill_price=Decimal("100"),
            chases=0,
            status="RESIDUAL",
        )
        exc = VenueError(
            "venue connection lost", code=-1000, http_status=500,
            path="/fapi/v1/order", payload_digest="0" * 12,
        )
        exc.partial_outcomes = (outcome,)
        raise exc

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)

    weights = pd.DataFrame(
        {"AAAUSDT": [0.02], "BUSDT": [-0.02]}, index=pd.DatetimeIndex([DECISION_TIME])
    )
    weights_path = tmp_path / "weights.parquet"
    weights.to_parquet(weights_path, index=True)

    ledger_path = tmp_path / "ledger_cash_halt.json"
    settings = LiveSettings(mode="paper", notional_equity_usdt=2000.0, ledger_path=str(ledger_path))
    report = run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)

    assert report.status == "HALT"
    state = load_ledger(ledger_path)
    expected_cash = Decimal("2000") - (raised_fill * Decimal("100")) - (raised_fill * Decimal("100") * Decimal("5") / Decimal("10000"))
    assert state.cash_usdt == expected_cash


