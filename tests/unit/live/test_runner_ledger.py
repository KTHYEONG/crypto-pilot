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


def test_run_shadow_cycle_skips_already_executed_decision_without_venue_calls(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest
    import src.live.runner as runner_mod
    from src.live.ledger import LedgerState, save_ledger
    from src.live.settings import LiveSettings

    def _forbidden(*a, **k):
        raise AssertionError("already executed decision must not reach venue or weights")

    monkeypatch.setattr(runner_mod, "_market_client", _forbidden)
    monkeypatch.setattr(runner_mod, "_order_client", _forbidden)
    monkeypatch.setattr(runner_mod, "latest_target_weights", _forbidden)
    decision = pd.Timestamp("2026-08-24 00:00Z")
    ledger_path = tmp_path / "ledger.json"
    save_ledger(ledger_path, LedgerState(last_executed_decision_time=decision))
    settings = LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(ledger_path))

    same = runner_mod.run_shadow_cycle(settings, decision, tmp_path / "missing.parquet", now=decision + pd.Timedelta(hours=2))
    older = runner_mod.run_shadow_cycle(settings, decision - pd.Timedelta(days=1), tmp_path / "missing.parquet", now=decision + pd.Timedelta(hours=2))

    assert (same.status, same.reason, same.intent_count) == ("COMPLETE", "already_executed", 0)
    assert (older.status, older.reason) == ("COMPLETE", "already_executed")


def test_run_shadow_cycle_proceeds_when_decision_newer_than_last_executed(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.live.runner as runner_mod
    from src.live.ledger import LedgerState, save_ledger
    from src.live.settings import LiveSettings

    reached: list[pd.Timestamp] = []

    def _weights(path, decision_time, **kwargs):
        reached.append(decision_time)
        raise ValueError("weights reached")

    monkeypatch.setattr(runner_mod, "latest_target_weights", _weights)
    decision = pd.Timestamp("2026-08-24 00:00Z")
    ledger_path = tmp_path / "ledger.json"
    save_ledger(ledger_path, LedgerState(last_executed_decision_time=decision - pd.Timedelta(days=1)))
    settings = LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(ledger_path))

    report = runner_mod.run_shadow_cycle(settings, decision, tmp_path / "w.parquet", now=decision + pd.Timedelta(hours=2))

    assert reached == [decision]
    assert report.status == "HALT"
    assert report.reason == "weights reached"


def test_persist_confirmed_fills_stamps_or_preserves_last_executed(tmp_path) -> None:
    from decimal import Decimal
    import pandas as pd
    import src.live.runner as runner_mod
    from src.live.executor import ExecutionOutcome
    from src.live.ledger import LedgerState, load_ledger
    from src.live.planner import OrderIntent

    previous = pd.Timestamp("2026-08-23 00:00Z")
    decision = pd.Timestamp("2026-08-24 00:00Z")
    base = LedgerState(positions={}, equity_high_water_mark=Decimal("2000"), last_executed_decision_time=previous)
    intent = OrderIntent(
        symbol="AAAUSDT", side="BUY", quantity=Decimal("1"), reduce_only=False,
        target_qty=Decimal("1"), current_qty=Decimal("0"), client_order_prefix="run1",
        leg_index=0, decision_price=Decimal("100"),
    )
    outcome = ExecutionOutcome(
        symbol="AAAUSDT", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"),
        avg_fill_price=Decimal("100"), chases=0, status="FILLED",
    )

    kept_path = tmp_path / "kept.json"
    kept = runner_mod._persist_confirmed_fills(kept_path, base, [], [], Decimal("2000"))
    stamped_path = tmp_path / "stamped.json"
    stamped = runner_mod._persist_confirmed_fills(
        stamped_path, base, [intent], [outcome], Decimal("2000"), executed_decision_time=decision,
    )

    assert kept.last_executed_decision_time == previous
    assert load_ledger(kept_path).last_executed_decision_time == previous
    assert stamped.last_executed_decision_time == decision
    assert load_ledger(stamped_path).last_executed_decision_time == decision
    assert stamped.positions == {"AAAUSDT": Decimal("1")}


def test_accrue_ledger_funding_preserves_last_executed(tmp_path, monkeypatch) -> None:
    from decimal import Decimal
    import pandas as pd
    import src.live.runner as runner_mod
    from src.live.ledger import LedgerState, PositionSnapshot, load_ledger

    executed = pd.Timestamp("2026-08-23 00:00Z")
    now = pd.Timestamp("2026-08-24 12:00Z")
    epoch = pd.Timestamp("2026-08-24 08:00Z")
    seed, _ = runner_mod._accrue_ledger_funding(
        LedgerState(positions={}, last_executed_decision_time=executed), now, tmp_path / "seed.json",
    )
    monkeypatch.setattr(runner_mod, "_load_paper_funding", lambda symbols: {"AAAUSDT": pd.Series([0.001], index=pd.DatetimeIndex([epoch]))})
    monkeypatch.setattr(runner_mod, "_load_paper_marks", lambda symbols: {"AAAUSDT": pd.Series([100.0], index=pd.DatetimeIndex([epoch]))})
    held, accrual = runner_mod._accrue_ledger_funding(
        LedgerState(
            positions={"AAAUSDT": Decimal("1")}, cash_usdt=Decimal("1000"),
            funding_accrued_through=pd.Timestamp("2026-08-24 00:00Z"), last_executed_decision_time=executed,
        ),
        now, tmp_path / "held.json",
    )

    assert seed.last_executed_decision_time == executed
    assert seed.funding_accrued_through == now
    assert held.last_executed_decision_time == executed
    assert held.cash_usdt == Decimal("1000") - Decimal("0.1")
    assert held.position_history == (PositionSnapshot(effective_from=pd.Timestamp("2026-08-24 00:00Z"), positions={"AAAUSDT": Decimal("1")}),)
    assert held.funding_watermarks == {"AAAUSDT": epoch}
    assert accrual.lag_by_symbol == {"AAAUSDT": now - epoch}
    reloaded = load_ledger(tmp_path / "held.json")
    assert reloaded.last_executed_decision_time == executed
    assert reloaded.funding_watermarks == {"AAAUSDT": epoch}
def test_run_shadow_cycle_failed_execution_does_not_stamp_last_executed(artifact, monkeypatch, tmp_path) -> None:
    from decimal import Decimal
    import src.live.runner as runner_mod
    from src.live.errors import VenueError
    from src.live.executor import ExecutionOutcome
    from src.live.ledger import load_ledger
    from src.live.settings import LiveSettings
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        first = intents[0]
        exc = VenueError("venue connection lost", code=-1000, http_status=500, path="/fapi/v1/order", payload_digest="0" * 12)
        exc.partial_outcomes = (
            ExecutionOutcome(
                symbol=first.symbol, filled_qty=Decimal("0.1"), unfilled_qty=first.quantity - Decimal("0.1"),
                avg_fill_price=Decimal("100"), chases=0, status="RESIDUAL",
            ),
        )
        raise exc

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)
    ledger_path = tmp_path / "ledger.json"
    settings = LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(ledger_path))

    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "HALT"
    state = load_ledger(ledger_path)
    assert state.positions["AAAUSDT"] == Decimal("0.1")
    assert state.last_executed_decision_time is None



def test_persist_confirmed_fills_appends_position_snapshot_only_on_change(tmp_path) -> None:
    from decimal import Decimal
    import pandas as pd
    import src.live.runner as runner_mod
    from src.live.executor import ExecutionOutcome
    from src.live.ledger import LedgerState, PositionSnapshot, load_ledger
    from src.live.planner import OrderIntent

    t0 = pd.Timestamp("2026-08-23 01:26Z")
    now = pd.Timestamp("2026-08-24 01:26Z")
    base = LedgerState(
        positions={"AAAUSDT": Decimal("1")}, cash_usdt=Decimal("1000"),
        funding_watermarks={"AAAUSDT": pd.Timestamp("2026-08-24 00:00Z")},
        position_history=(PositionSnapshot(effective_from=t0, positions={"AAAUSDT": Decimal("1")}),),
    )
    intent = OrderIntent(
        symbol="AAAUSDT", side="BUY", quantity=Decimal("1"), reduce_only=False,
        target_qty=Decimal("2"), current_qty=Decimal("1"), client_order_prefix="run1",
        leg_index=0, decision_price=Decimal("100"),
    )
    outcome = ExecutionOutcome(symbol="AAAUSDT", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED")

    unchanged = runner_mod._persist_confirmed_fills(tmp_path / "a.json", base, [], [], Decimal("2000"), snapshot_at=now)
    filled = runner_mod._persist_confirmed_fills(tmp_path / "b.json", base, [intent], [outcome], Decimal("2000"), snapshot_at=now)

    assert unchanged.position_history == base.position_history
    assert filled.position_history == (
        PositionSnapshot(effective_from=t0, positions={"AAAUSDT": Decimal("1")}),
        PositionSnapshot(effective_from=now, positions={"AAAUSDT": Decimal("2")}),
    )
    assert load_ledger(tmp_path / "b.json").funding_watermarks == base.funding_watermarks


def test_accrue_ledger_funding_stamps_accrual_start_on_bootstrap(tmp_path, monkeypatch) -> None:
    from decimal import Decimal

    import pandas as pd

    import src.live.runner as runner_mod
    from src.live.ledger import LedgerState, load_ledger

    monkeypatch.setattr(runner_mod, "_load_paper_funding", lambda symbols: {})
    monkeypatch.setattr(runner_mod, "_load_paper_marks", lambda symbols: {})
    now = pd.Timestamp("2026-09-15 01:05Z")
    later = pd.Timestamp("2026-09-16 01:05Z")

    # Given: 레거시 원장(보유 있음, 이력/through 없음)
    legacy = LedgerState(positions={"AAAUSDT": Decimal("1")}, cash_usdt=Decimal("1000"))

    # When
    first, _ = runner_mod._accrue_ledger_funding(legacy, now, tmp_path / "ledger.json")

    # Then: 부트스트랩 시각이 accrual 시작 마커로 영속
    assert first.funding_accrual_started_at == now
    assert load_ledger(tmp_path / "ledger.json").funding_accrual_started_at == now

    # When: 이력이 생긴 뒤 재호출해도 마커 불변
    second, _ = runner_mod._accrue_ledger_funding(first, later, tmp_path / "ledger.json")
    assert second.funding_accrual_started_at == now

    # Given: funding_accrued_through 가 있는 원장은 그 시각으로 부트스트랩
    through = pd.Timestamp("2026-09-14 02:00Z")
    seeded, _ = runner_mod._accrue_ledger_funding(
        LedgerState(positions={"AAAUSDT": Decimal("1")}, cash_usdt=Decimal("1000"), funding_accrued_through=through), now, tmp_path / "seeded.json"
    )
    assert seeded.funding_accrual_started_at == through

    # Given: 보유 없음 -> 부트스트랩 없음 -> 마커 없음
    flat, _ = runner_mod._accrue_ledger_funding(LedgerState(positions={}, cash_usdt=Decimal("1000")), now, tmp_path / "flat.json")
    assert flat.funding_accrual_started_at is None


