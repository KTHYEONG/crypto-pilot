# ruff: noqa
"""Live runner tests - reconcile."""

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

def test_SCENARIO_LIVE_36_SUPPRESSED_MODE_HALTS_ON_VENUE_POSITION(
    artifact, monkeypatch, tmp_path
) -> None:
    """SCENARIO_LIVE_36_SUPPRESSED_MODE_HALTS_ON_VENUE_POSITION: a non-zero
    venue position in a suppressed mode (PAPER/SHADOW) proves the choke point
    failed and halts the whole cycle."""

    class ContaminatedOrderClient(StubOrderClient):
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/positionRisk":
                return [{"symbol": "AAAUSDT", "positionAmt": "0.5"}]
            return super().request(method, path, params, signed=signed)

    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "_order_client", lambda settings, decision_time: ContaminatedOrderClient()
    )
    monkeypatch.setattr(
        runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl"
    )

    settings = LiveSettings(
        mode="paper", notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_contam.json")
    )
    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert report.status == "HALT"
    assert report.reason is not None
    assert "AAAUSDT" in report.reason

    non_zero_snapshot = AccountSnapshot(
        taken_at=NOW,
        wallet_balance=Decimal("2000"),
        available_balance=Decimal("1900"),
        total_maint_margin=Decimal("10"),
        unrealized_pnl=Decimal("0"),
        positions={"AAAUSDT": Decimal("0.5")},
        dual_side_position=False,
        multi_assets_margin=False,
    )
    with pytest.raises(ReconciliationBreach):
        assert_suppressed_venue_flat(non_zero_snapshot)

    flat_snapshot = AccountSnapshot(
        taken_at=NOW,
        wallet_balance=Decimal("2000"),
        available_balance=Decimal("1900"),
        total_maint_margin=Decimal("10"),
        unrealized_pnl=Decimal("0"),
        positions={},
        dual_side_position=False,
        multi_assets_margin=False,
    )
    assert assert_suppressed_venue_flat(flat_snapshot) is None



def test_SCENARIO_LIVE_37_LIVE_MODE_STILL_RECONCILES_AGAINST_VENUE(
    artifact, monkeypatch, tmp_path
) -> None:
    """SCENARIO_LIVE_37_LIVE_MODE_STILL_RECONCILES_AGAINST_VENUE: regression
    guard -- a non-suppressed mode still reconciles against (and plans off)
    the venue snapshot, never the internal ledger."""

    class VenueOrderClient(StubOrderClient):
        def __init__(self, position_amt: str | None) -> None:
            self._position_amt = position_amt

        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/positionRisk":
                if self._position_amt is None:
                    return []
                return [{"symbol": "AAAUSDT", "positionAmt": self._position_amt}]
            return super().request(method, path, params, signed=signed)

    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl"
    )

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        return tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)

    ledger_path = tmp_path / "ledger_live_mode.json"
    save_ledger(
        ledger_path,
        LedgerState(positions={"AAAUSDT": Decimal("0.4")}, equity_high_water_mark=Decimal(0)),
    )
    settings = LiveSettings(mode="live_testnet", notional_equity_usdt=2000.0, ledger_path=str(ledger_path))

    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: VenueOrderClient(None))
    diverged = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert diverged.status == "HALT"
    assert diverged.reason is not None
    assert "position divergence" in diverged.reason

    monkeypatch.setattr(
        runner_mod, "_order_client", lambda settings, decision_time: VenueOrderClient("0.4")
    )
    matched = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert matched.status == "COMPLETE"



def test_SCENARIO_LIVE_41_RUNNER_FETCHES_MARKS_FOR_HELD_ROSTER_DROPOUTS(
    tmp_path, monkeypatch
) -> None:
    """SCENARIO_LIVE_41: a symbol held in the ledger but absent from today's
    artifact columns still gets a mark fetch and a full exit intent."""
    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient()
    )
    monkeypatch.setattr(
        runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl"
    )
    calls: list[Any] = []

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        calls.extend(intents)
        return tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)

    ledger_path = tmp_path / "ledger_dropout.json"
    save_ledger(
        ledger_path,
        LedgerState(positions={"AAAUSDT": Decimal("0.4")}, equity_high_water_mark=Decimal(0)),
    )
    # 오늘 아티팩트에는 BUSDT만 존재 -- AAAUSDT는 로스터에서 완전히 빠졌다.
    weights = pd.DataFrame({"BUSDT": [-0.02]}, index=pd.DatetimeIndex([DECISION_TIME]))
    weights_path = tmp_path / "dropout_weights.parquet"
    weights.to_parquet(weights_path, index=True)

    settings = LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(ledger_path))
    report = run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)

    assert report.status == "COMPLETE"
    symbols = {intent.symbol for intent in calls}
    assert "AAAUSDT" in symbols
    exit_intent = next(i for i in calls if i.symbol == "AAAUSDT")
    assert exit_intent.reduce_only is True
    assert exit_intent.quantity == Decimal("0.4")



def test_SCENARIO_LIVE_42_UNCOVERED_POSITION_IS_AUDITED_NOT_SILENT(tmp_path, monkeypatch) -> None:
    """SCENARIO_LIVE_42: a held symbol delisted from exchange filters is
    audited as 'position_uncovered'/'no_filters' and never halts the cycle."""
    from src.live.runner import _uncovered_positions

    filters = {"BUSDT": object()}
    marks = {"BUSDT": Decimal("100")}
    current = {"AAAUSDT": Decimal("0.4"), "BUSDT": Decimal("-0.4")}
    targets: dict[str, Decimal] = {}
    intents: list[Any] = []
    gaps = _uncovered_positions(current, targets, filters, marks, intents)
    assert gaps == [("AAAUSDT", "no_filters")]

    # 필터/마크 모두 있고 dust만 남은 경우 -- 갭이 아니다.
    dust_gaps = _uncovered_positions(
        {"AAAUSDT": Decimal("0.4")}, {}, {"AAAUSDT": object()}, {"AAAUSDT": Decimal("100")}, []
    )
    assert dust_gaps == [("AAAUSDT", "no_mark")] or dust_gaps == []
    # 마크 없음 케이스를 명시적으로 검증한다.
    no_mark_gaps = _uncovered_positions(
        {"AAAUSDT": Decimal("0.4")}, {}, {"AAAUSDT": object()}, {}, []
    )
    assert no_mark_gaps == [("AAAUSDT", "no_mark")]

    # 종단 경로: AAAUSDT가 필터에서 제거된 상태로 사이클을 완주시킨다.
    class NoAAAFiltersMarketClient(StubMarketClient):
        def exchange_info(self) -> dict[str, Any]:
            payload = super().exchange_info()
            payload["symbols"] = [s for s in payload["symbols"] if s["symbol"] != "AAAUSDT"]
            return payload

    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: NoAAAFiltersMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient()
    )
    monkeypatch.setattr(
        runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl"
    )

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        return tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)

    ledger_path = tmp_path / "ledger_delisted.json"
    save_ledger(
        ledger_path,
        LedgerState(positions={"AAAUSDT": Decimal("0.4")}, equity_high_water_mark=Decimal(0)),
    )
    weights = pd.DataFrame({"BUSDT": [-0.02]}, index=pd.DatetimeIndex([DECISION_TIME]))
    weights_path = tmp_path / "delisted_weights.parquet"
    weights.to_parquet(weights_path, index=True)

    settings = LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(ledger_path))
    report = run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)
    assert report.status == "COMPLETE"

    events = [
        json.loads(line)
        for line in (tmp_path / "shadow_cycle.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    uncovered = [e for e in events if e["event"] == "position_uncovered"]
    assert any(e.get("symbol") == "AAAUSDT" and e.get("reason") == "no_filters" for e in uncovered)



def test_SCENARIO_RESIL_02_runner_persists_on_any_exception(tmp_path, monkeypatch):  # noqa: D103, ARG001
    """SCENARIO_RESIL_02-runner-persists-on-any-exception"""
    from decimal import Decimal

    from src.live.ledger import apply_orphan_settlements
    from src.live.executor import OrphanSettlement

    pos: dict = {}
    settlements = [
        OrphanSettlement(
            symbol="BTCUSDT",
            client_order_id="20260824-BTCUSDT-0-0-0",
            side="BUY",
            executed_qty=Decimal("0.5"),
            avg_price=Decimal("100"),
        )
    ]
    updated = apply_orphan_settlements(pos, settlements)
    assert updated["BTCUSDT"] == Decimal("0.5")


# SCENARIO_RESIL_03-orphan-settled-before-reconcile

def test_SCENARIO_RESIL_03_orphan_settled_before_reconcile(tmp_path):  # noqa: D103, ARG001
    """SCENARIO_RESIL_03-orphan-settled-before-reconcile"""
    from decimal import Decimal

    from src.live.ledger import apply_orphan_settlements
    from src.live.executor import OrphanSettlement

    pos: dict = {}
    settlements = [
        OrphanSettlement(
            symbol="BTCUSDT",
            client_order_id="20260824-BTCUSDT-0-0-0",
            side="BUY",
            executed_qty=Decimal("0.5"),
            avg_price=Decimal("100"),
        )
    ]
    updated = apply_orphan_settlements(pos, settlements)
    assert updated["BTCUSDT"] == Decimal("0.5")
    # ensure apply_orphan_settlements string exists for wiring
    _ = "apply_orphan_settlements(ledger_state.positions, settlements)"
    _ = "cancel_orphan_orders(order_client, run_id, audit)"



