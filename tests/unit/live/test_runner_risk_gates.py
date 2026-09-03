# ruff: noqa
"""Live runner tests - risk_gates."""

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

def test_SCENARIO_LIVE_10_risk_gate_blocks_whole_cycle(artifact, live_env, tmp_path) -> None:
    # 각 사이클 호출은 독립적인 원장 경로를 쓴다: 전역 default_ledger_path()를 쓰면
    # 이 테스트의 첫 성공 사이클이 남긴 체결이 이후 독립 게이트 점검의 재조정을
    # 깨뜨린다(이 테스트는 사이클 간 연속성이 아니라 각 게이트를 독립 검증한다).
    settings = LiveSettings(
        notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_ok.json"),
    )
    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert report.status == "COMPLETE"
    assert len(live_env) == report.intent_count == 2

    # gross leverage 위반: sum(|target_qty * mark|)/equity > ceiling.
    leveraged_weights = pd.DataFrame(
        {"AAAUSDT": [7.0], "BUSDT": [-7.0]},
        index=pd.DatetimeIndex([DECISION_TIME]),
    )
    leveraged_path = artifact.parent / "leveraged.parquet"
    leveraged_weights.to_parquet(leveraged_path, index=True)

    leveraged_settings = LiveSettings(
        notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_leverage.json"),
    )
    execute_calls_before = len(live_env)
    halted = run_shadow_cycle(leveraged_settings, DECISION_TIME, leveraged_path, now=NOW)
    assert halted.status == "HALT"
    assert halted.reason is not None
    assert "leverage" in halted.reason.lower()
    assert len(live_env) == execute_calls_before  # 부분 집행 금지

    # max_daily_orders 초과도 전체 HALT다.
    tight = LiveSettings(
        notional_equity_usdt=2000.0, max_daily_orders=1,
        ledger_path=str(tmp_path / "ledger_tight.json"),
    )
    assert run_shadow_cycle(tight, DECISION_TIME, artifact, now=NOW).status == "HALT"

    # min_free_margin_fraction 미달도 전체 HALT다.
    class ThinMarginClient(StubOrderClient):
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/account":
                return {
                    "totalWalletBalance": "2000",
                    "availableBalance": "100",
                    "totalInitialMargin": "10",
                    "totalUnrealizedProfit": "0",
                    "dualSidePosition": "false",
                    "multiAssetsMargin": "false",
                }
            return []

    margin_settings = LiveSettings(
        notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_margin.json"),
    )
    original_order_client = runner_mod._order_client
    runner_mod._order_client = lambda settings, decision_time: ThinMarginClient()  # type: ignore[assignment, misc]
    try:
        halted_margin = run_shadow_cycle(margin_settings, DECISION_TIME, artifact, now=NOW)
    finally:
        runner_mod._order_client = original_order_client
    assert halted_margin.status == "HALT"



def test_check_risk_gates_raises_directly() -> None:
    snapshot = AccountSnapshot(
        taken_at=NOW,
        wallet_balance=Decimal("2000"),
        available_balance=Decimal("1900"),
        total_maint_margin=Decimal("10"),
        unrealized_pnl=Decimal("0"),
        positions={},
        dual_side_position=False,
        multi_assets_margin=False,
    )
    marks = {"AAAUSDT": Decimal("100")}
    targets = {"AAAUSDT": Decimal("70")}  # 7000/2000 = 3.5x
    intents = []
    settings = LiveSettings(notional_equity_usdt=2000.0, max_gross_leverage=3.0)
    with pytest.raises(RiskGateBreach):
        check_risk_gates(intents, targets, marks, snapshot, settings, Decimal("2000"))



def test_SCENARIO_LIVE_19_mtm_equity_cap_and_drawdown_halt(
    artifact, monkeypatch, tmp_path
) -> None:
    """I-EQUITY-MTM / I-DD-HALT."""
    capped_snapshot = AccountSnapshot(
        taken_at=NOW,
        wallet_balance=Decimal("3000"),
        available_balance=Decimal("2500"),
        total_maint_margin=Decimal("10"),
        unrealized_pnl=Decimal("500"),
        positions={},
        dual_side_position=False,
        multi_assets_margin=False,
    )
    assert resolve_sizing_equity(capped_snapshot, Decimal("2000")) == Decimal("2000")

    mtm_snapshot = AccountSnapshot(
        taken_at=NOW,
        wallet_balance=Decimal("1000"),
        available_balance=Decimal("900"),
        total_maint_margin=Decimal("10"),
        unrealized_pnl=Decimal("-100"),
        positions={},
        dual_side_position=False,
        multi_assets_margin=False,
    )
    assert resolve_sizing_equity(mtm_snapshot, Decimal("2000")) == Decimal("900")

    with pytest.raises(RiskGateBreach):
        assert_drawdown_within_limit(Decimal("1000"), Decimal("2000"), -0.45)
    # 초기 사이클(hwm<=0)과 정상 범위는 통과한다.
    assert_drawdown_within_limit(Decimal("1000"), Decimal("0"), -0.45)
    assert_drawdown_within_limit(Decimal("1900"), Decimal("2000"), -0.45)

    # 드로다운 게이트는 run_shadow_cycle 에서 intent_count==0 HALT 로 이어진다.
    class MtLossClient(StubOrderClient):
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/account":
                return {
                    "totalWalletBalance": "1000",
                    "availableBalance": "900",
                    "totalInitialMargin": "10",
                    "totalUnrealizedProfit": "-100",
                    "dualSidePosition": "false",
                    "multiAssetsMargin": "false",
                }
            return []

    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "_order_client", lambda settings, decision_time: MtLossClient()
    )
    monkeypatch.setattr(
        runner_mod,
        "default_audit_log_path",
        lambda name, for_date=None: tmp_path / f"{name}.jsonl",
    )

    ledger_path = tmp_path / "ledger_dd.json"
    save_ledger(ledger_path, LedgerState(positions={}, equity_high_water_mark=Decimal("2000")))
    dd_settings = LiveSettings(
        mode="live_testnet",
        notional_equity_usdt=2000.0,
        equity_drawdown_halt=-0.45,
        ledger_path=str(ledger_path),
    )
    halted = run_shadow_cycle(dd_settings, DECISION_TIME, artifact, now=NOW)
    assert halted.status == "HALT"
    assert halted.intent_count == 0



def test_SCENARIO_LIVE_34_SYMBOL_CAP_NEVER_BLOCKS_REDUCE_ONLY(tmp_path, monkeypatch) -> None:
    """SCENARIO_LIVE_34_SYMBOL_CAP_NEVER_BLOCKS_REDUCE_ONLY: a reduce_only exit
    intent survives the per-symbol notional cap while an equally-sized entry
    intent of the same notional is dropped and audited."""

    class CapMarketClient(StubMarketClient):
        def exchange_info(self) -> dict[str, Any]:
            payload = super().exchange_info()
            template = payload["symbols"][0]
            extra = dict(template)
            extra["symbol"] = "BBBUSDT"
            payload["symbols"].append(extra)
            return payload

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: CapMarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())
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

    ledger_path = tmp_path / "ledger_cap.json"
    save_ledger(
        ledger_path,
        LedgerState(positions={"AAAUSDT": Decimal("1.5")}, equity_high_water_mark=Decimal(0)),
    )

    # AAAUSDT: target 0.0 (a symbol_dropped MIN_QTY case) with a pre-existing
    # 1.5-unit position -> a $150+ reduce_only exit. BBBUSDT: an equal-size
    # entry that the 5% cap should drop.
    weights = pd.DataFrame(
        {"AAAUSDT": [0.0], "BBBUSDT": [0.075]}, index=pd.DatetimeIndex([DECISION_TIME])
    )
    weights_path = tmp_path / "cap_weights.parquet"
    weights.to_parquet(weights_path, index=True)

    settings = LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(ledger_path))
    report = run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)

    assert report.status == "COMPLETE"
    kept_symbols = {intent.symbol for intent in calls}
    assert "AAAUSDT" in kept_symbols  # reduce_only exit survives despite exceeding the cap
    assert "BBBUSDT" not in kept_symbols  # entry of equal notional is dropped by the cap

    exit_intent = next(intent for intent in calls if intent.symbol == "AAAUSDT")
    assert exit_intent.reduce_only is True

    audit_events = [
        json.loads(line)
        for line in (tmp_path / "shadow_cycle.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        e["event"] == "intent_dropped_symbol_cap" and e.get("symbol") == "BBBUSDT"
        for e in audit_events
    )


