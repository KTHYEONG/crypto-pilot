"""SCENARIO_LIVE_10: 리스크 게이트는 사이클 전체를 차단한다(부분 집행 금지)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd
import pytest

import src.live.runner as runner_mod
from src.live.account import AccountSnapshot
from src.live.errors import RiskGateBreach
from src.live.executor import ExecutionOutcome
from src.live.runner import check_risk_gates, run_shadow_cycle
from src.live.settings import LiveSettings


DECISION_TIME = pd.Timestamp("2026-08-24 00:00Z")
NOW = DECISION_TIME + pd.Timedelta(hours=2)


class StubMarketClient:
    def exchange_info(self) -> dict[str, Any]:
        def symbol_entry(symbol: str) -> dict[str, Any]:
            return {
                "symbol": symbol,
                "contractType": "PERPETUAL",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "quantityPrecision": 3,
                "pricePrecision": 2,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "100000"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "1"},
                ],
            }

        return {"symbols": [symbol_entry("AAAUSDT"), symbol_entry("BUSDT")]}

    def book_ticker(self, symbol: str) -> dict[str, str]:
        return {"bidPrice": "100.00", "askPrice": "101.00"}


class StubOrderClient:
    def request(self, method: str, path: str, params=None, *, signed=False) -> Any:
        if path == "/fapi/v2/account":
            return {
                "totalWalletBalance": "2000",
                "availableBalance": "1900",
                "totalInitialMargin": "10",
                "dualSidePosition": "false",
                "multiAssetsMargin": "false",
            }
        if path == "/fapi/v2/positionRisk":
            return []
        raise AssertionError(f"unexpected path {path}")


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
    monkeypatch.setattr(runner_mod, "_market_client", lambda settings: StubMarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings: StubOrderClient())
    calls: list[Any] = []

    def fake_execute(client, intent, filters, policy, audit, clock) -> ExecutionOutcome:
        calls.append(intent)
        return ExecutionOutcome(
            symbol=intent.symbol,
            filled_qty=intent.quantity,
            unfilled_qty=Decimal("0"),
            avg_fill_price=Decimal("100"),
            chases=0,
            status="FILLED",
        )

    monkeypatch.setattr(runner_mod, "execute_intent", fake_execute)
    monkeypatch.setattr(
        runner_mod,
        "default_audit_log_path",
        lambda name: tmp_path / f"{name}.jsonl",
    )
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
                    "dualSidePosition": "false",
                    "multiAssetsMargin": "false",
                }
            return []

    margin_settings = LiveSettings(
        notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_margin.json"),
    )
    original_order_client = runner_mod._order_client
    runner_mod._order_client = lambda settings: ThinMarginClient()  # type: ignore[assignment]
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
        positions={},
        dual_side_position=False,
        multi_assets_margin=False,
    )
    marks = {"AAAUSDT": Decimal("100")}
    targets = {"AAAUSDT": Decimal("70")}  # 7000/2000 = 3.5x
    intents = []
    settings = LiveSettings(notional_equity_usdt=2000.0, max_gross_leverage=3.0)
    with pytest.raises(RiskGateBreach):
        check_risk_gates(intents, targets, marks, snapshot, settings)

#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_10_RISK_GATE_BLOCKS_WHOLE_CYCLE",
)
