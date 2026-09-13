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



def test_apply_ruin_guard_flattens_at_backtest_equity_floor() -> None:
    from decimal import Decimal
    import pandas as pd
    import pytest
    from src.live.runner import apply_ruin_guard
    from src.mhs.params import REFERENCE_PASS_EQUITY_FLOOR

    assert REFERENCE_PASS_EQUITY_FLOOR == 0.5
    weights = pd.Series({"AAAUSDT": 0.2, "BUSDT": -0.1}, name=pd.Timestamp("2026-09-01", tz="UTC"))
    flat, breached = apply_ruin_guard(weights, Decimal("1000"), Decimal("2000"))
    assert breached is True
    assert (flat == 0.0).all() and list(flat.index) == list(weights.index) and flat.name == weights.name
    kept, breached_above = apply_ruin_guard(weights, Decimal("1000.01"), Decimal("2000"))
    assert breached_above is False
    pd.testing.assert_series_equal(kept, weights)
    with pytest.raises(ValueError, match="starting_capital"):
        apply_ruin_guard(weights, Decimal("1000"), Decimal("0"))


def test_run_shadow_cycle_ruin_guard_flattens_and_no_symbol_cap(tmp_path, monkeypatch) -> None:
    from decimal import Decimal
    import pandas as pd
    import src.live.runner as runner_mod
    from src.live.executor import ExecutionOutcome
    from src.live.ledger import LedgerState, save_ledger
    from src.live.settings import ExecutionMode, LiveSettings

    class MarketClient:
        def exchange_info(self):
            return {
                "symbols": [{
                    "symbol": "AAAUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT", "status": "TRADING",
                    "quantityPrecision": 3, "pricePrecision": 2,
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "100000"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "1"},
                    ],
                }],
                "rateLimits": [
                    {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 2400},
                    {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
                    {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 300},
                ],
            }

        def book_ticker(self, symbol):
            return {"bidPrice": "100.00", "askPrice": "102.00"}

        def book_tickers(self):
            return {"AAAUSDT": {"bidPrice": "100.00", "askPrice": "102.00", "symbol": "AAAUSDT"}}

        def premium_index(self):
            return {}

    class OrderClient:
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/account":
                return {"totalWalletBalance": "2000", "availableBalance": "1900", "totalInitialMargin": "10", "totalUnrealizedProfit": "0", "dualSidePosition": "false", "multiAssetsMargin": "false"}
            if path == "/fapi/v2/positionRisk":
                return []
            raise AssertionError(path)

        def sync_server_time(self):
            return None

        def open_orders(self):
            return []

    decision_time = pd.Timestamp("2026-08-24 00:00Z")
    now = decision_time + pd.Timedelta(hours=2)
    calls: list[object] = []

    def fake_execute(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        calls.extend(intents)
        return tuple(ExecutionOutcome(symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"), avg_fill_price=Decimal("101"), chases=0, status="FILLED") for i in intents)

    monkeypatch.setattr(runner_mod, "_market_client", lambda s, dt: MarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda s, dt: OrderClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")
    monkeypatch.setattr(runner_mod, "_load_paper_funding", lambda symbols: {s: pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC")) for s in symbols})
    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute)

    def settings_for(name):
        return LiveSettings(mode=ExecutionMode.PAPER, notional_equity_usdt=2000.0, ledger_path=str(tmp_path / f"{name}.json"), fills_dir=str(tmp_path / f"{name}_fills"), orderbook_capture_enabled=False, microstructure_dir=str(tmp_path / f"{name}_micro"), execution_quality_dir=str(tmp_path / f"{name}_eq"), portfolio_state_dir=str(tmp_path / f"{name}_port"), tax_ledger_dir=str(tmp_path / f"{name}_tax"))

    ruin_weights = tmp_path / "ruin.parquet"
    pd.DataFrame({"AAAUSDT": [0.02]}, index=pd.DatetimeIndex([decision_time])).to_parquet(ruin_weights, index=True)
    save_ledger(tmp_path / "ruin.json", LedgerState(positions={"AAAUSDT": Decimal("1")}, equity_high_water_mark=Decimal("2000"), cash_usdt=Decimal("800"), funding_accrued_through=now - pd.Timedelta(minutes=1)))
    ruin_report = runner_mod.run_shadow_cycle(settings_for("ruin"), decision_time, ruin_weights, now=now)
    assert ruin_report.status == "COMPLETE"
    assert len(calls) == 1
    assert calls[0].side == "SELL" and calls[0].reduce_only is True and calls[0].quantity == Decimal("1")

    calls.clear()
    big_weights = tmp_path / "big.parquet"
    pd.DataFrame({"AAAUSDT": [0.20]}, index=pd.DatetimeIndex([decision_time])).to_parquet(big_weights, index=True)
    big_report = runner_mod.run_shadow_cycle(settings_for("big"), decision_time, big_weights, now=now)
    assert big_report.status == "COMPLETE"
    assert len(calls) == 1
    assert calls[0].side == "BUY" and calls[0].reduce_only is False and calls[0].quantity > Decimal("3.9")
