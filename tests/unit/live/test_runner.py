"""SCENARIO_LIVE_10/18/19: 리스크 게이트, 원장 내구성, MTM 에쿼티/드로다운 HALT."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import src.live.runner as runner_mod
from src.live.account import AccountSnapshot, assert_drawdown_within_limit, resolve_sizing_equity
from src.live.errors import RiskGateBreach, VenueError
from src.live.executor import ExecutionOutcome
from src.live.ledger import LedgerState, load_ledger, save_ledger
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

        return {
            "symbols": [symbol_entry("AAAUSDT"), symbol_entry("BUSDT")],
            "rateLimits": [
                {"filterType": "REQUEST_WEIGHT", "interval": "1m", "limit": 2400},
                {"filterType": "ORDERS", "interval": "1m", "limit": 300},
                {"filterType": "ORDERS", "interval": "10s", "limit": 1200},
            ],
        }

    def book_ticker(self, symbol: str) -> dict[str, str]:
        return {"bidPrice": "100.00", "askPrice": "101.00"}


class StubOrderClient:
    def request(self, method: str, path: str, params=None, *, signed=False) -> Any:
        if path == "/fapi/v2/account":
            return {
                "totalWalletBalance": "2000",
                "availableBalance": "1900",
                "totalInitialMargin": "10",
                "totalUnrealizedProfit": "0",
                "dualSidePosition": "false",
                "multiAssetsMargin": "false",
            }
        if path == "/fapi/v2/positionRisk":
            return []
        raise AssertionError(f"unexpected path {path}")

    def sync_server_time(self) -> None:
        return None

    def open_orders(self) -> list[dict[str, Any]]:
        return []


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

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None):
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

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None):
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

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None):
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
        notional_equity_usdt=2000.0,
        equity_drawdown_halt=-0.45,
        ledger_path=str(ledger_path),
    )
    halted = run_shadow_cycle(dd_settings, DECISION_TIME, artifact, now=NOW)
    assert halted.status == "HALT"
    assert halted.intent_count == 0


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_10_RISK_GATE_BLOCKS_WHOLE_CYCLE",
    "SCENARIO_LIVE_18",  # LEDGER_DURABILITY_ON_EXECUTION_FAILURE
    "SCENARIO_LIVE_19",  # MTM_EQUITY_CAP_AND_DRAWDOWN_HALT
    "SCENARIO_LIVE_DAEMON_08_RUNNER_AUDIT_KEYED_BY_DECISION_DATE",
    # SCENARIO_LIVE_DAEMON_10_EXISTING_SHADOW_CYCLE_TESTS_UPDATED:
    # 본 파일 포함 tests/unit/live 전체가 새 _market_client/_order_client
    # 시그니처(settings, decision_time)로 갱신되어 0 failed로 통과한다.
)
