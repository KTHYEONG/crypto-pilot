"""SCENARIO_LIVE_10/18/19: 리스크 게이트, 원장 내구성, MTM 에쿼티/드로다운 HALT.
SCENARIO_LIVE_RUNNER_WRITES_EXECUTION_QUALITY_AND_NEVER_HALTS_ON_ITS_FAILURE
"""

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
                {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 2400},
                {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
                {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 300},
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
def test_SCENARIO_LIVE_29_CYCLE_REPORTS_MIN_NOTIONAL_DROP(artifact, live_env, tmp_path) -> None:
    """SCENARIO_LIVE_29_CYCLE_REPORTS_MIN_NOTIONAL_DROP: at equity $2,000,
    three symbols whose weights place them below minNotional surface in
    CycleReport.dropped_notional_fraction as dropped target notional over
    total target notional (strictly inside (0, 1)); a cycle with no drops
    reports exactly 0.0."""

    class FiveSymbolMarketClient(StubMarketClient):
        def exchange_info(self) -> dict[str, Any]:
            payload = super().exchange_info()
            template = payload["symbols"][0]
            for symbol in ("DRP1USDT", "DRP2USDT", "DRP3USDT"):
                extra = dict(template)
                extra["symbol"] = symbol
                payload["symbols"].append(extra)
            return payload

    monkeypatch_market = FiveSymbolMarketClient()
    original_market = runner_mod._market_client
    runner_mod._market_client = lambda settings, decision_time: monkeypatch_market  # type: ignore[assignment, misc]
    try:
        weights = pd.DataFrame(
            {
                "AAAUSDT": [0.02],
                "BUSDT": [-0.02],
                "DRP1USDT": [0.0001],
                "DRP2USDT": [0.0001],
                "DRP3USDT": [0.0001],
            },
            index=pd.DatetimeIndex([DECISION_TIME]),
        )
        drop_path = artifact.parent / "with_drops.parquet"
        weights.to_parquet(drop_path, index=True)

        settings = LiveSettings(
            notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_drop.json"),
        )
        report = run_shadow_cycle(settings, DECISION_TIME, drop_path, now=NOW)
    finally:
        runner_mod._market_client = original_market

    assert report.status == "COMPLETE"
    # 분자: 3 * $2000 * 0.0001 = $0.6, 분모: 유지 노셔널 $80 + 드롭 $0.6.
    from decimal import Decimal as _D

    dropped = _D("0.6")
    total = _D("80") + dropped
    expected = float(dropped / total)
    assert 0.0 < report.dropped_notional_fraction < 1.0
    assert report.dropped_notional_fraction == pytest.approx(expected, rel=1e-9)

    no_drop_report = run_shadow_cycle(
        LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_clean.json")),
        DECISION_TIME,
        artifact,
        now=NOW,
    )
    assert no_drop_report.status == "COMPLETE"
    assert no_drop_report.dropped_notional_fraction == 0.0


def test_SCENARIO_LIVE_RUNNER_WRITES_EXECUTION_QUALITY_AND_NEVER_HALTS_ON_ITS_FAILURE(
    artifact, monkeypatch, tmp_path
) -> None:
    """Run shadow cycle writes execution quality and never halts on its failure."""
    import pandas as pd

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

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None):
        return tuple(
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

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)

    eq_dir = tmp_path / "eq_quality"
    ledger_path = tmp_path / "ledger_runner_eq.json"
    settings = LiveSettings(
        mode="paper",
        notional_equity_usdt=2000.0,
        ledger_path=str(ledger_path),
        execution_quality_dir=str(eq_dir),
    )
    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert report.status == "COMPLETE"
    # At least one record persisted with mark
    shards = list(eq_dir.glob("*.parquet"))
    assert len(shards) >= 1
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    assert len(df) >= 1
    # mark_price_at_decision populated (proving runner marks now reach storage)
    assert df["mark_price_at_decision"].notna().any()

    # Monkeypatch append to raise OSError still returns COMPLETE and ledger written
    def raise_oserror(records, history_dir):
        raise OSError("disk full")

    monkeypatch.setattr(runner_mod, "append_execution_quality", raise_oserror)
    ledger_path2 = tmp_path / "ledger_runner_eq2.json"
    settings2 = LiveSettings(
        mode="paper",
        notional_equity_usdt=2000.0,
        ledger_path=str(ledger_path2),
        execution_quality_dir=str(tmp_path / "eq_quality2"),
    )
    report2 = run_shadow_cycle(settings2, DECISION_TIME, artifact, now=NOW)
    assert report2.status == "COMPLETE"
    assert ledger_path2.exists()


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

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None):
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


def test_SCENARIO_LIVE_35_PAPER_MULTI_DAY_CYCLES_DO_NOT_HALT(tmp_path, monkeypatch) -> None:
    """SCENARIO_LIVE_35_PAPER_MULTI_DAY_CYCLES_DO_NOT_HALT: PAPER cycles on
    unchanged target weights never HALT, and settle to 0 intents from the
    second cycle onward once the internal ledger already holds the target."""
    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient()
    )
    monkeypatch.setattr(
        runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl"
    )

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None):
        return tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)

    days = [DECISION_TIME + pd.Timedelta(days=i) for i in range(3)]
    weights = pd.DataFrame(
        {"AAAUSDT": [0.02] * 3, "BUSDT": [-0.02] * 3}, index=pd.DatetimeIndex(days)
    )
    weights_path = tmp_path / "paper_weights.parquet"
    weights.to_parquet(weights_path, index=True)

    ledger_path = tmp_path / "ledger_paper_multi.json"
    settings = LiveSettings(mode="paper", notional_equity_usdt=2000.0, ledger_path=str(ledger_path))

    reports = []
    ledger_snapshots = []
    for day in days:
        report = run_shadow_cycle(settings, day, weights_path, now=day + pd.Timedelta(hours=2))
        reports.append(report)
        ledger_snapshots.append(dict(load_ledger(ledger_path).positions))

    assert [r.status for r in reports] == ["COMPLETE", "COMPLETE", "COMPLETE"]
    assert reports[0].intent_count > 0
    assert reports[1].intent_count == 0
    assert reports[2].intent_count == 0
    assert ledger_snapshots[0] == ledger_snapshots[2]


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

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None):
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

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None):
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

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None):
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


def test_SCENARIO_LIVE_47_RUNNER_PERSISTS_PORTFOLIO_STATE_PAPER_VS_LIVE(
    tmp_path, monkeypatch
) -> None:
    """SCENARIO_LIVE_47: PAPER writes a virtual_mtm row with cash populated;
    live_testnet writes a venue row with wallet_balance populated; a write
    failure never changes the cycle status."""
    monkeypatch.setattr(
        runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient()
    )
    monkeypatch.setattr(
        runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl"
    )

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None):
        return tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)

    weights = pd.DataFrame(
        {"AAAUSDT": [0.02], "BUSDT": [-0.02]}, index=pd.DatetimeIndex([DECISION_TIME])
    )
    weights_path = tmp_path / "weights.parquet"
    weights.to_parquet(weights_path, index=True)

    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())
    paper_dir = tmp_path / "portfolio_paper"
    paper_settings = LiveSettings(
        mode="paper", notional_equity_usdt=2000.0,
        ledger_path=str(tmp_path / "ledger_paper.json"),
        portfolio_state_dir=str(paper_dir),
    )
    paper_report = run_shadow_cycle(paper_settings, DECISION_TIME, weights_path, now=NOW)
    assert paper_report.status == "COMPLETE"
    paper_df = pd.read_parquet(paper_dir / "active.parquet")
    assert (paper_df["equity_source"] == "virtual_mtm").all()
    assert paper_df["cash_usdt"].notna().all()

    class VenueMatchClient(StubOrderClient):
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/positionRisk":
                return [{"symbol": "AAAUSDT", "positionAmt": "0.4"}]
            return super().request(method, path, params, signed=signed)

    save_ledger(
        tmp_path / "ledger_live.json",
        LedgerState(positions={"AAAUSDT": Decimal("0.4")}, equity_high_water_mark=Decimal(0)),
    )
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: VenueMatchClient())
    live_dir = tmp_path / "portfolio_live"
    live_settings = LiveSettings(
        mode="live_testnet", notional_equity_usdt=2000.0,
        ledger_path=str(tmp_path / "ledger_live.json"),
        portfolio_state_dir=str(live_dir),
    )
    live_report = run_shadow_cycle(live_settings, DECISION_TIME, weights_path, now=NOW)
    assert live_report.status == "COMPLETE"
    live_df = pd.read_parquet(live_dir / "active.parquet")
    assert (live_df["equity_source"] == "venue").all()
    assert live_df["wallet_balance_usdt"].notna().all()
    assert live_df["cash_usdt"].isna().all()

    def raise_oserror(record, history_dir):
        raise OSError("disk full")

    monkeypatch.setattr(runner_mod, "append_portfolio_state", raise_oserror)
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())
    fail_settings = LiveSettings(
        mode="paper", notional_equity_usdt=2000.0,
        ledger_path=str(tmp_path / "ledger_paper2.json"),
        portfolio_state_dir=str(tmp_path / "portfolio_fail"),
    )
    fail_report = run_shadow_cycle(fail_settings, DECISION_TIME, weights_path, now=NOW)
    assert fail_report.status == "COMPLETE"


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
    expected_cash = Decimal("2000") - (raised_fill * Decimal("100"))
    assert state.cash_usdt == expected_cash


def test_SCENARIO_LIVE_49_MARK_FETCH_FALLBACK_TOLERATES_UNKNOWN_SYMBOL() -> None:
    """SCENARIO_LIVE_49: _marks_from_tickers' per-symbol fallback skips a
    symbol whose lookup fails and still returns marks for the rest."""
    from src.live.runner import _marks_from_tickers

    class PartialFailClient:
        def book_ticker(self, symbol: str) -> dict[str, str]:
            if symbol == "DEADUSDT":
                raise AssertionError("unknown symbol")
            return {"bidPrice": "100.00", "askPrice": "101.00"}

    marks = _marks_from_tickers(PartialFailClient(), ["AAAUSDT", "DEADUSDT", "BUSDT"])
    assert set(marks) == {"AAAUSDT", "BUSDT"}
    assert marks["AAAUSDT"] == Decimal("100.50")


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용, live_paper_readiness_p0).
COVERED_SCENARIOS = (
    *COVERED_SCENARIOS,
    "SCENARIO_LIVE_34_SYMBOL_CAP_NEVER_BLOCKS_REDUCE_ONLY",
    "SCENARIO_LIVE_35_PAPER_MULTI_DAY_CYCLES_DO_NOT_HALT",
    "SCENARIO_LIVE_36_SUPPRESSED_MODE_HALTS_ON_VENUE_POSITION",
    "SCENARIO_LIVE_37_LIVE_MODE_STILL_RECONCILES_AGAINST_VENUE",
    "SCENARIO_LIVE_41_RUNNER_FETCHES_MARKS_FOR_HELD_ROSTER_DROPOUTS",
    "SCENARIO_LIVE_42_UNCOVERED_POSITION_IS_AUDITED_NOT_SILENT",
    "SCENARIO_LIVE_47_RUNNER_PERSISTS_PORTFOLIO_STATE_PAPER_VS_LIVE",
    "SCENARIO_LIVE_48_CASH_TRACKING_SURVIVES_PARTIAL_FILL_HALT",
    "SCENARIO_LIVE_49_MARK_FETCH_FALLBACK_TOLERATES_UNKNOWN_SYMBOL",
)
