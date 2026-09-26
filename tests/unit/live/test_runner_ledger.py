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


def _journal_fake_fills(journal, attempt, intents, outcomes, filled_at):
    """Mirror fake execution outcomes into the order journal (spec-03 durability).

    Fakes returning bare ExecutionOutcomes persist nothing through the runner's
    journal-backed commit path, so every fake journals its fills before
    returning or raising: fills from each outcome's own ``fills`` tuples when
    present (preserving qty/price/fee/reason/liquidity), else one taker fill
    per outcome at its average price.
    """
    if journal is None or attempt is None:
        return
    sides = {intent.symbol: intent.side for intent in intents}
    for outcome in outcomes:
        if outcome.filled_qty <= 0:
            continue
        side = sides.get(outcome.symbol, "BUY")
        if outcome.fills:
            for qty, price, fee_bps, reason, liquidity, ts in outcome.fills:
                journal.record_fill(
                    kind="execution",
                    attempt_seq=attempt.attempt_seq,
                    symbol=outcome.symbol,
                    side=side,
                    quantity=qty,
                    price=price,
                    fee_bps=fee_bps,
                    liquidity=liquidity,
                    reason=reason,
                    filled_at=ts,
                    client_order_id=None,
                    leg_index=0,
                    cumulative_executed_qty=None,
                    simulated=True,
                )
        else:
            journal.record_fill(
                kind="execution",
                attempt_seq=attempt.attempt_seq,
                symbol=outcome.symbol,
                side=side,
                quantity=outcome.filled_qty,
                price=outcome.avg_fill_price if outcome.avg_fill_price else Decimal("100"),
                fee_bps=5.0,
                liquidity="taker",
                reason="timeout_taker",
                filled_at=filled_at,
                client_order_id=None,
                leg_index=0,
                cumulative_executed_qty=None,
                simulated=True,
            )

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
    closes = pd.DataFrame(
        {"AAAUSDT": [100.0], "BUSDT": [100.0]},
        index=pd.DatetimeIndex([DECISION_TIME]),
    )
    closes.to_parquet(tmp_path / "deployed_decision_ohlcv_close.parquet", index=True)
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
        _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, (outcome,), NOW)
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
    settings = LiveSettings(
        notional_equity_usdt=2000.0,
        ledger_path=str(ledger_path),
        order_journal_path=str(tmp_path / "order_journal.jsonl"),
        fills_dir=str(tmp_path / "fills"),
        tax_ledger_dir=str(tmp_path / "tax"),
        execution_quality_dir=str(tmp_path / "eq"),
        portfolio_state_dir=str(tmp_path / "portfolio"),
    )
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
        _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, (outcome,), NOW)
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
    weights_path = tmp_path / "deployed_target_weights.parquet"
    weights.to_parquet(weights_path, index=True)
    pd.DataFrame(
        {"AAAUSDT": [100.0], "BUSDT": [100.0]}, index=pd.DatetimeIndex([DECISION_TIME])
    ).to_parquet(tmp_path / "deployed_decision_ohlcv_close.parquet", index=True)

    ledger_path = tmp_path / "ledger_cash_halt.json"
    settings = LiveSettings(
        mode="paper",
        notional_equity_usdt=2000.0,
        ledger_path=str(ledger_path),
        order_journal_path=str(tmp_path / "order_journal.jsonl"),
        fills_dir=str(tmp_path / "fills"),
        tax_ledger_dir=str(tmp_path / "tax"),
        execution_quality_dir=str(tmp_path / "eq"),
        portfolio_state_dir=str(tmp_path / "portfolio"),
    )
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


def test_accrue_ledger_funding_preserves_last_executed(tmp_path, monkeypatch) -> None:
    from decimal import Decimal
    import pandas as pd
    import src.live.runner as runner_mod
    from src.live.ledger import LedgerState, PositionSnapshot, load_ledger

    executed = pd.Timestamp("2026-08-23 00:00Z")
    now = pd.Timestamp("2026-08-24 12:00Z")
    epoch = pd.Timestamp("2026-08-24 08:00Z")
    seed, _, _ = runner_mod._accrue_ledger_funding(
        LedgerState(positions={}, last_executed_decision_time=executed), now, tmp_path / "seed.json",
    )
    monkeypatch.setattr(runner_mod, "_load_paper_funding", lambda symbols: {"AAAUSDT": pd.Series([0.001], index=pd.DatetimeIndex([epoch]))})
    monkeypatch.setattr(runner_mod, "_load_paper_trade_closes", lambda symbols: {"AAAUSDT": pd.Series([100.0], index=pd.DatetimeIndex([epoch]))})
    held, accrual, _ = runner_mod._accrue_ledger_funding(
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
        _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, exc.partial_outcomes, NOW)
        raise exc

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)
    ledger_path = tmp_path / "ledger.json"
    settings = LiveSettings(
        notional_equity_usdt=2000.0,
        ledger_path=str(ledger_path),
        order_journal_path=str(tmp_path / "order_journal.jsonl"),
        fills_dir=str(tmp_path / "fills"),
        tax_ledger_dir=str(tmp_path / "tax"),
        execution_quality_dir=str(tmp_path / "eq"),
        portfolio_state_dir=str(tmp_path / "portfolio"),
    )

    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "HALT"
    state = load_ledger(ledger_path)
    assert state.positions["AAAUSDT"] == Decimal("0.1")
    assert state.last_executed_decision_time is None



def test_accrue_ledger_funding_stamps_accrual_start_on_bootstrap(tmp_path, monkeypatch) -> None:
    from decimal import Decimal

    import pandas as pd

    import src.live.runner as runner_mod
    from src.live.ledger import LedgerState, load_ledger

    monkeypatch.setattr(runner_mod, "_load_paper_funding", lambda symbols: {})
    monkeypatch.setattr(runner_mod, "_load_paper_trade_closes", lambda symbols: {})
    now = pd.Timestamp("2026-09-15 01:05Z")
    later = pd.Timestamp("2026-09-16 01:05Z")

    # Given: 레거시 원장(보유 있음, 이력/through 없음)
    legacy = LedgerState(positions={"AAAUSDT": Decimal("1")}, cash_usdt=Decimal("1000"))

    # When
    first, _, _ = runner_mod._accrue_ledger_funding(legacy, now, tmp_path / "ledger.json")

    # Then: 부트스트랩 시각이 accrual 시작 마커로 영속
    assert first.funding_accrual_started_at == now
    assert load_ledger(tmp_path / "ledger.json").funding_accrual_started_at == now

    # When: 이력이 생긴 뒤 재호출해도 마커 불변
    second, _, _ = runner_mod._accrue_ledger_funding(first, later, tmp_path / "ledger.json")
    assert second.funding_accrual_started_at == now

    # Given: funding_accrued_through 가 있는 원장은 그 시각으로 부트스트랩
    through = pd.Timestamp("2026-09-14 02:00Z")
    seeded, _, _ = runner_mod._accrue_ledger_funding(
        LedgerState(positions={"AAAUSDT": Decimal("1")}, cash_usdt=Decimal("1000"), funding_accrued_through=through), now, tmp_path / "seeded.json"
    )
    assert seeded.funding_accrual_started_at == through

    # Given: 보유 없음 -> 부트스트랩 없음 -> 마커 없음
    flat, _, _ = runner_mod._accrue_ledger_funding(LedgerState(positions={}, cash_usdt=Decimal("1000")), now, tmp_path / "flat.json")
    assert flat.funding_accrual_started_at is None


def _paper_cycle_harness(tmp_path, monkeypatch):
    """Paper cycle with seeded cash and two taker fills; returns (settings, paths)."""
    import pandas as pd

    import src.live.runner as runner_mod
    from decimal import Decimal
    from src.live.executor import ExecutionOutcome
    from src.live.ledger import LedgerState, save_ledger
    from src.live.settings import LiveSettings
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())
    monkeypatch.setattr(
        runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl"
    )

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        outcomes = []
        for intent in intents:
            outcomes.append(
                ExecutionOutcome(
                    symbol=intent.symbol,
                    filled_qty=intent.quantity,
                    unfilled_qty=Decimal("0"),
                    avg_fill_price=Decimal("100"),
                    chases=0,
                    status="FILLED",
                    fills=((intent.quantity, Decimal("100"), 5.0, "immediate_taker", "taker", NOW),),
                )
            )
        _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, outcomes, NOW)
        audit.record("intents_executed", count=len(outcomes))
        return outcomes

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)
    weights = pd.DataFrame(
        {"AAAUSDT": [0.02], "BUSDT": [-0.02]}, index=pd.DatetimeIndex([DECISION_TIME])
    )
    weights_path = tmp_path / "deployed_target_weights.parquet"
    weights.to_parquet(weights_path, index=True)
    closes = pd.DataFrame(
        {"AAAUSDT": [100.0], "BUSDT": [100.0]}, index=pd.DatetimeIndex([DECISION_TIME])
    )
    closes.to_parquet(tmp_path / "deployed_decision_ohlcv_close.parquet", index=True)
    ledger_path = tmp_path / "ledger.json"
    save_ledger(
        ledger_path,
        LedgerState(positions={}, equity_high_water_mark=Decimal("2000"), cash_usdt=Decimal("2000")),
    )
    settings = LiveSettings(
        mode="paper",
        notional_equity_usdt=2000.0,
        ledger_path=str(ledger_path),
        order_journal_path=str(tmp_path / "order_journal.jsonl"),
        tax_ledger_dir=str(tmp_path / "tax"),
        fills_dir=str(tmp_path / "fills"),
        execution_quality_dir=str(tmp_path / "eq"),
        portfolio_state_dir=str(tmp_path / "portfolio"),
        microstructure_dir=str(tmp_path / "micro"),
    )
    return runner_mod, settings, weights_path


def _audit_events(tmp_path, event: str) -> list[dict]:
    import json

    path = tmp_path / "shadow_cycle.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("event") == event
    ]


def test_funding_records_persist_before_ledger_save(tmp_path, monkeypatch) -> None:
    """save_ledger crashes after the tax append; a retry writes each FUNDING_FEE id exactly once."""
    from decimal import Decimal

    import pandas as pd

    import src.live.runner as runner_mod
    from src.live.ledger import LedgerState

    epoch = pd.Timestamp("2026-08-24 08:00Z")
    monkeypatch.setattr(
        runner_mod, "_load_paper_funding",
        lambda symbols: {"AAAUSDT": pd.Series([0.001], index=pd.DatetimeIndex([epoch]))},
    )
    monkeypatch.setattr(
        runner_mod, "_load_paper_trade_closes",
        lambda symbols: {"AAAUSDT": pd.Series([100.0], index=pd.DatetimeIndex([epoch]))},
    )
    tax_dir = tmp_path / "tax"
    state = LedgerState(
        positions={"AAAUSDT": Decimal("1")},
        equity_high_water_mark=Decimal("2000"),
        cash_usdt=Decimal("1000"),
        funding_accrued_through=pd.Timestamp("2026-08-24 00:00Z"),
    )
    now = pd.Timestamp("2026-08-24 12:00Z")
    real_save = runner_mod.save_ledger

    def _raise_once(path, next_state):
        raise RuntimeError("crash after tax append")

    monkeypatch.setattr(runner_mod, "save_ledger", _raise_once)
    import pytest

    with pytest.raises(RuntimeError, match="crash after tax append"):
        runner_mod._accrue_ledger_funding(
            state, now, tmp_path / "ledger.json", tax_dir=tax_dir, run_id="run1", mode="paper"
        )
    monkeypatch.setattr(runner_mod, "save_ledger", real_save)
    _, _, funding_records = runner_mod._accrue_ledger_funding(
        state, now, tmp_path / "ledger.json", tax_dir=tax_dir, run_id="run1", mode="paper"
    )
    assert len(funding_records) == 1
    shard = tax_dir / "tax_ledger_202608.jsonl"
    lines = [line for line in shard.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    import json

    assert json.loads(lines[0])["record_id"] == funding_records[0].record_id


def test_paper_cycle_trade_tax_rows_equal_fills_rows(tmp_path, monkeypatch) -> None:
    """Paper TRADE tax records correspond one-to-one to the written fills; no globals fallback."""
    import inspect

    import src.live.runner as runner_mod
    from src.live.fills import load_fills
    from src.live.tax_ledger import load_tax_records
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW

    runner_mod, settings, weights_path = _paper_cycle_harness(tmp_path, monkeypatch)
    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)
    assert report.status == "COMPLETE"
    fills = load_fills(tmp_path / "fills")
    assert len(fills) == 2
    tax = load_tax_records(tmp_path / "tax", year=2026)
    trades = tax[tax["kind"] == "TRADE"]
    assert len(trades) == len(fills)
    for _, fill in fills.iterrows():
        match = trades[trades["symbol"] == fill["symbol"]]
        assert len(match) == 1
        assert abs(match.iloc[0]["quantity"]) == abs(fill["quantity_delta"])
        assert match.iloc[0]["price"] == fill["fill_price"]
    source = inspect.getsource(runner_mod._commit_and_record)
    assert 'globals().get("fill_events")' not in source
    assert "simulated_tax_records(events" in source


def test_paper_cycle_emits_reconcile_audit(tmp_path, monkeypatch) -> None:
    """Every paper cycle records exactly one ledger_reconcile audit event with ok=True."""
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW

    runner_mod, settings, weights_path = _paper_cycle_harness(tmp_path, monkeypatch)
    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)
    assert report.status == "COMPLETE"
    events = _audit_events(tmp_path, "ledger_reconcile")
    assert len(events) == 1
    assert events[0]["ok"] is True


def test_paper_cycle_mismatch_alerts_once_without_halting(tmp_path, monkeypatch) -> None:
    """Cash perturbed by 1 USDT: COMPLETE, ok=False audited, one mismatch email."""
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW

    runner_mod, settings, weights_path = _paper_cycle_harness(tmp_path, monkeypatch)
    real_records = runner_mod.simulated_tax_records

    def _phantom_record(events, mode):
        records = real_records(events, mode)
        if records:
            import dataclasses

            phantom = dataclasses.replace(
                records[0],
                record_id=records[0].record_id + ":phantom",
                quote_qty=records[0].quote_qty + 1.0,
            )
            return (*records, phantom)
        return records

    monkeypatch.setattr(runner_mod, "simulated_tax_records", _phantom_record)
    calls: list[dict] = []

    def _fake_dispatch(settings, *, event, detail, decision_time, dedupe_key, now):
        calls.append({"event": event, "detail": detail, "decision_time": decision_time, "dedupe_key": dedupe_key})
        return True

    monkeypatch.setattr(runner_mod, "dispatch_alert", _fake_dispatch)
    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)
    assert report.status == "COMPLETE"
    events = _audit_events(tmp_path, "ledger_reconcile")
    assert len(events) == 1
    assert events[0]["ok"] is False
    assert len(calls) == 1
    assert calls[0]["event"] == "ledger_reconcile_mismatch"


def _seed_weights_artifact(tmp_path):
    """Two-symbol weights plus decision closes for DECISION_TIME."""
    weights = pd.DataFrame(
        {"AAAUSDT": [0.02], "BUSDT": [-0.02]}, index=pd.DatetimeIndex([DECISION_TIME])
    )
    weights_path = tmp_path / "deployed_target_weights.parquet"
    weights.to_parquet(weights_path, index=True)
    pd.DataFrame(
        {"AAAUSDT": [100.0], "BUSDT": [100.0]}, index=pd.DatetimeIndex([DECISION_TIME])
    ).to_parquet(tmp_path / "deployed_decision_ohlcv_close.parquet", index=True)
    return weights_path


def test_interrupted_cycle_resumes_without_restamping(tmp_path, monkeypatch) -> None:
    """An ExecutionInterrupted attempt commits its journaled fill but never stamps; rerun completes."""
    from src.live.executor import ExecutionInterrupted, ExecutionOutcome
    from src.live.fills import load_fills
    from src.live.ledger import LedgerState, load_ledger, save_ledger
    from src.live.settings import LiveSettings
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")

    seen: dict = {}
    calls = {"count": 0}

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, outcome_sink=None, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            first = intents[0]
            outcome = ExecutionOutcome(
                symbol=first.symbol, filled_qty=first.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            seen["symbol"] = first.symbol
            seen["qty"] = first.quantity
            _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, (outcome,), NOW)
            if outcome_sink is not None:
                outcome_sink.append(outcome)
            exc = ExecutionInterrupted("shutdown requested")
            exc.partial_outcomes = (outcome,)
            seen["first_symbols"] = sorted(i.symbol for i in intents)
            raise exc
        seen["resume_intents"] = list(intents)
        return ()

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)
    weights_path = _seed_weights_artifact(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    save_ledger(ledger_path, LedgerState(positions={}, equity_high_water_mark=Decimal("2000"), cash_usdt=Decimal("2000")))
    settings = LiveSettings(
        mode="paper",
        notional_equity_usdt=2000.0,
        ledger_path=str(ledger_path),
        order_journal_path=str(tmp_path / "order_journal.jsonl"),
        fills_dir=str(tmp_path / "fills"),
        tax_ledger_dir=str(tmp_path / "tax"),
        execution_quality_dir=str(tmp_path / "eq"),
        portfolio_state_dir=str(tmp_path / "portfolio"),
    )

    first = runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)
    assert first.status == "INTERRUPTED"
    assert first.reason == "shutdown_requested"
    interrupted = load_ledger(ledger_path)
    assert interrupted.last_executed_decision_time is None
    assert interrupted.positions[seen["symbol"]] == seen["qty"]
    assert len(load_fills(tmp_path / "fills")) == 1

    second = runner_mod.run_shadow_cycle(
        settings, DECISION_TIME, weights_path, now=NOW + pd.Timedelta(hours=1)
    )
    assert second.status == "COMPLETE"
    completed = load_ledger(ledger_path)
    assert completed.last_executed_decision_time == DECISION_TIME
    # 재개는 이미 체결된 종목을 다시 주문하지 않고 남은 종목만 재계획한다.
    resumed = sorted(i.symbol for i in seen["resume_intents"])
    assert seen["symbol"] not in resumed
    assert resumed == [sym for sym in seen["first_symbols"] if sym != seen["symbol"]]


def test_hard_kill_fills_repaired_at_next_start(tmp_path, monkeypatch) -> None:
    """Journal fills orphaned by a kill before commit are applied at the next cycle start."""
    from src.live.ledger import LedgerState, load_ledger, save_ledger
    from src.live.order_journal import OrderJournal
    from src.live.settings import LiveSettings
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient

    journal_path = tmp_path / "order_journal.jsonl"
    journal = OrderJournal(journal_path)
    attempt = journal.begin_attempt(
        decision_time=DECISION_TIME, run_id="20260824", mode="shadow",
        pre_trade_equity=Decimal("2000"), sizing_anchor="decision_ohlcv_close",
        decision_marks={"AAAUSDT": Decimal("100")}, started_at=NOW,
    )
    for qty, price in ((Decimal("0.2"), Decimal("100")), (Decimal("0.1"), Decimal("101"))):
        journal.record_fill(
            kind="execution", attempt_seq=attempt.attempt_seq, symbol="AAAUSDT", side="BUY",
            quantity=qty, price=price, fee_bps=5.0, liquidity="taker", reason="timeout_taker",
            filled_at=NOW, client_order_id=None, leg_index=0,
            cumulative_executed_qty=None, simulated=True,
        )

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")
    monkeypatch.setattr(runner_mod, "execute_intents", lambda *a, **k: ())

    weights_path = _seed_weights_artifact(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    save_ledger(ledger_path, LedgerState(positions={}, equity_high_water_mark=Decimal("2000"), cash_usdt=Decimal("2000")))
    settings = LiveSettings(
        mode="shadow",
        notional_equity_usdt=2000.0,
        ledger_path=str(ledger_path),
        order_journal_path=str(journal_path),
        fills_dir=str(tmp_path / "fills"),
        tax_ledger_dir=str(tmp_path / "tax"),
        execution_quality_dir=str(tmp_path / "eq"),
        portfolio_state_dir=str(tmp_path / "portfolio"),
    )

    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)
    assert report.status == "COMPLETE"
    assert load_ledger(ledger_path).positions["AAAUSDT"] == Decimal("0.3")


def test_already_executed_flush_emits_pending_evidence_once(tmp_path, monkeypatch) -> None:
    """A stamped ledger with recorded < applied flushes pending rows once with no venue calls."""
    from src.live.fills import load_fills
    from src.live.ledger import LedgerState, commit_journal_fills, load_ledger, mark_fills_recorded
    from src.live.order_journal import OrderJournal
    from src.live.settings import LiveSettings
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW

    journal_path = tmp_path / "order_journal.jsonl"
    journal = OrderJournal(journal_path)
    attempt = journal.begin_attempt(
        decision_time=DECISION_TIME, run_id="20260824", mode="paper",
        pre_trade_equity=Decimal("2000"), sizing_anchor="decision_ohlcv_close",
        decision_marks={"AAAUSDT": Decimal("100")}, started_at=NOW,
    )
    for qty, price in ((Decimal("0.2"), Decimal("100")), (Decimal("0.1"), Decimal("101"))):
        journal.record_fill(
            kind="execution", attempt_seq=attempt.attempt_seq, symbol="AAAUSDT", side="BUY",
            quantity=qty, price=price, fee_bps=5.0, liquidity="taker", reason="timeout_taker",
            filled_at=NOW, client_order_id=None, leg_index=0,
            cumulative_executed_qty=None, simulated=True,
        )

    ledger_path = tmp_path / "ledger.json"
    base = LedgerState(
        positions={}, equity_high_water_mark=Decimal("2000"), cash_usdt=Decimal("2000"),
        last_executed_decision_time=DECISION_TIME,
    )
    committed = commit_journal_fills(
        ledger_path, base, journal.fills_after(-1), equity=None,
        track_cash=True, starting_capital=Decimal("2000"), executed_decision_time=DECISION_TIME,
    )
    assert committed.journal_applied_fill_seq == 1
    assert mark_fills_recorded(ledger_path, committed, 0).journal_recorded_fill_seq == 0

    def _forbidden(*a, **k):
        raise AssertionError("already executed decision must not reach venue or weights")

    monkeypatch.setattr(runner_mod, "_market_client", _forbidden)
    monkeypatch.setattr(runner_mod, "_order_client", _forbidden)
    monkeypatch.setattr(runner_mod, "latest_target_weights", _forbidden)
    monkeypatch.setattr(runner_mod, "dispatch_alert", lambda *a, **k: True)
    settings = LiveSettings(
        mode="paper",
        notional_equity_usdt=2000.0,
        ledger_path=str(ledger_path),
        order_journal_path=str(journal_path),
        fills_dir=str(tmp_path / "fills"),
        tax_ledger_dir=str(tmp_path / "tax"),
        execution_quality_dir=str(tmp_path / "eq"),
        portfolio_state_dir=str(tmp_path / "portfolio"),
    )

    report = runner_mod.run_shadow_cycle(settings, DECISION_TIME, tmp_path / "missing.parquet", now=NOW)
    assert (report.status, report.reason) == ("COMPLETE", "already_executed")
    flushed = load_ledger(ledger_path)
    assert (flushed.journal_applied_fill_seq, flushed.journal_recorded_fill_seq) == (1, 1)
    assert flushed.positions["AAAUSDT"] == Decimal("0.3")
    fills = load_fills(tmp_path / "fills")
    assert len(fills) == 1
    assert set(fills["fill_id"].tolist()) == {"journal:1"}

    rerun = runner_mod.run_shadow_cycle(settings, DECISION_TIME, tmp_path / "missing.parquet", now=NOW)
    assert (rerun.status, rerun.reason) == ("COMPLETE", "already_executed")
    assert len(load_fills(tmp_path / "fills")) == 1





def test_recorded_watermark_failure_is_retried_without_duplicate_evidence(tmp_path, monkeypatch) -> None:
    """If advancing the recorded watermark fails after evidence was written, the cycle still completes; the
    already-executed rerun re-emits the same fills idempotently (fill_id / record_id) and advances the watermark."""
    from src.live.fills import load_fills
    from src.live.ledger import load_ledger
    from src.live.tax_ledger import load_tax_records
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW

    runner_mod, settings, weights_path = _paper_cycle_harness(tmp_path, monkeypatch)
    real_mark = runner_mod.mark_fills_recorded

    def _fail(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(runner_mod, "mark_fills_recorded", _fail)
    first = runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)
    assert first.status == "COMPLETE"
    assert len(_audit_events(tmp_path, "recorded_watermark_failed")) == 1
    lagging = load_ledger(tmp_path / "ledger.json")
    assert lagging.journal_recorded_fill_seq < lagging.journal_applied_fill_seq

    monkeypatch.setattr(runner_mod, "mark_fills_recorded", real_mark)
    rerun = runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)

    assert rerun.reason == "already_executed"
    healed = load_ledger(tmp_path / "ledger.json")
    assert healed.journal_recorded_fill_seq == healed.journal_applied_fill_seq
    assert len(load_fills(tmp_path / "fills")) == 2
    tax = load_tax_records(tmp_path / "tax", year=2026)
    assert len(tax[tax["kind"] == "TRADE"]) == 2


def test_already_executed_flush_failure_still_skips_cycle(tmp_path, monkeypatch) -> None:
    """The evidence flush on an already-executed day is best-effort: its failure never re-runs the cycle."""
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW

    runner_mod, settings, weights_path = _paper_cycle_harness(tmp_path, monkeypatch)
    real_mark = runner_mod.mark_fills_recorded
    monkeypatch.setattr(runner_mod, "mark_fills_recorded", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    assert runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW).status == "COMPLETE"
    monkeypatch.setattr(runner_mod, "mark_fills_recorded", real_mark)

    def _broken_commit(*args, **kwargs):
        raise OSError("journal unreadable")

    monkeypatch.setattr(runner_mod, "_commit_and_record", _broken_commit)
    rerun = runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)

    assert (rerun.status, rerun.reason, rerun.intent_count) == ("COMPLETE", "already_executed", 0)


@pytest.mark.parametrize("ending", ["interrupted", "halt", "abort"])
def test_failed_commit_after_early_end_never_masks_cause_and_is_repaired(tmp_path, monkeypatch, ending) -> None:
    """When committing an early-ended attempt fails, the original outcome (INTERRUPTED / HALT / crash) stands, the
    failure is audited with its stage, and the next cycle applies the journaled fill exactly once."""
    from src.live.errors import VenueError
    from src.live.executor import ExecutionInterrupted, ExecutionOutcome
    from src.live.ledger import LedgerState, load_ledger, save_ledger
    from src.live.settings import LiveSettings
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")
    seen: dict = {}

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, outcome_sink=None, **kwargs):
        if seen:
            return ()
        first = intents[0]
        outcome = ExecutionOutcome(symbol=first.symbol, filled_qty=first.quantity, unfilled_qty=Decimal("0"),
                                   avg_fill_price=Decimal("100"), chases=0, status="FILLED")
        seen.update(symbol=first.symbol, qty=first.quantity)
        _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, (outcome,), NOW)
        if ending == "interrupted":
            exc: BaseException = ExecutionInterrupted("shutdown requested")
        elif ending == "halt":
            exc = VenueError("venue", code=-1022, http_status=400, path="/fapi/v1/order", payload_digest="0" * 12)
        else:
            exc = RuntimeError("executor crashed")
        exc.partial_outcomes = (outcome,)  # type: ignore[attr-defined]
        raise exc

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)
    real_commit = runner_mod._commit_and_record
    commit_calls = {"n": 0}

    def _commit_fails_during_attempt(*args, **kwargs):
        commit_calls["n"] += 1
        if seen and commit_calls["n"] == 2:  # 1st = pre-trade commit, 2nd = the early-end commit
            raise OSError("ledger write failed")
        return real_commit(*args, **kwargs)

    monkeypatch.setattr(runner_mod, "_commit_and_record", _commit_fails_during_attempt)
    weights_path = _seed_weights_artifact(tmp_path)
    ledger_path = tmp_path / "ledger.json"
    save_ledger(ledger_path, LedgerState(positions={}, equity_high_water_mark=Decimal("2000"), cash_usdt=Decimal("2000")))
    settings = LiveSettings(mode="paper", notional_equity_usdt=2000.0, ledger_path=str(ledger_path),
                            order_journal_path=str(tmp_path / "order_journal.jsonl"), fills_dir=str(tmp_path / "fills"),
                            tax_ledger_dir=str(tmp_path / "tax"), execution_quality_dir=str(tmp_path / "eq"),
                            portfolio_state_dir=str(tmp_path / "portfolio"))

    if ending == "interrupted":
        assert runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW).status == "INTERRUPTED"
    elif ending == "halt":
        assert runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW).status == "HALT"
    else:
        with pytest.raises(RuntimeError, match="executor crashed"):
            runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)

    failed = [e for e in _audit_events(tmp_path, "commit_failed")]
    assert [e["stage"] for e in failed] == [ending]
    assert load_ledger(ledger_path).positions.get(seen["symbol"], Decimal(0)) == Decimal(0)

    runner_mod.run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW + pd.Timedelta(hours=1))

    assert load_ledger(ledger_path).positions[seen["symbol"]] == seen["qty"]


def test_operator_resync_fills_produce_no_trade_evidence(tmp_path) -> None:
    """Operator resync adjustments move the ledger but are not trades: no fill/tax evidence is built for them."""
    from src.live.order_journal import OrderJournal
    from tests.unit.live._runner_stubs import DECISION_TIME, NOW

    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = journal.begin_attempt(decision_time=DECISION_TIME, run_id="r", mode="live", pre_trade_equity=Decimal("1000"),
                                    sizing_anchor="equity", decision_marks={}, started_at=NOW)
    common = dict(symbol="AAAUSDT", side="BUY", quantity=Decimal("1"), price=Decimal("100"), fee_bps=0.0,
                  liquidity="taker", filled_at=NOW, client_order_id=None, leg_index=0,
                  cumulative_executed_qty=None, simulated=False)
    journal.record_fill(kind="execution", attempt_seq=attempt.attempt_seq, reason="timeout_taker", **common)
    journal.record_fill(kind="operator_resync", attempt_seq=None, reason="operator_resync", **common)

    events = runner_mod._fill_events_from_journal(journal, journal.fills_after(-1), fallback=attempt)

    assert [e.fill_id for e in events] == ["journal:0"]


def _delist_delivery_ms(delivery: pd.Timestamp) -> int:
    return int(pd.Timestamp(delivery).value // 1_000_000)


def _settling_exchange_info(symbol: str, status: str, delivery: pd.Timestamp) -> dict[str, Any]:
    return {
        "symbols": [
            {"symbol": symbol, "status": status, "deliveryDate": _delist_delivery_ms(delivery)}
        ]
    }


def _flat_evidence(symbol: str, delivery: pd.Timestamp, price: str, bars: int = 5):
    from src.live.venue_listing import SettlementEvidence

    return SettlementEvidence(
        symbol=symbol,
        delivery_time=pd.Timestamp(delivery),
        price=Decimal(price),
        flat_bars=bars,
        source="flat_1h_klines",
    )


def test_delisting_paper_long_books_cash_at_evidenced_price() -> None:
    """PAPER long settles at the evidenced price minus fee; reconcile explains the move."""
    from src.live.ledger import LedgerState
    from src.live.runner import book_delisting_settlements, delisting_settlement_tax_records
    from src.live.settings import ExecutionMode
    from src.live.tax_ledger import reconcile_cycle_cash

    delivery = pd.Timestamp("2026-08-20 09:00Z")
    now = pd.Timestamp("2026-08-24 00:00Z")
    state = LedgerState(positions={"XUSDT": Decimal("10")}, cash_usdt=Decimal("1000"))
    exchange_info = _settling_exchange_info("XUSDT", "SETTLING", delivery)
    evidence = {"XUSDT": _flat_evidence("XUSDT", delivery, "2.5")}

    booked_state, booked = book_delisting_settlements(
        state,
        mode=ExecutionMode.PAPER,
        exchange_info=exchange_info,
        venue_positions={},
        evidence=evidence,
        fee_bps=Decimal("5"),
        now=now,
    )

    assert booked_state.positions.get("XUSDT", Decimal(0)) == 0
    assert booked_state.cash_usdt == Decimal("1000") + Decimal("25") - Decimal("0.0125")
    assert len(booked) == 1
    assert booked[0].evidence_source == "flat_1h_klines"
    assert booked[0].delivery_time == delivery
    assert booked_state.position_history
    assert booked_state.position_history[-1].effective_from == now

    records = delisting_settlement_tax_records(booked, mode="paper")
    assert len(records) == 1
    record = records[0]
    assert record.kind == "TRADE"
    assert record.side == "SELL"
    assert record.source == "delisting_settlement"
    assert record.quantity == 10.0
    assert record.price == 2.5
    assert record.quote_qty == 25.0
    assert record.fee == 0.0125

    reconciliation = reconcile_cycle_cash(
        Decimal("1000"), booked_state.cash_usdt, records, (), tolerance_usdt=Decimal("0.01")
    )
    assert reconciliation.within_tolerance


def test_delisting_paper_short_settlement_sign() -> None:
    """PAPER short settlement decreases cash by notional plus fee with a BUY record."""
    from src.live.ledger import LedgerState
    from src.live.runner import book_delisting_settlements, delisting_settlement_tax_records
    from src.live.settings import ExecutionMode

    delivery = pd.Timestamp("2026-08-20 09:00Z")
    now = pd.Timestamp("2026-08-24 00:00Z")
    state = LedgerState(positions={"XUSDT": Decimal("-4")}, cash_usdt=Decimal("1000"))
    exchange_info = _settling_exchange_info("XUSDT", "SETTLING", delivery)
    evidence = {"XUSDT": _flat_evidence("XUSDT", delivery, "3")}

    booked_state, booked = book_delisting_settlements(
        state,
        mode=ExecutionMode.PAPER,
        exchange_info=exchange_info,
        venue_positions={},
        evidence=evidence,
        fee_bps=Decimal("5"),
        now=now,
    )

    assert booked_state.positions.get("XUSDT", Decimal(0)) == 0
    assert booked_state.cash_usdt == Decimal("1000") - Decimal("12") - Decimal("0.006")
    records = delisting_settlement_tax_records(booked, mode="paper")
    assert len(records) == 1
    assert records[0].side == "BUY"
    assert records[0].source == "delisting_settlement"


def test_delisting_paper_unresolved_fails_closed(tmp_path) -> None:
    """A held past-delivery name without evidence is never booked and fails closed."""
    from src.live.audit import AuditLog
    from src.live.ledger import LedgerState
    from src.live.runner import book_delisting_settlements
    from src.live.settings import ExecutionMode, LiveSettings
    from src.common.errors import DataIntegrityError
    from tests.unit.live._runner_stubs import DECISION_TIME

    delivery = pd.Timestamp("2026-08-20 09:00Z")
    now = pd.Timestamp("2026-08-24 00:00Z")
    state = LedgerState(positions={"XUSDT": Decimal("10")}, cash_usdt=Decimal("1000"))
    exchange_info = _settling_exchange_info("XUSDT", "SETTLING", delivery)

    untouched, booked = book_delisting_settlements(
        state,
        mode=ExecutionMode.PAPER,
        exchange_info=exchange_info,
        venue_positions={},
        evidence={},
        fee_bps=Decimal("5"),
        now=now,
    )
    assert booked == ()
    assert untouched.positions["XUSDT"] == Decimal("10")
    assert untouched.cash_usdt == Decimal("1000")

    audit = AuditLog(tmp_path / "shadow_cycle.jsonl")
    settings = LiveSettings(mode="paper", notional_equity_usdt=2000.0)
    with pytest.raises(DataIntegrityError, match="without settlement evidence"):
        runner_mod._settle_delisted_paper_positions(
            state,
            {"XUSDT": delivery},
            now,
            tmp_path / "ledger.json",
            audit,
            settings,
            DECISION_TIME,
        )
    assert state.positions["XUSDT"] == Decimal("10")
    assert state.cash_usdt == Decimal("1000")


def test_delisting_live_flat_zeroes_ghost_quantity(tmp_path) -> None:
    """LIVE zeroes a venue-flat settled quantity with no cash move and an audit record."""
    import json

    from src.live.audit import AuditLog
    from src.live.ledger import LedgerState
    from src.live.runner import book_delisting_settlements
    from src.live.settings import ExecutionMode

    delivery = pd.Timestamp("2026-08-20 09:00Z")
    now = pd.Timestamp("2026-08-24 00:00Z")
    state = LedgerState(positions={"XUSDT": Decimal("1.5")})
    exchange_info = _settling_exchange_info("XUSDT", "SETTLING", delivery)

    booked_state, booked = book_delisting_settlements(
        state,
        mode=ExecutionMode.LIVE_TESTNET,
        exchange_info=exchange_info,
        venue_positions={},
        evidence={},
        fee_bps=Decimal(0),
        now=now,
    )

    assert booked_state.positions.get("XUSDT", Decimal(0)) == 0
    assert booked_state.cash_usdt is None
    assert len(booked) == 1
    assert booked[0].price is None
    assert booked[0].fee == Decimal(0)
    assert booked[0].evidence_source == "venue_flat_position"

    audit_path = tmp_path / "shadow_cycle.jsonl"
    audit = AuditLog(audit_path)
    for settlement in booked:
        audit.record(
            "delisting_settlement_booked",
            symbol=settlement.symbol,
            qty=str(settlement.quantity),
            price=None if settlement.price is None else str(settlement.price),
            fee=str(settlement.fee),
            delivery_time=settlement.delivery_time.isoformat(),
            evidence_source=settlement.evidence_source,
        )
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    booked_events = [e for e in events if e.get("event") == "delisting_settlement_booked"]
    assert len(booked_events) == 1
    assert booked_events[0]["symbol"] == "XUSDT"
    assert booked_events[0]["evidence_source"] == "venue_flat_position"


def test_delisting_live_relist_after_booking_cannot_breach() -> None:
    """A booked (zeroed) symbol reconciles cleanly whether relisted or purged."""
    from src.live.account import reconcile_or_halt, synthetic_flat_snapshot
    from src.live.ledger import LedgerState
    from src.live.runner import book_delisting_settlements
    from src.live.settings import ExecutionMode

    delivery = pd.Timestamp("2026-08-20 09:00Z")
    now = pd.Timestamp("2026-08-24 00:00Z")
    state = LedgerState(positions={"XUSDT": Decimal("1.5")})
    exchange_info = _settling_exchange_info("XUSDT", "SETTLING", delivery)

    booked_state, booked = book_delisting_settlements(
        state,
        mode=ExecutionMode.LIVE_TESTNET,
        exchange_info=exchange_info,
        venue_positions={},
        evidence={},
        fee_bps=Decimal(0),
        now=now,
    )
    assert booked

    snapshot = synthetic_flat_snapshot(now)
    relisted_info = {
        "symbols": [
            {"symbol": "XUSDT", "status": "TRADING", "deliveryDate": 9999999999999}
        ]
    }
    purged_info = {"symbols": [{"symbol": "OTHERUSDT", "status": "TRADING"}]}
    assert booked_state.positions.get("XUSDT", Decimal(0)) == 0
    reconcile_or_halt(
        snapshot, booked_state.positions, qty_tolerance_fraction=0.001, settled_symbols=()
    )
    assert relisted_info["symbols"][0]["status"] == "TRADING"
    assert all(e.get("symbol") != "XUSDT" for e in purged_info["symbols"])
    reconcile_or_halt(
        snapshot, booked_state.positions, qty_tolerance_fraction=0.001, settled_symbols=()
    )


def test_delisting_live_nonzero_venue_position_never_booked() -> None:
    """LIVE never books while the venue still holds the position."""
    from src.live.ledger import LedgerState
    from src.live.runner import book_delisting_settlements
    from src.live.settings import ExecutionMode

    delivery = pd.Timestamp("2026-08-20 09:00Z")
    now = pd.Timestamp("2026-08-24 00:00Z")
    state = LedgerState(positions={"XUSDT": Decimal("1.5")})
    exchange_info = _settling_exchange_info("XUSDT", "SETTLING", delivery)

    same, booked = book_delisting_settlements(
        state,
        mode=ExecutionMode.LIVE_TESTNET,
        exchange_info=exchange_info,
        venue_positions={"XUSDT": Decimal("1.5")},
        evidence={},
        fee_bps=Decimal(0),
        now=now,
    )

    assert booked == ()
    assert same.positions["XUSDT"] == Decimal("1.5")


def test_delisting_booking_idempotent_across_retries() -> None:
    """A retry after booking finds zero quantity and books nothing new."""
    from src.live.ledger import LedgerState
    from src.live.runner import book_delisting_settlements
    from src.live.settings import ExecutionMode

    delivery = pd.Timestamp("2026-08-20 09:00Z")
    now = pd.Timestamp("2026-08-24 00:00Z")
    state = LedgerState(positions={"XUSDT": Decimal("10")}, cash_usdt=Decimal("1000"))
    exchange_info = _settling_exchange_info("XUSDT", "SETTLING", delivery)
    evidence = {"XUSDT": _flat_evidence("XUSDT", delivery, "2.5")}

    first, booked = book_delisting_settlements(
        state,
        mode=ExecutionMode.PAPER,
        exchange_info=exchange_info,
        venue_positions={},
        evidence=evidence,
        fee_bps=Decimal("5"),
        now=now,
    )
    assert len(booked) == 1

    second, rebooked = book_delisting_settlements(
        first,
        mode=ExecutionMode.PAPER,
        exchange_info=exchange_info,
        venue_positions={},
        evidence=evidence,
        fee_bps=Decimal("5"),
        now=now,
    )
    assert rebooked == ()
    assert second.cash_usdt == first.cash_usdt
    assert second.positions.get("XUSDT", Decimal(0)) == 0


def test_delisting_funding_stops_at_delivery() -> None:
    """Funding accrual capped at delivery emits no FUNDING_FEE record after it."""
    from src.live.ledger import PositionSnapshot, accrue_funding_by_watermark

    held_from = pd.Timestamp("2026-08-18 00:00Z")
    delivery = pd.Timestamp("2026-08-20 09:00Z")
    now = pd.Timestamp("2026-08-24 00:00Z")
    before = pd.Timestamp("2026-08-19 00:00Z")
    after = pd.Timestamp("2026-08-21 00:00Z")
    history = (
        PositionSnapshot(effective_from=held_from, positions={"XUSDT": Decimal("1")}),
    )
    funding = pd.Series(
        [0.001, 0.001], index=pd.DatetimeIndex([before, after])
    )
    closes = pd.Series(
        [100.0, 100.0],
        index=pd.DatetimeIndex([before.floor("h"), after.floor("h")]),
    )

    accrual = accrue_funding_by_watermark(
        history,
        {},
        {"XUSDT": funding},
        {"XUSDT": closes},
        now,
        closed_at={"XUSDT": delivery},
    )

    assert len(accrual.events) == 1
    assert accrual.events[0].epoch <= delivery
    assert all(event.epoch <= delivery for event in accrual.events)


def test_delisting_booking_validates_inputs() -> None:
    """Naive timestamps, negative fees, and missing PAPER cash fail closed."""
    from src.live.ledger import LedgerState
    from src.live.runner import book_delisting_settlements
    from src.live.settings import ExecutionMode
    from src.common.errors import DataIntegrityError

    delivery = pd.Timestamp("2026-08-20 09:00Z")
    aware_now = pd.Timestamp("2026-08-24 00:00Z")
    naive_now = pd.Timestamp("2026-08-24 00:00")
    state = LedgerState(positions={"XUSDT": Decimal("10")}, cash_usdt=Decimal("1000"))
    exchange_info = _settling_exchange_info("XUSDT", "SETTLING", delivery)
    evidence = {"XUSDT": _flat_evidence("XUSDT", delivery, "2.5")}

    with pytest.raises(DataIntegrityError):
        book_delisting_settlements(
            state,
            mode=ExecutionMode.PAPER,
            exchange_info=exchange_info,
            venue_positions={},
            evidence=evidence,
            fee_bps=Decimal("5"),
            now=naive_now,
        )
    with pytest.raises(DataIntegrityError):
        book_delisting_settlements(
            state,
            mode=ExecutionMode.PAPER,
            exchange_info=exchange_info,
            venue_positions={},
            evidence=evidence,
            fee_bps=Decimal("-1"),
            now=aware_now,
        )
    cashless = LedgerState(positions={"XUSDT": Decimal("10")}, cash_usdt=None)
    with pytest.raises(DataIntegrityError):
        book_delisting_settlements(
            cashless,
            mode=ExecutionMode.PAPER,
            exchange_info=exchange_info,
            venue_positions={},
            evidence=evidence,
            fee_bps=Decimal("5"),
            now=aware_now,
        )


def test_delisting_shadow_books_nothing() -> None:
    """SHADOW never books settlements; the venue owns reconciliation there."""
    from src.live.runner import book_delisting_settlements
    from src.live.settings import ExecutionMode

    delivery = pd.Timestamp("2026-08-20 09:00Z")
    now = pd.Timestamp("2026-08-24 00:00Z")
    state = LedgerState(positions={"XUSDT": Decimal("1.5")}, cash_usdt=Decimal("1000"))
    exchange_info = _settling_exchange_info("XUSDT", "SETTLING", delivery)
    evidence = {"XUSDT": _flat_evidence("XUSDT", delivery, "2.5")}

    out_state, booked = book_delisting_settlements(
        state,
        mode=ExecutionMode.SHADOW,
        exchange_info=exchange_info,
        venue_positions={},
        evidence=evidence,
        fee_bps=Decimal("5"),
        now=now,
    )

    assert booked == ()
    assert out_state == state


def test_delisting_paper_skips_unlisted_and_future_delivery() -> None:
    """Held names absent from the schedule or not yet delivered stay untouched."""
    from src.live.runner import book_delisting_settlements
    from src.live.settings import ExecutionMode

    now = pd.Timestamp("2026-08-24 00:00Z")
    future = pd.Timestamp("2026-09-01 09:00Z")
    state = LedgerState(
        positions={"GHOSTUSDT": Decimal("2"), "FUTUREUSDT": Decimal("3")},
        cash_usdt=Decimal("1000"),
    )
    exchange_info = {
        "symbols": [
            {"symbol": "FUTUREUSDT", "status": "TRADING", "deliveryDate": _delist_delivery_ms(future)},
        ]
    }

    out_state, booked = book_delisting_settlements(
        state,
        mode=ExecutionMode.PAPER,
        exchange_info=exchange_info,
        venue_positions={},
        evidence={},
        fee_bps=Decimal("5"),
        now=now,
    )

    assert booked == ()
    assert out_state == state


def test_delisting_tax_records_skip_live_cashless_settlements() -> None:
    """LIVE settlements carry no price, so they emit no TRADE record."""
    from src.live.runner import DelistingSettlement, delisting_settlement_tax_records

    live = DelistingSettlement(
        symbol="XUSDT",
        quantity=Decimal("1.5"),
        price=None,
        fee=Decimal("0"),
        delivery_time=pd.Timestamp("2026-08-20 09:00Z"),
        evidence_source="venue_flat_position",
    )

    assert delisting_settlement_tax_records((live,), mode="live_testnet") == ()
