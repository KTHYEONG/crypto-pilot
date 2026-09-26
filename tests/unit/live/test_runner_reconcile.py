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
    assert_suppressed_venue_flat,
    resolve_sizing_equity,
)
from src.live.errors import ReconciliationBreach, RiskGateBreach, VenueError
from src.live.executor import ExecutionOutcome
from src.live.ledger import LedgerState, load_ledger, save_ledger
from src.live.runner import check_risk_gates, run_shadow_cycle
from src.live.settings import LiveSettings

from tests.unit.live._runner_stubs import DECISION_TIME, NOW, StubMarketClient, StubOrderClient


def _tmp_state_kwargs(tmp_path: Path, stem: str) -> dict[str, str]:
    """Isolate every filesystem side channel of one cycle under tmp_path."""
    base = tmp_path / stem
    return {
        "order_journal_path": str(base / "order_journal.jsonl"),
        "fills_dir": str(base / "fills"),
        "tax_ledger_dir": str(base / "tax"),
        "execution_quality_dir": str(base / "execution_quality"),
        "portfolio_state_dir": str(base / "portfolio_state"),
    }


def _journal_fake_fills(journal: Any, attempt: Any, intents: Any, outcomes: Any) -> None:
    """Mirror fake execution outcomes into the order journal (kind="execution").

    The runner persists executions only through the journal; a fake that
    returns bare outcomes without journaling persists nothing.
    """
    if journal is None:
        return
    attempt_seq = attempt.attempt_seq if attempt is not None else None
    for i, (intent, outcome) in enumerate(zip(intents, outcomes)):
        if outcome.filled_qty <= 0:
            continue
        if outcome.fills:
            _q, _p, fee_bps, reason, liquidity, _ts = outcome.fills[0]
        else:
            fee_bps, reason, liquidity = 5.0, "timeout_taker", "taker"
        journal.record_fill(
            kind="execution",
            attempt_seq=attempt_seq,
            symbol=intent.symbol,
            side=intent.side,
            quantity=outcome.filled_qty,
            price=outcome.avg_fill_price if outcome.avg_fill_price else Decimal("100"),
            fee_bps=float(fee_bps),
            liquidity=str(liquidity),
            reason=str(reason),
            filled_at=NOW,
            client_order_id=f"fake-{attempt_seq}-{i}-{intent.symbol}",
            leg_index=int(getattr(intent, "leg_index", 0)),
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
        _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, outcomes)
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
        mode="paper", notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_contam.json"),
        **_tmp_state_kwargs(tmp_path, "state_contam"),
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
        outcomes = tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )
        _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, outcomes)
        return outcomes

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)

    ledger_path = tmp_path / "ledger_live_mode.json"
    save_ledger(
        ledger_path,
        LedgerState(positions={"AAAUSDT": Decimal("0.4")}, equity_high_water_mark=Decimal(0)),
    )
    settings = LiveSettings(mode="live_testnet", notional_equity_usdt=2000.0, ledger_path=str(ledger_path),
                             **_tmp_state_kwargs(tmp_path, "state_live_mode"))

    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: VenueOrderClient(None))
    diverged = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert diverged.status == "DEGRADED"
    assert diverged.reason is not None
    assert "reconciliation_breach" in diverged.reason

    monkeypatch.setattr(
        runner_mod, "_order_client", lambda settings, decision_time: VenueOrderClient("0.4")
    )
    matched_settings = LiveSettings(mode="live_testnet", notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_live_mode_matched.json"),
                             **_tmp_state_kwargs(tmp_path, "state_live_mode_matched"))
    save_ledger(
        Path(matched_settings.ledger_path),
        LedgerState(positions={"AAAUSDT": Decimal("0.4")}, equity_high_water_mark=Decimal(0)),
    )
    matched = run_shadow_cycle(matched_settings, DECISION_TIME, artifact, now=NOW)
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
        outcomes = tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )
        _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, outcomes)
        return outcomes

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)

    ledger_path = tmp_path / "ledger_dropout.json"
    save_ledger(
        ledger_path,
        LedgerState(positions={"AAAUSDT": Decimal("0.4")}, equity_high_water_mark=Decimal(0)),
    )
    # 오늘 아티팩트에는 BUSDT만 존재 -- AAAUSDT는 로스터에서 완전히 빠졌다.
    weights = pd.DataFrame({"BUSDT": [-0.02]}, index=pd.DatetimeIndex([DECISION_TIME]))
    weights_path = tmp_path / "dropout_deployed_target_weights.parquet"
    weights.to_parquet(weights_path, index=True)
    from src.live.deployed_weights import decision_ohlcv_close_path
    pd.DataFrame({"BUSDT": [100.0]}, index=pd.DatetimeIndex([DECISION_TIME])).to_parquet(
        decision_ohlcv_close_path(weights_path), index=True
    )

    settings = LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(ledger_path),
                             **_tmp_state_kwargs(tmp_path, "state_dropout"))
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
    # exchangeInfo 등재는 유지(SETTLING)해야 absent HALT가 아닌 uncovered 경로를 탄다.
    class NoAAAFiltersMarketClient(StubMarketClient):
        def exchange_info(self) -> dict[str, Any]:
            payload = super().exchange_info()
            for s in payload["symbols"]:
                if s["symbol"] == "AAAUSDT":
                    s["status"] = "SETTLING"
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
        outcomes = tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )
        _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, outcomes)
        return outcomes

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)

    ledger_path = tmp_path / "ledger_delisted.json"
    save_ledger(
        ledger_path,
        LedgerState(positions={"AAAUSDT": Decimal("0.4")}, equity_high_water_mark=Decimal(0)),
    )
    weights = pd.DataFrame({"BUSDT": [-0.02]}, index=pd.DatetimeIndex([DECISION_TIME]))
    weights_path = tmp_path / "delisted_deployed_target_weights.parquet"
    weights.to_parquet(weights_path, index=True)
    from src.live.deployed_weights import decision_ohlcv_close_path as _close_path
    pd.DataFrame({"BUSDT": [100.0]}, index=pd.DatetimeIndex([DECISION_TIME])).to_parquet(
        _close_path(weights_path), index=True
    )

    settings = LiveSettings(notional_equity_usdt=2000.0, ledger_path=str(ledger_path),
                             **_tmp_state_kwargs(tmp_path, "state_delisted"))
    report = run_shadow_cycle(settings, DECISION_TIME, weights_path, now=NOW)
    assert report.status == "COMPLETE"

    events = [
        json.loads(line)
        for line in (tmp_path / "shadow_cycle.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    uncovered = [e for e in events if e["event"] == "position_uncovered"]
    assert any(e.get("symbol") == "AAAUSDT" and e.get("reason") == "no_filters" for e in uncovered)




def test_run_shadow_cycle_live_mode_sets_venue_leverage_before_execution(artifact, monkeypatch, tmp_path) -> None:
    from decimal import Decimal
    from src.live.errors import VenueError

    calls: list[tuple[str, str, dict]] = []
    executed: list[list[str]] = []

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        executed.append([intent.symbol for intent in intents])
        outcomes = tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )
        _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, outcomes)
        return outcomes

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)
    ledger_path = tmp_path / "ledger_live.json"
    save_ledger(ledger_path, LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    settings = LiveSettings(mode="live_testnet", notional_equity_usdt=2000.0, ledger_path=str(ledger_path),
                             **_tmp_state_kwargs(tmp_path, "state_live"))

    def _audit_events() -> list[dict]:
        path = tmp_path / "shadow_cycle.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    class RecordingClient(StubOrderClient):
        def request(self, method, path, params=None, *, signed=False):
            calls.append((method, path, dict(params or {})))
            return super().request(method, path, params, signed=signed)

    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: RecordingClient())

    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "COMPLETE"
    assert [(path, params) for method, path, params in calls if method == "POST"] == [
        ("/fapi/v1/marginType", {"symbol": "AAAUSDT", "marginType": "CROSSED"}),
        ("/fapi/v1/marginType", {"symbol": "BUSDT", "marginType": "CROSSED"}),
        ("/fapi/v1/leverage", {"symbol": "AAAUSDT", "leverage": 14}),
        ("/fapi/v1/leverage", {"symbol": "BUSDT", "leverage": 14}),
    ]
    assert executed == [["AAAUSDT", "BUSDT"]]


def test_run_shadow_cycle_live_mode_rejects_intent_over_notional_cap(artifact, monkeypatch, tmp_path) -> None:
    from decimal import Decimal
    from src.live.errors import VenueError

    calls: list[tuple[str, str, dict]] = []
    executed: list[list[str]] = []

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        executed.append([intent.symbol for intent in intents])
        outcomes = tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )
        _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, outcomes)
        return outcomes

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)
    ledger_path = tmp_path / "ledger_live.json"
    save_ledger(ledger_path, LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    settings = LiveSettings(mode="live_testnet", notional_equity_usdt=2000.0, ledger_path=str(ledger_path),
                             **_tmp_state_kwargs(tmp_path, "state_live"))

    def _audit_events() -> list[dict]:
        path = tmp_path / "shadow_cycle.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    class TightCapClient(StubOrderClient):
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v1/leverageBracket":
                rows = super().request(method, path, params, signed=signed)
                rows[0]["brackets"][0]["notionalCap"] = 10
                return rows
            return super().request(method, path, params, signed=signed)

    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: TightCapClient())

    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "COMPLETE"
    assert executed == [["BUSDT"]]
    rejected = [e for e in _audit_events() if e["event"] == "notional_cap_rejected"]
    assert [e["symbol"] for e in rejected] == ["AAAUSDT"]
    assert rejected[0]["notional_cap"] == "10"


def test_run_shadow_cycle_live_mode_halts_when_margin_change_blocked(artifact, monkeypatch, tmp_path) -> None:
    from decimal import Decimal
    from src.live.errors import VenueError

    calls: list[tuple[str, str, dict]] = []
    executed: list[list[str]] = []

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        executed.append([intent.symbol for intent in intents])
        outcomes = tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )
        _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, outcomes)
        return outcomes

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)
    ledger_path = tmp_path / "ledger_live.json"
    save_ledger(ledger_path, LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    settings = LiveSettings(mode="live_testnet", notional_equity_usdt=2000.0, ledger_path=str(ledger_path),
                             **_tmp_state_kwargs(tmp_path, "state_live"))

    def _audit_events() -> list[dict]:
        path = tmp_path / "shadow_cycle.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    class BlockedClient(StubOrderClient):
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v1/marginType":
                raise VenueError("venue rejected request", code=-4048, http_status=400, path=path, payload_digest="000000000000")
            return super().request(method, path, params, signed=signed)

    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: BlockedClient())

    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "HALT"
    assert "margin type change to CROSSED blocked for AAAUSDT (code=-4048)" in report.reason
    assert executed == []


def test_run_shadow_cycle_live_mode_accepts_settled_delisting_only_after_delivery(artifact, monkeypatch, tmp_path) -> None:
    from decimal import Decimal
    from src.live.errors import VenueError

    calls: list[tuple[str, str, dict]] = []
    executed: list[list[str]] = []

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        executed.append([intent.symbol for intent in intents])
        outcomes = tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )
        _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, outcomes)
        return outcomes

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)
    ledger_path = tmp_path / "ledger_live.json"
    save_ledger(ledger_path, LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    settings = LiveSettings(mode="live_testnet", notional_equity_usdt=2000.0, ledger_path=str(ledger_path),
                             **_tmp_state_kwargs(tmp_path, "state_live"))

    def _audit_events() -> list[dict]:
        path = tmp_path / "shadow_cycle.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _market_with(delivery_ms: int):
        class DelistedMarketClient(StubMarketClient):
            def exchange_info(self):
                payload = super().exchange_info()
                payload["symbols"].append({
                    "symbol": "ZZZUSDT", "contractType": "PERPETUAL", "quoteAsset": "USDT",
                    "status": "SETTLING", "deliveryDate": delivery_ms,
                })
                return payload
        return DelistedMarketClient()

    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: StubOrderClient())
    save_ledger(ledger_path, LedgerState(positions={"ZZZUSDT": Decimal("1.5")}, equity_high_water_mark=Decimal(0)))

    future_ms = int((NOW + pd.Timedelta(days=1)).value // 1_000_000)
    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: _market_with(future_ms))
    before = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert before.status == "DEGRADED"
    assert before.reason is not None
    assert "reconciliation_breach" in before.reason

    past_ms = int((NOW - pd.Timedelta(hours=1)).value // 1_000_000)
    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: _market_with(past_ms))
    save_ledger(ledger_path, LedgerState(positions={"ZZZUSDT": Decimal("1.5")}, equity_high_water_mark=Decimal(0)))
    after = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert after.status == "COMPLETE"
    pending = [e for e in _audit_events() if e["event"] == "delisting_settlement_pending"]
    assert [(e["symbol"], e["ledger_qty"]) for e in pending] == [("ZZZUSDT", "1.5")]


def test_run_shadow_cycle_paper_mode_never_touches_venue_leverage(artifact, monkeypatch, tmp_path) -> None:
    from decimal import Decimal
    from src.live.errors import VenueError

    calls: list[tuple[str, str, dict]] = []
    executed: list[list[str]] = []

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl")

    def fake_execute_intents(client, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        executed.append([intent.symbol for intent in intents])
        outcomes = tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )
        _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, outcomes)
        return outcomes

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)
    ledger_path = tmp_path / "ledger_live.json"
    save_ledger(ledger_path, LedgerState(positions={}, equity_high_water_mark=Decimal(0)))
    settings = LiveSettings(mode="live_testnet", notional_equity_usdt=2000.0, ledger_path=str(ledger_path),
                             **_tmp_state_kwargs(tmp_path, "state_live"))

    def _audit_events() -> list[dict]:
        path = tmp_path / "shadow_cycle.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    class RecordingClient(StubOrderClient):
        def request(self, method, path, params=None, *, signed=False):
            calls.append((method, path, dict(params or {})))
            return super().request(method, path, params, signed=signed)

    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: RecordingClient())
    paper = LiveSettings(mode="paper", notional_equity_usdt=2000.0, ledger_path=str(tmp_path / "ledger_paper.json"),
                          **_tmp_state_kwargs(tmp_path, "state_paper"))

    report = run_shadow_cycle(paper, DECISION_TIME, artifact, now=NOW)

    assert report.status == "COMPLETE"
    touched = {path for _, path, _ in calls}
    assert touched.isdisjoint({"/fapi/v1/leverageBracket", "/fapi/v1/marginType", "/fapi/v1/leverage"})



class _BreachOrderClient(StubOrderClient):
    """LIVE fake with venue AAAUSDT 0.5 against an empty ledger."""

    def __init__(self) -> None:
        self.force_calls: list[tuple[int, int, int]] = []
        self.cancels: list[str] = []
        self._open: list[dict] = []

    def request(self, method, path, params=None, *, signed=False):
        if path == "/fapi/v2/positionRisk":
            return [{"symbol": "AAAUSDT", "positionAmt": "0.5"}]
        return super().request(method, path, params, signed=signed)

    def open_orders(self):
        return list(self._open)

    def cancel_order(self, symbol, orig_client_order_id):
        self.cancels.append(orig_client_order_id)
        return {}

    def query_order(self, symbol, orig_client_order_id):
        return {"status": "CANCELED", "side": "BUY", "avgPrice": "100", "executedQty": "0"}

    def force_orders(self, *, start_time_ms, end_time_ms, limit=100):
        self.force_calls.append((start_time_ms, end_time_ms, limit))
        return []


def _live_derisk_env(monkeypatch, tmp_path, client):
    """Wire market/order fakes, journaling executor fake and alert capture."""
    import src.live.orderbook as ob_mod

    monkeypatch.setattr(runner_mod, "_market_client", lambda settings, decision_time: StubMarketClient())
    monkeypatch.setattr(runner_mod, "_order_client", lambda settings, decision_time: client)
    monkeypatch.setattr(
        runner_mod, "default_audit_log_path", lambda name, for_date=None: tmp_path / f"{name}.jsonl"
    )
    monkeypatch.setattr(ob_mod, "capture_order_books", lambda *a, **k: [])
    monkeypatch.setattr(ob_mod, "append_order_book_snapshots", lambda *a, **k: [])
    posted: list[Any] = []
    alerts: list[tuple[str, str]] = []

    def fake_execute_intents(c, intents, filters, policy, audit, clock, sleep_fn, *, rate_limits=None, **kwargs):
        posted.extend(intents)
        outcomes = tuple(
            ExecutionOutcome(
                symbol=i.symbol, filled_qty=i.quantity, unfilled_qty=Decimal("0"),
                avg_fill_price=Decimal("100"), chases=0, status="FILLED",
            )
            for i in intents
        )
        _journal_fake_fills(kwargs.get("journal"), kwargs.get("attempt"), intents, outcomes)
        return outcomes

    monkeypatch.setattr(runner_mod, "execute_intents", fake_execute_intents)
    monkeypatch.setattr(
        runner_mod, "dispatch_alert",
        lambda settings, *, event, detail, decision_time, dedupe_key, now: alerts.append((event, detail)) or True,
    )
    return posted, alerts


def _live_settings(tmp_path, name: str, **overrides: Any):
    ledger_path = tmp_path / f"ledger_{name}.json"
    kwargs = {"mode": "live_testnet", "notional_equity_usdt": 2000.0, "ledger_path": str(ledger_path)}
    kwargs.update(_tmp_state_kwargs(tmp_path, f"state_{name}"))
    kwargs.update(overrides)
    return LiveSettings(**kwargs)


def test_unexplained_breach_enters_persistent_derisk_mode(artifact, monkeypatch, tmp_path) -> None:
    """Unexplained breach enters persistent de-risk mode."""
    client = _BreachOrderClient()
    posted, alerts = _live_derisk_env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "derisk_enter")

    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "DEGRADED"
    assert report.reason == "reconciliation_breach"
    assert [i.symbol for i in posted] == ["AAAUSDT"]
    assert all(i.reduce_only for i in posted)
    state = load_ledger(Path(settings.ledger_path))
    assert state.derisk_since is not None
    assert state.derisk_reasons == ("reconciliation_breach",)
    assert state.last_executed_decision_time is None
    degraded = [detail for event, detail in alerts if event == "cycle_degraded"]
    assert degraded and "force_close_auto_adopt=off" in degraded[0]


def test_derisk_mode_persists_after_breach_disappears(artifact, monkeypatch, tmp_path) -> None:
    """De-risk mode persists after the breach disappears."""
    from src.live.ledger import enter_derisk

    client = _BreachOrderClient()
    posted, _alerts = _live_derisk_env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "derisk_persist")
    ledger_path = Path(settings.ledger_path)
    save_ledger(ledger_path, LedgerState(positions={"AAAUSDT": Decimal("0.5")}, equity_high_water_mark=Decimal(0)))
    ledger_state = load_ledger(ledger_path)
    enter_derisk(ledger_path, ledger_state, reasons=("reconciliation_breach",), now=NOW)

    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "DEGRADED"
    assert all(i.reduce_only for i in posted)
    assert load_ledger(ledger_path).derisk_since is not None


def test_auto_adoption_is_off_by_default(artifact, monkeypatch, tmp_path) -> None:
    """Auto-adoption is off by default."""
    client = _BreachOrderClient()
    posted, alerts = _live_derisk_env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "derisk_off_default")

    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert client.force_calls == []
    assert report.status == "DEGRADED"
    state = load_ledger(Path(settings.ledger_path))
    assert state.derisk_since is not None
    degraded = [detail for event, detail in alerts if event == "cycle_degraded"]
    assert degraded and "force_close_auto_adopt=off" in degraded[0]


def test_explained_adl_resumes_normal_rebalance(artifact, monkeypatch, tmp_path) -> None:
    """Explained ADL resumes a normal rebalance."""
    from src.live.order_journal import OrderJournal

    force_ms = int(NOW.value // 1_000_000)

    class _AdlClient(_BreachOrderClient):
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/positionRisk":
                return [{"symbol": "AAAUSDT", "positionAmt": "0.5"}]
            return StubOrderClient.request(self, method, path, params, signed=signed)

        def force_orders(self, *, start_time_ms, end_time_ms, limit=100):
            self.force_calls.append((start_time_ms, end_time_ms, limit))
            return [{
                "symbol": "AAAUSDT", "side": "SELL", "executedQty": "0.5", "avgPrice": "100",
                "autoCloseType": "ADL", "orderId": "adl-1", "updateTime": force_ms,
            }]

    client = _AdlClient()
    posted, _alerts = _live_derisk_env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "derisk_adopt", venue_force_close_auto_adopt=True)
    save_ledger(
        Path(settings.ledger_path),
        LedgerState(positions={"AAAUSDT": Decimal("1.0")}, equity_high_water_mark=Decimal(0)),
    )

    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "COMPLETE"
    assert sorted(i.symbol for i in posted) == ["AAAUSDT", "BUSDT"]
    journal = OrderJournal(Path(settings.order_journal_path))
    adopted = [f for f in journal.fills_after(-1) if f.kind == "venue_force_close"]
    assert len(adopted) == 1
    assert (adopted[0].symbol, adopted[0].side) == ("AAAUSDT", "SELL")
    state = load_ledger(Path(settings.ledger_path))
    assert state.derisk_since is None
    assert state.positions["AAAUSDT"] == Decimal("0.4")


def test_adopted_force_closes_are_not_adopted_twice(artifact, monkeypatch, tmp_path) -> None:
    """Adopted force closes are not adopted twice."""
    from src.live.order_journal import OrderJournal

    force_ms = int(NOW.value // 1_000_000)

    class _AdlClient(_BreachOrderClient):
        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/positionRisk":
                return [{"symbol": "AAAUSDT", "positionAmt": "0.5"}]
            return StubOrderClient.request(self, method, path, params, signed=signed)

        def force_orders(self, *, start_time_ms, end_time_ms, limit=100):
            self.force_calls.append((start_time_ms, end_time_ms, limit))
            return [{
                "symbol": "AAAUSDT", "side": "SELL", "executedQty": "0.5", "avgPrice": "100",
                "autoCloseType": "ADL", "orderId": "adl-1", "updateTime": force_ms,
            }]

    client = _AdlClient()
    posted, _alerts = _live_derisk_env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "derisk_no_double", venue_force_close_auto_adopt=True)
    save_ledger(
        Path(settings.ledger_path),
        LedgerState(positions={"AAAUSDT": Decimal("1.0")}, equity_high_water_mark=Decimal(0)),
    )

    first = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)
    assert first.status == "COMPLETE"
    second = run_shadow_cycle(settings, DECISION_TIME + pd.Timedelta(days=1), artifact, now=NOW + pd.Timedelta(days=1))
    journal = OrderJournal(Path(settings.order_journal_path))
    adopted = [f for f in journal.fills_after(-1) if f.kind == "venue_force_close"]
    assert len(adopted) == 1


def test_disabled_derisk_mode_keeps_fail_closed_behavior(artifact, monkeypatch, tmp_path) -> None:
    """Disabled de-risk mode keeps fail-closed behavior."""
    client = _BreachOrderClient()
    posted, _alerts = _live_derisk_env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "derisk_disabled", derisk_mode_enabled=False)

    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "HALT"
    assert report.reason is not None
    assert "position divergence" in report.reason
    assert posted == []


def test_foreign_open_order_excludes_symbol_and_derisks(artifact, monkeypatch, tmp_path) -> None:
    """Foreign open order excludes its symbol and de-risks."""
    from src.live.order_journal import OrderJournal

    class _ForeignClient(StubOrderClient):
        def __init__(self) -> None:
            self.cancels: list[str] = []

        def request(self, method, path, params=None, *, signed=False):
            if path == "/fapi/v2/positionRisk":
                return [
                    {"symbol": "AAAUSDT", "positionAmt": "1.0"},
                    {"symbol": "BUSDT", "positionAmt": "-1.0"},
                ]
            return super().request(method, path, params, signed=signed)

        def open_orders(self):
            return [{"symbol": "AAAUSDT", "clientOrderId": "web_manual_123"}]

        def cancel_order(self, symbol, orig_client_order_id):
            self.cancels.append(orig_client_order_id)
            return {}

        def query_order(self, symbol, orig_client_order_id):
            return {"status": "CANCELED", "side": "BUY", "avgPrice": "100", "executedQty": "0"}

    client = _ForeignClient()
    posted, _alerts = _live_derisk_env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "derisk_foreign")
    save_ledger(
        Path(settings.ledger_path),
        LedgerState(
            positions={"AAAUSDT": Decimal("1.0"), "BUSDT": Decimal("-1.0")},
            equity_high_water_mark=Decimal(0),
        ),
    )

    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "DEGRADED"
    assert report.reason == "foreign_open_orders"
    assert client.cancels == []
    assert [i.symbol for i in posted] == ["BUSDT"]
    state = load_ledger(Path(settings.ledger_path))
    assert state.derisk_since is not None
    assert state.derisk_reasons == ("foreign_open_orders",)


def test_disabled_derisk_mode_halts_on_foreign_orders(artifact, monkeypatch, tmp_path) -> None:
    """Disabled de-risk mode raises on foreign orders exactly as before."""

    class _ForeignClient(StubOrderClient):
        def open_orders(self):
            return [{"symbol": "AAAUSDT", "clientOrderId": "web_manual_123"}]

    client = _ForeignClient()
    _live_derisk_env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "derisk_foreign_disabled", derisk_mode_enabled=False)

    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "HALT"
    assert report.reason is not None
    assert "foreign" in report.reason.lower()


def test_live_tax_failure_does_not_halt_degraded_cycle(artifact, monkeypatch, tmp_path) -> None:
    """A LIVE tax-collection failure is fail-soft on a degraded cycle."""
    client = _BreachOrderClient()
    posted, _alerts = _live_derisk_env(monkeypatch, tmp_path, client)

    def _boom(*args, **kwargs):
        raise OSError("tax disk full")

    monkeypatch.setattr(runner_mod, "collect_and_persist_live_tax", _boom)
    settings = _live_settings(tmp_path, "derisk_tax_soft")

    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "DEGRADED"
    assert [i.symbol for i in posted] == ["AAAUSDT"]


def test_force_close_lookback_starts_after_last_operator_resync(artifact, monkeypatch, tmp_path) -> None:
    """Force closes absorbed by an operator resync are never re-offered as breach explanations."""
    from src.live.order_journal import OrderJournal

    client = _BreachOrderClient()
    client.force_orders = lambda *, start_time_ms, end_time_ms, limit=100: (
        client.force_calls.append((start_time_ms, end_time_ms, limit)) or []
    )
    _posted, _alerts = _live_derisk_env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "derisk_lookback", venue_force_close_auto_adopt=True)
    resync_at = NOW - pd.Timedelta(hours=2)
    journal = OrderJournal(Path(settings.order_journal_path))
    journal.record_fill(
        kind="operator_resync", attempt_seq=None, symbol="AAAUSDT", side="BUY",
        quantity=Decimal("1.0"), price=Decimal("100"), fee_bps=0.0, liquidity="taker",
        reason="operator_resync", filled_at=resync_at, client_order_id=None, leg_index=0,
        cumulative_executed_qty=None, simulated=False,
    )
    save_ledger(
        Path(settings.ledger_path),
        LedgerState(
            positions={"AAAUSDT": Decimal("1.0")}, equity_high_water_mark=Decimal(0),
            journal_applied_fill_seq=0, journal_recorded_fill_seq=0,
        ),
    )

    run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert client.force_calls
    assert client.force_calls[0][0] >= int(resync_at.value // 1_000_000)


def test_regressed_journal_halts_and_alerts(artifact, monkeypatch, tmp_path) -> None:
    """A journal shorter than the ledger's applied watermark halts instead of dropping later fills."""
    client = _BreachOrderClient()
    posted, alerts = _live_derisk_env(monkeypatch, tmp_path, client)
    settings = _live_settings(tmp_path, "journal_regressed")
    save_ledger(
        Path(settings.ledger_path),
        LedgerState(
            positions={"AAAUSDT": Decimal("0.5")}, equity_high_water_mark=Decimal(0),
            journal_applied_fill_seq=4, journal_recorded_fill_seq=4,
        ),
    )

    report = run_shadow_cycle(settings, DECISION_TIME, artifact, now=NOW)

    assert report.status == "HALT"
    assert report.reason is not None and "order journal regressed" in report.reason
    assert posted == []
    assert [event for event, _ in alerts].count("order_journal_regressed") == 1


def test_notify_event_discriminator_keeps_same_cycle_alerts_distinct(monkeypatch) -> None:
    """Two same-event alerts of one cycle get distinct dedupe keys so neither is dropped."""
    keys: list[str] = []
    monkeypatch.setattr(
        runner_mod, "dispatch_alert",
        lambda settings, *, event, detail, decision_time, dedupe_key, now: keys.append(dedupe_key) or True,
    )
    settings = LiveSettings(mode="paper")
    for code in (-4164, -4140):
        runner_mod._notify_event(
            settings, event="intent_reject_cluster", detail="x", decision_time=DECISION_TIME, now=NOW,
            discriminator=f"code={code}",
        )
    assert len(set(keys)) == 2
