"""PAPER tape-through fill invariants for the resting GTX simulator."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from src.live.audit import AuditLog
from src.live.executor import (
    PAPER_TAPE_ACK_DELAY_MS,
    PassiveExecutionPolicy,
    _simulate_paper_fill,
    execute_intents,
)
from src.live.filters import SymbolFilters
from src.live.order_journal import OrderJournal
from src.live.planner import OrderIntent
from src.live.rest import PaperResponse


class _PaperBookClient:
    def __init__(self, touches: list[tuple[str, str]]) -> None:
        self._touches = list(touches)
        self._cursor = 0

    def book_tickers(self):
        idx = min(self._cursor, len(self._touches) - 1)
        self._cursor += 1
        bid, ask = self._touches[idx]
        return {"AAAUSDT": {"bidPrice": bid, "askPrice": ask}}

    def new_order(self, params: dict[str, Any]) -> Any:
        return PaperResponse.suppressed("POST", "/fapi/v1/order", "0" * 12)


class _FakeTape:
    def __init__(self, prints: list[tuple[int, float, bool]], live: bool = True) -> None:
        self._prints = list(prints)
        self._live = live

    def traded_through(self, symbol: str, side: str, limit: Decimal, since_ms: int):
        threshold = float(limit)
        for transact_ms, price, buyer_maker in self._prints:
            if transact_ms < since_ms:
                continue
            if side == "BUY" and buyer_maker and price < threshold:
                return transact_ms
            if side == "SELL" and not buyer_maker and price > threshold:
                return transact_ms
        return None

    def is_live(self, symbol: str, now_ms: int) -> bool:
        return self._live


def _intent() -> OrderIntent:
    return OrderIntent(
        symbol="AAAUSDT", side="BUY", quantity=Decimal("1.000"), reduce_only=False,
        target_qty=Decimal("1.000"), current_qty=Decimal("0"),
        client_order_prefix="run1", leg_index=0, decision_price=Decimal("100"),
    )


def _filters() -> SymbolFilters:
    return SymbolFilters(
        symbol="AAAUSDT", tick_size=Decimal("0.10"), step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"), min_notional=Decimal("5"),
        max_qty=Decimal("1000000"), quantity_precision=3, price_precision=2,
    )


def _anchored() -> PassiveExecutionPolicy:
    return PassiveExecutionPolicy(
        poll_interval_s=3.0, passive_deadline_s=50.0, window_deadline_s=600.0,
        taker_cap_bps=15.0, max_slices=1, passive_pricing="anchored",
    )


def _attempt(journal: OrderJournal):
    import pandas as pd

    now = pd.Timestamp("2026-09-14T00:00:00Z")
    return journal.begin_attempt(
        decision_time=now, run_id="test", mode="PAPER", pre_trade_equity=Decimal("1000"),
        sizing_anchor="equity", decision_marks={}, started_at=now,
    )


class _Clock:
    def __init__(self, start: float, step: float) -> None:
        self._t = start - step
        self._step = step

    def __call__(self) -> float:
        self._t += self._step
        return self._t


def _events(audit_path):
    import json as _json

    return [_json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]


def test_tape_through_print_fills_resting_gtx_touch_misses(tmp_path) -> None:
    client = _PaperBookClient([("100.00", "100.20")] * 30)
    audit_path = tmp_path / "tape_fill.jsonl"
    journal = OrderJournal(tmp_path / "j.jsonl")
    attempt = _attempt(journal)
    tape = _FakeTape([(1_007_000, 99.9, True)], live=True)
    outcomes = execute_intents(
        client, [_intent()], {"AAAUSDT": _filters()}, _anchored(),
        AuditLog(audit_path), _Clock(1000.0, 3.0), lambda s: None,
        journal=journal, attempt=attempt, paper_tape=tape,
    )
    assert outcomes[0].status == "FILLED"
    assert outcomes[0].fills[0][1] == Decimal("100.0")
    assert outcomes[0].fills[0][4] == "maker"
    fills = journal.fills_after(-1)
    assert len(fills) == 1
    assert fills[0].price == Decimal("100.0")


def test_ack_delay_excludes_prints_at_post_time(tmp_path) -> None:
    from src.live.executor import _IntentRuntime

    rt = _IntentRuntime(intent=_intent(), filters=_filters(),
                        fee_schedule=_anchored().fee_schedule)
    rt.posted_at = 1000.0
    tape = _FakeTape([(1_000_200, 99.9, True)], live=True)
    touch = (Decimal("100.00"), Decimal("100.20"))
    assert PAPER_TAPE_ACK_DELAY_MS == 50
    assert _simulate_paper_fill(rt, touch, "GTX", Decimal("100.0"), Decimal("1"), tape=tape, now=1003.0) == Decimal("1")
    early = _FakeTape([(1_000_000 + 10, 99.9, True)], live=True)
    assert _simulate_paper_fill(rt, touch, "GTX", Decimal("100.0"), Decimal("1"), tape=early, now=1000.0 + 0.01) == Decimal("0")


def test_repeg_resets_eligibility_window(tmp_path) -> None:
    client = _PaperBookClient([("100.00", "100.20")] * 300)
    tape = _FakeTape([(1_000_200, 99.9, True)], live=True)
    outcomes = execute_intents(
        client, [_intent()], {"AAAUSDT": _filters()},
        PassiveExecutionPolicy(
            poll_interval_s=3.0, passive_deadline_s=50.0, window_deadline_s=600.0,
            taker_cap_bps=15.0, max_slices=1, passive_pricing="anchored_repeg",
            repeg_interval_s=3.0,
        ),
        AuditLog(tmp_path / "repeg.jsonl"), _Clock(2000.0, 3.0), lambda s: None,
        paper_tape=tape,
    )
    assert not any(fill[4] == "maker" and fill[1] == Decimal("100.0") for fill in outcomes[0].fills)


def test_degraded_tape_falls_back_to_touch_rule(tmp_path) -> None:
    client = _PaperBookClient([("100.00", "100.20")] * 300)
    audit_path = tmp_path / "degraded.jsonl"
    tape = _FakeTape([(1_000_200, 99.9, True)], live=False)
    outcomes = execute_intents(
        client, [_intent()], {"AAAUSDT": _filters()}, _anchored(),
        AuditLog(audit_path), _Clock(1000.0, 3.0), lambda s: None,
        paper_tape=tape,
    )
    assert not any(fill[4] == "maker" for fill in outcomes[0].fills)
    degraded = [e for e in _events(audit_path) if e["event"] == "paper_tape_degraded"]
    assert len(degraded) == 1
    assert degraded[0]["symbol"] == "AAAUSDT"


def test_no_tape_reproduces_legacy_paper_fills(tmp_path) -> None:
    touches = [("100.00", "100.20")] * 300
    first = execute_intents(
        _PaperBookClient(touches), [_intent()], {"AAAUSDT": _filters()}, _anchored(),
        AuditLog(tmp_path / "legacy_a.jsonl"), _Clock(1000.0, 3.0), lambda s: None,
    )
    dead = _FakeTape([], live=False)
    second = execute_intents(
        _PaperBookClient(touches), [_intent()], {"AAAUSDT": _filters()}, _anchored(),
        AuditLog(tmp_path / "legacy_b.jsonl"), _Clock(1000.0, 3.0), lambda s: None,
        paper_tape=dead,
    )
    assert [(o.status, o.filled_qty) for o in first] == [(o.status, o.filled_qty) for o in second]
    assert [(o.maker_qty, o.taker_qty) for o in first] == [(o.maker_qty, o.taker_qty) for o in second]


class _RaisingTape(_FakeTape):
    def __init__(self, where: str) -> None:
        super().__init__([], live=True)
        self._where = where

    def is_live(self, symbol: str, now_ms: int) -> bool:
        if self._where == "is_live":
            raise RuntimeError("tape boom")
        return True

    def traded_through(self, symbol: str, side: str, limit, since_ms: int):
        if self._where == "traded_through":
            raise RuntimeError("tape boom")
        return super().traded_through(symbol, side, limit, since_ms)


def test_evil_tape_never_breaks_simulator(tmp_path) -> None:
    from src.live.executor import _IntentRuntime

    rt = _IntentRuntime(intent=_intent(), filters=_filters(),
                        fee_schedule=_anchored().fee_schedule)
    rt.posted_at = 1000.0
    touch = (Decimal("100.00"), Decimal("100.20"))
    assert _simulate_paper_fill(
        rt, touch, "GTX", Decimal("100.0"), Decimal("1"),
        tape=_RaisingTape("is_live"), now=1003.0,
    ) == Decimal("0")
    assert _simulate_paper_fill(
        rt, touch, "GTX", Decimal("100.0"), Decimal("1"),
        tape=_RaisingTape("traded_through"), now=1003.0,
    ) == Decimal("0")


def test_evil_tape_degrades_to_touch_rule_without_audit_spam(tmp_path) -> None:
    client = _PaperBookClient([("100.00", "100.20")] * 300)
    audit_path = tmp_path / "evil.jsonl"
    outcomes = execute_intents(
        client, [_intent()], {"AAAUSDT": _filters()}, _anchored(),
        AuditLog(audit_path), _Clock(1000.0, 3.0), lambda s: None,
        paper_tape=_RaisingTape("is_live"),
    )
    assert not any(fill[4] == "maker" for fill in outcomes[0].fills)
    degraded = [e for e in _events(audit_path) if e["event"] == "paper_tape_degraded"]
    assert len(degraded) == 1


def test_resting_fill_without_degraded_tracker_stays_conservative(tmp_path) -> None:
    from src.live.executor import _IntentRuntime, _paper_resting_fill

    rt = _IntentRuntime(intent=_intent(), filters=_filters(),
                        fee_schedule=_anchored().fee_schedule)
    rt.posted_at = 1000.0
    rt.active_price = Decimal("100.0")
    audit = AuditLog(tmp_path / "resting.jsonl")
    touch = (Decimal("100.00"), Decimal("100.20"))
    assert _paper_resting_fill(
        rt, touch, Decimal("1"), now=1006.0, audit=audit,
        paper_tape=_FakeTape([], live=False),
    ) == Decimal("0")
