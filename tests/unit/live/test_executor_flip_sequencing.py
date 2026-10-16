"""Flip leg sequencing part 1: executor gate, window budget, outcome reason."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest
import pandas as pd

from src.live.errors import OrderObsolete, VenueError
from src.live.executor import (
    OUTCOME_STATUSES,
    ExecutionInterrupted,
    ExecutionOutcome,
    FeeSchedule,
    PassiveExecutionPolicy,
    _IntentRuntime,
    _effective_window_deadline_s,
    _gate_state,
    _gated_passive_deadline_s,
    _link_close_dependencies,
    _poll_or_post,
    _run_loop,
    _shutdown_finalize,
    execute_intents,
)
from src.live.filters import SymbolFilters
from src.live.planner import OrderIntent
from src.live.order_journal import OrderJournal
from src.live.rest import PaperResponse, ShadowResponse


class MemAudit:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, event: str, **fields: Any) -> None:
        self.events.append({"event": event, **fields})

    def of(self, event: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["event"] == event]


class SteppingClock:
    def __init__(self, step: float) -> None:
        self._step = step
        self._t = -step

    def __call__(self) -> float:
        self._t += self._step
        return self._t


class FlipVenue:
    """Net-position tracker: plain orders move inventory, flat reduce-only is obsolete."""

    def __init__(
        self,
        position: dict[str, Decimal] | None = None,
        *,
        fill_after: int | dict[tuple[str, bool], int | None] | None = 1,
        post_reject: Any = None,
        fill_open_on_post: bool = False,
    ) -> None:
        self.position = dict(position or {})
        self.orders: list[dict[str, Any]] = []
        self._polls: dict[str, int] = {}
        self._meta: dict[str, tuple[str, str, Decimal]] = {}
        self._settled: set[str] = set()
        self._canceled: set[str] = set()
        self._expired: set[str] = set()
        self.events: list[tuple[str, str]] = []
        self._fill_open_on_post = fill_open_on_post
        self._fill_after = fill_after
        self._post_reject = post_reject

    def book_ticker(self, symbol: str) -> dict[str, str]:
        return {"bidPrice": "100.00", "askPrice": "100.20"}

    def new_order(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._post_reject is not None:
            exc = self._post_reject(params)
            if exc is not None:
                raise exc
        if params.get("reduceOnly") == "true" and self.position.get(params["symbol"], Decimal("0")) == 0:
            raise OrderObsolete("flat")
        oid = str(params["newClientOrderId"])
        self.orders.append(params)
        self._polls[oid] = 0
        self._meta[oid] = (params["symbol"], params["side"], Decimal(params["quantity"]))
        self.events.append(("post", oid))
        if self._fill_open_on_post and params.get("reduceOnly") != "true":
            self._fill(oid)
        return {"orderId": len(self.orders)}

    def _fill(self, oid: str) -> None:
        sym, side, qty = self._meta[oid]
        self._settled.add(oid)
        self.position[sym] = self.position.get(sym, Decimal("0")) + (qty if side == "BUY" else -qty)
        self.events.append(("fill", oid))

    def _need(self, symbol: str, params: dict[str, Any]) -> int | None:
        if isinstance(self._fill_after, dict):
            return self._fill_after.get((symbol, params.get("reduceOnly") == "true"), 1)
        return self._fill_after

    def query_order(self, symbol: str, oid: str) -> dict[str, Any]:
        params = next(o for o in self.orders if str(o["newClientOrderId"]) == oid)
        sym, side, qty = self._meta[oid]
        now_ms = int(time.time() * 1000)
        if oid in self._settled:
            return {"status": "FILLED", "executedQty": str(qty), "avgPrice": "100.10", "time": now_ms}
        position = self.position.get(sym, Decimal("0"))
        if params.get("reduceOnly") == "true" and (
            abs(position) < qty or (position > 0) == (side == "BUY")
        ):
            self._expired.add(oid)
        if oid in self._expired:
            return {"status": "EXPIRED", "executedQty": "0", "time": now_ms}
        if oid in self._canceled:
            return {"status": "CANCELED", "executedQty": "0", "time": now_ms}
        self._polls[oid] += 1
        need = self._need(sym, params)
        if need is not None and self._polls[oid] >= need:
            self._fill(oid)
            return {"status": "FILLED", "executedQty": str(qty), "avgPrice": "100.10", "time": now_ms}
        return {"status": "NEW", "executedQty": "0", "time": now_ms}

    def cancel_order(self, symbol: str, oid: str) -> dict[str, Any]:
        self._canceled.add(oid)
        return {"status": "CANCELED"}

    def posts(self, *, reduce_only: bool | None = None) -> list[dict[str, Any]]:
        if reduce_only is None:
            return list(self.orders)
        want = "true" if reduce_only else None
        return [o for o in self.orders if (o.get("reduceOnly") == "true") == (want == "true")]


def _filters(symbol: str) -> SymbolFilters:
    return SymbolFilters(
        symbol=symbol,
        tick_size=Decimal("0.10"),
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        min_notional=Decimal("5"),
        max_qty=Decimal("1000000"),
        quantity_precision=3,
        price_precision=2,
    )


def _policy(**overrides: Any) -> PassiveExecutionPolicy:
    base: dict[str, Any] = {
        "poll_interval_s": 3.0,
        "passive_deadline_s": 30.0,
        "window_deadline_s": 120.0,
        "max_ioc_attempts": 3,
        "max_slices": 1,
    }
    base.update(overrides)
    return PassiveExecutionPolicy(**base)


def _flip_intents(symbol: str = "AAAUSDT", qty: str = "1.000") -> list[OrderIntent]:
    q = Decimal(qty)
    return [
        OrderIntent(symbol, "BUY", q, True, Decimal(qty), -q, "flip", 0, Decimal("100.00")),
        OrderIntent(symbol, "BUY", q, False, Decimal(qty), -q, "flip", 1, Decimal("100.00")),
    ]


def _rt(intent: OrderIntent) -> _IntentRuntime:
    return _IntentRuntime(
        intent=intent,
        filters=_filters(intent.symbol),
        fee_schedule=FeeSchedule(2.0, 5.0),
    )


def test_open_leg_waits_for_close_leg() -> None:
    venue = FlipVenue({"AAAUSDT": Decimal("-1")})
    audit = MemAudit()
    outcomes = execute_intents(
        venue, _flip_intents(), {"AAAUSDT": _filters("AAAUSDT")},
        _policy(), audit, SteppingClock(3.0), lambda s: None,
    )
    assert [o.status for o in outcomes] == ["FILLED", "FILLED"]
    assert venue.position["AAAUSDT"] == Decimal("1")
    first_open = next(i for i, o in enumerate(venue.orders) if o.get("reduceOnly") != "true")
    assert first_open > 0
    assert venue.orders[0].get("reduceOnly") == "true"
    assert audit.of("flip_open_waiting")


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_historical_starvation_is_gone(side: str) -> None:
    direction = Decimal("1") if side == "BUY" else Decimal("-1")
    close, open_ = [
        replace(i, side=side, current_qty=-direction, target_qty=direction, decision_price=Decimal("100.10"))
        for i in _flip_intents()
    ]
    unlinked = [_rt(close), _rt(open_)]
    baseline = FlipVenue({"AAAUSDT": -direction}, fill_open_on_post=True)
    _run_loop(baseline, unlinked, _policy(), MemAudit(), SteppingClock(3.0), lambda s: None, None)
    assert baseline.position["AAAUSDT"] == 0
    assert baseline._expired
    assert unlinked[0].terminal_status == "OBSOLETE"
    venue = FlipVenue({"AAAUSDT": -direction}, fill_open_on_post=True)
    audit = MemAudit()
    outcomes = execute_intents(
        venue, [close, open_], {"AAAUSDT": _filters("AAAUSDT")},
        _policy(), audit, SteppingClock(3.0), lambda s: None,
    )
    assert venue.position["AAAUSDT"] == direction
    assert all(o.status == "FILLED" for o in outcomes)
    assert not venue._expired
    close_id, open_id = [o["newClientOrderId"] for o in venue.orders]
    assert venue.events.index(("fill", close_id)) < venue.events.index(("post", open_id))


def test_unfilled_close_leg_blocks_open_leg() -> None:
    venue = FlipVenue({"AAAUSDT": Decimal("-1")}, fill_after=None)
    audit = MemAudit()
    outcomes = execute_intents(
        venue, _flip_intents(), {"AAAUSDT": _filters("AAAUSDT")},
        _policy(passive_deadline_s=6.0, window_deadline_s=60.0, max_ioc_attempts=2),
        audit, SteppingClock(3.0), lambda s: None,
    )
    assert outcomes[0].status == "RESIDUAL"
    assert outcomes[1].status == "RESIDUAL"
    assert outcomes[1].block_reason == "blocked_by_close_leg"
    assert venue.posts(reduce_only=False) == []
    assert audit.of("flip_open_blocked")


def test_rejected_close_leg_blocks_open_leg() -> None:
    def reject(params: dict[str, Any]) -> Any:
        if params.get("reduceOnly") == "true":
            return VenueError("reduce-only failed", code=-4118, http_status=400,
                              path="/fapi/v1/order", payload_digest="0" * 12)
        return None

    venue = FlipVenue({"AAAUSDT": Decimal("-1")}, post_reject=reject)
    audit = MemAudit()
    outcomes = execute_intents(
        venue, _flip_intents(), {"AAAUSDT": _filters("AAAUSDT")},
        _policy(), audit, SteppingClock(3.0), lambda s: None,
    )
    assert outcomes[0].status == "REJECTED"
    assert outcomes[1].status == "RESIDUAL"
    assert outcomes[1].block_reason == "blocked_by_close_leg"
    assert venue.posts(reduce_only=False) == []
    assert audit.of("flip_open_blocked")


def test_obsolete_close_leg_unlocks_open_leg() -> None:
    venue = FlipVenue({"AAAUSDT": Decimal("0")})
    audit = MemAudit()
    outcomes = execute_intents(
        venue, _flip_intents(), {"AAAUSDT": _filters("AAAUSDT")},
        _policy(), audit, SteppingClock(3.0), lambda s: None,
    )
    assert outcomes[0].status == "OBSOLETE"
    assert outcomes[1].status == "FILLED"
    assert outcomes[1].block_reason is None
    assert venue.position["AAAUSDT"] == Decimal("1")


def test_sub_minimum_close_leg_unlocks_open_leg() -> None:
    tiny = OrderIntent("AAAUSDT", "BUY", Decimal("0.0001"), True,
                       Decimal("1"), Decimal("-1"), "flip", 0, Decimal("100.00"))
    open_ = OrderIntent("AAAUSDT", "BUY", Decimal("1"), False,
                        Decimal("1"), Decimal("-1"), "flip", 1, Decimal("100.00"))
    venue = FlipVenue({"AAAUSDT": Decimal("-1")})
    audit = MemAudit()
    outcomes = execute_intents(
        venue, [tiny, open_], {"AAAUSDT": _filters("AAAUSDT")},
        _policy(), audit, SteppingClock(3.0), lambda s: None,
    )
    assert outcomes[0].status == "RESIDUAL_SUB_MINIMUM"
    assert outcomes[1].status == "FILLED"


def test_waiting_leg_has_no_clock_and_no_venue_call() -> None:
    venue = FlipVenue({"AAAUSDT": Decimal("-1")})
    audit = MemAudit()
    policy = _policy()
    close_rt, open_rt = _rt(_flip_intents()[0]), _rt(_flip_intents()[1])
    _link_close_dependencies([close_rt, open_rt])
    touch = (Decimal("100.00"), Decimal("100.20"))
    _poll_or_post(venue, open_rt, touch, 3.0, policy, audit)
    assert venue.orders == []
    assert open_rt.passive_started_at == 0.0
    assert len(audit.of("flip_open_waiting")) == 1
    _poll_or_post(venue, open_rt, touch, 6.0, policy, audit)
    assert venue.orders == []
    assert len(audit.of("flip_open_waiting")) == 1
    close_rt.terminal_status = "FILLED"
    close_rt.finalized_at = 6.0
    _poll_or_post(venue, open_rt, touch, 9.0, policy, audit)
    assert len(venue.orders) == 1
    assert open_rt.passive_started_at == 9.0


def test_window_extends_only_for_gated_cycles() -> None:
    policy = _policy(poll_interval_s=3.0, passive_deadline_s=18.0, window_deadline_s=30.0)
    plain = [_rt(OrderIntent("AAAUSDT", "BUY", Decimal("1"), False,
                             Decimal("1"), Decimal("0"), "p", 0, Decimal("100.00")))]
    flip = [_rt(i) for i in _flip_intents()]
    _link_close_dependencies(plain)
    _link_close_dependencies(flip)
    assert _effective_window_deadline_s(policy, plain) == 30.0
    assert _effective_window_deadline_s(policy, flip) == 48.0
    plain_ticks = math.ceil(30.0 / 3.0) + 1
    flip_ticks = math.ceil(48.0 / 3.0) + 1

    def run(intents: list[OrderIntent]) -> int:
        venue = FlipVenue({"AAAUSDT": Decimal("-1")}, fill_after=None)
        sleeps: list[float] = []
        execute_intents(venue, intents, {"AAAUSDT": _filters("AAAUSDT")},
                        policy, MemAudit(), lambda: 0.0, sleeps.append)
        return len(sleeps)

    assert run([plain[0].intent]) == plain_ticks
    assert run([r.intent for r in flip]) == flip_ticks


def test_ioc_reserve_survives_hard_stop() -> None:
    policy = _policy(window_deadline_s=60.0, passive_deadline_s=30.0)
    open_rt = _rt(_flip_intents()[1])
    close_rt = _rt(_flip_intents()[0])
    _link_close_dependencies([close_rt, open_rt])
    open_rt.hard_stop_at = 100.0
    open_rt.passive_started_at = 60.0
    assert _gated_passive_deadline_s(open_rt, policy) == pytest.approx(10.0)
    open_rt.passive_started_at = 95.0
    assert _gated_passive_deadline_s(open_rt, policy) == 0.0
    venue = FlipVenue({"AAAUSDT": Decimal("0")})
    audit = MemAudit()
    close_rt.terminal_status = "FILLED"
    open_rt.passive_started_at = 0.0
    _poll_or_post(venue, open_rt, (Decimal("100.00"), Decimal("100.20")),
                  95.0, policy, audit)
    assert open_rt.phase == "ioc"
    assert venue.orders
    assert venue.orders[-1]["timeInForce"] == "IOC"


def test_other_symbols_are_not_delayed() -> None:
    sym_b = OrderIntent("BBBUSDT", "BUY", Decimal("1"), False,
                        Decimal("1"), Decimal("0"), "p", 0, Decimal("100.00"))
    venue = FlipVenue({"AAAUSDT": Decimal("-1"), "BBBUSDT": Decimal("0")})
    audit = MemAudit()
    outcomes = execute_intents(
        venue, [*_flip_intents(), sym_b],
        {"AAAUSDT": _filters("AAAUSDT"), "BBBUSDT": _filters("BBBUSDT")},
        _policy(), audit, SteppingClock(3.0), lambda s: None,
    )
    assert all(o.status == "FILLED" for o in outcomes)
    first_b = next(i for i, o in enumerate(venue.orders) if o["symbol"] == "BBBUSDT")
    first_open = next(i for i, o in enumerate(venue.orders)
                      if o["symbol"] == "AAAUSDT" and o.get("reduceOnly") != "true")
    assert first_b < first_open


def test_lone_close_leg_has_no_dependency() -> None:
    venue = FlipVenue({"AAAUSDT": Decimal("1")})
    close = OrderIntent("AAAUSDT", "SELL", Decimal("1"), True,
                        Decimal("0"), Decimal("1"), "solo", 0, Decimal("100.20"))
    audit = MemAudit()
    outcomes = execute_intents(
        venue, [close], {"AAAUSDT": _filters("AAAUSDT")},
        _policy(), audit, SteppingClock(3.0), lambda s: None,
    )
    assert outcomes[0].status == "FILLED"
    assert not audit.of("flip_open_waiting")
    assert not audit.of("flip_open_blocked")


def test_shutdown_while_waiting() -> None:
    venue = FlipVenue({"AAAUSDT": Decimal("-1")}, fill_after=None)
    audit = MemAudit()

    class Shutdown:
        requested = False

    shutdown = Shutdown()

    def sleep(s: float) -> None:
        shutdown.requested = True

    with pytest.raises(ExecutionInterrupted) as exc:
        execute_intents(
            venue, _flip_intents(), {"AAAUSDT": _filters("AAAUSDT")},
            _policy(), audit, SteppingClock(3.0), sleep, shutdown=shutdown,
        )
    partials = exc.value.partial_outcomes
    assert partials is not None
    assert len(partials) == 2
    assert partials[1].status == "RESIDUAL"
    assert partials[1].block_reason is None
    assert venue.posts(reduce_only=False) == []


def test_immediate_taker_paper_path_unchanged() -> None:
    venue = FlipVenue({"AAAUSDT": Decimal("-1")})
    audit = MemAudit()
    outcomes = execute_intents(
        venue, _flip_intents(), {"AAAUSDT": _filters("AAAUSDT")},
        _policy(), audit, lambda: 0.0, lambda s: None,
        paper_fill_model="immediate_taker",
    )
    assert [o.status for o in outcomes] == ["FILLED", "FILLED"]
    assert not audit.of("flip_open_waiting")
    assert not audit.of("flip_open_blocked")


def test_outcome_carries_block_reason() -> None:
    venue = FlipVenue({"AAAUSDT": Decimal("-1")}, fill_after=None)
    audit = MemAudit()
    outcomes = execute_intents(
        venue, _flip_intents(), {"AAAUSDT": _filters("AAAUSDT")},
        _policy(passive_deadline_s=6.0, window_deadline_s=60.0, max_ioc_attempts=2),
        audit, SteppingClock(3.0), lambda s: None,
    )
    blocked = outcomes[1]
    assert isinstance(blocked, ExecutionOutcome)
    assert blocked.block_reason == "blocked_by_close_leg"
    assert blocked.status in OUTCOME_STATUSES
    intent_outcomes = audit.of("intent_outcome")
    by_symbol = [e for e in intent_outcomes if e.get("block_reason") == "blocked_by_close_leg"]
    assert len(by_symbol) == 1


def test_linking_and_gate_states() -> None:
    a0 = OrderIntent("AAAUSDT", "BUY", Decimal("1"), True, Decimal("1"), Decimal("-1"), "f", 0, Decimal("100"))
    a1 = OrderIntent("AAAUSDT", "BUY", Decimal("1"), True, Decimal("1"), Decimal("-1"), "f", 1, Decimal("100"))
    a2 = OrderIntent("AAAUSDT", "BUY", Decimal("1"), False, Decimal("1"), Decimal("-1"), "f", 2, Decimal("100"))
    b0 = OrderIntent("BBBUSDT", "BUY", Decimal("1"), False, Decimal("1"), Decimal("0"), "p", 0, Decimal("100"))
    rts = [_rt(a0), _rt(a1), _rt(a2), _rt(b0)]
    _link_close_dependencies(rts)
    assert rts[0].close_dependency is None
    assert rts[1].close_dependency is None
    assert rts[2].close_dependency is rts[1]
    assert rts[3].close_dependency is None
    assert _gate_state(rts[3]) == "open"
    rts[1].terminal_status = None
    assert _gate_state(rts[2]) == "wait"
    for terminal, expected in [("RESIDUAL", "blocked"), ("REJECTED", "blocked"),
                               ("FILLED", "open"), ("OBSOLETE", "open"),
                               ("RESIDUAL_SUB_MINIMUM", "open"), ("SHADOW", "open")]:
        rts[1].terminal_status = terminal
        assert _gate_state(rts[2]) == expected


@pytest.mark.parametrize("reject_close", [False, True])
def test_blocked_open_has_no_journal_submit(tmp_path, reject_close: bool) -> None:
    def reject(params: dict[str, Any]) -> VenueError | None:
        if reject_close and params.get("reduceOnly") == "true":
            return VenueError("reduce-only failed", code=-4118, http_status=400,
                              path="/fapi/v1/order", payload_digest="0" * 12)
        return None

    journal = OrderJournal(tmp_path / "orders.jsonl")
    now = pd.Timestamp.now(tz="UTC")
    attempt = journal.begin_attempt(
        decision_time=now, run_id="flip", mode="LIVE", pre_trade_equity=Decimal("1000"),
        sizing_anchor="equity", decision_marks={}, started_at=now,
    )
    venue = FlipVenue({"AAAUSDT": Decimal("-1")}, fill_after=None, post_reject=reject)
    audit = MemAudit()
    outcomes = execute_intents(
        venue, _flip_intents(), {"AAAUSDT": _filters("AAAUSDT")},
        _policy(passive_deadline_s=6.0, window_deadline_s=12.0, max_ioc_attempts=100),
        audit, SteppingClock(3.0), lambda s: None, journal=journal, attempt=attempt,
    )
    assert outcomes[1].block_reason == "blocked_by_close_leg"
    submits = journal.submits()
    assert submits
    assert all(s.leg_index == 0 for s in submits)
    assert not venue.posts(reduce_only=False)
    assert len(audit.of("flip_open_blocked")) == 1


def test_window_end_blocks_even_when_open_precedes_close() -> None:
    venue = FlipVenue({"AAAUSDT": Decimal("-1")}, fill_after=None)
    audit = MemAudit()
    outcomes = execute_intents(
        venue, list(reversed(_flip_intents())), {"AAAUSDT": _filters("AAAUSDT")},
        _policy(passive_deadline_s=6.0, window_deadline_s=12.0, max_ioc_attempts=100),
        audit, SteppingClock(3.0), lambda s: None,
    )
    assert outcomes[0].block_reason == "blocked_by_close_leg"
    assert outcomes[0].status == "RESIDUAL"
    assert not venue.posts(reduce_only=False)
    assert len(audit.of("flip_open_blocked")) == 1


@pytest.mark.parametrize("close_status", [None, "FILLED", "REJECTED"])
def test_shutdown_finalizes_waiter_without_block_reason(close_status: str | None) -> None:
    runtimes = [_rt(i) for i in _flip_intents()]
    _link_close_dependencies(runtimes)
    runtimes[0].terminal_status = close_status
    runtimes[0].finalized_at = 5.0
    unsettled = _shutdown_finalize(FlipVenue(), runtimes, MemAudit(), lambda: 10.0, 20.0)
    assert unsettled == 0
    assert runtimes[1].terminal_status == "RESIDUAL"
    assert runtimes[1].finalized_at == 10.0
    assert runtimes[1].block_reason is None
    if close_status is not None:
        assert runtimes[0].terminal_status == close_status
        assert runtimes[0].finalized_at == 5.0


@pytest.mark.parametrize("unlock_at", [150.0, 165.0])
def test_late_unlock_preserves_ioc_reserve_and_hard_stop(unlock_at: float) -> None:
    now = [100.0]
    posts: list[tuple[float, dict[str, Any]]] = []

    class LateVenue(FlipVenue):
        def _need(self, symbol: str, params: dict[str, Any]) -> int | None:
            return 1 if params.get("reduceOnly") == "true" and now[0] >= unlock_at else None

        def new_order(self, params: dict[str, Any]) -> dict[str, Any]:
            posts.append((now[0], params))
            return super().new_order(params)

    def sleep(seconds: float) -> None:
        now[0] += seconds

    venue = LateVenue({"AAAUSDT": Decimal("-1")})
    outcomes = execute_intents(
        venue, _flip_intents(), {"AAAUSDT": _filters("AAAUSDT")},
        _policy(poll_interval_s=5.0, passive_deadline_s=30.0,
                window_deadline_s=60.0, max_ioc_attempts=100),
        MemAudit(), lambda: now[0], sleep,
    )
    open_posts = [(at, p["timeInForce"]) for at, p in posts if p.get("reduceOnly") != "true"]
    assert open_posts[0] == (unlock_at, "GTX" if unlock_at == 150.0 else "IOC")
    assert next(at for at, tif in open_posts if tif == "IOC") == max(160.0, unlock_at)
    assert all(at < 190.0 for at, _ in posts)
    assert now[0] == 190.0
    assert outcomes[0].status == "FILLED"
    assert outcomes[1].status == "RESIDUAL"
    assert outcomes[1].block_reason is None


@pytest.mark.parametrize("mode", ["paper", "shadow"])
def test_cooperative_simulated_modes_respect_close_gate(mode: str) -> None:
    class SimulatedVenue(FlipVenue):
        def new_order(self, params: dict[str, Any]) -> Any:
            self.orders.append(params)
            if mode == "shadow":
                return ShadowResponse.suppressed("POST", "/fapi/v1/order", "0" * 12)
            return PaperResponse.suppressed("POST", "/fapi/v1/order", "0" * 12)

    venue = SimulatedVenue()
    audit = MemAudit()
    outcomes = execute_intents(
        venue, _flip_intents(), {"AAAUSDT": _filters("AAAUSDT")},
        _policy(passive_deadline_s=6.0, window_deadline_s=12.0),
        audit, SteppingClock(3.0), lambda s: None, paper_fill_model="peg_chase",
    )
    assert len(outcomes) == 2
    assert all(o.status == ("SHADOW" if mode == "shadow" else "FILLED") for o in outcomes)
    first_open = next(i for i, p in enumerate(venue.orders) if p.get("reduceOnly") != "true")
    assert all(p.get("reduceOnly") == "true" for p in venue.orders[:first_open])
