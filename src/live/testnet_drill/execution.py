"""Bounded drill entry to the production executor and recovery paths."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd

from src.live.errors import VenueError
from src.live.executor import (
    ExecutionOutcome,
    FeeSchedule,
    PassiveExecutionPolicy,
    execute_intents,
    strict_passive_repeg_execution_policy,
)
from src.live.filters import quantize_order
from src.live.order_cancel import cancel_and_confirm
from src.live.planner import OrderIntent
from src.live.recovery import recover_unresolved_orders, settle_unresolved_orders
from src.live.rest import parse_rate_limits
from src.live.testnet_drill.guard import DrillRefused
from src.live.testnet_drill.session import DrillSession


def execution_policy(session: DrillSession) -> PassiveExecutionPolicy:
    """Build the same strict passive re-peg policy as the live runner."""
    settings = session.settings
    return strict_passive_repeg_execution_policy(
        FeeSchedule(settings.maker_fee_bps, settings.taker_fee_bps),
        settings.taker_slippage_bps,
        settings.passive_timeout_minutes,
        settings.repeg_interval_s,
    )


def make_intent(
    session: DrillSession,
    symbol: str,
    *,
    quantity: Decimal | None = None,
    side: str = "BUY",
    reduce_only: bool = False,
    leg_index: int = 0,
) -> OrderIntent:
    """Create an exactly quantized, budgeted intent using current observable touch."""
    quote = session.client.book_ticker(symbol)
    price = Decimal(str(quote["bidPrice" if side == "BUY" else "askPrice"]))
    desired = quantity if quantity is not None else session.config.probe_notional_usdt / price
    order = quantize_order(
        session.filters[symbol], desired if side == "BUY" else -desired, price, reduce_only=reduce_only
    )
    if order is None:
        raise DrillRefused("drill refused: probe_below_venue_minimum")
    if abs(order.quantity) * order.price > session.config.max_probe_notional_usdt:
        raise DrillRefused("drill refused: intent_notional_cap")
    return OrderIntent(
        symbol,
        side,
        abs(order.quantity),
        reduce_only,
        order.quantity,
        Decimal(0),
        session.client_order_prefix.removeprefix("mh"),
        leg_index,
        order.price,
    )


def settle_journal(session: DrillSession) -> None:
    """Settle durable submissions through production recovery without venue-based repair."""
    recovery = recover_unresolved_orders(
        session.client,
        session.journal,
        session.audit,
        now=pd.Timestamp.now(tz="UTC"),
        lookback=pd.Timedelta(hours=session.settings.journal_recovery_lookback_hours),
        taker_fee_bps=session.settings.taker_fee_bps,
    )
    settled = settle_unresolved_orders(
        session.client,
        session.journal,
        session.audit,
        recovery.unresolved,
        taker_fee_bps=session.settings.taker_fee_bps,
    )
    if settled.still_open:
        raise DrillRefused("drill refused: unsettled_orders")


def submit_probe(
    session: DrillSession,
    params: Mapping[str, Any],
    *,
    raw: bool = False,
    before_send: Callable[[], None] | None = None,
) -> Any:
    """Journal a direct probe before guarded venue delivery."""
    order = dict(params)
    order.setdefault("newClientOrderId", session.next_client_order_id(str(order["symbol"])))
    order_id = str(order["newClientOrderId"])
    if any(submit.client_order_id == order_id for submit in session.journal.submits()):
        if before_send is not None:
            before_send()
        return session.guarded_new_order(order, raw=raw)
    now = pd.Timestamp.now(tz="UTC")
    attempt = session.journal.begin_attempt(
        decision_time=now,
        run_id=session.run_id,
        mode=session.settings.mode.value,
        pre_trade_equity=Decimal(str(session.client.account()["totalWalletBalance"])),
        sizing_anchor="testnet_drill",
        decision_marks={},
        started_at=now,
    )
    session.journal.record_submit(
        order_id,
        str(order["symbol"]),
        session.journal.next_submit_seq(),
        attempt_seq=attempt.attempt_seq,
        side=str(order["side"]),
        quantity=Decimal(str(order["quantity"])),
        reduce_only=str(order.get("reduceOnly", "false")).lower() == "true",
        leg_index=0,
    )
    try:
        if before_send is not None:
            before_send()
        return session.guarded_new_order(order, raw=raw)
    except (VenueError, DrillRefused):
        session.journal.record_terminal(order_id, "NOT_PLACED")
        raise


def cancel_probe(session: DrillSession, symbol: str, order_id: str) -> Any:
    """Confirm cancellation and settle all executed deltas through production recovery."""
    confirmation = cancel_and_confirm(session.client, symbol, order_id)
    settle_journal(session)
    return confirmation


@dataclass(slots=True)
class CycleEvidence:
    """Recorded executor outcomes and per-tick venue observations."""

    intents: tuple[OrderIntent, ...]
    policy: PassiveExecutionPolicy
    outcomes: tuple[ExecutionOutcome, ...] = ()
    escaped: str | None = None
    open_counts: list[int] = field(default_factory=list)
    sleeps: list[float] = field(default_factory=list)
    wire_start: int = 0
    wire_end: int = 0
    post_quotes: list[dict[str, Any]] = field(default_factory=list)
    submit_start: int = 0


class _GuardedClient:
    def __init__(self, session: DrillSession, trace: CycleEvidence, clock: Callable[[], float]) -> None:
        self.session = session
        self.trace = trace
        self.clock = clock

    def __getattr__(self, name: str) -> Any:
        return getattr(self.session.client, name)

    def new_order(self, params: Mapping[str, Any]) -> Any:
        # Observed via depth so bookTicker fault rules only ever hit the executor's own reads.
        book = self.session.client.depth(str(params["symbol"]), limit=5)
        self.trace.post_quotes.append(
            {**dict(params), "bid": book["bids"][0][0], "ask": book["asks"][0][0], "observed_at": self.clock()}
        )
        return self.session.guarded_new_order(params)


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """One audit event with the wire position and clock reading at the moment it was written."""

    wire_index: int
    at: float
    event: str
    fields: Mapping[str, Any]


class AuditTap:
    """Record audit events in order against the wire log while preserving the durable audit write."""

    def __init__(self, session: DrillSession, clock: Callable[[], float] | None = None) -> None:
        self.events: list[AuditEvent] = []
        self._session = session
        self._clock = clock if clock is not None else (lambda: time.time())
        self._original: Any = None

    def __enter__(self) -> AuditTap:
        audit = self._session.audit
        self._original = audit.record

        def record(event: str, **fields: Any) -> Any:
            result = self._original(event, **fields)
            self.events.append(AuditEvent(len(self._session.transport.records), self._clock(), event, dict(fields)))
            return result

        setattr(audit, "record", record)  # noqa: B010 - instance-level tap, restored on exit
        return self

    def __exit__(self, *exc: object) -> None:
        setattr(self._session.audit, "record", self._original)  # noqa: B010 - restore the tapped method

    def named(self, event: str) -> list[AuditEvent]:
        """Events with the given name, in write order."""
        return [item for item in self.events if item.event == event]


def run_cycle(
    session: DrillSession,
    intents: Sequence[OrderIntent],
    *,
    on_tick: Callable[[int], None] | None = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    stop: Callable[[], bool] | None = None,
) -> CycleEvidence:
    """Run bounded production execution with guarded posts and recorded tick samples."""
    clock_fn = clock if clock is not None else time.time
    sleep_fn = sleep if sleep is not None else time.sleep
    policy = execution_policy(session)
    trace = CycleEvidence(tuple(intents), policy, wire_start=len(session.transport.records),
                          submit_start=session.journal.next_submit_seq())
    start = clock_fn()

    class Deadline:
        @property
        def requested(self) -> bool:
            return clock_fn() - start >= session.config.scenario_timeout_s or (stop is not None and stop())

    def tick_wait(duration: float) -> None:
        rows = session.client.open_orders()
        submits = {s.client_order_id: (s.symbol, s.leg_index or 0) for s in session.journal.submits()}
        counts: dict[tuple[str, int], int] = {}
        for row in rows:
            key = submits.get(str(row["clientOrderId"]))
            if key is not None:
                counts[key] = counts.get(key, 0) + 1
        trace.open_counts.append(max(counts.values(), default=0))
        if on_tick is not None:
            on_tick(len(trace.sleeps))
        trace.sleeps.append(duration)
        sleep_fn(min(duration, max(0, session.config.scenario_timeout_s - (clock_fn() - start))))

    now = pd.Timestamp.now(tz="UTC")
    attempt = session.journal.begin_attempt(
        decision_time=now,
        run_id=session.run_id,
        mode=session.settings.mode.value,
        pre_trade_equity=Decimal(str(session.client.account()["totalWalletBalance"])),
        sizing_anchor="testnet_drill",
        decision_marks={it.symbol: it.decision_price for it in intents},
        started_at=now,
    )
    try:
        trace.outcomes = execute_intents(
            _GuardedClient(session, trace, clock_fn),
            intents,
            session.filters,
            policy,
            session.audit,
            clock_fn,
            tick_wait,
            rate_limits=parse_rate_limits(session.client.exchange_info()),
            journal=session.journal,
            attempt=attempt,
            shutdown=Deadline(),
            shutdown_cleanup_budget_s=session.config.scenario_timeout_s,
        )
    except Exception as exc:
        # Any escape is evidence (H1/H11 observe exactly this); it must not abort the scenario harness.
        trace.escaped = type(exc).__name__
        trace.outcomes = tuple(getattr(exc, "partial_outcomes", None) or ())
    finally:
        trace.wire_end = len(session.transport.records)
    return trace
