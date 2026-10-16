"""Venue-backed lifecycle experiments on a single account against natural market flow."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from decimal import Decimal
from itertools import pairwise
from typing import Any

from src.live.executor import MAX_SLICE_NOTIONAL
from src.live.planner import plan_orders
from src.live.testnet_drill.execution import (
    cancel_probe,
    make_intent,
    run_cycle,
    submit_probe,
)
from src.live.testnet_drill.guard import DrillRefused
from src.live.testnet_drill.scenarios.base import ScenarioEvidence, ev, isolated_run
from src.live.testnet_drill.session import DrillSession, _read_positions


def _limit(session: DrillSession, *, distance: bool = False) -> dict[str, Any]:
    intent = make_intent(session, session.config.symbols[0])
    price = intent.decision_price
    if distance:
        from src.live.filters import quantize_order

        order = quantize_order(
            session.filters[intent.symbol],
            intent.quantity,
            price * (1 - Decimal(session.config.resting_distance_bps) / 10000),
            reduce_only=False,
        )
        if order is None:
            raise DrillRefused("drill refused: resting_order_minimum")
        price = order.price
    return {
        "symbol": intent.symbol,
        "side": "BUY",
        "type": "LIMIT",
        "timeInForce": "GTX",
        "quantity": str(intent.quantity),
        "price": str(price),
        "newClientOrderId": session.next_client_order_id(intent.symbol),
    }


def _codes(session: DrillSession, start: int = 0) -> list[int]:
    codes = []
    for record in session.transport.records[start:]:
        if record.response_body:
            try:
                payload = json.loads(record.response_body)
            except (ValueError, UnicodeError):
                continue
            if isinstance(payload, dict) and isinstance(payload.get("code"), int) and payload["code"] < 0:
                codes.append(payload["code"])
    return codes


def _primary_orders(session: DrillSession, symbol: str) -> list[dict[str, Any]]:
    return [
        row
        for row in session.client.open_orders()
        if row["symbol"] == symbol and str(row["clientOrderId"]).startswith(session.client_order_prefix + "-")
    ]


def _partial(session: DrillSession) -> ScenarioEvidence:
    symbol = session.config.symbols[0]
    intent = make_intent(session, symbol)

    trace = run_cycle(session, [intent])
    fills = [fill for fill in session.journal.fills_after(-1) if fill.symbol == symbol and fill.kind == "execution"]
    by_id: dict[str, list[Any]] = {}
    for fill in fills:
        if fill.client_order_id is not None:
            by_id.setdefault(fill.client_order_id, []).append(fill)
    trades = session.client.user_trades(symbol)
    if len(trades) >= 1000:
        raise DrillRefused("drill refused: partial_trade_window_truncated")
    monotonic = all(
        all(a.cumulative_executed_qty < b.cumulative_executed_qty for a, b in pairwise(group))
        for group in by_id.values()
    )
    parity = True
    compared = False
    mismatch: tuple[Decimal, Decimal] | None = None
    for client_id, group in by_id.items():
        order = session.client.query_order(symbol, client_id)
        actual = sorted(
            (row for row in trades if str(row["orderId"]) == str(order["orderId"])), key=lambda row: int(row["id"])
        )
        parity &= sum((fill.quantity for fill in group), Decimal(0)) == Decimal(str(order["executedQty"]))
        if len(group) < 2 or len(actual) < 2:
            continue
        compared = True
        consumed = Decimal(0)
        for fill in group:
            end = consumed + fill.quantity
            position = Decimal(0)
            notional = Decimal(0)
            for row in actual:
                amount = Decimal(str(row["qty"]))
                overlap = max(Decimal(0), min(end, position + amount) - max(consumed, position))
                notional += overlap * Decimal(str(row["price"]))
                position += amount
            price = notional / fill.quantity
            if price > 0 and abs(fill.price - price) / price > session.config.price_rel_tolerance:
                mismatch = (fill.price, price)
            consumed = end
    delta_price, trade_price = mismatch if mismatch is not None else (Decimal(0), Decimal(0))
    return ev(
        "C5",
        cumulative_monotonic=monotonic,
        deltas_sum_to_venue=parity,
        multi_step_observed=compared,
        delta_price=str(delta_price),
        user_trade_price=str(trade_price),
        price_rel_tolerance=str(session.config.price_rel_tolerance),
        uses_cumulative_avg=mismatch is not None,
        escaped_exception=trace.escaped,
    )


def _race(session: DrillSession) -> ScenarioEvidence:
    repetitions = []
    started = time.monotonic()
    for index in range(session.config.race_repetitions):
        if time.monotonic() - started >= session.config.scenario_timeout_s:
            raise DrillRefused("drill refused: race_timeout")
        order = _limit(session)
        posted = submit_probe(session, order)
        if posted.get("status") not in ("NEW", "PARTIALLY_FILLED"):
            raise DrillRefused("drill refused: race_order_did_not_rest")
        offset = session.config.race_offsets_s[index % len(session.config.race_offsets_s)]

        time.sleep(abs(offset))
        confirmation = cancel_probe(session, order["symbol"], order["newClientOrderId"])
        query = session.client.query_order(order["symbol"], order["newClientOrderId"])
        fills = session.journal.fills_after(-1)
        keys = [
            (fill.client_order_id, fill.cumulative_executed_qty) for fill in fills if fill.client_order_id is not None
        ]
        fold = sum(
            (
                fill.quantity if fill.side == "BUY" else -fill.quantity
                for fill in fills
                if fill.symbol == order["symbol"]
            ),
            Decimal(0),
        )
        parity = _read_positions(session.client).get(
            order["symbol"], Decimal(0)
        ) == fold and session.journal.observed_qty(order["newClientOrderId"]) == Decimal(str(query["executedQty"]))
        repetitions.append(
            {
                "id": index,
                "terminal_confirmed": confirmation.closed,
                "qty_match": parity,
                "no_double": len(keys) == len(set(keys)),
                "primary_seq": len(session.transport.records),
                "executed_qty": str(query["executedQty"]),
            }
        )
    return ev("C6", repetitions=repetitions)


_REPEG_CLOCK_SLACK_S = 0.5


def _repeg_lifetimes(audit_path: Any) -> list[float]:
    """Seconds each order rested before a re-peg cancelled it, from the audit stream."""
    from datetime import datetime

    posted: dict[str, datetime] = {}
    lifetimes: list[float] = []
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        client_id = row.get("client_order_id")
        at = datetime.fromisoformat(row["ts"])
        if row.get("event") == "order_posted" and client_id is not None:
            posted[client_id] = at
        elif row.get("event") == "order_cancelled" and row.get("reason") == "repeg" and client_id in posted:
            lifetimes.append((at - posted[client_id]).total_seconds())
    return lifetimes


def _repeg(session: DrillSession) -> ScenarioEvidence:
    symbol = session.config.symbols[0]
    intent = make_intent(session, symbol)
    trace = run_cycle(session, [intent])
    lifetimes = _repeg_lifetimes(session.workspace / "audit.jsonl")
    interval_ok = all(seconds >= trace.policy.repeg_interval_s - _REPEG_CLOCK_SLACK_S for seconds in lifetimes)
    return ev(
        "C9",
        open_order_snapshots=trace.open_counts,
        cumulative_filled=str(sum((outcome.filled_qty for outcome in trace.outcomes), Decimal(0))),
        intent_qty=str(intent.quantity),
        interval_respected=interval_ok,
        repegs=len(lifetimes),
    )


def _resting_cancel(session: DrillSession) -> ScenarioEvidence:
    symbol = session.config.symbols[0]
    order = _limit(session, distance=True)
    posted = submit_probe(session, order)
    query = session.client.query_order(symbol, order["newClientOrderId"])
    confirmation = cancel_probe(session, symbol, order["newClientOrderId"])
    return ev(
        "C1",
        status_path=[query["status"], confirmation.status],
        closed=confirmation.closed,
        fills=sum(fill.client_order_id == order["newClientOrderId"] for fill in session.journal.fills_after(-1)),
        journal_submit=any(s.client_order_id == order["newClientOrderId"] for s in session.journal.submits()),
        journal_terminal=session.journal.terminal(order["newClientOrderId"]) is not None,
        posted=posted,
    )


def _opened(session: DrillSession, intents: list[Any], refusal: str) -> Any:
    opening = run_cycle(session, intents)
    if not opening.outcomes or opening.outcomes[0].filled_qty <= 0:
        raise DrillRefused(f"drill refused: {refusal}")
    return opening


def _ordinary_intents(session: DrillSession, scenario_id: str) -> list[Any]:
    symbol = session.config.symbols[0]
    intents = [make_intent(session, symbol)]
    if scenario_id == "C4":
        depth = session.client.depth(symbol)
        amount = Decimal(str(depth["asks"][0][1])) + session.filters[symbol].step_size
        return [make_intent(session, symbol, quantity=amount)]
    if scenario_id == "C8":
        _opened(session, intents, "flip_requires_open_position")
        return list(
            plan_orders(
                {symbol: -intents[0].quantity},
                _read_positions(session.client),
                session.filters,
                {symbol: intents[0].decision_price},
                session.client_order_prefix.removeprefix("mh"),
            )
        )
    if scenario_id == "C10" and session.config.probe_notional_usdt <= MAX_SLICE_NOTIONAL:
        raise DrillRefused("drill refused: slice_probe_notional_must_exceed_slice_limit")
    if scenario_id in ("C11", "C13"):
        return _multi_symbol_intents(session, scenario_id)
    if scenario_id == "C7":
        return [replace(intents[0], reduce_only=True)]
    if scenario_id == "C14":
        opening = run_cycle(session, intents)
        available = Decimal(str(session.client.account()["availableBalance"]))
        if not opening.outcomes or available > session.config.probe_notional_usdt:
            raise DrillRefused("drill refused: margin_release_requires_near_limit_account")
        return [replace(intents[0], side="SELL", reduce_only=True, leg_index=0), replace(intents[0], leg_index=1)]
    return intents


def _multi_symbol_intents(session: DrillSession, scenario_id: str) -> list[Any]:
    minimum = 2 if scenario_id == "C11" else 3
    if len(session.config.symbols) < minimum:
        raise DrillRefused("drill refused: insufficient_probe_symbols")
    intents = [make_intent(session, sym, leg_index=index) for index, sym in enumerate(session.config.symbols)]
    if scenario_id == "C11":
        return intents
    opening = _opened(session, [intents[0]], "throttle_probe_requires_reducing_intent")
    intents = [replace(intents[index % len(intents)], leg_index=index) for index in range(10)]
    intents[0] = replace(intents[0], side="SELL", reduce_only=True, quantity=opening.outcomes[0].filled_qty)
    return intents


@dataclass(frozen=True, slots=True)
class _Cycle:
    """Everything an ordinary lifecycle evidence builder reads after one production cycle."""

    session: DrillSession
    symbol: str
    intents: list[Any]
    trace: Any
    fills: tuple[Any, ...]
    posted: list[Any]
    codes: list[int]
    total: Decimal
    requested: Decimal
    exact: bool
    residual: bool
    positions: dict[str, Decimal]
    common: dict[str, Any]


def _observe_cycle(session: DrillSession, intents: list[Any]) -> _Cycle:
    trace = run_cycle(session, intents)
    outcomes = trace.outcomes
    exact = len(outcomes) == len(intents) and all(
        outcome.filled_qty + outcome.unfilled_qty == intent.quantity
        for outcome, intent in zip(outcomes, intents, strict=True)
    )
    return _Cycle(
        session=session,
        symbol=session.config.symbols[0],
        intents=intents,
        trace=trace,
        fills=session.journal.fills_after(-1),
        posted=[
            row
            for row in session.transport.records[trace.wire_start : trace.wire_end]
            if row.method == "POST" and row.path == "/fapi/v1/order"
        ],
        codes=_codes(session, trace.wire_start),
        total=sum((outcome.filled_qty for outcome in outcomes), Decimal(0)),
        requested=sum((intent.quantity for intent in intents), Decimal(0)),
        exact=exact,
        residual=exact
        and any(
            outcome.unfilled_qty > 0 and outcome.status in ("RESIDUAL", "RESIDUAL_SUB_MINIMUM", "OBSOLETE")
            for outcome in outcomes
        ),
        positions=_read_positions(session.client),
        common={"escaped_exception": trace.escaped, "wire_start": trace.wire_start, "wire_end": trace.wire_end},
    )


def _evidence_c2(c: _Cycle) -> ScenarioEvidence:
    return ev(
        "C2",
        **c.common,
        code=next((code for code in c.codes if code == -5022), None),
        chases=sum(outcome.chases for outcome in c.trace.outcomes),
        window=c.trace.policy.max_chases,
        no_resting_taker=all(fill.liquidity == "maker" for fill in c.fills if fill.reason == "maker_fill"),
    )


def _evidence_c3(c: _Cycle) -> ScenarioEvidence:
    session, outcomes = c.session, c.trace.outcomes
    submit_ids = {s.client_order_id for s in session.journal.submits() if s.submit_seq >= c.trace.submit_start}
    venue_ids = {str(session.client.query_order(c.symbol, client_id)["orderId"]) for client_id in submit_ids}
    trades = [row for row in session.client.user_trades(c.symbol) if str(row["orderId"]) in venue_ids]
    venue_qty = sum((Decimal(str(row["qty"])) for row in trades), Decimal(0))
    venue_notional = sum((Decimal(str(row["qty"])) * Decimal(str(row["price"])) for row in trades), Decimal(0))
    journal_notional = sum(
        (o.filled_qty * o.avg_fill_price for o in outcomes if o.avg_fill_price is not None), Decimal(0)
    )
    vwap_parity = (
        venue_qty == c.total
        and venue_notional > 0
        and abs(journal_notional - venue_notional) / venue_notional <= session.config.price_rel_tolerance
    )
    return ev(
        "C3",
        **c.common,
        status=outcomes[0].status if outcomes else None,
        qty_exact=c.total == c.requested,
        vwap_within_tolerance=vwap_parity,
        maker_false=bool(trades) and all(row["maker"] is False for row in trades),
    )


def _evidence_c4(c: _Cycle) -> ScenarioEvidence:
    ioc = [
        json.loads(row.response_body)
        for row in c.posted
        if b'"status"' in row.response_body and "timeInForce=IOC" in row.url
    ]
    partial: dict[str, Any] = next(
        (row for row in ioc if row.get("status") == "EXPIRED" and Decimal(str(row.get("executedQty", 0))) > 0), {}
    )
    return ev(
        "C4",
        **c.common,
        status=partial.get("status"),
        executed_qty=partial.get("executedQty", "0"),
        requested_qty=str(c.requested),
        ioc_attempts=len(ioc),
        max_ioc_attempts=c.trace.policy.max_ioc_attempts,
        residual_exact=c.residual,
        booked_once=len({(fill.client_order_id, fill.cumulative_executed_qty) for fill in c.fills}) == len(c.fills),
    )


def _evidence_c7(c: _Cycle) -> ScenarioEvidence:
    return ev(
        "C7",
        **c.common,
        never_flips=c.positions.get(c.symbol, Decimal(0)) == 0,
        contained_to_intent=c.trace.escaped is None,
        codes=c.codes,
    )


def _evidence_c8(c: _Cycle) -> ScenarioEvidence:
    submits = [
        s
        for s in c.session.journal.submits()
        if s.submit_seq >= c.trace.submit_start and s.symbol == c.symbol
    ]
    open_submits = [s for s in submits if s.reduce_only is False]
    close_submits = [s for s in submits if s.reduce_only is True]
    close_filled_at = [
        term.recorded_at
        for s in close_submits
        if (term := c.session.journal.terminal(s.client_order_id)) is not None
        and term.status == "FILLED"
    ]
    if open_submits and close_filled_at:
        first_open = min(s.recorded_at for s in open_submits)
        sequenced = first_open >= max(close_filled_at)
    else:
        sequenced = False
    final_pos = c.positions.get(c.symbol, Decimal(0))
    target = c.intents[-1].target_qty
    step = c.session.filters[c.symbol].step_size
    within_step = abs(final_pos - target) <= step
    initial = c.intents[0].current_qty
    position = initial
    never_crossed = True
    attempt_seqs = {s.attempt_seq for s in submits}
    for fill in c.fills:
        if fill.symbol == c.symbol and fill.attempt_seq in attempt_seqs:
            position += fill.quantity if fill.side == "BUY" else -fill.quantity
            never_crossed &= position * initial >= 0
    return ev(
        "C8",
        **c.common,
        final_equals_target=bool(within_step),
        residual_explicit=c.residual,
        has_obsolete_leg=any(outcome.status == "OBSOLETE" for outcome in c.trace.outcomes),
        open_posted_after_close_filled=bool(sequenced),
        blocked_open_leg=any(
            getattr(outcome, "block_reason", None) == "blocked_by_close_leg"
            for outcome in c.trace.outcomes
        ),
        open_leg_posts=len(open_submits),
        final_position=str(final_pos),
        target_qty=str(target),
        position_never_crossed_zero=bool(never_crossed and final_pos * initial >= 0),
    )


def _evidence_c10(c: _Cycle) -> ScenarioEvidence:
    ids = [row.client_order_id for row in c.session.journal.submits()]
    return ev(
        "C10",
        **c.common,
        slice_sum=str(c.total),
        intent_qty=str(c.requested),
        residual_explicit=c.residual,
        slice_ids_unique=len(ids) == len(set(ids)),
        final_equals_target=c.positions.get(c.symbol, Decimal(0)) == c.intents[0].target_qty,
    )


def _evidence_c11(c: _Cycle) -> ScenarioEvidence:
    return ev("C11", **c.common, r1_r6_pass=False, symbol_count=len({intent.symbol for intent in c.intents}))


def _evidence_c12(c: _Cycle) -> ScenarioEvidence:
    cap_bps = Decimal(str(c.trace.policy.taker_cap_bps))
    samples = [sample for sample in c.trace.post_quotes if sample["timeInForce"] == "IOC"]

    def within_cap(sample: dict[str, Any]) -> bool:
        touch = Decimal(str(sample["ask" if sample["side"] == "BUY" else "bid"]))
        tick = c.session.filters[str(sample["symbol"])].tick_size
        return abs(Decimal(str(sample["price"])) - touch) <= touch * cap_bps / 10000 + tick

    return ev(
        "C12",
        **c.common,
        escalated_to_ioc=any("timeInForce=IOC" in row.url for row in c.posted),
        price_within_cap=bool(samples) and all(within_cap(sample) for sample in samples),
        open_empty=not c.session.client.open_orders(),
        residual_exact=c.exact,
    )


def _evidence_c13(c: _Cycle) -> ScenarioEvidence:
    return ev(
        "C13",
        **c.common,
        no_429=not any(row.status_code == 429 for row in c.session.transport.records),
        intent_count=len(c.intents),
        symbol_count=len({intent.symbol for intent in c.intents}),
        throttle_respected=bool(c.trace.sleeps) and min(c.trace.sleeps) >= c.trace.policy.poll_interval_s,
        reducing_first=bool(c.posted) and "reduceOnly=true" in c.posted[0].url,
    )


def _evidence_c14(c: _Cycle) -> ScenarioEvidence:
    return ev(
        "C14",
        **c.common,
        reducing_first=bool(c.posted) and "reduceOnly=true" in c.posted[0].url,
        no_spurious_2019=-2019 not in c.codes,
        margin_near_limit=True,
    )


_ORDINARY_EVIDENCE: dict[str, Callable[[_Cycle], ScenarioEvidence]] = {
    "C2": _evidence_c2,
    "C3": _evidence_c3,
    "C4": _evidence_c4,
    "C7": _evidence_c7,
    "C8": _evidence_c8,
    "C10": _evidence_c10,
    "C11": _evidence_c11,
    "C12": _evidence_c12,
    "C13": _evidence_c13,
    "C14": _evidence_c14,
}


def _ordinary(session: DrillSession, scenario_id: str) -> ScenarioEvidence:
    if scenario_id == "C1":
        return _resting_cancel(session)
    return _ORDINARY_EVIDENCE[scenario_id](_observe_cycle(session, _ordinary_intents(session, scenario_id)))


def runner(scenario_id: str) -> Callable[[DrillSession, Any], ScenarioEvidence]:
    """Bind one lifecycle experiment to the common isolation and conservation wrapper."""

    def run(session: DrillSession, config: Any) -> ScenarioEvidence:
        if scenario_id == "C5":
            return _partial(session)
        if scenario_id == "C6":
            return _race(session)
        if scenario_id == "C9":
            return _repeg(session)
        return _ordinary(session, scenario_id)

    return isolated_run(run)
