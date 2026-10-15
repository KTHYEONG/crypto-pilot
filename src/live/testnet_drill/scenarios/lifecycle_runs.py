"""Venue-backed lifecycle experiments with an independently owned counterparty."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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
from src.live.testnet_drill.session import DrillSession, _read_positions, open_counterparty_session


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


def _counterparty_order(aux: DrillSession, row: dict[str, Any], quantity: Decimal, *, resting: bool = False) -> Any:
    return aux.guarded_new_order(
        {
            "symbol": row["symbol"],
            "side": "SELL" if row["side"] == "BUY" else "BUY",
            "type": "LIMIT",
            "timeInForce": "GTC" if resting else "IOC",
            "quantity": str(quantity),
            "price": str(row["price"]),
        }
    )


def _partial(session: DrillSession) -> ScenarioEvidence:
    aux = open_counterparty_session(session)
    symbol = session.config.symbols[0]
    intent = make_intent(session, symbol)
    step = session.filters[symbol].step_size
    portions = [intent.quantity // (step * 2) * step]
    portions.append(intent.quantity - portions[0])
    if min(portions) <= 0:
        raise DrillRefused("drill refused: partial_probe_needs_two_steps")
    sweeps: list[dict[str, Any]] = []

    def sweep(tick: int) -> None:
        orders = _primary_orders(session, symbol)
        if orders and len(sweeps) < len(portions):
            row = orders[0]
            amount = portions[len(sweeps)]
            response = _counterparty_order(aux, row, amount)
            sweeps.append(
                {
                    "primary_seq": len(session.transport.records),
                    "aux_seq": len(aux.transport.records),
                    "response": response,
                }
            )
            time.sleep(session.config.counterparty_step_wait_s)

    trace = run_cycle(session, [intent], on_tick=sweep)
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
        deltas_sum_to_venue=parity and compared and len(sweeps) >= 2,
        delta_price=str(delta_price),
        user_trade_price=str(trade_price),
        price_rel_tolerance=str(session.config.price_rel_tolerance),
        uses_cumulative_avg=mismatch is not None,
        sweeps=sweeps,
        escaped_exception=trace.escaped,
    )


def _race(session: DrillSession) -> ScenarioEvidence:
    aux = open_counterparty_session(session)
    repetitions = []
    started = time.monotonic()
    for index in range(session.config.race_repetitions):
        if time.monotonic() - started >= session.config.scenario_timeout_s:
            raise DrillRefused("drill refused: race_timeout")
        order = _limit(session)
        posted = submit_probe(session, order)
        if posted.get("status") not in ("NEW", "PARTIALLY_FILLED"):
            raise DrillRefused("drill refused: race_order_did_not_rest")
        row = session.client.query_order(order["symbol"], order["newClientOrderId"])
        offset = session.config.race_offsets_s[index % len(session.config.race_offsets_s)]

        def cancel(order: dict[str, Any] = order, offset: float = offset) -> Any:
            time.sleep(max(0, -offset))
            return cancel_probe(session, order["symbol"], order["newClientOrderId"])

        def trade(row: dict[str, Any] = row, order: dict[str, Any] = order, offset: float = offset) -> Any:
            time.sleep(max(0, offset))
            return _counterparty_order(aux, row, Decimal(order["quantity"]))

        with ThreadPoolExecutor(max_workers=2) as pool:
            cancelled, traded = pool.submit(cancel), pool.submit(trade)
            confirmation = cancelled.result()
            traded.result()
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
                "aux_seq": len(aux.transport.records),
            }
        )
    return ev("C6", repetitions=repetitions)


def _repeg(session: DrillSession) -> ScenarioEvidence:
    aux = open_counterparty_session(session)
    symbol = session.config.symbols[0]
    intent = make_intent(session, symbol)
    quotes = []
    last_id: str | None = None

    def move(tick: int) -> None:
        nonlocal last_id
        if last_id is not None:
            aux.client.cancel_order(symbol, last_id)
        quote = aux.client.book_ticker(symbol)
        price = min(Decimal(str(quote["askPrice"])) - aux.filters[symbol].tick_size,
                    Decimal(str(quote["bidPrice"])) + aux.filters[symbol].tick_size * (1 + tick % 2))
        from src.live.filters import quantize_order

        order = quantize_order(aux.filters[symbol], intent.quantity, price, reduce_only=False)
        if order is None:
            raise DrillRefused("drill refused: quote_minimum")
        last_id = aux.next_client_order_id(symbol)
        aux.guarded_new_order(
            {
                "symbol": symbol,
                "side": "BUY",
                "type": "LIMIT",
                "timeInForce": "GTC",
                "price": str(order.price),
                "quantity": str(abs(order.quantity)),
                "newClientOrderId": last_id,
            }
        )
        quotes.append({"primary_seq": len(session.transport.records), "aux_seq": len(aux.transport.records)})

    trace = run_cycle(session, [intent], on_tick=move)
    times = [sample["observed_at"] for sample in trace.post_quotes if sample["timeInForce"] == "GTX"]
    interval_ok = len(times) >= 2 and all(b - a >= trace.policy.repeg_interval_s for a, b in pairwise(times))
    return ev(
        "C9",
        open_order_snapshots=trace.open_counts,
        cumulative_filled=str(sum((outcome.filled_qty for outcome in trace.outcomes), Decimal(0))),
        intent_qty=str(intent.quantity),
        interval_respected=interval_ok,
        quote_records=quotes,
    )


def _ordinary(session: DrillSession, scenario_id: str) -> ScenarioEvidence:
    symbol = session.config.symbols[0]
    if scenario_id == "C1":
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
    intents = [make_intent(session, symbol)]
    if scenario_id == "C4":
        depth = session.client.depth(symbol)
        amount = Decimal(str(depth["asks"][0][1])) + session.filters[symbol].step_size
        intents = [make_intent(session, symbol, quantity=amount)]
    elif scenario_id == "C8":
        opening = run_cycle(session, intents)
        if not opening.outcomes or opening.outcomes[0].filled_qty <= 0:
            raise DrillRefused("drill refused: flip_requires_open_position")
        positions = _read_positions(session.client)
        intents = plan_orders(
            {symbol: -intents[0].quantity},
            positions,
            session.filters,
            {symbol: intents[0].decision_price},
            session.client_order_prefix.removeprefix("mh"),
        )
    elif scenario_id == "C10":
        if session.config.probe_notional_usdt <= MAX_SLICE_NOTIONAL:
            raise DrillRefused("drill refused: slice_probe_notional_must_exceed_slice_limit")
    elif scenario_id in ("C11", "C13"):
        minimum = 2 if scenario_id == "C11" else 3
        if len(session.config.symbols) < minimum:
            raise DrillRefused("drill refused: insufficient_probe_symbols")
        intents = [make_intent(session, sym, leg_index=index) for index, sym in enumerate(session.config.symbols)]
        if scenario_id == "C13":
            opening = run_cycle(session, [intents[0]])
            if not opening.outcomes or opening.outcomes[0].filled_qty <= 0:
                raise DrillRefused("drill refused: throttle_probe_requires_reducing_intent")
            intents = [replace(intents[index % len(intents)], leg_index=index) for index in range(10)]
            intents[0] = replace(intents[0], side="SELL", reduce_only=True, quantity=opening.outcomes[0].filled_qty)
    elif scenario_id == "C7":
        intents = [replace(intents[0], reduce_only=True)]
    elif scenario_id == "C14":
        opening = run_cycle(session, intents)
        account = session.client.account()
        if not opening.outcomes or Decimal(str(account["availableBalance"])) > session.config.probe_notional_usdt:
            raise DrillRefused("drill refused: margin_release_requires_near_limit_account")
        intents = [replace(intents[0], side="SELL", reduce_only=True, leg_index=0), replace(intents[0], leg_index=1)]
    trace = run_cycle(session, intents)
    outcomes = trace.outcomes
    fills = session.journal.fills_after(-1)
    posted = [
        row
        for row in session.transport.records[trace.wire_start : trace.wire_end]
        if row.method == "POST" and row.path == "/fapi/v1/order"
    ]
    codes = _codes(session, trace.wire_start)
    total = sum((outcome.filled_qty for outcome in outcomes), Decimal(0))
    requested = sum((intent.quantity for intent in intents), Decimal(0))
    exact = len(outcomes) == len(intents) and all(
        outcome.filled_qty + outcome.unfilled_qty == intent.quantity
        for outcome, intent in zip(outcomes, intents, strict=True)
    )
    residual = exact and any(
        outcome.unfilled_qty > 0 and outcome.status in ("RESIDUAL", "RESIDUAL_SUB_MINIMUM", "OBSOLETE")
        for outcome in outcomes
    )
    positions = _read_positions(session.client)
    common = {"escaped_exception": trace.escaped, "wire_start": trace.wire_start, "wire_end": trace.wire_end}
    if scenario_id == "C2":
        return ev(
            scenario_id,
            **common,
            code=next((code for code in codes if code == -5022), None),
            chases=sum(outcome.chases for outcome in outcomes),
            window=trace.policy.max_chases,
            no_resting_taker=all(fill.liquidity == "maker" for fill in fills if fill.reason == "maker_fill"),
        )
    if scenario_id == "C3":
        submit_ids = {s.client_order_id for s in session.journal.submits() if s.submit_seq >= trace.submit_start}
        venue_ids = {str(session.client.query_order(symbol, client_id)["orderId"]) for client_id in submit_ids}
        trades = [row for row in session.client.user_trades(symbol) if str(row["orderId"]) in venue_ids]
        venue_qty = sum((Decimal(str(row["qty"])) for row in trades), Decimal(0))
        venue_notional = sum((Decimal(str(row["qty"])) * Decimal(str(row["price"])) for row in trades), Decimal(0))
        journal_notional = sum((outcome.filled_qty * outcome.avg_fill_price for outcome in outcomes if outcome.avg_fill_price is not None), Decimal(0))
        vwap_parity = venue_qty == total and venue_notional > 0 and abs(journal_notional - venue_notional) / venue_notional <= session.config.price_rel_tolerance
        return ev(
            scenario_id,
            **common,
            status=outcomes[0].status if outcomes else None,
            qty_exact=total == requested,
            vwap_within_tolerance=vwap_parity,
            maker_false=bool(trades) and all(row["maker"] is False for row in trades),
        )
    if scenario_id == "C4":
        ioc = [
            json.loads(row.response_body)
            for row in posted
            if b'"status"' in row.response_body and "timeInForce=IOC" in row.url
        ]
        partial: dict[str, Any] = next(
            (row for row in ioc if row.get("status") == "EXPIRED" and Decimal(str(row.get("executedQty", 0))) > 0), {}
        )
        return ev(
            scenario_id,
            **common,
            status=partial.get("status"),
            executed_qty=partial.get("executedQty", "0"),
            requested_qty=str(requested),
            ioc_attempts=len(ioc),
            max_ioc_attempts=trace.policy.max_ioc_attempts,
            residual_exact=residual,
            booked_once=len({(fill.client_order_id, fill.cumulative_executed_qty) for fill in fills}) == len(fills),
        )
    if scenario_id == "C7":
        return ev(
            scenario_id,
            **common,
            never_flips=positions.get(symbol, Decimal(0)) == 0,
            contained_to_intent=trace.escaped is None,
            codes=codes,
        )
    if scenario_id == "C8":
        return ev(
            scenario_id,
            **common,
            final_equals_target=positions.get(symbol, Decimal(0)) == intents[-1].target_qty,
            residual_explicit=residual,
            has_obsolete_leg=any(outcome.status == "OBSOLETE" for outcome in outcomes),
        )
    if scenario_id == "C10":
        ids = [row.client_order_id for row in session.journal.submits()]
        return ev(
            scenario_id,
            **common,
            slice_sum=str(total),
            intent_qty=str(requested),
            residual_explicit=residual,
            slice_ids_unique=len(ids) == len(set(ids)),
            final_equals_target=positions.get(symbol, Decimal(0)) == intents[0].target_qty,
        )
    if scenario_id == "C11":
        return ev(scenario_id, **common, r1_r6_pass=False, symbol_count=len({intent.symbol for intent in intents}))
    if scenario_id == "C12":
        ioc_posts = [row for row in posted if "timeInForce=IOC" in row.url]
        samples = [sample for sample in trace.post_quotes if sample["timeInForce"] == "IOC"]
        cap_ok = bool(samples) and all(
            abs(Decimal(str(sample["price"])) - Decimal(str(sample["ask" if sample["side"] == "BUY" else "bid"])))
            <= Decimal(str(sample["ask" if sample["side"] == "BUY" else "bid"])) * Decimal(str(trace.policy.taker_cap_bps)) / 10000
               + session.filters[str(sample["symbol"])].tick_size
            for sample in samples)
        return ev(
            scenario_id,
            **common,
            escalated_to_ioc=bool(ioc_posts),
            price_within_cap=cap_ok,
            open_empty=not session.client.open_orders(),
            residual_exact=exact,
        )
    if scenario_id == "C13":
        return ev(
            scenario_id,
            **common,
            no_429=not any(row.status_code == 429 for row in session.transport.records),
            intent_count=len(intents),
            symbol_count=len({intent.symbol for intent in intents}),
            throttle_respected=bool(trace.sleeps) and min(trace.sleeps) >= trace.policy.poll_interval_s,
            reducing_first=bool(posted) and "reduceOnly=true" in posted[0].url,
        )
    return ev(
        scenario_id,
        **common,
        reducing_first=bool(posted) and "reduceOnly=true" in posted[0].url,
        no_spurious_2019=-2019 not in codes,
        margin_near_limit=True,
    )


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
