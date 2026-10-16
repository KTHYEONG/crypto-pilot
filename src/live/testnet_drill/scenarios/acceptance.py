"""Group B acceptance, filter and error-code scenarios (B1..B11)."""

from __future__ import annotations

import contextlib
from dataclasses import replace
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from typing import Any
from uuid import uuid4

from src.live.errors import BINANCE_ERROR_POLICY, VenueError, resolve_error_action
from src.live.filters import quantize_order
from src.live.testnet_drill.execution import cancel_probe, make_intent, run_cycle, settle_journal, submit_probe
from src.live.testnet_drill.guard import DrillRefused
from src.live.testnet_drill.report import Verdict
from src.live.testnet_drill.scenarios.base import (
    ScenarioEvidence,
    ScenarioSpec,
    ev,
    isolated_run,
    observation,
    result_of,
)
from src.live.testnet_drill.scenarios.lifecycle_runs import _codes


def _wrap(body: Any) -> Any:
    return isolated_run(body)


def _params(session: Any, symbol: str) -> dict[str, Any]:
    intent = make_intent(session, symbol)
    return {
        "symbol": symbol,
        "side": "BUY",
        "type": "LIMIT",
        "timeInForce": "GTX",
        "price": str(intent.decision_price),
        "quantity": str(intent.quantity),
    }


def _run_b1(session: Any, config: Any) -> ScenarioEvidence:
    per_symbol: dict[str, Any] = {}
    raw_codes: dict[str, Any] = {}
    for symbol in config.symbols:
        filters = session.filters[symbol]
        params = _params(session, symbol)
        mark = Decimal(params["price"])
        valid = []
        for offset in (-1, 0, 1):
            qty = Decimal(params["quantity"]) + filters.step_size * offset
            order = quantize_order(filters, qty, mark - filters.tick_size * abs(offset), reduce_only=False)
            if order is not None:
                candidate = {
                    **params,
                    "quantity": str(abs(order.quantity)),
                    "price": str(order.price),
                    "newClientOrderId": session.next_client_order_id(symbol),
                }
                try:
                    response = submit_probe(session, candidate)
                    valid.append(isinstance(response, dict) and "orderId" in response)
                    cancel_probe(session, symbol, candidate["newClientOrderId"])
                except VenueError:
                    valid.append(False)
        per_symbol[symbol] = {"quantizer_accepted": bool(valid) and all(valid)}
        try:
            submit_probe(
                session, {**params, "quantity": str(Decimal(params["quantity"]) - filters.step_size / 2)}, raw=True
            )
        except VenueError as exc:
            raw_codes[symbol] = exc.code
        else:
            raw_codes[symbol] = None
    return ev("B1", per_symbol=per_symbol, raw_codes=raw_codes)


def _eval_b1(evidence: ScenarioEvidence) -> Any:
    per_symbol = dict(evidence.payload.get("per_symbol", {}))
    raw_codes = dict(evidence.payload.get("raw_codes", {}))
    grid_ok = all(bool(info.get("quantizer_accepted", False)) for info in per_symbol.values()) if per_symbol else False
    raw_ok = all(code in (-1111, -4014, -4023) for code in raw_codes.values()) if raw_codes else True
    passed = bool(grid_ok and raw_ok)
    obs = tuple(
        observation(int(code), resolve_error_action(int(code)).value) for code in raw_codes.values() if code is not None
    )
    return result_of(
        evidence,
        "B1",
        Verdict.PASS if passed else Verdict.FAIL,
        (("quantization_grid", passed),),
        observations=obs,
        hypotheses=(),
    )


def _run_boundary(kind: str) -> Any:
    def run(session: Any, config: Any) -> ScenarioEvidence:
        symbol = config.symbols[0]
        filters = session.filters[symbol]
        params = _params(session, symbol)
        price = Decimal(params["price"])
        if kind == "B2":
            # Snap up to the step grid so only notional, not precision, separates the boundary levels.
            steps = (filters.min_notional / price / filters.step_size).to_integral_value(rounding=ROUND_CEILING)
            center = max(steps * filters.step_size, filters.min_qty)
        else:
            center = filters.min_qty
        rows = []
        for reduce_only in (False, True):
            if reduce_only:
                submit_probe(
                    session, {"symbol": symbol, "side": "SELL", "type": "MARKET", "quantity": params["quantity"]}
                )
                settle_journal(session)
            for level, offset in (("below_min", -1), ("exact_min", 0), ("above_min", 1)):
                qty = center + filters.step_size * offset
                client_accept = qty > 0 and quantize_order(filters, qty, price, reduce_only=reduce_only) is not None
                candidate = {
                    **params,
                    "quantity": str(qty),
                    "reduceOnly": str(reduce_only).lower(),
                    "newClientOrderId": session.next_client_order_id(symbol),
                }
                accepted = False
                code = None
                if qty > 0:
                    try:
                        response = submit_probe(session, candidate, raw=True)
                        accepted = isinstance(response, dict) and "orderId" in response
                        cancel_probe(session, symbol, candidate["newClientOrderId"])
                    except VenueError as exc:
                        code = exc.code
                rows.append(
                    {
                        "level": level,
                        "reduce_only": reduce_only,
                        "client_accept": client_accept,
                        "venue_accept": accepted,
                        "code": code,
                    }
                )
        exact = next(row for row in rows if row["level"] == "exact_min" and not row["reduce_only"])
        return ev(
            kind,
            level="exact_min",
            client_decision="accept" if exact["client_accept"] else "reject",
            venue_decision="accept" if exact["venue_accept"] else "reject",
            agreed=all(row["client_accept"] == row["venue_accept"] for row in rows),
            boundary_rows=rows,
        )

    return run


def _eval_min_agreement(scenario_id: str) -> Any:
    def evaluate(evidence: ScenarioEvidence) -> Any:
        client_decision = str(evidence.payload.get("client_decision", ""))
        venue_decision = str(evidence.payload.get("venue_decision", ""))
        level = str(evidence.payload.get("level", ""))
        if level == "exact_min" and client_decision != venue_decision:
            return result_of(
                evidence,
                scenario_id,
                Verdict.FAIL,
                ((f"{scenario_id.lower()}_agreement", False),),
                hypotheses=(),
                proposal_rationale="",
            )
        passed = bool(evidence.payload.get("agreed", True))
        return result_of(
            evidence,
            scenario_id,
            Verdict.PASS if passed else Verdict.FAIL,
            ((f"{scenario_id.lower()}_agreement", passed),),
            hypotheses=(),
        )

    return evaluate


def _run_b4(session: Any, config: Any) -> ScenarioEvidence:
    symbol = config.symbols[0]
    intent = make_intent(session, symbol)
    far = quantize_order(session.filters[symbol], intent.quantity, intent.decision_price * 2, reduce_only=False)
    if far is None:
        raise DrillRefused("drill refused: far_price_minimum")
    start = len(session.transport.records)
    with contextlib.suppress(VenueError):
        submit_probe(
            session,
            {
                "symbol": symbol,
                "side": "BUY",
                "type": "LIMIT",
                "timeInForce": "IOC",
                "quantity": str(abs(far.quantity)),
                "price": str(far.price),
            },
            raw=True,
        )
    return ev("B4", code=next(iter(_codes(session, start)), None))


def _eval_b4(evidence: ScenarioEvidence) -> Any:
    code = evidence.payload.get("code")
    if code is None:
        return result_of(evidence, "B4", Verdict.OBSERVED, (("percent_price_not_enforced", True),), hypotheses=())
    action = resolve_error_action(int(code)).value
    passed = int(code) == -4131 and action == "benign_reprice"
    return result_of(
        evidence,
        "B4",
        Verdict.PASS if passed else Verdict.OBSERVED,
        (("percent_price_rejected_by_venue", True),),
        observations=(observation(int(code), action),),
        hypotheses=(),
        proposed_action=None if int(code) in BINANCE_ERROR_POLICY else action,
        proposal_rationale=f"percent price code {code}",
    )


def _run_b5(session: Any, config: Any) -> ScenarioEvidence:
    from src.live.filters import quantize_order

    symbol = config.symbols[0]
    filters = session.filters[symbol]
    oversized = filters.max_qty * 2
    planner_emits_above = False
    try:
        submit_probe(
            session,
            {
                "symbol": symbol,
                "side": "BUY",
                "type": "LIMIT",
                "quantity": str(oversized),
                "price": str(make_intent(session, symbol).decision_price),
                "timeInForce": "GTX",
            },
            raw=True,
        )
        code = None
    except VenueError as exc:
        code = exc.code
    quantized = quantize_order(filters, oversized, Decimal("100"), reduce_only=False)
    if quantized is not None and abs(quantized.quantity) > filters.max_qty:
        planner_emits_above = True
    return ev("B5", code=code, planner_emits_above=planner_emits_above)


def _eval_b5(evidence: ScenarioEvidence) -> Any:
    if bool(evidence.payload.get("planner_emits_above", False)):
        return result_of(evidence, "B5", Verdict.FAIL, (("max_qty_clip", False),), hypotheses=())
    code = evidence.payload.get("code")
    if code is None:
        return result_of(evidence, "B5", Verdict.FAIL, (("max_qty_clip", False),), hypotheses=())
    action = resolve_error_action(int(code)).value
    obs = (observation(int(code), action),)
    if int(code) not in BINANCE_ERROR_POLICY:
        return result_of(
            evidence,
            "B5",
            Verdict.OBSERVED,
            (("max_qty_observed", True),),
            observations=obs,
            hypotheses=("H1",),
            proposed_action=action,
            proposal_rationale=f"unregistered max-qty code {code}",
        )
    return result_of(evidence, "B5", Verdict.PASS, (("max_qty_clip", True),), observations=obs, hypotheses=())


def _run_b6(session: Any, config: Any) -> ScenarioEvidence:
    from src.live.testnet_drill.scenarios.account_config import _restore_configuration

    symbol = config.symbols[0]
    original = next(row["leverage"] for row in session.client.position_risk() if row["symbol"] == symbol)
    with _restore_configuration(session, "/fapi/v1/leverage", {"symbol": symbol, "leverage": original}):
        session.client.request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": 1}, signed=True)
        account = session.client.account()
        seed = make_intent(session, symbol)
        qty = Decimal(str(account["availableBalance"])) / seed.decision_price + session.filters[symbol].step_size
        opening = make_intent(session, symbol, quantity=qty)
        trace = run_cycle(session, [opening])
        reduce = run_cycle(session, [replace(opening, reduce_only=True, leg_index=1)])
        codes = _codes(session, trace.wire_start)
        reduce_codes = _codes(session, reduce.wire_start)
    return ev(
        "B6",
        code=next((code for code in codes if code == -2019), None),
        retries=codes.count(-2019),
        max_margin_rejects=trace.policy.max_margin_rejects,
        reduce_only_waited=-2019 in reduce_codes,
        status=trace.outcomes[0].status if trace.outcomes else None,
    )


def _eval_b6(evidence: ScenarioEvidence) -> Any:
    code = evidence.payload.get("code", -2019)
    retries = int(evidence.payload.get("retries", 0))
    max_rejects = int(evidence.payload.get("max_margin_rejects", 3))
    reduce_only_waited = bool(evidence.payload.get("reduce_only_waited", False))
    action = resolve_error_action(int(code)).value if code is not None else "fail_closed"
    passed = (
        int(code) == -2019
        and 0 < retries <= max_rejects + 1  # max waits plus the terminal rejection
        and not reduce_only_waited
        and evidence.payload["status"] == "REJECTED"
    )
    obs = (observation(int(code), action),) if code is not None else ()
    return result_of(
        evidence,
        "B6",
        Verdict.PASS if passed else Verdict.FAIL,
        (("margin_wait_bounded", passed),),
        observations=obs,
        hypotheses=(),
    )


def _run_b7(session: Any, config: Any) -> ScenarioEvidence:
    from src.live.account import max_notional_at_leverage, parse_leverage_brackets

    symbol = config.symbols[0]
    brackets = parse_leverage_brackets(session.client.request("GET", "/fapi/v1/leverageBracket", signed=True))
    leverage = int(next(row["leverage"] for row in session.client.position_risk() if row["symbol"] == symbol))
    cap = max_notional_at_leverage(brackets[symbol], leverage)
    seed = make_intent(session, symbol)
    excessive = make_intent(session, symbol, quantity=cap / seed.decision_price + session.filters[symbol].step_size)
    trace = run_cycle(session, [excessive, replace(seed, leg_index=1)])
    return ev(
        "B7",
        code=next(iter(_codes(session, trace.wire_start)), None),
        others_unaffected=trace.escaped is None and len(trace.outcomes) == 2 and trace.outcomes[1].status == "FILLED",
    )


def _eval_b7(evidence: ScenarioEvidence) -> Any:
    code = evidence.payload.get("code", -2027)
    others_unaffected = bool(evidence.payload.get("others_unaffected", True))
    action = resolve_error_action(int(code)).value if code is not None else "fail_closed"
    passed = int(code) == -2027 and others_unaffected and action == "intent_reject"
    verdict = Verdict.PASS if passed else (Verdict.OBSERVED if others_unaffected else Verdict.FAIL)
    return result_of(
        evidence,
        "B7",
        verdict,
        (("position_cap_isolation", others_unaffected),),
        observations=(observation(int(code), action),) if code is not None else (),
        hypotheses=("H1",) if code not in BINANCE_ERROR_POLICY else (),
        proposed_action=action if code not in BINANCE_ERROR_POLICY else None,
        proposal_rationale=f"cap code {code}",
    )


def _run_b8(session: Any, config: Any) -> ScenarioEvidence:
    seed = make_intent(session, config.symbols[0])
    trace = run_cycle(
        session,
        [
            seed,
            replace(seed, quantity=seed.quantity + session.filters[seed.symbol].step_size / 2, leg_index=1),
            replace(seed, leg_index=2),
        ],
    )
    submits = session.journal.submits()
    rejected = [s for s in submits if s.leg_index == 1]
    return ev(
        "B8",
        outcomes=[outcome.status for outcome in trace.outcomes],
        escaped_exception=trace.escaped is not None,
        terminal_once=bool(rejected) and all(session.journal.terminal(s.client_order_id) is not None for s in rejected),
        codes=_codes(session, trace.wire_start),
    )


def _eval_b8(evidence: ScenarioEvidence) -> Any:
    outcomes = list(evidence.payload.get("outcomes", []))
    escaped = bool(evidence.payload.get("escaped_exception", False))
    terminal_once = bool(evidence.payload.get("terminal_once", True))
    passed = outcomes == ["FILLED", "REJECTED", "FILLED"] and not escaped and terminal_once
    return result_of(
        evidence,
        "B8",
        Verdict.PASS if passed else Verdict.FAIL,
        (("reject_isolation", passed),),
        hypotheses=(),
    )


def _run_b9(session: Any, config: Any) -> ScenarioEvidence:
    from src.live.testnet_drill.scenarios.lifecycle_runs import _limit

    code = None
    for _ in range(config.max_open_orders_probe + 1):
        try:
            submit_probe(session, _limit(session, distance=True), raw=True)
        except VenueError as exc:
            code = exc.code
            break
    return ev("B9", code=code, cycle_halted=code is not None and resolve_error_action(code).value == "fail_closed")


def _eval_b9(evidence: ScenarioEvidence) -> Any:
    code = evidence.payload.get("code")
    halted = bool(evidence.payload.get("cycle_halted", False))
    if code is None:
        return result_of(evidence, "B9", Verdict.OBSERVED, (("order_limit_not_reached", True),), hypotheses=())
    action = resolve_error_action(int(code)).value
    return result_of(
        evidence,
        "B9",
        Verdict.OBSERVED,
        (("order_limit_observed", True),),
        observations=(observation(int(code), action),),
        hypotheses=("H1",),
        hypothesis_outcome="INCONCLUSIVE",
        proposed_action=action,
        proposal_rationale=f"limit code {code} halt={halted}",
    )


def _run_b10(session: Any, config: Any) -> ScenarioEvidence:
    from src.live.testnet_drill.scenarios.lifecycle_runs import _limit

    params = _limit(session, distance=True)
    submit_probe(session, params, raw=True)
    code = None
    queries = []
    try:
        submit_probe(session, params, raw=True)
    except VenueError as exc:
        code = exc.code
    queries.append(session.client.query_order(params["symbol"], params["newClientOrderId"]))
    cancel_probe(session, params["symbol"], params["newClientOrderId"])
    for variant in (
        params,
        {**params, "timeInForce": "IOC", "price": str(session.client.book_ticker(params["symbol"])["askPrice"])},
    ):
        try:
            submit_probe(session, variant, raw=True)
            queries.append(session.client.query_order(params["symbol"], params["newClientOrderId"]))
            cancel_probe(session, params["symbol"], params["newClientOrderId"])
        except VenueError as exc:
            code = exc.code
    from src.live.order_journal import OrderJournal

    reopened = OrderJournal(session.journal.path)
    return ev(
        "B10",
        code=code,
        query_returned=str([row["orderId"] for row in queries]),
        submit_seq_journal_derived=reopened.next_submit_seq() == session.journal.next_submit_seq(),
        queries=queries,
    )


def _eval_b10(evidence: ScenarioEvidence) -> Any:
    code = evidence.payload.get("code", -4116)
    which_returned = str(evidence.payload.get("query_returned", ""))
    submit_seq_ok = bool(evidence.payload.get("submit_seq_journal_derived", True))
    action = resolve_error_action(int(code)).value if code is not None else "fail_closed"
    passed = submit_seq_ok and bool(which_returned)
    obs = (observation(int(code), action),) if code is not None else ()
    return result_of(
        evidence,
        "B10",
        Verdict.OBSERVED,
        (("id_semantics_observed", passed),),
        observations=obs,
        hypotheses=("H1", "H8"),
        proposed_action=action if code not in BINANCE_ERROR_POLICY else None,
        proposal_rationale=f"reuse code {code} query={which_returned}",
    )


def _run_b11(session: Any, config: Any) -> ScenarioEvidence:
    from src.live.planner import build_client_order_id

    candidate = build_client_order_id("r" * 17, config.symbols[0], 0, 0, 0)
    ok_36 = len(candidate) == 36
    rejected_37 = False
    try:
        build_client_order_id("r" * 18, config.symbols[0], 0, 0, 0)
    except ValueError:
        rejected_37 = True
    prefix = session.client_order_prefix + "-"
    ids = [
        prefix + uuid4().hex[: 36 - len(prefix)],
        build_client_order_id(
            session.client_order_prefix.removeprefix("mh"), "비ASCII", 0, 0, session.journal.next_submit_seq()
        ),
        datetime.now(UTC).strftime("%Y%m%d-") + prefix + uuid4().hex[:6],
    ]
    accepted = []
    for order_id in ids:
        params = {**_params(session, config.symbols[0]), "newClientOrderId": order_id}
        try:
            response = submit_probe(session, params, raw=True)
            accepted.append(isinstance(response, dict) and "orderId" in response)
            cancel_probe(session, params["symbol"], order_id)
        except VenueError:
            accepted.append(False)
    return ev(
        "B11",
        ok_36=ok_36,
        rejected_37=rejected_37,
        venue_accepted=accepted[0],
        non_ascii_accepted=accepted[1],
        legacy_accepted=accepted[2],
    )


def _eval_b11(evidence: ScenarioEvidence) -> Any:
    passed = all(
        evidence.payload[key] is True
        for key in ("ok_36", "rejected_37", "venue_accepted", "non_ascii_accepted", "legacy_accepted")
    )
    return result_of(
        evidence,
        "B11",
        Verdict.PASS if passed else Verdict.FAIL,
        (("client_id_charset", passed),),
        hypotheses=(),
    )


def _eval_raw_guard(evidence: ScenarioEvidence) -> Any:
    refused = bool(evidence.payload.get("refused_before_transport", False))
    return result_of(
        evidence,
        evidence.scenario_id,
        Verdict.PASS if refused else Verdict.FAIL,
        (("raw_notional_guard", refused),),
        hypotheses=(),
    )


def scenario_specs() -> tuple[ScenarioSpec, ...]:
    """Group B scenario registry."""
    return (
        ScenarioSpec("B1", "B", "quantization grid", (), _wrap(_run_b1), _eval_b1),
        ScenarioSpec("B2", "B", "min notional boundary", (), _wrap(_run_boundary("B2")), _eval_min_agreement("B2")),
        ScenarioSpec("B3", "B", "min qty boundary", (), _wrap(_run_boundary("B3")), _eval_min_agreement("B3")),
        ScenarioSpec("B4", "B", "percent price", (), _wrap(_run_b4), _eval_b4),
        ScenarioSpec("B5", "B", "max qty", ("H1",), _wrap(_run_b5), _eval_b5),
        ScenarioSpec("B6", "B", "margin insufficient", (), _wrap(_run_b6), _eval_b6),
        ScenarioSpec("B7", "B", "position cap at leverage", ("H1",), _wrap(_run_b7), _eval_b7),
        ScenarioSpec("B8", "B", "reject isolation", (), _wrap(_run_b8), _eval_b8),
        ScenarioSpec("B9", "B", "open order limit", ("H1",), _wrap(_run_b9), _eval_b9),
        ScenarioSpec("B10", "B", "client order id semantics", ("H1", "H8"), _wrap(_run_b10), _eval_b10),
        ScenarioSpec("B11", "B", "client order id charset", (), _wrap(_run_b11), _eval_b11),
    )


def evaluate_raw_notional_guard(evidence: ScenarioEvidence) -> Any:
    """Guard helper for raw sends above the notional cap."""
    return _eval_raw_guard(evidence)
