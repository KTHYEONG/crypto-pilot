"""Group D transport-fault scenarios (D1..D10).

Every runner installs real ``FaultRule``s on the session transport and derives evidence from the wire log,
the audit stream (``AuditTap``) and the order journal; no evidence field is asserted by construction.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from decimal import Decimal
from typing import Any

from src.live.errors import LiveTradingError, TransientReadError, VenueError
from src.live.executor import UNKNOWN_SUBMISSION_MISS_LIMIT
from src.live.rest import _GET_RETRY_BACKOFF_SECONDS, OrderStatusUnknown
from src.live.testnet_drill.execution import (
    AuditTap,
    CycleEvidence,
    cancel_probe,
    make_intent,
    run_cycle,
    settle_journal,
    submit_probe,
)
from src.live.testnet_drill.faults import FaultAction, FaultRule, WireRecord
from src.live.testnet_drill.guard import DrillRefused
from src.live.testnet_drill.report import Verdict
from src.live.testnet_drill.scenarios.base import (
    ScenarioEvidence,
    ScenarioSpec,
    ev,
    isolated_run,
    result_of,
)
from src.live.testnet_drill.scenarios.wire import (
    BOOK_TICKER_PATH,
    ORDER_PATH,
)
from src.live.testnet_drill.scenarios.wire import (
    body as _body,
)
from src.live.testnet_drill.scenarios.wire import (
    code as _code,
)
from src.live.testnet_drill.scenarios.wire import (
    event_for as _event_for,
)
from src.live.testnet_drill.scenarios.wire import (
    executed as _executed,
)
from src.live.testnet_drill.scenarios.wire import (
    fills_unique as _fills_unique,
)
from src.live.testnet_drill.scenarios.wire import (
    journal_terminal_lines as _journal_terminal_lines,
)
from src.live.testnet_drill.scenarios.wire import (
    mutations as _mutations,
)
from src.live.testnet_drill.scenarios.wire import (
    order_id as _order_id,
)
from src.live.testnet_drill.scenarios.wire import (
    param as _param,
)
from src.live.testnet_drill.scenarios.wire import (
    records as _records,
)
from src.live.testnet_drill.scenarios.wire import (
    resting_params as _resting_params,
)
from src.live.testnet_drill.scenarios.wire import (
    signed_fold as _signed_fold,
)
from src.live.testnet_drill.scenarios.wire import (
    venue_position as _venue_position,
)

_UNKNOWN_EVENTS = frozenset(
    {"order_status_unknown", "order_unknown_adopted", "order_unknown_not_placed", "order_unknown_left_unresolved"}
)
_GET_BUDGET = len(_GET_RETRY_BACKOFF_SECONDS) + 1


@contextmanager
def _faulted_cycle(session: Any, rules: Sequence[FaultRule]) -> Iterator[AuditTap]:
    with AuditTap(session) as tap, session.transport.inject(rules):
        yield tap


def _cycle(session: Any, rules: Sequence[FaultRule], **kwargs: Any) -> tuple[CycleEvidence, AuditTap, int]:
    intent = make_intent(session, session.config.symbols[0])
    start = len(session.transport.records)
    with _faulted_cycle(session, rules) as tap:
        trace = run_cycle(session, [intent], **kwargs)
    return trace, tap, start


def _no_post_between(posts: Sequence[WireRecord], after_seq: int, until_index: int) -> bool:
    return not any(after_seq < r.seq <= until_index for r in posts)


def _run_lost_response(session: Any, scenario_id: str, *, tif: str | None) -> ScenarioEvidence:
    params = (("timeInForce", tif),) if tif else ()
    rule = FaultRule("POST", ORDER_PATH, params=params, action=FaultAction.DELIVER_DROP_RESPONSE)
    trace, tap, start = _cycle(session, [rule])
    posts = _mutations(_records(session, start, trace.wire_end), "POST")
    lost = next((r for r in posts if tif is None or _param(r, "timeInForce") == tif), None)
    if lost is None or lost.delivered is not True:
        raise DrillRefused(f"drill refused: {scenario_id.lower()}_fault_not_reached")
    symbol, lost_id = str(_param(lost, "symbol")), _order_id(lost)
    adopted = _event_for(tap, {"order_unknown_adopted"}, lost_id)
    single = (
        adopted is not None
        and sum(_order_id(r) == lost_id for r in posts) == 1
        and _no_post_between(posts, lost.seq, adopted.wire_index)
        and max(trace.open_counts, default=0) <= 1
    )
    executed = _executed(session, symbol, str(lost_id))
    booked = sum((f.quantity for f in session.journal.fills_after(-1) if f.client_order_id == lost_id), Decimal(0))
    common = {
        "lost_seq": lost.seq,
        "lost_id": lost_id,
        "adopted_at_wire": adopted.wire_index if adopted is not None else None,
        "watermark_match": session.journal.observed_qty(str(lost_id)) == executed,
        "escaped_exception": trace.escaped,
    }
    if scenario_id == "D1":
        return ev("D1", **common, unresolved_adopted=adopted is not None, single_post=single, booked_once=_fills_unique(session))
    return ev(
        "D2",
        **common,
        booked_from_query=adopted is not None and executed > 0 and booked == executed,
        position_parity=_venue_position(session, symbol) == _signed_fold(session, symbol),
        no_repost=single,
    )


def _run_d1(session: Any, config: Any) -> ScenarioEvidence:
    return _run_lost_response(session, "D1", tif="GTX")


def _run_d2(session: Any, config: Any) -> ScenarioEvidence:
    return _run_lost_response(session, "D2", tif="IOC")


def _run_d3(session: Any, config: Any) -> ScenarioEvidence:
    trace, tap, start = _cycle(session, [FaultRule("POST", ORDER_PATH, action=FaultAction.DROP_REQUEST)])
    records = _records(session, start, trace.wire_end)
    posts = _mutations(records, "POST")
    dropped = next((r for r in posts if r.delivered is False), None)
    if dropped is None:
        raise DrillRefused("drill refused: d3_fault_not_reached")
    dropped_id = _order_id(dropped)
    not_placed = _event_for(tap, {"order_unknown_not_placed"}, dropped_id)
    boundary = not_placed.wire_index if not_placed is not None else trace.wire_end
    misses = [
        r
        for r in _mutations(records, "GET")
        if _order_id(r) == dropped_id and dropped.seq < r.seq <= boundary and _code(r) == -2013
    ]
    horizon = float(session.client.unknown_outcome_horizon_s)
    horizon_ok = not_placed is not None and not_placed.at - dropped.at >= horizon
    terminal = session.journal.terminal(str(dropped_id))
    return ev(
        "D3",
        reposted_early=not _no_post_between(posts, dropped.seq, boundary),
        reposted_after_horizon=not_placed is not None and any(r.seq > boundary for r in posts),
        miss_limit_reached=horizon_ok and len(misses) >= UNKNOWN_SUBMISSION_MISS_LIMIT,
        journal_not_placed=terminal is not None and terminal.status == "NOT_PLACED",
        never_two_live=max(trace.open_counts, default=0) <= 1,
        post_count=len(posts),
        post_seqs=[r.seq for r in posts],
        miss_count=len(misses),
        horizon_s=horizon,
    )


def _run_d4(session: Any, config: Any) -> ScenarioEvidence:
    params = _resting_params(session)
    order_id = params["newClientOrderId"]
    submit_probe(session, params)
    start = len(session.transport.records)
    error: str | None = None
    with session.transport.inject([FaultRule("DELETE", ORDER_PATH, action=FaultAction.DELIVER_DROP_RESPONSE)]):
        try:
            cancel_probe(session, params["symbol"], order_id)
        except LiveTradingError as exc:
            error = type(exc).__name__
    records = _records(session, start)
    deletes = [r for r in _mutations(records, "DELETE") if _order_id(r) == order_id]
    last_delete = deletes[-1].seq if deletes else 0
    confirming = [
        r
        for r in _mutations(records, "GET")
        if _order_id(r) == order_id and r.seq > last_delete and isinstance(_body(r), dict)
    ]
    terminal = session.journal.terminal(order_id)
    return ev(
        "D4",
        delete_count=len(deletes),
        confirmed_by_query=any(_body(r).get("status") in ("CANCELED", "FILLED", "EXPIRED") for r in confirming),
        terminal_once=terminal is not None and _journal_terminal_lines(session, order_id) == 1,
        cancel_error=error,
    )


def _run_d5(session: Any, config: Any) -> ScenarioEvidence:
    symbol = session.config.symbols[0]
    trace_a, _, start_a = _cycle(
        session, [FaultRule("POST", ORDER_PATH, action=FaultAction.SYNTHETIC_STATUS, status=429, body=b"{}")]
    )
    posts_a = _mutations(_records(session, start_a, trace_a.wire_end), "POST")
    limited = next((r for r in posts_a if r.status_code == 429), None)
    limited_id = _order_id(limited) if limited is not None else None
    terminal = session.journal.terminal(str(limited_id)) if limited_id else None

    rule_b = FaultRule(
        "DELETE", ORDER_PATH, action=FaultAction.SYNTHETIC_STATUS, status=429, body=b'{"code":-1003,"msg":"Too many requests"}'
    )
    trace_b, _, start_b = _cycle(session, [rule_b])
    deletes_b = _mutations(_records(session, start_b, trace_b.wire_end), "DELETE")
    first_limited = next((r for r in deletes_b if r.status_code == 429), None)
    cancel_id = _order_id(first_limited) if first_limited is not None else None
    open_ids = {str(row.get("clientOrderId")) for row in session.client.open_orders() if row.get("symbol") == symbol}
    if first_limited is None:
        cancel_outcome = "not_reached"
    elif any(_order_id(r) == cancel_id and r.seq > first_limited.seq for r in deletes_b):
        cancel_outcome = "retried"
    elif trace_b.escaped is not None:
        cancel_outcome = "aborted"
    else:
        cancel_outcome = "abandoned"
    resting_left = cancel_id is not None and cancel_id in open_ids
    if resting_left:
        cancel_probe(session, symbol, str(cancel_id))

    retry_after = int(config.retry_after_probe_s)
    rule_c = FaultRule(
        "GET", BOOK_TICKER_PATH, action=FaultAction.SYNTHETIC_STATUS, status=429, headers=(("Retry-After", str(retry_after)),)
    )
    trace_c, _, start_c = _cycle(session, [rule_c])
    records_c = _records(session, start_c, trace_c.wire_end)
    limited_read = next((r for r in records_c if r.path == BOOK_TICKER_PATH and r.status_code == 429), None)
    following = next((r for r in records_c if limited_read is not None and r.seq > limited_read.seq), None)
    sleep_s = following.at - limited_read.at if limited_read is not None and following is not None else 0.0
    elapsed = limited_read.at - records_c[0].at if limited_read is not None and records_c else 0.0
    remaining = max(0.0, float(config.scenario_timeout_s) - elapsed)
    return ev(
        "D5",
        not_placed=terminal is not None and terminal.status == "NOT_PLACED",
        retried_later=limited is not None and any(r.seq > limited.seq and _order_id(r) != limited_id for r in posts_a),
        no_halt=trace_a.escaped is None,
        cancel_outcome=cancel_outcome,
        resting_left=resting_left,
        rate_limited_read=limited_read is not None,
        retry_after_s=retry_after,
        sleep_s=sleep_s,
        remaining_window_s=remaining,
        sleep_exceeds_window=limited_read is not None and sleep_s > remaining,
        escaped=[trace_a.escaped, trace_b.escaped, trace_c.escaped],
    )


def _run_d6(session: Any, config: Any) -> ScenarioEvidence:
    ban_body = b'{"code":-1003,"msg":"Way too many requests; IP banned"}'
    phases: list[dict[str, Any]] = []
    for method, path in (("GET", BOOK_TICKER_PATH), ("POST", ORDER_PATH)):
        rule = FaultRule(
            method, path, action=FaultAction.SYNTHETIC_STATUS, status=418, headers=(("Retry-After", "120"),), body=ban_body
        )
        trace, tap, start = _cycle(session, [rule])
        records = _records(session, start, trace.wire_end)
        ban = next((r for r in records if r.status_code == 418), None)
        after = [r for r in records if ban is not None and r.seq > ban.seq]
        captured = any(
            any("ban" in key.lower() or "retry" in key.lower() for key in event.fields)
            for event in tap.events
            if ban is not None and event.wire_index >= ban.seq
        )
        phases.append(
            {
                "method": method,
                "banned": ban is not None,
                "escaped": trace.escaped,
                "records_after": len(after),
                "captured": captured,
            }
        )
    if not all(phase["banned"] for phase in phases):
        raise DrillRefused("drill refused: d6_fault_not_reached")
    return ev(
        "D6",
        records_after_ban=sum(int(phase["records_after"]) for phase in phases),
        ban_horizon_captured=any(bool(phase["captured"]) for phase in phases),
        halted=all(phase["escaped"] is not None for phase in phases),
        phases=phases,
    )


_ERROR_BODIES = (
    (403, b"<html><body>Request blocked</body></html>"),
    (502, b"Unknown error, please check your request or try again later."),
    (503, b"Service Unavailable"),
)


def _run_d7(session: Any, config: Any) -> ScenarioEvidence:
    symbol = session.config.symbols[0]
    reads: list[dict[str, Any]] = []
    posts: list[dict[str, Any]] = []
    for status, body in _ERROR_BODIES:
        start = len(session.transport.records)
        rule = FaultRule("GET", BOOK_TICKER_PATH, action=FaultAction.SYNTHETIC_STATUS, status=status, body=body, times=_GET_BUDGET)
        with session.transport.inject([rule]):
            try:
                session.client.book_ticker(symbol)
                outcome = "returned"
            except LiveTradingError as exc:
                outcome = type(exc).__name__
        attempts = sum(r.path == BOOK_TICKER_PATH for r in _records(session, start))
        expected = "VenueError" if status < 500 else "TransientReadError"
        reads.append({"status": status, "outcome": outcome, "attempts": attempts, "bounded": outcome == expected and attempts <= _GET_BUDGET})
    for status, body in _ERROR_BODIES:
        params = _resting_params(session)
        start = len(session.transport.records)
        rule = FaultRule("POST", ORDER_PATH, action=FaultAction.SYNTHETIC_STATUS, status=status, body=body)
        with session.transport.inject([rule]):
            try:
                submit_probe(session, params)
                outcome = "returned"
            except LiveTradingError as exc:
                outcome = type(exc).__name__
        if outcome == "OrderStatusUnknown":
            settle_journal(session)
        sends = [r for r in _mutations(_records(session, start), "POST") if _order_id(r) == params["newClientOrderId"]]
        posts.append({"status": status, "outcome": outcome, "sends": len(sends)})
    own_open = [
        row
        for row in session.client.open_orders()
        if str(row.get("clientOrderId", "")).startswith(session.client_order_prefix + "-")
    ]
    server = [p for p in posts if p["status"] >= 500]
    client_side = [p for p in posts if p["status"] < 500]
    return ev(
        "D7",
        get_retries_bounded=all(bool(r["bounded"]) for r in reads),
        post_unknown_no_repost=all(p["outcome"] == OrderStatusUnknown.__name__ and p["sends"] == 1 for p in server),
        post_fail_closed=all(p["outcome"] == VenueError.__name__ for p in client_side),
        no_duplicate_exposure=not own_open and _venue_position(session, symbol) == _signed_fold(session, symbol),
        post_resent_after_5xx=any(int(p["sends"]) > 1 for p in server),
        reads=reads,
        posts=posts,
    )


_CORRUPTIONS = (
    ("corrupt", FaultAction.CORRUPT_BODY, b"<html>gateway</html>", 0),
    ("truncated", FaultAction.TRUNCATE_BODY, b"{}", 12),
    ("empty", FaultAction.CORRUPT_BODY, b"", 0),
)


def _run_d8(session: Any, config: Any) -> ScenarioEvidence:
    variants = []
    for name, action, body, cut in _CORRUPTIONS:
        rule = FaultRule("POST", ORDER_PATH, action=action, body=body, truncate_to=cut)
        trace, tap, start = _cycle(session, [rule])
        post = next(iter(_mutations(_records(session, start, trace.wire_end), "POST")), None)
        if post is None:
            raise DrillRefused("drill refused: d8_fault_not_reached")
        order_id = _order_id(post)
        if _event_for(tap, {"order_unknown_adopted"}, order_id) is not None:
            path = "resolved_by_query"
        elif _event_for(tap, _UNKNOWN_EVENTS, order_id) is not None:
            path = "unknown_outcome_query"
        elif trace.escaped is not None and trace.escaped not in ("OrderStatusUnknown", "ExecutionInterrupted"):
            path = f"escaped:{trace.escaped}"
        elif _event_for(tap, {"order_posted"}, order_id) is not None:
            path = "treated_as_active"
        else:
            path = "unobserved"
        variants.append({"variant": name, "path": path, "order_id": order_id, "venue_accepted": post.status_code == 200})
    return ev(
        "D8",
        resolved_by_query=all(v["path"] == "resolved_by_query" for v in variants),
        treated_as_active_without_query=any(
            v["path"] == "treated_as_active" or str(v["path"]).startswith("escaped:") for v in variants
        ),
        variants=variants,
    )


def _run_d9(session: Any, config: Any) -> ScenarioEvidence:
    params = _resting_params(session)
    order_id = params["newClientOrderId"]
    delay = session.config.recv_window_ms / 1000 + float(config.recv_window_overshoot_s)
    start = len(session.transport.records)
    with session.transport.inject([FaultRule("POST", ORDER_PATH, action=FaultAction.DELAY_DELIVER, delay_s=delay)]):
        try:
            submit_probe(session, params)
        except LiveTradingError as exc:
            error: str | None = type(exc).__name__
        else:
            error = None
    records = _records(session, start)
    posts = [r for r in _mutations(records, "POST") if _order_id(r) == order_id]
    first, last = (posts[0].seq, posts[-1].seq) if posts else (0, 0)
    live = [row for row in session.client.open_orders() if row.get("clientOrderId") == order_id]
    cancel_probe(session, params["symbol"], order_id)
    return ev(
        "D9",
        post_count=len(posts),
        first_code=_code(posts[0]) if posts else None,
        resync_count=sum(r.path == "/fapi/v1/time" and first < r.seq < last for r in records),
        no_duplicate=len(live) <= 1 and _fills_unique(session),
        send_error=error,
    )


def _run_d10(session: Any, config: Any) -> ScenarioEvidence:
    budget = _GET_BUDGET
    intent = make_intent(session, session.config.symbols[0])
    policy_limit = _max_read_failures(session)
    phase_one = FaultRule("GET", BOOK_TICKER_PATH, action=FaultAction.SYNTHETIC_STATUS, status=503, body=b"busy", times=2 * budget)
    phase_two = FaultRule(
        "GET", BOOK_TICKER_PATH, action=FaultAction.SYNTHETIC_STATUS, status=503, body=b"busy", times=(policy_limit + 1) * budget
    )
    with ExitStack() as stack:
        tap = stack.enter_context(AuditTap(session))
        stack.enter_context(session.transport.inject([phase_one]))
        installed: list[int] = []

        def escalate(tick: int) -> None:
            skipped = len(tap.named("tick_skipped_read_failure"))
            if not installed and skipped >= 2 and tick >= skipped:
                installed.append(len(tap.events))
                stack.enter_context(session.transport.inject([phase_two]))

        trace = run_cycle(session, [intent], on_tick=escalate)
        book_events = [e for e in tap.named("tick_skipped_read_failure") if e.fields.get("scope") == "books"]
    open_after = [
        row
        for row in session.client.open_orders()
        if str(row.get("clientOrderId", "")).startswith(session.client_order_prefix + "-")
    ]
    later = [e for e in book_events if installed and tap.events.index(e) >= installed[0]]
    order_read = _order_read_containment(session)
    exceeded = trace.escaped == TransientReadError.__name__
    return ev(
        "D10",
        tick_skipped=len(book_events) >= 2,
        counters_reset=bool(later) and later[0].fields.get("consecutive") == 1,
        aborted_with_cleanup=exceeded and not open_after,
        no_live_order=not open_after,
        max_failures_exceeded=exceeded,
        order_read_contained=order_read,
        consecutive_trace=[e.fields.get("consecutive") for e in book_events],
    )


def _max_read_failures(session: Any) -> int:
    from src.live.testnet_drill.execution import execution_policy

    return int(execution_policy(session).max_consecutive_read_failures)


def _order_read_containment(session: Any) -> bool:
    """Two transient GET order failures after the first post must not escape the window."""
    rule = FaultRule("GET", ORDER_PATH, action=FaultAction.SYNTHETIC_STATUS, status=503, body=b"busy", times=2 * _GET_BUDGET)
    with ExitStack() as stack:
        tap = stack.enter_context(AuditTap(session))
        armed: list[bool] = []

        def arm(tick: int) -> None:
            if not armed and tap.named("order_posted"):
                armed.append(True)
                stack.enter_context(session.transport.inject([rule]))

        trace = run_cycle(session, [make_intent(session, session.config.symbols[0])], on_tick=arm)
    return bool(armed) and trace.escaped != TransientReadError.__name__


def _eval_d1(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = p["unresolved_adopted"] is True and p["single_post"] is True and p["booked_once"] is True
    return result_of(
        evidence,
        "D1",
        Verdict.PASS if passed else Verdict.FAIL,
        (("lost_response_adopted", passed),),
        hypotheses=("H6",),
        hypothesis_outcome=_h6(p),
    )


def _h6(payload: Any) -> str:
    match = payload.get("watermark_match")
    return "REFUTED" if match is True else ("CONFIRMED" if match is False else "INCONCLUSIVE")


def _eval_d2(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = p["booked_from_query"] is True and p["position_parity"] is True and p["no_repost"] is True
    return result_of(
        evidence,
        "D2",
        Verdict.PASS if passed else Verdict.FAIL,
        (("ioc_fill_adopted", passed),),
        hypotheses=("H6",),
        hypothesis_outcome=_h6(p),
    )


def _eval_d3(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    if p["reposted_early"] is True:
        return result_of(
            evidence,
            "D3",
            Verdict.FAIL,
            (("no_early_repost", False),),
            proposal_rationale=f"wire reposted before horizon post_seqs={p.get('post_seqs', p.get('post_count'))}",
        )
    passed = (
        p["reposted_after_horizon"] is True
        and p["miss_limit_reached"] is True
        and p["journal_not_placed"] is True
        and p["never_two_live"] is True
    )
    return result_of(evidence, "D3", Verdict.PASS if passed else Verdict.FAIL, (("horizon_repost", passed),))


def _eval_d4(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = int(p["delete_count"]) == 1 and p["confirmed_by_query"] is True and p["terminal_once"] is True
    return result_of(evidence, "D4", Verdict.PASS if passed else Verdict.FAIL, (("single_cancel", passed),))


def _eval_d5(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    sleep_ok = p["rate_limited_read"] is True and float(p["sleep_s"]) >= float(p["retry_after_s"])
    passed = p["not_placed"] is True and p["retried_later"] is True and p["no_halt"] is True and sleep_ok
    h3 = "INCONCLUSIVE"
    if p["cancel_outcome"] == "retried" and p["resting_left"] is False:
        h3 = "REFUTED"
    elif p["resting_left"] is True:
        h3 = "CONFIRMED"
    h10 = "INCONCLUSIVE" if p["rate_limited_read"] is not True else ("CONFIRMED" if p["sleep_exceeds_window"] else "REFUTED")
    return result_of(
        evidence,
        "D5",
        Verdict.PASS if passed else Verdict.FAIL,
        (("rate_limit_contained", passed),),
        hypotheses=("H3", "H10"),
        hypothesis_outcomes=(("H3", h3), ("H10", h10)),
    )


def _eval_d6(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    after = int(p["records_after_ban"])
    if after > 0 or p["halted"] is not True:
        return result_of(
            evidence,
            "D6",
            Verdict.FAIL,
            (("no_calls_after_ban", False),),
            hypotheses=("H4",),
            proposal_rationale=f"wire continued after 418 records={after}",
        )
    captured = p["ban_horizon_captured"] is True
    return result_of(
        evidence,
        "D6",
        Verdict.PASS,
        (("ban_halted", True),),
        hypotheses=("H4",),
        hypothesis_outcome="REFUTED" if captured else "CONFIRMED",
    )


def _eval_d7(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    if p["post_resent_after_5xx"] is True:
        return result_of(evidence, "D7", Verdict.FAIL, (("no_blind_repost", False),), proposal_rationale="POST re-sent after 5xx")
    passed = (
        p["get_retries_bounded"] is True
        and p["post_unknown_no_repost"] is True
        and p["post_fail_closed"] is True
        and p["no_duplicate_exposure"] is True
    )
    return result_of(evidence, "D7", Verdict.PASS if passed else Verdict.FAIL, (("error_body_contained", passed),))


def _eval_d8(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    if p["treated_as_active_without_query"] is True:
        return result_of(
            evidence,
            "D8",
            Verdict.FAIL,
            (("corrupt_body_queried", False),),
            hypotheses=("H11",),
            hypothesis_outcome="CONFIRMED",
        )
    passed = p["resolved_by_query"] is True
    return result_of(
        evidence,
        "D8",
        Verdict.PASS if passed else Verdict.FAIL,
        (("corrupt_body_queried", passed),),
        hypotheses=("H11",),
        hypothesis_outcome="REFUTED" if passed else "INCONCLUSIVE",
    )


def _eval_d9(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = (
        int(p["post_count"]) == 2
        and p["first_code"] == -1021
        and int(p["resync_count"]) == 1
        and p["no_duplicate"] is True
    )
    return result_of(evidence, "D9", Verdict.PASS if passed else Verdict.FAIL, (("clock_resend_once", passed),))


def _eval_d10(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = (
        p["tick_skipped"] is True
        and p["counters_reset"] is True
        and p["max_failures_exceeded"] is True
        and p["aborted_with_cleanup"] is True
        and p["no_live_order"] is True
        and p["order_read_contained"] is True
    )
    return result_of(evidence, "D10", Verdict.PASS if passed else Verdict.FAIL, (("transient_reads_bounded", passed),))


_D_RUNNERS: dict[str, Any] = {
    "D1": _run_d1,
    "D2": _run_d2,
    "D3": _run_d3,
    "D4": _run_d4,
    "D5": _run_d5,
    "D6": _run_d6,
    "D7": _run_d7,
    "D8": _run_d8,
    "D9": _run_d9,
    "D10": _run_d10,
}

_D_EVALUATORS: dict[str, Any] = {
    "D1": _eval_d1,
    "D2": _eval_d2,
    "D3": _eval_d3,
    "D4": _eval_d4,
    "D5": _eval_d5,
    "D6": _eval_d6,
    "D7": _eval_d7,
    "D8": _eval_d8,
    "D9": _eval_d9,
    "D10": _eval_d10,
}

_D_TITLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "D1": ("lost response resting", ("H6",)),
    "D2": ("lost response ioc fill", ("H6",)),
    "D3": ("dropped request", ()),
    "D4": ("lost response cancel", ()),
    "D5": ("rate limits", ("H3", "H10")),
    "D6": ("ip ban synthetic", ("H4",)),
    "D7": ("non json error bodies", ()),
    "D8": ("corrupt success body", ("H11",)),
    "D9": ("delay past recv window", ()),
    "D10": ("transient reads in window", ()),
}


def scenario_specs() -> tuple[ScenarioSpec, ...]:
    """Group D scenario registry."""
    return tuple(
        ScenarioSpec(scenario_id, "D", title, hypotheses, isolated_run(_D_RUNNERS[scenario_id]), _D_EVALUATORS[scenario_id])
        for scenario_id, (title, hypotheses) in sorted(_D_TITLES.items(), key=lambda item: int(item[0][1:]))
    )
