"""Group F account/venue-state scenarios (F1..F7)."""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from decimal import Decimal
from itertools import pairwise
from typing import Any

import pandas as pd

from src.live.errors import LiveTradingError
from src.live.testnet_drill.guard import DrillRefused
from src.live.testnet_drill.report import Verdict
from src.live.testnet_drill.scenarios.base import (
    ScenarioEvidence,
    ScenarioSpec,
    ev,
    isolated_run,
    result_of,
)
from src.live.testnet_drill.scenarios.wire import BOOK_TICKER_PATH, ORDER_PATH

_SNAPSHOT_DEPENDENT_CHECKS = frozenset({"position_reconciliation", "venue_leverage_plan"})


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _resting(session: Any, **overrides: Any) -> dict[str, Any]:
    from src.live.testnet_drill.scenarios.lifecycle_runs import _limit

    return {**_limit(session, distance=True), **overrides}


def _open_long(session: Any) -> Decimal:
    """Open a probe-sized long through a journaled taker probe and production settlement."""
    from src.live.testnet_drill.execution import make_intent, settle_journal, submit_probe

    symbol = session.config.symbols[0]
    intent = make_intent(session, symbol)
    ask = str(session.client.book_ticker(symbol)["askPrice"])
    submit_probe(
        session,
        {"symbol": symbol, "side": "BUY", "type": "LIMIT", "timeInForce": "IOC", "quantity": str(intent.quantity), "price": ask},
    )
    settle_journal(session)
    return intent.quantity


def _positions(session: Any) -> dict[str, Decimal]:
    from src.live.testnet_drill.session import _read_positions

    return _read_positions(session.client)


def _sweep(session: Any) -> Any:
    from src.live.order_cancel import cancel_orphan_orders

    return cancel_orphan_orders(
        session.client,
        session.client_order_prefix,
        session.audit,
        journal=session.journal,
        now=_now(),
        taker_fee_bps=float(session.settings.taker_fee_bps),
    )


def _fresh(session: Any) -> Any:
    from src.live.order_journal import OrderJournal

    return replace(session, journal=OrderJournal(session.journal.path))


def _with_settings(session: Any, **update: Any) -> Any:
    return replace(session, settings=session.settings.model_copy(update=update))


def _restart(session: Any) -> tuple[Any, str | None]:
    """Production restart on a freshly read journal; callers must re-read the journal afterwards too."""
    from src.live.testnet_drill.scenarios.restart_recovery import restart_recovery

    try:
        return restart_recovery(_fresh(session)), None
    except LiveTradingError as exc:
        return None, type(exc).__name__


def _run_f1(session: Any, config: Any) -> ScenarioEvidence:
    from src.live.derisk import derisk_filter
    from src.live.testnet_drill.execution import AuditTap, make_intent

    symbol = session.config.symbols[0]
    foreign_id = f"{session.foreign_order_prefix}-{session.next_client_order_id(symbol).rsplit('-', 1)[-1]}"
    session.guarded_new_order(_resting(session, newClientOrderId=foreign_id), scope="foreign")
    try:
        with AuditTap(session) as tap:
            sweep = _sweep(session)
        still_open = any(row.get("clientOrderId") == foreign_id for row in session.client.open_orders())
        plan = derisk_filter([make_intent(session, symbol)], _positions(session), excluded_symbols=sweep.foreign_symbols)
        blocked = {intent.symbol: reason for intent, reason in plan.blocked}
        _, halted = _restart(_with_settings(session, derisk_mode_enabled=False))
        recon, derisk_error = _restart(session)
    finally:
        with contextlib.suppress(LiveTradingError):
            session.client.cancel_order(symbol, foreign_id)
    return ev(
        "F1",
        foreign_audited=any(e.fields.get("symbol") == symbol for e in tap.named("foreign_open_order")),
        foreign_cancelled=not still_open,
        symbol_excluded=symbol in sweep.foreign_symbols and blocked.get(symbol) == "foreign_order_symbol",
        halt_when_derisk_off=halted == "ForeignOpenOrderError",
        derisk_when_enabled=derisk_error is None
        and recon is not None
        and "foreign_open_orders" in recon.derisk_reasons,
    )


def _run_f2(session: Any, config: Any) -> ScenarioEvidence:
    from src.live.ledger_resync import run_ledger_resync
    from src.live.testnet_drill.execution import AuditTap, make_intent, run_cycle
    from src.live.testnet_drill.seams import production_bound

    # Several production entries append to the one journal file through their own handles, so every step below
    # starts from a fresh read (stale handles would reuse sequence numbers).
    symbol = session.config.symbols[0]
    qty = _open_long(_fresh(session))
    _restart(session)
    variants = []
    for name, side, amount in (("increase", "BUY", qty), ("decrease", "SELL", qty), ("full_flatten", "SELL", None)):
        held = _positions(session).get(symbol, Decimal(0))
        if side == "SELL" and held <= 0:
            _open_long(_fresh(session))
            _restart(session)
            held = _positions(session).get(symbol, Decimal(0))
        delta = min(amount, held) if amount is not None and side == "SELL" else (amount if amount is not None else held)
        if delta <= 0:
            raise DrillRefused("drill refused: f2_requires_position")
        # External change: deliberately NOT journaled, so only the venue sees it.
        session.guarded_new_order(
            {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": str(delta),
                **({"reduceOnly": "true"} if side == "SELL" else {}),
            }
        )
        strict = _with_settings(session, venue_force_close_auto_adopt=False)
        _, halted = _restart(_with_settings(strict, derisk_mode_enabled=False))
        recon, error = _restart(strict)
        with production_bound(session):
            plan = run_ledger_resync(session.settings, apply=False, now=_now())
            applied = run_ledger_resync(session.settings, apply=True, now=_now())
        after, after_error = _restart(strict)
        fresh = _fresh(session)
        long_now = _positions(fresh).get(symbol, Decimal(0)) > 0
        intent = make_intent(fresh, symbol, side="SELL" if long_now else "BUY", reduce_only=long_now)
        with AuditTap(fresh) as tap:
            run_cycle(fresh, [intent])
        variants.append(
            {
                "variant": name,
                "delta": str(delta),
                "halt_without_derisk": halted == "ReconciliationBreach",
                "breach_detected": error is None and recon is not None and "reconciliation_breach" in recon.derisk_reasons,
                "adoption_listed": any(b.symbol == symbol for b in plan.adjustments) and applied.applied,
                "breach_cleared": after_error is None and after is not None and not after.derisk_reasons,
                "next_cycle_trades": bool(tap.named("order_posted")),
            }
        )
    return ev(
        "F2",
        breach_detected=all(v["breach_detected"] for v in variants),
        derisk_or_halt=all(v["halt_without_derisk"] for v in variants),
        adoption_listed=all(v["adoption_listed"] for v in variants),
        breach_cleared=all(v["breach_cleared"] and v["next_cycle_trades"] for v in variants),
        variants=variants,
    )


def _open_stale_session(session: Any) -> Any:
    from src.live.testnet_drill.guard import _PRODUCTION_PATH_FIELDS
    from src.live.testnet_drill.session import open_drill_session

    settings = session.settings.model_copy(update=dict.fromkeys(_PRODUCTION_PATH_FIELDS))
    stale_run = ("st" + session.run_id)[:16]
    return open_drill_session(settings, session.config, run_id=stale_run, workspace=session.workspace / "stale_run")


def _run_f3(session: Any, config: Any) -> ScenarioEvidence:
    from src.live.testnet_drill.execution import AuditTap, submit_probe
    from src.live.testnet_drill.session import close_session_handles

    symbol = session.config.symbols[0]
    stale = _open_stale_session(session)
    try:
        params = _resting(stale, price=str(stale.client.book_ticker(symbol)["bidPrice"]))
        submit_probe(stale, params)
    finally:
        close_session_handles(stale)
    with AuditTap(session) as tap:
        sweep = _sweep(session)
    order = session.client.query_order(symbol, params["newClientOrderId"])
    executed = Decimal(str(order.get("executedQty", "0")))
    settled = [f for f in session.journal.fills_after(-1) if f.client_order_id == params["newClientOrderId"]]
    unconfirmed = {u.client_order_id for u in sweep.unconfirmed}
    withheld_ok = True
    if params["newClientOrderId"] in unconfirmed:
        recon, _ = _restart(session)
        withheld_ok = recon is not None and symbol in recon.frozen_symbols
    return ev(
        "F3",
        orphan_cancelled=order.get("status") in ("CANCELED", "FILLED")
        and _audited(tap, "orphan_cancelled", params["newClientOrderId"]),
        orphan_settled=sum((f.quantity for f in settled), Decimal(0)) == executed
        and all(f.kind == "orphan_settlement" and Decimal(str(f.fee_bps)) == Decimal(str(session.settings.taker_fee_bps)) for f in settled),
        symbol_withheld_on_unconfirmed=withheld_ok,
        executed_qty=str(executed),
    )


def _audited(tap: Any, event: str, order_id: str) -> bool:
    return any(e.fields.get("client_order_id") == order_id for e in tap.named(event))


def _run_f4(session: Any, config: Any) -> ScenarioEvidence:
    from src.live.account import RECONCILE_QTY_TOLERANCE_FRACTION
    from src.live.testnet_drill.execution import cancel_probe, submit_probe

    symbol = session.config.symbols[0]
    params = _resting(session)
    params["price"] = str(session.client.book_ticker(symbol)["bidPrice"])
    submit_probe(session, params)
    trials, mismatches, exceed = int(config.snapshot_trials), 0, False
    samples = []
    try:
        for _ in range(trials):
            account = session.client.account()
            risk = _positions(session).get(symbol, Decimal(0))
            account_qty = next(
                (Decimal(str(row.get("positionAmt", "0"))) for row in account.get("positions", []) if row.get("symbol") == symbol),
                Decimal(0),
            )
            samples.append({"account": str(account_qty), "position_risk": str(risk)})
            if account_qty != risk:
                mismatches += 1
                scale = max(abs(account_qty), abs(risk))
                exceed |= abs(account_qty - risk) > Decimal(str(RECONCILE_QTY_TOLERANCE_FRACTION)) * scale
    finally:
        cancel_probe(session, symbol, params["newClientOrderId"])
    return ev("F4", inconsistency_rate=mismatches / trials, breach_would_exceed=exceed, samples=samples)


def _run_f5(session: Any, config: Any) -> ScenarioEvidence:
    if float(config.funding_soak_s) <= 0:
        return ev("F5", funding_events=0, funding_parity=False, skipped=True, sign_convention_ok=False)
    import logging

    logger = logging.getLogger("TestnetDrill")
    symbol = session.config.symbols[0]
    start_ms = int(time.time() * 1000)
    _open_long(session)
    deadline = time.monotonic() + float(config.funding_soak_s)
    rows: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        rows = [
            r
            for r in session.client.income(start_time_ms=start_ms, end_time_ms=int(time.time() * 1000))
            if r.get("incomeType") == "FUNDING_FEE" and r.get("asset") == "USDT" and r.get("symbol") == symbol
        ]
        logger.info("[DATA] testnet-drill F5 soak funding_events=%d remaining_s=%.0f", len(rows), deadline - time.monotonic())
        time.sleep(min(float(config.funding_poll_s), max(0.0, deadline - time.monotonic())))
    position = _positions(session).get(symbol, Decimal(0))
    rates = session.client.request(
        "GET", "/fapi/v1/fundingRate", {"symbol": symbol, "startTime": start_ms, "limit": 1000}
    )
    by_time = {int(r["fundingTime"]): r for r in rates if isinstance(r, dict)}
    expected, actual, signs_ok = Decimal(0), Decimal(0), True
    for row in rows:
        amount = Decimal(str(row["income"]))
        actual += amount
        rate_row = min(by_time.items(), key=lambda item: abs(item[0] - int(row["time"])), default=(0, None))[1]
        if rate_row is None:
            signs_ok = False
            continue
        rate, mark = Decimal(str(rate_row["fundingRate"])), Decimal(str(rate_row.get("markPrice") or "0"))
        expected += -position * mark * rate
        if rate != 0 and amount != 0 and (amount > 0) != (position * rate < 0):
            signs_ok = False
    return ev(
        "F5",
        funding_events=len(rows),
        funding_parity=len(rows) >= int(config.funding_soak_min_funding_events)
        and abs(actual - expected) <= config.fee_abs_tolerance_usdt,
        sign_convention_ok=signs_ok,
        skipped=False,
        funding_sum=str(actual),
        expected_sum=str(expected),
    )


@contextmanager
def _override_window(session: Any, rules: list[Any], start_tick: int, stop_tick: int) -> Iterator[Any]:
    with ExitStack() as outer:
        window: list[ExitStack] = []

        def on_tick(tick: int) -> None:
            if tick == start_tick and not window:
                stack = ExitStack()
                stack.enter_context(session.transport.inject(rules))
                window.append(stack)
            elif tick == stop_tick and window:
                window[0].close()

        outer.callback(lambda: window[0].close() if window else None)
        yield on_tick


def _run_f6(session: Any, config: Any) -> ScenarioEvidence:
    from src.live.rest import parse_rate_limits
    from src.live.testnet_drill.execution import execution_policy, make_intent, run_cycle
    from src.live.testnet_drill.faults import FaultAction, FaultRule

    limits = parse_rate_limits(session.client.exchange_info())
    policy = execution_policy(session)
    hot = str(math.floor(policy.rate_weight_budget_fraction * limits.request_weight_1m) + 1)
    rules = [
        FaultRule(method, path, action=FaultAction.OVERRIDE_HEADERS, headers=(("X-MBX-USED-WEIGHT-1M", hot),), times=10**6)
        for method, path in (
            ("GET", BOOK_TICKER_PATH),
            ("GET", ORDER_PATH),
            ("POST", ORDER_PATH),
            ("DELETE", ORDER_PATH),
            ("GET", "/fapi/v1/openOrders"),
            ("GET", "/fapi/v1/depth"),
            ("GET", "/fapi/v2/account"),
            ("GET", "/fapi/v2/positionRisk"),
        )
    ]
    intents = [make_intent(session, symbol, leg_index=i) for i, symbol in enumerate(session.config.symbols)]
    start = len(session.transport.records)
    with _override_window(session, rules, 1, 4) as on_tick:
        trace = run_cycle(session, intents, on_tick=on_tick)
    records = session.transport.records[start : trace.wire_end]
    observed = [
        int(value)
        for r in records
        for key, value in r.response_headers
        if key.lower() == "x-mbx-used-weight-1m" and value != hot
    ]
    sleeps = trace.sleeps
    doubled = any(b > a for a, b in pairwise(sleeps))
    recovered = any(b < a for a, b in pairwise(sleeps))
    return ev(
        "F6",
        limits_match=bool(observed) and max(observed) <= limits.request_weight_1m,
        throttle_reacts=doubled and recovered,
        no_429=not any(r.status_code == 429 for r in records),
        sleeps=sleeps,
        max_used_weight=max(observed, default=None),
    )


def _run_f7(session: Any, config: Any) -> ScenarioEvidence:
    from src.live.preflight import run_preflight
    from src.live.rest import BinanceFuturesRestClient
    from src.live.testnet_drill.scenarios.account_config import _restore_configuration
    from src.live.testnet_drill.scenarios.cycle_e2e import write_weights_artifact
    from src.live.testnet_drill.seams import GuardedVenueClient

    client = GuardedVenueClient(session)
    today = _now().normalize()
    artifact = write_weights_artifact(session, {today: dict.fromkeys(session.config.symbols[:1], 0.0)})
    sealed = artifact.with_name(artifact.name + ".enc")
    artifact = sealed if sealed.exists() else artifact

    def checks(settings: Any, order_client: Any) -> dict[str, tuple[bool, str]]:
        report = run_preflight(settings, artifact, market_client=client, order_client=order_client)
        return {c.name: (c.passed, c.detail) for c in report.checks}

    clean = checks(session.settings, client)
    keyless = BinanceFuturesRestClient(
        session.settings.order_base_url, None, None, session.settings.mode, session.audit,
        recv_window_ms=session.config.recv_window_ms, session=session.transport,
    )
    no_key = checks(session.settings, keyless)
    with _restore_configuration(session, "/fapi/v1/positionSide/dual", {"dualSidePosition": "false"}):
        session.client.request("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "true"}, signed=True)
        dual = checks(session.settings, client)
    tax_off = checks(session.settings.model_copy(update={"tax_collection_enabled": False}), client)

    def flipped(other: dict[str, tuple[bool, str]]) -> list[str]:
        return sorted(name for name in clean if clean[name][0] != other.get(name, clean[name])[0])

    def flips_exactly(other: dict[str, tuple[bool, str]], named: str) -> bool:
        # Checks that consume the account snapshot fail with it; anything else flipping is a mis-attributed check.
        changed = set(flipped(other))
        return named in changed and changed <= {named, *_SNAPSHOT_DEPENDENT_CHECKS}

    tax = tax_off.get("tax_collection_ready", (False, ""))
    return ev(
        "F7",
        clean_pass=all(passed for passed, _ in clean.values()),
        removal_flips=flips_exactly(no_key, "account_configuration") and flips_exactly(dual, "account_configuration"),
        warning_only=tax[0] is True and tax[1].startswith("WARNING:") and not flipped(tax_off),
        clean_checks={name: passed for name, (passed, _) in clean.items()},
        flips={"key": flipped(no_key), "dual": flipped(dual), "tax": flipped(tax_off)},
    )


def _eval_f1(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = (
        p["foreign_audited"] is True
        and p["foreign_cancelled"] is False
        and p["symbol_excluded"] is True
        and p["halt_when_derisk_off"] is True
        and p["derisk_when_enabled"] is True
    )
    return result_of(evidence, "F1", Verdict.PASS if passed else Verdict.FAIL, (("foreign_de_risk", passed),))


def _eval_f2(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = all(p[key] is True for key in ("breach_detected", "derisk_or_halt", "adoption_listed", "breach_cleared"))
    return result_of(evidence, "F2", Verdict.PASS if passed else Verdict.FAIL, (("external_position", passed),))


def _eval_f3(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = p["orphan_cancelled"] is True and p["orphan_settled"] is True and p["symbol_withheld_on_unconfirmed"] is True
    return result_of(evidence, "F3", Verdict.PASS if passed else Verdict.FAIL, (("stale_namespace", passed),))


def _eval_f4(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    confirmed = float(p["inconsistency_rate"]) > 0 and p["breach_would_exceed"] is True
    return result_of(
        evidence,
        "F4",
        Verdict.OBSERVED,
        (("snapshot_atomicity_measured", True),),
        hypotheses=("H9",),
        hypothesis_outcome="CONFIRMED" if confirmed else "INCONCLUSIVE",
    )


def _eval_f5(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    if p["skipped"] is True:
        return result_of(evidence, "F5", Verdict.SKIPPED, (("funding_soak", True),))
    passed = p["funding_parity"] is True and p["sign_convention_ok"] is True
    return result_of(evidence, "F5", Verdict.PASS if passed else Verdict.FAIL, (("funding_soak", passed),))


def _eval_f6(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = p["limits_match"] is True and p["throttle_reacts"] is True and p["no_429"] is True
    return result_of(evidence, "F6", Verdict.PASS if passed else Verdict.FAIL, (("rate_budget", passed),))


def _eval_f7(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = p["clean_pass"] is True and p["removal_flips"] is True and p["warning_only"] is True
    return result_of(evidence, "F7", Verdict.PASS if passed else Verdict.FAIL, (("preflight", passed),))


_F_RUNNERS: dict[str, Any] = {
    "F1": _run_f1,
    "F2": _run_f2,
    "F3": _run_f3,
    "F4": _run_f4,
    "F5": _run_f5,
    "F6": _run_f6,
    "F7": _run_f7,
}

_F_EVALUATORS: dict[str, Any] = {
    "F1": _eval_f1,
    "F2": _eval_f2,
    "F3": _eval_f3,
    "F4": _eval_f4,
    "F5": _eval_f5,
    "F6": _eval_f6,
    "F7": _eval_f7,
}

_F_TITLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "F1": ("foreign open order", ()),
    "F2": ("external position change", ()),
    "F3": ("stale namespace order", ()),
    "F4": ("snapshot atomicity", ("H9",)),
    "F5": ("funding soak", ()),
    "F6": ("rate limit budget", ()),
    "F7": ("preflight end to end", ()),
}


def scenario_specs() -> tuple[ScenarioSpec, ...]:
    """Group F scenario registry."""
    return tuple(
        ScenarioSpec(scenario_id, "F", title, hypotheses, isolated_run(_F_RUNNERS[scenario_id]), _F_EVALUATORS[scenario_id])
        for scenario_id, (title, hypotheses) in sorted(_F_TITLES.items(), key=lambda item: int(item[0][1:]))
    )
