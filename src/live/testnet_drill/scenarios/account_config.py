"""Group A account-configuration scenarios (A1..A8)."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import ROUND_DOWN, Decimal
from typing import Any

import pandas as pd

from src.live.account import assert_venue_configuration, fetch_account_snapshot
from src.live.errors import BINANCE_ERROR_POLICY, RiskGateBreach, VenueError, resolve_error_action
from src.live.testnet_drill.execution import make_intent, submit_probe
from src.live.testnet_drill.report import Verdict
from src.live.testnet_drill.scenarios.base import (
    ScenarioEvidence,
    ScenarioSpec,
    ev,
    isolated_run,
    observation,
    result_of,
)
from src.live.testnet_drill.session import DirtySessionError, _flatten_positions, _read_positions


@contextmanager
def _restore_configuration(session: Any, path: str, params: dict[str, Any]) -> Iterator[None]:
    try:
        yield
    finally:
        try:
            if path == "/fapi/v1/marginType":
                _flatten_positions(session, _read_positions(session.client))
            session.client.request("POST", path, params, signed=True)
        except Exception as exc:
            session.audit.record("drill_dirty", path=path, error=type(exc).__name__)
            residuals: dict[str, str] = {}
            with contextlib.suppress(Exception):
                residuals = {symbol: str(qty) for symbol, qty in _read_positions(session.client).items()}
            raise DirtySessionError(f"DIRTY: restoration failed at {path}", residuals=residuals) from exc


def _run_with_slate(body: Any) -> Any:
    return isolated_run(body, with_slate=True)


def _run_a1(session: Any, config: Any, slate: Any) -> ScenarioEvidence:
    from src.live.rest import BinanceFuturesRestClient
    from src.live.testnet_drill.faults import redact

    testnet_ok = False
    mainnet_code: int | None = None
    try:
        session.client.request("GET", "/fapi/v2/account", signed=True)
        testnet_ok = True
    except VenueError as exc:
        return ev("A1", testnet_accepted=False, mainnet_code=None, redacted=redact(str(exc)))
    if session.settings.api_key is None or session.settings.api_secret is None:
        return ev("A1", error="mainnet read credentials unavailable")
    read_client = BinanceFuturesRestClient(
        session.settings.order_base_url,
        session.settings.api_key,
        session.settings.api_secret,
        session.settings.mode,
        session.audit,
        recv_window_ms=config.recv_window_ms,
        session=session.transport,
    )
    try:
        read_client.sync_server_time()
        read_client.account()
    except VenueError as exc:
        mainnet_code = exc.code
    evidence_text = redact(f"testnet_ok={testnet_ok} code={mainnet_code}")
    return ev("A1", testnet_accepted=testnet_ok, mainnet_code=mainnet_code, redacted=evidence_text)


def _eval_a1(evidence: ScenarioEvidence) -> Any:
    testnet_ok = bool(evidence.payload.get("testnet_accepted", False))
    code = evidence.payload.get("mainnet_code")
    action = resolve_error_action(int(code)).value if code is not None else "fail_closed"
    passed = testnet_ok and code == -2015 and "key" not in str(evidence.payload.get("redacted", "")).lower()
    obs = (observation(int(code), action, path="/fapi/v2/account"),) if code is not None else ()
    return result_of(
        evidence,
        "A1",
        Verdict.PASS if passed else Verdict.FAIL,
        (("key_routing", passed),),
        observations=obs,
        hypotheses=("H1",),
        hypothesis_outcome="CONFIRMED" if passed else "INCONCLUSIVE",
        proposed_action="fail_closed" if code is not None and code not in BINANCE_ERROR_POLICY else None,
    )


def _run_a2(session: Any, config: Any, slate: Any) -> ScenarioEvidence:
    intent = make_intent(session, config.symbols[0])
    original = session.client._time_offset_ms
    offset = config.recv_window_ms // 2
    start = len(session.transport.records)
    try:
        session.client._time_offset_ms = original + offset
        account = session.client.account()
        response = submit_probe(
            session,
            {
                "symbol": intent.symbol,
                "side": "BUY",
                "type": "LIMIT",
                "timeInForce": "GTX",
                "quantity": str(intent.quantity),
                "price": str(intent.decision_price),
            },
            before_send=lambda: setattr(session.client, "_time_offset_ms", original + offset),
        )
    finally:
        session.client._time_offset_ms = original
    records = session.transport.records[start:]
    return ev(
        "A2",
        accepted=isinstance(account, dict),
        order_accepted=isinstance(response, dict) and "orderId" in response,
        resynced=any(row.path == "/fapi/v1/time" for row in records),
        offset_ms=offset,
    )


def _eval_a2(evidence: ScenarioEvidence) -> Any:
    passed = (
        evidence.payload["accepted"] is True
        and evidence.payload["order_accepted"] is True
        and evidence.payload["resynced"] is False
    )
    verdict = Verdict.PASS if passed else Verdict.FAIL
    return result_of(evidence, "A2", verdict, (("skew_in_window", passed),), hypotheses=())


def _run_a3(session: Any, config: Any, slate: Any) -> ScenarioEvidence:
    import json

    from src.live.rest import _MUTATION_MAX_SENDS

    intent = make_intent(session, config.symbols[0])
    original = session.client._time_offset_ms
    phases = []
    try:
        for sign in (-1, 1):
            start = len(session.transport.records)
            skew = sign * (config.recv_window_ms + 1000)

            def set_offset(skew: int = skew) -> None:
                session.client._time_offset_ms = original + skew

            response = submit_probe(
                session,
                {
                    "symbol": intent.symbol,
                    "side": "BUY",
                    "type": "LIMIT",
                    "timeInForce": "GTX",
                    "quantity": str(intent.quantity),
                    "price": str(intent.decision_price),
                },
                before_send=set_offset,
            )
            records = session.transport.records[start:]
            posts = [row for row in records if row.method == "POST" and row.path == "/fapi/v1/order"]
            first = json.loads(posts[0].response_body) if posts else {}
            phases.append(
                {
                    "sends": len(posts),
                    "resyncs": sum(row.path == "/fapi/v1/time" for row in records),
                    "first_code": first.get("code"),
                    "accepted": isinstance(response, dict) and "orderId" in response,
                }
            )
    finally:
        session.client._time_offset_ms = original
    return ev(
        "A3",
        sends=max(phase["sends"] for phase in phases),
        resyncs=max(phase["resyncs"] for phase in phases),
        max_sends=_MUTATION_MAX_SENDS,
        first_code=-1021 if all(p["first_code"] == -1021 for p in phases) else None,
        accepted=all(p["accepted"] for p in phases),
        phases=phases,
    )


def _eval_a3(evidence: ScenarioEvidence) -> Any:
    sends = int(evidence.payload.get("sends", 0))
    resyncs = int(evidence.payload.get("resyncs", 0))
    max_sends = int(evidence.payload.get("max_sends", 2))
    passed = (
        sends == max_sends == 2
        and resyncs == 1
        and evidence.payload["accepted"] is True
        and evidence.payload["first_code"] == -1021
    )
    return result_of(
        evidence, "A3", Verdict.PASS if passed else Verdict.FAIL, (("clock_resync_bounded", passed),), hypotheses=()
    )


def _config_guard_run(scenario_id: str, hypotheses: tuple[str, ...]) -> Any:
    def run(session: Any, config: Any) -> ScenarioEvidence:
        @_run_with_slate
        def _inner(session: Any, config: Any, slate: Any) -> ScenarioEvidence:
            path, key = (
                ("/fapi/v1/positionSide/dual", "dualSidePosition")
                if scenario_id == "A4"
                else ("/fapi/v1/multiAssetsMargin", "multiAssetsMargin")
            )
            original = session.client.request("GET", path, signed=True)[key]
            if _read_positions(session.client) or session.client.open_orders():
                raise RiskGateBreach("mode probe requires a flat account without orders")
            with _restore_configuration(session, path, {key: str(original).lower()}):
                session.client.request("POST", path, {key: "true"}, signed=True)
                snapshot = fetch_account_snapshot(session.client, now=pd.Timestamp.now(tz="UTC"))
                try:
                    assert_venue_configuration(snapshot)
                    gate = "passed"
                except RiskGateBreach:
                    gate = "RiskGateBreach"
            return ev(scenario_id, gate=gate, raw_code=None, restored=True)

        return _inner(session, config)  # type: ignore[no-any-return]

    return run


def _eval_config_guard(scenario_id: str) -> Any:
    def evaluate(evidence: ScenarioEvidence) -> Any:
        gate = str(evidence.payload.get("gate", ""))
        passed = gate == "RiskGateBreach" and evidence.payload["restored"] is True
        code = evidence.payload.get("raw_code")
        obs: tuple[Any, ...] = ()
        if code is not None:
            action = resolve_error_action(int(code)).value
            obs = (observation(int(code), action),)
            return result_of(
                evidence,
                scenario_id,
                Verdict.OBSERVED,
                (("config_gate", passed),),
                observations=obs,
                hypotheses=("H1",),
                hypothesis_outcome="INCONCLUSIVE",
                proposed_action=action if int(code) not in BINANCE_ERROR_POLICY else None,
                proposal_rationale=f"observed {code} at {scenario_id}",
            )
        return result_of(
            evidence,
            scenario_id,
            Verdict.PASS if passed else Verdict.FAIL,
            (("config_gate", passed),),
            observations=obs,
            hypotheses=(),
        )

    return evaluate


def _run_a6(session: Any, config: Any, slate: Any) -> ScenarioEvidence:
    from src.live.account import BLOCKED_MARGIN_TYPE_CHANGE_CODES, ensure_venue_leverage, parse_leverage_brackets

    symbol = config.symbols[0]
    risk = session.client.position_risk()
    original = next(row["marginType"] for row in risk if row["symbol"] == symbol)
    restore = {"symbol": symbol, "marginType": "ISOLATED" if str(original).lower() == "isolated" else "CROSSED"}
    code = None
    gate = "passed"
    with _restore_configuration(session, "/fapi/v1/marginType", restore):
        session.client.request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": "ISOLATED"}, signed=True)
        quote = session.client.book_ticker(symbol)
        price = max(Decimal(quote["bidPrice"]), Decimal(quote["askPrice"]))
        step = session.filters[symbol].step_size
        qty = (config.probe_notional_usdt / price / step).to_integral_value(rounding=ROUND_DOWN) * step
        submit_probe(session, {"symbol": symbol, "side": "BUY", "type": "MARKET", "quantity": str(qty)})
        brackets = parse_leverage_brackets(session.client.request("GET", "/fapi/v1/leverageBracket", signed=True))
        try:
            ensure_venue_leverage(
                session.client,
                (symbol,),
                brackets,
                max_gross_leverage=session.settings.max_gross_leverage,
                buffer_fraction=session.settings.leverage_buffer_fraction,
                audit=session.audit,
            )
        except RiskGateBreach as exc:
            gate = type(exc).__name__
            if isinstance(exc.__cause__, VenueError):
                code = exc.__cause__.code
        except VenueError as exc:
            code = exc.code
    return ev("A6", code=code, blocked=sorted(BLOCKED_MARGIN_TYPE_CHANGE_CODES), gate=gate, restored=True)


def _eval_a6(evidence: ScenarioEvidence) -> Any:
    code = evidence.payload.get("code")
    blocked = set(evidence.payload.get("blocked", ()))
    if code is None or evidence.payload["restored"] is not True:
        return result_of(evidence, "A6", Verdict.FAIL, (("margin_block", False),), hypotheses=())
    action = resolve_error_action(int(code)).value
    if int(code) in blocked and evidence.payload["gate"] == "RiskGateBreach" and evidence.payload["restored"] is True:
        return result_of(
            evidence,
            "A6",
            Verdict.PASS,
            (("margin_block", True),),
            observations=(observation(int(code), action, path="/fapi/v1/marginType"),),
            hypotheses=(),
        )
    return result_of(
        evidence,
        "A6",
        Verdict.OBSERVED,
        (("margin_block_observed", True),),
        observations=(observation(int(code), action, path="/fapi/v1/marginType"),),
        hypotheses=("H1",),
        proposed_action=action,
        proposal_rationale=f"unlisted margin code {code}",
    )


def _run_a7(session: Any, config: Any, slate: Any) -> ScenarioEvidence:
    symbol = config.symbols[0]
    original = next(row["leverage"] for row in session.client.position_risk() if row["symbol"] == symbol)
    bracket = session.client.request("GET", "/fapi/v1/leverageBracket", {"symbol": symbol}, signed=True)
    requested = max(int(row["initialLeverage"]) for row in bracket[0]["brackets"]) + 1
    with _restore_configuration(session, "/fapi/v1/leverage", {"symbol": symbol, "leverage": original}):
        try:
            session.client.request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": requested}, signed=True)
            code = None
        except VenueError as exc:
            code = exc.code
    return ev("A7", code=code, action=resolve_error_action(code).value, silent_success=code is None, restored=True)


def _eval_a7(evidence: ScenarioEvidence) -> Any:
    if evidence.payload["silent_success"] is not False or evidence.payload["restored"] is not True:
        return result_of(evidence, "A7", Verdict.FAIL, (("no_silent_success", False),), hypotheses=())
    code = evidence.payload.get("code")
    action = str(evidence.payload.get("action", "fail_closed"))
    obs = (observation(int(code), action, path="/fapi/v1/leverage"),) if code is not None else ()
    return result_of(
        evidence,
        "A7",
        Verdict.OBSERVED if code not in BINANCE_ERROR_POLICY else Verdict.PASS,
        (("leverage_guard", True),),
        observations=obs,
        hypotheses=("H1",),
        proposed_action=action if code not in BINANCE_ERROR_POLICY else None,
        proposal_rationale=f"bracket cap code {code}",
    )


def _run_a8(session: Any, config: Any, slate: Any) -> ScenarioEvidence:
    payload = session.client.request("GET", "/fapi/v2/account", signed=True)
    keys = sorted(payload.keys()) if isinstance(payload, dict) else []
    total_maint = payload.get("totalMaintMargin") if isinstance(payload, dict) else None
    total_initial = payload.get("totalInitialMargin") if isinstance(payload, dict) else None
    snapshot = fetch_account_snapshot(session.client, now=pd.Timestamp.now(tz="UTC"))
    return ev(
        "A8",
        keys=keys,
        total_maint=total_maint,
        total_initial=total_initial,
        snapshot_total_maint=snapshot.total_maint_margin,
    )


def _eval_a8(evidence: ScenarioEvidence) -> Any:
    keys = list(evidence.payload.get("keys", ()))
    passed = "totalMaintMargin" in keys and "totalInitialMargin" in keys
    confirmed = (
        passed
        and evidence.payload.get("total_maint") is not None
        and str(evidence.payload.get("total_maint")) != str(evidence.payload.get("total_initial"))
        and Decimal(str(evidence.payload["snapshot_total_maint"])) == Decimal(str(evidence.payload["total_initial"]))
    )
    return result_of(
        evidence,
        "A8",
        Verdict.PASS if passed else Verdict.FAIL,
        (("schema_present", passed),),
        hypotheses=("H5",),
        hypothesis_outcome="CONFIRMED" if confirmed else "INCONCLUSIVE",
    )


def scenario_specs() -> tuple[ScenarioSpec, ...]:
    """Group A scenario registry."""
    return (
        ScenarioSpec("A1", "A", "key routing", ("H1",), _run_with_slate(_run_a1), _eval_a1),
        ScenarioSpec("A2", "A", "clock skew in window", (), _run_with_slate(_run_a2), _eval_a2),
        ScenarioSpec("A3", "A", "clock skew out of window", (), _run_with_slate(_run_a3), _eval_a3),
        ScenarioSpec("A4", "A", "dual side mode", (), _config_guard_run("A4", ()), _eval_config_guard("A4")),
        ScenarioSpec("A5", "A", "multi assets mode", (), _config_guard_run("A5", ()), _eval_config_guard("A5")),
        ScenarioSpec("A6", "A", "margin type blocked", (), _run_with_slate(_run_a6), _eval_a6),
        ScenarioSpec("A7", "A", "leverage over bracket", (), _run_with_slate(_run_a7), _eval_a7),
        ScenarioSpec("A8", "A", "account payload schema", ("H5",), _run_with_slate(_run_a8), _eval_a8),
    )
