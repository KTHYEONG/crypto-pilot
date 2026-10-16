"""Group G runner-cycle end-to-end scenarios (G1..G4).

Same-day reruns of group G require the durable offset reservations under scratch.
Keep that reservation file across runs: deleting it can re-issue historical ids.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from src.live.errors import LiveTradingError
from src.live.order_journal import OrderJournal
from src.live.testnet_drill.guard import DrillRefused
from src.live.testnet_drill.report import Verdict
from src.live.testnet_drill.scenarios.base import (
    ScenarioEvidence,
    ScenarioSpec,
    ev,
    isolated_run,
    result_of,
)

_WEIGHTS_NAME = "deployed_target_weights.parquet"


def write_weights_artifact(session: Any, rows: Mapping[pd.Timestamp, Mapping[str, float]]) -> Path:
    """Write target weights and their decision closes with the production writer pair (sealed when a key is set)."""
    from src.live.deployed_weights import append_weight_row, decision_ohlcv_close_path

    path: Path = session.workspace / _WEIGHTS_NAME
    key = session.settings.artifact_key
    for day, weights in sorted(rows.items()):
        closes = {symbol: float(_mid(session, symbol)) for symbol in weights}
        append_weight_row(path, day, pd.Series(dict(weights), dtype=float), artifact_key=key)
        append_weight_row(decision_ohlcv_close_path(path), day, pd.Series(closes, dtype=float), artifact_key=key)
    return path


def _mid(session: Any, symbol: str) -> Decimal:
    quote = session.client.book_ticker(symbol)
    return (Decimal(str(quote["bidPrice"])) + Decimal(str(quote["askPrice"]))) / 2


def _probe_weight(session: Any) -> float:
    """Weight whose target notional is the probe notional at the current venue equity."""
    account = session.client.account()
    equity = Decimal(str(account["totalWalletBalance"])) + Decimal(str(account.get("totalUnrealizedProfit", "0")))
    if equity <= 0:
        raise DrillRefused("drill refused: non_positive_equity")
    return float(session.config.probe_notional_usdt / equity)


def _symbols(session: Any) -> tuple[str, str]:
    if len(session.config.symbols) < 2:
        raise DrillRefused("drill refused: g_requires_two_symbols")
    return session.config.symbols[0], session.config.symbols[1]


def _decision_days(count: int, *, seed: str = "") -> list[pd.Timestamp]:
    import fcntl
    import hashlib
    import json
    import os

    from src.live.testnet_drill.guard import drill_scratch_root

    root = drill_scratch_root()
    root.mkdir(parents=True, exist_ok=True)
    anchor = pd.Timestamp("2020-01-01", tz="UTC")
    slots = ((pd.Timestamp.now(tz="UTC").normalize() - anchor).days - 30) // count
    start = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % max(slots, 1)
    with (root / "g_decision_windows.jsonl").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        used = {day for line in handle for day in json.loads(line)["days"]}
        for offset in range(slots):
            slot = (start + offset) % slots
            days = [anchor + pd.Timedelta(days=slot * count + i) for i in range(count)]
            labels = [day.isoformat() for day in days]
            if used.isdisjoint(labels):
                handle.write(json.dumps({"run_id": seed, "days": labels}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                return days
    raise DrillRefused("drill refused: g_decision_window_exhausted")


def cycle_now(day: pd.Timestamp) -> pd.Timestamp:
    """Earliest causally valid order time for a decision day (the signal availability lag)."""
    from src.live.signal import _SIGNAL_LAG

    return day + _SIGNAL_LAG


def _cycle(session: Any, weights_path: Path, day: pd.Timestamp) -> tuple[Any, str | None]:
    from src.live.runner import run_shadow_cycle
    from src.live.testnet_drill.seams import production_bound

    with production_bound(session):
        try:
            report = run_shadow_cycle(session.settings, day, weights_path, now=cycle_now(day))
        except LiveTradingError as exc:
            return None, type(exc).__name__
    return report, None


def _fresh(session: Any) -> Any:
    return replace(session, journal=OrderJournal(session.journal.path))


def _artifacts_agree(session: Any) -> dict[str, bool]:
    from src.live.fills import load_fills

    fills = load_fills(Path(session.settings.fills_dir))
    journal = session.journal.fills_after(-1)
    per_symbol: dict[str, Decimal] = {}
    for f in journal:
        per_symbol[f.symbol] = per_symbol.get(f.symbol, Decimal(0)) + (f.quantity if f.side == "BUY" else -f.quantity)
    parquet: dict[str, Decimal] = {}
    for symbol, delta in zip(fills.get("symbol", []), fills.get("quantity_delta", []), strict=False):
        parquet[str(symbol)] = parquet.get(str(symbol), Decimal(0)) + Decimal(str(delta))
    tax_dir = Path(session.settings.tax_ledger_dir)
    tax_lines = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in tax_dir.glob("tax_ledger_*.jsonl"))
    return {
        "fills_parquet": {k: v for k, v in parquet.items() if v} == {k: v for k, v in per_symbol.items() if v},
        "execution_quality": not journal or any(Path(session.settings.execution_quality_dir).rglob("*")),
        # With collection disabled the runner audits tax_collection_disabled and writes no records by design.
        "tax_ledger": not journal or not session.settings.tax_collection_enabled or tax_lines > 0,
        "portfolio_state": any(Path(session.settings.portfolio_state_dir).rglob("*.parquet")),
    }


def _reconciled(session: Any) -> bool:
    from src.live.testnet_drill.reconcile import evaluate_reconciliation, reconcile_session

    evidence = reconcile_session(session, since_ms=int(session.opened_at * 1000), settle_wait_s=session.config.settle_wait_s)
    return all(item.passed for item in evaluate_reconciliation(evidence, session.config))


def _script(session: Any) -> tuple[Path, list[pd.Timestamp]]:
    first, second = _symbols(session)
    weight = _probe_weight(session)
    days = _decision_days(3, seed=getattr(session, "run_id", ""))
    rows = {
        days[0]: {first: weight, second: weight},
        days[1]: {first: -weight, second: weight},
        days[2]: {first: 0.0, second: 0.0},
    }
    return write_weights_artifact(session, rows), days


def _run_g1(session: Any, config: Any) -> ScenarioEvidence:
    from src.live.testnet_drill.session import _read_positions

    weights_path, days = _script(session)
    cycles = []
    for day in days:
        report, error = _cycle(session, weights_path, day)
        session = _fresh(session)
        cycles.append(
            {
                "decision_time": day.isoformat(),
                "status": getattr(report, "status", None),
                "reason": getattr(report, "reason", None),
                "error": error,
                "reconciled": error is None and _reconciled(session),
                "artifacts": _artifacts_agree(session),
            }
        )
        if error is not None or getattr(report, "status", None) != "COMPLETE":
            break
    positions = _read_positions(session.client)
    completed = sum(1 for c in cycles if c["error"] is None and c["status"] == "COMPLETE")
    return ev(
        "G1",
        cycles_completed=completed,
        r1_r6_pass=len(cycles) == 3 and all(c["reconciled"] for c in cycles),
        fills_agree=len(cycles) == 3 and all(all(c["artifacts"].values()) for c in cycles),
        final_flat=not any(positions.get(s, Decimal(0)) for s in _symbols(session)),
        cycles=cycles,
    )


def _posts(session: Any, start: int) -> int:
    return sum(1 for r in session.transport.records[start:] if r.method == "POST" and r.path == "/fapi/v1/order")


def _run_g2(session: Any, config: Any) -> ScenarioEvidence:
    weights_path, days = _script(session)
    _, error = _cycle(session, weights_path, days[0])
    if error is not None:
        raise DrillRefused(f"drill refused: g2_first_cycle_failed {error}")
    session = _fresh(session)
    fills_before = len(session.journal.fills_after(-1))
    start = len(session.transport.records)
    rerun, rerun_error = _cycle(session, weights_path, days[0])
    session = _fresh(session)
    keys = [(f.client_order_id, f.cumulative_executed_qty) for f in session.journal.fills_after(-1) if f.client_order_id]
    return ev(
        "G2",
        new_orders=_posts(session, start),
        decision_guard_honored=rerun_error is None and getattr(rerun, "reason", None) == "already_executed",
        no_duplicate_fills=len(session.journal.fills_after(-1)) == fills_before and len(keys) == len(set(keys)),
    )


def _run_g3(session: Any, config: Any) -> ScenarioEvidence:
    from src.live.testnet_drill.scenarios.restart_recovery import spawn
    from src.live.testnet_drill.session import _read_positions
    from src.live.testnet_drill.worker import KILL_EXIT_CODE

    weights_path, days = _script(session)
    first, _ = _cycle(session, weights_path, days[0])
    session = _fresh(session)
    run = spawn(
        session,
        {
            "action": "runner_cycle",
            "weights_path": str(weights_path),
            "decision_time": days[1].isoformat(),
            "kill_on_audit_event": {"event": "fill"},
        },
        [],
    )
    if run.exit_code != KILL_EXIT_CODE:
        raise DrillRefused(f"drill refused: kill_point_not_reached exit={run.exit_code}")
    session = _fresh(session)
    rerun, error = _cycle(session, weights_path, days[1])
    session = _fresh(session)
    again, again_error = _cycle(session, weights_path, days[1])
    target = _target_positions(session, weights_path, days[1])
    venue = _read_positions(session.client)
    converged_flag = error is None and getattr(rerun, "status", None) == "COMPLETE" and _converged(session, target, venue, rerun)
    silent = not _converged(session, target, venue, rerun)
    return ev(
        "G3",
        converged=converged_flag,
        silent_shortfall=bool(silent),
        no_double_execution=again_error is None and getattr(again, "reason", None) == "already_executed",
        first_status=getattr(first, "status", None),
        rerun_status=getattr(rerun, "status", None),
        target={k: str(v) for k, v in target.items()},
        venue={k: str(v) for k, v in venue.items()},
    )


def _target_positions(session: Any, weights_path: Path, day: pd.Timestamp) -> dict[str, Decimal]:
    """Quantities an uninterrupted run targets for ``day``, sized by the production sizing function."""
    from src.live.signal import latest_decision_ohlcv_close, latest_target_weights
    from src.live.sizing import target_quantities

    key = session.settings.artifact_key
    weights = latest_target_weights(weights_path, day, artifact_key=key)
    closes = {str(k): Decimal(str(v)) for k, v in latest_decision_ohlcv_close(weights_path, day, artifact_key=key).items()}
    account = session.client.account()
    equity = Decimal(str(account["totalWalletBalance"])) + Decimal(str(account.get("totalUnrealizedProfit", "0")))
    targets, _ = target_quantities(weights, closes, session.filters, equity, sizing_marks=closes)
    return {symbol: targets.get(symbol, Decimal(0)) for symbol in weights.index.astype(str)}


def _converged(session: Any, target: Mapping[str, Decimal], venue: Mapping[str, Decimal], rerun: Any) -> bool:
    """Venue equals target within one lot step, or the rerun left an explicit residual."""
    outcomes = getattr(rerun, "outcomes", ()) or ()

    def _explicit(symbol: str) -> bool:
        intents = getattr(rerun, "intents", ())
        for index, outcome in enumerate(outcomes):
            if getattr(outcome, "symbol", None) != symbol:
                continue
            if getattr(outcome, "status", None) == "OBSOLETE":
                continue
            if getattr(outcome, "block_reason", None) == "blocked_by_close_leg":
                return True
            if outcome.status == "RESIDUAL" and index < len(intents):
                intent = intents[index]
                total = outcome.filled_qty + outcome.unfilled_qty
                if (
                    intent.symbol == symbol
                    and total == intent.quantity
                    and outcome.filled_qty >= 0
                    and outcome.unfilled_qty > 0
                ):
                    return True
        return False

    return all(
        abs(venue.get(symbol, Decimal(0)) - qty) <= session.filters[symbol].step_size
        or _explicit(symbol)
        for symbol, qty in target.items()
    )


def _eval_g1(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = (
        int(p["cycles_completed"]) == 3
        and p["r1_r6_pass"] is True
        and p["fills_agree"] is True
        and p["final_flat"] is True
    )
    return result_of(evidence, "G1", Verdict.PASS if passed else Verdict.FAIL, (("three_cycle_script", passed),))


def _eval_g2(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = int(p["new_orders"]) == 0 and p["decision_guard_honored"] is True and p["no_duplicate_fills"] is True
    return result_of(evidence, "G2", Verdict.PASS if passed else Verdict.FAIL, (("idempotent_rerun", passed),))


def _eval_g3(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    silent = p.get("silent_shortfall", False) is True
    passed = p["converged"] is True and p["no_double_execution"] is True and not silent
    return result_of(
        evidence,
        "G3",
        Verdict.PASS if passed else Verdict.FAIL,
        (("restart_in_cycle", passed), ("no_silent_shortfall", not silent)),
    )


# IOC limit beyond the touch so a thin top-of-book cannot leave the seeding order partially filled.
_G4_CROSS_FRACTION = Decimal("0.002")


def _await_terminal(session: Any, symbol: str, client_order_id: str, config: Any) -> dict[str, Any]:
    """Poll the venue until the order leaves NEW/PARTIALLY_FILLED (the POST ack is always NEW) or the settle budget ends."""
    import time

    deadline = time.monotonic() + config.settle_wait_s
    while True:
        payload = dict(session.client.query_order(symbol, client_order_id))
        if str(payload.get("status", "")) not in ("NEW", "PARTIALLY_FILLED") or time.monotonic() >= deadline:
            return payload
        time.sleep(config.settle_poll_s)


def _run_g4(session: Any, config: Any) -> ScenarioEvidence:
    """Reuse a filled venue id from a fresh journal while the replacement rests."""
    import time

    from src.live.executor import FeeSchedule, _IntentRuntime, _sync_venue_order
    from src.live.order_identity import StaleOrderIdentity
    from src.live.recovery import recover_unresolved_orders
    from src.live.testnet_drill.execution import make_intent, submit_probe
    from src.live.testnet_drill.scenarios.lifecycle_runs import _limit

    symbol = _symbols(session)[0]
    intent = make_intent(session, symbol)
    resting = _limit(session, distance=True)
    order_id = resting["newClientOrderId"]
    quote = session.client.book_ticker(symbol)
    tick = session.filters[symbol].tick_size
    crossing = (Decimal(str(quote["askPrice"])) * (1 + _G4_CROSS_FRACTION) / tick).to_integral_value(
        rounding=ROUND_CEILING
    ) * tick
    # Seeded outside the session journal and flattened at once: a venue lookup of the reused id is ambiguous
    # by construction, so the session's own R1..R6 reconciliation must never reference this id.
    first = session.guarded_new_order({**resting, "timeInForce": "IOC", "price": str(crossing)})
    seeded = _await_terminal(session, symbol, order_id, config)
    if seeded.get("status") != "FILLED":
        raise DrillRefused("drill refused: g4_initial_order_not_filled")
    session.guarded_new_order({
        "symbol": symbol, "side": "SELL", "type": "MARKET", "quantity": str(seeded["executedQty"]), "reduceOnly": "true",
    })
    # The guard adds the learned clock margin to its tolerance and the old order's venue time sits on the
    # venue clock, so up to two margins separate "older than tolerance" on the wire from the local gap.
    tolerance = session.settings.order_identity_tolerance_ms + 2 * session.client.clock_offset_margin_ms
    if tolerance / 1000 + 1 >= config.scenario_timeout_s:
        raise DrillRefused("drill refused: g4_clock_wait_exceeds_budget")
    time.sleep(tolerance / 1000 + 1)
    journal_b = OrderJournal(session.workspace / "g4-journal-b.jsonl")
    replacement = submit_probe(replace(session, journal=journal_b), resting)
    if replacement.get("status") != "NEW":
        raise DrillRefused("drill refused: g4_replacement_did_not_rest")
    if replacement["orderId"] == first["orderId"]:
        raise DrillRefused("drill refused: g4_venue_did_not_reuse_id")
    rt = _IntentRuntime(
        intent=intent, filters=session.filters[symbol],
        fee_schedule=FeeSchedule(session.settings.maker_fee_bps, session.settings.taker_fee_bps),
        journal=journal_b, journal_enabled=True, active_id=order_id,
        active_price=Decimal(resting["price"]), active_post_qty=intent.quantity,
        order_identity_tolerance_ms=session.settings.order_identity_tolerance_ms,
    )
    stale_lookup = session.client.query_order(symbol, order_id).get("orderId") == first["orderId"]
    executor_client = _StaleLookupRecorder(session.client, first["orderId"])
    recovery_client = _StaleLookupRecorder(session.client, first["orderId"])
    now = pd.Timestamp.now(tz="UTC")
    executor_tripped = False
    recovery_tripped = False
    try:
        try:
            _sync_venue_order(executor_client, rt, now=now.timestamp(), audit=session.audit, touch=None)
        except StaleOrderIdentity:
            executor_tripped = True
        try:
            recover_unresolved_orders(
                recovery_client, journal_b, session.audit, now=now,
                lookback=pd.Timedelta(hours=72), taker_fee_bps=session.settings.taker_fee_bps,
                order_identity_tolerance_ms=session.settings.order_identity_tolerance_ms,
            )
        except StaleOrderIdentity:
            recovery_tripped = True
    finally:
        session.client.request("DELETE", "/fapi/v1/order", {
            "symbol": symbol, "orderId": replacement["orderId"],
        }, signed=True)
    fills = [f for f in journal_b.fills_after(-1) if f.client_order_id == order_id]
    tripped = executor_tripped and recovery_tripped
    return ev(
        "G4",
        guard_tripped=tripped,
        stale_lookup_observed=stale_lookup,
        executor_tripped=executor_tripped,
        recovery_tripped=recovery_tripped,
        stale_served_executor=executor_client.served,
        stale_served_recovery=recovery_client.served,
        no_phantom_fill=not fills and rt.filled_total == Decimal("0"),
        first_order_id=first["orderId"], replacement_order_id=replacement["orderId"],
        h12="REFUTED" if tripped else "CONFIRMED" if fills else "INCONCLUSIVE",
    )


def _eval_g4(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    phantom = p["no_phantom_fill"] is not True
    served_executor = int(p.get("stale_served_executor", 1 if p.get("stale_lookup_observed") else 0)) > 0
    served_recovery = int(p.get("stale_served_recovery", 0)) > 0
    if not phantom and not served_executor and not served_recovery:
        # Every lookup the guarded consumers made returned the current order: nothing needed guarding.
        return result_of(evidence, "G4", Verdict.OBSERVED, (("stale_lookup_not_served", True),), hypotheses=("H12",))
    executor_ok = not served_executor or p.get("executor_tripped") is True
    recovery_ok = not served_recovery or p.get("recovery_tripped") is True
    passed = executor_ok and recovery_ok and not phantom
    return result_of(
        evidence, "G4", Verdict.PASS if passed else Verdict.FAIL, (("id_reuse_guard", passed),),
        hypotheses=("H12",),
        hypothesis_outcome="REFUTED" if passed else "CONFIRMED" if phantom else "INCONCLUSIVE",
    )


class _StaleLookupRecorder:
    """Pass-through venue client that counts order lookups served from the superseded order."""

    def __init__(self, client: Any, superseded_order_id: Any) -> None:
        self._client = client
        self._superseded = superseded_order_id
        self.served = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def query_order(self, symbol: str, client_order_id: str) -> Any:
        payload = self._client.query_order(symbol, client_order_id)
        if isinstance(payload, dict) and payload.get("orderId") == self._superseded:
            self.served += 1
        return payload


_G_TITLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "G1": ("three cycle script", ()),
    "G2": ("idempotent rerun", ()),
    "G3": ("restart in cycle", ()),
    "G4": ("id reuse guard", ("H12",)),
}

def _swept(body: Callable[[Any, Any], ScenarioEvidence]) -> Callable[[Any, Any], ScenarioEvidence]:
    """Runner orders carry the runner's own namespace ids, which teardown leaves alone; sweep them via production."""

    def run(session: Any, config: Any) -> ScenarioEvidence:
        from src.live.order_cancel import cancel_orphan_orders

        try:
            return body(session, config)
        finally:
            cancel_orphan_orders(
                session.client,
                session.client_order_prefix,
                session.audit,
                journal=OrderJournal(session.journal.path),
                now=pd.Timestamp.now(tz="UTC"),
                taker_fee_bps=float(session.settings.taker_fee_bps),
            )

    return run


_G_RUNNERS: dict[str, Any] = {"G1": _swept(_run_g1), "G2": _swept(_run_g2), "G3": _swept(_run_g3), "G4": _run_g4}
_G_EVALUATORS: dict[str, Any] = {"G1": _eval_g1, "G2": _eval_g2, "G3": _eval_g3, "G4": _eval_g4}


def scenario_specs() -> tuple[ScenarioSpec, ...]:
    """Group G scenario registry."""
    return tuple(
        ScenarioSpec(scenario_id, "G", title, hypotheses, isolated_run(_G_RUNNERS[scenario_id]), _G_EVALUATORS[scenario_id])
        for scenario_id, (title, hypotheses) in sorted(_G_TITLES.items(), key=lambda item: int(item[0][1:]))
    )
