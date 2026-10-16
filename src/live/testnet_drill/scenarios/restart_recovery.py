"""Group E kill/restart recovery scenarios (E1..E10).

Each scenario spawns ``worker`` with a kill rule, checks the child died as configured, snapshots venue truth,
runs the production restart sequence and records evidence from venue queries and the reopened journal.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.live.errors import VenueError
from src.live.order_journal import OrderJournal
from src.live.testnet_drill.config import config_to_json
from src.live.testnet_drill.faults import FaultAction
from src.live.testnet_drill.guard import DrillRefused
from src.live.testnet_drill.report import Verdict
from src.live.testnet_drill.scenarios.base import (
    ScenarioEvidence,
    ScenarioSpec,
    ev,
    isolated_run,
    result_of,
)
from src.live.testnet_drill.scenarios.wire import ORDER_PATH
from src.live.testnet_drill.worker import KILL_EXIT_CODE

RESTART_ANCHOR = "src.live.runner:_reconcile_pre_trade"
_TERMINAL = ("CANCELED", "FILLED", "EXPIRED", "REJECTED")


def restart_recovery(session: Any, *, decision_time: pd.Timestamp | None = None) -> Any:
    """Run the production pre-trade restart sequence (sweep, recover, settle, commit) through the runner entry.

    Raises whatever the runner raises; a corrupt ledger is never replaced by an empty one.
    """
    from src.live.account import fetch_account_snapshot
    from src.live.ledger import load_ledger
    from src.live.runner import _reconcile_pre_trade  # anchor: RESTART_ANCHOR

    now = pd.Timestamp.now(tz="UTC")
    ledger_path = Path(session.settings.ledger_path)
    return _reconcile_pre_trade(
        session.settings,
        order_client=session.client,
        journal=session.journal,
        audit=session.audit,
        ledger_path=ledger_path,
        ledger_state=load_ledger(ledger_path),
        snapshot=fetch_account_snapshot(session.client, now=now),
        exchange_info=session.client.exchange_info(),
        run_id=session.run_id,
        decision_time=decision_time if decision_time is not None else now,
        now=now,
    )


@dataclass(frozen=True, slots=True)
class WorkerRun:
    """Exit status and milestone lines of one child run."""

    exit_code: int
    milestones: tuple[dict[str, Any], ...]


def launch_worker(session_dir: Path, run_id: str, script_path: Path, rules_path: Path, timeout_s: float) -> WorkerRun:
    """Spawn the drill worker in a child process; the parent never kills itself."""
    argv = [
        sys.executable,
        "-m",
        "src.live.testnet_drill.worker",
        "--session-dir",
        str(session_dir),
        "--run-id",
        run_id,
        "--script",
        str(script_path),
        "--rules",
        str(rules_path),
    ]
    repo_root = Path(__file__).resolve().parents[4]
    proc = subprocess.run(  # noqa: S603 - fixed interpreter and module, arguments are workspace paths
        argv, capture_output=True, text=True, timeout=timeout_s, check=False, cwd=repo_root
    )
    lines = []
    for line in proc.stdout.splitlines():
        try:
            lines.append(json.loads(line))
        except ValueError:
            continue
    return WorkerRun(proc.returncode, tuple(lines))


def spawn(session: Any, script: dict[str, Any], rules: list[dict[str, Any]]) -> WorkerRun:
    """Write script/rules inside the workspace and run one child against the shared journal."""
    folder = session.workspace / "worker"
    folder.mkdir(parents=True, exist_ok=True)
    index = len(list(folder.glob("*_script.json")))
    script_path, rules_path = folder / f"{index:02d}_script.json", folder / f"{index:02d}_rules.json"
    script_path.write_text(json.dumps({**script, "config": config_to_json(session.config)}), encoding="utf-8")
    rules_path.write_text(json.dumps(rules), encoding="utf-8")
    return launch_worker(session.workspace, session.run_id, script_path, rules_path, session.config.restart_kill_timeout_s)


def _reopen(session: Any) -> Any:
    return replace(session, journal=OrderJournal(session.journal.path))


def _rule(method: str, action: FaultAction, **extra: Any) -> dict[str, Any]:
    return {"method": method, "path": ORDER_PATH, "action": action.value, **extra}


def _resting(session: Any) -> dict[str, Any]:
    from src.live.testnet_drill.scenarios.lifecycle_runs import _limit

    return _limit(session, distance=True)


def _taker(session: Any) -> dict[str, Any]:
    from src.live.testnet_drill.execution import make_intent

    symbol = session.config.symbols[0]
    intent = make_intent(session, symbol)
    ask = str(session.client.book_ticker(symbol)["askPrice"])
    return {
        "symbol": symbol,
        "side": "BUY",
        "type": "LIMIT",
        "timeInForce": "IOC",
        "quantity": str(intent.quantity),
        "price": ask,
        "newClientOrderId": session.next_client_order_id(symbol),
    }


def _own_open(session: Any) -> list[dict[str, Any]]:
    return [
        row
        for row in session.client.open_orders()
        if str(row.get("clientOrderId", "")).startswith(session.client_order_prefix + "-")
    ]


def _venue_truth(session: Any) -> dict[str, Any]:
    from src.live.testnet_drill.session import _read_positions

    return {
        "positions": {k: str(v) for k, v in _read_positions(session.client).items()},
        "open_orders": [str(row.get("clientOrderId")) for row in _own_open(session)],
    }


def _query(session: Any, symbol: str, order_id: str) -> tuple[dict[str, Any] | None, int | None]:
    try:
        return session.client.query_order(symbol, order_id), None
    except VenueError as exc:
        return None, exc.code


def _fills(session: Any, order_id: str, kind: str | None = None) -> list[Any]:
    return [
        f
        for f in session.journal.fills_after(-1)
        if f.client_order_id == order_id and (kind is None or f.kind == kind)
    ]


def _unique_fill_keys(session: Any) -> bool:
    keys = [(f.client_order_id, f.cumulative_executed_qty) for f in session.journal.fills_after(-1) if f.client_order_id]
    return len(keys) == len(set(keys))


def _killed_and_restarted(
    session: Any, script: dict[str, Any], rules: list[dict[str, Any]]
) -> tuple[WorkerRun, Any, dict[str, Any]]:
    run = spawn(session, script, rules)
    if run.exit_code != KILL_EXIT_CODE:
        raise DrillRefused(f"drill refused: kill_point_not_reached exit={run.exit_code}")
    fresh = _reopen(session)
    before = _venue_truth(fresh)
    restart_recovery(fresh)
    return run, _reopen(fresh), before


def _common(run: WorkerRun, before: dict[str, Any]) -> dict[str, Any]:
    return {"exit_code": run.exit_code, "pre_restart_venue": before, "milestones": list(run.milestones)}


def _run_e1(session: Any, config: Any) -> ScenarioEvidence:
    params = _resting(session)
    run, fresh, before = _killed_and_restarted(
        session, {"action": "probe", "params": params}, [_rule("POST", FaultAction.KILL_BEFORE_DELIVERY)]
    )
    order, code = _query(fresh, params["symbol"], params["newClientOrderId"])
    terminal = fresh.journal.terminal(params["newClientOrderId"])
    from src.live.testnet_drill.session import _read_positions

    exposure = len(_own_open(fresh)) + sum(1 for qty in _read_positions(fresh.client).values() if qty != 0)
    return ev(
        "E1",
        **_common(run, before),
        query_code=code if order is None else None,
        terminal_status=terminal.status if terminal is not None else None,
        venue_exposure=exposure,
        orphan_count=len(_own_open(fresh)),
    )


def _run_e2(session: Any, config: Any) -> ScenarioEvidence:
    params = _resting(session)
    run, fresh, before = _killed_and_restarted(
        session, {"action": "probe", "params": params}, [_rule("POST", FaultAction.KILL_AFTER_DELIVERY)]
    )
    order, _ = _query(fresh, params["symbol"], params["newClientOrderId"])
    return ev(
        "E2",
        **_common(run, before),
        settled_terminal=order is not None
        and order.get("status") in _TERMINAL
        and fresh.journal.terminal(params["newClientOrderId"]) is not None,
        no_fills=order is not None
        and Decimal(str(order.get("executedQty", "0"))) == 0
        and not _fills(fresh, params["newClientOrderId"]),
        no_orphan=not _own_open(fresh),
    )


def _run_e3(session: Any, config: Any) -> ScenarioEvidence:
    params = _taker(session)
    run, fresh, before = _killed_and_restarted(
        session, {"action": "probe", "params": params}, [_rule("POST", FaultAction.KILL_AFTER_DELIVERY)]
    )
    order_id = params["newClientOrderId"]
    order, _ = _query(fresh, params["symbol"], order_id)
    recovered = _fills(fresh, order_id, "recovered")
    taker = Decimal(str(fresh.settings.taker_fee_bps))
    return ev(
        "E3",
        **_common(run, before),
        recovered_qty=str(sum((f.quantity for f in recovered), Decimal(0))),
        venue_qty=str(order.get("executedQty", "0")) if order is not None else "missing",
        fee_taker=bool(recovered) and all(Decimal(str(f.fee_bps)) == taker for f in recovered),
        single_booking=not _fills(fresh, order_id, "execution") and _unique_fill_keys(fresh),
    )


def _run_e4(session: Any, config: Any) -> ScenarioEvidence:
    params = _taker(session)
    order_id = params["newClientOrderId"]
    rule = _rule(
        "GET", FaultAction.KILL_ON_RESPONSE, params=[["origClientOrderId", order_id]], occurrence=1
    )
    run, fresh, before = _killed_and_restarted(session, {"action": "settle", "params": params}, [rule])
    order, _ = _query(fresh, params["symbol"], order_id)
    executed = Decimal(str(order.get("executedQty", "0"))) if order is not None else Decimal(-1)
    booked = _fills(fresh, order_id)
    kinds = sorted({f.kind for f in booked})
    return ev(
        "E4",
        **_common(run, before),
        recovered_once=executed > 0 and sum((f.quantity for f in booked), Decimal(0)) == executed,
        no_double_count=len(kinds) <= 1 and _unique_fill_keys(fresh),
        kinds="+".join(kinds),
    )


def _ledger(session: Any) -> Any:
    from src.live.ledger import load_ledger

    return load_ledger(Path(session.settings.ledger_path))


def _fold(session: Any, symbol: str) -> Decimal:
    return sum(
        (
            f.quantity if f.side == "BUY" else -f.quantity
            for f in session.journal.fills_after(-1)
            if f.symbol == symbol
        ),
        Decimal(0),
    )


def _run_e5(session: Any, config: Any) -> ScenarioEvidence:
    from src.live.testnet_drill.session import _read_positions

    symbol = session.config.symbols[0]
    run = spawn(session, {"action": "cycle", "kill_on_audit_event": {"event": "fill"}}, [])
    if run.exit_code != KILL_EXIT_CODE:
        raise DrillRefused(f"drill refused: kill_point_not_reached exit={run.exit_code}")
    fresh = _reopen(session)
    ahead = fresh.journal.last_fill_seq() > _ledger(fresh).journal_applied_fill_seq
    before = _venue_truth(fresh)
    restart_recovery(fresh)
    fresh = _reopen(fresh)
    first = _ledger(fresh)
    restart_recovery(fresh)
    second = _ledger(_reopen(fresh))
    venue = _read_positions(fresh.client).get(symbol, Decimal(0))
    return ev(
        "E5",
        **_common(run, before),
        journal_ahead_before=ahead,
        ledger_equal_after=first.journal_applied_fill_seq == fresh.journal.last_fill_seq()
        and first.positions.get(symbol, Decimal(0)) == _fold(fresh, symbol) == venue,
        single_commit=second.positions == first.positions
        and second.journal_applied_fill_seq == first.journal_applied_fill_seq,
    )


def _run_e6(session: Any, config: Any) -> ScenarioEvidence:
    params = _resting(session)
    run, fresh, before = _killed_and_restarted(
        session, {"action": "cancel", "params": params}, [_rule("DELETE", FaultAction.KILL_AFTER_DELIVERY)]
    )
    order_id = params["newClientOrderId"]
    order, _ = _query(fresh, params["symbol"], order_id)
    return ev(
        "E6",
        **_common(run, before),
        confirmed_or_recancelled=order is not None
        and order.get("status") in _TERMINAL
        and fresh.journal.terminal(order_id) is not None,
        fill_booked=order is not None
        and fresh.journal.observed_qty(order_id) == Decimal(str(order.get("executedQty", "0"))),
    )


def _run_e7(session: Any, config: Any) -> ScenarioEvidence:
    from src.live.testnet_drill.execution import AuditTap, make_intent, run_cycle
    from src.live.testnet_drill.session import _read_positions

    symbol = session.config.symbols[0]
    script = {"action": "cycle", "kill_on_audit_event": {"event": "order_cancelled", "fields": {"reason": "repeg"}}}
    run, fresh, before = _killed_and_restarted(session, script, [])
    venue = _read_positions(fresh.client).get(symbol, Decimal(0))
    no_live = not _own_open(fresh)
    intent = make_intent(fresh, symbol)
    with AuditTap(fresh) as tap:
        run_cycle(fresh, [intent])
    return ev(
        "E7",
        **_common(run, before),
        no_live_order=no_live,
        no_double_exposure=venue == _fold(fresh, symbol) and abs(venue) <= intent.quantity,
        next_cycle_posts=bool(tap.named("order_posted")),
    )


def _run_e8(session: Any, config: Any) -> ScenarioEvidence:
    params = _resting(session)
    order_id = params["newClientOrderId"]
    first = spawn(session, {"action": "probe", "params": params}, [_rule("POST", FaultAction.KILL_AFTER_DELIVERY)])
    second = spawn(session, {"action": "recovery"}, [_rule("DELETE", FaultAction.KILL_BEFORE_DELIVERY)])
    if first.exit_code != KILL_EXIT_CODE or second.exit_code != KILL_EXIT_CODE:
        raise DrillRefused(f"drill refused: kill_point_not_reached exit={first.exit_code},{second.exit_code}")
    fresh = _reopen(session)
    before = _venue_truth(fresh)
    restart_recovery(fresh)
    fresh = _reopen(fresh)
    order, _ = _query(fresh, params["symbol"], order_id)
    executed = Decimal(str(order.get("executedQty", "0"))) if order is not None else Decimal(0)
    terminal_lines = sum(
        1
        for line in fresh.journal.path.read_text(encoding="utf-8").splitlines()
        if '"terminal"' in line and order_id in line
    )
    return ev(
        "E8",
        exit_codes=[first.exit_code, second.exit_code],
        pre_restart_venue=before,
        recovery_passes=2,
        recovered_fill_count=len(_fills(fresh, order_id)),
        single_pass_fill_count=1 if executed > 0 else 0,
        idempotent=terminal_lines == 1 and _unique_fill_keys(fresh) and not _own_open(fresh),
    )


def _audit_events(session: Any, name: str) -> list[dict[str, Any]]:
    rows = []
    for line in session.audit.path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("event") == name:
            rows.append(row)
    return rows


def _run_e9(session: Any, config: Any) -> ScenarioEvidence:
    run = spawn(session, {"action": "cycle", "sigterm_on_audit_event": "order_posted"}, [])
    finished = next((m for m in run.milestones if m.get("event") == "finished"), {})
    if run.exit_code != 0 or finished.get("escaped") != "ExecutionInterrupted":
        raise DrillRefused(f"drill refused: shutdown_not_observed exit={run.exit_code}")
    interrupted = _audit_events(session, "execution_interrupted")
    unsettled = int(interrupted[-1].get("unsettled", -1)) if interrupted else -1
    fresh = _reopen(session)
    before = _venue_truth(fresh)
    restart_recovery(fresh)
    fresh = _reopen(fresh)
    open_submits = [s for s in fresh.journal.submits() if fresh.journal.terminal(s.client_order_id) is None]
    return ev(
        "E9",
        **_common(run, before),
        unsettled=unsettled,
        unsettled_within_budget=0 <= unsettled <= 1,
        settled_after_restart=not _own_open(fresh) and not open_submits,
        decision_executed=_ledger(fresh).last_executed_decision_time is not None,
    )


def _run_e10(session: Any, config: Any) -> ScenarioEvidence:
    params = _resting(session)
    run = spawn(session, {"action": "probe", "params": params}, [_rule("POST", FaultAction.KILL_AFTER_DELIVERY)])
    if run.exit_code != KILL_EXIT_CODE:
        raise DrillRefused(f"drill refused: kill_point_not_reached exit={run.exit_code}")
    path = session.journal.path
    intact = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b'{"event": "fill", "client_order_id": "' + params["newClientOrderId"].encode()[:8])
    fresh = _reopen(session)
    before = _venue_truth(fresh)
    try:
        restart_recovery(fresh)
        fresh = _reopen(fresh)
        tail_ok = path.read_bytes().startswith(intact) and any(
            s.client_order_id == params["newClientOrderId"] for s in fresh.journal.submits()
        )
    except DataIntegrityError:
        tail_ok = False
    lines = intact.splitlines(keepends=True)
    corrupt = session.workspace / "journal_mid_corruption.jsonl"
    corrupt.write_bytes(b"".join([*lines[:1], b"{not json}\n", *lines[1:]]))
    try:
        OrderJournal(corrupt).submits()
        mid_closed = False
    except DataIntegrityError:
        mid_closed = True
    return ev(
        "E10",
        **_common(run, before),
        tail_corrupt_at="tail",
        truncated_tail_ok=tail_ok,
        mid_file_fails_closed=mid_closed and len(lines) >= 2,
    )


def _eval_e1(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = (
        p["query_code"] == -2013
        and p["terminal_status"] in ("NOT_FOUND", "NOT_PLACED")
        and int(p["venue_exposure"]) == 0
        and int(p["orphan_count"]) == 0
    )
    return result_of(evidence, "E1", Verdict.PASS if passed else Verdict.FAIL, (("not_found_terminal", passed),))


def _eval_e2(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = p["settled_terminal"] is True and p["no_fills"] is True and p["no_orphan"] is True
    return result_of(evidence, "E2", Verdict.PASS if passed else Verdict.FAIL, (("resting_recovered", passed),), hypotheses=("H6",))


def _eval_e3(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = (
        Decimal(str(p["recovered_qty"])) == Decimal(str(p["venue_qty"]))
        and Decimal(str(p["venue_qty"])) > 0
        and p["fee_taker"] is True
        and p["single_booking"] is True
    )
    return result_of(evidence, "E3", Verdict.PASS if passed else Verdict.FAIL, (("ioc_fill_recovered", passed),), hypotheses=("H6",))


def _eval_e4(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    if "execution" in str(p.get("kinds", "")) and "recovered" in str(p.get("kinds", "")):
        return result_of(evidence, "E4", Verdict.FAIL, (("single_booking", False),))
    pre_restart = p.get("pre_restart_venue") or {}
    if not pre_restart.get("positions") and not str(p.get("kinds", "")) and p["no_double_count"] is True:
        # The IOC expired unfilled before the kill, so there was no fill to recover.
        return result_of(evidence, "E4", Verdict.OBSERVED, (("ioc_unfilled_nothing_to_recover", True),))
    passed = p["recovered_once"] is True and p["no_double_count"] is True
    return result_of(evidence, "E4", Verdict.PASS if passed else Verdict.FAIL, (("single_booking", passed),))


def _eval_e5(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = p["journal_ahead_before"] is True and p["ledger_equal_after"] is True and p["single_commit"] is True
    return result_of(evidence, "E5", Verdict.PASS if passed else Verdict.FAIL, (("watermark_commit", passed),))


def _eval_e6(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = p["confirmed_or_recancelled"] is True and p["fill_booked"] is True
    return result_of(evidence, "E6", Verdict.PASS if passed else Verdict.FAIL, (("cancel_recovered", passed),))


def _eval_e7(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = p["no_live_order"] is True and p["no_double_exposure"] is True and p["next_cycle_posts"] is True
    return result_of(evidence, "E7", Verdict.PASS if passed else Verdict.FAIL, (("repeg_recovered", passed),))


def _eval_e8(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = (
        int(p["recovery_passes"]) == 2
        and int(p["recovered_fill_count"]) == int(p["single_pass_fill_count"])
        and p["idempotent"] is True
    )
    return result_of(evidence, "E8", Verdict.PASS if passed else Verdict.FAIL, (("idempotent_recovery", passed),))


def _eval_e9(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = (
        p["unsettled_within_budget"] is True
        and p["settled_after_restart"] is True
        and p["decision_executed"] is False
    )
    return result_of(evidence, "E9", Verdict.PASS if passed else Verdict.FAIL, (("graceful_restart", passed),))


def _eval_e10(evidence: ScenarioEvidence) -> Any:
    p = evidence.payload
    passed = p["truncated_tail_ok"] is True and p["mid_file_fails_closed"] is True
    return result_of(evidence, "E10", Verdict.PASS if passed else Verdict.FAIL, (("tail_corruption", passed),))


_E_RUNNERS: dict[str, Any] = {
    "E1": _run_e1,
    "E2": _run_e2,
    "E3": _run_e3,
    "E4": _run_e4,
    "E5": _run_e5,
    "E6": _run_e6,
    "E7": _run_e7,
    "E8": _run_e8,
    "E9": _run_e9,
    "E10": _run_e10,
}

_E_EVALUATORS: dict[str, Any] = {
    "E1": _eval_e1,
    "E2": _eval_e2,
    "E3": _eval_e3,
    "E4": _eval_e4,
    "E5": _eval_e5,
    "E6": _eval_e6,
    "E7": _eval_e7,
    "E8": _eval_e8,
    "E9": _eval_e9,
    "E10": _eval_e10,
}

_E_TITLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "E1": ("before send", ()),
    "E2": ("after send resting", ("H6",)),
    "E3": ("after send ioc fill", ("H6",)),
    "E4": ("fill seen not journaled", ()),
    "E5": ("after journal fill", ()),
    "E6": ("during cancel", ()),
    "E7": ("during repeg", ()),
    "E8": ("kill in recovery", ()),
    "E9": ("graceful shutdown", ()),
    "E10": ("journal tail corruption", ()),
}


def scenario_specs() -> tuple[ScenarioSpec, ...]:
    """Group E scenario registry."""
    return tuple(
        ScenarioSpec(scenario_id, "E", title, hypotheses, isolated_run(_E_RUNNERS[scenario_id]), _E_EVALUATORS[scenario_id])
        for scenario_id, (title, hypotheses) in sorted(_E_TITLES.items(), key=lambda item: int(item[0][1:]))
    )
