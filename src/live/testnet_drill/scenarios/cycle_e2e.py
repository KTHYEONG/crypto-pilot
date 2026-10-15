"""Group G runner-cycle end-to-end scenarios (G1..G3).

Cycles run the production ``run_shadow_cycle`` bound to the drill session (``seams.production_bound``): orders pass
the per-order notional cap, and target weights are sized so each leg is about ``probe_notional_usdt``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from decimal import Decimal
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


def _decision_days(count: int) -> list[pd.Timestamp]:
    today = pd.Timestamp.now(tz="UTC").normalize()
    return [today - pd.Timedelta(days=count - 1 - i) for i in range(count)]


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
    days = _decision_days(3)
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
    return ev(
        "G3",
        converged=error is None and getattr(rerun, "status", None) == "COMPLETE" and _converged(session, target, venue, rerun),
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
    """Venue equals target within one lot step, or the rerun recorded an explicit residual for that symbol."""
    residual = {o.symbol for o in getattr(rerun, "outcomes", ()) if o.status in ("RESIDUAL", "RESIDUAL_SUB_MINIMUM")}
    return all(
        abs(venue.get(symbol, Decimal(0)) - qty) <= session.filters[symbol].step_size or symbol in residual
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
    passed = p["converged"] is True and p["no_double_execution"] is True
    return result_of(evidence, "G3", Verdict.PASS if passed else Verdict.FAIL, (("restart_in_cycle", passed),))


_G_TITLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "G1": ("three cycle script", ()),
    "G2": ("idempotent rerun", ()),
    "G3": ("restart in cycle", ()),
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


_G_RUNNERS: dict[str, Any] = {"G1": _swept(_run_g1), "G2": _swept(_run_g2), "G3": _swept(_run_g3)}
_G_EVALUATORS: dict[str, Any] = {"G1": _eval_g1, "G2": _eval_g2, "G3": _eval_g3}


def scenario_specs() -> tuple[ScenarioSpec, ...]:
    """Group G scenario registry."""
    return tuple(
        ScenarioSpec(scenario_id, "G", title, hypotheses, isolated_run(_G_RUNNERS[scenario_id]), _G_EVALUATORS[scenario_id])
        for scenario_id, (title, hypotheses) in sorted(_G_TITLES.items(), key=lambda item: int(item[0][1:]))
    )
