"""일일 섬도우 사이클 오케스트레이션.

어떤 게이트든 위반하면 주문을 하나도 생성하지 않고 HALT를 반환한다(부분 집행 금지).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

from src.live.account import (
    AccountSnapshot,
    assert_venue_configuration,
    fetch_account_snapshot,
    reconcile_or_halt,
)
from src.live.audit import AuditLog, default_audit_log_path
from src.live.errors import LiveTradingError, RiskGateBreach
from src.live.executor import ExecutionOutcome, PassiveExecutionPolicy, execute_intent
from src.live.filters import parse_exchange_filters
from src.live.ledger import apply_outcomes, default_ledger_path, load_ledger, save_ledger
from src.live.planner import OrderIntent, plan_orders
from src.live.rest import BinanceFuturesRestClient
from src.live.settings import LiveSettings
from src.live.signal import assert_signal_available, latest_target_weights
from src.live.sizing import target_quantities

logger = logging.getLogger("LiveRunner")

_RECONCILE_TOLERANCE_FRACTION = 0.001


@dataclass(frozen=True, slots=True)
class CycleReport:
    """사이클 결과 요약. status는 'COMPLETE' 또는 'HALT'다."""

    status: str
    reason: str | None
    decision_time: pd.Timestamp
    intent_count: int
    outcomes: tuple[ExecutionOutcome, ...] = ()


def check_risk_gates(
    intents: Sequence[OrderIntent],
    targets: dict[str, Decimal],
    marks: dict[str, Decimal],
    snapshot: AccountSnapshot,
    settings: LiveSettings,
) -> None:
    """§2.8 사이클 수준 리스크 게이트. 위반 시 RiskGateBreach로 전체 HALT."""
    equity = Decimal(str(settings.notional_equity_usdt))
    gross_notional = sum(
        (abs(qty * marks[symbol]) for symbol, qty in targets.items() if symbol in marks),
        Decimal(0),
    )
    if equity > 0 and gross_notional / equity > Decimal(str(settings.max_gross_leverage)):
        raise RiskGateBreach(
            f"gross leverage {gross_notional / equity} exceeds "
            f"ceiling {settings.max_gross_leverage}"
        )
    if len(intents) > settings.max_daily_orders:
        raise RiskGateBreach(
            f"daily order count {len(intents)} exceeds cap {settings.max_daily_orders}"
        )
    turnover = sum(
        (abs(intent.quantity * marks[intent.symbol]) for intent in intents if intent.symbol in marks),
        Decimal(0),
    )
    if equity > 0 and turnover / equity > Decimal(str(settings.max_daily_turnover_fraction)):
        raise RiskGateBreach(f"turnover {turnover / equity} exceeds cap {settings.max_daily_turnover_fraction}")
    if snapshot.wallet_balance > 0:
        free_fraction = snapshot.available_balance / snapshot.wallet_balance
        if free_fraction < Decimal(str(settings.min_free_margin_fraction)):
            raise RiskGateBreach(f"free margin fraction {free_fraction} below floor")


def run_shadow_cycle(
    settings: LiveSettings,
    decision_time: pd.Timestamp,
    artifact_path: Path,
    *,
    now: pd.Timestamp | None = None,
) -> CycleReport:
    """블루프린트 §2.2 순서 그대로 실행한다."""
    now_ts = now if now is not None else pd.Timestamp.now(tz="UTC")
    try:
        # 1) 인과성 게이트: 결정 시각 T의 주문은 T+1h 이전에 생성될 수 없다.
        assert_signal_available(decision_time, now_ts)

        audit = AuditLog(default_audit_log_path("shadow_cycle"))
        audit.context.update(run_id=decision_time.strftime("%Y%m%dT%H%M%SZ"), mode=settings.mode.value)

        market_client = _market_client(settings)
        filters = parse_exchange_filters(market_client.exchange_info())

        order_client = _order_client(settings)
        snapshot = fetch_account_snapshot(order_client, now=now_ts)
        assert_venue_configuration(snapshot)
        ledger_path = Path(settings.ledger_path) if settings.ledger_path else default_ledger_path()
        ledger_positions = load_ledger(ledger_path)
        reconcile_or_halt(snapshot, ledger_positions, qty_tolerance_fraction=_RECONCILE_TOLERANCE_FRACTION)

        weights = latest_target_weights(artifact_path, decision_time)
        marks = {
            symbol: (Decimal(str(ticker["bidPrice"])) + Decimal(str(ticker["askPrice"]))) / 2
            for symbol, ticker in (
                (sym, market_client.book_ticker(sym)) for sym in map(str, weights.index)
            )
        }

        targets, dropped = target_quantities(
            weights, marks, filters, Decimal(str(settings.notional_equity_usdt))
        )
        for item in dropped:
            audit.record("symbol_dropped", symbol=item.symbol, reason=item.reason)

        run_id = decision_time.strftime("%Y%m%dT%H%M%SZ")
        intents = plan_orders(targets, snapshot.positions, filters, marks, run_id)

        # 사이클 수준 리스크 게이트를 먼저 검사한다(부분 집행 금지).
        check_risk_gates(intents, targets, marks, snapshot, settings)

        # 종목별 노셔널 상한: 초과 심볼만 드롭한다(비중 재분배 없음).
        equity = Decimal(str(settings.notional_equity_usdt))
        kept: list[OrderIntent] = []
        for intent in intents:
            mark = marks.get(intent.symbol)
            if mark is not None and abs(intent.quantity * mark) / equity > Decimal(
                str(settings.max_symbol_notional_fraction)
            ):
                audit.record("intent_dropped_symbol_cap", symbol=intent.symbol)
                continue
            kept.append(intent)

        policy = PassiveExecutionPolicy()
        executed_intents: list[OrderIntent] = []
        outcomes: list[ExecutionOutcome] = []
        for intent in kept:
            symbol_filters = filters.get(intent.symbol)
            if symbol_filters is None:
                continue
            outcomes.append(execute_intent(order_client, intent, symbol_filters, policy, audit, _clock))
            executed_intents.append(intent)

        save_ledger(ledger_path, apply_outcomes(ledger_positions, executed_intents, outcomes))

        report = CycleReport(
            status="COMPLETE",
            reason=None,
            decision_time=decision_time,
            intent_count=len(outcomes),
            outcomes=tuple(outcomes),
        )
        audit.record("cycle_complete", intents=len(outcomes))
        return report
    except (LiveTradingError, ValueError) as exc:
        logger.error("[SYS] shadow cycle halted reason=%s", exc)
        return CycleReport(
            status="HALT",
            reason=str(exc),
            decision_time=decision_time,
            intent_count=0,
        )


def _clock() -> float:
    import time

    return time.time()


def _market_client(settings: LiveSettings) -> BinanceFuturesRestClient:
    return BinanceFuturesRestClient(
        settings.market_data_base_url,
        settings.api_key,
        settings.api_secret,
        settings.mode,
        AuditLog(default_audit_log_path("market_data")),
        recv_window_ms=settings.recv_window_ms,
    )


def _order_client(settings: LiveSettings) -> BinanceFuturesRestClient:
    return BinanceFuturesRestClient(
        settings.order_base_url,
        settings.order_api_key or settings.api_key,
        settings.order_api_secret or settings.api_secret,
        settings.mode,
        AuditLog(default_audit_log_path("orders")),
        recv_window_ms=settings.recv_window_ms,
    )
