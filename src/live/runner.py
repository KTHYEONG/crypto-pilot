"""일일 섬도우 사이클 오케스트레이션.

어떤 게이트든 위반하면 주문을 하나도 생성하지 않고 HALT를 반환한다(부분 집행 금지).
I-LEDGER-DURABLE: 집행 구간은 try/finally 로 감싸 어떤 예외 경로에서도 이미 확인된
체결은 원장에 영속된 후 재전파된다.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from src.live.account import (
    AccountSnapshot,
    assert_drawdown_within_limit,
    assert_venue_configuration,
    fetch_account_snapshot,
    reconcile_or_halt,
    resolve_sizing_equity,
)
from src.live.audit import AuditLog, default_audit_log_path
from src.live.errors import LiveTradingError, RiskGateBreach
from src.live.executor import (
    ExecutionOutcome,
    PassiveExecutionPolicy,
    cancel_orphan_orders,
    execute_intents,
)
from src.live.filters import parse_exchange_filters
from src.live.ledger import LedgerState, apply_outcomes, default_ledger_path, load_ledger, save_ledger
from src.live.planner import OrderIntent, plan_orders
from src.live.rest import BinanceFuturesRestClient, parse_rate_limits
from src.live.settings import LiveSettings
from src.live.signal import assert_signal_available, assert_signal_fresh, latest_target_weights
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
    equity: Decimal,
) -> None:
    """사이클 수준 리스크 게이트. 위반 시 RiskGateBreach로 전체 HALT.

    분모는 settings 상수가 아니라 주입된 equity(resolve_sizing_equity 결과)다.
    """
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
    """게이트 순서: 인과성/스테일 -> 거래소 메타 -> 계좌 -> 고아 정리 -> 재조정 ->
    에쿼티/드로다운 -> 신호 -> 목표수량 -> 계획 -> 리스크 게이트 -> 집행 -> 원장 영속."""
    now_ts = now if now is not None else pd.Timestamp.now(tz="UTC")
    try:
        # 1) 인과성 하한 게이트: 결정 시각 T의 주문은 T+1h 이전에 생성될 수 없다.
        assert_signal_available(decision_time, now_ts)
        # 2) 스테일 상한 게이트: N일 정지 후 복귀 시 과거 신호 실주문을 차단한다.
        assert_signal_fresh(
            decision_time, now_ts, pd.Timedelta(hours=settings.max_signal_staleness_hours)
        )

        audit = AuditLog(default_audit_log_path("shadow_cycle", for_date=decision_time))
        run_id = decision_time.strftime("%Y%m%d")
        audit.context.update(run_id=run_id, mode=settings.mode.value)

        market_client = _market_client(settings, decision_time)
        exchange_info_payload: dict[str, Any] = market_client.exchange_info()
        filters = parse_exchange_filters(exchange_info_payload)
        rate_limits = parse_rate_limits(exchange_info_payload)

        order_client = _order_client(settings, decision_time)
        # -1021 이후가 아니라 사전에 시계를 동기화한다.
        order_client.sync_server_time()
        snapshot = fetch_account_snapshot(order_client, now=now_ts)
        assert_venue_configuration(snapshot)

        ledger_path = Path(settings.ledger_path) if settings.ledger_path else default_ledger_path()
        ledger_state = load_ledger(ledger_path)
        ledger_positions = ledger_state.positions

        # 3) 고아 주문 정리는 재조정 '이전에' 이뤄져야 한다(GTX 잔존 -> 원장 괴리 방지).
        cancel_orphan_orders(order_client, run_id, audit)
        reconcile_or_halt(snapshot, ledger_positions, qty_tolerance_fraction=_RECONCILE_TOLERANCE_FRACTION)

        # 4) I-EQUITY-MTM / I-DD-HALT.
        equity = resolve_sizing_equity(snapshot, Decimal(str(settings.notional_equity_usdt)))
        assert_drawdown_within_limit(equity, ledger_state.equity_high_water_mark, settings.equity_drawdown_halt)

        weights = latest_target_weights(artifact_path, decision_time, artifact_key=settings.artifact_key)
        marks = _marks_from_tickers(market_client, [str(symbol) for symbol in weights.index])

        targets, dropped = target_quantities(weights, marks, filters, equity)
        for item in dropped:
            audit.record("symbol_dropped", symbol=item.symbol, reason=item.reason)

        intents = plan_orders(targets, snapshot.positions, filters, marks, run_id)

        # 사이클 수준 리스크 게이트를 먼저 검사한다(부분 집행 금지).
        check_risk_gates(intents, targets, marks, snapshot, settings, equity)

        # 종목별 노셔널 상한: 초과 심볼만 드롭한다(비중 재분배 없음).
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
        try:
            outcomes = list(execute_intents(order_client, kept, filters, policy, audit, _clock, time.sleep, rate_limits=rate_limits))
        except LiveTradingError as exc:
            _persist_confirmed_fills(ledger_path, ledger_state, kept, _partial_outcomes(exc), equity)
            raise
        _persist_confirmed_fills(ledger_path, ledger_state, kept, outcomes, equity)

        report = CycleReport(
            status="COMPLETE",
            reason=None,
            decision_time=decision_time,
            intent_count=len(outcomes),
            outcomes=tuple(outcomes),
        )
        audit.record("cycle_complete", intents=len(outcomes))
        return report
    except (LiveTradingError, ValueError, OSError) as exc:
        logger.error("[SYS] shadow cycle halted reason=%s", exc)
        return CycleReport(
            status="HALT",
            reason=str(exc),
            decision_time=decision_time,
            intent_count=0,
        )


def _partial_outcomes(exc: LiveTradingError) -> list[ExecutionOutcome]:
    """execute_intents 가 부분 체결을 예외에 붙여 재전파한 경우 회복한다."""
    partial = exc.partial_outcomes or ()
    return list(partial)


def _persist_confirmed_fills(
    ledger_path: Path,
    base_state: LedgerState,
    intents: Sequence[OrderIntent],
    outcomes: Sequence[ExecutionOutcome],
    equity: Decimal,
) -> None:
    """확인된 체결과 단조 증가한 hwm 을 원자적으로 영속한다."""
    if not outcomes:
        # 체결이 없어도 hwm 은 ratchet 한다.
        save_ledger(
            ledger_path,
            LedgerState(positions=dict(base_state.positions), equity_high_water_mark=max(base_state.equity_high_water_mark, equity)),
        )
        return
    paired_intents = list(intents)[: len(outcomes)]
    updated_positions = apply_outcomes(base_state.positions, paired_intents, outcomes)
    save_ledger(
        ledger_path,
        LedgerState(
            positions=updated_positions,
            equity_high_water_mark=max(base_state.equity_high_water_mark, equity),
        ),
    )


def _marks_from_tickers(client: Any, symbols: Sequence[str]) -> dict[str, Decimal]:
    """전 종목 호가를 배치 1회로 수집해 mid mark 를 만든다(N+1 호출 금지)."""
    wanted = list(symbols)
    batch_getter = getattr(client, "book_tickers", None)
    payload_map: dict[str, Any]
    if callable(batch_getter):
        payload_map = batch_getter()
    else:
        payload_map = {symbol: client.book_ticker(symbol) for symbol in wanted}
    marks: dict[str, Decimal] = {}
    for symbol in wanted:
        payload = payload_map.get(symbol)
        if payload is None:
            continue
        marks[symbol] = (
            Decimal(str(payload["bidPrice"])) + Decimal(str(payload["askPrice"]))
        ) / 2
    return marks


def _clock() -> float:
    return time.time()


def _market_client(settings: LiveSettings, decision_time: pd.Timestamp) -> BinanceFuturesRestClient:
    return BinanceFuturesRestClient(
        settings.market_data_base_url,
        settings.api_key,
        settings.api_secret,
        settings.mode,
        AuditLog(default_audit_log_path("market_data", for_date=decision_time)),
        recv_window_ms=settings.recv_window_ms,
    )


def _order_client(settings: LiveSettings, decision_time: pd.Timestamp) -> BinanceFuturesRestClient:
    return BinanceFuturesRestClient(
        settings.order_base_url,
        settings.order_api_key or settings.api_key,
        settings.order_api_secret or settings.api_secret,
        settings.mode,
        AuditLog(default_audit_log_path("orders", for_date=decision_time)),
        recv_window_ms=settings.recv_window_ms,
    )
