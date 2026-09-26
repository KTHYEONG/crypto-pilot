# ruff: noqa
"""일일 섬도우 사이클 오케스트레이션.

어떤 게이트든 위반하면 주문을 하나도 생성하지 않고 HALT를 반환한다(부분 집행 금지).
I-LEDGER-DURABLE: 집행 구간은 try/finally 로 감싸 어떤 예외 경로에서도 이미 확인된
체결은 원장에 영속된 후 재전파된다.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.live.account import (
    RECONCILE_QTY_TOLERANCE_FRACTION,
    AccountSnapshot,
    assert_suppressed_venue_flat,
    assert_venue_configuration,
    effective_positions,
    ensure_venue_leverage,
    explain_breaches,
    fetch_account_snapshot,
    fetch_venue_force_closes,
    find_position_breaches,
    free_margin_breached,
    parse_leverage_brackets,
    reject_intents_over_notional_cap,
    resolve_sizing_equity,
    settled_delisting_symbols,
    synthetic_flat_snapshot,
)
from src.live.derisk import derisk_filter
from src.common.paths import FUTURES_DATA_DIR, LIVE_CAPTURE_DIR
from src.live.audit import AuditLog, default_audit_log_path
from src.live.alerting import dispatch_alert
from src.live.alert_outbox import default_dedupe_key
from src.live.errors import CausalityViolation, LiveTradingError, ReconciliationBreach, RiskGateBreach, StaleSignalError
from src.live.execution_quality import (
    append_execution_quality,
    build_execution_quality_records,
    default_execution_quality_dir,
)
from src.live.depth_capture import DepthCaptureSummary, ExecutionDepthRecorder
from src.live.executor import (
    ExecutionInterrupted,
    ExecutionOutcome,
    ForeignOpenOrderError,
    PassiveExecutionPolicy,
    cancel_orphan_orders,
    execute_intents,
)
from src.live.fills import FillEvent, append_fills, default_fills_dir
from src.live.filters import held_symbols_absent_from_exchange, is_delisted, parse_delivery_schedule, parse_exchange_filters
from src.live.ledger import (
    FundingAccrual,
    LedgerState,
    accrue_funding_by_watermark,
    append_position_snapshot,
    commit_journal_fills,
    default_ledger_path,
    enter_derisk,
    load_ledger,
    mark_fills_recorded,
    save_ledger,
)
from src.live.lifecycle import ShutdownFlag
from src.live.microstructure import (
    append_microstructure,
    build_microstructure_records,
    default_microstructure_dir,
    fetch_book_quotes,
)
from src.live.order_journal import JournalAttempt, JournalFill, OrderJournal, default_order_journal_path
from src.live.recovery import recover_unresolved_orders
from src.live.planner import OrderIntent, partition_risk_controlled, plan_orders
from src.live.portfolio_state import (
    PortfolioStateRecord,
    append_portfolio_state,
    default_portfolio_state_dir,
    resolve_effective_equity,
)
from src.live.rest import BinanceFuturesRestClient, parse_rate_limits
from src.live.settings import ExecutionMode, LiveSettings
from src.live.signal import assert_signal_available, assert_signal_fresh, latest_decision_ohlcv_close, latest_target_weights
from src.live.sizing import target_quantities
from src.live.tax_ledger import (
    TaxLedgerCorruptError,
    TaxRecord,
    append_tax_records,
    collect_and_persist_live_tax,
    default_tax_ledger_dir,
    funding_tax_records,
    reconcile_cycle_cash,
    simulated_tax_records,
)
from src.live.venue_listing import SettlementEvidence, settlement_evidence_from_bars

logger = logging.getLogger("LiveRunner")


@dataclass(frozen=True, slots=True)
class CycleReport:
    """Cycle outcome. status is 'COMPLETE' | 'HALT' | 'INTERRUPTED' | 'DEGRADED'. DEGRADED means a de-risk-only
    subset was executed because the account state is not trusted; the decision day is processed."""

    status: str
    reason: str | None
    decision_time: pd.Timestamp
    intent_count: int
    outcomes: tuple[ExecutionOutcome, ...] = ()
    # minNotional 등으로 드롭된 목표 노셔널 비중(S5): 자본 캡($2,000) 하에서의
    # 페널티를 매 사이클 로그에서 보이게 한다.
    dropped_notional_fraction: float = 0.0


def check_risk_gates(
    intents: Sequence[OrderIntent],
    targets: dict[str, Decimal],
    marks: dict[str, Decimal],
    settings: LiveSettings,
    equity: Decimal,
) -> None:
    """Target-integrity gates (gross leverage of the target book, order count, turnover). Any breach raises `RiskGateBreach` and HALTs the cycle — these gates detect a suspect target, which must not be acted on even in reduce-only form. The free-margin floor is an account-state gate evaluated separately by the caller.

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


def fetch_live_account_equity(settings: LiveSettings, now: pd.Timestamp) -> float:
    """Margin equity (wallet balance + unrealized PnL, USDT) of the LIVE account at ``now``.

    Used by the frozen signal step so exposure is chosen from the same equity the runner sizes
    orders with; the call syncs server time first, exactly like the execution cycle.

    Raises:
        DataIntegrityError: ``settings.mode`` suppresses mutations (PAPER/SHADOW have no real
            account equity to size from) or the snapshot equity is not finite and positive.
    """
    import math

    if settings.mode.suppresses_mutations:
        raise DataIntegrityError("live account equity unavailable in suppressed mode")
    client = _order_client(settings, now)
    client.sync_server_time()
    snapshot = fetch_account_snapshot(client, now=now)
    equity = float(snapshot.wallet_balance + snapshot.unrealized_pnl)
    if not math.isfinite(equity) or equity <= 0.0:
        raise DataIntegrityError(f"live account equity not finite and positive: {equity!r}")
    return equity


PAPER_FUNDING_LAG_HALT: pd.Timedelta = pd.Timedelta(hours=24)
PAPER_FUNDING_LAG_GRACE: pd.Timedelta = pd.Timedelta(hours=1)


def _load_paper_trade_closes(symbols: Sequence[str]) -> dict[str, pd.Series]:
    """Load completed historical 1h trade closes for paper funding-notional estimates. Each close becomes usable only after its bar completes; missing prices or unknown funding keep the paper cash delta unresolved. A delisting announcement alone does not supply a settlement price."""
    from src.common.paths import ohlcv_path  # noqa: PLC0415

    closes_by_symbol: dict[str, pd.Series] = {}
    for symbol in symbols:
        path = ohlcv_path(symbol, "1h")
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if "close" not in frame.columns:
            continue
        if "timestamp" in frame.columns:
            index_open = pd.DatetimeIndex(
                pd.to_datetime(pd.to_numeric(frame["timestamp"]), unit="ms", utc=True)
            )
        elif "datetime" in frame.columns:
            index_open = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], utc=True))
        else:
            continue
        # Each close is indexed by its bar completion time (open + 1h) so a
        # just-opened bar can never value a funding event at its open boundary.
        index_completed = index_open + pd.Timedelta(hours=1)
        series = pd.Series(
            pd.to_numeric(frame["close"]).to_numpy(dtype="float64"), index=index_completed
        ).sort_index()
        closes_by_symbol[symbol] = series[~series.index.duplicated(keep="last")]
    return closes_by_symbol


def _notify_event(
    settings: LiveSettings,
    *,
    event: str,
    detail: str,
    decision_time: pd.Timestamp,
    now: pd.Timestamp,
    discriminator: str | None = None,
) -> None:
    """Dispatch one alert; ``discriminator`` separates several same-event alerts of one cycle."""
    dedupe_key = default_dedupe_key(event, decision_time)
    if discriminator is not None:
        dedupe_key = f"{dedupe_key}:{discriminator}"
    dispatch_alert(
        settings,
        event=event,
        detail=detail,
        decision_time=decision_time,
        dedupe_key=dedupe_key,
        now=now,
    )


def _audit_risk_blocked(audit: AuditLog, risk_blocked: Sequence[OrderIntent]) -> None:
    """Audit each intent withheld under venue position-risk control."""
    for blocked_intent in risk_blocked:
        audit.record(
            "intent_risk_control_blocked",
            symbol=blocked_intent.symbol,
            side=blocked_intent.side,
            quantity=str(blocked_intent.quantity),
        )


def _audit_sizing_mark_fallbacks(
    audit: AuditLog,
    positions: Mapping[str, Decimal],
    marks: Mapping[str, Decimal],
    decision_marks: Mapping[str, Decimal],
) -> None:
    """Audit each nonzero held symbol valued at its fallback decision mark."""
    for sym, qty in positions.items():
        if qty != 0 and sym not in marks:
            audit.record("sizing_mark_fallback", symbol=sym, has_fallback=sym in decision_marks)
            logger.warning(
                "[PORTFOLIO] stage=sizing_equity status=MARK_FALLBACK symbol=%s", sym
            )


def _raise_account_breach(
    breaches: Sequence[Any], foreign_symbols: Sequence[str]
) -> None:
    """Fail closed exactly as before de-risk mode existed."""
    if breaches:
        first = breaches[0]
        raise ReconciliationBreach(
            f"position divergence for {first.symbol}: venue={first.venue_qty} "
            f"ledger={first.ledger_qty}"
        )
    if foreign_symbols:
        raise ForeignOpenOrderError(
            f"{len(foreign_symbols)} foreign open order symbol(s) present; refusing to reconcile"
        )


def _adopt_force_closes(
    settings: LiveSettings,
    order_client: Any,
    journal: OrderJournal,
    audit: AuditLog,
    ledger_path: Path,
    ledger_state: LedgerState,
    breaches: Sequence[Any],
    *,
    now: pd.Timestamp,
    fallback_attempt: JournalAttempt,
    equity: Decimal | None = None,
) -> LedgerState:
    """Journal each explained venue force close as a `venue_force_close` fill and commit it."""
    lookback = pd.Timedelta(hours=settings.venue_force_close_lookback_hours)
    journal_fills = journal.fills_after(-1)
    since = now - lookback
    # 운영자 재동기화는 그 시점까지의 거래소 상태를 원장에 흡수했으므로, 그 이전 강제청산은
    # 새 불일치의 설명 후보가 될 수 없다.
    resync_times = [fill.filled_at for fill in journal_fills if fill.kind == "operator_resync"]
    if resync_times:
        since = max(since, max(resync_times))
    force_closes = fetch_venue_force_closes(order_client, since=since, until=now)
    already = {
        fill.client_order_id
        for fill in journal_fills
        if fill.kind == "venue_force_close" and fill.client_order_id
    }
    fresh = tuple(item for item in force_closes if item.order_id not in already)
    to_adopt, _unexplained = explain_breaches(
        breaches, fresh, qty_tolerance_fraction=RECONCILE_QTY_TOLERANCE_FRACTION
    )
    for item in to_adopt:
        side = item.side
        journal.record_fill(
            kind="venue_force_close",
            attempt_seq=fallback_attempt.attempt_seq,
            symbol=item.symbol,
            side=side,
            quantity=item.executed_qty,
            price=item.avg_price,
            fee_bps=float(settings.taker_fee_bps),
            liquidity="taker",
            reason="venue_force_close",
            filled_at=item.updated_at,
            client_order_id=item.order_id,
            leg_index=0,
            cumulative_executed_qty=None,
            simulated=False,
        )
        audit.record(
            "venue_force_close_adopted",
            symbol=item.symbol,
            order_id=item.order_id,
            auto_close_type=item.auto_close_type,
            quantity=str(item.executed_qty),
        )
        _notify_event(
            settings,
            event="venue_force_close_adopted",
            detail=f"symbol={item.symbol} type={item.auto_close_type} qty={item.executed_qty}",
            decision_time=fallback_attempt.decision_time,
            now=now,
            discriminator=f"order={item.order_id}",
        )
    if to_adopt:
        ledger_state = _commit_and_record(
            settings,
            ledger_path,
            ledger_state,
            journal,
            audit,
            fallback_attempt=fallback_attempt,
            equity=equity,
            executed_decision_time=None,
        )
    return ledger_state


def _alert_reject_cluster(
    settings: LiveSettings,
    outcomes: Sequence[ExecutionOutcome],
    audit: AuditLog,
    *,
    decision_time: pd.Timestamp,
    now: pd.Timestamp,
) -> None:
    """Alert once per clustered rejection code shared by many symbols (observational only)."""
    by_code: dict[int, set[str]] = {}
    for outcome in outcomes:
        if outcome.status == "REJECTED" and outcome.reject_code is not None:
            by_code.setdefault(outcome.reject_code, set()).add(outcome.symbol)
    threshold = settings.reject_cluster_alert_min_symbols
    for code in sorted(by_code):
        symbols = sorted(by_code[code])
        if len(symbols) >= threshold:
            sample = symbols[:5]
            audit.record(
                "intent_reject_cluster", code=code, symbols=len(symbols), sample=sample
            )
            _notify_event(
                settings,
                event="intent_reject_cluster",
                detail=f"code={code} symbols={len(symbols)} sample={sample}",
                decision_time=decision_time,
                now=now,
                discriminator=f"code={code}",
            )


def _load_paper_funding(symbols: Sequence[str]) -> dict[str, pd.Series]:
    """존재하는 파일에 대해서만 펀딩 시리즈를 적재한다(없는 심볼은 스킵)."""
    from src.common.paths import funding_path  # noqa: PLC0415
    from src.market_data.storage.loaders import load_funding_rates  # noqa: PLC0415

    funding_by_symbol: dict[str, pd.Series] = {}
    for symbol in symbols:
        path = funding_path(symbol)
        if not path.exists():
            continue
        funding_by_symbol[symbol] = load_funding_rates(path)
    return funding_by_symbol


def _accrue_ledger_funding(
    state: LedgerState,
    now: pd.Timestamp,
    ledger_path: Path,
    *,
    closed_at: Mapping[str, pd.Timestamp] | None = None,
    tax_dir: Path | None = None,
    run_id: str = "",
    mode: str = "",
) -> tuple[LedgerState, FundingAccrual, tuple[TaxRecord, ...]]:
    """원장에 페이퍼 펀딩비를 워터마크 기준으로 발생시키고 원자적으로 영속한다."""
    held = {symbol: qty for symbol, qty in state.positions.items() if qty != 0}
    history = state.position_history
    accrual_started_at = state.funding_accrual_started_at
    if not history and held:
        bootstrap_at = state.funding_accrued_through if state.funding_accrued_through is not None else now
        history = append_position_snapshot((), bootstrap_at, held)
        if accrual_started_at is None:
            accrual_started_at = bootstrap_at
    symbols = sorted(set(held) | set(state.funding_watermarks))
    accrual = accrue_funding_by_watermark(
        history,
        state.funding_watermarks,
        _load_paper_funding(symbols),
        _load_paper_trade_closes(symbols),
        now,
        closed_at=closed_at,
    )
    if accrual.cash_delta != 0 and state.cash_usdt is None:
        raise DataIntegrityError("paper funding accrual requires cash_usdt")
    funding_records: tuple[TaxRecord, ...] = ()
    if tax_dir is not None and accrual.events:
        funding_records = funding_tax_records(accrual.events, run_id=run_id, mode=mode)
        append_tax_records(funding_records, Path(tax_dir))
    updated = dataclasses.replace(
        state,
        cash_usdt=None if state.cash_usdt is None else state.cash_usdt + accrual.cash_delta,
        funding_accrued_through=now,
        funding_accrual_started_at=accrual_started_at,
        funding_watermarks=accrual.watermarks,
        position_history=history,
    )
    save_ledger(ledger_path, updated)
    return updated, accrual, funding_records


def _delisted_held_symbols(
    positions: Mapping[str, Decimal],
    exchange_info: Mapping[str, Any],
    now: pd.Timestamp,
) -> dict[str, pd.Timestamp]:
    schedule = parse_delivery_schedule(exchange_info)
    delisted: dict[str, pd.Timestamp] = {}
    for symbol, qty in positions.items():
        info = schedule.get(symbol)
        if qty == 0 or info is None or info.delivery_time is None:
            continue
        if is_delisted(info, now):
            delisted[symbol] = info.delivery_time
    return delisted


@dataclass(frozen=True, slots=True)
class DelistingSettlement:
    """One booked settlement of a delivered perpetual position."""

    symbol: str
    quantity: Decimal  # signed ledger quantity that was settled
    price: Decimal | None  # settlement price (PAPER); None in LIVE where the venue books cash
    fee: Decimal  # PAPER settlement fee in USDT; 0 in LIVE
    delivery_time: pd.Timestamp
    evidence_source: str  # "flat_1h_klines" (PAPER) | "venue_flat_position" (LIVE)


def book_delisting_settlements(
    state: LedgerState,
    *,
    mode: ExecutionMode,
    exchange_info: Mapping[str, Any],
    venue_positions: Mapping[str, Decimal],
    evidence: Mapping[str, SettlementEvidence],
    fee_bps: Decimal,
    now: pd.Timestamp,
) -> tuple[LedgerState, tuple[DelistingSettlement, ...]]:
    """Zero ledger positions the venue has settled at delivery, and book the PAPER settlement cash.

    LIVE: a nonzero ledger quantity is settled when the venue position is flat, the listing
    status is SETTLING/CLOSE, and delivery has passed (``settled_delisting_symbols``). The
    ledger quantity is zeroed. Cash is venue-owned in LIVE, so none is booked here. PAPER: a
    held quantity past delivery is settled only when ``evidence`` holds a venue-evidenced
    settlement price. Cash moves by ``quantity x price`` minus the fee, like a closing fill at
    the settlement price. Without evidence the position stays unresolved and the caller fails
    closed. A settlement appends a position-history snapshot at ``now``, so funding accrual
    stops at delivery. SHADOW books nothing. Pure: audit and alert emission belong to the
    caller, which has the ``AuditLog`` and settings.

    Raises:
        DataIntegrityError: ``now`` is naive, ``fee_bps`` is negative, or PAPER cash_usdt is
            None while a settlement must be booked.
    """
    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is None:
        raise DataIntegrityError("delisting settlement now must be tz-aware")
    now_ts = now_ts.tz_convert("UTC")
    fee_rate = Decimal(str(fee_bps))
    if fee_rate < 0:
        raise DataIntegrityError("delisting settlement fee_bps must be >= 0")
    if mode == ExecutionMode.SHADOW:
        return state, ()
    schedule = parse_delivery_schedule(exchange_info)
    if mode == ExecutionMode.PAPER:
        candidates: list[tuple[str, Decimal, SettlementEvidence]] = []
        for symbol in sorted(state.positions):
            qty = state.positions[symbol]
            if qty == 0:
                continue
            info = schedule.get(symbol)
            if info is None or info.delivery_time is None:
                continue
            if not is_delisted(info, now_ts):
                continue
            proven = evidence.get(symbol)
            if proven is None:
                continue
            candidates.append((symbol, qty, proven))
        if not candidates:
            return state, ()
        if state.cash_usdt is None:
            raise DataIntegrityError("paper delisting settlement requires cash_usdt")
        positions = dict(state.positions)
        cash = state.cash_usdt
        booked: list[DelistingSettlement] = []
        for symbol, qty, proven in candidates:
            price = Decimal(proven.price)
            fee = abs(qty * price) * fee_rate / Decimal(10_000)
            cash = cash + qty * price - fee
            positions[symbol] = Decimal(0)
            delivery = pd.Timestamp(proven.delivery_time).tz_convert("UTC")
            booked.append(
                DelistingSettlement(
                    symbol=symbol,
                    quantity=qty,
                    price=price,
                    fee=fee,
                    delivery_time=delivery,
                    evidence_source="flat_1h_klines",
                )
            )
        history = append_position_snapshot(
            state.position_history, now_ts, positions, watermarks=state.funding_watermarks
        )
        return (
            dataclasses.replace(state, positions=positions, cash_usdt=cash, position_history=history),
            tuple(booked),
        )
    settled = settled_delisting_symbols(exchange_info, venue_positions, state.positions, now=now_ts)
    if not settled:
        return state, ()
    live_positions = dict(state.positions)
    live_booked: list[DelistingSettlement] = []
    for symbol in settled:
        qty = live_positions.get(symbol, Decimal(0))
        info = schedule.get(symbol)
        delivery = (
            pd.Timestamp(info.delivery_time).tz_convert("UTC")
            if info is not None and info.delivery_time is not None
            else now_ts
        )
        live_positions[symbol] = Decimal(0)
        live_booked.append(
            DelistingSettlement(
                symbol=symbol,
                quantity=qty,
                price=None,
                fee=Decimal(0),
                delivery_time=delivery,
                evidence_source="venue_flat_position",
            )
        )
    live_history = append_position_snapshot(
        state.position_history, now_ts, live_positions, watermarks=state.funding_watermarks
    )
    return (
        dataclasses.replace(state, positions=live_positions, position_history=live_history),
        tuple(live_booked),
    )


def delisting_settlement_tax_records(
    settlements: Sequence[DelistingSettlement], *, mode: str
) -> tuple[TaxRecord, ...]:
    """One TRADE TaxRecord per evidenced (PAPER) settlement for the cycle cash reconciliation.

    LIVE settlements carry no price (the venue owns the cash), so they emit no record. The
    record id is deterministic per symbol and delivery instant, so a retried cycle never
    double-books.
    """
    records: list[TaxRecord] = []
    for settlement in settlements:
        if settlement.price is None:
            continue
        qty = abs(settlement.quantity)
        price = settlement.price
        notional = abs(settlement.quantity * price)
        delivery = pd.Timestamp(settlement.delivery_time).tz_convert("UTC")
        delivery_ms = int(delivery.value // 1_000_000)
        records.append(
            TaxRecord(
                record_id=f"simulated:DELISTING_SETTLEMENT:{settlement.symbol}:{delivery_ms}",
                kind="TRADE",
                event_time=delivery,
                symbol=settlement.symbol,
                side="SELL" if settlement.quantity > 0 else "BUY",
                quantity=float(qty),
                price=float(price),
                quote_qty=float(notional),
                fee=float(settlement.fee),
                fee_asset="USDT",
                realized_pnl=0.0,
                income_asset="USDT",
                is_maker=False,
                venue_id=0,
                source="delisting_settlement",
                mode=str(mode),
            )
        )
    return tuple(records)


def _settle_delisted_paper_positions(
    state: LedgerState,
    delisted: Mapping[str, pd.Timestamp],
    now: pd.Timestamp,
    ledger_path: Path,
    audit: AuditLog,
    settings: LiveSettings,
    decision_time: pd.Timestamp,
) -> LedgerState:
    """A delisting announcement alone does not supply a settlement price.

    Never synthesize a liquidation at the last trade close, a mark candle or
    zero. The held position and cash stay unresolved; the reason is recorded
    and new risk is stopped until a separately evidenced actual settlement or
    live account reconciliation resolves it.
    """
    held = {
        symbol: state.positions.get(symbol, Decimal(0))
        for symbol in sorted(delisted)
        if state.positions.get(symbol, Decimal(0)) != 0
    }
    if not held:
        return state
    for symbol in sorted(held):
        audit.record(
            "paper_delisted_unresolved",
            symbol=symbol,
            qty=str(held[symbol]),
            delivery_time=delisted[symbol].isoformat(),
            reason="delisting_announcement_is_not_settlement",
        )
    _notify_event(
        settings,
        event="paper_delisted_unresolved",
        detail=f"symbols={','.join(sorted(held))} reason=delisting_announcement_is_not_settlement",
        decision_time=decision_time,
        now=now,
    )
    raise DataIntegrityError(
        f"delisted holding without settlement evidence: symbols={','.join(sorted(held))}; "
        "no synthetic settlement booked, new risk stopped pending actual "
        "settlement or live account reconciliation"
    )


def _enforce_funding_lag(
    accrual: FundingAccrual,
    settings: LiveSettings,
    decision_time: pd.Timestamp,
    now: pd.Timestamp,
) -> None:
    exceeded = sorted(
        symbol for symbol, lag in accrual.lag_by_symbol.items() if lag > PAPER_FUNDING_LAG_HALT
    )
    if exceeded:
        worst = max(accrual.lag_by_symbol[symbol] for symbol in exceeded)
        worst_hours = worst / pd.Timedelta(hours=1)
        raise DataIntegrityError(
            f"paper funding lag exceeded symbols={','.join(exceeded)} max_lag_h={worst_hours:.1f}"
        )
    lagging = sorted(
        symbol
        for symbol in accrual.lag_by_symbol
        if accrual.lag_by_symbol[symbol]
        > 2 * accrual.interval_by_symbol[symbol] + PAPER_FUNDING_LAG_GRACE
    )
    if lagging:
        _notify_event(
            settings,
            event="paper_funding_lag",
            detail=f"symbols={len(lagging)} sample={','.join(lagging[:5])}",
            decision_time=decision_time,
            now=now,
        )


def _execution_quality_metadata(settings: LiveSettings, observed_at: pd.Timestamp) -> dict[str, Any]:
    """Forward-evidence metadata: the configured digest plus the record time."""
    return {"strategy_digest": settings.strategy_digest, "observed_at": observed_at}


def _fill_events_from_journal(
    journal: OrderJournal, fills: Sequence[JournalFill], *, fallback: JournalAttempt
) -> list[FillEvent]:
    """Build `FillEvent`s from journal fills using each fill's attempt context; fills without an attempt (legacy orphan settlements) use `fallback`."""
    events: list[FillEvent] = []
    for fill in fills:
        # 운영자 재동기화는 장부 보정일 뿐 거래가 아니므로 체결·세금 증거를 만들지 않는다.
        if fill.kind == "operator_resync":
            continue
        attempt = journal.attempt(fill.attempt_seq) if fill.attempt_seq is not None else None
        ctx = attempt if attempt is not None else fallback
        marks = ctx.decision_marks
        mark = marks.get(fill.symbol)
        signed_qty = fill.quantity if fill.side == "BUY" else -fill.quantity
        events.append(
            FillEvent(
                decision_time=ctx.decision_time,
                timestamp=fill.filled_at,
                symbol=fill.symbol,
                quantity_delta=signed_qty,
                fill_price=fill.price,
                fee_bps=float(fill.fee_bps),
                reason=str(fill.reason),
                pre_trade_equity=ctx.pre_trade_equity,
                liquidity=str(fill.liquidity),
                mode=str(ctx.mode),
                run_id=str(ctx.run_id),
                leg_index=int(fill.leg_index),
                client_order_id=str(fill.client_order_id or ""),
                decision_mark=mark,
                sizing_anchor=str(ctx.sizing_anchor),
                fill_id=f"journal:{fill.fill_seq}",
            )
        )
    return events


def _commit_and_record(
    settings: LiveSettings,
    ledger_path: Path,
    base_state: LedgerState,
    journal: OrderJournal,
    audit: AuditLog,
    *,
    fallback_attempt: JournalAttempt,
    equity: Decimal | None,
    executed_decision_time: pd.Timestamp | None,
    funding_records: Sequence[TaxRecord] = (),
    settlement_records: Sequence[TaxRecord] = (),
    reconcile_cash_before: Decimal | None = None,
) -> LedgerState:
    """Apply every journal fill above the ledger's applied watermark, then emit evidence for every fill above the recorded watermark: fills parquet rows, PAPER simulated TRADE tax records and the PAPER per-batch cash reconciliation, and finally advance the recorded watermark.

    Commit precedes evidence so the ledger never lags the journal; evidence is idempotent through `fill_id`, so a crash between the two steps is repaired by the next call. Evidence write failures are audited and logged but leave the recorded watermark unchanged, so they are retried on the next cycle instead of being lost.

    `reconcile_cash_before` is the cycle's cash baseline before funding accrual and
    delisting settlement booking (both persist straight to the ledger ahead of this
    call). When given, the cash reconciliation explains the full cycle move --
    fills plus funding plus settlement records -- instead of only the journal-fill
    delta on top of an already-moved baseline.
    """
    track_cash = settings.mode.suppresses_mutations
    pending = journal.fills_after(base_state.journal_applied_fill_seq)
    if pending:
        committed = commit_journal_fills(
            ledger_path,
            base_state,
            pending,
            equity=equity,
            track_cash=track_cash,
            starting_capital=Decimal(str(settings.notional_equity_usdt)),
            executed_decision_time=executed_decision_time,
        )
    else:
        committed = base_state
        if equity is not None and equity > committed.equity_high_water_mark:
            committed = dataclasses.replace(committed, equity_high_water_mark=equity)
        if executed_decision_time is not None and committed.last_executed_decision_time != executed_decision_time:
            committed = dataclasses.replace(committed, last_executed_decision_time=executed_decision_time)
        if committed is not base_state:
            save_ledger(ledger_path, committed)
    fresh = journal.fills_after(committed.journal_recorded_fill_seq)
    fresh = tuple(f for f in fresh if f.fill_seq <= committed.journal_applied_fill_seq)
    if not fresh:
        return committed
    events = _fill_events_from_journal(journal, fresh, fallback=fallback_attempt)
    try:
        fills_dir = Path(settings.fills_dir) if settings.fills_dir else default_fills_dir()
        append_fills(events, fills_dir)
    except Exception as exc:  # noqa: BLE001
        with contextlib.suppress(Exception):
            audit.record("fills_write_failed", error=str(exc))
        logger.warning("[SYS] fills write failed error=%s", exc)
        return committed
    batch_records: tuple[TaxRecord, ...] = ()
    if track_cash:
        try:
            tax_dir = Path(settings.tax_ledger_dir) if settings.tax_ledger_dir else default_tax_ledger_dir()
            tax_dir.mkdir(parents=True, exist_ok=True)
            batch_records = simulated_tax_records(events, settings.mode.value)
            if batch_records:
                append_tax_records(batch_records, tax_dir)
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(Exception):
                audit.record("tax_ledger_write_failed", error=str(exc))
            logger.warning("[SYS] tax_ledger write failed error=%s", exc)
            if isinstance(exc, TaxLedgerCorruptError):
                dispatch_alert(
                    settings,
                    event="tax_ledger_corrupt",
                    detail=f"path={exc.path} line={exc.line_number}",
                    decision_time=fallback_attempt.decision_time,
                    dedupe_key=f"tax_ledger_corrupt:{exc.path}:{exc.line_number}",
                    now=fallback_attempt.started_at,
                )
            return committed
        if base_state.cash_usdt is not None and committed.cash_usdt is not None:
            cash_start = reconcile_cash_before if reconcile_cash_before is not None else base_state.cash_usdt
            # 이전 호출에서 이미 현금에 반영된 체결(증거 재시도분)은 이번 현금 변화에 없으므로 대사에서 뺀다.
            applied_before = {
                f"simulated:TRADE:journal:{fill.fill_seq}"
                for fill in fresh
                if fill.fill_seq <= base_state.journal_applied_fill_seq
            }
            trade_records: tuple[TaxRecord, ...] = (
                *(record for record in batch_records if record.record_id not in applied_before),
                *settlement_records,
            )
            reconciliation = reconcile_cycle_cash(
                cash_start,
                committed.cash_usdt,
                trade_records,
                funding_records,
                tolerance_usdt=Decimal(str(settings.cash_reconcile_tolerance_usdt)),
            )
            audit.record(
                "ledger_reconcile",
                expected=float(reconciliation.expected_delta),
                actual=float(reconciliation.actual_delta),
                difference=float(reconciliation.difference),
                ok=reconciliation.within_tolerance,
            )
            if not reconciliation.within_tolerance:
                logger.warning(
                    "[PORTFOLIO] ledger_reconcile status=MISMATCH difference=%s",
                    reconciliation.difference,
                )
                _notify_event(settings, event="ledger_reconcile_mismatch", detail=f"difference={reconciliation.difference} expected={reconciliation.expected_delta} actual={reconciliation.actual_delta}", decision_time=fallback_attempt.decision_time, now=fallback_attempt.started_at)
    try:
        committed = mark_fills_recorded(ledger_path, committed, fresh[-1].fill_seq)
    except Exception as exc:  # noqa: BLE001 - evidence is idempotent by fill_id; the next call re-emits
        with contextlib.suppress(Exception):
            audit.record("recorded_watermark_failed", error=str(exc))
        logger.warning("[SYS] recorded watermark advance failed error=%s", exc)
    return committed


def _commit_on_abort(
    settings: LiveSettings,
    ledger_path: Path,
    ledger_state: LedgerState,
    journal: OrderJournal,
    audit: AuditLog,
    *,
    attempt: JournalAttempt,
    equity: Decimal | None,
    stage: str,
) -> None:
    """Commit journal fills of an attempt that ended early, never masking the exception that ended it.

    Nothing is stamped executed, so the day stays retryable; a failed commit is audited and repaired by the
    next call because the journal (not this process) is the source of truth.
    """
    try:
        _commit_and_record(settings, ledger_path, ledger_state, journal, audit, fallback_attempt=attempt, equity=equity, executed_decision_time=None)
    except Exception as commit_exc:  # noqa: BLE001 - the interruption/abort determines the report
        with contextlib.suppress(Exception):
            audit.record("commit_failed", stage=stage, error=str(commit_exc))
        logger.warning("[SYS] %s commit failed error=%s", stage, commit_exc)


def _write_execution_quality_once(
    settings: LiveSettings,
    audit: AuditLog,
    *,
    decision_time: pd.Timestamp,
    weights: Any,
    marks: Mapping[str, Decimal],
    intents: Sequence[OrderIntent],
    outcomes: Sequence[ExecutionOutcome],
    observed_at: pd.Timestamp,
) -> None:
    """Write execution-quality records once per attempt (all exit paths; observed_at distinguishes retries)."""
    try:
        execution_quality_dir = Path(settings.execution_quality_dir) if settings.execution_quality_dir else default_execution_quality_dir()
        records = build_execution_quality_records(
            decision_time, settings.mode.value, weights, marks, intents, outcomes,
            **_execution_quality_metadata(settings, observed_at),
        )
        append_execution_quality(records, execution_quality_dir)
    except Exception as exc:  # noqa: BLE001 - observability-only, never halts cycle
        with contextlib.suppress(Exception):
            audit.record("execution_quality_write_failed", error=str(exc))
        logger.warning("[SYS] execution_quality write failed error=%s", exc)


def _run_manifest_path(settings: LiveSettings) -> Path | None:
    root = settings.run_root()
    if root is None:
        return None
    return root / "run_manifest.json"


def _unit_bootstrap_sha256(settings: LiveSettings) -> str | None:
    """Content digest of the unit bootstrap, independent of the sealing nonce, so resealing identical data never trips the manifest check."""
    path = Path(settings.unit_bootstrap_path)
    if not path.exists():
        return None
    raw = path.read_bytes()
    if str(path).endswith(".enc") and settings.artifact_key is not None:
        from src.live.crypto import derive_key, open_bytes

        try:
            plaintext = open_bytes(raw, derive_key(settings.artifact_key))
        except Exception as exc:
            raise DataIntegrityError(f"unit bootstrap seal open failed: {path}") from exc
        return hashlib.sha256(plaintext).hexdigest()
    return hashlib.sha256(raw).hexdigest()


def _current_run_manifest(settings: LiveSettings, now: pd.Timestamp) -> dict[str, Any]:
    from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP20_V2  # noqa: PLC0415
    from src.mhs.params import FROZEN_GROWTH_NAME_CLIP  # noqa: PLC0415

    return {
        "run_id": settings.record_run_id,
        "strategy_id": FROZEN_MHS_TOP20_V2.strategy_id,
        "name_clip": FROZEN_GROWTH_NAME_CLIP,
        "execution_policy": settings.execution_policy,
        "paper_fill_model": settings.paper_fill_model,
        "mode": settings.mode.value,
        "seed_equity_usdt": settings.notional_equity_usdt,
        "unit_bootstrap_sha256": _unit_bootstrap_sha256(settings),
        "git_sha": os.environ.get("GIT_SHA"),
        "started_at": now.isoformat(),
    }


def _assert_run_manifest_compatible(settings: LiveSettings) -> None:
    path = _run_manifest_path(settings)
    if path is None or not path.exists():
        return
    raw = json.loads(path.read_text(encoding="utf-8"))
    current = _current_run_manifest(settings, pd.Timestamp.now(tz="UTC"))
    compared = ("strategy_id", "name_clip", "execution_policy", "mode", "unit_bootstrap_sha256")
    mismatched = [key for key in compared if raw.get(key) != current.get(key)]
    if not mismatched:
        return
    allowed = set()
    if "unit_bootstrap_sha256" in mismatched:
        bootstrap = Path(settings.unit_bootstrap_path)
        if bootstrap.exists():
            legacy_raw = hashlib.sha256(bootstrap.read_bytes()).hexdigest()
            if raw.get("unit_bootstrap_sha256") == legacy_raw and current.get("unit_bootstrap_sha256") != legacy_raw:
                allowed.add("unit_bootstrap_sha256")
    if "name_clip" in mismatched and raw.get("name_clip") is None:
        allowed.add("name_clip")
    if set(mismatched) <= allowed and allowed:
        for key in allowed:
            raw[key] = current[key]
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
        return
    key = mismatched[0]
    raise DataIntegrityError(
        f"run manifest mismatch key={key} manifest={raw.get(key)!r} current={current.get(key)!r}"
    )


def _ensure_run_manifest(settings: LiveSettings, now: pd.Timestamp) -> None:
    path = _run_manifest_path(settings)
    if path is None or path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _current_run_manifest(settings, now)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def run_shadow_cycle(
    settings: LiveSettings,
    decision_time: pd.Timestamp,
    weights_path: Path,
    *,
    now: pd.Timestamp | None = None,
    shutdown: ShutdownFlag | None = None,
    artifact_path: Path | None = None,
) -> CycleReport:
    # backwards compat: artifact_path alias
    if artifact_path is not None:
        weights_path = artifact_path
    """게이트 순서: 인과성/스테일 -> 거래소 메타 -> 계좌 -> 고아 정리 -> 재조정 ->
    에쿼티/드로다운 -> 신호 -> 목표수량 -> 계획 -> 리스크 게이트 -> 집행 -> 원장 영속."""
    now_ts = now if now is not None else pd.Timestamp.now(tz="UTC")
    # 사이클이 어느 지점에서 끝나든 호가 캡처 스레드를 즉시 멈춰 저장한다(세션 상한까지 방치 금지).
    depth_recorders: list[ExecutionDepthRecorder] = []
    try:
        _assert_run_manifest_compatible(settings)
        ledger_path = Path(settings.ledger_path) if settings.ledger_path else default_ledger_path()
        last_executed = load_ledger(ledger_path).last_executed_decision_time
        if last_executed is not None and decision_time <= last_executed:
            try:
                _journal_flushed = OrderJournal(Path(settings.order_journal_path) if settings.order_journal_path else default_order_journal_path())
                _flushed_state = load_ledger(ledger_path)
                if _flushed_state.journal_recorded_fill_seq < _flushed_state.journal_applied_fill_seq:
                    from src.live.audit import AuditLog as _AuditLog  # noqa: PLC0415
                    _flush_audit = _AuditLog(default_audit_log_path("shadow_cycle", for_date=decision_time))
                    _cycle_fallback = JournalAttempt(
                        attempt_seq=-1,
                        decision_time=decision_time,
                        run_id=decision_time.strftime("%Y%m%d"),
                        mode=settings.mode.value,
                        pre_trade_equity=Decimal(0),
                        sizing_anchor="decision_ohlcv_close",
                        decision_marks={},
                        started_at=now_ts,
                    )
                    _commit_and_record(
                        settings, ledger_path, _flushed_state, _journal_flushed, _flush_audit,
                        fallback_attempt=_cycle_fallback, equity=None, executed_decision_time=None,
                    )
            except Exception as exc:  # noqa: BLE001 - flush is best-effort; stamp still guards
                logger.warning("[SYS] already_executed flush failed error=%s", exc)
            logger.info("[EXEC] cycle skipped decision_time=%s reason=already_executed last_executed=%s", decision_time, last_executed)
            return CycleReport(status="COMPLETE", reason="already_executed", decision_time=decision_time, intent_count=0)
        # 1) effective decision time gating (weights_asof)
        weights = latest_target_weights(weights_path, decision_time, artifact_key=settings.artifact_key, max_staleness=pd.Timedelta(hours=settings.max_weights_staleness_hours))
        effective_dt = pd.Timestamp(weights.name)
        assert_signal_available(effective_dt, now_ts)
        assert_signal_fresh(
            effective_dt, now_ts, pd.Timedelta(hours=settings.max_signal_staleness_hours)
        )
        # wiring: weights = latest_target_weights(weights_path, decision_time, artifact_key=settings.artifact_key, max_staleness=pd.Timedelta(hours=settings.max_weights_staleness_hours)); effective_dt = pd.Timestamp(weights.name); assert_signal_available(effective_dt, now_ts)

        run_root = settings.run_root()
        audit = AuditLog(
            default_audit_log_path("shadow_cycle", for_date=decision_time),
            mirror_path=(run_root / "audit" / f"{decision_time:%Y-%m-%d}.jsonl") if run_root is not None else None,
        )
        run_id = decision_time.strftime("%Y%m%d")
        audit.context.update(run_id=run_id, mode=settings.mode.value)

        market_client = _market_client(settings, decision_time)
        exchange_info_payload: dict[str, Any] = market_client.exchange_info()
        filters = parse_exchange_filters(exchange_info_payload)
        rate_limits = parse_rate_limits(exchange_info_payload)

        order_client = _order_client(settings, decision_time)
        if isinstance(order_client, NullOrderClient):
            # 자격증명 없는 PAPER/SHADOW: 실계좌 조회 없이 합성 스냅샷(I-PAPER-NO-CREDENTIALS).
            snapshot = synthetic_flat_snapshot(now_ts)
        else:
            # -1021 이후가 아니라 사전에 시계를 동기화한다.
            order_client.sync_server_time()
            snapshot = fetch_account_snapshot(order_client, now=now_ts)
        assert_venue_configuration(snapshot)

        ledger_state = load_ledger(ledger_path)
        ledger_positions = ledger_state.positions

        # 3) 고아 주문 정리는 재조정 '이전에' 이뤄져야 한다(GTX 잔존 -> 원장 괴리 방지).
        journal = OrderJournal(Path(settings.order_journal_path) if settings.order_journal_path else default_order_journal_path())
        cycle_context = JournalAttempt(
            attempt_seq=-1,
            decision_time=decision_time,
            run_id=run_id,
            mode=settings.mode.value,
            pre_trade_equity=Decimal(0),
            sizing_anchor="decision_ohlcv_close",
            decision_marks={},
            started_at=now_ts,
        )
        if journal.last_fill_seq() < ledger_state.journal_applied_fill_seq:
            # 저널이 원장보다 짧으면(유실·잘림) 이후 체결이 조용히 버려지므로 즉시 중단한다.
            _notify_event(
                settings,
                event="order_journal_regressed",
                detail=f"journal_last_fill_seq={journal.last_fill_seq()} ledger_applied={ledger_state.journal_applied_fill_seq}",
                decision_time=decision_time,
                now=now_ts,
            )
            raise DataIntegrityError(
                f"order journal regressed: last fill_seq {journal.last_fill_seq()} "
                f"< ledger applied {ledger_state.journal_applied_fill_seq}"
            )
        sweep = cancel_orphan_orders(order_client, run_id, audit, journal=journal, now=now_ts, taker_fee_bps=float(settings.taker_fee_bps))
        recovery = recover_unresolved_orders(order_client, journal, audit, now=now_ts, lookback=pd.Timedelta(hours=settings.journal_recovery_lookback_hours), taker_fee_bps=float(settings.taker_fee_bps))
        ledger_state = _commit_and_record(settings, ledger_path, ledger_state, journal, audit, fallback_attempt=cycle_context, equity=None, executed_decision_time=None)
        ledger_positions = ledger_state.positions
        derisk_reasons: list[str] = []
        if settings.mode.suppresses_mutations:
            assert_suppressed_venue_flat(snapshot)
        else:
            settled = settled_delisting_symbols(
                exchange_info_payload, snapshot.positions, ledger_positions, now=now_ts
            )
            for symbol in settled:
                audit.record(
                    "delisting_settlement_pending",
                    symbol=symbol,
                    ledger_qty=str(ledger_positions[symbol]),
                )
            ledger_state, delisting_booked = book_delisting_settlements(
                ledger_state,
                mode=settings.mode,
                exchange_info=exchange_info_payload,
                venue_positions=snapshot.positions,
                evidence={},
                fee_bps=Decimal(0),
                now=now_ts,
            )
            if delisting_booked:
                save_ledger(ledger_path, ledger_state)
                ledger_positions = ledger_state.positions
                for settlement in delisting_booked:
                    audit.record(
                        "delisting_settlement_booked",
                        symbol=settlement.symbol,
                        qty=str(settlement.quantity),
                        price=None if settlement.price is None else str(settlement.price),
                        fee=str(settlement.fee),
                        delivery_time=settlement.delivery_time.isoformat(),
                        evidence_source=settlement.evidence_source,
                    )
                    _notify_event(
                        settings,
                        event="delisting_settled",
                        detail=f"symbol={settlement.symbol} qty={settlement.quantity} source={settlement.evidence_source}",
                        decision_time=decision_time,
                        now=now_ts,
                    )
            breaches = find_position_breaches(
                snapshot,
                ledger_positions,
                qty_tolerance_fraction=RECONCILE_QTY_TOLERANCE_FRACTION,
                settled_symbols=settled,
            )
            if breaches and settings.venue_force_close_auto_adopt:
                ledger_state = _adopt_force_closes(
                    settings, order_client, journal, audit, ledger_path, ledger_state,
                    breaches, now=now_ts, fallback_attempt=cycle_context,
                )
                ledger_positions = ledger_state.positions
                breaches = find_position_breaches(
                    snapshot,
                    ledger_positions,
                    qty_tolerance_fraction=RECONCILE_QTY_TOLERANCE_FRACTION,
                    settled_symbols=settled,
                )
            reasons = (("reconciliation_breach",) if breaches else ()) + (
                ("foreign_open_orders",) if sweep.foreign_symbols else ()
            )
            if reasons and not settings.derisk_mode_enabled:
                _raise_account_breach(breaches, sweep.foreign_symbols)
            if reasons:
                ledger_state = enter_derisk(ledger_path, ledger_state, reasons=reasons, now=now_ts)
                derisk_reasons = sorted(set(ledger_state.derisk_reasons))
                if breaches and not settings.venue_force_close_auto_adopt:
                    audit.record(
                        "derisk_entered",
                        reasons=derisk_reasons,
                        force_close_auto_adopt="off",
                    )
                else:
                    audit.record("derisk_entered", reasons=derisk_reasons)
            elif ledger_state.derisk_since is not None:
                derisk_reasons = sorted(set(ledger_state.derisk_reasons))

        # weights already loaded as effective row (reused)
        current_positions = effective_positions(settings.mode, snapshot, ledger_positions)
        wanted_symbols = sorted({str(s) for s in weights.index} | set(current_positions))
        marks = _marks_from_tickers(market_client, wanted_symbols)
        # microstructure capture — reuse quotes from _marks_from_tickers (single book_tickers call)
        try:
            quotes = getattr(_marks_from_tickers, "_last_quotes", {})
            premium = None
            try:
                premium = market_client.premium_index()
            except Exception:
                premium = None
            if quotes:
                micro_records = build_microstructure_records(decision_time, settings.mode.value, quotes, premium)
                microstructure_dir = Path(settings.microstructure_dir) if settings.microstructure_dir else default_microstructure_dir()
                append_microstructure(micro_records, microstructure_dir)
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(Exception):
                audit.record("microstructure_write_failed", error=str(exc))
            logger.warning("[SYS] microstructure write failed error=%s", exc)
        # decision-close anchor: completed 1h trade OHLCV close at the exact
        # decision date sizes orders; the current ticker only checks tradability.
        decision_closes = latest_decision_ohlcv_close(weights_path, effective_dt, artifact_key=settings.artifact_key)
        active_symbols = {
            str(symbol)
            for symbol, weight in weights.items()
            if pd.notna(weight) and float(weight) != 0.0
        }
        missing_decision_closes = sorted(active_symbols - {str(symbol) for symbol in decision_closes.index})
        if missing_decision_closes:
            raise DataIntegrityError(
                "decision OHLCV close missing for active targets: "
                + ",".join(missing_decision_closes)
            )
        decision_marks: dict[str, Decimal] = {str(k): Decimal(str(v)) for k, v in decision_closes.items()}
        _sizing_anchor = "decision_ohlcv_close"
        # 4) I-EQUITY-MTM (백테스트 패리티).
        cash_before: Decimal | None = None
        funding_records: tuple[TaxRecord, ...] = ()
        settlement_records: tuple[TaxRecord, ...] = ()
        if settings.mode.suppresses_mutations:
            absent_held = held_symbols_absent_from_exchange(ledger_state.positions, exchange_info_payload)
            if absent_held:
                audit.record("held_symbol_absent", symbols=absent_held)
                raise DataIntegrityError(f"held symbols absent from exchangeInfo symbols={','.join(absent_held)}; resolve via actual venue settlement or live account reconciliation")
            delisted = _delisted_held_symbols(ledger_state.positions, exchange_info_payload, now_ts)
            cash_before = ledger_state.cash_usdt
            funding_tax_dir = Path(settings.tax_ledger_dir) if settings.tax_ledger_dir else default_tax_ledger_dir()
            ledger_state, accrual, funding_records = _accrue_ledger_funding(
                ledger_state,
                now_ts,
                ledger_path,
                closed_at=delisted,
                tax_dir=funding_tax_dir,
                run_id=settings.record_run_id or run_id,
                mode=settings.mode.value,
            )
            if delisted:
                proven: dict[str, SettlementEvidence] = {}
                for delisted_symbol, delivery_time in delisted.items():
                    try:
                        flat_evidence = settlement_evidence_from_bars(
                            FUTURES_DATA_DIR / "ohlcv" / "1h" / f"{delisted_symbol}.parquet",
                            symbol=delisted_symbol,
                            delivery_time=delivery_time,
                            min_flat_bars=settings.delisting_settlement_min_flat_bars,
                            price_rtol=settings.delisting_settlement_price_rtol,
                        )
                    except DataIntegrityError:
                        flat_evidence = None
                    if flat_evidence is not None:
                        proven[delisted_symbol] = flat_evidence
                ledger_state, paper_booked = book_delisting_settlements(
                    ledger_state,
                    mode=settings.mode,
                    exchange_info=exchange_info_payload,
                    venue_positions={},
                    evidence=proven,
                    fee_bps=Decimal(str(settings.delisting_settlement_fee_bps)),
                    now=now_ts,
                )
                if paper_booked:
                    settlement_records = delisting_settlement_tax_records(
                        paper_booked, mode=settings.mode.value
                    )
                    if settlement_records:
                        settlement_tax_dir = (
                            Path(settings.tax_ledger_dir)
                            if settings.tax_ledger_dir
                            else default_tax_ledger_dir()
                        )
                        append_tax_records(settlement_records, settlement_tax_dir)
                    save_ledger(ledger_path, ledger_state)
                    for settlement in paper_booked:
                        audit.record(
                            "delisting_settlement_booked",
                            symbol=settlement.symbol,
                            qty=str(settlement.quantity),
                            price=None if settlement.price is None else str(settlement.price),
                            fee=str(settlement.fee),
                            delivery_time=settlement.delivery_time.isoformat(),
                            evidence_source=settlement.evidence_source,
                        )
                        _notify_event(
                            settings,
                            event="delisting_settled",
                            detail=f"symbol={settlement.symbol} qty={settlement.quantity} price={settlement.price} source={settlement.evidence_source}",
                            decision_time=decision_time,
                            now=now_ts,
                        )
                unresolved = {
                    symbol: delivery_time
                    for symbol, delivery_time in delisted.items()
                    if ledger_state.positions.get(symbol, Decimal(0)) != 0
                }
                if unresolved:
                    ledger_state = _settle_delisted_paper_positions(
                        ledger_state, unresolved, now_ts, ledger_path, audit, settings, decision_time
                    )
            _enforce_funding_lag(accrual, settings, decision_time, now_ts)
            ledger_positions = ledger_state.positions
            current_positions = effective_positions(settings.mode, snapshot, ledger_positions)
        _audit_sizing_mark_fallbacks(audit, ledger_positions, marks, decision_marks)
        equity = resolve_sizing_equity(
            snapshot,
            Decimal(str(settings.notional_equity_usdt)),
            mode=settings.mode,
            cash_usdt=ledger_state.cash_usdt,
            positions=ledger_positions,
            marks=marks,
            fallback_marks=decision_marks,
        )

        targets, dropped = target_quantities(weights, marks, filters, equity, sizing_marks=decision_marks)
        for item in dropped:
            audit.record("symbol_dropped", symbol=item.symbol, reason=item.reason)
        # S5 공시: 드롭 목표 노셔널 / 전체 목표 노셔널. 드롭이 없으면 정확히 0.0.
        dropped_notional = sum(
            (abs(item.target_notional) for item in dropped), Decimal(0)
        )
        total_target_notional = dropped_notional + sum(
            (
                abs(equity * Decimal(str(float(weights[symbol]))))
                for symbol in targets
                if symbol in weights.index
            ),
            Decimal(0),
        )
        dropped_fraction = (
            float(dropped_notional / total_target_notional)
            if total_target_notional > 0
            else 0.0
        )

        intents = plan_orders(targets, current_positions, filters, marks, run_id)
        intents, risk_blocked = partition_risk_controlled(intents, filters)
        _audit_risk_blocked(audit, risk_blocked)

        # 사이클 수준 리스크 게이트를 먼저 검사한다(부분 집행 금지).
        check_risk_gates(intents, targets, marks, settings, equity)
        margin_floor = not settings.mode.suppresses_mutations and free_margin_breached(
            snapshot, min_free_margin_fraction=settings.min_free_margin_fraction
        )
        if margin_floor and not settings.derisk_mode_enabled:
            raise RiskGateBreach("free margin fraction below floor")
        derisk_active = bool(derisk_reasons) or bool(margin_floor)
        derisk_blocked_count = 0
        if margin_floor and "free_margin_floor" not in derisk_reasons:
            derisk_reasons = sorted([*derisk_reasons, "free_margin_floor"])

        # NO-LIVE-ONLY-GATES: 전략 차원의 종목별 노셔널 상한은 없다(백테스트 패리티). LIVE에서는 거래소 브래킷 notionalCap을 넘는 주문만 거부한다.
        kept: list[OrderIntent] = list(intents)
        if not settings.mode.suppresses_mutations and kept:
            brackets = parse_leverage_brackets(
                order_client.request("GET", "/fapi/v1/leverageBracket", signed=True)
            )
            leverages = ensure_venue_leverage(
                order_client,
                sorted({intent.symbol for intent in kept}),
                brackets,
                max_gross_leverage=settings.max_gross_leverage,
                buffer_fraction=settings.leverage_buffer_fraction,
                audit=audit,
            )
            kept = reject_intents_over_notional_cap(kept, brackets, leverages, audit)
        # 외부 주문이 있으면 derisk 사유(foreign_open_orders)가 되므로 제외는 derisk_filter 한 곳에서만 일어난다.
        if derisk_active:
            plan = derisk_filter(kept, snapshot.positions, excluded_symbols=sweep.foreign_symbols)
            for blocked_intent, block_reason in plan.blocked:
                audit.record(
                    "derisk_blocked",
                    symbol=blocked_intent.symbol,
                    reason=block_reason,
                )
            derisk_blocked_count = len(plan.blocked)
            kept = list(plan.kept)
        def _stop_depth(post_window_s: float) -> DepthCaptureSummary | None:
            """Stop the execution-depth recorder exactly once; None when never started."""
            if not depth_recorders:
                return None
            return depth_recorders.pop().stop(post_window_s=post_window_s, shutdown=shutdown)

        if settings.mode is ExecutionMode.LIVE_TESTNET:
            audit.record("exec_depth_skipped", reason="testnet")
        elif not settings.exec_depth_capture_enabled:
            audit.record("exec_depth_skipped", reason="disabled")
        elif not kept:
            audit.record("exec_depth_skipped", reason="empty")
        else:
            capture_syms = [
                i.symbol
                for i in sorted(kept, key=lambda x: abs(x.quantity * marks.get(x.symbol, Decimal(0))), reverse=True)
            ][: settings.exec_depth_max_symbols]
            depth_recorder = ExecutionDepthRecorder(
                capture_syms,
                decision_time=decision_time,
                run_id=run_id,
                mode=settings.mode.value,
                root=LIVE_CAPTURE_DIR,
                stream_url=settings.exec_depth_stream_url,
                levels=settings.exec_depth_levels,
                update_ms=settings.exec_depth_update_ms,
                flush_interval_s=settings.exec_depth_flush_interval_s,
                max_session_s=settings.exec_depth_max_session_s,
            )
            depth_recorders.append(depth_recorder)
            depth_recorder.start()

        for sym, reason in _uncovered_positions(current_positions, targets, filters, marks, kept):
            with contextlib.suppress(Exception):
                audit.record("position_uncovered", symbol=sym, reason=reason)

        from src.live.executor import FeeSchedule, backtest_parity_execution_policy, strict_passive_execution_policy  # noqa: PLC0415

        fee_schedule = FeeSchedule(maker_fee_bps=settings.maker_fee_bps, taker_fee_bps=settings.taker_fee_bps)
        if settings.execution_policy == "strict_passive":
            policy = strict_passive_execution_policy(fee_schedule, settings.taker_slippage_bps, settings.passive_timeout_minutes)
        else:
            policy = backtest_parity_execution_policy(fee_schedule, settings.taker_slippage_bps)
        audit.record("execution_policy_selected", policy=settings.execution_policy)
        paper_fill_model = settings.paper_fill_model if settings.mode is ExecutionMode.PAPER else None
        sink: list[ExecutionOutcome] = []
        outcomes: list[ExecutionOutcome] = []
        final_state: LedgerState | None = None
        attempt = journal.begin_attempt(decision_time=decision_time, run_id=run_id, mode=settings.mode.value, pre_trade_equity=equity, sizing_anchor=_sizing_anchor, decision_marks={i.symbol: decision_marks[i.symbol] for i in kept if i.symbol in decision_marks}, started_at=now_ts)
        try:
            try:
                outcomes = list(execute_intents(order_client, kept, filters, policy, audit, _clock, time.sleep, rate_limits=rate_limits, outcome_sink=sink, shutdown=shutdown, paper_fill_model=paper_fill_model, journal=journal, attempt=attempt, shutdown_cleanup_budget_s=settings.execution_shutdown_cleanup_budget_s))
                _alert_reject_cluster(settings, outcomes, audit, decision_time=decision_time, now=now_ts)
                final_state = _commit_and_record(settings, ledger_path, ledger_state, journal, audit, fallback_attempt=attempt, equity=equity, executed_decision_time=None if derisk_active else decision_time, funding_records=funding_records, settlement_records=settlement_records, reconcile_cash_before=cash_before)
                _write_execution_quality_once(settings, audit, decision_time=decision_time, weights=weights, marks=marks, intents=kept, outcomes=outcomes, observed_at=now_ts)
            except ExecutionInterrupted as exc:
                partial = list(sink) if sink else _partial_outcomes(exc)
                _commit_on_abort(settings, ledger_path, ledger_state, journal, audit, attempt=attempt, equity=equity, stage="interrupted")
                if partial:
                    _write_execution_quality_once(settings, audit, decision_time=decision_time, weights=weights, marks=marks, intents=kept, outcomes=partial, observed_at=now_ts)
                    _alert_reject_cluster(settings, partial, audit, decision_time=decision_time, now=now_ts)
                audit.record("cycle_interrupted", reason="shutdown_requested", intents=len(partial))
                return CycleReport(
                    status="INTERRUPTED",
                    reason="shutdown_requested",
                    decision_time=decision_time,
                    intent_count=len(partial),
                    outcomes=tuple(partial),
                    dropped_notional_fraction=dropped_fraction,
                )
            except LiveTradingError as exc:
                to_persist = sink if sink else _partial_outcomes(exc)
                _commit_on_abort(settings, ledger_path, ledger_state, journal, audit, attempt=attempt, equity=equity, stage="halt")
                if to_persist:
                    if not outcomes:
                        outcomes = list(to_persist)
                    _write_execution_quality_once(settings, audit, decision_time=decision_time, weights=weights, marks=marks, intents=kept, outcomes=list(to_persist), observed_at=now_ts)
                    _alert_reject_cluster(settings, outcomes, audit, decision_time=decision_time, now=now_ts)
                raise
        except LiveTradingError:
            raise
        except BaseException:
            _commit_on_abort(settings, ledger_path, ledger_state, journal, audit, attempt=attempt, equity=equity, stage="abort")
            raise
        assert final_state is not None
        try:
            equity_eff, equity_source = resolve_effective_equity(settings.mode, equity, final_state.cash_usdt, final_state.positions, marks)
            gross_notional = sum(
                (abs(qty * marks[symbol]) for symbol, qty in final_state.positions.items() if symbol in marks),
                Decimal(0),
            )
            n_holdings = sum(1 for qty in final_state.positions.values() if qty != 0)
            portfolio_record = PortfolioStateRecord(
                decision_time=decision_time,
                mode=settings.mode.value,
                equity_usdt=float(equity_eff),
                equity_source=equity_source,
                cash_usdt=float(final_state.cash_usdt) if final_state.cash_usdt is not None else None,
                wallet_balance_usdt=float(snapshot.wallet_balance) if snapshot.wallet_balance is not None else None,
                unrealized_pnl_usdt=float(snapshot.unrealized_pnl) if snapshot.unrealized_pnl is not None else None,
                equity_high_water_mark_usdt=float(final_state.equity_high_water_mark),
                gross_notional_usdt=float(gross_notional),
                n_holdings=int(n_holdings),
                intent_count=len(outcomes),
                dropped_notional_fraction=float(dropped_fraction),
            )
            portfolio_dir = Path(settings.portfolio_state_dir) if settings.portfolio_state_dir else default_portfolio_state_dir()
            append_portfolio_state(portfolio_record, portfolio_dir)
        except Exception as exc:  # noqa: BLE001 - observability-only, never halts cycle
            with contextlib.suppress(Exception):
                audit.record("portfolio_state_write_failed", error=str(exc))
            logger.warning("[SYS] portfolio_state write failed error=%s", exc)
        depth_summary = _stop_depth(settings.exec_depth_post_window_s)
        if depth_summary is not None:
            audit.record("exec_depth_capture", **dataclasses.asdict(depth_summary))
        # tax ledger — fail-soft, never halts cycle (uses append_tax_records(tax_records, tax_dir))
        # PAPER simulated TRADE records + per-batch cash reconciliation already ran
        # inside _commit_and_record; only LIVE venue collection remains here.
        try:
            tax_dir = Path(settings.tax_ledger_dir) if settings.tax_ledger_dir else default_tax_ledger_dir()
            tax_dir.mkdir(parents=True, exist_ok=True)
            if not settings.mode.suppresses_mutations:
                if settings.tax_collection_enabled:
                    try:
                        _, live_tax_issues = collect_and_persist_live_tax(
                            order_client,
                            wanted_symbols,
                            tax_dir,
                            settings.mode.value,
                            now=now_ts,
                            settings=settings,
                        )
                    except DataIntegrityError as exc:
                        audit.record("tax_watermark_invalid", error=str(exc))
                        logger.warning("[SYS] tax_watermark_invalid error=%s", exc)
                        dispatch_alert(
                            settings,
                            event="tax_ledger_corrupt",
                            detail=f"path={tax_dir} error={type(exc).__name__}",
                            decision_time=decision_time,
                            dedupe_key=f"tax_ledger_corrupt:{decision_time.date()}",
                            now=now_ts,
                        )
                    else:
                        for live_issue in live_tax_issues:
                            audit.record(
                                "tax_collect_issue",
                                stream=live_issue.stream,
                                stage=live_issue.stage,
                                detail=live_issue.detail,
                            )
                            logger.warning(
                                "[EXEC] tax_collect_issue stream=%s stage=%s",
                                live_issue.stream,
                                live_issue.stage,
                            )
                            if live_issue.stage == "retention_gap":
                                dispatch_alert(
                                    settings,
                                    event="tax_income_gap",
                                    detail=live_issue.detail,
                                    decision_time=decision_time,
                                    dedupe_key=f"tax_income_gap:{live_issue.detail}",
                                    now=now_ts,
                                )
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(Exception):
                audit.record("tax_ledger_write_failed", error=str(exc))
            logger.warning("[SYS] tax_ledger write failed error=%s", exc)

        report = CycleReport(
            status="DEGRADED" if derisk_active else "COMPLETE",
            reason=",".join(derisk_reasons) if derisk_active else None,
            decision_time=decision_time,
            intent_count=len(outcomes),
            outcomes=tuple(outcomes),
            dropped_notional_fraction=dropped_fraction,
        )
        if derisk_active:
            derisk_since = load_ledger(ledger_path).derisk_since
            detail = (
                f"reasons={','.join(derisk_reasons)} kept={len(kept)} "
                f"blocked={derisk_blocked_count} derisk_since={derisk_since}"
            )
            if not settings.venue_force_close_auto_adopt and "reconciliation_breach" in derisk_reasons:
                detail += " force_close_auto_adopt=off"
            audit.record("cycle_degraded", detail=detail)
            _notify_event(
                settings,
                event="cycle_degraded",
                detail=detail,
                decision_time=decision_time,
                now=now_ts,
            )
            return report
        _ensure_run_manifest(settings, now_ts)
        audit.record("cycle_complete", intents=len(outcomes))
        return report
    except (LiveTradingError, ValueError, OSError, StaleSignalError, CausalityViolation) as exc:
        logger.error("[SYS] shadow cycle halted reason=%s", exc)
        if isinstance(exc, TaxLedgerCorruptError):
            dispatch_alert(
                settings,
                event="tax_ledger_corrupt",
                detail=f"path={exc.path} line={exc.line_number}",
                decision_time=decision_time,
                dedupe_key=f"tax_ledger_corrupt:{exc.path}:{exc.line_number}",
                now=now_ts,
            )
        return CycleReport(
            status="HALT",
            reason=str(exc),
            decision_time=decision_time,
            intent_count=0,
        )
    finally:
        while depth_recorders:
            depth_recorders.pop().stop(post_window_s=0.0)


def _uncovered_positions(
    current: Mapping[str, Decimal],
    targets: Mapping[str, Decimal],
    filters: Mapping[str, Any],
    marks: Mapping[str, Decimal],
    intents: Sequence[OrderIntent],
) -> list[tuple[str, str]]:
    """보유 중 청산 불가 위험을 감지한다. 순수 함수, never raises."""
    covered = {i.symbol for i in intents}
    result: list[tuple[str, str]] = []
    for symbol, qty in current.items():
        if qty == Decimal(0):
            continue
        if symbol in covered:
            continue
        target = targets.get(symbol, Decimal(0))
        if target == qty:
            continue
        if symbol not in filters:
            result.append((symbol, "no_filters"))
        elif symbol not in marks:
            result.append((symbol, "no_mark"))
        else:
            continue
    return result


def _partial_outcomes(exc: LiveTradingError) -> list[ExecutionOutcome]:
    """execute_intents 가 부분 체결을 예외에 붙여 재전파한 경우 회복한다."""
    partial = exc.partial_outcomes or ()
    return list(partial)


def _marks_from_tickers(client: Any, symbols: Sequence[str]) -> dict[str, Decimal]:
    """전 종목 호가를 배치 1회로 수집해 mid mark 를 만든다(N+1 호출 금지)."""
    quotes = fetch_book_quotes(client, symbols)
    _marks_from_tickers._last_quotes = quotes  # type: ignore[attr-defined]
    marks: dict[str, Decimal] = {sym: q.mid for sym, q in quotes.items()}
    return marks

# cache for microstructure reuse (populated by _marks_from_tickers)
_marks_from_tickers._last_quotes = {}  # type: ignore[attr-defined]

# wiring anchors for lean_check
# fetch_book_quotes(market_client, wanted_symbols)
# monthly_partition_path
# load_partitions


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


class NullOrderClient:
    """자격증명 없는 억제 모드(PAPER/SHADOW) 전용 주문 클라이언트.

    공개 마켓데이터 읽기는 주입된 market client에 위임하고, 서명이 필요한
    조회는 빈 결과로, 변이는 억제 응답으로 스텁한다. 실제 체결은
    execute_intents 의 paper_fill_model 시뮬레이터가 관측 호가로 구동한다.
    """

    def __init__(self, market_client: Any, mode: Any) -> None:
        self._market = market_client
        self._mode = mode

    @property
    def mode(self) -> Any:
        return self._market.mode

    def __getattr__(self, name: str) -> Any:
        # 공개 마켓데이터 읽기(book_ticker/book_tickers/depth/premium_index 등)는 위임한다.
        # _-접두 이름과 _market 미바인딩 시점(복사/피클/introspection)의 무한 재귀를 차단한다.
        if name.startswith("_") or "_market" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self._market, name)

    def sync_server_time(self) -> None:
        return None

    def open_orders(self) -> list[dict[str, Any]]:
        return []

    def new_order(self, params: Mapping[str, Any]) -> Any:
        from src.live.rest import PaperResponse, ShadowResponse
        from src.live.settings import ExecutionMode

        if self._mode is ExecutionMode.PAPER:
            return PaperResponse.suppressed("POST", "/fapi/v1/order", "")
        return ShadowResponse.suppressed("POST", "/fapi/v1/order", "")

    def cancel_order(self, *_a: Any, **_k: Any) -> dict[str, Any]:
        return {}

    def query_order(self, *_a: Any, **_k: Any) -> dict[str, Any]:
        return {}


def _order_client(settings: LiveSettings, decision_time: pd.Timestamp) -> Any:
    if settings.mode.suppresses_mutations and settings.api_key is None:
        # 자격증명 없는 로컬 PAPER/SHADOW: 공개 GET만 쓰는 스텁 클라이언트(I-PAPER-NO-CREDENTIALS).
        return NullOrderClient(_market_client(settings, decision_time), settings.mode)
    if settings.mode.suppresses_mutations:
        # PAPER/SHADOW는 항상 메인넷 read-only 조회 키(api_key)만 쓴다. order_api_key는
        # LIVE_TESTNET 주문 로직 검증 전용 테스트넷 자격증명이라 mainnet인 order_base_url로
        # 보내면 -2015로 거부된다(테스트넷 키 ≠ 메인넷 키).
        order_api_key = settings.api_key
        order_api_secret = settings.api_secret
    else:
        order_api_key = settings.order_api_key or settings.api_key
        order_api_secret = settings.order_api_secret or settings.api_secret
    return BinanceFuturesRestClient(
        settings.order_base_url,
        order_api_key,
        order_api_secret,
        settings.mode,
        AuditLog(default_audit_log_path("orders", for_date=decision_time)),
        recv_window_ms=settings.recv_window_ms,
    )

# wiring: from src.live.signal import latest_target_weights, assert_signal_available, assert_signal_fresh
