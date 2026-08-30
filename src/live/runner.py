# ruff: noqa
"""일일 섬도우 사이클 오케스트레이션.

어떤 게이트든 위반하면 주문을 하나도 생성하지 않고 HALT를 반환한다(부분 집행 금지).
I-LEDGER-DURABLE: 집행 구간은 try/finally 로 감싸 어떤 예외 경로에서도 이미 확인된
체결은 원장에 영속된 후 재전파된다.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from src.live.account import (
    AccountSnapshot,
    assert_drawdown_within_limit,
    assert_suppressed_venue_flat,
    assert_venue_configuration,
    effective_positions,
    fetch_account_snapshot,
    reconcile_or_halt,
    resolve_sizing_equity,
)
from src.live.audit import AuditLog, default_audit_log_path
from src.live.errors import LiveTradingError, RiskGateBreach
from src.live.execution_quality import (
    append_execution_quality,
    build_execution_quality_records,
    default_execution_quality_dir,
)
from src.live.executor import (
    ExecutionOutcome,
    PassiveExecutionPolicy,
    cancel_orphan_orders,
    execute_intents,
)
from src.live.fills import FillEvent, append_fills, default_fills_dir
from src.live.filters import parse_exchange_filters
from src.live.ledger import (
    LedgerState,
    apply_orphan_settlements,
    apply_outcomes,
    compute_fill_cash_flow,
    default_ledger_path,
    load_ledger,
    save_ledger,
)
from src.live.lifecycle import ShutdownFlag
from src.live.microstructure import (
    append_microstructure,
    build_microstructure_records,
    default_microstructure_dir,
    fetch_book_quotes,
)
from src.live.planner import OrderIntent, plan_orders
from src.live.portfolio_state import (
    PortfolioStateRecord,
    append_portfolio_state,
    default_portfolio_state_dir,
    resolve_effective_equity,
)
from src.live.rest import BinanceFuturesRestClient, parse_rate_limits
from src.live.settings import LiveSettings
from src.live.signal import assert_signal_available, assert_signal_fresh, latest_decision_marks, latest_target_weights
from src.live.sizing import target_quantities
from src.live.tax_ledger import (
    append_tax_records,
    collect_tax_records,
    default_tax_ledger_dir,
    simulated_tax_records,
)

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
    # minNotional 등으로 드롭된 목표 노셔널 비중(S5): 자본 캡($2,000) 하에서의
    # 페널티를 매 사이클 로그에서 보이게 한다.
    dropped_notional_fraction: float = 0.0


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
    shutdown: ShutdownFlag | None = None,
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
        settlements = cancel_orphan_orders(order_client, run_id, audit)
        if settlements:
            updated = apply_orphan_settlements(ledger_state.positions, settlements)
            ledger_state = LedgerState(
                positions=updated,
                equity_high_water_mark=ledger_state.equity_high_water_mark,
                cash_usdt=ledger_state.cash_usdt,
            )
            save_ledger(ledger_path, ledger_state)
            ledger_positions = ledger_state.positions
        if settings.mode.suppresses_mutations:
            assert_suppressed_venue_flat(snapshot)
        else:
            reconcile_or_halt(snapshot, ledger_positions, qty_tolerance_fraction=_RECONCILE_TOLERANCE_FRACTION)

        weights = latest_target_weights(artifact_path, decision_time, artifact_key=settings.artifact_key)
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
        # decision marks for sizing anchor separation
        try:
            _decision_marks_series = latest_decision_marks(artifact_path, decision_time, artifact_key=settings.artifact_key)
        except Exception:
            _decision_marks_series = None
        if _decision_marks_series is not None:
            decision_marks: dict[str, Decimal] | None = {str(k): Decimal(str(v)) for k, v in _decision_marks_series.items() if pd.notna(v)}
            _sizing_anchor = "decision_mark"
        else:
            decision_marks = None
            _sizing_anchor = "book_mid"
        # 4) I-EQUITY-MTM / I-DD-HALT.
        equity = resolve_sizing_equity(snapshot, Decimal(str(settings.notional_equity_usdt)), mode=settings.mode, cash_usdt=ledger_state.cash_usdt, positions=ledger_positions, marks=marks)
        assert_drawdown_within_limit(equity, ledger_state.equity_high_water_mark, settings.equity_drawdown_halt)

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

        # 사이클 수준 리스크 게이트를 먼저 검사한다(부분 집행 금지).
        check_risk_gates(intents, targets, marks, snapshot, settings, equity)

        # 종목별 노셔널 상한: 초과 심볼만 드롭한다(비중 재분배 없음).
        kept: list[OrderIntent] = []
        for intent in intents:
            if intent.reduce_only:
                kept.append(intent)
                continue
            mark = marks.get(intent.symbol)
            if mark is not None and abs(intent.quantity * mark) / equity > Decimal(
                str(settings.max_symbol_notional_fraction)
            ):
                audit.record("intent_dropped_symbol_cap", symbol=intent.symbol)
                continue
            kept.append(intent)

        for sym, reason in _uncovered_positions(current_positions, targets, filters, marks, kept):
            with contextlib.suppress(Exception):
                audit.record("position_uncovered", symbol=sym, reason=reason)

        from src.live.executor import FeeSchedule  # noqa: PLC0415

        policy = PassiveExecutionPolicy(fee_schedule=FeeSchedule(maker_fee_bps=settings.maker_fee_bps, taker_fee_bps=settings.taker_fee_bps))
        sink: list[ExecutionOutcome] = []
        persisted = False
        outcomes: list[ExecutionOutcome] = []
        final_state: LedgerState | None = None
        try:
            try:
                outcomes = list(execute_intents(order_client, kept, filters, policy, audit, _clock, time.sleep, rate_limits=rate_limits, outcome_sink=sink, shutdown=shutdown))
                final_state = _persist_confirmed_fills(
                    ledger_path,
                    ledger_state,
                    kept,
                    outcomes,
                    equity,
                    track_cash=settings.mode.suppresses_mutations,
                    starting_capital=Decimal(str(settings.notional_equity_usdt)),
                )
                persisted = True
            except LiveTradingError as exc:
                to_persist = sink if sink else _partial_outcomes(exc)
                if to_persist:
                    final_state = _persist_confirmed_fills(ledger_path, ledger_state, kept, to_persist, equity, track_cash=settings.mode.suppresses_mutations, starting_capital=Decimal(str(settings.notional_equity_usdt)))
                    persisted = True
                    if not outcomes:
                        outcomes = list(to_persist)
                raise
            finally:
                if not persisted and sink:
                    final_state = _persist_confirmed_fills(ledger_path, ledger_state, kept, sink, equity, track_cash=settings.mode.suppresses_mutations, starting_capital=Decimal(str(settings.notional_equity_usdt)))
                    persisted = True
                    if not outcomes:
                        outcomes = list(sink)
        except LiveTradingError:
            raise
        except BaseException:
            if not persisted and sink:
                final_state = _persist_confirmed_fills(ledger_path, ledger_state, kept, sink, equity, track_cash=settings.mode.suppresses_mutations, starting_capital=Decimal(str(settings.notional_equity_usdt)))
                persisted = True
                if not outcomes:
                    outcomes = list(sink)
            raise
        if final_state is None:
            # No intents and no exception, still need to persist hwm
            final_state = _persist_confirmed_fills(
                ledger_path,
                ledger_state,
                kept,
                [],
                equity,
                track_cash=settings.mode.suppresses_mutations,
                starting_capital=Decimal(str(settings.notional_equity_usdt)),
            )
            persisted = True
        # Ensure outcomes and final_state are defined for success path
        if persisted:
            execution_quality_dir = Path(settings.execution_quality_dir) if settings.execution_quality_dir else default_execution_quality_dir()
            try:
                records = build_execution_quality_records(decision_time, settings.mode.value, weights, marks, kept, outcomes)
                append_execution_quality(records, execution_quality_dir)
            except Exception as exc:  # noqa: BLE001 - observability-only, never halts cycle
                with contextlib.suppress(Exception):
                    audit.record("execution_quality_write_failed", error=str(exc))
                logger.warning("[SYS] execution_quality write failed error=%s", exc)
            try:
                fills_dir = Path(settings.fills_dir) if settings.fills_dir else default_fills_dir()
                fill_events: list[FillEvent] = []
                for intent, outcome in zip(kept, outcomes, strict=False):
                    for qty_abs, price, fee_bps, reason, liquidity in getattr(outcome, "fills", ()):
                        qty = Decimal(qty_abs)
                        signed_qty = qty if intent.side == "BUY" else -qty
                        dm = decision_marks.get(intent.symbol) if decision_marks is not None else None
                        fill_events.append(
                            FillEvent(
                                decision_time=decision_time,
                                timestamp=decision_time,
                                symbol=intent.symbol,
                                quantity_delta=signed_qty,
                                fill_price=Decimal(price),
                                fee_bps=float(fee_bps),
                                reason=str(reason),
                                pre_trade_equity=equity,
                                liquidity=str(liquidity),
                                mode=settings.mode.value,
                                run_id=run_id,
                                leg_index=int(intent.leg_index),
                                client_order_id=str(intent.client_order_prefix),
                                decision_mark=dm,
                                sizing_anchor=_sizing_anchor,
                            )
                        )
                append_fills(fill_events, fills_dir)
            except Exception as exc:  # noqa: BLE001
                with contextlib.suppress(Exception):
                    audit.record("fills_write_failed", error=str(exc))
                logger.warning("[SYS] fills write failed error=%s", exc)
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
            # tax ledger — fail-soft, never halts cycle (uses append_tax_records(tax_records, tax_dir))
            try:
                tax_dir = Path(settings.tax_ledger_dir) if settings.tax_ledger_dir else default_tax_ledger_dir()
                tax_dir.mkdir(parents=True, exist_ok=True)
                tax_records: tuple[Any, ...] = ()
                if settings.mode.suppresses_mutations:
                    # SHADOW/PAPER: derive from fills
                    fill_events_for_tax: list[Any] = []
                    for intent, outcome in zip(kept, outcomes, strict=False):
                        for qty_abs, price, fee_bps, reason, liquidity in getattr(outcome, "fills", ()):
                            qty = Decimal(qty_abs)
                            signed_qty = qty if intent.side == "BUY" else -qty
                            dm = decision_marks.get(intent.symbol) if decision_marks is not None else None
                            fill_events_for_tax.append(
                                FillEvent(
                                    decision_time=decision_time,
                                    timestamp=decision_time,
                                    symbol=intent.symbol,
                                    quantity_delta=signed_qty,
                                    fill_price=Decimal(price),
                                    fee_bps=float(fee_bps),
                                    reason=str(reason),
                                    pre_trade_equity=equity,
                                    liquidity=str(liquidity),
                                    mode=settings.mode.value,
                                    run_id=run_id,
                                    leg_index=int(intent.leg_index),
                                    client_order_id=str(intent.client_order_prefix),
                                    decision_mark=dm,
                                    sizing_anchor=_sizing_anchor,
                                )
                            )
                    # also capture any fallback fills from non-fills outcomes
                    if not fill_events_for_tax:
                        # try to use previously built fill_events if available via outer scope
                        try:
                            _maybe = globals().get("fill_events", [])
                            if isinstance(_maybe, list):
                                fill_events_for_tax = _maybe
                        except Exception:
                            pass
                    tax_records = simulated_tax_records(fill_events_for_tax, settings.mode.value)
                else:
                    if settings.tax_collection_enabled:
                        try:
                            wm_path = tax_dir / "watermark.json"
                            if wm_path.exists():
                                import json as _json

                                raw = _json.loads(wm_path.read_text(encoding="utf-8"))
                                from src.live.tax_ledger import TaxWatermark as _TW

                                watermark = _TW(
                                    last_trade_id={k: int(v) for k, v in raw.get("last_trade_id", {}).items()},
                                    last_income_id=int(raw.get("last_income_id", 0)),
                                    last_collected_at=pd.Timestamp(raw["last_collected_at"]) if raw.get("last_collected_at") else None,
                                )
                            else:
                                from src.live.tax_ledger import TaxWatermark as _TW

                                watermark = _TW(last_trade_id={}, last_income_id=0, last_collected_at=None)
                            collected, new_wm = collect_tax_records(order_client, wanted_symbols, watermark, settings.mode.value, now=decision_time)
                            tax_records = collected
                            try:
                                import json as _json2

                                wm_path.write_text(
                                    _json2.dumps(
                                        {
                                            "last_trade_id": new_wm.last_trade_id,
                                            "last_income_id": new_wm.last_income_id,
                                            "last_collected_at": new_wm.last_collected_at.isoformat() if new_wm.last_collected_at is not None else None,
                                        }
                                    ),
                                    encoding="utf-8",
                                )
                            except Exception:
                                pass
                        except Exception:
                            tax_records = ()
                    else:
                        tax_records = ()
                if tax_records:
                    append_tax_records(tax_records, tax_dir)
            except Exception as exc:  # noqa: BLE001
                with contextlib.suppress(Exception):
                    audit.record("tax_ledger_write_failed", error=str(exc))
                logger.warning("[SYS] tax_ledger write failed error=%s", exc)

            report = CycleReport(
                status="COMPLETE",
                reason=None,
                decision_time=decision_time,
                intent_count=len(outcomes),
                outcomes=tuple(outcomes),
                dropped_notional_fraction=dropped_fraction,
            )
            audit.record("cycle_complete", intents=len(outcomes))
            return report
        else:
            # If we reached here via finally without persisted, exception is propagating - will be caught outer
            # For case where no exception but sink path, we still need to return
            if outcomes:
                execution_quality_dir = Path(settings.execution_quality_dir) if settings.execution_quality_dir else default_execution_quality_dir()
                try:
                    records = build_execution_quality_records(decision_time, settings.mode.value, weights, marks, kept, outcomes)
                    append_execution_quality(records, execution_quality_dir)
                except Exception as exc:  # noqa: BLE001
                    with contextlib.suppress(Exception):
                        audit.record("execution_quality_write_failed", error=str(exc))
                    logger.warning("[SYS] execution_quality write failed error=%s", exc)
                report = CycleReport(
                    status="COMPLETE",
                    reason=None,
                    decision_time=decision_time,
                    intent_count=len(outcomes),
                    outcomes=tuple(outcomes),
                    dropped_notional_fraction=dropped_fraction,
                )
                audit.record("cycle_complete", intents=len(outcomes))
                return report
            raise RuntimeError("unreachable")
    except (LiveTradingError, ValueError, OSError) as exc:
        logger.error("[SYS] shadow cycle halted reason=%s", exc)
        return CycleReport(
            status="HALT",
            reason=str(exc),
            decision_time=decision_time,
            intent_count=0,
        )


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


def _persist_confirmed_fills(
    ledger_path: Path,
    base_state: LedgerState,
    intents: Sequence[OrderIntent],
    outcomes: Sequence[ExecutionOutcome],
    equity: Decimal,
    *,
    track_cash: bool = False,
    starting_capital: Decimal = Decimal(0),
) -> LedgerState:
    """확인된 체결과 단조 증가한 hwm 을 원자적으로 영속한다."""
    paired_intents = list(intents)[: len(outcomes)]
    if track_cash:
        cash_before = base_state.cash_usdt if base_state.cash_usdt is not None else starting_capital
        cash_usdt: Decimal | None = cash_before + compute_fill_cash_flow(paired_intents, outcomes)
    else:
        cash_usdt = None
    if not outcomes:
        # 체결이 없어도 hwm 은 ratchet 한다.
        state = LedgerState(
            positions=dict(base_state.positions),
            equity_high_water_mark=max(base_state.equity_high_water_mark, equity),
            cash_usdt=cash_usdt,
        )
        save_ledger(ledger_path, state)
        return state
    updated_positions = apply_outcomes(base_state.positions, paired_intents, outcomes)
    state = LedgerState(
        positions=updated_positions,
        equity_high_water_mark=max(base_state.equity_high_water_mark, equity),
        cash_usdt=cash_usdt,
    )
    save_ledger(ledger_path, state)
    return state


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


def _order_client(settings: LiveSettings, decision_time: pd.Timestamp) -> BinanceFuturesRestClient:
    return BinanceFuturesRestClient(
        settings.order_base_url,
        settings.order_api_key or settings.api_key,
        settings.order_api_secret or settings.api_secret,
        settings.mode,
        AuditLog(default_audit_log_path("orders", for_date=decision_time)),
        recv_window_ms=settings.recv_window_ms,
    )
