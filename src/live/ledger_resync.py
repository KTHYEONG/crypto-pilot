"""Operator-acknowledged adoption of the venue position snapshot into the ledger after an unexplained reconciliation breach. The only path that clears the de-risk-only flag."""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.live.account import (
    RECONCILE_QTY_TOLERANCE_FRACTION,
    AccountSnapshot,
    PositionBreach,
    fetch_account_snapshot,
    find_position_breaches,
    reconcile_or_halt,
    settled_delisting_symbols,
)
from src.live.audit import AuditLog, default_audit_log_path
from src.live.daemon_idle import assert_daemon_idle
from src.live.errors import LiveTradingError
from src.live.executor import _LEGACY_CLIENT_ORDER_ID, cancel_orphan_orders
from src.live.ledger import clear_derisk, default_ledger_path, load_ledger
from src.live.order_journal import OrderJournal, default_order_journal_path
from src.live.planner import CLIENT_ORDER_NAMESPACE
from src.live.recovery import recover_unresolved_orders
from src.live.scheduler import _resolve_heartbeat_path
from src.live.settings import LiveSettings

logger = logging.getLogger("LiveLedgerResync")


def _is_our_order(client_order_id: str) -> bool:
    return client_order_id.startswith(CLIENT_ORDER_NAMESPACE) or bool(_LEGACY_CLIENT_ORDER_ID.match(client_order_id))


def _breaches(
    snapshot: AccountSnapshot, exchange_info: Mapping[str, Any], positions: dict[str, Decimal], now: pd.Timestamp
) -> tuple[PositionBreach, ...]:
    # 정산 대기 중인 상폐 심볼은 재동기화로 지우면 정산 기록이 사라지므로 불일치에서 제외한다.
    settled = settled_delisting_symbols(exchange_info, snapshot.positions, positions, now=now)
    return find_position_breaches(
        snapshot, positions, qty_tolerance_fraction=RECONCILE_QTY_TOLERANCE_FRACTION, settled_symbols=settled
    )



@dataclass(frozen=True, slots=True)
class ResyncPlan:
    adjustments: tuple[PositionBreach, ...]  # per-symbol venue - ledger gaps to adopt
    backup_path: Path | None  # set when applied
    applied: bool


def _default_backup_dir(settings: LiveSettings) -> Path:
    if settings.ledger_resync_backup_dir:
        return Path(settings.ledger_resync_backup_dir)
    from src.common.paths import DATA_DIR

    return DATA_DIR / "state" / "ledger_backups"


def run_ledger_resync(
    settings: LiveSettings,
    *,
    apply: bool,
    now: pd.Timestamp,
) -> ResyncPlan:
    """Compute the venue-minus-ledger gap for every symbol and, with `apply=True`, journal each gap as an `operator_resync` fill, commit it, verify that reconciliation now passes and clear the de-risk flag.

    Steps are ordered so the ledger is never left half-adopted: reject unless the mode mutates the venue and the daemon is idle; sweep and recover our own orders so the ledger already contains every fill we can prove; back up the current ledger file; journal adjustments; commit; verify with `reconcile_or_halt`; clear the flag; audit `ledger_resynced`.

    Args: settings: live settings (must be a LIVE mode). apply: False computes and logs the plan only. now: tz-aware UTC wall clock.
    Returns: ResyncPlan.
    Raises: LiveTradingError: mode is PAPER/SHADOW, the daemon heartbeat shows a busy stage younger than the busy-stale threshold, foreign open orders exist (the operator must resolve them first), or post-commit verification still breaches.
    """
    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is None:
        raise ValueError("now must be tz-aware")
    now_ts = now_ts.tz_convert("UTC")
    if settings.mode.suppresses_mutations:
        raise LiveTradingError("ledger resync requires a LIVE mode")
    from src.live.runner import (
        _market_client,
        _marks_from_tickers,
        _order_client,
    )

    ledger_path = Path(settings.ledger_path) if settings.ledger_path else default_ledger_path()
    journal_path = (
        Path(settings.order_journal_path)
        if settings.order_journal_path
        else default_order_journal_path()
    )
    if apply:
        try:
            assert_daemon_idle(_resolve_heartbeat_path(settings), now_ts)
        except DataIntegrityError as exc:
            raise LiveTradingError(str(exc)) from exc

    audit = AuditLog(default_audit_log_path("ledger_resync", for_date=now_ts))
    try:
        market_client = _market_client(settings, now_ts)
        order_client = _order_client(settings, now_ts)
        order_client.sync_server_time()
        exchange_info = market_client.exchange_info()
        journal = OrderJournal(journal_path)
        run_id = now_ts.strftime("%Y%m%d")
        if not apply:
            snapshot = fetch_account_snapshot(order_client, now=now_ts)
            foreign = sorted(
                {
                    str(entry.get("symbol", ""))
                    for entry in order_client.open_orders()
                    if not _is_our_order(str(entry.get("clientOrderId", ""))) and str(entry.get("symbol", ""))
                }
            )
            ledger_state = load_ledger(ledger_path)
            breaches = _breaches(snapshot, exchange_info, dict(ledger_state.positions), now_ts)
            audit.record(
                "ledger_resync_planned",
                adjustments=len(breaches),
                symbols=sorted(b.symbol for b in breaches),
                foreign_symbols=foreign,
            )
            return ResyncPlan(adjustments=breaches, backup_path=None, applied=False)
        sweep = cancel_orphan_orders(
            order_client, run_id, audit, journal=journal, now=now_ts, taker_fee_bps=float(settings.taker_fee_bps)
        )
        if sweep.foreign_symbols:
            raise LiveTradingError(
                f"foreign open orders present on {','.join(sweep.foreign_symbols)}; "
                "operator must resolve them first"
            )
        recover_unresolved_orders(
            order_client,
            journal,
            audit,
            now=now_ts,
            lookback=pd.Timedelta(hours=settings.journal_recovery_lookback_hours),
            taker_fee_bps=float(settings.taker_fee_bps),
        )
        # 스냅샷은 고아 주문 정리·복구 이후에 떠야 그 사이 체결까지 반영된 거래소 상태와 비교된다.
        snapshot = fetch_account_snapshot(order_client, now=now_ts)
        ledger_state = load_ledger(ledger_path)
        from src.live.runner import _commit_and_record as _commit

        cycle_context = journal.begin_attempt(
            decision_time=now_ts,
            run_id=run_id,
            mode=settings.mode.value,
            pre_trade_equity=Decimal(0),
            sizing_anchor="ledger_resync",
            decision_marks={},
            started_at=now_ts,
        )
        ledger_state = _commit(
            settings,
            ledger_path,
            ledger_state,
            journal,
            audit,
            fallback_attempt=cycle_context,
            equity=None,
            executed_decision_time=None,
        )
        breaches = _breaches(snapshot, exchange_info, dict(ledger_state.positions), now_ts)
        if not breaches:
            ledger_state = clear_derisk(ledger_path, ledger_state)
            audit.record("ledger_resynced", adjustments=0)
            return ResyncPlan(adjustments=(), backup_path=None, applied=True)
        backup_dir = _default_backup_dir(settings)
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = now_ts.strftime("%Y%m%dT%H%M%SZ")
        backup_path = backup_dir / f"live_position_ledger.{stamp}.json"
        raw = ledger_path.read_bytes() if ledger_path.exists() else b"{}"
        tmp_backup = backup_path.with_suffix(backup_path.suffix + ".tmp")
        tmp_backup.write_bytes(raw)
        with tmp_backup.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_backup, backup_path)
        # 디렉터리 fsync 를 지원하지 않는 파일시스템에서도 백업 파일 자체는 이미 fsync 되었다.
        with contextlib.suppress(OSError):
            dir_fd = os.open(backup_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        marks = _marks_from_tickers(market_client, sorted({b.symbol for b in breaches}))
        unpriced = sorted(b.symbol for b in breaches if not (marks.get(b.symbol) or Decimal(0)) > 0)
        if unpriced:
            # 재동기화 체결의 가격은 라벨이지만 임의 값을 원장 증거로 남기지 않는다.
            raise LiveTradingError(f"no valid mark for resync symbols {','.join(unpriced)}; retry later")
        for breach in breaches:
            signed_gap = breach.gap
            side = "BUY" if signed_gap > 0 else "SELL"
            price = marks[breach.symbol]
            journal.record_fill(
                kind="operator_resync",
                attempt_seq=cycle_context.attempt_seq,
                symbol=breach.symbol,
                side=side,
                quantity=abs(signed_gap),
                price=price,
                fee_bps=0.0,
                liquidity="taker",
                reason="operator_resync",
                filled_at=now_ts,
                client_order_id=None,
                leg_index=0,
                cumulative_executed_qty=None,
                simulated=False,
            )
        ledger_state = _commit(
            settings,
            ledger_path,
            ledger_state,
            journal,
            audit,
            fallback_attempt=cycle_context,
            equity=None,
            executed_decision_time=None,
        )
        try:
            reconcile_or_halt(
                snapshot,
                ledger_state.positions,
                qty_tolerance_fraction=RECONCILE_QTY_TOLERANCE_FRACTION,
                settled_symbols=settled_delisting_symbols(
                    exchange_info, snapshot.positions, ledger_state.positions, now=now_ts
                ),
            )
        except LiveTradingError as exc:
            raise LiveTradingError(f"post-commit verification still breaches: {exc}") from exc
        ledger_state = clear_derisk(ledger_path, ledger_state)
        audit.record(
            "ledger_resynced",
            adjustments=len(breaches),
            symbols=sorted(b.symbol for b in breaches),
        )
        from src.live.runner import _notify_event

        _notify_event(
            settings,
            event="ledger_resynced",
            detail=f"adjustments={len(breaches)} " + ",".join(f"{b.symbol}={b.gap}" for b in breaches),
            decision_time=now_ts,
            now=now_ts,
        )
        return ResyncPlan(adjustments=breaches, backup_path=backup_path, applied=True)
    finally:
        with contextlib.suppress(Exception):
            audit.close()
