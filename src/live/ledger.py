"""내부 포지션 원장: 거래소 스냅샷과 별개로 우리가 의도한 체결을 누적 추적한다.

I-RECONCILE-FIRST가 대조하는 '내부 원장'의 유일한 소스. SHADOW에서는 실제 체결이
전송되지 않으므로 이 원장은 항상 0으로 남아, 거래소 스냅샷(역시 0)과 자명하게
일치한다. LIVE_TESTNET에서 실제 체결이 발생해야 원장이 갱신된다.
I-DD-HALT: equity_high_water_mark 는 여기에 단조 증가로 영속된다.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.paths import DATA_DIR
from src.live.executor import ExecutionOutcome, OrphanSettlement
from src.live.planner import OrderIntent

_HWM_KEY = "equity_high_water_mark"
_POSITIONS_KEY = "positions"
_CASH_KEY = "cash_usdt"
_FUNDING_THROUGH_KEY = "funding_accrued_through"
_LAST_EXECUTED_KEY = "last_executed_decision_time"
_WATERMARKS_KEY = "funding_watermarks"
_HISTORY_KEY = "position_history"
_ACCRUAL_STARTED_KEY = "funding_accrual_started_at"
_BACKFILLED_KEY = "funding_backfilled_through"

POSITION_HISTORY_MAX: int = 4


def default_ledger_path() -> Path:
    return DATA_DIR / "state" / "live_position_ledger.json"


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    effective_from: pd.Timestamp
    positions: dict[str, Decimal]


@dataclass(frozen=True, slots=True)
class FundingEvent:
    """One accrued paper funding settlement: amount = -(rate * quantity * price), cash sign convention."""

    symbol: str
    epoch: pd.Timestamp  # tz-aware UTC funding time
    rate: Decimal
    quantity: Decimal  # signed position held at epoch
    price: Decimal
    amount: Decimal
    price_source: str = "trade_close_1h"


@dataclass(frozen=True, slots=True)
class FundingAccrual:
    cash_delta: Decimal
    watermarks: dict[str, pd.Timestamp]
    lag_by_symbol: dict[str, pd.Timedelta]
    interval_by_symbol: dict[str, pd.Timedelta]
    events: tuple[FundingEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class LedgerState:
    """원장 상태: 포지션 맵 + 에쿼티 고수위선(단조 증가)."""

    positions: dict[str, Decimal] = field(default_factory=dict)
    equity_high_water_mark: Decimal = Decimal(0)
    cash_usdt: Decimal | None = None
    funding_accrued_through: pd.Timestamp | None = None
    last_executed_decision_time: pd.Timestamp | None = None
    funding_watermarks: dict[str, pd.Timestamp] = field(default_factory=dict)
    position_history: tuple[PositionSnapshot, ...] = ()
    funding_accrual_started_at: pd.Timestamp | None = None
    funding_backfilled_through: pd.Timestamp | None = None


def _parse_utc(value: Any, name: str, path: Path) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(str(value))
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f"ledger {name} is not a timestamp: {path}") from exc
    if parsed.tzinfo is None:
        raise DataIntegrityError(f"ledger {name} must be tz-aware: {path}")
    return parsed.tz_convert("UTC")


def _parse_positions(raw: Any, name: str, path: Path) -> dict[str, Decimal]:
    if not isinstance(raw, dict):
        raise DataIntegrityError(f"ledger {name} must be an object: {path}")
    try:
        return {str(symbol): Decimal(str(qty)) for symbol, qty in raw.items()}
    except (InvalidOperation, ValueError, TypeError, AttributeError) as exc:
        raise DataIntegrityError(f"ledger {name} quantity is not numeric: {path}") from exc


def load_ledger(path: Path) -> LedgerState:
    """path가 없으면 빈 원장을 반환한다. 레거시 평면 dict 는 hwm=0 으로 승격한다."""
    if not path.exists():
        return LedgerState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataIntegrityError(f"ledger file corrupt: {path}") from exc
    if not isinstance(raw, dict):
        raise DataIntegrityError(f"ledger file must be a JSON object: {path}")
    if _POSITIONS_KEY not in raw:
        # 레거시 평면 레이아웃({symbol: qty}) 하위 호환.
        positions_raw = raw
        hwm = Decimal(0)
        cash_usdt: Decimal | None = None
        funding_accrued_through: pd.Timestamp | None = None
        last_executed: pd.Timestamp | None = None
        funding_accrual_started_at: pd.Timestamp | None = None
        funding_backfilled_through: pd.Timestamp | None = None
    else:
        positions_raw = raw[_POSITIONS_KEY]
        hwm_raw = raw.get(_HWM_KEY, "0")
        try:
            hwm = Decimal(str(hwm_raw))
        except (InvalidOperation, ValueError, TypeError, AttributeError) as exc:
            raise DataIntegrityError(f"ledger hwm is not numeric: {path}") from exc
        if _CASH_KEY in raw:
            cash_raw = raw[_CASH_KEY]
            try:
                cash_usdt = Decimal(str(cash_raw))
            except (InvalidOperation, ValueError, TypeError, AttributeError) as exc:
                raise DataIntegrityError(f"ledger cash_usdt is not numeric: {path}") from exc
        else:
            cash_usdt = None
        if _FUNDING_THROUGH_KEY in raw and raw[_FUNDING_THROUGH_KEY] is not None:
            try:
                parsed = pd.Timestamp(str(raw[_FUNDING_THROUGH_KEY]))
            except (ValueError, TypeError) as exc:
                raise DataIntegrityError(f"ledger funding_accrued_through is not a timestamp: {path}") from exc
            parsed = parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
            funding_accrued_through = parsed
        else:
            funding_accrued_through = None
        if _LAST_EXECUTED_KEY in raw and raw[_LAST_EXECUTED_KEY] is not None:
            try:
                parsed_exec = pd.Timestamp(str(raw[_LAST_EXECUTED_KEY]))
            except (ValueError, TypeError) as exc:
                raise DataIntegrityError(f"ledger last_executed_decision_time is not a timestamp: {path}") from exc
            if parsed_exec.tzinfo is None:
                raise DataIntegrityError(f"ledger last_executed_decision_time must be tz-aware: {path}")
            last_executed = parsed_exec.tz_convert("UTC")
        else:
            last_executed = None
        if raw.get(_ACCRUAL_STARTED_KEY) is not None:
            funding_accrual_started_at = _parse_utc(raw.get(_ACCRUAL_STARTED_KEY), "funding_accrual_started_at", path)
        else:
            funding_accrual_started_at = None
        if raw.get(_BACKFILLED_KEY) is not None:
            funding_backfilled_through = _parse_utc(raw.get(_BACKFILLED_KEY), "funding_backfilled_through", path)
        else:
            funding_backfilled_through = None
    if not isinstance(positions_raw, dict):
        raise DataIntegrityError(f"ledger positions must be an object: {path}")
    try:
        positions = {str(symbol): Decimal(str(qty)) for symbol, qty in positions_raw.items()}
    except (InvalidOperation, ValueError, TypeError, AttributeError) as exc:
        raise DataIntegrityError(f"ledger position quantity is not numeric: {path}") from exc
    if _POSITIONS_KEY not in raw:
        watermarks: dict[str, pd.Timestamp] = {}
        history: tuple[PositionSnapshot, ...] = ()
    else:
        watermarks_raw = raw.get(_WATERMARKS_KEY, {})
        if not isinstance(watermarks_raw, dict):
            raise DataIntegrityError(f"ledger funding_watermarks must be an object: {path}")
        watermarks = {
            str(symbol): _parse_utc(value, "funding_watermarks", path)
            for symbol, value in watermarks_raw.items()
        }
        history_raw = raw.get(_HISTORY_KEY, [])
        if not isinstance(history_raw, list):
            raise DataIntegrityError(f"ledger position_history must be a list: {path}")
        parsed_history: list[PositionSnapshot] = []
        for entry in history_raw:
            if (
                not isinstance(entry, dict)
                or "effective_from" not in entry
                or "positions" not in entry
            ):
                raise DataIntegrityError(f"ledger position_history entry malformed: {path}")
            parsed_history.append(
                PositionSnapshot(
                    effective_from=_parse_utc(entry["effective_from"], "position_history", path),
                    positions=_parse_positions(entry["positions"], "position_history", path),
                )
            )
        parsed_history.sort(key=lambda snap: snap.effective_from)
        history = tuple(parsed_history)
    return LedgerState(
        positions=positions,
        equity_high_water_mark=hwm,
        cash_usdt=cash_usdt,
        funding_accrued_through=funding_accrued_through,
        last_executed_decision_time=last_executed,
        funding_watermarks=watermarks,
        position_history=history,
        funding_accrual_started_at=funding_accrual_started_at,
        funding_backfilled_through=funding_backfilled_through,
    )


def save_ledger(path: Path, state: LedgerState) -> None:
    """임시파일 + os.replace 로 원자적 기록한다(부분 기록 JSON 은 영구 HALT 로 이어진다)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        _POSITIONS_KEY: {symbol: str(qty) for symbol, qty in state.positions.items() if qty != 0},
        _HWM_KEY: str(state.equity_high_water_mark),
    }
    if state.cash_usdt is not None:
        payload[_CASH_KEY] = str(state.cash_usdt)
    if state.funding_accrued_through is not None:
        ts = pd.Timestamp(state.funding_accrued_through)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        payload[_FUNDING_THROUGH_KEY] = ts.isoformat()
    if state.last_executed_decision_time is not None:
        payload[_LAST_EXECUTED_KEY] = pd.Timestamp(state.last_executed_decision_time).tz_convert("UTC").isoformat()
    if state.funding_watermarks:
        payload[_WATERMARKS_KEY] = {
            symbol: pd.Timestamp(ts).tz_convert("UTC").isoformat()
            for symbol, ts in sorted(state.funding_watermarks.items())
        }
    if state.position_history:
        payload[_HISTORY_KEY] = [
            {
                "effective_from": pd.Timestamp(snap.effective_from).tz_convert("UTC").isoformat(),
                "positions": {symbol: str(qty) for symbol, qty in sorted(snap.positions.items())},
            }
            for snap in state.position_history
        ]
    if state.funding_accrual_started_at is not None:
        payload[_ACCRUAL_STARTED_KEY] = pd.Timestamp(state.funding_accrual_started_at).tz_convert("UTC").isoformat()
    if state.funding_backfilled_through is not None:
        payload[_BACKFILLED_KEY] = pd.Timestamp(state.funding_backfilled_through).tz_convert("UTC").isoformat()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, path)


def compute_fill_cash_flow(
    intents: Sequence[OrderIntent],
    outcomes: Sequence[ExecutionOutcome],
) -> Decimal:
    """체결 현금흐름을 계산한다. BUY는 음수, SELL은 양수."""
    total = Decimal(0)
    for intent, outcome in zip(intents, outcomes, strict=True):
        # I-FEE-ACCOUNTED: fills가 있으면 per-fill 수수료 포함, 없으면 보수적 taker fallback
        fills = getattr(outcome, "fills", ())
        if fills:
            for qty_abs, price, fee_bps, _reason, _liq, _filled_at in fills:
                qty = Decimal(qty_abs)
                px = Decimal(price)
                fee = abs(qty * px) * Decimal(str(fee_bps)) / Decimal(10_000)
                signed = qty if intent.side == "BUY" else -qty
                total += -signed * px - fee
            continue
        if outcome.filled_qty <= 0 or outcome.avg_fill_price is None:
            continue
        signed = outcome.filled_qty if intent.side == "BUY" else -outcome.filled_qty
        # 구 스키마 fallback: 수수료는 taker로 보수적 부과
        fee_bps = 5.0
        try:
            from src.mhs.types import ExecutionSpec  # noqa: PLC0415

            fee_bps = ExecutionSpec().taker_fee_bps
        except Exception:  # noqa: BLE001, S110
            pass
        fee = abs(outcome.filled_qty * outcome.avg_fill_price) * Decimal(str(fee_bps)) / Decimal(10_000)
        total += -signed * outcome.avg_fill_price - fee
    return total


def apply_outcomes(
    positions: Mapping[str, Decimal],
    intents: Sequence[OrderIntent],
    outcomes: Sequence[ExecutionOutcome],
) -> dict[str, Decimal]:
    """체결된 수량만큼 원장을 갱신한다. 방향은 intent.side에서, 크기는 outcome.filled_qty에서 온다."""
    if len(intents) != len(outcomes):
        raise ValueError("intents and outcomes must be the same length and order")
    updated = dict(positions)
    for intent, outcome in zip(intents, outcomes, strict=True):
        if outcome.symbol != intent.symbol:
            raise ValueError(f"outcome/intent symbol mismatch: {outcome.symbol} != {intent.symbol}")
        signed_fill = outcome.filled_qty if intent.side == "BUY" else -outcome.filled_qty
        updated[intent.symbol] = updated.get(intent.symbol, Decimal(0)) + signed_fill
    return updated


def apply_orphan_settlements(
    positions: Mapping[str, Decimal], settlements: Sequence[OrphanSettlement]
) -> dict[str, Decimal]:
    updated = dict(positions)
    for s in settlements:
        signed = s.executed_qty if s.side == "BUY" else -s.executed_qty
        updated[s.symbol] = updated.get(s.symbol, Decimal(0)) + signed
    return updated


def _require_tz_aware(value: pd.Timestamp, name: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise ValueError(f"{name} must be tz-aware")
    return ts


def append_position_snapshot(
    history: Sequence[PositionSnapshot],
    effective_from: pd.Timestamp,
    positions: Mapping[str, Decimal],
) -> tuple[PositionSnapshot, ...]:
    ts = _require_tz_aware(effective_from, "effective_from").tz_convert("UTC")
    nonzero = {symbol: qty for symbol, qty in positions.items() if qty != 0}
    if history and history[-1].positions == nonzero:
        return tuple(history)
    return (*history, PositionSnapshot(effective_from=ts, positions=nonzero))[
        -POSITION_HISTORY_MAX:
    ]


def position_at(
    history: Sequence[PositionSnapshot], symbol: str, at: pd.Timestamp
) -> Decimal:
    latest: PositionSnapshot | None = None
    for snap in history:
        if snap.effective_from >= at:
            break
        latest = snap
    if latest is None:
        raise DataIntegrityError(
            f"paper funding epoch {at.isoformat()} predates position history for {symbol}"
        )
    return latest.positions.get(symbol, Decimal(0))


def _held_since(history: Sequence[PositionSnapshot], symbol: str) -> pd.Timestamp | None:
    if not history or history[-1].positions.get(symbol, Decimal(0)) == 0:
        return None
    start = history[-1].effective_from
    for snap in reversed(history[:-1]):
        if snap.positions.get(symbol, Decimal(0)) == 0:
            break
        start = snap.effective_from
    return start


def _released_at(history: Sequence[PositionSnapshot], symbol: str) -> pd.Timestamp | None:
    last_held: int | None = None
    for idx, snap in enumerate(history):
        if snap.positions.get(symbol, Decimal(0)) != 0:
            last_held = idx
    if last_held is None or last_held == len(history) - 1:
        return None
    return history[last_held + 1].effective_from


def _funding_interval(series: pd.Series | None) -> pd.Timedelta:
    if series is None or len(series) < 2:
        return pd.Timedelta(hours=8)
    idx = series.sort_index().index[-4:]
    diffs = pd.Series(idx).diff().dropna()
    median = diffs.median()
    return pd.Timedelta(hours=max(1, round(median / pd.Timedelta(hours=1))))


def accrue_funding_by_watermark(
    history: Sequence[PositionSnapshot],
    watermarks: Mapping[str, pd.Timestamp],
    funding_by_symbol: Mapping[str, pd.Series],
    trade_close_by_symbol: Mapping[str, pd.Series],
    now: pd.Timestamp,
    closed_at: Mapping[str, pd.Timestamp] | None = None,
) -> FundingAccrual:
    """Accrue observed funding rates against held paper units using completed trade-price notional estimates. The result is a paper cash estimate, with explicit unresolved status when price or funding evidence is missing. Each accrued settlement is also returned as a FundingEvent so the cash change is reconstructable from records."""
    now_ts = _require_tz_aware(now, "now").tz_convert("UTC")
    closed = dict(closed_at or {})
    current = history[-1].positions if history else {}
    total = Decimal(0)
    updated: dict[str, pd.Timestamp] = {}
    lags: dict[str, pd.Timedelta] = {}
    intervals: dict[str, pd.Timedelta] = {}
    events: list[FundingEvent] = []
    for symbol in sorted(set(watermarks) | set(current)):
        watermark = watermarks.get(symbol) or _held_since(history, symbol)
        if watermark is None:
            continue
        upper = min(now_ts, closed[symbol]) if symbol in closed else now_ts
        series = funding_by_symbol.get(symbol)
        trade_closes = trade_close_by_symbol.get(symbol)
        if series is not None:
            window = series[(series.index > watermark) & (series.index <= upper)].sort_index()
            for epoch, rate in window.items():
                qty = position_at(history, symbol, epoch)
                if qty != 0:
                    try:
                        rate_f = float(rate)
                    except (TypeError, ValueError):
                        break
                    if not math.isfinite(rate_f):
                        break
                    bar = epoch.floor("h")
                    if trade_closes is None or bar not in trade_closes.index:
                        break
                    try:
                        px_f = float(trade_closes.loc[bar])
                    except (TypeError, ValueError, KeyError):
                        break
                    if not math.isfinite(px_f) or px_f <= 0:
                        break
                    amount = -(Decimal(str(rate)) * qty * Decimal(str(px_f)))
                    total += amount
                    events.append(
                        FundingEvent(
                            symbol=symbol,
                            epoch=pd.Timestamp(epoch).tz_convert("UTC"),
                            rate=Decimal(str(rate)),
                            quantity=qty,
                            price=Decimal(str(px_f)),
                            amount=amount,
                            price_source="trade_close_1h",
                        )
                    )
                watermark = epoch
        intervals[symbol] = _funding_interval(series)
        if symbol in current and symbol not in closed:
            updated[symbol] = watermark
            lags[symbol] = now_ts - watermark
        else:
            released = _released_at(history, symbol)
            if symbol in closed or (released is not None and watermark < released):
                updated[symbol] = watermark
                if symbol not in closed and released is not None and watermark < released:
                    lags[symbol] = now_ts - watermark
    return FundingAccrual(total, updated, lags, intervals, tuple(events))
