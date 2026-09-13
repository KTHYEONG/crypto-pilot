"""내부 포지션 원장: 거래소 스냅샷과 별개로 우리가 의도한 체결을 누적 추적한다.

I-RECONCILE-FIRST가 대조하는 '내부 원장'의 유일한 소스. SHADOW에서는 실제 체결이
전송되지 않으므로 이 원장은 항상 0으로 남아, 거래소 스냅샷(역시 0)과 자명하게
일치한다. LIVE_TESTNET에서 실제 체결이 발생해야 원장이 갱신된다.
I-DD-HALT: equity_high_water_mark 는 여기에 단조 증가로 영속된다.
"""

from __future__ import annotations

import json
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


def default_ledger_path() -> Path:
    return DATA_DIR / "state" / "live_position_ledger.json"


@dataclass(frozen=True, slots=True)
class LedgerState:
    """원장 상태: 포지션 맵 + 에쿼티 고수위선(단조 증가)."""

    positions: dict[str, Decimal] = field(default_factory=dict)
    equity_high_water_mark: Decimal = Decimal(0)
    cash_usdt: Decimal | None = None
    funding_accrued_through: pd.Timestamp | None = None


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
    if not isinstance(positions_raw, dict):
        raise DataIntegrityError(f"ledger positions must be an object: {path}")
    try:
        positions = {str(symbol): Decimal(str(qty)) for symbol, qty in positions_raw.items()}
    except (InvalidOperation, ValueError, TypeError, AttributeError) as exc:
        raise DataIntegrityError(f"ledger position quantity is not numeric: {path}") from exc
    return LedgerState(
        positions=positions,
        equity_high_water_mark=hwm,
        cash_usdt=cash_usdt,
        funding_accrued_through=funding_accrued_through,
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
            for qty_abs, price, fee_bps, _reason, _liq in fills:
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


def accrue_paper_funding(
    positions: Mapping[str, Decimal],
    funding_by_symbol: Mapping[str, pd.Series],
    marks: Mapping[str, Decimal],
    start_exclusive: pd.Timestamp,
    end_inclusive: pd.Timestamp,
) -> Decimal:
    """페이퍼 펀딩비 현금 증분을 계산한다. 반환값은 cash delta(지급은 음수).

    start_exclusive < t <= end_inclusive 구간에 속한 settlement마다
    -rate * qty * mark 를 합산한다. 0이 아닌 모든 포지션은 펀딩 시리즈와
    mark를 반드시 가져야 하며, 없으면 fail-closed 한다.
    """
    start = _require_tz_aware(start_exclusive, "start_exclusive")
    end = _require_tz_aware(end_inclusive, "end_inclusive")
    if end < start:
        raise ValueError("end_inclusive must be >= start_exclusive: end precedes start")
    total = Decimal(0)
    for symbol, qty in positions.items():
        if qty == 0:
            continue
        if symbol not in funding_by_symbol:
            raise DataIntegrityError(f"paper funding series missing for {symbol}")
        if symbol not in marks:
            raise DataIntegrityError(f"paper funding mark missing for {symbol}")
        series = funding_by_symbol[symbol]
        mark = marks[symbol]
        window = series[(series.index > start) & (series.index <= end)]
        for rate in window:
            total += -(Decimal(str(rate)) * qty * mark)
    return total
