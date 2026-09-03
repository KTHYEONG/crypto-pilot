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
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.common.errors import DataIntegrityError
from src.common.paths import DATA_DIR
from src.live.executor import ExecutionOutcome, OrphanSettlement
from src.live.planner import OrderIntent

_HWM_KEY = "equity_high_water_mark"
_POSITIONS_KEY = "positions"
_CASH_KEY = "cash_usdt"


def default_ledger_path() -> Path:
    return DATA_DIR / "state" / "live_position_ledger.json"


@dataclass(frozen=True, slots=True)
class LedgerState:
    """원장 상태: 포지션 맵 + 에쿼티 고수위선(단조 증가)."""

    positions: dict[str, Decimal] = field(default_factory=dict)
    equity_high_water_mark: Decimal = Decimal(0)
    cash_usdt: Decimal | None = None


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
    else:
        positions_raw = raw[_POSITIONS_KEY]
        hwm_raw = raw.get(_HWM_KEY, "0")
        try:
            hwm = Decimal(str(hwm_raw))
        except Exception as exc:  # noqa: BLE001
            raise DataIntegrityError(f"ledger hwm is not numeric: {path}") from exc
        if _CASH_KEY in raw:
            cash_raw = raw[_CASH_KEY]
            try:
                cash_usdt = Decimal(str(cash_raw))
            except Exception as exc:  # noqa: BLE001
                raise DataIntegrityError(f"ledger cash_usdt is not numeric: {path}") from exc
        else:
            cash_usdt = None
    if not isinstance(positions_raw, dict):
        raise DataIntegrityError(f"ledger positions must be an object: {path}")
    try:
        positions = {str(symbol): Decimal(str(qty)) for symbol, qty in positions_raw.items()}
    except Exception as exc:  # noqa: BLE001
        raise DataIntegrityError(f"ledger position quantity is not numeric: {path}") from exc
    return LedgerState(positions=positions, equity_high_water_mark=hwm, cash_usdt=cash_usdt)


def save_ledger(path: Path, state: LedgerState) -> None:
    """임시파일 + os.replace 로 원자적 기록한다(부분 기록 JSON 은 영구 HALT 로 이어진다)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        _POSITIONS_KEY: {symbol: str(qty) for symbol, qty in state.positions.items() if qty != 0},
        _HWM_KEY: str(state.equity_high_water_mark),
    }
    if state.cash_usdt is not None:
        payload[_CASH_KEY] = str(state.cash_usdt)
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
