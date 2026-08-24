"""내부 포지션 원장: 거래소 스냅샷과 별개로 우리가 의도한 체결을 누적 추적한다.

I-RECONCILE-FIRST가 대조하는 '내부 원장'의 유일한 소스. SHADOW에서는 실제 체결이
전송되지 않으므로 이 원장은 항상 0으로 남아, 거래소 스냅샷(역시 0)과 자명하게
일치한다. LIVE_TESTNET에서 실제 체결이 발생해야 원장이 갱신된다.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path

from src.common.config import DATA_DIR
from src.common.errors import DataIntegrityError
from src.live.executor import ExecutionOutcome
from src.live.planner import OrderIntent


def default_ledger_path() -> Path:
    return DATA_DIR / "state" / "live_position_ledger.json"


def load_ledger(path: Path) -> dict[str, Decimal]:
    """path가 없으면 빈 원장(전량 미보유)을 반환한다."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataIntegrityError(f"ledger file corrupt: {path}") from exc
    return {str(symbol): Decimal(str(qty)) for symbol, qty in raw.items()}


def save_ledger(path: Path, positions: Mapping[str, Decimal]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {symbol: str(qty) for symbol, qty in positions.items() if qty != 0}
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


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
