"""Account snapshot, venue configuration guard, and first reconciliation.

I-RECONCILE-FIRST: 불일치 시 자동 보정 없이 예외만 발생시킨다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.live.errors import ReconciliationBreach, RiskGateBreach
from src.live.settings import ExecutionMode


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """거래소 계좌/포지션의 불변 스냅샷. positions는 심볼 -> 부호 있는 순포지션."""

    taken_at: pd.Timestamp
    wallet_balance: Decimal
    available_balance: Decimal
    total_maint_margin: Decimal
    unrealized_pnl: Decimal
    positions: Mapping[str, Decimal]
    dual_side_position: bool
    multi_assets_margin: bool


def _required_number(payload: Mapping[str, Any], key: str) -> Decimal:
    if key not in payload:
        raise DataIntegrityError(f"account payload missing required key {key}")
    try:
        return Decimal(str(payload[key]))
    except Exception as exc:  # noqa: BLE001
        raise DataIntegrityError(f"account payload key {key} is not numeric") from exc


def fetch_account_snapshot(client: Any, *, now: pd.Timestamp) -> AccountSnapshot:
    """GET /fapi/v2/account 및 /fapi/v2/positionRisk로 스냅샷을 구성한다."""
    account = client.request("GET", "/fapi/v2/account", signed=True)
    position_risk = client.request("GET", "/fapi/v2/positionRisk", signed=True)
    if not isinstance(account, dict) or "totalWalletBalance" not in account:
        raise DataIntegrityError("account endpoint returned an unexpected schema")
    if not isinstance(position_risk, list):
        raise DataIntegrityError("positionRisk endpoint returned an unexpected schema")

    positions: dict[str, Decimal] = {}
    for entry in position_risk:
        if not isinstance(entry, dict) or "symbol" not in entry or "positionAmt" not in entry:
            raise DataIntegrityError("positionRisk row missing symbol/positionAmt")
        qty = Decimal(str(entry["positionAmt"]))
        if qty != 0:
            positions[str(entry["symbol"])] = qty

    dual_side_raw = account.get("dualSidePosition")
    multi_assets_raw = account.get("multiAssetsMargin")
    if dual_side_raw is None or multi_assets_raw is None:
        raise DataIntegrityError("account payload missing dualSidePosition/multiAssetsMargin")

    return AccountSnapshot(
        taken_at=now,
        wallet_balance=_required_number(account, "totalWalletBalance"),
        available_balance=_required_number(account, "availableBalance"),
        total_maint_margin=_required_number(account, "totalInitialMargin"),
        unrealized_pnl=_required_number(account, "totalUnrealizedProfit"),
        positions=positions,
        dual_side_position=str(dual_side_raw).lower() == "true",
        multi_assets_margin=str(multi_assets_raw).lower() == "true",
    )


def resolve_sizing_equity(snapshot: AccountSnapshot, cap_usdt: Decimal) -> Decimal:
    """I-EQUITY-MTM: E = min(wallet_balance + unrealized_pnl, cap_usdt).

    cap 은 '목표 노셔널'이 아니라 사이징 에쿼티의 절대 상한 캡이다.
    결과가 0 이하면 RiskGateBreach 로 전체 HALT 한다.
    """
    equity = min(snapshot.wallet_balance + snapshot.unrealized_pnl, cap_usdt)
    if equity <= Decimal(0):
        raise RiskGateBreach(
            f"sizing equity {equity} must be positive "
            f"(wallet={snapshot.wallet_balance} uPnL={snapshot.unrealized_pnl} cap={cap_usdt})"
        )
    return equity


def assert_drawdown_within_limit(
    equity: Decimal, high_water_mark: Decimal, limit_fraction: float
) -> None:
    """I-DD-HALT: equity/hwm - 1 <= limit_fraction 이면 RiskGateBreach.

    high_water_mark <= 0 은 초기 사이클로 간주해 게이트를 통과시킨다.
    """
    if high_water_mark <= Decimal(0):
        return
    drawdown = equity / high_water_mark - Decimal(1)
    if drawdown <= Decimal(str(limit_fraction)):
        raise RiskGateBreach(
            f"equity drawdown {drawdown} breaches halt limit {limit_fraction} "
            f"(equity={equity} hwm={high_water_mark})"
        )


def assert_venue_configuration(snapshot: AccountSnapshot) -> None:
    """one-way / USDT 단일 마진 가정이 깨지면 HALT 한다."""
    if snapshot.multi_assets_margin:
        raise RiskGateBreach("multi-assets margin is active; USDT single-margin assumption broken")
    if snapshot.dual_side_position:
        raise RiskGateBreach("dual-side position mode is active; one-way assumption broken")


def reconcile_or_halt(
    snapshot: AccountSnapshot,
    ledger_positions: Mapping[str, Decimal],
    *,
    qty_tolerance_fraction: float,
) -> None:
    """거래소 스냅샷과 내부 원장을 대조한다. 불일치 시 절대 보정하지 않고 breach만 발생시킨다."""
    if qty_tolerance_fraction < 0:
        raise ValueError("qty_tolerance_fraction must be >= 0")
    epsilon = Decimal("1e-12")
    symbols = set(ledger_positions) | {
        sym for sym, qty in snapshot.positions.items() if qty != 0
    }
    for symbol in sorted(symbols):
        ledger_qty = ledger_positions.get(symbol, Decimal(0))
        venue_qty = snapshot.positions.get(symbol, Decimal(0))
        denominator = max(abs(ledger_qty), epsilon)
        deviation = abs(venue_qty - ledger_qty) / denominator
        if deviation > Decimal(str(qty_tolerance_fraction)):
            raise ReconciliationBreach(
                f"position divergence for {symbol}: venue={venue_qty} "
                f"ledger={ledger_qty} tolerance={qty_tolerance_fraction}"
            )


def assert_suppressed_venue_flat(snapshot: AccountSnapshot) -> None:
    """억제 모드에서 거래소 포지션이 모두 0임을 증명한다."""
    non_zero = sorted(
        symbol for symbol, qty in snapshot.positions.items() if qty != Decimal(0)
    )
    if non_zero:
        raise ReconciliationBreach(
            f"suppressed venue position non-zero for {', '.join(non_zero)}: "
            f"venue={snapshot.positions}"
        )


def effective_positions(
    mode: ExecutionMode,
    snapshot: AccountSnapshot,
    ledger_positions: Mapping[str, Decimal],
) -> Mapping[str, Decimal]:
    """억제 모드이면 원장을, 라이브이면 거래소 스냅샷을 반환한다."""
    if mode.suppresses_mutations:
        return ledger_positions
    return snapshot.positions
