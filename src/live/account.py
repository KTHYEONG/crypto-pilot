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


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """거래소 계좌/포지션의 불변 스냅샷. positions는 심볼 -> 부호 있는 순포지션."""

    taken_at: pd.Timestamp
    wallet_balance: Decimal
    available_balance: Decimal
    total_maint_margin: Decimal
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
        positions=positions,
        dual_side_position=str(dual_side_raw).lower() == "true",
        multi_assets_margin=str(multi_assets_raw).lower() == "true",
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
