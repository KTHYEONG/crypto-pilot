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


def _fetch_flag(client: Any, path: str, key: str) -> Any:
    """전용 설정 엔드포인트에서 단일 boolean 플래그를 읽는다(응답 스키마 불일치는 fail-closed)."""
    payload = client.request("GET", path, signed=True)
    if not isinstance(payload, dict) or key not in payload:
        raise DataIntegrityError(f"{path} returned an unexpected schema")
    return payload[key]


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

    # dualSidePosition 은 /fapi/v2(v3)/account 응답에 없다 -- 전용 엔드포인트에서 조회한다.
    # multiAssetsMargin 은 account 페이로드에 있으면 그대로, 없으면(v3) 전용 엔드포인트로 폴백.
    dual_side_raw = account.get("dualSidePosition")
    if dual_side_raw is None:
        dual_side_raw = _fetch_flag(client, "/fapi/v1/positionSide/dual", "dualSidePosition")
    multi_assets_raw = account.get("multiAssetsMargin")
    if multi_assets_raw is None:
        multi_assets_raw = _fetch_flag(client, "/fapi/v1/multiAssetsMargin", "multiAssetsMargin")
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


def resolve_sizing_equity(
    snapshot: AccountSnapshot,
    cap_usdt: Decimal,
    *,
    mode: ExecutionMode | None = None,
    cash_usdt: Decimal | None = None,
    positions: Mapping[str, Decimal] | None = None,
    marks: Mapping[str, Decimal] | None = None,
) -> Decimal:
    """I-EQUITY-MODE / I-EQUITY-MTM: 모드에 따라 가상 MTM 분기."""
    if mode is not None and mode.suppresses_mutations:
        # virtual MTM: cash + Σ qty*mark, 첫 사이클 cash None이면 cap으로 시드
        cash = cash_usdt if cash_usdt is not None else cap_usdt
        total = Decimal(cash)
        if positions is not None and marks is not None:
            for sym, qty in positions.items():
                mk = marks.get(sym)
                if mk is not None:
                    total += qty * mk
        # 합성 원장은 캡을 적용하지 않는다: 백테스트의 자유 복리 vol-target 북과의
        # 정합성을 위해 cap_usdt 는 첫 사이클 현금 시드로만 쓰인다(I-PAPER-IS-BACKTEST-CONTINUATION).
        equity = total
        if equity <= Decimal(0):
            raise RiskGateBreach(
                f"sizing equity {equity} must be positive "
                f"(virtual_mtm={total} seed={cap_usdt})"
            )
        return equity
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


def synthetic_flat_snapshot(now: pd.Timestamp) -> AccountSnapshot:
    """자격증명 없는 PAPER/SHADOW용 합성 스냅샷: 플랫·one-way·단일 마진.

    억제 모드는 베뉴를 건드리지 않으므로 실제 계좌 조회 없이 이 스냅샷으로
    venue-config / flatness 가드를 통과시킨다(I-PAPER-NO-CREDENTIALS).
    """
    ts = pd.Timestamp(now)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return AccountSnapshot(
        taken_at=ts,
        wallet_balance=Decimal(0),
        available_balance=Decimal(0),
        total_maint_margin=Decimal(0),
        unrealized_pnl=Decimal(0),
        positions={},
        dual_side_position=False,
        multi_assets_margin=False,
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
