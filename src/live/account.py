"""Account snapshot, venue configuration guard, and first reconciliation.

I-RECONCILE-FIRST: 불일치 시 자동 보정 없이 예외만 발생시킨다.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.live.errors import ReconciliationBreach, RiskGateBreach, VenueError
from src.live.settings import ExecutionMode

if TYPE_CHECKING:
    from src.live.audit import AuditLog
    from src.live.planner import OrderIntent

VENUE_MARGIN_TYPE: str = "CROSSED"
DELISTED_SYMBOL_STATUSES: frozenset[str] = frozenset({"SETTLING", "CLOSE"})
_CROSS_MARGIN_ALIASES: frozenset[str] = frozenset({"cross", "crossed"})
BLOCKED_MARGIN_TYPE_CHANGE_CODES: frozenset[int] = frozenset({-4047, -4048})
BLOCKED_LEVERAGE_CHANGE_CODES: frozenset[int] = frozenset({-4161})


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


@dataclass(frozen=True, slots=True)
class LeverageBracket:
    """레버리지 브래킷: 초기 레버리지와 노셔널 상/하한."""

    bracket: int
    initial_leverage: int
    notional_cap: Decimal
    notional_floor: Decimal


@dataclass(frozen=True, slots=True)
class VenueSymbolConfig:
    """positionRisk 행: cross 마진 종류(소문자)와 레버리지. 없으면 None."""

    symbol: str
    margin_type: str | None
    leverage: int | None


@dataclass(frozen=True, slots=True)
class VenueLeveragePlan:
    """심볼별 목표 레버리지와 변경 대상(심볼 정렬)."""

    target_leverage: Mapping[str, int]
    margin_type_changes: tuple[str, ...]
    leverage_changes: tuple[str, ...]


def parse_leverage_brackets(payload: Any) -> dict[str, tuple[LeverageBracket, ...]]:
    """GET /fapi/v1/leverageBracket 응답을 심볼별 브래킷 튜플로 파싱한다."""
    if not isinstance(payload, list):
        raise DataIntegrityError("leverageBracket returned an unexpected schema")
    parsed: dict[str, tuple[LeverageBracket, ...]] = {}
    for row in payload:
        if not isinstance(row, dict) or "symbol" not in row:
            raise DataIntegrityError("leverageBracket row malformed")
        brackets_raw = row.get("brackets")
        if not isinstance(brackets_raw, list) or not brackets_raw:
            raise DataIntegrityError("leverageBracket row malformed")
        brackets: list[LeverageBracket] = []
        for entry in brackets_raw:
            try:
                brackets.append(
                    LeverageBracket(
                        bracket=int(entry["bracket"]),
                        initial_leverage=int(entry["initialLeverage"]),
                        notional_cap=Decimal(str(entry["notionalCap"])),
                        notional_floor=Decimal(str(entry["notionalFloor"])),
                    )
                )
            except (KeyError, TypeError) as exc:
                raise DataIntegrityError("leverageBracket bracket missing required keys") from exc
        parsed[str(row["symbol"])] = tuple(sorted(brackets, key=lambda item: item.bracket))
    return parsed


def parse_position_config(payload: Any) -> dict[str, VenueSymbolConfig]:
    """GET /fapi/v2/positionRisk 응답을 심볼별 마진/레버리지 설정으로 파싱한다."""
    if not isinstance(payload, list):
        raise DataIntegrityError("positionRisk endpoint returned an unexpected schema")
    configs: dict[str, VenueSymbolConfig] = {}
    for row in payload:
        if not isinstance(row, dict) or "symbol" not in row:
            raise DataIntegrityError("positionRisk row missing symbol")
        margin_type = str(row["marginType"]).lower() if "marginType" in row else None
        leverage = int(row["leverage"]) if "leverage" in row else None
        configs[str(row["symbol"])] = VenueSymbolConfig(
            symbol=str(row["symbol"]), margin_type=margin_type, leverage=leverage
        )
    return configs


def required_leverage(max_gross_leverage: float, buffer_fraction: float, bracket_max_leverage: int) -> int:
    """버퍼를 반영한 필요 레버리지: min(브래킷 상한, ceil(ceiling/(1-buffer))), 최소 1."""
    needed = math.ceil(max_gross_leverage / (1.0 - buffer_fraction))
    return max(1, min(int(bracket_max_leverage), needed))


def max_notional_at_leverage(brackets: Sequence[LeverageBracket], leverage: int) -> Decimal:
    """레버리지 L에서 허용되는 최대 노셔널: 초기 레버리지가 L 이상인 브래킷 중 최대 cap."""
    eligible = [b.notional_cap for b in brackets if b.initial_leverage >= leverage]
    return max(eligible) if eligible else Decimal(0)


def plan_venue_leverage(
    symbols: Collection[str],
    brackets: Mapping[str, Sequence[LeverageBracket]],
    configs: Mapping[str, VenueSymbolConfig],
    *,
    max_gross_leverage: float,
    buffer_fraction: float,
) -> VenueLeveragePlan:
    """목표 레버리지와 변경 대상(CROSSED/레버리지 불일치)을 계획한다. 멱등하다."""
    targets: dict[str, int] = {}
    margin_type_changes: list[str] = []
    leverage_changes: list[str] = []
    for symbol in sorted(set(symbols)):
        symbol_brackets = brackets.get(symbol)
        if not symbol_brackets:
            raise RiskGateBreach(f"no leverage bracket for {symbol}")
        target = required_leverage(max_gross_leverage, buffer_fraction, symbol_brackets[0].initial_leverage)
        targets[symbol] = target
        config = configs.get(symbol)
        if config is None or config.margin_type not in _CROSS_MARGIN_ALIASES:
            margin_type_changes.append(symbol)
        if config is None or config.leverage != target:
            leverage_changes.append(symbol)
    return VenueLeveragePlan(
        target_leverage=targets,
        margin_type_changes=tuple(margin_type_changes),
        leverage_changes=tuple(leverage_changes),
    )


def ensure_venue_leverage(
    client: Any,
    symbols: Collection[str],
    brackets: Mapping[str, Sequence[LeverageBracket]],
    *,
    max_gross_leverage: float,
    buffer_fraction: float,
    audit: AuditLog,
) -> dict[str, int]:
    """마진 타입(CROSSED) 변경을 먼저, 레버리지 변경을 나중에 적용한다. 차단 코드는 fail-closed."""
    configs = parse_position_config(client.request("GET", "/fapi/v2/positionRisk", signed=True))
    plan = plan_venue_leverage(
        symbols, brackets, configs, max_gross_leverage=max_gross_leverage, buffer_fraction=buffer_fraction
    )
    for symbol in plan.margin_type_changes:
        try:
            client.request(
                "POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": VENUE_MARGIN_TYPE}, signed=True
            )
        except VenueError as exc:
            if exc.code in BLOCKED_MARGIN_TYPE_CHANGE_CODES:
                raise RiskGateBreach(
                    f"margin type change to {VENUE_MARGIN_TYPE} blocked for {symbol} (code={exc.code})"
                ) from exc
            raise
        audit.record("venue_margin_type_set", symbol=symbol, margin_type=VENUE_MARGIN_TYPE)
    for symbol in plan.leverage_changes:
        leverage = plan.target_leverage[symbol]
        try:
            client.request(
                "POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True
            )
        except VenueError as exc:
            if exc.code in BLOCKED_LEVERAGE_CHANGE_CODES:
                raise RiskGateBreach(
                    f"leverage change to {leverage} blocked for {symbol} (code={exc.code})"
                ) from exc
            raise
        audit.record("venue_leverage_set", symbol=symbol, leverage=leverage)
    return dict(plan.target_leverage)


def reject_intents_over_notional_cap(
    intents: Sequence[OrderIntent],
    brackets: Mapping[str, Sequence[LeverageBracket]],
    leverages: Mapping[str, int],
    audit: AuditLog,
) -> list[OrderIntent]:
    """설정 레버리지에서 브래킷 cap을 초과하는 노출증가 intent를 버린다. reduce-only는 항상 통과."""
    kept: list[OrderIntent] = []
    for intent in intents:
        if intent.reduce_only:
            kept.append(intent)
            continue
        cap = max_notional_at_leverage(brackets[intent.symbol], leverages[intent.symbol])
        target_notional = abs(intent.target_qty) * intent.decision_price
        if target_notional > cap:
            audit.record(
                "notional_cap_rejected",
                symbol=intent.symbol,
                target_notional=str(target_notional),
                notional_cap=str(cap),
                leverage=leverages[intent.symbol],
            )
            continue
        kept.append(intent)
    return kept


def settled_delisting_symbols(
    exchange_info: Mapping[str, Any],
    venue_positions: Mapping[str, Decimal],
    ledger_positions: Mapping[str, Decimal],
    *,
    now: pd.Timestamp,
) -> tuple[str, ...]:
    """원장에만 남은 상장폐지 포지션 중 정산 완료(flat + SETTLING/CLOSE + delivery 경과)를 반환한다."""
    entries = {
        str(entry["symbol"]): entry
        for entry in exchange_info.get("symbols", ())
        if isinstance(entry, Mapping) and "symbol" in entry
    }
    now_ms = int(pd.Timestamp(now).value // 1_000_000)
    settled: list[str] = []
    for symbol in sorted(ledger_positions):
        if ledger_positions[symbol] == 0:
            continue
        if venue_positions.get(symbol, Decimal(0)) != 0:
            continue
        entry = entries.get(symbol)
        if entry is None:
            continue
        if entry.get("status") not in DELISTED_SYMBOL_STATUSES:
            continue
        delivery = entry.get("deliveryDate")
        if delivery is None:
            continue
        if now_ms < int(delivery):
            continue
        settled.append(symbol)
    return tuple(settled)


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
    """LIVE modes size from the venue's own margin equity (wallet balance plus unrealized PnL) with no ceiling, so the account compounds exactly as the growth policy was evaluated; ``cap_usdt`` only seeds the first PAPER/SHADOW cycle's virtual cash."""
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
    equity = snapshot.wallet_balance + snapshot.unrealized_pnl
    if equity <= Decimal(0):
        raise RiskGateBreach(
            f"sizing equity {equity} must be positive "
            f"(wallet={snapshot.wallet_balance} uPnL={snapshot.unrealized_pnl} cap={cap_usdt})"
        )
    return equity


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
    settled_symbols: Collection[str] = (),
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
        if symbol in settled_symbols and venue_qty == 0:
            continue
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
