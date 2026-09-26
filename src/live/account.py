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
class PositionBreach:
    symbol: str
    venue_qty: Decimal
    ledger_qty: Decimal

    @property
    def gap(self) -> Decimal:
        """venue_qty - ledger_qty (the quantity the ledger is missing)."""
        return self.venue_qty - self.ledger_qty


@dataclass(frozen=True, slots=True)
class VenueForceClose:
    symbol: str
    side: str  # 'BUY' | 'SELL'
    executed_qty: Decimal  # > 0
    avg_price: Decimal  # > 0
    auto_close_type: str  # 'LIQUIDATION' | 'ADL'
    order_id: str
    updated_at: pd.Timestamp  # tz-aware UTC


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
    fallback_marks: Mapping[str, Decimal] | None = None,
) -> Decimal:
    """Sizing equity for the cycle.

    LIVE modes use the venue's margin equity (wallet balance + unrealized PnL) with no ceiling.
    PAPER/SHADOW value the virtual book as cash + Σ qty·mark; a held symbol without a valid live
    mark is valued at its causal decision snapshot close from ``fallback_marks``, and if neither
    exists the cycle fails closed instead of silently valuing the position at zero (which
    over-sized every target by the missing notional).

    Raises:
        RiskGateBreach: non-positive equity, or a nonzero PAPER/SHADOW position with neither a
            live mark nor a fallback mark.
    """
    if mode is not None and mode.suppresses_mutations:
        # virtual MTM: cash + Σ qty*mark, 첫 사이클 cash None이면 cap으로 시드
        cash = cash_usdt if cash_usdt is not None else cap_usdt
        total = Decimal(cash)
        if positions is not None and marks is not None:
            for sym, qty in positions.items():
                if qty == 0:
                    continue
                mk = marks.get(sym)
                if mk is None and fallback_marks is not None:
                    candidate = fallback_marks.get(sym)
                    if candidate is not None:
                        try:
                            finite = math.isfinite(float(candidate))
                        except (OverflowError, ValueError):
                            finite = False
                        if finite and candidate > 0:
                            mk = candidate
                if mk is None:
                    raise RiskGateBreach(
                        f"sizing equity missing mark for held {sym}; "
                        "neither a live mark nor a fallback mark exists"
                    )
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


def find_position_breaches(
    snapshot: AccountSnapshot,
    ledger_positions: Mapping[str, Decimal],
    *,
    qty_tolerance_fraction: float,
    settled_symbols: Collection[str] = (),
) -> tuple[PositionBreach, ...]:
    """Compare the venue snapshot with the ledger and return every symbol whose relative deviation exceeds `qty_tolerance_fraction`, sorted by symbol. Pure; never corrects anything. Settled delisted symbols that are flat on the venue are exempt (delisting settlement is booked separately)."""
    if qty_tolerance_fraction < 0:
        raise ValueError("qty_tolerance_fraction must be >= 0")
    epsilon = Decimal("1e-12")
    symbols = set(ledger_positions) | {
        sym for sym, qty in snapshot.positions.items() if qty != 0
    }
    breaches: list[PositionBreach] = []
    for symbol in sorted(symbols):
        ledger_qty = ledger_positions.get(symbol, Decimal(0))
        venue_qty = snapshot.positions.get(symbol, Decimal(0))
        if symbol in settled_symbols and venue_qty == 0:
            continue
        denominator = max(abs(ledger_qty), epsilon)
        deviation = abs(venue_qty - ledger_qty) / denominator
        if deviation > Decimal(str(qty_tolerance_fraction)):
            breaches.append(
                PositionBreach(symbol=symbol, venue_qty=venue_qty, ledger_qty=ledger_qty)
            )
    return tuple(breaches)


def reconcile_or_halt(
    snapshot: AccountSnapshot,
    ledger_positions: Mapping[str, Decimal],
    *,
    qty_tolerance_fraction: float,
    settled_symbols: Collection[str] = (),
) -> None:
    """Raise `ReconciliationBreach` naming the first breach returned by `find_position_breaches`. Kept for callers that must fail closed (resync verification, `derisk_mode_enabled=False`)."""
    breaches = find_position_breaches(
        snapshot,
        ledger_positions,
        qty_tolerance_fraction=qty_tolerance_fraction,
        settled_symbols=settled_symbols,
    )
    if breaches:
        first = breaches[0]
        raise ReconciliationBreach(
            f"position divergence for {first.symbol}: venue={first.venue_qty} "
            f"ledger={first.ledger_qty} tolerance={qty_tolerance_fraction}"
        )


def fetch_venue_force_closes(
    client: Any, *, since: pd.Timestamp, until: pd.Timestamp
) -> tuple[VenueForceClose, ...]:
    """Fetch the account's venue-initiated liquidation and ADL orders (`GET /fapi/v1/forceOrders`, both auto-close types) filled in `[since, until]`, paginating by time until the range is exhausted.

    Raises: DataIntegrityError: the response is not a list, or an entry lacks symbol/side/executedQty/avgPrice/updateTime or has non-positive quantity or price. Only filled quantity is returned; entries with zero executed quantity are skipped.
    """
    since_ms = int(pd.Timestamp(since).tz_convert("UTC").value // 1_000_000)
    until_ms = int(pd.Timestamp(until).tz_convert("UTC").value // 1_000_000)
    out: list[VenueForceClose] = []
    cursor = since_ms
    seen: set[str] = set()
    while True:
        payload = client.force_orders(start_time_ms=cursor, end_time_ms=until_ms, limit=100)
        if not isinstance(payload, list):
            raise DataIntegrityError("forceOrders endpoint returned an unexpected schema")
        if not payload:
            break
        max_ts: int | None = None
        for entry in payload:
            if not isinstance(entry, dict):
                raise DataIntegrityError("forceOrders row malformed")
            try:
                symbol = str(entry["symbol"])
                side = str(entry["side"])
                qty = Decimal(str(entry["executedQty"]))
                price = Decimal(str(entry["avgPrice"]))
                ts_ms = int(entry["updateTime"])
                order_id = str(entry.get("orderId", entry.get("id", "")))
                auto_close = str(entry.get("autoCloseType", ""))
            except (KeyError, TypeError, ValueError) as exc:
                raise DataIntegrityError("forceOrders row missing required keys") from exc
            if side not in ("BUY", "SELL"):
                raise DataIntegrityError("forceOrders row has unknown side")
            if qty < 0:
                raise DataIntegrityError("forceOrders row has negative executedQty")
            if qty == 0:
                continue
            if price <= 0:
                raise DataIntegrityError("forceOrders row has non-positive avgPrice")
            if auto_close not in ("LIQUIDATION", "ADL"):
                raise DataIntegrityError("forceOrders row has unknown autoCloseType")
            updated_at = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
            if max_ts is None or ts_ms > max_ts:
                max_ts = ts_ms
            key = f"{symbol}|{order_id}|{ts_ms}|{qty}|{price}"
            if key in seen:
                continue
            seen.add(key)
            out.append(
                VenueForceClose(
                    symbol=symbol,
                    side=side,
                    executed_qty=qty,
                    avg_price=price,
                    auto_close_type=auto_close,
                    order_id=order_id,
                    updated_at=updated_at,
                )
            )
        if max_ts is None or max_ts >= until_ms or len(payload) < 100:
            break
        cursor = max_ts + 1
    out.sort(key=lambda item: (item.symbol, item.order_id, int(item.updated_at.value // 1_000_000)))
    return tuple(out)


def explain_breaches(
    breaches: Sequence[PositionBreach],
    force_closes: Sequence[VenueForceClose],
    *,
    qty_tolerance_fraction: float,
) -> tuple[tuple[VenueForceClose, ...], tuple[PositionBreach, ...]]:
    """Split breaches into those fully explained by venue force closes and those that remain unexplained.

    A breach is explained only when the signed sum of force-close quantities for its symbol equals its `gap` within `qty_tolerance_fraction` of the larger absolute quantity. Partial explanations count as unexplained — adopting part of a gap would hide the rest. Returns (force closes to adopt, unexplained breaches).
    """
    by_symbol: dict[str, list[VenueForceClose]] = {}
    for item in force_closes:
        by_symbol.setdefault(item.symbol, []).append(item)
    to_adopt: list[VenueForceClose] = []
    unexplained: list[PositionBreach] = []
    epsilon = Decimal("1e-12")
    for breach in breaches:
        closes = by_symbol.get(breach.symbol, [])
        signed = sum(
            (item.executed_qty if item.side == "BUY" else -item.executed_qty for item in closes),
            Decimal(0),
        )
        gap = breach.gap
        denominator = max(abs(gap), abs(signed), epsilon)
        deviation = abs(signed - gap) / denominator
        if closes and deviation <= Decimal(str(qty_tolerance_fraction)):
            to_adopt.extend(closes)
        else:
            unexplained.append(breach)
    to_adopt.sort(key=lambda item: (item.symbol, item.order_id))
    return tuple(to_adopt), tuple(unexplained)


def free_margin_breached(snapshot: AccountSnapshot, *, min_free_margin_fraction: float) -> bool:
    """True when the wallet balance is positive and `available_balance / wallet_balance` is below the floor. A zero wallet (credential-less synthetic snapshot) is never a breach."""
    if snapshot.wallet_balance <= 0:
        return False
    return (
        snapshot.available_balance / snapshot.wallet_balance
        < Decimal(str(min_free_margin_fraction))
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
