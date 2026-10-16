"""Venue-truth reconciler for testnet drill sessions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from src.common.errors import DataIntegrityError
from src.live.testnet_drill.config import DrillConfig


@dataclass(frozen=True, slots=True)
class ExpectationResult:
    """One named reconciliation expectation."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ReconciliationEvidence:
    """Venue truth next to the journal fold."""

    symbols: tuple[str, ...] = ()
    journal_positions: tuple[tuple[str, Decimal], ...] = ()
    venue_positions: tuple[tuple[str, Decimal], ...] = ()
    journal_observed: tuple[tuple[str, Decimal], ...] = ()
    venue_executed: tuple[tuple[str, Decimal], ...] = ()
    journal_trade_qty: tuple[tuple[str, Decimal], ...] = ()
    journal_trade_vwap: tuple[tuple[str, Decimal], ...] = ()
    journal_trade_fee: tuple[tuple[str, Decimal], ...] = ()
    journal_maker_qty: tuple[tuple[str, Decimal], ...] = ()
    venue_trade_qty: tuple[tuple[str, Decimal], ...] = ()
    venue_trade_vwap: tuple[tuple[str, Decimal], ...] = ()
    venue_trade_commission: tuple[tuple[str, Decimal], ...] = ()
    venue_maker_count: tuple[tuple[str, int], ...] = ()
    journal_maker_count: tuple[tuple[str, int], ...] = ()
    estimated_fee_orders: tuple[str, ...] = ()
    wallet_start: Decimal | None = None
    wallet_end: Decimal = Decimal(0)
    income_sums: tuple[tuple[str, Decimal], ...] = ()
    open_orders: tuple[str, ...] = ()
    fill_keys: tuple[tuple[str, Decimal], ...] = ()
    read_errors: tuple[str, ...] = ()


def _as_map(pairs: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return dict(pairs)


def _abs_diff(left: Decimal, right: Decimal) -> Decimal:
    return abs(Decimal(left) - Decimal(right))


def _trade_parity(evidence: ReconciliationEvidence, price_tol: Decimal, fee_tol: Decimal) -> ExpectationResult:
    journal_qty = _as_map(evidence.journal_trade_qty)
    journal_vwap = _as_map(evidence.journal_trade_vwap)
    journal_fee = _as_map(evidence.journal_trade_fee)
    venue_qty = _as_map(evidence.venue_trade_qty)
    venue_vwap = _as_map(evidence.venue_trade_vwap)
    venue_fee = _as_map(evidence.venue_trade_commission)
    journal_maker = _as_map(evidence.journal_maker_count)
    estimated = set(evidence.estimated_fee_orders)
    venue_maker = _as_map(evidence.venue_maker_count)
    r3_bad: list[str] = []
    maker_notes: list[str] = []
    for order_id in sorted(set(journal_qty) | set(venue_qty)):
        jq = Decimal(str(journal_qty.get(order_id, 0)))
        vq = Decimal(str(venue_qty.get(order_id, 0)))
        if jq != vq:
            r3_bad.append(f"{order_id} qty journal={jq} venue={vq}")
            continue
        if jq > 0 and vq > 0 and (order_id not in journal_vwap or order_id not in venue_vwap):
            r3_bad.append(f"{order_id} missing VWAP")
        if jq > 0 and vq > 0 and order_id in journal_vwap and order_id in venue_vwap:
            jw = Decimal(str(journal_vwap[order_id]))
            vw = Decimal(str(venue_vwap[order_id]))
            denom = abs(vw) if vw != 0 else Decimal(1)
            if abs(jw - vw) / denom > price_tol:
                r3_bad.append(f"{order_id} vwap journal={jw} venue={vw}")
        jf = Decimal(str(journal_fee.get(order_id, 0)))
        vf = Decimal(str(venue_fee.get(order_id, 0)))
        if order_id in estimated:
            if vf - jf > fee_tol:
                r3_bad.append(f"{order_id} estimated fee undercuts venue journal={jf} venue={vf}")
            continue
        if _abs_diff(jf, vf) > fee_tol:
            r3_bad.append(f"{order_id} fee journal={jf} venue={vf}")
        jm = int(journal_maker.get(order_id, 0))
        vm = int(venue_maker.get(order_id, 0))
        if jm != vm:
            maker_notes.append(f"{order_id} maker journal={jm} venue={vm}")
    r3_detail = "; ".join(r3_bad + ([f"maker_mismatch: {', '.join(maker_notes)}"] if maker_notes else []))
    r3 = ExpectationResult("trade_parity", not r3_bad, r3_detail)
    return r3


def evaluate_reconciliation(evidence: ReconciliationEvidence, config: DrillConfig) -> tuple[ExpectationResult, ...]:
    """Pure R1..R6 checks over recorded evidence."""
    qty_tol = config.qty_tolerance
    price_tol = config.price_rel_tolerance
    fee_tol = config.fee_abs_tolerance_usdt
    numbers = [
        evidence.wallet_end,
        *(
            value
            for pairs in (
                evidence.journal_positions,
                evidence.venue_positions,
                evidence.journal_observed,
                evidence.venue_executed,
                evidence.journal_trade_qty,
                evidence.venue_trade_qty,
                evidence.journal_trade_vwap,
                evidence.venue_trade_vwap,
                evidence.journal_trade_fee,
                evidence.venue_trade_commission,
                evidence.income_sums,
            )
            for _, value in pairs
        ),
    ]
    if evidence.wallet_start is not None:
        numbers.append(evidence.wallet_start)
    if evidence.read_errors or any(not value.is_finite() for value in numbers):
        return tuple(
            ExpectationResult(name, False, "; ".join(evidence.read_errors) or "non-finite evidence")
            for name in (
                "position_parity",
                "order_watermark_parity",
                "trade_parity",
                "wallet_parity",
                "no_residual_orders",
                "no_duplicate_fills",
            )
        )
    journal_pos = _as_map(evidence.journal_positions)
    venue_pos = _as_map(evidence.venue_positions)
    symbols = list(evidence.symbols) or sorted(set(journal_pos) | set(venue_pos))
    r1_bad: list[str] = []
    for symbol in symbols:
        diff = _abs_diff(
            Decimal(str(journal_pos.get(symbol, 0))),
            Decimal(str(venue_pos.get(symbol, 0))),
        )
        if diff > qty_tol:
            r1_bad.append(f"{symbol} journal={journal_pos.get(symbol, 0)} venue={venue_pos.get(symbol, 0)}")
    r1 = ExpectationResult("position_parity", not r1_bad, "; ".join(r1_bad))

    journal_obs = _as_map(evidence.journal_observed)
    venue_exec = _as_map(evidence.venue_executed)
    r2_bad: list[str] = []
    for order_id in sorted(set(journal_obs) | set(venue_exec)):
        diff = _abs_diff(
            Decimal(str(journal_obs.get(order_id, 0))),
            Decimal(str(venue_exec.get(order_id, 0))),
        )
        if order_id not in journal_obs or order_id not in venue_exec or diff != 0:
            r2_bad.append(f"{order_id} journal={journal_obs.get(order_id, 0)} venue={venue_exec.get(order_id, 0)}")
    r2 = ExpectationResult("order_watermark_parity", not r2_bad, "; ".join(r2_bad))

    r3 = _trade_parity(evidence, price_tol, fee_tol)

    income = _as_map(evidence.income_sums)
    wallet_delta = evidence.wallet_end - evidence.wallet_start if evidence.wallet_start is not None else None
    explained = sum(
        (Decimal(str(income.get(kind, 0))) for kind in ("REALIZED_PNL", "COMMISSION", "FUNDING_FEE")),
        Decimal(0),
    )
    r4 = ExpectationResult(
        "wallet_parity",
        wallet_delta is not None and _abs_diff(wallet_delta, explained) <= fee_tol,
        f"delta={wallet_delta} explained={explained}",
    )

    r5 = ExpectationResult(
        "no_residual_orders",
        len(evidence.open_orders) == 0,
        f"open={len(evidence.open_orders)}",
    )

    seen: set[tuple[str, Decimal]] = set()
    dupes: list[str] = []
    for order_id, cumulative in evidence.fill_keys:
        key = (order_id, cumulative)
        if key in seen:
            dupes.append(f"{order_id}@{cumulative}")
        seen.add(key)
    r6 = ExpectationResult("no_duplicate_fills", not dupes, "; ".join(dupes))
    return (r1, r2, r3, r4, r5, r6)


# Terminal statuses of orders the venue never accepted; only these may answer -2013 with zero observed quantity.
_NEVER_ON_VENUE = frozenset({"NOT_PLACED", "NOT_FOUND", "REJECTED", "OBSOLETE"})


def _number(value: Any) -> Decimal:
    amount = Decimal(str(value))
    if not amount.is_finite():
        raise DataIntegrityError("non-finite venue value")
    return amount


def _rows(payload: Any, endpoint: str) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise DataIntegrityError(f"{endpoint}: expected object rows")
    return payload


@dataclass(slots=True)
class _Aggregate:
    """Per-order quantity, notional, fee and maker-count sums."""

    qty: dict[str, Decimal]
    notional: dict[str, Decimal]
    fees: dict[str, Decimal]
    makers: dict[str, int]
    estimated: set[str]

    @classmethod
    def empty(cls) -> _Aggregate:
        return cls({}, {}, {}, {}, set())

    def add(self, key: str, amount: Decimal, price: Decimal, fee: Decimal, maker: bool) -> None:
        self.qty[key] = self.qty.get(key, Decimal(0)) + amount
        self.notional[key] = self.notional.get(key, Decimal(0)) + amount * price
        self.fees[key] = self.fees.get(key, Decimal(0)) + fee
        self.makers[key] = self.makers.get(key, 0) + int(maker)

    def vwap(self) -> tuple[tuple[str, Decimal], ...]:
        return tuple((key, self.notional[key] / value) for key, value in self.qty.items())


def _venue_positions(client: Any, symbols: tuple[str, ...]) -> dict[str, Decimal]:
    positions: dict[str, Decimal] = {}
    for row in _rows(client.position_risk(), "positionRisk"):
        if row["symbol"] not in symbols:
            continue
        if row.get("positionSide", "BOTH") != "BOTH":
            raise DataIntegrityError("positionRisk: hedge position cannot be netted")
        positions[row["symbol"]] = _number(row["positionAmt"])
    missing = set(symbols) - positions.keys()
    if missing:
        raise DataIntegrityError(f"positionRisk: missing symbols {sorted(missing)}")
    return positions


_ORDER_FILL_KINDS = frozenset({"execution", "recovered", "orphan_settlement"})
# Journaled position adjustments without a venue order (operator resync, adopted force-close): they move the
# position fold (R1) but have no userTrades counterpart, so they stay out of trade parity (R3).
_ADJUSTMENT_KINDS = frozenset({"operator_resync", "venue_force_close"})
# Recovery books the conservative taker fee because an order query does not reveal maker/taker, so the
# journal fee may exceed the venue commission for these orders but must never undercut it.
_ESTIMATED_FEE_KINDS = frozenset({"recovered", "orphan_settlement"})


def _journal_fold(fills: Any) -> tuple[dict[str, Decimal], _Aggregate]:
    positions: dict[str, Decimal] = {}
    trades = _Aggregate.empty()
    for fill in fills:
        if fill.kind not in _ORDER_FILL_KINDS and fill.kind not in _ADJUSTMENT_KINDS:
            continue
        positions[fill.symbol] = positions.get(fill.symbol, Decimal(0)) + (
            fill.quantity if fill.side == "BUY" else -fill.quantity
        )
        if fill.kind in _ORDER_FILL_KINDS and fill.client_order_id is not None:
            fee = fill.quantity * fill.price * Decimal(str(fill.fee_bps)) / 10000
            trades.add(fill.client_order_id, fill.quantity, fill.price, fee, fill.liquidity == "maker")
            if fill.kind in _ESTIMATED_FEE_KINDS:
                trades.estimated.add(fill.client_order_id)
    return positions, trades


def _venue_orders(
    client: Any, journal: Any, orders: dict[str, str]
) -> tuple[dict[str, Decimal], dict[tuple[str, str], str]]:
    from src.live.errors import VenueError

    executed: dict[str, Decimal] = {}
    venue_ids: dict[tuple[str, str], str] = {}
    for client_id, symbol in orders.items():
        try:
            order = client.query_order(symbol, client_id)
        except VenueError as exc:
            terminal = journal.terminal(client_id)
            if (
                exc.code != -2013
                or terminal is None
                or terminal.status not in _NEVER_ON_VENUE
                or journal.observed_qty(client_id) != 0
            ):
                raise
            executed[client_id] = Decimal(0)
            continue
        if not isinstance(order, dict):
            raise DataIntegrityError(f"order: invalid response for {client_id}")
        executed[client_id] = _number(order["executedQty"])
        key = (symbol, str(order["orderId"]))
        if key in venue_ids:
            raise DataIntegrityError("orderId maps to multiple client ids")
        venue_ids[key] = client_id
    return executed, venue_ids


def _book_trade(trades: _Aggregate, key: str, row: dict[str, Any]) -> None:
    if row["commissionAsset"] != "USDT" or not isinstance(row["maker"], bool):
        raise DataIntegrityError("userTrades: unsupported commission asset or maker flag")
    amount, price = _number(row["qty"]), _number(row["price"])
    if amount <= 0 or price <= 0:
        raise DataIntegrityError("userTrades: non-positive execution")
    trades.add(key, amount, price, _number(row["commission"]), row["maker"])


def _venue_trades(
    client: Any, symbols: tuple[str, ...], venue_ids: dict[tuple[str, str], str], since_ms: int, until_ms: int
) -> _Aggregate:
    trades = _Aggregate.empty()
    for symbol in symbols:
        cursor: int | None = None
        seen: set[int] = set()
        while True:
            params: dict[str, Any] = {"symbol": symbol, "limit": 1000}
            if cursor is None:
                params.update(startTime=since_ms, endTime=until_ms)
            else:
                params["fromId"] = cursor
            rows = _rows(client.request("GET", "/fapi/v1/userTrades", params, signed=True), "userTrades")
            for row in rows:
                trade_id = int(row["id"])
                if trade_id in seen:
                    raise DataIntegrityError("userTrades: duplicate or non-advancing page")
                seen.add(trade_id)
                key = venue_ids.get((symbol, str(row["orderId"])))
                if key is not None and since_ms <= int(row["time"]) <= until_ms:
                    _book_trade(trades, key, row)
            if len(rows) < 1000 or max(int(row["time"]) for row in rows) > until_ms:
                break
            next_cursor = max(int(row["id"]) for row in rows) + 1
            if cursor is not None and next_cursor <= cursor:
                raise DataIntegrityError("userTrades: cursor did not advance")
            cursor = next_cursor
    return trades


def _income_sums(client: Any, since_ms: int, until_ms: int) -> dict[str, Decimal]:
    income: dict[str, Decimal] = {}
    page = 1
    seen: set[tuple[str, str, str]] = set()
    while True:
        params = {"startTime": since_ms, "endTime": until_ms, "limit": 1000, "page": page}
        rows = _rows(client.request("GET", "/fapi/v1/income", params, signed=True), "income")
        for row in rows:
            key = (str(row["tranId"]), str(row["incomeType"]), str(row["asset"]))
            if key in seen:
                raise DataIntegrityError("income: duplicate or non-advancing page")
            seen.add(key)
            if row["asset"] == "USDT" and row["incomeType"] in ("REALIZED_PNL", "COMMISSION", "FUNDING_FEE"):
                income[row["incomeType"]] = income.get(row["incomeType"], Decimal(0)) + _number(row["income"])
        if len(rows) < 1000:
            return income
        page += 1


def _snapshot(session: Any, since_ms: int, until_ms: int) -> ReconciliationEvidence:
    client = session.client
    symbols = session.config.symbols
    journal = session.journal
    fills = journal.fills_after(-1)
    orders = {s.client_order_id: s.symbol for s in journal.submits()}
    orders.update({f.client_order_id: f.symbol for f in fills if f.client_order_id is not None})
    positions = _venue_positions(client, symbols)
    journal_positions, booked = _journal_fold(fills)
    executed, venue_ids = _venue_orders(client, journal, orders)
    traded = _venue_trades(client, symbols, venue_ids, since_ms, until_ms)
    income = _income_sums(client, since_ms, until_ms)
    account = client.account()
    if not isinstance(account, dict):
        raise DataIntegrityError("account: invalid response")
    open_orders = _rows(client.open_orders(), "openOrders")
    return ReconciliationEvidence(
        symbols=symbols,
        journal_positions=tuple(journal_positions.items()),
        venue_positions=tuple(positions.items()),
        journal_observed=tuple((key, journal.observed_qty(key)) for key in orders),
        venue_executed=tuple(executed.items()),
        journal_trade_qty=tuple(booked.qty.items()),
        journal_trade_vwap=booked.vwap(),
        journal_trade_fee=tuple(booked.fees.items()),
        journal_maker_count=tuple(booked.makers.items()),
        estimated_fee_orders=tuple(sorted(booked.estimated)),
        venue_trade_qty=tuple(traded.qty.items()),
        venue_trade_vwap=traded.vwap(),
        venue_trade_commission=tuple(traded.fees.items()),
        venue_maker_count=tuple(traded.makers.items()),
        wallet_start=session.starting_wallet_balance,
        wallet_end=_number(account["totalWalletBalance"]),
        income_sums=tuple(income.items()),
        open_orders=tuple(str(row["clientOrderId"]) for row in open_orders),
        fill_keys=tuple(
            (f.client_order_id, f.cumulative_executed_qty)
            for f in fills
            if f.client_order_id is not None and f.cumulative_executed_qty is not None
        ),
    )


def reconcile_session(session: Any, *, since_ms: int, settle_wait_s: float) -> ReconciliationEvidence:
    """Observe final venue truth; incomplete or unavailable evidence never passes."""
    import math

    from src.live.errors import TransientReadError

    if not math.isfinite(settle_wait_s) or settle_wait_s < 0:
        raise ValueError("settle_wait_s must be finite and non-negative")
    deadline = time.monotonic() + settle_wait_s
    while True:
        try:
            evidence = _snapshot(session, since_ms, int(time.time() * 1000))
        except TransientReadError as exc:
            evidence = ReconciliationEvidence(symbols=session.config.symbols, read_errors=(type(exc).__name__,))
        if (
            all(item.passed for item in evaluate_reconciliation(evidence, session.config))
            or time.monotonic() >= deadline
        ):
            return evidence
        time.sleep(min(session.config.settle_poll_s, max(0, deadline - time.monotonic())))
