"""Tax ledger — immutable JSONL with watermark idempotent collection."""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace as _replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.common.durable_io import durable_write_text
from src.common.errors import DataIntegrityError
from src.common.paths import DATA_DIR
from src.live.order_journal import truncate_durably

# Re-exported so `from src.live.tax_ledger import TaxRecord` keeps working.
from src.live.tax_schema import (
    ONE_WAY_POSITION_SIDE,
    parse_tax_decimal,
    tax_event_sort_key,
    tax_record_from_row,
    tax_record_to_row,
    validate_tax_record,
)
from src.live.tax_schema import (
    TAX_RECORD_KINDS as TAX_RECORD_KINDS,
)
from src.live.tax_schema import (
    TaxRecord as TaxRecord,
)

if TYPE_CHECKING:
    from src.live.fills import FillEvent

INCOME_TYPE_KIND: Mapping[str, str] = {
    "REALIZED_PNL": "REALIZED_PNL",
    "DELIVERED_SETTELMENT": "REALIZED_PNL",
    "FUNDING_FEE": "FUNDING_FEE",
    "COMMISSION": "COMMISSION",
    "TRANSFER": "TRANSFER",
    "INTERNAL_TRANSFER": "TRANSFER",
    "CROSS_COLLATERAL_TRANSFER": "TRANSFER",
    "STRATEGY_UMFUTURES_TRANSFER": "TRANSFER",
    "COIN_SWAP_DEPOSIT": "TRANSFER",
    "COIN_SWAP_WITHDRAW": "TRANSFER",
    "AUTO_EXCHANGE": "TRANSFER",
    "INSURANCE_CLEAR": "UNCLASSIFIED",
    "WELCOME_BONUS": "UNCLASSIFIED",
    "REFERRAL_KICKBACK": "UNCLASSIFIED",
    "COMMISSION_REBATE": "UNCLASSIFIED",
    "API_REBATE": "UNCLASSIFIED",
    "CONTEST_REWARD": "UNCLASSIFIED",
    "POSITION_LIMIT_INCREASE_FEE": "UNCLASSIFIED",
    "FEE_RETURN": "UNCLASSIFIED",
    "BFUSD_REWARD": "UNCLASSIFIED",
    "OPTIONS_PREMIUM_FEE": "UNCLASSIFIED",
    "OPTIONS_SETTLE_PROFIT": "UNCLASSIFIED",
}
"""Venue incomeType (verbatim, upper-case) -> ledger kind.

Explicit table; anything absent maps to UNCLASSIFIED and is reported as an issue so new venue
types are noticed instead of being silently folded into another bucket.
"""

_LEGACY_INCOME_TYPES: frozenset[str] = frozenset({"REALIZED_PNL", "FUNDING_FEE", "COMMISSION", "TRANSFER"})


class TaxLedgerCorruptError(DataIntegrityError):
    """A tax shard line other than an unterminated final line is unparseable or lacks record_id.

    Attributes:
        path: Shard path.
        line_number: 1-based line number of the first corrupt line.
    """

    def __init__(self, path: Path | str, line_number: int, detail: str = "") -> None:
        self.path = Path(path)
        self.line_number = int(line_number)
        message = f"tax shard corrupt: {self.path} line {self.line_number}"
        if detail:
            message += f": {detail}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class TaxWatermark:
    last_trade_id: dict[str, int]
    last_collected_at: pd.Timestamp | None
    """Income is known complete for event times <= this instant (UTC)."""


@dataclass(frozen=True, slots=True)
class TaxCollectionIssue:
    stream: str  # "trades:<SYMBOL>" | "income"
    stage: str   # "fetch" | "parse" | "classify" | "page_cap" | "retention_gap" | "id_conflict"
    detail: str


def classify_income_type(income_type: str) -> tuple[str, bool]:
    """Map a raw venue incomeType to a ledger kind.

    Args:
        income_type: Raw ``incomeType`` string from ``/fapi/v1/income``.

    Returns:
        ``(kind, known)``. ``known`` is False when the type is absent from ``INCOME_TYPE_KIND``; the
        kind is then ``"UNCLASSIFIED"``.

    Raises:
        DataIntegrityError: ``income_type`` is empty or not a string (the row cannot be attributed).
    """
    if not isinstance(income_type, str) or not income_type:
        raise DataIntegrityError(f"venue incomeType is not attributable: {income_type!r}")
    key = income_type.upper()
    if key in INCOME_TYPE_KIND:
        return INCOME_TYPE_KIND[key], True
    return "UNCLASSIFIED", False


def _report_issue(issues: list[TaxCollectionIssue] | None, stream: str, stage: str, detail: str) -> None:
    if issues is not None:
        issues.append(TaxCollectionIssue(stream=stream, stage=stage, detail=detail))


def _issue_detail(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _as_utc(value: Any) -> pd.Timestamp:
    # 호출부 계약상 항상 tz-aware; naive 는 TypeError 로 드러나야 한다(임의 UTC 가정 금지).
    return pd.Timestamp(value).tz_convert("UTC")


def _venue_field(entry: dict[str, Any], key: str, what: str) -> Any:
    if entry.get(key) is None:
        raise DataIntegrityError(f"{what} entry lacks {key}: {entry!r}")
    return entry[key]


def _venue_int(entry: dict[str, Any], key: str, what: str) -> int:
    raw = _venue_field(entry, key, what)
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise DataIntegrityError(f"{what} entry has unparseable {key}: {entry!r}") from exc


def _venue_decimal(entry: dict[str, Any], key: str, what: str) -> Decimal:
    raw = entry.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (str, int)):
        raise DataIntegrityError(f"{what} entry has unparseable {key}: {entry!r}")
    try:
        return parse_tax_decimal(raw, field=f"{what}.{key}")
    except DataIntegrityError as exc:
        raise DataIntegrityError(f"{what} entry has unparseable {key}: {entry!r}") from exc


def _parse_trade_row(entry: Any, sym: str) -> tuple[TaxRecord, int, str]:
    """Convert one ``/fapi/v1/userTrades`` entry; every field the tax record needs is required.

    Missing or unparseable venue fields raise instead of defaulting: a trade priced at 0 or stamped
    with the collection time would silently corrupt the tax ledger.
    """
    if not isinstance(entry, dict):
        raise DataIntegrityError(f"trade entry is not an object: {entry!r}")
    venue_id = _venue_int(entry, "id", "trade")
    price = _venue_decimal(entry, "price", "trade")
    qty = _venue_decimal(entry, "qty", "trade")
    quote_qty = _venue_decimal(entry, "quoteQty", "trade")
    fee = _venue_decimal(entry, "commission", "trade")
    realized_pnl = _venue_decimal(entry, "realizedPnl", "trade")
    event_time = pd.Timestamp(_venue_int(entry, "time", "trade"), unit="ms", tz="UTC")
    is_buyer = _venue_field(entry, "buyer", "trade")
    if not isinstance(is_buyer, bool):
        raise DataIntegrityError(f"trade entry has non-boolean buyer: {entry!r}")
    side = "BUY" if is_buyer else "SELL"
    fee_asset = entry.get("commissionAsset")
    if not isinstance(fee_asset, str) or not fee_asset:
        raise DataIntegrityError(f"trade entry lacks commissionAsset: {entry!r}")
    position_side = entry.get("positionSide")
    if not isinstance(position_side, str) or position_side != ONE_WAY_POSITION_SIDE:
        raise DataIntegrityError(
            f"trade entry positionSide {position_side!r} is not one-way "
            f"(expected {ONE_WAY_POSITION_SIDE!r}): {entry!r}"
        )
    symbol = str(entry.get("symbol") or sym)
    is_maker = bool(entry.get("maker", False))
    income_asset = fee_asset or "USDT"
    return (
        TaxRecord(
            record_id=f"venue:TRADE:{venue_id}",
            kind="TRADE",
            event_time=event_time,
            symbol=symbol,
            side=side,
            quantity=qty,
            price=price,
            quote_qty=quote_qty,
            fee=fee,
            fee_asset=fee_asset,
            realized_pnl=realized_pnl,
            income_asset=income_asset,
            is_maker=is_maker,
            venue_id=venue_id,
            source="venue",
            mode="",
            position_side=ONE_WAY_POSITION_SIDE,
        ),
        venue_id,
        symbol,
    )


def _parse_income_row(entry: Any, issues: list[TaxCollectionIssue] | None) -> tuple[TaxRecord, int, pd.Timestamp]:
    """Convert one ``/fapi/v1/income`` entry; returns (record, tran_id, event_time).

    ``tranId``, ``incomeType``, ``income`` and ``time`` are required; ``symbol`` is legitimately
    empty for account-level rows (transfers, rebates).
    """
    if not isinstance(entry, dict):
        raise DataIntegrityError(f"income entry is not an object: {entry!r}")
    tran_id = _venue_int(entry, "tranId", "income")
    raw_type = str(_venue_field(entry, "incomeType", "income"))
    kind, known = classify_income_type(raw_type)
    if not known and issues is not None:
        issues.append(
            TaxCollectionIssue(
                stream="income", stage="classify",
                detail=f"unknown incomeType {raw_type!r} tranId {tran_id}",
            )
        )
    raw_upper = raw_type.upper()
    inc = _venue_decimal(entry, "income", "income")
    event_time = pd.Timestamp(_venue_int(entry, "time", "income"), unit="ms", tz="UTC")
    asset = str(entry.get("asset") or "")
    symbol = str(entry.get("symbol") or "")
    return (
        TaxRecord(
            record_id=f"venue:{raw_upper}:{tran_id}",
            kind=kind,
            event_time=event_time,
            symbol=symbol,
            side="",
            quantity=Decimal(0),
            price=Decimal(0),
            quote_qty=Decimal(0),
            fee=Decimal(0),
            fee_asset="",
            realized_pnl=inc,
            income_asset=asset,
            is_maker=False,
            venue_id=tran_id,
            source="venue",
            mode="",
            income_type=raw_upper,
        ),
        tran_id,
        event_time,
    )


def _collect_trades(
    client: Any,
    symbols: Sequence[str],
    watermark: TaxWatermark,
    mode: str,
    *,
    now_ts: pd.Timestamp,
    trades_page_limit: int,
    max_pages: int,
    pages_used: list[int],
    issues: list[TaxCollectionIssue] | None,
) -> tuple[list[TaxRecord], dict[str, int]]:
    """Page userTrades per symbol by fromId until a short page arrives."""
    records: list[TaxRecord] = []
    new_last_trade: dict[str, int] = dict(watermark.last_trade_id)
    for sym in symbols:
        last_id = watermark.last_trade_id.get(sym)
        from_id: int | None = (last_id + 1) if last_id is not None else None
        parsed_max: int | None = None
        failed_min: int | None = None
        while True:
            if pages_used[0] >= max_pages:
                _report_issue(issues, f"trades:{sym}", "page_cap", f"page budget exhausted at fromId {from_id}")
                break
            try:
                if from_id is not None:
                    page = client.user_trades(sym, from_id=from_id, limit=trades_page_limit)
                else:
                    page = client.user_trades(sym, limit=trades_page_limit)
            except Exception as exc:  # noqa: BLE001 - fetch failure keeps the stream watermark
                _report_issue(issues, f"trades:{sym}", "fetch", _issue_detail(exc))
                break
            pages_used[0] += 1
            if not isinstance(page, list):
                _report_issue(issues, f"trades:{sym}", "fetch", "non-list payload")
                break
            page_failed = False
            for entry in page:
                try:
                    rec, venue_id, _ = _parse_trade_row(entry, sym)
                except DataIntegrityError as exc:
                    page_failed = True
                    # id 를 알 수 있으면 그 직전까지만 워터마크를 전진시켜 다음 주기에 재수집한다.
                    with contextlib.suppress(TypeError, ValueError, AttributeError):
                        failed_id = int(entry["id"])
                        failed_min = failed_id if failed_min is None else min(failed_min, failed_id)
                    _report_issue(issues, f"trades:{sym}", "parse", _issue_detail(exc))
                    continue
                # fromId 페이징이라 같은 id 가 두 번 오지 않는다(venue 계약).
                records.append(_replace(rec, mode=str(mode)))
                parsed_max = venue_id if parsed_max is None else max(parsed_max, venue_id)
            # 짧은 페이지 = 소진. 파싱 실패가 섞인 페이지는 다음 fromId 를 신뢰할 수 없어 멈춘다.
            if len(page) < trades_page_limit or page_failed or parsed_max is None:
                break
            from_id = parsed_max + 1
        if failed_min is not None:
            new_last_trade[sym] = max(new_last_trade.get(sym, -1), failed_min - 1)
        elif parsed_max is not None:
            new_last_trade[sym] = max(new_last_trade.get(sym, -1), parsed_max)
    return records, new_last_trade


def _collect_income(
    client: Any,
    watermark: TaxWatermark,
    mode: str,
    *,
    now_ts: pd.Timestamp,
    income_page_limit: int,
    income_window: pd.Timedelta,
    income_overlap: pd.Timedelta,
    income_retention: pd.Timedelta,
    max_pages: int,
    pages_used: list[int],
    issues: list[TaxCollectionIssue] | None,
) -> tuple[list[TaxRecord], pd.Timestamp | None, pd.Timestamp | None]:
    """Read income in bounded [startTime, endTime] windows with overlap and paging.

    Returns (records, complete_through, conflict_cap). complete_through is the last instant known
    complete; None when nothing completed. conflict_cap bounds the watermark on id conflicts.
    """
    records: list[TaxRecord] = []
    # 창 겹침 재조회에서 같은 record_id 가 다시 오므로 내용 서명으로 중복/충돌을 가른다.
    seen_ids: dict[str, tuple[str, Decimal, str, str]] = {}
    retention_floor = now_ts - income_retention
    if watermark.last_collected_at is not None:
        start0 = _as_utc(watermark.last_collected_at) - income_overlap
        if _as_utc(watermark.last_collected_at) < retention_floor:
            _report_issue(
                issues, "income", "retention_gap",
                f"uncovered [{_as_utc(watermark.last_collected_at).isoformat()}..{retention_floor.isoformat()}]",
            )
            start0 = retention_floor
    else:
        start0 = retention_floor
    if start0 > now_ts:
        start0 = now_ts
    complete_through: pd.Timestamp | None = None
    conflict_cap: pd.Timestamp | None = None
    window_start = start0
    while window_start < now_ts:
        window_end = min(window_start + income_window, now_ts)
        page_start = window_start
        window_failed = False
        prev_last_ms: int | None = None
        while True:
            if pages_used[0] >= max_pages:
                _report_issue(issues, "income", "page_cap", f"page budget exhausted at {page_start.isoformat()}")
                # 다음 페이지는 page_start(포함)부터 이어지므로 그 직전까지만 완결로 확정한다.
                capped = page_start - pd.Timedelta(milliseconds=1)
                if complete_through is None or capped > complete_through:
                    complete_through = capped
                return records, complete_through, conflict_cap
            try:
                page = client.income(
                    start_time_ms=int(page_start.timestamp() * 1000),
                    end_time_ms=int(window_end.timestamp() * 1000),
                    limit=income_page_limit,
                )
            except Exception as exc:  # noqa: BLE001 - fetch failure leaves the watermark at the window start
                _report_issue(issues, "income", "fetch", _issue_detail(exc))
                window_failed = True
                break
            pages_used[0] += 1
            if not isinstance(page, list):
                _report_issue(issues, "income", "fetch", "non-list payload")
                window_failed = True
                break
            if not page:
                break
            page_last: pd.Timestamp | None = None
            for entry in page:
                try:
                    rec, _tran_id, event_time = _parse_income_row(entry, issues)
                except DataIntegrityError as exc:
                    _report_issue(issues, "income", "parse", _issue_detail(exc))
                    window_failed = True
                    continue
                page_last = event_time if page_last is None else max(page_last, event_time)
                rec = _replace(rec, mode=str(mode))
                signature = (rec.symbol, rec.realized_pnl, rec.event_time.isoformat(), rec.income_asset)
                if rec.record_id in seen_ids:
                    if seen_ids[rec.record_id] != signature:
                        _report_issue(issues, "income", "id_conflict", f"conflicting {rec.record_id}")
                        if conflict_cap is None or event_time < conflict_cap:
                            conflict_cap = event_time
                    continue
                seen_ids[rec.record_id] = signature
                records.append(rec)
            if page_last is None:
                break
            last_ms = int(page_last.timestamp() * 1000)
            start_ms = int(page_start.timestamp() * 1000)
            if len(page) >= income_page_limit and (last_ms == start_ms or (prev_last_ms is not None and last_ms <= prev_last_ms)):
                _report_issue(issues, "income", "page_cap", f"page stalled at {page_last.isoformat()}")
                window_failed = True
                break
            prev_last_ms = last_ms
            if len(page) < income_page_limit:
                break
            page_start = page_last
        if window_failed:
            break
        complete_through = window_end
        window_start = window_end
    else:
        # 모든 창을 소진(창이 0개인 미래 워터마크 포함) = now 까지 완결.
        complete_through = now_ts
    return records, complete_through, conflict_cap


def collect_tax_records(
    client: Any,
    symbols: Sequence[str],
    watermark: TaxWatermark,
    mode: str,
    *,
    now: pd.Timestamp,
    income_page_limit: int,
    trades_page_limit: int,
    income_window: pd.Timedelta,
    income_overlap: pd.Timedelta,
    income_retention: pd.Timedelta,
    max_pages: int,
    issues: list[TaxCollectionIssue] | None = None,
) -> tuple[tuple[TaxRecord, ...], TaxWatermark]:
    """Pull every venue trade and income row newer than the watermark, paging until exhausted.

    Income is read in explicit ``[startTime, endTime]`` windows from ``last_collected_at -
    income_overlap`` up to ``now``; each window is paged by advancing ``startTime`` to the last row's
    time (inclusive, ties resolved by record_id) until a page is shorter than the limit. userTrades is
    paged per symbol by ``fromId``. Deduplication is by record_id only; no scalar id comparison is
    applied across income types. The returned watermark never moves past data that was actually
    fetched and parsed: a fetch failure, parse failure, page cap or stalled page leaves
    ``last_collected_at`` at the last instant known complete.

    Args:
        now: Wall-clock UTC upper bound of this collection (never the decision time).
        income_page_limit / trades_page_limit: Venue page sizes.
        income_window: Width of one bounded income request window.
        income_overlap: Re-read margin behind the watermark.
        income_retention: Venue income retention; a watermark older than ``now - income_retention``
            is an unrecoverable gap.
        max_pages: Total page budget across income and trades for this call.

    Returns:
        (records sorted by event_time, new watermark).
    """
    now_ts = _as_utc(now)
    collected: list[TaxCollectionIssue] = []
    pages_used = [0]
    trade_records, new_last_trade = _collect_trades(
        client, symbols, watermark, mode, now_ts=now_ts,
        trades_page_limit=trades_page_limit, max_pages=max_pages,
        pages_used=pages_used, issues=collected,
    )
    income_records, complete_through, conflict_cap = _collect_income(
        client, watermark, mode, now_ts=now_ts,
        income_page_limit=income_page_limit, income_window=income_window,
        income_overlap=income_overlap, income_retention=income_retention,
        max_pages=max_pages, pages_used=pages_used, issues=collected,
    )
    if issues is not None:
        issues.extend(collected)
    new_collected_at: pd.Timestamp | None = watermark.last_collected_at
    if complete_through is not None and (
        new_collected_at is None or complete_through > _as_utc(new_collected_at)
    ):
        # 겹침 재조회 구간은 이미 완결이므로 워터마크는 절대 뒤로 가지 않는다.
        new_collected_at = complete_through
    if new_collected_at is not None and _as_utc(new_collected_at) > now_ts:
        new_collected_at = now_ts
    if conflict_cap is not None and new_collected_at is not None and new_collected_at > conflict_cap:
        new_collected_at = conflict_cap
    new_watermark = TaxWatermark(
        last_trade_id=new_last_trade,
        last_collected_at=new_collected_at,
    )
    records_sorted = sorted(trade_records + income_records, key=lambda r: r.event_time)
    return tuple(records_sorted), new_watermark


def _sim_decimal(value: Any, fill_id: str, what: str) -> Decimal:
    """Exact Decimal for a simulated fill amount; a float would inject binary rounding."""
    if isinstance(value, bool) or not isinstance(value, (Decimal, int)):
        raise DataIntegrityError(f"simulated fill {fill_id} has non-decimal {what}: {value!r}")
    amount = value if isinstance(value, Decimal) else Decimal(value)
    if not amount.is_finite():
        raise DataIntegrityError(f"simulated fill {fill_id} has non-finite {what}")
    return amount


def simulated_tax_records(fill_events: Sequence[FillEvent], mode: str) -> tuple[TaxRecord, ...]:
    """Build PAPER/SHADOW simulated TRADE tax records from persisted fill events.

    The record identity is the fill's durable journal identity, so a retried attempt of the same
    decision day (interrupted days are resumed) never reuses the id of an earlier attempt's fill
    and re-emitting a journal range after a crash is deduplicated instead of double-booked.

    Args:
        fill_events: Fill events carrying ``fill_id`` (``journal:<fill_seq>``), Decimal signed
            ``quantity_delta``, Decimal positive finite ``fill_price``, ``fee_bps`` and a tz-aware
            ``timestamp``.
        mode: Execution mode label stored on each record.

    Returns:
        One TRADE record per fill event with a nonzero quantity delta, sorted by event time. The
        fee is ``quantity * price * Decimal(str(fee_bps)) / 10000``, the exact formula of the paper
        cash ledger, so the per-batch cash reconciliation of these records has zero difference.

    Raises:
        DataIntegrityError: a fill lacks ``fill_id``, or its price is not finite and positive, its
            quantity delta is not finite, or its timestamp is missing or unparseable.
    """
    records: list[TaxRecord] = []
    for ev in fill_events:
        fill_id = getattr(ev, "fill_id", None)
        if not isinstance(fill_id, str) or not fill_id.startswith("journal:") or not fill_id[8:].isdigit():
            raise DataIntegrityError(f"simulated fill lacks a journal fill_id: {ev!r}")
        qty_delta = _sim_decimal(ev.quantity_delta, fill_id, "quantity delta")
        if qty_delta == 0:
            continue
        price = _sim_decimal(ev.fill_price, fill_id, "price")
        if price <= 0:
            raise DataIntegrityError(f"simulated fill {fill_id} has non-positive price {price!r}")
        try:
            fee_rate = Decimal(str(ev.fee_bps))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise DataIntegrityError(f"simulated fill {fill_id} has invalid fee_bps {ev.fee_bps!r}") from exc
        if not fee_rate.is_finite() or fee_rate < 0:
            raise DataIntegrityError(f"simulated fill {fill_id} has invalid fee_bps {ev.fee_bps!r}")
        # NaT(누락)·naive 모두 tzinfo 가 없어 여기서 거부된다.
        event_time = pd.Timestamp(ev.timestamp)
        if event_time.tzinfo is None:
            raise DataIntegrityError(f"simulated fill {fill_id} has missing or naive timestamp")
        event_time = event_time.tz_convert("UTC")
        quantity = abs(qty_delta)
        side = "BUY" if qty_delta > 0 else "SELL"
        quote_qty = quantity * price
        fee = quote_qty * fee_rate / Decimal(10_000)
        fee_asset = "USDT"
        symbol = str(ev.symbol)
        is_maker = str(ev.liquidity) == "maker"
        venue_id = int(fill_id[8:])
        rec = TaxRecord(
            record_id=f"simulated:TRADE:{fill_id}",
            kind="TRADE",
            event_time=event_time,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            quote_qty=quote_qty,
            fee=fee,
            fee_asset=fee_asset,
            realized_pnl=Decimal(0),
            income_asset=fee_asset,
            is_maker=is_maker,
            venue_id=venue_id,
            source="simulated",
            mode=str(mode),
        )
        records.append(rec)
    records.sort(key=lambda r: r.event_time)
    return tuple(records)


def funding_tax_records(events: Sequence[Any], *, run_id: str, mode: str) -> tuple[TaxRecord, ...]:
    """Map paper funding events to simulated FUNDING_FEE tax records with deterministic ids.

    The id is "simulated:FUNDING_FEE:<run_id>:<symbol>:<epoch_ms>". A settlement accrues at most once
    per symbol and epoch within a run, so replaying the same accrual after a crash yields the same ids.
    """
    records: list[TaxRecord] = []
    for ev in events:
        epoch = pd.Timestamp(ev.epoch)
        epoch = epoch.tz_localize("UTC") if epoch.tzinfo is None else epoch.tz_convert("UTC")
        epoch_ms = int(epoch.timestamp() * 1000)
        quantity = _sim_decimal(ev.quantity, str(ev.symbol), "quantity")
        price = _sim_decimal(ev.price, str(ev.symbol), "price")
        amount = _sim_decimal(ev.amount, str(ev.symbol), "amount")
        records.append(
            TaxRecord(
                record_id=f"simulated:FUNDING_FEE:{run_id}:{ev.symbol}:{epoch_ms}",
                kind="FUNDING_FEE",
                event_time=epoch,
                symbol=str(ev.symbol),
                side="",
                quantity=quantity,
                price=price,
                quote_qty=quantity * price,
                fee=Decimal(0),
                fee_asset="USDT",
                realized_pnl=amount,
                income_asset="USDT",
                is_maker=False,
                venue_id=0,
                source="simulated",
                mode=str(mode),
            )
        )
    return tuple(records)


def _tax_shard_path(ledger_dir: Path, event_time: pd.Timestamp) -> Path:
    ts = pd.Timestamp(event_time)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return Path(ledger_dir) / f"tax_ledger_{ts.strftime('%Y%m')}.jsonl"


def _read_shard_lines(shard: Path) -> tuple[list[tuple[int, dict[str, Any]]], bytes | None]:
    """Read one shard's complete rows and detect a torn tail.

    Returns (rows, torn_prefix). rows holds (1-based line number, parsed object) for every
    newline-terminated line. torn_prefix is the raw bytes of an unterminated final line, or None.
    A line counts as torn only if it is the last line and the file does not end with ``\\n``.

    Raises:
        TaxLedgerCorruptError: a complete line is not valid JSON, holds a non-finite JSON
            constant, or lacks record_id.
    """
    raw = shard.read_bytes() if shard.exists() else b""
    if not raw:
        return [], None
    torn_prefix: bytes | None = None
    body = raw
    if not raw.endswith(b"\n"):
        head, _, tail = raw.rpartition(b"\n")
        torn_prefix = tail
        body = head + (b"\n" if head else b"")
    rows: list[tuple[int, dict[str, Any]]] = []

    def _reject(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {value}")

    for lineno, line in enumerate(body.split(b"\n")[:-1] if body else [], start=1):
        if not line.strip():
            continue
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TaxLedgerCorruptError(shard, lineno, "not utf-8") from exc
        try:
            obj = json.loads(text, parse_float=Decimal, parse_constant=_reject)
        except ValueError as exc:
            reason = "non-finite JSON constant" if "non-finite JSON constant" in str(exc) else "not valid JSON"
            raise TaxLedgerCorruptError(shard, lineno, reason) from exc
        if not isinstance(obj, dict) or not obj.get("record_id"):
            raise TaxLedgerCorruptError(shard, lineno, "lacks record_id")
        rows.append((lineno, obj))
    return rows, torn_prefix


def _partition_fresh_tax_records(records: Sequence[TaxRecord], directory: Path) -> dict[Path, list[TaxRecord]]:
    """Bucket records by destination shard and drop ids already present (verified first).

    An unterminated final line is a torn tail, not data: it is ignored here and truncated by
    :func:`append_tax_records` before the next append.

    Raises:
        TaxLedgerCorruptError: an existing shard line other than a torn tail is not valid JSON
            or lacks record_id.
    """
    buckets: dict[Path, list[TaxRecord]] = {}
    for r in records:
        buckets.setdefault(_tax_shard_path(directory, r.event_time), []).append(r)
    # Verify first: load existing ids for every touched shard before deciding anything.
    existing_by_shard: dict[Path, set[str]] = {}
    for shard in sorted(buckets):
        seen: set[str] = set()
        rows, _torn = _read_shard_lines(shard)
        for _lineno, obj in rows:
            seen.add(str(obj["record_id"]))
        existing_by_shard[shard] = seen
    fresh: dict[Path, list[TaxRecord]] = {}
    for shard in sorted(buckets):
        seen = existing_by_shard[shard]
        batch_seen: set[str] = set()
        fresh_rows: list[TaxRecord] = []
        for r in buckets[shard]:
            if r.record_id in seen or r.record_id in batch_seen:
                continue
            batch_seen.add(r.record_id)
            fresh_rows.append(r)
        if fresh_rows:
            fresh[shard] = fresh_rows
    return fresh


def _fsync_dir(dir_path: Path) -> None:
    fd = os.open(str(dir_path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append_tax_records(records: Sequence[TaxRecord], ledger_dir: Path) -> list[Path]:
    """Append tax records to monthly JSONL shards, skipping record_ids already present.

    Idempotent by record_id. An unterminated final line (a write torn by a kill, ENOSPC or a backup
    copied mid-append) is not data: it is ignored when collecting existing ids and truncated before
    the next append, mirroring ``OrderJournal``. Every source that feeds this function regenerates
    identical record_ids on replay (deterministic simulated ids, venue ids re-fetched because the
    watermark is saved only after the append), so dropping a torn row loses nothing. Each shard's new
    rows are written, flushed and fsynced before the function moves on; a newly created shard also
    fsyncs its directory entry.

    Every record is validated with ``validate_tax_record`` before any shard is read or written;
    a single invalid record aborts the whole call with nothing written.

    Returns:
        Shard paths that received at least one new row.

    Raises:
        DataIntegrityError: a record fails ``validate_tax_record``.
        TaxLedgerCorruptError: a complete (newline-terminated) line, or a non-final line, is not
            valid JSON or lacks record_id. Nothing is appended to any shard in that case.
    """
    if not records:
        return []
    for record in records:
        validate_tax_record(record)
    directory = Path(ledger_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fresh = _partition_fresh_tax_records(records, directory)
    # Truncate torn tails only after every touched shard verified clean above.
    for shard in sorted(fresh):
        _rows, torn = _read_shard_lines(shard)
        if torn is not None:
            truncate_durably(shard, shard.stat().st_size - len(torn))
    written: list[Path] = []
    for shard in sorted(fresh):
        is_new = not shard.exists()
        payload = "".join(json.dumps(tax_record_to_row(r), ensure_ascii=False) + "\n" for r in fresh[shard])
        with shard.open("a", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        if is_new:
            _fsync_dir(shard.parent)
        written.append(shard)
    return sorted(written)


def read_tax_ledger(ledger_dir: Path | str) -> tuple[TaxRecord, ...]:
    """Load every tax record in ``ledger_dir`` as validated, Decimal-typed records.

    The yearly summary folds positions from inception, so this always loads all shards; there is
    no year filter. Read-only: an unterminated final line (torn tail) is ignored, never
    truncated. Only ``tax_ledger_YYYYMM.jsonl`` shards are read.

    A record_id present more than once with byte-identical rebuilt content is kept once; the same
    record_id with different content is corruption, because append is idempotent by record_id and
    a second, different row can only come from tampering or a producer bug.

    Args:
        ledger_dir: Ledger directory. Required: callers resolve it from settings so that a summary
            can never silently read a default directory that belongs to another run.

    Returns:
        Records sorted by ``tax_event_sort_key``. Empty tuple when the directory is absent or has
        no shards.

    Raises:
        TaxLedgerCorruptError: a complete line is not valid JSON, lacks record_id, holds a
            non-finite constant, fails ``tax_record_from_row`` (detail carries the reason), or
            conflicts with an earlier row of the same record_id. ``path``/``line_number`` point at
            the offending line.
    """
    dir_path = Path(ledger_dir)
    if not dir_path.exists():
        return ()
    shards = sorted(dir_path.glob("tax_ledger_*.jsonl"))
    if not shards:
        return ()
    seen: dict[str, TaxRecord] = {}
    for shard in shards:
        rows, _torn = _read_shard_lines(shard)
        for lineno, obj in rows:
            rid = str(obj["record_id"])
            try:
                record = tax_record_from_row(obj)
            except DataIntegrityError as exc:
                raise TaxLedgerCorruptError(shard, lineno, str(exc)) from exc
            if rid in seen:
                if tax_record_to_row(seen[rid]) != tax_record_to_row(record):
                    raise TaxLedgerCorruptError(shard, lineno, f"conflicting duplicate record_id {rid!r}")
                continue
            seen[rid] = record
    return tuple(sorted(seen.values(), key=tax_event_sort_key))


def load_tax_watermark(path: Path) -> TaxWatermark:
    """Read watermark.json; an absent file means an empty watermark.

    A legacy ``last_income_id`` key is ignored: income deduplication is by ``record_id`` only.

    Raises:
        DataIntegrityError: file exists but is not valid JSON or has malformed fields (fail-closed).
    """
    watermark_path = Path(path)
    if not watermark_path.exists():
        return TaxWatermark(last_trade_id={}, last_collected_at=None)
    try:
        raw = json.loads(watermark_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataIntegrityError(f"tax watermark unreadable: {watermark_path}") from exc
    if not isinstance(raw, dict):
        raise DataIntegrityError(f"tax watermark must be a JSON object: {watermark_path}")
    try:
        last_trade_id = {str(k): int(v) for k, v in dict(raw.get("last_trade_id", {})).items()}
        collected_raw = raw.get("last_collected_at")
        if collected_raw is None:
            last_collected_at = None
        else:
            last_collected_at = pd.Timestamp(collected_raw)
            if last_collected_at.tzinfo is None:
                last_collected_at = last_collected_at.tz_localize("UTC")
            else:
                last_collected_at = last_collected_at.tz_convert("UTC")
    except (ValueError, TypeError, AttributeError) as exc:
        raise DataIntegrityError(f"tax watermark malformed: {watermark_path}") from exc
    return TaxWatermark(
        last_trade_id=last_trade_id,
        last_collected_at=last_collected_at,
    )


def save_tax_watermark(path: Path, watermark: TaxWatermark) -> None:
    """Persist watermark.json crash-safely: temp file written, flushed and fsynced, os.replace, then
    the directory fsynced, so a power loss leaves either the old or the new watermark, never an
    empty file."""
    watermark_path = Path(path)
    watermark_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_trade_id": dict(watermark.last_trade_id),
        "last_collected_at": watermark.last_collected_at.isoformat()
        if watermark.last_collected_at is not None
        else None,
    }
    durable_write_text(watermark_path, json.dumps(payload, sort_keys=True))


def collect_and_persist_live_tax(
    client: Any,
    symbols: Sequence[str],
    tax_dir: Path,
    mode: str,
    *,
    now: pd.Timestamp,
    settings: Any,
) -> tuple[int, tuple[TaxCollectionIssue, ...]]:
    """Load watermark, collect, append (durably), then save the watermark.

    Raises:
        DataIntegrityError: watermark unreadable.
        TaxLedgerCorruptError: a shard is corrupt beyond a torn tail. Nothing is written and the
            watermark is not saved.
    """
    directory = Path(tax_dir)
    directory.mkdir(parents=True, exist_ok=True)
    watermark = load_tax_watermark(directory / "watermark.json")
    found: list[TaxCollectionIssue] = []
    records, new_watermark = collect_tax_records(
        client,
        symbols,
        watermark,
        mode,
        now=now,
        income_page_limit=int(settings.tax_income_page_limit),
        trades_page_limit=int(settings.tax_trades_page_limit),
        income_window=pd.Timedelta(days=int(settings.tax_income_window_days)),
        income_overlap=pd.Timedelta(seconds=float(settings.tax_income_overlap_s)),
        income_retention=pd.Timedelta(days=int(settings.tax_income_retention_days)),
        max_pages=int(settings.tax_max_pages_per_cycle),
        issues=found,
    )
    new_rows = sum(len(rows) for rows in _partition_fresh_tax_records(records, directory).values())
    append_tax_records(records, directory)
    save_tax_watermark(directory / "watermark.json", new_watermark)
    return new_rows, tuple(found)


@dataclass(frozen=True, slots=True)
class CashReconciliation:
    expected_delta: Decimal
    actual_delta: Decimal
    difference: Decimal
    within_tolerance: bool


def reconcile_cycle_cash(
    cash_before: Decimal,
    cash_after: Decimal,
    trade_records: Sequence[TaxRecord],
    funding_records: Sequence[TaxRecord],
    *,
    tolerance_usdt: Decimal,
) -> CashReconciliation:
    """Check that one paper cycle's cash change equals the cash implied by its own records.

    expected = sum(funding.realized_pnl) - sum(signed trade quote flow) - sum(trade.fee), where a BUY consumes
    quote_qty and a SELL returns it. A mismatch means the ledger moved cash that no record explains.
    """
    expected = Decimal(0)
    for r in funding_records:
        if r.kind != "FUNDING_FEE":
            raise ValueError(f"funding_records must all be FUNDING_FEE, got {r.kind!r}")
        expected += r.realized_pnl
    for r in trade_records:
        if r.kind != "TRADE":
            raise ValueError(f"trade_records must all be TRADE, got {r.kind!r}")
        quote = r.quote_qty
        fee = r.fee
        if r.side == "BUY":
            expected += -quote - fee
        elif r.side == "SELL":
            expected += quote - fee
        elif r.side == "" and quote == 0:
            expected += -fee
        else:
            raise ValueError(f"trade record side must be BUY or SELL, got {r.side!r}")
    actual = Decimal(cash_after) - Decimal(cash_before)
    difference = actual - expected
    return CashReconciliation(
        expected_delta=expected,
        actual_delta=actual,
        difference=difference,
        within_tolerance=abs(difference) <= tolerance_usdt,
    )


def default_tax_ledger_dir() -> Path:
    return DATA_DIR / "state" / "live_tax_ledger"
