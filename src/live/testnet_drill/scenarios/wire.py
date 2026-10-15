"""Wire-log, journal and venue helpers shared by the fault, restart and account-state scenario groups."""

from __future__ import annotations

import json
from collections.abc import Sequence
from decimal import Decimal
from typing import Any
from urllib.parse import parse_qs, urlsplit

from src.live.testnet_drill.execution import AuditTap
from src.live.testnet_drill.faults import WireRecord

ORDER_PATH = "/fapi/v1/order"
BOOK_TICKER_PATH = "/fapi/v1/ticker/bookTicker"


def param(record: WireRecord, key: str) -> str | None:
    values = parse_qs(urlsplit(record.url).query).get(key)
    return values[0] if values else None


def order_id(record: WireRecord) -> str | None:
    return param(record, "newClientOrderId") or param(record, "origClientOrderId")


def records(session: Any, start: int, end: int | None = None) -> list[WireRecord]:
    return list(session.transport.records[start:end])


def mutations(records: Sequence[WireRecord], method: str) -> list[WireRecord]:
    return [r for r in records if r.method == method and r.path == ORDER_PATH]


def body(record: WireRecord) -> Any:
    try:
        return json.loads(record.response_body)
    except (ValueError, UnicodeError):
        return None


def code(record: WireRecord) -> int | None:
    payload = body(record)
    if isinstance(payload, dict) and isinstance(payload.get("code"), int):
        return int(payload["code"])
    return None


def event_for(tap: AuditTap, names: frozenset[str] | set[str], order_id: str | None) -> Any:
    return next((e for e in tap.events if e.event in names and e.fields.get("client_order_id") == order_id), None)


def signed_fold(session: Any, symbol: str) -> Decimal:
    return sum(
        (
            f.quantity if f.side == "BUY" else -f.quantity
            for f in session.journal.fills_after(-1)
            if f.symbol == symbol and f.kind in ("execution", "recovered", "orphan_settlement")
        ),
        Decimal(0),
    )


def venue_position(session: Any, symbol: str) -> Decimal:
    from src.live.testnet_drill.session import _read_positions

    return _read_positions(session.client).get(symbol, Decimal(0))


def fills_unique(session: Any) -> bool:
    keys = [
        (f.client_order_id, f.cumulative_executed_qty)
        for f in session.journal.fills_after(-1)
        if f.client_order_id is not None
    ]
    return len(keys) == len(set(keys))


def executed(session: Any, symbol: str, order_id: str) -> Decimal:
    return Decimal(str(session.client.query_order(symbol, order_id)["executedQty"]))


def journal_terminal_lines(session: Any, order_id: str) -> int:
    count = 0
    for line in session.journal.path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("event") == "terminal" and row.get("client_order_id") == order_id:
            count += 1
    return count


def resting_params(session: Any) -> dict[str, Any]:
    from src.live.testnet_drill.scenarios.lifecycle_runs import _limit

    return _limit(session, distance=True)
