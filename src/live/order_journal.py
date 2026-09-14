"""Live order write-ahead journal: lifetime-unique submit_seq and observed fills.

INV-CLIENT-ID-ONCE: every mutating newClientOrderId embeds a journal submit_seq
recorded (fsync) before the send; a submit_seq is never reused across processes.
INV-FILL-DELTA: orphan settlement quantity = exchange executedQty - journal
observed qty for that client id. Crash between journal write and ledger save
under-books (reconciliation HALT, fail-closed), never double-books.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.common.errors import DataIntegrityError
from src.common.paths import DATA_DIR


def default_order_journal_path() -> Path:
    """Default journal location under data/state."""
    return DATA_DIR / "state" / "live_order_journal.jsonl"


class OrderJournal:
    """Append-only JSONL journal of order submissions and observed fills.

    Construction performs no I/O; the file is lazily loaded on first use.
    A torn tail (unparseable last line without trailing newline, i.e. a crash
    mid-write before any send) is ignored on load and truncated before the
    next append. Any other corruption fails closed with DataIntegrityError.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._loaded = False
        self._next_seq = 0
        self._observed: dict[str, Decimal] = {}
        self._torn_offset: int | None = None

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._path.exists():
            return
        raw = self._path.read_text(encoding="utf-8")
        ends_with_newline = raw.endswith("\n")
        lines = raw.splitlines()
        for index, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) and not ends_with_newline:
                    self._torn_offset = raw.rfind("\n") + 1
                    return
                raise DataIntegrityError(
                    f"order journal corrupt at line {index}: {self._path}"
                ) from None
            event = record.get("event") if isinstance(record, dict) else None
            if event == "submit":
                seq = int(record["submit_seq"])
                if seq + 1 > self._next_seq:
                    self._next_seq = seq + 1
            elif event == "observed":
                order_id = str(record["client_order_id"])
                qty = Decimal(str(record["executed_qty"]))
                if qty > self._observed.get(order_id, Decimal(0)):
                    self._observed[order_id] = qty
            else:
                raise DataIntegrityError(
                    f"order journal unknown event at line {index}: {self._path}"
                )

    def next_submit_seq(self) -> int:
        self._ensure_loaded()
        return self._next_seq

    def record_submit(self, client_order_id: str, symbol: str, submit_seq: int) -> None:
        self._ensure_loaded()
        if submit_seq != self._next_seq:
            raise ValueError(
                f"submit_seq {submit_seq} does not match next journal sequence {self._next_seq}"
            )
        self._append(
            {
                "event": "submit",
                "client_order_id": client_order_id,
                "symbol": symbol,
                "submit_seq": submit_seq,
            }
        )
        self._next_seq = submit_seq + 1

    def record_observed(self, client_order_id: str, executed_qty: Decimal) -> None:
        self._ensure_loaded()
        if executed_qty <= self._observed.get(client_order_id, Decimal(0)):
            return
        self._observed[client_order_id] = executed_qty
        self._append(
            {
                "event": "observed",
                "client_order_id": client_order_id,
                "executed_qty": str(executed_qty),
            }
        )

    def observed_qty(self, client_order_id: str) -> Decimal:
        self._ensure_loaded()
        return self._observed.get(client_order_id, Decimal(0))

    def _append(self, record: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._torn_offset is not None:
            raw = self._path.read_text(encoding="utf-8")
            self._path.write_text(raw[: self._torn_offset], encoding="utf-8")
            self._torn_offset = None
        line = json.dumps({**record, "ts": datetime.now(UTC).isoformat()}, sort_keys=True) + "\n"
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
