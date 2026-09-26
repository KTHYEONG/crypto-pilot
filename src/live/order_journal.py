# ruff: noqa: RUF002 -- docstrings use the U+2212 MINUS SIGN verbatim.
"""Live order write-ahead journal: submissions, attempts, fills and terminal states.

INV-CLIENT-ID-ONCE: every mutating newClientOrderId embeds a journal submit_seq recorded (fsync) before the send; a submit_seq is never reused across processes.
INV-FILL-WAL: every executed quantity (live, paper, orphan, recovered, venue- or operator-adjusted) is appended here as a `fill` record with a lifetime-unique `fill_seq` before any in-memory or ledger state reflects it. The ledger applies fills strictly by `fill_seq` watermark, so replay after any crash is idempotent and never double-books.
INV-FILL-DELTA: orphan/recovery settlement quantity = venue executedQty − journal observed qty for that client id, where observed qty is the maximum `cumulative_executed_qty` (or legacy `observed` line) recorded for the id.
Schema v2 is marked by a `schema` line; submits written before the last marker (schema v1) carry no side/quantity and are excluded from restart recovery.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.paths import DATA_DIR

FillKind = Literal[
    "execution",
    "orphan_settlement",
    "recovered",
    "venue_force_close",
    "operator_resync",
]

_FILL_KINDS: frozenset[str] = frozenset(
    {
        "execution",
        "orphan_settlement",
        "recovered",
        "venue_force_close",
        "operator_resync",
    }
)
_SIDES: frozenset[str] = frozenset({"BUY", "SELL"})
_LIQUIDITY: frozenset[str] = frozenset({"maker", "taker"})

JOURNAL_SCHEMA_VERSION: int = 2


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(UTC))


def _iso(ts: pd.Timestamp) -> str:
    # 모든 기록 경로가 tz-aware 를 먼저 검증하므로 naive 는 여기서 TypeError 로 드러난다.
    return str(pd.Timestamp(ts).tz_convert("UTC").isoformat())


def _parse_ts(value: Any, *, line: int, path: Path) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataIntegrityError(
            f"order journal bad timestamp at line {line}: {path}"
        ) from exc
    if stamp.tzinfo is None:
        raise DataIntegrityError(
            f"order journal naive timestamp at line {line}: {path}"
        )
    return stamp


def _parse_decimal(value: Any, *, line: int, path: Path) -> Decimal:
    try:
        return Decimal(str(value))
    except (ValueError, TypeError, ArithmeticError) as exc:
        raise DataIntegrityError(
            f"order journal bad decimal at line {line}: {path}"
        ) from exc


@dataclass(frozen=True, slots=True)
class JournalAttempt:
    """Context of one execution attempt, persisted before any order of the attempt is sent.

    Carries everything needed to rebuild fill evidence (fills parquet rows, simulated tax
    records) from the journal alone after a crash, without the in-memory runner state.
    """

    attempt_seq: int
    decision_time: pd.Timestamp  # tz-aware UTC decision label
    run_id: str
    mode: str
    pre_trade_equity: Decimal
    sizing_anchor: str
    decision_marks: Mapping[str, Decimal]
    started_at: pd.Timestamp  # tz-aware UTC wall clock


@dataclass(frozen=True, slots=True)
class JournalSubmit:
    """An order submission recorded (fsync) before the send."""

    client_order_id: str
    symbol: str
    submit_seq: int
    attempt_seq: int | None  # None only for legacy (schema v1) lines
    side: str | None  # 'BUY' | 'SELL'; None only for legacy lines
    quantity: Decimal | None
    reduce_only: bool | None
    leg_index: int | None
    recorded_at: pd.Timestamp


def truncate_durably(path: Path, size: int) -> None:
    """Cut ``path`` to ``size`` bytes in place and fsync it.

    Torn-tail repair must never pass through an empty file: a truncate-then-rewrite that is
    interrupted loses every committed line, while an in-place truncate either happened or not.
    """
    with path.open("r+b") as handle:
        handle.truncate(size)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True, slots=True)
class JournalFill:
    """One durable fill delta. The only source from which the ledger learns about executions."""

    fill_seq: int
    kind: FillKind
    attempt_seq: int | None
    symbol: str
    side: str  # 'BUY' | 'SELL'
    quantity: Decimal  # strictly positive delta
    price: Decimal  # strictly positive
    fee_bps: float  # >= 0
    liquidity: str  # 'maker' | 'taker'
    reason: str  # member of fills.FILL_REASONS, or 'operator_resync'
    filled_at: pd.Timestamp  # tz-aware UTC confirmation time
    client_order_id: str | None  # None only for 'operator_resync'
    leg_index: int
    cumulative_executed_qty: Decimal | None  # venue executedQty after this delta; None for paper/adjustments
    simulated: bool


@dataclass(frozen=True, slots=True)
class JournalTerminal:
    client_order_id: str
    status: str  # venue/terminal status label, e.g. FILLED, CANCELED, EXPIRED, NOT_PLACED
    recorded_at: pd.Timestamp


def default_order_journal_path() -> Path:
    """Default journal location under data/state."""
    return DATA_DIR / "state" / "live_order_journal.jsonl"


class OrderJournal:
    """Append-only JSONL journal of attempts, submissions, fills and terminal states.

    Construction performs no I/O; the file is lazily loaded on first use.
    A torn tail (unparseable last line without trailing newline, i.e. a crash
    mid-write before any send) is ignored on load and truncated before the
    next append. Any other corruption fails closed with DataIntegrityError.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._loaded = False
        self._next_submit_seq = 0
        self._next_fill_seq = 0
        self._next_attempt_seq = 0
        self._observed_legacy: dict[str, Decimal] = {}
        self._attempts: dict[int, JournalAttempt] = {}
        self._submits: list[JournalSubmit] = []
        self._fills: list[JournalFill] = []
        self._terminals: dict[str, JournalTerminal] = {}
        self._has_schema = False
        self._had_lines = False
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
        if any(line.strip() for line in lines):
            self._had_lines = True
        for index, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) and not ends_with_newline:
                    self._torn_offset = len(raw.encode("utf-8")) - len(line.encode("utf-8"))
                    return
                raise DataIntegrityError(
                    f"order journal corrupt at line {index}: {self._path}"
                ) from None
            try:
                self._apply_record(record, line=index)
            except DataIntegrityError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise DataIntegrityError(
                    f"order journal corrupt at line {index}: {self._path}"
                ) from exc

    def _apply_record(self, record: Any, *, line: int) -> None:
        if not isinstance(record, dict):
            raise DataIntegrityError(
                f"order journal corrupt at line {line}: {self._path}"
            )
        event = record.get("event")
        if event == "schema":
            self._has_schema = True
        elif event == "submit":
            seq = int(record["submit_seq"])
            if seq + 1 > self._next_submit_seq:
                self._next_submit_seq = seq + 1
            attempt_seq = record.get("attempt_seq")
            side = record.get("side")
            quantity_raw = record.get("quantity")
            quantity = (
                _parse_decimal(quantity_raw, line=line, path=self._path)
                if quantity_raw is not None
                else None
            )
            reduce_only = record.get("reduce_only")
            leg_index = record.get("leg_index")
            recorded_raw = record.get("recorded_at", record.get("ts"))
            if recorded_raw is None:
                recorded_at = pd.Timestamp.min.tz_localize("UTC")
            else:
                recorded_at = _parse_ts(recorded_raw, line=line, path=self._path)
            self._submits.append(
                JournalSubmit(
                    client_order_id=str(record["client_order_id"]),
                    symbol=str(record["symbol"]),
                    submit_seq=seq,
                    attempt_seq=int(attempt_seq) if attempt_seq is not None else None,
                    side=str(side) if side is not None else None,
                    quantity=quantity,
                    reduce_only=bool(reduce_only) if reduce_only is not None else None,
                    leg_index=int(leg_index) if leg_index is not None else None,
                    recorded_at=recorded_at,
                )
            )
        elif event == "observed":
            order_id = str(record["client_order_id"])
            qty = _parse_decimal(record["executed_qty"], line=line, path=self._path)
            if qty > self._observed_legacy.get(order_id, Decimal(0)):
                self._observed_legacy[order_id] = qty
        elif event == "attempt":
            seq = int(record["attempt_seq"])
            if seq + 1 > self._next_attempt_seq:
                self._next_attempt_seq = seq + 1
            marks_raw = record.get("decision_marks", {})
            if not isinstance(marks_raw, dict):
                raise DataIntegrityError(
                    f"order journal corrupt at line {line}: {self._path}"
                )
            marks = {
                str(sym): _parse_decimal(val, line=line, path=self._path)
                for sym, val in marks_raw.items()
            }
            self._attempts[seq] = JournalAttempt(
                attempt_seq=seq,
                decision_time=_parse_ts(record["decision_time"], line=line, path=self._path),
                run_id=str(record["run_id"]),
                mode=str(record["mode"]),
                pre_trade_equity=_parse_decimal(
                    record["pre_trade_equity"], line=line, path=self._path
                ),
                sizing_anchor=str(record["sizing_anchor"]),
                decision_marks=marks,
                started_at=_parse_ts(record["started_at"], line=line, path=self._path),
            )
        elif event == "fill":
            seq = int(record["fill_seq"])
            if seq + 1 > self._next_fill_seq:
                self._next_fill_seq = seq + 1
            cumulative_raw = record.get("cumulative_executed_qty")
            cumulative = (
                _parse_decimal(cumulative_raw, line=line, path=self._path)
                if cumulative_raw is not None
                else None
            )
            attempt_raw = record.get("attempt_seq")
            client_raw = record.get("client_order_id")
            kind_raw = str(record["kind"])
            if kind_raw not in _FILL_KINDS:
                raise DataIntegrityError(
                    f"order journal unknown fill kind at line {line}: {self._path}"
                )
            self._fills.append(
                JournalFill(
                    fill_seq=seq,
                    kind=cast(FillKind, kind_raw),
                    attempt_seq=int(attempt_raw) if attempt_raw is not None else None,
                    symbol=str(record["symbol"]),
                    side=str(record["side"]),
                    quantity=_parse_decimal(record["quantity"], line=line, path=self._path),
                    price=_parse_decimal(record["price"], line=line, path=self._path),
                    fee_bps=float(record["fee_bps"]),
                    liquidity=str(record["liquidity"]),
                    reason=str(record["reason"]),
                    filled_at=_parse_ts(record["filled_at"], line=line, path=self._path),
                    client_order_id=str(client_raw) if client_raw is not None else None,
                    leg_index=int(record["leg_index"]),
                    cumulative_executed_qty=cumulative,
                    simulated=bool(record["simulated"]),
                )
            )
        elif event == "terminal":
            order_id = str(record["client_order_id"])
            recorded_raw = record.get("recorded_at", record.get("ts"))
            recorded_at = (
                _utc_now()
                if recorded_raw is None
                else _parse_ts(recorded_raw, line=line, path=self._path)
            )
            self._terminals[order_id] = JournalTerminal(
                client_order_id=order_id,
                status=str(record["status"]),
                recorded_at=recorded_at,
            )
        else:
            raise DataIntegrityError(
                f"order journal unknown event at line {line}: {self._path}"
            )

    def begin_attempt(
        self,
        *,
        decision_time: pd.Timestamp,
        run_id: str,
        mode: str,
        pre_trade_equity: Decimal,
        sizing_anchor: str,
        decision_marks: Mapping[str, Decimal],
        started_at: pd.Timestamp,
    ) -> JournalAttempt:
        """Persist the context of one execution attempt and return it with a new lifetime-unique `attempt_seq`. Must be called before the first order or simulated fill of the attempt so every fill can be rebuilt into evidence rows from the journal alone."""
        self._ensure_loaded()
        for label, stamp in (("decision_time", decision_time), ("started_at", started_at)):
            if pd.Timestamp(stamp).tzinfo is None:
                raise ValueError(f"{label} must be tz-aware")
        seq = self._next_attempt_seq
        attempt = JournalAttempt(
            attempt_seq=seq,
            decision_time=pd.Timestamp(decision_time),
            run_id=run_id,
            mode=mode,
            pre_trade_equity=Decimal(pre_trade_equity),
            sizing_anchor=sizing_anchor,
            decision_marks={str(k): Decimal(v) for k, v in dict(decision_marks).items()},
            started_at=pd.Timestamp(started_at),
        )
        self._ensure_v2_schema()
        self._append(
            {
                "event": "attempt",
                "attempt_seq": seq,
                "decision_time": _iso(attempt.decision_time),
                "run_id": run_id,
                "mode": mode,
                "pre_trade_equity": str(attempt.pre_trade_equity),
                "sizing_anchor": sizing_anchor,
                "decision_marks": {k: str(v) for k, v in attempt.decision_marks.items()},
                "started_at": _iso(attempt.started_at),
            }
        )
        self._attempts[seq] = attempt
        self._next_attempt_seq = seq + 1
        return attempt

    def record_submit(
        self,
        client_order_id: str,
        symbol: str,
        submit_seq: int,
        *,
        attempt_seq: int,
        side: str,
        quantity: Decimal,
        reduce_only: bool,
        leg_index: int,
    ) -> None:
        self._ensure_loaded()
        if submit_seq != self._next_submit_seq:
            raise ValueError(
                f"submit_seq {submit_seq} does not match next journal sequence {self._next_submit_seq}"
            )
        if side not in _SIDES:
            raise ValueError(f"unknown side: {side!r}")
        qty = Decimal(quantity)
        if qty <= 0:
            raise ValueError(f"quantity must be positive: {quantity!r}")
        if int(leg_index) < 0:
            raise ValueError(f"leg_index must be non-negative: {leg_index!r}")
        recorded_at = _utc_now()
        self._ensure_v2_schema()
        self._append(
            {
                "event": "submit",
                "client_order_id": client_order_id,
                "symbol": symbol,
                "submit_seq": submit_seq,
                "attempt_seq": int(attempt_seq),
                "side": side,
                "quantity": str(qty),
                "reduce_only": bool(reduce_only),
                "leg_index": int(leg_index),
                "recorded_at": _iso(recorded_at),
            }
        )
        self._submits.append(
            JournalSubmit(
                client_order_id=client_order_id,
                symbol=symbol,
                submit_seq=submit_seq,
                attempt_seq=int(attempt_seq),
                side=side,
                quantity=qty,
                reduce_only=bool(reduce_only),
                leg_index=int(leg_index),
                recorded_at=recorded_at,
            )
        )
        self._next_submit_seq = submit_seq + 1

    def record_fill(
        self,
        *,
        kind: FillKind,
        attempt_seq: int | None,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        fee_bps: float,
        liquidity: str,
        reason: str,
        filled_at: pd.Timestamp,
        client_order_id: str | None,
        leg_index: int,
        cumulative_executed_qty: Decimal | None,
        simulated: bool,
    ) -> JournalFill:
        """Append one fill delta (fsync) and return it with its assigned `fill_seq`. Raises `ValueError` for a non-positive quantity or price, an unknown side, liquidity or kind, or a naive `filled_at`; raises `OSError` when the append cannot be made durable — callers must treat that as fatal for the attempt because an unjournaled fill could later be lost."""
        self._ensure_loaded()
        if str(kind) not in _FILL_KINDS:
            raise ValueError(f"unknown kind: {kind!r}")
        if side not in _SIDES:
            raise ValueError(f"unknown side: {side!r}")
        if liquidity not in _LIQUIDITY:
            raise ValueError(f"unknown liquidity: {liquidity!r}")
        qty = Decimal(quantity)
        if qty <= 0:
            raise ValueError(f"quantity must be positive: {quantity!r}")
        px = Decimal(price)
        if px <= 0:
            raise ValueError(f"price must be positive: {price!r}")
        if float(fee_bps) < 0:
            raise ValueError(f"fee_bps must be >= 0: {fee_bps!r}")
        stamp = pd.Timestamp(filled_at)
        if stamp.tzinfo is None:
            raise ValueError("filled_at must be tz-aware")
        cumulative = (
            Decimal(cumulative_executed_qty)
            if cumulative_executed_qty is not None
            else None
        )
        if (
            cumulative is not None
            and client_order_id is not None
            and cumulative <= self.observed_qty(client_order_id)
        ):
            raise ValueError(
                f"cumulative_executed_qty {cumulative} does not exceed observed qty "
                f"for {client_order_id}"
            )
        seq = self._next_fill_seq
        fill = JournalFill(
            fill_seq=seq,
            kind=kind,
            attempt_seq=int(attempt_seq) if attempt_seq is not None else None,
            symbol=symbol,
            side=side,
            quantity=qty,
            price=px,
            fee_bps=float(fee_bps),
            liquidity=liquidity,
            reason=reason,
            filled_at=stamp,
            client_order_id=client_order_id,
            leg_index=int(leg_index),
            cumulative_executed_qty=cumulative,
            simulated=bool(simulated),
        )
        self._ensure_v2_schema()
        self._append(
            {
                "event": "fill",
                "fill_seq": seq,
                "kind": str(kind),
                "attempt_seq": fill.attempt_seq,
                "symbol": symbol,
                "side": side,
                "quantity": str(qty),
                "price": str(px),
                "fee_bps": float(fee_bps),
                "liquidity": liquidity,
                "reason": reason,
                "filled_at": _iso(stamp),
                "client_order_id": client_order_id,
                "leg_index": int(leg_index),
                "cumulative_executed_qty": str(cumulative) if cumulative is not None else None,
                "simulated": bool(simulated),
            }
        )
        self._fills.append(fill)
        self._next_fill_seq = seq + 1
        return fill

    def record_terminal(self, client_order_id: str, status: str) -> None:
        """Mark a client order id as terminal (no further fills possible). Restart recovery never queries ids with a terminal record."""
        self._ensure_loaded()
        recorded_at = _utc_now()
        self._ensure_v2_schema()
        self._append(
            {
                "event": "terminal",
                "client_order_id": client_order_id,
                "status": status,
                "recorded_at": _iso(recorded_at),
            }
        )
        self._terminals[client_order_id] = JournalTerminal(
            client_order_id=client_order_id,
            status=status,
            recorded_at=recorded_at,
        )

    def observed_qty(self, client_order_id: str) -> Decimal:
        self._ensure_loaded()
        best = self._observed_legacy.get(client_order_id, Decimal(0))
        for fill in self._fills:
            if fill.client_order_id != client_order_id:
                continue
            cumulative = fill.cumulative_executed_qty
            if cumulative is not None and cumulative > best:
                best = cumulative
        return best

    def next_submit_seq(self) -> int:
        self._ensure_loaded()
        return self._next_submit_seq

    def last_fill_seq(self) -> int:
        self._ensure_loaded()
        return self._next_fill_seq - 1

    def fills_after(self, fill_seq: int) -> tuple[JournalFill, ...]:
        self._ensure_loaded()
        return tuple(f for f in self._fills if f.fill_seq > fill_seq)

    def attempt(self, attempt_seq: int) -> JournalAttempt | None:
        self._ensure_loaded()
        return self._attempts.get(int(attempt_seq))

    def unresolved_submits(self, *, since: pd.Timestamp) -> tuple[JournalSubmit, ...]:
        """Schema-v2 submits recorded at or after `since` that have no terminal record, in `submit_seq` order."""
        self._ensure_loaded()
        floor = pd.Timestamp(since)
        selected = [
            s
            for s in self._submits
            if s.attempt_seq is not None
            and s.recorded_at >= floor
            and s.client_order_id not in self._terminals
        ]
        selected.sort(key=lambda s: s.submit_seq)
        return tuple(selected)

    def _ensure_v2_schema(self) -> None:
        if self._has_schema:
            return
        self._append({"event": "schema", "version": JOURNAL_SCHEMA_VERSION})
        self._has_schema = True

    def _append(self, record: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._torn_offset is not None:
            truncate_durably(self._path, self._torn_offset)
            self._torn_offset = None
        line = json.dumps(record, sort_keys=True) + "\n"
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
