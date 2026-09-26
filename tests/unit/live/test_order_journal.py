# ruff: noqa
"""Order journal unit tests (spec 03: journal-backed fill durability)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.live.order_journal import OrderJournal


def _ts() -> pd.Timestamp:
    return pd.Timestamp(datetime(2026, 1, 2, 12, 0, 0, tzinfo=UTC))


def _begin(journal: OrderJournal, *, seq_check: int | None = None):
    attempt = journal.begin_attempt(
        decision_time=_ts(),
        run_id="run-1",
        mode="PAPER",
        pre_trade_equity=Decimal("1000"),
        sizing_anchor="equity",
        decision_marks={"AAAUSDT": Decimal("1")},
        started_at=_ts(),
    )
    if seq_check is not None:
        assert attempt.attempt_seq == seq_check
    return attempt


def _submit(journal: OrderJournal, client_order_id: str, seq: int, attempt_seq: int = 0) -> None:
    journal.record_submit(
        client_order_id,
        "AAAUSDT",
        seq,
        attempt_seq=attempt_seq,
        side="BUY",
        quantity=Decimal("1"),
        reduce_only=False,
        leg_index=0,
    )


def _fill_kwargs(**over: Any) -> Any:
    base: dict[str, Any] = {
        "kind": "execution",
        "attempt_seq": 0,
        "symbol": "AAAUSDT",
        "side": "BUY",
        "quantity": Decimal("0.5"),
        "price": Decimal("100"),
        "fee_bps": 5.0,
        "liquidity": "taker",
        "reason": "maker_fill",
        "filled_at": _ts(),
        "client_order_id": "mhA",
        "leg_index": 0,
        "cumulative_executed_qty": None,
        "simulated": False,
    }
    base.update(over)
    return base


def _non_schema_lines(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("event") != "schema"
    ]


def _submit_seqs(path: Path) -> list[int]:
    return [r["submit_seq"] for r in _non_schema_lines(path) if r.get("event") == "submit"]


def test_order_journal_submit_seq_is_monotonic_across_reloads(tmp_path) -> None:
    path = tmp_path / "state" / "journal.jsonl"
    journal = OrderJournal(path)
    assert journal.next_submit_seq() == 0
    _begin(journal, seq_check=0)
    _submit(journal, "mh20260914-AAAAAAAAAA-0-0-0", 0)
    _submit(journal, "mh20260914-AAAAAAAAAA-0-0-1", 1)

    reloaded = OrderJournal(path)

    assert reloaded.next_submit_seq() == 2
    assert _submit_seqs(path) == [0, 1]


def test_order_journal_rejects_out_of_order_submit_seq(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = OrderJournal(path)

    with pytest.raises(ValueError, match="submit_seq"):
        journal.record_submit(
            "mh20260914-AAAAAAAAAA-0-0-5",
            "AAAUSDT",
            5,
            attempt_seq=0,
            side="BUY",
            quantity=Decimal("1"),
            reduce_only=False,
            leg_index=0,
        )

    assert not path.exists()


def test_order_journal_observed_qty_from_legacy_and_fill_maximum(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_text(
        json.dumps({"event": "observed", "client_order_id": "mhX", "executed_qty": "0.4"}) + "\n",
        encoding="utf-8",
    )
    journal = OrderJournal(path)
    assert journal.observed_qty("mhX") == Decimal("0.4")
    _begin(journal)
    journal.record_fill(**_fill_kwargs(client_order_id="mhX", quantity=Decimal("0.6"), cumulative_executed_qty=Decimal("1")))
    assert journal.observed_qty("mhX") == Decimal("1")
    assert OrderJournal(path).observed_qty("mhX") == Decimal("1")


def test_order_journal_tolerates_torn_tail_and_truncates_before_append(tmp_path) -> None:
    complete = json.dumps({"event": "submit", "client_order_id": "mhA", "symbol": "AAAUSDT", "submit_seq": 0}) + "\n"
    path = tmp_path / "journal.jsonl"
    path.write_text(complete + '{"event": "submit", "client_ord', encoding="utf-8")

    journal = OrderJournal(path)
    assert journal.next_submit_seq() == 1
    _begin(journal, seq_check=0)
    _submit(journal, "mhB", 1)

    assert _submit_seqs(path) == [0, 1]


def test_order_journal_raises_on_mid_file_corruption(tmp_path) -> None:
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text("not-json\n" + json.dumps({"event": "submit", "client_order_id": "mhA", "symbol": "A", "submit_seq": 0}) + "\n", encoding="utf-8")
    unknown = tmp_path / "unknown.jsonl"
    unknown.write_text(json.dumps({"event": "mystery"}) + "\n", encoding="utf-8")

    with pytest.raises(DataIntegrityError, match="corrupt"):
        OrderJournal(corrupt).next_submit_seq()
    with pytest.raises(DataIntegrityError, match="unknown event"):
        OrderJournal(unknown).next_submit_seq()


def test_order_journal_construction_performs_no_io(tmp_path) -> None:
    from src.common.paths import DATA_DIR
    from src.live.order_journal import default_order_journal_path

    path = tmp_path / "missing_dir" / "journal.jsonl"
    journal = OrderJournal(path)

    assert journal.path == path
    assert not path.parent.exists()
    assert default_order_journal_path() == DATA_DIR / "state" / "live_order_journal.jsonl"


# --- Spec 03 invariant scenarios ---


def test_fill_sequence_is_lifetime_unique_across_reloads(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = OrderJournal(path)
    _begin(journal)
    for _ in range(3):
        journal.record_fill(**_fill_kwargs())

    reloaded = OrderJournal(path)
    new_fill = reloaded.record_fill(**_fill_kwargs())
    assert new_fill.fill_seq == 3
    assert [f.fill_seq for f in reloaded.fills_after(1)] == [2, 3]


def test_observed_qty_derives_from_cumulative_fill_records(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_text(
        json.dumps({"event": "observed", "client_order_id": "A", "executed_qty": "1.0"}) + "\n",
        encoding="utf-8",
    )
    journal = OrderJournal(path)
    _begin(journal)
    journal.record_fill(**_fill_kwargs(client_order_id="A", quantity=Decimal("0.5"), cumulative_executed_qty=Decimal("1.5")))
    assert journal.observed_qty("A") == Decimal("1.5")
    with pytest.raises(ValueError):
        journal.record_fill(**_fill_kwargs(client_order_id="A", quantity=Decimal("0.1"), cumulative_executed_qty=Decimal("1.5")))


def test_schema_marker_precedes_first_v2_line_in_legacy_file(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_text(
        json.dumps({"event": "submit", "client_order_id": "mhOld", "symbol": "AAAUSDT", "submit_seq": 0}) + "\n",
        encoding="utf-8",
    )
    journal = OrderJournal(path)
    journal.begin_attempt(
        decision_time=_ts(),
        run_id="run-1",
        mode="PAPER",
        pre_trade_equity=Decimal("1000"),
        sizing_anchor="equity",
        decision_marks={},
        started_at=_ts(),
    )
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[1] == {"event": "schema", "version": 2}
    since = pd.Timestamp(datetime(2020, 1, 1, tzinfo=UTC))
    assert journal.unresolved_submits(since=since) == ()


def test_torn_fill_tail_is_ignored_and_truncated(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = OrderJournal(path)
    _begin(journal)
    journal.record_fill(**_fill_kwargs())
    assert journal.last_fill_seq() == 0
    torn = '{"event": "fill", "fill_seq": 1, "kind": "execu'
    with path.open("a", encoding="utf-8") as handle:
        handle.write(torn)  # no trailing newline: torn tail

    reloaded = OrderJournal(path)
    assert reloaded.last_fill_seq() == 0
    new_fill = reloaded.record_fill(**_fill_kwargs())
    assert new_fill.fill_seq == 1
    raw = path.read_text(encoding="utf-8")
    assert torn not in raw
    assert raw.endswith("\n")


def test_unknown_event_fails_closed(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = OrderJournal(path)
    _begin(journal)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "bogus"}) + "\n")
        handle.write(json.dumps({"event": "submit", "client_order_id": "mhZ", "symbol": "AAAUSDT", "submit_seq": 99}) + "\n")

    with pytest.raises(DataIntegrityError):
        OrderJournal(path).next_submit_seq()


def test_terminal_submits_are_excluded_from_recovery(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = OrderJournal(path)
    attempt = _begin(journal)
    _submit(journal, "mhA", 0, attempt_seq=attempt.attempt_seq)
    _submit(journal, "mhB", 1, attempt_seq=attempt.attempt_seq)
    journal.record_terminal("mhA", "FILLED")

    since = pd.Timestamp(datetime(2020, 1, 1, tzinfo=UTC))
    unresolved = journal.unresolved_submits(since=since)
    assert [s.client_order_id for s in unresolved] == ["mhB"]


def test_record_fill_rejects_invalid_inputs(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = OrderJournal(path)
    _begin(journal)
    with pytest.raises(ValueError):
        journal.record_fill(**_fill_kwargs(quantity=Decimal("0")))
    with pytest.raises(ValueError):
        journal.record_fill(**_fill_kwargs(price=Decimal("-1")))
    with pytest.raises(ValueError):
        journal.record_fill(**_fill_kwargs(side="HOLD"))
    with pytest.raises(ValueError):
        journal.record_fill(**_fill_kwargs(liquidity="dark"))
    with pytest.raises(ValueError):
        journal.record_fill(**_fill_kwargs(kind="nope"))
    with pytest.raises(ValueError):
        journal.record_fill(**_fill_kwargs(filled_at=pd.Timestamp("2026-01-01 00:00:00")))
    with pytest.raises(ValueError):
        journal.record_fill(**_fill_kwargs(fee_bps=-0.1))
    assert journal.last_fill_seq() == -1


@pytest.mark.parametrize(
    "kwargs",
    [{"side": "HOLD"}, {"quantity": Decimal("0")}, {"leg_index": -1}],
    ids=["side", "quantity", "leg_index"],
)
def test_record_submit_rejects_invalid_inputs(tmp_path, kwargs) -> None:
    """Submit preconditions are enforced before anything is appended (the sequence stays unused)."""
    journal = OrderJournal(tmp_path / "journal.jsonl")
    attempt = _begin(journal)
    base: dict[str, Any] = {"side": "BUY", "quantity": Decimal("1"), "reduce_only": False, "leg_index": 0}
    base.update(kwargs)
    with pytest.raises(ValueError):
        journal.record_submit("mhA", "AAAUSDT", 0, attempt_seq=attempt.attempt_seq, **base)
    assert journal.next_submit_seq() == 0


def test_begin_attempt_rejects_naive_timestamps(tmp_path) -> None:
    journal = OrderJournal(tmp_path / "journal.jsonl")
    with pytest.raises(ValueError):
        journal.begin_attempt(
            decision_time=pd.Timestamp("2026-01-02 12:00:00"), run_id="r", mode="PAPER",
            pre_trade_equity=Decimal("1000"), sizing_anchor="equity", decision_marks={}, started_at=_ts(),
        )


def _mutate_attempt_record(path: Path, **fields: Any) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        record = json.loads(line)
        if record.get("event") == "attempt":
            record.update(fields)
            for key, value in list(record.items()):
                if value is _DROP:
                    del record[key]
        out.append(json.dumps(record))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


_DROP = object()


@pytest.mark.parametrize(
    "fields",
    [
        {"decision_time": "not-a-time"},
        {"decision_time": "2026-01-02T12:00:00"},
        {"pre_trade_equity": "abc"},
        {"decision_marks": ["AAAUSDT"]},
        {"run_id": _DROP},
    ],
    ids=["bad_timestamp", "naive_timestamp", "bad_decimal", "marks_not_object", "missing_key"],
)
def test_corrupt_attempt_record_fails_closed(tmp_path, fields) -> None:
    """A complete (newline-terminated) attempt line with an invalid field is corruption, not a torn tail."""
    path = tmp_path / "journal.jsonl"
    _begin(OrderJournal(path))
    _mutate_attempt_record(path, **fields)

    with pytest.raises(DataIntegrityError):
        OrderJournal(path).next_submit_seq()


def test_non_object_line_fails_closed(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    _begin(OrderJournal(path))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("[1, 2]\n")
    with pytest.raises(DataIntegrityError):
        OrderJournal(path).next_submit_seq()


def test_blank_lines_are_ignored_on_load(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = OrderJournal(path)
    attempt = _begin(journal)
    _submit(journal, "mhA", 0, attempt_seq=attempt.attempt_seq)
    path.write_text(path.read_text(encoding="utf-8").replace("\n", "\n\n", 1), encoding="utf-8")
    assert OrderJournal(path).next_submit_seq() == 1


def test_torn_tail_after_non_ascii_rows_truncates_at_byte_offset(tmp_path) -> None:
    """Repair cuts exactly the torn bytes even when earlier lines hold multi-byte characters."""
    path = tmp_path / "journal.jsonl"
    journal = OrderJournal(path)
    _begin(journal)
    journal.record_fill(**_fill_kwargs(symbol="龙虾USDT"))
    committed = path.read_bytes()
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "fill", "fill_seq": 1, "symbol": "龙')

    reloaded = OrderJournal(path)
    reloaded.record_fill(**_fill_kwargs())
    raw = path.read_bytes()
    assert raw.startswith(committed)
    assert all(json.loads(line) for line in raw.decode("utf-8").splitlines())
    assert OrderJournal(path).last_fill_seq() == 1


def test_truncate_durably_keeps_prefix(tmp_path) -> None:
    """In-place truncation keeps the committed prefix byte-exact."""
    from src.live.order_journal import truncate_durably

    path = tmp_path / "f.jsonl"
    path.write_bytes(b'{"a":1}\n{"b":')
    truncate_durably(path, 8)
    assert path.read_bytes() == b'{"a":1}\n'


def test_unknown_fill_kind_fails_closed(tmp_path) -> None:
    """A complete fill line with an unrecognised kind is corruption; the journal refuses to load."""
    path = tmp_path / "journal.jsonl"
    journal = OrderJournal(path)
    attempt = _begin(journal)
    journal.record_fill(
        kind="execution", attempt_seq=attempt.attempt_seq, symbol="AAAUSDT", side="BUY",
        quantity=Decimal("1"), price=Decimal("100"), fee_bps=4.5, liquidity="taker", reason="test",
        filled_at=pd.Timestamp("2026-09-14T00:00:00Z"), client_order_id=None, leg_index=0,
        cumulative_executed_qty=None, simulated=False,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    record["kind"] = "mystery_kind"
    lines[-1] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(DataIntegrityError, match="unknown fill kind"):
        OrderJournal(path).next_submit_seq()
