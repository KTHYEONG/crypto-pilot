# ruff: noqa
"""Order journal unit tests (live_order_idempotency contract)."""

from __future__ import annotations


def test_order_journal_submit_seq_is_monotonic_across_reloads(tmp_path) -> None:
    from src.live.order_journal import OrderJournal

    path = tmp_path / "state" / "journal.jsonl"
    journal = OrderJournal(path)
    assert journal.next_submit_seq() == 0
    journal.record_submit("mh20260914-AAAAAAAAAA-0-0-0", "AAAUSDT", 0)
    journal.record_submit("mh20260914-AAAAAAAAAA-0-0-1", "AAAUSDT", 1)

    reloaded = OrderJournal(path)

    assert reloaded.next_submit_seq() == 2
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2

def test_order_journal_rejects_out_of_order_submit_seq(tmp_path) -> None:
    import pytest
    from src.live.order_journal import OrderJournal

    path = tmp_path / "journal.jsonl"
    journal = OrderJournal(path)

    with pytest.raises(ValueError, match="submit_seq"):
        journal.record_submit("mh20260914-AAAAAAAAAA-0-0-5", "AAAUSDT", 5)

    assert not path.exists()

def test_order_journal_observed_qty_keeps_maximum_and_skips_non_increasing_writes(tmp_path) -> None:
    from decimal import Decimal
    from src.live.order_journal import OrderJournal

    path = tmp_path / "journal.jsonl"
    journal = OrderJournal(path)
    journal.record_observed("mhX", Decimal("0.4"))
    journal.record_observed("mhX", Decimal("0.4"))
    journal.record_observed("mhX", Decimal("0.3"))
    journal.record_observed("mhX", Decimal("1"))

    assert journal.observed_qty("mhX") == Decimal("1")
    assert journal.observed_qty("mhY") == Decimal("0")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert OrderJournal(path).observed_qty("mhX") == Decimal("1")

def test_order_journal_tolerates_torn_tail_and_truncates_before_append(tmp_path) -> None:
    import json
    from src.live.order_journal import OrderJournal

    path = tmp_path / "journal.jsonl"
    complete = json.dumps({"event": "submit", "client_order_id": "mhA", "symbol": "AAAUSDT", "submit_seq": 0}) + "\n"
    path.write_text(complete + '{"event": "submit", "client_ord', encoding="utf-8")

    journal = OrderJournal(path)
    assert journal.next_submit_seq() == 1
    journal.record_submit("mhB", "AAAUSDT", 1)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["submit_seq"] for line in lines] == [0, 1]

def test_order_journal_raises_on_mid_file_corruption(tmp_path) -> None:
    import json
    import pytest
    from src.common.errors import DataIntegrityError
    from src.live.order_journal import OrderJournal

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
    from src.live.order_journal import OrderJournal, default_order_journal_path

    path = tmp_path / "missing_dir" / "journal.jsonl"
    journal = OrderJournal(path)

    assert journal.path == path
    assert not path.parent.exists()
    assert default_order_journal_path() == DATA_DIR / "state" / "live_order_journal.jsonl"

