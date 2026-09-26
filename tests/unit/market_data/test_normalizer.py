"""Invariant guards for raw-to-derived checkpointed derivation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from src.capture.journal import SegmentWriter, hot_segment_path, last_complete_offset
from src.common.errors import DataIntegrityError
from src.market_data.streams.coverage import load_coverage
from src.market_data.streams.normalizer import (
    DedupeWindow,
    NormalizerCheckpoint,
    NormalizerConfig,
    load_checkpoint,
    normalize_once,
    save_checkpoint,
)

BOOK_BODY = (
    '[{"symbol":"BTCUSDT","bidPrice":"70000.5","bidQty":"1.2","askPrice":"70001.0","askQty":"0.8",'
    '"time":1758679200000},{"symbol":"ETHUSDT","bidPrice":"3000.0","bidQty":"2.0","askPrice":"3000.5",'
    '"askQty":"1.5","time":1758679200000}]'
)
PREMIUM_BODY = (
    '[{"symbol":"BTCUSDT","markPrice":"70000.0","indexPrice":"69999.0","estimatedSettlePrice":"70001.0",'
    '"lastFundingRate":"0.0001","nextFundingTime":1758681600000,"interestRate":"0.0003","time":1758679200000}]'
)
FRAME_TEXT = (
    '{"e":"forceOrder","E":1758679201000,"o":{"s":"BTCUSDT","S":"SELL","o":"LIMIT","f":"GTC",'
    '"q":"1.0","p":"70000.0","ap":"70000.0","X":"FILLED","l":"1.0","z":"1.0","T":1758679201000}}'
)


def _ns(hour: int = 10, minute: int = 0, second: int = 0, day: int = 26) -> int:
    return int(datetime(day=day, month=9, year=2026, hour=hour, minute=minute, second=second,
                        tzinfo=UTC).timestamp() * 1_000_000_000)


def _now(day: int = 26, hour: int = 12) -> pd.Timestamp:
    return pd.Timestamp(datetime(day=day, month=9, year=2026, hour=hour, tzinfo=UTC))


def _rest(stream: str, slot: str, grid: str, recv_ns: int, body: str = BOOK_BODY, status: int = 200) -> dict[str, Any]:
    return {"v": 1, "stream": stream, "slot": slot, "kind": "rest", "recv_ns": recv_ns,
            "grid": grid, "status": status, "body": body}


def _rest_error(stream: str, slot: str, grid: str, recv_ns: int) -> dict[str, Any]:
    return {"v": 1, "stream": stream, "slot": slot, "kind": "rest_error", "recv_ns": recv_ns,
            "grid": grid, "status": None, "error": "TimeoutError: boom"}


def _ws_open(slot: str, recv_ns: int) -> dict[str, Any]:
    return {"v": 1, "stream": "force_order", "slot": slot, "kind": "ws_open",
            "recv_ns": recv_ns, "url": "wss://example.invalid"}


def _frame(slot: str, recv_ns: int, text: str = FRAME_TEXT) -> dict[str, Any]:
    return {"v": 1, "stream": "force_order", "slot": slot, "kind": "frame",
            "recv_ns": recv_ns, "frame": text}


def _ws_close(slot: str, recv_ns: int, reason: str = "server_close") -> dict[str, Any]:
    return {"v": 1, "stream": "force_order", "slot": slot, "kind": "ws_close",
            "recv_ns": recv_ns, "reason": reason}


def _journal(root: Path, stream: str, slot: str, records: list[dict[str, Any]]) -> None:
    writer = SegmentWriter(root, stream, slot)
    for record in records:
        writer.add(record)
    writer.flush()


def _cycle(root: Path, liq: Path, checkpoint: NormalizerCheckpoint,
           config: NormalizerConfig | None = None, now: pd.Timestamp | None = None,
           dedupe: DedupeWindow | None = None):
    checkpoint_path = root / "raw" / "normalizer_checkpoint.json"
    if dedupe is None:
        dedupe = DedupeWindow(rest_window_s=7200.0, ws_window_s=7200.0)
    new_checkpoint, report = normalize_once(
        root, liq, config or NormalizerConfig(), checkpoint=checkpoint,
        now=now or _now(), dedupe=dedupe,
    )
    save_checkpoint(checkpoint_path, new_checkpoint)
    return load_checkpoint(checkpoint_path), report


def _empty() -> NormalizerCheckpoint:
    return NormalizerCheckpoint(files={}, coverage={})


def _parquet_rows(root: Path, dataset: str) -> pd.DataFrame:
    paths = sorted((root / dataset).rglob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def test_cycle_derives_both_rest_streams(tmp_path: Path) -> None:
    """Hot samples land in the existing layout with grid captured_at and recv fetched_at."""
    grid = "2026-09-26T10:00:00Z"
    _journal(tmp_path, "book_ticker", "blue", [_rest("book_ticker", "blue", grid, _ns(10, 0, 5))])
    _journal(tmp_path, "premium_index", "blue",
             [_rest("premium_index", "blue", "2026-09-26T10:00:00Z", _ns(10, 0, 7), body=PREMIUM_BODY)])
    checkpoint, report = _cycle(tmp_path, tmp_path / "liq", _empty())
    books = _parquet_rows(tmp_path, "book_ticker")
    assert set(books["symbol"]) == {"BTCUSDT", "ETHUSDT"}
    assert (books["captured_at"] == pd.Timestamp(grid)).all()
    assert (books["fetched_at_ms"] == _ns(10, 0, 5) // 1_000_000).all()
    premiums = _parquet_rows(tmp_path, "premium_index")
    assert set(premiums["symbol"]) == {"BTCUSDT"}
    dest = hot_segment_path(tmp_path, "book_ticker", "blue", _ns(10))
    assert checkpoint.files["book_ticker/20260926/10.blue.jsonl.gz"].offset == last_complete_offset(dest)
    assert report.rows_written["book_ticker"] == 2
    assert report.more_pending is False


def test_crash_replay_is_idempotent(tmp_path: Path) -> None:
    """Re-deriving from a pre-write checkpoint yields byte-identical parquet and coverage."""
    grid = "2026-09-26T10:00:00Z"
    _journal(tmp_path, "book_ticker", "blue", [_rest("book_ticker", "blue", grid, _ns(10, 0, 5))])
    _journal(tmp_path, "force_order", "blue", [
        _ws_open("blue", _ns(10, 0, 1)), _frame("blue", _ns(10, 0, 2)),
        _frame("blue", _ns(10, 0, 2) + 1_000_000_000), _ws_close("blue", _ns(10, 0, 3)),
    ])
    liq = tmp_path / "liq"
    first, _ = _cycle(tmp_path, liq, _empty())
    before_parquet = {str(p): p.read_bytes() for p in sorted(tmp_path.rglob("*.parquet"))}
    before_merged = load_coverage(tmp_path, "liquidations", start=_now(26, 10), end=_now(26, 11))
    second, _ = _cycle(tmp_path, liq, _empty())
    assert {str(p): p.read_bytes() for p in sorted(tmp_path.rglob("*.parquet"))} == before_parquet
    after_merged = load_coverage(tmp_path, "liquidations", start=_now(26, 10), end=_now(26, 11))
    pd.testing.assert_frame_equal(after_merged, before_merged)
    assert second.files == first.files
    merged = load_coverage(tmp_path, "liquidations", start=_now(26, 10), end=_now(26, 11))
    assert len(merged) == 1


def test_failed_write_is_fully_retried_with_shared_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A derived write failure discards staged dedupe marks, so the retry writes every record."""
    import src.market_data.streams.normalizer as normalizer_mod

    grid = "2026-09-26T10:00:00Z"
    _journal(tmp_path, "book_ticker", "blue", [_rest("book_ticker", "blue", grid, _ns(10, 0, 5))])
    _journal(tmp_path, "force_order", "blue", [
        _ws_open("blue", _ns(10, 0, 1)), _frame("blue", _ns(10, 0, 2)), _ws_close("blue", _ns(10, 0, 3)),
    ])
    liq = tmp_path / "liq"
    dedupe = DedupeWindow(rest_window_s=7200.0, ws_window_s=7200.0)
    real_write = normalizer_mod.write_hourly_partition

    def _disk_full(*args: object, **kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(normalizer_mod, "write_hourly_partition", _disk_full)
    with pytest.raises(OSError, match="No space left"):
        normalize_once(tmp_path, liq, NormalizerConfig(), checkpoint=_empty(), now=_now(), dedupe=dedupe)
    assert dedupe.drain_fresh_outcomes() == []
    monkeypatch.setattr(normalizer_mod, "write_hourly_partition", real_write)
    checkpoint, report = normalize_once(tmp_path, liq, NormalizerConfig(), checkpoint=_empty(), now=_now(), dedupe=dedupe)
    assert report.rows_written["book_ticker"] > 0
    assert report.rows_written["force_order"] == 1
    assert report.duplicates_dropped["book_ticker"] == 0
    assert report.duplicates_dropped["force_order"] == 0
    assert len(_parquet_rows(tmp_path, "book_ticker")) > 0
    assert all(cursor.offset > 0 for cursor in checkpoint.files.values())


def test_earlier_twin_in_later_cycle_is_derived(tmp_path: Path) -> None:
    """A slot twin with an earlier receipt arriving in a later cycle still reaches the earliest-receipt merge."""
    grid = "2026-09-26T10:00:00Z"
    dedupe = DedupeWindow(rest_window_s=7200.0, ws_window_s=7200.0)
    _journal(tmp_path, "book_ticker", "green", [_rest("book_ticker", "green", grid, _ns(10, 0, 6))])
    checkpoint, _ = _cycle(tmp_path, tmp_path / "liq", _empty(), dedupe=dedupe)
    _journal(tmp_path, "book_ticker", "blue", [_rest("book_ticker", "blue", grid, _ns(10, 0, 5))])
    _, report = _cycle(tmp_path, tmp_path / "liq", checkpoint, dedupe=dedupe)
    books = _parquet_rows(tmp_path, "book_ticker")
    assert report.duplicates_dropped["book_ticker"] == 0
    assert (books["fetched_at_ms"] == _ns(10, 0, 5) // 1_000_000).all()


def test_two_slots_keep_earliest_receipt(tmp_path: Path) -> None:
    """Overlapping slots dedupe to one row per key with the smaller fetched_at."""
    grid = "2026-09-26T10:00:00Z"
    _journal(tmp_path, "book_ticker", "blue", [_rest("book_ticker", "blue", grid, _ns(10, 0, 5))])
    _journal(tmp_path, "book_ticker", "green", [_rest("book_ticker", "green", grid, _ns(10, 0, 5) + 100_000_000)])
    _, report = _cycle(tmp_path, tmp_path / "liq", _empty())
    books = _parquet_rows(tmp_path, "book_ticker")
    assert len(books) == 2
    assert (books["fetched_at_ms"] == _ns(10, 0, 5) // 1_000_000).all()
    assert report.duplicates_dropped["book_ticker"] == 1


def test_identical_ws_frames_are_one_event(tmp_path: Path) -> None:
    """Byte-identical frames from two slots append once but attest on both spans."""
    _journal(tmp_path, "force_order", "blue", [
        _ws_open("blue", _ns(10, 0, 1)), _frame("blue", _ns(10, 0, 2)),
        _frame("blue", _ns(10, 0, 6)), _ws_close("blue", _ns(10, 0, 7)),
    ])
    _journal(tmp_path, "force_order", "green", [
        _ws_open("green", _ns(10, 0, 1)), _frame("green", _ns(10, 0, 4)),
        _frame("green", _ns(10, 0, 8)), _ws_close("green", _ns(10, 0, 9)),
    ])
    _, report = _cycle(tmp_path, tmp_path / "liq", _empty())
    events = _parquet_rows(tmp_path / "liq", "")
    assert len(events) == 1
    assert events.iloc[0]["ingested_at"] == pd.Timestamp(_ns(10, 0, 2), unit="ns", tz="UTC")
    assert report.duplicates_dropped["force_order"] == 3
    merged = load_coverage(tmp_path, "liquidations", start=_now(26, 10), end=_now(26, 11))
    assert len(merged) == 1
    assert merged.iloc[0]["start"] == pd.Timestamp(_ns(10, 0, 2), unit="ns", tz="UTC")
    assert merged.iloc[0]["end"] == pd.Timestamp(_ns(10, 0, 8), unit="ns", tz="UTC")


def test_torn_tail_not_parsed_not_skipped(tmp_path: Path) -> None:
    """A truncated member stops the checkpoint; the completed write derives next cycle."""
    grid = "2026-09-26T10:00:00Z"
    writer = SegmentWriter(tmp_path, "book_ticker", "blue")
    writer.add(_rest("book_ticker", "blue", grid, _ns(10, 0, 5)))
    writer.flush()
    writer.add(_rest("book_ticker", "blue", "2026-09-26T10:01:00Z", _ns(10, 1, 5)))
    writer.flush()
    dest = hot_segment_path(tmp_path, "book_ticker", "blue", _ns(10))
    full = dest.read_bytes()
    dest.write_bytes(full[: len(full) - 7])
    checkpoint, _ = _cycle(tmp_path, tmp_path / "liq", _empty())
    books = _parquet_rows(tmp_path, "book_ticker")
    assert set(books["captured_at"]) == {pd.Timestamp(grid)}
    member_end = last_complete_offset(dest)
    assert checkpoint.files["book_ticker/20260926/10.blue.jsonl.gz"].offset == member_end
    assert member_end < len(full) - 7
    dest.write_bytes(full)
    checkpoint2, _ = _cycle(tmp_path, tmp_path / "liq", checkpoint)
    books = _parquet_rows(tmp_path, "book_ticker")
    assert len(books) == 4
    assert checkpoint2.files["book_ticker/20260926/10.blue.jsonl.gz"].offset == dest.stat().st_size


def test_pong_only_connection_attests_nothing(tmp_path: Path) -> None:
    """Open and close with no frames write no coverage interval."""
    _journal(tmp_path, "force_order", "blue", [_ws_open("blue", _ns(10, 0, 1)), _ws_close("blue", _ns(10, 0, 3))])
    _cycle(tmp_path, tmp_path / "liq", _empty())
    assert not (tmp_path / "coverage").exists() or not list((tmp_path / "coverage").rglob("*.jsonl"))


def test_restart_mid_connection_continues_segment(tmp_path: Path) -> None:
    """Checkpointed tracker state lets a restarted normalizer extend the same span."""
    t1, t2, t3 = _ns(10, 0, 1), _ns(10, 0, 2), _ns(10, 0, 5)
    _journal(tmp_path, "force_order", "blue", [_ws_open("blue", t1), _frame("blue", t1), _frame("blue", t2)])
    checkpoint, _ = _cycle(tmp_path, tmp_path / "liq", _empty())
    assert checkpoint.coverage["blue"].open_last is not None
    _journal(tmp_path, "force_order", "blue", [_frame("blue", t3), _ws_close("blue", t3 + 1)])
    checkpoint2, _ = _cycle(tmp_path, tmp_path / "liq", checkpoint)
    assert checkpoint2.coverage["blue"].open_last is None
    merged = load_coverage(tmp_path, "liquidations", start=_now(26, 10), end=_now(26, 11))
    assert len(merged) == 1
    assert merged.iloc[0]["start"] == pd.Timestamp(t1, unit="ns", tz="UTC")
    assert merged.iloc[0]["end"] == pd.Timestamp(t3, unit="ns", tz="UTC")


def test_rejected_and_failed_grids_feed_counters(tmp_path: Path) -> None:
    """rest_error, 500 and malformed rows never raise; failures and rejects are counted."""
    rows = ",".join(
        f'{{"symbol":"R{i:02d}USDT","bidPrice":"{70000 + i}","bidQty":"1","askPrice":"{70001 + i}",'
        f'"askQty":"1","time":1758679200000}}' for i in range(20)
    )
    bad_row_body = (
        f'[{{"symbol":"BTCUSDT","bidPrice":"not-a-price","bidQty":"1","askPrice":"2","askQty":"1",'
        f'"time":1758679200000}},{rows}]'
    )
    _journal(tmp_path, "book_ticker", "blue", [
        _rest_error("book_ticker", "blue", "2026-09-26T10:00:00Z", _ns(10, 0, 1)),
        _rest("book_ticker", "blue", "2026-09-26T10:01:00Z", _ns(10, 1, 1), status=500, body="<html>busy</html>"),
        _rest("book_ticker", "blue", "2026-09-26T10:02:00Z", _ns(10, 2, 1), body=bad_row_body),
    ])
    dedupe = DedupeWindow(rest_window_s=7200.0, ws_window_s=7200.0)
    _, report = _cycle(tmp_path, tmp_path / "liq", _empty(), dedupe=dedupe)
    assert report.rows_written.get("book_ticker", 0) == 20
    outcomes = [item for item in dedupe.rest_outcomes if item[1] == "book_ticker"]
    assert any(not item[2] for item in outcomes)
    assert any(item[2] and item[4] == 1 for item in outcomes)


def _build_backlog(tmp_path: Path) -> None:
    """Write ~3 MiB of hot snapshots as small members across three hours."""
    symbols = ",".join(
        f'{{"symbol":"S{i:04d}USDT","bidPrice":"{70000 + i}","bidQty":"1","askPrice":"{70001 + i}",'
        f'"askQty":"1","time":1758679200000}}' for i in range(3000)
    )
    big_body = f"[{symbols}]"
    assert len(big_body) > 100_000
    grids = [f"2026-09-26T{h:02d}:{m:02d}:00Z" for h in (10, 11, 12, 13, 14) for m in range(0, 60, 2)]
    writer = SegmentWriter(tmp_path, "book_ticker", "blue")
    for index, grid in enumerate(grids):
        moment = pd.Timestamp(grid)
        recv_ns = int(moment.value) + 5_000_000_000
        writer.add(_rest("book_ticker", "blue", grid, recv_ns, body=big_body))
        if index % 7 == 6:
            writer.flush()
    writer.flush()


def _drain(root: Path, liq: Path, config: NormalizerConfig) -> tuple[NormalizerCheckpoint, list[Any]]:
    checkpoint = _empty()
    reports = []
    for _ in range(12):
        checkpoint, report = _cycle(root, liq, checkpoint, config=config)
        reports.append(report)
        if not report.more_pending:
            break
    return checkpoint, reports


def _derived_bytes(root: Path) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in sorted((root / "book_ticker").rglob("*.parquet"))}


def test_byte_budget_bounds_catch_up(tmp_path: Path) -> None:
    """A 3x backlog drains in bounded cycles with the same result as one unbounded pass."""
    _build_backlog(tmp_path)
    total_compressed = sum(p.stat().st_size for p in tmp_path.rglob("*.jsonl.gz"))
    assert total_compressed > 3 * 1_048_576
    config = NormalizerConfig(max_bytes_per_cycle=1_048_576)
    checkpoint, reports = _drain(tmp_path, tmp_path / "liq", config)
    assert reports[0].more_pending is True
    assert reports[0].bytes_read > 0.9 * 1_048_576
    assert len(reports) >= 2
    assert reports[-1].more_pending is False


def test_unbounded_pass_matches_bounded_drain(tmp_path: Path) -> None:
    """One unbounded cycle derives exactly what bounded cycles derive."""
    _build_backlog(tmp_path)
    checkpoint, _ = _drain(tmp_path, tmp_path / "liq", NormalizerConfig(max_bytes_per_cycle=1_048_576))
    bounded_files, bounded_offsets = _derived_bytes(tmp_path), dict(checkpoint.files)
    for path in (tmp_path / "book_ticker").rglob("*.parquet"):
        path.unlink()
    checkpoint2, _ = _drain(tmp_path, tmp_path / "liq", NormalizerConfig(max_bytes_per_cycle=1_073_741_824))
    assert _derived_bytes(tmp_path) == bounded_files
    assert dict(checkpoint2.files) == bounded_offsets


def test_corrupt_checkpoint_fails_closed(tmp_path: Path) -> None:
    """An undecodable checkpoint raises before any derived file is touched."""
    grid = "2026-09-26T10:00:00Z"
    _journal(tmp_path, "book_ticker", "blue", [_rest("book_ticker", "blue", grid, _ns(10, 0, 5))])
    _, _ = _cycle(tmp_path, tmp_path / "liq", _empty())
    checkpoint_path = tmp_path / "raw" / "normalizer_checkpoint.json"
    before = {str(p): p.read_bytes() for p in sorted(tmp_path.rglob("*.parquet"))}
    checkpoint_path.write_text("{corrupt", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="checkpoint"):
        load_checkpoint(checkpoint_path)
    assert {str(p): p.read_bytes() for p in sorted(tmp_path.rglob("*.parquet"))} == before


def test_unknown_slot_rejected(tmp_path: Path) -> None:
    """Records outside the capture slots count as parse failures and derive nothing."""
    grid = "2026-09-26T10:00:00Z"
    _journal(tmp_path, "book_ticker", "blue", [_rest("book_ticker", "red", grid, _ns(10, 0, 5))])
    _, report = _cycle(tmp_path, tmp_path / "liq", _empty())
    assert report.parse_failures == 1
    assert not list(tmp_path.rglob("*.parquet"))


def test_empty_hot_tree_is_noop(tmp_path: Path) -> None:
    """Zero records write nothing, change nothing and report zero lag."""
    checkpoint, report = _cycle(tmp_path, tmp_path / "liq", _empty())
    assert checkpoint.files == {}
    assert all(state.open_last is None for state in checkpoint.coverage.values())
    assert report.records_read == 0
    assert report.lag_s == 0
    assert report.more_pending is False
    assert not list(tmp_path.rglob("*.parquet"))


def test_checkpoint_drops_deleted_files(tmp_path: Path) -> None:
    """Entries for compacted-away files disappear on the next save."""
    grid = "2026-09-26T10:00:00Z"
    _journal(tmp_path, "book_ticker", "blue", [_rest("book_ticker", "blue", grid, _ns(10, 0, 5))])
    checkpoint, _ = _cycle(tmp_path, tmp_path / "liq", _empty())
    assert len(checkpoint.files) == 1
    for path in (tmp_path / "raw" / "hot").rglob("*.jsonl.gz"):
        path.unlink()
    checkpoint2, _ = _cycle(tmp_path, tmp_path / "liq", checkpoint)
    assert checkpoint2.files == {}


def test_config_validators_reject_bad_values() -> None:
    """Every normalizer cadence, window and bound validates its contract."""
    with pytest.raises(ValidationError, match="normalize_interval_s"):
        NormalizerConfig(normalize_interval_s=0.0)
    with pytest.raises(ValidationError, match="segment_final_grace_s"):
        NormalizerConfig(segment_final_grace_s=10.0)
    with pytest.raises(ValidationError, match="ws_dedupe_window_s"):
        NormalizerConfig(ws_dedupe_window_s=10.0)
    with pytest.raises(ValidationError, match="rest_dedupe_window_s"):
        NormalizerConfig(rest_dedupe_window_s=10.0)
    with pytest.raises(ValidationError, match="max_bytes_per_cycle"):
        NormalizerConfig(max_bytes_per_cycle=100)
    with pytest.raises(ValidationError, match="book_ticker_interval_s"):
        NormalizerConfig(book_ticker_interval_s=7)
    with pytest.raises(ValidationError, match="premium_index_interval_s"):
        NormalizerConfig(premium_index_interval_s=7)
    with pytest.raises(ValidationError, match="snapshot_max_rejected_fraction"):
        NormalizerConfig(snapshot_max_rejected_fraction=1.0)
    with pytest.raises(ValidationError, match="grid_health_window_s"):
        NormalizerConfig(grid_health_window_s=0.0)
    with pytest.raises(ValidationError, match="reference_capture_after_utc"):
        NormalizerConfig(reference_capture_after_utc="nope")
    with pytest.raises(ValidationError, match="compaction_grace_s"):
        NormalizerConfig(compaction_grace_s=0.0)
    with pytest.raises(ValidationError, match="archive_lzma_preset"):
        NormalizerConfig(archive_lzma_preset=10)
    with pytest.raises(ValidationError, match="raw_archive_local_retention_days"):
        NormalizerConfig(raw_archive_local_retention_days=1)
    with pytest.raises(ValidationError, match="parquet_local_retention_days"):
        NormalizerConfig(parquet_local_retention_days=1)
    with pytest.raises(ValidationError, match="backup_status_max_age_h"):
        NormalizerConfig(backup_status_max_age_h=0.0)
    with pytest.raises(ValidationError, match="partial_sweep_age_s"):
        NormalizerConfig(partial_sweep_age_s=0.0)
    with pytest.raises(ValidationError, match="retention_interval_s"):
        NormalizerConfig(retention_interval_s=0.0)
    with pytest.raises(ValidationError, match="heartbeat_interval_s"):
        NormalizerConfig(heartbeat_interval_s=0.0)
    with pytest.raises(ValidationError, match="compaction_grace_s"):
        NormalizerConfig(compaction_grace_s=60.0, segment_final_grace_s=180.0)
    with pytest.raises(ValidationError, match="parquet_local_retention_days"):
        NormalizerConfig(parquet_local_retention_days=2, raw_archive_local_retention_days=30)
    with pytest.raises(ValidationError, match="retention_interval_s"):
        NormalizerConfig(retention_interval_s=5.0, normalize_interval_s=30.0)
    with pytest.raises(ValidationError, match="grid_health_window_s"):
        NormalizerConfig(grid_health_window_s=100.0)
    with pytest.raises(ValidationError, match="partial_sweep_age_s"):
        NormalizerConfig(partial_sweep_age_s=100.0)
    assert NormalizerConfig().normalize_interval_s == 30.0


def test_checkpoint_schema_fails_closed(tmp_path: Path) -> None:
    """Every checkpoint schema violation raises instead of restarting from zero."""
    bad_payloads = [
        "{oops",
        "[1,2]",
        {"files": []},
        {"files": {"a": []}},
        {"files": {"a": {"offset": "x"}}},
        {"files": {"a": {"offset": -1}}},
        {"files": {"a": {"offset": True}}},
        {"coverage": []},
        {"coverage": {"blue": []}},
        {"coverage": {"blue": {"open_last": "yesterday"}}},
        {"coverage": {"blue": {"cursor": "yesterday"}}},
    ]
    for index, payload in enumerate(bad_payloads):
        path = tmp_path / f"checkpoint_{index}.json"
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
        with pytest.raises(DataIntegrityError, match="checkpoint"):
            load_checkpoint(path)
    assert load_checkpoint(tmp_path / "absent.json") == _empty()


def test_save_checkpoint_tolerates_fsync_failures(tmp_path: Path, monkeypatch) -> None:
    """Unfsyncable directories still leave a readable checkpoint behind."""
    import os as _os

    dest = tmp_path / "normalizer_checkpoint.json"
    save_checkpoint(dest, _empty())
    assert load_checkpoint(dest) == _empty()
    monkeypatch.setattr(_os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    save_checkpoint(dest, _empty())
    monkeypatch.undo()
    calls: list[int] = []
    real_fsync = _os.fsync

    def selective_fsync(fd: int) -> None:
        calls.append(fd)
        if len(calls) > 1:
            raise OSError("no")
        real_fsync(fd)

    monkeypatch.setattr(_os, "fsync", selective_fsync)
    save_checkpoint(dest, _empty())
    assert load_checkpoint(dest) == _empty()
    assert len(calls) >= 2


def test_save_checkpoint_write_failure_cleans_up(tmp_path: Path, monkeypatch) -> None:
    """A failed replace raises and leaves no partial behind."""
    dest = tmp_path / "normalizer_checkpoint.json"
    monkeypatch.setattr("os.replace", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    with pytest.raises(OSError, match="no"):
        save_checkpoint(dest, _empty())
    assert list(tmp_path.glob("*.partial")) == []


def test_dedupe_window_prunes_and_tracks(tmp_path: Path) -> None:
    """Huge frame tables prune by horizon; outcomes feed the heartbeat windows."""
    window = DedupeWindow(rest_window_s=7200.0, ws_window_s=7200.0)
    now_ns = _ns(10, 0, 0)
    assert window.check_rest("book_ticker", "g", now_ns, now_ns=now_ns) is False
    window.mark_rest_success("book_ticker", "g", now_ns)
    assert window.check_rest("book_ticker", "g", now_ns, now_ns=now_ns) is True
    stale = now_ns - 10_000_000_000_000
    for i in range(1_000):
        window.check_frame(f"d{i}", stale, now_ns=stale)
    window.commit()
    assert window.check_frame("fresh", now_ns, now_ns=now_ns) is False
    assert "d0" not in window._ws_seen
    assert len(window._ws_seen) == 0
    assert window.check_frame("fresh", now_ns, now_ns=now_ns) is True
    window.record_rest_outcome(now_ns, "book_ticker", True, 2, 0, 0.0)
    assert window.drain_fresh_outcomes() == []
    window.commit()
    assert window.drain_fresh_outcomes()[0][2] is True
    assert window.drain_fresh_outcomes() == []
    assert window.ws_last_recv_ns == now_ns


def test_two_file_budget_peeks_and_skips(tmp_path: Path) -> None:
    """An exhausted budget peeks later files for lag without consuming them."""
    from src.market_data.streams.normalizer import _gather_new_records

    grid = "2026-09-26T10:00:00Z"
    _journal(tmp_path, "book_ticker", "blue", [_rest("book_ticker", "blue", grid, _ns(10, 0, 5))])
    _journal(tmp_path, "premium_index", "blue",
             [_rest("premium_index", "blue", grid, _ns(10, 0, 7), body=PREMIUM_BODY)])
    tiny = NormalizerConfig().model_copy(update={"max_bytes_per_cycle": 100})
    torn_file = tmp_path / "raw" / "hot" / "force_order" / "20260926" / "10.blue.jsonl.gz"
    torn_file.parent.mkdir(parents=True, exist_ok=True)
    torn_file.write_bytes(b"\x1f\x8b")
    gathered, consumed, used, pending, oldest = _gather_new_records(
        tmp_path / "raw" / "hot", _empty(), tiny)
    assert used > 0
    assert pending is True
    assert oldest == _ns(10, 0, 7)
    assert set(consumed) == {"book_ticker/20260926/10.blue.jsonl.gz"}


def test_gather_clamps_shrunk_files(tmp_path: Path) -> None:
    """A hot file truncated below the checkpoint offset restarts from zero."""
    from src.market_data.streams.normalizer import FileCursor, NormalizerCheckpoint, _gather_new_records

    grid = "2026-09-26T10:00:00Z"
    _journal(tmp_path, "book_ticker", "blue", [_rest("book_ticker", "blue", grid, _ns(10, 0, 5))])
    dest = hot_segment_path(tmp_path, "book_ticker", "blue", _ns(10))
    checkpoint = NormalizerCheckpoint(
        files={"book_ticker/20260926/10.blue.jsonl.gz": FileCursor(offset=10**9, final=False)},
        coverage={},
    )
    dest.write_bytes(dest.read_bytes()[:10])
    gathered, consumed, used, pending, oldest = _gather_new_records(
        tmp_path / "raw" / "hot", checkpoint, NormalizerConfig())
    assert used == 0
    assert pending is False


def test_rest_branch_rejections(tmp_path: Path) -> None:
    """Non-string grids, bad labels and non-text bodies fail the grid, never the cycle."""
    import gzip as _gzip
    import json as _json

    _journal(tmp_path, "book_ticker", "blue", [
        {"v": 1, "stream": "book_ticker", "slot": "blue", "kind": "rest",
         "recv_ns": _ns(10, 0, 5), "grid": 12345, "status": 200, "body": BOOK_BODY},
        {"v": 1, "stream": "book_ticker", "slot": "blue", "kind": "rest",
         "recv_ns": _ns(10, 1, 5), "grid": "not-a-grid", "status": 200, "body": BOOK_BODY},
    ])
    dest = hot_segment_path(tmp_path, "book_ticker", "blue", _ns(10))
    line = _json.dumps({"v": 1, "stream": "book_ticker", "slot": "blue", "kind": "rest",
                        "recv_ns": _ns(10, 2, 5), "grid": "2026-09-26T10:02:00Z",
                        "status": 200, "body": 12345}, separators=(",", ":")) + "\n"
    with open(dest, "ab") as handle:
        handle.write(_gzip.compress(line.encode(), compresslevel=6))
    _, report = _cycle(tmp_path, tmp_path / "liq", _empty())
    assert report.parse_failures == 2
    assert report.rows_written.get("book_ticker", 0) == 0
    assert not list((tmp_path / "book_ticker").rglob("*.parquet"))


def test_ws_branch_rejections(tmp_path: Path) -> None:
    """Bad frames, undecodable texts and unknown kinds count parse failures."""
    _journal(tmp_path, "force_order", "blue", [
        _ws_open("blue", _ns(10, 0, 1)),
        {"v": 1, "stream": "force_order", "slot": "blue", "kind": "frame",
         "recv_ns": _ns(10, 0, 2), "frame": 12345},
        {"v": 1, "stream": "force_order", "slot": "blue", "kind": "frame",
         "recv_ns": _ns(10, 0, 3), "frame": "not json"},
        {"v": 1, "stream": "force_order", "slot": "blue", "kind": "frame",
         "recv_ns": _ns(10, 0, 4), "frame": "[1,2]"},
        {"v": 1, "stream": "force_order", "slot": "blue", "kind": "mystery",
         "recv_ns": _ns(10, 0, 5)},
        _ws_close("blue", _ns(10, 0, 6)),
    ])
    _, report = _cycle(tmp_path, tmp_path / "liq", _empty())
    assert report.parse_failures == 4
    assert not list((tmp_path / "liq").rglob("*.parquet"))


def test_final_flags_and_pending_bytes(tmp_path: Path) -> None:
    """FINAL flags refresh per cycle; pending bytes count only complete backlog."""
    import os as _os

    from src.market_data.streams.normalizer import (
        FileCursor,
        NormalizerCheckpoint,
        _file_final_flags,
        _pending_complete_bytes,
    )

    grid = "2026-09-26T10:00:00Z"
    _journal(tmp_path, "book_ticker", "blue", [_rest("book_ticker", "blue", grid, _ns(10, 0, 5))])
    hot_root = tmp_path / "raw" / "hot"
    rel = "book_ticker/20260926/10.blue.jsonl.gz"
    old = _now().value // 1_000_000_000 - 3600
    _os.utime(hot_root / rel, (old, old))
    refreshed = _file_final_flags(hot_root, {}, _ns(12), 60.0)
    assert refreshed == {}
    checkpoint, _ = _cycle(tmp_path, tmp_path / "liq", _empty(), now=_now())
    assert checkpoint.files[rel].final is True
    assert _pending_complete_bytes(hot_root, checkpoint) == 0
    corrupt_rel = "book_ticker/20260926/11.blue.jsonl.gz"
    (hot_root / "book_ticker" / "20260926" / "11.blue.jsonl.gz").write_bytes(b"GARBAGE" * 100)
    garbage = NormalizerCheckpoint(files={corrupt_rel: FileCursor(offset=0, final=False)}, coverage={})
    assert _pending_complete_bytes(hot_root, garbage) == (hot_root / rel).stat().st_size + 700
    bad_name = NormalizerCheckpoint(files={"book_ticker/20260926/xx.blue.jsonl.gz": FileCursor(offset=0, final=True)},
                                    coverage={})
    refreshed = _file_final_flags(hot_root, bad_name.files, _ns(12), 60.0)
    assert refreshed == {}


def test_run_recovers_from_corrupt_checkpoint(tmp_path: Path) -> None:
    """A corrupt checkpoint counts failed cycles while the heartbeat reports failing."""
    from src.live.lifecycle import ShutdownFlag
    from src.market_data.streams.normalizer import run_normalizer

    grid = "2026-09-26T10:00:00Z"
    _journal(tmp_path, "book_ticker", "blue", [_rest("book_ticker", "blue", grid, _ns(10, 0, 5))])
    checkpoint_path = tmp_path / "raw" / "normalizer_checkpoint.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("{corrupt", encoding="utf-8")
    now = _now()
    flag = ShutdownFlag()
    sleeps = 0

    def _sleep(delay: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps >= 3:
            flag.requested = True

    run_normalizer(
        tmp_path, tmp_path / "liq", NormalizerConfig(heartbeat_interval_s=3600.0),
        backup_status_path=tmp_path / "missing.json", shutdown=flag,
        now_fn=lambda: now, sleep_fn=_sleep,
    )
    heartbeat = json.loads((tmp_path / "recorder_heartbeat.json").read_text())
    assert heartbeat["normalizer"]["consecutive_failures"] >= 1
    assert heartbeat["normalizer"]["last_error"] is not None


def test_run_drains_pending_without_sleeping(tmp_path: Path) -> None:
    """more_pending repeats derivation immediately inside one outer iteration."""
    from src.live.lifecycle import ShutdownFlag
    from src.market_data.streams.normalizer import run_normalizer

    grids = [f"2026-09-26T10:{m:02d}:00Z" for m in range(0, 30, 5)]
    writer = SegmentWriter(tmp_path, "book_ticker", "blue")
    for grid in grids:
        moment = pd.Timestamp(grid)
        writer.add(_rest("book_ticker", "blue", grid, int(moment.value) + 5_000_000_000))
        writer.flush()
    now = _now()
    flag = ShutdownFlag()
    sleeps = 0

    def _sleep(delay: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps >= 4:
            flag.requested = True

    tiny = NormalizerConfig().model_copy(update={"max_bytes_per_cycle": 100})
    run_normalizer(
        tmp_path, tmp_path / "liq", tiny,
        backup_status_path=tmp_path / "missing.json", shutdown=flag,
        now_fn=lambda: now, sleep_fn=_sleep,
    )
    books = _parquet_rows(tmp_path, "book_ticker")
    assert len(books) == len(grids) * 2


def test_retention_stage_failures_are_counted(tmp_path: Path, monkeypatch) -> None:
    """Sweep, compaction-list, compaction-item and prune failures never escape the pass."""
    from src.market_data.streams import normalizer as normalizer_mod
    from src.market_data.streams.normalizer import _run_retention_pass

    checkpoint = _empty()
    compaction_state: dict = {}
    retention_state: dict = {}
    now = _now()
    checkpoint_path = tmp_path / "raw" / "normalizer_checkpoint.json"

    def _boom(*args, **kwargs):
        raise RuntimeError("stage down")

    monkeypatch.setattr(normalizer_mod, "sweep_partials", _boom)
    out = _run_retention_pass(tmp_path, tmp_path / "liq", NormalizerConfig(), checkpoint,
                              tmp_path / "missing.json", compaction_state, retention_state,
                              now, checkpoint_path)
    assert out[0].files == checkpoint.files
    assert out[0].coverage == checkpoint.coverage
    monkeypatch.undo()
    monkeypatch.setattr(normalizer_mod, "due_compactions", _boom)
    _run_retention_pass(tmp_path, tmp_path / "liq", NormalizerConfig(), checkpoint,
                        tmp_path / "missing.json", dict(compaction_state), dict(retention_state),
                        now, checkpoint_path)
    monkeypatch.undo()
    monkeypatch.setattr(normalizer_mod, "prune_backed_up", _boom)
    _run_retention_pass(tmp_path, tmp_path / "liq", NormalizerConfig(), checkpoint,
                        tmp_path / "missing.json", dict(compaction_state), dict(retention_state),
                        now, checkpoint_path)


def test_compaction_item_failures_recorded(tmp_path: Path, monkeypatch) -> None:
    """Item-level compaction errors record last_result error and continue."""
    from src.common.errors import DataIntegrityError
    from src.market_data.streams import normalizer as normalizer_mod
    from src.market_data.streams.normalizer import _run_retention_pass

    def _due(*args, **kwargs):
        return [("book_ticker", "20260920")]

    def _fail(*args, **kwargs):
        raise DataIntegrityError("bad day")

    def _fail_other(*args, **kwargs):
        raise RuntimeError("weird")

    monkeypatch.setattr(normalizer_mod, "due_compactions", _due)
    monkeypatch.setattr(normalizer_mod, "compact_day", _fail)
    _, compaction_state, _ = _run_retention_pass(
        tmp_path, tmp_path / "liq", NormalizerConfig(), _empty(),
        tmp_path / "missing.json", {}, {}, _now(), tmp_path / "raw" / "normalizer_checkpoint.json")
    assert compaction_state["last_result"] == "error"
    assert compaction_state["last_day"] == "20260920"
    monkeypatch.setattr(normalizer_mod, "compact_day", _fail_other)
    _, compaction_state, _ = _run_retention_pass(
        tmp_path, tmp_path / "liq", NormalizerConfig(), _empty(),
        tmp_path / "missing.json", {}, {}, _now(), tmp_path / "raw" / "normalizer_checkpoint.json")
    assert compaction_state["last_result"] == "error"


def test_heartbeat_publish_skip_and_failure(tmp_path: Path, monkeypatch, caplog) -> None:
    """A fresh publish is skipped; a failed write warns without raising."""
    import logging

    from src.market_data.streams import normalizer as normalizer_mod
    from src.market_data.streams.heartbeat_v3 import new_stream_states
    from src.market_data.streams.normalizer import _maybe_publish_heartbeat

    states = new_stream_states()
    dedupe = DedupeWindow(rest_window_s=7200.0, ws_window_s=7200.0)
    now = _now()
    config = NormalizerConfig()
    args: dict = {
        "capture_root": tmp_path, "config": config, "states": states,
        "intervals": {"book_ticker": 60, "premium_index": 300}, "dedupe": dedupe,
        "checkpoint": _empty(), "consecutive_failures": 0, "last_success_at": None,
        "last_error": None, "last_report": None,
        "compaction_state": {}, "retention_state": {},
        "started_at": now, "now": now,
    }
    first = _maybe_publish_heartbeat(**args, last_run=None)
    assert first == now
    assert _maybe_publish_heartbeat(**args, last_run=now) == now

    def _fail_write(root: Path, payload: object) -> Path:
        raise OSError("read-only")

    monkeypatch.setattr(normalizer_mod, "write_heartbeat_atomic", _fail_write)
    with caplog.at_level(logging.WARNING):
        _maybe_publish_heartbeat(**args, last_run=None)
    assert "HEARTBEAT_FAILED" in caplog.text


def test_heartbeat_publishes_overdue_hot_days_and_restart_window(tmp_path: Path) -> None:
    """Heartbeat lists hot days past the compaction grace and counts expected points since start only."""
    import json

    from src.market_data.streams.heartbeat_v3 import new_stream_states
    from src.market_data.streams.normalizer import _maybe_publish_heartbeat

    (tmp_path / "raw" / "hot" / "book_ticker" / "20260920").mkdir(parents=True)
    (tmp_path / "raw" / "hot" / "book_ticker" / "20260926").mkdir(parents=True)
    now = _now()
    states = new_stream_states()
    _maybe_publish_heartbeat(
        capture_root=tmp_path, config=NormalizerConfig(), states=states,
        intervals={"book_ticker": 60, "premium_index": 300},
        dedupe=DedupeWindow(rest_window_s=7200.0, ws_window_s=7200.0),
        checkpoint=_empty(), consecutive_failures=0, last_success_at=None, last_error=None,
        last_report=None, compaction_state={}, retention_state={},
        started_at=now - pd.Timedelta(minutes=10), now=now, last_run=None,
    )
    payload = json.loads((tmp_path / "recorder_heartbeat.json").read_text())
    assert payload["compaction"]["archived_days_pending"] == [{"stream": "book_ticker", "day": "20260920"}]
    assert states["book_ticker"]["window_expected_points"] == 10
    assert states["premium_index"]["window_expected_points"] == 2


def test_valid_config_exercises_every_validator() -> None:
    """An explicitly valued config runs every validator success path."""
    config = NormalizerConfig(
        normalize_interval_s=30.0,
        segment_final_grace_s=180.0,
        ws_dedupe_window_s=7200.0,
        rest_dedupe_window_s=7200.0,
        max_bytes_per_cycle=67_108_864,
        book_ticker_interval_s=60,
        premium_index_interval_s=300,
        snapshot_max_rejected_fraction=0.05,
        grid_health_window_s=3600.0,
        reference_capture_after_utc="00:05",
        compaction_grace_s=1800.0,
        archive_lzma_preset=6,
        raw_archive_local_retention_days=30,
        parquet_local_retention_days=180,
        backup_status_max_age_h=72.0,
        partial_sweep_age_s=3600.0,
        retention_interval_s=3600.0,
        heartbeat_interval_s=30.0,
    )
    assert config.book_ticker_interval_s == 60
    with pytest.raises(ValidationError):
        NormalizerConfig.model_validate({"normalize_interval_s": "fast"})


def test_branch_helpers_direct() -> None:
    """Timestamp, grid and receipt extractors fail closed on every shape."""
    from src.market_data.streams.normalizer import _grid_ns, _recv_ns_of, _segment_final

    assert _recv_ns_of({}) == 0
    assert _recv_ns_of({"recv_ns": "x"}) == 0
    assert _recv_ns_of({"recv_ns": True}) == 0
    assert _recv_ns_of({"recv_ns": -5}) == 0
    assert _recv_ns_of({"recv_ns": 7}) == 7
    assert _grid_ns("2026-09-26T10:00:00") is None
    assert _grid_ns("nope") is None
    assert _grid_ns(None) is None
    assert _segment_final(Path("/nope"), hour=10, day="99999999",
                          now_ns=0, grace_s=60.0) is False
    assert _segment_final(Path("/nope"), hour=10, day="20260926",
                          now_ns=10**20, grace_s=60.0) is False


def test_gather_skips_stray_names_and_vanished_files(tmp_path: Path, monkeypatch) -> None:
    """Stray files never route; a file vanishing mid-gather is skipped safely."""
    from pathlib import Path as _Path

    from src.market_data.streams.normalizer import _gather_new_records

    grid = "2026-09-26T10:00:00Z"
    _journal(tmp_path, "book_ticker", "blue", [_rest("book_ticker", "blue", grid, _ns(10, 0, 5))])
    stream_dir = tmp_path / "raw" / "hot" / "book_ticker"
    (stream_dir / "notes.txt").write_text("x")
    (stream_dir / "2026-9").mkdir()
    target = hot_segment_path(tmp_path, "book_ticker", "blue", _ns(10))
    real_stat = _Path.stat

    def _vanishing(self: Path, *args: object, **kwargs: object) -> object:
        if self == target:
            raise OSError("gone")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(_Path, "stat", _vanishing)
    gathered, consumed, used, pending, oldest = _gather_new_records(
        tmp_path / "raw" / "hot", _empty(), NormalizerConfig())
    assert gathered == []
    assert used == 0
    assert pending is False


def test_parse_failure_payloads_and_naive_grids(tmp_path: Path) -> None:
    """Unparseable bodies and naive grid labels are failed attempts, never crashes."""
    import gzip as _gzip
    import json as _json

    _journal(tmp_path, "book_ticker", "blue", [
        {"v": 1, "stream": "book_ticker", "slot": "blue", "kind": "rest",
         "recv_ns": _ns(10, 0, 5), "grid": "2026-09-26T10:00:00", "status": 200, "body": BOOK_BODY},
    ])
    dest = hot_segment_path(tmp_path, "book_ticker", "blue", _ns(10))
    for body, grid, recv in (('"oops"', "2026-09-26T10:01:00Z", _ns(10, 1, 5)),
                             (BOOK_BODY, "2026-09-26T10:02:00Z", _ns(10, 2, 5))):
        line = _json.dumps({"v": 1, "stream": "book_ticker", "slot": "blue", "kind": "rest",
                            "recv_ns": recv, "grid": grid, "status": 200, "body": body},
                           separators=(",", ":")) + "\n"
        with open(dest, "ab") as handle:
            handle.write(_gzip.compress(line.encode(), compresslevel=6))
    _, report = _cycle(tmp_path, tmp_path / "liq", _empty())
    assert report.rows_written.get("book_ticker", 0) == 2
    assert report.parse_failures == 1


def test_ws_red_slot_and_bad_hour_file(tmp_path: Path) -> None:
    """A red-slot forceOrder record fails; an unparseable hour name is never FINAL."""
    from src.market_data.streams.normalizer import _file_final_flags

    _journal(tmp_path, "force_order", "blue", [
        {"v": 1, "stream": "force_order", "slot": "red", "kind": "frame",
         "recv_ns": _ns(10, 0, 2), "frame": FRAME_TEXT},
    ])
    _, report = _cycle(tmp_path, tmp_path / "liq", _empty())
    assert report.parse_failures == 1
    hot_root = tmp_path / "raw" / "hot"
    stray = hot_root / "force_order" / "20260926" / "xx.blue.jsonl.gz"
    stray.write_bytes(b"junk")
    from src.market_data.streams.normalizer import FileCursor

    refreshed = _file_final_flags(
        hot_root, {"force_order/20260926/xx.blue.jsonl.gz": FileCursor(offset=0, final=True)},
        _ns(12), 60.0)
    assert next(iter(refreshed.values())).final is False


def test_pending_bytes_skips_vanished_files(tmp_path: Path, monkeypatch) -> None:
    """A file vanishing during backlog accounting is skipped safely."""
    from pathlib import Path as _Path

    from src.market_data.streams.normalizer import _pending_complete_bytes

    grid = "2026-09-26T10:00:00Z"
    _journal(tmp_path, "book_ticker", "blue", [_rest("book_ticker", "blue", grid, _ns(10, 0, 5))])
    hot_root = tmp_path / "raw" / "hot"
    assert _pending_complete_bytes(hot_root, _empty()) > 0
    real_stat = _Path.stat
    target = hot_root / "book_ticker" / "20260926" / "10.blue.jsonl.gz"

    def _vanishing(self: Path, *args: object, **kwargs: object) -> object:
        if self == target:
            raise OSError("gone")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(_Path, "stat", _vanishing)
    assert _pending_complete_bytes(hot_root, _empty()) == 0


def test_sleep_interval_returns_at_deadline() -> None:
    """A clock that outruns the cadence ends the sleep without shutdown."""
    from src.live.lifecycle import ShutdownFlag
    from src.market_data.streams.normalizer import _sleep_interval

    now = [pd.Timestamp("2026-09-26T12:00:00Z")]
    flag = ShutdownFlag()
    calls = []

    def _sleep(delay: float) -> None:
        calls.append(delay)
        now[0] += pd.Timedelta(seconds=delay * 10)

    _sleep_interval(_sleep, flag, 30.0, lambda: now[0])
    assert calls
    assert flag.requested is False




def test_checkpoint_naive_timestamps_rejected(tmp_path: Path) -> None:
    """Timezone-naive coverage instants fail closed instead of silently shifting."""
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({"files": {}, "coverage": {"blue": {"open_last": "2026-09-26T10:00:00"}}}))
    with pytest.raises(DataIntegrityError, match="checkpoint"):
        load_checkpoint(path)


def test_rest_invalid_json_body_is_failed_attempt(tmp_path: Path) -> None:
    """A non-JSON text body counts as a failed grid, never a cycle error."""
    _journal(tmp_path, "book_ticker", "blue", [
        {"v": 1, "stream": "book_ticker", "slot": "blue", "kind": "rest",
         "recv_ns": _ns(10, 0, 5), "grid": "2026-09-26T10:00:00Z", "status": 200, "body": "not json{{"},
    ])
    _, report = _cycle(tmp_path, tmp_path / "liq", _empty())
    assert report.rows_written.get("book_ticker", 0) == 0
    assert report.parse_failures == 0


def test_utc_now_returns_aware_now() -> None:
    """The default loop clock returns tz-aware UTC."""
    from src.market_data.streams.normalizer import _utc_now

    now = _utc_now()
    assert now.tzinfo is not None


def test_run_counts_normalize_exceptions(tmp_path: Path, monkeypatch) -> None:
    """A raising derivation stage counts a failed cycle and keeps the loop alive."""
    import src.market_data.streams.normalizer as normalizer_mod
    from src.live.lifecycle import ShutdownFlag

    calls = {"n": 0}
    real_once = normalizer_mod.normalize_once
    flag = ShutdownFlag()

    def _boom(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            if calls["n"] == 2:
                flag.requested = True
            raise RuntimeError("stage down")
        return real_once(*args, **kwargs)

    monkeypatch.setattr(normalizer_mod, "normalize_once", _boom)
    now = _now()
    clock = [now]
    sleeps = 0

    def _sleep(delay: float) -> None:
        nonlocal sleeps
        sleeps += 1
        clock[0] += pd.Timedelta(seconds=delay)
        if sleeps >= 200:
            flag.requested = True

    normalizer_mod.run_normalizer(
        tmp_path, tmp_path / "liq", NormalizerConfig(heartbeat_interval_s=3600.0),
        backup_status_path=tmp_path / "missing.json", shutdown=flag,
        now_fn=lambda: clock[0], sleep_fn=_sleep,
    )
    heartbeat = json.loads((tmp_path / "recorder_heartbeat.json").read_text())
    assert heartbeat["normalizer"]["consecutive_failures"] >= 1
    assert calls["n"] >= 2


def test_retention_pass_sweeps_before_pruning(tmp_path: Path) -> None:
    """The retention pass sweeps temps, then prunes, logging the sweep count."""
    from src.market_data.streams.normalizer import _run_retention_pass

    raw = tmp_path / "raw"
    raw.mkdir()
    doomed = raw / "z.partial"
    doomed.write_bytes(b"x")
    old = _now().value // 1_000_000_000 - 7200
    import os as _os

    _os.utime(doomed, (old, old))
    checkpoint, compaction_state, retention_state = _run_retention_pass(
        tmp_path, tmp_path / "liq", NormalizerConfig(), _empty(),
        tmp_path / "missing.json", {}, {}, _now(), tmp_path / "raw" / "normalizer_checkpoint.json")
    assert not doomed.exists()
    assert retention_state["prune_blocked"] is True
    assert retention_state["blocked_reason"] == "status_missing"


def _write_backup_status(path: Path, *, finished: pd.Timestamp, rc: int = 0) -> None:
    path.write_text(
        json.dumps(
            {
                "started_at": (finished - pd.Timedelta(minutes=1)).isoformat(),
                "finished_at": finished.isoformat(),
                "rc": rc,
            }
        ),
        encoding="utf-8",
    )


def test_blocked_since_is_set_kept_persisted_and_cleared(tmp_path: Path) -> None:
    """The block start survives passes and restarts, and clears with the block."""
    from src.market_data.streams.normalizer import _run_retention_pass, load_checkpoint

    ckpt_path = tmp_path / "raw" / "normalizer_checkpoint.json"
    status_path = tmp_path / "last_success.json"
    first = _now()
    checkpoint, _, state = _run_retention_pass(
        tmp_path,
        tmp_path / "liq",
        NormalizerConfig(),
        _empty(),
        status_path,
        {},
        {"pruned_files_total": 0},
        first,
        ckpt_path,
    )
    assert state["prune_blocked"] is True
    assert state["blocked_since"] == first.isoformat()
    assert checkpoint.retention_blocked_since == first.isoformat()
    assert load_checkpoint(ckpt_path).retention_blocked_since == first.isoformat()

    later = first + pd.Timedelta(hours=2)
    checkpoint, _, state = _run_retention_pass(
        tmp_path,
        tmp_path / "liq",
        NormalizerConfig(),
        load_checkpoint(ckpt_path),
        status_path,
        {},
        state,
        later,
        ckpt_path,
    )
    assert state["blocked_since"] == first.isoformat()

    _write_backup_status(status_path, finished=later)
    checkpoint, _, state = _run_retention_pass(
        tmp_path,
        tmp_path / "liq",
        NormalizerConfig(),
        checkpoint,
        status_path,
        {},
        state,
        later,
        ckpt_path,
    )
    assert state["prune_blocked"] is False
    assert state["blocked_since"] is None
    assert checkpoint.retention_blocked_since is None
    assert load_checkpoint(ckpt_path).retention_blocked_since is None
    assert isinstance(state["footprint_bytes"], int)


def test_checkpoint_rejects_malformed_blocked_since(tmp_path: Path) -> None:
    """A checkpoint with a non-ISO block start fails closed instead of resetting the timer."""
    from src.market_data.streams.normalizer import load_checkpoint

    path = tmp_path / "ckpt.json"
    path.write_text(json.dumps({"files": {}, "coverage": {}, "retention_blocked_since": "not-a-time"}))
    with pytest.raises(DataIntegrityError):
        load_checkpoint(path)


def test_retention_due_only_when_block_can_clear(tmp_path: Path) -> None:
    """Between cadence ticks the pass re-runs only if it is blocked and a fresh success status exists."""
    from src.market_data.streams.normalizer import _retention_due

    config = NormalizerConfig()
    status_path = tmp_path / "last_success.json"
    now = _now()
    recent = now - pd.Timedelta(seconds=60)
    blocked: dict[str, Any] = {"prune_blocked": True}
    assert _retention_due(None, blocked, config, status_path, now) is True
    assert _retention_due(recent, blocked, config, status_path, now) is False
    _write_backup_status(status_path, finished=now)
    assert _retention_due(recent, blocked, config, status_path, now) is True
    assert _retention_due(recent, {"prune_blocked": False}, config, status_path, now) is False
    _write_backup_status(status_path, finished=now - pd.Timedelta(hours=100))
    assert _retention_due(recent, blocked, config, status_path, now) is False
    assert _retention_due(now - pd.Timedelta(hours=2), {"prune_blocked": False}, config, status_path, now) is True


def test_run_clears_prune_block_within_one_cycle_of_status_appearing(tmp_path: Path) -> None:
    """The loop re-evaluates a blocked prune every cycle, not once per retention interval."""
    import src.market_data.streams.normalizer as normalizer_mod
    from src.live.lifecycle import ShutdownFlag

    status_path = tmp_path / "last_success.json"
    clock = [_now()]
    flag = ShutdownFlag()
    sleeps = 0
    seen: list[bool] = []

    def _sleep(delay: float) -> None:
        nonlocal sleeps
        sleeps += 1
        clock[0] += pd.Timedelta(seconds=delay)
        heartbeat = tmp_path / "recorder_heartbeat.json"
        if heartbeat.exists():
            seen.append(json.loads(heartbeat.read_text())["retention"]["prune_blocked"])
        if sleeps == 40:
            _write_backup_status(status_path, finished=clock[0])
        if sleeps >= 120:
            flag.requested = True

    normalizer_mod.run_normalizer(
        tmp_path,
        tmp_path / "liq",
        NormalizerConfig(heartbeat_interval_s=30.0),
        backup_status_path=status_path,
        shutdown=flag,
        now_fn=lambda: clock[0],
        sleep_fn=_sleep,
    )
    assert True in seen
    assert seen[-1] is False
    heartbeat = json.loads((tmp_path / "recorder_heartbeat.json").read_text())
    assert heartbeat["retention"]["blocked_since"] is None
    assert (
        normalizer_mod.load_checkpoint(tmp_path / "raw" / "normalizer_checkpoint.json").retention_blocked_since is None
    )


def test_restart_keeps_blocked_since_from_checkpoint(tmp_path: Path) -> None:
    """A restarted normalizer publishes the persisted block start instead of restarting the clock."""
    import src.market_data.streams.normalizer as normalizer_mod
    from src.live.lifecycle import ShutdownFlag

    started = _now()
    ckpt_path = tmp_path / "raw" / "normalizer_checkpoint.json"
    normalizer_mod.save_checkpoint(
        ckpt_path,
        NormalizerCheckpoint(
            files={},
            coverage={},
            retention_blocked_since=(started - pd.Timedelta(hours=5)).isoformat(),
        ),
    )
    clock = [started]
    flag = ShutdownFlag()

    def _sleep(delay: float) -> None:
        clock[0] += pd.Timedelta(seconds=delay)
        flag.requested = True

    normalizer_mod.run_normalizer(
        tmp_path,
        tmp_path / "liq",
        NormalizerConfig(heartbeat_interval_s=30.0),
        backup_status_path=tmp_path / "missing.json",
        shutdown=flag,
        now_fn=lambda: clock[0],
        sleep_fn=_sleep,
    )
    heartbeat = json.loads((tmp_path / "recorder_heartbeat.json").read_text())
    assert heartbeat["retention"]["prune_blocked"] is True
    assert heartbeat["retention"]["blocked_since"] == (started - pd.Timedelta(hours=5)).isoformat()


def test_config_rejects_derive_lag_shorter_than_cycle() -> None:
    """The derive-lag allowance must cover at least one normalize cycle."""
    with pytest.raises(ValidationError):
        NormalizerConfig(normalize_interval_s=30.0, derive_lag_allowance_s=10.0)


def test_successful_compaction_drops_only_that_days_cursors(tmp_path: Path, monkeypatch) -> None:
    """A compacted day's hot cursors are removed while other days and the block start survive."""
    from src.market_data.streams import normalizer as normalizer_mod
    from src.market_data.streams.normalizer import FileCursor, _run_retention_pass

    monkeypatch.setattr(normalizer_mod, "due_compactions", lambda *a, **k: [("book_ticker", "20260920")])
    monkeypatch.setattr(normalizer_mod, "compact_day", lambda *a, **k: True)
    since = (_now() - pd.Timedelta(hours=1)).isoformat()
    checkpoint = NormalizerCheckpoint(
        files={
            "book_ticker/20260920/00.blue.jsonl.gz": FileCursor(offset=5, final=True),
            "book_ticker/20260921/00.blue.jsonl.gz": FileCursor(offset=7, final=False),
        },
        coverage={},
        retention_blocked_since=since,
    )
    out, compaction_state, retention_state = _run_retention_pass(
        tmp_path, tmp_path / "liq", NormalizerConfig(), checkpoint, tmp_path / "missing.json",
        {}, {"pruned_files_total": 0}, _now(), tmp_path / "raw" / "normalizer_checkpoint.json",
    )
    assert set(out.files) == {"book_ticker/20260921/00.blue.jsonl.gz"}
    assert compaction_state["last_result"] == "ok"
    assert out.retention_blocked_since == since
    assert retention_state["blocked_since"] == since
