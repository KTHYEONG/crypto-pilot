"""Invariant scenarios for the paper cohort of the fill calibration report."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import src.application.fill_calibration_paper as fcp
from src.application.fill_calibration import run_fill_calibration
from src.common.errors import DataIntegrityError
from src.market_data.public_archive.catalog import AGG_TRADES_SLICE_COLUMNS
from tests.unit.application._fill_calibration_helpers import (
    _ts,
    _request,
    _toy_parts,
    _install_tape,
    _slice_frame,
    _coverage_row,
    _day_ms,
    _paper_evidence,
    _paper_run,
)


def test_paper_confusion_uses_observed_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal

    parts = _toy_parts()
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    day = date(2023, 1, 2)
    paper_dir = tmp_path / "run"
    (paper_dir / "audit").mkdir(parents=True)
    post_ts = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=23, minutes=1)
    base_ms = int(post_ts.value // 10**6)
    tape_frame = _slice_frame(day, [(base_ms + (i + 1) * 10_000, i + 1, 100.0, 3.0, True) for i in range(4)])
    (paper_dir / "audit" / "a.jsonl").write_text(
        json.dumps(
            {
                "event": "order_posted",
                "client_order_id": "c1",
                "symbol": "AAAUSDT",
                "side": "BUY",
                "time_in_force": "GTX",
                "price": "100",
                "quantity": "5",
                "ts": post_ts.isoformat(),
            }
        )
        + "\n"
        + json.dumps(
            {
                "event": "fill",
                "qty": "5",
                "client_order_id": "c1",
                "symbol": "AAAUSDT",
                "ts": (post_ts + pd.Timedelta(minutes=1)).isoformat(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (paper_dir / "run_manifest.json").write_text(json.dumps({"mode": "paper"}), encoding="utf-8")
    live_root = tmp_path / "live"
    depth_dir = live_root / "exec_depth" / "20230102"
    depth_dir.mkdir(parents=True)
    depth_row = pd.DataFrame(
        {
            "symbol": ["AAAUSDT"],
            "transact_time_ms": [int(post_ts.value // 10**6)],
            "bid_px_0": [100.0],
            "bid_qty_0": [10.0],
            "ask_px_0": [101.0],
            "ask_qty_0": [10.0],
        }
    )
    depth_row.to_parquet(depth_dir / "part_230000_0.parquet")
    from src.market_data.public_archive import store

    monkeypatch.setattr(
        store,
        "load_coverage",
        lambda root, dataset: {
            ("AAAUSDT", day): _coverage_row("AAAUSDT", day, "ok"),
            ("BBBUSDT", day): _coverage_row("BBBUSDT", day, "ok"),
            ("AAAUSDT", date(2023, 1, 3)): _coverage_row("AAAUSDT", date(2023, 1, 3), "ok"),
            ("BBBUSDT", date(2023, 1, 3)): _coverage_row("BBBUSDT", date(2023, 1, 3), "ok"),
        },
    )
    for sym in ("AAAUSDT", "BBBUSDT"):
        p = store.partition_path(tmp_path / "archive", "aggTrades", sym, "202301", kind="slice")
        p.parent.mkdir(parents=True, exist_ok=True)
        tape_frame.to_parquet(p)
    req = _request(tmp_path, paper_run_dir=paper_dir, live_capture_root=live_root)
    report = run_fill_calibration(req, output_dir=tmp_path / "out")
    assert report.paper is not None
    through = report.paper["variants"]["through_only"]["confusion"]
    queued = report.paper["variants"]["queue_observed"]["confusion"]
    front = report.paper["variants"]["front_of_queue"]["confusion"]
    assert through["paper_only"] == 1
    assert queued["both_filled"] == 1
    assert front["both_filled"] == 1
    scored = pd.read_parquet(report.intents_path)
    paper_fills = scored[scored["cohort"] == "paper"].set_index("rule")["fill"].to_dict()
    assert paper_fills == pytest.approx({"through_only": 0.0, "queue_observed": 2 / 5, "front_of_queue": 1.0})


def test_non_paper_run_dir_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps({"mode": "live"}), encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        run_fill_calibration(
            _request(tmp_path, paper_run_dir=run_dir, live_capture_root=tmp_path), output_dir=tmp_path / "out"
        )


def test_paper_partial_fill_does_not_terminate_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stamp = _ts("2023-01-02T23:01:00Z")
    base = int(stamp.value // 10**6)
    _, live = _paper_evidence(tmp_path, monkeypatch, [(base + 40_000, 1, 99.0, 10.0, True)])
    records = [
        {"event": "order_posted", "client_order_id": "partial", "side": "BUY", "symbol": "AAAUSDT", "price": "100", "quantity": "5", "time_in_force": "GTX", "ts": stamp.isoformat()},
        {"event": "fill", "client_order_id": "partial", "qty": "2", "ts": (stamp + pd.Timedelta(seconds=10)).isoformat()},
        {"event": "fill", "client_order_id": "partial", "qty": "3", "ts": (stamp + pd.Timedelta(minutes=1)).isoformat()},
    ]
    run = _paper_run(tmp_path / "paper", [json.dumps(row) + "\n" for row in records])
    report = run_fill_calibration(_request(tmp_path, paper_run_dir=run, live_capture_root=live), output_dir=tmp_path / "out")
    assert report.paper["variants"]["through_only"]["confusion"]["both_filled"] == 1
    frame = pd.read_parquet(report.intents_path)
    paper = frame[(frame["cohort"] == "paper") & (frame["rule"] == "through_only")].iloc[0]
    assert paper["fill"] == 1.0
    assert paper["paper_fill_time_min"] == pytest.approx(1 / 6)
    assert paper["fill_time_min"] == pytest.approx(2 / 3)


def test_paper_cancel_ignores_late_fill_and_gap_is_notional_weighted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stamp, live = _paper_evidence(tmp_path, monkeypatch)
    rows = []
    for cid, price, qty in (("filled", 100, 5), ("missed", 90, 50)):
        rows.extend([
            {"event": "order_posted", "client_order_id": cid, "side": "BUY", "symbol": "AAAUSDT", "price": str(price), "quantity": str(qty), "time_in_force": "GTX", "ts": stamp.isoformat()},
            {"event": "fill", "client_order_id": cid, "qty": str(qty), "ts": (stamp + pd.Timedelta(minutes=1)).isoformat()},
        ])
    rows.extend([
        {"event": "order_posted", "client_order_id": "late", "side": "BUY", "symbol": "AAAUSDT", "price": "100", "quantity": "5", "time_in_force": "GTX", "ts": stamp.isoformat()},
        {"event": "order_cancelled", "client_order_id": "late", "ts": (stamp + pd.Timedelta(seconds=5)).isoformat()},
        {"event": "fill", "client_order_id": "late", "qty": "5", "ts": (stamp + pd.Timedelta(minutes=2)).isoformat()},
    ])
    run = _paper_run(tmp_path / "paper", [json.dumps(row) + "\n" for row in rows])
    report = run_fill_calibration(_request(tmp_path, paper_run_dir=run, live_capture_root=live), output_dir=tmp_path / "out")
    assert report.paper["variants"]["through_only"]["fill_rate_gap"] == pytest.approx(-4500 / 5500)
    assert report.paper["variants"]["through_only"]["confusion"] == {"both_filled": 1, "paper_only": 1, "neither": 1, "tape_only": 0}


def test_real_paper_post_side_comes_from_read_only_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stamp = _ts("2023-01-02T23:01:00Z")
    base = int(stamp.value // 10**6)
    _, live = _paper_evidence(tmp_path, monkeypatch, [(base + 20_000, 1, 101.0, 10.0, False)])
    run = _paper_run(tmp_path / "paper", [])
    events = [
        {"event": "order_posted", "client_order_id": "sell", "symbol": "AAAUSDT", "price": "100", "quantity": "5", "time_in_force": "GTX", "ts": stamp.isoformat()},
        {"event": "fill", "client_order_id": "sell", "qty": "5", "ts": (stamp + pd.Timedelta(minutes=1)).isoformat()},
    ]
    (run / "audit" / "a.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    with pytest.raises(DataIntegrityError, match="no side"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run, live_capture_root=live), output_dir=tmp_path / "missing")
    journal = run / "order_journal.jsonl"
    journal.write_text(json.dumps({"event": "submit", "client_order_id": "sell", "side": "SELL"}) + "\n")
    before = journal.read_bytes()
    report = run_fill_calibration(_request(tmp_path, paper_run_dir=run, live_capture_root=live), output_dir=tmp_path / "out")
    assert report.paper["variants"]["through_only"]["confusion"]["both_filled"] == 1
    assert journal.read_bytes() == before


def test_queue_snapshot_cannot_arrive_after_post(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stamp, live = _paper_evidence(tmp_path, monkeypatch)
    directory = live / "exec_depth" / "20230102"
    directory.mkdir(parents=True)
    pd.DataFrame({
        "symbol": ["AAAUSDT"], "received_at": [stamp + pd.Timedelta(milliseconds=1)],
        "transact_time_ms": [int(stamp.value // 10**6) - 100],
        "bid_px_0": [100.0], "bid_qty_0": [1000.0], "ask_px_0": [101.0], "ask_qty_0": [1.0],
    }).to_parquet(directory / "part.parquet")
    run = _paper_run(tmp_path / "paper", [
        json.dumps({"event": "order_posted", "client_order_id": "p", "side": "BUY", "symbol": "AAAUSDT", "price": "100", "quantity": "5", "time_in_force": "GTX", "ts": stamp.isoformat()}) + "\n",
        json.dumps({"event": "order_cancelled", "client_order_id": "p", "ts": (stamp + pd.Timedelta(minutes=1)).isoformat()}) + "\n",
    ])
    report = run_fill_calibration(_request(tmp_path, paper_run_dir=run, live_capture_root=live), output_dir=tmp_path / "out")
    assert report.paper["queue_fallback_count"] == 1


def test_paper_periods_and_requested_window_never_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal
    from src.market_data.public_archive import store

    design_stamp, live = _paper_evidence(tmp_path, monkeypatch)
    holdout_stamp = _ts("2026-07-02T23:01:00Z")
    day = holdout_stamp.date()
    frame = _slice_frame(day, [(int(holdout_stamp.value // 10**6) + 20_000, 1, 99.0, 10.0, True)])
    path = store.partition_path(tmp_path / "archive", "aggTrades", "AAAUSDT", "202607", kind="slice")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    coverage = store.load_coverage(tmp_path / "archive", "aggTrades")
    coverage[("AAAUSDT", day)] = _coverage_row("AAAUSDT", day, "ok")
    monkeypatch.setattr(store, "load_coverage", lambda *args: coverage)
    events = []
    for cid, stamp in (("d", design_stamp), ("h", holdout_stamp)):
        events.extend([
            {"event": "order_posted", "client_order_id": cid, "side": "BUY", "symbol": "AAAUSDT", "price": "100", "quantity": "5", "time_in_force": "GTX", "ts": stamp.isoformat()},
            {"event": "fill", "client_order_id": cid, "qty": "5", "ts": (stamp + pd.Timedelta(minutes=1)).isoformat()},
        ])
    run = _paper_run(tmp_path / "paper", [json.dumps(row) + "\n" for row in events])
    full = run_fill_calibration(_request(tmp_path, evaluation_end=_ts("2026-07-03T00:00:00Z"), paper_run_dir=run, live_capture_root=live), output_dir=tmp_path / "full")
    assert set(full.paper["periods"]) == {"design", "holdout"}
    assert "variants" not in full.paper
    design = run_fill_calibration(_request(tmp_path, paper_run_dir=run, live_capture_root=live), output_dir=tmp_path / "design")
    assert set(design.paper["periods"]) == {"design"}
    assert cal._round_json(design.paper["periods"]["design"]) == cal._round_json(full.paper["periods"]["design"])


def test_journal_corruption_and_direction_conflicts_are_rejected(tmp_path: Path) -> None:

    stamp = _ts("2023-01-02T23:01:00Z")
    run = _paper_run(tmp_path, [json.dumps({"event": "order_posted", "client_order_id": "p", "side": "BUY", "symbol": "AAAUSDT", "price": "100", "quantity": "5", "time_in_force": "GTX", "ts": stamp.isoformat()}) + "\n"])
    journal = run / "order_journal.jsonl"
    journal.write_text("[]\n")
    with pytest.raises(DataIntegrityError, match="malformed paper order journal"):
        fcp.load_paper_posts(run)
    journal.write_text(json.dumps({"event": "submit", "client_order_id": "p", "side": "SELL"}) + "\n")
    with pytest.raises(DataIntegrityError, match="disagrees with journal"):
        fcp.load_paper_posts(run)
    journal.write_text("")
    with (run / "audit" / "a.jsonl").open("a") as handle:
        handle.write(json.dumps({"event": "fill", "client_order_id": "p", "qty": "0", "ts": stamp.isoformat()}) + "\n")
    with pytest.raises(DataIntegrityError, match="fill quantity"):
        fcp.load_paper_posts(run)
    with pytest.raises(DataIntegrityError, match="NaT"):
        fcp.parse_ts("NaT")
    with pytest.raises(DataIntegrityError, match="quantities must be non-negative"):
        fcp.queue_ahead("BUY", 100, pd.Series({"bid_px_0": 100, "bid_qty_0": -1}), 1)
    with pytest.raises(DataIntegrityError, match="malformed exec_depth quantities"):
        fcp.queue_ahead("BUY", 100, pd.Series({"bid_px_0": 100}), 1)


def test_partition_pruning_caching_and_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import itertools

    stamp, live = _paper_evidence(tmp_path, monkeypatch)
    ignored = live / "exec_depth" / "20180101"
    ignored.mkdir(parents=True)
    (ignored / "corrupt.parquet").write_bytes(b"must never be read")
    posts = []
    for cid in ("a", "b"):
        posts.extend([
            {"event": "order_posted", "client_order_id": cid, "side": "BUY", "symbol": "AAAUSDT", "price": "100", "quantity": "5", "time_in_force": "GTX", "ts": stamp.isoformat()},
            {"event": "order_cancelled", "client_order_id": cid, "ts": (stamp + pd.Timedelta(minutes=1)).isoformat()},
        ])
    run = _paper_run(tmp_path / "paper", [json.dumps(row) + "\n" for row in posts])
    reads = []
    real_read = pd.read_parquet
    def read(path, **kwargs):
        if "aggTrades_slice" in str(path):
            reads.append((str(path), kwargs.get("columns")))
        return real_read(path, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", read)
    clock = itertools.count(step=61)
    monkeypatch.setattr(fcp.time, "monotonic", lambda: next(clock))
    with caplog.at_level("INFO"):
        report = run_fill_calibration(_request(tmp_path, paper_run_dir=run, live_capture_root=live), output_dir=tmp_path / "out")
    assert len(reads) == 3
    assert all(tuple(columns) == AGG_TRADES_SLICE_COLUMNS for _, columns in reads)
    assert "cohort=paper done=" in caplog.text
    assert "cohort=backtest done=" in caplog.text
    assert report.paper["variants"]["through_only"]["confusion"]["tape_only"] == 2


def test_paper_audit_error_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal

    parts = _toy_parts()
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    day = date(2023, 1, 2)
    cov = {(s, day): _coverage_row(s, day, "ok") for s in ("AAAUSDT", "BBBUSDT")} | {
        (s, date(2023, 1, 3)): _coverage_row(s, date(2023, 1, 3), "ok") for s in ("AAAUSDT", "BBBUSDT")
    }
    frame = _slice_frame(day, [(_day_ms(day, 23.1), 1, 100.0, 1.0, True)])
    _install_tape(monkeypatch, tmp_path / "archive", {("AAAUSDT", "202301"): frame, ("BBBUSDT", "202301"): frame}, cov)
    live = tmp_path / "live"
    live.mkdir()
    with pytest.raises(DataIntegrityError, match="non-paper"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=_paper_run(tmp_path, [], mode="live"), live_capture_root=live), output_dir=tmp_path / "o0")
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text("not json", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="manifest"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_dir, live_capture_root=live), output_dir=tmp_path / "o1")
    run_dir3 = tmp_path / "run3"
    run_dir3.mkdir()
    (run_dir3 / "run_manifest.json").write_text(json.dumps({"mode": "paper"}), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="audit directory missing"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_dir3, live_capture_root=live), output_dir=tmp_path / "o2")
    (run_dir3 / "audit").mkdir()
    with pytest.raises(DataIntegrityError, match="no jsonl"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_dir3, live_capture_root=live), output_dir=tmp_path / "o3")
    post_ts = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=23, minutes=1)
    run_ok = _paper_run(tmp_path / "ok", ["\n", json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n", "oops\n"])
    with pytest.raises(DataIntegrityError, match="malformed paper audit line"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok, live_capture_root=live), output_dir=tmp_path / "o4")
    run_ok2 = _paper_run(tmp_path / "ok2", ["[1]\n"])
    with pytest.raises(DataIntegrityError, match="malformed paper audit line"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok2, live_capture_root=live), output_dir=tmp_path / "o5")
    run_ok3 = _paper_run(
        tmp_path / "ok3",
        [
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "IOC", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
            json.dumps({"event": "order_posted", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "bad", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
        ],
    )
    with pytest.raises(DataIntegrityError, match="malformed order_posted"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok3, live_capture_root=live), output_dir=tmp_path / "o6")
    run_ok4 = _paper_run(
        tmp_path / "ok4",
        [
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "0", "ts": post_ts.isoformat()}) + "\n",
        ],
    )
    with pytest.raises(DataIntegrityError, match="malformed order_posted"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok4, live_capture_root=live), output_dir=tmp_path / "o7")
    run_ok5 = _paper_run(
        tmp_path / "ok5",
        [
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5"}) + "\n",
        ],
    )
    with pytest.raises(DataIntegrityError, match="missing ts"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok5, live_capture_root=live), output_dir=tmp_path / "o8")
    run_ok6 = _paper_run(
        tmp_path / "ok6",
        [
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": "not-a-time"}) + "\n",
        ],
    )
    with pytest.raises(DataIntegrityError, match="malformed audit ts"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok6, live_capture_root=live), output_dir=tmp_path / "o9")
    run_ok7 = _paper_run(
        tmp_path / "ok7",
        [
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": "2023-01-02 23:01:00"}) + "\n",
            json.dumps({"event": "order_cancelled", "client_order_id": "c1", "ts": "2023-01-02 23:02:00"}) + "\n",
        ],
    )
    report = run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok7, live_capture_root=live), output_dir=tmp_path / "o10")
    assert report.paper is not None
    run_ok8 = _paper_run(
        tmp_path / "ok8",
        [
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
            json.dumps({"event": "order_cancelled", "client_order_id": "c1", "ts": "bad-ts"}) + "\n",
        ],
    )
    with pytest.raises(DataIntegrityError, match="malformed order_cancelled"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok8, live_capture_root=live), output_dir=tmp_path / "o11")
    run_ok9 = _paper_run(
        tmp_path / "ok9",
        [
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
            json.dumps({"event": "fill", "client_order_id": "c1", "ts": "bad-ts"}) + "\n",
        ],
    )
    with pytest.raises(DataIntegrityError, match="malformed fill record"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok9, live_capture_root=live), output_dir=tmp_path / "o12")


def test_paper_depth_and_outcome_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal
    from src.market_data.public_archive import store

    parts = _toy_parts()
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    day = date(2023, 1, 2)
    post_ts = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=23, minutes=1)
    base_ms = int(post_ts.value // 10**6)
    tape_frame = _slice_frame(day, [(base_ms + 10_000, 1, 100.0, 50.0, False)])
    for sym in ("AAAUSDT", "BBBUSDT"):
        p = store.partition_path(tmp_path / "archive", "aggTrades", sym, "202301", kind="slice")
        p.parent.mkdir(parents=True, exist_ok=True)
        tape_frame.to_parquet(p)
    monkeypatch.setattr(store, "load_coverage", lambda root, dataset: {(s, day): _coverage_row(s, day, "ok") for s in ("AAAUSDT", "BBBUSDT")} | {(s, date(2023, 1, 3)): _coverage_row(s, date(2023, 1, 3), "ok") for s in ("AAAUSDT", "BBBUSDT")})
    live = tmp_path / "live"
    depth_dir = live / "exec_depth" / "20230102"
    depth_dir.mkdir(parents=True)
    depth_row = pd.DataFrame({"symbol": ["AAAUSDT"], "transact_time_ms": [base_ms], "bid_px_0": [100.0], "bid_qty_0": [4.0], "ask_px_0": [101.0], "ask_qty_0": [4.0]})
    depth_row.to_parquet(depth_dir / "part_0.parquet")
    run_cancel = _paper_run(
        tmp_path / "cancel",
        [
            json.dumps({"event": "order_posted", "client_order_id": "s1", "symbol": "AAAUSDT", "side": "SELL", "time_in_force": "GTX", "price": "101", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
            json.dumps({"event": "order_cancelled", "client_order_id": "s1", "ts": (post_ts + pd.Timedelta(minutes=5)).isoformat()}) + "\n",
            json.dumps({"event": "order_posted", "client_order_id": "u1", "symbol": "AAAUSDT", "side": "BUY", "time_in_force": "GTX", "price": "50", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
            json.dumps({"event": "order_posted", "client_order_id": "u2", "symbol": "AAAUSDT", "side": "BUY", "time_in_force": "GTX", "price": "50", "quantity": "5", "ts": (post_ts - pd.Timedelta(minutes=5)).isoformat()}) + "\n",
            json.dumps({"event": "order_cancelled", "client_order_id": "u2", "ts": post_ts.isoformat()}) + "\n",
        ],
    )
    report = run_fill_calibration(_request(tmp_path, paper_run_dir=run_cancel, live_capture_root=live), output_dir=tmp_path / "oc")
    assert report.paper is not None
    assert report.exclusions.get("paper_post_unresolved", 0) >= 1
    both = sum(v["confusion"]["tape_only"] + v["confusion"]["neither"] for v in report.paper["variants"].values())
    assert both >= 3
    assert report.paper["variants"]["queue_observed"]["confusion"]["neither"] == 1
    assert report.paper["optimistic_full_fill_share"] == 1.0
    run_stale = _paper_run(
        tmp_path / "stale",
        [json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "side": "BUY", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n", json.dumps({"event": "fill", "client_order_id": "c1", "ts": (post_ts + pd.Timedelta(minutes=1)).isoformat()}) + "\n"],
    )
    (depth_dir / "part_0.parquet").unlink()
    pd.DataFrame({"symbol": ["AAAUSDT"], "received_at": [post_ts - pd.Timedelta(seconds=30)], "bid_px_0": [100.0], "bid_qty_0": [10.0], "ask_px_0": [102.0], "ask_qty_0": [10.0]}).to_parquet(depth_dir / "part_1.parquet")
    report2 = run_fill_calibration(_request(tmp_path, paper_run_dir=run_stale, live_capture_root=live), output_dir=tmp_path / "os")
    assert report2.paper is not None
    assert report2.paper["queue_fallback_count"] >= 1


def test_paper_depth_error_tables_and_sell_books(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal

    parts = _toy_parts()
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    day = date(2023, 1, 2)
    live = tmp_path / "live"
    (live / "exec_depth" / "20230102").mkdir(parents=True)
    bad_depth = pd.DataFrame({"transact_time_ms": [1]})
    bad_depth.to_parquet(live / "exec_depth" / "20230102" / "part_0.parquet")
    post_ts = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=23, minutes=1)
    run_dir = _paper_run(tmp_path / "d1", [json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n", json.dumps({"event": "order_cancelled", "client_order_id": "c1", "ts": (post_ts + pd.Timedelta(minutes=1)).isoformat()}) + "\n"])
    base_cov = {(s, day): _coverage_row(s, day, "ok") for s in ("AAAUSDT", "BBBUSDT")} | {
        (s, date(2023, 1, 3)): _coverage_row(s, date(2023, 1, 3), "ok") for s in ("AAAUSDT", "BBBUSDT")
    }
    tape_frame = _slice_frame(day, [(_day_ms(day, 23.1), 1, 100.0, 1.0, True)])
    _install_tape(monkeypatch, tmp_path / "archive", {("AAAUSDT", "202301"): tape_frame, ("BBBUSDT", "202301"): tape_frame}, base_cov)
    with pytest.raises(DataIntegrityError, match="missing symbol column"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_dir, live_capture_root=live), output_dir=tmp_path / "od1")
    for f in (live / "exec_depth" / "20230102").glob("*.parquet"):
        f.unlink()
    pd.DataFrame({"symbol": ["AAAUSDT"]}).to_parquet(live / "exec_depth" / "20230102" / "part_0.parquet")
    with pytest.raises(DataIntegrityError, match="missing time column"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_dir, live_capture_root=live), output_dir=tmp_path / "od2")
    for f in (live / "exec_depth" / "20230102").glob("*.parquet"):
        f.unlink()
    (live / "exec_depth" / "20230102" / "junk.txt").write_text("x", encoding="utf-8")
    (live / "exec_depth" / "20230102" / "junk.parquet").write_bytes(b"garbage-bytes-not-parquet")
    with pytest.raises(DataIntegrityError, match="unreadable exec_depth"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_dir, live_capture_root=live), output_dir=tmp_path / "od3")
    for f in (live / "exec_depth" / "20230102").glob("*"):
        f.unlink()
    assert fcp.queue_ahead("BUY", 105.0, pd.Series({"bid_px_0": 100.0, "bid_qty_0": 3.0}), 1) == (0.0, None)
    assert fcp.queue_ahead("BUY", 100.0, pd.Series({"bid_px_0": 100.0, "bid_qty_0": 3.0}), 1) == (3.0, 3.0)
    ask_row = pd.Series({"ask_px_0": 100.0, "ask_qty_0": 3.0, "ask_px_1": 101.0, "ask_qty_1": 4.0, "bid_px_0": 99.0, "bid_qty_0": 5.0, "bid_px_1": 98.0, "bid_qty_1": 6.0})
    assert fcp.queue_ahead("SELL", 99.0, ask_row, 2) == (0.0, None)
    assert fcp.queue_ahead("SELL", 100.0, ask_row, 2) == (3.0, 3.0)
    assert fcp.queue_ahead("SELL", 100.5, ask_row, 2) == (0.0, None)
    assert fcp.queue_ahead("SELL", 105.0, ask_row, 2) == (None, None)
    assert fcp.queue_ahead("SELL", 100.0, pd.Series({"ask_px_0": float("nan"), "ask_qty_0": 1.0}), 1) == (None, None)
    assert fcp.queue_ahead("BUY", 50.0, pd.Series({"bid_px_0": 100.0, "bid_qty_0": 3.0, "bid_px_1": 90.0, "bid_qty_1": 4.0}), 2) == (None, None)
    bid_row = pd.Series({"bid_px_0": 100.0, "bid_qty_0": 3.0, "bid_px_1": 90.0, "bid_qty_1": 4.0})
    assert fcp.queue_ahead("BUY", 95.0, bid_row, 2) == (0.0, None)
    assert fcp.queue_ahead("BUY", 100.0, pd.Series({"bid_px_0": float("nan"), "bid_qty_0": 1.0}), 1) == (None, None)


def test_paper_remaining_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal
    from src.market_data.public_archive import store

    parts = _toy_parts()
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    day = date(2023, 1, 2)
    post_ts = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=23, minutes=1)
    base_ms = int(post_ts.value // 10**6)
    tape_frame = _slice_frame(
        day,
        [
            (_day_ms(day, 23.1), 1, 100.0, 50.0, True),
            (base_ms + 30_000, 2, 101.0, 1.0, False),
        ],
    )
    feb_day = date(2023, 2, 6)
    feb_post = pd.Timestamp(feb_day, tz="UTC") + pd.Timedelta(hours=23, minutes=1)
    cov = {(s, day): _coverage_row(s, day, "ok") for s in ("AAAUSDT", "BBBUSDT")} | {
        (s, date(2023, 1, 3)): _coverage_row(s, date(2023, 1, 3), "ok") for s in ("AAAUSDT", "BBBUSDT")
    }
    cov[(("AAAUSDT", feb_day))] = _coverage_row("AAAUSDT", feb_day, "ok")
    _install_tape(monkeypatch, tmp_path / "archive", {("AAAUSDT", "202301"): tape_frame, ("BBBUSDT", "202301"): tape_frame}, cov)
    live = tmp_path / "live"
    live.mkdir()
    jan5 = pd.Timestamp(date(2023, 1, 5), tz="UTC") + pd.Timedelta(hours=23, minutes=1)
    run_cov = _paper_run(
        tmp_path / "cov",
        [
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": jan5.isoformat()}) + "\n",
            json.dumps({"event": "order_cancelled", "client_order_id": "c1", "ts": (jan5 + pd.Timedelta(minutes=1)).isoformat()}) + "\n",
            json.dumps({"event": "order_posted", "client_order_id": "c2", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
            json.dumps({"event": "order_cancelled", "client_order_id": "c2", "ts": (post_ts - pd.Timedelta(minutes=10)).isoformat()}) + "\n",
            json.dumps({"event": "order_posted", "client_order_id": "s1", "symbol": "AAAUSDT", "side": "SELL", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
            json.dumps({"event": "order_cancelled", "client_order_id": "s1", "ts": (post_ts + pd.Timedelta(minutes=5)).isoformat()}) + "\n",
        ],
    )
    report = run_fill_calibration(_request(tmp_path, paper_run_dir=run_cov, live_capture_root=live, evaluation_end=_ts("2023-03-01T00:00:00Z")), output_dir=tmp_path / "ocov")
    assert report.paper is not None
    assert report.exclusions.get("paper_no_tape_coverage", 0) >= 1
    assert report.exclusions.get("paper_post_unresolved", 0) >= 1
    assert report.paper["variants"]["through_only"]["confusion"]["tape_only"] >= 1
    import pathlib as _pl

    real_read = _pl.Path.read_text

    def _boom_read(self, *args, **kwargs):
        if self.name == "a.jsonl":
            raise OSError("disk gone")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(_pl.Path, "read_text", _boom_read)
    with pytest.raises(DataIntegrityError, match="unreadable paper audit"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_cov, live_capture_root=live), output_dir=tmp_path / "oord")
    monkeypatch.setattr(_pl.Path, "read_text", real_read)
    feb_lines = [
        json.dumps({"event": "order_posted", "client_order_id": "f1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": feb_post.isoformat()}) + "\n",
        json.dumps({"event": "order_cancelled", "client_order_id": "f1", "ts": (feb_post + pd.Timedelta(minutes=1)).isoformat()}) + "\n",
    ]
    feb_month = feb_post.strftime("%Y%m")
    real_partition = store.partition_path

    def _key_boom(root, dataset, symbol, month, *, kind="derived"):
        if month == feb_month:
            raise ValueError("bad key")
        return real_partition(root, dataset, symbol, month, kind=kind)

    run_feb = _paper_run(tmp_path / "feb", feb_lines)
    monkeypatch.setattr(store, "partition_path", _key_boom)
    with pytest.raises(DataIntegrityError, match="malformed tape partition key"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_feb, live_capture_root=live, evaluation_end=_ts("2023-03-01T00:00:00Z")), output_dir=tmp_path / "ofeb0")
    monkeypatch.setattr(store, "partition_path", real_partition)
    with pytest.raises(DataIntegrityError, match="tape partition missing"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_feb, live_capture_root=live, evaluation_end=_ts("2023-03-01T00:00:00Z")), output_dir=tmp_path / "ofeb1")
    feb_path = real_partition(tmp_path / "archive", "aggTrades", "AAAUSDT", feb_month, kind="slice")
    feb_path.parent.mkdir(parents=True, exist_ok=True)
    feb_path.write_bytes(b"garbage-bytes-not-parquet")
    with pytest.raises(DataIntegrityError, match="unreadable tape partition"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_feb, live_capture_root=live, evaluation_end=_ts("2023-03-01T00:00:00Z")), output_dir=tmp_path / "ofeb2")
    pd.DataFrame({"transact_time_ms": [1]}).to_parquet(feb_path)
    with pytest.raises(DataIntegrityError, match="missing columns"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_feb, live_capture_root=live, evaluation_end=_ts("2023-03-01T00:00:00Z")), output_dir=tmp_path / "ofeb3")
    bad = pd.DataFrame({"transact_time_ms": [int(feb_post.value // 10**6) + 5_000], "agg_trade_id": [1], "price": [float("nan")], "quantity": [1.0], "is_buyer_maker": [True]})
    bad.to_parquet(feb_path)
    with pytest.raises(DataIntegrityError, match="malformed tape slice"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_feb, live_capture_root=live, evaluation_end=_ts("2023-03-01T00:00:00Z")), output_dir=tmp_path / "ofeb4")
    run_empty = _paper_run(
        tmp_path / "empty",
        [json.dumps({"event": "order_cancelled", "client_order_id": "ghost", "ts": feb_post.isoformat()}) + "\n"],
    )
    empty_report = run_fill_calibration(_request(tmp_path, paper_run_dir=run_empty, live_capture_root=live), output_dir=tmp_path / "oempty")
    assert empty_report.paper is not None
    assert empty_report.paper["queue_fallback_count"] == 0


def test_observed_queue_ignores_stale_or_levelless_snapshots() -> None:
    fresh = pd.DataFrame({"_row_ms": [1_000.0], "bid_px_0": [100.0], "bid_qty_0": [3.0], "ask_px_0": [101.0], "ask_qty_0": [2.0]})
    assert fcp._observed_queue(fresh, "BUY", 100.0, 2_000) == (3.0, 3.0)
    assert fcp._observed_queue(fresh, "BUY", 100.0, 3_001) == (None, None)
    assert fcp._observed_queue(fresh, "BUY", 100.0, 999) == (None, None)
    assert fcp._observed_queue(fresh[["_row_ms", "bid_px_0", "bid_qty_0"]], "BUY", 100.0, 2_000) == (None, None)
    assert fcp._observed_queue(None, "BUY", 100.0, 2_000) == (None, None)
