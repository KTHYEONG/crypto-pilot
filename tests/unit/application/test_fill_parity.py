"""Invariant scenarios for the daily fill parity monitor."""

from __future__ import annotations

import json
import os
import stat
from datetime import date
from pathlib import Path

import pandas as pd

from src.application.fill_parity import (
    FillParityRow,
    FillParityThresholds,
    append_parity_row,
    measure_fill_parity,
    trailing_breach,
)


def _day(year: int = 2023, month: int = 1, day: int = 2) -> date:
    return date(year, month, day)


def _make_run(
    tmp_path: Path,
    day: date,
    *,
    run_id: str = "run1",
    legs: list[dict] | None = None,
) -> Path:
    run_dir = tmp_path / run_id
    eq_dir = run_dir / "execution_quality"
    fills_dir = run_dir / "fills"
    audit_dir = run_dir / "audit"
    eq_dir.mkdir(parents=True, exist_ok=True)
    fills_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({"mode": "paper"}), encoding="utf-8")
    (audit_dir / "a.jsonl").write_text("", encoding="utf-8")
    legs = legs if legs is not None else [
        {"symbol": "BTCUSDT", "leg_index": 0, "ref": 100.0, "avg": 100.0, "qty": 1.0, "fee": 2.0, "side": "BUY"}
    ]
    ts = pd.Timestamp(day, tz="UTC")
    eq_rows = []
    fill_rows = []
    for leg in legs:
        qty = float(leg["qty"])
        side = leg["side"]
        delta = qty if side == "BUY" else -qty
        eq_rows.append({
            "decision_time": ts,
            "symbol": leg["symbol"],
            "mode": "paper",
            "target_weight": 0.01,
            "mark_price_at_decision": float(leg["ref"]),
            "avg_fill_price": float(leg["avg"]),
            "filled_qty": qty,
            "unfilled_qty": 0.0,
            "status": "filled",
            "chases": 0,
            "slippage_bps": 0.0,
            "leg_index": int(leg["leg_index"]),
            "run_id": run_id,
            "latency_seconds": 0.1,
            "sizing_anchor": "book_mid",
            "maker_fill_fraction": 1.0,
            "strategy_digest": None,
            "observed_at": ts,
        })
        fill_rows.append({
            "decision_time": ts,
            "timestamp": ts + pd.Timedelta(minutes=5),
            "symbol": leg["symbol"],
            "quantity_delta": delta,
            "fill_price": float(leg["avg"]),
            "fee_bps": float(leg["fee"]),
            "reason": "maker_fill",
            "pre_trade_equity": 1000.0,
            "liquidity": "maker",
            "mode": "paper",
            "run_id": run_id,
            "leg_index": int(leg["leg_index"]),
            "client_order_id": f"{leg['symbol']}-{leg['leg_index']}",
            "decision_mark": float(leg["ref"]),
            "sizing_anchor": "book_mid",
            "fill_id": f"journal:{leg['symbol']}-{leg['leg_index']}",
        })
    pd.DataFrame(eq_rows).to_parquet(eq_dir / "execution_quality_202301.parquet")
    pd.DataFrame(fill_rows).to_parquet(fills_dir / "fills_202301.parquet")
    return run_dir


def _install_tape(archive_root: Path, day: date, symbol: str, prices: list[float], *, pre_release: tuple[float, float] | None = None) -> None:
    from src.market_data.public_archive import store

    month = day.strftime("%Y%m")
    release = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=23)
    start_ms = int(release.value // 10**6)
    rows = []
    trade_id = 1
    if pre_release is not None:
        offset_s, price = pre_release
        t_ms = int((release - pd.Timedelta(seconds=offset_s)).value // 10**6)
        rows.append((t_ms, trade_id, price, 1.0, True))
        trade_id += 1
    for i, price in enumerate(prices):
        t_ms = start_ms + i * 180_000 + 60_000
        rows.append((t_ms, trade_id, price, 1.0, True))
        trade_id += 1
    frame = pd.DataFrame({
        "transact_time_ms": [r[0] for r in rows],
        "agg_trade_id": [r[1] for r in rows],
        "price": [r[2] for r in rows],
        "quantity": [r[3] for r in rows],
        "is_buyer_maker": [r[4] for r in rows],
    })
    path = store.partition_path(archive_root, "aggTrades", symbol, month, kind="slice")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def test_identical_paper_and_rule_cost_gives_zero_gap(tmp_path: Path) -> None:
    day = _day()
    run_dir = _make_run(tmp_path / "r", day, run_id="run1")
    archive = tmp_path / "archive"
    _install_tape(archive, day, "BTCUSDT", [99.0] * 10)
    row = measure_fill_parity(day, paper_run_dir=run_dir, archive_root=archive, live_capture_root=tmp_path / "live")
    assert row.status == "OK"
    assert row.paper_is_bps == 2.0
    assert row.rule_is_bps == 2.0
    assert row.gap_bps == 0.0


def test_rule_taker_fallback_when_price_never_trades_through(tmp_path: Path) -> None:
    from src.core.params import ACCOUNT_TAKER_FEE_BPS

    day = _day()
    run_dir = _make_run(tmp_path / "r", day, run_id="run1")
    archive = tmp_path / "archive"
    _install_tape(archive, day, "BTCUSDT", [101.0] * 10)
    row = measure_fill_parity(day, paper_run_dir=run_dir, archive_root=archive, live_capture_root=tmp_path / "live")
    assert row.status == "OK"
    assert row.rule_is_bps == (101.0 - 100.0) / 100.0 * 1e4 + ACCOUNT_TAKER_FEE_BPS


def test_release_bar_timing_excludes_pre_release_trades(tmp_path: Path) -> None:
    day = _day()
    run_dir = _make_run(tmp_path / "r", day, run_id="run1")
    archive = tmp_path / "archive"
    _install_tape(archive, day, "BTCUSDT", [101.0] * 10, pre_release=(1.0, 90.0))
    row = measure_fill_parity(day, paper_run_dir=run_dir, archive_root=archive, live_capture_root=tmp_path / "live")
    assert row.status == "OK"
    assert row.rule_is_bps is not None
    assert row.rule_is_bps > 100.0


def test_notional_weighting(tmp_path: Path) -> None:
    day = _day()
    legs = [
        {"symbol": "BTCUSDT", "leg_index": 0, "ref": 100.0, "avg": 100.0, "qty": 0.01, "fee": 2.0, "side": "BUY"},
        {"symbol": "ETHUSDT", "leg_index": 1, "ref": 100.0, "avg": 101.0, "qty": 0.03, "fee": 2.0, "side": "BUY"},
    ]
    run_dir = _make_run(tmp_path / "r", day, run_id="run1", legs=legs)
    archive = tmp_path / "archive"
    _install_tape(archive, day, "BTCUSDT", [99.0] * 10)
    _install_tape(archive, day, "ETHUSDT", [99.0] * 10)
    row = measure_fill_parity(day, paper_run_dir=run_dir, archive_root=archive, live_capture_root=tmp_path / "live")
    assert row.status == "OK"
    assert row.n_intents == 2
    paper1 = 2.0
    paper2 = (101.0 - 100.0) / 100.0 * 1e4 + 2.0
    n1 = 0.01 * 100.0
    n2 = 0.03 * 101.0
    expected = (paper1 * n1 + paper2 * n2) / (n1 + n2)
    assert row.paper_is_bps is not None
    assert abs(row.paper_is_bps - expected) < 1e-9


def test_missing_tape_is_skipped_not_raised(tmp_path: Path) -> None:
    day = _day()
    run_dir = _make_run(tmp_path / "r", day, run_id="run1")
    row = measure_fill_parity(day, paper_run_dir=run_dir, archive_root=tmp_path / "archive", live_capture_root=tmp_path / "live")
    assert row.status == "SKIPPED"
    assert row.detail == "tape_missing"
    assert row.through_agreement is None
    assert row.paper_is_bps is None
    assert row.rule_is_bps is None
    assert row.gap_bps is None


def _ok_row(day: date, run_id: str, n_intents: int, gap: float, agreement: float = 0.97, n_posts: int = 10) -> FillParityRow:
    return FillParityRow(
        day=day, run_id=run_id, n_intents=n_intents, n_posts=n_posts,
        through_agreement=agreement, paper_is_bps=gap + 2.0, rule_is_bps=2.0,
        gap_bps=gap, status="OK", detail=None,
    )


def test_trailing_breach_needs_enough_evidence(tmp_path: Path) -> None:
    path = tmp_path / "fill_parity.jsonl"
    for i in range(7):
        append_parity_row(_ok_row(date(2023, 1, 2 + i), "run1", 3, 5.0), path)
    assert trailing_breach(path, FillParityThresholds(trailing_days=7, min_intents=30)) is None


def test_trailing_gap_breach(tmp_path: Path) -> None:
    path = tmp_path / "fill_parity.jsonl"
    for i in range(7):
        append_parity_row(_ok_row(date(2023, 1, 2 + i), "run1", 30, 2.0, agreement=0.99, n_posts=10), path)
    breach = trailing_breach(path, FillParityThresholds())
    assert breach is not None
    assert "gap" in breach
    path2 = tmp_path / "fill_parity2.jsonl"
    for i in range(7):
        append_parity_row(_ok_row(date(2023, 1, 2 + i), "run1", 30, 1.0, agreement=0.97, n_posts=10), path2)
    assert trailing_breach(path2, FillParityThresholds()) is None


def test_last_line_wins_per_day_run(tmp_path: Path) -> None:
    path = tmp_path / "fill_parity.jsonl"
    append_parity_row(_ok_row(date(2023, 1, 2), "run1", 30, 5.0), path)
    append_parity_row(_ok_row(date(2023, 1, 2), "run1", 30, 0.0), path)
    for i in range(1, 7):
        append_parity_row(_ok_row(date(2023, 1, 2 + i), "run1", 30, 0.0), path)
    assert trailing_breach(path, FillParityThresholds()) is None


def test_report_file_readable_by_backup_user(tmp_path: Path) -> None:
    path = tmp_path / "fill_parity.jsonl"
    append_parity_row(_ok_row(date(2023, 1, 2), "run1", 1, 0.0), path)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o644


def test_partition_without_the_days_window_is_tape_missing_not_ok(tmp_path: Path) -> None:
    day = _day()
    legs = [{"symbol": "BTCUSDT", "leg_index": 0, "ref": 100.0, "avg": 100.0, "qty": 0.01, "fee": 2.0, "side": "BUY"}]
    run_dir = _make_run(tmp_path / "r", day, run_id="run1", legs=legs)
    archive = tmp_path / "archive"
    _install_tape(archive, _day(day=1), "BTCUSDT", [99.0] * 10)
    row = measure_fill_parity(day, paper_run_dir=run_dir, archive_root=archive, live_capture_root=tmp_path / "live")
    assert row.status == "SKIPPED"
    assert row.detail == "tape_missing"
