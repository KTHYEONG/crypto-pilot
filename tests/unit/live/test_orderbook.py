"""OrderBook capture tests for paper_execution_fidelity."""

from __future__ import annotations

import math
from decimal import Decimal

import pandas as pd


def test_fetch_order_book_parses_levels_and_last_update_id(tmp_path) -> None:
    from src.live.orderbook import fetch_order_book

    class StubClient:
        def depth(self, symbol, *, limit=20):
            return {
                "lastUpdateId": 42,
                "bids": [["100.0", "2.0"], ["99.5", "5.0"]],
                "asks": [["100.5", "1.0"], ["101.0", "3.0"]],
            }

    ts = pd.Timestamp("2026-09-01 00:00:00", tz="UTC")
    decision_time = pd.Timestamp("2026-09-01 00:00:00", tz="UTC")
    snap = fetch_order_book(
        StubClient(), "BTCUSDT", decision_time, mode="paper", capture_seq=0, limit=20, now=ts
    )
    assert snap.bids[0] == (Decimal("100.0"), Decimal("2.0"))
    assert snap.bids[1] == (Decimal("99.5"), Decimal("5.0"))
    assert snap.asks[0] == (Decimal("100.5"), Decimal("1.0"))
    assert snap.last_update_id == 42
    assert snap.captured_at == ts
    assert snap.captured_at.tzinfo is not None
    assert snap.capture_seq == 0
    assert snap.mode == "paper"


def test_capture_order_books_is_bounded_by_duration_and_interval(tmp_path) -> None:
    from src.live.orderbook import capture_order_books

    class StubClient:
        def depth(self, symbol, *, limit=20):
            return {"lastUpdateId": 1, "bids": [["100", "1"]], "asks": [["101", "1"]]}

    clock_time = [0.0]
    sleeps: list[float] = []

    def clock():
        return clock_time[0]

    def sleep_fn(s):
        sleeps.append(s)
        clock_time[0] += s

    decision_time = pd.Timestamp("2026-09-01 00:00:00", tz="UTC")
    snapshots = capture_order_books(
        StubClient(),
        ["BTCUSDT", "ETHUSDT"],
        decision_time,
        mode="paper",
        duration_s=30,
        interval_s=10,
        depth_limit=20,
        max_symbols=40,
        clock=clock,
        sleep_fn=sleep_fn,
        now_fn=lambda: pd.Timestamp("2026-09-01 00:00:00", tz="UTC"),
    )
    assert len(snapshots) == 6
    assert {s.capture_seq for s in snapshots} == {0, 1, 2}
    assert all(v == 10.0 for v in sleeps)
    # total ticks <= ceil(30/10)+1 =4
    assert len({s.capture_seq for s in snapshots}) <= math.ceil(30 / 10) + 1


def test_capture_order_books_failsoft_skips_failing_symbol(tmp_path) -> None:
    from src.live.orderbook import capture_order_books

    class StubClient:
        def depth(self, symbol, *, limit=20):
            if symbol == "ETHUSDT":
                raise RuntimeError("fail")
            return {"lastUpdateId": 1, "bids": [["100", "1"]], "asks": [["101", "1"]]}

    clock_val = [0.0]

    def clock():
        return clock_val[0]

    snapshots = capture_order_books(
        StubClient(),
        ["BTCUSDT", "ETHUSDT"],
        pd.Timestamp("2026-09-01 00:00:00", tz="UTC"),
        mode="paper",
        duration_s=0,
        interval_s=10,
        depth_limit=20,
        max_symbols=40,
        clock=clock,
        sleep_fn=lambda s: None,
        now_fn=lambda: pd.Timestamp("2026-09-01 00:00:00", tz="UTC"),
    )
    # With duration 0, at least one tick? Actually duration 0 -> single tick before break?
    # Our loop should capture at start then break before next tick if duration 0.
    # So expect only BTCUSDT captured.
    assert all(s.symbol == "BTCUSDT" for s in snapshots)
    assert not any(s.symbol == "ETHUSDT" for s in snapshots)


def test_capture_order_books_stops_on_shutdown(tmp_path) -> None:
    from src.live.lifecycle import ShutdownFlag
    from src.live.orderbook import capture_order_books

    class StubClient:
        def depth(self, symbol, *, limit=20):
            return {"lastUpdateId": 1, "bids": [["100", "1"]], "asks": [["101", "1"]]}

    # requested=True from start
    flag = ShutdownFlag()
    flag.requested = True  # type: ignore[attr-defined]
    # Some implementations use attribute requested; ensure it's True
    import contextlib as _ctx

    with _ctx.suppress(Exception):
        flag.requested = True
    # Use object with requested attr
    class Flag:
        requested = True

    snapshots = capture_order_books(
        StubClient(),
        ["BTCUSDT", "ETHUSDT"],
        pd.Timestamp("2026-09-01 00:00:00", tz="UTC"),
        mode="paper",
        duration_s=30,
        interval_s=10,
        depth_limit=20,
        max_symbols=40,
        clock=lambda: 0.0,
        sleep_fn=lambda s: None,
        now_fn=lambda: pd.Timestamp("2026-09-01 00:00:00", tz="UTC"),
        shutdown=Flag(),
    )
    assert snapshots == []

    # flip after tick 0 via side-effecting sleep
    class Flag2:
        requested = False

    flag2 = Flag2()
    clock_calls = [0.0]

    def clock2():
        return clock_calls[0]

    def sleep_side(s):
        # advance clock and set flag
        clock_calls[0] += 10.0
        flag2.requested = True

    # need clock that increments to simulate ticks; but we use flag to stop after first tick
    # Use a clock that returns 0 on first tick, 10 after sleep etc.
    # Simplify: capture loop checks shutdown each tick start. After first tick sleep sets flag.
    # second tick should break before capture.
    snapshots2 = capture_order_books(
        StubClient(),
        ["BTCUSDT", "ETHUSDT"],
        pd.Timestamp("2026-09-01 00:00:00", tz="UTC"),
        mode="paper",
        duration_s=30,
        interval_s=10,
        depth_limit=20,
        max_symbols=40,
        clock=lambda: clock_calls[0],
        sleep_fn=sleep_side,
        now_fn=lambda: pd.Timestamp("2026-09-01 00:00:00", tz="UTC"),
        shutdown=flag2,
    )
    assert len(snapshots2) == 2  # only tick 0 captured (2 symbols)
    assert all(s.capture_seq == 0 for s in snapshots2)


def test_capture_order_books_truncates_to_max_symbols(tmp_path) -> None:
    from src.live.orderbook import capture_order_books

    class StubClient:
        def depth(self, symbol, *, limit=20):
            return {"lastUpdateId": 1, "bids": [["100", "1"]], "asks": [["101", "1"]]}

    symbols = [f"S{i}USDT" for i in range(50)]
    snapshots = capture_order_books(
        StubClient(),
        symbols,
        pd.Timestamp("2026-09-01 00:00:00", tz="UTC"),
        mode="paper",
        duration_s=0,
        interval_s=10,
        depth_limit=20,
        max_symbols=40,
        clock=lambda: 0.0,
        sleep_fn=lambda s: None,
        now_fn=lambda: pd.Timestamp("2026-09-01 00:00:00", tz="UTC"),
    )
    distinct = {s.symbol for s in snapshots}
    assert distinct == set(symbols[:40])
    assert "S40USDT" not in distinct
    assert "S49USDT" not in distinct


def test_append_order_book_snapshots_writes_daily_zstd_and_dedupes(tmp_path) -> None:
    from decimal import Decimal

    from src.live.orderbook import OrderBookSnapshot, append_order_book_snapshots

    ts = pd.Timestamp("2026-09-01 12:00:00", tz="UTC")
    dt = pd.Timestamp("2026-09-01 00:00:00", tz="UTC")

    def snap(last_id):
        return OrderBookSnapshot(
            symbol="BTCUSDT",
            captured_at=ts,
            decision_time=dt,
            mode="paper",
            capture_seq=0,
            bids=((Decimal("100"), Decimal("1")),),
            asks=((Decimal("101"), Decimal("1")),),
            last_update_id=last_id,
        )

    s1 = snap(1)
    s2 = snap(1)  # duplicate
    s3 = snap(2)  # distinct
    snapshots = [s1, s2, s3]
    # first append
    paths = append_order_book_snapshots(snapshots, tmp_path)
    assert len(paths) == 1
    # second append same (dedup should keep 2 rows)
    paths2 = append_order_book_snapshots(snapshots, tmp_path)
    assert len(paths2) == 1
    files = list(tmp_path.glob("live_orderbook_*.parquet"))
    assert len(files) == 1
    assert files[0].name == "live_orderbook_20260901.parquet"

    df = pd.read_parquet(files[0])
    assert len(df) == 2
    for col in ["bid_px_00", "bid_qty_00", "ask_px_19", "ask_qty_19", "depth_levels", "mid", "spread_bps", "mode", "decision_time", "capture_seq", "last_update_id"]:
        assert col in df.columns, f"missing {col}"
    # dtypes
    assert df["bid_qty_00"].dtype == "float32"
    assert df["bid_px_00"].dtype == "float64"


def test_load_order_book_snapshots_roundtrip(tmp_path) -> None:
    from decimal import Decimal

    from src.live.orderbook import OrderBookSnapshot, append_order_book_snapshots, load_order_book_snapshots

    ts = pd.Timestamp("2026-09-01 12:00:00", tz="UTC")
    dt = pd.Timestamp("2026-09-01 00:00:00", tz="UTC")
    snap = OrderBookSnapshot(
        symbol="BTCUSDT",
        captured_at=ts,
        decision_time=dt,
        mode="paper",
        capture_seq=0,
        bids=((Decimal("100"), Decimal("1")),),
        asks=((Decimal("101"), Decimal("1")),),
        last_update_id=1,
    )
    append_order_book_snapshots([snap], tmp_path)
    df = load_order_book_snapshots(tmp_path)
    assert len(df) == 1
    # captured_at tz-aware
    assert pd.api.types.is_datetime64_any_dtype(df["captured_at"])
    # check tz-aware via dtype
    assert str(df["captured_at"].dtype) == "datetime64[ns, UTC]"
    # since after all rows -> empty
    df2 = load_order_book_snapshots(tmp_path, since=pd.Timestamp("2026-09-02 00:00:00", tz="UTC"))
    assert len(df2) == 0
    # missing dir -> empty
    df3 = load_order_book_snapshots(tmp_path / "missing")
    assert len(df3) == 0
