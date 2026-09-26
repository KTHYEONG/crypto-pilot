"""Invariant scenarios for PIT venue listing snapshots (spec 05)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.live.venue_listing import (
    VenueListingSnapshot,
    delisting_blocked_decisions,
    latest_venue_listing,
    latest_venue_listing_or_none,
    load_venue_listing_history,
    parse_venue_listing,
    settlement_evidence_from_bars,
    write_venue_listing_snapshot,
)

CAPTURE = pd.Timestamp("2026-09-10T00:00:00Z")
HORIZON = pd.Timedelta(days=365)


def _ms(stamp: pd.Timestamp) -> int:
    return int(stamp.value // 1_000_000)


def _row(
    symbol: str,
    *,
    status: str = "TRADING",
    contract: str = "PERPETUAL",
    delivery: pd.Timestamp | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "symbol": symbol,
        "status": status,
        "contractType": contract,
        "underlyingType": "COIN",
        "quoteAsset": "USDT",
    }
    if delivery is not None:
        row["deliveryDate"] = _ms(delivery)
    return row


def test_announced_perpetual_detected() -> None:
    payload = {
        "symbols": [
            _row("AAAUSDT", delivery=CAPTURE + pd.Timedelta(days=10)),
            _row("BBBUSDT", delivery=CAPTURE + pd.Timedelta(days=3650)),
        ]
    }
    snapshot = parse_venue_listing(payload, captured_at=CAPTURE, previous=None, announcement_horizon=HORIZON)
    assert snapshot.entries["AAAUSDT"].announced_delisting is True
    assert snapshot.entries["AAAUSDT"].delisting_first_seen_at == CAPTURE
    assert snapshot.entries["BBBUSDT"].announced_delisting is False
    assert snapshot.entries["BBBUSDT"].delisting_first_seen_at is None


def test_quarterly_contracts_never_announced() -> None:
    payload = {"symbols": [_row("BTCUSDT_260925", contract="CURRENT_QUARTER", delivery=CAPTURE + pd.Timedelta(days=10))]}
    snapshot = parse_venue_listing(payload, captured_at=CAPTURE, previous=None, announcement_horizon=HORIZON)
    assert snapshot.entries["BTCUSDT_260925"].announced_delisting is False


def test_first_seen_carried_forward() -> None:
    day1 = pd.Timestamp("2026-09-01T00:00:00Z")
    day3 = pd.Timestamp("2026-09-03T00:00:00Z")
    delivery = day1 + pd.Timedelta(days=10)
    first = parse_venue_listing(
        {"symbols": [_row("XUSDT", delivery=delivery)]}, captured_at=day1, previous=None, announcement_horizon=HORIZON
    )
    third = parse_venue_listing(
        {"symbols": [_row("XUSDT", delivery=delivery)]}, captured_at=day3, previous=first, announcement_horizon=HORIZON
    )
    assert third.entries["XUSDT"].delisting_first_seen_at == day1


def test_malformed_row_isolated() -> None:
    payload = {"symbols": [_row("OKUSDT", delivery=CAPTURE + pd.Timedelta(days=3650)), {"symbol": "BADUSDT"}]}
    snapshot = parse_venue_listing(payload, captured_at=CAPTURE, previous=None, announcement_horizon=HORIZON)
    assert "OKUSDT" in snapshot.entries
    assert "BADUSDT" not in snapshot.entries


def test_empty_listing_fails_closed() -> None:
    with pytest.raises(DataIntegrityError, match="symbols"):
        parse_venue_listing({}, captured_at=CAPTURE, previous=None, announcement_horizon=HORIZON)
    with pytest.raises(DataIntegrityError, match="symbols"):
        parse_venue_listing({"symbols": "nope"}, captured_at=CAPTURE, previous=None, announcement_horizon=HORIZON)
    with pytest.raises(DataIntegrityError, match=r"naive|timezone"):
        parse_venue_listing(
            {"symbols": [_row("AUSDT")]},
            captured_at=pd.Timestamp("2026-09-10"),
            previous=None,
            announcement_horizon=HORIZON,
        )


def _announced_snapshot(day: pd.Timestamp, delivery: pd.Timestamp) -> VenueListingSnapshot:
    return parse_venue_listing(
        {"symbols": [_row("XUSDT", delivery=delivery), _row("YUSDT", delivery=day + pd.Timedelta(days=3650))]},
        captured_at=day,
        previous=None,
        announcement_horizon=HORIZON,
    )


def test_blocks_are_point_in_time(tmp_path: Path) -> None:
    day5 = pd.Timestamp("2026-09-05T00:00:00Z")
    delivery = pd.Timestamp("2026-09-09T09:00:00Z")
    snapshot = _announced_snapshot(day5, delivery)
    write_venue_listing_snapshot(snapshot, tmp_path, slot_day=day5)
    history = load_venue_listing_history(tmp_path, through_day=pd.Timestamp("2026-09-08T00:00:00Z"))
    index = pd.date_range("2026-09-01", "2026-09-08", freq="D", tz="UTC")
    frame = delisting_blocked_decisions(
        history,
        index,
        ["XUSDT", "YUSDT"],
        holding_end_offset=pd.Timedelta(hours=24),
        lead=pd.Timedelta(hours=48),
    )
    assert list(frame.columns) == ["XUSDT", "YUSDT"]
    assert frame.dtypes.eq(bool).all()
    assert frame.loc[index[:4], "XUSDT"].tolist() == [False] * 4
    assert frame["XUSDT"].tolist() == [False] * 6 + [True] * 2
    assert frame["YUSDT"].tolist() == [False] * 8


def test_withdrawn_announcement_unblocks_only_forward(tmp_path: Path) -> None:
    delivery = pd.Timestamp("2026-09-08T09:00:00Z")
    for day in ("2026-09-05", "2026-09-06"):
        slot = pd.Timestamp(f"{day}T00:00:00Z")
        write_venue_listing_snapshot(_announced_snapshot(slot, delivery), tmp_path, slot_day=slot)
    slot7 = pd.Timestamp("2026-09-07T00:00:00Z")
    write_venue_listing_snapshot(
        parse_venue_listing(
            {"symbols": [_row("XUSDT", delivery=slot7 + pd.Timedelta(days=3650))]},
            captured_at=slot7,
            previous=None,
            announcement_horizon=HORIZON,
        ),
        tmp_path,
        slot_day=slot7,
    )
    history = load_venue_listing_history(tmp_path, through_day=pd.Timestamp("2026-09-08T00:00:00Z"))
    index = pd.date_range("2026-09-05", "2026-09-08", freq="D", tz="UTC")
    frame = delisting_blocked_decisions(
        history, index, ["XUSDT"], holding_end_offset=pd.Timedelta(hours=24), lead=pd.Timedelta(hours=48)
    )
    assert frame["XUSDT"].tolist() == [False, True, False, False]


def _write_hourly(path: Path, delivery: pd.Timestamp, *, flat_price: float, flat_bars: int, traded_after: bool) -> None:
    stamps: list[int] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []
    base = delivery - pd.Timedelta(hours=6)
    for hour in range(5):
        ts = base + pd.Timedelta(hours=hour)
        stamps.append(_ms(ts))
        opens.append(1.0)
        highs.append(1.1)
        lows.append(0.9)
        closes.append(1.0)
        volumes.append(10.0)
    for hour in range(flat_bars):
        ts = delivery + pd.Timedelta(hours=hour)
        stamps.append(_ms(ts))
        opens.append(flat_price)
        highs.append(flat_price)
        lows.append(flat_price)
        closes.append(flat_price)
        volumes.append(0.0)
    if traded_after:
        ts = delivery + pd.Timedelta(hours=flat_bars)
        stamps.append(_ms(ts))
        opens.append(flat_price)
        highs.append(flat_price + 0.01)
        lows.append(flat_price)
        closes.append(flat_price)
        volumes.append(5.0)
    pd.DataFrame(
        {"timestamp": stamps, "open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}
    ).to_parquet(path, index=False)


def test_settlement_evidence_from_flat_bars(tmp_path: Path) -> None:
    delivery = pd.Timestamp("2026-09-09T09:00:00Z")
    path = tmp_path / "XUSDT.parquet"
    _write_hourly(path, delivery, flat_price=0.214, flat_bars=5, traded_after=False)
    evidence = settlement_evidence_from_bars(
        path, symbol="XUSDT", delivery_time=delivery, min_flat_bars=3, price_rtol=1e-9
    )
    assert evidence is not None
    assert evidence.price == Decimal("0.214")
    assert evidence.flat_bars == 5
    assert evidence.source == "flat_1h_klines"


def test_insufficient_or_inconsistent_flat_run(tmp_path: Path) -> None:
    delivery = pd.Timestamp("2026-09-09T09:00:00Z")
    few = tmp_path / "FEWUSDT.parquet"
    _write_hourly(few, delivery, flat_price=0.214, flat_bars=2, traded_after=False)
    assert (
        settlement_evidence_from_bars(few, symbol="FEWUSDT", delivery_time=delivery, min_flat_bars=3, price_rtol=1e-9)
        is None
    )
    noisy = tmp_path / "NOISYUSDT.parquet"
    _write_hourly(noisy, delivery, flat_price=0.214, flat_bars=3, traded_after=False)
    frame = pd.read_parquet(noisy)
    frame.loc[frame.index[-1], "close"] = 0.3
    frame.to_parquet(noisy, index=False)
    assert (
        settlement_evidence_from_bars(noisy, symbol="NOISYUSDT", delivery_time=delivery, min_flat_bars=3, price_rtol=1e-9)
        is None
    )
    traded = tmp_path / "TRADEDUSDT.parquet"
    _write_hourly(traded, delivery, flat_price=0.214, flat_bars=5, traded_after=True)
    assert (
        settlement_evidence_from_bars(
            traded, symbol="TRADEDUSDT", delivery_time=delivery, min_flat_bars=3, price_rtol=1e-9
        )
        is None
    )


def test_stale_listing_rejected(tmp_path: Path) -> None:
    captured = pd.Timestamp("2026-09-10T00:00:00Z")
    write_venue_listing_snapshot(_announced_snapshot(captured, captured + pd.Timedelta(days=10)), tmp_path, slot_day=captured)
    with pytest.raises(DataIntegrityError, match="stale"):
        latest_venue_listing(
            tmp_path, now=captured + pd.Timedelta(hours=40), max_age=pd.Timedelta(hours=30)
        )
    assert latest_venue_listing(tmp_path, now=captured + pd.Timedelta(hours=29), max_age=pd.Timedelta(hours=30)) is not None


def test_snapshot_round_trip_and_recapture_guard(tmp_path: Path) -> None:
    assert latest_venue_listing_or_none(tmp_path) is None
    assert load_venue_listing_history(tmp_path, through_day=CAPTURE) == ()
    snapshot = _announced_snapshot(CAPTURE, CAPTURE + pd.Timedelta(days=10))
    path = write_venue_listing_snapshot(snapshot, tmp_path, slot_day=CAPTURE)
    assert path.name == "20260910.json.gz"
    assert latest_venue_listing_or_none(tmp_path) is not None
    assert load_venue_listing_history(tmp_path, through_day=CAPTURE - pd.Timedelta(days=1)) == ()
    moved = parse_venue_listing(
        {"symbols": [_row("XUSDT", delivery=CAPTURE + pd.Timedelta(days=10))]},
        captured_at=CAPTURE + pd.Timedelta(hours=1),
        previous=None,
        announcement_horizon=HORIZON,
    )
    with pytest.raises(DataIntegrityError, match="first-seen"):
        write_venue_listing_snapshot(moved, tmp_path, slot_day=CAPTURE)
    with pytest.raises(DataIntegrityError, match=r"naive|timezone"):
        write_venue_listing_snapshot(snapshot, tmp_path, slot_day=pd.Timestamp("2026-09-10"))


def test_settlement_bars_missing_columns_fail_closed(tmp_path: Path) -> None:
    delivery = pd.Timestamp("2026-09-09T09:00:00Z")
    path = tmp_path / "BADUSDT.parquet"
    pd.DataFrame({"timestamp": [_ms(delivery)], "close": [1.0]}).to_parquet(path, index=False)
    with pytest.raises(DataIntegrityError, match="columns"):
        settlement_evidence_from_bars(path, symbol="BADUSDT", delivery_time=delivery, min_flat_bars=3, price_rtol=1e-9)
    with pytest.raises(DataIntegrityError, match=r"naive|timezone"):
        settlement_evidence_from_bars(
            path, symbol="BADUSDT", delivery_time=pd.Timestamp("2026-09-09"), min_flat_bars=3, price_rtol=1e-9
        )


def test_naive_delivery_and_malformed_slot_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "20260910.json.gz").write_bytes(b"not a snapshot")
    with pytest.raises(DataIntegrityError, match=r"unreadable|malformed"):
        load_venue_listing_history(tmp_path, through_day=CAPTURE)


def test_assorted_malformed_rows_skipped() -> None:
    delivery = CAPTURE + pd.Timedelta(days=10)
    payload = {
        "symbols": [
            "junk",
            {"symbol": "", "status": "TRADING", "contractType": "PERPETUAL"},
            {"symbol": "NOSTATUSUSDT", "contractType": "PERPETUAL"},
            {"symbol": "NOCONTRACTUSDT", "status": "TRADING"},
            {"symbol": "BOOLUSDT", "status": "TRADING", "contractType": "PERPETUAL", "deliveryDate": True},
            {"symbol": "STRUSDT", "status": "TRADING", "contractType": "PERPETUAL", "deliveryDate": str(_ms(delivery))},
            {"symbol": "EMPTYSTRUSDT", "status": "TRADING", "contractType": "PERPETUAL", "deliveryDate": "   "},
            {"symbol": "FLOATUSDT", "status": "TRADING", "contractType": "PERPETUAL", "deliveryDate": float(_ms(delivery))},
            {"symbol": "BADFLOATUSDT", "status": "TRADING", "contractType": "PERPETUAL", "deliveryDate": 1.5},
            {"symbol": "MAPUSDT", "status": "TRADING", "contractType": "PERPETUAL", "deliveryDate": {"ms": 1}},
            _row("OKUSDT", delivery=CAPTURE + pd.Timedelta(days=3650)),
        ]
    }
    snapshot = parse_venue_listing(payload, captured_at=CAPTURE, previous=None, announcement_horizon=HORIZON)
    assert snapshot.entries["STRUSDT"].announced_delisting is True
    assert snapshot.entries["FLOATUSDT"].announced_delisting is True
    assert snapshot.entries["OKUSDT"].announced_delisting is False
    assert snapshot.entries["EMPTYSTRUSDT"].announced_delisting is False
    assert snapshot.entries["EMPTYSTRUSDT"].delivery_time is None
    for absent in ("BOOLUSDT", "BADFLOATUSDT", "MAPUSDT", "NOSTATUSUSDT", "NOCONTRACTUSDT"):
        assert absent not in snapshot.entries
    with pytest.raises(DataIntegrityError, match="no listing entry"):
        parse_venue_listing({"symbols": ["junk", {"symbol": "BADUSDT"}]}, captured_at=CAPTURE, previous=None, announcement_horizon=HORIZON)


def _write_raw_snapshot(root: Path, name: str, obj: Any) -> None:
    import gzip
    import json

    with gzip.open(root / name, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(obj))


def test_corrupt_snapshot_files_fail_closed(tmp_path: Path) -> None:
    good = {
        "captured_at": CAPTURE.isoformat(),
        "entries": [
            {
                "symbol": "XUSDT",
                "status": "TRADING",
                "contract_type": "PERPETUAL",
                "underlying_type": "COIN",
                "quote_asset": "USDT",
                "delivery_time": (CAPTURE + pd.Timedelta(days=10)).isoformat(),
                "announced_delisting": True,
                "delisting_first_seen_at": CAPTURE.isoformat(),
            }
        ],
    }
    cases = {
        "20260901.json.gz": ["not", "a", "mapping"],
        "20260902.json.gz": {"entries": good["entries"]},
        "20260903.json.gz": {"captured_at": "not-a-time", "entries": good["entries"]},
        "20260904.json.gz": {"captured_at": CAPTURE.isoformat(), "entries": []},
        "20260905.json.gz": {"captured_at": CAPTURE.isoformat(), "entries": ["nope"]},
        "20260906.json.gz": {
            "captured_at": CAPTURE.isoformat(),
            "entries": [{"symbol": "", "status": "TRADING", "contract_type": "PERPETUAL", "announced_delisting": False}],
        },
        "20260910.json.gz": {
            "captured_at": CAPTURE.isoformat(),
            "entries": [{"symbol": "XUSDT", "status": "", "contract_type": "PERPETUAL", "announced_delisting": False}],
        },
        "20260911.json.gz": {
            "captured_at": CAPTURE.isoformat(),
            "entries": [{"symbol": "XUSDT", "status": "TRADING", "contract_type": "", "announced_delisting": False}],
        },
        "20260907.json.gz": {
            "captured_at": CAPTURE.isoformat(),
            "entries": [{"symbol": "XUSDT", "status": "TRADING", "contract_type": "PERPETUAL", "announced_delisting": "yes"}],
        },
        "20260908.json.gz": {
            "captured_at": CAPTURE.isoformat(),
            "entries": [{"symbol": "XUSDT", "status": "TRADING", "contract_type": "PERPETUAL"}],
        },
        "20260909.json.gz": {
            "captured_at": CAPTURE.isoformat(),
            "entries": [
                {
                    "symbol": "XUSDT",
                    "status": "TRADING",
                    "contract_type": "PERPETUAL",
                    "delivery_time": "2026-09-10",
                    "announced_delisting": True,
                    "delisting_first_seen_at": CAPTURE.isoformat(),
                }
            ],
        },
    }
    for name, obj in cases.items():
        _write_raw_snapshot(tmp_path, name, obj)
        with pytest.raises(DataIntegrityError, match="malformed"):
            load_venue_listing_history(tmp_path, through_day=pd.Timestamp(f"{name[:4]}-{name[4:6]}-{name[6:8]}T00:00:00Z"))
        (tmp_path / name).unlink()
    _write_raw_snapshot(tmp_path, "20260910.json.gz", good)
    assert len(load_venue_listing_history(tmp_path, through_day=CAPTURE)) == 1
    with pytest.raises(DataIntegrityError, match="no venue listing snapshot"):
        latest_venue_listing(tmp_path / "empty", now=CAPTURE, max_age=pd.Timedelta(hours=30))


def test_recapture_without_announcement_keeps_slot(tmp_path: Path) -> None:
    delivery = CAPTURE + pd.Timedelta(days=10)
    first = parse_venue_listing(
        {"symbols": [_row("XUSDT", delivery=delivery), _row("YUSDT", delivery=CAPTURE + pd.Timedelta(days=3650))]},
        captured_at=CAPTURE,
        previous=None,
        announcement_horizon=HORIZON,
    )
    write_venue_listing_snapshot(first, tmp_path, slot_day=CAPTURE)
    cleared = parse_venue_listing(
        {"symbols": [_row("XUSDT", delivery=CAPTURE + pd.Timedelta(days=3650)), _row("YUSDT")]},
        captured_at=CAPTURE + pd.Timedelta(hours=1),
        previous=first,
        announcement_horizon=HORIZON,
    )
    assert cleared.entries["XUSDT"].announced_delisting is False
    assert cleared.entries["XUSDT"].delisting_first_seen_at is None
    write_venue_listing_snapshot(cleared, tmp_path, slot_day=CAPTURE)
    reloaded = latest_venue_listing_or_none(tmp_path)
    assert reloaded is not None
    assert reloaded.entries["XUSDT"].announced_delisting is False


def test_blocked_decisions_ignores_unknown_and_undated_symbols() -> None:
    from src.live.venue_listing import VenueListingEntry, VenueListingSnapshot

    delivery = pd.Timestamp("2026-09-09T09:00:00Z")
    announced = VenueListingEntry(
        symbol="XUSDT",
        status="TRADING",
        contract_type="PERPETUAL",
        underlying_type="COIN",
        quote_asset="USDT",
        delivery_time=delivery,
        announced_delisting=True,
        delisting_first_seen_at=pd.Timestamp("2026-09-07T00:00:00Z"),
    )
    undated = VenueListingEntry(
        symbol="WUSDT",
        status="TRADING",
        contract_type="PERPETUAL",
        underlying_type="COIN",
        quote_asset="USDT",
        delivery_time=None,
        announced_delisting=True,
        delisting_first_seen_at=None,
    )
    history = (VenueListingSnapshot(captured_at=pd.Timestamp("2026-09-05T00:00:00Z"), entries={"XUSDT": announced, "WUSDT": undated}),)
    index = pd.date_range("2026-09-05", "2026-09-08", freq="D", tz="UTC")
    frame = delisting_blocked_decisions(
        history, index, ["XUSDT", "WUSDT", "GHOSTUSDT"], holding_end_offset=pd.Timedelta(hours=24), lead=pd.Timedelta(hours=48)
    )
    assert frame["XUSDT"].tolist() == [False, False, True, True]
    assert frame["WUSDT"].tolist() == [False] * 4
    assert frame["GHOSTUSDT"].tolist() == [False] * 4
    naive_index = pd.DatetimeIndex([stamp.tz_localize(None) for stamp in index])
    naive_frame = delisting_blocked_decisions(
        history, naive_index, ["XUSDT"], holding_end_offset=pd.Timedelta(hours=24), lead=pd.Timedelta(hours=48)
    )
    assert naive_frame["XUSDT"].tolist() == frame["XUSDT"].tolist()
    naive_seen = VenueListingSnapshot(
        captured_at=pd.Timestamp("2026-09-05T00:00:00Z"),
        entries={
            "XUSDT": VenueListingEntry(
                symbol="XUSDT",
                status="TRADING",
                contract_type="PERPETUAL",
                underlying_type="COIN",
                quote_asset="USDT",
                delivery_time=delivery,
                announced_delisting=True,
                delisting_first_seen_at=pd.Timestamp("2026-09-05T00:00:00"),
            )
        },
    )
    naive_seen_frame = delisting_blocked_decisions(
        (naive_seen,), index, ["XUSDT"], holding_end_offset=pd.Timedelta(hours=24), lead=pd.Timedelta(hours=48)
    )
    assert naive_seen_frame["XUSDT"].tolist() == [False, False, True, True]


def test_settlement_evidence_edge_cases(tmp_path: Path) -> None:
    delivery = pd.Timestamp("2026-09-09T09:00:00Z")
    with pytest.raises(DataIntegrityError, match="min_flat_bars"):
        settlement_evidence_from_bars(
            tmp_path / "missing.parquet", symbol="XUSDT", delivery_time=delivery, min_flat_bars=0, price_rtol=1e-9
        )
    with pytest.raises(DataIntegrityError, match="unreadable"):
        settlement_evidence_from_bars(
            tmp_path / "missing.parquet", symbol="XUSDT", delivery_time=delivery, min_flat_bars=3, price_rtol=1e-9
        )
    empty = tmp_path / "EMPTYUSDT.parquet"
    pd.DataFrame(
        {"timestamp": pd.Series([], dtype="int64"), "open": [], "high": [], "low": [], "close": [], "volume": []}
    ).to_parquet(empty, index=False)
    assert (
        settlement_evidence_from_bars(empty, symbol="EMPTYUSDT", delivery_time=delivery, min_flat_bars=3, price_rtol=1e-9)
        is None
    )
    all_nan = tmp_path / "NANUSDT.parquet"
    pd.DataFrame(
        {
            "timestamp": [_ms(delivery + pd.Timedelta(hours=h)) for h in range(4)],
            "open": [float("nan")] * 4,
            "high": [float("nan")] * 4,
            "low": [float("nan")] * 4,
            "close": [float("nan")] * 4,
            "volume": [float("nan")] * 4,
        }
    ).to_parquet(all_nan, index=False)
    assert (
        settlement_evidence_from_bars(all_nan, symbol="NANUSDT", delivery_time=delivery, min_flat_bars=3, price_rtol=1e-9)
        is None
    )


def test_settlement_evidence_rejects_gaps_and_accepts_zero_price(tmp_path: Path) -> None:
    delivery = pd.Timestamp("2026-09-09T09:00:00Z")
    gapped = tmp_path / "GAPUSDT.parquet"
    stamps = [_ms(delivery + pd.Timedelta(hours=h)) for h in (0, 1, 3, 4)]
    pd.DataFrame(
        {"timestamp": stamps, "open": [0.5] * 4, "high": [0.5] * 4, "low": [0.5] * 4, "close": [0.5] * 4, "volume": [0.0] * 4}
    ).to_parquet(gapped, index=False)
    assert (
        settlement_evidence_from_bars(gapped, symbol="GAPUSDT", delivery_time=delivery, min_flat_bars=3, price_rtol=1e-9)
        is None
    )
    zero = tmp_path / "ZEROUSDT.parquet"
    stamps = [_ms(delivery + pd.Timedelta(hours=h)) for h in range(3)]
    pd.DataFrame(
        {"timestamp": stamps, "open": [0.0] * 3, "high": [0.0] * 3, "low": [0.0] * 3, "close": [0.0] * 3, "volume": [0.0] * 3}
    ).to_parquet(zero, index=False)
    evidence = settlement_evidence_from_bars(
        zero, symbol="ZEROUSDT", delivery_time=delivery, min_flat_bars=3, price_rtol=1e-9
    )
    assert evidence is not None
    assert evidence.price == Decimal("0")
    nonzero = tmp_path / "NONZEROUSDT.parquet"
    pd.DataFrame(
        {
            "timestamp": stamps,
            "open": [0.0, 0.0, 0.1],
            "high": [0.0, 0.0, 0.1],
            "low": [0.0, 0.0, 0.1],
            "close": [0.0, 0.0, 0.1],
            "volume": [0.0, 0.0, 0.0],
        }
    ).to_parquet(nonzero, index=False)
    assert (
        settlement_evidence_from_bars(
            nonzero, symbol="NONZEROUSDT", delivery_time=delivery, min_flat_bars=3, price_rtol=1e-9
        )
        is None
    )


def test_settlement_evidence_rejects_close_disagreement(tmp_path: Path) -> None:
    delivery = pd.Timestamp("2026-09-09T09:00:00Z")
    path = tmp_path / "DRIFTUSDT.parquet"
    stamps = [_ms(delivery + pd.Timedelta(hours=h)) for h in range(3)]
    pd.DataFrame(
        {
            "timestamp": stamps,
            "open": [0.5, 0.5, 0.6],
            "high": [0.5, 0.5, 0.6],
            "low": [0.5, 0.5, 0.6],
            "close": [0.5, 0.5, 0.6],
            "volume": [0.0, 0.0, 0.0],
        }
    ).to_parquet(path, index=False)
    assert (
        settlement_evidence_from_bars(path, symbol="DRIFTUSDT", delivery_time=delivery, min_flat_bars=3, price_rtol=1e-9)
        is None
    )


def test_retry_after_midnight_applies_by_slot_day_not_capture_time(tmp_path: Path) -> None:
    """00:00 UTC 이후 재시도로 D+1에 재캡처된 slot D도 결정 D부터 적용되어야 한다."""
    from src.live.venue_listing import (
        VenueListingEntry,
        VenueListingSnapshot,
        delisting_blocked_decisions,
        load_venue_listing_history,
        write_venue_listing_snapshot,
    )

    delivery = pd.Timestamp("2026-09-08T09:00:00Z")
    entry = VenueListingEntry(
        symbol="XUSDT",
        status="TRADING",
        contract_type="PERPETUAL",
        underlying_type="COIN",
        quote_asset="USDT",
        delivery_time=delivery,
        announced_delisting=True,
        delisting_first_seen_at=pd.Timestamp("2026-09-06T23:00:00Z"),
    )
    slot = pd.Timestamp("2026-09-06T00:00:00Z")
    retry_capture = VenueListingSnapshot(captured_at=pd.Timestamp("2026-09-07T00:30:00Z"), entries={"XUSDT": entry})
    write_venue_listing_snapshot(retry_capture, tmp_path, slot_day=slot)
    history = load_venue_listing_history(tmp_path, through_day=pd.Timestamp("2026-09-06T00:00:00Z"))
    assert len(history) == 1
    assert history[0].slot_day == slot
    index = pd.DatetimeIndex([slot])
    frame = delisting_blocked_decisions(
        history, index, ["XUSDT"], holding_end_offset=pd.Timedelta(hours=24), lead=pd.Timedelta(hours=48)
    )
    assert frame["XUSDT"].tolist() == [True]
