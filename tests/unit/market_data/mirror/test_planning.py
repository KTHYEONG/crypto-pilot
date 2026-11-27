from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.market_data.mirror.config import MirrorConfig
from src.market_data.mirror.datasets import MIRROR_DATASET_SPECS
from src.market_data.mirror.planning import (
    SourceRef,
    list_dataset,
    plan_dataset,
    read_listing_cache,
    write_listing_cache,
)
from tests.unit.market_data.mirror.conftest import FakeVision

DAILY = MIRROR_DATASET_SPECS["bookDepth"]
MONTHLY = MIRROR_DATASET_SPECS["fundingRate"]
LISTED_AT = "2024-03-02T00:00:00+00:00"


def _ref(symbol: str, period: str, etag: str = "e1", cadence_daily: bool = True) -> SourceRef:
    if cadence_daily:
        key = f"data/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{period}.zip"
    else:
        key = f"data/futures/um/monthly/fundingRate/{symbol}-fundingRate-{period}.zip"
    return SourceRef(key=key, etag=etag, size=10, symbol=symbol, period=period)


def _plan(dataset, refs, mirrored=None, today=date(2024, 3, 2), grace=0, listing_ok=True):  # type: ignore[no-untyped-def]
    return plan_dataset(
        dataset,
        refs,
        listing_ok=listing_ok,
        mirrored=mirrored or {},
        today=today,
        close_grace_days=grace,
        listed_at=LISTED_AT,
    )


def test_pending_is_listing_minus_index() -> None:
    a = _ref("BTCUSDT", "2024-01-01")
    b = _ref("BTCUSDT", "2024-01-02")
    revised = _ref("BTCUSDT", "2024-01-03", etag="new")
    mirrored = {a.key: frozenset({a.etag}), revised.key: frozenset({"old"})}

    plan = _plan(DAILY, [a, b, revised], mirrored=mirrored)

    pending = [s.key for p in plan.partitions for s in p.sources]
    assert pending == [b.key, revised.key]


def test_outbox_pairs_are_excluded_by_caller_mapping() -> None:
    a = _ref("BTCUSDT", "2024-01-01")
    plan = _plan(DAILY, [a], mirrored={a.key: frozenset({a.etag})})
    assert plan.partitions == ()


def test_open_month_excluded_daily_only() -> None:
    feb_last = _ref("BTCUSDT", "2024-02-29")
    mar_1 = _ref("BTCUSDT", "2024-03-01")
    monthly = _ref("BTCUSDT", "2024-02", cadence_daily=False)

    closed_edge = _plan(DAILY, [feb_last], today=date(2024, 3, 6), grace=5)
    open_month = _plan(DAILY, [feb_last], today=date(2024, 3, 6), grace=6)
    assert _plan(DAILY, [feb_last], today=date(2024, 3, 5), grace=5).partitions == ()
    monthly_plan = _plan(MONTHLY, [monthly], today=date(2024, 2, 2), grace=5)
    assert len(closed_edge.partitions) == 1
    assert closed_edge.partitions[0].month == "2024-02"
    assert open_month.partitions == ()
    assert _plan(DAILY, [mar_1], today=date(2024, 3, 6), grace=5).partitions == ()
    assert len(monthly_plan.partitions) == 1


def test_listing_failure_yields_nothing() -> None:
    a = _ref("BTCUSDT", "2024-01-01")
    plan = _plan(DAILY, [a], mirrored={"gone.zip": frozenset({"x"})}, listing_ok=False)
    assert plan.listing_ok is False
    assert plan.partitions == ()
    assert plan.vanished_keys == ()


def test_removed_symbol_reported_vanished() -> None:
    kept = _ref("BTCUSDT", "2024-01-01")
    plan = _plan(DAILY, [kept], mirrored={kept.key: frozenset({kept.etag}), "removed.zip": frozenset({"x"})})
    assert plan.vanished_keys == ("removed.zip",)


def test_oldest_first_and_pack_grouping() -> None:
    refs = [_ref("ETHUSDT", "2024-01-02"), _ref("BTCUSDT", "2024-01-01"), _ref("BTCUSDT", "2024-02-01")]
    plan = _plan(DAILY, refs)
    assert [(p.month, p.partition) for p in plan.partitions] == [
        ("2024-01", "BTCUSDT"),
        ("2024-01", "ETHUSDT"),
        ("2024-02", "BTCUSDT"),
    ]
    metrics = MIRROR_DATASET_SPECS["metrics"]
    grouped = _plan(metrics, [_ref("BTCUSDT", "2024-01-01"), _ref("ETHUSDT", "2024-01-01")])
    assert [(p.month, p.partition) for p in grouped.partitions] == [("2024-01", "ALL")]
    assert plan.latest_period == "2024-02-01"


def test_list_dataset_ignores_unparseable_and_uses_known_symbols(fake_vision: FakeVision) -> None:
    base = "data/futures/um/daily/bookDepth/"
    fake_vision.put(f"{base}BTCUSDT/BTCUSDT-bookDepth-2024-01-01.zip", b"z")
    fake_vision.put(f"{base}BTCUSDT/README.txt", b"x")
    fake_vision.put(f"{base}BTCUSDT/BTCUSDT-bookDepth-notadate.zip", b"y")
    fake_vision.put(f"{base}DEADUSDT/DEADUSDT-bookDepth-2023-12-31.zip", b"w")
    fake_vision.put(f"{base}BTC_PERP/BTC_PERP-bookDepth-2024-01-01.zip", b"q")

    ok, refs = list_dataset(fake_vision, DAILY, known_symbols={"DEADUSDT"}, workers=2)

    assert ok is True
    assert sorted((r.symbol, r.period) for r in refs) == [("BTCUSDT", "2024-01-01"), ("DEADUSDT", "2023-12-31")]


def test_list_dataset_empty_symbols_fails_closed(fake_vision: FakeVision) -> None:
    assert list_dataset(fake_vision, DAILY, known_symbols=set(), workers=2) == (False, ())


def test_listing_cache_respects_max_age(tmp_path: Path) -> None:
    path = tmp_path / "plan" / "bookDepth.json"
    now = datetime(2024, 3, 2, 12, tzinfo=UTC)
    refs = (_ref("BTCUSDT", "2024-01-01"),)
    write_listing_cache(path, now.isoformat(), refs)

    fresh = read_listing_cache(path, now=now, max_age_hours=12)
    stale = read_listing_cache(path, now=datetime(2024, 3, 3, 12, tzinfo=UTC), max_age_hours=12)

    assert fresh is not None
    assert fresh[1] == refs
    assert stale is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"datasets": ("nope",)}, "unknown"),
        ({"datasets": ("bookDepth", "bookDepth")}, "duplicate"),
        ({"download_workers": 9}, "download_workers"),
        ({"verbatim_max_fraction": 1.5}, "verbatim_max_fraction"),
        ({"outbox_cap_gib": 0}, "outbox_cap_gib"),
        ({"close_grace_days": -1}, "close_grace_days"),
    ],
)
def test_config_rejects_invalid_values(tmp_path: Path, overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MirrorConfig(root=tmp_path, **overrides)  # type: ignore[arg-type]


def test_config_orders_datasets_canonically(tmp_path: Path) -> None:
    config = MirrorConfig(root=tmp_path, datasets=("bookDepth", "metrics"))
    assert config.datasets == ("metrics", "bookDepth")
