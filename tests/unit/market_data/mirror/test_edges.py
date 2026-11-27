from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from src.common.errors import DataIntegrityError
from src.market_data.mirror import build as build_mod
from src.market_data.mirror import confirm as confirm_mod
from src.market_data.mirror.confirm import confirm_uploads, parse_remote_listing, restore_source
from src.market_data.mirror.datasets import MIRROR_DATASET_SPECS
from src.market_data.mirror.index import Sidecar, SourceRecord, load_index, read_sidecar, sidecar_path, write_sidecar
from src.market_data.mirror.planning import list_dataset
from tests.unit.market_data.mirror.conftest import FakeVision, zip_of
from tests.unit.market_data.mirror.test_build import _config, _csv, _key, _run

NOW = datetime(2024, 3, 2, 12, tzinfo=UTC)
DAILY = MIRROR_DATASET_SPECS["bookDepth"]
MONTHLY = MIRROR_DATASET_SPECS["fundingRate"]


def test_leftover_temp_files_removed_and_listing_cache_reused(fake_vision: FakeVision, tmp_path: Path) -> None:
    fake_vision.put(_key("BTCUSDT", "2024-01-01"), zip_of(_csv("2024-01-01")))
    config = _config(tmp_path)
    leftover = config.work_dir / "downloads" / "bookDepth" / "stale.partial"
    leftover.parent.mkdir(parents=True)
    leftover.write_bytes(b"x")

    _run(fake_vision, config)
    assert not leftover.exists()
    second = _run(fake_vision, config)

    assert second.built_sources == 0
    assert fake_vision.downloads == [_key("BTCUSDT", "2024-01-01")]


def test_build_rejects_naive_clock(fake_vision: FakeVision, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        build_mod.build_chunk(
            fake_vision,
            _config(tmp_path),
            remote_bytes=0,
            drive_free_bytes=10**15,
            now=datetime(2024, 3, 2),
            deadline=None,
        )


def test_period_end_monthly_and_daily() -> None:
    assert build_mod._period_end(MONTHLY, "2024-02") == date(2024, 2, 29)
    assert build_mod._period_end(MONTHLY, "2023-12") == date(2023, 12, 31)
    assert build_mod._period_end(DAILY, "2024-01-05") == date(2024, 1, 5)


def test_daily_outbox_cap_stops_before_next_partition(fake_vision: FakeVision, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from src.market_data.mirror import packing

    fake_vision.put(_key("BTCUSDT", "2024-01-01"), zip_of(_csv("2024-01-01")))
    fake_vision.put(_key("ETHUSDT", "2024-01-01"), zip_of(_csv("2024-01-01")))
    monkeypatch.setattr(packing, "GIB_BYTES", 1)
    config = _config(tmp_path, outbox_cap_gib=1, drive_budget_gib=10**6)

    report = _run(fake_vision, config)

    assert report.more is True
    assert report.built_sources == 1


def test_future_deadline_builds_everything(fake_vision: FakeVision, tmp_path: Path) -> None:
    fake_vision.put(_key("BTCUSDT", "2024-01-01"), zip_of(_csv("2024-01-01")))
    fake_vision.put(_key("ETHUSDT", "2024-01-01"), zip_of(_csv("2024-01-01")))
    future = datetime(2999, 1, 1, tzinfo=UTC)

    report = _run(fake_vision, _config(tmp_path), deadline=future)

    assert report.built_sources == 2
    assert report.more is False


def test_list_dataset_monthly_and_invalid_periods(fake_vision: FakeVision) -> None:
    base = "data/futures/um/monthly/fundingRate/"
    fake_vision.put(f"{base}BTCUSDT/BTCUSDT-fundingRate-2024-01.zip", b"a")
    fake_vision.put(f"{base}BTCUSDT/BTCUSDT-fundingRate-2024-13.zip", b"b")
    fake_vision.put(f"{base}BTCUSDT/BTCUSDT-fundingRate-2024-01-01.zip", b"c")

    ok, refs = list_dataset(fake_vision, MONTHLY, known_symbols=set(), workers=1)

    assert ok is True
    assert [(r.symbol, r.period) for r in refs] == [("BTCUSDT", "2024-01")]


def test_list_dataset_symbol_listing_failure_fails_closed(fake_vision: FakeVision, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fake_vision.put(_key("BTCUSDT", "2024-01-01"), b"z")
    fake_vision.put(_key("ETHUSDT", "2024-01-01"), b"z")
    original = fake_vision.list_objects

    def _partial(*, prefix: str, timeout: int | None = None):  # type: ignore[no-untyped-def]
        return None if "ETHUSDT" in prefix else original(prefix=prefix)

    monkeypatch.setattr(fake_vision, "list_objects", _partial)

    assert list_dataset(fake_vision, DAILY, known_symbols=set(), workers=2) == (False, ())


def test_parse_remote_listing_rejects_non_list() -> None:
    with pytest.raises(DataIntegrityError, match="not a list"):
        parse_remote_listing(json.dumps({"Path": "x"}))


def test_parse_remote_listing_accepts_missing_hashes() -> None:
    entries = parse_remote_listing(json.dumps([{"Path": "objects/a", "Size": 3}]))
    assert entries[0].md5 is None


def test_commit_sidecar_identical_and_conflicting(tmp_path: Path) -> None:
    root = tmp_path / "m"
    outbox = root / "outbox" / "index" / "bookDepth" / "2024-01"
    target = root / "index" / "bookDepth" / "2024-01"
    outbox.mkdir(parents=True)
    target.mkdir(parents=True)
    (outbox / "a.json.zst").write_bytes(b"same")
    (target / "a.json.zst").write_bytes(b"same")
    confirm_mod._commit_sidecar(root, outbox / "a.json.zst", Path("bookDepth/2024-01/a.json.zst"))
    assert not (outbox / "a.json.zst").exists()

    (outbox / "b.json.zst").write_bytes(b"new")
    (target / "b.json.zst").write_bytes(b"old")
    with pytest.raises(DataIntegrityError, match="differs"):
        confirm_mod._commit_sidecar(root, outbox / "b.json.zst", Path("bookDepth/2024-01/b.json.zst"))


def test_confirm_size_mismatch_on_sidecar_stays_unconfirmed(fake_vision: FakeVision, tmp_path: Path) -> None:
    from tests.unit.market_data.mirror.test_confirm import _outbox_pairs, _remote_json

    config = _outbox_pairs(fake_vision, tmp_path / "m")
    entries = json.loads(_remote_json(config.root))
    for entry in entries:
        if entry["Path"].startswith("index/"):
            entry["Size"] += 1
    report = confirm_uploads(config.root, parse_remote_listing(json.dumps(entries)), now=NOW)
    assert (report.confirmed, report.unconfirmed) == (0, 2)


def test_restore_rejects_several_etags(tmp_path: Path) -> None:
    payload = b"object-bytes"
    obj = tmp_path / "obj.parquet"
    obj.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    records = tuple(
        SourceRecord(
            key="k.zip",
            etag=etag,
            zip_sha256="0" * 64,
            symbol="BTCUSDT",
            period="2024-01-01",
            source_index=0,
            storage="verbatim",
            layout=None,
        )
        for etag in ("e1", "e2")
    )
    sidecar_file = tmp_path / "s.json.zst"
    write_sidecar(
        sidecar_file,
        Sidecar(
            format_version=1,
            dataset="bookDepth",
            month="2024-01",
            object_path="objects/verbatim/bookDepth/2024-01/x.zip",
            object_bytes=len(payload),
            object_md5="0" * 32,
            object_sha256=sha,
            created_at=NOW.isoformat(),
            sources=records,
        ),
    )
    with pytest.raises(ValueError, match="several ETags"):
        restore_source(sidecar_file, obj, "k.zip", tmp_path / "out.zip")


def test_list_dataset_rejects_impossible_calendar_day(fake_vision: FakeVision) -> None:
    fake_vision.put("data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2024-13-45.zip", b"z")
    assert list_dataset(fake_vision, DAILY, known_symbols=set(), workers=1) == (True, ())


def test_malformed_listing_cache_is_discarded(tmp_path: Path) -> None:
    from src.market_data.mirror.planning import read_listing_cache

    path = tmp_path / "bookDepth.json"
    path.write_text("{broken", encoding="utf-8")
    assert read_listing_cache(path, now=NOW, max_age_hours=12) is None


def test_confirm_on_empty_root_is_noop(tmp_path: Path) -> None:
    (tmp_path / "outbox" / "objects" / "bookDepth" / "2024-01").mkdir(parents=True)
    report = confirm_uploads(tmp_path, [], now=NOW)
    assert (report.confirmed, report.unconfirmed) == (0, 0)


def _crafted_sidecar(tmp_path: Path, payload: bytes, **record_overrides: object) -> tuple[Path, Path]:
    obj = tmp_path / "obj.parquet"
    obj.write_bytes(payload)
    record = SourceRecord(
        key="k.zip",
        etag="e1",
        zip_sha256="0" * 64,
        symbol="BTCUSDT",
        period="2024-01-01",
        source_index=0,
        storage="verbatim",
        layout=None,
    )
    record = dataclasses.replace(record, **record_overrides)  # type: ignore[arg-type]
    sidecar_file = tmp_path / "s.json.zst"
    write_sidecar(
        sidecar_file,
        Sidecar(
            format_version=1,
            dataset="bookDepth",
            month="2024-01",
            object_path="objects/verbatim/bookDepth/2024-01/x.zip",
            object_bytes=len(payload),
            object_md5="0" * 32,
            object_sha256=hashlib.sha256(payload).hexdigest(),
            created_at=NOW.isoformat(),
            sources=(record,),
        ),
    )
    return sidecar_file, obj


def test_restore_verbatim_digest_mismatch_raises(tmp_path: Path) -> None:
    sidecar_file, obj = _crafted_sidecar(tmp_path, b"zip-bytes")
    with pytest.raises(DataIntegrityError, match="verbatim zip sha256"):
        restore_source(sidecar_file, obj, "k.zip", tmp_path / "out.zip")
    assert not (tmp_path / "out.zip").exists()


def test_restore_columnar_csv_mismatch_raises(fake_vision: FakeVision, tmp_path: Path) -> None:
    from tests.unit.market_data.mirror.test_confirm import _outbox_pairs

    config = _outbox_pairs(fake_vision, tmp_path / "m")
    sc = next(s for s in load_index(config.outbox_index_dir, "bookDepth") if s.sources[0].storage == "columnar")
    layout = sc.sources[0].layout
    assert layout is not None
    bad_record = dataclasses.replace(sc.sources[0], layout=dataclasses.replace(layout, csv_sha256="0" * 64))
    sidecar_file = sidecar_path(config.outbox_index_dir, sc.object_path)
    write_sidecar(sidecar_file, dataclasses.replace(sc, sources=(bad_record,)))
    obj = config.outbox_objects_dir / sc.object_path[len("objects/") :]

    with pytest.raises(DataIntegrityError, match="rebuilt csv"):
        restore_source(sidecar_file, obj, bad_record.key, tmp_path / "out.csv")
    assert not (tmp_path / "out.csv").exists()


def test_pre_check_blocks_when_outbox_already_over_budget(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from src.market_data.mirror import packing

    fake = FakeVision()
    fake.put(_key("BTCUSDT", "2024-01-01"), zip_of(_csv("2024-01-01")))
    monkeypatch.setattr(packing, "GIB_BYTES", 10)
    config = _config(tmp_path, drive_budget_gib=1, outbox_cap_gib=10**6)
    staged = config.outbox_objects_dir / "bookDepth" / "2024-01" / "OLD-aaaaaaaaaaaa.parquet"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"x" * 50)
    sidecar_file = sidecar_path(config.outbox_index_dir, "objects/bookDepth/2024-01/OLD-aaaaaaaaaaaa.parquet")
    write_sidecar(
        sidecar_file,
        Sidecar(
            format_version=1,
            dataset="bookDepth",
            month="2024-01",
            object_path="objects/bookDepth/2024-01/OLD-aaaaaaaaaaaa.parquet",
            object_bytes=50,
            object_md5="0" * 32,
            object_sha256="0" * 64,
            created_at=NOW.isoformat(),
            sources=(),
        ),
    )

    report = _run(fake, config)

    assert report.capacity_blocked == "BUDGET"
    assert fake.downloads == []


def test_deadline_reached_during_downloads_keeps_partition_unbuilt(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from src.market_data.mirror import packing

    fake = FakeVision()
    fake.put(_key("BTCUSDT", "2024-01-01"), zip_of(_csv("2024-01-01")))
    fake.put(_key("BTCUSDT", "2024-01-02"), zip_of(_csv("2024-01-02")))
    early = datetime(2024, 3, 2, 11, tzinfo=UTC)
    late = datetime(2024, 3, 2, 13, tzinfo=UTC)
    clock = iter([early, late, late, late])
    monkeypatch.setattr(packing, "utc_now", lambda: next(clock))
    monkeypatch.setattr(build_mod, "utc_now", lambda: early)
    config = _config(tmp_path)

    report = _run(fake, config, deadline=datetime(2024, 3, 2, 12, tzinfo=UTC))

    assert report.built_sources == 0
    assert report.more is True
    assert list(config.outbox_objects_dir.rglob("*.parquet")) == []


def test_confirm_without_outbox_and_skips_orphan_objects(tmp_path: Path) -> None:
    assert confirm_uploads(tmp_path / "absent", [], now=NOW).confirmed == 0
    orphan = tmp_path / "outbox" / "objects" / "bookDepth" / "2024-01" / "X-aaaaaaaaaaaa.parquet"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"orphan")
    report = confirm_uploads(tmp_path, [], now=NOW)
    assert (report.confirmed, report.unconfirmed) == (0, 0)
    assert orphan.exists()


def test_failed_symbol_discovery_never_lists_known_symbols_only(
    fake_vision: FakeVision, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_vision.put(_key("BTCUSDT", "2024-01-01"), b"z")
    monkeypatch.setattr(fake_vision, "list_symbols_from_s3_xml_listing", lambda **_: [])

    assert list_dataset(fake_vision, DAILY, known_symbols={"BTCUSDT"}, workers=1) == (False, ())


@pytest.mark.parametrize("object_path", ["objects/../../outside", "objects//x", "outside/x", "objects", "objects/./x"])
def test_sidecar_object_path_cannot_escape_objects_dir(tmp_path: Path, object_path: str) -> None:
    sidecar_file, _ = _crafted_sidecar(tmp_path, b"zip-bytes")
    good = read_sidecar(sidecar_file)
    write_sidecar(sidecar_file, dataclasses.replace(good, object_path=object_path))

    with pytest.raises(DataIntegrityError, match="object_path"):
        read_sidecar(sidecar_file)
