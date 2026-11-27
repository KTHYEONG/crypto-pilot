from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.market_data.mirror import build as build_mod
from src.market_data.mirror import packing
from src.market_data.mirror.config import MirrorConfig
from src.market_data.mirror.index import load_index
from tests.unit.market_data.mirror.conftest import FakeVision, zip_of

NOW = datetime(2024, 3, 2, 12, 0, tzinfo=UTC)
ROOT_PREFIX = "data/futures/um/daily/bookDepth/"
HEADER = b"timestamp,percentage,depth,notional\n"


def _csv(day: str, rows: int = 2) -> bytes:
    body = b"".join(f"{day} 00:00:0{i},-5.00,1234.5,5678.9\n".encode() for i in range(rows))
    return HEADER + body.rstrip(b"\n")


def _key(symbol: str, day: str) -> str:
    return f"{ROOT_PREFIX}{symbol}/{symbol}-bookDepth-{day}.zip"


def _config(tmp_path: Path, **overrides: object) -> MirrorConfig:
    params: dict[str, object] = {
        "root": tmp_path / "mirror",
        "datasets": ("bookDepth",),
        "close_grace_days": 0,
        "download_workers": 2,
        "drive_free_floor_gib": 1,
        "local_free_floor_gib": 1,
    }
    params.update(overrides)
    return MirrorConfig(**params)  # type: ignore[arg-type]


def _run(fake: FakeVision, config: MirrorConfig, *, deadline: datetime | None = None):  # type: ignore[no-untyped-def]
    return build_mod.build_chunk(
        fake,
        config,
        remote_bytes=0,
        drive_free_bytes=10**15,
        now=NOW,
        deadline=deadline,
    )


def _drop_listing_cache(config: MirrorConfig) -> None:
    for cache in (config.status_dir / "plan").glob("*.json"):
        cache.unlink()


def _outbox_sidecars(config: MirrorConfig) -> list[Path]:
    return sorted(config.outbox_index_dir.rglob("*.json.zst"))


def test_publishes_object_then_sidecar(fake_vision: FakeVision, tmp_path: Path) -> None:
    fake_vision.put(_key("BTCUSDT", "2024-01-01"), zip_of(_csv("2024-01-01")))
    fake_vision.put(_key("BTCUSDT", "2024-01-02"), zip_of(_csv("2024-01-02")))
    fake_vision.put(_key("ETHUSDT", "2024-01-01"), zip_of(_csv("2024-01-01")))
    config = _config(tmp_path)

    report = _run(fake_vision, config)

    assert report.built_sources == 3
    assert report.failed_sources == 0
    assert report.pending_sources == 0
    assert len(_outbox_sidecars(config)) == 2
    assert not list((config.work_dir / "downloads").rglob("*.zip"))
    sources = [r for sc in load_index(config.outbox_index_dir, "bookDepth") for r in sc.sources]
    assert sorted(r.key for r in sources) == sorted(fake_vision.objects)


def test_etag_md5_mismatch_not_published(fake_vision: FakeVision, tmp_path: Path) -> None:
    key = _key("BTCUSDT", "2024-01-01")
    fake_vision.put(key, zip_of(_csv("2024-01-01")), etag="0" * 32)
    config = _config(tmp_path)

    report = _run(fake_vision, config)

    assert report.failed_sources == 1
    assert report.built_sources == 0
    assert _outbox_sidecars(config) == []


def test_multipart_etag_uses_checksum(fake_vision: FakeVision, tmp_path: Path) -> None:
    import hashlib

    key = _key("BTCUSDT", "2024-01-01")
    data = zip_of(_csv("2024-01-01"))
    fake_vision.put(key, data, etag="abc123-2")
    fake_vision.put(f"{key}.CHECKSUM", f"{hashlib.sha256(data).hexdigest()}  x.zip\n".encode())

    report = _run(fake_vision, _config(tmp_path))

    assert report.built_sources == 1
    assert report.failed_sources == 0


def test_unencodable_goes_verbatim_unaltered(fake_vision: FakeVision, tmp_path: Path) -> None:
    key = _key("BTCUSDT", "2024-01-01")
    crlf = zip_of(HEADER.replace(b"\n", b"\r\n") + b"2024-01-01 00:00:00,-5.00,1.0,2.0\r\n")
    fake_vision.put(key, crlf)
    config = _config(tmp_path)

    report = _run(fake_vision, config)

    assert report.verbatim_sources == 1
    assert report.built_sources == 0
    objects = list((config.outbox_objects_dir / "verbatim").rglob("*.zip"))
    assert len(objects) == 1
    assert objects[0].read_bytes() == crlf


def test_budget_blocks_publish(fake_vision: FakeVision, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_vision.put(_key("BTCUSDT", "2024-01-01"), zip_of(_csv("2024-01-01")))
    monkeypatch.setattr(packing, "GIB_BYTES", 10)
    config = _config(tmp_path, drive_budget_gib=1)

    report = _run(fake_vision, config)

    assert report.capacity_blocked == "BUDGET"
    assert report.built_sources == 0
    assert report.pending_sources == 1
    assert _outbox_sidecars(config) == []


def test_outbox_cap_sets_more(fake_vision: FakeVision, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_vision.put(_key("BTCUSDT", "2024-01-01"), zip_of(_csv("2024-01-01")))
    fake_vision.put(_key("ETHUSDT", "2024-01-01"), zip_of(_csv("2024-01-01")))
    monkeypatch.setattr(packing, "GIB_BYTES", 1)
    config = _config(tmp_path, outbox_cap_gib=1, drive_budget_gib=10**6)

    report = _run(fake_vision, config)

    assert report.more is True
    assert report.built_objects == 1


def test_orphan_object_recovered(fake_vision: FakeVision, tmp_path: Path) -> None:
    config = _config(tmp_path)
    orphan = config.outbox_objects_dir / "bookDepth" / "2024-01" / "BTCUSDT-deadbeef0000.parquet"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"partial")

    _run(fake_vision, config)

    assert not orphan.exists()


def test_format_drift_holds_verbatim_and_stops_dataset(fake_vision: FakeVision, tmp_path: Path) -> None:
    crlf = HEADER.replace(b"\n", b"\r\n") + b"2024-01-01 00:00:00,-5.00,1.0,2.0\r\n"
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        fake_vision.put(_key(symbol, "2024-01-01"), zip_of(crlf))
    config = _config(tmp_path, verbatim_min_sources=2, verbatim_max_fraction=0.5)

    report = _run(fake_vision, config)

    assert "format_drift:bookDepth" in report.anomalies
    assert report.verbatim_sources == 1
    assert len(_outbox_sidecars(config)) == 1
    assert report.pending_sources == 2


def test_vanished_reported_once_and_nothing_deleted(fake_vision: FakeVision, tmp_path: Path) -> None:
    btc = _key("BTCUSDT", "2024-01-01")
    fake_vision.put(btc, zip_of(_csv("2024-01-01")))
    fake_vision.put(_key("ETHUSDT", "2024-01-01"), zip_of(_csv("2024-01-01")))
    config = _config(tmp_path)
    _run(fake_vision, config)
    fake_vision.objects.pop(btc)
    _drop_listing_cache(config)

    first = _run(fake_vision, config)
    _drop_listing_cache(config)
    second = _run(fake_vision, config)

    assert "vanished_upstream:bookDepth:1" in first.anomalies
    assert not any(a.startswith("vanished_upstream") for a in second.anomalies)
    assert len(_outbox_sidecars(config)) == 2


def test_listing_failure_reported(fake_vision: FakeVision, tmp_path: Path) -> None:
    fake_vision.listing_fail = True
    report = _run(fake_vision, _config(tmp_path))

    assert report.listing_failed == ("bookDepth",)
    assert report.built_sources == 0


def test_deadline_stops_before_download(fake_vision: FakeVision, tmp_path: Path) -> None:
    fake_vision.put(_key("BTCUSDT", "2024-01-01"), zip_of(_csv("2024-01-01")))
    past = datetime(2020, 1, 1, tzinfo=UTC)

    report = _run(fake_vision, _config(tmp_path), deadline=past)

    assert fake_vision.downloads == []
    assert report.more is True


def test_rejects_negative_inputs(fake_vision: FakeVision, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_mod.build_chunk(
            fake_vision, _config(tmp_path), remote_bytes=-1, drive_free_bytes=0, now=NOW, deadline=None
        )


def test_format_drift_publishes_nothing_for_a_changed_partition(fake_vision: FakeVision, tmp_path: Path) -> None:
    changed = b"timestamp,percentage,depth,notional,new_column\n2024-01-01 00:00:00,-5.00,1.0,2.0,3\n"
    for day in ("2024-01-01", "2024-01-02", "2024-01-03"):
        fake_vision.put(_key("BTCUSDT", day), zip_of(changed))
    config = _config(tmp_path, verbatim_min_sources=2)

    report = _run(fake_vision, config)
    again = _run(fake_vision, config)

    assert "format_drift:bookDepth" in report.anomalies
    assert "format_drift:bookDepth" in again.anomalies
    assert (report.built_objects, again.built_objects) == (0, 0)
    assert _outbox_sidecars(config) == []
    assert len(fake_vision.downloads) == 3, "held downloads are reused, not fetched again"


def test_isolated_unencodable_source_is_not_drift(fake_vision: FakeVision, tmp_path: Path) -> None:
    fake_vision.put(_key("BTCUSDT", "2024-01-01"), zip_of(_csv("2024-01-01")))
    fake_vision.put(_key("BTCUSDT", "2024-01-02"), zip_of(b"broken,header\n1,2\n"))
    config = _config(tmp_path)

    report = _run(fake_vision, config)

    assert not any(a.startswith("format_drift") for a in report.anomalies)
    assert (report.built_sources, report.verbatim_sources) == (1, 1)


def test_deadline_stops_new_downloads_inside_a_large_partition(
    fake_vision: FakeVision, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for number in range(1, 21):
        day = f"2024-01-{number:02d}"
        fake_vision.put(_key("BTCUSDT", day), zip_of(_csv(day)))
    deadline = datetime(2024, 3, 2, 12, 30, tzinfo=UTC)

    def _clock() -> datetime:
        late = len(fake_vision.downloads) >= 4
        return datetime(2024, 3, 2, 13 if late else 12, tzinfo=UTC)

    monkeypatch.setattr(packing, "utc_now", _clock)
    monkeypatch.setattr(build_mod, "utc_now", lambda: datetime(2024, 3, 2, 12, tzinfo=UTC))
    config = _config(tmp_path)

    report = _run(fake_vision, config, deadline=deadline)

    assert report.more is True
    assert report.built_objects == 0
    assert 4 <= len(fake_vision.downloads) <= 4 + config.download_workers


def test_local_floor_blocks_before_downloading(
    fake_vision: FakeVision, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_vision.put(_key("BTCUSDT", "2024-01-01"), zip_of(_csv("2024-01-01")))
    monkeypatch.setattr(packing, "local_free_bytes", lambda _: (1 << 30) + 10)

    report = _run(fake_vision, _config(tmp_path))

    assert report.capacity_blocked == "LOCAL_FREE_FLOOR"
    assert fake_vision.downloads == []


def test_rezipped_identical_csv_gets_a_distinct_sidecar_path(fake_vision: FakeVision, tmp_path: Path) -> None:
    key = _key("BTCUSDT", "2024-01-01")
    fake_vision.put(key, zip_of(_csv("2024-01-01"), name="a.csv"))
    config = _config(tmp_path)
    _run(fake_vision, config)
    first = load_index(config.outbox_index_dir, "bookDepth")[0]
    committed = config.index_dir / "bookDepth" / "2024-01"
    committed.mkdir(parents=True)
    for sidecar_file in _outbox_sidecars(config):
        sidecar_file.rename(committed / sidecar_file.name)
    for obj in config.outbox_objects_dir.rglob("*.parquet"):
        obj.unlink()

    fake_vision.put(key, zip_of(_csv("2024-01-01"), name="renamed.csv"))
    _drop_listing_cache(config)
    report = _run(fake_vision, config)

    second = load_index(config.outbox_index_dir, "bookDepth")[0]
    assert report.built_sources == 1
    assert second.object_sha256 == first.object_sha256, "same CSV encodes to the same bytes"
    assert second.object_path != first.object_path
