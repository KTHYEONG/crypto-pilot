from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.common.errors import DataIntegrityError
from src.market_data.mirror import build as build_mod
from src.market_data.mirror.confirm import confirm_uploads, parse_remote_listing, restore_source
from src.market_data.mirror.config import MirrorConfig
from src.market_data.mirror.index import load_index, sidecar_path
from tests.unit.market_data.mirror.conftest import FakeVision, md5_hex, zip_of

NOW = datetime(2024, 3, 2, 12, tzinfo=UTC)
PREFIX = "data/futures/um/daily/bookDepth/"
HEADER = b"timestamp,percentage,depth,notional\n"
GOOD_CSV = HEADER + b"2024-01-01 00:00:00,-5.00,1234.5,5678.9\n2024-01-01 00:00:01,-4.00,1.0,2.0"
CRLF_CSV = HEADER.replace(b"\n", b"\r\n") + b"2024-01-01 00:00:00,-5.00,1.0,2.0\r\n"
BTC_KEY = f"{PREFIX}BTCUSDT/BTCUSDT-bookDepth-2024-01-01.zip"
ETH_KEY = f"{PREFIX}ETHUSDT/ETHUSDT-bookDepth-2024-01-01.zip"


def _config(root: Path) -> MirrorConfig:
    return MirrorConfig(
        root=root,
        datasets=("bookDepth",),
        close_grace_days=0,
        drive_free_floor_gib=1,
        local_free_floor_gib=1,
    )


def _outbox_pairs(fake: FakeVision, root: Path) -> MirrorConfig:
    config = _config(root)
    fake.put(BTC_KEY, zip_of(GOOD_CSV))
    fake.put(ETH_KEY, zip_of(CRLF_CSV))
    build_mod.build_chunk(fake, config, remote_bytes=0, drive_free_bytes=10**15, now=NOW, deadline=None)
    return config


def _remote_json(root: Path, *, drop_objects: bool = False, size_delta: int = 0, null_object_md5: bool = False) -> str:
    entries = []
    for base, prefix in (
        (root / "outbox" / "objects", "objects/"),
        (root / "outbox" / "index", "index/"),
        (root / "index", "index/"),
    ):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = prefix + path.relative_to(base).as_posix()
            if drop_objects and rel.startswith("objects/"):
                continue
            data = path.read_bytes()
            md5 = None if null_object_md5 and rel.startswith("objects/") else md5_hex(data)
            entries.append({"Path": rel, "Size": len(data) + size_delta, "Hashes": {"md5": md5}})
    return json.dumps(entries)


def _pick(config: MirrorConfig, storage: str):  # type: ignore[no-untyped-def]
    return next(sc for sc in load_index(config.outbox_index_dir, "bookDepth") if sc.sources[0].storage == storage)


@pytest.mark.parametrize(
    "broken",
    [{"drop_objects": True}, {"size_delta": 1}, {"null_object_md5": True}],
)
def test_confirm_requires_object_and_sidecar_md5(fake_vision: FakeVision, tmp_path: Path, broken: dict) -> None:  # type: ignore[type-arg]
    config = _outbox_pairs(fake_vision, tmp_path / "m")

    report = confirm_uploads(config.root, parse_remote_listing(_remote_json(config.root, **broken)), now=NOW)

    assert report.confirmed == 0
    assert report.unconfirmed == 2
    assert load_index(config.index_dir, "bookDepth") == ()


def test_confirm_is_idempotent_and_repairs_crash(fake_vision: FakeVision, tmp_path: Path) -> None:
    config = _outbox_pairs(fake_vision, tmp_path / "m")
    columnar = _pick(config, "columnar")
    object_rel = columnar.object_path[len("objects/") :]
    object_bytes = (config.outbox_objects_dir / object_rel).read_bytes()

    first = confirm_uploads(config.root, parse_remote_listing(_remote_json(config.root)), now=NOW)
    assert (first.confirmed, first.unconfirmed) == (2, 0)
    assert not (config.outbox_objects_dir / object_rel).exists()

    again = confirm_uploads(config.root, parse_remote_listing(_remote_json(config.root)), now=NOW)
    assert (again.confirmed, again.unconfirmed) == (0, 0)

    # Crash after the sidecar move but before the object delete leaves the object behind.
    leftover = config.outbox_objects_dir / object_rel
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_bytes(object_bytes)
    cleaned = confirm_uploads(config.root, parse_remote_listing(_remote_json(config.root)), now=NOW)
    assert cleaned.unconfirmed == 0
    assert not leftover.exists()
    assert sidecar_path(config.index_dir, columnar.object_path).exists()


def test_malformed_listing_changes_nothing(fake_vision: FakeVision, tmp_path: Path) -> None:
    config = _outbox_pairs(fake_vision, tmp_path / "m")
    before = sorted(p.as_posix() for p in config.root.rglob("*") if p.is_file())
    with pytest.raises(DataIntegrityError):
        parse_remote_listing("{not json")
    with pytest.raises(DataIntegrityError):
        parse_remote_listing(json.dumps([{"Path": "x"}]))
    assert sorted(p.as_posix() for p in config.root.rglob("*") if p.is_file()) == before


def test_restore_roundtrip_columnar_and_verbatim(fake_vision: FakeVision, tmp_path: Path) -> None:
    config = _outbox_pairs(fake_vision, tmp_path / "m")
    columnar = _pick(config, "columnar")
    verbatim = _pick(config, "verbatim")

    out_csv = tmp_path / "out" / "btc.csv"
    restored = restore_source(
        sidecar_path(config.outbox_index_dir, columnar.object_path),
        config.outbox_objects_dir / columnar.object_path[len("objects/") :],
        columnar.sources[0].key,
        out_csv,
    )
    assert out_csv.read_bytes() == GOOD_CSV
    assert restored.sha256 == columnar.sources[0].layout.csv_sha256  # type: ignore[union-attr]

    out_zip = tmp_path / "out" / "eth.zip"
    restore_source(
        sidecar_path(config.outbox_index_dir, verbatim.object_path),
        config.outbox_objects_dir / verbatim.object_path[len("objects/") :],
        verbatim.sources[0].key,
        out_zip,
    )
    assert out_zip.read_bytes() == zip_of(CRLF_CSV)


def test_restore_mismatch_leaves_no_output(fake_vision: FakeVision, tmp_path: Path) -> None:
    config = _outbox_pairs(fake_vision, tmp_path / "m")
    columnar = _pick(config, "columnar")
    sidecar = sidecar_path(config.outbox_index_dir, columnar.object_path)
    obj = config.outbox_objects_dir / columnar.object_path[len("objects/") :]
    corrupted = bytearray(obj.read_bytes())
    corrupted[-1] ^= 0xFF
    obj.write_bytes(bytes(corrupted))
    out = tmp_path / "out.csv"

    with pytest.raises(DataIntegrityError):
        restore_source(sidecar, obj, columnar.sources[0].key, out)
    assert not out.exists()
    assert not out.with_name(out.name + ".tmp").exists()


def test_restore_rejects_unknown_key(fake_vision: FakeVision, tmp_path: Path) -> None:
    config = _outbox_pairs(fake_vision, tmp_path / "m")
    columnar = _pick(config, "columnar")
    with pytest.raises(ValueError, match="not found"):
        restore_source(
            sidecar_path(config.outbox_index_dir, columnar.object_path),
            config.outbox_objects_dir / columnar.object_path[len("objects/") :],
            "unknown-key",
            tmp_path / "x.csv",
        )
