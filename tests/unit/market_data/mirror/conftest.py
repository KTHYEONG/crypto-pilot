from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from src.market_data.binance.vision import BucketObject


def md5_hex(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def zip_of(csv_body: bytes, name: str = "part.csv") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, csv_body)
    return buffer.getvalue()


class FakeVision:
    """In-memory stand-in for BinanceVisionDownloader; never touches the network."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.etags: dict[str, str] = {}
        self.listing_fail = False
        self.downloads: list[str] = []

    def put(self, key: str, data: bytes, etag: str | None = None) -> None:
        self.objects[key] = data
        if etag is not None:
            self.etags[key] = etag

    def list_symbols_from_s3_xml_listing(self, *, dataset_prefix: str, timeout: int | None = None) -> list[str]:
        if self.listing_fail:
            return []
        return sorted(
            {key[len(dataset_prefix) :].split("/")[0] for key in self.objects if key.startswith(dataset_prefix)}
        )

    def list_objects(self, *, prefix: str, timeout: int | None = None) -> list[BucketObject] | None:
        if self.listing_fail:
            return None
        return [
            BucketObject(key=key, etag=self.etags.get(key, md5_hex(data)), size=len(data))
            for key, data in sorted(self.objects.items())
            if key.startswith(prefix)
        ]

    def download_object(self, *, key: str, dest: Path, timeout: int | None = None) -> bool:
        if key not in self.objects:
            return False
        self.downloads.append(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.objects[key])
        return True


@pytest.fixture
def fake_vision() -> FakeVision:
    return FakeVision()
