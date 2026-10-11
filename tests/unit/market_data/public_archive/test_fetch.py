from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from src.common.errors import DataIntegrityError
from src.market_data.binance.vision import BinanceVisionDownloader
from src.market_data.public_archive.catalog import is_perpetual_usdt
from src.market_data.public_archive.fetch import fetch_verified_zip, read_single_csv


def _zip_bytes(members: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, body in members.items():
            zf.writestr(name, body)
    return buffer.getvalue()


def _downloader_with(mapping: dict[str, bytes | None]) -> BinanceVisionDownloader:
    downloader = BinanceVisionDownloader()

    def _fake(*parts: str) -> bytes | None:
        key = "/".join(parts)
        if key not in mapping:
            raise AssertionError(f"unexpected key: {key}")
        return mapping[key]

    downloader.fetch_archive_bytes = _fake  # type: ignore[method-assign]
    return downloader


def test_checksum_mismatch_raises_and_writes_nothing(tmp_path) -> None:
    payload = _zip_bytes({"a.csv": "x,y\n1,2\n"})
    mapping = {
        "daily/metrics/BTCUSDT/f": payload,
        "daily/metrics/BTCUSDT/f.CHECKSUM": b"0" * 64 + b"  f\n",
    }
    downloader = _downloader_with(mapping)
    with pytest.raises(DataIntegrityError):
        fetch_verified_zip(downloader, "data/futures/um/daily/metrics/BTCUSDT/f")
    assert list(tmp_path.iterdir()) == []


def test_missing_zip_returns_none() -> None:
    downloader = _downloader_with({"daily/metrics/BTCUSDT/f": None})
    assert fetch_verified_zip(downloader, "data/futures/um/daily/metrics/BTCUSDT/f") is None


def test_fetch_verified_zip_ok() -> None:
    payload = _zip_bytes({"a.csv": "x,y\n1,2\n"})
    digest = hashlib.sha256(payload).hexdigest()
    mapping = {
        "daily/metrics/BTCUSDT/f": payload,
        "daily/metrics/BTCUSDT/f.CHECKSUM": f"{digest}  f\n".encode(),
    }
    downloader = _downloader_with(mapping)
    body, sha = fetch_verified_zip(downloader, "data/futures/um/daily/metrics/BTCUSDT/f") or (None, None)
    assert body == payload
    assert sha == digest


def test_single_csv_required() -> None:
    payload = _zip_bytes({"a.csv": "x\n1\n", "b.csv": "x\n2\n"})
    with pytest.raises(DataIntegrityError):
        read_single_csv(payload)
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("note.txt", "hi")
    with pytest.raises(DataIntegrityError):
        read_single_csv(empty.getvalue())


def test_perpetual_filter() -> None:
    assert is_perpetual_usdt("BTCUSDT") is True
    assert is_perpetual_usdt("BTCUSDT_250627") is False
    assert is_perpetual_usdt("币安人生USDT") is True
    assert is_perpetual_usdt("ETHBUSD") is False
