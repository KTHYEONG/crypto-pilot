from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.common.errors import DataIntegrityError
from src.research.provenance.fingerprints import (
    combined_data_hash,
    hash_declared_sources,
    sha256_file,
)


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_file_chunked_over_large_input(tmp_path: Path) -> None:
    payload = b"x" * (3 * (1 << 20)) + b"tail"
    path = tmp_path / "large.parquet"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_hash_declared_sources_returns_one_digest_per_entry_in_order(tmp_path: Path) -> None:
    sources = {
        "first": tmp_path / "a.parquet",
        "second": tmp_path / "b.parquet",
    }
    sources["first"].write_bytes(b"first-payload")
    sources["second"].write_bytes(b"second-payload")

    hashes = hash_declared_sources("BTCUSDT", sources)

    assert list(hashes) == ["first", "second"]
    assert all(len(digest) == 64 for digest in hashes.values())
    assert hashes["first"] != hashes["second"]


def test_hash_declared_sources_fails_closed_on_missing_file(tmp_path: Path) -> None:
    sources = {
        "first": tmp_path / "present.parquet",
        "second": tmp_path / "absent.parquet",
    }
    sources["first"].write_bytes(b"payload")

    with pytest.raises(
        DataIntegrityError,
        match=r"second data missing for BTCUSDT: .*absent\.parquet",
    ):
        hash_declared_sources("BTCUSDT", sources)


def test_combined_data_hash_is_insertion_order_independent() -> None:
    first = {"a": "1" * 64, "b": "2" * 64}
    second = {"b": "2" * 64, "a": "1" * 64}
    assert combined_data_hash(first) == combined_data_hash(second)


def test_combined_data_hash_matches_canonical_serialization() -> None:
    hashes = {"b": "2" * 64, "a": "1" * 64}
    expected = hashlib.sha256(
        __import__("json").dumps(hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert combined_data_hash(hashes) == expected
