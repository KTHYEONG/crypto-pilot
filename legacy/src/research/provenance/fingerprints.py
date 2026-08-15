"""Generic byte and aggregate fingerprints shared across research domains.

This module owns only generic mechanisms: bounded-memory SHA-256 over declared
files, fail-closed hashing of a domain-prepared source mapping, and an
insertion-order-independent aggregate digest. Domain provenance modules own the
source-name-to-path mapping and therefore decide what counts as a candidate's
input data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from src.common.errors import DataIntegrityError

_CHUNK_BYTES = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_declared_sources(symbol: str, source_paths: Mapping[str, Path]) -> dict[str, str]:
    """Fingerprint every declared source in order, failing closed on a missing file.

    Existence validation happens here so a partial mapping is never returned: a
    missing declared file raises ``DataIntegrityError`` before any caller sees a
    digest, preserving caller-provided source keys and ordering.
    """
    hashes: dict[str, str] = {}
    for name, path in source_paths.items():
        if not path.exists():
            raise DataIntegrityError(f"{name} data missing for {symbol}: {path}")
        hashes[name] = sha256_file(path)
    return hashes


def combined_data_hash(data_hashes: Mapping[str, str]) -> str:
    """SHA-256 over the canonical sorted JSON serialization of the hash mapping."""
    return hashlib.sha256(
        json.dumps(dict(data_hashes), sort_keys=True).encode("utf-8")
    ).hexdigest()
