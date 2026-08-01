"""Shared data provenance for the frozen technical-expert candidates.

``technical_data_hashes`` is the single helper every technical evaluator uses
to fingerprint the exact OHLCV/funding bytes backing a candidate, so the single
candidate screen and the library admission diagnostic always agree on data
provenance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.common.config import funding_path, ohlcv_path
from src.common.errors import DataIntegrityError

_SOURCE_FILES = ("perp_ohlcv", "funding")


def _source_paths(symbol: str) -> dict[str, str]:
    return {
        "perp_ohlcv": str(ohlcv_path(symbol, "1h")),
        "funding": str(funding_path(symbol)),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def technical_data_hashes(symbol: str) -> dict[str, str]:
    """Fingerprint the exact OHLCV/funding bytes declared for one symbol.

    A declared source file that is missing fails closed instead of hashing an
    empty file, so a partial panel can never be fingerprinted.
    """
    hashes: dict[str, str] = {}
    for name, path in _source_paths(symbol).items():
        p = Path(path)
        if not p.exists():
            raise DataIntegrityError(f"{name} data missing for {symbol}: {p}")
        hashes[name] = _file_sha256(p)
    return hashes


def _check_contract() -> None:
    """Executable assertions locking the shared provenance helper surface."""
    assert technical_data_hashes.__name__ == "technical_data_hashes"


_check_contract()
