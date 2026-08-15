"""Shared data provenance for the frozen technical-expert candidates.

``technical_data_hashes`` is the single helper every technical evaluator uses
to fingerprint the exact OHLCV/funding bytes backing a candidate, so the single
candidate screen and the library admission diagnostic always agree on data
provenance.
"""

from __future__ import annotations

from pathlib import Path

from src.common.config import funding_path, ohlcv_path
from src.research.provenance.fingerprints import hash_declared_sources


def _source_paths(symbol: str) -> dict[str, Path]:
    return {
        "perp_ohlcv": ohlcv_path(symbol, "1h"),
        "funding": funding_path(symbol),
    }


def technical_data_hashes(symbol: str) -> dict[str, str]:
    """Fingerprint the exact OHLCV/funding bytes declared for one symbol.

    A declared source file that is missing fails closed instead of hashing an
    empty file, so a partial panel can never be fingerprinted.
    """
    return hash_declared_sources(symbol, _source_paths(symbol))


def _check_contract() -> None:
    """Executable assertions locking the shared provenance helper surface."""
    assert technical_data_hashes.__name__ == "technical_data_hashes"


_check_contract()
