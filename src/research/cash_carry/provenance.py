"""Data provenance for the cash-and-carry research domain.

The source-name-to-path mapping is owned here so the cash-and-carry evaluation
and any future workflow fingerprint exactly the same declared bytes.
"""

from __future__ import annotations

from pathlib import Path

from src.common.config import borrow_path, funding_path, ohlcv_path, spot_ohlcv_path
from src.research.provenance.fingerprints import hash_declared_sources


def source_paths(symbol: str) -> dict[str, Path]:
    """Ordered declared source mapping for one cash-and-carry symbol."""
    return {
        "spot_ohlcv": spot_ohlcv_path(symbol, "1h"),
        "perp_ohlcv": ohlcv_path(symbol, "1h"),
        "funding": funding_path(symbol),
        "borrow": borrow_path(symbol),
    }


def cash_carry_data_hashes(symbol: str) -> dict[str, str]:
    """Fingerprint the exact spot/perp/funding/borrow bytes for one symbol."""
    return hash_declared_sources(symbol, source_paths(symbol))
