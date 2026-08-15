"""Data provenance for the open-interest deleveraging research domain.

The source-name-to-path mapping is owned here so the OI-deleveraging screen and
any future workflow fingerprint exactly the same declared bytes.
"""

from __future__ import annotations

from pathlib import Path

from src.common.config import funding_path, metrics_path, ohlcv_path
from src.research.provenance.fingerprints import hash_declared_sources


def source_paths(symbol: str) -> dict[str, Path]:
    """Ordered declared source mapping for one OI-deleveraging symbol."""
    return {
        "perp_ohlcv": ohlcv_path(symbol, "1h"),
        "funding": funding_path(symbol),
        "metrics": metrics_path(symbol),
    }


def oi_deleveraging_data_hashes(symbol: str) -> dict[str, str]:
    """Fingerprint the exact perp/funding/metrics bytes for one symbol."""
    return hash_declared_sources(symbol, source_paths(symbol))
