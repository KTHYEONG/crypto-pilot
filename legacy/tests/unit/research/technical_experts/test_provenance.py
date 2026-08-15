from __future__ import annotations

from pathlib import Path

import pytest

from src.common.errors import DataIntegrityError
from src.research.technical_experts import provenance


def _patch_paths(monkeypatch, ohlcv: Path, funding: Path) -> None:
    monkeypatch.setattr(provenance, "ohlcv_path", lambda symbol, timeframe: ohlcv)
    monkeypatch.setattr(provenance, "funding_path", lambda symbol: funding)


def test_technical_data_hashes_fingerprints_declared_bytes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    ohlcv = tmp_path / "ohlcv" / "1h" / "BTCUSDT.parquet"
    funding = tmp_path / "funding" / "BTCUSDT.parquet"
    ohlcv.parent.mkdir(parents=True)
    funding.parent.mkdir(parents=True)
    ohlcv.write_bytes(b"bar-data")
    funding.write_bytes(b"funding-data")
    _patch_paths(monkeypatch, ohlcv, funding)

    hashes = provenance.technical_data_hashes("BTCUSDT")

    assert list(hashes) == ["perp_ohlcv", "funding"]
    assert hashes == {
        "perp_ohlcv": "276f98ada012eb876472a489a34494a101856a48dc7e837df3c80b820fe71807",
        "funding": "93a3bf520502d187ee60f4d2eed329495a2f71d622c2a925d51bafd568be6ad4",
    }


def test_technical_data_hashes_fails_closed_on_missing_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_paths(monkeypatch, tmp_path / "missing.parquet", tmp_path / "f.parquet")
    with pytest.raises(DataIntegrityError, match="data missing"):
        provenance.technical_data_hashes("BTCUSDT")
