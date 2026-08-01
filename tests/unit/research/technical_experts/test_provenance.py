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

    assert set(hashes) == {"perp_ohlcv", "funding"}
    assert len(hashes["perp_ohlcv"]) == 64
    assert len(hashes["funding"]) == 64
    assert hashes["perp_ohlcv"] != hashes["funding"]


def test_technical_data_hashes_fails_closed_on_missing_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_paths(monkeypatch, tmp_path / "missing.parquet", tmp_path / "f.parquet")
    with pytest.raises(DataIntegrityError, match="data missing"):
        provenance.technical_data_hashes("BTCUSDT")
