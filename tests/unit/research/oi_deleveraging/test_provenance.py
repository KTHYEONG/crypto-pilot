from __future__ import annotations

from pathlib import Path

import pytest

from src.common.errors import DataIntegrityError
from src.research.oi_deleveraging import provenance


def _patch_paths(monkeypatch, paths: dict[str, Path]) -> None:
    monkeypatch.setattr(
        provenance, "ohlcv_path", lambda symbol, timeframe: paths["perp_ohlcv"],
    )
    monkeypatch.setattr(provenance, "funding_path", lambda symbol: paths["funding"])
    monkeypatch.setattr(provenance, "metrics_path", lambda symbol: paths["metrics"])


def _source_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "perp_ohlcv": tmp_path / "perp" / "BTCUSDT.parquet",
        "funding": tmp_path / "funding" / "BTCUSDT.parquet",
        "metrics": tmp_path / "metrics" / "BTCUSDT.parquet",
    }


def test_oi_deleveraging_data_hashes_locks_source_keys_and_digests(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _source_paths(tmp_path)
    payloads = {
        "perp_ohlcv": b"perp-bytes",
        "funding": b"funding-bytes",
        "metrics": b"metrics-bytes",
    }
    for name, payload in payloads.items():
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        paths[name].write_bytes(payload)
    _patch_paths(monkeypatch, paths)

    hashes = provenance.oi_deleveraging_data_hashes("BTCUSDT")

    assert list(hashes) == ["perp_ohlcv", "funding", "metrics"]
    assert hashes == {
        "perp_ohlcv": "96aa9186a3132045bfc1f69102e208929f75bf0f410aa991235bb54a87539fd0",
        "funding": "12bc1689ad354296733ac04c1866bf83242ea6fb2a4a262d3f97b09c7146f6e9",
        "metrics": "f914db7bb771338acf962a50e740cdb3c252bcee8205e6a1f2680da3a8ad997e",
    }


def test_oi_deleveraging_data_hashes_fails_closed_on_missing_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _source_paths(tmp_path)
    for name, payload in {
        "perp_ohlcv": b"perp-bytes",
        "funding": b"funding-bytes",
        "metrics": b"metrics-bytes",
    }.items():
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        paths[name].write_bytes(payload)
    paths["metrics"].unlink()
    _patch_paths(monkeypatch, paths)

    with pytest.raises(DataIntegrityError, match=r"metrics data missing for BTCUSDT"):
        provenance.oi_deleveraging_data_hashes("BTCUSDT")
