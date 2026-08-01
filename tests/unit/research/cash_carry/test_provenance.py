from __future__ import annotations

from pathlib import Path

import pytest

from src.common.errors import DataIntegrityError
from src.research.cash_carry import provenance


def _patch_paths(monkeypatch, paths: dict[str, Path]) -> None:
    monkeypatch.setattr(
        provenance, "spot_ohlcv_path", lambda symbol, timeframe: paths["spot_ohlcv"],
    )
    monkeypatch.setattr(
        provenance, "ohlcv_path", lambda symbol, timeframe: paths["perp_ohlcv"],
    )
    monkeypatch.setattr(provenance, "funding_path", lambda symbol: paths["funding"])
    monkeypatch.setattr(provenance, "borrow_path", lambda symbol: paths["borrow"])


def _write_sources(paths: dict[str, Path], payloads: dict[str, bytes]) -> None:
    for name, payload in payloads.items():
        paths[name].parent.mkdir(parents=True, exist_ok=True)
        paths[name].write_bytes(payload)


def _source_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "spot_ohlcv": tmp_path / "spot" / "BTCUSDT.parquet",
        "perp_ohlcv": tmp_path / "perp" / "BTCUSDT.parquet",
        "funding": tmp_path / "funding" / "BTCUSDT.parquet",
        "borrow": tmp_path / "borrow" / "BTCUSDT.parquet",
    }


def test_cash_carry_data_hashes_locks_source_keys_and_digests(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _source_paths(tmp_path)
    payloads = {
        "spot_ohlcv": b"spot-bytes",
        "perp_ohlcv": b"perp-bytes",
        "funding": b"funding-bytes",
        "borrow": b"borrow-bytes",
    }
    _write_sources(paths, payloads)
    _patch_paths(monkeypatch, paths)

    hashes = provenance.cash_carry_data_hashes("BTCUSDT")

    assert list(hashes) == ["spot_ohlcv", "perp_ohlcv", "funding", "borrow"]
    assert hashes == {
        "spot_ohlcv": "c26dcd8a08f2932d9bf5908d235c86890fb2f400e2fcdd6a4ec5f963435de633",
        "perp_ohlcv": "96aa9186a3132045bfc1f69102e208929f75bf0f410aa991235bb54a87539fd0",
        "funding": "12bc1689ad354296733ac04c1866bf83242ea6fb2a4a262d3f97b09c7146f6e9",
        "borrow": "1a3e135fdcccfe91a1f06db3f317f1330f983804d0fdad76dcc78de11d7de4e8",
    }


def test_cash_carry_data_hashes_fails_closed_on_missing_borrow(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _source_paths(tmp_path)
    _write_sources(
        paths,
        {
            "spot_ohlcv": b"spot-bytes",
            "perp_ohlcv": b"perp-bytes",
            "funding": b"funding-bytes",
            "borrow": b"borrow-bytes",
        },
    )
    paths["borrow"].unlink()
    _patch_paths(monkeypatch, paths)

    with pytest.raises(DataIntegrityError, match=r"borrow data missing for BTCUSDT"):
        provenance.cash_carry_data_hashes("BTCUSDT")
