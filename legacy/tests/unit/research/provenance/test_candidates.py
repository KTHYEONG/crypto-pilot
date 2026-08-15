from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from src.research.cash_carry.contracts import CarryCostModel, CashCarrySpec
from src.research.provenance.candidates import (
    compute_candidate_id,
    load_registered_candidate,
    register_candidate,
)

_HASHES = {
    "spot_ohlcv": "a" * 64,
    "perp_ohlcv": "b" * 64,
    "funding": "c" * 64,
    "borrow": "d" * 64,
}
_MANIFEST: dict[str, object] = {"ohlcv/1h": {"BTCUSDT": {"row_count": 100}}}
_PATHS = {
    "spot_ohlcv": "data/spot/ohlcv/1h/BTCUSDT.parquet",
    "perp_ohlcv": "data/futures/ohlcv/1h/BTCUSDT.parquet",
    "funding": "data/futures/funding/BTCUSDT.parquet",
    "borrow": "data/spot/borrow/BTCUSDT.parquet",
}


def _kwargs(**overrides):
    base = {
        "hypothesis_id": "cash_and_carry_basis",
        "symbol": "BTCUSDT",
        "observation_end": "2025-12-31 23:59:59+00:00",
        "spec": CashCarrySpec(symbol="BTCUSDT"),
        "costs": CarryCostModel(),
        "source_paths": dict(_PATHS),
        "data_hashes": dict(_HASHES),
        "manifest": _MANIFEST,
        "code_hash": "code123",
        "return_source": "spot_perp_funding_carry",
    }
    base.update(overrides)
    return base


class TestCandidateRegistry:
    def test_register_then_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "candidate_registry.json"
        registration = register_candidate(registry_path=path, **_kwargs())
        loaded = load_registered_candidate(registration.candidate_id, registry_path=path)
        assert loaded is not None
        assert loaded.candidate_id == registration.candidate_id
        assert loaded.symbol == "BTCUSDT"
        assert loaded.observation_end == "2025-12-31 23:59:59+00:00"
        assert loaded.data_hashes == _HASHES
        assert loaded.code_hash == "code123"
        assert loaded.status == "REGISTERED"

    def test_register_is_append_only_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "candidate_registry.json"
        first = register_candidate(registry_path=path, **_kwargs())
        second = register_candidate(registry_path=path, **_kwargs())
        assert first.candidate_id == second.candidate_id
        assert path.read_text(encoding="utf-8").count("candidate_id") == 1

    def test_duplicate_id_with_different_payload_is_error(self, tmp_path: Path) -> None:
        # SC-REG-01: a candidate_id already registered with a different payload
        # is an immutable conflict, never silently overwritten.
        path = tmp_path / "candidate_registry.json"
        kwargs = _kwargs()
        candidate_id = compute_candidate_id(
            hypothesis_id=kwargs["hypothesis_id"],
            symbol=kwargs["symbol"],
            observation_end=kwargs["observation_end"],
            spec=kwargs["spec"],
            costs=kwargs["costs"],
            data_hashes=kwargs["data_hashes"],
            manifest=kwargs["manifest"],
            code_hash=kwargs["code_hash"],
        )
        conflict = dict(kwargs)
        conflict["spec"] = CashCarrySpec(symbol="BTCUSDT", initial_margin_rate=0.2)
        conflict_record = {
            "candidate_id": candidate_id,
            "hypothesis_id": kwargs["hypothesis_id"],
            "symbol": kwargs["symbol"],
            "observation_end": kwargs["observation_end"],
            "spec": asdict(conflict["spec"]),
            "costs": asdict(kwargs["costs"]),
            "source_paths": kwargs["source_paths"],
            "data_hashes": kwargs["data_hashes"],
            "manifest": kwargs["manifest"],
            "code_hash": kwargs["code_hash"],
            "return_source": kwargs["return_source"],
            "registration_ts": "2026-01-01T00:00:00+00:00",
            "status": "REGISTERED",
        }
        path.write_text(json.dumps([conflict_record]), encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            register_candidate(registry_path=path, **kwargs)

    def test_candidate_id_binds_data_and_code(self) -> None:
        base_id = compute_candidate_id(
            hypothesis_id="cash_and_carry_basis",
            symbol="BTCUSDT",
            observation_end="2025-12-31 23:59:59+00:00",
            spec=CashCarrySpec(symbol="BTCUSDT"),
            costs=CarryCostModel(),
            data_hashes=_HASHES,
            manifest=_MANIFEST,
            code_hash="code123",
        )
        other = compute_candidate_id(
            hypothesis_id="cash_and_carry_basis",
            symbol="BTCUSDT",
            observation_end="2025-12-31 23:59:59+00:00",
            spec=CashCarrySpec(symbol="BTCUSDT"),
            costs=CarryCostModel(),
            data_hashes={**_HASHES, "borrow": "f" * 64},
            manifest=_MANIFEST,
            code_hash="code123",
        )
        assert base_id != other

    def test_missing_data_hash_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "candidate_registry.json"
        hashes = dict(_HASHES)
        hashes.pop("funding")
        with pytest.raises(ValueError, match="funding"):
            register_candidate(registry_path=path, **_kwargs(data_hashes=hashes))

    def test_load_unknown_candidate_returns_none(self, tmp_path: Path) -> None:
        assert load_registered_candidate("nope", registry_path=tmp_path / "missing.json") is None
