from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.expert_portfolio.catalog import ExpertLibraryCatalog
from src.research.expert_portfolio.contracts import ExpertPortfolioSpec
from src.research.expert_portfolio.registry import (
    FORBIDDEN_RETURN_SOURCES,
    is_registered_library,
    load_expert_library,
    resolve_registered_library,
)
from src.research.provenance.ledger import LedgerEvent, append_event
from src.research.provenance.registration import register_expert_library

_DIRECTIONAL_FINGERPRINT = {
    "experts": [],
    "code_hash": "a" * 64,
    "data_hashes": {"AUSDT": {"ohlcv_1h": "b" * 64}},
}


def _write_registry(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def _valid_library() -> dict[str, object]:
    return {
        "library_id": "lib-a",
        "experts": [
            {
                "expert_id": "e1",
                "return_source": "cointegration_residual",
                "family": "pair_residual",
                "symbols": ["A", "B"],
                "runner": "run_pair_residual",
                "code_hash": "abc",
            },
        ],
        "gross_exposure": 1.0,
        "family_exposure_limit": 0.5,
        "symbol_exposure_limit": 0.5,
        "min_history_bars": 30,
        "confidence": 0.90,
    }


class TestLegacyJsonRegistry:
    def test_unregistered_library_raises(self, tmp_path: Path) -> None:
        registry = tmp_path / "registry.json"
        _write_registry(registry, [_valid_library()])
        assert is_registered_library("lib-a", registry)
        assert not is_registered_library("nope", registry)
        with pytest.raises(ValueError, match="not registered"):
            load_expert_library("nope", registry)

    def test_missing_registry_file_is_unregistered(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not registered"):
            load_expert_library("lib-a", tmp_path / "missing.json")

    def test_registered_library_round_trips(self, tmp_path: Path) -> None:
        registry = tmp_path / "registry.json"
        _write_registry(registry, [_valid_library()])
        spec = load_expert_library("lib-a", registry)
        assert isinstance(spec, ExpertPortfolioSpec)
        assert spec.experts[0].expert_id == "e1"
        assert spec.experts[0].symbols == ("A", "B")
        assert spec.family_exposure_limit == 0.5
        assert spec.symbol_exposure_limit == 0.5

    def test_anti_pattern_return_source_is_rejected(self, tmp_path: Path) -> None:
        registry = tmp_path / "registry.json"
        library = _valid_library()
        assert "funding_signed_directional" in FORBIDDEN_RETURN_SOURCES
        library["experts"][0]["return_source"] = "funding_signed_directional"  # type: ignore[index]
        _write_registry(registry, [library])
        with pytest.raises(ValueError, match="rejected return source"):
            load_expert_library("lib-a", registry)

    def test_incomplete_component_is_rejected(self, tmp_path: Path) -> None:
        registry = tmp_path / "registry.json"
        library = _valid_library()
        del library["experts"][0]["code_hash"]  # type: ignore[index]
        _write_registry(registry, [library])
        with pytest.raises(ValueError, match="must not be empty"):
            load_expert_library("lib-a", registry)

        empty_symbols = _valid_library()
        empty_symbols["experts"][0]["symbols"] = []  # type: ignore[index]
        _write_registry(registry, [empty_symbols])
        with pytest.raises(ValueError, match="incomplete"):
            load_expert_library("lib-a", registry)

    def test_empty_registry_rejects_all(self, tmp_path: Path) -> None:
        registry = tmp_path / "registry.json"
        _write_registry(registry, [])
        with pytest.raises(ValueError, match="not registered"):
            load_expert_library("lib-a", registry)


class TestResolveRegisteredLibrary:
    def test_resolve_registered_library_rejects_fingerprint_drift(
        self,
        tmp_path: Path,
        expert_library_catalog: ExpertLibraryCatalog,
        expert_library_blueprint,
    ) -> None:
        # PL-EXPERT-001: the valid library resolves; a modified source fails
        # closed instead of running under old evidence.
        ledger = tmp_path / "runs.jsonl"
        registration = register_expert_library(
            "valid_library", catalog=expert_library_catalog, ledger_path=ledger,
        )
        resolved = resolve_registered_library(
            "valid_library", catalog=expert_library_catalog, ledger_path=ledger,
        )
        assert resolved.library_id == "valid_library"
        assert resolved.registration_id == registration.registration_id
        assert resolved.spec.experts[0].expert_id == "e1"

        engine = next(iter(expert_library_blueprint.code_units.values()))
        engine.write_text("# drifted\nVALUE = 99\n", encoding="utf-8")
        with pytest.raises(ValueError, match="drift"):
            resolve_registered_library(
                "valid_library", catalog=expert_library_catalog, ledger_path=ledger,
            )

    def test_unregistered_library_fails_closed(self, tmp_path: Path) -> None:
        ledger = tmp_path / "runs.jsonl"
        with pytest.raises(ValueError, match="not in the catalog"):
            resolve_registered_library(
                "missing", catalog=ExpertLibraryCatalog(blueprints={}), ledger_path=ledger,
            )

    def test_no_active_registration_fails_closed(
        self, tmp_path: Path, expert_library_catalog,
    ) -> None:
        ledger = tmp_path / "runs.jsonl"
        with pytest.raises(ValueError, match="no ACTIVE registration"):
            resolve_registered_library(
                "valid_library", catalog=expert_library_catalog, ledger_path=ledger,
            )

    def test_retired_library_cannot_be_evaluated(
        self, tmp_path: Path, expert_library_catalog,
    ) -> None:
        ledger = tmp_path / "runs.jsonl"
        append_event(
            LedgerEvent(
                record_type="retirement",
                payload={
                    "registration_id": "retired-1",
                    "library_id": "valid_library",
                    "status": "RETIRED",
                    "registered_at": "2026-01-01T00:00:00+00:00",
                    "fingerprint": dict(_DIRECTIONAL_FINGERPRINT),
                },
            ),
            ledger_path=ledger,
        )
        with pytest.raises(ValueError, match="RETIRED"):
            resolve_registered_library(
                "valid_library", catalog=expert_library_catalog, ledger_path=ledger,
            )

    def test_ignores_evaluation_events_when_resolving(
        self, tmp_path: Path, expert_library_catalog,
    ) -> None:
        ledger = tmp_path / "runs.jsonl"
        append_event(
            LedgerEvent(
                record_type="evaluation",
                payload={"workflow": "expert_portfolio", "ts": "2026-01-01T00:00:00+00:00"},
            ),
            ledger_path=ledger,
        )
        with pytest.raises(ValueError, match="no ACTIVE registration"):
            resolve_registered_library(
                "valid_library", catalog=expert_library_catalog, ledger_path=ledger,
            )
