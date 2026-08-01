from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.expert_portfolio.contracts import ExpertPortfolioSpec
from src.research.expert_portfolio.registry import (
    FORBIDDEN_RETURN_SOURCES,
    is_registered_library,
    load_expert_library,
)


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


def test_unregistered_library_raises(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    _write_registry(registry, [_valid_library()])
    assert is_registered_library("lib-a", registry)
    assert not is_registered_library("nope", registry)
    with pytest.raises(ValueError, match="not registered"):
        load_expert_library("nope", registry)


def test_missing_registry_file_is_unregistered(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not registered"):
        load_expert_library("lib-a", tmp_path / "missing.json")


def test_registered_library_round_trips(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    _write_registry(registry, [_valid_library()])
    spec = load_expert_library("lib-a", registry)
    assert isinstance(spec, ExpertPortfolioSpec)
    assert spec.experts[0].expert_id == "e1"
    assert spec.experts[0].symbols == ("A", "B")
    assert spec.family_exposure_limit == 0.5
    assert spec.symbol_exposure_limit == 0.5


def test_anti_pattern_return_source_is_rejected(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    library = _valid_library()
    assert "funding_signed_directional" in FORBIDDEN_RETURN_SOURCES
    library["experts"][0]["return_source"] = "funding_signed_directional"  # type: ignore[index]
    _write_registry(registry, [library])
    with pytest.raises(ValueError, match="rejected return source"):
        load_expert_library("lib-a", registry)


def test_incomplete_component_is_rejected(tmp_path: Path) -> None:
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


def test_empty_registry_rejects_all(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    _write_registry(registry, [])
    with pytest.raises(ValueError, match="not registered"):
        load_expert_library("lib-a", registry)
