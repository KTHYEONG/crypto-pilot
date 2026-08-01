"""Source-controlled declarative expert-library catalog.

A blueprint names the eligible experts, the supported causal runner keys, the
canonical code and data files whose content is fingerprinted, the sealed
``observation_end``, and the allocator constraints.  The catalog is declarative
and never reads a mutable JSON registry: every fingerprint is derived from the
declared files at registration time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from src.research.expert_portfolio.contracts import ExpertDefinition, ExpertPortfolioSpec
from src.research.technical_experts.catalog import resolve_technical_candidate


@dataclass(frozen=True, slots=True)
class ExpertLibraryBlueprint:
    """One declarative, source-controlled expert library definition."""

    library_id: str
    experts: tuple[ExpertDefinition, ...]
    supported_runners: frozenset[str]
    code_units: Mapping[str, Path]
    data_files: Mapping[str, Path]
    observation_end: str
    gross_exposure: float = 1.0
    family_exposure_limit: float = 1.0
    symbol_exposure_limit: float = 1.0
    min_history_bars: int = 30
    confidence: float = 0.90

    def __post_init__(self) -> None:
        if not self.library_id:
            raise ValueError("library_id must not be empty")
        if not self.experts:
            raise ValueError("experts must contain at least one expert")
        if not self.supported_runners:
            raise ValueError("supported_runners must not be empty")
        if not self.code_units:
            raise ValueError("code_units must not be empty")
        if not self.data_files:
            raise ValueError("data_files must not be empty")
        self.to_spec()
        for expert in self.experts:
            if expert.runner not in self.supported_runners:
                raise ValueError(
                    f"expert {expert.expert_id} uses runner '{expert.runner}' which is not "
                    f"supported by blueprint {self.library_id}"
                )

    def to_spec(self) -> ExpertPortfolioSpec:
        """Build the immutable evaluation spec from this blueprint."""
        return ExpertPortfolioSpec(
            experts=self.experts,
            gross_exposure=self.gross_exposure,
            family_exposure_limit=self.family_exposure_limit,
            symbol_exposure_limit=self.symbol_exposure_limit,
            min_history_bars=self.min_history_bars,
            confidence=self.confidence,
        )


@dataclass(frozen=True, slots=True)
class ExpertLibraryCatalog:
    """Immutable set of library blueprints keyed by ``library_id``."""

    blueprints: Mapping[str, ExpertLibraryBlueprint]

    def __post_init__(self) -> None:
        if len(set(self.blueprints)) != len(self.blueprints):
            raise ValueError("catalog library ids must be unique")

    def get(self, library_id: str) -> ExpertLibraryBlueprint:
        try:
            return self.blueprints[library_id]
        except KeyError as exc:
            raise ValueError(f"library '{library_id}' is not in the catalog") from exc

    def __getitem__(self, library_id: str) -> ExpertLibraryBlueprint:
        return self.get(library_id)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    ).encode("utf-8")


def compute_blueprint_fingerprint(blueprint: ExpertLibraryBlueprint) -> dict[str, object]:
    """Deterministic fingerprint over the declarative blueprint.

    Derived exclusively from the blueprint definitions, the allocator
    configuration, the sealed ``observation_end``, and the content hashes of
    the declared code and data files.  Caller-supplied hash strings are never
    part of the fingerprint; a missing declared file fails closed.
    """
    code_digest = hashlib.sha256()
    for unit_id in sorted(blueprint.code_units):
        code_digest.update(unit_id.encode("utf-8"))
        code_digest.update(Path(blueprint.code_units[unit_id]).read_bytes())
    fingerprint: dict[str, object] = {
        "experts": [asdict(expert) for expert in blueprint.experts],
        "gross_exposure": blueprint.gross_exposure,
        "family_exposure_limit": blueprint.family_exposure_limit,
        "symbol_exposure_limit": blueprint.symbol_exposure_limit,
        "min_history_bars": blueprint.min_history_bars,
        "confidence": blueprint.confidence,
        "observation_end": blueprint.observation_end,
        "code_hash": code_digest.hexdigest(),
        "data_hashes": {
            data_id: _file_sha256(Path(path))
            for data_id, path in sorted(blueprint.data_files.items())
        },
    }
    # Normalise to the exact structure that survives a JSON round-trip so that
    # the stored fingerprint always equals the freshly derived one (tuples
    # become lists, etc.).
    return cast("dict[str, object]", json.loads(_canonical_bytes(fingerprint)))


def registration_id_from_fingerprint(fingerprint: Mapping[str, object]) -> str:
    """Deterministic identity of a registration over its canonical fingerprint."""
    return hashlib.sha256(_canonical_bytes(dict(fingerprint))).hexdigest()


def build_technical_price_v1_blueprint(
    experts: tuple[ExpertDefinition, ...],
    code_units: Mapping[str, Path],
    data_files: Mapping[str, Path],
    observation_end: str = "2025-12-31",
) -> ExpertLibraryBlueprint:
    """Declarative blueprint for the conditional ``technical_price_v1`` library.

    Admission is deliberately narrow: every expert must run the technical
    runner and resolve to a frozen candidate, no family and no underlying
    symbol may appear more than once, and the blueprint requires at least one
    expert. ``default_catalog`` never emits this library: it becomes
    source-controlled only after recorded ``HOLDOUT_PASS`` evidence and a
    human correlation review.
    """
    if not experts:
        raise ValueError("technical_price_v1 requires at least one approved expert")
    families: set[str] = set()
    symbols: set[str] = set()
    for expert in experts:
        if expert.runner != "run_technical_expert":
            raise ValueError(
                f"technical_price_v1 expert {expert.expert_id} must use runner "
                "'run_technical_expert'"
            )
        resolve_technical_candidate(expert.return_source)
        if expert.family in families:
            raise ValueError(
                f"technical_price_v1 admits at most one candidate per family, "
                f"duplicate family '{expert.family}'"
            )
        families.add(expert.family)
        if len(expert.symbols) != 1:
            raise ValueError(
                f"technical_price_v1 experts must hold exactly one symbol, got "
                f"{expert.symbols}"
            )
        if expert.symbols[0] in symbols:
            raise ValueError(
                f"technical_price_v1 admits at most one candidate per symbol, "
                f"duplicate symbol '{expert.symbols[0]}'"
            )
        symbols.add(expert.symbols[0])
    return ExpertLibraryBlueprint(
        library_id="technical_price_v1",
        experts=experts,
        supported_runners=frozenset({"run_technical_expert"}),
        code_units=code_units,
        data_files=data_files,
        observation_end=observation_end,
    )


def default_catalog() -> ExpertLibraryCatalog:
    """Return the currently source-controlled expert library catalog.

    No independent strategy has yet earned ACTIVE status, so the default
    catalog declares no blueprints.  A new strategy earns ACTIVE status only
    through its own evidence; its blueprint is added here after review.
    """
    return ExpertLibraryCatalog(blueprints={})
