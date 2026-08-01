from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from src.research.expert_portfolio.catalog import (
    ExpertLibraryCatalog,
    compute_blueprint_fingerprint,
)
from src.research.expert_portfolio.contracts import ExpertDefinition, ExpertPortfolioSpec
from src.research.expert_portfolio.sources import FORBIDDEN_RETURN_SOURCES
from src.research.provenance.ledger import RUNS_LOG_PATH, load_events
from src.research.provenance.registration import RegistrationRecord, _to_registration_record

LIBRARY_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "docs" / "results" / "expert_library_registry.json"
)


def _load_registry(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"expert library registry is not a JSON list: {path}")
    return records


def is_registered_library(library_id: str, registry_path: Path = LIBRARY_REGISTRY_PATH) -> bool:
    """Return whether ``library_id`` exists in the append-only registry."""
    return any(record.get("library_id") == library_id for record in _load_registry(registry_path))


def load_expert_library(
    library_id: str,
    registry_path: Path = LIBRARY_REGISTRY_PATH,
) -> ExpertPortfolioSpec:
    """Load a pre-registered expert library, rejecting unregistered or ineligible entries.

    An unregistered library, an expert whose return source is a recorded
    anti-pattern, or an incomplete component definition raises ``ValueError``:
    the evaluator refuses anything that was not pre-registered before the sealed
    evaluation.
    """
    if not library_id:
        raise ValueError("library_id must not be empty")
    for record in _load_registry(registry_path):
        if record.get("library_id") != library_id:
            continue
        raw_experts = record.get("experts")
        if not isinstance(raw_experts, list) or not raw_experts:
            raise ValueError(f"library {library_id} has no experts in the registry")
        definitions: list[ExpertDefinition] = []
        for raw in raw_experts:
            if not isinstance(raw, dict):
                raise ValueError(f"library {library_id} contains a malformed expert record")
            source = str(raw.get("return_source", ""))
            if source in FORBIDDEN_RETURN_SOURCES:
                raise ValueError(
                    f"expert {raw.get('expert_id', '')} in library {library_id} uses "
                    f"rejected return source '{source}' and cannot be loaded"
                )
            symbols_raw = raw.get("symbols")
            if not isinstance(symbols_raw, (list, tuple)) or not symbols_raw:
                raise ValueError(
                    f"expert {raw.get('expert_id', '')} in library {library_id} is incomplete"
                )
            definitions.append(ExpertDefinition(
                expert_id=str(raw["expert_id"]),
                return_source=source,
                family=str(raw.get("family", "")),
                symbols=tuple(str(s) for s in symbols_raw),
                runner=str(raw.get("runner", "")),
                code_hash=str(raw.get("code_hash", "")),
            ))
        return ExpertPortfolioSpec(
            experts=tuple(definitions),
            gross_exposure=_numeric_field(record, "gross_exposure", 1.0),
            family_exposure_limit=_numeric_field(record, "family_exposure_limit", 1.0),
            symbol_exposure_limit=_numeric_field(record, "symbol_exposure_limit", 1.0),
            min_history_bars=int(_numeric_field(record, "min_history_bars", 30.0)),
            confidence=_numeric_field(record, "confidence", 0.90),
        )
    raise ValueError(f"library '{library_id}' is not registered")


def _numeric_field(record: dict[str, object], name: str, default: float) -> float:
    """Read a numeric registry field, failing closed on a non-numeric value."""
    value = record.get(name, default)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"library registry field '{name}' must be numeric, got {value!r}")


@dataclass(frozen=True, slots=True)
class RegisteredExpertLibrary:
    """One ACTIVE, fingerprint-verified library resolved to its evaluation spec."""

    library_id: str
    registration_id: str
    spec: ExpertPortfolioSpec
    registration: RegistrationRecord


def resolve_registered_library(
    library_id: str,
    *,
    catalog: ExpertLibraryCatalog,
    ledger_path: Path = RUNS_LOG_PATH,
) -> RegisteredExpertLibrary:
    """Resolve an ACTIVE registration whose fingerprint still matches the catalog.

    An unregistered library, a RETIRED registration, or any fingerprint drift
    between the registered fingerprint and the current blueprint fails closed
    with ``ValueError`` so component evaluation never begins under stale
    evidence.  The forbidden-return-source policy is checked again at
    resolution.
    """
    blueprint = catalog[library_id]
    for expert in blueprint.experts:
        if expert.return_source in FORBIDDEN_RETURN_SOURCES:
            raise ValueError(
                f"expert {expert.expert_id} uses forbidden return source "
                f"'{expert.return_source}' and cannot be evaluated"
            )
    active_record: Mapping[str, object] | None = None
    for event in load_events(ledger_path):
        if event.record_type not in ("registration", "retirement"):
            continue
        if event.payload.get("library_id") != library_id:
            continue
        status = event.payload.get("status")
        if status == "RETIRED":
            raise ValueError(f"library '{library_id}' is RETIRED and cannot be evaluated")
        if status == "ACTIVE":
            active_record = event.payload
            break
    if active_record is None:
        raise ValueError(f"library '{library_id}' has no ACTIVE registration in the ledger")
    registered_fingerprint = cast(Mapping[str, object], active_record["fingerprint"])
    current_fingerprint = compute_blueprint_fingerprint(blueprint)
    if registered_fingerprint != current_fingerprint:
        raise ValueError(
            f"library '{library_id}' fingerprint drift: re-register and re-evaluate, "
            "never run under old evidence"
        )
    return RegisteredExpertLibrary(
        library_id=library_id,
        registration_id=str(active_record["registration_id"]),
        spec=blueprint.to_spec(),
        registration=_to_registration_record(active_record),
    )
