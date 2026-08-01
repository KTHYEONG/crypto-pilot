from __future__ import annotations

import json
from pathlib import Path

from src.research.expert_portfolio.contracts import ExpertDefinition, ExpertPortfolioSpec

LIBRARY_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "docs" / "results" / "expert_library_registry.json"
)

# Pre-registered anti-pattern return sources: previously rejected hypotheses must
# never be re-labelled as new experts and re-enter promotion by renaming.
FORBIDDEN_RETURN_SOURCES: frozenset[str] = frozenset({
    "donchian_multi_symbol_diversification",
    "bollinger_mean_reversion",
    "cross_sectional_momentum",
    "cash_carry",
    "taker_flow",
    "funding_signed_directional",
})


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
