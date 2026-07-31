from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from src.core.types import CarryCostModel, CashCarrySpec

REGISTRY_PATH = Path(__file__).resolve().parent.parent.parent / "docs" / "decisions" / "candidate_registry.json"

_REQUIRED_DATA_HASHES = ("spot_ohlcv", "perp_ohlcv", "funding", "borrow")


@dataclass(frozen=True, slots=True)
class CandidateRegistration:
    """Immutable pre-registered candidate identity and sealed fingerprint."""

    candidate_id: str
    hypothesis_id: str
    symbol: str
    observation_end: str
    spec: dict[str, object]
    costs: dict[str, object]
    source_paths: dict[str, str]
    data_hashes: dict[str, str]
    manifest: dict[str, object]
    code_hash: str
    return_source: str
    registration_ts: str
    status: str

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        missing = set(_REQUIRED_DATA_HASHES) - set(self.data_hashes)
        if missing:
            raise ValueError(f"data_hashes must include all four inputs, missing: {sorted(missing)}")
        if not self.code_hash:
            raise ValueError("code_hash must not be empty")
        if not self.hypothesis_id:
            raise ValueError("hypothesis_id must not be empty")


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    ).encode("utf-8")


def compute_candidate_id(
    *,
    hypothesis_id: str,
    symbol: str,
    observation_end: str,
    spec: CashCarrySpec,
    costs: CarryCostModel,
    data_hashes: dict[str, str],
    manifest: dict[str, object],
    code_hash: str,
) -> str:
    """Deterministic candidate identity over the frozen fingerprint.

    The identity is bound to the hypothesis, the sealed observation window, the
    immutable spec/costs, all four dataset content hashes, the manifest
    evidence, and the code hash: any of those changing yields a new candidate.
    """
    payload = {
        "hypothesis_id": hypothesis_id,
        "symbol": symbol,
        "observation_end": observation_end,
        "spec": _canonical_bytes(spec).decode("utf-8"),
        "costs": _canonical_bytes(costs).decode("utf-8"),
        "data_hashes": data_hashes,
        "manifest": manifest,
        "code_hash": code_hash,
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _load_registry(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"candidate registry is not a JSON list: {path}")
    return records


def _save_registry(records: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp.json")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path.replace(path)


def _payload(record: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in record.items() if k not in {"registration_ts", "status"}}


def register_candidate(
    *,
    hypothesis_id: str,
    symbol: str,
    observation_end: str,
    spec: CashCarrySpec,
    costs: CarryCostModel,
    source_paths: dict[str, str],
    data_hashes: dict[str, str],
    manifest: dict[str, object],
    code_hash: str,
    return_source: str,
    registry_path: Path = REGISTRY_PATH,
) -> CandidateRegistration:
    """Append-only pre-registration of a candidate before any evaluation.

    A duplicate ``candidate_id`` with a different payload is an error; mutating
    an existing registration is forbidden. Re-registering an identical
    fingerprint is idempotent and returns the existing record.
    """
    candidate_id = compute_candidate_id(
        hypothesis_id=hypothesis_id,
        symbol=symbol,
        observation_end=observation_end,
        spec=spec,
        costs=costs,
        data_hashes=data_hashes,
        manifest=manifest,
        code_hash=code_hash,
    )
    registration = CandidateRegistration(
        candidate_id=candidate_id,
        hypothesis_id=hypothesis_id,
        symbol=symbol,
        observation_end=observation_end,
        spec=asdict(spec),
        costs=asdict(costs),
        source_paths=dict(source_paths),
        data_hashes=dict(data_hashes),
        manifest=manifest,
        code_hash=code_hash,
        return_source=return_source,
        registration_ts=datetime.now(UTC).isoformat(),
        status="REGISTERED",
    )
    record: dict[str, object] = {
        "candidate_id": registration.candidate_id,
        "hypothesis_id": registration.hypothesis_id,
        "symbol": registration.symbol,
        "observation_end": registration.observation_end,
        "spec": registration.spec,
        "costs": registration.costs,
        "source_paths": registration.source_paths,
        "data_hashes": registration.data_hashes,
        "manifest": registration.manifest,
        "code_hash": registration.code_hash,
        "return_source": registration.return_source,
        "registration_ts": registration.registration_ts,
        "status": registration.status,
    }
    records = _load_registry(registry_path)
    for existing in records:
        if existing.get("candidate_id") == candidate_id:
            if _payload(existing) != _payload(record):
                raise ValueError(
                    f"duplicate candidate_id {candidate_id} registered with a different payload"
                )
            return _record_to_registration(existing)
    records.append(record)
    _save_registry(records, registry_path)
    return registration


def _record_to_registration(record: dict[str, object]) -> CandidateRegistration:
    return CandidateRegistration(
        candidate_id=str(record["candidate_id"]),
        hypothesis_id=str(record["hypothesis_id"]),
        symbol=str(record["symbol"]),
        observation_end=str(record["observation_end"]),
        spec=cast(dict[str, object], record["spec"]),
        costs=cast(dict[str, object], record["costs"]),
        source_paths=cast(dict[str, str], record["source_paths"]),
        data_hashes=cast(dict[str, str], record["data_hashes"]),
        manifest=cast(dict[str, object], record["manifest"]),
        code_hash=str(record["code_hash"]),
        return_source=str(record["return_source"]),
        registration_ts=str(record.get("registration_ts", "")),
        status=str(record.get("status", "REGISTERED")),
    )


def load_registered_candidate(
    candidate_id: str,
    registry_path: Path = REGISTRY_PATH,
) -> CandidateRegistration | None:
    """Return the registered candidate or ``None`` when not registered."""
    for record in _load_registry(registry_path):
        if record.get("candidate_id") == candidate_id:
            return _record_to_registration(record)
    return None


def _check_contract() -> None:
    assert {f.name for f in fields(CandidateRegistration)} == {
        "candidate_id", "hypothesis_id", "symbol", "observation_end", "spec",
        "costs", "source_paths", "data_hashes", "manifest", "code_hash",
        "return_source", "registration_ts", "status",
    }
    assert register_candidate.__name__ == "register_candidate"
    assert load_registered_candidate.__name__ == "load_registered_candidate"


_check_contract()
