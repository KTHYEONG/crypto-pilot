"""Generic immutable registration lifecycle and expert-library registration.

A registration is a pre-run record appended to the ledger before any
evaluation.  The registration id derives deterministically from the canonical
fingerprint (blueprint definitions + code hashes + data hashes); caller-supplied
hash strings are never trusted.  Equal fingerprints are idempotent, a different
fingerprint for an already-ACTIVE entity fails closed, and the historical
directional anti-pattern is only ever migrated as RETIRED.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd

from src.research.expert_portfolio.catalog import (
    ExpertLibraryBlueprint,
    ExpertLibraryCatalog,
    compute_blueprint_fingerprint,
    registration_id_from_fingerprint,
)
from src.research.expert_portfolio.sources import FORBIDDEN_RETURN_SOURCES
from src.research.provenance.ledger import RUNS_LOG_PATH, LedgerEvent, append_event, load_events

_LEGACY_FINGERPRINT_KEYS = (
    "domain",
    "candidate_id",
    "hypothesis_id",
    "symbols",
    "rules",
    "parameters",
    "data_hashes",
    "code_hash",
    "observation_end",
    "return_source",
)


@dataclass(frozen=True, slots=True)
class RegistrationRecord:
    """One immutable registration read back from the ledger."""

    registration_id: str
    library_id: str
    status: str
    fingerprint: Mapping[str, object]
    registered_at: str
    record: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class MigrationReport:
    """Result of one legacy-candidate-registry migration run."""

    appended_count: int
    existing_count: int


def _registration_payload(
    *,
    registration_id: str,
    library_id: str,
    status: str,
    fingerprint: Mapping[str, object],
    registered_at: str,
    extra_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "registration_id": registration_id,
        "library_id": library_id,
        "status": status,
        "registered_at": registered_at,
        "fingerprint": dict(fingerprint),
    }
    if extra_payload:
        payload.update(extra_payload)
    return payload


def _to_registration_record(record: Mapping[str, object]) -> RegistrationRecord:
    return RegistrationRecord(
        registration_id=str(record["registration_id"]),
        library_id=str(record["library_id"]),
        status=str(record["status"]),
        fingerprint=cast(Mapping[str, object], record["fingerprint"]),
        registered_at=str(record.get("registered_at", "")),
        record=dict(record),
    )


def register_registration(
    *,
    library_id: str,
    fingerprint: Mapping[str, object],
    status: str = "ACTIVE",
    registered_at: str | None = None,
    extra_payload: Mapping[str, object] | None = None,
    ledger_path: Path = RUNS_LOG_PATH,
) -> RegistrationRecord:
    """Append or return one idempotent immutable registration event.

    The registration id derives only from the canonical fingerprint.  An equal
    fingerprint for the same entity is a no-op; a different fingerprint for an
    already-ACTIVE entity fails closed instead of overwriting history.
    ``status`` selects the record type: ACTIVE appends a ``registration``
    event, RETIRED appends a ``retirement`` event.
    """
    if status not in ("ACTIVE", "RETIRED"):
        raise ValueError(f"status must be ACTIVE or RETIRED, got {status!r}")
    registration_id = registration_id_from_fingerprint(fingerprint)
    record_type = "registration" if status == "ACTIVE" else "retirement"
    for event in load_events(ledger_path):
        if event.record_type not in ("registration", "retirement"):
            continue
        prior = event.payload
        if prior.get("registration_id") == registration_id:
            return _to_registration_record(prior)
        if (
            status == "ACTIVE"
            and prior.get("library_id") == library_id
            and prior.get("status") == "ACTIVE"
        ):
            raise ValueError(
                f"library '{library_id}' is already ACTIVE under a different fingerprint; "
                "re-register only after the existing registration is retired"
            )
    payload = _registration_payload(
        registration_id=registration_id,
        library_id=library_id,
        status=status,
        fingerprint=fingerprint,
        registered_at=registered_at or datetime.now(UTC).isoformat(),
        extra_payload=extra_payload,
    )
    append_event(LedgerEvent(record_type=record_type, payload=payload), ledger_path=ledger_path)
    return _to_registration_record(payload)


def _last_data_timestamp(path: Path) -> pd.Timestamp:
    """Return the newest available timestamp for a declared data file."""
    frame = pd.read_parquet(path)
    if isinstance(frame.index, pd.DatetimeIndex) and len(frame.index):
        return pd.Timestamp(frame.index.max())
    for column in ("timestamp", "datetime", "open_time"):
        if column in frame.columns:
            ts = pd.to_datetime(
                pd.to_numeric(frame[column], errors="coerce"), unit="ms", utc=True, errors="coerce",
            ).dropna()
            if len(ts):
                return pd.Timestamp(ts.max())
    raise ValueError(f"cannot determine available data range for {path}")


def _check_observation_end(blueprint: ExpertLibraryBlueprint) -> None:
    end = pd.Timestamp(blueprint.observation_end)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    for data_id, path in blueprint.data_files.items():
        last = _last_data_timestamp(Path(path))
        if last.tzinfo is None:
            last = last.tz_localize("UTC")
        if end > last:
            raise ValueError(
                f"observation_end {end.isoformat()} is beyond available data "
                f"({last.isoformat()}) for {data_id}"
            )


def _validate_blueprint(blueprint: ExpertLibraryBlueprint) -> None:
    """Fail closed on any definition that must never become ACTIVE."""
    blueprint.to_spec()
    for expert in blueprint.experts:
        if expert.return_source in FORBIDDEN_RETURN_SOURCES:
            raise ValueError(
                f"expert {expert.expert_id} uses forbidden return source "
                f"'{expert.return_source}' and cannot be registered"
            )
        if expert.runner not in blueprint.supported_runners:
            raise ValueError(
                f"expert {expert.expert_id} uses runner '{expert.runner}' which is not "
                f"supported by blueprint {blueprint.library_id}"
            )
    for unit_id, path in blueprint.code_units.items():
        if not Path(path).exists():
            raise ValueError(f"code unit {unit_id} is missing: {path}")
    for data_id, path in blueprint.data_files.items():
        if not Path(path).exists():
            raise ValueError(f"data file {data_id} is missing: {path}")
    _check_observation_end(blueprint)


def register_expert_library(
    library_id: str,
    *,
    catalog: ExpertLibraryCatalog,
    ledger_path: Path = RUNS_LOG_PATH,
) -> RegistrationRecord:
    """Fingerprint and register one expert library as idempotent ACTIVE.

    All code and data hashes are derived here from the blueprint's declared
    files; user-provided hash strings are not accepted.  An unknown library, a
    forbidden return source, a runner mismatch, missing files, or an
    ``observation_end`` beyond available data all raise ``ValueError`` before
    any event is written.
    """
    blueprint = catalog[library_id]
    _validate_blueprint(blueprint)
    fingerprint = compute_blueprint_fingerprint(blueprint)
    return register_registration(
        library_id=library_id,
        fingerprint=fingerprint,
        status="ACTIVE",
        extra_payload={"spec_fingerprint": blueprint.to_spec().fingerprint()},
        ledger_path=ledger_path,
    )


def _legacy_fingerprint(record: Mapping[str, object]) -> dict[str, object]:
    return {key: record[key] for key in _LEGACY_FINGERPRINT_KEYS if key in record}


def migrate_legacy_candidate_registry(
    source_path: Path,
    ledger_path: Path = RUNS_LOG_PATH,
) -> MigrationReport:
    """Migrate the legacy ``candidate_registry.json`` rows as RETIRED events.

    Each legacy row becomes one immutable ``retirement`` event that preserves
    the original payload, ``candidate_id``, rules, and data hashes and maps the
    legacy ``registration_ts`` to ``registered_at``.  Rerunning is idempotent;
    an unrecognized row raises ``ValueError``.  Deleting the source file is a
    separate, confirmed repository migration step.
    """
    if not source_path.exists():
        return MigrationReport(appended_count=0, existing_count=0)
    with source_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"candidate registry is not a JSON list: {source_path}")

    appended = 0
    existing = 0
    existing_ids = {
        event.payload.get("registration_id")
        for event in load_events(ledger_path)
        if event.record_type in ("registration", "retirement")
    }
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"candidate registry contains a non-object row in {source_path}")
        candidate_id = str(record.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError(f"candidate registry row is missing candidate_id in {source_path}")
        fingerprint = _legacy_fingerprint(record)
        registration_id = registration_id_from_fingerprint(fingerprint)
        if registration_id in existing_ids:
            existing += 1
            continue
        preserved = {
            key: value for key, value in record.items() if key not in {"registration_ts", "status"}
        }
        payload = _registration_payload(
            registration_id=registration_id,
            library_id=candidate_id,
            status="RETIRED",
            fingerprint=fingerprint,
            registered_at=str(record.get("registration_ts", "")) or datetime.now(UTC).isoformat(),
            extra_payload={"legacy": preserved},
        )
        append_event(LedgerEvent(record_type="retirement", payload=payload), ledger_path=ledger_path)
        existing_ids.add(registration_id)
        appended += 1
    return MigrationReport(appended_count=appended, existing_count=existing)
