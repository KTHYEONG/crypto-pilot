"""Versioned append-only provenance ledger.

Every pre-run registration, completed evaluation, and retirement is one
immutable JSONL line in ``docs/results/runs.jsonl``.  New records carry
``schema_version: 1`` and exactly one ``record_type``; rows written before this
ledger existed carry no version and are normalised to ``schema_version=0`` with
``record_type=evaluation`` by the read boundary.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

RUNS_LOG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "docs" / "results" / "runs.jsonl"

RECORD_TYPES = frozenset({"registration", "evaluation", "retirement"})
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})
# schema_version 0 is reserved for legacy no-version rows normalised on read.
LEDGER_SCHEMA_VERSIONS = frozenset({0, 1})


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    """One immutable ledger record before it is serialised to a JSONL line.

    ``payload`` is the domain-specific body of the event.  ``recorded_at`` is
    filled by :func:`append_event` when the event is actually written so that
    the returned event exactly matches the durable line.
    """

    record_type: str
    payload: Mapping[str, object]
    schema_version: int = 1
    recorded_at: str = ""

    def __post_init__(self) -> None:
        if self.record_type not in RECORD_TYPES:
            raise ValueError(
                f"record_type must be one of {sorted(RECORD_TYPES)}, got {self.record_type!r}"
            )
        if self.schema_version not in LEDGER_SCHEMA_VERSIONS:
            raise ValueError(
                f"schema_version must be one of {sorted(LEDGER_SCHEMA_VERSIONS)}, "
                f"got {self.schema_version}"
            )
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a mapping")


def append_event(event: LedgerEvent, ledger_path: Path = RUNS_LOG_PATH) -> LedgerEvent:
    """Append exactly one fsynced JSONL line under an exclusive append lock.

    The append is immutable: a prior line is never rewritten or merged, and a
    rejected event appends nothing.  The returned event carries the exact
    ``recorded_at`` that was durably written.
    """
    recorded_at = event.recorded_at or datetime.now(UTC).isoformat()
    line = json.dumps(
        {
            "record_type": event.record_type,
            "schema_version": event.schema_version,
            "recorded_at": recorded_at,
            "payload": dict(event.payload),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return replace(event, recorded_at=recorded_at)


def load_events(ledger_path: Path = RUNS_LOG_PATH) -> list[LedgerEvent]:
    """Return every event in the ledger in append order.

    A malformed line or an unsupported schema version raises a diagnostic
    ``ValueError`` that names the line number; records are never silently
    skipped.  Legacy no-version rows normalise to ``record_type=evaluation``
    with ``schema_version=0``.
    """
    if not ledger_path.exists():
        return []
    events: list[LedgerEvent] = []
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed ledger line {lineno}: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"ledger line {lineno} must be a JSON object")
        try:
            schema_version = int(data.get("schema_version", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ledger line {lineno} has a malformed schema_version") from exc
        if schema_version == 0:
            events.append(LedgerEvent(
                record_type="evaluation",
                payload=data,
                schema_version=0,
                recorded_at=str(data.get("ts", "")),
            ))
            continue
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"ledger line {lineno} uses unsupported schema_version {schema_version}"
            )
        record_type = data.get("record_type")
        if record_type not in RECORD_TYPES:
            raise ValueError(f"ledger line {lineno} has unknown record_type {record_type!r}")
        raw_payload = data.get("payload")
        if not isinstance(raw_payload, dict):
            raise ValueError(f"ledger line {lineno} payload must be a JSON object")
        events.append(LedgerEvent(
            record_type=str(record_type),
            payload=raw_payload,
            schema_version=schema_version,
            recorded_at=str(data.get("recorded_at", "")),
        ))
    return events


def build_evaluation_event(
    *,
    workflow: str,
    ts: str,
    git_sha: str | None,
    git_dirty: bool,
    metrics: Mapping[str, object],
    reliability: Mapping[str, object],
    promotion: Mapping[str, object] | None,
    parent_registration_id: str | None = None,
    **extra: object,
) -> LedgerEvent:
    """Build the single common evaluation-event shape used by every writer.

    All evaluation records share the immutable request/outcome pieces
    (``workflow``, ``git`` provenance, ``metrics``, ``reliability``, promotion,
    and the optional ``parent_registration_id`` link) plus workflow-specific
    fields passed as ``extra``.  Writers never serialise JSON themselves.
    """
    payload: dict[str, object] = {
        "workflow": workflow,
        "ts": ts,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "metrics": dict(metrics),
        "reliability": dict(reliability),
        "promotion": dict(promotion) if promotion is not None else None,
        "parent_registration_id": parent_registration_id,
    }
    payload.update(extra)
    return LedgerEvent(record_type="evaluation", payload=payload)


def load_evaluation_runs(ledger_path: Path = RUNS_LOG_PATH) -> pd.DataFrame:
    """Return evaluation rows only, normalising legacy rows to ``schema_version=0``.

    v1 registration and retirement events are excluded so run comparisons can
    never mix pre-run registrations into the result set.  Legacy no-version
    rows remain available with ``record_type=evaluation`` and their original
    comparison columns.
    """
    rows: list[dict[str, object]] = []
    for event in load_events(ledger_path):
        if event.record_type != "evaluation":
            continue
        row = dict(event.payload)
        row["record_type"] = event.record_type
        row["schema_version"] = event.schema_version
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)
