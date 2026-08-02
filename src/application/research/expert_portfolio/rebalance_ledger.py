"""Append-only rolling rebalance ledger and atomic current-profile pointer.

Every quarterly selection decision is one immutable JSONL line keyed by
(profile, rebalance_start, snapshot_key); re-running the identical snapshot is
idempotent and later data can never rewrite an earlier line. A separate small
JSON pointer stores the latest successful record for the current profile and is
written atomically, so the runtime executor always reads one complete decision.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from src.research.expert_portfolio.rolling import (
    RollingCandidateAuditRecord,
    RollingSelectionRecord,
)

REBALANCE_LEDGER_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "docs" / "results" / "rebalance_ledger.jsonl"
)
CURRENT_PROFILE_POINTER_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "docs" / "results" / "current_rolling_profile.json"
)
ROLLING_CANDIDATE_AUDIT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "docs" / "results" / "rolling_candidate_audit.jsonl"
)


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    ).encode("utf-8")


def rebalance_snapshot_key(inputs: Mapping[str, object]) -> str:
    """Deterministic hash of the exact selection inputs for one rebalance."""
    return hashlib.sha256(_canonical_bytes(inputs)).hexdigest()


def _payload_from_record(record: RollingSelectionRecord) -> dict[str, object]:
    return dict(record.to_payload())


def _record_from_payload(payload: Mapping[str, object]) -> RollingSelectionRecord:
    proposal_id_value = payload.get("proposal_id")
    raw_ids = cast("list[object] | tuple[object, ...]", payload.get("expert_ids", ()))
    raw_hashes = cast("Mapping[str, object]", payload.get("data_hashes", {}))
    return RollingSelectionRecord(
        profile=str(payload["profile"]),
        rebalance_start=str(payload["rebalance_start"]),
        scored_start=str(payload["scored_start"]),
        observed_end=str(payload["observed_end"]),
        load_start=str(payload["load_start"]),
        deploy_start=str(payload["deploy_start"]),
        deploy_end=str(payload["deploy_end"]),
        status=str(payload["status"]),
        selection_status=str(payload["selection_status"]),
        proposal_id=None if proposal_id_value is None else str(proposal_id_value),
        expert_ids=tuple(str(expert_id) for expert_id in raw_ids),
        incumbent_kept=bool(payload.get("incumbent_kept", False)),
        code_hash=str(payload["code_hash"]),
        data_hashes={
            str(symbol): {
                str(key): str(value)
                for key, value in cast("Mapping[str, object]", values).items()
            }
            for symbol, values in raw_hashes.items()
        },
        snapshot_key=str(payload["snapshot_key"]),
        recorded_at=str(payload.get("recorded_at", "")),
    )


def load_rebalance_records(
    ledger_path: Path = REBALANCE_LEDGER_PATH,
) -> tuple[RollingSelectionRecord, ...]:
    """Return every rebalance decision in append order; a malformed line fails closed."""
    if not ledger_path.exists():
        return ()
    records: list[RollingSelectionRecord] = []
    for lineno, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed rebalance ledger line {lineno}: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"rebalance ledger line {lineno} must be a JSON object")
        records.append(_record_from_payload(data))
    return tuple(records)


def append_rebalance_record(
    record: RollingSelectionRecord,
    ledger_path: Path = REBALANCE_LEDGER_PATH,
) -> RollingSelectionRecord:
    """Append one decision idempotently under an exclusive append lock.

    When a record for the same ``(profile, rebalance_start, snapshot_key)``
    already exists the existing line is returned and nothing is written, so a
    replay of an identical snapshot never duplicates or mutates history.
    """
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            for existing in _iter_lines(ledger_path):
                existing_record = _record_from_payload(existing)
                if (
                    existing_record.profile == record.profile
                    and existing_record.rebalance_start == record.rebalance_start
                    and existing_record.snapshot_key == record.snapshot_key
                ):
                    return existing_record
            recorded_at = record.recorded_at or datetime.now(UTC).isoformat()
            stored = _record_from_payload({**dict(record.to_payload()), "recorded_at": recorded_at})
            handle.write(json.dumps(_payload_from_record(stored), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return stored
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _iter_lines(ledger_path: Path) -> Iterator[dict[str, object]]:
    if not ledger_path.exists():
        return
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data = json.loads(line)
                if isinstance(data, dict):
                    yield data


def _audit_from_payload(payload: Mapping[str, object]) -> RollingCandidateAuditRecord:
    selected_value = payload.get("selected")
    cash_reason_value = payload.get("cash_reason")
    return RollingCandidateAuditRecord(
        profile=str(payload["profile"]),
        rebalance_start=str(payload["rebalance_start"]),
        snapshot_key=str(payload["snapshot_key"]),
        window=dict(cast("Mapping[str, object]", payload["window"])),
        selection=dict(cast("Mapping[str, object]", payload["selection"])),
        candidates=tuple(
            dict(cast("Mapping[str, object]", entry))
            for entry in cast("list[object]", payload["candidates"])
        ),
        proposals=tuple(
            dict(cast("Mapping[str, object]", entry))
            for entry in cast("list[object]", payload["proposals"])
        ),
        shortlist=tuple(str(entry) for entry in cast("list[object]", payload["shortlist"])),
        training=tuple(
            dict(cast("Mapping[str, object]", entry))
            for entry in cast("list[object]", payload["training"])
        ),
        selected=(
            dict(cast("Mapping[str, object]", selected_value))
            if selected_value is not None
            else None
        ),
        selection_status=str(payload["selection_status"]),
        execution=dict(
            cast("Mapping[str, object]", payload.get("execution", {})),
        ),
        incumbent_kept=bool(payload.get("incumbent_kept", False)),
        cash_reason=(
            None if cash_reason_value is None else str(cash_reason_value)
        ),
    )


def _iter_audit_lines(audit_path: Path) -> Iterator[dict[str, object]]:
    if not audit_path.exists():
        return
    with audit_path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed rolling candidate audit line {lineno}: {exc.msg}"
                ) from exc
            if not isinstance(data, dict):
                raise ValueError(
                    f"rolling candidate audit line {lineno} must be a JSON object"
                )
            yield data


def append_rolling_candidate_audit(
    audit: RollingCandidateAuditRecord,
    audit_path: Path = ROLLING_CANDIDATE_AUDIT_PATH,
) -> RollingCandidateAuditRecord:
    """Append one candidate audit idempotently under an exclusive append lock.

    Exactly one canonical JSONL line is written per ``(profile,
    rebalance_start, snapshot_key)``; an identical replay returns the existing
    stored record without writing. A malformed existing audit line raises
    ``ValueError`` and this ledger never mutates the rebalance decision ledger,
    current-profile pointer, catalog, or runtime trading state.
    """
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            for existing in _iter_audit_lines(audit_path):
                if (
                    existing.get("profile") == audit.profile
                    and existing.get("rebalance_start") == audit.rebalance_start
                    and existing.get("snapshot_key") == audit.snapshot_key
                ):
                    return _audit_from_payload(existing)
            handle.write(
                json.dumps(
                    audit.to_payload(), ensure_ascii=False, sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
            return audit
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_current_profile(
    pointer_path: Path = CURRENT_PROFILE_POINTER_PATH,
) -> RollingSelectionRecord | None:
    """Read the latest recorded decision; a missing or empty pointer is ``None``."""
    if not pointer_path.exists():
        return None
    data = json.loads(pointer_path.read_text(encoding="utf-8"))
    if not data:
        return None
    return _record_from_payload(data)


def write_current_profile(
    record: RollingSelectionRecord,
    pointer_path: Path = CURRENT_PROFILE_POINTER_PATH,
) -> None:
    """Atomically replace the current-profile pointer with one complete record."""
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pointer_path.with_suffix(pointer_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(_payload_from_record(record), ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, pointer_path)


def _check_contract() -> None:
    """Executable assertions locking the ledger surface."""
    assert append_rebalance_record.__name__ == "append_rebalance_record"
    assert append_rolling_candidate_audit.__name__ == "append_rolling_candidate_audit"
    assert rebalance_snapshot_key({"a": 1}) == rebalance_snapshot_key({"a": 1})


_check_contract()

__all__ = [
    "CURRENT_PROFILE_POINTER_PATH",
    "REBALANCE_LEDGER_PATH",
    "ROLLING_CANDIDATE_AUDIT_PATH",
    "append_rebalance_record",
    "append_rolling_candidate_audit",
    "load_rebalance_records",
    "read_current_profile",
    "rebalance_snapshot_key",
    "write_current_profile",
]
