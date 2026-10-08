"""Equivalent-run reuse admission for supervised MHS process evaluations."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import sqlite3
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from src.common.errors import DataIntegrityError

# Rejection warnings keep the supervisor's logger name so operational log routing is unchanged.
_logger = logging.getLogger("src.lab.mhs.app.supervisor")

ReuseRejectionReason = Literal[
    "not_finalized",
    "status_not_completed",
    "primary_not_valid",
    "terminal_not_certified",
    "unmanaged_run",
    "result_unrecorded",
    "evidence_unrecorded",
    "artifact_missing",
    "artifact_size_mismatch",
    "artifact_digest_mismatch",
    "artifact_unreadable",
]
"""Closed set of reasons a fingerprint-matching run is not admissible for reuse."""


@dataclass(frozen=True, slots=True)
class ReuseRejection:
    """One fingerprint-matching run that was examined and refused for reuse.

    Attributes:
        run_id: Registry identity of the refused run.
        reason: First failed admissibility check, in contract order.
        detail: Short machine-readable context (stored status, offending
            artifact ``role:path``, or observed vs recorded size); None when
            the reason is self-describing.
    """

    run_id: str
    reason: ReuseRejectionReason
    detail: str | None


@dataclass(frozen=True, slots=True)
class ReusedRun:
    """Verified prior execution whose recorded evidence can stand in for a fresh run.

    Attributes:
        run_id: Registry identity of the admitted run.
        finalized_at: Stored UTC ISO8601 finalization timestamp.
        result_path: Absolute path of the verified result envelope.
        targets_path: Absolute path of the verified exact-target parquet, or
            None when the run recorded no ``targets`` artifact.
        evidence_retained: False when retention reclaimed any of the run's
            managed detail rows (``retained = 0``); the compact envelope stays
            authoritative either way.
    """

    run_id: str
    finalized_at: str
    result_path: Path
    targets_path: Path | None
    evidence_retained: bool


@dataclass(frozen=True, slots=True)
class ReuseLookup:
    """Outcome of one equivalent-run lookup.

    Attributes:
        reused: The admitted run, or None when a fresh execution is required.
        rejections: Every examined, refused candidate in examination order.
    """

    reused: ReusedRun | None
    rejections: tuple[ReuseRejection, ...]


_RUN_STATUS_LITERALS: frozenset[str] = frozenset(
    {"completed", "failed", "timed_out", "signaled", "resource_rejected", "interrupted"}
)
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_RUNS_COLUMNS = frozenset({"run_id", "request_json", "managed_directory"})
_FINALIZATION_COLUMNS = frozenset(
    {"run_id", "finalized_at", "status", "primary_valid", "terminal_certified", "outcome_json"}
)
_ARTIFACT_COLUMNS = frozenset(
    {"run_id", "role", "path", "sha256", "byte_count", "managed", "evidence_id", "retained"}
)

_Row = tuple[object, ...]


@dataclass(frozen=True, slots=True)
class _RegistrySnapshot:
    candidates_managed: dict[str, str | None]
    candidate_ids: list[str]
    final_rows: list[_Row]
    artifact_rows: list[_Row]


def _hash_file(path: Path) -> tuple[str, int]:
    """Stream content identity and size without parsing evidence payloads."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _parse_reuse_finalized_at(value: object, run_id: str, registry_path: Path) -> datetime.datetime:
    if not isinstance(value, str) or not value:
        raise DataIntegrityError(f"corrupt finalization finalized_at for run {run_id} in {registry_path}")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError as exc:
        raise DataIntegrityError(
            f"corrupt finalization finalized_at for run {run_id} in {registry_path}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != datetime.timedelta(0):
        raise DataIntegrityError(
            f"corrupt finalization finalized_at for run {run_id} in {registry_path}"
        )
    return parsed


def _reuse_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}  # noqa: S608


def _require_reuse_schema(conn: sqlite3.Connection, tables: set[str], registry_path: Path) -> None:
    for required in ("finalizations", "artifacts"):
        if required not in tables:
            raise DataIntegrityError(f"registry is missing table {required}: {registry_path}")
    runs_cols = _reuse_table_columns(conn, "runs")
    finals_cols = _reuse_table_columns(conn, "finalizations")
    artifacts_cols = _reuse_table_columns(conn, "artifacts")
    if not _RUNS_COLUMNS.issubset(runs_cols):
        raise DataIntegrityError(f"registry runs table has missing columns: {registry_path}")
    if not _FINALIZATION_COLUMNS.issubset(finals_cols):
        raise DataIntegrityError(f"registry finalizations table has missing columns: {registry_path}")
    if not _ARTIFACT_COLUMNS.issubset(artifacts_cols):
        raise DataIntegrityError(f"registry artifacts table has missing columns: {registry_path}")


def _fingerprint_candidates(
    conn: sqlite3.Connection, fingerprint: str, registry_path: Path
) -> dict[str, str | None]:
    run_rows = conn.execute("SELECT run_id, request_json, managed_directory FROM runs").fetchall()
    decoded: list[tuple[str, dict[str, object], str | None]] = []
    for raw_run_id, raw_request, managed in run_rows:
        run_id = str(raw_run_id)
        try:
            parsed_request = json.loads(str(raw_request))
        except (ValueError, TypeError) as exc:
            raise DataIntegrityError(f"corrupt run request {run_id} in {registry_path}") from exc
        if not isinstance(parsed_request, dict):
            raise DataIntegrityError(f"run request {run_id} is not a JSON object in {registry_path}")
        decoded.append((run_id, parsed_request, None if managed is None else str(managed)))
    candidates_managed: dict[str, str | None] = {}
    for run_id, request, managed in decoded:
        value = request.get("fingerprint")
        if isinstance(value, str) and value == fingerprint:
            candidates_managed[run_id] = managed
    return candidates_managed


def _read_snapshot_rows(
    conn: sqlite3.Connection, fingerprint: str, registry_path: Path
) -> _RegistrySnapshot | None:
    conn.execute("BEGIN")
    tables = {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    if "runs" not in tables:
        conn.execute("ROLLBACK")
        return None
    _require_reuse_schema(conn, tables, registry_path)
    candidates_managed = _fingerprint_candidates(conn, fingerprint, registry_path)
    if not candidates_managed:
        conn.execute("ROLLBACK")
        return None
    candidate_ids = sorted(candidates_managed)
    placeholders = ",".join("?" for _ in candidate_ids)
    final_rows = conn.execute(
        f"SELECT run_id, finalized_at, status, primary_valid, terminal_certified, outcome_json"  # noqa: S608
        f" FROM finalizations WHERE run_id IN ({placeholders})",
        candidate_ids,
    ).fetchall()
    artifact_rows = conn.execute(
        f"SELECT run_id, role, path, sha256, byte_count, managed, evidence_id, retained"  # noqa: S608
        f" FROM artifacts WHERE run_id IN ({placeholders})",
        candidate_ids,
    ).fetchall()
    conn.execute("ROLLBACK")
    return _RegistrySnapshot(candidates_managed, candidate_ids, final_rows, artifact_rows)


def _read_registry_snapshot(registry_path: Path, fingerprint: str) -> _RegistrySnapshot | None:
    try:
        conn = sqlite3.connect(registry_path.absolute().as_uri() + "?mode=ro", uri=True, timeout=5.0)
    except sqlite3.OperationalError:
        raise
    except (sqlite3.DatabaseError, OSError) as exc:
        raise DataIntegrityError(f"registry is unreadable: {registry_path}") from exc
    try:
        try:
            return _read_snapshot_rows(conn, fingerprint, registry_path)
        except sqlite3.OperationalError:
            raise
        except DataIntegrityError:
            with suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise
        except sqlite3.DatabaseError as exc:
            with suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise DataIntegrityError(f"registry is unreadable: {registry_path}") from exc
    finally:
        conn.close()


def _validate_finalization_row(run_id: str, row: _Row, registry_path: Path) -> None:
    _, finalized_at, status, primary_valid, terminal_certified, outcome_json = row
    if str(status) not in _RUN_STATUS_LITERALS:
        raise DataIntegrityError(f"corrupt finalization status for run {run_id} in {registry_path}")
    if primary_valid not in (0, 1, None):
        raise DataIntegrityError(f"corrupt finalization primary_valid for run {run_id} in {registry_path}")
    if terminal_certified not in (0, 1, None):
        raise DataIntegrityError(
            f"corrupt finalization terminal_certified for run {run_id} in {registry_path}"
        )
    _parse_reuse_finalized_at(finalized_at, run_id, registry_path)
    try:
        outcome = json.loads(str(outcome_json))
    except (ValueError, TypeError) as exc:
        raise DataIntegrityError(f"corrupt finalization outcome for run {run_id} in {registry_path}") from exc
    if not isinstance(outcome, dict):
        raise DataIntegrityError(
            f"finalization outcome for run {run_id} is not a JSON object in {registry_path}"
        )
    evidence_value = outcome.get("evidence_id", None)
    if evidence_value is not None and not (isinstance(evidence_value, str) and evidence_value != ""):
        raise DataIntegrityError(f"corrupt finalization evidence_id for run {run_id} in {registry_path}")


def _validate_artifact_rows(run_id: str, rows: list[_Row], registry_path: Path) -> None:
    for row in rows:
        _, _role, path, sha256, byte_count, _managed, _evidence_id, retained = row
        if not isinstance(path, str) or not Path(path).is_absolute() or "\x00" in path:
            raise DataIntegrityError(f"corrupt artifact path for run {run_id} in {registry_path}")
        if not isinstance(sha256, str) or _SHA256_RE.match(sha256) is None:
            raise DataIntegrityError(f"corrupt artifact sha256 for run {run_id} in {registry_path}")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise DataIntegrityError(f"corrupt artifact byte_count for run {run_id} in {registry_path}")
        if retained not in (0, 1):
            raise DataIntegrityError(f"corrupt artifact retained flag for run {run_id} in {registry_path}")
    result_count = sum(1 for row in rows if str(row[1]) == "result")
    if result_count > 1:
        raise DataIntegrityError(f"duplicate result artifacts for run {run_id} in {registry_path}")


def _reject(run_id: str, reason: ReuseRejectionReason, detail: str | None) -> ReuseRejection:
    rejection = ReuseRejection(run_id=run_id, reason=reason, detail=detail)
    _logger.warning("[DATA] reuse_rejected run_id=%s reason=%s detail=%s", run_id, reason, detail)
    return rejection


def _finalization_rejection(run_id: str, final_row: _Row, managed: str | None) -> ReuseRejection | None:
    _, _finalized_at, status, primary_valid, terminal_certified, _outcome_json = final_row
    if str(status) != "completed":
        return _reject(run_id, "status_not_completed", str(status))
    if primary_valid != 1:
        return _reject(run_id, "primary_not_valid", None)
    if terminal_certified != 1:
        return _reject(run_id, "terminal_not_certified", None)
    if managed is None:
        return _reject(run_id, "unmanaged_run", None)
    return None


def _recorded_evidence_rejection(
    run_id: str, rows: list[_Row], evidence_id: str | None
) -> ReuseRejection | None:
    result_rows = [row for row in rows if str(row[1]) == "result"]
    if not result_rows:
        return _reject(run_id, "result_unrecorded", "no_result_row")
    if int(cast(int, result_rows[0][7])) != 1:
        return _reject(run_id, "result_unrecorded", "result_retained=0")
    if evidence_id is not None and not any(
        (None if row[6] is None else str(row[6])) == evidence_id for row in rows
    ):
        return _reject(run_id, "evidence_unrecorded", evidence_id)
    return None


def _artifact_rejection(run_id: str, row: _Row) -> ReuseRejection | None:
    path_text = str(row[2])
    recorded_sha = str(row[3]).lower()
    recorded_size = int(cast(int, row[4]))
    label = f"{row[1]!s}:{path_text}"
    try:
        file_stat = os.stat(path_text)
    except (FileNotFoundError, NotADirectoryError):
        return _reject(run_id, "artifact_missing", label)
    except OSError:
        return _reject(run_id, "artifact_unreadable", label)
    if not stat.S_ISREG(file_stat.st_mode):
        return _reject(run_id, "artifact_missing", label)
    if int(file_stat.st_size) != recorded_size:
        return _reject(
            run_id,
            "artifact_size_mismatch",
            f"{label} observed={int(file_stat.st_size)} recorded={recorded_size}",
        )
    try:
        digest, _ = _hash_file(Path(path_text))
    except OSError:
        return _reject(run_id, "artifact_unreadable", label)
    if digest.lower() != recorded_sha:
        return _reject(run_id, "artifact_digest_mismatch", label)
    return None


def _retained_artifacts_rejection(run_id: str, rows: list[_Row]) -> ReuseRejection | None:
    retained_rows = [row for row in rows if int(cast(int, row[7])) == 1]
    retained_rows.sort(key=lambda row: (str(row[1]), str(row[2])))
    retained_rows.sort(key=lambda row: 0 if str(row[1]) == "result" else 1)
    for row in retained_rows:
        failure = _artifact_rejection(run_id, row)
        if failure is not None:
            return failure
    return None


def _admit_candidate(
    run_id: str, final_row: _Row, managed: str | None, rows: list[_Row]
) -> ReusedRun | ReuseRejection:
    outcome = json.loads(str(final_row[5]))
    evidence_value = outcome.get("evidence_id", None)
    evidence_id = str(evidence_value) if isinstance(evidence_value, str) and evidence_value != "" else None
    rejection = (
        _finalization_rejection(run_id, final_row, managed)
        or _recorded_evidence_rejection(run_id, rows, evidence_id)
        or _retained_artifacts_rejection(run_id, rows)
    )
    if rejection is not None:
        return rejection
    result_row = next(row for row in rows if str(row[1]) == "result")
    targets_rows = [row for row in rows if str(row[1]) == "targets" and int(cast(int, row[7])) == 1]
    targets_rows.sort(key=lambda row: str(row[2]))
    return ReusedRun(
        run_id=run_id,
        finalized_at=str(final_row[1]),
        result_path=Path(str(result_row[2])),
        targets_path=Path(str(targets_rows[0][2])) if targets_rows else None,
        evidence_retained=all(int(cast(int, row[7])) == 1 for row in rows),
    )


def find_reused_run(registry_path: Path, fingerprint: str) -> ReuseLookup:
    """Find the newest finalized run whose evidence can replace a fresh execution.

    A fingerprint match only proves that the request was equivalent. It does
    not prove that the earlier execution succeeded or that its evidence still
    exists. Reuse is therefore admitted only for a run that completed with
    certified financial validity and whose recorded artifacts still exist
    byte-for-byte. Otherwise a failed, interrupted, financially invalid,
    deleted or tampered run could stand in silently as the answer to the
    request. Refusals are expected lifecycle states of a prunable local
    registry, so they become a fresh run, never an error. Corruption of the
    registry ledger itself fails closed, because absent evidence and
    unreadable evidence must never look the same.

    Admissibility, checked in this order (the first failure is the reason):
    a finalization row exists; ``status == "completed"``;
    ``primary_valid == 1``; ``terminal_certified == 1``;
    ``managed_directory`` is not NULL; exactly one ``result`` row exists and
    has ``retained = 1``; when the outcome records an ``evidence_id``, at
    least one artifact row carries it; every ``retained = 1`` artifact row,
    result first, is a regular file whose size equals ``byte_count`` and
    whose streamed SHA-256 equals ``sha256``. Rows with ``retained = 0``
    were reclaimed by retention and are not verified on disk.

    Args:
        registry_path: Execution registry SQLite file.
        fingerprint: ``request_fingerprint`` of the request being served.
    Returns:
        The newest admissible run by ``finalized_at`` (ties: larger ``run_id``),
        plus a rejection for every candidate examined before it and for every
        unfinalized candidate. Returns an empty lookup when the registry file is
        absent or has no ``runs`` table.
    Side effects:
        Writes nothing to the registry or the filesystem. Emits one
        ``WARNING`` ``[DATA]`` line per rejection.
    Raises:
        DataIntegrityError: The path exists but is not a regular file or not a
            readable SQLite database; ``runs`` exists while ``finalizations`` or
            ``artifacts`` (or a required column) is missing; any
            ``request_json`` is not a JSON object; a candidate's finalization or
            artifact rows are outside their contract domains; or a candidate has
            more than one ``result`` row.
        sqlite3.OperationalError: Transient lock or busy failures, unchanged.
    """
    if not os.path.lexists(registry_path):
        return ReuseLookup(None, ())
    if not registry_path.is_file():
        raise DataIntegrityError(f"registry path is not a regular file: {registry_path}")
    snapshot = _read_registry_snapshot(registry_path, fingerprint)
    if snapshot is None:
        return ReuseLookup(None, ())
    candidate_ids = snapshot.candidate_ids
    finals_by_run: dict[str, _Row] = {str(row[0]): row for row in snapshot.final_rows}
    artifacts_by_run: dict[str, list[_Row]] = {run_id: [] for run_id in candidate_ids}
    for row in snapshot.artifact_rows:
        artifacts_by_run.setdefault(str(row[0]), []).append(row)

    for run_id, row in finals_by_run.items():
        _validate_finalization_row(run_id, row, registry_path)
    for run_id in candidate_ids:
        _validate_artifact_rows(run_id, artifacts_by_run.get(run_id, []), registry_path)

    rejections: list[ReuseRejection] = []
    unfinalized = sorted(run_id for run_id in candidate_ids if run_id not in finals_by_run)
    for run_id in unfinalized:
        rejections.append(_reject(run_id, "not_finalized", None))  # noqa: PERF401 -- logging side effect per rejection
    finalized_ids = [run_id for run_id in candidate_ids if run_id in finals_by_run]
    ordered = sorted(
        finalized_ids,
        key=lambda rid: (_parse_reuse_finalized_at(finals_by_run[rid][1], rid, registry_path), rid),
        reverse=True,
    )
    for run_id in ordered:
        verdict = _admit_candidate(
            run_id,
            finals_by_run[run_id],
            snapshot.candidates_managed[run_id],
            artifacts_by_run.get(run_id, []),
        )
        if isinstance(verdict, ReusedRun):
            return ReuseLookup(verdict, tuple(rejections))
        rejections.append(verdict)
    return ReuseLookup(None, tuple(rejections))
