"""Local SQLite execution registry owning registrations, finalizations and artifacts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from src.backtests.contracts import ArtifactReference, RunFinalization, RunRegistration

_REGISTRY_SCHEMA_VERSION = 1

_DDL_STATEMENTS: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS schema_meta (version INTEGER PRIMARY KEY)",
    """CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        strategy_id TEXT NOT NULL,
        registered_at TEXT NOT NULL,
        request_json TEXT NOT NULL,
        managed_directory TEXT,
        pinned INTEGER NOT NULL DEFAULT 0,
        resolved INTEGER NOT NULL DEFAULT 0,
        deployment_referenced INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS finalizations (
        run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
        finalized_at TEXT NOT NULL,
        status TEXT NOT NULL,
        primary_valid INTEGER,
        terminal_certified INTEGER,
        outcome_json TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS artifacts (
        run_id TEXT NOT NULL REFERENCES runs(run_id),
        role TEXT NOT NULL,
        path TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        byte_count INTEGER NOT NULL,
        managed INTEGER NOT NULL,
        evidence_id TEXT,
        retained INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY (run_id, role, path)
    )""",
    """CREATE TABLE IF NOT EXISTS trials (
        namespace TEXT NOT NULL,
        identity_key TEXT NOT NULL,
        first_seen TEXT NOT NULL,
        provenance_json TEXT NOT NULL,
        PRIMARY KEY (namespace, identity_key)
    )""",
    """CREATE TABLE IF NOT EXISTS history_records (
        source_id TEXT NOT NULL,
        ordinal INTEGER NOT NULL,
        namespace TEXT NOT NULL,
        record_json TEXT NOT NULL,
        admitted INTEGER NOT NULL,
        identity_key TEXT,
        PRIMARY KEY (source_id, ordinal)
    )""",
    """CREATE TABLE IF NOT EXISTS migrations (
        source_id TEXT PRIMARY KEY,
        source_path TEXT NOT NULL,
        source_sha256 TEXT NOT NULL,
        imported_at TEXT NOT NULL
    )""",
)

_INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_finalizations_status ON finalizations(status)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_evidence_id ON artifacts(evidence_id)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_retained ON artifacts(retained)",
    "CREATE INDEX IF NOT EXISTS idx_runs_registered_at ON runs(registered_at)",
)

_REGISTRY_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "runs": frozenset({"run_id", "strategy_id", "registered_at", "request_json", "managed_directory", "pinned", "resolved", "deployment_referenced"}),
    "finalizations": frozenset({"run_id", "finalized_at", "status", "primary_valid", "terminal_certified", "outcome_json"}),
    "artifacts": frozenset({"run_id", "role", "path", "sha256", "byte_count", "managed", "evidence_id", "retained"}),
    "trials": frozenset({"namespace", "identity_key", "first_seen", "provenance_json"}),
    "history_records": frozenset({"source_id", "ordinal", "namespace", "record_json", "admitted", "identity_key"}),
    "migrations": frozenset({"source_id", "source_path", "source_sha256", "imported_at"}),
}


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=5.0, isolation_level="DEFERRED")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    with conn:
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _trial_identity(strategy_id: str, request_json: str) -> tuple[str, str]:
    digest = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
    return strategy_id, digest


def _optional_flag(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def initialize_registry(path: Path) -> None:
    """Create or validate the local execution registry without discarding existing evidence. Args: SQLite file path. Returns: None. Raises: OSError or sqlite3.Error on persistence failure; ValueError for an unsupported schema."""
    if not isinstance(path, Path):
        raise ValueError(f"registry path must be a Path, got {path!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(path)
    try:
        with conn:
            for statement in _DDL_STATEMENTS:
                conn.execute(statement)
            row = conn.execute("SELECT version FROM schema_meta").fetchone()
            if row is None:
                conn.execute("INSERT INTO schema_meta (version) VALUES (?)", (_REGISTRY_SCHEMA_VERSION,))
            elif int(row[0]) != _REGISTRY_SCHEMA_VERSION:
                raise ValueError(f"unsupported registry schema version: {row[0]!r}")
        for table, required in _REGISTRY_REQUIRED_COLUMNS.items():
            actual = frozenset(r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall())  # noqa: S608
            if not required.issubset(actual):
                raise ValueError(f"unsupported registry schema for table {table}: {sorted(actual)}")
        with conn:
            for statement in _INDEX_STATEMENTS:
                conn.execute(statement)
    finally:
        conn.close()


def register_run(path: Path, registration: RunRegistration) -> None:
    """Register one execution before workload launch. Args: registry path and immutable registration. Returns: None. Raises: ValueError for a conflicting identity; sqlite3.Error on persistence failure."""
    request_json = json.dumps(registration.request, sort_keys=True, separators=(",", ":"))
    managed = None if registration.managed_directory is None else str(registration.managed_directory)
    conn = _connect(path)
    try:
        with conn:
            existing = conn.execute(
                "SELECT strategy_id, registered_at, request_json, managed_directory FROM runs WHERE run_id = ?",
                (registration.run_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing[0] != registration.strategy_id
                    or existing[1] != registration.registered_at
                    or existing[2] != request_json
                    or existing[3] != managed
                ):
                    raise ValueError(f"conflicting registration for run_id {registration.run_id!r}")
            else:
                conn.execute(
                    "INSERT INTO runs (run_id, strategy_id, registered_at, request_json, managed_directory, pinned, resolved, deployment_referenced) VALUES (?, ?, ?, ?, ?, 0, 0, 0)",
                    (registration.run_id, registration.strategy_id, registration.registered_at, request_json, managed),
                )
            namespace, identity_key = _trial_identity(registration.strategy_id, request_json)
            conn.execute(
                "INSERT OR IGNORE INTO trials (namespace, identity_key, first_seen, provenance_json) VALUES (?, ?, ?, ?)",
                (namespace, identity_key, registration.registered_at, request_json),
            )
    finally:
        conn.close()


def finalize_run(path: Path, finalization: RunFinalization, artifacts: tuple[ArtifactReference, ...]) -> None:
    """Atomically finalize observed execution and publish its verified artifact references. Financial invalidity is not process failure. Args: registry, final outcome and artifacts. Returns: None. Raises: ValueError for missing or conflicting registration; sqlite3.Error on transaction failure."""
    for artifact in artifacts:
        if artifact.run_id != finalization.run_id:
            raise ValueError(f"artifact run {artifact.run_id!r} does not match finalization {finalization.run_id!r}")
    outcome_json = json.dumps(finalization.outcome, sort_keys=True, separators=(",", ":"))
    conn = _connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            registered = conn.execute("SELECT run_id FROM runs WHERE run_id = ?", (finalization.run_id,)).fetchone()
            if registered is None:
                raise ValueError(f"unknown run_id {finalization.run_id!r}")
            stored = conn.execute(
                "SELECT finalized_at, status, primary_valid, terminal_certified, outcome_json FROM finalizations WHERE run_id = ?",
                (finalization.run_id,),
            ).fetchone()
            if stored is not None:
                if (
                    stored[0] != finalization.finalized_at
                    or stored[1] != finalization.status
                    or stored[2] != _optional_flag(finalization.primary_valid)
                    or stored[3] != _optional_flag(finalization.terminal_certified)
                    or stored[4] != outcome_json
                ):
                    raise ValueError(f"conflicting finalization for run_id {finalization.run_id!r}")
                stored_artifacts = conn.execute(
                    "SELECT role, path, sha256, byte_count, managed, evidence_id FROM artifacts WHERE run_id = ? ORDER BY role, path",
                    (finalization.run_id,),
                ).fetchall()
                expected = sorted(
                    ((a.role, str(a.path), a.sha256, a.byte_count, 1 if a.managed else 0, a.evidence_id) for a in artifacts),
                    key=lambda item: (item[0], item[1]),
                )
                if [tuple(r) for r in stored_artifacts] != expected:
                    raise ValueError(f"conflicting artifacts for run_id {finalization.run_id!r}")
                conn.execute("COMMIT")
                return
            conn.execute(
                "INSERT INTO finalizations (run_id, finalized_at, status, primary_valid, terminal_certified, outcome_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    finalization.run_id,
                    finalization.finalized_at,
                    finalization.status,
                    _optional_flag(finalization.primary_valid),
                    _optional_flag(finalization.terminal_certified),
                    outcome_json,
                ),
            )
            conn.execute(
                "DELETE FROM artifacts WHERE run_id = ? AND role = ?",
                (finalization.run_id, "publication_lease"),
            )
            for artifact in artifacts:
                conn.execute(
                    "INSERT INTO artifacts (run_id, role, path, sha256, byte_count, managed, evidence_id, retained) VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
                    (
                        artifact.run_id,
                        artifact.role,
                        str(artifact.path),
                        artifact.sha256,
                        artifact.byte_count,
                        1 if artifact.managed else 0,
                        artifact.evidence_id,
                    ),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()


def set_run_protection(path: Path, run_id: str, *, pinned: bool, resolved: bool, deployment_referenced: bool) -> None:
    """Record explicit evidence protection without rewriting execution or financial outcomes. Args: registry, run identity and protection flags. Returns: None. Raises: KeyError for an unknown run; sqlite3.Error on persistence failure."""
    if not isinstance(pinned, bool) or not isinstance(resolved, bool) or not isinstance(deployment_referenced, bool):
        raise ValueError("protection flags must be bool")
    conn = _connect(path)
    try:
        with conn:
            cursor = conn.execute(
                "UPDATE runs SET pinned = ?, resolved = ?, deployment_referenced = ? WHERE run_id = ?",
                (1 if pinned else 0, 1 if resolved else 0, 1 if deployment_referenced else 0, run_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"unknown run_id {run_id!r}")
    finally:
        conn.close()


__all__ = ["finalize_run", "initialize_registry", "register_run", "set_run_protection"]
