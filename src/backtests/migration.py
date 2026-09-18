"""Explicit legacy history and execution artifact importer."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from src.backtests.contracts import ArtifactReference, JsonValue, RunFinalization, RunRegistration
from src.common.errors import DataIntegrityError

logger = logging.getLogger("BacktestMigration")

_HISTORY_NAMESPACE = "mhs_legacy_horizon"
_RUN_STRATEGY_ID = "legacy_mhs_backtest"
_ALLOWED_RUN_STATUS = ("completed", "failed", "timed_out", "signaled", "resource_rejected", "interrupted")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    return _sha256_hex(path.read_bytes())


def _hash_history_dir(source: Path) -> str:
    digest = hashlib.sha256()
    for shard in sorted(source.glob("*.jsonl")):
        digest.update(shard.read_bytes())
    ledger = source / "trials_ledger.json"
    if ledger.is_file():
        digest.update(ledger.read_bytes())
    digest.update(str(source.resolve()).encode("utf-8"))
    return digest.hexdigest()


def _hash_run_dir(source: Path) -> str:
    digest = hashlib.sha256()
    for name in ("run.json", "primary.json", "failure.json"):
        candidate = source / name
        if candidate.is_file():
            digest.update(candidate.read_bytes())
    digest.update(str(source.resolve()).encode("utf-8"))
    return digest.hexdigest()


def _source_identity(source: Path, content_sha: str) -> str:
    return _sha256_hex(f"{source.resolve()}|{content_sha}".encode("utf-8"))


def _is_hex32(value: str) -> bool:
    if len(value) != 32:
        return False
    return all(c in "0123456789abcdefABCDEF" for c in value)


def _derive_run_id(basename: str) -> str:
    if _is_hex32(basename):
        return basename.lower()
    return uuid.uuid5(uuid.NAMESPACE_URL, basename).hex


def _map_run_status(raw: Any) -> str:
    text = str(raw).lower() if raw is not None else ""
    if text in ("complete", "completed"):
        return "completed"
    if text in _ALLOWED_RUN_STATUS:
        return text
    return "failed"


def _normalize_gap(gap: Any) -> dict[str, Any]:
    if not isinstance(gap, dict):
        return {"code": "unknown", "symbol": "unknown", "timestamp": "unknown"}
    return {
        "code": gap.get("code", "unknown"),
        "symbol": gap.get("symbol", "unknown"),
        "timestamp": gap.get("timestamp", "unknown"),
        "decision_time": gap.get("decision_time", "unknown"),
        "signal_time": gap.get("signal_time", "unknown"),
        "execution_bound": gap.get("execution_bound", "unknown"),
    }


def _already_migrated(conn: sqlite3.Connection, source_id: str) -> bool:
    row = conn.execute("SELECT source_id FROM migrations WHERE source_id = ?", (source_id,)).fetchone()
    return row is not None


def _record_migration(conn: sqlite3.Connection, source_id: str, source: Path, sha: str) -> None:
    conn.execute(
        "INSERT INTO migrations (source_id, source_path, source_sha256, imported_at) VALUES (?, ?, ?, ?)",
        (source_id, str(source.resolve()), sha, datetime.now(UTC).isoformat()),
    )


def _migrate_history_source(conn: sqlite3.Connection, source: Path, dry_run: bool, diagnostics: list[str]) -> str:
    from src.mhs.run_history import _sparse_identity_key, is_trial_record, trial_identity_key

    sha = _hash_history_dir(source)
    source_id = _source_identity(source, sha)
    if _already_migrated(conn, source_id):
        diagnostics.append(f"{source}: skipped (already migrated)")
        return "skipped"
    shards = sorted(source.glob("*.jsonl"))
    ledger_path = source / "trials_ledger.json"
    ledger_raw: dict[str, str] = {}
    if ledger_path.is_file():
        try:
            loaded = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataIntegrityError(f"corrupt trials ledger: {source}") from exc
        if not isinstance(loaded, dict):
            raise DataIntegrityError(f"corrupt trials ledger: {source}")
        ledger_raw = {str(k): str(v) for k, v in loaded.items()}
    raw_records: list[dict[str, Any]] = []
    for shard in shards:
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except ValueError as exc:
                raise DataIntegrityError(f"corrupt history shard: {shard}") from exc
            if isinstance(parsed, dict):
                raw_records.append(parsed)
    if dry_run:
        diagnostics.append(f"{source}: preview {len(raw_records)} records")
        return "imported"
    with conn:
        for ordinal, record in enumerate(raw_records):
            admitted = is_trial_record(record)
            identity = trial_identity_key(record) if admitted else None
            conn.execute(
                "INSERT INTO history_records (source_id, ordinal, namespace, record_json, admitted, identity_key)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    source_id,
                    ordinal,
                    _HISTORY_NAMESPACE,
                    json.dumps(record, sort_keys=True, ensure_ascii=False),
                    1 if admitted else 0,
                    identity,
                ),
            )
            if admitted and identity is not None:
                first = record.get("run_at") if isinstance(record.get("run_at"), str) else datetime.now(UTC).isoformat()
                conn.execute(
                    "INSERT OR IGNORE INTO trials (namespace, identity_key, first_seen, provenance_json)"
                    " VALUES (?, ?, ?, ?)",
                    (_HISTORY_NAMESPACE, identity, str(first), json.dumps(record, sort_keys=True)[:2000]),
                )
        collapsed: dict[str, str] = {}
        for original, first_seen in ledger_raw.items():
            sparse = _sparse_identity_key(original)
            if sparse in collapsed:
                collapsed[sparse] = min(collapsed[sparse], first_seen)
            else:
                collapsed[sparse] = first_seen
        for sparse_key, first_seen in collapsed.items():
            conn.execute(
                "INSERT OR IGNORE INTO trials (namespace, identity_key, first_seen, provenance_json)"
                " VALUES (?, ?, ?, ?)",
                (_HISTORY_NAMESPACE, sparse_key, first_seen, json.dumps({"source_key": sparse_key})[:2000]),
            )
        _record_migration(conn, source_id, source, sha)
    diagnostics.append(f"{source}: imported {len(raw_records)} records")
    return "imported"


def _migrate_run_source(conn: sqlite3.Connection, source: Path, dry_run: bool, diagnostics: list[str]) -> tuple[str, bool]:
    from src.backtests.registry import finalize_run, register_run, set_run_protection

    sha = _hash_run_dir(source)
    source_id = _source_identity(source, sha)
    if _already_migrated(conn, source_id):
        diagnostics.append(f"{source}: skipped (already migrated)")
        return "skipped", False
    run_path = source / "run.json"
    run_doc = json.loads(run_path.read_text(encoding="utf-8"))
    if not isinstance(run_doc, dict):
        raise DataIntegrityError(f"corrupt run record: {source}")
    primary_doc: dict[str, Any] | None = None
    primary_sha: str | None = None
    primary_path = source / "primary.json"
    if primary_path.is_file():
        primary_sha = _hash_file(primary_path)
        loaded_primary = json.loads(primary_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_primary, dict):
            raise DataIntegrityError(f"corrupt primary evidence: {source}")
        primary_doc = loaded_primary
    base_section = primary_doc.get("base", {}) if isinstance(primary_doc, dict) else {}
    stress_section = primary_doc.get("stress", {}) if isinstance(primary_doc, dict) else {}
    base_terminal = base_section.get("terminal", {}) if isinstance(base_section, dict) else {}
    stress_terminal = stress_section.get("terminal", {}) if isinstance(stress_section, dict) else {}
    if not isinstance(base_terminal, dict):
        base_terminal = {}
    if not isinstance(stress_terminal, dict):
        stress_terminal = {}
    base_valid = base_terminal.get("primary_valid")
    stress_valid = stress_terminal.get("primary_valid")
    if base_valid is True and stress_valid is True:
        primary_valid: bool | None = True
    elif base_valid is False or stress_valid is False:
        primary_valid = False
    else:
        primary_valid = None
    base_cert = base_terminal.get("terminal_certified")
    stress_cert = stress_terminal.get("terminal_certified")
    if base_cert is True and stress_cert is True:
        terminal_certified: bool | None = True
    elif base_cert is False or stress_cert is False:
        terminal_certified = False
    else:
        terminal_certified = None
    raw_gaps: list[Any] = []
    for terminal in (base_terminal, stress_terminal):
        gaps = terminal.get("data_gaps")
        if isinstance(gaps, list):
            raw_gaps.extend(gaps)
    normalized_gaps = [_normalize_gap(gap) for gap in raw_gaps]
    status = _map_run_status(run_doc.get("status"))
    run_id = _derive_run_id(source.name)
    registered_at = datetime.now(UTC).isoformat()
    request: dict[str, JsonValue] = {
        "source_basename": source.name,
        "source_path": str(source.resolve()),
        "source_sha256": sha,
        "start": run_doc.get("start"),
        "end": run_doc.get("end"),
        "data_root": run_doc.get("data_root"),
        "code_identity": None,
    }
    outcome = cast(
        dict[str, JsonValue],
        {
            "source_basename": source.name,
            "source_path": str(source.resolve()),
            "source_sha256": sha,
            "raw_status": run_doc.get("status"),
            "cpu_seconds": run_doc.get("cpu_seconds"),
            "wall_seconds": run_doc.get("wall_seconds"),
            "gnu_max_individual_rss_bytes": run_doc.get("gnu_max_individual_rss_bytes"),
            "sampled_tree_pss_peak_bytes": run_doc.get("sampled_tree_pss_peak_bytes"),
            "sampled_tree_uss_peak_bytes": run_doc.get("sampled_tree_uss_peak_bytes"),
            "process_swap_growth_bytes": run_doc.get("process_swap_growth_bytes"),
            "signal_number": run_doc.get("signal_number"),
            "exit_code": run_doc.get("exit_code"),
            "termination_reason": run_doc.get("termination_reason"),
            "primary_sha256": primary_sha,
            "gap_count": len(normalized_gaps),
            "gaps": normalized_gaps[:50],
            "run_document": run_doc,
        },
    )
    if dry_run:
        diagnostics.append(f"{source}: preview run {run_id}")
        return "imported", status != "completed" or primary_valid is not True
    registration = RunRegistration(
        run_id=run_id,
        strategy_id=_RUN_STRATEGY_ID,
        registered_at=registered_at,
        request=request,
        managed_directory=None,
    )
    registry_path = _registry_path_of(conn)
    register_run(registry_path, registration)
    artifacts: list[ArtifactReference] = []
    for role, candidate in (("run", run_path), ("primary", primary_path), ("failure", source / "failure.json")):
        if candidate.is_file():
            artifacts.append(
                ArtifactReference(
                    run_id=run_id,
                    role=role,
                    path=candidate.resolve(),
                    sha256=_hash_file(candidate),
                    byte_count=candidate.stat().st_size,
                    managed=False,
                    evidence_id=primary_sha if role == "primary" else None,
                )
            )
    finalization = RunFinalization(
        run_id=run_id,
        status=status,  # type: ignore[arg-type]
        finalized_at=datetime.now(UTC).isoformat(),
        primary_valid=primary_valid,
        terminal_certified=terminal_certified,
        outcome=outcome,
    )
    finalize_run(registry_path, finalization, tuple(artifacts))
    resolved = status == "completed" and primary_valid is True and terminal_certified is True
    set_run_protection(registry_path, run_id, pinned=False, resolved=resolved, deployment_referenced=False)
    with conn:
        _record_migration(conn, source_id, source, sha)
    diagnostics.append(f"{source}: imported run {run_id}")
    return "imported", not resolved


def _registry_path_of(conn: sqlite3.Connection) -> Path:
    row = conn.execute("PRAGMA database_list").fetchone()
    return Path(str(row[2]))


def migrate_legacy_backtests(
    *,
    registry_path: Path,
    history_directories: tuple[Path, ...],
    run_directories: tuple[Path, ...],
    dry_run: bool = True,
) -> dict[str, JsonValue]:
    """Import explicitly selected legacy histories and execution artifacts without losing trial identity or failure evidence. Args: target registry, exact source directories and non-mutating preview control. Returns: imported, skipped, protected and rejected source counts with diagnostics. Raises: ValueError for unsafe source selection; DataIntegrityError for inconsistent or corrupt evidence; OSError or sqlite3.Error for persistence failure."""
    from src.backtests.registry import initialize_registry

    if not isinstance(registry_path, Path):
        raise ValueError(f"registry_path must be a Path, got {registry_path!r}")
    for source in (*history_directories, *run_directories):
        if not isinstance(source, Path) or not source.is_dir():
            raise ValueError(f"source must be an existing directory, got {source!r}")
    imported = 0
    skipped = 0
    protected = 0
    rejected = 0
    diagnostics: list[str] = []
    if dry_run and not registry_path.is_file():
        for source in (*history_directories, *run_directories):
            diagnostics.append(f"{source}: preview (no registry change)")
            imported += 1
        return cast(
            dict[str, JsonValue],
            {"imported": imported, "skipped": skipped, "protected": protected, "rejected": rejected, "diagnostics": diagnostics},
        )
    if not dry_run:
        initialize_registry(registry_path)
    conn = sqlite3.connect(str(registry_path), timeout=5.0, isolation_level="DEFERRED")
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        for source in history_directories:
            try:
                outcome = _migrate_history_source(conn, source, dry_run, diagnostics)
            except DataIntegrityError as exc:
                rejected += 1
                diagnostics.append(f"{source}: rejected ({exc})")
                continue
            if outcome == "skipped":
                skipped += 1
            else:
                imported += 1
        for source in run_directories:
            try:
                outcome, is_protected = _migrate_run_source(conn, source, dry_run, diagnostics)
            except DataIntegrityError as exc:
                rejected += 1
                diagnostics.append(f"{source}: rejected ({exc})")
                continue
            if outcome == "skipped":
                skipped += 1
            else:
                imported += 1
                if is_protected:
                    protected += 1
    finally:
        conn.close()
    return cast(
        dict[str, JsonValue],
        {"imported": imported, "skipped": skipped, "protected": protected, "rejected": rejected, "diagnostics": diagnostics},
    )


def verify_legacy_history_migration(*, registry_path: Path, source: Path) -> dict[str, JsonValue]:
    """Verify that a legacy trial-history source is durably represented by the registry.

    Args:
        registry_path: Canonical SQLite registry holding imported trial provenance.
        source: Existing JSONL history directory selected for retirement.
    Returns:
        Counts and semantic trial-set observations for the source and registry.
    Raises:
        DataIntegrityError: The source is unreadable, was not imported, or differs semantically.
    """
    from src.mhs.run_history import _sparse_identity_key, is_trial_record, trial_identity_key

    if not isinstance(source, Path) or not source.is_dir():
        raise DataIntegrityError(f"unreadable history source: {source!r}")
    shards = sorted(source.glob("*.jsonl"))
    try:
        shard_bytes = [shard.read_bytes() for shard in shards]
    except OSError as exc:
        raise DataIntegrityError(f"unreadable history source: {source}") from exc
    ledger_path = source / "trials_ledger.json"
    ledger_raw: dict[str, str] = {}
    if ledger_path.is_file():
        try:
            loaded = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DataIntegrityError(f"corrupt trials ledger: {source}") from exc
        if not isinstance(loaded, dict):
            raise DataIntegrityError(f"corrupt trials ledger: {source}")
        ledger_raw = {str(k): str(v) for k, v in loaded.items()}
    raw_records: list[dict[str, Any]] = []
    for shard, data in zip(shards, shard_bytes, strict=True):
        for line in data.decode("utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except ValueError as exc:
                raise DataIntegrityError(f"corrupt history shard: {shard}") from exc
            if isinstance(parsed, dict):
                raw_records.append(parsed)
    sha = _hash_history_dir(source)
    source_id = _source_identity(source, sha)
    if not isinstance(registry_path, Path) or not registry_path.is_file():
        raise DataIntegrityError(f"history source was not imported: {source}")
    conn = sqlite3.connect(str(registry_path), timeout=5.0)
    try:
        try:
            row = conn.execute("SELECT source_id FROM migrations WHERE source_id = ?", (source_id,)).fetchone()
            registry_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM history_records WHERE source_id = ?", (source_id,)
                ).fetchone()[0]
            )
            trial_rows = conn.execute(
                "SELECT identity_key, first_seen FROM trials WHERE namespace = ?", (_HISTORY_NAMESPACE,)
            ).fetchall()
        except sqlite3.Error as exc:
            raise DataIntegrityError(f"history source was not imported: {source}") from exc
        if row is None:
            raise DataIntegrityError(f"history source changed or was not imported: {source}")
    finally:
        conn.close()
    if registry_count != len(raw_records):
        raise DataIntegrityError(f"history record count differs for source: {source}")
    expected: dict[str, str] = {}
    for record in raw_records:
        if not is_trial_record(record):
            continue
        identity = trial_identity_key(record)
        if identity is None:
            continue
        sparse = _sparse_identity_key(identity)
        first = record.get("run_at")
        if isinstance(first, str):
            expected[sparse] = min(expected[sparse], first) if sparse in expected and expected[sparse] else first
        else:
            expected.setdefault(sparse, "")
    for original, first_seen in ledger_raw.items():
        sparse = _sparse_identity_key(original)
        if sparse in expected and expected[sparse]:
            expected[sparse] = min(expected[sparse], str(first_seen))
        elif sparse not in expected:
            expected[sparse] = str(first_seen)
    registry_map: dict[str, str] = {}
    for identity_key, first_seen in trial_rows:
        sparse = _sparse_identity_key(str(identity_key))
        seen = str(first_seen)
        registry_map[sparse] = min(registry_map[sparse], seen) if sparse in registry_map else seen
    for sparse, first_seen in expected.items():
        actual = registry_map.get(sparse)
        if actual is None:
            raise DataIntegrityError(f"history trial identity missing for source: {source}")
        if first_seen and actual != first_seen:
            raise DataIntegrityError(f"history trial provenance differs for source: {source}")
    return cast(
        dict[str, JsonValue],
        {
            "source": str(source),
            "registry": str(registry_path),
            "source_records": len(raw_records),
            "registry_records": registry_count,
            "distinct_trial_identities": len(expected),
            "verified": True,
        },
    )


__all__ = ["migrate_legacy_backtests", "verify_legacy_history_migration"]
