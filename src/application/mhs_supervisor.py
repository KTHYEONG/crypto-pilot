"""Supervised source-owned three-minute process evaluation.

The supervisor launches the source-owned worker module in a dedicated process
group, waits for actual termination, samples the workload tree, and persists
an atomic outcome report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from src.backtests.contracts import (
    ArtifactReference,
    JsonValue,
    RetentionPolicy,
    RunFinalization,
    RunRegistration,
    utc_now_iso8601,
)
from src.backtests.registry import finalize_run, initialize_registry, register_run
from src.backtests.retention import apply_retention, plan_retention
from src.common.paths import BACKTESTS_DIR, FUTURES_DATA_DIR
from src.mhs.process import ProcessExecutionPolicy
from src.mhs.reporting.inventory import PROCESS_INVENTORY_CERTIFICATION_LEVEL
from src.mhs.resources import (
    MhsMemoryBudget,
    current_mhs_headroom_bytes,
    resolve_mhs_memory_budget,
)

_logger = logging.getLogger(__name__)

GRACE_SECONDS: float = 5.0
HEARTBEAT_SECONDS: float = 30.0
CPU_SCOPE: str = "waited workload user+system CPU seconds via GNU time when available"
MEMORY_SCOPE: str = (
    "sampled child-tree PSS/USS bytes at poll interval; "
    "GNU max RSS is maximum individual-process bytes"
)
_RESOURCE_ERROR_CODES: frozenset[str] = frozenset(
    {"MEMORY_BUDGET", "MEMORY_RESERVE", "SWAP_GROWTH", "RESOURCE_TELEMETRY"}
)
_GNU_LINE_RE = re.compile(
    r"MHS_GNU_TIME elapsed=([0-9.eE+-]+) user=([0-9.eE+-]+) sys=([0-9.eE+-]+) maxrss=([0-9]+)"
)


@dataclass(frozen=True, slots=True)
class MhsSupervisedRun:
    """Observed completed subprocess outcome and resource scope. GNU maximum RSS is a maximum individual-process metric; tree PSS/USS are sampled sums, not instantaneous guarantees. Unknown observations stay null. A signal or failed exit is not operating-system OOM evidence by itself."""

    status: Literal["completed", "failed", "timed_out", "signaled", "resource_rejected", "interrupted"]
    command: tuple[str, ...]
    exit_code: int | None
    signal_number: int | None
    start: str
    end: str
    data_root: str | None
    result_output_path: str
    log_path: str
    domain_artifact_written: bool
    wall_seconds: float
    cpu_seconds: float | None
    gnu_max_individual_rss_bytes: int | None
    sampled_tree_pss_peak_bytes: int | None
    sampled_tree_uss_peak_bytes: int | None
    min_available_bytes: int | None
    process_swap_growth_bytes: int | None
    samples_taken: int
    sample_interval_seconds: float
    cpu_scope: str
    memory_scope: str
    termination_reason: str | None
    memory_budget: MhsMemoryBudget
    run_id: str


def _resolve_run_id(run_id: str | None) -> str:
    """Use the given execution identity or mint a fresh UUID hex, rejecting path-bearing values."""
    candidate = run_id if run_id is not None else uuid.uuid4().hex
    if len(candidate) != 32 or any(c not in "0123456789abcdefABCDEF" for c in candidate):
        raise ValueError(f"run_id must be UUID hex, got {run_id!r}")
    return candidate


_DATA_MANIFEST_DIRNAME = "mhs_execution"
_DATA_MANIFEST_NAME = "input_manifest.json"


def _hash_python_sources(source_root: Path, extra_files: tuple[Path, ...]) -> str | None:
    """Content identity over result-affecting Python sources.

    Hashes every ``*.py`` file under ``source_root`` (sorted by path, each
    entry contributing its anchor-relative POSIX path plus raw bytes) followed
    by ``extra_files`` in order. Any unreadable file fails closed to ``None``
    (provenance unavailable) instead of silently hashing a partial tree.
    """
    try:
        anchor = source_root.resolve()
        members = sorted(
            (path for path in source_root.rglob("*.py") if path.is_file()),
            key=lambda path: path.as_posix(),
        )
        members.extend(extra_files)
        digest = hashlib.sha256()
        for path in members:
            try:
                relative = path.resolve().relative_to(anchor).as_posix()
            except ValueError:
                relative = path.as_posix()
            digest.update(relative.encode("utf-8") + b"\x00")
            digest.update(path.read_bytes() + b"\x00")
    except OSError:
        return None
    return digest.hexdigest()


def _code_identity() -> str | None:
    """Content identity over the MHS strategy source tree.

    Covers every Python module under ``src/mhs`` plus the supervised worker
    entry point: any byte change in result-affecting strategy code yields a
    new identity. ``None`` when provenance is unavailable.
    """
    src_root = Path(__file__).resolve().parent.parent
    return _hash_python_sources(src_root / "mhs", (src_root / "application" / "mhs_worker.py",))


def _verified_manifest_digest(root: Path, observed: dict[str, tuple[int, int]]) -> str | None:
    """Sealed byte-level identity for the input corpus, when freshly attested.

    Trusts ``<root>/mhs_execution/input_manifest.json`` only when it attests
    exactly the parquet files present (same relative-path set) with matching
    size and mtime metadata; any addition, removal, rewrite, or unsealed
    corpus falls through to the metadata snapshot. Returns the lowercase
    sealed digest, else ``None``.
    """
    manifest = root / _DATA_MANIFEST_DIRNAME / _DATA_MANIFEST_NAME
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        attested = {
            str(entry["relative_path"]): (int(entry["size_bytes"]), int(entry["mtime_ns"]))
            for entry in raw["files"]
            if isinstance(entry, dict)
            and isinstance(entry.get("relative_path"), str)
            and isinstance(entry.get("size_bytes"), int)
            and isinstance(entry.get("mtime_ns"), int)
        }
        digest = str(raw["digest"])
        complete = len(attested) == len(raw["files"]) and set(attested) == set(observed)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
    if not complete:
        return None
    if any(attested[rel] != observed[rel] for rel in attested):
        return None
    if not re.fullmatch(r"[0-9a-f]{64}", digest.lower()):
        return None
    return digest.lower()


def _snapshot_data_tree(root: Path) -> str:
    """Content identity over the sealed-or-snapshotted input data corpus.

    A freshly attested sealed manifest yields its byte-level digest (sealed at
    collection time over file bytes). Otherwise the identity is a snapshot
    digest over sorted ``(relative path, size, mtime_ns)`` entries for every
    parquet file under ``root``: any rewrite, restatement, addition, or
    removal changes it without rehashing the multi-gigabyte corpus per run.
    File mtimes are input-content provenance, never wall-clock run inputs.
    Missing roots hash deterministically as absent.
    """
    try:
        if not root.is_dir():
            return "absent"
        observed: dict[str, tuple[int, int]] = {}
        for path in root.rglob("*.parquet"):
            if not path.is_file():
                continue
            stat = path.stat()
            observed[path.relative_to(root).as_posix()] = (int(stat.st_size), int(stat.st_mtime_ns))
        sealed = _verified_manifest_digest(root, observed)
        if sealed is not None:
            return f"sealed:{sealed}"
        canonical = json.dumps(
            [{"path": rel, "size": size, "mtime_ns": mtime} for rel, (size, mtime) in sorted(observed.items())],
            separators=(",", ":"),
        ).encode("utf-8")
        return "snapshot:" + hashlib.sha256(canonical).hexdigest()
    except OSError:
        return "unreadable"


def _data_identity(data_root: str | None) -> str:
    """Resolve the input-data identity root and snapshot it."""
    root = Path(data_root).resolve() if data_root is not None else FUTURES_DATA_DIR.resolve()
    return _snapshot_data_tree(root)


def _registration_request(
    *, start: pd.Timestamp, end: pd.Timestamp, data_root: str | None,
    tracking_error_threshold: float | None, timeout_seconds: float | None, poll_seconds: float,
    memory_budget: MhsMemoryBudget | None = None, execution_timeframe: str = "3m",
) -> dict[str, JsonValue]:
    """Record lifecycle request provenance without fabricating input content hashes."""
    budget = resolve_mhs_memory_budget(memory_budget)
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "strategy_id": PROCESS_INVENTORY_CERTIFICATION_LEVEL,
        "execution_timeframe": execution_timeframe,
        "tracking_error_threshold": tracking_error_threshold,
        "timeout_seconds": timeout_seconds,
        "poll_seconds": float(poll_seconds),
        "data_root": data_root,
        "code_identity": _code_identity(),
        "memory_budget": {
            "total_tree_pss_bytes": budget.total_tree_pss_bytes,
            "replay_tree_pss_bytes": budget.replay_tree_pss_bytes,
            "min_available_bytes": budget.min_available_bytes,
        },
        "fingerprint": request_fingerprint(
            start=start, end=end, data_root=data_root,
            tracking_error_threshold=tracking_error_threshold,
            memory_budget=budget, execution_timeframe=execution_timeframe,
        ),
    }


def request_fingerprint(
    *, start: pd.Timestamp, end: pd.Timestamp, data_root: str | None,
    tracking_error_threshold: float | None, memory_budget: MhsMemoryBudget | None = None,
    execution_timeframe: str = "3m",
) -> str:
    """Immutable request fingerprint for equivalent-run reuse.

    Inputs are strategy identity, UTC interval, execution timeframe, strategy
    controls, resource controls, data-root identity, input-data content
    identity (sealed manifest digest or file snapshot), and source-tree
    content identity. Output paths, timestamps, and telemetry never enter
    the digest.
    """
    budget = resolve_mhs_memory_budget(memory_budget)
    canonical = {
        "strategy_id": PROCESS_INVENTORY_CERTIFICATION_LEVEL,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "execution_timeframe": execution_timeframe,
        "tracking_error_threshold": tracking_error_threshold,
        "data_root": data_root,
        "data_identity": _data_identity(data_root),
        "code_identity": _code_identity(),
        "memory_budget": {
            "total_tree_pss_bytes": budget.total_tree_pss_bytes,
            "replay_tree_pss_bytes": budget.replay_tree_pss_bytes,
            "min_available_bytes": budget.min_available_bytes,
        },
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def find_reused_run(registry_path: Path, fingerprint: str) -> tuple[str, str | None] | None:
    """Return the existing finalized run identity matching one fingerprint, if any."""
    if not registry_path.is_file():
        return None
    conn = sqlite3.connect(str(registry_path), timeout=5.0)
    try:
        try:
            rows = conn.execute("SELECT run_id, request_json FROM runs").fetchall()
            finalized = {r[0]: r[1:] for r in conn.execute("SELECT run_id, status, primary_valid FROM finalizations").fetchall()}
        except sqlite3.Error:
            return None
    finally:
        conn.close()
    for run_id, request_json in rows:
        try:
            request = json.loads(str(request_json))
        except ValueError:
            continue
        if not isinstance(request, dict) or request.get("fingerprint") != fingerprint:
            continue
        if str(run_id) not in finalized:
            continue
        status_row = finalized[str(run_id)]
        validity = None if status_row[1] is None else bool(status_row[1])
        return str(run_id), str(status_row[0]) if validity is None else f"{status_row[0]} valid={validity}"
    return None


def _read_compact_flags(primary: Path) -> tuple[bool | None, bool | None, dict[str, bool | None], str | None]:
    """Combine tier financial flags from a compact summary, leaving missing evidence null."""
    try:
        payload = json.loads(primary.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None, {}, None
    if not isinstance(payload, dict):
        return None, None, {}, None
    raw_identity = payload.get("evidence_id")
    evidence_id = (
        raw_identity
        if isinstance(raw_identity, str)
        and len(raw_identity) == 64
        and all(c in "0123456789abcdefABCDEF" for c in raw_identity)
        else None
    )
    tier_flags: dict[str, bool | None] = {}
    for tier in ("base", "stress"):
        node = payload.get(tier)
        terminal = node.get("terminal") if isinstance(node, dict) else None
        for flag in ("primary_valid", "terminal_certified"):
            raw = terminal.get(flag) if isinstance(terminal, dict) else None
            tier_flags[f"{tier}_{flag}"] = raw if isinstance(raw, bool) else None
    combined: dict[str, bool | None] = {}
    for flag in ("primary_valid", "terminal_certified"):
        values = (tier_flags[f"base_{flag}"], tier_flags[f"stress_{flag}"])
        if values[0] is True and values[1] is True:
            combined[flag] = True
        elif values[0] is False or values[1] is False:
            combined[flag] = False
        else:
            combined[flag] = None
    return combined["primary_valid"], combined["terminal_certified"], tier_flags, evidence_id


def _hash_file(path: Path) -> tuple[str, int]:
    """Stream content identity and size without parsing evidence payloads."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _local_artifact(run_id: str, role: str, raw_path: str | None, evidence_root: Path) -> ArtifactReference | None:
    """Reference one owned-or-sibling lifecycle file, skipping paths the workload never wrote."""
    if raw_path is None:
        return None
    path = Path(raw_path).absolute()
    if not path.is_file():
        return None
    sha256, size = _hash_file(path)
    return ArtifactReference(
        run_id=run_id,
        role=role,
        path=path,
        sha256=sha256,
        byte_count=size,
        managed=path.is_relative_to(evidence_root),
        evidence_id=None,
    )


def _detail_artifacts(run_id: str, evidence_root: Path, evidence_id: str) -> list[ArtifactReference]:
    """Reference verified managed bundle files, releasing nothing when the manifest is unavailable."""
    try:
        manifest = json.loads((evidence_root / evidence_id / "manifest.json").read_text(encoding="utf-8"))
        entries = manifest["files"]
        refs = [
            ArtifactReference(
                run_id=run_id,
                role=str(entry["role"]),
                path=evidence_root / evidence_id / str(entry["name"]),
                sha256=str(entry["sha256"]),
                byte_count=int(entry["size"]),
                managed=True,
                evidence_id=evidence_id,
            )
            for entry in entries
        ]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    return refs


ENVELOPE_SCHEMA_VERSION: int = 1
_BOUNDED_LOG_MAX_BYTES: int = 65536


def _read_domain_payload(staging: Path) -> dict[str, JsonValue] | None:
    """Read the worker staging domain object, if present and well-formed."""
    try:
        payload = json.loads(staging.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _build_result_envelope(
    *, run_id: str, start: pd.Timestamp, end: pd.Timestamp, data_root: str | None,
    fingerprint: str, command: tuple[str, ...], run: MhsSupervisedRun,
    domain: dict[str, JsonValue] | None, evidence_id: str | None,
    result_output: Path, log_path: Path | None, targets_output: Path | None,
) -> dict[str, JsonValue]:
    financial: JsonValue = domain if domain is not None else None
    return {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "run": {
            "run_id": run_id,
            "strategy_id": PROCESS_INVENTORY_CERTIFICATION_LEVEL,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "data_root": data_root,
            "code_identity": _code_identity(),
            "fingerprint": fingerprint,
            "command": list(command),
        },
        "execution": {
            "status": run.status,
            "exit_code": run.exit_code,
            "signal_number": run.signal_number,
            "termination_reason": run.termination_reason,
            "wall_seconds": run.wall_seconds,
            "cpu_seconds": run.cpu_seconds,
            "gnu_max_individual_rss_bytes": run.gnu_max_individual_rss_bytes,
            "sampled_tree_pss_peak_bytes": run.sampled_tree_pss_peak_bytes,
            "sampled_tree_uss_peak_bytes": run.sampled_tree_uss_peak_bytes,
            "min_available_bytes": run.min_available_bytes,
            "process_swap_growth_bytes": run.process_swap_growth_bytes,
        },
        "financial": financial,
        "evidence": {
            "evidence_id": evidence_id,
            "result_path": str(result_output),
            "log_path": str(log_path) if log_path is not None else None,
            "targets_path": str(targets_output) if targets_output is not None else None,
        },
    }


def _finalize_lifecycle(
    registry_path: Path, run_id: str, run: MhsSupervisedRun, evidence_root: Path,
    result_output: Path, targets_output: Path | None,
) -> None:
    """Atomically finalize the observed execution after durable outcome persistence."""
    domain = _read_domain_payload(Path(run.result_output_path)) if run.domain_artifact_written else None
    primary_valid, terminal_certified, tier_flags, evidence_id = _read_compact_flags(
        Path(run.result_output_path)
    ) if run.domain_artifact_written else (None, None, {}, None)
    outcome: dict[str, JsonValue] = {
        "status": run.status,
        "exit_code": run.exit_code,
        "signal_number": run.signal_number,
        "termination_reason": run.termination_reason,
        "wall_seconds": run.wall_seconds,
        "cpu_seconds": run.cpu_seconds,
        "domain_artifact_written": run.domain_artifact_written,
        "base_primary_valid": tier_flags.get("base_primary_valid"),
        "stress_primary_valid": tier_flags.get("stress_primary_valid"),
        "primary_valid": primary_valid,
        "base_terminal_certified": tier_flags.get("base_terminal_certified"),
        "stress_terminal_certified": tier_flags.get("stress_terminal_certified"),
        "terminal_certified": terminal_certified,
        "evidence_id": evidence_id,
        "command": list(run.command),
    }
    finalization = RunFinalization(
        run_id=run_id,
        status=run.status,
        finalized_at=utc_now_iso8601(),
        primary_valid=primary_valid,
        terminal_certified=terminal_certified,
        outcome=outcome,
    )
    _ = domain
    artifacts = [
        artifact
        for artifact in (
            _local_artifact(run_id, "result", str(result_output), evidence_root),
            _local_artifact(
                run_id, "log", run.log_path if run.status != "completed" else None, evidence_root
            ),
            _local_artifact(run_id, "targets", str(targets_output) if targets_output is not None else None, evidence_root),
        )
        if artifact is not None
    ]
    if evidence_id is not None:
        artifacts.extend(_detail_artifacts(run_id, evidence_root, evidence_id))
    finalize_run(registry_path, finalization, tuple(artifacts))


def _retain_bounded_log(log_path: Path) -> Path | None:
    """Keep a bounded diagnostic log reference for non-completed executions."""
    try:
        data = log_path.read_bytes()
    except OSError:
        return None
    if len(data) > _BOUNDED_LOG_MAX_BYTES:
        log_path.write_bytes(data[-_BOUNDED_LOG_MAX_BYTES:])
    return log_path


def _publish_envelope(
    *, start: pd.Timestamp, end: pd.Timestamp, data_root: str | None, fingerprint: str,
    registry_path: Path, run_id: str, run: MhsSupervisedRun, evidence_root: Path,
    result_output: Path, targets_output: Path | None, staging: Path, log_path: Path,
    command: tuple[str, ...],
) -> dict[str, JsonValue]:
    """Atomically publish one result envelope and finalize its registry lifecycle."""
    domain = _read_domain_payload(staging) if staging.is_file() else None
    _, _, _, evidence_id = _read_compact_flags(staging) if staging.is_file() else (None, None, {}, None)
    kept_log: Path | None = None
    if run.status == "completed":
        with suppress(OSError):
            log_path.unlink()
    else:
        kept_log = _retain_bounded_log(log_path)
    envelope = _build_result_envelope(
        run_id=run_id, start=start, end=end, data_root=data_root, fingerprint=fingerprint,
        command=command, run=run, domain=domain, evidence_id=evidence_id,
        result_output=result_output, log_path=kept_log, targets_output=targets_output,
    )
    _atomic_write_json(result_output, envelope)
    _finalize_lifecycle(registry_path, run_id, run, evidence_root, result_output, targets_output)
    with suppress(OSError):
        staging.unlink()
    return envelope


def _record_operational_metadata(registry_path: Path, run_id: str, payload: dict[str, JsonValue]) -> None:
    """Record mutable cleanup observations beside the immutable execution outcome."""
    conn = sqlite3.connect(str(registry_path), timeout=5.0, isolation_level="DEFERRED")
    try:
        conn.execute("PRAGMA busy_timeout = 5000")
        with conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS run_operations "
                "(run_id TEXT PRIMARY KEY REFERENCES runs(run_id), operational_json TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO run_operations (run_id, operational_json) VALUES (?, ?)",
                (run_id, json.dumps(payload, sort_keys=True)),
            )
    finally:
        conn.close()


def _run_retention(registry_path: Path, run_id: str, evidence_root: Path, policy: RetentionPolicy) -> None:
    """Reclaim planned details without disguising cleanup failures as computation failures."""
    try:
        plan = plan_retention(registry_path, evidence_root, policy)
    except Exception as exc:  # noqa: BLE001
        _record_operational_metadata(
            registry_path, run_id,
            {"cleanup_error": str(exc), "budget_satisfied": None, "removed_evidence_ids": [], "reclaimed_bytes": 0},
        )
        _logger.warning("[RETENTION] status=plan_failed run_id=%s error=%s", run_id, exc)
        return
    try:
        result = apply_retention(registry_path, evidence_root, plan)
    except Exception as exc:  # noqa: BLE001
        _record_operational_metadata(
            registry_path, run_id,
            {
                "cleanup_error": str(exc),
                "budget_satisfied": plan.budget_satisfied,
                "removed_evidence_ids": [],
                "reclaimed_bytes": 0,
            },
        )
        _logger.warning("[RETENTION] status=apply_failed run_id=%s error=%s", run_id, exc)
        return
    _record_operational_metadata(
        registry_path, run_id,
        {
            "cleanup_error": None,
            "budget_satisfied": bool(plan.budget_satisfied and result.budget_satisfied),
            "removed_evidence_ids": list(result.removed_evidence_ids),
            "reclaimed_bytes": result.reclaimed_bytes,
        },
    )
    if not (plan.budget_satisfied and result.budget_satisfied):
        _logger.warning(
            "[RETENTION] status=budget_unsatisfied run_id=%s reclaimed_bytes=%d",
            run_id, result.reclaimed_bytes,
        )


def _workload_pss_uss(pid: int) -> tuple[int, int]:
    """Sampled child-tree PSS/USS sums excluding the supervisor."""
    import psutil

    root = psutil.Process(pid)
    procs = [root, *root.children(recursive=True)]
    pss = 0
    uss = 0
    for proc in procs:
        try:
            info = proc.memory_full_info()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        pss_value = getattr(info, "pss", None)
        uss_value = getattr(info, "uss", None)
        if pss_value is None or uss_value is None:
            raise OSError("process PSS/USS telemetry unavailable")
        pss += int(pss_value)
        uss += int(uss_value)
    return pss, uss


def _workload_swap_bytes(pid: int) -> int | None:
    """Return swap bytes for the supervised process tree, or null if unavailable."""
    import psutil

    root = psutil.Process(pid)
    procs = [root, *root.children(recursive=True)]
    total = 0
    observed = False
    for proc in procs:
        try:
            info = proc.memory_full_info()
            value = getattr(info, "swap", None)
            if value is None:
                raise OSError("process swap telemetry unavailable")
            total += int(value)
            observed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total if observed else None


def _gnu_time_prefix() -> list[str]:
    """GNU time wrapper when available, else an empty prefix."""
    with suppress(Exception):  # noqa: BLE001
        if os.path.isfile("/usr/bin/time") and os.access("/usr/bin/time", os.X_OK):
            return ["/usr/bin/time", "-f", "MHS_GNU_TIME elapsed=%e user=%U sys=%S maxrss=%M"]
    if shutil.which("time") is not None:
        return ["time", "-f", "MHS_GNU_TIME elapsed=%e user=%U sys=%S maxrss=%M"]
    return []


def _parse_gnu_metrics(log_path: Path) -> tuple[float | None, int | None]:
    """Best-effort GNU user+system CPU and max individual RSS from the log."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None, None
    match: re.Match[str] | None = None
    for line in text.splitlines():
        found = _GNU_LINE_RE.search(line)
        if found is not None:
            match = found
    if match is None:
        return None, None
    try:
        cpu = float(match.group(2)) + float(match.group(3))
        rss = int(match.group(4)) * 1024
    except (ValueError, ArithmeticError):
        return None, None
    return cpu, rss


def _terminate_group(proc: subprocess.Popen[bytes], timeout: float = GRACE_SECONDS) -> None:
    """Terminate only the launched process group with bounded graceful wait."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        return
    with suppress(ProcessLookupError, PermissionError, OSError):  # noqa: BLE001
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + max(timeout, 0.0)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)
    if proc.poll() is None:
        with suppress(ProcessLookupError, PermissionError, OSError):  # noqa: BLE001
            os.killpg(pgid, signal.SIGKILL)
        with suppress(Exception):  # noqa: BLE001
            proc.wait(timeout=5.0)


def _atomic_write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    """Atomically persist supervisor JSON without partial artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=str(path.parent), suffix=".tmp", delete=False, encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name
            tmp.write(json.dumps(payload, sort_keys=True, indent=2))
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path is not None:
            with suppress(Exception):  # noqa: BLE001
                os.unlink(tmp_path)
        raise


def _primary_completed(path: Path) -> bool:
    """True only when a new completed primary JSON is present."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return isinstance(payload, dict) and payload.get("status") == "completed"


def _failure_resource_code(path: Path) -> str | None:
    """Typed worker failure code from the dedicated failure artifact, if present."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if isinstance(payload, dict):
        code = payload.get("error_code")
        if isinstance(code, str):
            return code
    return None


def _validate_positive_interval(value: float | None, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{label} must be a positive finite number, got {value!r}")


def run_mhs_process_backtest(
    *, start: pd.Timestamp, end: pd.Timestamp, data_root: str | None,
    result_output: Path, targets_output: Path | None = None,
    tracking_error_threshold: float | None = None,
    timeout_seconds: float | None = None, poll_seconds: float = 0.25,
    memory_budget: MhsMemoryBudget | None = None,
    registry_path: Path | None = None, run_id: str | None = None,
    retention_policy: RetentionPolicy | None = None,
) -> MhsSupervisedRun:
    """Execute one 3-minute MHS evaluation and atomically publish its complete outcome.

    The envelope keeps process completion distinct from financial validity so a
    successful subprocess can never be mistaken for deployable evidence.
    """
    if not isinstance(start, pd.Timestamp) or start.tzinfo is None:
        raise ValueError("start must be a timezone-aware Timestamp")
    if not isinstance(end, pd.Timestamp) or end.tzinfo is None:
        raise ValueError("end must be a timezone-aware Timestamp")
    if start >= end:
        raise ValueError("start must precede end")
    if not isinstance(result_output, Path) or result_output.suffix != ".json":
        raise ValueError("result_output must be a .json path")
    if targets_output is not None and (
        not isinstance(targets_output, Path) or targets_output.suffix != ".parquet"
    ):
        raise ValueError("targets_output must be a parquet path")
    _validate_positive_interval(poll_seconds, "poll_seconds")
    if timeout_seconds is not None:
        _validate_positive_interval(timeout_seconds, "timeout_seconds")
    if tracking_error_threshold is not None and not isinstance(
        tracking_error_threshold, (int, float)
    ):
        raise ValueError("tracking_error_threshold must be numeric")
    ProcessExecutionPolicy(tracking_error_threshold=tracking_error_threshold)
    if retention_policy is not None and not isinstance(retention_policy, RetentionPolicy):
        raise ValueError(f"retention_policy must be a RetentionPolicy or None, got {retention_policy!r}")
    resolved_registry = Path(registry_path) if registry_path is not None else BACKTESTS_DIR / "registry.sqlite3"
    resolved_run_id = _resolve_run_id(run_id)
    evidence_root = resolved_registry.resolve().parent / "evidence"
    log_path = result_output.parent / f"{result_output.stem}.log"
    candidates: list[tuple[str, Path]] = [
        ("result_output", result_output),
        ("log", log_path),
    ]
    if targets_output is not None:
        candidates.append(("targets_output", targets_output))
    seen: set[Path] = set()
    for label, candidate in candidates:
        resolved = candidate.resolve() if isinstance(candidate, Path) else candidate
        if resolved in seen:
            raise ValueError(f"{label} must be distinct from other destinations")
        seen.add(resolved)
        if os.path.lexists(candidate):
            raise ValueError(f"{label} must be fresh: {candidate} already exists")
    budget = resolve_mhs_memory_budget(memory_budget)
    fingerprint = request_fingerprint(
        start=start, end=end, data_root=data_root,
        tracking_error_threshold=tracking_error_threshold,
        memory_budget=budget,
    )
    initialize_registry(resolved_registry)
    register_run(
        resolved_registry,
        RunRegistration(
            run_id=resolved_run_id,
            strategy_id=PROCESS_INVENTORY_CERTIFICATION_LEVEL,
            registered_at=utc_now_iso8601(),
            request=_registration_request(
                start=start, end=end, data_root=data_root,
                tracking_error_threshold=tracking_error_threshold,
                timeout_seconds=timeout_seconds, poll_seconds=poll_seconds,
                memory_budget=budget,
            ),
            managed_directory=result_output.parent,
        ),
    )
    staging = result_output.parent / ".staging_domain.json"
    command = [
        sys.executable, "-m", "src.application.mhs_worker",
        "--start", start.isoformat(), "--end", end.isoformat(),
        "--result-output", str(staging),
        "--total-tree-pss-bytes", str(budget.total_tree_pss_bytes),
        "--replay-tree-pss-bytes", str(budget.replay_tree_pss_bytes),
        "--min-available-bytes", str(budget.min_available_bytes),
        "--evidence-root", str(evidence_root),
        "--registry-path", str(resolved_registry),
        "--run-id", resolved_run_id,
    ]
    if data_root is not None:
        command += ["--data-root", str(data_root)]
    if targets_output is not None:
        command += ["--targets-output", str(targets_output)]
    if tracking_error_threshold is not None:
        command += ["--rebalance-tracking-error-threshold", str(tracking_error_threshold)]
    scoped_command = _gnu_time_prefix() + command
    wall_start = time.monotonic()
    samples = 0
    pss_peak: int | None = None
    uss_peak: int | None = None
    min_available: int | None = None
    swap_baseline: int | None = None
    swap_growth: int | None = None
    timed_out = False
    resource_reason: str | None = None
    interrupted = False
    last_heartbeat = wall_start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "w", encoding="utf-8")  # noqa: PTH123, SIM115
    proc: subprocess.Popen[bytes] | None = None
    try:
        try:
            proc = subprocess.Popen(  # noqa: S603
                scoped_command, stdout=log_handle, stderr=subprocess.STDOUT,
                start_new_session=True, shell=False,
            )
        except OSError as exc:
            log_handle.close()
            failed = MhsSupervisedRun(
                status="failed",
                command=tuple(scoped_command),
                exit_code=None,
                signal_number=None,
                start=start.isoformat(),
                end=end.isoformat(),
                data_root=data_root,
                result_output_path=str(staging),
                log_path=str(log_path),
                domain_artifact_written=staging.exists(),
                wall_seconds=time.monotonic() - wall_start,
                cpu_seconds=None,
                gnu_max_individual_rss_bytes=None,
                sampled_tree_pss_peak_bytes=None,
                sampled_tree_uss_peak_bytes=None,
                min_available_bytes=None,
                process_swap_growth_bytes=None,
                samples_taken=0,
                sample_interval_seconds=float(poll_seconds),
                cpu_scope=CPU_SCOPE,
                memory_scope=MEMORY_SCOPE,
                termination_reason=f"launch failed: {exc}",
                memory_budget=budget,
                run_id=resolved_run_id,
            )
            _publish_envelope(
                start=start, end=end, data_root=data_root, fingerprint=fingerprint,
                registry_path=resolved_registry, run_id=resolved_run_id, run=failed,
                evidence_root=evidence_root, result_output=result_output,
                targets_output=targets_output, staging=staging, log_path=log_path,
                command=tuple(scoped_command),
            )
            raise
        pid = proc.pid
        try:
            swap_baseline = _workload_swap_bytes(pid)
        except Exception:  # noqa: BLE001 - optional measurement
            swap_baseline = None
        while True:
            try:
                returncode = proc.wait(timeout=float(poll_seconds))
                break
            except subprocess.TimeoutExpired:
                pass
            wall = time.monotonic() - wall_start
            if wall - (last_heartbeat - wall_start) >= HEARTBEAT_SECONDS:
                _logger.info(
                    "[SUPERVISOR] heartbeat wall_s=%.1f pid=%d", wall, pid,
                )
                last_heartbeat = time.monotonic()
            if timeout_seconds is not None and wall >= float(timeout_seconds):
                timed_out = True
                _terminate_group(proc)
                returncode = proc.wait()
                break
            try:
                pss, uss = _workload_pss_uss(pid)
                headroom = current_mhs_headroom_bytes()
            except Exception as tel_exc:  # noqa: BLE001
                resource_reason = f"missing safety telemetry: {tel_exc}"
                _terminate_group(proc)
                returncode = proc.wait()
                break
            samples += 1
            pss_peak = pss if pss_peak is None else max(pss_peak, pss)
            uss_peak = uss if uss_peak is None else max(uss_peak, uss)
            min_available = headroom if min_available is None else min(min_available, headroom)
            try:
                swap_current = _workload_swap_bytes(pid)
            except Exception:  # noqa: BLE001 - optional measurement
                swap_current = None
            if swap_baseline is not None and swap_current is not None:
                growth = swap_current - swap_baseline
                if growth > 0:
                    swap_growth = growth if swap_growth is None else max(swap_growth, growth)
            if pss > budget.total_tree_pss_bytes:
                resource_reason = (
                    f"sampled tree PSS {pss} exceeds {budget.total_tree_pss_bytes}"
                )
                _terminate_group(proc)
                returncode = proc.wait()
                break
            if headroom < budget.min_available_bytes:
                resource_reason = (
                    f"headroom {headroom} below {budget.min_available_bytes}"
                )
                _terminate_group(proc)
                returncode = proc.wait()
                break
            if swap_growth is not None and swap_growth > 0:
                resource_reason = f"swap growth {swap_growth} bytes observed"
                _terminate_group(proc)
                returncode = proc.wait()
                break
    except KeyboardInterrupt:
        interrupted = True
        if proc is not None:
            _terminate_group(proc)
            with suppress(Exception):  # noqa: BLE001
                returncode = proc.wait()
        else:
            returncode = None
    finally:
        with suppress(Exception):  # noqa: BLE001
            log_handle.close()
    wall_seconds = time.monotonic() - wall_start
    if proc is None:
        raise OSError("child process failed to launch")
    cpu_seconds, gnu_rss = _parse_gnu_metrics(log_path)
    domain_written = staging.exists()
    failure_code = _failure_resource_code(staging) if domain_written else None
    exit_code: int | None = returncode if returncode is not None and returncode >= 0 else None
    signal_number: int | None = -returncode if returncode is not None and returncode < 0 else None
    termination_reason: str | None = None
    if timed_out:
        status: Literal[
            "completed", "failed", "timed_out", "signaled", "resource_rejected", "interrupted"
        ] = "timed_out"
        termination_reason = f"deadline exceeded after {wall_seconds:.1f}s"
        exit_code = returncode if returncode is not None and returncode >= 0 else exit_code
        signal_number = -returncode if returncode is not None and returncode < 0 else signal_number
    elif resource_reason is not None:
        status = "resource_rejected"
        termination_reason = resource_reason
    elif interrupted:
        status = "interrupted"
        termination_reason = "supervisor interrupted"
    elif failure_code in _RESOURCE_ERROR_CODES:
        status = "resource_rejected"
        termination_reason = f"worker reported {failure_code}"
    elif returncode is not None and returncode < 0:
        status = "signaled"
        termination_reason = f"signal {-returncode}"
    elif returncode == 0 and _primary_completed(staging):
        status = "completed"
    elif returncode == 0:
        status = "failed"
        termination_reason = "exit zero without a new completed domain artifact"
    else:
        status = "failed"
        termination_reason = f"exit_code={returncode}"
    run = MhsSupervisedRun(
        status=status,
        command=tuple(scoped_command),
        exit_code=exit_code,
        signal_number=signal_number,
        start=start.isoformat(),
        end=end.isoformat(),
        data_root=data_root,
        result_output_path=str(staging),
        log_path=str(log_path),
        domain_artifact_written=bool(domain_written),
        wall_seconds=float(wall_seconds),
        cpu_seconds=cpu_seconds,
        gnu_max_individual_rss_bytes=gnu_rss,
        sampled_tree_pss_peak_bytes=pss_peak,
        sampled_tree_uss_peak_bytes=uss_peak,
        min_available_bytes=min_available,
        process_swap_growth_bytes=swap_growth,
        samples_taken=int(samples),
        sample_interval_seconds=float(poll_seconds),
        cpu_scope=CPU_SCOPE,
        memory_scope=MEMORY_SCOPE,
        termination_reason=termination_reason,
        memory_budget=budget,
        run_id=resolved_run_id,
    )
    _publish_envelope(
        start=start, end=end, data_root=data_root, fingerprint=fingerprint,
        registry_path=resolved_registry, run_id=resolved_run_id, run=run,
        evidence_root=evidence_root, result_output=result_output,
        targets_output=targets_output, staging=staging, log_path=log_path,
        command=tuple(scoped_command),
    )
    if retention_policy is not None:
        _run_retention(resolved_registry, resolved_run_id, evidence_root, retention_policy)
    return run
