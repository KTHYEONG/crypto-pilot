"""Outcome-blind retention planning and reclamation over managed evidence bundles."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from src.backtests.contracts import RetentionPlan, RetentionPolicy, RetentionResult


def _validated_root(evidence_root: Path) -> Path:
    if not isinstance(evidence_root, Path) or evidence_root.is_symlink() or not evidence_root.is_dir():
        raise ValueError(f"evidence_root must be an owned directory, got {evidence_root!r}")
    return evidence_root


def _validated_evidence_id(evidence_id: str) -> str:
    if (
        not isinstance(evidence_id, str)
        or not evidence_id
        or "/" in evidence_id
        or "\\" in evidence_id
        or ".." in evidence_id
    ):
        raise ValueError(f"evidence_id must be a plain content identity, got {evidence_id!r}")
    return evidence_id


def _connect(path: Path) -> sqlite3.Connection:
    if not isinstance(path, Path) or not path.is_file():
        raise sqlite3.Error(f"registry unavailable: {path!r}")
    conn = sqlite3.connect(str(path), timeout=5.0, isolation_level="DEFERRED")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _run_protected(conn: sqlite3.Connection, run_id: str) -> bool:
    row = conn.execute("SELECT pinned, resolved, deployment_referenced FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return True
    if int(row[0]) == 1 or int(row[2]) == 1:
        return True
    final = conn.execute(
        "SELECT status, primary_valid, terminal_certified FROM finalizations WHERE run_id = ?", (run_id,)
    ).fetchone()
    if final is None:
        return True
    if int(row[1]) == 1:
        return False
    if str(final[0]) != "completed":
        return True
    if final[1] is None or int(final[1]) != 1:
        return True
    return final[2] is None or int(final[2]) != 1


def _bundle_snapshots(conn: sqlite3.Connection) -> tuple[dict[str, set[str]], dict[str, int], dict[str, str]]:
    members: dict[str, set[str]] = {}
    sizes: dict[str, int] = {}
    for evidence_id, run_id, byte_count in conn.execute(
        "SELECT evidence_id, run_id, byte_count FROM artifacts WHERE managed = 1 AND retained = 1 AND evidence_id IS NOT NULL"
    ).fetchall():
        key = str(evidence_id)
        members.setdefault(key, set()).add(str(run_id))
        sizes[key] = max(sizes.get(key, 0), int(byte_count))
    finalized = {str(r[0]): str(r[1]) for r in conn.execute("SELECT run_id, finalized_at FROM finalizations").fetchall()}
    oldest: dict[str, str] = {}
    for key, runs in members.items():
        stamps = sorted(finalized[r] for r in runs if r in finalized)
        if stamps:
            oldest[key] = stamps[0]
    return members, sizes, oldest


def _within_budget(remaining_bytes: int, remaining_runs: int, policy: RetentionPolicy) -> bool:
    if policy.max_detail_bytes is not None and remaining_bytes > policy.max_detail_bytes:
        return False
    return not (policy.max_detail_runs is not None and remaining_runs > policy.max_detail_runs)


def _managed_finalized_runs(conn: sqlite3.Connection, members: dict[str, set[str]]) -> set[str]:
    finalized = {str(r[0]) for r in conn.execute("SELECT run_id FROM finalizations").fetchall()}
    return {r for runs in members.values() for r in runs if r in finalized}


def plan_retention(registry_path: Path, evidence_root: Path, policy: RetentionPolicy) -> RetentionPlan:
    """Plan outcome-blind reclamation of exclusively managed, unprotected detail bundles. Args: registry, owned evidence root and explicit budgets. Returns: a non-mutating plan including feasibility. Raises: ValueError for unsafe ownership or roots; sqlite3.Error for unavailable registry evidence."""
    root = _validated_root(evidence_root)
    if not isinstance(policy, RetentionPolicy):
        raise ValueError(f"policy must be a RetentionPolicy, got {policy!r}")
    conn = _connect(registry_path)
    try:
        members, sizes, oldest = _bundle_snapshots(conn)
        guarded = {key: any(_run_protected(conn, run) for run in runs) for key, runs in members.items()}
        total_runs = len(_managed_finalized_runs(conn, members))
    finally:
        conn.close()
    for key in oldest:
        if not guarded[key] and (root / key).is_symlink():
            guarded[key] = True
    protected_bytes = sum(sizes[key] for key, flag in guarded.items() if flag)
    ordered = sorted(((oldest[key], key) for key in oldest if not guarded[key]), key=lambda item: (item[0], item[1]))
    if policy.max_detail_bytes is None and policy.max_detail_runs is None:
        return RetentionPlan(evidence_ids=(), reclaimable_bytes=0, protected_bytes=protected_bytes, budget_satisfied=True)
    total_bytes = protected_bytes + sum(sizes[key] for _, key in ordered)
    planned: list[str] = []
    reclaimed = 0
    covered: set[str] = set()
    for _, key in ordered:
        if _within_budget(total_bytes - reclaimed, total_runs - len(covered), policy):
            break
        planned.append(key)
        reclaimed += sizes[key]
        covered.update(members[key])
    satisfied = _within_budget(total_bytes - reclaimed, total_runs - len(covered), policy)
    return RetentionPlan(
        evidence_ids=tuple(planned), reclaimable_bytes=reclaimed, protected_bytes=protected_bytes, budget_satisfied=satisfied
    )


def apply_retention(registry_path: Path, evidence_root: Path, plan: RetentionPlan) -> RetentionResult:
    """Recheck ownership and protection before reclaiming planned managed details. Args: registry, owned evidence root and prior plan. Returns: actual reclaimed evidence and feasibility. Raises: OSError for deletion failure; ValueError for unsafe paths."""
    root = _validated_root(evidence_root)
    if not isinstance(plan, RetentionPlan):
        raise ValueError(f"plan must be a RetentionPlan, got {plan!r}")
    conn = _connect(registry_path)
    try:
        removed: list[str] = []
        reclaimed = 0
        for evidence_id in plan.evidence_ids:
            _validated_evidence_id(evidence_id)
            bundle = root / evidence_id
            if bundle.is_symlink():
                continue
            refs = conn.execute(
                "SELECT run_id, byte_count FROM artifacts WHERE evidence_id = ? AND retained = 1", (evidence_id,)
            ).fetchall()
            if not refs:
                continue
            if any(_run_protected(conn, str(item[0])) for item in refs):
                continue
            physical = max(int(item[1]) for item in refs)
            if bundle.is_dir():
                shutil.rmtree(bundle)
            conn.execute("UPDATE artifacts SET retained = 0 WHERE evidence_id = ?", (evidence_id,))
            conn.commit()
            removed.append(evidence_id)
            reclaimed += physical
        conn.commit()
    finally:
        conn.close()
    return RetentionResult(
        removed_evidence_ids=tuple(removed),
        reclaimed_bytes=reclaimed,
        budget_satisfied=len(removed) == len(plan.evidence_ids),
    )


__all__ = ["apply_retention", "plan_retention"]
