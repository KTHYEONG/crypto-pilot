"""Invariant guards for outcome-blind detail retention."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from src.backtests.contracts import (
    ArtifactReference,
    JsonValue,
    RetentionPlan,
    RetentionPolicy,
    RunFinalization,
    RunRegistration,
)
from src.backtests.registry import finalize_run, initialize_registry, register_run, set_run_protection
from src.backtests.retention import apply_retention, plan_retention

_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)  # noqa: UP017


def _ts(seconds: int) -> str:
    return (_BASE + timedelta(seconds=seconds)).isoformat()


def _managed_run(
    db: Path,
    root: Path,
    run_id: str,
    evidence_id: str,
    size: int,
    reg_seconds: int,
    fin_seconds: int,
    strategy_id: str = "strat-a",
    status: Any = "completed",
    primary_valid: bool | None = True,
    terminal_certified: bool | None = True,
    outcome: dict[str, JsonValue] | None = None,
    pinned: bool = False,
    resolved: bool = True,
    deployment_referenced: bool = False,
    create_bundle: bool = True,
) -> None:
    register_run(
        db,
        RunRegistration(
            run_id=run_id,
            strategy_id=strategy_id,
            registered_at=_ts(reg_seconds),
            request={"window": "3m"},
            managed_directory=None,
        ),
    )
    if create_bundle:
        bundle = root / evidence_id
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "detail.bin").write_bytes(b"x" * size)
    finalize_run(
        db,
        RunFinalization(
            run_id=run_id,
            status=status,
            finalized_at=_ts(fin_seconds),
            primary_valid=primary_valid,
            terminal_certified=terminal_certified,
            outcome={"pnl": 0.0} if outcome is None else outcome,
        ),
        (
            ArtifactReference(
                run_id=run_id,
                role="detail",
                path=Path(f"/evidence/{run_id[:8]}/detail.bin"),
                sha256="cd" * 32,
                byte_count=size,
                managed=True,
                evidence_id=evidence_id,
            ),
        ),
    )
    set_run_protection(db, run_id, pinned=pinned, resolved=resolved, deployment_referenced=deployment_referenced)


def _setup(db_name: Path) -> tuple[Path, Path]:
    db = db_name / "registry.sqlite3"
    root = db_name / "evidence"
    root.mkdir(parents=True, exist_ok=True)
    initialize_registry(db)
    return db, root


def test_shared_bundle_preserved_by_pinned_reference(tmp_path: Path) -> None:
    db, root = _setup(tmp_path)
    evidence_id = "a" * 64
    pinned_run, open_run = uuid.uuid4().hex, uuid.uuid4().hex
    _managed_run(db, root, pinned_run, evidence_id, 300, 0, 60, pinned=True, resolved=False)
    _managed_run(db, root, open_run, evidence_id, 300, 10, 120)
    plan = plan_retention(db, root, RetentionPolicy(max_detail_bytes=100))
    assert plan.evidence_ids == ()
    assert plan.protected_bytes == 300
    assert plan.budget_satisfied is False
    result = apply_retention(db, root, plan)
    assert result.removed_evidence_ids == ()
    assert (root / evidence_id).is_dir()


def test_unresolved_failures_preserved_and_reported_unsatisfied(tmp_path: Path) -> None:
    db, root = _setup(tmp_path)
    failed_run, unknown_run = uuid.uuid4().hex, uuid.uuid4().hex
    _managed_run(
        db, root, failed_run, "b" * 64, 200, 0, 60, status="failed", primary_valid=None,
        terminal_certified=None, resolved=False,
    )
    _managed_run(
        db, root, unknown_run, "c" * 64, 200, 10, 120, primary_valid=None,
        terminal_certified=None, resolved=False,
    )
    plan = plan_retention(db, root, RetentionPolicy(max_detail_bytes=50))
    assert plan.evidence_ids == ()
    assert plan.protected_bytes == 400
    assert plan.budget_satisfied is False


def test_quota_reclaims_oldest_first_outcome_blind(tmp_path: Path) -> None:
    db, root = _setup(tmp_path)
    oldest, middle, newest = "d" * 64, "e" * 64, "f" * 64
    _managed_run(db, root, uuid.uuid4().hex, oldest, 100, 0, 60, outcome={"pnl": 0.9})
    _managed_run(db, root, uuid.uuid4().hex, middle, 100, 10, 120, outcome={"pnl": 0.1})
    _managed_run(db, root, uuid.uuid4().hex, newest, 100, 20, 180, outcome={"pnl": -0.5})
    conn = sqlite3.connect(str(db))
    try:
        trials_before = int(conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0])
    finally:
        conn.close()
    plan = plan_retention(db, root, RetentionPolicy(max_detail_bytes=150))
    assert plan.evidence_ids == (oldest, middle)
    assert plan.reclaimable_bytes == 200
    assert plan.budget_satisfied is True
    result = apply_retention(db, root, plan)
    assert result.removed_evidence_ids == (oldest, middle)
    assert result.reclaimed_bytes == 200
    assert result.budget_satisfied is True
    assert not (root / oldest).exists()
    assert (root / newest).is_dir()
    conn = sqlite3.connect(str(db))
    try:
        trials_after = int(conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0])
        retained = conn.execute("SELECT retained FROM artifacts WHERE evidence_id = ?", (oldest,)).fetchall()
    finally:
        conn.close()
    assert trials_after == trials_before
    assert retained
    assert all(int(r[0]) == 0 for r in retained)


def test_run_budget_reclaims_oldest_first(tmp_path: Path) -> None:
    db, root = _setup(tmp_path)
    first, second, third = "1" * 64, "2" * 64, "3" * 64
    _managed_run(db, root, uuid.uuid4().hex, first, 100, 0, 60)
    _managed_run(db, root, uuid.uuid4().hex, second, 100, 10, 120)
    _managed_run(db, root, uuid.uuid4().hex, third, 100, 20, 180)
    plan = plan_retention(db, root, RetentionPolicy(max_detail_runs=1))
    assert plan.evidence_ids == (first, second)
    assert plan.budget_satisfied is True


def test_none_budgets_never_evict(tmp_path: Path) -> None:
    db, root = _setup(tmp_path)
    _managed_run(db, root, uuid.uuid4().hex, "4" * 64, 100, 0, 60)
    plan = plan_retention(db, root, RetentionPolicy())
    assert plan.evidence_ids == ()
    assert plan.budget_satisfied is True
    result = apply_retention(db, root, plan)
    assert result.removed_evidence_ids == ()
    assert result.budget_satisfied is True
    assert (root / ("4" * 64)).is_dir()


def test_ownership_escape_skips_external_files(tmp_path: Path) -> None:
    db, root = _setup(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"keep")
    evidence_id = "5" * 64
    run_id = uuid.uuid4().hex
    register_run(
        db,
        RunRegistration(
            run_id=run_id, strategy_id="strat-a", registered_at=_ts(0), request={}, managed_directory=None
        ),
    )
    finalize_run(
        db,
        RunFinalization(
            run_id=run_id, status="completed", finalized_at=_ts(60),
            primary_valid=True, terminal_certified=True, outcome={},
        ),
        (
            ArtifactReference(
                run_id=run_id, role="custom-export", path=outside / "export.json",
                sha256="ef" * 32, byte_count=4, managed=False, evidence_id=None,
            ),
            ArtifactReference(
                run_id=run_id, role="detail", path=outside / "detail.bin",
                sha256="ef" * 32, byte_count=4, managed=True, evidence_id=evidence_id,
            ),
        ),
    )
    set_run_protection(db, run_id, pinned=False, resolved=True, deployment_referenced=False)
    (root / evidence_id).symlink_to(outside, target_is_directory=True)
    plan = plan_retention(db, root, RetentionPolicy(max_detail_bytes=1))
    assert plan.evidence_ids == ()
    result = apply_retention(db, root, plan)
    assert result.removed_evidence_ids == ()
    assert sentinel.read_bytes() == b"keep"
    assert (root / evidence_id).is_symlink()


def test_apply_rechecks_new_protection(tmp_path: Path) -> None:
    db, root = _setup(tmp_path)
    first_id, second_id = "6" * 64, "7" * 64
    first_run, second_run = uuid.uuid4().hex, uuid.uuid4().hex
    _managed_run(db, root, first_run, first_id, 100, 0, 60)
    _managed_run(db, root, second_run, second_id, 100, 10, 120)
    plan = plan_retention(db, root, RetentionPolicy(max_detail_bytes=10))
    assert plan.evidence_ids == (first_id, second_id)
    set_run_protection(db, second_run, pinned=True, resolved=False, deployment_referenced=False)
    result = apply_retention(db, root, plan)
    assert result.removed_evidence_ids == (first_id,)
    assert result.budget_satisfied is False
    assert (root / second_id).is_dir()
    assert not (root / first_id).exists()


def test_running_reference_blocks_reclamation(tmp_path: Path) -> None:
    db, root = _setup(tmp_path)
    run_id = uuid.uuid4().hex
    register_run(
        db,
        RunRegistration(
            run_id=run_id, strategy_id="strat-a", registered_at=_ts(0), request={}, managed_directory=None
        ),
    )
    evidence_id = "8" * 64
    bundle = root / evidence_id
    bundle.mkdir()
    (bundle / "detail.bin").write_bytes(b"x" * 200)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT INTO artifacts (run_id, role, path, sha256, byte_count, managed, evidence_id, retained) VALUES (?, ?, ?, ?, ?, 1, ?, 1)",
            (run_id, "detail", "/evidence/running/detail.bin", "ab" * 32, 200, evidence_id),
        )
        conn.commit()
    finally:
        conn.close()
    plan = plan_retention(db, root, RetentionPolicy(max_detail_bytes=10))
    assert plan.evidence_ids == ()
    assert plan.protected_bytes == 200
    assert plan.budget_satisfied is False


def test_dangling_reference_fails_closed(tmp_path: Path) -> None:
    db, root = _setup(tmp_path)
    evidence_id = "9" * 64
    bundle = root / evidence_id
    bundle.mkdir()
    (bundle / "detail.bin").write_bytes(b"x" * 50)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "INSERT INTO artifacts (run_id, role, path, sha256, byte_count, managed, evidence_id, retained) VALUES (?, ?, ?, ?, ?, 1, ?, 1)",
            ("f" * 32, "detail", "/evidence/dangling/detail.bin", "ab" * 32, 50, evidence_id),
        )
        conn.commit()
    finally:
        conn.close()
    plan = plan_retention(db, root, RetentionPolicy(max_detail_bytes=10))
    assert plan.evidence_ids == ()
    assert plan.protected_bytes == 50


def test_apply_rejects_unsafe_evidence_identity(tmp_path: Path) -> None:
    db, root = _setup(tmp_path)
    plan = RetentionPlan(
        evidence_ids=("../escape",), reclaimable_bytes=0, protected_bytes=0, budget_satisfied=False
    )
    with pytest.raises(ValueError, match=r".+"):
        apply_retention(db, root, plan)


def test_healthy_unresolved_run_is_reclaimable(tmp_path: Path) -> None:
    db, root = _setup(tmp_path)
    evidence_id = "c0" * 32
    _managed_run(db, root, uuid.uuid4().hex, evidence_id, 100, 0, 60, resolved=False)
    plan = plan_retention(db, root, RetentionPolicy(max_detail_bytes=10))
    assert plan.evidence_ids == (evidence_id,)
    assert plan.budget_satisfied is True
    result = apply_retention(db, root, plan)
    assert result.removed_evidence_ids == (evidence_id,)


def test_apply_skips_bundle_replaced_by_symlink(tmp_path: Path) -> None:
    db, root = _setup(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.bin"
    sentinel.write_bytes(b"keep")
    evidence_id = "d0" * 32
    _managed_run(db, root, uuid.uuid4().hex, evidence_id, 100, 0, 60)
    plan = plan_retention(db, root, RetentionPolicy(max_detail_bytes=10))
    assert plan.evidence_ids == (evidence_id,)
    bundle = root / evidence_id
    for child in bundle.iterdir():
        child.unlink()
    bundle.rmdir()
    bundle.symlink_to(outside, target_is_directory=True)
    result = apply_retention(db, root, plan)
    assert result.removed_evidence_ids == ()
    assert result.budget_satisfied is False
    assert sentinel.read_bytes() == b"keep"


def test_apply_skips_unknown_planned_evidence(tmp_path: Path) -> None:
    db, root = _setup(tmp_path)
    plan = RetentionPlan(evidence_ids=("0" * 64,), reclaimable_bytes=0, protected_bytes=0, budget_satisfied=False)
    result = apply_retention(db, root, plan)
    assert result.removed_evidence_ids == ()
    assert result.budget_satisfied is False


def test_missing_bundle_dir_reclaimed_by_reference(tmp_path: Path) -> None:
    db, root = _setup(tmp_path)
    evidence_id = "a0" * 32
    _managed_run(db, root, uuid.uuid4().hex, evidence_id, 80, 0, 60, create_bundle=False)
    plan = plan_retention(db, root, RetentionPolicy(max_detail_bytes=10))
    assert plan.evidence_ids == (evidence_id,)
    result = apply_retention(db, root, plan)
    assert result.removed_evidence_ids == (evidence_id,)
    assert result.reclaimed_bytes == 80
    conn = sqlite3.connect(str(db))
    try:
        retained = conn.execute("SELECT retained FROM artifacts WHERE evidence_id = ?", (evidence_id,)).fetchall()
    finally:
        conn.close()
    assert all(int(r[0]) == 0 for r in retained)


def test_unmanaged_artifacts_never_planned(tmp_path: Path) -> None:
    db, root = _setup(tmp_path)
    run_id = uuid.uuid4().hex
    register_run(
        db,
        RunRegistration(
            run_id=run_id, strategy_id="strat-a", registered_at=_ts(0), request={}, managed_directory=None
        ),
    )
    finalize_run(
        db,
        RunFinalization(
            run_id=run_id, status="completed", finalized_at=_ts(60),
            primary_valid=True, terminal_certified=True, outcome={},
        ),
        (
            ArtifactReference(
                run_id=run_id, role="custom-export", path=Path("/evidence/export.json"),
                sha256="ab" * 32, byte_count=999, managed=False, evidence_id="b0" * 32,
            ),
        ),
    )
    set_run_protection(db, run_id, pinned=False, resolved=True, deployment_referenced=False)
    plan = plan_retention(db, root, RetentionPolicy(max_detail_bytes=1))
    assert plan.evidence_ids == ()


def test_plan_rejects_missing_registry(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    with pytest.raises(sqlite3.Error):
        plan_retention(tmp_path / "missing.sqlite3", root, RetentionPolicy(max_detail_bytes=1))


def test_plan_rejects_unsafe_root(tmp_path: Path) -> None:
    db, _root = _setup(tmp_path)
    with pytest.raises(ValueError, match=r".+"):
        plan_retention(db, tmp_path / "no-such-dir", RetentionPolicy(max_detail_bytes=1))


def test_plan_rejects_bad_policy(tmp_path: Path) -> None:
    db, root = _setup(tmp_path)
    with pytest.raises(ValueError, match=r".+"):
        plan_retention(db, root, cast(RetentionPolicy, {"max_detail_bytes": 1}))


def test_apply_rejects_bad_root(tmp_path: Path) -> None:
    db, _root = _setup(tmp_path)
    plan = RetentionPlan(evidence_ids=(), reclaimable_bytes=0, protected_bytes=0, budget_satisfied=True)
    with pytest.raises(ValueError, match=r".+"):
        apply_retention(db, tmp_path / "no-such-dir", plan)


def test_apply_rejects_bad_plan(tmp_path: Path) -> None:
    _db, root = _setup(tmp_path)
    with pytest.raises(ValueError, match=r".+"):
        apply_retention(_db, root, cast(RetentionPlan, ()))
