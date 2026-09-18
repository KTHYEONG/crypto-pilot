"""Invariant guards for the local execution registry."""

from __future__ import annotations

import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from src.backtests.contracts import (
    ArtifactReference,
    JsonValue,
    RetentionPlan,
    RetentionPolicy,
    RetentionResult,
    RunFinalization,
    RunRegistration,
    utc_now_iso8601,
)
from src.backtests.registry import finalize_run, initialize_registry, register_run, set_run_protection

_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)  # noqa: UP017


def _ts(seconds: int) -> str:
    return (_BASE + timedelta(seconds=seconds)).isoformat()


def _registration(
    run_id: str,
    strategy_id: str = "strat-a",
    seconds: int = 0,
    request: dict[str, JsonValue] | None = None,
) -> RunRegistration:
    return RunRegistration(
        run_id=run_id,
        strategy_id=strategy_id,
        registered_at=_ts(seconds),
        request={"window": "3m", "budget": 7} if request is None else request,
        managed_directory=None,
    )


def _finalization(
    run_id: str,
    seconds: int = 60,
    status: Any = "completed",
    primary_valid: bool | None = True,
    terminal_certified: bool | None = True,
    outcome: dict[str, JsonValue] | None = None,
) -> RunFinalization:
    return RunFinalization(
        run_id=run_id,
        status=status,
        finalized_at=_ts(seconds),
        primary_valid=primary_valid,
        terminal_certified=terminal_certified,
        outcome={"exit_code": 0, "telemetry": None} if outcome is None else outcome,
    )


def _artifact(
    run_id: str,
    role: str = "detail",
    name: str = "bundle.bin",
    size: int = 100,
    managed: bool = True,
    evidence_id: str | None = None,
) -> ArtifactReference:
    return ArtifactReference(
        run_id=run_id,
        role=role,
        path=Path(f"/evidence/{run_id[:8]}/{name}"),
        sha256="ab" * 32,
        byte_count=size,
        managed=managed,
        evidence_id=evidence_id,
    )


def _db(tmp_path: Path) -> Path:
    path = tmp_path / "registry.sqlite3"
    initialize_registry(path)
    return path


def _counts(db: Path) -> dict[str, int]:
    conn = sqlite3.connect(str(db))
    try:
        return {
            "runs": int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]),
            "finalizations": int(conn.execute("SELECT COUNT(*) FROM finalizations").fetchone()[0]),
            "artifacts": int(conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]),
            "trials": int(conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]),
        }
    finally:
        conn.close()


def test_distinct_runs_share_single_trial(tmp_path: Path) -> None:
    db = _db(tmp_path)
    first, second = uuid.uuid4().hex, uuid.uuid4().hex
    register_run(db, _registration(first))
    register_run(db, _registration(second, seconds=5))
    counts = _counts(db)
    assert counts["runs"] == 2
    assert counts["trials"] == 1


def test_concurrent_registrations_persist(tmp_path: Path) -> None:
    db = _db(tmp_path)
    first, second = uuid.uuid4().hex, uuid.uuid4().hex
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item: register_run(db, item), [_registration(first), _registration(second, seconds=5)]))
    assert _counts(db)["runs"] == 2


def test_repeated_finalization_is_idempotent(tmp_path: Path) -> None:
    db = _db(tmp_path)
    run_id = uuid.uuid4().hex
    register_run(db, _registration(run_id))
    finalization = _finalization(run_id)
    artifacts = (_artifact(run_id, evidence_id="a" * 64),)
    finalize_run(db, finalization, artifacts)
    finalize_run(db, finalization, artifacts)
    conn = sqlite3.connect(str(db))
    try:
        assert int(conn.execute("SELECT COUNT(*) FROM finalizations").fetchone()[0]) == 1
        assert int(conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]) == 1
    finally:
        conn.close()


def test_conflicting_finalization_rejected_and_original_kept(tmp_path: Path) -> None:
    db = _db(tmp_path)
    run_id = uuid.uuid4().hex
    register_run(db, _registration(run_id))
    finalize_run(db, _finalization(run_id), (_artifact(run_id, evidence_id="a" * 64),))
    with pytest.raises(ValueError, match=r".+"):
        finalize_run(db, _finalization(run_id, outcome={"exit_code": 1}), (_artifact(run_id, evidence_id="a" * 64),))
    conn = sqlite3.connect(str(db))
    try:
        stored = conn.execute("SELECT outcome_json FROM finalizations WHERE run_id = ?", (run_id,)).fetchone()[0]
    finally:
        conn.close()
    assert "exit_code" in stored
    assert "0" in stored


def test_failed_artifact_insert_rolls_back_finalization(tmp_path: Path) -> None:
    db = _db(tmp_path)
    run_id = uuid.uuid4().hex
    register_run(db, _registration(run_id))
    first = _artifact(run_id, evidence_id="a" * 64)
    with pytest.raises(sqlite3.Error):
        finalize_run(db, _finalization(run_id), (first, first))
    conn = sqlite3.connect(str(db))
    try:
        assert int(conn.execute("SELECT COUNT(*) FROM finalizations").fetchone()[0]) == 0
        assert int(conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]) == 0
    finally:
        conn.close()


def test_failed_finalization_keeps_publication_lease(tmp_path: Path) -> None:
    db = _db(tmp_path)
    run_id = uuid.uuid4().hex
    register_run(db, _registration(run_id))
    lease = _artifact(run_id, role="publication_lease", evidence_id="a" * 64)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT INTO artifacts (run_id, role, path, sha256, byte_count, managed, evidence_id, retained) VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
            (lease.run_id, lease.role, str(lease.path), lease.sha256, lease.byte_count, 1, lease.evidence_id),
        )
        conn.commit()
    finally:
        conn.close()
    duplicate = _artifact(run_id, evidence_id="a" * 64)
    with pytest.raises(sqlite3.Error):
        finalize_run(db, _finalization(run_id), (duplicate, duplicate))
    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM artifacts WHERE run_id = ? AND role = ?", (run_id, "publication_lease")
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_execution_status_and_financial_validity_stored_separately(tmp_path: Path) -> None:
    db = _db(tmp_path)
    run_id = uuid.uuid4().hex
    register_run(db, _registration(run_id))
    finalize_run(
        db, _finalization(run_id, primary_valid=False, terminal_certified=None), (_artifact(run_id),)
    )
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT status, primary_valid, terminal_certified FROM finalizations WHERE run_id = ?", (run_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == "completed"
    assert row[1] == 0
    assert row[2] is None


def test_null_observations_preserved_as_null(tmp_path: Path) -> None:
    db = _db(tmp_path)
    run_id = uuid.uuid4().hex
    register_run(db, _registration(run_id))
    finalize_run(
        db,
        _finalization(run_id, primary_valid=None, terminal_certified=None, outcome={"peak": None, "exit_code": 0}),
        (_artifact(run_id),),
    )
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT primary_valid, terminal_certified, outcome_json FROM finalizations WHERE run_id = ?", (run_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row[0] is None
    assert row[1] is None
    assert '"peak":null' in str(row[2]).replace(" ", "")


def test_repeated_registration_is_idempotent(tmp_path: Path) -> None:
    db = _db(tmp_path)
    run_id = uuid.uuid4().hex
    registration = _registration(run_id)
    register_run(db, registration)
    register_run(db, registration)
    assert _counts(db)["runs"] == 1


def test_conflicting_registration_rejected(tmp_path: Path) -> None:
    db = _db(tmp_path)
    run_id = uuid.uuid4().hex
    register_run(db, _registration(run_id))
    with pytest.raises(ValueError, match=r".+"):
        register_run(db, _registration(run_id, strategy_id="strat-b"))


def test_finalize_unknown_run_rejected(tmp_path: Path) -> None:
    db = _db(tmp_path)
    run_id = uuid.uuid4().hex
    with pytest.raises(ValueError, match=r".+"):
        finalize_run(db, _finalization(run_id), ())


def test_finalize_artifact_run_mismatch_rejected(tmp_path: Path) -> None:
    db = _db(tmp_path)
    run_id = uuid.uuid4().hex
    register_run(db, _registration(run_id))
    with pytest.raises(ValueError, match=r".+"):
        finalize_run(db, _finalization(run_id), (_artifact(uuid.uuid4().hex),))


def test_finalize_conflicting_artifacts_rejected(tmp_path: Path) -> None:
    db = _db(tmp_path)
    run_id = uuid.uuid4().hex
    register_run(db, _registration(run_id))
    finalize_run(db, _finalization(run_id), (_artifact(run_id, evidence_id="a" * 64),))
    with pytest.raises(ValueError, match=r".+"):
        finalize_run(db, _finalization(run_id), (_artifact(run_id, evidence_id="b" * 64),))


def test_set_run_protection_records_flags(tmp_path: Path) -> None:
    db = _db(tmp_path)
    run_id = uuid.uuid4().hex
    register_run(db, _registration(run_id))
    set_run_protection(db, run_id, pinned=True, resolved=False, deployment_referenced=True)
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute("SELECT pinned, resolved, deployment_referenced FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    finally:
        conn.close()
    assert tuple(row) == (1, 0, 1)


def test_set_run_protection_unknown_run_raises(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with pytest.raises(KeyError):
        set_run_protection(db, uuid.uuid4().hex, pinned=True, resolved=False, deployment_referenced=False)


def test_set_run_protection_rejects_non_bool_flags(tmp_path: Path) -> None:
    db = _db(tmp_path)
    run_id = uuid.uuid4().hex
    register_run(db, _registration(run_id))
    with pytest.raises(ValueError, match=r".+"):
        set_run_protection(db, run_id, pinned=cast(bool, 1), resolved=False, deployment_referenced=False)


def test_initialize_registry_rejects_unsupported_version(tmp_path: Path) -> None:
    db = tmp_path / "registry.sqlite3"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE schema_meta (version INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO schema_meta (version) VALUES (999)")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(ValueError, match=r".+"):
        initialize_registry(db)


def test_initialize_registry_rejects_incomplete_schema(tmp_path: Path) -> None:
    db = tmp_path / "registry.sqlite3"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE schema_meta (version INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO schema_meta (version) VALUES (1)")
        conn.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(ValueError, match=r".+"):
        initialize_registry(db)


def test_initialize_registry_rejects_non_path() -> None:
    with pytest.raises(ValueError, match=r".+"):
        initialize_registry(cast(Path, "not-a-path"))


def test_run_id_rejects_path_components() -> None:
    with pytest.raises(ValueError, match=r".+"):
        _registration("a" * 31 + "/")


def test_run_id_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match=r".+"):
        _registration("abc")


def test_run_id_rejects_non_hex() -> None:
    with pytest.raises(ValueError, match=r".+"):
        _registration("z" * 32)


def test_registration_rejects_empty_strategy() -> None:
    with pytest.raises(ValueError, match=r".+"):
        _registration(uuid.uuid4().hex, strategy_id="")


def test_registration_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match=r".+"):
        RunRegistration(
            run_id=uuid.uuid4().hex,
            strategy_id="s",
            registered_at="2026-01-01T00:00:00",
            request={},
            managed_directory=None,
        )


def test_registration_rejects_empty_timestamp() -> None:
    with pytest.raises(ValueError, match=r".+"):
        RunRegistration(
            run_id=uuid.uuid4().hex,
            strategy_id="s",
            registered_at="",
            request={},
            managed_directory=None,
        )


def test_registration_accepts_zulu_timestamp() -> None:
    registration = RunRegistration(
        run_id=uuid.uuid4().hex,
        strategy_id="s",
        registered_at="2026-01-01T00:00:00Z",
        request={},
        managed_directory=None,
    )
    assert registration.registered_at.endswith("Z")


def test_registration_accepts_nested_metadata() -> None:
    registration = _registration(
        uuid.uuid4().hex,
        request={"tags": ["a", 1, None], "nested": {"score": 1.5}},
    )
    assert registration.request["tags"] == ["a", 1, None]


def test_registration_rejects_non_utc_offset() -> None:
    with pytest.raises(ValueError, match=r".+"):
        RunRegistration(
            run_id=uuid.uuid4().hex,
            strategy_id="s",
            registered_at="2026-01-01T00:00:00+05:00",
            request={},
            managed_directory=None,
        )


def test_registration_rejects_unparsable_timestamp() -> None:
    with pytest.raises(ValueError, match=r".+"):
        RunRegistration(
            run_id=uuid.uuid4().hex,
            strategy_id="s",
            registered_at="not-a-time",
            request={},
            managed_directory=None,
        )


def test_registration_rejects_non_finite_metadata() -> None:
    with pytest.raises(ValueError, match=r".+"):
        _registration(uuid.uuid4().hex, request={"v": cast(JsonValue, float("inf"))})


def test_registration_rejects_non_json_metadata() -> None:
    with pytest.raises(ValueError, match=r".+"):
        _registration(uuid.uuid4().hex, request=cast(dict[str, JsonValue], {"v": (1, 2)}))


def test_registration_rejects_non_string_metadata_key() -> None:
    with pytest.raises(ValueError, match=r".+"):
        _registration(uuid.uuid4().hex, request=cast(dict[str, JsonValue], {1: "x"}))


def test_registration_rejects_nested_non_string_key() -> None:
    with pytest.raises(ValueError, match=r".+"):
        _registration(uuid.uuid4().hex, request=cast(dict[str, JsonValue], {"nested": {1: "x"}}))


def test_registration_rejects_non_mapping_request() -> None:
    with pytest.raises(ValueError, match=r".+"):
        RunRegistration(
            run_id=uuid.uuid4().hex,
            strategy_id="s",
            registered_at=_ts(0),
            request=cast(dict[str, JsonValue], ["nope"]),
            managed_directory=None,
        )


def test_registration_rejects_non_path_directory() -> None:
    with pytest.raises(ValueError, match=r".+"):
        RunRegistration(
            run_id=uuid.uuid4().hex,
            strategy_id="s",
            registered_at=_ts(0),
            request={},
            managed_directory=cast(Path, "/evidence/staging"),
        )


def test_finalization_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match=r".+"):
        _finalization(uuid.uuid4().hex, status="exploded")


def test_finalization_rejects_non_bool_flags() -> None:
    with pytest.raises(ValueError, match=r".+"):
        _finalization(uuid.uuid4().hex, primary_valid=cast(bool, 1))


def test_finalization_rejects_non_bool_terminal() -> None:
    with pytest.raises(ValueError, match=r".+"):
        _finalization(uuid.uuid4().hex, terminal_certified=cast(bool, "yes"))


def test_artifact_rejects_empty_role() -> None:
    with pytest.raises(ValueError, match=r".+"):
        _artifact(uuid.uuid4().hex, role="")


def test_artifact_rejects_relative_path() -> None:
    with pytest.raises(ValueError, match=r".+"):
        ArtifactReference(
            run_id=uuid.uuid4().hex,
            role="detail",
            path=Path("relative/path"),
            sha256="ab" * 32,
            byte_count=10,
            managed=True,
            evidence_id=None,
        )


def test_artifact_rejects_bad_sha() -> None:
    with pytest.raises(ValueError, match=r".+"):
        ArtifactReference(
            run_id=uuid.uuid4().hex,
            role="detail",
            path=Path("/evidence/x"),
            sha256="nope",
            byte_count=10,
            managed=True,
            evidence_id=None,
        )


def test_artifact_rejects_bool_byte_count() -> None:
    with pytest.raises(ValueError, match=r".+"):
        ArtifactReference(
            run_id=uuid.uuid4().hex,
            role="detail",
            path=Path("/evidence/x"),
            sha256="ab" * 32,
            byte_count=cast(int, True),
            managed=True,
            evidence_id=None,
        )


def test_artifact_rejects_negative_byte_count() -> None:
    with pytest.raises(ValueError, match=r".+"):
        _artifact(uuid.uuid4().hex, size=-1)


def test_artifact_rejects_non_bool_managed() -> None:
    with pytest.raises(ValueError, match=r".+"):
        ArtifactReference(
            run_id=uuid.uuid4().hex,
            role="detail",
            path=Path("/evidence/x"),
            sha256="ab" * 32,
            byte_count=10,
            managed=cast(bool, 1),
            evidence_id=None,
        )


def test_artifact_rejects_bad_evidence_id() -> None:
    with pytest.raises(ValueError, match=r".+"):
        _artifact(uuid.uuid4().hex, evidence_id="../escape")


def test_retention_policy_rejects_bool_budget() -> None:
    with pytest.raises(ValueError, match=r".+"):
        RetentionPolicy(max_detail_bytes=cast(int, True))


def test_retention_policy_rejects_non_integer_budget() -> None:
    with pytest.raises(ValueError, match=r".+"):
        RetentionPolicy(max_detail_runs=cast(int, "5"))


def test_retention_policy_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError, match=r".+"):
        RetentionPolicy(max_detail_bytes=0)


def test_retention_plan_rejects_bad_identities() -> None:
    with pytest.raises(ValueError, match=r".+"):
        RetentionPlan(
            evidence_ids=cast(tuple[str, ...], ["a"]),
            reclaimable_bytes=0,
            protected_bytes=0,
            budget_satisfied=True,
        )


def test_retention_plan_rejects_negative_bytes() -> None:
    with pytest.raises(ValueError, match=r".+"):
        RetentionPlan(evidence_ids=(), reclaimable_bytes=-1, protected_bytes=0, budget_satisfied=True)


def test_retention_plan_rejects_non_bool_feasibility() -> None:
    with pytest.raises(ValueError, match=r".+"):
        RetentionPlan(
            evidence_ids=(),
            reclaimable_bytes=0,
            protected_bytes=0,
            budget_satisfied=cast(bool, 1),
        )


def test_retention_result_rejects_bad_identities() -> None:
    with pytest.raises(ValueError, match=r".+"):
        RetentionResult(
            removed_evidence_ids=cast(tuple[str, ...], ["a"]), reclaimed_bytes=0, budget_satisfied=True
        )


def test_retention_result_rejects_negative_bytes() -> None:
    with pytest.raises(ValueError, match=r".+"):
        RetentionResult(removed_evidence_ids=(), reclaimed_bytes=-1, budget_satisfied=True)


def test_retention_result_rejects_non_bool_feasibility() -> None:
    with pytest.raises(ValueError, match=r".+"):
        RetentionResult(removed_evidence_ids=(), reclaimed_bytes=0, budget_satisfied=cast(bool, 0))


def test_utc_now_helper_returns_utc_iso8601() -> None:
    stamp = utc_now_iso8601()
    assert stamp.endswith("+00:00")
