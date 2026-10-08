"""Fail-closed equivalent-run reuse invariants for `lab process-backtest`."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from pathlib import Path

import pytest

import src.lab.mhs.app.reuse as reuse
import src.lab.mhs.app.supervisor as sup
from src.backtests.contracts import ArtifactReference, RunFinalization, RunRegistration
from src.backtests.registry import finalize_run, initialize_registry, register_run
from src.common.errors import DataIntegrityError


def _fp() -> str:
    return uuid.uuid4().hex


def _digest_size(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def _seed(
    tmp_path: Path,
    registry: Path,
    fingerprint: str,
    *,
    run_id: str | None = None,
    finalized_at: str = "2026-01-02T00:00:00+00:00",
    status: str = "completed",
    primary_valid: bool | None = True,
    terminal_certified: bool | None = True,
    with_targets: bool = True,
    detail_count: int = 2,
    with_evidence: bool = True,
    managed: bool = True,
    ensure_registry: bool = True,
) -> tuple[str, dict[str, Path]]:
    run_id = run_id or uuid.uuid4().hex
    if ensure_registry and not registry.exists():
        initialize_registry(registry)
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result = run_dir / "result.json"
    result.write_bytes(b'{"ok": true, "run": "%s"}' % run_id.encode())
    paths: dict[str, Path] = {"result": result}
    if with_targets:
        targets = run_dir / "targets.parquet"
        targets.write_bytes(b"targets-" + run_id.encode())
        paths["targets"] = targets
    evidence_id: str | None = None
    if with_evidence and detail_count > 0:
        evidence_id = f"ev-{run_id[:16]}"
        for index in range(detail_count):
            detail = run_dir / f"detail-{index}.parquet"
            detail.write_bytes(b"detail-%s-%d" % (run_id.encode(), index))
            paths[f"detail-{index}"] = detail
    register_run(
        registry,
        RunRegistration(
            run_id=run_id, strategy_id="process_inventory_3m",
            registered_at="2026-01-01T00:00:00+00:00",
            request={"fingerprint": fingerprint, "window": "3m"},
            managed_directory=run_dir if managed else None,
        ),
    )
    artifacts: list[ArtifactReference] = []
    sha, size = _digest_size(result)
    artifacts.append(
        ArtifactReference(
            run_id=run_id, role="result", path=result, sha256=sha,
            byte_count=size, managed=True, evidence_id=None,
        )
    )
    if with_targets:
        sha, size = _digest_size(paths["targets"])
        artifacts.append(
            ArtifactReference(
                run_id=run_id, role="targets", path=paths["targets"], sha256=sha,
                byte_count=size, managed=True, evidence_id=None,
            )
        )
    if evidence_id is not None:
        for index in range(detail_count):
            key = f"detail-{index}"
            sha, size = _digest_size(paths[key])
            artifacts.append(
                ArtifactReference(
                    run_id=run_id, role="detail", path=paths[key], sha256=sha,
                    byte_count=size, managed=True, evidence_id=evidence_id,
                )
            )
    outcome: dict = {"ok": True}
    if evidence_id is not None:
        outcome["evidence_id"] = evidence_id
    finalize_run(
        registry,
        RunFinalization(
            run_id=run_id, status=status, finalized_at=finalized_at,  # type: ignore[arg-type]
            primary_valid=primary_valid, terminal_certified=terminal_certified,
            outcome=outcome,
        ),
        tuple(artifacts),
    )
    return run_id, paths


def _registry_tables(registry: Path) -> set[str]:
    conn = sqlite3.connect(str(registry))
    try:
        return {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        conn.close()


def test_completed_valid_run_with_verified_artifacts_is_reused(tmp_path) -> None:
    registry = tmp_path / "registry.sqlite3"
    fingerprint = _fp()
    run_id, paths = _seed(tmp_path, registry, fingerprint)
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is not None
    assert lookup.reused.run_id == run_id
    assert lookup.reused.result_path == paths["result"]
    assert lookup.reused.targets_path == paths["targets"]
    assert lookup.reused.evidence_retained is True
    assert lookup.rejections == ()


def test_non_completed_statuses_are_rejected_for_fresh_run(tmp_path) -> None:
    registry = tmp_path / "registry.sqlite3"
    initialize_registry(registry)
    fingerprint = _fp()
    for index, status in enumerate(
        ("failed", "timed_out", "signaled", "resource_rejected", "interrupted")
    ):
        _seed(
            tmp_path, registry, fingerprint, run_id=f"{index:02x}" + "a" * 30,
            status=status, ensure_registry=False,
        )
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is None
    by_run = {rejection.run_id: rejection for rejection in lookup.rejections}
    assert len(by_run) == 5
    for rejection in lookup.rejections:
        assert rejection.reason == "status_not_completed"
        assert rejection.detail in (
            "failed", "timed_out", "signaled", "resource_rejected", "interrupted",
        )


def test_completed_but_financially_invalid_is_rejected(tmp_path) -> None:
    registry = tmp_path / "registry.sqlite3"
    initialize_registry(registry)
    fingerprint = _fp()
    _seed(tmp_path, registry, fingerprint, run_id="a" * 32, primary_valid=False, ensure_registry=False)
    _seed(tmp_path, registry, fingerprint, run_id="b" * 32, primary_valid=None, ensure_registry=False)
    _seed(
        tmp_path, registry, fingerprint, run_id="c" * 32,
        terminal_certified=False, ensure_registry=False,
    )
    _seed(
        tmp_path, registry, fingerprint, run_id="d" * 32,
        terminal_certified=None, ensure_registry=False,
    )
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is None
    by_run = {rejection.run_id: rejection for rejection in lookup.rejections}
    assert by_run["a" * 32].reason == "primary_not_valid"
    assert by_run["b" * 32].reason == "primary_not_valid"
    assert by_run["c" * 32].reason == "terminal_not_certified"
    assert by_run["d" * 32].reason == "terminal_not_certified"


def test_unfinalized_registration_is_rejected(tmp_path) -> None:
    registry = tmp_path / "registry.sqlite3"
    initialize_registry(registry)
    fingerprint = _fp()
    run_id = uuid.uuid4().hex
    register_run(
        registry,
        RunRegistration(
            run_id=run_id, strategy_id="process_inventory_3m",
            registered_at="2026-01-01T00:00:00+00:00",
            request={"fingerprint": fingerprint}, managed_directory=tmp_path,
        ),
    )
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is None
    assert len(lookup.rejections) == 1
    assert lookup.rejections[0].run_id == run_id
    assert lookup.rejections[0].reason == "not_finalized"


def test_deleted_managed_directory_is_rejected(tmp_path) -> None:
    import shutil

    registry = tmp_path / "registry.sqlite3"
    fingerprint = _fp()
    run_id, paths = _seed(tmp_path, registry, fingerprint)
    before = set((tmp_path / "runs").iterdir())
    shutil.rmtree(paths["result"].parent)
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is None
    assert len(lookup.rejections) == 1
    assert lookup.rejections[0].reason == "artifact_missing"
    assert lookup.rejections[0].detail == f"result:{paths['result']}"
    assert set((tmp_path / "runs").iterdir()) == before - {tmp_path / "runs" / run_id}
    assert not (tmp_path / "runs" / run_id).exists()


def test_tampered_result_envelope_is_rejected(tmp_path) -> None:
    registry = tmp_path / "registry.sqlite3"
    fingerprint = _fp()
    _, paths = _seed(tmp_path, registry, fingerprint, with_evidence=False, with_targets=False)
    result = paths["result"]
    original = result.read_bytes()
    same_length = bytearray(original)
    same_length[0] ^= 0x01
    result.write_bytes(bytes(same_length))
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is None
    assert lookup.rejections[0].reason == "artifact_digest_mismatch"
    result.write_bytes(original + b"x")
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is None
    assert lookup.rejections[0].reason == "artifact_size_mismatch"


def test_run_without_result_artifact_or_ownership_is_rejected(tmp_path) -> None:
    registry = tmp_path / "registry.sqlite3"
    initialize_registry(registry)
    fingerprint = _fp()
    bare_id = "a" * 32
    register_run(
        registry,
        RunRegistration(
            run_id=bare_id, strategy_id="process_inventory_3m",
            registered_at="2026-01-01T00:00:00+00:00",
            request={"fingerprint": fingerprint}, managed_directory=tmp_path / "runs" / bare_id,
        ),
    )
    finalize_run(
        registry,
        RunFinalization(
            run_id=bare_id, status="completed", finalized_at="2026-01-02T00:00:00+00:00",
            primary_valid=True, terminal_certified=True, outcome={"ok": True},
        ),
        (),
    )
    unmanaged_id, _ = _seed(tmp_path, registry, fingerprint, managed=False, ensure_registry=False)
    pruned_id, _ = _seed(tmp_path, registry, fingerprint, ensure_registry=False)
    conn = sqlite3.connect(str(registry))
    try:
        with conn:
            conn.execute(
                "UPDATE artifacts SET retained = 0 WHERE run_id = ? AND role = 'result'",
                (pruned_id,),
            )
    finally:
        conn.close()
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is None
    by_run = {rejection.run_id: rejection for rejection in lookup.rejections}
    assert by_run[bare_id].reason == "result_unrecorded"
    assert by_run[unmanaged_id].reason == "unmanaged_run"
    assert by_run[pruned_id].reason == "result_unrecorded"


def test_retention_pruned_details_remain_reusable_with_intact_result(tmp_path) -> None:
    registry = tmp_path / "registry.sqlite3"
    fingerprint = _fp()
    run_id, paths = _seed(tmp_path, registry, fingerprint)
    conn = sqlite3.connect(str(registry))
    try:
        with conn:
            conn.execute(
                "UPDATE artifacts SET retained = 0 WHERE run_id = ? AND role = 'detail'",
                (run_id,),
            )
    finally:
        conn.close()
    for key, path in paths.items():
        if key.startswith("detail-"):
            path.unlink()
    result = paths["result"]
    if not result.exists():
        pytest.fail("result must survive detail reclamation")
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is not None
    assert lookup.reused.run_id == run_id
    assert lookup.reused.evidence_retained is False


def test_retained_detail_file_missing_is_rejected(tmp_path) -> None:
    registry = tmp_path / "registry.sqlite3"
    fingerprint = _fp()
    _, paths = _seed(tmp_path, registry, fingerprint)
    detail = paths["detail-0"]
    detail.unlink()
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is None
    assert lookup.rejections[0].reason == "artifact_missing"


def test_recorded_evidence_without_artifact_rows_is_rejected(tmp_path) -> None:
    registry = tmp_path / "registry.sqlite3"
    fingerprint = _fp()
    run_id, paths = _seed(
        tmp_path, registry, fingerprint, with_evidence=False, with_targets=False,
    )
    conn = sqlite3.connect(str(registry))
    try:
        with conn:
            conn.execute(
                "UPDATE finalizations SET outcome_json = ? WHERE run_id = ?",
                (json.dumps({"ok": True, "evidence_id": "ev-missing"}), run_id),
            )
    finally:
        conn.close()
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is None
    assert lookup.rejections[0].reason == "evidence_unrecorded"


def test_missing_targets_file_is_rejected(tmp_path) -> None:
    registry = tmp_path / "registry.sqlite3"
    fingerprint = _fp()
    _, paths = _seed(tmp_path, registry, fingerprint)
    paths["targets"].unlink()
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is None
    assert lookup.rejections[0].reason == "artifact_missing"
    assert "targets:" in (lookup.rejections[0].detail or "")


def test_newest_admissible_candidate_wins(tmp_path) -> None:
    registry = tmp_path / "registry.sqlite3"
    initialize_registry(registry)
    fingerprint = _fp()
    old_id, _ = _seed(
        tmp_path, registry, fingerprint, finalized_at="2026-01-02T00:00:00+00:00",
        ensure_registry=False,
    )
    new_id, _ = _seed(
        tmp_path, registry, fingerprint, finalized_at="2026-01-03T00:00:00+00:00",
        ensure_registry=False,
    )
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is not None
    assert lookup.reused.run_id == new_id
    tie_fp = _fp()
    tie_a = "a" * 32
    tie_b = "f" * 32
    _seed(
        tmp_path, registry, tie_fp, run_id=tie_a,
        finalized_at="2026-01-04T00:00:00+00:00", ensure_registry=False,
    )
    _seed(
        tmp_path, registry, tie_fp, run_id=tie_b,
        finalized_at="2026-01-04T00:00:00+00:00", ensure_registry=False,
    )
    lookup = sup.find_reused_run(registry, tie_fp)
    assert lookup.reused is not None
    assert lookup.reused.run_id == max(tie_a, tie_b)
    assert old_id != new_id


def test_newer_rejected_run_does_not_shadow_older_admissible_run(tmp_path) -> None:
    registry = tmp_path / "registry.sqlite3"
    initialize_registry(registry)
    fingerprint = _fp()
    good_id, _ = _seed(
        tmp_path, registry, fingerprint, finalized_at="2026-01-02T00:00:00+00:00",
        ensure_registry=False,
    )
    bad_id, _ = _seed(
        tmp_path, registry, fingerprint, finalized_at="2026-01-03T00:00:00+00:00",
        status="failed", ensure_registry=False,
    )
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is not None
    assert lookup.reused.run_id == good_id
    assert [rejection.run_id for rejection in lookup.rejections] == [bad_id]
    assert lookup.rejections[0].reason == "status_not_completed"


def test_candidates_older_than_admitted_run_are_not_examined(tmp_path) -> None:
    registry = tmp_path / "registry.sqlite3"
    initialize_registry(registry)
    fingerprint = _fp()
    stale_id, stale_paths = _seed(
        tmp_path, registry, fingerprint, finalized_at="2026-01-02T00:00:00+00:00",
        ensure_registry=False,
    )
    stale_paths["result"].unlink()
    _seed(
        tmp_path, registry, fingerprint, finalized_at="2026-01-03T00:00:00+00:00",
        ensure_registry=False,
    )
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is not None
    assert lookup.rejections == ()
    assert stale_id not in [rejection.run_id for rejection in lookup.rejections]


def test_non_matching_and_legacy_rows_are_ignored(tmp_path) -> None:
    registry = tmp_path / "registry.sqlite3"
    initialize_registry(registry)
    register_run(
        registry,
        RunRegistration(
            run_id="a" * 32, strategy_id="process_inventory_3m",
            registered_at="2026-01-01T00:00:00+00:00",
            request={"fingerprint": "other", "window": "3m"},
            managed_directory=tmp_path,
        ),
    )
    register_run(
        registry,
        RunRegistration(
            run_id="b" * 32, strategy_id="process_inventory_3m",
            registered_at="2026-01-01T00:00:00+00:00",
            request={"window": "3m"}, managed_directory=tmp_path,
        ),
    )
    register_run(
        registry,
        RunRegistration(
            run_id="c" * 32, strategy_id="process_inventory_3m",
            registered_at="2026-01-01T00:00:00+00:00",
            request={"fingerprint": 123}, managed_directory=tmp_path,
        ),
    )
    lookup = sup.find_reused_run(registry, "missing-fingerprint")
    assert lookup.reused is None
    assert lookup.rejections == ()


def test_absent_or_uninitialized_registry_yields_empty_lookup(tmp_path) -> None:
    fingerprint = _fp()
    missing = tmp_path / "missing.sqlite3"
    assert sup.find_reused_run(missing, fingerprint) == sup.ReuseLookup(None, ())
    bare = tmp_path / "bare.sqlite3"
    conn = sqlite3.connect(str(bare))
    try:
        conn.execute("CREATE TABLE unrelated (a TEXT)")
        conn.commit()
    finally:
        conn.close()
    before = _registry_tables(bare)
    assert sup.find_reused_run(bare, fingerprint) == sup.ReuseLookup(None, ())
    assert _registry_tables(bare) == before


def test_corrupt_registry_raises_data_integrity_error(tmp_path) -> None:
    fingerprint = _fp()
    garbage = tmp_path / "garbage.sqlite3"
    garbage.write_text("not a database", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        sup.find_reused_run(garbage, fingerprint)
    directory = tmp_path / "as-directory"
    directory.mkdir()
    with pytest.raises(DataIntegrityError):
        sup.find_reused_run(directory, fingerprint)
    partial = tmp_path / "partial.sqlite3"
    conn = sqlite3.connect(str(partial))
    try:
        conn.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, request_json TEXT, managed_directory TEXT)")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError):
        sup.find_reused_run(partial, fingerprint)
    bad_json = tmp_path / "badjson.sqlite3"
    initialize_registry(bad_json)
    register_run(
        bad_json,
        RunRegistration(
            run_id="a" * 32, strategy_id="process_inventory_3m",
            registered_at="2026-01-01T00:00:00+00:00",
            request={"fingerprint": "other"}, managed_directory=tmp_path,
        ),
    )
    conn = sqlite3.connect(str(bad_json))
    try:
        with conn:
            conn.execute("UPDATE runs SET request_json = 'not-json' WHERE run_id = ?", ("a" * 32,))
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError):
        sup.find_reused_run(bad_json, "other")
    bad_status = tmp_path / "badstatus.sqlite3"
    bad_fp = _fp()
    _seed(tmp_path, bad_status, bad_fp)
    conn = sqlite3.connect(str(bad_status))
    try:
        with conn:
            conn.execute("UPDATE finalizations SET status = 'bogus'")
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError):
        sup.find_reused_run(bad_status, bad_fp)
    dup = tmp_path / "dup.sqlite3"
    dup_fp = _fp()
    dup_id, _ = _seed(tmp_path, dup, dup_fp, with_evidence=False, with_targets=False)
    conn = sqlite3.connect(str(dup))
    try:
        row = conn.execute(
            "SELECT role, path, sha256, byte_count, managed, evidence_id, retained FROM artifacts WHERE run_id = ?",
            (dup_id,),
        ).fetchone()
        with conn:
            conn.execute(
                "INSERT INTO artifacts (run_id, role, path, sha256, byte_count, managed, evidence_id, retained)"
                " VALUES (?, 'result', ?, ?, ?, ?, ?, 1)",
                (dup_id, str(tmp_path / "runs" / dup_id / "result-copy.json"), row[2], row[3], row[4], row[5]),
            )
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError):
        sup.find_reused_run(dup, dup_fp)


def test_lookup_is_side_effect_free(tmp_path) -> None:
    registry = tmp_path / "registry.sqlite3"
    initialize_registry(registry)
    fingerprint = _fp()
    good_id, _ = _seed(
        tmp_path, registry, fingerprint, finalized_at="2026-01-02T00:00:00+00:00",
        ensure_registry=False,
    )
    bad_id, bad_paths = _seed(
        tmp_path, registry, fingerprint, finalized_at="2026-01-03T00:00:00+00:00",
        status="failed", ensure_registry=False,
    )
    assert bad_paths["result"].exists()

    def _snapshot() -> tuple:
        conn = sqlite3.connect(str(registry))
        try:
            runs = conn.execute("SELECT * FROM runs ORDER BY run_id").fetchall()
            finals = conn.execute("SELECT * FROM finalizations ORDER BY run_id").fetchall()
            artifacts = conn.execute("SELECT * FROM artifacts ORDER BY run_id, role, path").fetchall()
            return runs, finals, artifacts
        finally:
            conn.close()

    def _file_state() -> dict:
        state = {}
        for path in sorted((tmp_path / "runs").rglob("*")):
            if path.is_file():
                stat = path.stat()
                state[str(path)] = (path.read_bytes(), stat.st_mtime_ns)
        return state

    before_rows = _snapshot()
    before_files = _file_state()
    first = sup.find_reused_run(registry, fingerprint)
    mid_rows = _snapshot()
    second = sup.find_reused_run(registry, fingerprint)
    assert first == second
    assert first.reused is not None
    assert first.reused.run_id == good_id
    assert [rejection.run_id for rejection in first.rejections] == [bad_id]
    assert mid_rows == before_rows
    assert _snapshot() == before_rows
    assert _file_state() == before_files


def test_rejections_emit_structured_data_warnings(tmp_path, caplog) -> None:
    registry = tmp_path / "registry.sqlite3"
    fingerprint = _fp()
    run_id, _ = _seed(tmp_path, registry, fingerprint, status="failed")
    with caplog.at_level(logging.WARNING, logger="src.lab.mhs.app.supervisor"):
        lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is None
    matches = [record for record in caplog.records if record.getMessage().startswith("[DATA] reuse_rejected")]
    assert len(matches) == 1
    assert run_id in matches[0].getMessage()
    assert lookup.rejections[0].reason in matches[0].getMessage()


def test_corrupt_finalized_at_variants_raise(tmp_path) -> None:
    for bad in ("", "not-a-date", "2026-01-02T00:00:00", "2026-01-02T01:00:00+01:00"):
        registry = tmp_path / f"fin-{abs(hash(bad)) % 10_000}.sqlite3"
        fingerprint = _fp()
        _seed(tmp_path, registry, fingerprint)
        conn = sqlite3.connect(str(registry))
        try:
            with conn:
                conn.execute("UPDATE finalizations SET finalized_at = ?", (bad,))
        finally:
            conn.close()
        with pytest.raises(DataIntegrityError):
            sup.find_reused_run(registry, fingerprint)


def test_zulu_finalized_at_is_accepted(tmp_path) -> None:
    registry = tmp_path / "zulu.sqlite3"
    fingerprint = _fp()
    run_id, _ = _seed(tmp_path, registry, fingerprint)
    conn = sqlite3.connect(str(registry))
    try:
        with conn:
            conn.execute(
                "UPDATE finalizations SET finalized_at = ? WHERE run_id = ?",
                ("2026-01-02T00:00:00Z", run_id),
            )
    finally:
        conn.close()
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is not None
    assert lookup.reused.run_id == run_id


def test_missing_tables_and_columns_raise(tmp_path) -> None:
    fingerprint = _fp()
    no_finals = tmp_path / "nofinals.sqlite3"
    initialize_registry(no_finals)
    register_run(
        no_finals,
        RunRegistration(
            run_id="a" * 32, strategy_id="s", registered_at="2026-01-01T00:00:00+00:00",
            request={"fingerprint": fingerprint}, managed_directory=tmp_path,
        ),
    )
    conn = sqlite3.connect(str(no_finals))
    try:
        with conn:
            conn.execute("DROP TABLE finalizations")
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError):
        sup.find_reused_run(no_finals, fingerprint)
    for table, column in (
        ("runs", "managed_directory"), ("finalizations", "outcome_json"), ("artifacts", "managed"),
    ):
        registry = tmp_path / f"nocol-{table}.sqlite3"
        fingerprint = _fp()
        _seed(tmp_path, registry, fingerprint)
        conn = sqlite3.connect(str(registry))
        try:
            with conn:
                conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        finally:
            conn.close()
        with pytest.raises(DataIntegrityError):
            sup.find_reused_run(registry, fingerprint)


def test_non_object_request_raises(tmp_path) -> None:
    registry = tmp_path / "listreq.sqlite3"
    initialize_registry(registry)
    register_run(
        registry,
        RunRegistration(
            run_id="a" * 32, strategy_id="s", registered_at="2026-01-01T00:00:00+00:00",
            request={"fingerprint": "x"}, managed_directory=tmp_path,
        ),
    )
    conn = sqlite3.connect(str(registry))
    try:
        with conn:
            conn.execute("UPDATE runs SET request_json = '[1, 2]' WHERE run_id = ?", ("a" * 32,))
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError):
        sup.find_reused_run(registry, "x")


@pytest.mark.parametrize(
    "setup",
    ["status", "primary", "terminal", "outcome_json", "outcome_list", "evidence_empty", "evidence_int"],
)
def test_corrupt_finalization_domains_raise(tmp_path, setup: str) -> None:
    registry = tmp_path / f"corrupt-fin-{setup}.sqlite3"
    fingerprint = _fp()
    _seed(tmp_path, registry, fingerprint)
    conn = sqlite3.connect(str(registry))
    try:
        with conn:
            if setup == "status":
                conn.execute("UPDATE finalizations SET status = 'bogus'")
            elif setup == "primary":
                conn.execute("UPDATE finalizations SET primary_valid = 2")
            elif setup == "terminal":
                conn.execute("UPDATE finalizations SET terminal_certified = 2")
            elif setup == "outcome_json":
                conn.execute("UPDATE finalizations SET outcome_json = 'not-json'")
            elif setup == "outcome_list":
                conn.execute("UPDATE finalizations SET outcome_json = '[1, 2]'")
            elif setup == "evidence_empty":
                conn.execute(
                    "UPDATE finalizations SET outcome_json = ?",
                    (json.dumps({"evidence_id": ""}),),
                )
            else:
                conn.execute(
                    "UPDATE finalizations SET outcome_json = ?",
                    (json.dumps({"evidence_id": 123}),),
                )
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError):
        sup.find_reused_run(registry, fingerprint)


@pytest.mark.parametrize(
    "setup",
    ["path", "null_byte_path", "sha", "byte_count", "retained"],
)
def test_corrupt_artifact_domains_raise(tmp_path, setup: str) -> None:
    registry = tmp_path / f"corrupt-art-{setup}.sqlite3"
    fingerprint = _fp()
    _seed(tmp_path, registry, fingerprint, with_evidence=False, with_targets=False)
    conn = sqlite3.connect(str(registry))
    try:
        with conn:
            if setup == "path":
                conn.execute("UPDATE artifacts SET path = 'relative/result.json'")
            elif setup == "null_byte_path":
                conn.execute("UPDATE artifacts SET path = ?", ("/result\x00.json",))
            elif setup == "sha":
                conn.execute("UPDATE artifacts SET sha256 = 'zz'")
            elif setup == "byte_count":
                conn.execute("UPDATE artifacts SET byte_count = -1")
            else:
                conn.execute("UPDATE artifacts SET retained = 2")
    finally:
        conn.close()
    with pytest.raises(DataIntegrityError):
        sup.find_reused_run(registry, fingerprint)


def test_filesystem_edge_cases_reject_without_read(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "fsedge.sqlite3"
    fingerprint = _fp()
    _, paths = _seed(tmp_path, registry, fingerprint, with_evidence=False, with_targets=False)
    result = paths["result"]
    result.unlink()
    result.mkdir()
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is None
    assert lookup.rejections[0].reason == "artifact_missing"
    result.rmdir()
    result.write_bytes(b"x" * 8)
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    conn = sqlite3.connect(str(registry))
    try:
        with conn:
            conn.execute(
                "UPDATE artifacts SET path = ?", (str(blocker / "child.json"),),
            )
    finally:
        conn.close()
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is None
    assert lookup.rejections[0].reason == "artifact_missing"


def test_unreadable_artifact_paths_reject(tmp_path, monkeypatch) -> None:
    import os as _os

    registry = tmp_path / "unread.sqlite3"
    fingerprint = _fp()
    _seed(tmp_path, registry, fingerprint, with_evidence=False, with_targets=False)
    real_stat = _os.stat

    def _denied(path, *args: object, **kwargs: object) -> object:
        if "/runs/" in str(path):
            raise PermissionError("injected stat boom")
        return real_stat(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_os, "stat", _denied)
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is None
    assert lookup.rejections[0].reason == "artifact_unreadable"
    monkeypatch.setattr(_os, "stat", real_stat)
    monkeypatch.setattr(reuse, "_hash_file", lambda path: (_ for _ in ()).throw(OSError("hash boom")))
    lookup = sup.find_reused_run(registry, fingerprint)
    assert lookup.reused is None
    assert lookup.rejections[0].reason == "artifact_unreadable"


def test_registry_connect_and_lock_errors(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "registry.sqlite3"
    fingerprint = _fp()
    _seed(tmp_path, registry, fingerprint)
    real_connect = sqlite3.connect

    def _raise_operational(*args, **kwargs) -> object:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(sqlite3, "connect", _raise_operational)
    with pytest.raises(sqlite3.OperationalError):
        sup.find_reused_run(registry, fingerprint)

    def _raise_database(*args, **kwargs) -> object:
        raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(sqlite3, "connect", _raise_database)
    with pytest.raises(DataIntegrityError):
        sup.find_reused_run(registry, fingerprint)
    monkeypatch.setattr(sqlite3, "connect", real_connect)

    class _LockedConnection:
        def __init__(self, real: sqlite3.Connection) -> None:
            self._real = real

        def execute(self, *args: object, **kwargs: object) -> object:
            raise sqlite3.OperationalError("database is locked")

        def close(self) -> None:
            self._real.close()

    real = real_connect(str(registry), timeout=5.0)
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: _LockedConnection(real))
    try:
        with pytest.raises(sqlite3.OperationalError):
            sup.find_reused_run(registry, fingerprint)
    finally:
        monkeypatch.setattr(sqlite3, "connect", real_connect)
        real.close()


def test_registry_removed_before_connect_is_not_recreated(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "registry.sqlite3"
    fingerprint = _fp()
    _seed(tmp_path, registry, fingerprint)
    real_connect = sqlite3.connect

    def _remove_before_connect(*args, **kwargs):
        registry.unlink()
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _remove_before_connect)
    with pytest.raises(sqlite3.OperationalError):
        sup.find_reused_run(registry, fingerprint)
    assert not registry.exists()


def test_concurrent_finalization_cannot_tear_lookup_snapshot(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "registry.sqlite3"
    fingerprint = _fp()
    initialize_registry(registry)
    run_id = "a" * 32
    result = tmp_path / "result.json"
    result.write_bytes(b"verified result")
    register_run(
        registry,
        RunRegistration(
            run_id=run_id, strategy_id="s", registered_at="2026-01-01T00:00:00Z",
            request={"fingerprint": fingerprint}, managed_directory=tmp_path,
        ),
    )
    digest, size = _digest_size(result)
    artifact = ArtifactReference(
        run_id=run_id, role="result", path=result, sha256=digest,
        byte_count=size, managed=True, evidence_id=None,
    )
    finalization = RunFinalization(
        run_id=run_id, status="completed", finalized_at="2026-01-02T00:00:00Z",
        primary_valid=True, terminal_certified=True, outcome={},
    )
    real_connect = sqlite3.connect
    published = False

    class _ConcurrentConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            nonlocal published
            cursor = super().execute(sql, parameters)
            if sql.startswith("SELECT run_id, request_json") and not published:
                finalize_run(registry, finalization, (artifact,))
                published = True
            return cursor

    def _connect(*args, **kwargs):
        if kwargs.get("uri"):
            kwargs["factory"] = _ConcurrentConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", _connect)
    lookup = sup.find_reused_run(registry, fingerprint)
    assert published
    assert lookup.reused is None
    assert lookup.rejections == (sup.ReuseRejection(run_id, "not_finalized", None),)
    next_lookup = sup.find_reused_run(registry, fingerprint)
    assert next_lookup.reused is not None
    assert next_lookup.reused.run_id == run_id
