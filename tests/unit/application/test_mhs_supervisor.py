"""Invariant scenarios for the source-owned supervised runner."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import time
from pathlib import Path

import pandas as pd
import pytest

import src.application.mhs_supervisor as sup


def _stamps():
    return pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-01-04", tz="UTC")


def _result(tmp_path: Path, name: str = "run") -> Path:
    return tmp_path / name / "result.json"


def _registry(tmp_path: Path) -> Path:
    return tmp_path / "registry.sqlite3"


class _FakeProc:
    def __init__(self, returncode: int, delay: float = 0.0, on_start=None) -> None:
        self.pid = 987654
        self._returncode = returncode
        self._delay = delay
        self._start = time.monotonic()
        self._on_start = on_start
        self.terminated = False
        if on_start is not None:
            on_start(self)

    def poll(self):
        if self.terminated:
            return self._returncode
        if time.monotonic() - self._start >= self._delay:
            return self._returncode
        return None

    def wait(self, timeout=None):
        if self.terminated:
            return self._returncode
        deadline = None if timeout is None else time.monotonic() + timeout
        while time.monotonic() - self._start < self._delay:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(cmd=[], timeout=timeout)
            time.sleep(0.005)
        return self._returncode


def _install_fake(monkeypatch, returncode=0, delay=0.0, on_start=None):
    def _factory(*args, **kwargs):
        return _FakeProc(returncode, delay, on_start)

    monkeypatch.setattr(subprocess, "Popen", _factory)
    monkeypatch.setattr(sup, "_gnu_time_prefix", lambda: [])
    monkeypatch.setattr(sup, "_parse_gnu_metrics", lambda *a, **k: (None, None))
    monkeypatch.setattr(sup, "_workload_memory", lambda pid: (100, 50, 0))
    monkeypatch.setattr(sup, "current_mhs_headroom_bytes", lambda: 8 * 2**30)
    monkeypatch.setattr(sup, "_terminate_group", lambda proc, timeout=5.0: setattr(proc, "terminated", True))


def _completed_domain() -> dict:
    return {
        "status": "completed",
        "execution_timeframe": "3m",
        "base": {"terminal": {"primary_valid": True, "terminal_certified": True}},
        "stress": {"terminal": {"primary_valid": True, "terminal_certified": True}},
    }


def _failed_domain() -> dict:
    return {"status": "failed", "error_code": "DATA_INTEGRITY", "stage": "process_execution_piece"}


def test_completed_envelope_combines_domains(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path)

    def _on_start(proc):
        staging = result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(_completed_domain()))

    _install_fake(monkeypatch, 0, 0.0, _on_start)
    run = sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05, registry_path=_registry(tmp_path))
    assert run.status == "completed"
    envelope = json.loads(result.read_text(encoding="utf-8"))
    assert envelope["schema_version"] == 1
    assert envelope["execution"]["status"] == "completed"
    assert envelope["financial"]["status"] == "completed"
    assert envelope["run"]["run_id"] == run.run_id


def test_failure_envelope_is_single_path(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path)

    def _on_start(proc):
        staging = result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(_failed_domain()))

    _install_fake(monkeypatch, 1, 0.0, _on_start)
    run = sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05, registry_path=_registry(tmp_path))
    assert run.status == "failed"
    assert result.is_file()
    assert not (result.parent / "failure.json").exists()
    assert not (result.parent / "primary.json").exists()
    assert not (result.parent / "run.json").exists()
    envelope = json.loads(result.read_text(encoding="utf-8"))
    assert envelope["financial"]["error_code"] == "DATA_INTEGRITY"


def test_successful_log_is_transient(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path)

    def _on_start(proc):
        staging = result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(_completed_domain()))

    _install_fake(monkeypatch, 0, 0.0, _on_start)
    run = sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05, registry_path=_registry(tmp_path))
    assert run.status == "completed"
    log = result.parent / "result.log"
    assert not log.exists()
    assert result.is_file()


def test_failed_log_is_diagnostic(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path, "bad")

    def _on_start(proc):
        staging = result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(_failed_domain()))
        (result.parent / "bad.log").write_text("boom\n" * 10)

    _install_fake(monkeypatch, 1, 0.0, _on_start)
    run = sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05, registry_path=_registry(tmp_path))
    assert run.status == "failed"
    log = result.parent / "result.log"
    assert log.is_file()
    assert log.stat().st_size <= sup._BOUNDED_LOG_MAX_BYTES + 1024


def test_retention_protects_envelope_and_shared_evidence(tmp_path) -> None:
    from src.backtests.contracts import ArtifactReference, RetentionPolicy, RunFinalization, RunRegistration
    from src.backtests.registry import finalize_run, initialize_registry, register_run, set_run_protection
    from src.backtests.retention import apply_retention, plan_retention

    registry = tmp_path / "registry.sqlite3"
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    initialize_registry(registry)
    for run_id, resolved in (("a" * 32, False), ("b" * 32, True)):
        register_run(
            registry,
            RunRegistration(
                run_id=run_id, strategy_id="process_inventory_3m",
                registered_at="2026-01-01T00:00:00+00:00",
                request={"window": "3m"}, managed_directory=None,
            ),
        )
        finalize_run(
            registry,
            RunFinalization(
                run_id=run_id, status="completed", finalized_at="2026-01-02T00:00:00+00:00",
                primary_valid=True, terminal_certified=resolved, outcome={"ok": True},
            ),
            (
                ArtifactReference(
                    run_id=run_id, role="result", path=tmp_path / f"{run_id}.json",
                    sha256="a" * 64, byte_count=10, managed=False, evidence_id=None,
                ),
            ),
        )
    shared = "shared-evidence"
    (evidence_root / shared).mkdir()
    (evidence_root / shared / "manifest.json").write_text("{}", encoding="utf-8")
    eligible = "eligible-evidence"
    (evidence_root / eligible).mkdir()
    (evidence_root / eligible / "manifest.json").write_text("{}", encoding="utf-8")
    set_run_protection(registry, "b" * 32, pinned=False, resolved=True, deployment_referenced=False)
    conn = sqlite3.connect(str(registry))
    try:
        with conn:
            for run_id, evidence_id in (("a" * 32, shared), ("b" * 32, shared), ("b" * 32, eligible)):
                conn.execute(
                    "INSERT INTO artifacts (run_id, role, path, sha256, byte_count, managed, evidence_id, retained)"
                    " VALUES (?, ?, ?, ?, ?, 1, ?, 1)",
                    (run_id, "detail", str(evidence_root / evidence_id / "x.parquet"), "b" * 64, 100, evidence_id),
                )
    finally:
        conn.close()
    plan = plan_retention(registry, evidence_root, RetentionPolicy(max_detail_bytes=1, max_detail_runs=1))
    assert shared not in plan.evidence_ids
    assert eligible in plan.evidence_ids
    result = apply_retention(registry, evidence_root, plan)
    assert shared not in result.removed_evidence_ids
    assert eligible in result.removed_evidence_ids


def test_atomic_publication_rejects_occupied_output(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path)
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text("{}", encoding="utf-8")

    def _boom(*args, **kwargs):
        raise AssertionError("worker must not start")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    with pytest.raises(ValueError, match="fresh"):
        sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05, registry_path=_registry(tmp_path))
    assert json.loads(result.read_text(encoding="utf-8")) == {}
    link = tmp_path / "link" / "result.json"
    link.parent.mkdir(parents=True, exist_ok=True)
    import os

    os.symlink(tmp_path / "missing.json", link)
    with pytest.raises(ValueError, match="fresh"):
        sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=link, poll_seconds=0.05, registry_path=_registry(tmp_path))


def test_supervised_signal_and_timeout_stay_distinct(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path, "sig")
    _install_fake(monkeypatch, -15, 0.0)
    run = sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05, registry_path=_registry(tmp_path))
    assert run.status == "signaled"
    assert run.signal_number == 15
    result2 = _result(tmp_path, "timeout")
    _install_fake(monkeypatch, 0, 30.0)
    run2 = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result2, poll_seconds=0.05, timeout_seconds=0.2,
        registry_path=_registry(tmp_path),
    )
    assert run2.status == "timed_out"


def test_supervised_resource_stop_without_success_claim(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    for label in ("pss", "headroom", "swap"):
        result = _result(tmp_path, f"res_{label}")
        _install_fake(monkeypatch, 0, 5.0)
        if label == "pss":
            monkeypatch.setattr(sup, "_workload_memory", lambda pid: (10 * 2**30, 1, 0))
        elif label == "headroom":
            monkeypatch.setattr(sup, "current_mhs_headroom_bytes", lambda: 100)
        else:
            calls = {"n": 0}

            def _swap(_calls=calls):
                _calls["n"] += 1
                return (100, 50, 0 if _calls["n"] < 2 else 10**9)

            monkeypatch.setattr(sup, "_workload_memory", lambda pid: _swap())
        run = sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05, registry_path=_registry(tmp_path))
        assert run.status == "resource_rejected"


def test_fingerprint_helpers_are_deterministic(tmp_path) -> None:
    start, end = _stamps()
    first = sup.request_fingerprint(start=start, end=end, data_root=None, tracking_error_threshold=None)
    second = sup.request_fingerprint(start=start, end=end, data_root=None, tracking_error_threshold=None)
    assert first == second
    assert sup.find_reused_run(tmp_path / "missing.sqlite3", first) is None


def test_invalid_run_identity_rejected_before_launch(tmp_path, monkeypatch) -> None:
    import subprocess as _subprocess

    def _no_launch(*args, **kwargs):
        raise AssertionError("worker must not launch")

    monkeypatch.setattr(_subprocess, "Popen", _no_launch)
    start, end = _stamps()
    with pytest.raises(ValueError, match="UUID"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=_result(tmp_path),
            poll_seconds=0.05, run_id="bogus", registry_path=_registry(tmp_path),
        )
    with pytest.raises(ValueError, match="result_output"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=tmp_path / "bad.csv",  # type: ignore[arg-type]
            poll_seconds=0.05, registry_path=_registry(tmp_path),
        )
    with pytest.raises(ValueError, match="retention_policy"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=_result(tmp_path),
            poll_seconds=0.05, retention_policy="policy",  # type: ignore[arg-type]
            registry_path=_registry(tmp_path),
        )


def test_code_identity_null_when_worker_unreadable(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path, "noid")

    def _boom(*args, **kwargs):
        raise AssertionError("worker must not launch")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    real_read_bytes = Path.read_bytes

    def _read_boom(self, *args, **kwargs):
        if self.name == "mhs_worker.py":
            raise OSError("injected provenance boom")
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _read_boom)
    with pytest.raises(ValueError, match="source identity"):
        sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05, registry_path=_registry(tmp_path))
    assert sup._code_identity() is None
    assert not result.exists()


def test_find_reused_run_skips_unusable_registry_entries(tmp_path) -> None:
    import sqlite3 as _sqlite3

    start, end = _stamps()
    fingerprint = sup.request_fingerprint(start=start, end=end, data_root=None, tracking_error_threshold=None)
    assert sup.find_reused_run(tmp_path / "absent.sqlite3", fingerprint) is None
    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_text("not a database", encoding="utf-8")
    assert sup.find_reused_run(corrupt, fingerprint) is None
    empty = tmp_path / "empty.sqlite3"
    conn = _sqlite3.connect(str(empty))
    try:
        conn.execute("CREATE TABLE t (a TEXT)")
        conn.commit()
    finally:
        conn.close()
    assert sup.find_reused_run(empty, fingerprint) is None
    registry = tmp_path / "registry.sqlite3"
    from src.backtests.contracts import RunFinalization, RunRegistration
    from src.backtests.registry import finalize_run, initialize_registry, register_run

    initialize_registry(registry)
    register_run(
        registry,
        RunRegistration(
            run_id="c" * 32, strategy_id="process_inventory_3m",
            registered_at="2026-01-01T00:00:00+00:00",
            request={"fingerprint": "other", "window": "3m"}, managed_directory=None,
        ),
    )
    assert sup.find_reused_run(registry, fingerprint) is None
    register_run(
        registry,
        RunRegistration(
            run_id="d" * 32, strategy_id="process_inventory_3m",
            registered_at="2026-01-01T00:00:00+00:00",
            request={"fingerprint": fingerprint, "window": "3m"}, managed_directory=None,
        ),
    )
    assert sup.find_reused_run(registry, fingerprint) is None
    finalize_run(
        registry,
        RunFinalization(
            run_id="d" * 32, status="completed", finalized_at="2026-01-02T00:00:00+00:00",
            primary_valid=True, terminal_certified=True, outcome={"ok": True},
        ),
        (),
    )
    found = sup.find_reused_run(registry, fingerprint)
    assert found is not None
    assert found[0] == "d" * 32
    conn = _sqlite3.connect(str(registry))
    try:
        with conn:
            conn.execute("UPDATE runs SET request_json = 'not-json' WHERE run_id = ?", ("d" * 32,))
    finally:
        conn.close()
    assert sup.find_reused_run(registry, fingerprint) is None


def test_corrupt_and_non_mapping_domain_stay_null(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    for name, payload in (("corrupt", "{not-json"), ("listed", "[1, 2]")):
        result = _result(tmp_path, name)

        def _on_start(proc, _payload=payload, _result=result):
            staging = _result.parent / ".staging_domain.json"
            staging.parent.mkdir(parents=True, exist_ok=True)
            staging.write_text(_payload)

        _install_fake(monkeypatch, 0, 0.0, _on_start)
        run = sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05, registry_path=_registry(tmp_path))
        assert run.status == "failed"
        envelope = json.loads(result.read_text(encoding="utf-8"))
        assert envelope["financial"] is None


def test_mixed_validity_and_missing_bundle_finalize(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path, "mixed")
    domain = {
        "status": "completed", "execution_timeframe": "3m",
        "base": {"terminal": {"primary_valid": True, "terminal_certified": True}},
        "stress": {"terminal": {"primary_valid": False, "terminal_certified": False}},
        "evidence_id": "a" * 64,
    }

    def _on_start(proc):
        staging = result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(domain))

    _install_fake(monkeypatch, 0, 0.0, _on_start)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result,
        targets_output=result.parent / "targets.parquet", poll_seconds=0.05,
        registry_path=_registry(tmp_path),
    )
    assert run.status == "completed"
    envelope = json.loads(result.read_text(encoding="utf-8"))
    assert envelope["financial"]["evidence_id"] == "a" * 64


def test_detail_artifacts_reference_verified_bundles(tmp_path) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_id = "b" * 64
    bundle = evidence_root / evidence_id
    bundle.mkdir(parents=True)
    payload = b"x" * 16
    (bundle / "detail.parquet").write_bytes(payload)
    import hashlib as _hashlib
    import json as _json

    digest = _hashlib.sha256(payload).hexdigest()
    (bundle / "manifest.json").write_text(
        _json.dumps({"files": [{"role": "detail", "name": "detail.parquet", "sha256": digest, "size": len(payload)}]}),
        encoding="utf-8",
    )
    refs = sup._detail_artifacts("c" * 32, evidence_root, evidence_id)
    assert len(refs) == 1
    assert refs[0].evidence_id == evidence_id
    assert sup._detail_artifacts("c" * 32, evidence_root, "missing") == []
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    assert sup._detail_artifacts("c" * 32, evidence_root, evidence_id) == []


def test_bounded_log_handles_missing_and_oversized(tmp_path) -> None:
    assert sup._retain_bounded_log(tmp_path / "absent.log") is None
    big = tmp_path / "big.log"
    big.write_bytes(b"y" * (sup._BOUNDED_LOG_MAX_BYTES + 1024))
    assert sup._retain_bounded_log(big) == big
    assert big.stat().st_size == sup._BOUNDED_LOG_MAX_BYTES
    assert sup._read_domain_payload(tmp_path / "absent.json") is None
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{bad", encoding="utf-8")
    assert sup._read_domain_payload(corrupt) is None
    listed = tmp_path / "listed.json"
    listed.write_text("[1, 2]", encoding="utf-8")
    assert sup._read_domain_payload(listed) is None


def test_gnu_metrics_helpers(tmp_path) -> None:
    assert isinstance(sup._gnu_time_prefix(), list)
    assert sup._parse_gnu_metrics(tmp_path / "absent.log") == (None, None)
    plain = tmp_path / "plain.log"
    plain.write_text("nothing here\n", encoding="utf-8")
    assert sup._parse_gnu_metrics(plain) == (None, None)
    good = tmp_path / "good.log"
    good.write_text("MHS_GNU_TIME elapsed=1.5 user=1.0 sys=0.5 maxrss=2048\n", encoding="utf-8")
    assert sup._parse_gnu_metrics(good) == (1.5, 2048 * 1024)
    bad = tmp_path / "bad.log"
    bad.write_text("MHS_GNU_TIME elapsed=e user=+ sys=- maxrss=7\n", encoding="utf-8")
    assert sup._parse_gnu_metrics(bad) == (None, None)


def test_failure_resource_code_reads_worker_evidence(tmp_path) -> None:
    assert sup._failure_resource_code(tmp_path / "absent.json") is None
    garbage = tmp_path / "garbage.json"
    garbage.write_text("not json", encoding="utf-8")
    assert sup._failure_resource_code(garbage) is None
    listed = tmp_path / "list.json"
    listed.write_text("[1, 2]", encoding="utf-8")
    assert sup._failure_resource_code(listed) is None
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    assert sup._failure_resource_code(plain) is None
    typed = tmp_path / "typed.json"
    typed.write_text(json.dumps({"status": "failed", "error_code": "SWAP_GROWTH"}), encoding="utf-8")
    assert sup._failure_resource_code(typed) == "SWAP_GROWTH"


def test_primary_completed_reads_new_completed_artifact(tmp_path) -> None:
    assert sup._primary_completed(tmp_path / "absent.json") is False
    garbage = tmp_path / "garbage.json"
    garbage.write_text("not json", encoding="utf-8")
    assert sup._primary_completed(garbage) is False
    listed = tmp_path / "list.json"
    listed.write_text("[1, 2]", encoding="utf-8")
    assert sup._primary_completed(listed) is False
    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    assert sup._primary_completed(failed) is False
    completed = tmp_path / "completed.json"
    completed.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    assert sup._primary_completed(completed) is True


def test_validate_positive_interval_rejects_non_finite() -> None:
    for bad in (True, False, 0, -1.0, float("nan"), float("inf"), "fast", None):
        with pytest.raises(ValueError, match=r".+"):
            sup._validate_positive_interval(bad, "poll_seconds")
    sup._validate_positive_interval(0.25, "poll_seconds")
    sup._validate_positive_interval(2, "timeout_seconds")


def test_terminate_group_handles_missing_and_live_groups(monkeypatch) -> None:
    proc = _FakeProc(0, 0.0)
    sup._terminate_group(proc)
    assert proc.poll() == 0
    calls: list = []
    monkeypatch.setattr(sup.os, "getpgid", lambda pid: 4321)
    monkeypatch.setattr(sup.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))

    class _Scripted:
        def __init__(self, polls) -> None:
            self.pid = 111
            self._polls = list(polls)
            self.waited = None

        def poll(self):
            return self._polls.pop(0) if self._polls else 0

        def wait(self, timeout=None):
            self.waited = timeout
            return 0

    graceful = _Scripted([None, 0])
    sup._terminate_group(graceful)
    assert (4321, sup.signal.SIGTERM) in calls
    assert sup.signal.SIGKILL not in [sig for _, sig in calls]
    calls.clear()
    stuck = _Scripted([None] * 100)
    sup._terminate_group(stuck, timeout=0.0)
    assert (4321, sup.signal.SIGKILL) in calls
    assert stuck.waited == 5.0


def test_atomic_write_json_cleans_up_on_replace_failure(tmp_path, monkeypatch) -> None:
    target = tmp_path / "run.json"

    def _boom(src, dst):
        raise OSError("replace unavailable")

    monkeypatch.setattr(sup.os, "replace", _boom)
    with pytest.raises(OSError, match="replace unavailable"):
        sup._atomic_write_json(target, {"a": 1})
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def _ensure_evidence(db: Path) -> Path:
    root = db.resolve().parent / "evidence"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _operational(db: Path, run_id: str) -> dict:
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            "SELECT operational_json FROM run_operations WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row is not None
        return json.loads(row[0])
    finally:
        conn.close()


def test_failed_launch_skips_retention_accounting(tmp_path, monkeypatch) -> None:
    import subprocess as _subprocess

    from src.backtests.contracts import RetentionPolicy

    start, end = _stamps()
    result = _result(tmp_path)
    db = tmp_path / "registry.sqlite3"
    run_id = "e" * 32

    def _boom(*args, **kwargs):
        raise OSError("injected launch boom")

    monkeypatch.setattr(_subprocess, "Popen", _boom)
    with pytest.raises(OSError, match="injected launch boom"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05,
            registry_path=db, run_id=run_id,
            retention_policy=RetentionPolicy(max_detail_bytes=1),
        )
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'run_operations'").fetchall()
    finally:
        conn.close()
    assert rows == []


def test_cleanup_failure_keeps_computation_outcome(tmp_path, monkeypatch, caplog) -> None:
    from src.backtests.contracts import RetentionPolicy

    start, end = _stamps()
    result = _result(tmp_path)
    db = tmp_path / "registry.sqlite3"
    run_id = "f" * 32

    def _on_start(proc):
        staging = result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(_completed_domain()))

    def _boom(*args, **kwargs):
        raise OSError("injected cleanup boom")

    _install_fake(monkeypatch, 0, 0.0, _on_start)
    monkeypatch.setattr(sup, "apply_retention", _boom)
    _ensure_evidence(db)
    with caplog.at_level("WARNING", logger="src.application.mhs_supervisor"):
        run = sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05,
            registry_path=db, run_id=run_id,
            retention_policy=RetentionPolicy(max_detail_bytes=1),
        )
    assert run.status == "completed"
    metadata = _operational(db, run_id)
    assert "injected cleanup boom" in (metadata["cleanup_error"] or "")
    assert any("apply_failed" in record.message for record in caplog.records)


def test_unsatisfied_budget_reported_explicitly(tmp_path, monkeypatch, caplog) -> None:
    from src.backtests.contracts import RetentionPlan, RetentionPolicy

    start, end = _stamps()
    result = _result(tmp_path)
    db = tmp_path / "registry.sqlite3"
    run_id = "a" * 32

    def _on_start(proc):
        staging = result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(_completed_domain()))

    _install_fake(monkeypatch, 0, 0.0, _on_start)
    monkeypatch.setattr(
        sup, "plan_retention",
        lambda *args, **kwargs: RetentionPlan(
            evidence_ids=(), reclaimable_bytes=0, protected_bytes=100, budget_satisfied=False
        ),
    )
    _ensure_evidence(db)
    with caplog.at_level("WARNING", logger="src.application.mhs_supervisor"):
        run = sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05,
            registry_path=db, run_id=run_id,
            retention_policy=RetentionPolicy(max_detail_bytes=1),
        )
    assert run.status == "completed"
    metadata = _operational(db, run_id)
    assert metadata["cleanup_error"] is None
    assert metadata["budget_satisfied"] is False
    assert any("budget_unsatisfied" in record.message for record in caplog.records)


def test_plan_failure_recorded_separately(tmp_path, monkeypatch, caplog) -> None:
    from src.backtests.contracts import RetentionPolicy

    start, end = _stamps()
    result = _result(tmp_path)
    db = tmp_path / "registry.sqlite3"
    run_id = "b" * 32

    def _on_start(proc):
        staging = result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(_completed_domain()))

    def _boom(*args, **kwargs):
        raise ValueError("injected plan boom")

    _install_fake(monkeypatch, 0, 0.0, _on_start)
    monkeypatch.setattr(sup, "plan_retention", _boom)
    with caplog.at_level("WARNING", logger="src.application.mhs_supervisor"):
        run = sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05,
            registry_path=db, run_id=run_id,
            retention_policy=RetentionPolicy(max_detail_bytes=1),
        )
    assert run.status == "completed"
    metadata = _operational(db, run_id)
    assert "injected plan boom" in (metadata["cleanup_error"] or "")
    assert metadata["budget_satisfied"] is None
    assert any("plan_failed" in record.message for record in caplog.records)


def test_satisfied_retention_records_feasibility(tmp_path, monkeypatch) -> None:
    from src.backtests.contracts import RetentionPolicy

    start, end = _stamps()
    result = _result(tmp_path)
    db = tmp_path / "registry.sqlite3"
    run_id = "c" * 32

    def _on_start(proc):
        staging = result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(_completed_domain()))

    _install_fake(monkeypatch, 0, 0.0, _on_start)
    _ensure_evidence(db)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05,
        registry_path=db, run_id=run_id,
        retention_policy=RetentionPolicy(max_detail_bytes=1),
    )
    assert run.status == "completed"
    metadata = _operational(db, run_id)
    assert metadata["cleanup_error"] is None
    assert metadata["budget_satisfied"] is True


def _write_parquet_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_source_snapshot_changes_when_strategy_source_changes(tmp_path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "alpha.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "beta.py").write_text("VALUE = 2\n", encoding="utf-8")
    before = sup._hash_python_sources(package, ())
    assert before is not None
    assert sup._hash_python_sources(package, ()) == before
    (package / "beta.py").write_text("VALUE = 3\n", encoding="utf-8")
    assert sup._hash_python_sources(package, ()) != before
    (package / "gamma.py").write_text("VALUE = 4\n", encoding="utf-8")
    assert sup._hash_python_sources(package, ()) != before


def test_source_snapshot_null_when_unreadable(tmp_path, monkeypatch) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "alpha.py").write_text("VALUE = 1\n", encoding="utf-8")
    real_read_bytes = Path.read_bytes

    def _boom(self, *args, **kwargs):
        raise OSError("injected provenance boom")

    monkeypatch.setattr(Path, "read_bytes", _boom)
    assert sup._hash_python_sources(package, ()) is None
    monkeypatch.setattr(Path, "read_bytes", real_read_bytes)
    assert sup._hash_python_sources(package, ()) is not None


def test_data_snapshot_changes_when_input_bytes_change(tmp_path) -> None:
    root = tmp_path / "data"
    _write_parquet_bytes(root / "ohlcv" / "3m" / "AAAUSDT.parquet", b"frame-v1")
    _write_parquet_bytes(root / "funding" / "AAAUSDT.parquet", b"funding-v1")
    before = sup._snapshot_data_tree(root)
    assert before.startswith("snapshot:")
    assert sup._snapshot_data_tree(root) == before
    (root / "ohlcv" / "3m" / "AAAUSDT.parquet").write_bytes(b"frame-v1+x")
    assert sup._snapshot_data_tree(root) != before
    _write_parquet_bytes(root / "ohlcv" / "3m" / "BBBUSDT.parquet", b"frame-new")
    assert sup._snapshot_data_tree(root) != before


def test_data_snapshot_absent_for_missing_root(tmp_path) -> None:
    assert sup._snapshot_data_tree(tmp_path / "absent") == "absent"


def test_sealed_manifest_upgrades_data_identity(tmp_path) -> None:
    import pandas as pd

    from src.mhs.data_provenance import seal_mhs_input_manifest

    root = tmp_path / "data"
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2022-01-01", periods=4, freq="3min", tz="UTC"), "v": [1.0, 2.0, 3.0, 4.0]}
    )
    target = root / "ohlcv" / "3m" / "AAAUSDT.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target)
    digest = seal_mhs_input_manifest([target], data_root=root, output_path=root / "mhs_execution" / "input_manifest.json")
    assert sup._snapshot_data_tree(root) == f"sealed:{digest}"
    target.write_bytes(target.read_bytes() + b"\x00")
    resealed = sup._snapshot_data_tree(root)
    assert resealed != f"sealed:{digest}"
    assert resealed.startswith("snapshot:")


def test_reused_run_rejected_after_input_data_change(tmp_path) -> None:
    from src.backtests.contracts import RunFinalization, RunRegistration
    from src.backtests.registry import finalize_run, initialize_registry, register_run

    start, end = _stamps()
    root = tmp_path / "data"
    _write_parquet_bytes(root / "ohlcv" / "3m" / "AAAUSDT.parquet", b"frame-v1")
    kwargs = {"start": start, "end": end, "tracking_error_threshold": None}
    first = sup.request_fingerprint(data_root=str(root), **kwargs)
    assert sup.request_fingerprint(data_root=str(root), **kwargs) == first
    registry = tmp_path / "registry.sqlite3"
    initialize_registry(registry)
    register_run(
        registry,
        RunRegistration(
            run_id="e" * 32, strategy_id="process_inventory_3m",
            registered_at="2026-01-01T00:00:00+00:00",
            request={"fingerprint": first}, managed_directory=None,
        ),
    )
    finalize_run(
        registry,
        RunFinalization(
            run_id="e" * 32, status="completed", finalized_at="2026-01-02T00:00:00+00:00",
            primary_valid=True, terminal_certified=True, outcome={"ok": True},
        ),
        (),
    )
    assert sup.find_reused_run(registry, first)[0] == "e" * 32
    (root / "ohlcv" / "3m" / "AAAUSDT.parquet").write_bytes(b"frame-v2-restated")
    second = sup.request_fingerprint(data_root=str(root), **kwargs)
    assert second != first
    assert sup.find_reused_run(registry, second) is None
    assert sup.find_reused_run(registry, first)[0] == "e" * 32


def _write_manifest(root: Path, payload: object) -> None:
    manifest = root / "mhs_execution" / "input_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload), encoding="utf-8")


def test_corrupt_manifest_falls_back_to_snapshot(tmp_path) -> None:
    root = tmp_path / "data"
    _write_parquet_bytes(root / "ohlcv" / "3m" / "AAAUSDT.parquet", b"frame-v1")
    plain = sup._snapshot_data_tree(root)
    assert plain.startswith("snapshot:")
    cases = [
        "[1, 2]",
        {"digest": "not-hex", "files": []},
        {"digest": "a" * 64, "files": [{"relative_path": "x.parquet", "size_bytes": "big", "mtime_ns": 1}]},
        {"digest": "b" * 64, "files": [{"relative_path": "gone.parquet", "size_bytes": 3, "mtime_ns": 4}]},
    ]
    for payload in cases:
        _write_manifest(root, payload)
        assert sup._snapshot_data_tree(root) == plain
    target = root / "ohlcv" / "3m" / "AAAUSDT.parquet"
    stat = target.stat()
    _write_manifest(
        root,
        {
            "digest": "not-hex",
            "files": [
                {
                    "relative_path": "ohlcv/3m/AAAUSDT.parquet",
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            ],
        },
    )
    assert sup._snapshot_data_tree(root) == plain
    (root / "weird.parquet").mkdir()
    assert sup._snapshot_data_tree(root) == plain


def test_data_snapshot_unreadable_when_walk_fails(tmp_path, monkeypatch) -> None:
    root = tmp_path / "data"
    root.mkdir()

    def _boom(self, *args, **kwargs):
        raise OSError("injected walk boom")

    monkeypatch.setattr(Path, "rglob", _boom)
    assert sup._snapshot_data_tree(root) == "unreadable"


def test_supervised_launch_shares_source_identity(tmp_path, monkeypatch) -> None:
    """Registry request and worker command contain the same source digest."""
    import sqlite3 as _sqlite3
    import subprocess as _subprocess

    start, end = _stamps()
    result = _result(tmp_path, "shared")
    db = tmp_path / "registry.sqlite3"
    run_id = "a" * 32
    digest = "e" * 64
    monkeypatch.setattr(sup, "_code_identity", lambda: digest)
    seen: dict = {}

    def _capture(*args, **kwargs):
        seen["command"] = list(args[0])
        proc = _FakeProc(1, 0.0)
        return proc

    monkeypatch.setattr(_subprocess, "Popen", _capture)
    monkeypatch.setattr(sup, "_gnu_time_prefix", lambda: [])
    monkeypatch.setattr(sup, "_parse_gnu_metrics", lambda *a, **k: (None, None))
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result,
        poll_seconds=0.05, registry_path=db, run_id=run_id,
    )
    assert run.status == "failed"
    assert "--procedure-code-digest" in seen["command"]
    assert seen["command"][seen["command"].index("--procedure-code-digest") + 1] == digest
    assert seen["command"][1:3] == ["-m", "src.application.mhs_worker"]
    assert not any(str(part).startswith("src.cli") for part in seen["command"])
    assert not any(str(part).startswith("tools") for part in seen["command"])
    conn = _sqlite3.connect(str(db))
    try:
        row = conn.execute("SELECT request_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    finally:
        conn.close()
    request = json.loads(row[0])
    assert request["code_identity"] == digest
    assert request["fingerprint"] == sup.request_fingerprint(
        start=start, end=end, data_root=None, tracking_error_threshold=None, code_digest=digest,
    )


def test_supervised_launch_blocked_without_source_identity(tmp_path, monkeypatch) -> None:
    """Source hashing failure blocks worker launch and run registration."""
    import subprocess as _subprocess

    start, end = _stamps()
    result = _result(tmp_path, "blocked")
    db = tmp_path / "registry.sqlite3"

    def _no_launch(*args, **kwargs):
        raise AssertionError("worker must not launch")

    monkeypatch.setattr(_subprocess, "Popen", _no_launch)
    monkeypatch.setattr(sup, "_code_identity", lambda: None)
    with pytest.raises(ValueError, match="source identity"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=result,
            poll_seconds=0.05, registry_path=db, run_id="b" * 32,
        )
    assert not result.exists()
    assert not db.exists()
    monkeypatch.setattr(sup, "_code_identity", lambda: "NOT-HEX")
    with pytest.raises(ValueError, match="source identity"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=result,
            poll_seconds=0.05, registry_path=db, run_id="b" * 32,
        )


def test_fingerprint_changes_with_source_identity(tmp_path) -> None:
    """Distinct source identities yield distinct fingerprints blocking reuse."""
    from src.backtests.contracts import RunFinalization, RunRegistration
    from src.backtests.registry import finalize_run, initialize_registry, register_run

    start, end = _stamps()
    first = sup.request_fingerprint(
        start=start, end=end, data_root=None, tracking_error_threshold=None, code_digest="a" * 64,
    )
    second = sup.request_fingerprint(
        start=start, end=end, data_root=None, tracking_error_threshold=None, code_digest="b" * 64,
    )
    assert first != second
    registry = tmp_path / "registry.sqlite3"
    initialize_registry(registry)
    register_run(
        registry,
        RunRegistration(
            run_id="c" * 32, strategy_id="process_inventory_3m",
            registered_at="2026-01-01T00:00:00+00:00",
            request={"fingerprint": first}, managed_directory=None,
        ),
    )
    finalize_run(
        registry,
        RunFinalization(
            run_id="c" * 32, status="completed", finalized_at="2026-01-02T00:00:00+00:00",
            primary_valid=True, terminal_certified=True, outcome={"ok": True},
        ),
        (),
    )
    assert sup.find_reused_run(registry, first)[0] == "c" * 32
    assert sup.find_reused_run(registry, second) is None


def test_supervised_run_reports_wall_time_and_tree_pss(tmp_path, monkeypatch) -> None:
    """Supervised runs report wall time and sampled tree PSS."""
    start, end = _stamps()
    result = _result(tmp_path, "pss")

    def _on_start(proc):
        staging = result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(_completed_domain()))

    _install_fake(monkeypatch, 0, 0.3, _on_start)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05,
        registry_path=_registry(tmp_path),
    )
    assert run.status == "completed"
    assert run.wall_seconds >= 0.0
    assert run.sampled_tree_pss_peak_bytes == 100
    assert "PSS" in run.memory_scope


def test_registry_path_is_mandatory(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path)
    calls: list = []

    def _recording_factory(*args, **kwargs):
        calls.append(args)
        return _FakeProc(0, 0.0)

    _install_fake(monkeypatch)
    monkeypatch.setattr(subprocess, "Popen", _recording_factory)
    with pytest.raises(TypeError, match="registry_path"):
        sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05)  # type: ignore[call-arg]
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_explicit_registry_receives_run_and_nothing_else(tmp_path, monkeypatch) -> None:
    from src.common.paths import BACKTESTS_DIR

    start, end = _stamps()
    result = _result(tmp_path, "explicit")
    registry = tmp_path / "reg" / "registry.sqlite3"
    operator_registry = BACKTESTS_DIR / "registry.sqlite3"
    before_exists = operator_registry.exists()
    before_stat = operator_registry.stat() if before_exists else None
    before_key = (before_stat.st_size, before_stat.st_mtime_ns) if before_stat is not None else None

    def _on_start(proc):
        staging = result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(_completed_domain()))

    _install_fake(monkeypatch, 0, 0.0, _on_start)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05,
        registry_path=registry,
    )
    assert run.status == "completed"
    conn = sqlite3.connect(str(registry))
    try:
        rows = conn.execute("SELECT run_id FROM runs").fetchall()
    finally:
        conn.close()
    assert [row[0] for row in rows] == [run.run_id]
    assert operator_registry.exists() == before_exists
    if before_stat is not None:
        after_stat = operator_registry.stat()
        assert (after_stat.st_size, after_stat.st_mtime_ns) == before_key


def _fake_psutil_tree(monkeypatch, payloads):
    import psutil as _psutil

    calls: dict[int, int] = {}

    class _Proc:
        def __init__(self, index, payload) -> None:
            self._index = index
            self._payload = payload

        def memory_full_info(self):
            calls[self._index] = calls.get(self._index, 0) + 1
            if isinstance(self._payload, Exception):
                raise self._payload
            return self._payload

        def children(self, recursive=True):
            return []

    root_payload, *child_payloads = payloads
    children = [_Proc(i + 1, payload) for i, payload in enumerate(child_payloads)]

    class _Root(_Proc):
        def children(self, recursive=True):
            return children

    root = _Root(0, root_payload)

    def _factory(pid=None, *args, **kwargs):
        return root

    monkeypatch.setattr(_psutil, "Process", _factory)
    return calls


def _ns(pss, uss, swap="__unset__"):
    from types import SimpleNamespace

    kwargs = {"pss": pss, "uss": uss}
    if swap != "__unset__":
        kwargs["swap"] = swap
    return SimpleNamespace(**kwargs)


def test_workload_memory_reads_each_process_once(monkeypatch) -> None:
    """One sweep returns summed PSS, USS and swap with one read per process."""
    calls = _fake_psutil_tree(
        monkeypatch, [_ns(100, 10, 1), _ns(200, 20, 2), _ns(300, 30, 3)],
    )
    assert sup._workload_memory(1234) == (600, 60, 6)
    assert calls == {0: 1, 1: 1, 2: 1}


def test_workload_memory_skips_vanished_and_denied_processes(monkeypatch) -> None:
    """Vanished and access-denied processes leave all three sums."""
    import psutil as _psutil

    calls = _fake_psutil_tree(
        monkeypatch,
        [_ns(100, 10, 1), _psutil.NoSuchProcess(pid=9), _psutil.AccessDenied(pid=10)],
    )
    assert sup._workload_memory(1234) == (100, 10, 1)
    assert calls.get(0) == 1


def test_workload_memory_missing_pss_or_uss_fails(monkeypatch) -> None:
    """A process without PSS or USS fails the safety sample."""
    _fake_psutil_tree(monkeypatch, [_ns(None, 10, 1)])
    with pytest.raises(OSError, match="PSS/USS telemetry unavailable"):
        sup._workload_memory(1)
    _fake_psutil_tree(monkeypatch, [_ns(100, None, 1)])
    with pytest.raises(OSError, match="PSS/USS telemetry unavailable"):
        sup._workload_memory(1)


def test_workload_memory_missing_swap_is_unknown_not_zero(monkeypatch) -> None:
    """Missing or unusable swap nulls the sample swap without touching PSS/USS."""
    _fake_psutil_tree(monkeypatch, [_ns(100, 10), _ns(200, 20, 5)])
    pss, uss, swap = sup._workload_memory(1)
    assert (pss, uss, swap) == (300, 30, None)
    _fake_psutil_tree(monkeypatch, [_ns(100, 10, "x")])
    pss, uss, swap = sup._workload_memory(1)
    assert (pss, uss, swap) == (100, 10, None)


def test_workload_memory_root_failure_propagates(monkeypatch) -> None:
    """Root enumeration failures propagate to the missing-telemetry path."""
    import psutil as _psutil

    def _boom(pid=None, *args, **kwargs):
        raise _psutil.NoSuchProcess(pid=pid)

    monkeypatch.setattr(_psutil, "Process", _boom)
    with pytest.raises(_psutil.NoSuchProcess):
        sup._workload_memory(999)


def test_one_sweep_per_poll(tmp_path, monkeypatch) -> None:
    """Each counted sample plus the launch baseline costs exactly one sweep."""
    start, end = _stamps()
    result = _result(tmp_path, "sweep")

    def _on_start(proc):
        staging = result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(_completed_domain()))

    _install_fake(monkeypatch, 0, 0.3, _on_start)
    calls = {"n": 0}

    def _counted(pid):
        calls["n"] += 1
        return (100, 50, 0)

    monkeypatch.setattr(sup, "_workload_memory", _counted)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result,
        poll_seconds=0.05, registry_path=_registry(tmp_path),
    )
    assert run.status == "completed"
    assert calls["n"] == run.samples_taken + 1
    assert run.samples_taken >= 1


def test_peaks_equal_scripted_sample_extremes(tmp_path, monkeypatch) -> None:
    """Sampled peaks and floors equal the extremes of the scripted sequence."""
    start, end = _stamps()
    result = _result(tmp_path, "peaks")

    def _on_start(proc):
        staging = result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(_completed_domain()))

    _install_fake(monkeypatch, 0, 0.4, _on_start)
    pss_seq = [300, 900, 500]
    uss_seq = [30, 20, 90]
    head_seq = [8 * 2**30, 5 * 2**30, 7 * 2**30]
    state = {"n": 0}

    def _scripted(pid):
        i = min(state["n"], 2)
        return (pss_seq[i], uss_seq[i], 0)

    def _head():
        i = min(max(state["n"] - 1, 0), 2)
        return head_seq[i]

    def _counted(pid):
        out = _scripted(pid)
        state["n"] += 1
        return out

    monkeypatch.setattr(sup, "_workload_memory", _counted)
    monkeypatch.setattr(sup, "current_mhs_headroom_bytes", _head)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result,
        poll_seconds=0.05, registry_path=_registry(tmp_path),
    )
    assert run.status == "completed"
    assert run.sampled_tree_pss_peak_bytes == 900
    assert run.sampled_tree_uss_peak_bytes == 90
    assert run.min_available_bytes == 5 * 2**30
    assert run.samples_taken >= 3


def test_swap_growth_uses_same_sample_swap(tmp_path, monkeypatch) -> None:
    """Swap growth compares the same-sample swap against the launch baseline."""
    start, end = _stamps()
    result = _result(tmp_path, "growth")
    _install_fake(monkeypatch, 0, 5.0)
    swaps = iter([100, 100, 160, 120])

    def _scripted(pid):
        try:
            value = next(swaps)
        except StopIteration:
            value = 120
        return (100, 50, value)

    monkeypatch.setattr(sup, "_workload_memory", _scripted)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result,
        poll_seconds=0.05, registry_path=_registry(tmp_path),
    )
    assert run.status == "resource_rejected"
    assert run.termination_reason == "swap growth 60 bytes observed"
    assert run.process_swap_growth_bytes == 60


def test_telemetry_loss_rejects_without_counting(tmp_path, monkeypatch) -> None:
    """A failing safety sample rejects without counting the poll."""
    start, end = _stamps()
    result = _result(tmp_path, "loss")
    _install_fake(monkeypatch, 0, 5.0)

    def _gone(pid):
        raise OSError("gone")

    monkeypatch.setattr(sup, "_workload_memory", _gone)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result,
        poll_seconds=0.05, registry_path=_registry(tmp_path),
    )
    assert run.status == "resource_rejected"
    assert run.termination_reason == "missing safety telemetry: gone"
    assert run.samples_taken == 0


def test_unknown_baseline_never_reports_growth(tmp_path, monkeypatch) -> None:
    """An unobservable launch baseline never yields swap growth."""
    start, end = _stamps()
    result = _result(tmp_path, "nobase")

    def _on_start(proc):
        staging = result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(_completed_domain()))

    _install_fake(monkeypatch, 0, 0.15, _on_start)
    calls = {"n": 0}

    def _scripted(pid):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("no baseline")
        return (100, 50, 10**9)

    monkeypatch.setattr(sup, "_workload_memory", _scripted)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result,
        poll_seconds=0.05, registry_path=_registry(tmp_path),
    )
    assert run.status == "completed"
    assert run.process_swap_growth_bytes is None


def test_status_precedence_unchanged(tmp_path, monkeypatch) -> None:
    """An elapsed deadline outranks a simultaneous resource breach."""
    start, end = _stamps()
    result = _result(tmp_path, "precedence")
    _install_fake(monkeypatch, 0, 30.0)
    monkeypatch.setattr(sup, "_workload_memory", lambda pid: (10 * 2**30, 1, 0))
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result,
        poll_seconds=0.05, timeout_seconds=0.01, registry_path=_registry(tmp_path),
    )
    assert run.status == "timed_out"


def test_workload_memory_all_skipped_returns_zeros(monkeypatch) -> None:
    """Every process skipped yields zero PSS/USS and unknown swap."""
    import psutil as _psutil

    _fake_psutil_tree(
        monkeypatch,
        [_psutil.NoSuchProcess(pid=1), _psutil.AccessDenied(pid=2)],
    )
    assert sup._workload_memory(1) == (0, 0, None)
