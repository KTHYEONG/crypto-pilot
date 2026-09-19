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
    monkeypatch.setattr(sup, "_workload_pss_uss", lambda pid: (100, 50))
    monkeypatch.setattr(sup, "current_mhs_headroom_bytes", lambda: 8 * 2**30)
    monkeypatch.setattr(sup, "_workload_swap_bytes", lambda pid: 0)
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
    run = sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05)
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
    run = sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05)
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
    run = sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05)
    assert run.status == "completed"
    log = result.parent / "result.log"
    assert not log.exists()
    registry = Path("data/backtests/registry.sqlite3")
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
    run = sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05)
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
        sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05)
    assert json.loads(result.read_text(encoding="utf-8")) == {}
    link = tmp_path / "link" / "result.json"
    link.parent.mkdir(parents=True, exist_ok=True)
    import os

    os.symlink(tmp_path / "missing.json", link)
    with pytest.raises(ValueError, match="fresh"):
        sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=link, poll_seconds=0.05)


def test_supervised_signal_and_timeout_stay_distinct(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path, "sig")
    _install_fake(monkeypatch, -15, 0.0)
    run = sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05)
    assert run.status == "signaled"
    assert run.signal_number == 15
    result2 = _result(tmp_path, "timeout")
    _install_fake(monkeypatch, 0, 30.0)
    run2 = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result2, poll_seconds=0.05, timeout_seconds=0.2,
    )
    assert run2.status == "timed_out"


def test_supervised_resource_stop_without_success_claim(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    for label in ("pss", "headroom", "swap"):
        result = _result(tmp_path, f"res_{label}")
        _install_fake(monkeypatch, 0, 5.0)
        if label == "pss":
            monkeypatch.setattr(sup, "_workload_pss_uss", lambda pid: (10 * 2**30, 1))
        elif label == "headroom":
            monkeypatch.setattr(sup, "current_mhs_headroom_bytes", lambda: 100)
        else:
            calls = {"n": 0}

            def _swap(_calls=calls):
                _calls["n"] += 1
                return 0 if _calls["n"] < 2 else 10**9

            monkeypatch.setattr(sup, "_workload_swap_bytes", lambda pid: _swap())
        run = sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05)
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
            poll_seconds=0.05, run_id="bogus",
        )
    with pytest.raises(ValueError, match="result_output"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=tmp_path / "bad.csv",  # type: ignore[arg-type]
            poll_seconds=0.05,
        )
    with pytest.raises(ValueError, match="retention_policy"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=_result(tmp_path),
            poll_seconds=0.05, retention_policy="policy",  # type: ignore[arg-type]
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
        sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05)
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
        run = sup.run_mhs_process_backtest(start=start, end=end, data_root=None, result_output=result, poll_seconds=0.05)
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
