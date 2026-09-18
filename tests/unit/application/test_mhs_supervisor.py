"""Invariant scenarios for the source-owned supervised runner."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import pytest

import src.application.mhs_supervisor as sup


def _stamps():
    return pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-01-04", tz="UTC")


def _paths(tmp_path: Path, name: str = "run"):
    base = tmp_path / name
    return (
        base.with_name(f"{base.name}.primary.json"),
        base.with_name(f"{base.name}.failure.json"),
        base.with_name(f"{base.name}.run.json"),
    )


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


def test_supervised_awaits_real_completion(tmp_path, monkeypatch) -> None:
    """Real completion waited: return waits for actual exit and requires zero exit plus completed 3m evidence."""
    start, end = _stamps()
    output, failure, run_out = _paths(tmp_path)

    def _on_start(proc):
        output.write_text(json.dumps({"status": "completed", "execution_timeframe": "3m"}))

    _install_fake(monkeypatch, 0, 0.4, _on_start)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, output=output,
        failure_output=failure, run_output=run_out, poll_seconds=0.05,
    )
    assert run.status == "completed"
    assert run.wall_seconds >= 0.4
    assert run.primary_artifact_written is True
    assert run.command[0].endswith("python") or "python" in run.command[0]
    assert list(run.command[1:3]) == ["-m", "src.application.mhs_worker"]
    assert "src.cli.main" not in " ".join(run.command)


def test_supervised_ordinary_failure_not_oom(tmp_path, monkeypatch) -> None:
    """Rejected versus signaled: nonzero exit with traceback stays failed, never signaled or OOM."""
    start, end = _stamps()
    output, failure, run_out = _paths(tmp_path)
    log = run_out.parent / f"{run_out.stem}.log"

    def _on_start(proc):
        log.write_text("Traceback (most recent call last):\n  boom\n")

    _install_fake(monkeypatch, 1, 0.0, _on_start)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, output=output,
        failure_output=failure, run_output=run_out, poll_seconds=0.05,
    )
    assert run.status == "failed"
    assert run.exit_code == 1
    assert run.signal_number is None
    assert "OOM" not in (run.termination_reason or "")


def test_supervised_signal_and_timeout_stay_distinct(tmp_path, monkeypatch) -> None:
    """Externally signaled and deadline-expired outcomes keep their reasons."""
    start, end = _stamps()
    output, failure, run_out = _paths(tmp_path, "sig")
    _install_fake(monkeypatch, -15, 0.0)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, output=output,
        failure_output=failure, run_output=run_out, poll_seconds=0.05,
    )
    assert run.status == "signaled"
    assert run.signal_number == 15
    output2, failure2, run_out2 = _paths(tmp_path, "timeout")
    _install_fake(monkeypatch, 0, 30.0)
    run2 = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, output=output2,
        failure_output=failure2, run_output=run_out2, poll_seconds=0.05,
        timeout_seconds=0.2,
    )
    assert run2.status == "timed_out"
    assert "deadline" in (run2.termination_reason or "")


def test_supervised_resource_stop_without_success_claim(tmp_path, monkeypatch) -> None:
    """Interrupted group: unsafe sampled telemetry triggers resource_rejected cleanup."""
    start, end = _stamps()
    for label in ("pss", "headroom", "swap"):
        output, failure, run_out = _paths(tmp_path, f"res_{label}")
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
        run = sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, output=output,
            failure_output=failure, run_output=run_out, poll_seconds=0.05,
        )
        assert run.status == "resource_rejected"
        assert run.termination_reason
        assert run.primary_artifact_written is False
    output, failure, run_out = _paths(tmp_path, "res_missing")

    def _missing(pid):
        raise OSError("no telemetry")

    _install_fake(monkeypatch, 0, 5.0)
    monkeypatch.setattr(sup, "_workload_pss_uss", _missing)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, output=output,
        failure_output=failure, run_output=run_out, poll_seconds=0.05,
    )
    assert run.status == "resource_rejected"


def test_supervised_fresh_artifacts_and_null_metrics(tmp_path, monkeypatch) -> None:
    """Optional measurements unknown: pre-existing bytes survive while absent metrics stay null."""
    start, end = _stamps()
    output, failure, run_out = _paths(tmp_path)
    output.write_text('{"keep": true}', encoding="utf-8")
    with pytest.raises(ValueError, match=r".+"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, output=output,
            failure_output=failure, run_output=run_out, poll_seconds=0.05,
        )
    assert json.loads(output.read_text(encoding="utf-8")) == {"keep": True}
    output.unlink()
    _install_fake(monkeypatch, 1, 0.0)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, output=output,
        failure_output=failure, run_output=run_out, poll_seconds=0.05,
    )
    assert run.gnu_max_individual_rss_bytes is None
    assert run.cpu_seconds is None
    assert run.cpu_scope
    assert run.memory_scope
    assert "seconds" in run.cpu_scope or "CPU" in run.cpu_scope
    assert "bytes" in run.memory_scope or "RSS" in run.memory_scope
    payload = json.loads(run_out.read_text(encoding="utf-8"))
    assert payload["samples_taken"] == run.samples_taken
    assert "cagr" not in json.dumps(payload)


def test_supervised_zero_exit_without_evidence_fails(tmp_path, monkeypatch) -> None:
    """Nonzero with primary: exit zero with no new completed artifact never reuses prior evidence."""
    start, end = _stamps()
    output, failure, run_out = _paths(tmp_path)
    _install_fake(monkeypatch, 0, 0.0)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, output=output,
        failure_output=failure, run_output=run_out, poll_seconds=0.05,
    )
    assert run.status == "failed"
    assert "completed primary artifact" in (run.termination_reason or "")


def test_supervised_nonzero_exit_preserves_primary(tmp_path, monkeypatch) -> None:
    """Nonzero with primary: a fresh primary artifact stays preserved while the outcome stays failed."""
    start, end = _stamps()
    output, failure, run_out = _paths(tmp_path)

    def _on_start(proc):
        output.write_text(json.dumps({"status": "completed", "execution_timeframe": "3m"}))

    _install_fake(monkeypatch, 1, 0.0, _on_start)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, output=output,
        failure_output=failure, run_output=run_out, poll_seconds=0.05,
    )
    assert run.status == "failed"
    assert run.primary_artifact_written is True
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "completed"


def test_supervised_worker_receives_resolved_budget(tmp_path, monkeypatch) -> None:
    """Resource profile propagated: worker arguments and outcome share identical limits; total governs."""
    from src.mhs.resources import MhsMemoryBudget

    start, end = _stamps()
    output, failure, run_out = _paths(tmp_path)
    targets = tmp_path / "targets.parquet"
    budget = MhsMemoryBudget(
        total_tree_pss_bytes=6 * 2**30,
        replay_tree_pss_bytes=2 * 2**30,
        min_available_bytes=1 * 2**30,
    )
    seen: dict = {}

    def _factory(*args, **kwargs):
        seen["command"] = list(args[0])
        output.write_text(json.dumps({"status": "completed", "execution_timeframe": "3m"}))
        return _FakeProc(0, 0.2)

    monkeypatch.setattr(subprocess, "Popen", _factory)
    monkeypatch.setattr(sup, "_gnu_time_prefix", lambda: [])
    monkeypatch.setattr(sup, "_parse_gnu_metrics", lambda *a, **k: (None, None))
    monkeypatch.setattr(sup, "_workload_pss_uss", lambda pid: (4 * 2**30, 1 * 2**30))
    monkeypatch.setattr(sup, "current_mhs_headroom_bytes", lambda: 8 * 2**30)
    monkeypatch.setattr(sup, "_workload_swap_bytes", lambda pid: 0)
    monkeypatch.setattr(sup, "_terminate_group", lambda proc, timeout=5.0: setattr(proc, "terminated", True))
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=str(tmp_path), output=output,
        failure_output=failure, run_output=run_out, targets_output=targets,
        tracking_error_threshold=0.2, poll_seconds=0.05, memory_budget=budget,
    )
    command = seen["command"]
    assert list(command[1:3]) == ["-m", "src.application.mhs_worker"]
    for flag, value in (
        ("--total-tree-pss-bytes", 6 * 2**30),
        ("--replay-tree-pss-bytes", 2 * 2**30),
        ("--min-available-bytes", 1 * 2**30),
    ):
        assert command[command.index(flag) + 1] == str(value)
    assert run.status == "completed"
    assert run.memory_budget == budget
    payload = json.loads(run_out.read_text(encoding="utf-8"))
    assert payload["memory_budget"] == {
        "total_tree_pss_bytes": 6 * 2**30,
        "replay_tree_pss_bytes": 2 * 2**30,
        "min_available_bytes": 1 * 2**30,
    }


def test_supervised_typed_failure_code_rejected(tmp_path, monkeypatch) -> None:
    """Rejected versus signaled: only typed resource evidence establishes resource_rejected."""
    start, end = _stamps()
    output, failure, run_out = _paths(tmp_path, "typed")

    def _on_start(proc):
        failure.write_text(json.dumps({"status": "failed", "error_code": "MEMORY_BUDGET"}))

    _install_fake(monkeypatch, 1, 0.0, _on_start)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, output=output,
        failure_output=failure, run_output=run_out, poll_seconds=0.05,
    )
    assert run.status == "resource_rejected"
    assert "MEMORY_BUDGET" in (run.termination_reason or "")
    output2, failure2, run_out2 = _paths(tmp_path, "plain")

    def _on_start_plain(proc):
        failure2.write_text(json.dumps({"status": "failed", "error_code": "DATA_INTEGRITY"}))

    _install_fake(monkeypatch, 1, 0.0, _on_start_plain)
    run2 = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, output=output2,
        failure_output=failure2, run_output=run_out2, poll_seconds=0.05,
    )
    assert run2.status == "failed"
    assert "OOM" not in (run2.termination_reason or "")


def test_supervised_interrupt_terminates_group(tmp_path, monkeypatch) -> None:
    """Interrupted group: interruption terminates the launched group and persists a non-success outcome."""

    class _InterruptProc(_FakeProc):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._raised = False

        def wait(self, timeout=None):
            if not self._raised:
                self._raised = True
                raise KeyboardInterrupt
            return self._returncode

    terminated: list = []

    def _factory(*args, **kwargs):
        return _InterruptProc(0, 0.0)

    def _mark(proc, timeout=5.0):
        proc.terminated = True
        terminated.append(proc)

    monkeypatch.setattr(subprocess, "Popen", _factory)
    monkeypatch.setattr(sup, "_gnu_time_prefix", lambda: [])
    monkeypatch.setattr(sup, "_parse_gnu_metrics", lambda *a, **k: (None, None))
    monkeypatch.setattr(sup, "_workload_pss_uss", lambda pid: (100, 50))
    monkeypatch.setattr(sup, "current_mhs_headroom_bytes", lambda: 8 * 2**30)
    monkeypatch.setattr(sup, "_workload_swap_bytes", lambda pid: 0)
    monkeypatch.setattr(sup, "_terminate_group", _mark)
    start, end = _stamps()
    output, failure, run_out = _paths(tmp_path)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, output=output,
        failure_output=failure, run_output=run_out, poll_seconds=0.05,
    )
    assert run.status == "interrupted"
    assert terminated
    payload = json.loads(run_out.read_text(encoding="utf-8"))
    assert payload["status"] == "interrupted"


def test_supervised_launch_interrupt_reports_launch_failure(tmp_path, monkeypatch) -> None:
    """A launch-time interruption surfaces as a launch failure without an outcome artifact."""
    start, end = _stamps()
    output, failure, run_out = _paths(tmp_path)

    def _factory(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", _factory)
    with pytest.raises(OSError, match="failed to launch"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, output=output,
            failure_output=failure, run_output=run_out, poll_seconds=0.05,
        )
    assert not run_out.exists()


def test_supervised_unknown_swap_stays_null(tmp_path, monkeypatch) -> None:
    """Optional measurements unknown: unavailable workload swap telemetry stays null without failing."""
    start, end = _stamps()
    output, failure, run_out = _paths(tmp_path)

    def _on_start(proc):
        output.write_text(json.dumps({"status": "completed", "execution_timeframe": "3m"}))

    _install_fake(monkeypatch, 0, 0.0, _on_start)
    monkeypatch.setattr(sup, "_workload_swap_bytes", lambda pid: None)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, output=output,
        failure_output=failure, run_output=run_out, poll_seconds=0.05,
    )
    assert run.status == "completed"
    assert run.process_swap_growth_bytes is None


def test_supervised_swap_baseline_starts_at_launch(tmp_path, monkeypatch) -> None:
    """The workload swap baseline is captured before the first polling delay."""
    start, end = _stamps()
    output, failure, run_out = _paths(tmp_path)
    calls = {"count": 0}

    def _swap(_pid):
        calls["count"] += 1
        return 0 if calls["count"] == 1 else 1

    def _on_start(_proc):
        output.write_text(json.dumps({"status": "completed", "execution_timeframe": "3m"}))

    _install_fake(monkeypatch, 0, 0.2, _on_start)
    monkeypatch.setattr(sup, "_workload_swap_bytes", _swap)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, output=output,
        failure_output=failure, run_output=run_out, poll_seconds=0.05,
    )
    assert run.status == "resource_rejected"
    assert run.process_swap_growth_bytes == 1


def test_supervised_rejects_bad_intervals_and_destinations(tmp_path, monkeypatch) -> None:
    """Fresh destinations and finite controls: invalid intervals or occupied paths never launch."""
    start, end = _stamps()

    def _boom(*args, **kwargs):
        raise AssertionError("child must not launch")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    output, failure, run_out = _paths(tmp_path)
    cases = [
        {"poll_seconds": True},
        {"poll_seconds": 0},
        {"poll_seconds": float("nan")},
        {"poll_seconds": float("inf")},
        {"poll_seconds": "fast"},
        {"timeout_seconds": False},
        {"timeout_seconds": -1.0},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"tracking_error_threshold": "bad"},
        {"tracking_error_threshold": -0.5},
        {"targets_output": tmp_path / "targets.txt"},
        {"output": tmp_path / "primary.csv"},
        {"output": "plain.json"},
        {"start": pd.Timestamp("2022-01-01")},
    ]
    for case in cases:
        call = {
            "start": start,
            "end": end,
            "data_root": None,
            "output": output,
            "failure_output": failure,
            "run_output": run_out,
        }
        call.update(case)
        with pytest.raises(ValueError, match=r".+"):
            sup.run_mhs_process_backtest(**call)
    from src.mhs.process_backtest import PROCESS_REPORT_PATH

    with pytest.raises(ValueError, match=r".+"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, output=PROCESS_REPORT_PATH,
            failure_output=failure, run_output=run_out,
        )
    with pytest.raises(ValueError, match=r".+"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, output=output,
            failure_output=output, run_output=run_out,
        )
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r".+"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, output=output,
            failure_output=failure, run_output=run_out,
        )
    assert json.loads(output.read_text(encoding="utf-8")) == {}


def test_gnu_metrics_helpers(tmp_path) -> None:
    """GNU and scope helpers report honest nulls instead of invented measurements."""
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
    """Terminal classification reads the serialized worker failure code without parsing text."""
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
    """Completion requires a fresh completed primary artifact, never a bare file."""
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
    """Timeout and poll controls reject bool, non-positive, NaN and infinity."""
    for bad in (True, False, 0, -1.0, float("nan"), float("inf"), "fast", None):
        with pytest.raises(ValueError, match=r".+"):
            sup._validate_positive_interval(bad, "poll_seconds")
    sup._validate_positive_interval(0.25, "poll_seconds")
    sup._validate_positive_interval(2, "timeout_seconds")


def test_terminate_group_handles_missing_and_live_groups(monkeypatch) -> None:
    """Group termination returns for missing groups and escalates only for live ones."""
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
    """Supervisor persistence never leaves partial outcome artifacts behind."""
    target = tmp_path / "run.json"

    def _boom(src, dst):
        raise OSError("replace unavailable")

    monkeypatch.setattr(sup.os, "replace", _boom)
    with pytest.raises(OSError, match="replace unavailable"):
        sup._atomic_write_json(target, {"a": 1})
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
