"""Invariant scenarios for the supervised production 3m runner."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import pytest

import tools.mhs_process_backtest_run as sup


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
    monkeypatch.setattr(sup, "_system_headroom_bytes", lambda: 8 * 2**30)
    monkeypatch.setattr(sup, "_swap_used_bytes", lambda: 0)
    monkeypatch.setattr(sup, "_terminate_group", lambda proc, timeout=5.0: setattr(proc, "terminated", True))


def test_supervised_awaits_real_completion(tmp_path, monkeypatch) -> None:
    """Child lifetime is fully waited and reflected in wall measurement."""
    import tools.mhs_process_backtest_run as sup

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


def test_supervised_ordinary_failure_not_oom(tmp_path, monkeypatch) -> None:
    """Nonzero exit with traceback stays failed, never signaled or OOM."""
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
    """Unsafe sampled telemetry triggers resource_rejected cleanup."""
    start, end = _stamps()
    for label in ("pss", "headroom", "swap"):
        output, failure, run_out = _paths(tmp_path, f"res_{label}")
        _install_fake(monkeypatch, 0, 5.0)
        if label == "pss":
            monkeypatch.setattr(sup, "_workload_pss_uss", lambda pid: (10 * 2**30, 1))
        elif label == "headroom":
            monkeypatch.setattr(sup, "_system_headroom_bytes", lambda: 100)
        else:
            calls = {"n": 0}

            def _swap(_calls=calls):
                _calls["n"] += 1
                return 0 if _calls["n"] < 2 else 10**9

            monkeypatch.setattr(sup, "_swap_used_bytes", _swap)
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
    """Pre-existing bytes survive while absent GNU metrics stay null."""
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
    """Exit zero with no new completed artifact never reuses prior evidence."""
    start, end = _stamps()
    output, failure, run_out = _paths(tmp_path)
    _install_fake(monkeypatch, 0, 0.0)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, output=output,
        failure_output=failure, run_output=run_out, poll_seconds=0.05,
    )
    assert run.status == "failed"
    assert "completed primary artifact" in (run.termination_reason or "")
    assert sup.main is not None
    code = sup.main([
        "--start", "2022-01-01T00:00:00+00:00", "--end", "2022-01-04T00:00:00+00:00",
        "--output", str(output), "--failure-output", str(failure),
        "--run-output", str(tmp_path / "other.run.json"),
    ])
    assert code == 1
