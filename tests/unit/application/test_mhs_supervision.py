"""Termination-decomposition invariants for the supervised runner.

Companion to ``test_mhs_supervisor.py``: classify precedence, single-sample
semantics, interrupt reap, launch/log failure publication, heartbeat cadence
and the single run builder. Kept in a separate module so the lifecycle suite
stays within the 60 KB test file budget.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import src.application.mhs_supervisor as sup


def _stamps():
    return pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-01-04", tz="UTC")


def _result(tmp_path: Path, name: str) -> Path:
    return tmp_path / name / "result.json"


def _registry(tmp_path: Path) -> Path:
    return tmp_path / "registry.sqlite3"


def _finalizations(db: Path, run_id: str) -> int:
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT run_id FROM finalizations WHERE run_id = ?", (run_id,)).fetchall()
    finally:
        conn.close()
    return len(rows)


def _completed_domain() -> dict:
    return {
        "status": "completed",
        "execution_timeframe": "3m",
        "base": {"terminal": {"primary_valid": True, "terminal_certified": True}},
        "stress": {"terminal": {"primary_valid": True, "terminal_certified": True}},
    }


class _ScriptedProc:
    """Fake worker with scripted wait/poll behavior; empty waits idle with timeouts."""

    def __init__(self, waits=(), polls=()) -> None:
        self.pid = 987654
        self._waits = list(waits)
        self._polls = list(polls)
        self.terminated = False

    def poll(self):
        if self._polls:
            item = self._polls.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        return 0 if self.terminated else None

    def wait(self, timeout=None):
        item = self._waits.pop(0) if self._waits else "timeout"
        if item == "timeout":
            raise subprocess.TimeoutExpired(cmd=[], timeout=timeout)
        if isinstance(item, BaseException):
            raise item
        return item


def _install_scripted(monkeypatch, proc: _ScriptedProc):
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(sup, "_gnu_time_prefix", lambda: [])
    monkeypatch.setattr(sup, "_parse_gnu_metrics", lambda *a, **k: (None, None))
    monkeypatch.setattr(sup, "_workload_memory", lambda pid: (100, 50, 0))
    monkeypatch.setattr(sup, "current_mhs_headroom_bytes", lambda: 8 * 2**30)
    monkeypatch.setattr(sup, "_terminate_group", lambda p, timeout=5.0: setattr(p, "terminated", True))


def _base_kwargs(**override):
    kwargs = {
        "launch_error": None, "timed_out": False, "resource_reason": None,
        "interrupted": False, "failure_code": None, "returncode": 0,
        "primary_completed": True, "wall_seconds": 12.34,
    }
    kwargs.update(override)
    return kwargs


def test_classify_termination_precedence_is_total_and_ordered() -> None:
    cases = [
        ({"launch_error": "x", "timed_out": True, "interrupted": True, "returncode": -9}, "failed"),
        ({"timed_out": True, "resource_reason": "r", "interrupted": True}, "timed_out"),
        ({"resource_reason": "r", "interrupted": True, "failure_code": "MEMORY_BUDGET"}, "resource_rejected"),
        ({"interrupted": True, "failure_code": "MEMORY_BUDGET", "returncode": -9}, "interrupted"),
        ({"failure_code": "MEMORY_BUDGET"}, "resource_rejected"),
        ({"failure_code": "DATA_INTEGRITY"}, "completed"),
        ({"failure_code": "DATA_INTEGRITY", "primary_completed": False}, "failed"),
        ({"returncode": -9}, "signaled"),
        ({"returncode": 0, "primary_completed": True}, "completed"),
        ({"returncode": 0, "primary_completed": False}, "failed"),
        ({"returncode": 1, "primary_completed": False}, "failed"),
        ({"returncode": None, "primary_completed": False}, "failed"),
    ]
    for override, status in cases:
        outcome = sup._classify_termination(**_base_kwargs(**override))
        assert outcome.status == status
        rc = override.get("returncode", 0)
        assert outcome.exit_code == (rc if rc is not None and rc >= 0 else None)
        assert outcome.signal_number == (-rc if rc is not None and rc < 0 else None)


def test_classify_termination_reason_strings_are_stable() -> None:
    assert sup._classify_termination(**_base_kwargs(launch_error="x")).termination_reason == "launch failed: x"
    assert sup._classify_termination(**_base_kwargs(timed_out=True)).termination_reason == "deadline exceeded after 12.3s"
    assert sup._classify_termination(**_base_kwargs(resource_reason="r")).termination_reason == "r"
    assert sup._classify_termination(**_base_kwargs(interrupted=True)).termination_reason == "supervisor interrupted"
    assert sup._classify_termination(**_base_kwargs(failure_code="MEMORY_BUDGET")).termination_reason == "worker reported MEMORY_BUDGET"
    assert sup._classify_termination(**_base_kwargs(returncode=-9, primary_completed=False)).termination_reason == "signal 9"
    assert sup._classify_termination(**_base_kwargs()).termination_reason is None
    assert sup._classify_termination(**_base_kwargs(primary_completed=False)).termination_reason == (
        "exit zero without a new completed domain artifact"
    )
    assert sup._classify_termination(**_base_kwargs(returncode=1, primary_completed=False)).termination_reason == "exit_code=1"


def test_sample_once_counts_and_rejects_in_order(monkeypatch) -> None:
    from src.mhs.resources import MhsMemoryBudget

    budget = MhsMemoryBudget()
    state = sup._SupervisionState(pid=1234, swap_baseline=0, last_heartbeat=0.0)
    monkeypatch.setattr(sup, "_workload_memory", lambda pid: (budget.total_tree_pss_bytes + 1, 7, 0))
    monkeypatch.setattr(sup, "current_mhs_headroom_bytes", lambda: 0)
    reason = sup._sample_once(state, budget)
    assert reason == f"sampled tree PSS {budget.total_tree_pss_bytes + 1} exceeds {budget.total_tree_pss_bytes}"
    assert state.samples == 1
    assert state.pss_peak == budget.total_tree_pss_bytes + 1
    assert state.uss_peak == 7
    assert state.min_available == 0


def test_sample_once_telemetry_loss_is_uncounted_rejection(monkeypatch) -> None:
    from src.mhs.resources import MhsMemoryBudget

    budget = MhsMemoryBudget()
    state = sup._SupervisionState(pid=1, swap_baseline=None, last_heartbeat=0.0)

    def _gone(pid):
        raise OSError("gone")

    monkeypatch.setattr(sup, "_workload_memory", _gone)
    assert sup._sample_once(state, budget) == "missing safety telemetry: gone"
    assert state.samples == 0
    assert state.pss_peak is None


def test_sample_once_boundary_equality_is_within_budget(monkeypatch) -> None:
    from src.mhs.resources import MhsMemoryBudget

    budget = MhsMemoryBudget()
    state = sup._SupervisionState(pid=1, swap_baseline=100, last_heartbeat=0.0)
    monkeypatch.setattr(sup, "_workload_memory", lambda pid: (budget.total_tree_pss_bytes, 9, 100))
    monkeypatch.setattr(sup, "current_mhs_headroom_bytes", lambda: budget.min_available_bytes)
    assert sup._sample_once(state, budget) is None
    assert state.swap_growth is None


def test_sample_once_swap_growth_only_above_baseline(monkeypatch) -> None:
    from src.mhs.resources import MhsMemoryBudget

    budget = MhsMemoryBudget()
    state = sup._SupervisionState(pid=1, swap_baseline=100, last_heartbeat=0.0)
    current = {"bytes": 100}
    monkeypatch.setattr(sup, "_workload_memory", lambda pid: (100, 50, current["bytes"]))
    monkeypatch.setattr(sup, "current_mhs_headroom_bytes", lambda: 8 * 2**30)
    assert sup._sample_once(state, budget) is None
    assert state.swap_growth is None
    current["bytes"] = 150
    assert sup._sample_once(state, budget) == "swap growth 50 bytes observed"


def test_sample_once_unavailable_swap_keeps_run_monitored(monkeypatch) -> None:
    from src.mhs.resources import MhsMemoryBudget

    budget = MhsMemoryBudget()
    state = sup._SupervisionState(pid=1, swap_baseline=100, last_heartbeat=0.0)
    monkeypatch.setattr(sup, "_workload_memory", lambda pid: (100, 50, None))
    monkeypatch.setattr(sup, "current_mhs_headroom_bytes", lambda: 8 * 2**30)
    assert sup._sample_once(state, budget) is None
    assert state.samples == 1
    assert state.swap_growth is None


def test_interrupt_before_launch_publishes_interrupted_envelope(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path, "ki_launch")
    db = _registry(tmp_path)
    run_id = "a" * 32

    def _ki(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", _ki)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result,
        poll_seconds=0.05, registry_path=db, run_id=run_id,
    )
    assert run.status == "interrupted"
    assert run.exit_code is None
    assert run.signal_number is None
    assert run.samples_taken == 0
    envelope = json.loads(result.read_text(encoding="utf-8"))
    assert envelope["execution"]["status"] == "interrupted"
    assert _finalizations(db, run_id) == 1


def test_interrupt_during_wait_loop_reaps_and_publishes(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path, "ki_wait")
    db = _registry(tmp_path)
    run_id = "b" * 32
    proc = _ScriptedProc(waits=[KeyboardInterrupt(), -15])
    calls: list = []
    _install_scripted(monkeypatch, proc)
    monkeypatch.setattr(
        sup, "_terminate_group",
        lambda p, timeout=5.0: (calls.append(p), setattr(p, "terminated", True)),
    )
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result,
        poll_seconds=0.05, registry_path=db, run_id=run_id,
    )
    assert run.status == "interrupted"
    assert run.signal_number == 15
    assert calls == [proc]
    assert _finalizations(db, run_id) == 1


def test_interrupt_with_failing_reap_never_leaves_return_code_unbound(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    for name, run_id, polls in (
        ("ki_poll_none", "c" * 32, [None]),
        ("ki_poll_boom", "d" * 32, [OSError("poll gone")]),
    ):
        result = _result(tmp_path, name)
        proc = _ScriptedProc(waits=[KeyboardInterrupt(), OSError("wait gone")], polls=polls)
        _install_scripted(monkeypatch, proc)
        run = sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=result,
            poll_seconds=0.05, registry_path=_registry(tmp_path), run_id=run_id,
        )
        assert run.status == "interrupted"
        assert run.exit_code is None
        assert run.signal_number is None
        assert result.is_file()
        assert _finalizations(_registry(tmp_path), run_id) == 1


def test_second_interrupt_during_reap_still_publishes(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path, "ki_second")
    db = _registry(tmp_path)
    run_id = "e" * 32
    proc = _ScriptedProc(waits=[KeyboardInterrupt(), KeyboardInterrupt()], polls=[None])
    _install_scripted(monkeypatch, proc)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result,
        poll_seconds=0.05, registry_path=db, run_id=run_id,
    )
    assert run.status == "interrupted"
    assert result.is_file()
    assert _finalizations(db, run_id) == 1


def _install_raise_once_terminate(monkeypatch, calls: list):
    def _once(proc, timeout=5.0):
        calls.append(proc)
        if len(calls) == 1:
            raise KeyboardInterrupt
        proc.terminated = True

    monkeypatch.setattr(sup, "_terminate_group", _once)


def test_deadline_outranks_interrupt_during_termination(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path, "deadline_ki")
    db = _registry(tmp_path)
    run_id = "f" * 32
    proc = _ScriptedProc()
    _install_scripted(monkeypatch, proc)
    calls: list = []
    _install_raise_once_terminate(monkeypatch, calls)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result,
        poll_seconds=0.02, timeout_seconds=0.2, registry_path=db, run_id=run_id,
    )
    assert run.status == "timed_out"
    assert calls
    assert _finalizations(db, run_id) == 1


def test_resource_stop_outranks_interrupt_during_termination(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path, "resource_ki")
    db = _registry(tmp_path)
    run_id = "a1" * 16
    proc = _ScriptedProc()
    _install_scripted(monkeypatch, proc)
    monkeypatch.setattr(sup, "_workload_memory", lambda pid: (10**12, 0, 0))
    calls: list = []
    _install_raise_once_terminate(monkeypatch, calls)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result,
        poll_seconds=0.02, registry_path=db, run_id=run_id,
    )
    assert run.status == "resource_rejected"
    assert run.termination_reason is not None
    assert run.termination_reason.startswith("sampled tree PSS")
    assert _finalizations(db, run_id) == 1


def test_worker_resource_code_ranks_below_interrupt(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path, "code_ki")
    db = _registry(tmp_path)
    run_id = "b1" * 16

    def _on_start(proc):
        staging = result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps({"status": "failed", "error_code": "MEMORY_BUDGET"}))

    proc = _ScriptedProc(waits=[KeyboardInterrupt(), 0])
    _install_scripted(monkeypatch, proc)

    def _factory(*args, **kwargs):
        _on_start(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", _factory)
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=result,
        poll_seconds=0.05, registry_path=db, run_id=run_id,
    )
    assert run.status == "interrupted"
    assert _finalizations(db, run_id) == 1


def test_log_open_failure_publishes_failed_envelope_then_raises(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    result = _result(tmp_path, "logfail")
    db = _registry(tmp_path)
    run_id = "c1" * 16
    log_path = result.parent / "result.log"
    real_open = open

    def _gated(path, *args, **kwargs):
        if str(path) == str(log_path):
            raise OSError("disk full")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _gated)
    _install_scripted(monkeypatch, _ScriptedProc())
    with pytest.raises(OSError, match="disk full"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=result,
            poll_seconds=0.05, registry_path=db, run_id=run_id,
        )
    envelope = json.loads(result.read_text(encoding="utf-8"))
    assert envelope["execution"]["status"] == "failed"
    assert envelope["execution"]["termination_reason"] == "launch failed: disk full"
    assert _finalizations(db, run_id) == 1
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'run_operations'").fetchall()
    finally:
        conn.close()
    assert rows == []


def test_log_handle_closed_on_every_path(tmp_path, monkeypatch) -> None:
    start, end = _stamps()
    real_open = open
    handles: list = []

    def _tracking(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(sup, "open", _tracking, raising=False)

    def _completed_on_start(proc, _result):
        staging = _result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(_completed_domain()))

    ok_result = _result(tmp_path, "handle_ok")
    ok_proc = _ScriptedProc(waits=[0])
    _install_scripted(monkeypatch, ok_proc)

    def _with_domain(*args, **kwargs):
        _completed_on_start(None, ok_result)
        return ok_proc

    monkeypatch.setattr(subprocess, "Popen", _with_domain)
    sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=ok_result,
        poll_seconds=0.05, registry_path=_registry(tmp_path), run_id="d1" * 16,
    )

    def _boom(*args, **kwargs):
        raise OSError("no launch")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    with pytest.raises(OSError, match="no launch"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=_result(tmp_path, "handle_fail"),
            poll_seconds=0.05, registry_path=_registry(tmp_path), run_id="e1" * 16,
        )

    def _ki(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "Popen", _ki)
    sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=_result(tmp_path, "handle_ki"),
        poll_seconds=0.05, registry_path=_registry(tmp_path), run_id="f1" * 16,
    )
    assert handles
    assert all(getattr(handle, "closed", False) for handle in handles)


def test_heartbeat_cadence_bounded_by_interval(tmp_path, monkeypatch, caplog) -> None:
    import logging
    import math

    start, end = _stamps()
    result = _result(tmp_path, "beat")

    def _on_start(proc):
        staging = result.parent / ".staging_domain.json"
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text(json.dumps(_completed_domain()))

    _install_scripted(monkeypatch, _ScriptedProc())
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _DelayedProc(_on_start))
    monkeypatch.setattr(sup, "HEARTBEAT_SECONDS", 0.1)
    with caplog.at_level(logging.INFO, logger="src.application.mhs_supervisor"):
        run = sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=result,
            poll_seconds=0.02, registry_path=_registry(tmp_path),
        )
    beats = [r for r in caplog.records if r.getMessage().startswith("[SYS] heartbeat wall_s=")]
    assert run.status == "completed"
    assert len(beats) >= 1
    assert len(beats) <= max(1, math.floor(run.wall_seconds / 0.1))


class _DelayedProc(_ScriptedProc):
    """Real-time worker replacement with a fixed exit delay for cadence tests."""

    def __init__(self, on_start, delay: float = 0.5) -> None:
        super().__init__()
        import time as _time

        self._deadline = _time.monotonic() + delay
        on_start(self)

    def wait(self, timeout=None):
        import time as _time

        if _time.monotonic() < self._deadline:
            if timeout is not None:
                _time.sleep(timeout)
            raise subprocess.TimeoutExpired(cmd=[], timeout=timeout)
        return 0


def test_single_run_builder_parity(tmp_path, monkeypatch) -> None:
    import ast as _ast

    from src.mhs.resources import MhsMemoryBudget

    start, end = _stamps()
    fail_result = _result(tmp_path, "bld_fail")

    def _boom(*args, **kwargs):
        raise OSError("no launch")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    with pytest.raises(OSError, match="no launch"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=fail_result,
            poll_seconds=0.05, registry_path=_registry(tmp_path), run_id="a2" * 16,
        )
    failed_env = json.loads(fail_result.read_text(encoding="utf-8"))
    assert failed_env["execution"]["sampled_tree_pss_peak_bytes"] is None
    assert failed_env["execution"]["sampled_tree_uss_peak_bytes"] is None
    assert failed_env["execution"]["min_available_bytes"] is None
    assert failed_env["execution"]["process_swap_growth_bytes"] is None

    outcome = sup._TerminationOutcome(
        status="failed", termination_reason="launch failed: x", exit_code=None, signal_number=None,
    )
    staging = tmp_path / "staging.json"
    log_path = tmp_path / "staging.log"
    built = sup._build_supervised_run(
        outcome=outcome, command=("worker",), start=start, end=end, data_root=None,
        staging=staging, log_path=log_path, domain_written=False, wall_seconds=0.1,
        cpu_seconds=None, gnu_rss=None, state=None, poll_seconds=0.05,
        budget=MhsMemoryBudget(), run_id="b2" * 16,
    )
    assert built.samples_taken == 0
    assert built.sampled_tree_pss_peak_bytes is None
    assert built.cpu_scope == sup.CPU_SCOPE
    assert built.memory_scope == sup.MEMORY_SCOPE
    assert built.sample_interval_seconds == 0.05
    assert built.result_output_path == str(staging)

    ok_result = _result(tmp_path, "bld_ok")

    def _on_start(proc):
        staging_ok = ok_result.parent / ".staging_domain.json"
        staging_ok.parent.mkdir(parents=True, exist_ok=True)
        staging_ok.write_text(json.dumps(_completed_domain()))

    _install_scripted(monkeypatch, _ScriptedProc(waits=[0]))
    monkeypatch.setattr(
        subprocess, "Popen",
        lambda *a, **k: (_on_start(None), _ScriptedProc(waits=[0]))[1],
    )
    run = sup.run_mhs_process_backtest(
        start=start, end=end, data_root=None, result_output=ok_result,
        poll_seconds=0.05, registry_path=_registry(tmp_path), run_id="c2" * 16,
    )
    assert run.cpu_scope == sup.CPU_SCOPE
    assert run.memory_scope == sup.MEMORY_SCOPE
    assert run.sample_interval_seconds == 0.05

    source = Path(sup.__file__).read_text(encoding="utf-8")
    module = _ast.parse(source)
    builder = next(
        node for node in _ast.walk(module)
        if isinstance(node, _ast.FunctionDef) and node.name == "_build_supervised_run"
    )
    assert sum(
        isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name) and node.func.id == "MhsSupervisedRun"
        for node in _ast.walk(builder)
    ) == 1
    main = next(
        node for node in _ast.walk(module)
        if isinstance(node, _ast.FunctionDef) and node.name == "run_mhs_process_backtest"
    )
    call_ids = [
        node.func.id for node in _ast.walk(main)
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name)
    ]
    assert call_ids.count("_publish_envelope") == 1
    assert "MhsSupervisedRun" not in call_ids


def test_supervised_request_validation_rejects_bad_controls(tmp_path) -> None:
    start, end = _stamps()
    naive_start = pd.Timestamp("2022-01-01")
    naive_end = pd.Timestamp("2022-01-04")
    fresh = tmp_path / "fresh" / "result.json"
    with pytest.raises(ValueError, match="timezone-aware"):
        sup.run_mhs_process_backtest(
            start=naive_start, end=end, data_root=None, result_output=fresh,
            poll_seconds=0.05, registry_path=_registry(tmp_path),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        sup.run_mhs_process_backtest(
            start=start, end=naive_end, data_root=None, result_output=fresh,
            poll_seconds=0.05, registry_path=_registry(tmp_path),
        )
    with pytest.raises(ValueError, match="precede"):
        sup.run_mhs_process_backtest(
            start=end, end=start, data_root=None, result_output=fresh,
            poll_seconds=0.05, registry_path=_registry(tmp_path),
        )
    with pytest.raises(ValueError, match="parquet"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=fresh,
            targets_output=tmp_path / "bad.txt", poll_seconds=0.05,
            registry_path=_registry(tmp_path),
        )
    with pytest.raises(ValueError, match="poll_seconds"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=fresh,
            poll_seconds=0.0, registry_path=_registry(tmp_path),
        )
    with pytest.raises(ValueError, match="timeout_seconds"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=fresh,
            poll_seconds=0.05, timeout_seconds=-1.0, registry_path=_registry(tmp_path),
        )
    with pytest.raises(ValueError, match="numeric"):
        sup.run_mhs_process_backtest(
            start=start, end=end, data_root=None, result_output=fresh,
            tracking_error_threshold="bad", poll_seconds=0.05,
            registry_path=_registry(tmp_path),
        )
