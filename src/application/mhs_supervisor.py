"""Supervised source-owned three-minute process evaluation.

The supervisor launches the source-owned worker module in a dedicated process
group, waits for actual termination, samples the workload tree, and persists
an atomic outcome report.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from src.mhs.process import ProcessExecutionPolicy
from src.mhs.process_backtest import (
    PROCESS_INVENTORY_REPORT_PATH,
    PROCESS_POLICY_REPORT_PATH,
    PROCESS_REPORT_PATH,
)
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
    output_path: str
    failure_output_path: str
    log_path: str
    primary_artifact_written: bool
    failure_artifact_written: bool
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


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
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
    output: Path, failure_output: Path, run_output: Path,
    targets_output: Path | None = None,
    tracking_error_threshold: float | None = None,
    timeout_seconds: float | None = None,
    poll_seconds: float = 0.25,
    memory_budget: MhsMemoryBudget | None = None,
) -> MhsSupervisedRun:
    """Wait for a source-owned three-minute evaluation and persist observed outcome.

    Args:
        start: Explicit timezone-aware source start.
        end: Explicit timezone-aware registered evaluation end.
        data_root: Existing OHLCV root override.
        output: Fresh primary inventory report destination.
        failure_output: Fresh domain-failure evidence destination.
        run_output: Fresh supervisor outcome destination.
        targets_output: Optional fresh exact-target parquet destination.
        tracking_error_threshold: Existing optional process adoption control.
        timeout_seconds: Optional positive finite wall timeout.
        poll_seconds: Positive finite resource observation interval.
        memory_budget: Stage ceilings and reserve shared with the worker.
    Returns:
        Outcome after process-group termination and atomic evidence persistence.
    Raises:
        ValueError: Controls or fresh/distinct destinations are invalid.
        OSError: Launch or supervisor evidence persistence fails.
    """
    if not isinstance(start, pd.Timestamp) or start.tzinfo is None:
        raise ValueError("start must be a timezone-aware Timestamp")
    if not isinstance(end, pd.Timestamp) or end.tzinfo is None:
        raise ValueError("end must be a timezone-aware Timestamp")
    if start >= end:
        raise ValueError("start must precede end")
    for label, candidate, suffix in (
        ("output", output, ".json"),
        ("failure_output", failure_output, ".json"),
        ("run_output", run_output, ".json"),
    ):
        if not isinstance(candidate, Path) or candidate.suffix != suffix:
            raise ValueError(f"{label} must be a {suffix} path")
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
    reserved = {
        PROCESS_REPORT_PATH.resolve(),
        PROCESS_POLICY_REPORT_PATH.resolve(),
        PROCESS_INVENTORY_REPORT_PATH.resolve(),
    }
    log_path = run_output.parent / f"{run_output.stem}.log"
    candidates: list[tuple[str, Path]] = [
        ("output", output),
        ("failure_output", failure_output),
        ("run_output", run_output),
        ("log", log_path),
    ]
    if targets_output is not None:
        candidates.append(("targets_output", targets_output))
    seen: set[Path] = set()
    for label, candidate in candidates:
        resolved = candidate.resolve() if isinstance(candidate, Path) else candidate
        if resolved in reserved:
            raise ValueError(f"{label} conflicts with reserved evidence")
        if resolved in seen:
            raise ValueError(f"{label} must be distinct from other destinations")
        seen.add(resolved)
        if os.path.lexists(candidate):
            raise ValueError(f"{label} must be fresh: {candidate} already exists")
    budget = resolve_mhs_memory_budget(memory_budget)
    command = [
        sys.executable, "-m", "src.application.mhs_worker",
        "--start", start.isoformat(), "--end", end.isoformat(),
        "--output", str(output), "--failure-output", str(failure_output),
        "--total-tree-pss-bytes", str(budget.total_tree_pss_bytes),
        "--replay-tree-pss-bytes", str(budget.replay_tree_pss_bytes),
        "--min-available-bytes", str(budget.min_available_bytes),
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
    log_handle = open(log_path, "w", encoding="utf-8")  # noqa: PTH123
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(  # noqa: S603
            scoped_command, stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=True, shell=False,
        )
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
    primary_written = output.exists()
    failure_written = failure_output.exists()
    failure_code = _failure_resource_code(failure_output) if failure_written else None
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
    elif returncode == 0 and _primary_completed(output):
        status = "completed"
    elif returncode == 0:
        status = "failed"
        termination_reason = "exit zero without a new completed primary artifact"
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
        output_path=str(output),
        failure_output_path=str(failure_output),
        log_path=str(log_path),
        primary_artifact_written=bool(primary_written),
        failure_artifact_written=bool(failure_written),
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
    )
    _atomic_write_json(run_output, dataclasses.asdict(run))
    return run
