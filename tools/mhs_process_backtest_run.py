"""Supervised production 3m process evaluation runner.

The supervisor launches the existing production CLI in a dedicated process
group, waits for actual termination, samples the workload tree, and persists
an atomic outcome report. It never imports a scratch strategy.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
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

_logger = logging.getLogger(__name__)

PSS_LIMIT_BYTES: int = int(2.5 * 2**30)
HEADROOM_FLOOR_BYTES: int = 2 * 2**30
GRACE_SECONDS: float = 5.0
HEARTBEAT_SECONDS: float = 30.0
CPU_SCOPE: str = "waited workload user+system CPU seconds via GNU time when available"
MEMORY_SCOPE: str = (
    "sampled child-tree PSS/USS bytes at poll interval; "
    "GNU max RSS is maximum individual-process bytes"
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


def _system_headroom_bytes() -> int:
    """Minimum of host available and known cgroup remaining bytes."""
    import psutil

    available = int(psutil.virtual_memory().available)
    remaining: int | None = None
    try:
        with open("/sys/fs/cgroup/memory.current", encoding="utf-8") as handle:
            current = int(handle.read().strip())
        with open("/sys/fs/cgroup/memory.max", encoding="utf-8") as handle:
            raw = handle.read().strip()
        if raw not in ("", "max"):
            remaining = int(raw) - current
    except (OSError, ValueError):
        remaining = None
    if remaining is not None and remaining >= 0:
        return min(available, remaining)
    return available


def _swap_used_bytes() -> int | None:
    """Current swap used bytes, or null when unreadable."""
    try:
        import psutil

        return int(psutil.swap_memory().used)
    except Exception:  # noqa: BLE001
        return None


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


def run_mhs_process_backtest(
    *, start: pd.Timestamp, end: pd.Timestamp, data_root: str | None,
    output: Path, failure_output: Path, run_output: Path,
    targets_output: Path | None = None,
    tracking_error_threshold: float | None = None,
    timeout_seconds: float | None = None,
    poll_seconds: float = 0.25,
) -> MhsSupervisedRun:
    """Wait for production evaluation termination and persist scoped measurements.

    Args:
        start: Explicit timezone-aware input start.
        end: Explicit timezone-aware registered evaluation end.
        data_root: Existing OHLCV root override.
        output: Fresh primary performance destination.
        failure_output: Fresh dedicated domain failure destination.
        run_output: Fresh supervisor outcome destination.
        targets_output: Optional fresh exact-target parquet destination.
        tracking_error_threshold: Existing process adoption control.
        timeout_seconds: Positive optional wall timeout, not a strategy horizon.
        poll_seconds: Positive observation interval in seconds.

    Returns:
        Final child outcome after process-group completion and atomic reporting.

    Raises:
        ValueError: Controls or fresh/distinct evidence destinations are invalid.
        OSError: Child launch or supervisor evidence persistence fails.
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
    if not isinstance(poll_seconds, (int, float)) or not float(poll_seconds) > 0:
        raise ValueError("poll_seconds must be positive")
    if timeout_seconds is not None and (
        not isinstance(timeout_seconds, (int, float)) or not float(timeout_seconds) > 0
    ):
        raise ValueError("timeout_seconds must be positive")
    if tracking_error_threshold is not None and not isinstance(
        tracking_error_threshold, (int, float)
    ):
        raise ValueError("tracking_error_threshold must be numeric")
    try:
        from src.mhs.process_backtest import (
            PROCESS_INVENTORY_REPORT_PATH,
            PROCESS_POLICY_REPORT_PATH,
            PROCESS_REPORT_PATH,
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"reserved destination lookup failed: {exc}") from exc
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
    command = [
        sys.executable, "-m", "src.cli.main",
        "research", "run", "portfolio", "mhs-process-backtest",
        "--start", start.isoformat(), "--end", end.isoformat(),
        "--output", str(output), "--failure-output", str(failure_output),
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
    swap_baseline = _swap_used_bytes()
    swap_growth: int | None = None
    timed_out = False
    resource_reason: str | None = None
    interrupted = False
    last_heartbeat = wall_start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        log_handle = open(log_path, "w", encoding="utf-8")  # noqa: PTH123
    except OSError:
        raise
    proc: subprocess.Popen[bytes] | None = None
    try:
        try:
            proc = subprocess.Popen(  # noqa: S603
                scoped_command, stdout=log_handle, stderr=subprocess.STDOUT,
                start_new_session=True, shell=False,
            )
        except OSError:
            raise
        pid = proc.pid
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
                headroom = _system_headroom_bytes()
            except Exception as tel_exc:  # noqa: BLE001
                resource_reason = f"missing safety telemetry: {tel_exc}"
                _terminate_group(proc)
                returncode = proc.wait()
                break
            samples += 1
            pss_peak = pss if pss_peak is None else max(pss_peak, pss)
            uss_peak = uss if uss_peak is None else max(uss_peak, uss)
            min_available = headroom if min_available is None else min(min_available, headroom)
            swap_current = _swap_used_bytes()
            if swap_baseline is not None and swap_current is not None:
                growth = swap_current - swap_baseline
                if growth > 0:
                    swap_growth = growth if swap_growth is None else max(swap_growth, growth)
            if pss > PSS_LIMIT_BYTES:
                resource_reason = f"sampled tree PSS {pss} exceeds {PSS_LIMIT_BYTES}"
                _terminate_group(proc)
                returncode = proc.wait()
                break
            if headroom < HEADROOM_FLOOR_BYTES:
                resource_reason = f"headroom {headroom} below {HEADROOM_FLOOR_BYTES}"
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
    if returncode is None:
        try:
            returncode = proc.wait(timeout=GRACE_SECONDS)
        except Exception:  # noqa: BLE001
            returncode = proc.poll()
    cpu_seconds, gnu_rss = _parse_gnu_metrics(log_path)
    primary_written = output.exists()
    failure_written = failure_output.exists()
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
    )
    _atomic_write_json(run_output, dataclasses.asdict(run))  # type: ignore[arg-type]
    return run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supervised production 3m process evaluation.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failure-output", required=True)
    parser.add_argument("--run-output", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--targets-output", default=None)
    parser.add_argument("--rebalance-tracking-error-threshold", type=float, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the production process CLI under explicit resource supervision.

    Args:
        argv: Explicit controls or command-line arguments.

    Returns:
        Zero only for completed evaluation; nonzero for every other outcome.

    Raises:
        SystemExit: CLI arguments are invalid.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        start = pd.Timestamp(args.start, tz="UTC")
        end = pd.Timestamp(args.end, tz="UTC")
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"invalid start/end: {exc}") from exc
    try:
        run = run_mhs_process_backtest(
            start=start,
            end=end,
            data_root=args.data_root,
            output=Path(args.output),
            failure_output=Path(args.failure_output),
            run_output=Path(args.run_output),
            targets_output=Path(args.targets_output) if args.targets_output else None,
            tracking_error_threshold=args.rebalance_tracking_error_threshold,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    except ValueError as exc:
        raise SystemExit(f"invalid supervisor arguments: {exc}") from exc
    return 0 if run.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
