from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import psutil

_logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RuntimeProbeSnapshot:
    run_id: str
    sample_seq: int
    stage_path: str
    pid: int
    ppid: int
    role: Literal["parent", "child"]
    rss_mb: float
    pss_mb: float
    tree_rss_mb: float
    tree_pss_mb: float
    parent_vmhwm_mb: float
    sample_elapsed_ms: float
    status: str


@dataclass(frozen=True, slots=True)
class RuntimeSpanSummary:
    stage_path: str
    calls: int
    elapsed_ms: float
    rss_delta_mb: float
    tree_rss_peak_mb: float
    tree_pss_peak_mb: float
    peak_pid: int
    peak_role: str


_EMPTY_SPAN_SUMMARY = RuntimeSpanSummary(
    stage_path="",
    calls=0,
    elapsed_ms=0.0,
    rss_delta_mb=0.0,
    tree_rss_peak_mb=0.0,
    tree_pss_peak_mb=0.0,
    peak_pid=0,
    peak_role="",
)

_SAMPLE_MS_DEFAULT = 250
_HOT_SAMPLE_MS_DEFAULT = 50
_SAMPLE_MS_MIN = 50
_SAMPLE_MS_MAX = 1000
_HOT_SAMPLE_MS_MIN = 50
_HOT_SAMPLE_MS_MAX = 250
_SAMPLE_SLOW_THRESHOLD_MS = 50.0
_DEGRADE_SAMPLE_MS = 250
_JSONL_PATH = Path("logs/futures/optimization/l2_runtime_probe.jsonl")


def _clamp(value: int, lo: int, hi: int, name: str, tag: str) -> int:
    if lo <= value <= hi:
        return value
    _logger.warning(
        "[SYS] stage=l2_probe status=config_fallback key=%s value=%d range=%d..%d tag=%s",
        name, value, lo, hi, tag,
    )
    return max(lo, min(hi, value))


def _pss_mib(proc: psutil.Process) -> float:
    try:
        info = proc.memory_full_info()
        raw = getattr(info, "pss", None)
        pss: float = float(raw) if raw is not None else -1.0
        if pss > 0.0:
            return pss / (1024.0 * 1024.0)
    except (psutil.AccessDenied, psutil.NoSuchProcess, FileNotFoundError, RuntimeError):
        pass
    return -1.0


def _rss_mib(proc: psutil.Process) -> float:
    try:
        rss: float = float(proc.memory_info().rss)
        return rss / (1024.0 * 1024.0)
    except (psutil.AccessDenied, psutil.NoSuchProcess, FileNotFoundError, RuntimeError):
        return -1.0


def _parent_vmhwm_mib() -> float:
    try:
        with Path(f"/proc/{os.getpid()}/status").open() as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except (FileNotFoundError, OSError, ValueError):
        pass
    return -1.0


def _collect_tree_snap(parent_pid: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[int] = set()

    def _walk(p: psutil.Process) -> None:
        pid = p.pid
        if pid in seen:
            return
        seen.add(pid)
        try:
            ppid = p.ppid()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            ppid = -1
        try:
            name = p.name() or ""
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            name = ""
        role: Literal["parent", "child"] = "parent" if pid == parent_pid else "child"
        rss = _rss_mib(p)
        pss = _pss_mib(p)
        status = "ok" if rss >= 0 else "denied"
        out.append({
            "pid": pid,
            "ppid": ppid,
            "role": role,
            "rss_mb": rss if rss >= 0 else -1.0,
            "pss_mb": pss,
            "name": name,
            "status": status,
        })
        try:
            for child in p.children(recursive=True):
                _walk(child)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

    try:
        _walk(psutil.Process(parent_pid))
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        out.append({
            "pid": parent_pid,
            "ppid": 0,
            "role": "parent",
            "rss_mb": -1.0,
            "pss_mb": -1.0,
            "name": "",
            "status": "denied",
        })
    return out


class _SpanState:
    __slots__ = (
        "enter_rss",
        "enter_tree_pss",
        "enter_tree_rss",
        "name",
        "peak_pid",
        "peak_role",
        "peak_tree_pss",
        "peak_tree_rss",
        "t_ns",
    )

    def __init__(self, name: str) -> None:
        self.name = name
        self.t_ns = time.perf_counter_ns()
        self.enter_rss: float = 0.0
        self.enter_tree_rss: float = 0.0
        self.enter_tree_pss: float = 0.0
        self.peak_tree_rss: float = 0.0
        self.peak_tree_pss: float = 0.0
        self.peak_pid: int = 0
        self.peak_role: str = ""


class _SpanTracker:
    __slots__ = (
        "_calls",
        "_current_span",
        "_peak_pid",
        "_peak_role",
        "_peak_tree_pss",
        "_peak_tree_rss",
        "_stacks",
        "_total_elapsed_ns",
    )

    def __init__(self) -> None:
        self._stacks: list[_SpanState] = []
        self._total_elapsed_ns: int = 0
        self._calls: int = 0
        self._peak_tree_rss: float = 0.0
        self._peak_tree_pss: float = 0.0
        self._peak_pid: int = 0
        self._peak_role: str = ""
        self._current_span: _SpanState | None = None

    def enter(self, name: str, rss: float, tree_rss: float, tree_pss: float) -> None:
        s = _SpanState(name)
        s.enter_rss = rss
        s.enter_tree_rss = tree_rss
        s.enter_tree_pss = tree_pss
        s.peak_tree_rss = tree_rss
        s.peak_tree_pss = tree_pss
        self._stacks.append(s)
        self._current_span = s

    def update_peak(self, tree_rss: float, tree_pss: float, peak_pid: int, peak_role: str) -> None:
        current = self._current_span
        if current is None:
            return
        if tree_rss > current.peak_tree_rss:
            current.peak_tree_rss = tree_rss
            current.peak_pid = peak_pid
        if tree_pss > current.peak_tree_pss:
            current.peak_tree_pss = tree_pss
        current.peak_role = peak_role
        if tree_rss > self._peak_tree_rss:
            self._peak_tree_rss = tree_rss
            self._peak_pid = peak_pid
        if tree_pss > self._peak_tree_pss:
            self._peak_tree_pss = tree_pss
        self._peak_role = peak_role

    def exit(self) -> _SpanState | None:
        if not self._stacks:
            return None
        s = self._stacks.pop()
        elapsed_ns = time.perf_counter_ns() - s.t_ns
        self._total_elapsed_ns += elapsed_ns
        self._calls += 1
        self._current_span = self._stacks[-1] if self._stacks else None
        return s

    def summary(self, name: str, rss_delta_mb: float) -> RuntimeSpanSummary:
        elapsed_ms = self._total_elapsed_ns / 1_000_000.0 if self._total_elapsed_ns > 0 else 0.0
        return RuntimeSpanSummary(
            stage_path=name,
            calls=self._calls,
            elapsed_ms=elapsed_ms,
            rss_delta_mb=rss_delta_mb,
            tree_rss_peak_mb=self._peak_tree_rss,
            tree_pss_peak_mb=self._peak_tree_pss,
            peak_pid=self._peak_pid,
            peak_role=self._peak_role,
        )


class _SampleRecord:
    __slots__ = ("reason", "sample_seq", "snapshots", "stage_path")

    def __init__(self, sample_seq: int, stage_path: str, snapshots: list[dict[str, Any]], reason: str) -> None:
        self.sample_seq = sample_seq
        self.stage_path = stage_path
        self.snapshots = snapshots
        self.reason = reason


class L2RuntimeProbe:
    def __init__(
        self,
        *,
        enabled: bool,
        sample_interval_ms: int,
        hot_sample_interval_ms: int,
        jsonl_enabled: bool,
        jsonl_path: Path,
    ) -> None:
        self._enabled = enabled
        self._sample_interval_ms = sample_interval_ms
        self._hot_sample_interval_ms = hot_sample_interval_ms
        self._jsonl_enabled = jsonl_enabled
        self._jsonl_path = jsonl_path

        self._run_id: str = ""
        self._run_start_ns: int = 0
        self._sampler_thread: threading.Thread | None = None
        self._sampler_stop = threading.Event()
        self._sample_seq: int = 0
        self._degraded: bool = False
        self._slow_sample_count: int = 0
        self._degraded_reported: bool = False
        self._peak_owner_stage: str = ""
        self._peak_pid: int = 0
        self._peak_role: str = ""
        self._peak_tree_rss: float = 0.0
        self._peak_tree_pss: float = 0.0
        self._lock = threading.Lock()

        self._parent_pid: int = os.getpid()
        self._prepare_peak_tree_pss_ever: float = 0.0
        self._peak_sample_seq: int = 0

        self._spans: dict[str, _SpanTracker] = {}
        self._span_stack: list[str] = []

        self._records: list[dict[str, Any]] = []

        self._enter_rss: float = 0.0
        self._enter_tree_rss: float = 0.0
        self._enter_tree_pss: float = 0.0

    @classmethod
    def from_environment(cls, *, logger: Any, base_dir: Path) -> L2RuntimeProbe:
        _env_val = os.environ.get("L2_RUNTIME_PROBE_ENABLED", "false").lower()
        _enabled = logger.isEnabledFor(logging.DEBUG) and _env_val in ("true", "1", "yes")
        sample_ms = _clamp(
            int(os.environ.get("L2_RUNTIME_PROBE_SAMPLE_MS", str(_SAMPLE_MS_DEFAULT))),
            _SAMPLE_MS_MIN, _SAMPLE_MS_MAX, "L2_RUNTIME_PROBE_SAMPLE_MS", "from_env",
        )
        hot_ms = _clamp(
            int(os.environ.get("L2_RUNTIME_PROBE_HOT_SAMPLE_MS", str(_HOT_SAMPLE_MS_DEFAULT))),
            _HOT_SAMPLE_MS_MIN, _HOT_SAMPLE_MS_MAX, "L2_RUNTIME_PROBE_HOT_SAMPLE_MS", "from_env",
        )
        _jsonl_val = os.environ.get("L2_RUNTIME_PROBE_JSONL_ENABLED", "true").lower()
        jsonl_enabled = _jsonl_val in ("true", "1", "yes") if _enabled else False
        jsonl_path = base_dir / _JSONL_PATH
        return cls(
            enabled=_enabled,
            sample_interval_ms=sample_ms,
            hot_sample_interval_ms=hot_ms,
            jsonl_enabled=jsonl_enabled,
            jsonl_path=jsonl_path,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_run(self, *, stage: str) -> None:
        if not self._enabled:
            return
        import uuid
        if not self._run_id:
            self._run_id = str(uuid.uuid4())
        self._run_start_ns = time.perf_counter_ns()
        self._sample_seq = 0
        self._degraded = False
        self._slow_sample_count = 0
        self._degraded_reported = False
        self._records.clear()
        self._spans.clear()
        self._span_stack.clear()

        parent = psutil.Process(self._parent_pid)
        self._enter_rss = _rss_mib(parent)
        tree_snaps = _collect_tree_snap(self._parent_pid)
        self._enter_tree_rss = sum(s.get("rss_mb", 0.0) for s in tree_snaps if s.get("rss_mb", -1.0) >= 0.0)
        self._enter_tree_pss = sum(s.get("pss_mb", 0.0) for s in tree_snaps if s.get("pss_mb", -1.0) >= 0.0)

        if self._jsonl_enabled:
            try:
                self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                _logger.warning(
                    "[SYS] stage=l2_probe status=degraded reason=jsonl_mkdir_failed err=%s",
                    exc,
                )
                self._jsonl_enabled = False

        self._sampler_stop.clear()
        self._sampler_thread = threading.Thread(
            target=self._sampler_loop,
            daemon=True,
            name="l2-probe-sampler",
        )
        self._sampler_thread.start()

    def stop_run(self, *, outcome: str) -> tuple[RuntimeSpanSummary, ...]:
        if not self._enabled:
            return ()
        self._sampler_stop.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=2.0)

        if self._peak_tree_rss > 0.0:
            _logger.info(
                "[SYS] stage=l2_peak run_id=%s owner_stage=%s pid=%d role=%s "
                "rss_mb=%.3f pss_mb=%.3f tree_rss_mb=%.3f sample_seq=%d outcome=%s",
                self._run_id,
                self._peak_owner_stage,
                self._peak_pid,
                self._peak_role,
                self._peak_tree_rss,
                self._peak_tree_pss,
                self._peak_tree_rss,
                self._peak_sample_seq,
                outcome,
            )

        summaries: list[RuntimeSpanSummary] = []
        parent = psutil.Process(self._parent_pid)
        exit_rss = _rss_mib(parent)

        for name in sorted(self._spans):
            tr = self._spans[name]
            rss_delta = exit_rss - self._enter_rss if exit_rss >= 0 and self._enter_rss >= 0 else 0.0
            summaries.append(tr.summary(name, rss_delta_mb=rss_delta))

        for s in sorted(summaries, key=lambda x: x.elapsed_ms, reverse=True):
            _logger.info(
                "[SYS] stage=l2_span event=end run_id=%s stage_path=%s "
                "elapsed_ms=%.3f tree_rss_peak_mb=%.3f tree_pss_peak_mb=%.3f "
                "peak_pid=%d peak_role=%s calls=%d",
                self._run_id,
                s.stage_path,
                s.elapsed_ms,
                s.tree_rss_peak_mb,
                s.tree_pss_peak_mb,
                s.peak_pid,
                s.peak_role,
                s.calls,
            )

        self._sampler_thread = None
        return tuple(summaries)

    @contextmanager
    def span(self, stage: str, /, **fields: str | int | float | bool) -> Iterator[None]:
        if not self._enabled:
            yield
            return
        stage_path = "/".join([*self._span_stack, stage]) if self._span_stack else stage
        self._span_stack.append(stage)
        if stage_path not in self._spans:
            self._spans[stage_path] = _SpanTracker()
        tracker = self._spans[stage_path]

        with self._lock:
            parent = psutil.Process(self._parent_pid)
            rss = _rss_mib(parent)
            tree_snaps = _collect_tree_snap(self._parent_pid)
            tree_rss = sum(s.get("rss_mb", 0.0) for s in tree_snaps if s.get("rss_mb", -1.0) >= 0.0)
            tree_pss = sum(s.get("pss_mb", 0.0) for s in tree_snaps if s.get("pss_mb", -1.0) >= 0.0)
            tracker.enter(stage_path, rss, tree_rss, tree_pss)

        try:
            yield
        finally:
            with self._lock:
                _tree_snaps = _collect_tree_snap(self._parent_pid)
                _exit_tree_rss = sum(s.get("rss_mb", 0.0) for s in _tree_snaps if s.get("rss_mb", -1.0) >= 0.0)
                _exit_tree_pss = sum(s.get("pss_mb", 0.0) for s in _tree_snaps if s.get("pss_mb", -1.0) >= 0.0)
                _peak_pid = self._peak_pid or self._parent_pid
                tracker.update_peak(_exit_tree_rss, _exit_tree_pss, _peak_pid, self._peak_role or "parent")
                tracker.exit()

            if fields:
                field_str = " ".join(f"{k}={v}" for k, v in fields.items())
                _logger.info(
                    "[SYS] stage=l2_span event=end run_id=%s stage_path=%s %s",
                    self._run_id, stage_path, field_str,
                )
            self._span_stack.pop()

    def record(
        self,
        category: Literal["SYS", "DATA", "ALGO", "EVAL"],
        stage: str,
        /,
        **fields: str | int | float | bool,
    ) -> None:
        if not self._enabled:
            return
        field_str = " ".join(f"{k}={v}" for k, v in fields.items())
        if category == "EVAL":
            _logger.info(
                "[%s] stage=%s run_id=%s %s",
                category, stage, self._run_id, field_str,
            )
        else:
            _logger.info(
                "[%s] stage=%s run_id=%s %s",
                category, stage, self._run_id, field_str,
            )

    def snapshot_now(self, *, reason: str) -> RuntimeProbeSnapshot | None:
        if not self._enabled:
            return None
        t0 = time.perf_counter_ns()
        parent = psutil.Process(self._parent_pid)
        rss = _rss_mib(parent)
        pss = _pss_mib(parent)
        parent_vmhwm = _parent_vmhwm_mib()
        tree_snaps = _collect_tree_snap(self._parent_pid)
        tree_rss = sum(s.get("rss_mb", 0.0) for s in tree_snaps if s.get("rss_mb", -1.0) >= 0.0)
        tree_pss = sum(s.get("pss_mb", 0.0) for s in tree_snaps if s.get("pss_mb", -1.0) >= 0.0)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        stage_path = "/".join(self._span_stack) if self._span_stack else "root"
        with self._lock:
            seq = self._sample_seq
            self._sample_seq += 1

        role: Literal["parent", "child"] = "parent"
        snap = RuntimeProbeSnapshot(
            run_id=self._run_id,
            sample_seq=seq,
            stage_path=stage_path,
            pid=self._parent_pid,
            ppid=os.getppid(),
            role=role,
            rss_mb=rss if rss >= 0 else -1.0,
            pss_mb=pss,
            tree_rss_mb=tree_rss,
            tree_pss_mb=tree_pss,
            parent_vmhwm_mb=parent_vmhwm,
            sample_elapsed_ms=elapsed_ms,
            status="ok" if rss >= 0 else "degraded",
        )
        return snap

    def _sample_loop_active(self, interval_ms: int, stage_path: str) -> None:
        count = 0
        max_samples = max(1, int(30000.0 / max(interval_ms, 1)))
        t_start = time.perf_counter_ns()
        while not self._sampler_stop.is_set():
            elapsed_ns = time.perf_counter_ns() - t_start
            if elapsed_ns >= interval_ms * 1_000_000:
                t0 = time.perf_counter_ns()
                self._do_sample(reason="active_span", stage_path=stage_path)
                sample_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                if sample_ms > _SAMPLE_SLOW_THRESHOLD_MS:
                    self._slow_sample_count += 1
                else:
                    self._slow_sample_count = max(0, self._slow_sample_count - 1)
                if self._slow_sample_count >= 2 and not self._degraded:
                    self._degraded = True
                    _logger.warning(
                        "[SYS] stage=l2_probe status=degraded reason=sample_slow "
                        "slow_count=%d interval_ms=%d",
                        self._slow_sample_count, _DEGRADE_SAMPLE_MS,
                    )
                count += 1
                if count >= max_samples:
                    break
                t_start = time.perf_counter_ns()
            else:
                remaining = (interval_ms * 1_000_000 - elapsed_ns) // 1_000_000
                self._sampler_stop.wait(timeout=max(0.001, remaining / 1000.0))

    def _sampler_loop(self) -> None:
        interval_s = self._sample_interval_ms / 1000.0
        while not self._sampler_stop.is_set():
            t0 = time.perf_counter_ns()
            self._do_sample(reason="periodic", stage_path="/".join(self._span_stack) if self._span_stack else "root")
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            if elapsed_ms > _SAMPLE_SLOW_THRESHOLD_MS:
                self._slow_sample_count += 1
            else:
                self._slow_sample_count = max(0, self._slow_sample_count - 1)
            if self._slow_sample_count >= 2 and not self._degraded:
                self._degraded = True
                _logger.warning(
                    "[SYS] stage=l2_probe status=degraded reason=sample_slow "
                    "slow_count=%d interval_ms=%d",
                    self._slow_sample_count, _DEGRADE_SAMPLE_MS,
                )
            wait = interval_s - elapsed_ms / 1000.0
            if wait > 0:
                self._sampler_stop.wait(timeout=wait)

    def _do_sample(self, *, reason: str, stage_path: str) -> None:
        t0 = time.perf_counter_ns()
        try:
            tree = _collect_tree_snap(self._parent_pid)
        except Exception:
            return
        tree_rss = sum(s.get("rss_mb", 0.0) for s in tree if s.get("rss_mb", -1.0) >= 0.0)
        tree_pss = sum(s.get("pss_mb", 0.0) for s in tree if s.get("pss_mb", -1.0) >= 0.0)
        sample_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        with self._lock:
            seq = self._sample_seq
            self._sample_seq += 1

        for snap in tree:
            snap["sample_seq"] = seq
            snap["run_id"] = self._run_id
            snap["stage_path"] = stage_path
            snap["sample_elapsed_ms"] = round(sample_ms, 3)
            snap["reason"] = reason

        if tree_rss > self._peak_tree_rss:
            self._peak_tree_rss = tree_rss
            self._peak_tree_pss = tree_pss
            self._peak_owner_stage = stage_path
            self._peak_sample_seq = seq
            for _t in tree:
                if _t.get("role") == "parent" or _t.get("rss_mb", 0.0) == max(
                    (x.get("rss_mb", 0.0) for x in tree), default=0.0
                ):
                    self._peak_pid = _t.get("pid", 0)
                    self._peak_role = _t.get("role", "")
                    break

        with self._lock:
            for _st in self._spans.values():
                _st.update_peak(tree_rss, tree_pss, self._peak_pid, self._peak_role)

        if self._jsonl_enabled:
            try:
                with self._jsonl_path.open("a") as f:
                    for entry in tree:
                        f.write(json.dumps(entry, default=str) + "\n")
            except OSError as exc:
                if not self._degraded_reported:
                    self._degraded_reported = True
                    _logger.warning(
                        "[SYS] stage=l2_probe status=degraded reason=jsonl_write_failed err=%s",
                        exc,
                    )
