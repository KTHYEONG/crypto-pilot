"""Sequential supervisor: run all 4 cross-TF replay labels, then diagnose.

[ADR_20260715_L0_L1_DIAGNOSTIC_PIPELINE_INTEGRITY] New orchestrator wiring the
previously-orphaned diagnose_snapshots()/write_cross_tf_diagnosis() contract to
the 4 replay run artifacts.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

from src.domain.futures.alpha_foundry.contracts import CrossTfDiagnosticRun, CrossTfStageSnapshot
from src.domain.futures.strategy.tiered_workflow.cross_tf_diagnostics import (
    STAGE_ORDER,
    diagnose_snapshots,
    snapshot_from_raw_stage_entry,
    write_cross_tf_diagnosis,
)

_LABELS: tuple[CrossTfDiagnosticRun, ...] = ("control", "control_repeat", "treatment", "fusion_ablation")
_OUT_DIR = Path("logs/futures/diagnostics/l1_cross_tf")


@dataclass(frozen=True, slots=True)
class SupervisorRunRecord:
    label: str
    returncode: int
    signal: int | None
    exit_code: int | None
    reason: str | None
    wall_s: float
    peak_rss_mb: float
    last_stage: str | None


def _last_stage_reached(trace: dict[str, dict[str, object]]) -> str | None:
    reached = [s for s in STAGE_ORDER if trace.get(s)]
    return reached[-1] if reached else None


def _run_one_label(label: str) -> SupervisorRunRecord:
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-m", "src.domain.futures.strategy.run_l1_cross_tf_replay", label])  # noqa: S603
    peak_rss_mb = 0.0
    try:
        ps_proc = psutil.Process(proc.pid)
        while proc.poll() is None:
            peak_rss_mb = max(peak_rss_mb, ps_proc.memory_info().rss / (1024 * 1024))
            time.sleep(1.0)
    except psutil.NoSuchProcess:
        pass
    returncode = proc.wait()
    wall_s = time.perf_counter() - t0
    payload = json.loads((_OUT_DIR / f"{label}.json").read_text(encoding="utf-8"))
    runner_result = payload.get("runner_result")
    return SupervisorRunRecord(
        label=label,
        returncode=returncode,
        signal=(-returncode if returncode < 0 else None),
        exit_code=(runner_result or {}).get("exit_code"),
        reason=(runner_result or {}).get("reason"),
        wall_s=wall_s,
        peak_rss_mb=peak_rss_mb,
        last_stage=_last_stage_reached(payload),
    )


def run_supervised() -> int:
    records = [_run_one_label(label) for label in _LABELS]
    (_OUT_DIR / "supervisor.json").write_text(
        json.dumps([asdict(r) for r in records], sort_keys=True),
        encoding="utf-8",
    )
    snapshots: list[CrossTfStageSnapshot] = []
    for label in _LABELS:
        payload = json.loads((_OUT_DIR / f"{label}.json").read_text(encoding="utf-8"))
        for stage in STAGE_ORDER:
            for tf, entry in payload.get(stage, {}).items():
                snapshots.append(snapshot_from_raw_stage_entry(run=label, stage=stage, timeframe=tf, entry=entry))
    diagnosis = diagnose_snapshots(snapshots=snapshots)
    write_cross_tf_diagnosis(diagnosis=diagnosis, output_path=_OUT_DIR / "diagnosis.json")
    all_exit_ok = all(r.exit_code == 0 for r in records)
    return 0 if (all_exit_ok and diagnosis.complete) else 1


if __name__ == "__main__":
    raise SystemExit(run_supervised())
