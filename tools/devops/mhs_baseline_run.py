"""Durable instrumented MHS baseline runner (acceptance gate SCENARIO_MHS_PERF_ACCEPT_01).

Runs the all-defaults ``MhsRunConfig`` through ``run_mhs_diagnostic`` with a
whole-process-tree memory sampler attached, then writes wall time, the recorded
worker plan, tree memory peaks, and stage measurements to ``logs/scratch/``.
The report itself is NOT persisted here -- that path is measured separately by
the end-to-end acceptance scenario.

Run::

    uv run python tools/devops/mhs_baseline_run.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run_instrumented_baseline(
    *,
    sample_interval_seconds: float = 1.0,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """One instrumented baseline run; returns the summary written to disk.

    The worker plan is read from the report (every fork point records its
    granted count via the wired observer); tree memory peaks come from the
    externally sampled ``_TreeMemorySampler`` when it observed any sample,
    falling back to the report-attached stats otherwise.
    """
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.resources import _TreeMemorySampler
    from src.mhs.pipeline.config import MhsRunConfig
    from src.mhs.pipeline.orchestrator import run_mhs_diagnostic

    defaults = MhsDiagnosticRequest()
    config = MhsRunConfig(**{
        f.name: getattr(defaults, f.name)
        for f in MhsRunConfig.__dataclass_fields__.values()
        if hasattr(defaults, f.name)
    })
    sampler = _TreeMemorySampler(interval_seconds=sample_interval_seconds)
    sampler.start()
    try:
        wall_start = time.perf_counter()
        report = run_mhs_diagnostic(config)
        wall_seconds = time.perf_counter() - wall_start
    finally:
        tree_stats = sampler.stop()

    report_stats = report.tree_memory
    pss_peak = tree_stats.tree_pss_peak_bytes or (
        report_stats.tree_pss_peak_bytes if report_stats else 0
    )
    uss_peak = tree_stats.tree_uss_peak_bytes or (
        report_stats.tree_uss_peak_bytes if report_stats else 0
    )
    min_available = tree_stats.min_system_available_bytes or (
        report_stats.min_system_available_bytes if report_stats else 0
    )
    gib = 2**30
    summary: dict[str, Any] = {
        "wall_seconds": round(wall_seconds, 3),
        "worker_plan": dict(report.worker_plan),
        "tree_pss_peak_gb": round(pss_peak / gib, 3),
        "tree_uss_peak_gb": round(uss_peak / gib, 3),
        "min_system_available_gb": round(min_available / gib, 3),
        "stage_measurements": [
            {
                "stage": m.stage,
                "elapsed_ms": m.elapsed_ms,
                "rss_bytes": m.rss_bytes,
                "peak_rss_bytes": m.peak_rss_bytes,
                "active_symbols": m.active_symbols,
                "grid_bars": m.grid_bars,
            }
            for m in report.resource_measurements
        ],
        # Acceptance-scenario extras (SCENARIO_MHS_PERF_ACCEPT_01 checks).
        "status": report.status,
        "eligible_symbols": report.eligible_symbols,
        "trials_attempted": report.trials_attempted,
        "granted_book_workers": report.worker_plan.get("books"),
    }
    target = out_path or ROOT / "logs" / "scratch" / "mhs_baseline_instrumented.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    summary = run_instrumented_baseline(
        sample_interval_seconds=args.sample_interval, out_path=args.out,
    )
    slim = {k: v for k, v in summary.items() if k != "stage_measurements"}
    sys.stdout.write(json.dumps(slim, indent=2) + "\n")  # noqa: T201


if __name__ == "__main__":
    main()
